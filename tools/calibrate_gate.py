"""Arrival-certificate calibration (Yoga_edge_framework_v3.md §3).

Fits the per-node gate thresholds against takeover-oracle labels and emits
`<node_dir>/gate.yaml` — the ONE artifact consumed identically by the edge
training scoreboard, the offline edge eval, and the inference handoff.

Certificate channels (all deterministic, computed from harvested state blobs):
  C1  pose_dist to the node's hold reference: mean AND per-body max
  C2  support: required contact set loaded (force > thresh) + load share
      >= alpha * mg + NO force on forbidden bodies (= all contact-candidate
      bodies not in the required set)
  C3  |up_z - hold up_z| within tolerance
  C4  support-polygon CoM margin  > m_min (per-node, from clean-hold telemetry)
  C5  quiescence: mean |dof_vel| and horizontal CoM speed below thresholds
 (C6  dwell is a runtime notion — consumers hold the gate for gate.dwell_s.)

Fit rule: thresholds = percentiles of the ORACLE-PASS distribution (loosest
gate consistent with the node's real competence), then verified: joint
precision vs oracle labels + hard negatives must clear --min_precision, else
the most discriminative channel is tightened stepwise. Hard negatives = other
nodes' hold states (oracle-FAIL by assumption — a warrior3 hold is not in
downdog's basin) + this node's own harvest fail-window states.

Usage (CPU, no Isaac):
  env_isaaclab/bin/python tools/calibrate_gate.py \
      --node_dir output/yoga_nodes_v2/downdog_lt_ampft \
      --neg_states output/yoga_nodes_v2/crow_lt_comfwd/hold_states.pt \
                   output/yoga_nodes_v2/warrior3_lt_ampft/hold_states.pt
Outputs: <node_dir>/gate.yaml, <node_dir>/gate_report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import envs.char_env as char_env
import util.torch_util as torch_util

LOAD_N = 5.0          # N; a unit "carries load" (used to DERIVE the required set)
REQ_N = 2.0           # N; a required unit merely must not be AIRBORNE per-frame
                      # (weight-bearing is enforced by the load-share channel;
                      # momentary per-unit lightness is normal within a hold and
                      # is smoothed by the runtime dwell, not the instant check)
FORBID_N = 5.0        # N; force on a forbidden unit above this = violation
# three-tier unit classes from the oracle-PASS load fractions: REQUIRED
# (loaded in ~all pass states), OPTIONAL (sometimes loaded — no constraint;
# e.g. downdog hands, which B freely lifts and replants inside its basin),
# FORBIDDEN (rarely loaded — e.g. crow's dab feet).
REQUIRED_FRAC = 0.85
FORBIDDEN_FRAC = 0.25

# Contact-candidate bodies are grouped into support UNITS before deriving the
# required/forbidden sets: heel-vs-toe (Ankle/Toe geoms) and wrist-vs-palm
# trade load freely within a planted limb (the triangle toe-load false-negative
# lesson), so the gate reasons about limbs, not geoms.
UNIT_OF = {}
for _s in "LR":
    UNIT_OF[f"{_s}_Wrist"] = f"hand:{_s}"; UNIT_OF[f"{_s}_Hand"] = f"hand:{_s}"
    UNIT_OF[f"{_s}_Ankle"] = f"foot:{_s}"; UNIT_OF[f"{_s}_Toe"] = f"foot:{_s}"
    UNIT_OF[f"{_s}_Elbow"] = f"elbow:{_s}"; UNIT_OF[f"{_s}_Knee"] = f"knee:{_s}"
PCTL = 97.5           # gate percentile of the oracle-PASS distribution
# physical floors: the gate may never be tighter than these (A4 defusal)
FLOORS = dict(pose_mean=0.05, pose_max=0.10, up_z_tol=0.05,
              dof_speed=0.5, com_speed=0.05)
# ... nor looser than these caps (unexercised-channel guard)
CAPS = dict(pose_mean=0.40, pose_max=0.90, up_z_tol=0.60,
            dof_speed=6.0, com_speed=0.60)


def pose_dist_to_ref(body_pos, root_rot, ref_body_pos, ref_root_rot):
    cur_rel = body_pos[:, 1:, :] - body_pos[:, 0:1, :]
    cur_local = char_env.convert_to_local_body_pos(root_rot, cur_rel)
    ref_rel = ref_body_pos[:, 1:, :] - ref_body_pos[:, 0:1, :]
    ref_local = char_env.convert_to_local_body_pos(ref_root_rot, ref_rel)
    d = torch.linalg.vector_norm(cur_local - ref_local, dim=-1)
    return d.mean(dim=-1), d.max(dim=-1)[0]


def unit_forces(f, body_names, contact_ids):
    """Sum per-body force norms into support units. Returns (unit_names,
    unit_f [K,U])."""
    units = {}
    for i in contact_ids:
        u = UNIT_OF.get(body_names[i], body_names[i])
        units.setdefault(u, []).append(i)
    unit_names = sorted(units)
    uf = torch.stack([f[:, units[u]].sum(dim=-1) for u in unit_names], dim=-1)
    return unit_names, uf


def channels(blob, hold_ref, required_units, forbidden_units, alpha_mg):
    """Per-state certificate channel values + per-channel pass booleans."""
    bp, rr = blob["body_pos"], blob["root_rot"]
    ref_bp = hold_ref["body_pos"].expand(bp.shape[0], -1, -1)
    ref_rr = hold_ref["root_rot"].expand(bp.shape[0], -1)
    pd_mean, pd_max = pose_dist_to_ref(bp, rr, ref_bp, ref_rr)

    f = torch.linalg.vector_norm(blob["contact_forces"], dim=-1)   # [K,B]
    body_names = blob["meta"]["body_names"]
    contact_ids = blob["meta"]["contact_body_ids"].tolist()
    unit_names, uf = unit_forces(f, body_names, contact_ids)
    ureq = [unit_names.index(u) for u in required_units if u in unit_names]
    uforb = [unit_names.index(u) for u in forbidden_units if u in unit_names]
    # load share sums ALL non-forbidden (support-eligible) units: total weight
    # must ride plausible support, whichever limbs currently carry it.
    usup = [i for i, u in enumerate(unit_names) if u not in forbidden_units]
    req_loaded = (uf[:, ureq] > REQ_N).all(dim=-1) if ureq else \
        torch.ones(f.shape[0], dtype=torch.bool)
    load_share = uf[:, usup].sum(dim=-1) if usup else torch.zeros(f.shape[0])
    forb = (uf[:, uforb] > FORBID_N).any(dim=-1) if uforb else \
        torch.zeros(f.shape[0], dtype=torch.bool)

    up_z = blob["up_z"]
    margin_col = blob["meta"]["feature_names"].index("support_margin")
    margin = blob["features"][:, margin_col]
    com_speed = torch.linalg.vector_norm(blob["com_vel"][:, :2], dim=-1)
    dof_speed = blob["dof_speed"]

    return dict(pose_mean=pd_mean, pose_max=pd_max, req_loaded=req_loaded,
                load_share=load_share, forbidden=forb, up_z=up_z,
                margin=margin, com_speed=com_speed, dof_speed=dof_speed)


def apply_gate(ch, g, hold_up_z, alpha_mg):
    ok = (ch["pose_mean"] < g["pose_mean"]) \
        & (ch["pose_max"] < g["pose_max"]) \
        & ch["req_loaded"] \
        & (ch["load_share"] >= alpha_mg) \
        & ~ch["forbidden"] \
        & ((ch["up_z"] - hold_up_z).abs() < g["up_z_tol"]) \
        & (ch["margin"] > g["margin_min"]) \
        & (ch["com_speed"] < g["com_speed"]) \
        & (ch["dof_speed"] < g["dof_speed"])
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", required=True)
    p.add_argument("--states", default=None)
    p.add_argument("--oracle", default=None)
    p.add_argument("--neg_states", nargs="*", default=[])
    p.add_argument("--min_precision", type=float, default=0.95)
    p.add_argument("--dwell_s", type=float, default=1.0)
    args = p.parse_args()

    node = os.path.basename(args.node_dir)
    states_path = args.states or os.path.join(args.node_dir, "hold_states.pt")
    oracle_path = args.oracle or (states_path + ".oracle.pt")
    blob = torch.load(states_path, map_location="cpu")
    orc = torch.load(oracle_path, map_location="cpu")
    sel = orc["sel"]
    sub = {k: (blob[k][sel] if torch.is_tensor(blob[k]) and blob[k].shape[:1] == blob["root_pos"].shape[:1]
               else blob[k]) for k in blob}
    verdict = orc["verdict"]

    meta = blob["meta"]
    body_names = meta["body_names"]
    contact_ids = meta["contact_body_ids"].tolist()
    char_weight = meta["char_weight"]
    hold_ref = blob["hold_ref"]
    hold_up_z = float(hold_ref["up_z"][0])

    # oracle validity precheck (framework §3.2): B from its own CLEAN held states
    lab = sub["hold_label"]; sea = sub["seasoned"]; noise = sub["noise"]
    clean = (lab == 1.0) & sea & (noise <= 0.02)
    validity = float(verdict[clean].float().mean()) if clean.any() else float("nan")
    print(f"[{node}] oracle validity (clean held states): {validity:.3f} "
          f"({int(clean.sum())} states)")

    # required support UNITS from oracle-PASS states' measured loads
    f = torch.linalg.vector_norm(sub["contact_forces"], dim=-1)
    pass_states = verdict
    unit_names, uf = unit_forces(f, body_names, contact_ids)
    load_frac = (uf[pass_states] > LOAD_N).float().mean(dim=0)
    required_units = [unit_names[i] for i in range(len(unit_names))
                      if load_frac[i] >= REQUIRED_FRAC]
    forbidden_units = [unit_names[i] for i in range(len(unit_names))
                       if load_frac[i] <= FORBIDDEN_FRAC]
    optional_units = [u for u in unit_names
                      if u not in required_units and u not in forbidden_units]
    print("per-unit load fractions (oracle-PASS):",
          {unit_names[i]: round(float(load_frac[i]), 3) for i in range(len(unit_names))})
    print(f"required={required_units} optional={optional_units} forbidden={forbidden_units}")
    required_names, forbidden_names = required_units, forbidden_units
    # load share floor: percentile of pass states' support-load (vs mg),
    # over all non-forbidden units (matches channels())
    usup = [i for i, u in enumerate(unit_names) if u not in forbidden_units]
    share_pass = uf[pass_states][:, usup].sum(dim=-1) / char_weight
    alpha = float(np.percentile(share_pass.numpy(), 100.0 - PCTL))
    alpha = min(max(alpha, 0.30), 0.90)
    alpha_mg = alpha * char_weight

    ch = channels(sub, hold_ref, required_units, forbidden_units, alpha_mg)

    # thresholds from the oracle-PASS distribution, floored + capped
    def fit(name, series, upper=True):
        v = series[pass_states].numpy()
        x = float(np.percentile(v, PCTL if upper else 100.0 - PCTL))
        return float(np.clip(x, FLOORS.get(name, -1e9), CAPS.get(name, 1e9))) if upper \
            else x
    gate = dict(
        pose_mean=fit("pose_mean", ch["pose_mean"]),
        pose_max=fit("pose_max", ch["pose_max"]),
        up_z_tol=float(np.clip(np.percentile((ch["up_z"][pass_states] - hold_up_z).abs().numpy(), PCTL),
                               FLOORS["up_z_tol"], CAPS["up_z_tol"])),
        margin_min=float(np.percentile(ch["margin"][pass_states].numpy(), 100.0 - PCTL)),
        com_speed=fit("com_speed", ch["com_speed"]),
        dof_speed=fit("dof_speed", ch["dof_speed"]),
    )

    # per-channel recall diagnostic (pass rate among oracle-PASS states)
    per_ch = dict(
        pose_mean=float((ch["pose_mean"][pass_states] < gate["pose_mean"]).float().mean()),
        pose_max=float((ch["pose_max"][pass_states] < gate["pose_max"]).float().mean()),
        req_loaded=float(ch["req_loaded"][pass_states].float().mean()),
        load_share=float((ch["load_share"][pass_states] >= alpha_mg).float().mean()),
        not_forbidden=float((~ch["forbidden"][pass_states]).float().mean()),
        up_z=float(((ch["up_z"][pass_states] - hold_up_z).abs() < gate["up_z_tol"]).float().mean()),
        margin=float((ch["margin"][pass_states] > gate["margin_min"]).float().mean()),
        com_speed=float((ch["com_speed"][pass_states] < gate["com_speed"]).float().mean()),
        dof_speed=float((ch["dof_speed"][pass_states] < gate["dof_speed"]).float().mean()),
    )
    print("per-channel recall on oracle-PASS states:",
          {k: round(v, 3) for k, v in per_ch.items()})

    # evaluation set: own states (oracle-labeled) + hard negatives
    own_ok = apply_gate(ch, gate, hold_up_z, alpha_mg)
    neg_blobs = []
    for np_path in args.neg_states:
        nb = torch.load(np_path, map_location="cpu")
        nch = channels(nb, hold_ref, required_units, forbidden_units, alpha_mg)
        neg_blobs.append((os.path.basename(os.path.dirname(np_path)), nb, nch))

    def joint_stats(g):
        ok = apply_gate(ch, g, hold_up_z, alpha_mg)
        tp = int((ok & verdict).sum()); fp = int((ok & ~verdict).sum())
        fn = int((~ok & verdict).sum())
        neg_fp = 0; neg_n = 0
        for _, nb, nch in neg_blobs:
            nok = apply_gate(nch, g, hold_up_z, alpha_mg)
            neg_fp += int(nok.sum()); neg_n += len(nok)
        prec = tp / max(tp + fp + neg_fp, 1)
        rec = tp / max(tp + fn, 1)
        return prec, rec, dict(tp=tp, fp=fp, fn=fn, neg_fp=neg_fp, neg_n=neg_n)

    prec, rec, st = joint_stats(gate)
    # tighten stepwise (pose first — the most discriminative channel) until the
    # precision bar is met
    tighten_seq = ["pose_mean", "pose_max", "com_speed", "dof_speed", "up_z_tol"]
    ti = 0
    while prec < args.min_precision and ti < 40:
        k = tighten_seq[ti % len(tighten_seq)]
        gate[k] = max(gate[k] * 0.85, FLOORS.get(k, 0.0))
        prec, rec, st = joint_stats(gate)
        ti += 1
    if prec < args.min_precision:
        print(f"!! GATE DEGENERATE: precision {prec:.3f} < {args.min_precision} "
              f"after {ti} tighten steps — do NOT use this gate.yaml")

    report = dict(
        node=node, oracle_validity_clean=round(validity, 4),
        n_labeled=int(len(verdict)), oracle_pass_rate=round(float(verdict.float().mean()), 4),
        required_contacts=required_names, forbidden_contacts=forbidden_names,
        load_share_alpha=round(alpha, 3), gate={k: round(v, 4) for k, v in gate.items()},
        precision=round(prec, 4), recall=round(rec, 4), counts=st,
        tighten_steps=ti,
        hard_negative_sources=[n for n, _, _ in neg_blobs],
    )
    print(json.dumps(report, indent=2))

    gate_yaml = dict(
        node=node, version="edge-framework-v3",
        theta_pose_mean=round(gate["pose_mean"], 4),
        theta_pose_max=round(gate["pose_max"], 4),
        up_z=round(hold_up_z, 4), up_z_tol=round(gate["up_z_tol"], 4),
        required_contacts=required_names, forbidden_contacts=forbidden_names,
        load_share_alpha=round(alpha, 3), load_n=LOAD_N, forbid_n=FORBID_N,
        com_margin_min=round(gate["margin_min"], 4),
        com_speed_max=round(gate["com_speed"], 4),
        dof_speed_max=round(gate["dof_speed"], 4),
        dwell_s=args.dwell_s,
        provenance=dict(
            oracle_validity_clean=round(validity, 4),
            oracle_hold_seconds=float(orc["hold_seconds"]),
            n_states=int(len(verdict)), precision=round(prec, 4),
            recall=round(rec, 4), states_file=states_path,
            hard_negatives=[n for n, _, _ in neg_blobs],
            fit_percentile=PCTL),
    )
    out_yaml = os.path.join(args.node_dir, "gate.yaml")
    with open(out_yaml, "w") as fh:
        yaml.safe_dump(gate_yaml, fh, sort_keys=False)
    with open(os.path.join(args.node_dir, "gate_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out_yaml}")


if __name__ == "__main__":
    main()
