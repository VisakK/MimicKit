"""Bridge Certifier — ONE command turns an edge->node bridge into a measured
failure-signature vector + a routed repair rung (structured orchestration;
the Yoga_plank_vinyasa_and_baseline.md §3 prescription, built).

What it does (single Isaac boot for the edge; oracle in a subprocess):
  1. PRECHECK (no sim): destination gate + hold_ref; node oracle validity
     (gate.yaml provenance); the edge clip's OWN tail vs the destination gate
     (the demo-tail seam predictor — the certificate-recall~0 root cause).
  2. ROLL the edge honestly from t=0 (init_time_range pinned; per-env action-
     noise spectrum 0..--noise for a delivered-state spread, env 0 exact),
     to the settle time (default env episode_length - 0.5 — the warrior3
     truncation lesson). Per step: destination-gate channels, gait telemetry
     (loaded-body slide speed, contact flicker, CoM jerk), CoM-margin trace.
  3. CAPTURE surviving envs' full states at settle, SE(2)-aligned to the
     node's hold_ref -> cert_arrivals.pt (same blob the B1 loop consumes).
  4. RECOVERY ORACLE (subprocess tools/takeover_oracle.py): does the node's
     own policy hold the delivered states? (the campaign's ground truth).
  5. EMIT bridge_cert.json {channels, signature, verdict}: the signature is
     computed by tools/route_repair.py against skills/repair_router.yaml
     (single owner of all thresholds), and the verdict is the ONE
     pre-registered repair rung to execute next.

Usage (repo root, env python, env -u DISPLAY; ONE Isaac job at a time):
  env -u DISPLAY /home/visakii/Documents/moves/env_isaaclab/bin/python \
      tools/certify_bridge.py \
      --edge_dir output/yoga_edges_v3/downdog_to_tadasana_deepmimic \
      --target_node_dir output/yoga_nodes_v2/tadasana_lt_ampft
Outputs (in --edge_dir unless --out_dir): bridge_cert.json, route_verdict.json,
cert_arrivals.pt (+ .oracle.pt)
"""
import argparse
import json
import os
import subprocess
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import envs.env_builder as env_builder
import envs.base_env as base_env
import envs.char_env as char_env
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.util as util

# tools/ and skillgraph/ go on the path AFTER the mimickit imports — tools/
# shadows the `util` package otherwise
sys.path.append(os.path.join(REPO, "tools"))
sys.path.append(os.path.join(REPO, "mimickit", "skillgraph"))
import state_features as sf
import calibrate_gate as cg
import route_repair
import takeover_oracle

PY = sys.executable
STATE_KEYS = ["root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]


def heading_align(state, ref_root_pos, ref_root_rot):
    """SE(2)-align a batch into the node's reference frame (harvest semantics)."""
    q_s = torch_util.calc_heading_quat(state["root_rot"])
    q_r = torch_util.calc_heading_quat(ref_root_rot.expand_as(state["root_rot"]))
    q = torch_util.quat_mul(q_r, torch_util.quat_conjugate(q_s))
    out = {k: v.clone() for k, v in state.items()}
    out["root_rot"] = torch_util.quat_mul(q, state["root_rot"])
    for k in ("root_vel", "root_ang_vel"):
        out[k] = torch_util.quat_rotate(q, state[k])
    rp = state["root_pos"].clone()
    rp[:, 0:2] = ref_root_pos[0, 0:2]
    out["root_pos"] = rp
    return out


def local_pose(bp, rr):
    """Heading-local root-relative body positions (the gate's pose space)."""
    rel = bp[:, 1:, :] - bp[:, 0:1, :]
    return char_env.convert_to_local_body_pos(rr, rel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edge_dir", required=True)
    p.add_argument("--edge_model", default="model.pt")
    p.add_argument("--target_node_dir", required=True)
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--init_states", default=None,
                   help="CHAINED cert: a delivered-distribution blob (.pt, "
                        "STATE_KEYS[+motion_time]) from the predecessor stage. "
                        "Envs reset FROM these states (cycled over N, SE(2)-"
                        "aligned to this edge's t=0 lead-in reference, motion "
                        "clock at 0) instead of the canonical t=0 reference "
                        "pose — the protocol's 'certify on the harvested "
                        "delivered distribution, not canonical starts'.")
    p.add_argument("--noise", type=float, default=0.08,
                   help="max per-env action-noise std (spread; env 0 exact)")
    p.add_argument("--settle_at", type=float, default=None,
                   help="nominal capture horizon (s); default = "
                        "min(episode_length, motion_length) - 0.5. Each env's "
                        "DELIVERY is its last live state (rolling capture), so "
                        "clip-end SUCC/TIME completions count as deliveries.")
    p.add_argument("--late_window_s", type=float, default=1.5,
                   help="hold-quality window ending at settle")
    p.add_argument("--hold_seconds", type=float, default=5.0,
                   help="recovery-oracle hold")
    p.add_argument("--router", default=os.path.join(REPO, "skills", "repair_router.yaml"))
    p.add_argument("--skip_oracle", action="store_true",
                   help="kinematic-only cert (recovery reported as -1)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29650)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or args.edge_dir
    os.makedirs(out_dir, exist_ok=True)
    B = args.target_node_dir
    gate = yaml.safe_load(open(os.path.join(B, "gate.yaml")))
    b_blob = torch.load(os.path.join(B, "hold_states.pt"), map_location="cpu")
    hold_ref = b_blob["hold_ref"]
    b_char_weight = b_blob["meta"]["char_weight"]
    node_validity = float(gate.get("provenance", {}).get("oracle_validity_clean", 1.0))

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)
    env_cfg_path = os.path.join(args.edge_dir, "env_config.yaml")
    env = env_builder.build_env(env_cfg_path, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(os.path.join(args.edge_dir, "agent_config.yaml"),
                                      env, device)
    agent.load(os.path.join(args.edge_dir, args.edge_model))
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)
    env._init_time_range = [0.0, 0.0]   # the honest exam: t=0, no spread

    chained_from = None
    if args.init_states:
        # chained cert: reset every env from the predecessor's delivered
        # distribution, aligned into this edge's t=0 lead-in frame
        blob = torch.load(args.init_states, map_location="cpu")
        K_in = blob["root_pos"].shape[0]
        pad = torch.arange(args.num_envs) % K_in
        in_state = {k: blob[k][pad].to(device) for k in STATE_KEYS}
        mids0 = torch.zeros(1, dtype=torch.long, device=device)
        t00 = torch.zeros(1, device=device)
        rp0, rr0, _, _, _, _ = env._motion_lib.calc_motion_frame(mids0, t00)
        aligned0 = heading_align(in_state, rp0, rr0)
        aligned0["motion_time"] = torch.zeros(args.num_envs, device=device)
        env._reset_char = takeover_oracle.make_inject_reset(env, aligned0)
        chained_from = os.path.basename(args.init_states)
        print(f"[cert] CHAINED init: {K_in} delivered states from "
              f"{args.init_states} -> {args.num_envs} envs")

    kcm = env._kin_char_model
    body_names = kcm.get_body_names()
    contact_ids = env._obs_contact_body_ids
    key_ids = env._key_body_ids
    com_w = env._com_body_weights
    cid = env._get_char_id()
    e = env._engine
    control_freq = int(round(1.0 / e.get_timestep()))
    dt = 1.0 / control_freq
    ep_len = float(yaml.safe_load(open(env_cfg_path))["env"].get("episode_length", 12.0))
    mlen = float(env._motion_lib.get_motion_length(
        torch.zeros(1, dtype=torch.long, device=device))[0])
    natural_end = min(ep_len, mlen)
    settle = args.settle_at if args.settle_at is not None else natural_end - 0.5
    settle_step = int(round(settle * control_freq))
    N = args.num_envs
    NULL, FAIL = base_env.DoneFlags.NULL.value, base_env.DoneFlags.FAIL.value
    force_thr = float(getattr(env, "_obs_contact_force_threshold", 1.0))

    # ---- destination-gate machinery (unit indices vs the edge env's contact list)
    ref_bp = hold_ref["body_pos"].to(device)
    ref_rr = hold_ref["root_rot"].to(device)
    ref_local = local_pose(ref_bp, ref_rr)              # [1, B-1, 3]
    units = {}
    for i in contact_ids.tolist():
        units.setdefault(cg.UNIT_OF.get(body_names[i], body_names[i]), []).append(i)
    unit_names = sorted(units)
    req_units = [u for u in gate["required_contacts"] if u in unit_names]
    req_idx = [unit_names.index(u) for u in req_units]
    forb_idx = [unit_names.index(u) for u in gate["forbidden_contacts"] if u in unit_names]
    sup_idx = [i for i, u in enumerate(unit_names) if u not in gate["forbidden_contacts"]]
    alpha_mg = gate["load_share_alpha"] * b_char_weight
    dwell_steps = int(round(float(gate["dwell_s"]) * control_freq))

    # ---- precheck: the edge clip's OWN tail (demo-tail seam predictor)
    t_tail = torch.tensor([max(0.0, mlen - 0.1)], device=device)
    mid0 = torch.zeros(1, dtype=torch.long, device=device)
    trp, trr, _, _, tjr, _ = env._motion_lib.calc_motion_frame(mid0, t_tail)
    tail_bp, _ = kcm.forward_kinematics(trp, trr, tjr)
    tail_local = local_pose(tail_bp, trr)               # [1, B-1, 3]
    edge_tail_vs_gate = float(torch.linalg.vector_norm(
        tail_local - ref_local, dim=-1).mean())

    # ---- hold-onset: the earliest time from which the REFERENCE stays within
    # 2*theta of the node hold_ref — hold-quality channels must measure the
    # HOLD, not the transit (the tadasana smoke-test lesson: a 1.5s window over
    # a 5.5s clip covered the rise and misread transit stepping as dirty gait)
    ts = torch.arange(0.0, max(mlen - 1e-3, 0.1), 0.1, device=device)
    mids_s = torch.zeros(len(ts), dtype=torch.long, device=device)
    rp_s, rr_s, _, _, jr_s, _ = env._motion_lib.calc_motion_frame(mids_s, ts)
    bp_s, _ = kcm.forward_kinematics(rp_s, rr_s, jr_s)
    dref = torch.linalg.vector_norm(
        local_pose(bp_s, rr_s) - ref_local, dim=-1).mean(dim=-1)
    inside = (dref < 2.0 * float(gate["theta_pose_mean"])).tolist()
    hold_onset = mlen - 1.0
    for i in range(len(inside)):
        if all(inside[i:]):
            hold_onset = float(ts[i])
            break

    # ---- style-agnostic limbs (the destination hub's port spec): excluded
    # from the style/extension metric — tadasana's arms gap is a SEAM (B1/E1),
    # not a survival exploit
    STYLE_PATTERNS = {"arms": ("Shoulder", "Elbow", "Wrist", "Hand"),
                      "legs": ("Hip", "Knee", "Ankle", "Toe", "Foot")}
    agnostic_bodies = set()
    graph_path = os.path.join(REPO, "skills", "graph.yaml")
    if os.path.exists(graph_path):
        gph = yaml.safe_load(open(graph_path))
        b_base = os.path.basename(os.path.normpath(B))
        for node in gph.get("nodes", []):
            nd = node if isinstance(node, dict) else gph["nodes"][node]
            if os.path.basename(str(nd.get("dir", ""))) == b_base:
                for limb in (nd.get("port", {}) or {}).get("style_agnostic", []):
                    pats = STYLE_PATTERNS.get(limb, (limb,))
                    agnostic_bodies |= {n for n in body_names
                                        if any(p in n for p in pats)}
                break

    # hold-quality window = [max(hold_onset, settle - late_window_s), settle]
    late_start_s = min(max(hold_onset, settle - args.late_window_s), settle - 0.5)
    late_from = int(round(late_start_s * control_freq))
    print(f"[cert] clip {mlen:.2f}s, hold-onset {hold_onset:.2f}s, "
          f"hold window [{late_start_s:.2f}, {settle:.2f}]s"
          + (f", style-agnostic: {sorted(agnostic_bodies)}" if agnostic_bodies else ""))

    # ---- rollout accumulators
    scale = torch.linspace(0.0, args.noise, N, device=device).unsqueeze(-1)
    obs, info = agent._reset_envs()
    fell = torch.zeros(N, dtype=torch.bool, device=device)
    ended = torch.zeros(N, dtype=torch.bool, device=device)
    first_fail_t = torch.full((N,), -1.0, device=device)
    dwell = torch.zeros(N, dtype=torch.long, device=device)
    arrived = torch.zeros(N, dtype=torch.bool, device=device)
    captured = None
    prev_com_v = None
    prev_loaded = None
    jerk_sum, jerk_n = 0.0, 0
    late = dict(n=0, req=0.0, forb=0.0, share=0.0, pd_sum=None, per_body=None,
                cz_sum=None, ext_sum=None, slide_sum=0.0, slide_n=0,
                toggles=0.0, margin_rows=[])

    for step in range(settle_step + 1):
        # rolling DELIVERY capture: each still-live env's latest pre-step state.
        # A clip-end SUCC/TIME completion keeps its last live state as the
        # delivery (the 5.5s-clip lesson: don't demand survival past the clip).
        live_pre = ~ended
        if bool(live_pre.any()):
            st_now = {k: getattr(e, "get_" + k)(cid) for k in STATE_KEYS}
            if captured is None:
                captured = {k: v.clone() for k, v in st_now.items()}
            else:
                for k in captured:
                    captured[k][live_pre] = st_now[k][live_pre]
        with torch.no_grad():
            a = agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        a = a + scale * torch.randn_like(a)

        obs, r, done, info = env.step(a)
        live = ~ended
        bp = e.get_body_pos(cid)
        rr = e.get_root_rot(cid)
        bv = e.get_body_vel(cid)
        f = torch.linalg.vector_norm(e.get_ground_contact_forces(cid), dim=-1)
        com_v = torch.einsum("nbk,b->nk", bv, com_w)
        if prev_com_v is not None and bool(live.any()):
            jerk = torch.linalg.vector_norm(com_v - prev_com_v, dim=-1) / dt
            jerk_sum += float(jerk[live].mean())
            jerk_n += 1
        prev_com_v = com_v.clone()

        # gate channels + dwell/arrival (certificate semantics)
        cur_local = local_pose(bp, rr)
        d = torch.linalg.vector_norm(cur_local - ref_local, dim=-1)   # [N, B-1]
        pd_mean = d.mean(dim=-1)
        uf = torch.stack([f[:, units[u]].sum(dim=-1) for u in unit_names], dim=-1)
        req = (uf[:, req_idx] > cg.REQ_N).all(dim=-1) if req_idx else \
            torch.ones(N, dtype=torch.bool, device=device)
        forb = (uf[:, forb_idx] > cg.FORBID_N).any(dim=-1) if forb_idx else \
            torch.zeros(N, dtype=torch.bool, device=device)
        share_ok = uf[:, sup_idx].sum(dim=-1) >= alpha_mg
        up_l = torch.zeros(N, 3, device=device); up_l[:, 2] = 1.0
        upz = torch_util.quat_rotate(rr, up_l)[:, 2]
        feats = sf.compute_state_features(
            root_pos=e.get_root_pos(cid), root_rot=rr, root_vel=e.get_root_vel(cid),
            root_ang_vel=e.get_root_ang_vel(cid), body_pos=bp,
            contact_forces=e.get_ground_contact_forces(cid),
            contact_body_ids=contact_ids, key_body_ids=key_ids, com_weights=com_w,
            force_threshold=force_thr,
            support_dirs=getattr(env, "_support_polygon_dirs", None))
        margin = feats[:, 7]
        com_speed = torch.linalg.vector_norm(com_v[:, :2], dim=-1)
        dof_speed = e.get_dof_vel(cid).abs().mean(dim=-1)
        gate_ok = ((pd_mean < gate["theta_pose_mean"])
                   & (d.max(dim=-1)[0] < gate["theta_pose_max"])
                   & req & share_ok & ~forb
                   & ((upz - float(hold_ref["up_z"][0])).abs() < gate["up_z_tol"])
                   & (margin > gate["com_margin_min"])
                   & (com_speed < gate["com_speed_max"])
                   & (dof_speed < gate["dof_speed_max"]))
        dwell = torch.where(gate_ok & live & ~arrived, dwell + 1,
                            torch.zeros_like(dwell))
        arrived |= (dwell >= dwell_steps) & live

        # late-window (hold-quality) accumulators over LIVE envs
        if step >= late_from and bool(live.any()):
            lf = live.float()
            denom = float(lf.sum())
            late["n"] += 1
            late["req"] += float((req.float() * lf).sum()) / denom
            late["forb"] += float(((~forb).float() * lf).sum()) / denom
            late["share"] += float((share_ok.float() * lf).sum()) / denom
            pb = d * lf.unsqueeze(-1)                      # [N, B-1]
            late["per_body"] = pb.sum(0) / max(denom, 1.0) if late["per_body"] is None \
                else late["per_body"] + pb.sum(0) / max(denom, 1.0)
            late["pd_sum"] = pd_mean[live].median().item() if late["pd_sum"] is None \
                else late["pd_sum"] + pd_mean[live].median().item()
            cz = bp[:, contact_ids, 2]
            late["cz_sum"] = (cz * lf.unsqueeze(-1)).sum(0) / denom if late["cz_sum"] is None \
                else late["cz_sum"] + (cz * lf.unsqueeze(-1)).sum(0) / denom
            ext = torch.linalg.vector_norm(bp[:, key_ids, :] - bp[:, 0:1, :], dim=-1)
            late["ext_sum"] = (ext * lf.unsqueeze(-1)).sum(0) / denom if late["ext_sum"] is None \
                else late["ext_sum"] + (ext * lf.unsqueeze(-1)).sum(0) / denom
            loaded = f[:, contact_ids] > force_thr        # [N, C]
            hspeed = torch.linalg.vector_norm(bv[:, contact_ids, :2], dim=-1)
            sel = loaded & live.unsqueeze(-1)
            if bool(sel.any()):
                late["slide_sum"] += float(hspeed[sel].mean())
                late["slide_n"] += 1
            # toggle metric uses a 5N floor: near-zero-force flutter around the
            # 1N obs threshold is not the intermittent-contact defect
            loaded_tog = f[:, contact_ids] > max(5.0, force_thr)
            if prev_loaded is not None:
                tog = (loaded_tog != prev_loaded) & live.unsqueeze(-1)
                late["toggles"] += float(tog.float().sum()) / max(denom, 1.0)
            late["margin_rows"].append(margin[live].median().item())
        prev_loaded = (f[:, contact_ids] > max(5.0, force_thr)).clone()

        new_fail = (done == FAIL) & ~ended
        first_fail_t[new_fail] = (step + 1) * dt
        fell |= new_fail
        ended |= (done != NULL)
        if bool(ended.all()):
            break

    # ---- delivery selection: every env that never FELL delivered its last live state
    keep = (~fell).nonzero(as_tuple=False).flatten()
    n_keep = int(len(keep))
    survive = n_keep / N
    fell_rate = float(fell.float().mean())
    completed = float((ended & ~fell).float().mean())
    print(f"[cert] t=0 rollout (natural end {natural_end:.2f}s, settle {settle:.2f}s): "
          f"delivered {n_keep}/{N} ({survive:.3f}), fell {fell_rate:.3f}, "
          f"clip-end completions {completed:.3f}")

    delivery_vs_tail = float("nan")
    recovery_rate = -1.0
    arrivals_path = os.path.join(out_dir, "cert_arrivals.pt")
    if n_keep > 0:
        state = {k: captured[k][keep] for k in STATE_KEYS}
        # delivered pose vs the edge's OWN tail (truncation/executor artifact tell)
        cap_jr = kcm.dof_to_rot(state["dof_pos"].to(device))
        cap_bp, _ = kcm.forward_kinematics(state["root_pos"].to(device),
                                           state["root_rot"].to(device), cap_jr)
        cap_local = local_pose(cap_bp, state["root_rot"].to(device))
        delivery_vs_tail = float(torch.linalg.vector_norm(
            cap_local - tail_local, dim=-1).mean())
        aligned = heading_align({k: v.cpu() for k, v in state.items()},
                                hold_ref["root_pos"], hold_ref["root_rot"])
        aligned["motion_time"] = hold_ref["motion_time"].expand(n_keep).clone()
        torch.save(aligned, arrivals_path)
        if not args.skip_oracle:
            print(f"[cert] recovery oracle on {n_keep} delivered states ...", flush=True)
            r2 = subprocess.run(
                [PY, "tools/takeover_oracle.py", "--node_dir", B,
                 "--states", arrivals_path, "--hold_seconds", str(args.hold_seconds),
                 "--master_port", str(args.master_port + 1)],
                capture_output=True, text=True)
            if r2.returncode != 0:
                print(r2.stderr[-2000:])
                raise SystemExit("takeover oracle failed")
            orc = torch.load(arrivals_path + ".oracle.pt", map_location="cpu")
            recovery_rate = float(orc["verdict"].float().mean())

    # ---- assemble channels
    nl = max(late["n"], 1)
    per_body = (late["per_body"] / nl) if late["per_body"] is not None else None
    worst = []
    if per_body is not None:
        top = torch.topk(per_body, min(5, per_body.numel()))
        worst = [{"body": body_names[1 + int(i)], "delta_cm": round(100 * float(v), 1)}
                 for v, i in zip(top.values, top.indices)]
    cz = (late["cz_sum"] / nl) if late["cz_sum"] is not None else None
    ref_cz = ref_bp[0, contact_ids, 2]
    support_z = {}
    max_sup_dz = 0.0
    if cz is not None:
        req_members = sorted({i for u in req_units for i in units[u]})
        for i, bi in enumerate(contact_ids.tolist()):
            dz_cm = round(100 * float(cz[i] - ref_cz[i]), 1)
            support_z[body_names[bi]] = dz_cm
            if bi in req_members:
                max_sup_dz = max(max_sup_dz, dz_cm)
    ext = (late["ext_sum"] / nl) if late["ext_sum"] is not None else None
    ref_ext = torch.linalg.vector_norm(
        ref_bp[0, key_ids, :] - ref_bp[0, 0:1, :], dim=-1)
    ext_delta = {}
    max_ext = 0.0
    if ext is not None:
        for i, bi in enumerate(key_ids.tolist()):
            nm = body_names[bi]
            dcm = round(100 * float(ext[i] - ref_ext[i]), 1)
            if nm in agnostic_bodies:
                ext_delta[nm + " (style-agnostic)"] = dcm
            else:
                ext_delta[nm] = dcm
                max_ext = max(max_ext, abs(dcm))
    mrows = late["margin_rows"]
    margin_start = mrows[0] if mrows else float("nan")
    margin_end = mrows[-1] if mrows else float("nan")
    margin_neg_frac = (sum(1 for m in mrows if m < 0) / len(mrows)) if mrows else 0.0

    channels = {
        "harness": {
            "episode_length": ep_len, "motion_length": round(mlen, 2),
            "ran_to_s": round(settle, 2), "init_at": 0.0,
            "clip_end_completion_rate": round(completed, 3),
            "delivery_vs_edge_tail_pose_mean": round(delivery_vs_tail, 4),
            "chained_from": chained_from,
            # explicit early settle (gate-fire cert, R2b semantics): the
            # harness rule must judge against THIS, not the natural end
            "settle_requested_s": args.settle_at,
        },
        "node": {
            "oracle_validity_clean": node_validity,
            "gate_theta_pose_mean": float(gate["theta_pose_mean"]),
            "gate_calibrated_on": "hold_states",
        },
        "demo": {"edge_tail_vs_gate_pose_mean": round(edge_tail_vs_gate, 4),
                 "hold_onset_s": round(hold_onset, 2),
                 "hold_window_s": [round(late_start_s, 2), round(settle, 2)]},
        "reachability": {
            "fell_rate_t0": round(fell_rate, 4),
            "survive_rate_at_settle": round(survive, 4),
            "first_fail_time_mean": round(float(
                first_fail_t[fell].mean()) if bool(fell.any()) else -1.0, 2),
        },
        "support": {
            "req_loaded_frac": round(late["req"] / nl, 3),
            "not_forbidden_frac": round(late["forb"] / nl, 3),
            "load_share_frac": round(late["share"] / nl, 3),
        },
        "certificate": {
            "arrival_rate": round(float(arrived.float().mean()), 4),
            "dwell_s": float(gate["dwell_s"]),
        },
        "recovery": {"recovery_rate": round(recovery_rate, 4), "n_states": n_keep,
                     "hold_seconds": args.hold_seconds},
        "pose": {"pose_mean_med": round((late["pd_sum"] or 0.0) / nl, 4),
                 "worst_bodies": worst},
        "style": {"max_support_z_delta_cm": round(max_sup_dz, 1),
                  "max_extension_delta_cm": round(max_ext, 1),
                  "support_z_delta_cm": support_z,
                  "extension_delta_cm": ext_delta,
                  "style_agnostic_excluded": sorted(agnostic_bodies)},
        "gait": {
            "loaded_slide_speed_mean": round(
                late["slide_sum"] / max(late["slide_n"], 1), 3),
            "contact_toggle_hz": round(
                late["toggles"] / max(late["n"] * dt * len(contact_ids), 1e-6), 2),
            "com_jerk_mean": round(jerk_sum / max(jerk_n, 1), 2),
        },
        "balance": {
            "margin_start_late": round(margin_start, 4),
            "margin_end": round(margin_end, 4),
            "margin_neg_frac_late": round(margin_neg_frac, 3),
        },
    }

    router = yaml.safe_load(open(args.router))
    sig = route_repair.compute_signature(channels, router["thresholds"])
    history = route_repair.load_history(
        os.path.join(args.edge_dir, "repair_history.json"))
    verdict = route_repair.route(sig, router, history)

    cert = {"edge": os.path.basename(args.edge_dir),
            "target_node": os.path.basename(B),
            "n_envs": N, "noise": args.noise,
            "channels": channels, "signature": sig, "verdict": verdict}
    cert_path = os.path.join(out_dir, "bridge_cert.json")
    with open(cert_path, "w") as fh:
        json.dump(cert, fh, indent=2)
    with open(os.path.join(out_dir, "route_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)

    print(json.dumps({"signature": sig,
                      "verdict": {k: verdict[k] for k in
                                  ("rung", "action", "target", "terminal")}},
                     indent=2))
    print(f"cert -> {cert_path}")
    print("VIEW (visual sign-off, needs a display):\n  "
          f"{PY} mimickit/run.py --mode test --visualize --num_envs 2 "
          f"--env_config {args.edge_dir}/env_config.yaml "
          f"--agent_config {args.edge_dir}/agent_config.yaml "
          f"--model_file {args.edge_dir}/{args.edge_model}")


if __name__ == "__main__":
    main()
