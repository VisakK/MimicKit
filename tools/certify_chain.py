"""Chain Certifier — the wavefront pass (Yoga_orchestration_protocol.md §2.5).

Per-edge certification composes UNSOUNDLY: the plank_vinyasa exit cascaded
through seams that each passed their LOCAL cert (canonical starts). This tool
walks a composed flow FRONT-TO-BACK and certifies every seam on its
predecessor's HARVESTED DELIVERED DISTRIBUTION:

  edge stage : tools/certify_bridge.py --init_states <prev distribution>
               -> full bridge cert + routed rung on the states the chain
               actually delivers; its surviving cert_arrivals.pt becomes the
               next distribution.
  node stage : tools/takeover_oracle.py on the delivered distribution
               (vel_scale = the executor's settle-and-hold handoff) with
               --save_end_states -> dwell pass rate is the seam verdict and
               the post-dwell survivors propagate.

Stage 0 runs canonical t=0 (that IS the flow's entry contract). The wave HALTS
when fewer than chain.min_states survivors propagate (the cascade is measured,
not imputed); otherwise failed seams are recorded and the wave continues, so
one pass maps EVERY unsound seam. Thresholds live ONLY in
skills/repair_router.yaml (chain: section).

A flow is chain-certified iff every edge seam routes R10_certified_local AND
every node dwell clears chain.node_dwell_pass_min — then it still needs visual
sign-off (flow_executor render), per the protocol.

Usage (repo root, env python, env -u DISPLAY; ONE Isaac job at a time —
stages run sequentially in subprocesses):
  env -u DISPLAY /home/visakii/Documents/moves/env_isaaclab/bin/python \
      tools/certify_chain.py --flow data/flows/plank_vinyasa.yaml \
      --out_dir output/yoga_flows/plank_vinyasa/chain_cert
Output: <out_dir>/chain_cert.json + per-stage subdirs (bridge certs, oracle
verdicts, propagated state blobs).
"""
import argparse
import json
import os
import subprocess
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
PY = sys.executable

NODE_ROOT = "output/yoga_nodes_v2"
PASS_RUNG = "R10_certified_local"


def resolve_target_node(stage):
    """Destination node dir for an edge stage: explicit target_node wins, else
    the '<X>_to_<Y>' label convention -> output/yoga_nodes_v2/<Y>_lt_ampft."""
    if stage.get("target_node"):
        return stage["target_node"]
    label = stage["label"]
    if "_to_" not in label:
        return None
    dest = label.split("_to_")[-1]
    return os.path.join(NODE_ROOT, f"{dest}_lt_ampft")


def load_flow(path):
    flow = yaml.safe_load(open(path))
    name = flow.get("name", os.path.splitext(os.path.basename(path))[0])
    stages = []
    for st in flow["stages"]:
        stype = st.get("type", "edge")
        label = st.get("edge") or st.get("node") or os.path.basename(st["dir"])
        s = {"type": stype, "label": label, "dir": st["dir"],
             "model": st.get("model", "model.pt"),
             "target_node": st.get("target_node"),
             # R2b gate-switch semantics: certify this edge at its gate-fire
             # time instead of the solo settle (approved row 2026-07-18)
             "cert_settle_at": st.get("cert_settle_at")}
        if stype == "node":
            s["dwell_seconds"] = float(st.get("dwell_seconds", 4.0))
        stages.append(s)
    return name, stages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", required=True, help="flow yaml (flow_executor schema)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--router", default=os.path.join(REPO, "skills", "repair_router.yaml"))
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--start_stage", type=int, default=0,
                   help="resume a partial pass: skip stages < this index "
                        "(their subdirs must already hold results)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29700)
    p.add_argument("--rand_seed", type=int, default=42)
    args = p.parse_args()

    name, stages = load_flow(args.flow)
    ch_th = yaml.safe_load(open(args.router)).get("chain", {})
    pass_min = float(ch_th.get("node_dwell_pass_min", 0.70))
    min_states = int(ch_th.get("min_states", 8))
    dwell_vs = float(ch_th.get("node_dwell_vel_scale", 0.0))
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[chain:{name}] {len(stages)} stages -> {args.out_dir}", flush=True)
    report = {"flow": name, "flow_yaml": args.flow, "router": args.router,
              "thresholds": {"node_dwell_pass_min": pass_min,
                             "min_states": min_states,
                             "node_dwell_vel_scale": dwell_vs},
              "stages": [], "chain_pass": False, "first_failed_seam": None,
              "halted_at": None}

    dist = None          # path to the current delivered distribution (.pt)
    n_dist = None        # states in it
    for i, st in enumerate(stages):
        label, stype = st["label"], st["type"]
        stage_dir = os.path.join(args.out_dir, f"stage{i}_{label}")
        os.makedirs(stage_dir, exist_ok=True)
        port = args.master_port + 10 * i
        entry = {"i": i, "label": label, "type": stype,
                 "n_states_in": n_dist if dist else args.num_envs,
                 "chained": bool(dist)}

        if i < args.start_stage:
            # resume: reload this stage's prior result to re-thread the wave
            if stype == "edge":
                cert = json.load(open(os.path.join(stage_dir, "bridge_cert.json")))
                entry["rung"] = cert["verdict"]["rung"]
                entry["seam_pass"] = entry["rung"] == PASS_RUNG
                dist = os.path.join(stage_dir, "cert_arrivals.pt")
            else:
                orc = torch.load(os.path.join(stage_dir, "dwell_oracle.pt"),
                                 map_location="cpu")
                rate = float(orc["verdict"].float().mean())
                entry["dwell_pass_rate"] = round(rate, 4)
                entry["seam_pass"] = rate >= pass_min
                dist = os.path.join(stage_dir, "dwell_end_states.pt")
            n_dist = int(torch.load(dist, map_location="cpu")["root_pos"].shape[0])
            entry["n_states_out"] = n_dist
            entry["resumed"] = True
            report["stages"].append(entry)
            print(f"[chain:{name}] stage {i} {label}: resumed "
                  f"({n_dist} states)", flush=True)
            continue

        if stype == "edge":
            node_dir = resolve_target_node(st)
            if node_dir is None or not os.path.exists(
                    os.path.join(node_dir, "gate.yaml")):
                entry.update(seam_pass=False, rung="NO_JUDGE",
                             error=f"no destination gate at {node_dir}")
                report["stages"].append(entry)
                report["halted_at"] = i
                print(f"[chain:{name}] stage {i} {label}: NO JUDGE "
                      f"({node_dir}) — halt", flush=True)
                break
            entry["target_node"] = node_dir
            cmd = [PY, "tools/certify_bridge.py",
                   "--edge_dir", st["dir"], "--edge_model", st["model"],
                   "--target_node_dir", node_dir, "--out_dir", stage_dir,
                   "--num_envs", str(args.num_envs), "--noise", str(args.noise),
                   "--device", args.device, "--master_port", str(port),
                   "--rand_seed", str(args.rand_seed)]
            if dist:
                cmd += ["--init_states", dist]
            if st.get("cert_settle_at") is not None:
                cmd += ["--settle_at", str(st["cert_settle_at"])]
                entry["cert_settle_at"] = st["cert_settle_at"]
            print(f"[chain:{name}] stage {i} {label} (edge -> "
                  f"{os.path.basename(node_dir)})"
                  + (f" ON {n_dist} delivered states" if dist else
                     " from canonical t=0"), flush=True)
            r = subprocess.run(cmd, cwd=REPO)
            if r.returncode != 0:
                entry.update(seam_pass=False, rung="CRASHED", rc=r.returncode)
                report["stages"].append(entry)
                report["halted_at"] = i
                break
            cert = json.load(open(os.path.join(stage_dir, "bridge_cert.json")))
            entry["rung"] = cert["verdict"]["rung"]
            entry["seam_pass"] = entry["rung"] == PASS_RUNG
            entry["signature"] = cert["signature"]
            entry["key_channels"] = {
                "survive": cert["channels"]["reachability"]["survive_rate_at_settle"],
                "arrival_rate": cert["channels"]["certificate"]["arrival_rate"],
                "recovery_rate": cert["channels"]["recovery"]["recovery_rate"],
                "pose_mean_med": cert["channels"]["pose"]["pose_mean_med"]}
            nxt = os.path.join(stage_dir, "cert_arrivals.pt")
        else:
            if dist is None:
                raise SystemExit(f"stage {i} ({label}): node dwell cannot be "
                                 "the first stage of a chain cert")
            nxt = os.path.join(stage_dir, "dwell_end_states.pt")
            print(f"[chain:{name}] stage {i} {label} (node dwell "
                  f"{st['dwell_seconds']}s) ON {n_dist} delivered states",
                  flush=True)
            r = subprocess.run(
                [PY, "tools/takeover_oracle.py", "--node_dir", st["dir"],
                 "--states", dist, "--hold_seconds", str(st["dwell_seconds"]),
                 "--vel_scale", str(dwell_vs), "--save_end_states", nxt,
                 "--num_envs", str(max(args.num_envs, 128)),
                 "--master_port", str(port), "--rand_seed", str(args.rand_seed),
                 "--out", os.path.join(stage_dir, "dwell_oracle.pt")],
                cwd=REPO)
            if r.returncode != 0:
                entry.update(seam_pass=False, rung="CRASHED", rc=r.returncode)
                report["stages"].append(entry)
                report["halted_at"] = i
                break
            orc = torch.load(os.path.join(stage_dir, "dwell_oracle.pt"),
                             map_location="cpu")
            rate = float(orc["verdict"].float().mean())
            entry["dwell_pass_rate"] = round(rate, 4)
            entry["seam_pass"] = rate >= pass_min

        if not os.path.exists(nxt):
            entry.update(n_states_out=0)
            report["stages"].append(entry)
            report["halted_at"] = i
            print(f"[chain:{name}] stage {i} {label}: zero survivors — "
                  "wave halts (cascade measured)", flush=True)
            break
        n_out = int(torch.load(nxt, map_location="cpu")["root_pos"].shape[0])
        entry["n_states_out"] = n_out
        report["stages"].append(entry)
        print(f"[chain:{name}] stage {i} {label}: "
              f"{'PASS' if entry['seam_pass'] else 'FAIL(' + entry.get('rung', 'dwell') + ')'}"
              f", {n_out} states propagate", flush=True)
        if n_out < min_states:
            report["halted_at"] = i
            print(f"[chain:{name}] {n_out} < min_states {min_states} — "
                  "wave halts (cascade measured)", flush=True)
            break
        dist, n_dist = nxt, n_out

    fails = [s for s in report["stages"] if not s.get("seam_pass")]
    report["first_failed_seam"] = fails[0]["label"] if fails else None
    report["chain_pass"] = (not fails and report["halted_at"] is None
                            and len(report["stages"]) == len(stages))
    out_path = os.path.join(args.out_dir, "chain_cert.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n[chain:{name}] " + ("CHAIN CERTIFIED (pending visual sign-off)"
          if report["chain_pass"] else
          f"NOT certified — first failed seam: {report['first_failed_seam']}"
          + (f", halted at stage {report['halted_at']}"
             if report["halted_at"] is not None else "")))
    for s in report["stages"]:
        mark = "PASS" if s.get("seam_pass") else "FAIL"
        extra = s.get("rung") or f"dwell {s.get('dwell_pass_rate')}"
        print(f"  stage {s['i']:>2} {mark}  {s['label']:<28} {extra:<24} "
              f"in={s.get('n_states_in')} out={s.get('n_states_out')}")
    print(f"chain cert -> {out_path}")


if __name__ == "__main__":
    main()
