"""On-body POSITIVE-CONTACT check for v3 ladder hold nodes (numpy only, no Isaac).

Reads the knee_support pair lists (knee_support_bodies -> the moving bodies,
knee_support_target_bodies -> the targets) from an ampft env yaml and, for each
moving body, reports the min-over-targets center distance from a rollout
telemetry.npz (tools/collect_crow_telemetry.py format: body_names,
body_pos [T, 24, 3]). This is exactly the quantity the reward_knee_support
attractor in mimickit/envs/deepmimic_env.py sees (body-origin distance, NOT
capsule-surface distance -- a seated contact typically reads 5-15 cm here, not
0), so it answers "is the press/wrap/grip actually seated?" without launching
the sim.

Frames 0..warmup_steady-1 are the RSI settle window (check_hold_node.py
collects with --warmup 10, so frame 0 is already ~0.33 s post-reset); steady
stats start at frame 15 by default.

Usage (also invoked automatically by tools/run_v3_ladder.sh -> onbody.txt):
  python tools/onbody_contact_check.py \
      --node_dir output/yoga_nodes_v2/<node>_lt_ampft \
      --env data/envs/amp_smpl_<node>_hold_lowtorque_ampft_env.yaml

Exits 0 with a message when the env has no pairs or knee_support weight 0
(most grounded nodes -- see tools/make_v3_envs.py for the per-node wiring).
"""

import argparse
import os
import sys

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", help="run dir containing telemetry.npz")
    p.add_argument("--telemetry", help="explicit path to telemetry.npz")
    p.add_argument("--env", required=True,
                   help="ampft env yaml carrying the knee_support pair lists")
    p.add_argument("--warmup_steady", type=int, default=15,
                   help="steady-window stats start at this frame (default 15)")
    args = p.parse_args()

    env = yaml.safe_load(open(args.env))["env"]
    w = float(env.get("reward_knee_support_w", 0.0))
    movers = env.get("knee_support_bodies", []) or []
    targets = env.get("knee_support_target_bodies", []) or []

    if w <= 0.0 or not movers or not targets:
        print(f"onbody: no positive-contact pairs to check in {args.env} "
              f"(reward_knee_support_w={w}, {len(movers)} movers, "
              f"{len(targets)} targets) -- nothing to do.")
        return 0

    path = args.telemetry or os.path.join(args.node_dir or "", "telemetry.npz")
    if not os.path.exists(path):
        print(f"onbody: telemetry not found: {path}", file=sys.stderr)
        return 1

    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["body_names"]]
    P = d["body_pos"]  # [T, B, 3]
    idx = {n: i for i, n in enumerate(names)}
    missing = [b for b in movers + targets if b not in idx]
    if missing:
        print(f"onbody: bodies missing from telemetry: {missing}", file=sys.stderr)
        return 1

    T = P.shape[0]
    s0 = min(args.warmup_steady, max(T - 1, 0))
    S = slice(s0, None)
    print(f"onbody: {path} ({T} frames; steady = frames {s0}+)")
    print(f"onbody: env {args.env}  w={w}  scale="
          f"{env.get('reward_knee_support_scale', 20.0)}")
    print(f"onbody: targets = {targets}")
    print(f"{'mover':10s} {'nearest(steady)':16s} {'mean cm':>8s} {'min cm':>8s} "
          f"{'std cm':>7s} {'settle-mean cm':>15s}")
    tgt_pos = np.stack([P[:, idx[t]] for t in targets], axis=1)  # [T, K, 3]
    for m in movers:
        dv = tgt_pos - P[:, idx[m]][:, None, :]        # [T, K, 3]
        dist = np.linalg.norm(dv, axis=2)              # [T, K]
        dmin = dist.min(axis=1) * 100.0                # cm, min over targets
        # which target is nearest, majority vote over the steady window
        near = np.bincount(dist[S].argmin(axis=1), minlength=len(targets)).argmax()
        settle = dmin[:s0].mean() if s0 > 0 else float("nan")
        print(f"{m:10s} {targets[near]:16s} {dmin[S].mean():8.1f} "
              f"{dmin[S].min():8.1f} {dmin[S].std():7.1f} {settle:15.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
