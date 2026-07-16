"""Isaac-Lab flow replay (yoga_flow_paper_plan.md §5.1).

Plays a flow's recorded per-frame kinematic states (root + dof, stitched continuous
by flow_executor.py) back through the REAL Isaac-Lab renderer, so you watch the actual
mesh character perform the whole flow in one window. This is a KINEMATIC replay (each
frame the character is set to the recorded pose, then rendered) — it reproduces exactly
the motion the flow policies produced, using Isaac Lab's renderer instead of the
matplotlib skeleton.

Isaac Lab's engine has no offscreen image capture here, so viewing needs a DISPLAY:
run this on the workstation/desktop (not headless). It opens a visualize=True window.
Use --headless_check to validate the state-setting path without a display (no render).

Idiom (verified against isaac_lab_engine.py): the setters stage into obj.data.* and
flag the obj for reset; update_sim_state() flushes those writes to the sim; render()
draws the current state and paces to real time.

Usage (repo root, env python, WITH a display):
  env_isaaclab/bin/python mimickit/skillgraph/flow_replay.py \
      --states output/yoga_flows/warrior_roundtrip/warrior_roundtrip_states.npz --loops 3
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True, help="<flow>_states.npz from flow_executor")
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--headless_check", action="store_true",
                   help="build headless + skip render (validate the set path only)")
    p.add_argument("--track", action="store_true", default=True,
                   help="camera follows the root (default on)")
    p.add_argument("--cam_offset", type=float, nargs=3, default=[3.5, -3.5, 2.2])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29810)
    p.add_argument("--rand_seed", type=int, default=42)
    args = p.parse_args()

    import torch
    import envs.env_builder as env_builder
    import util.mp_util as mp_util
    import util.util as util

    d = np.load(args.states, allow_pickle=True)
    root_pos = torch.tensor(d["root_pos"], dtype=torch.float32)
    root_rot = torch.tensor(d["root_rot"], dtype=torch.float32)
    dof_pos = torch.tensor(d["dof_pos"], dtype=torch.float32)
    control_freq = int(d["control_freq"])
    env_config = str(d["env_config"])
    T = root_pos.shape[0]
    bounds = d["stage_bounds"] if "stage_bounds" in d else None
    visualize = not args.headless_check

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)
    env = env_builder.build_env(env_config, 1, device, visualize=visualize)
    e = env._engine
    cid = env._get_char_id()
    ndof = dof_pos.shape[1]
    z3 = torch.zeros(1, 3, device=device)
    zdof = torch.zeros(1, ndof, device=device)

    def stage_at(t):
        if bounds is None:
            return ""
        for s, en, nm in bounds:
            if int(s) <= t < int(en):
                return nm
        return ""

    n = args.loops if not args.headless_check else 1
    n_frames = T if not args.headless_check else min(T, 30)
    print(f"[flow_replay] {T} frames @ {control_freq}fps, {n} loop(s), "
          f"visualize={visualize}", flush=True)
    for loop in range(n):
        prev_stage = None
        for t in range(n_frames):
            rp = root_pos[t:t + 1].to(device)
            e.set_root_pos(None, cid, rp)
            e.set_root_rot(None, cid, root_rot[t:t + 1].to(device))
            e.set_dof_pos(None, cid, dof_pos[t:t + 1].to(device))
            e.set_root_vel(None, cid, z3)
            e.set_root_ang_vel(None, cid, z3)
            e.set_dof_vel(None, cid, zdof)
            e.update_sim_state()
            if visualize:
                if args.track:
                    r = rp[0].cpu().numpy()
                    look = np.array([r[0], r[1], 1.0])
                    e.update_camera(look + np.array(args.cam_offset), look)
                e.render()
                st = stage_at(t)
                if st != prev_stage:
                    print(f"  loop {loop} t={t/control_freq:5.2f}s  stage: {st}", flush=True)
                    prev_stage = st
    print("[flow_replay] done", flush=True)


if __name__ == "__main__":
    main()
