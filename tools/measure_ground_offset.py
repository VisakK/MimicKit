"""Offline ground-clearance diagnostic for a motion clip.

Loads a clip through the SAME MotionLib + char_geoms + forward_kinematics path
the env uses at reset, then reports, per frame, the lowest world-z over all
collision geoms and WHICH body owns it. This is the kinematic ground truth for
choosing a per-clip `ground_offset` (manual constant lift) instead of trusting
`auto_ground_offset`, which lifts by the *global* min over the whole clip and so
can float the hold (Yoga attempt-2 post-mortem section 7).

No Isaac sim is booted (run.py --mode test never terminates); this is pure
torch + the kinematic char model, so it runs in a couple seconds on CPU.

Usage:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/measure_ground_offset.py \
      --motion_file data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_-a \
      --char_file data/assets/smpl/smpl_boxhands.xml \
      [--hold_start S --hold_end S] [--plot out.png]
"""
import argparse
import os
import sys

# Mirror run.py's import resolution (it runs as mimickit/run.py, so mimickit/ is
# sys.path[0] and modules import as `anim.*`, `util.*`).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))

import numpy as np
import torch

import anim.char_geoms as char_geoms
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import util.torch_util as torch_util


def per_body_min_geom_z(geoms, body_pos, body_rot):
    """Like char_geoms.compute_min_geom_z but keeps the per-body breakdown.

    Returns [N, B] lowest world-z per body (inf where a body has no geom)."""
    n, b_count = body_pos.shape[0], body_pos.shape[1]
    per_body = torch.full([n, b_count], float("inf"), dtype=body_pos.dtype)
    for b, body_geoms in enumerate(geoms):
        if len(body_geoms) == 0:
            continue
        b_pos = body_pos[:, b, :]
        b_rot = body_rot[:, b, :]
        body_min = torch.full([n], float("inf"), dtype=body_pos.dtype)
        for geom in body_geoms:
            pts = geom["points"]
            m = pts.shape[0]
            rot = b_rot.unsqueeze(1).expand(n, m, 4).reshape(-1, 4)
            p = pts.unsqueeze(0).expand(n, m, 3).reshape(-1, 3)
            world = torch_util.quat_rotate(rot, p).reshape(n, m, 3) + b_pos.unsqueeze(1)
            gmin = torch.min(world[..., 2], dim=1)[0] - geom["radius"]
            body_min = torch.minimum(body_min, gmin)
        per_body[:, b] = body_min
    return per_body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_file", required=True)
    p.add_argument("--char_file", default="data/assets/smpl/smpl_boxhands.xml")
    p.add_argument("--hold_start", type=float, default=None)
    p.add_argument("--hold_end", type=float, default=None)
    p.add_argument("--clearance", type=float, default=0.0,
                   help="target z of the deepest support geom after offset (m)")
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    device = "cpu"
    kin = kin_char_model.KinCharModel(device)
    kin.load_char_file(args.char_file)
    body_names = kin.get_body_names()

    # Raw clip: no offset applied, so the stored frames are the converter output.
    mlib = motion_lib.MotionLib(motion_file=args.motion_file,
                                kin_char_model=kin, device=device,
                                auto_ground_offset=False, ground_offset=0.0,
                                char_file=args.char_file)
    geoms = char_geoms.load_char_geoms(args.char_file, body_names, device)

    root_pos = mlib._frame_root_pos
    root_rot = mlib._frame_root_rot
    joint_rot = mlib._frame_joint_rot
    n_frames = root_pos.shape[0]
    fps = mlib._motion_fps[0].item()
    length = mlib._motion_lengths[0].item()
    times = np.arange(n_frames) / fps

    body_pos, body_rot = kin.forward_kinematics(root_pos, root_rot, joint_rot)
    per_body = per_body_min_geom_z(geoms, body_pos, body_rot)   # [N, B]
    min_z = torch.min(per_body, dim=1)[0].numpy()               # [N]
    low_body = torch.argmin(per_body, dim=1).numpy()            # [N]

    # Root up-vector z (pose orientation cue; ~ -1 inverted, ~ +1 upright).
    up = torch_util.quat_rotate(root_rot, torch.tensor([0.0, 0.0, 1.0]).expand(n_frames, 3))
    up_z = up[:, 2].numpy()

    g_min = float(min_z.min())
    g_argf = int(min_z.argmin())
    off_global = max(0.0, args.clearance - g_min)

    print("=" * 78)
    print("Clip : {}".format(os.path.basename(args.motion_file)))
    print("Char : {}".format(os.path.basename(args.char_file)))
    print("frames={}  fps={:.2f}  length={:.2f}s".format(n_frames, fps, length))
    print("-" * 78)
    print("RAW (no offset) lowest collision-geom z over the whole clip:")
    print("  global min  = {:+.4f} m  @ t={:.2f}s (frame {})  body='{}'".format(
        g_min, times[g_argf], g_argf, body_names[low_body[g_argf]]))
    print("  frame 0     = {:+.4f} m  body='{}'  (root up_z={:+.2f})".format(
        float(min_z[0]), body_names[low_body[0]], float(up_z[0])))
    print("  => auto_ground_offset would lift the WHOLE clip by {:+.4f} m".format(off_global))

    if args.hold_start is not None and args.hold_end is not None:
        m = (times >= args.hold_start) & (times <= args.hold_end)
        hz = min_z[m]
        h_min = float(hz.min())
        h_argf = int(np.where(m)[0][hz.argmin()])
        off_hold = max(0.0, args.clearance - h_min)
        print("-" * 78)
        print("HOLD window [{:.2f}, {:.2f}]s ({} frames):".format(
            args.hold_start, args.hold_end, int(m.sum())))
        print("  hold min    = {:+.4f} m  @ t={:.2f}s  body='{}'".format(
            h_min, times[h_argf], body_names[low_body[h_argf]]))
        print("  hold mean   = {:+.4f} m   hold up_z mean={:+.2f}".format(
            float(hz.mean()), float(up_z[m].mean())))
        print("  => offset to seat the HOLD at clearance = {:+.4f} m".format(off_hold))
        print("  HOLD FLOAT under auto (global) offset = {:+.4f} m".format(
            (h_min + off_global) - args.clearance))

    # Which bodies are the support during the lowest 15% of frames (the contact set).
    thr = np.percentile(min_z, 15)
    support = [(body_names[i], int((low_body[(min_z <= thr)] == i).sum()))
               for i in np.unique(low_body[min_z <= thr])]
    support.sort(key=lambda kv: -kv[1])
    print("-" * 78)
    print("Lowest-body tally over the deepest 15% of frames (support candidates):")
    for name, c in support:
        print("    {:<10s} {:d} frames".format(name, c))
    print("=" * 78)

    if args.plot is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(times, min_z, lw=1.2, color="tab:blue", label="raw min geom z")
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].axhline(g_min, color="tab:red", ls="--", lw=0.8,
                      label="global min ({:+.3f})".format(g_min))
        if args.hold_start is not None:
            ax[0].axvspan(args.hold_start, args.hold_end, color="tab:green",
                          alpha=0.15, label="hold window")
        ax[0].set_ylabel("min geom z (m)")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_title("Ground clearance — {}".format(os.path.basename(args.motion_file)))
        ax[1].plot(times, up_z, lw=1.2, color="tab:purple")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_ylabel("root up_z")
        ax[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print("saved plot -> {}".format(args.plot))


if __name__ == "__main__":
    main()
