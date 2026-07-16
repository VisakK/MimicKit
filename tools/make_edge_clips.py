"""Extract an EDGE CLIP (real transition demo segment) from a source motion
clip — the segment-mimic reference of Yoga_edge_framework_v3.md §2.1.

Cuts [transition_start - lead_in, transition_end + tail] (the A-hold lead-in
and B-hold tail come from the SAME clip, so the edge clip has ONE ground
datum), bakes a per-segment ground offset measured on the segment's own frames
(robust percentile of the per-frame collision-geom minimum — the clip-global
constant is exactly the L6 wound), saves with loop CLAMP (WRAP would teleport
the reference mid-episode), and emits a pre-flight report + strip render of
the RSI frames (the attempt-2 rule: render before training).

Usage (repo root, env python; frame indices are annotation frames @ clip fps):
  env_isaaclab/bin/python tools/make_edge_clips.py \
      --motion_file data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_-b \
      --start_frame 864 --end_frame 916 --lead_in 1.0 --tail 2.5 \
      --out data/motions/smpl_edges/crow_to_tadasana
Outputs: <out> (pickled Motion, CLAMP), <out>_preflight.png, report on stdout.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "mimickit"))   # must precede tools/
os.chdir(REPO)

import anim.motion as motion
import anim.kin_char_model as kin_char_model
import anim.char_geoms as char_geoms
import annotate_clips as AC

CHAR_FILE = "data/assets/smpl/smpl_boxhands.xml"   # geoms == lowtorque variant
PCTL_GROUND = 5.0     # robust per-frame-min percentile for the baked offset
SEAT_TOL_PEN = 0.015  # m; worst allowed penetration after baking
SEAT_TOL_FLOAT = 0.03 # m; worst allowed support float during holds (advisory)


def fk_min_z(frames, kcm, geoms):
    """Per-frame lowest collision-geom z + body positions, on CPU.
    Frame layout (motion_lib.extract_pose_data): [0:3] root pos,
    [3:6] root rot exp-map, [6:] joint dof."""
    import util.torch_util as torch_util
    rp = torch.tensor(frames[:, 0:3], dtype=torch.float32)
    rr = torch_util.exp_map_to_quat(torch.tensor(frames[:, 3:6], dtype=torch.float32))
    jdof = torch.tensor(frames[:, 6:], dtype=torch.float32)
    jr = torch_util.quat_pos(kcm.dof_to_rot(jdof))
    bp, br = kcm.forward_kinematics(rp, rr, jr)
    minz, _ = AC.per_body_witness(geoms, bp, br)   # [T,B]
    return minz, bp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_file", required=True)
    p.add_argument("--start_frame", type=int, required=True,
                   help="transition start (annotation frame index)")
    p.add_argument("--end_frame", type=int, required=True,
                   help="transition end (annotation frame index)")
    p.add_argument("--lead_in", type=float, default=1.0, help="s of A-hold before")
    p.add_argument("--tail", type=float, default=2.5, help="s of B-hold after")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    m = motion.load_motion(args.motion_file)
    fps = float(m.fps)
    frames = np.asarray(m.frames, dtype=np.float32)
    N = frames.shape[0]

    i0 = max(0, args.start_frame - int(round(args.lead_in * fps)))
    i1 = min(N - 1, args.end_frame + int(round(args.tail * fps)))
    seg = frames[i0:i1 + 1].copy()
    t_trans0 = (args.start_frame - i0) / fps
    t_trans1 = (args.end_frame - i0) / fps
    print(f"source {N} frames @ {fps:.0f}fps -> edge clip frames [{i0},{i1}] "
          f"({seg.shape[0]} frames, {(seg.shape[0]-1)/fps:.2f}s); transition "
          f"window inside clip: [{t_trans0:.2f},{t_trans1:.2f}]s")

    # ---- per-segment ground offset (measured on the segment's own frames) ----
    kcm = kin_char_model.KinCharModel("cpu")
    kcm.load_char_file(CHAR_FILE)
    geoms = char_geoms.load_char_geoms(CHAR_FILE, kcm.get_body_names(), "cpu")
    minz, _ = fk_min_z(seg, kcm, geoms)
    per_frame_min = minz.min(dim=1)[0].numpy()      # [T]
    offset = -float(np.percentile(per_frame_min, PCTL_GROUND))
    seg[:, 2] += offset
    after = per_frame_min + offset
    worst_pen = max(0.0, -float(after.min()))
    # float during the bracketing holds (first lead_in + last tail second)
    k_in = int(round(args.lead_in * fps)); k_tail = int(round(args.tail * fps))
    hold_float = float(np.median(np.concatenate([after[:max(k_in, 1)],
                                                 after[-max(k_tail, 1):]])))
    print(f"baked ground offset {offset:+.4f} m "
          f"(per-frame-min {PCTL_GROUND:.0f}th pctl); after: worst penetration "
          f"{worst_pen*100:.1f} cm, median hold seat {hold_float*100:+.1f} cm")
    ok_pen = worst_pen <= SEAT_TOL_PEN
    ok_float = abs(hold_float) <= SEAT_TOL_FLOAT
    if not ok_pen:
        print(f"  !! penetration exceeds {SEAT_TOL_PEN*100:.0f} cm — inspect the render")
    if not ok_float:
        print(f"  !! hold seating floats beyond {SEAT_TOL_FLOAT*100:.0f} cm — inspect")

    # velocity continuity inside the cut (no splice exists, but report the
    # max inter-frame root jump as a retarget-noise smoke check)
    root_jump = np.linalg.norm(np.diff(seg[:, 0:3], axis=0), axis=-1).max() * fps
    print(f"max root speed inside segment: {root_jump:.2f} m/s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=m.fps, frames=seg)
    out.save(args.out)
    print(f"saved CLAMP edge clip -> {args.out}")

    # ---- pre-flight strip render (RSI frames) --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    minz2, bp = fk_min_z(seg, kcm, geoms)
    T = seg.shape[0]
    n_snap = 10
    snaps = np.linspace(0, T - 1, n_snap).astype(int)
    parents = [int(kcm.get_parent_id(i)) for i in range(len(kcm.get_body_names()))]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    bpn = bp.numpy()
    # heading-agnostic side view: dominant horizontal travel axis vs z
    span = bpn[:, :, :2].reshape(-1, 2)
    dirv = np.linalg.svd(span - span.mean(0))[2][0]
    for k, s in enumerate(snaps):
        xo = k * 1.2
        pts_h = bpn[s, :, :2] @ dirv
        pts_h = pts_h - pts_h.mean() + xo
        for b, pa in enumerate(parents):
            if pa < 0:
                continue
            ax.plot([pts_h[b], pts_h[pa]], [bpn[s, b, 2], bpn[s, pa, 2]],
                    "k-", lw=1.5)
        ax.plot(pts_h, bpn[s, :, 2], "o", ms=2.5,
                color="tab:orange" if t_trans0 <= s / fps <= t_trans1 else "tab:blue")
        ax.text(xo, -0.12, f"{s/fps:.1f}s", ha="center", fontsize=8)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_aspect("equal"); ax.set_title(
        f"edge clip {os.path.basename(args.out)} — RSI-frame pre-flight "
        f"(orange = transition window)")
    ax2 = axes[1]
    tt = np.arange(T) / fps
    ax2.plot(tt, minz2.min(dim=1)[0].numpy() * 100, lw=1)
    ax2.axhspan(-SEAT_TOL_PEN * 100, 0, color="red", alpha=0.15)
    ax2.axvspan(t_trans0, t_trans1, color="orange", alpha=0.15)
    ax2.set_xlabel("s"); ax2.set_ylabel("min geom z (cm)")
    fig.tight_layout()
    png = args.out + "_preflight.png"
    fig.savefig(png, dpi=110)
    print(f"pre-flight strip -> {png}")
    print(f"PRE-FLIGHT {'PASS' if (ok_pen and ok_float) else 'CHECK'}")


if __name__ == "__main__":
    main()
