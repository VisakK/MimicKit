"""Trim a motion clip to its quasi-static HOLD sub-window and make it a looping
clip, so a policy can be trained to HOLD a pose (not transition into it). The hold
window is what makes a clean balance-node reference: removing the entry/exit (and,
for crow, the early [14,17]s sub-window where the LEFT toe sits at ~0.10 m, which
is what was teaching the toe-tap).

Default: trim [start,end]s, set loop_mode=WRAP. Options: --pingpong (forward then
reversed, for a seamless C0 loop) and --freeze_tail S (append S seconds of the last
frame). Saves a new pickled Motion next to the original.

Run (crow bilateral hold):
  env_isaaclab/bin/python tools/trim_hold_clip.py \
    --motion_file data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_-a \
    --out data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold \
    --start 17.0 --end 23.0
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))
import numpy as np
import anim.motion as motion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, required=True, help="hold start (s)")
    p.add_argument("--end", type=float, required=True, help="hold end (s)")
    p.add_argument("--pingpong", action="store_true",
                   help="append the reversed segment for a seamless loop")
    p.add_argument("--freeze_tail", type=float, default=0.0,
                   help="append this many seconds of the final frame")
    args = p.parse_args()

    m = motion.load_motion(args.motion_file)
    fps = float(m.fps)
    frames = np.asarray(m.frames, dtype=np.float32)
    N = frames.shape[0]

    i0 = max(0, int(round(args.start * fps)))
    i1 = min(N - 1, int(round(args.end * fps)))
    seg = frames[i0:i1 + 1].copy()
    print("source {} frames ({:.2f}s) -> window [{:.2f},{:.2f}]s = frames[{}:{}] ({} frames)"
          .format(N, (N - 1) / fps, args.start, args.end, i0, i1 + 1, seg.shape[0]))

    if args.pingpong and seg.shape[0] > 2:
        seg = np.concatenate([seg, seg[-2:0:-1]], axis=0)
        print("  pingpong -> {} frames".format(seg.shape[0]))

    if args.freeze_tail > 0:
        k = int(round(args.freeze_tail * fps))
        seg = np.concatenate([seg, np.repeat(seg[-1:], k, axis=0)], axis=0)
        print("  +{:.1f}s frozen tail ({} frames) -> {} frames".format(
            args.freeze_tail, k, seg.shape[0]))

    # seam diagnostic: how far apart are the loop endpoints (root + joints)?
    seam = float(np.linalg.norm(seg[0] - seg[-1]))
    print("  loop length = {:.2f}s   endpoint seam ||f0-f_last|| = {:.4f}".format(
        (seg.shape[0] - 1) / fps, seam))

    out = motion.Motion(loop_mode=motion.LoopMode.WRAP, fps=m.fps, frames=seg)
    out.save(args.out)
    print("saved WRAP-looping hold clip -> {}".format(args.out))


if __name__ == "__main__":
    main()
