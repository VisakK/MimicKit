"""Offline inspection for the one-legged-crow setup (no Isaac boot).

(1) Loads the crow HOLD clip, prints its shape/fps and, for a chosen frame, the
    L/R leg exp-map DoFs (with angle magnitudes) and a forward-kinematics readout
    of the key bodies (toe heights, knee->shoulder distances, hand geometry).
    This characterizes the canonical crow pose so we can author the extended pose.

(2) Loads an AMP checkpoint and prints its state_dict keys grouped by prefix +
    tensor shapes, so we can plan the disc-stripping surgery for a PPO warm-start.

Run:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/inspect_crow_assets.py \
      --motion_file data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold \
      --char_file data/assets/smpl/smpl_boxhands.xml \
      --ckpt output/model_yoga_amp_crow_hybrid_ft4_lowlr.pt --frame 0
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))
import numpy as np
import torch
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import anim.motion as motion
import util.torch_util as torch_util

LEG = [("L_Hip", 6), ("L_Knee", 9), ("L_Ankle", 12), ("L_Toe", 15),
       ("R_Hip", 18), ("R_Knee", 21), ("R_Ankle", 24), ("R_Toe", 27)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_file", default="data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold")
    p.add_argument("--char_file", default="data/assets/smpl/smpl_boxhands.xml")
    p.add_argument("--ckpt", default="output/model_yoga_amp_crow_hybrid_ft4_lowlr.pt")
    p.add_argument("--frame", type=int, default=0)
    args = p.parse_args()

    device = "cpu"
    kin = kin_char_model.KinCharModel(device)
    kin.load_char_file(args.char_file)
    body_names = kin.get_body_names()

    # --- (1) clip + crow pose ---
    raw = motion.load_motion(args.motion_file)
    frames = np.asarray(raw.frames, dtype=np.float32)
    print("=" * 78)
    print("CLIP {}  frames={} dim={} fps={} loop={}".format(
        os.path.basename(args.motion_file), frames.shape[0], frames.shape[1],
        raw.fps, raw.loop_mode))

    mlib = motion_lib.MotionLib(motion_file=args.motion_file, kin_char_model=kin,
                                device=device, auto_ground_offset=False,
                                ground_offset=0.0, char_file=args.char_file)
    rp, rr, jr = mlib._frame_root_pos, mlib._frame_root_rot, mlib._frame_joint_rot
    body_pos, body_rot = kin.forward_kinematics(rp, rr, jr)  # [N,B,3]
    bn = {n: i for i, n in enumerate(body_names)}

    f = args.frame
    print("-" * 78)
    print("FRAME {} root_pos={}".format(f, np.round(frames[f, 0:3], 4)))
    print("leg exp-map DoFs (col: [x y z]  |angle|deg):")
    for name, c in LEG:
        v = frames[f, c:c + 3]
        ang = np.degrees(np.linalg.norm(v))
        print("  {:<8s} c{:>2d}: [{:+.3f} {:+.3f} {:+.3f}]  {:6.1f} deg".format(
            name, c, v[0], v[1], v[2], ang))

    print("-" * 78)
    print("FK body positions at frame {} (x y z):".format(f))
    for name in ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip",
                 "R_Knee", "R_Ankle", "R_Toe", "L_Shoulder", "R_Shoulder",
                 "L_Hand", "R_Hand"]:
        pos = body_pos[f, bn[name]].numpy()
        print("  {:<11s} [{:+.3f} {:+.3f} {:+.3f}]".format(name, *pos))

    # crow characterization
    def d(a, b):
        return float(torch.linalg.vector_norm(body_pos[f, bn[a]] - body_pos[f, bn[b]]))
    print("-" * 78)
    print("crow characterization @frame {}:".format(f))
    print("  L_Toe z={:+.3f}  R_Toe z={:+.3f}".format(
        float(body_pos[f, bn['L_Toe'], 2]), float(body_pos[f, bn['R_Toe'], 2])))
    print("  L_Knee->L_Shoulder={:.3f}  R_Knee->R_Shoulder={:.3f}".format(
        d('L_Knee', 'L_Shoulder'), d('R_Knee', 'R_Shoulder')))
    print("  R_Hip->R_Knee={:.3f}  R_Knee->R_Ankle={:.3f}  R_Hip->R_Ankle={:.3f} (straight=sum)".format(
        d('R_Hip', 'R_Knee'), d('R_Knee', 'R_Ankle'), d('R_Hip', 'R_Ankle')))
    # left/forward geometry for the CoP "lean left" sign check
    root_xy = body_pos[f, 0, :2]
    hands_xy = 0.5 * (body_pos[f, bn['L_Hand'], :2] + body_pos[f, bn['R_Hand'], :2])
    fwd = (hands_xy - root_xy); fwd = fwd / (torch.linalg.vector_norm(fwd) + 1e-6)
    left = torch.tensor([-fwd[1], fwd[0]])  # cross(up=[0,0,1], fwd)
    lr_vec = (body_pos[f, bn['L_Hand'], :2] - body_pos[f, bn['R_Hand'], :2])
    print("  fwd(root->hands)_xy={}  left=cross(up,fwd)_xy={}".format(
        np.round(fwd.numpy(), 3), np.round(left.numpy(), 3)))
    print("  (L_Hand-R_Hand)_xy={}  dot(left, L-R)={:+.3f}  -> {}".format(
        np.round(lr_vec.numpy(), 3), float(torch.dot(left, lr_vec)),
        "left points toward L_Hand (CORRECT)" if float(torch.dot(left, lr_vec)) > 0
        else "left points toward R_Hand (FLIP SIGN)"))

    # toe-height spread across all frames (confirm both toes stay up in the hold)
    lt = body_pos[:, bn['L_Toe'], 2].numpy(); rt = body_pos[:, bn['R_Toe'], 2].numpy()
    print("-" * 78)
    print("across {} frames: L_Toe z [{:+.3f},{:+.3f}] mean {:+.3f} | R_Toe z [{:+.3f},{:+.3f}] mean {:+.3f}".format(
        frames.shape[0], lt.min(), lt.max(), lt.mean(), rt.min(), rt.max(), rt.mean()))

    # --- (2) checkpoint keys ---
    if args.ckpt and os.path.exists(args.ckpt):
        print("=" * 78)
        print("CKPT {}".format(args.ckpt))
        sd = torch.load(args.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and not any(
                hasattr(v, "shape") for v in list(sd.values())[:3]):
            sd = sd["state_dict"]
        groups = {}
        for k, v in sd.items():
            top = k.split(".")[0]
            groups.setdefault(top, []).append((k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__))
        for top in sorted(groups):
            keys = groups[top]
            is_disc = top.startswith("disc") or top.startswith("_disc")
            print("  [{}] {} tensors {}".format(
                top, len(keys), "<-- DISC (strip for PPO)" if is_disc else ""))
            for k, shp in keys[:3]:
                print("        {:<40s} {}".format(k, shp))
            if len(keys) > 3:
                print("        ... (+{} more)".format(len(keys) - 3))
    print("=" * 78)


if __name__ == "__main__":
    main()
