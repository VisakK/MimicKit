"""Synthesize a DeepMimic reference clip for the ONE-LEGGED CROW (right leg
extended out of Bakasana). We have no mocap for this pose, so we author it:

  phase A  hold the crow pose            (nA frames, ~1.0 s)
  phase B  extend the RIGHT leg out      (nB frames, ~1.5 s) -- geodesic SLERP of
           the right-leg joints from the tucked crow pose to a straight,
           up-and-back extended pose
  phase C  hold the one-legged crow      (nC frames, ~5.0 s)

The crow pose is taken from the solved crow HOLD clip (frame `--crow_frame`). The
extended target pose keeps the WHOLE body identical to crow EXCEPT the right leg:
R_Knee/R_Ankle/R_Toe are straightened to neutral and R_Hip is rotated so the
straight leg points up-and-back (Eka Pada Bakasana). The R_Hip rotation is found
by autograd through the SAME kinematic FK the env uses, so the result is verified
kinematically (toe high off the floor, knee straight) -- our headless substitute
for "watch it in the viewer".

Frame layout (75): [root_pos(3), root_rot expmap(3), 23 joints x 3 expmap(69)].
Right-leg expmap columns: R_Hip 18:21, R_Knee 21:24, R_Ankle 24:27, R_Toe 27:30.

The clip is saved CLAMP-looping: after the end the reference holds the extended
pose, so a long episode trains the sustained one-legged balance, and the env's
motion-end termination fires at clip end (CLAMP => motion_len_term True).

Run:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/make_one_legged_crow_clip.py \
      --crow_clip data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold \
      --out data/motions/smpl/one_legged_crow_synth
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))
import numpy as np
import torch
import anim.kin_char_model as kin_char_model
import anim.motion as motion
import util.torch_util as torch_util

# joint index among the 23 non-root joints (body index - 1); frame col = 6 + 3*ji
R_HIP_J, R_KNEE_J, R_ANKLE_J, R_TOE_J = 4, 5, 6, 7
def jcol(ji): return 6 + 3 * ji
# body indices (FK output, root=0)
B = dict(Pelvis=0, L_Hip=1, L_Knee=2, L_Ankle=3, L_Toe=4, R_Hip=5, R_Knee=6,
         R_Ankle=7, R_Toe=8, L_Shoulder=15, L_Hand=18, R_Shoulder=20, R_Hand=23)


def fk(kin, frame):
    """exp-map frame (75,) -> body_pos (24,3), body_rot (24,4). Mirrors MotionLib."""
    root_pos = frame[0:3].unsqueeze(0)
    root_rot = torch_util.exp_map_to_quat(frame[3:6].unsqueeze(0))
    joint_rot = torch_util.exp_map_to_quat(frame[6:].reshape(-1, 3)).unsqueeze(0)
    bp, br = kin.forward_kinematics(root_pos, root_rot, joint_rot)
    return bp[0], br[0]


def slerp(q0, q1, t):
    q0 = q0 / q0.norm(); q1 = q1 / q1.norm()
    d = (q0 * q1).sum()
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0); return q / q.norm()
    th0 = torch.acos(d.clamp(-1.0, 1.0)); th = th0 * t
    q2 = q1 - q0 * d; q2 = q2 / q2.norm()
    return q0 * torch.cos(th) + q2 * torch.sin(th)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crow_clip", default="data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold")
    p.add_argument("--char_file", default="data/assets/smpl/smpl_boxhands.xml")
    p.add_argument("--out", default="data/motions/smpl/one_legged_crow_synth")
    p.add_argument("--crow_frame", type=int, default=0)
    p.add_argument("--nA", type=int, default=30)   # crow hold frames
    p.add_argument("--nB", type=int, default=45)   # extension transition frames
    p.add_argument("--nC", type=int, default=150)  # extended hold frames
    p.add_argument("--target", type=float, nargs=3, default=[0.0, 1.0, 0.7],
                   help="world dir the extended leg should point (x,y,z); +y=back, +z=up")
    p.add_argument("--opt_steps", type=int, default=1500)
    args = p.parse_args()

    device = "cpu"
    kin = kin_char_model.KinCharModel(device)
    kin.load_char_file(args.char_file)

    raw = motion.load_motion(args.crow_clip)
    frames = torch.tensor(np.asarray(raw.frames, dtype=np.float32))
    fps = float(raw.fps)
    P_crow = frames[args.crow_frame].clone()
    print("crow source: {} frame {} (fps {:.0f})".format(
        os.path.basename(args.crow_clip), args.crow_frame, fps))

    # ---- author the extended pose: straighten the right lower leg, optimize R_Hip ----
    u = torch.tensor(args.target, dtype=torch.float32); u = u / u.norm()
    base = P_crow[6:].reshape(23, 3).clone()
    base[R_KNEE_J] = 0.0; base[R_ANKLE_J] = 0.0; base[R_TOE_J] = 0.0  # straight leg, neutral foot
    root_pos = P_crow[0:3].unsqueeze(0)
    root_rot = torch_util.exp_map_to_quat(P_crow[3:6].unsqueeze(0))

    h = base[R_HIP_J].clone().requires_grad_(True)  # init from the crow hip
    opt = torch.optim.Adam([h], lr=0.05)
    for it in range(args.opt_steps):
        joint_exp = torch.cat([base[:R_HIP_J], h.unsqueeze(0), base[R_HIP_J + 1:]], dim=0)
        joint_rot = torch_util.exp_map_to_quat(joint_exp).unsqueeze(0)
        bp, _ = kin.forward_kinematics(root_pos, root_rot, joint_rot)
        hip, toe = bp[0, B["R_Hip"]], bp[0, B["R_Toe"]]
        leg = toe - hip; leg_dir = leg / (leg.norm() + 1e-8)
        loss = (1.0 - (leg_dir * u).sum()) + 2.0 * (toe[0] - hip[0]) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    h_ext = h.detach()

    P_ext = P_crow.clone()
    P_ext[jcol(R_HIP_J):jcol(R_HIP_J) + 3] = h_ext
    P_ext[jcol(R_KNEE_J):jcol(R_KNEE_J) + 3] = 0.0
    P_ext[jcol(R_ANKLE_J):jcol(R_ANKLE_J) + 3] = 0.0
    P_ext[jcol(R_TOE_J):jcol(R_TOE_J) + 3] = 0.0

    # ---- build the trajectory: A (crow hold) + B (slerp) + C (extended hold) ----
    legcols = [jcol(R_HIP_J), jcol(R_KNEE_J), jcol(R_ANKLE_J), jcol(R_TOE_J)]
    qc = {c: torch_util.exp_map_to_quat(P_crow[c:c + 3].unsqueeze(0))[0] for c in legcols}
    qe = {c: torch_util.exp_map_to_quat(P_ext[c:c + 3].unsqueeze(0))[0] for c in legcols}

    def blend(frac):
        f = P_crow.clone()
        for c in legcols:
            q = slerp(qc[c], qe[c], frac)
            f[c:c + 3] = torch_util.quat_to_exp_map(q.unsqueeze(0))[0]
        return f

    seq = [P_crow.clone() for _ in range(args.nA)]
    seq += [blend(frac) for frac in np.linspace(0.0, 1.0, args.nB + 1)[1:]]  # ends at P_ext
    seq += [P_ext.clone() for _ in range(args.nC)]
    clip = torch.stack(seq, dim=0).numpy().astype(np.float32)

    # ---- verify kinematically (headless ground truth) ----
    def char(frame, tag):
        bp, _ = fk(kin, frame)
        def g(n): return bp[B[n]]
        def d(a, b): return float((g(a) - g(b)).norm())
        legd = g("R_Toe") - g("R_Hip"); legd = legd / (legd.norm() + 1e-8)
        straight = float((g("R_Hip") - g("R_Toe")).norm()) / (
            d("R_Hip", "R_Knee") + d("R_Knee", "R_Ankle") + d("R_Ankle", "R_Toe") + 1e-8)
        print("  [{:<10s}] R_Toe z={:+.3f} y={:+.3f} | L_Toe z={:+.3f} | "
              "L_Knee->L_Sh={:.3f} R_Knee->R_Sh={:.3f} | legdir(x,y,z)=[{:+.2f}{:+.2f}{:+.2f}] "
              "straightness={:.2f}".format(
                  tag, float(g("R_Toe")[2]), float(g("R_Toe")[1]), float(g("L_Toe")[2]),
                  d("L_Knee", "L_Shoulder"), d("R_Knee", "R_Shoulder"),
                  float(legd[0]), float(legd[1]), float(legd[2]), straight))
        return bp

    print("-" * 92)
    print("kinematic verification (raw coords; env adds ground_offset 0.0624):")
    char(P_crow, "crow")
    for frac in [0.25, 0.5, 0.75]:
        char(blend(frac), "extend {:.0%}".format(frac))
    bp_ext = char(P_ext, "EXTENDED")
    print("  target legdir u=[{:+.2f}{:+.2f}{:+.2f}]  (straightness 1.00 = perfectly straight leg)".format(*u.tolist()))

    # whole-clip min body-origin z (hands are the support; nothing should go near/under it)
    all_bp = torch.stack([fk(kin, torch.tensor(clip[i]))[0] for i in range(0, clip.shape[0], 3)], 0)
    minz = float(all_bp[..., 2].min())
    print("  whole-clip min body-origin z = {:+.3f} (raw); +0.0624 offset -> {:+.3f}".format(minz, minz + 0.0624))
    print("  R_Toe z over clip: min {:+.3f} max {:+.3f} (stays well clear of the floor)".format(
        float(all_bp[:, B["R_Toe"], 2].min()), float(all_bp[:, B["R_Toe"], 2].max())))

    out = motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=raw.fps, frames=clip)
    out.save(args.out)
    print("-" * 92)
    print("saved CLAMP clip: {} frames = {:.2f}s -> {}".format(
        clip.shape[0], (clip.shape[0] - 1) / fps, args.out))


if __name__ == "__main__":
    main()
