"""Synthesize a SLOW one-legged-crow reference for AMP+DeepMimic hybrid training.

Two changes from make_one_legged_crow_clip.py, both to PRESERVE CROW STYLE under an
AMP discriminator (whose demos are drawn from this clip):

  1. NATURAL BASE. The body (arms, head, torso, left leg) uses the real crow-HOLD
     clip's frames (genuine micro-balance motion), NOT a single frozen frame. Only
     the RIGHT-leg DoFs are overwritten. So the AMP demos show natural tucked-arm,
     lifted-head crow style instead of an unnaturally static pose.
  2. SLOW extension. A long crow hold (nA) then a slow right-leg extension (nB) then
     an extended hold (nC) -- the leg eases out over ~5 s so the policy can keep the
     crow style throughout instead of flailing through a fast balance change.

The extended RIGHT-leg pose is the same FK-optimized straight-up-and-back target as
before. The overlay slerps each right-leg joint from the base clip's CURRENT (natural
crow) right leg toward the extended pose by a blend that ramps 0 over [0,nA], 0->1
over the extension nB, and 1 over the hold nC -- so it is seamless with the natural
base at the start and reaches the held extended pose at the end. Saved CLAMP.

Run:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/make_one_legged_crow_slow_clip.py \
      --hold_clip data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold \
      --out data/motions/smpl/one_legged_crow_slow
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))
import numpy as np
import torch
import anim.kin_char_model as kin_char_model
import anim.motion as motion
import util.torch_util as torch_util

R_HIP_J, R_KNEE_J, R_ANKLE_J, R_TOE_J = 4, 5, 6, 7
def jcol(ji): return 6 + 3 * ji
LEGCOLS = [jcol(R_HIP_J), jcol(R_KNEE_J), jcol(R_ANKLE_J), jcol(R_TOE_J)]
B = dict(Pelvis=0, L_Toe=4, R_Hip=5, R_Knee=6, R_Ankle=7, R_Toe=8, L_Knee=2,
         L_Shoulder=15, R_Shoulder=20, L_Hand=18, R_Hand=23)


def fk(kin, frame):
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
    p.add_argument("--hold_clip", default="data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold")
    p.add_argument("--char_file", default="data/assets/smpl/smpl_boxhands.xml")
    p.add_argument("--out", default="data/motions/smpl/one_legged_crow_slow")
    p.add_argument("--crow_frame", type=int, default=0)
    p.add_argument("--nA", type=int, default=60)    # crow hold (2.0s) -- style established first
    p.add_argument("--nB", type=int, default=150)   # SLOW extension (5.0s)
    p.add_argument("--nC", type=int, default=90)    # extended hold (3.0s)
    p.add_argument("--target", type=float, nargs=3, default=[0.0, 1.0, 0.7])
    p.add_argument("--opt_steps", type=int, default=1500)
    args = p.parse_args()

    device = "cpu"
    kin = kin_char_model.KinCharModel(device); kin.load_char_file(args.char_file)
    base = torch.tensor(np.asarray(motion.load_motion(args.hold_clip).frames, dtype=np.float32))
    fps = float(motion.load_motion(args.hold_clip).fps)
    Nb = base.shape[0]
    P_crow = base[args.crow_frame].clone()
    print("natural base: {} ({} frames @ {:.0f}fps)".format(os.path.basename(args.hold_clip), Nb, fps))

    # --- FK-optimize the extended right-leg pose (straight, up-and-back) ---
    u = torch.tensor(args.target); u = u / u.norm()
    seed = P_crow[6:].reshape(23, 3).clone()
    seed[R_KNEE_J] = 0.0; seed[R_ANKLE_J] = 0.0; seed[R_TOE_J] = 0.0
    root_pos = P_crow[0:3].unsqueeze(0); root_rot = torch_util.exp_map_to_quat(P_crow[3:6].unsqueeze(0))
    h = seed[R_HIP_J].clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=0.05)
    for _ in range(args.opt_steps):
        je = torch.cat([seed[:R_HIP_J], h.unsqueeze(0), seed[R_HIP_J + 1:]], dim=0)
        bp, _ = kin.forward_kinematics(root_pos, root_rot, torch_util.exp_map_to_quat(je).unsqueeze(0))
        hip, toe = bp[0, B["R_Hip"]], bp[0, B["R_Toe"]]
        leg = toe - hip; ld = leg / (leg.norm() + 1e-8)
        loss = (1 - (ld * u).sum()) + 2.0 * (toe[0] - hip[0]) ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    ext_leg = {LEGCOLS[0]: h.detach(), LEGCOLS[1]: torch.zeros(3),
               LEGCOLS[2]: torch.zeros(3), LEGCOLS[3]: torch.zeros(3)}
    qe = {c: torch_util.exp_map_to_quat(ext_leg[c].unsqueeze(0))[0] for c in LEGCOLS}

    # --- build: natural body + slow right-leg overlay ---
    T = args.nA + args.nB + args.nC
    seq = []
    for i in range(T):
        f = base[i % Nb].clone()                                 # natural crow body
        if i < args.nA:              blend = 0.0
        elif i < args.nA + args.nB:  blend = (i - args.nA) / float(args.nB)
        else:                        blend = 1.0
        for c in LEGCOLS:
            q0 = torch_util.exp_map_to_quat(f[c:c + 3].unsqueeze(0))[0]   # base's natural right-leg joint
            q = slerp(q0, qe[c], blend)
            f[c:c + 3] = torch_util.quat_to_exp_map(q.unsqueeze(0))[0]
        seq.append(f)
    clip = torch.stack(seq, 0).numpy().astype(np.float32)

    # --- verify ---
    def rd(frame, tag):
        bp, _ = fk(kin, frame)
        def g(n): return bp[B[n]]
        def d(a, b): return float((g(a) - g(b)).norm())
        straight = float((g("R_Hip") - g("R_Toe")).norm()) / (
            d("R_Hip", "R_Knee") + d("R_Knee", "R_Ankle") + d("R_Ankle", "R_Toe") + 1e-8)
        print("  [{:<10s}] R_Toe z={:+.3f} reach={:.3f} straight={:.2f} | L_Toe z={:+.3f} "
              "L_Knee->L_Sh={:.3f} R_Knee->R_Sh={:.3f}".format(
                  tag, float(g("R_Toe")[2]), d("R_Toe", "Pelvis"), straight,
                  float(g("L_Toe")[2]), d("L_Knee", "L_Shoulder"), d("R_Knee", "R_Shoulder")))
    print("-" * 84)
    rd(torch.tensor(clip[args.nA // 2]), "crow")
    rd(torch.tensor(clip[args.nA + args.nB // 2]), "mid-extend")
    rd(torch.tensor(clip[-1]), "EXTENDED")
    allbp = torch.stack([fk(kin, torch.tensor(clip[i]))[0] for i in range(0, T, 4)], 0)
    print("  whole-clip min body-z={:+.3f} (raw)  R_Toe z range [{:+.3f},{:+.3f}]  L_Toe z range [{:+.3f},{:+.3f}]".format(
        float(allbp[..., 2].min()), float(allbp[:, B["R_Toe"], 2].min()), float(allbp[:, B["R_Toe"], 2].max()),
        float(allbp[:, B["L_Toe"], 2].min()), float(allbp[:, B["L_Toe"], 2].max())))

    motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=fps, frames=clip).save(args.out)
    print("-" * 84)
    print("saved CLAMP slow clip: {} frames = {:.2f}s (crow {:.1f}s + extend {:.1f}s + hold {:.1f}s) -> {}".format(
        T, (T - 1) / fps, args.nA / fps, args.nB / fps, args.nC / fps, args.out))


if __name__ == "__main__":
    main()
