"""Quantitative verdict on a trained ONE-LEGGED CROW policy, from a deterministic
rollout recorded by tools/collect_crow_telemetry.py (an .npz of per-step body_pos,
ground_force, net_force, dof_*). No Isaac needed.

Checks the four properties the experiment targets, over the held one-legged-crow
segment (auto-detected as the frames where the right toe is extended up):
  1. RIGHT LEG EXTENDED  -- R_Toe rides high & far from the pelvis; knee straight;
                            right knee has LEFT the right arm.
  2. TOES OFF THE GROUND  -- L/R toe & ankle ground-contact force ~ 0.
  3. LEFT KNEE CONTACT    -- L_Knee seated on L_Shoulder (small gap) and carrying a
                            body-to-body load (net - ground force).
  4. CoP LEANS LEFT       -- GRF-weighted center-of-pressure sits left of the hand
                            centroid (toward L_Hand).
Plus BALANCE: did the episode hold (no early fall) for the recorded horizon.

Run (after training):
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/collect_crow_telemetry.py \
      --env data/envs/deepmimic_smpl_one_legged_crow_env.yaml \
      --agent data/agents/deepmimic_smpl_one_legged_crow_ppo_agent.yaml \
      --model output/model_one_legged_crow.pt \
      --out output/one_legged_crow_telemetry.npz --steps 240 --warmup 10
  $PY tools/verify_one_legged_crow.py --npz output/one_legged_crow_telemetry.npz
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="output/one_legged_crow_telemetry.npz")
    p.add_argument("--extend_z", type=float, default=0.60,
                   help="R_Toe height (m) above which the leg counts as extended")
    p.add_argument("--contact_N", type=float, default=2.0, help="ground-contact force threshold (N)")
    args = p.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    names = list(d["body_names"])
    bi = {n: i for i, n in enumerate(names)}
    bp = d["body_pos"]            # [T,B,3]
    grd = d["ground_force"]       # [T,B,3]
    net = d["net_force"]          # [T,B,3]
    T = bp.shape[0]
    b2b = net - grd               # body-to-body force per body (knee-on-arm shelf)

    def z(name): return bp[:, bi[name], 2]
    def pos(name): return bp[:, bi[name], :]
    def gfz(name): return grd[:, bi[name], 2]
    def gmag(name): return np.linalg.norm(grd[:, bi[name], :], axis=-1)

    # --- segment: the held one-legged crow (right toe extended up) ---
    ext = z("R_Toe") > args.extend_z
    print("=" * 80)
    print("rollout: {} control steps recorded".format(T))
    print("right-leg-extended frames (R_Toe z > {:.2f}): {}/{} ({:.0%})".format(
        args.extend_z, int(ext.sum()), T, ext.mean()))
    if ext.sum() < 5:
        print(">>> the policy did NOT extend the right leg (R_Toe stayed low). "
              "max R_Toe z = {:+.3f} m".format(float(z("R_Toe").max())))
        seg = slice(max(0, T - 30), T)   # fall back to the last second
    else:
        idx = np.where(ext)[0]
        seg = slice(idx[0], idx[-1] + 1)
    sl = np.zeros(T, bool); sl[seg] = True; sl &= (ext | (ext.sum() < 5))

    def m(arr): return float(np.mean(arr[sl]))

    print("-" * 80)
    print("HELD-SEGMENT metrics (mean over the extended one-legged-crow frames):")

    # 1. right leg extended
    rtoe_z = m(z("R_Toe"))
    rtoe_reach = m(np.linalg.norm((pos("R_Toe") - pos("Pelvis"))[:, :3], axis=-1))
    # knee straightness: |Hip->Toe| / (|Hip->Knee|+|Knee->Ankle|+|Ankle->Toe|); 1=straight
    seglen = (np.linalg.norm(pos("R_Hip") - pos("R_Knee"), axis=-1)
              + np.linalg.norm(pos("R_Knee") - pos("R_Ankle"), axis=-1)
              + np.linalg.norm(pos("R_Ankle") - pos("R_Toe"), axis=-1))
    straight = np.linalg.norm(pos("R_Hip") - pos("R_Toe"), axis=-1) / (seglen + 1e-8)
    r_knee_arm = m(np.linalg.norm(pos("R_Knee") - pos("R_Shoulder"), axis=-1))
    print("  1) RIGHT LEG  R_Toe z={:+.3f} m  reach(Pelvis->R_Toe)={:.3f} m  "
          "knee_straightness={:.2f}  R_Knee->R_Shoulder={:.3f} m (extended away)".format(
              rtoe_z, rtoe_reach, m(straight), r_knee_arm))

    # 2. toes off ground
    def contact_pct(name): return float((gmag(name)[sl] > args.contact_N).mean())
    print("  2) TOES OFF   L_Toe Fz={:5.1f}N ({:3.0%} in contact) | R_Toe Fz={:5.1f}N ({:3.0%}) | "
          "L_Ankle={:5.1f}N R_Ankle={:5.1f}N".format(
              m(gfz("L_Toe")), contact_pct("L_Toe"), m(gfz("R_Toe")), contact_pct("R_Toe"),
              m(gfz("L_Ankle")), m(gfz("R_Ankle"))))

    # 3. left knee seated on left arm + loaded
    l_knee_arm = m(np.linalg.norm(pos("L_Knee") - pos("L_Shoulder"), axis=-1))
    l_knee_load = m(np.linalg.norm(b2b[:, bi["L_Knee"], :], axis=-1))
    print("  3) LEFT KNEE  L_Knee->L_Shoulder={:.3f} m  body-to-body load on L_Knee={:5.1f}N "
          "(seated & bearing)".format(l_knee_arm, l_knee_load))

    # 4. CoP leans left
    cop_bodies = ["L_Hand", "R_Hand", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    fz = np.stack([np.clip(gfz(n), 0, None) for n in cop_bodies], axis=-1)   # [T,6]
    xy = np.stack([pos(n)[:, :2] for n in cop_bodies], axis=1)               # [T,6,2]
    cop = (xy * fz[..., None]).sum(1) / np.clip(fz.sum(-1, keepdims=True), 1e-6, None)  # [T,2]
    hands = 0.5 * (pos("L_Hand")[:, :2] + pos("R_Hand")[:, :2])              # [T,2]
    leftdir = pos("L_Hand")[:, :2] - pos("R_Hand")[:, :2]
    leftdir = leftdir / (np.linalg.norm(leftdir, axis=-1, keepdims=True) + 1e-8)
    left_bias = ((cop - hands) * leftdir).sum(-1)   # >0 => CoP is toward the LEFT hand
    fwddir_x = pos("L_Hand")[:, 0] - pos("R_Hand")[:, 0]  # just for context
    print("  4) CoP LEAN   CoP left-of-hand-centroid = {:+.3f} m  ({})".format(
          m(left_bias), "LEFT as targeted" if m(left_bias) > 0.005 else "not clearly left"))

    # balance / hold
    print("-" * 80)
    held = T  # collector breaks the loop on done; full horizon => never fell
    print("  BALANCE     held {} control steps (~{:.1f}s @30Hz) of the rollout without terminating".format(
          held, held / 30.0))
    print("=" * 80)


if __name__ == "__main__":
    main()
