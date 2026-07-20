"""Synthesize an EDGE CLIP for a transition that has NO in-corpus demo, by
splicing two real HOLD segments (A = lead hold, B = tail hold) from different
source clips with a short kinematic BRIDGE between them (the plan's
"splice-at-shared-boundary" synthesis for plank<->side_plank).

Method (Yoga_edge_framework_v3.md routing = DeepMimic-first on a synthesized demo;
THE LESSON: "no clean demo -> synthesize/mirror, and expect to pay for it"):
  1. Take real hold frames A (from src_a) and B (from src_b) — quasi-static, so
     their boundary velocities are ~0 and a zero-velocity-ended bridge matches.
  2. ALIGN B onto A: rotate B about z so its head->feet horizontal axis matches
     A's (the roll happens IN PLACE, same facing), translate so B[0] pelvis xy
     = A[-1] pelvis xy. Ground each segment independently first (different source
     clips have different seat heights).
  3. BRIDGE A[-1] -> B[0] over bridge_s with smoothstep easing: lerp root pos,
     slerp root rot + every joint (dof<->quat). Both ends quasi-static => C1-ish.
  4. Concatenate [A | bridge | B], bake a final ground offset, save CLAMP, and
     emit the same pre-flight strip + report as make_edge_clips.

Usage (repo root, env python; frames @ src fps):
  env_isaaclab/bin/python tools/make_synth_edge_clip.py \
    --src_a_file data/motions/smpl/220923_Plank_Pose_or_Kumbhakasana_-a --src_a_win 530 565 \
    --src_b_file data/motions/smpl/220923_Side_Plank_Pose_or_Vasisthasana_-e --src_b_win 300 378 \
    --bridge_s 1.2 --out data/motions/smpl_edges/plank_to_side_plank
Outputs: <out> (pickled Motion, CLAMP), <out>_preflight.png, report on stdout.
"""
import argparse, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import anim.motion as motion
import anim.kin_char_model as kin_char_model
import anim.char_geoms as char_geoms
import annotate_clips as AC
import util.torch_util as tu

CHAR_FILE = "data/assets/smpl/smpl_boxhands.xml"   # geoms == lowtorque variant
PCTL_GROUND = 5.0
SEAT_TOL_PEN = 0.015


def load_kcm():
    kcm = kin_char_model.KinCharModel("cpu")
    kcm.load_char_file(CHAR_FILE)
    geoms = char_geoms.load_char_geoms(CHAR_FILE, kcm.get_body_names(), "cpu")
    return kcm, geoms


def fk(frames, kcm):
    rp = torch.tensor(frames[:, 0:3], dtype=torch.float32)
    rr = tu.exp_map_to_quat(torch.tensor(frames[:, 3:6], dtype=torch.float32))
    jr = tu.quat_pos(kcm.dof_to_rot(torch.tensor(frames[:, 6:], dtype=torch.float32)))
    bp, br = kcm.forward_kinematics(rp, rr, jr)
    return bp, br


def min_geom_z(frames, kcm, geoms):
    bp, br = fk(frames, kcm)
    minz, _ = AC.per_body_witness(geoms, bp, br)   # [T,B]
    return minz


def ground_segment(frames, kcm, geoms):
    """shift a segment in z so its own per-frame-min 5th pctl sits at 0."""
    mz = min_geom_z(frames, kcm, geoms).min(dim=1)[0].numpy()
    off = -float(np.percentile(mz, PCTL_GROUND))
    f = frames.copy(); f[:, 2] += off
    return f, off


def heading_xy(frame, kcm, head_i, toe_is):
    bp, _ = fk(frame[None], kcm)
    bp = bp[0].numpy()
    v = bp[head_i, :2] - bp[toe_is, :2].mean(0)      # head <- feet, horizontal
    return float(np.arctan2(v[1], v[0]))


def rot_z(frames, da, pivot_xy):
    """rotate a whole segment about world-z by da around pivot_xy (root rot + root xy)."""
    c, s = np.cos(da), np.sin(da)
    f = frames.copy()
    xy = f[:, 0:2] - pivot_xy
    f[:, 0] = xy[:, 0] * c - xy[:, 1] * s + pivot_xy[0]
    f[:, 1] = xy[:, 0] * s + xy[:, 1] * c + pivot_xy[1]
    qz = tu.exp_map_to_quat(torch.tensor([[0.0, 0.0, da]], dtype=torch.float32))  # [1,4]
    rr = tu.exp_map_to_quat(torch.tensor(f[:, 3:6], dtype=torch.float32))
    rr = tu.quat_pos(tu.quat_mul(qz.expand_as(rr), rr))
    f[:, 3:6] = tu.quat_to_exp_map(rr).numpy()
    return f


def bridge_frames(A_last, B_first, n, kcm):
    """smoothstep interp A_last -> B_first over n interior frames (exclusive)."""
    qa = tu.exp_map_to_quat(torch.tensor(A_last[3:6], dtype=torch.float32)[None])[0]
    qb = tu.exp_map_to_quat(torch.tensor(B_first[3:6], dtype=torch.float32)[None])[0]
    jqa = tu.quat_pos(kcm.dof_to_rot(torch.tensor(A_last[6:], dtype=torch.float32)[None]))[0]  # [J,4]
    jqb = tu.quat_pos(kcm.dof_to_rot(torch.tensor(B_first[6:], dtype=torch.float32)[None]))[0]
    pa = A_last[0:3]; pb = B_first[0:3]
    out = []
    for k in range(1, n + 1):
        u = k / (n + 1)
        t = u * u * (3 - 2 * u)                       # smoothstep
        rp = (1 - t) * pa + t * pb
        rq = tu.slerp(qa[None], qb[None], torch.tensor([t]))[0]
        rexp = tu.quat_to_exp_map(rq[None])[0].numpy()
        jq = tu.slerp(jqa, jqb, torch.full((jqa.shape[0],), float(t)))  # [J,4]
        jdof = kcm.rot_to_dof(jq[None])[0].numpy()
        fr = np.concatenate([rp, rexp, jdof]).astype(np.float32)
        out.append(fr)
    return np.stack(out, 0) if out else np.zeros((0, A_last.shape[0]), np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_a_file", required=True, help="LEAD hold source clip")
    p.add_argument("--src_a_win", type=int, nargs=2, required=True, metavar=("S", "E"))
    p.add_argument("--src_b_file", required=True, help="TAIL hold source clip")
    p.add_argument("--src_b_win", type=int, nargs=2, required=True, metavar=("S", "E"))
    p.add_argument("--bridge_s", type=float, default=1.2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    kcm, geoms = load_kcm()
    head_i = kcm.get_body_names().index("Head")
    toe_is = [kcm.get_body_names().index(b) for b in ("L_Toe", "R_Toe")]

    ma = motion.load_motion(args.src_a_file); fps = float(ma.fps)
    mb = motion.load_motion(args.src_b_file)
    A = np.asarray(ma.frames, np.float32)[args.src_a_win[0]:args.src_a_win[1] + 1].copy()
    B = np.asarray(mb.frames, np.float32)[args.src_b_win[0]:args.src_b_win[1] + 1].copy()

    # 1. independently ground each hold segment
    A, offA = ground_segment(A, kcm, geoms)
    B, offB = ground_segment(B, kcm, geoms)
    print(f"lead A={os.path.basename(args.src_a_file)}[{args.src_a_win}] {A.shape[0]}f seat {offA:+.3f} | "
          f"tail B={os.path.basename(args.src_b_file)}[{args.src_b_win}] {B.shape[0]}f seat {offB:+.3f}")

    # 2. align B onto A: match head->feet heading, then pelvis xy at the seam
    thA = heading_xy(A[-1], kcm, head_i, toe_is)
    thB = heading_xy(B[0], kcm, head_i, toe_is)
    da = thA - thB
    B = rot_z(B, da, pivot_xy=np.array([0.0, 0.0]))
    dxy = A[-1, 0:2] - B[0, 0:2]
    B[:, 0] += dxy[0]; B[:, 1] += dxy[1]
    print(f"align B: heading A {np.degrees(thA):.1f}deg B {np.degrees(thB):.1f}deg -> rot {np.degrees(da):+.1f}deg; "
          f"pelvis-xy shift {dxy}")

    # 3. bridge A[-1] -> B[0]
    n = int(round(args.bridge_s * fps))
    br = bridge_frames(A[-1], B[0], n, kcm)
    seg = np.concatenate([A, br, B], 0)
    t_tr0 = (A.shape[0] - 1) / fps
    t_tr1 = (A.shape[0] - 1 + n + 1) / fps
    print(f"bridge {n} interior frames ({args.bridge_s:.2f}s); total {seg.shape[0]}f "
          f"({(seg.shape[0]-1)/fps:.2f}s); transition window [{t_tr0:.2f},{t_tr1:.2f}]s")

    # 4. final global ground bake + smoke checks
    mz = min_geom_z(seg, kcm, geoms).min(dim=1)[0].numpy()
    off = -float(np.percentile(mz, PCTL_GROUND))
    seg[:, 2] += off
    after = mz + off
    worst = max(0.0, -float(after.min()))
    root_jump = np.linalg.norm(np.diff(seg[:, 0:3], axis=0), axis=-1).max() * fps
    print(f"final bake {off:+.4f}m; worst penetration {worst*100:.1f}cm "
          f"(bridge-min {after[A.shape[0]:A.shape[0]+n].min()*100 if n else 0:.1f}cm); "
          f"max root speed {root_jump:.2f} m/s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=ma.fps, frames=seg).save(args.out)
    print(f"saved CLAMP synth edge clip -> {args.out}")

    # strip render (same layout as make_edge_clips)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bp, _ = fk(seg, kcm); bpn = bp.numpy(); T = seg.shape[0]
    parents = [int(kcm.get_parent_id(i)) for i in range(len(kcm.get_body_names()))]
    snaps = np.linspace(0, T - 1, 10).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    span = bpn[:, :, :2].reshape(-1, 2); dirv = np.linalg.svd(span - span.mean(0))[2][0]
    for k, s in enumerate(snaps):
        xo = k * 1.2; ph = bpn[s, :, :2] @ dirv; ph = ph - ph.mean() + xo
        in_tr = t_tr0 <= s / fps <= t_tr1
        for b, pa in enumerate(parents):
            if pa < 0: continue
            ax.plot([ph[b], ph[pa]], [bpn[s, b, 2], bpn[s, pa, 2]], "k-", lw=1.5)
        ax.plot(ph, bpn[s, :, 2], "o", ms=2.5, color="tab:orange" if in_tr else "tab:blue")
        ax.text(xo, -0.12, f"{s/fps:.1f}s", ha="center", fontsize=8)
    ax.axhline(0.0, color="gray", lw=0.8); ax.set_aspect("equal")
    ax.set_title(f"SYNTH edge clip {os.path.basename(args.out)} — pre-flight (orange = bridge/transition)")
    ax2 = axes[1]; tt = np.arange(T) / fps
    ax2.plot(tt, min_geom_z(seg, kcm, geoms).min(dim=1)[0].numpy() * 100, lw=1)
    ax2.axhspan(-SEAT_TOL_PEN * 100, 0, color="red", alpha=0.15)
    ax2.axvspan(t_tr0, t_tr1, color="orange", alpha=0.15)
    ax2.set_xlabel("s"); ax2.set_ylabel("min geom z (cm)")
    fig.tight_layout(); png = args.out + "_preflight.png"; fig.savefig(png, dpi=110)
    print(f"pre-flight strip -> {png}")
    print(f"PRE-FLIGHT {'PASS' if worst <= SEAT_TOL_PEN else 'CHECK (inspect the roll)'}")


if __name__ == "__main__":
    main()
