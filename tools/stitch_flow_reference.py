"""Stitch the plank_vinyasa flow's 6 edge-clip reference segments (in flow order)
into ONE long reference trajectory for the MONOLITHIC baseline (yoga_flow_paper_plan.md
§5.2). Each edge clip = [lead-hold | transition | tail-hold]; consecutive clips share
the boundary hold (downdog/plank/side-plank), so we SE(2)-align each clip to the running
end (head->feet heading + pelvis xy — all internal seams are FLOOR poses so this is
well-defined), drop a short slice of the redundant lead-hold, and bridge the cross-take
gap. Output = one Motion (CLAMP), plus a boundary map (frame -> pose) and a strip.

Usage: env_isaaclab/bin/python tools/stitch_flow_reference.py
"""
import sys, os
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)
import make_synth_edge_clip as M
import anim.motion as motion

CLIPS = [
    ("tadasana", "data/motions/smpl_edges/tadasana_to_downdog"),
    ("downdog",  "data/motions/smpl_edges/downdog_to_plank"),
    ("plank",    "data/motions/smpl_edges/plank_to_side_plank"),
    ("side_plank","data/motions/smpl_edges/side_plank_to_plank"),
    ("plank",    "data/motions/smpl_edges/plank_to_downdog"),
    ("downdog",  "data/motions/smpl_edges/downdog_to_tadasana"),
]
OUT = "data/motions/smpl_edges/plank_vinyasa_monolith"


def main():
    kcm, geoms = M.load_kcm()
    head_i = kcm.get_body_names().index("Head")
    toe_is = [kcm.get_body_names().index(b) for b in ("L_Toe", "R_Toe")]

    m0 = motion.load_motion(CLIPS[0][1]); fps = float(m0.fps)
    running = np.asarray(m0.frames, np.float32).copy()
    drop = int(round(0.4 * fps)); nbr = int(round(0.3 * fps))
    seams = [(0, "tadasana(start)")]
    for name, path in CLIPS[1:]:
        B = np.asarray(motion.load_motion(path).frames, np.float32).copy()
        thA = M.heading_xy(running[-1], kcm, head_i, toe_is)
        thB = M.heading_xy(B[0], kcm, head_i, toe_is)
        da = thA - thB
        B = M.rot_z(B, da, np.array([0.0, 0.0]))
        dxy = running[-1, 0:2] - B[0, 0:2]; B[:, 0] += dxy[0]; B[:, 1] += dxy[1]
        K = min(B.shape[0] - 1, drop)
        br = M.bridge_frames(running[-1], B[K], nbr, kcm)
        seams.append((running.shape[0] + nbr, f"->{name} (via {os.path.basename(path)})"))
        running = np.concatenate([running, br, B[K:]], 0)

    running, off = M.ground_segment(running, kcm, geoms)
    T = running.shape[0]
    print(f"stitched {len(CLIPS)} clips -> {T} frames ({(T-1)/fps:.1f}s); final ground {off:+.3f}m")
    print("SEAMS (frame, pose):"); [print(f"   f{s[0]:4d} ({s[0]/fps:5.1f}s)  {s[1]}") for s in seams]
    mz = M.min_geom_z(running, kcm, geoms).min(dim=1)[0].numpy()
    rj = np.linalg.norm(np.diff(running[:, 0:3], axis=0), axis=-1).max() * fps
    print(f"worst penetration {max(0,-mz.min())*100:.1f}cm; max root speed {rj:.2f} m/s")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    motion.Motion(loop_mode=motion.LoopMode.CLAMP, fps=m0.fps, frames=running).save(OUT)
    print(f"saved -> {OUT}")

    # strip
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET
    bp, _ = kcm.forward_kinematics(torch.tensor(running[:, 0:3]),
            torch.tensor(M.tu.exp_map_to_quat(torch.tensor(running[:, 3:6])).numpy()),
            M.tu.quat_pos(kcm.dof_to_rot(torch.tensor(running[:, 6:]))))
    bp = bp.numpy()
    idx = {n: i for i, n in enumerate(kcm.get_body_names())}; edges = []
    w = ET.parse("data/assets/smpl/smpl_boxhands.xml").getroot().find("worldbody")
    def walk(e, pa):
        for c in e.findall("body"):
            nm = c.get("name")
            if pa in idx and nm in idx: edges.append((idx[pa], idx[nm]))
            walk(c, nm)
    for b in w.findall("body"): walk(b, b.get("name"))
    snaps = np.linspace(0, T - 1, 20).astype(int)
    fig, ax = plt.subplots(figsize=(28, 4))
    for kk, s in enumerate(snaps):
        xo = kk * 1.3; span = bp[s, :, :2] - bp[s, :, :2].mean(0); dv = np.linalg.svd(span)[2][0]
        x = span @ dv + xo
        for a, b2 in edges: ax.plot([x[a], x[b2]], [bp[s, a, 2], bp[s, b2, 2]], "k-", lw=1.1)
        ax.text(xo, -0.15, f"{s/fps:.1f}s", ha="center", fontsize=7)
    ax.axhline(0, color="gray", lw=0.7); ax.set_aspect("equal")
    ax.set_title(f"MONOLITH reference: {OUT} ({(T-1)/fps:.1f}s) — tadasana->downdog->plank->side_plank->plank->downdog->tadasana")
    fig.tight_layout(); fig.savefig(OUT + "_strip.png", dpi=95)
    print(f"strip -> {OUT}_strip.png")


if __name__ == "__main__":
    main()
