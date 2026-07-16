"""Multi-pose flow renderer (yoga_flow_paper_plan.md §5.1).

Renders a chained flow trajectory (concatenated per-frame body positions from the
flow executor) as ONE video: a 3D skeleton with a camera that tracks the root xy
and heading. Tracking the root hides the SE(2) alignment teleport between stages
(an artifact of the state handoff, not a real discontinuity) while leaving real
POSE seams visible — exactly what a sign-off needs to see.

No Isaac / no torch dependency: numpy + matplotlib(Agg) + ffmpeg only, so it runs
in the executor's parent process (which never boots the sim). ffmpeg at
/usr/bin/ffmpeg; falls back to gif via PillowWriter.
"""
import math

import numpy as np


CONTACT_N = 20.0   # N; a body drawn "loaded" (red) above this ground force


def _quat_rot_np(q, v):
    """Rotate v by quaternion q (xyzw), numpy, single quat + single vec."""
    xyz = np.asarray(q[:3], dtype=float)
    v = np.asarray(v, dtype=float)
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (q[3] * uv + uuv)


def _heading_deg(root_rot):
    fwd = _quat_rot_np(root_rot, [1.0, 0.0, 0.0])
    return math.degrees(math.atan2(fwd[1], fwd[0]))


def render_flow(body_pos, parents, contact, root_pos, root_rot, control_freq,
                out_path, stage_bounds=None, view_range=1.2, elev=12.0,
                azim_offset=-60.0, title=None):
    """body_pos [T,B,3], parents [B] (int, -1=root), contact [T,B] force-mag,
    root_pos [T,3], root_rot [T,4] xyzw. stage_bounds = [(start,end,name),...]
    for the caption. Writes out_path(.mp4 or .gif). Returns the written path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

    body_pos = np.asarray(body_pos)
    contact = np.asarray(contact)
    root_pos = np.asarray(root_pos)
    root_rot = np.asarray(root_rot)
    T, B, _ = body_pos.shape
    bones = [(int(p), i) for i, p in enumerate(parents) if int(p) >= 0]

    def stage_name(t):
        if not stage_bounds:
            return ""
        for s, e, nm in stage_bounds:
            if s <= t < e:
                return nm
        return stage_bounds[-1][2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    dt = 1.0 / control_freq

    def frame(t):
        ax.cla()
        bp = body_pos[t]
        cx, cy = root_pos[t, 0], root_pos[t, 1]
        # ground grid
        gx = np.linspace(cx - view_range, cx + view_range, 5)
        gy = np.linspace(cy - view_range, cy + view_range, 5)
        GX, GY = np.meshgrid(gx, gy)
        ax.plot_wireframe(GX, GY, np.zeros_like(GX), color="0.85", linewidth=0.5)
        # bones
        for a, b in bones:
            ax.plot([bp[a, 0], bp[b, 0]], [bp[a, 1], bp[b, 1]],
                    [bp[a, 2], bp[b, 2]], "-", color="0.15", linewidth=2.0)
        loaded = contact[t] > CONTACT_N
        if loaded.any():
            ax.scatter(bp[loaded, 0], bp[loaded, 1], bp[loaded, 2],
                       color="tab:red", s=28, depthshade=False)
        ax.scatter(bp[~loaded, 0], bp[~loaded, 1], bp[~loaded, 2],
                   color="0.4", s=8, depthshade=False)
        ax.set_xlim(cx - view_range, cx + view_range)
        ax.set_ylim(cy - view_range, cy + view_range)
        ax.set_zlim(0.0, 2.0)
        ax.set_box_aspect((1, 1, 2.0 / (2 * view_range)))
        ax.view_init(elev=elev, azim=_heading_deg(root_rot[t]) + azim_offset)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        cap = title or "flow"
        ax.set_title(f"{cap}\n{stage_name(t)}   t={t * dt:5.2f}s", fontsize=10)
        return []

    anim = FuncAnimation(fig, frame, frames=T, interval=1000 * dt, blit=False)
    fps = max(1, int(round(control_freq)))
    try:
        out = out_path if out_path.endswith(".mp4") else out_path + ".mp4"
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception as ex:  # noqa: BLE001
        print(f"[flow_render] ffmpeg failed ({ex}); writing gif")
        out = (out_path[:-4] if out_path.endswith(".mp4") else out_path) + ".gif"
        anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[flow_render] wrote {out} ({T} frames, {T * dt:.1f}s @ {fps}fps)")
    return out
