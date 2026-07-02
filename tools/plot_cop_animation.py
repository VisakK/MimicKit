"""Req 2: birds-eye (top-down) animation of the center of pressure as the policy
executes the crow. The CoP is the vertical-GRF-weighted mean horizontal position
over all ground-contacting bodies: CoP_xy = sum_i xy_i*Fz_i / sum_i Fz_i. Each
contact body is drawn as a circle sized by its Fz; the support polygon (convex
hull of loaded contacts) and the CoP star update per frame. Also writes a static
PNG of the whole CoP trajectory colored by time.

Outputs mp4 (ffmpeg) or gif fallback. Reads output/crow_telemetry.npz.
Run: env_isaaclab/bin/python tools/plot_cop_animation.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def convex_hull(pts):
    # minimal Andrew's monotone chain; pts [n,2]
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", default="output/crow_telemetry.npz")
    p.add_argument("--out", default="output/crow_cop")  # extension added
    p.add_argument("--fzmin", type=float, default=2.0, help="N; min Fz to count a body as loaded")
    args = p.parse_args()

    d = np.load(args.telemetry)
    names = [str(x) for x in d["body_names"]]
    dt = float(d["dt"])
    pos = d["body_pos"]            # [T,B,3]
    grd = d["ground_force"]        # [T,B,3]
    fz = np.clip(grd[:, :, 2], 0, None)   # [T,B] vertical GRF
    T = pos.shape[0]

    # bodies that ever bear load
    loaded = [i for i in range(len(names)) if fz[:, i].max() > args.fzmin]
    xy = pos[:, :, :2]

    # per-frame CoP
    cop = np.full((T, 2), np.nan)
    for t in range(T):
        w = fz[t, loaded]
        if w.sum() > 1e-6:
            cop[t] = (xy[t, loaded] * w[:, None]).sum(0) / w.sum()

    # axis bounds from loaded contacts
    P = xy[:, loaded, :].reshape(-1, 2)
    pad = 0.06
    xlo, xhi = P[:, 0].min()-pad, P[:, 0].max()+pad
    ylo, yhi = P[:, 1].min()-pad, P[:, 1].max()+pad

    # ---- static trajectory PNG ----
    figs, axs = plt.subplots(figsize=(6, 6))
    axs.plot(xy[:, loaded, 0], xy[:, loaded, 1], ".", ms=2, c="lightgray")
    sc = axs.scatter(cop[:, 0], cop[:, 1], c=np.arange(T)*dt, cmap="viridis", s=14)
    axs.set_aspect("equal"); axs.set_xlim(xlo, xhi); axs.set_ylim(ylo, yhi)
    axs.set_xlabel("x (m)"); axs.set_ylabel("y (m)")
    axs.set_title("CoP trajectory (birds-eye), colored by time")
    plt.colorbar(sc, ax=axs, label="time (s)")
    axs.grid(ls="dotted"); figs.tight_layout()
    figs.savefig(args.out + "_trajectory.png", dpi=130)

    # ---- animation ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal"); ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(ls="dotted")
    title = ax.set_title("")
    hull_line, = ax.plot([], [], "-", c="steelblue", lw=1.5, alpha=0.7)
    body_sc = ax.scatter([], [], s=[], c="tab:orange", alpha=0.8, zorder=3)
    cop_pt, = ax.plot([], [], "*", c="red", ms=18, zorder=4)
    cop_trail, = ax.plot([], [], "-", c="red", lw=1, alpha=0.4)
    labels = [ax.annotate(names[i].replace("_", ""), (0, 0), fontsize=7,
                          ha="center", va="bottom", color="dimgray") for i in loaded]

    def frame(t):
        w = fz[t, loaded]
        on = w > args.fzmin
        pts = xy[t, loaded]
        body_sc.set_offsets(pts)
        sizes = 20 + 6 * w  # marker area ~ Fz
        body_sc.set_sizes(sizes)
        for k, lab in enumerate(labels):
            lab.set_position((pts[k, 0], pts[k, 1] + 0.012))
            lab.set_visible(bool(on[k]))
        loaded_pts = pts[on]
        if loaded_pts.shape[0] >= 3:
            h = convex_hull(loaded_pts)
            hull_line.set_data(np.r_[h[:, 0], h[:1, 0]], np.r_[h[:, 1], h[:1, 1]])
        else:
            hull_line.set_data([], [])
        if not np.isnan(cop[t, 0]):
            cop_pt.set_data([cop[t, 0]], [cop[t, 1]])
            cop_trail.set_data(cop[:t+1, 0], cop[:t+1, 1])
        title.set_text("Center of pressure  t={:.2f}s   (marker size ~ Fz)".format(t*dt))
        return [body_sc, cop_pt, cop_trail, hull_line, title] + labels

    anim = FuncAnimation(fig, frame, frames=T, interval=1000*dt, blit=False)
    fps = max(1, int(round(1.0/dt)))
    try:
        anim.save(args.out + ".mp4", writer=FFMpegWriter(fps=fps, bitrate=2400))
        outname = args.out + ".mp4"
    except Exception as ex:
        print("ffmpeg failed ({}); writing gif".format(ex))
        anim.save(args.out + ".gif", writer=PillowWriter(fps=fps))
        outname = args.out + ".gif"
    print("loaded contact bodies:", [names[i] for i in loaded])
    print("saved", outname, "and", args.out + "_trajectory.png")


if __name__ == "__main__":
    main()
