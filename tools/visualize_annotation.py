"""Visualize the Stage-1 annotation of ONE motion clip, to eyeball correctness.

Re-runs tools/annotate_clips.annotate_clip(return_trace=True) so what is drawn is
exactly what was annotated (no re-derivation drift), then renders one figure:

  * TIMELINE: up_z (inversion), COM height, body speed over the clip, with the
    detected hold windows shaded (primary hold highlighted), entry/exit transition
    spans marked, and the fingerprint's peak-salience frame flagged.
  * CONTACT PIANO-ROLL: per-frame ground contact for every body -> verify the
    contact detection and the hold contact signatures.
  * SKELETON STRIP: rest_start -> PRIMARY HOLD -> rest_end, each an FK skeleton with
    contact bodies red, the support polygon drawn on the floor, the COM + its
    ground projection, and leg-arm body-on-body links -> verify the hub->pose->hub
    story and the hold fingerprint (contacts / support polygon / on-body support).

  env_isaaclab/bin/python tools/visualize_annotation.py --clip Bakasana_-a
  env_isaaclab/bin/python tools/visualize_annotation.py --clip Handstand --anim
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import anim.kin_char_model as kin_char_model
import anim.char_geoms as char_geoms
import annotate_clips as A


def resolve_clip(sub):
    if os.path.isfile(sub):
        return os.path.basename(sub), sub
    hits = [(n, p) for (n, p) in A.list_clips([sub])]
    if not hits:
        sys.exit("no clip matches '{}' under {}".format(sub, A.MOTION_DIR))
    if len(hits) > 1:
        print("multiple matches, using first:")
        for n, _ in hits[:10]:
            print("   ", n)
    return hits[0]


def hull_xy(pts):
    if len(pts) < 3:
        return pts
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts)
        return pts[h.vertices]
    except Exception:
        return pts


def com_of(P, body_names):
    w = np.array([A.MASS_FRACTIONS.get(n, 0.0) for n in body_names])
    return P.T @ w / w.sum()


def horiz_axis(P):
    """Dominant horizontal direction of the pose (for a heading-agnostic side view)."""
    xy = P[:, :2] - P[:, :2].mean(0)
    ev, evec = np.linalg.eigh(xy.T @ xy)
    return evec[:, -1]


def draw_side(ax, P, contact_mask, parents, body_names, floor, onbody, title):
    """Heading-aligned SIDE profile (dominant-horizontal u vs height z). Makes
    standing (tall) vs arm-balance (low/horizontal) vs inversion (upside-down)
    unambiguous."""
    d = horiz_axis(P)
    u = P[:, :2] @ d
    u = u - u.mean()
    z = P[:, 2]
    con = contact_mask.astype(bool)
    name2i = {n: i for i, n in enumerate(body_names)}
    for b in range(len(P)):
        p = parents[b]
        if p >= 0:
            ax.plot([u[b], u[p]], [z[b], z[p]], color="0.55", lw=1.6, zorder=1)
    for pair in onbody:                                   # leg-arm on-body links
        a, b = name2i.get(pair[0]), name2i.get(pair[1])
        if a is not None and b is not None:
            ax.plot([u[a], u[b]], [z[a], z[b]], color="green", ls="--", lw=1.8, zorder=4)
    ax.scatter(u[~con], z[~con], c="steelblue", s=22, zorder=2)
    ax.scatter(u[con], z[con], c="crimson", s=55, zorder=3, edgecolors="k", linewidths=0.4)
    hi = name2i["Head"]
    ax.scatter([u[hi]], [z[hi]], c="orange", s=90, marker="o", zorder=3,
               edgecolors="k", linewidths=0.5, label="head")
    com = com_of(P, body_names)
    cu = float(np.array([com[0], com[1]]) @ d) - float((P[:, :2] @ d).mean())
    ax.scatter([cu], [com[2]], marker="*", c="gold", s=170, edgecolors="k",
               linewidths=0.5, zorder=5)
    ax.axhline(floor, color="saddlebrown", lw=2, alpha=0.6)                 # ground
    ax.axvline(cu, color="gold", ls=":", lw=1)                             # COM plumb line
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_ylabel("height (m)", fontsize=8)
    ax.set_title(title, fontsize=9)


def draw_topdown(ax, P, contact_mask, wit, body_names, com_xy, title):
    """Top-down (x-y) view of the primary hold: bones faint, contact witness points,
    support-polygon hull, COM ground projection -> verify the support polygon."""
    con = contact_mask.astype(bool)
    ax.plot(P[:, 0], P[:, 1], ".", color="0.8", ms=3, zorder=1)
    cids = np.where(con)[0]
    spts = wit[cids]
    if len(spts) >= 3:
        poly = hull_xy(spts)
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="crimson", alpha=0.18,
                                edgecolor="crimson", lw=1.5, zorder=2))
    if len(spts):
        ax.scatter(spts[:, 0], spts[:, 1], c="crimson", s=40, zorder=3,
                   edgecolors="k", linewidths=0.3)
        for b in cids:
            ax.annotate(body_names[b], (wit[b, 0], wit[b, 1]), fontsize=6, zorder=4)
    ax.scatter([com_xy[0]], [com_xy[1]], marker="*", c="gold", s=170,
               edgecolors="k", linewidths=0.5, zorder=5, label="COM (top-down)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9); ax.legend(loc="upper right", fontsize=7)


def shade_segments(ax, ann, fps, y=None):
    seg = ann["segments"]
    pri = seg["primary_hold_index"]
    for k, h in enumerate(seg["holds"]):
        ax.axvspan(h["start_s"], h["end_s"],
                   color=("forestgreen" if k == pri else "yellowgreen"),
                   alpha=(0.35 if k == pri else 0.15), lw=0)
    for tr, col in [(seg["entry_transition"], "darkorange"),
                    (seg["exit_transition"], "red")]:
        if tr:
            ax.axvspan(tr["start_s"], tr["end_s"], color=col, alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="clip name or path substring")
    ap.add_argument("--out", default=None)
    ap.add_argument("--anim", action="store_true", help="also write a GIF of the skeleton")
    args = ap.parse_args()

    name, path = resolve_clip(args.clip)
    kcm = kin_char_model.KinCharModel(A.DEVICE)
    kcm.load_char_file(A.CHAR_FILE)
    body_names = kcm.get_body_names()
    parents = [int(kcm.get_parent_id(i)) for i in range(len(body_names))]
    geoms = char_geoms.load_char_geoms(A.CHAR_FILE, body_names, A.DEVICE)

    ann, tr = A.annotate_clip(name, path, kcm, body_names, parents, geoms, return_trace=True)

    nf, fps = tr["nf"], tr["fps"]
    t = np.arange(nf) / fps
    bp = tr["body_pos"].numpy()
    wit = tr["wit_xy"].numpy()
    inc = tr["in_contact"].numpy()
    uz = tr["up_z"].numpy()
    com = tr["com"].numpy()
    spd = tr["body_speed"].numpy()
    floor = tr["floor"]
    hs = ann["hold_signature"] or {}
    onbody = hs.get("body_on_body", [])

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.3, 1.6], hspace=0.35, wspace=0.22)

    # --- timeline ---
    axt = fig.add_subplot(gs[0, :])
    axt.plot(t, uz, label="up_z (inversion)", color="navy", lw=1.3)
    axt.plot(t, com[:, 2], label="COM height (m)", color="purple", lw=1.1)
    axt.plot(t, np.clip(spd, 0, 3), label="body speed (m/s)", color="gray", lw=0.8, alpha=0.7)
    axt.axhline(0, color="k", lw=0.5, alpha=0.4)
    shade_segments(axt, ann, fps)
    if tr["sig_win"] is not None:
        pk = (tr["sig_win"][0] + tr["sig_win"][1]) // 2
        axt.axvline(pk / fps, color="forestgreen", ls="--", lw=1.2, label="fingerprint frame")
    axt.set_xlim(0, t[-1]); axt.set_ylabel("value"); axt.set_xlabel("time (s)")
    axt.legend(loc="upper right", fontsize=8, ncol=2)
    axt.set_title("{}   [{}]   len {:.1f}s   ground_offset {:.3f}m".format(
        name, ann["suggested"]["support_mode"], ann["length_s"],
        ann["ground"]["recommended_ground_offset_m"]), fontsize=11)

    # --- contact piano-roll (bodies ever in contact) ---
    axc = fig.add_subplot(gs[1, :])
    ever = np.where(inc.any(0))[0]
    roll = inc[:, ever].T.astype(float)
    axc.imshow(roll, aspect="auto", cmap="Greys", interpolation="nearest",
               extent=[0, t[-1], len(ever) - 0.5, -0.5], vmin=0, vmax=1)
    axc.set_yticks(range(len(ever)))
    axc.set_yticklabels([body_names[b] for b in ever], fontsize=7)
    axc.set_xlabel("time (s)"); axc.set_title("per-frame ground contact", fontsize=10)
    shade_segments(axc, ann, fps)

    # --- pose strip: [rest_start side] [PRIMARY side] [PRIMARY top-down] ---
    seg = ann["segments"]
    holds = seg["holds"]
    def mid(h):
        return (h["start_frame"] + h["end_frame"]) // 2
    rs = mid(holds[0]) if holds else 0
    pk = ((tr["sig_win"][0] + tr["sig_win"][1]) // 2) if tr["sig_win"] is not None else rs

    ax_r = fig.add_subplot(gs[2, 0])
    draw_side(ax_r, bp[rs], inc[rs], parents, body_names, floor, [],
              "t={:.1f}s  rest_start / hub  up_z={:.2f}\n{}".format(
                  rs / fps, uz[rs], ann["endpoints"].get("rest_start", {}).get("support_mode", "")))
    ax_p = fig.add_subplot(gs[2, 1])
    draw_side(ax_p, bp[pk], inc[pk], parents, body_names, floor, onbody,
              "PRIMARY HOLD  up_z={:.2f}  [{}]\nlr_sym={}  {}".format(
                  uz[pk], ann["suggested"]["support_mode"], hs.get("lr_symmetry"),
                  ",".join(hs.get("contact_bodies", []))))
    ax_t = fig.add_subplot(gs[2, 2])
    draw_topdown(ax_t, bp[pk], inc[pk], wit[pk], body_names, com[pk, :2],
                 "top-down support\narea={}m2  span={}m  margin={}m".format(
                     hs.get("support_area_m2"), hs.get("support_span_m"), hs.get("com_margin_m")))

    out = args.out or os.path.join(A.OUT_DIR, name + ".png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("  primary hold: up_z={} mode={} contacts={}".format(
        hs.get("up_z"), ann["suggested"]["support_mode"], hs.get("contact_bodies")))
    print("  support: span={} area={} margin={} lr_sym={}  on-body={}".format(
        hs.get("support_span_m"), hs.get("support_area_m2"), hs.get("com_margin_m"),
        hs.get("lr_symmetry"), [p[:2] for p in onbody]))
    print("  segments: {} holds, entry={} exit={}".format(
        len(holds), seg["entry_transition"] and seg["entry_transition"]["dur_s"],
        seg["exit_transition"] and seg["exit_transition"]["dur_s"]))

    if args.anim:
        write_anim(name, bp, inc, wit, parents, body_names, floor, uz, t, fps,
                   ann, os.path.join(A.OUT_DIR, name + ".gif"))


def write_anim(name, bp, inc, wit, parents, body_names, floor, uz, t, fps, ann, out):
    import matplotlib.animation as animation
    nf = bp.shape[0]
    step = max(1, nf // 150)
    idx = list(range(0, nf, step))
    fig = plt.figure(figsize=(7, 8))
    axs = fig.add_subplot(2, 1, 1)
    axt = fig.add_subplot(2, 1, 2)
    axt.plot(t, uz, color="navy", lw=1); shade_segments(axt, ann, fps)
    axt.set_xlim(0, t[-1]); axt.set_ylabel("up_z"); axt.set_xlabel("time (s)")
    cursor = axt.axvline(0, color="red", lw=1.5)

    def upd(f):
        axs.clear()
        draw_side(axs, bp[f], inc[f], parents, body_names, floor, [], "t={:.1f}s".format(f / fps))
        cursor.set_xdata([f / fps, f / fps])
        return []
    anim = animation.FuncAnimation(fig, upd, frames=idx, interval=1000 / fps * step)
    anim.save(out, writer=animation.PillowWriter(fps=min(30, fps)))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
