"""Req 3: where on each palm does the load sit -- does the policy use the whole
palm or lean to one edge? Uses the per-body AVERAGE ground contact point
(contact_pt, from the contact sensor) transformed into the palm GEOM's local face
frame, so x-y is the palm surface. We show, per hand:
  (a) the load-centroid trajectory over time (scatter colored by time), and
  (b) an Fz-weighted 2D histogram over the palm footprint -- a pseudo pressure-
      density map accumulated over the episode.
The palm footprint rectangle (box half-extents 0.040 x 0.042 m, ~19deg geom tilt)
is drawn for reference.

LIMITATION: contact_pt is the AVERAGE contact point per frame (the sensor collapses
multiple contact points to their mean), so this is the first moment of the pressure
distribution over TIME, not an instantaneous within-frame pressure field. A true
per-frame pressure map would need the palm geom subdivided into a grid of pads.

Reads output/crow_telemetry.npz. Run: env_isaaclab/bin/python tools/plot_palm_pressure.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Palm box geoms from data/assets/smpl/smpl_boxhands.xml (pos, quat wxyz, half-size)
PALM = {
    "L_Hand": dict(pos=np.array([0.0021, 0.0108, -0.0056]),
                   quat=np.array([0.9866, 0.1359, 0.0904, 0.0]),
                   half=np.array([0.040, 0.042, 0.012])),
    "R_Hand": dict(pos=np.array([0.0021, -0.0108, -0.0056]),
                   quat=np.array([0.9866, -0.1359, 0.0904, 0.0]),
                   half=np.array([0.040, 0.042, 0.012])),
}


def quat_rot_inv_wxyz(q, v):
    # rotate vector v by the inverse of quaternion q (w,x,y,z)
    w, x, y, z = q
    # inverse = conjugate for unit quat
    qc = np.array([w, -x, -y, -z])
    return quat_rot_wxyz(qc, v)


def quat_rot_wxyz(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return (2*(u@v)*u + (w*w - u@u)*v + 2*w*np.cross(u, v))


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def to_palm_local(cp_world, body_pos, body_quat_xyzw, geom):
    # world -> body -> geom-local. Returns 3-vec in the palm face frame.
    bq = quat_xyzw_to_wxyz(body_quat_xyzw)
    v_body = quat_rot_inv_wxyz(bq, cp_world - body_pos)
    v_geom = quat_rot_inv_wxyz(geom["quat"], v_body - geom["pos"])
    return v_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default="output/crow_telemetry.npz")
    ap.add_argument("--out", default="output/crow_palm_pressure.png")
    ap.add_argument("--fzmin", type=float, default=2.0)
    args = ap.parse_args()

    d = np.load(args.telemetry)
    names = [str(x) for x in d["body_names"]]
    pos = d["body_pos"]; rot = d["body_rot"]
    cp = d["contact_pt"]; grd = d["ground_force"]
    if np.all(np.isnan(cp)):
        raise SystemExit("contact_pt is all-NaN: re-collect with enable_contact_points "
                         "(collect_crow_telemetry injects it automatically).")
    T = pos.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, hand in enumerate(["L_Hand", "R_Hand"]):
        bi = names.index(hand); geom = PALM[hand]
        loc, fz, tt = [], [], []
        for t in range(T):
            w = max(0.0, grd[t, bi, 2])
            c = cp[t, bi]
            if w > args.fzmin and np.all(np.isfinite(c)):
                v = to_palm_local(c, pos[t, bi], rot[t, bi], geom)
                loc.append(v[:2]); fz.append(w); tt.append(t)
        hx, hy = geom["half"][0], geom["half"][1]

        ax_tr = axes[0, col]; ax_hm = axes[1, col]
        for ax in (ax_tr, ax_hm):
            ax.add_patch(Rectangle((-hx, -hy), 2*hx, 2*hy, fill=False, ec="black", lw=1.5))
            ax.plot(0, 0, "k+", ms=9)  # palm-geom center
            ax.set_aspect("equal"); ax.set_xlabel("palm local x (m)  [finger axis]")
            ax.set_ylabel("palm local y (m)")
            ax.set_xlim(-hx*1.5, hx*1.5); ax.set_ylim(-hy*1.5, hy*1.5)

        if len(loc) == 0:
            ax_tr.set_title("{}: never loaded > {:.0f} N".format(hand, args.fzmin))
            continue
        loc = np.array(loc); fz = np.array(fz)
        mean_xy = (loc * fz[:, None]).sum(0) / fz.sum()

        sc = ax_tr.scatter(loc[:, 0], loc[:, 1], c=np.array(tt)*float(d["dt"]),
                           cmap="viridis", s=20)
        ax_tr.plot(*mean_xy, "r*", ms=16)
        plt.colorbar(sc, ax=ax_tr, label="time (s)", fraction=0.046)
        ax_tr.set_title("{} load-centroid path (red* = Fz-weighted mean)".format(hand))

        H, xe, ye = np.histogram2d(loc[:, 0], loc[:, 1], bins=24,
                                   range=[[-hx*1.5, hx*1.5], [-hy*1.5, hy*1.5]],
                                   weights=fz)
        ax_hm.imshow(H.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                     aspect="equal", cmap="inferno")
        ax_hm.plot(*mean_xy, "c*", ms=16)
        frac_x = mean_xy[0] / hx
        ax_hm.set_title("{} Fz-weighted load density  (mean x={:+.0f}% of half-width)"
                        .format(hand, 100*frac_x))

    fig.suptitle("Per-palm load distribution (palm face frame; +x = fingers)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
