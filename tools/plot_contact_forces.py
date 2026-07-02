"""Req 1: per-body contact force vs time, split into GROUND reaction and
BODY-TO-BODY (self) contact. Body-to-body force = net_contact - ground_contact,
which cleanly isolates e.g. the knee-on-upper-arm shelf (the knees touch only the
arms, not the floor). Reads output/crow_telemetry.npz (see collect_crow_telemetry.py).

Run: env_isaaclab/bin/python tools/plot_contact_forces.py   # any python w/ numpy+mpl
"""
import sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", default="output/crow_telemetry.npz")
    p.add_argument("--out", default="output/crow_contact_forces.png")
    p.add_argument("--thresh", type=float, default=5.0, help="N; a body is shown if its force ever exceeds this")
    args = p.parse_args()

    d = np.load(args.telemetry)
    names = [str(x) for x in d["body_names"]]
    dt = float(d["dt"]); W = float(d["char_weight"])
    grd = d["ground_force"]; net = d["net_force"]
    b2b = net - grd
    T = grd.shape[0]
    t = np.arange(T) * dt

    gmag = np.linalg.norm(grd, axis=-1)   # [T,B]
    bmag = np.linalg.norm(b2b, axis=-1)   # [T,B]

    g_bodies = [i for i in range(len(names)) if gmag[:, i].max() > args.thresh]
    b_bodies = [i for i in range(len(names)) if bmag[:, i].max() > args.thresh]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    cmap = plt.get_cmap("tab10")
    for k, i in enumerate(g_bodies):
        ax1.plot(t, gmag[:, i], label=names[i], color=cmap(k % 10), lw=1.6)
    ax1.axhline(W, ls="--", c="gray", lw=1, label="body weight ({:.0f} N)".format(W))
    ax1.set_title("Ground reaction force per body")
    ax1.set_ylabel("|F| ground (N)"); ax1.grid(ls="dotted"); ax1.legend(fontsize=8, ncol=2, loc="upper right")

    for k, i in enumerate(b_bodies):
        ax2.plot(t, bmag[:, i], label=names[i], color=cmap(k % 10), lw=1.6)
    ax2.set_title("Body-to-body (self) contact force per body  [net - ground]  "
                  "-- e.g. knee-on-arm shelf")
    ax2.set_ylabel("|F| body-body (N)"); ax2.set_xlabel("time (s)")
    ax2.grid(ls="dotted"); ax2.legend(fontsize=8, ncol=2, loc="upper right")

    fig.suptitle("Crow contact forces vs time", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("ground-contact bodies:", [names[i] for i in g_bodies])
    print("body-to-body bodies  :", [names[i] for i in b_bodies])
    print("saved", args.out)


if __name__ == "__main__":
    main()
