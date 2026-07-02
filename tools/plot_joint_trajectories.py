"""Req 4: per-joint trajectories (DoF angles) and joint torques over time. SMPL
has 69 DoFs (23 spherical joints x x/y/z). Produces a small-multiples grid (one
panel per joint, its 3 axes overlaid) for angles and for torques, plus a
mean-|torque| bar chart showing which joints do the most work in the hold.

Reads output/crow_telemetry.npz. Run: env_isaaclab/bin/python tools/plot_joint_trajectories.py
"""
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def group_by_joint(dof_names):
    # "L_Shoulder_x" -> joint "L_Shoulder", axis "x". Falls back to per-dof groups.
    groups = OrderedDict()
    for i, n in enumerate(dof_names):
        if "_" in n and n.rsplit("_", 1)[1] in ("x", "y", "z"):
            joint, axis = n.rsplit("_", 1)
        else:
            joint, axis = n, ""
        groups.setdefault(joint, []).append((i, axis))
    return groups


def grid_plot(t, data, groups, dof_names, title, ylabel, out):
    n = len(groups); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 2.2*rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    cmap = {"x": "tab:red", "y": "tab:green", "z": "tab:blue", "": "k"}
    for ax in axes[n:]:
        ax.axis("off")
    for k, (joint, members) in enumerate(groups.items()):
        ax = axes[k]
        for (i, axis) in members:
            ax.plot(t, data[:, i], lw=1.2, color=cmap.get(axis, "k"), label=axis or dof_names[i])
        ax.set_title(joint, fontsize=9); ax.grid(ls="dotted", lw=0.5)
        ax.tick_params(labelsize=7)
        if len(members) > 1:
            ax.legend(fontsize=6, ncol=3, loc="upper right", handlelength=1)
    for ax in axes[n-cols if n-cols > 0 else 0:n]:
        ax.set_xlabel("time (s)", fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.supylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default="output/crow_telemetry.npz")
    ap.add_argument("--prefix", default="output/crow_joint")
    args = ap.parse_args()

    d = np.load(args.telemetry)
    dof_names = [str(x) for x in d["dof_names"]]
    dt = float(d["dt"])
    ang = d["dof_pos"]; tau = d["dof_torque"]
    T = ang.shape[0]; t = np.arange(T) * dt
    groups = group_by_joint(dof_names)

    grid_plot(t, ang, groups, dof_names, "Joint angles (DoF trajectories)",
              "angle (rad)", args.prefix + "_angles.png")
    grid_plot(t, tau, groups, dof_names, "Joint torques", "torque (N*m)",
              args.prefix + "_torques.png")

    # mean |torque| per joint -> which joints work hardest in the hold
    eff = OrderedDict()
    for joint, members in groups.items():
        idx = [i for i, _ in members]
        eff[joint] = float(np.mean(np.abs(tau[:, idx])))
    order = sorted(eff, key=eff.get, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(order)), [eff[j] for j in order], color="tab:purple")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("mean |torque| (N*m)"); ax.grid(ls="dotted", axis="y")
    ax.set_title("Joint effort in the crow hold (mean |torque| per joint)")
    fig.tight_layout(); fig.savefig(args.prefix + "_effort.png", dpi=130)
    print("saved", args.prefix + "_effort.png")
    print("top effort joints:", order[:6])


if __name__ == "__main__":
    main()
