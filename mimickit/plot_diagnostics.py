"""Plot per-foot ground reaction forces and joint torques saved by DeepMimicEnv.

Run:
    python mimickit/plot_diagnostics.py output/diagnostics/diagnostics_<tag>.pt

Reads the .pt produced by DeepMimicEnv._save_logs (when log_ground_contact_forces
or log_joint_torques is enabled in the env yaml) and renders a multi-panel figure
with per-foot vertical/horizontal GRF traces and grouped joint-torque traces.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


FOOT_BODIES = ["right_foot", "left_foot"]
TORQUE_GROUPS = {
    "Hip (R)": ["right_hip_x", "right_hip_y", "right_hip_z"],
    "Knee (R)": ["right_knee"],
    "Ankle (R)": ["right_ankle_x", "right_ankle_y", "right_ankle_z"],
    "Hip (L)": ["left_hip_x", "left_hip_y", "left_hip_z"],
    "Knee (L)": ["left_knee"],
    "Ankle (L)": ["left_ankle_x", "left_ankle_y", "left_ankle_z"],
}


def _flatten_env(tensor, env_id):
    """Pull env_id slice from a [T, num_envs, ...] tensor."""
    if tensor.ndim < 2:
        return tensor
    return tensor[:, env_id]


def _time_axis(payload, num_steps, env_id):
    if "time" in payload:
        return _flatten_env(payload["time"], env_id).numpy()
    dt = payload.get("timestep") or (1.0 / 30.0)
    return np.arange(num_steps) * dt


def _plot_grf(ax_z, ax_xy, payload, env_id):
    grf = payload.get("ground_contact_forces")
    if grf is None:
        ax_z.set_title("(no GRF logged)")
        ax_xy.set_title("(no GRF logged)")
        return
    body_names = payload["body_names"]
    grf_env = _flatten_env(grf, env_id).numpy()  # [T, num_bodies, 3]
    t = _time_axis(payload, grf_env.shape[0], env_id)

    for foot in FOOT_BODIES:
        if foot not in body_names:
            continue
        bid = body_names.index(foot)
        ax_z.plot(t, grf_env[:, bid, 2], label=foot)
        ax_xy.plot(t, np.linalg.norm(grf_env[:, bid, :2], axis=-1), label=foot)

    ax_z.set_ylabel("Vertical GRF (N)")
    ax_z.set_title("Per-foot vertical GRF")
    ax_z.legend(loc="upper right")
    ax_z.grid(alpha=0.3)

    ax_xy.set_ylabel("|Horizontal GRF| (N)")
    ax_xy.set_xlabel("Time (s)")
    ax_xy.set_title("Per-foot horizontal GRF magnitude")
    ax_xy.legend(loc="upper right")
    ax_xy.grid(alpha=0.3)


def _plot_torques(axes, payload, env_id):
    torques = payload.get("joint_torques")
    if torques is None:
        for ax in axes.flat:
            ax.set_title("(no torques logged)")
        return
    dof_names = payload["dof_names"]
    tq_env = _flatten_env(torques, env_id).numpy()  # [T, dof]
    t = _time_axis(payload, tq_env.shape[0], env_id)

    flat_axes = list(axes.flat)
    for ax, (group_name, dof_list) in zip(flat_axes, TORQUE_GROUPS.items()):
        for d in dof_list:
            if d not in dof_names:
                continue
            ax.plot(t, tq_env[:, dof_names.index(d)], label=d)
        ax.set_title(group_name)
        ax.set_ylabel("Torque (Nm)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    for ax in flat_axes[-2:]:
        ax.set_xlabel("Time (s)")


def _summary(payload, env_id):
    grf = payload.get("ground_contact_forces")
    body_names = payload["body_names"]
    if grf is None:
        return
    grf_env = _flatten_env(grf, env_id).numpy()
    print("\n=== Summary (env {}) ===".format(env_id))
    for foot in FOOT_BODIES:
        if foot not in body_names:
            continue
        bid = body_names.index(foot)
        fz = grf_env[:, bid, 2]
        peak = float(fz.max())
        mean_active = float(fz[fz > 50.0].mean()) if np.any(fz > 50.0) else 0.0
        contact_frac = float((fz > 50.0).mean())
        # Crude impulse: integrate vertical GRF in N*s.
        dt = payload.get("timestep") or (1.0 / 30.0)
        impulse = float(np.trapz(np.clip(fz, 0.0, None), dx=dt))
        print("{:>12}: peak={:7.1f} N  mean_active={:6.1f} N  contact_frac={:.2f}  "
              "vert_impulse={:7.2f} N*s".format(foot, peak, mean_active, contact_frac, impulse))

    torques = payload.get("joint_torques")
    if torques is not None:
        tq = _flatten_env(torques, env_id).numpy()
        rms = np.sqrt(np.mean(tq * tq, axis=0))
        order = np.argsort(rms)[::-1][:5]
        dof_names = payload["dof_names"]
        print("Top-5 RMS joint torques:")
        for idx in order:
            print("  {:>20}: {:6.1f} Nm".format(dof_names[idx], rms[idx]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", nargs="?",
                        default="output/diagnostics/diagnostics_humanoid_run_default.pt")
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to write the figure instead of showing it.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print("Log file not found: {}".format(args.log_file), file=sys.stderr)
        sys.exit(1)

    payload = torch.load(args.log_file, map_location="cpu", weights_only=False)
    print("Loaded {}: keys={}".format(args.log_file, list(payload.keys())))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 3)
    ax_z = fig.add_subplot(gs[0, :])
    ax_xy = fig.add_subplot(gs[1, :], sharex=ax_z)
    torque_axes = np.array([
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])],
        [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])],
    ])

    _plot_grf(ax_z, ax_xy, payload, args.env_id)
    _plot_torques(torque_axes, payload, args.env_id)
    fig.suptitle(os.path.basename(args.log_file))
    fig.tight_layout()

    _summary(payload, args.env_id)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        fig.savefig(args.save, dpi=120)
        print("Wrote {}".format(args.save))
    else:
        plt.show()


if __name__ == "__main__":
    main()
