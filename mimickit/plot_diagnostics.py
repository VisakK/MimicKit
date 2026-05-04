"""Plot diagnostics saved by DeepMimicEnv during test rollouts.

Run:
    python mimickit/plot_diagnostics.py output/diagnostics/diagnostics_<tag>.pt
    python mimickit/plot_diagnostics.py <log> --save_dir output/figs

Reads the .pt produced by DeepMimicEnv._save_logs (when log_ground_contact_forces
or log_joint_torques is enabled in the env yaml) and renders five focused
figures:
    1. Ground reaction forces (per-foot vertical & horizontal)
    2. Center of mass velocity (timeseries + average horizontal speed)
    3. Joint angles (Hip/Knee/Ankle, R/L) — in degrees
    4. Joint torques (Hip/Knee/Ankle, R/L)
    5. Stride length, stride frequency, cost of transport (averages in title)

Older logs without `body_pos` / `char_mass` will show "missing data" panels for
the derived metrics; rerun test mode after the env update to populate them.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


FOOT_BODIES = ["right_foot", "left_foot"]
# Joint-name prefixes. DeepMimicEnv._save_logs writes multi-DOF joints as
# "<joint>_0", "<joint>_1", ... and 1-DOF joints as just "<joint>". A prefix
# match handles both transparently.
JOINT_GROUPS = {
    "Hip (R)": "right_hip",
    "Knee (R)": "right_knee",
    "Ankle (R)": "right_ankle",
    "Hip (L)": "left_hip",
    "Knee (L)": "left_knee",
    "Ankle (L)": "left_ankle",
}

GRAVITY = 9.81
CONTACT_THRESHOLD_N = 50.0  # vertical GRF threshold for "in contact"
COT_MIN_SPEED = 0.5         # m/s, clamp for CoT denominator


# ---------- shared helpers ----------

def _flatten_env(tensor, env_id):
    if tensor.ndim < 2:
        return tensor
    return tensor[:, env_id]


def _time_axis(payload, num_steps, env_id):
    if "time" in payload:
        t = _flatten_env(payload["time"], env_id).numpy()
        if t.shape[0] == num_steps:
            return _stitch_episodes(t, payload)
        # Older logs accumulated time on every _update_info while GRF/
        # torques skipped empty-id resets, so payload["time"] can be ~2x
        # the data length. Fall back to a synthesized axis.
        print("warning: payload['time'] has {} entries but data has {}; "
              "falling back to dt * arange".format(t.shape[0], num_steps),
              file=sys.stderr)
    dt = payload.get("timestep") or (1.0 / 30.0)
    return np.arange(num_steps) * dt


def _stitch_episodes(t, payload):
    """Make t monotonic across episode resets.

    The env logs `_time_buf`, which resets to 0 whenever an env resets
    (every episode during a test rollout, since `test_episodes` is
    typically > 1). Plotting that raw t produces a backward line from
    end-of-episode to start-of-next-episode that visually looks like
    "first and last point connected". We convert each negative diff into
    a single dt step so the axis becomes continuous; per-episode trends
    still read correctly because the within-episode diffs are unchanged.
    """
    if t.size <= 1:
        return t
    diffs = np.diff(t)
    if not np.any(diffs < 0):
        return t
    dt_default = float(payload.get("timestep") or (1.0 / 30.0))
    fwd = np.where(diffs >= 0, diffs, dt_default)
    cont = np.concatenate([[float(t[0])], float(t[0]) + np.cumsum(fwd)])
    print("info: stitched {} episode reset(s) in payload['time'] into a "
          "monotonic axis [{:.2f}, {:.2f}] s".format(
              int(np.sum(diffs < 0)), cont[0], cont[-1]),
          file=sys.stderr)
    return cont


def _com_trajectory(payload, env_id):
    """Return (t, com[T,3]) using body_pos[:,0] (root) as a COM proxy when
    per-body masses are not available, or None if body_pos was not logged."""
    body_pos = payload.get("body_pos")
    if body_pos is None:
        return None, None
    bp_env = _flatten_env(body_pos, env_id).numpy()  # [T, n_bodies, 3]
    t = _time_axis(payload, bp_env.shape[0], env_id)
    if "body_masses" in payload:
        masses = np.asarray(payload["body_masses"]).reshape(-1)
        masses = masses / masses.sum()
        com = np.einsum("tbk,b->tk", bp_env, masses)
    else:
        # Root as proxy. For humanoid running this is within a few cm of
        # the true COM since most mass is in the torso.
        com = bp_env[:, 0, :]
    return t, com


def _stride_events(grf_z, threshold=CONTACT_THRESHOLD_N):
    """Indices of heel-strike events: first sample of each new contact bout."""
    in_contact = grf_z > threshold
    starts = []
    prev = False
    for i, c in enumerate(in_contact):
        if c and not prev:
            starts.append(i)
        prev = c
    return np.array(starts, dtype=int)


def _foot_stride_metrics(grf_env, body_pos_env, body_names, t):
    """Per foot: (lengths[m], durations[s]) between consecutive heel strikes."""
    out = {}
    for foot in FOOT_BODIES:
        if foot not in body_names:
            continue
        bid = body_names.index(foot)
        strikes = _stride_events(grf_env[:, bid, 2])
        if strikes.size < 2:
            out[foot] = (np.array([]), np.array([]))
            continue
        if body_pos_env is not None:
            foot_xy = body_pos_env[:, bid, :2]
            lengths = np.linalg.norm(np.diff(foot_xy[strikes], axis=0), axis=-1)
        else:
            lengths = np.array([])
        durations = np.diff(t[strikes])
        out[foot] = (lengths, durations)
    return out


# ---------- per-figure plotters ----------

def _plot_grf(ax_z, ax_xy, payload, env_id):
    grf = payload.get("ground_contact_forces")
    if grf is None:
        for ax in (ax_z, ax_xy):
            ax.set_title("(no GRF logged)")
        return
    body_names = payload["body_names"]
    grf_env = _flatten_env(grf, env_id).numpy()
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


def _plot_com_velocity(ax, payload, env_id):
    """Plot COM velocity components and horizontal speed. Returns
    (t, horiz_speed) so downstream plots can reuse it for CoT."""
    t, com = _com_trajectory(payload, env_id)
    if com is None:
        ax.set_title("(no body_pos logged — rerun test mode)")
        return None, None
    com_vel = np.gradient(com, t, axis=0)
    horiz_speed = np.linalg.norm(com_vel[:, :2], axis=-1)

    ax.plot(t, horiz_speed, label="|v_horiz|", color="black", linewidth=1.5)
    ax.plot(t, com_vel[:, 0], label="vx", alpha=0.55)
    ax.plot(t, com_vel[:, 1], label="vy", alpha=0.55)
    ax.plot(t, com_vel[:, 2], label="vz", alpha=0.55)
    avg_v = float(np.mean(horiz_speed))
    ax.set_title("COM velocity  (avg |v_horiz| = {:.2f} m/s)".format(avg_v))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    return t, horiz_speed


def _plot_joint_groups(axes, payload, env_id, key, ylabel, scale_fn=None):
    """Generic per-joint-group plotter for dof_pos / joint_torques."""
    data = payload.get(key)
    if data is None:
        for ax in axes.flat:
            ax.set_title("(no {} logged)".format(key))
        return
    dof_names = payload["dof_names"]
    arr = _flatten_env(data, env_id).numpy()
    if scale_fn is not None:
        arr = scale_fn(arr)
    t = _time_axis(payload, arr.shape[0], env_id)

    flat_axes = list(axes.flat)
    for ax, (group_name, prefix) in zip(flat_axes, JOINT_GROUPS.items()):
        matched = False
        for i, d in enumerate(dof_names):
            if d == prefix or d.startswith(prefix + "_"):
                ax.plot(t, arr[:, i], label=d)
                matched = True
        ax.set_title(group_name)
        ax.set_ylabel(ylabel)
        if matched:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Time (s)")


def _plot_stride(ax_len, ax_freq, payload, env_id):
    grf = payload.get("ground_contact_forces")
    body_pos = payload.get("body_pos")
    if grf is None:
        for ax in (ax_len, ax_freq):
            ax.set_title("(no GRF logged)")
        return
    if body_pos is None:
        ax_len.set_title("(no body_pos — rerun to get stride length)")
    body_names = payload["body_names"]
    grf_env = _flatten_env(grf, env_id).numpy()
    bp_env = _flatten_env(body_pos, env_id).numpy() if body_pos is not None else None
    t = _time_axis(payload, grf_env.shape[0], env_id)
    metrics = _foot_stride_metrics(grf_env, bp_env, body_names, t)

    all_lengths, all_freqs = [], []
    for foot, (lengths, durations) in metrics.items():
        if lengths.size > 0:
            ax_len.plot(np.arange(1, lengths.size + 1), lengths, "o-", label=foot)
            all_lengths.extend(lengths.tolist())
        if durations.size > 0:
            freqs = 1.0 / durations
            ax_freq.plot(np.arange(1, freqs.size + 1), freqs, "o-", label=foot)
            all_freqs.extend(freqs.tolist())

    avg_len = "avg = {:.2f} m".format(np.mean(all_lengths)) if all_lengths else "no strides detected"
    avg_freq = "avg = {:.2f} Hz".format(np.mean(all_freqs)) if all_freqs else "no strides detected"
    ax_len.set_title("Stride length per stride  ({})".format(avg_len))
    ax_len.set_xlabel("Stride #")
    ax_len.set_ylabel("Length (m)")
    if all_lengths:
        ax_len.legend(fontsize=8, loc="best")
    ax_len.grid(alpha=0.3)

    ax_freq.set_title("Stride frequency per stride  ({})".format(avg_freq))
    ax_freq.set_xlabel("Stride #")
    ax_freq.set_ylabel("Frequency (Hz)")
    if all_freqs:
        ax_freq.legend(fontsize=8, loc="best")
    ax_freq.grid(alpha=0.3)


def _plot_cot(ax, payload, env_id, com_t=None, com_speed=None):
    torques = payload.get("joint_torques")
    dof_pos = payload.get("dof_pos")
    char_mass = payload.get("char_mass")
    missing = []
    if torques is None: missing.append("joint_torques")
    if dof_pos is None: missing.append("dof_pos")
    if char_mass is None: missing.append("char_mass")
    if missing:
        ax.set_title("(missing for CoT: {})".format(", ".join(missing)))
        return
    tq_env = _flatten_env(torques, env_id).numpy()
    dp_env = _flatten_env(dof_pos, env_id).numpy()
    t = _time_axis(payload, tq_env.shape[0], env_id)

    # dof_vel from finite diff. np.gradient handles uneven spacing.
    dof_vel = np.gradient(dp_env, t, axis=0)
    mech_power = np.sum(np.abs(tq_env * dof_vel), axis=-1)

    if com_speed is None or com_t is None or com_speed.shape[0] != t.shape[0]:
        com_t, com = _com_trajectory(payload, env_id)
        if com is None:
            ax.set_title("(no body_pos for CoT speed)")
            return
        com_vel = np.gradient(com, com_t, axis=0)
        com_speed = np.linalg.norm(com_vel[:, :2], axis=-1)

    v = np.clip(com_speed, COT_MIN_SPEED, None)
    cot = mech_power / (float(char_mass) * GRAVITY * v)

    ax.plot(t, cot, color="C2", linewidth=1.2)
    avg_cot = float(np.mean(cot))
    ax.set_title("Cost of transport  (avg = {:.3f}, dimensionless)".format(avg_cot))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CoT")
    ax.grid(alpha=0.3)


def _summary(payload, env_id):
    grf = payload.get("ground_contact_forces")
    body_names = payload["body_names"]
    print("\n=== Summary (env {}) ===".format(env_id))
    if grf is not None:
        grf_env = _flatten_env(grf, env_id).numpy()
        for foot in FOOT_BODIES:
            if foot not in body_names:
                continue
            bid = body_names.index(foot)
            fz = grf_env[:, bid, 2]
            peak = float(fz.max())
            mean_active = float(fz[fz > CONTACT_THRESHOLD_N].mean()) if np.any(fz > CONTACT_THRESHOLD_N) else 0.0
            contact_frac = float((fz > CONTACT_THRESHOLD_N).mean())
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

    if "char_mass" in payload:
        print("Character mass: {:.2f} kg".format(float(payload["char_mass"])))


# ---------- entry point ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", nargs="?",
                        default="output/diagnostics/diagnostics_humanoid_run_default.pt")
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="If set, save figures to this directory instead of showing them.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print("Log file not found: {}".format(args.log_file), file=sys.stderr)
        sys.exit(1)

    payload = torch.load(args.log_file, map_location="cpu", weights_only=False)
    print("Loaded {}: keys={}".format(args.log_file, list(payload.keys())))

    base_name = os.path.splitext(os.path.basename(args.log_file))[0]
    figures = []

    # 1. GRF
    fig_grf, (ax_grf_z, ax_grf_xy) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    _plot_grf(ax_grf_z, ax_grf_xy, payload, args.env_id)
    fig_grf.suptitle("{} — Ground reaction forces".format(base_name))
    fig_grf.tight_layout()
    figures.append(("grf", fig_grf))

    # 2. COM velocity (also returns horiz speed to share with CoT panel)
    fig_com, ax_com = plt.subplots(1, 1, figsize=(12, 4))
    com_t, com_speed = _plot_com_velocity(ax_com, payload, args.env_id)
    fig_com.suptitle("{} — Center of mass velocity".format(base_name))
    fig_com.tight_layout()
    figures.append(("com", fig_com))

    # 3. Joint angles (rad → deg for readability)
    fig_ang, ax_ang = plt.subplots(2, 3, figsize=(14, 7))
    _plot_joint_groups(ax_ang, payload, args.env_id,
                       key="dof_pos", ylabel="Angle (deg)",
                       scale_fn=lambda a: np.degrees(a))
    fig_ang.suptitle("{} — Joint angles".format(base_name))
    fig_ang.tight_layout()
    figures.append(("angles", fig_ang))

    # 4. Joint torques
    fig_tq, ax_tq = plt.subplots(2, 3, figsize=(14, 7))
    _plot_joint_groups(ax_tq, payload, args.env_id,
                       key="joint_torques", ylabel="Torque (Nm)")
    fig_tq.suptitle("{} — Joint torques".format(base_name))
    fig_tq.tight_layout()
    figures.append(("torques", fig_tq))

    # 5. Stride length / frequency / CoT
    fig_stride = plt.figure(figsize=(14, 8))
    gs = fig_stride.add_gridspec(2, 2)
    ax_len = fig_stride.add_subplot(gs[0, 0])
    ax_freq = fig_stride.add_subplot(gs[0, 1])
    ax_cot = fig_stride.add_subplot(gs[1, :])
    _plot_stride(ax_len, ax_freq, payload, args.env_id)
    _plot_cot(ax_cot, payload, args.env_id, com_t=com_t, com_speed=com_speed)
    fig_stride.suptitle("{} — Stride metrics & cost of transport".format(base_name))
    fig_stride.tight_layout()
    figures.append(("stride_cot", fig_stride))

    _summary(payload, args.env_id)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for tag, fig in figures:
            out = os.path.join(args.save_dir, "{}_{}.png".format(base_name, tag))
            fig.savefig(out, dpi=120)
            print("Wrote {}".format(out))
    else:
        plt.show()


if __name__ == "__main__":
    main()
