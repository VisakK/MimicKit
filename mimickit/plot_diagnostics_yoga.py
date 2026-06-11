"""Plot diagnostics saved by DeepMimicEnv for SMPL yoga / scorpion rollouts.

Run:
    python mimickit/plot_diagnostics_yoga.py output/diagnostics/diagnostics_smpl_scorpion.pt
    python mimickit/plot_diagnostics_yoga.py <log> --save_dir output/figs_yoga

Reads the .pt produced by DeepMimicEnv._save_logs (enable
`log_ground_contact_forces` and `log_joint_torques` in the env yaml) and renders
figures tailored to a static, hand-supported pose such as the scorpion
(vrischikasana) handstand back-bend on the SMPL skeleton:

    1. Ground reaction forces per *contacting* body  (vertical & horizontal)
    2. Left/right support-hand load asymmetry — the usual failure mode where
       the policy leans onto one hand
    3. Center-of-mass motion: height, velocity, and the medio-lateral lean of
       the COM relative to the line joining the two support hands
    4. Joint angles      (per SMPL body region, degrees)
    5. Joint torques     (per SMPL body region, Nm)
    6. Joint velocities  (per SMPL body region, rad/s)

Unlike plot_diagnostics.py (hard-coded for the humanoid foot/leg names), this
script keys everything off the SMPL body / dof names stored in the payload.
Contacting bodies are auto-detected from the logged forces, so it adapts to
whichever body parts touch the ground for a given pose. Joint velocities use
the logged `dof_vel`; older logs without it fall back to a finite difference of
`dof_pos`. Center of mass uses the logged per-body `body_masses`; older logs
fall back to the root body as a (poor, for an inverted pose) proxy.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


GRAVITY = 9.81
CONTACT_DETECT_N = 10.0      # peak |F| over the rollout to call a body "a contact"
CONTACT_THRESHOLD_N = 50.0   # instantaneous vertical GRF for "in contact at time t"
MAX_CONTACT_BODIES = 10      # cap how many bodies we draw on the GRF figure
EPS = 1e-8

# Preferred left/right support pairs for the asymmetry / lean panels. The first
# pair whose *both* bodies are actually in contact wins; otherwise we fall back
# to the highest-load L/R pair discovered in the data (see _pick_lr_pair).
LR_SUPPORT_PAIRS = [
    ("R_Hand", "L_Hand"),
    ("R_Wrist", "L_Wrist"),
    ("R_Elbow", "L_Elbow"),
    ("R_Toe", "L_Toe"),
    ("R_Ankle", "L_Ankle"),
]

# SMPL joint regions. DeepMimicEnv._save_logs writes multi-DOF joints as
# "<joint>_0", "<joint>_1", ... and 1-DOF joints as just "<joint>", so a prefix
# match handles both. L/R pairs share an axis so asymmetry is visible directly.
JOINT_GROUPS = {
    "Shoulder (R/L)": ["R_Shoulder", "L_Shoulder"],
    "Elbow (R/L)":    ["R_Elbow", "L_Elbow"],
    "Wrist (R/L)":    ["R_Wrist", "L_Wrist"],
    "Hip (R/L)":      ["R_Hip", "L_Hip"],
    "Knee (R/L)":     ["R_Knee", "L_Knee"],
    "Spine / Back":   ["Torso", "Spine", "Chest", "Neck"],
}

# Right side warm, left side cool, everything else a neutral palette. Index is
# the DOF number within a joint so the three axes of a spherical joint are
# distinguishable.
_R_COLORS = ["#d62728", "#ff7f0e", "#8c564b"]
_L_COLORS = ["#1f77b4", "#17becf", "#2ca02c"]
_OTHER_COLORS = ["#9467bd", "#7f7f7f", "#bcbd22", "#e377c2"]


def _side_color(name, k):
    if name.startswith("R_"):
        return _R_COLORS[k % len(_R_COLORS)]
    if name.startswith("L_"):
        return _L_COLORS[k % len(_L_COLORS)]
    return _OTHER_COLORS[k % len(_OTHER_COLORS)]


# ---------- shared helpers (mirrors plot_diagnostics.py) ----------

def _flatten_env(tensor, env_id):
    if tensor.ndim < 2:
        return tensor
    return tensor[:, env_id]


def _time_axis(payload, num_steps, env_id):
    if "time" in payload:
        t = _flatten_env(payload["time"], env_id).numpy()
        if t.shape[0] == num_steps:
            return _stitch_episodes(t, payload)
        print("warning: payload['time'] has {} entries but data has {}; "
              "falling back to dt * arange".format(t.shape[0], num_steps),
              file=sys.stderr)
    dt = payload.get("timestep") or (1.0 / 30.0)
    return np.arange(num_steps) * dt


def _stitch_episodes(t, payload):
    """Make t monotonic across episode resets (each reset zeroes _time_buf)."""
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
    """Return (t, com[T,3]) using the logged per-body masses for a true
    mass-weighted COM, or the root body as a proxy if masses are absent."""
    body_pos = payload.get("body_pos")
    if body_pos is None:
        return None, None
    bp_env = _flatten_env(body_pos, env_id).numpy()  # [T, n_bodies, 3]
    t = _time_axis(payload, bp_env.shape[0], env_id)
    if "body_masses" in payload:
        masses = np.asarray(payload["body_masses"]).reshape(-1).astype(np.float64)
        n = min(masses.shape[0], bp_env.shape[1])
        w = masses[:n] / max(masses[:n].sum(), EPS)
        com = np.einsum("tbk,b->tk", bp_env[:, :n, :], w)
    else:
        print("warning: no body_masses in payload; using root body as COM proxy "
              "(inaccurate for an inverted pose) — rerun test mode to populate it.",
              file=sys.stderr)
        com = bp_env[:, 0, :]
    return t, com


# ---------- contact-body discovery ----------

def _contacting_bodies(grf_env, body_names, detect_n=CONTACT_DETECT_N,
                       max_bodies=MAX_CONTACT_BODIES):
    """List of (name, body_id, peak_force) for bodies whose peak |F| over the
    rollout exceeds detect_n, sorted by peak descending."""
    if grf_env.shape[0] == 0:
        return []
    fmag = np.linalg.norm(grf_env, axis=-1)  # [T, n_bodies]
    out = []
    n = min(len(body_names), grf_env.shape[1])
    for bid in range(n):
        peak = float(fmag[:, bid].max())
        if peak > detect_n:
            out.append((body_names[bid], bid, peak))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:max_bodies]


def _pick_lr_pair(body_names, contacts):
    """Choose the dominant (right, left) support pair among contacting bodies.

    Prefers the canonical pairs in LR_SUPPORT_PAIRS (hands first); otherwise
    finds any R_/L_ bodies that share a suffix and picks the highest-load one.
    Returns (right_name, left_name) or None.
    """
    contact_peak = {n: p for n, _bid, p in contacts}
    for rname, lname in LR_SUPPORT_PAIRS:
        if rname in contact_peak and lname in contact_peak:
            return (rname, lname)

    by_suffix = {}
    for name, _bid, peak in contacts:
        if name.startswith("R_") or name.startswith("L_"):
            by_suffix.setdefault(name[2:], {})[name[0]] = (name, peak)
    best = None
    for sides in by_suffix.values():
        if "R" in sides and "L" in sides:
            score = sides["R"][1] + sides["L"][1]
            if best is None or score > best[0]:
                best = (score, sides["R"][0], sides["L"][0])
    if best is not None:
        return (best[1], best[2])
    return None


# ---------- per-figure plotters ----------

def _plot_grf_per_body(ax_z, ax_xy, payload, env_id, contacts):
    grf = payload.get("ground_contact_forces")
    if grf is None:
        for ax in (ax_z, ax_xy):
            ax.set_title("(no ground_contact_forces logged)")
        return
    grf_env = _flatten_env(grf, env_id).numpy()  # [T, n_bodies, 3]
    t = _time_axis(payload, grf_env.shape[0], env_id)

    if not contacts:
        ax_z.set_title("(no body exceeded {:.0f} N — nothing in contact?)".format(CONTACT_DETECT_N))
    for k, (name, bid, _peak) in enumerate(contacts):
        col = _side_color(name, 0) if name[:2] in ("R_", "L_") else _OTHER_COLORS[k % len(_OTHER_COLORS)]
        ax_z.plot(t, grf_env[:, bid, 2], label=name, color=col, linewidth=1.0)
        ax_xy.plot(t, np.linalg.norm(grf_env[:, bid, :2], axis=-1), label=name, color=col, linewidth=1.0)

    ax_z.set_ylabel("Vertical GRF (N)")
    ax_z.set_title("Per-body vertical ground reaction force (contacting bodies)")
    if contacts:
        ax_z.legend(fontsize=7, loc="upper right", ncol=2)
    ax_z.grid(alpha=0.3)

    ax_xy.set_ylabel("|Horizontal GRF| (N)")
    ax_xy.set_xlabel("Time (s)")
    ax_xy.set_title("Per-body horizontal ground reaction force magnitude")
    if contacts:
        ax_xy.legend(fontsize=7, loc="upper right", ncol=2)
    ax_xy.grid(alpha=0.3)


def _plot_support_asymmetry(ax_force, ax_asym, payload, env_id, pair):
    grf = payload.get("ground_contact_forces")
    if grf is None or pair is None:
        for ax in (ax_force, ax_asym):
            ax.set_title("(no GRF or no left/right support pair found)")
        return None
    rname, lname = pair
    body_names = payload["body_names"]
    grf_env = _flatten_env(grf, env_id).numpy()
    t = _time_axis(payload, grf_env.shape[0], env_id)
    fR = grf_env[:, body_names.index(rname), 2]
    fL = grf_env[:, body_names.index(lname), 2]

    ax_force.plot(t, fR, label=rname, color=_side_color(rname, 0), linewidth=1.1)
    ax_force.plot(t, fL, label=lname, color=_side_color(lname, 0), linewidth=1.1)
    ax_force.set_ylabel("Vertical GRF (N)")
    ax_force.set_title("Support-pair vertical load: {} vs {}".format(rname, lname))
    ax_force.legend(loc="upper right")
    ax_force.grid(alpha=0.3)

    total = fR + fL
    in_contact = total > CONTACT_THRESHOLD_N
    asym = np.where(in_contact, (fR - fL) / np.clip(total, EPS, None), np.nan)
    ax_asym.plot(t, asym, color="black", linewidth=1.2)
    ax_asym.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax_asym.set_ylim(-1.05, 1.05)
    ax_asym.set_ylabel("(R - L) / (R + L)")
    ax_asym.set_xlabel("Time (s)")

    mean_asym = float(np.nanmean(asym)) if np.any(in_contact) else float("nan")
    mean_fR = float(fR[in_contact].mean()) if np.any(in_contact) else 0.0
    mean_fL = float(fL[in_contact].mean()) if np.any(in_contact) else 0.0
    ax_asym.set_title("Load asymmetry  (mean = {:+.2f};  +1 = all on {}, -1 = all on {})"
                      .format(mean_asym, rname, lname))
    ax_asym.grid(alpha=0.3)
    return {"pair": pair, "mean_asym": mean_asym, "mean_fR": mean_fR, "mean_fL": mean_fL}


def _plot_com_motion(ax_pos, ax_vel, ax_lean, payload, env_id, pair):
    t, com = _com_trajectory(payload, env_id)
    if com is None:
        for ax in (ax_pos, ax_vel, ax_lean):
            ax.set_title("(no body_pos logged — rerun test mode)")
        return None

    ax_pos.plot(t, com[:, 0], label="x", alpha=0.8)
    ax_pos.plot(t, com[:, 1], label="y", alpha=0.8)
    ax_pos.plot(t, com[:, 2], label="z (height)", color="black", linewidth=1.5)
    ax_pos.set_ylabel("COM position (m)")
    ax_pos.set_title("Center of mass position  (mean height = {:.3f} m)".format(float(np.mean(com[:, 2]))))
    ax_pos.legend(fontsize=8, loc="upper right")
    ax_pos.grid(alpha=0.3)

    com_vel = np.gradient(com, t, axis=0)
    speed = np.linalg.norm(com_vel, axis=-1)
    ax_vel.plot(t, speed, label="|v|", color="black", linewidth=1.5)
    ax_vel.plot(t, com_vel[:, 0], label="vx", alpha=0.55)
    ax_vel.plot(t, com_vel[:, 1], label="vy", alpha=0.55)
    ax_vel.plot(t, com_vel[:, 2], label="vz", alpha=0.55)
    ax_vel.set_ylabel("COM velocity (m/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_title("Center of mass velocity  (mean |v| = {:.3f} m/s)".format(float(np.mean(speed))))
    ax_vel.legend(fontsize=8, loc="upper right")
    ax_vel.grid(alpha=0.3)

    lean_stats = _plot_com_lean(ax_lean, payload, env_id, com, t, pair)
    return {"mean_height": float(np.mean(com[:, 2])),
            "mean_speed": float(np.mean(speed)),
            "lean": lean_stats}


def _plot_com_lean(ax, payload, env_id, com, t, pair):
    """Signed medio-lateral offset of the COM from the midpoint of the two
    support hands, projected onto the line joining them. Positive = leaning
    toward the right-side support. The shaded band is the support base; COM
    outside it means the pose is tipping past a hand."""
    body_pos = payload.get("body_pos")
    if body_pos is None or pair is None:
        ax.set_title("(no body_pos / support pair for COM lean)")
        return None
    body_names = payload["body_names"]
    rname, lname = pair
    if rname not in body_names or lname not in body_names:
        ax.set_title("(support pair {} / {} not in body_names)".format(rname, lname))
        return None

    bp = _flatten_env(body_pos, env_id).numpy()  # [T, n_bodies, 3]
    pR = bp[:, body_names.index(rname), :2]
    pL = bp[:, body_names.index(lname), :2]
    mid = 0.5 * (pR + pL)
    axis_vec = pR - pL
    base = np.linalg.norm(axis_vec, axis=-1)            # hand-to-hand distance
    u = axis_vec / np.clip(base[:, None], EPS, None)    # unit L->R medio-lateral axis
    s = np.sum((com[:, :2] - mid) * u, axis=-1)         # signed COM offset toward R
    half_base = 0.5 * base

    ax.fill_between(t, -half_base, half_base, color="green", alpha=0.12,
                    label="support base ({}<->{})".format(lname, rname))
    ax.plot(t, half_base, color="green", lw=0.8, ls=":")
    ax.plot(t, -half_base, color="green", lw=0.8, ls=":")
    ax.plot(t, s, color="purple", linewidth=1.3, label="COM lateral offset")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Lateral offset (m)")
    ax.set_xlabel("Time (s)")

    mean_lean = float(np.mean(s))
    lean_ratio = float(np.mean(np.abs(s)) / max(np.mean(half_base), EPS))
    frac_outside = float(np.mean(np.abs(s) > np.clip(half_base, EPS, None)))
    toward = rname if mean_lean >= 0 else lname
    ax.set_title("COM medio-lateral lean  (mean {:+.3f} m toward {}; "
                 "|offset|/half-base = {:.2f}; {:.0%} of time outside base)"
                 .format(mean_lean, toward, lean_ratio, frac_outside))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    return {"mean_lean": mean_lean, "lean_ratio": lean_ratio, "frac_outside": frac_outside}


def _plot_joint_groups(axes, payload, env_id, ylabel, key=None, data_arr=None,
                       scale_fn=None):
    """Per-DOF timeseries grouped by SMPL body region. Either reads payload[key]
    or uses a pre-flattened data_arr ([T, n_dof] numpy, for finite-diff vel)."""
    flat_axes = list(axes.flat)
    if data_arr is None:
        data = payload.get(key)
        if data is None:
            for ax in flat_axes:
                ax.set_title("(no {} logged)".format(key))
            return
        arr = _flatten_env(data, env_id).numpy()
    else:
        arr = data_arr
    if scale_fn is not None:
        arr = scale_fn(arr)
    dof_names = payload["dof_names"]
    t = _time_axis(payload, arr.shape[0], env_id)

    for ax, (group_name, prefixes) in zip(flat_axes, JOINT_GROUPS.items()):
        matched = False
        for j, prefix in enumerate(prefixes):
            k = 0
            for i, d in enumerate(dof_names):
                if i >= arr.shape[1]:
                    break
                if d == prefix or d.startswith(prefix + "_"):
                    if prefix[:2] in ("R_", "L_"):
                        col = _side_color(prefix, k)
                    else:
                        col = _OTHER_COLORS[(j + k) % len(_OTHER_COLORS)]
                    ax.plot(t, arr[:, i], label=d, color=col, linewidth=1.0, alpha=0.9)
                    k += 1
                    matched = True
        ax.set_title(group_name)
        ax.set_ylabel(ylabel)
        if matched:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.grid(alpha=0.3)
    for ax in flat_axes[-3:]:
        ax.set_xlabel("Time (s)")


def _dof_vel_array(payload, env_id):
    """Flattened joint velocities [T, n_dof]; prefer logged dof_vel, else a
    finite difference of dof_pos. Returns (arr, is_estimated)."""
    if payload.get("dof_vel") is not None:
        return _flatten_env(payload["dof_vel"], env_id).numpy(), False
    dof_pos = payload.get("dof_pos")
    if dof_pos is None:
        return None, False
    dp = _flatten_env(dof_pos, env_id).numpy()
    t = _time_axis(payload, dp.shape[0], env_id)
    return np.gradient(dp, t, axis=0), True


# ---------- summary ----------

def _summary(payload, env_id, contacts, pair, com_stats, asym_stats):
    print("\n=== Yoga diagnostics summary (env {}) ===".format(env_id))
    if "char_mass" in payload:
        print("Character mass: {:.2f} kg  (weight {:.1f} N)".format(
            float(payload["char_mass"]), float(payload["char_mass"]) * GRAVITY))

    grf = payload.get("ground_contact_forces")
    if grf is not None and contacts:
        grf_env = _flatten_env(grf, env_id).numpy()
        dt = payload.get("timestep") or (1.0 / 30.0)
        print("Contacting bodies (peak |F| > {:.0f} N):".format(CONTACT_DETECT_N))
        for name, bid, peak in contacts:
            fz = grf_env[:, bid, 2]
            active = fz > CONTACT_THRESHOLD_N
            mean_active = float(fz[active].mean()) if np.any(active) else 0.0
            impulse = float(np.trapz(np.clip(fz, 0.0, None), dx=dt))
            print("  {:>11}: peak={:7.1f} N  mean_active={:6.1f} N  "
                  "contact_frac={:.2f}  vert_impulse={:7.2f} N*s"
                  .format(name, peak, mean_active, float(active.mean()), impulse))

    if asym_stats is not None:
        rname, lname = asym_stats["pair"]
        print("Support asymmetry [{} vs {}]: mean (R-L)/(R+L) = {:+.3f}  "
              "(mean R={:.1f} N, L={:.1f} N)".format(
                  rname, lname, asym_stats["mean_asym"],
                  asym_stats["mean_fR"], asym_stats["mean_fL"]))

    if com_stats is not None:
        print("COM: mean height={:.3f} m  mean speed={:.3f} m/s".format(
            com_stats["mean_height"], com_stats["mean_speed"]))
        lean = com_stats.get("lean")
        if lean is not None:
            print("COM lean: mean offset={:+.3f} m  |offset|/half-base={:.2f}  "
                  "outside base {:.0%} of the time".format(
                      lean["mean_lean"], lean["lean_ratio"], lean["frac_outside"]))

    torques = payload.get("joint_torques")
    if torques is not None:
        tq = _flatten_env(torques, env_id).numpy()
        rms = np.sqrt(np.mean(tq * tq, axis=0))
        dof_names = payload["dof_names"]
        order = np.argsort(rms)[::-1][:5]
        print("Top-5 RMS joint torques:")
        for idx in order:
            if idx < len(dof_names):
                print("  {:>14}: {:6.1f} Nm".format(dof_names[idx], rms[idx]))


# ---------- entry point ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file", nargs="?",
                        default="output/diagnostics/diagnostics_smpl_scorpion_rew_shaped.pt")
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="If set, save figures here instead of showing them.")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print("Log file not found: {}".format(args.log_file), file=sys.stderr)
        sys.exit(1)

    payload = torch.load(args.log_file, map_location="cpu", weights_only=False)
    print("Loaded {}: keys={}".format(args.log_file, list(payload.keys())))

    base_name = os.path.splitext(os.path.basename(args.log_file))[0]
    body_names = payload.get("body_names", [])

    # Discover which bodies actually touch the ground, and the dominant L/R
    # support pair, so every downstream panel adapts to the pose.
    contacts, pair = [], None
    grf = payload.get("ground_contact_forces")
    if grf is not None and body_names:
        grf_env = _flatten_env(grf, args.env_id).numpy()
        contacts = _contacting_bodies(grf_env, body_names)
        pair = _pick_lr_pair(body_names, contacts)
        print("Contacting bodies: {}".format([n for n, _b, _p in contacts]))
        print("Support pair for asymmetry/lean: {}".format(pair))

    figures = []

    # 1. Per-body ground reaction forces.
    fig_grf, (ax_grf_z, ax_grf_xy) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    _plot_grf_per_body(ax_grf_z, ax_grf_xy, payload, args.env_id, contacts)
    fig_grf.suptitle("{} — Ground reaction forces (per body)".format(base_name))
    fig_grf.tight_layout()
    figures.append(("grf_per_body", fig_grf))

    # 2. Left/right support asymmetry.
    fig_asym, (ax_af, ax_aa) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    asym_stats = _plot_support_asymmetry(ax_af, ax_aa, payload, args.env_id, pair)
    fig_asym.suptitle("{} — Support-hand load asymmetry".format(base_name))
    fig_asym.tight_layout()
    figures.append(("hand_asymmetry", fig_asym))

    # 3. Center of mass motion (+ medio-lateral lean).
    fig_com = plt.figure(figsize=(13, 9))
    gs = fig_com.add_gridspec(2, 2)
    ax_cp = fig_com.add_subplot(gs[0, 0])
    ax_cv = fig_com.add_subplot(gs[0, 1])
    ax_cl = fig_com.add_subplot(gs[1, :])
    com_stats = _plot_com_motion(ax_cp, ax_cv, ax_cl, payload, args.env_id, pair)
    fig_com.suptitle("{} — Center of mass motion".format(base_name))
    fig_com.tight_layout()
    figures.append(("com_motion", fig_com))

    # 4. Joint angles (rad -> deg).
    fig_ang, ax_ang = plt.subplots(2, 3, figsize=(15, 8))
    _plot_joint_groups(ax_ang, payload, args.env_id, ylabel="Angle (deg)",
                       key="dof_pos", scale_fn=np.degrees)
    fig_ang.suptitle("{} — Joint angles".format(base_name))
    fig_ang.tight_layout()
    figures.append(("joint_angles", fig_ang))

    # 5. Joint torques.
    fig_tq, ax_tq = plt.subplots(2, 3, figsize=(15, 8))
    _plot_joint_groups(ax_tq, payload, args.env_id, ylabel="Torque (Nm)",
                       key="joint_torques")
    fig_tq.suptitle("{} — Joint torques".format(base_name))
    fig_tq.tight_layout()
    figures.append(("joint_torques", fig_tq))

    # 6. Joint velocities (logged dof_vel, or finite-diff fallback).
    fig_vel, ax_vel = plt.subplots(2, 3, figsize=(15, 8))
    vel_arr, vel_est = _dof_vel_array(payload, args.env_id)
    _plot_joint_groups(ax_vel, payload, args.env_id, ylabel="Velocity (rad/s)",
                       data_arr=vel_arr)
    suffix = " (finite-diff of dof_pos)" if vel_est else ""
    fig_vel.suptitle("{} — Joint velocities{}".format(base_name, suffix))
    fig_vel.tight_layout()
    figures.append(("joint_velocities", fig_vel))

    _summary(payload, args.env_id, contacts, pair, com_stats, asym_stats)

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
