"""Headless verification of a trained v2 HOLD node.

The attempt-2 post-mortem rule: proxy scalars are smoke tests, the rendered
motion is the sign-off. This tool produces BOTH, without a display:

  1. Rolls the trained policy for one deterministic episode via
     tools/collect_crow_telemetry.py (pose-agnostic collector; boots Isaac)
     -> output/yoga_nodes_v2/<node>/telemetry.npz
  2. Computes a stability report against the pose's annotation fingerprint:
     survival time, support-body loading, fall-set / penalized-toe contact,
     root wobble, inversion (chest-pelvis up_z proxy)
     -> output/yoga_nodes_v2/<node>/check.json (+ console)
  3. Renders a strip figure -- skeleton side/front views at sampled times with
     contact bodies highlighted + force traces -- for the eyeball sign-off
     -> output/yoga_nodes_v2/<node>/strip.png

Usage (repo root, env python):
  tools/check_hold_node.py --node handstand [--skip_collect] [--steps 360]
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/visakii/Documents/moves/env_isaaclab/bin/python"
QUEUE = os.path.join(REPO, "tools/hold_nodes_queue.txt")
CHAR_XML = os.path.join(REPO, "data/assets/smpl/smpl_boxhands.xml")

# Annotation fingerprints (data/clip_annotations, corrected): expected chest-
# pelvis up_z (advisory, tol 0.4) and the bodies that must carry load.
EXPECT = {
    "handstand":     dict(up_z=-1.0, support=["L_Hand", "R_Hand"]),
    "headstand":     dict(up_z=-0.95, support=["L_Elbow", "R_Elbow"]),
    "pincha":        dict(up_z=-0.97, support=["L_Elbow", "R_Elbow"]),
    "scorpion":      dict(up_z=-0.99, support=["L_Hand", "R_Hand"]),
    "shoulderstand": dict(up_z=-0.98, support=["L_Shoulder", "R_Shoulder"]),
    "plow":          dict(up_z=-0.96, support=["L_Shoulder", "R_Shoulder"]),
    "firefly":       dict(up_z=0.87, support=["L_Hand", "R_Hand"]),
    "eight_angle":   dict(up_z=0.39, support=["L_Hand", "R_Hand"]),
    "side_crow":     dict(up_z=-0.18, support=["L_Hand", "R_Hand"]),
    "koundinyasana": dict(up_z=-0.95, support=["L_Hand", "R_Hand"]),
    "tree":          dict(up_z=0.93, support=["R_Toe", "R_Ankle"]),
    "warrior3":      dict(up_z=0.08, support=["R_Toe", "R_Ankle"]),
    "natarajasana":  dict(up_z=0.68, support=["R_Toe", "R_Ankle"]),
    "eagle":         dict(up_z=0.98, support=["L_Toe", "L_Ankle"]),
    "half_moon":     dict(up_z=-0.01, support=["R_Toe", "R_Ankle"]),
    "big_toe":       dict(up_z=0.94, support=["L_Toe", "L_Ankle"]),
    "warrior2":      dict(up_z=0.99, support=["L_Toe", "R_Toe"]),
    "triangle":      dict(up_z=0.55, support=["L_Toe", "R_Toe"]),
    "chair":         dict(up_z=0.89, support=["L_Ankle", "R_Ankle"]),
    "boat":          dict(up_z=0.32, support=["Pelvis"]),
    "side_plank":    dict(up_z=0.33, support=["L_Hand", "L_Toe", "L_Ankle"]),
    "camel":         dict(up_z=0.89, support=["L_Knee", "R_Knee"]),
    "wheel":         dict(up_z=0.0, support=["L_Hand", "R_Hand", "L_Toe", "R_Toe"]),
    "downdog":       dict(up_z=-0.30, support=["L_Hand", "R_Hand", "L_Toe", "R_Toe"]),
    # v3.1 connectors (2026-07-05): up_z here is a fallback proxy estimate --
    # the regenerated ref_fingerprints.json entries take precedence.
    "plank":         dict(up_z=0.15, support=["L_Hand", "R_Hand", "L_Toe", "R_Toe"]),
    "dolphin_plank": dict(up_z=0.15, support=["L_Elbow", "R_Elbow", "L_Toe", "R_Toe"]),
    "malasana":      dict(up_z=0.91, support=["L_Ankle", "R_Ankle"]),
    "tadasana":      dict(up_z=0.97, support=["L_Ankle", "R_Ankle"]),
    # off-queue nodes (pass --env/--agent explicitly); up_z here is the
    # chest-pelvis PROXY (empirically anchored on the verified crow rollout)
    "crow":            dict(up_z=-0.54, support=["L_Hand", "R_Hand"]),
    "one_legged_crow": dict(up_z=-0.40, support=["L_Hand", "R_Hand"]),
}

# Calibrated reference fingerprints (tools/ref_fingerprints.py): the same
# chest-pelvis proxy computed on the trimmed reference clip itself. Preferred
# over the annotation-derived EXPECT up_z (different up definition for tucked
# poses: crow proxy -0.54 vs annotation root-up +0.13).
REF_FP = {}
_fp_path = os.path.join(REPO, "tools/hold_nodes_ref_fingerprints.json")
if os.path.exists(_fp_path):
    with open(_fp_path) as f:
        REF_FP = json.load(f)

CONTACT_N = 2.0   # N; a body is "in contact" above this vertical GRF
LOAD_N = 5.0      # N; a support body counts as loaded above this mean force


def queue_entry(node):
    with open(QUEUE) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            nid, env, agent, samples = line.split()
            if nid == node:
                return env, agent
    raise SystemExit(f"node '{node}' not found in {QUEUE}")


def skeleton_edges(body_names):
    """Parent-child edges from the MJCF body tree, as index pairs."""
    idx = {n: i for i, n in enumerate(body_names)}
    edges = []
    def walk(elem, parent):
        for child in elem.findall("body"):
            name = child.get("name")
            if parent in idx and name in idx:
                edges.append((idx[parent], idx[name]))
            walk(child, name)
    root = ET.parse(CHAR_XML).getroot()
    world = root.find("worldbody")
    for b in world.findall("body"):
        walk(b, b.get("name"))
    return edges


def analyze(node, npz_path, env_yaml):
    d = np.load(npz_path, allow_pickle=True)
    body_pos = d["body_pos"]          # [T, B, 3]
    gf = d["ground_force"]            # [T, B, 3]
    names = [str(n) for n in d["body_names"]]
    dt = float(d["dt"])
    T = body_pos.shape[0]
    bi = {n: i for i, n in enumerate(names)}

    # fall set + penalized toes from the env yaml (source of truth)
    import yaml
    with open(os.path.join(REPO, env_yaml)) as f:
        env_cfg = yaml.safe_load(f)["env"]
    fall = env_cfg.get("contact_bodies", [])
    toe_pen = env_cfg.get("toe_force_pen_bodies", []) \
        if env_cfg.get("reward_toe_force_pen_w", 0) > 0 else []
    episode_s = float(env_cfg.get("episode_length", 12.0))

    fz = gf[:, :, 2]                              # vertical GRF [T, B]
    exp = EXPECT[node]
    rep = dict(node=node, steps=T, held_s=round(T * dt, 2),
               episode_s=episode_s,
               survived=bool(T * dt >= min(episode_s - 0.5, 10.0)))

    up = body_pos[:, bi["Chest"], :] - body_pos[:, bi["Pelvis"], :]
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-8)
    rep["up_z_mean"] = round(float(up[:, 2].mean()), 3)
    ref = REF_FP.get(node, {})
    rep["up_z_expected"] = ref.get("up_z_proxy", exp["up_z"])
    rep["up_z_ok"] = bool(abs(rep["up_z_mean"] - rep["up_z_expected"]) < 0.35)
    if "root_h" in ref:
        rep["root_h_expected"] = ref["root_h"]

    sup = {}
    for b in exp["support"]:
        if b in bi:
            sup[b] = dict(mean_fz=round(float(fz[:, bi[b]].mean()), 1),
                          contact_frac=round(float((fz[:, bi[b]] > CONTACT_N).mean()), 3))
    rep["support"] = sup
    loaded = [b for b, v in sup.items() if v["mean_fz"] > LOAD_N]
    rep["support_loaded_ok"] = bool(len(loaded) >= max(1, len(sup) // 2 + 1))

    rep["fall_contacts"] = {
        b: round(float((fz[:, bi[b]] > CONTACT_N).mean()), 3)
        for b in fall if b in bi and (fz[:, bi[b]] > CONTACT_N).any()}
    rep["fall_ok"] = bool(not rep["fall_contacts"])

    if toe_pen:
        tp = [bi[b] for b in toe_pen if b in bi]
        rep["toe_pen_mean_fz"] = round(float(fz[:, tp].mean()), 2)
        rep["toe_pen_contact_frac"] = round(float((fz[:, tp] > CONTACT_N).any(axis=1).mean()), 3)
    # top-loaded bodies (for poses without an explicit cop set, compare by eye)
    mean_fz = fz.mean(axis=0)
    top = np.argsort(mean_fz)[::-1][:6]
    rep["top_loaded"] = {names[i]: round(float(mean_fz[i]), 1)
                         for i in top if mean_fz[i] > 1.0}

    pel = body_pos[:, bi["Pelvis"], :]
    rep["root_xy_wobble_m"] = round(float(pel[:, :2].std(axis=0).mean()), 4)
    rep["root_z_mean"] = round(float(pel[:, 2].mean()), 3)

    rep["verdict"] = ("PASS" if rep["survived"] and rep["up_z_ok"]
                      and rep["support_loaded_ok"] and rep["fall_ok"]
                      else "CHECK")
    return rep, (body_pos, fz, names, dt)


def render_strip(node, data, out_png, n_frames=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body_pos, fz, names, dt = data
    T = body_pos.shape[0]
    edges = skeleton_edges(names)
    ts = np.linspace(0, T - 1, n_frames).astype(int)

    fig, axes = plt.subplots(2, n_frames, figsize=(3.2 * n_frames, 7.5))
    for k, t in enumerate(ts):
        p = body_pos[t]
        contact = fz[t] > CONTACT_N
        for view, ax in ((0, axes[0, k]), (1, axes[1, k])):
            a = view  # 0 = x-z (side), 1 = y-z (front)
            for i, j in edges:
                ax.plot([p[i, a], p[j, a]], [p[i, 2], p[j, 2]],
                        "-", color="0.3", lw=2)
            ax.scatter(p[~contact, a], p[~contact, 2], s=12, c="tab:blue", zorder=3)
            if contact.any():
                ax.scatter(p[contact, a], p[contact, 2], s=40, c="red",
                           marker="s", zorder=4)
            hi = names.index("Head")
            ax.scatter(p[hi, a], p[hi, 2], s=60, c="orange", zorder=5)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_aspect("equal")
            ax.set_ylim(-0.1, 2.0)
            c = p[:, a].mean()
            ax.set_xlim(c - 1.0, c + 1.0)
            if view == 0:
                ax.set_title(f"t={t*dt:.1f}s")
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("side (x-z)")
    axes[1, 0].set_ylabel("front (y-z)")
    fig.suptitle(f"{node} hold -- red squares = ground contact (> {CONTACT_N:.0f} N), "
                 f"orange = head, floor at z=0", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node", required=True)
    p.add_argument("--env", default="", help="override env yaml (off-queue nodes)")
    p.add_argument("--agent", default="", help="override agent yaml (off-queue nodes)")
    p.add_argument("--model", default="")
    p.add_argument("--steps", type=int, default=360)
    p.add_argument("--warmup", type=int, default=10,
                   help="settle steps before recording; 0 = record from the RSI reset")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    if args.env and args.agent:
        env, agent = args.env, args.agent
    else:
        env, agent = queue_entry(args.node)
    out_dir = args.out_dir or os.path.join(REPO, "output/yoga_nodes_v2", args.node)
    os.makedirs(out_dir, exist_ok=True)
    model = args.model or os.path.join(out_dir, "model.pt")
    npz = os.path.join(out_dir, "telemetry.npz")

    if not args.skip_collect:
        r = subprocess.run(
            [PY, "tools/collect_crow_telemetry.py", "--env", env, "--agent", agent,
             "--model", model, "--out", npz, "--steps", str(args.steps),
             "--warmup", str(args.warmup)],
            cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
            raise SystemExit(f"telemetry collection failed for {args.node}")

    rep, data = analyze(args.node, npz, env)
    render_strip(args.node, data, os.path.join(out_dir, "strip.png"))
    with open(os.path.join(out_dir, "check.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print(f"strip -> {os.path.join(out_dir, 'strip.png')}")


if __name__ == "__main__":
    main()
