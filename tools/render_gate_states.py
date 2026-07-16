"""Render oracle-PASS (and gate-pass) calibration states for the eyeball
sign-off (Yoga_edge_framework_v3.md §3.2 / critique A1: render before freezing
gate.yaml — B's policy can rescue ugly states, so the eye must see what the
gate admits). CPU-only: skeletons come from the harvested body_pos.

Grid: rows = {oracle-PASS+gate-PASS (admitted), oracle-PASS+gate-REJECT
(conservative), oracle-FAIL+gate-PASS (the FALSE POSITIVES — should be ~empty)}.

Usage:
  env_isaaclab/bin/python tools/render_gate_states.py \
      --node_dir output/yoga_nodes_v2/crow_lt_comfwd
Output: <node_dir>/gate_states.png
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
sys.path.insert(0, os.path.join(REPO, "tools"))
os.chdir(REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibrate_gate as cg

CHAR_XML = "data/assets/smpl/smpl_boxhands.xml"


def skeleton_edges(body_names):
    idx = {n: i for i, n in enumerate(body_names)}
    edges = []
    def walk(elem, parent):
        for child in elem.findall("body"):
            name = child.get("name")
            if parent in idx and name in idx:
                edges.append((idx[parent], idx[name]))
            walk(child, name)
    world = ET.parse(CHAR_XML).getroot().find("worldbody")
    for b in world.findall("body"):
        walk(b, b.get("name"))
    return edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", required=True)
    p.add_argument("--per_row", type=int, default=8)
    p.add_argument("--rand_seed", type=int, default=42)
    args = p.parse_args()

    node = os.path.basename(args.node_dir)
    blob = torch.load(os.path.join(args.node_dir, "hold_states.pt"), map_location="cpu")
    orc = torch.load(os.path.join(args.node_dir, "hold_states.pt.oracle.pt"), map_location="cpu")
    gate = yaml.safe_load(open(os.path.join(args.node_dir, "gate.yaml")))
    sel = orc["sel"]
    sub = {k: (blob[k][sel] if torch.is_tensor(blob[k]) and blob[k].shape[:1] == blob["root_pos"].shape[:1]
               else blob[k]) for k in blob}
    verdict = orc["verdict"]

    ch = cg.channels(sub, blob["hold_ref"], gate["required_contacts"],
                     gate["forbidden_contacts"],
                     gate["load_share_alpha"] * blob["meta"]["char_weight"])
    g = dict(pose_mean=gate["theta_pose_mean"], pose_max=gate["theta_pose_max"],
             up_z_tol=gate["up_z_tol"], margin_min=gate["com_margin_min"],
             com_speed=gate["com_speed_max"], dof_speed=gate["dof_speed_max"])
    ok = cg.apply_gate(ch, g, gate["up_z"], gate["load_share_alpha"] * blob["meta"]["char_weight"])

    rows = [("oracle-PASS & gate-PASS (admitted)", verdict & ok),
            ("oracle-PASS & gate-REJECT (conservative)", verdict & ~ok),
            ("oracle-FAIL & gate-PASS (FALSE POSITIVE!)", ~verdict & ok)]
    bn = blob["meta"]["body_names"]
    edges = skeleton_edges(bn)
    bp = sub["body_pos"].numpy()
    rng = np.random.default_rng(args.rand_seed)

    fig, axes = plt.subplots(len(rows), 1, figsize=(2.0 * args.per_row, 3.2 * len(rows)))
    for ax, (title, mask) in zip(axes, rows):
        ids = mask.nonzero(as_tuple=False).flatten().numpy()
        take = rng.choice(ids, size=min(args.per_row, len(ids)), replace=False) if len(ids) else []
        for k, s in enumerate(take):
            pts = bp[s]
            span = pts[:, :2] - pts[:, :2].mean(0)
            dirv = np.linalg.svd(span)[2][0]
            x = span @ dirv + k * 1.4
            for a, b in edges:
                ax.plot([x[a], x[b]], [pts[a, 2], pts[b, 2]], "k-", lw=1.2)
            ax.plot(x, pts[:, 2], "o", ms=2, color="tab:blue")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{title} — {int(mask.sum())} states")
        ax.set_aspect("equal")
    fig.suptitle(f"{node} gate calibration states")
    fig.tight_layout()
    out = os.path.join(args.node_dir, "gate_states.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  (rows: " + " | ".join(f"{t}: {int(m.sum())}" for t, m in rows) + ")")


if __name__ == "__main__":
    main()
