"""Zero-shot transition feasibility matrix for the skill graph (spec 11.3a, 12).

For every ordered pair (A -> B), score how far A's terminal states already fall
inside B's competence region by evaluating B's initiation classifier C_B on the
canonical features of A's saved terminal states. This is the LEARNABILITY gate
of Algorithm 11 (prioritise edges where p_init_B(S_A^term) is moderate, not ~0)
and the cheap, no-rollout half of the failure-diagnosis ladder (section 7.2):

  * p_init high  (median > zero_shot_thresh)  -> B likely already covers A's
        terminus; a handoff-timing sweep may give `zero_shot_ok` with no training.
  * p_init moderate (in [train_floor, zero_shot_thresh]) -> reachable but
        outside B's set: TRAIN a dedicated transition policy (RSI = S_A^term).
  * p_init ~0  (median < train_floor) -> likely dynamically far; candidate for an
        intermediate node A->C->B.

The diagonal (A->A) is a self-consistency check: B's classifier on B's own
terminal states should score high.

Usage (CPU):
  env_isaaclab/bin/python mimickit/skillgraph/feasibility.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import skillgraph.init_classifier as ic
import skillgraph.skill_io as sio


def classify_verdict(median_p, frac, zero_shot_thresh, train_floor):
    # Zero-shot feasibility only distinguishes "B's existing policy already
    # covers A's terminus" (a handoff sweep may suffice, no training) from "must
    # train a transition". A ~0 foothold does NOT mean untrainable -- a
    # transition policy is RSI'd from A's terminus and LEARNS to bridge the gap;
    # `needs_intermediate` is a *post-attempt* diagnosis (section 7.4), not a
    # feasibility call. frac (reachable fraction of S_A^term) ranks priority.
    if (median_p >= zero_shot_thresh):
        return "zero_shot_candidate"
    return "train_transition"


def run(args):
    skills = sio.list_skills(args.out_root)
    # only skills that have both a classifier and terminal states
    usable = []
    classifiers, terminals, feat_names = {}, {}, {}
    for s in skills:
        sdir = sio.skill_dir(s, args.out_root)
        clf_p = os.path.join(sdir, "classifier.pt")
        term_p = os.path.join(sdir, "terminal_states.pt")
        if (os.path.exists(clf_p) and os.path.exists(term_p)):
            model, meta = ic.load_classifier(clf_p)
            classifiers[s] = model
            terminals[s] = torch.load(term_p)
            feat_names[s] = meta.get("feature_names")
            usable.append(s)
    assert len(usable) >= 1, "no skills with classifier + terminal states under {}".format(args.out_root)
    print("skills with classifier + terminal states:", usable)

    # feature layouts must match across skills (same env-config family)
    ref_names = None
    for s in usable:
        if (feat_names[s] is not None):
            if (ref_names is None):
                ref_names = feat_names[s]
            else:
                assert feat_names[s] == ref_names, \
                    "feature layout mismatch between skills ({} vs ref)".format(s)

    n = len(usable)
    mat_mean = np.full((n, n), np.nan)
    mat_median = np.full((n, n), np.nan)
    mat_frac = np.full((n, n), np.nan)
    rows = []
    for i, A in enumerate(usable):
        featA = terminals[A]["features"].float()
        if (featA.shape[0] == 0):
            print("  [warn] {} has 0 terminal states; skipping as source".format(A))
            continue
        for j, B in enumerate(usable):
            p = classifiers[B].prob(featA).numpy()
            mat_mean[i, j] = float(np.mean(p))
            mat_median[i, j] = float(np.median(p))
            mat_frac[i, j] = float(np.mean(p >= args.threshold))
            if (A != B):
                verdict = classify_verdict(mat_median[i, j], mat_frac[i, j],
                                           args.zero_shot_thresh, args.train_floor)
                rows.append({"from": A, "to": B,
                             "p_init_mean": float(round(mat_mean[i, j], 3)),
                             "p_init_median": float(round(mat_median[i, j], 3)),
                             "frac_above_thresh": float(round(mat_frac[i, j], 3)),
                             "verdict": verdict})

    # ---- report ----
    print("\n=== zero-shot feasibility: median p_init_B(S_A^term) ===")
    print("rows = FROM (A terminus), cols = TO (B competence)")
    hdr = "{:>12s}".format("A\\B") + "".join("{:>12s}".format(b) for b in usable)
    print(hdr)
    for i, A in enumerate(usable):
        line = "{:>12s}".format(A)
        for j in range(n):
            v = mat_median[i, j]
            line += "{:>12s}".format("--" if np.isnan(v) else "{:.2f}".format(v))
        print(line)
    print("\n=== edge verdicts ===")
    for r in sorted(rows, key=lambda r: -r["p_init_median"]):
        print("  {:>10s} -> {:<10s}  median p_init={:.2f} mean={:.2f} frac>={:.2f}  => {}".format(
            r["from"], r["to"], r["p_init_median"], r["p_init_mean"],
            r["frac_above_thresh"], r["verdict"]))

    # ---- heatmap ----
    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.4 * n + 1.5))
    im = ax.imshow(mat_median, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(usable, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(usable)
    ax.set_xlabel("TO (B competence)"); ax.set_ylabel("FROM (A terminus)")
    ax.set_title("median p_init_B( S_A^term )")
    for i in range(n):
        for j in range(n):
            if (not np.isnan(mat_median[i, j])):
                ax.text(j, i, "{:.2f}".format(mat_median[i, j]), ha="center", va="center",
                        color="white" if mat_median[i, j] < 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(args.out_root, exist_ok=True)
    fig_path = os.path.join(args.out_root, "feasibility.png")
    fig.savefig(fig_path, dpi=120); plt.close(fig)

    # ---- write graph edges with the feasibility signal + verdict-derived status ----
    for r in rows:
        sio.update_edge(r["from"], r["to"], {
            "status": "untried",   # all confirmed/trained later by eval_transition
            "feasibility": {"p_init_mean": r["p_init_mean"], "p_init_median": r["p_init_median"],
                            "frac_above_thresh": r["frac_above_thresh"], "verdict": r["verdict"]},
        }, root=args.out_root)

    with open(os.path.join(args.out_root, "feasibility.json"), "w") as f:
        json.dump({"skills": usable, "median": mat_median.tolist(),
                   "mean": mat_mean.tolist(), "edges": rows}, f, indent=2)
    print("\nwrote {} + feasibility.json + graph edges".format(fig_path))
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "..", "..", "skills"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--zero_shot_thresh", type=float, default=0.6)
    p.add_argument("--train_floor", type=float, default=0.1)
    args = p.parse_args()
    args.out_root = os.path.normpath(args.out_root)
    run(args)


if __name__ == "__main__":
    main()
