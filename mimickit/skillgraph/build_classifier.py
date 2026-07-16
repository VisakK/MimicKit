"""Fit a skill's initiation classifier and produce the section-4 calibration
result (classifier vs raw critic at predicting hold-success).

This is the Stage-0 go/no-go deliverable (spec section 14): on held-out states,
the classifier should predict "will the skill hold from here" with AUC >= ~0.75
AND clearly beat the raw PPO critic, with a better-calibrated reliability curve.
The raw critic value was recorded on the real policy observation at collection
time, so this is a fair, in-distribution comparison.

Usage (CPU, on the dataset.pt written by collect_skill.py):
  env_isaaclab/bin/python mimickit/skillgraph/build_classifier.py --skill_id handstand
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


def build(args):
    sdir = sio.skill_dir(args.skill_id, args.out_root)
    ds = torch.load(os.path.join(sdir, "dataset.pt"))
    feats = ds["features"].float()
    labels = ds["label"].float()
    critic = ds["critic_value"].float().numpy()

    # held-out split shared by classifier and critic so the comparison is fair.
    g = torch.Generator().manual_seed(args.seed)
    N = feats.shape[0]
    perm = torch.randperm(N, generator=g)
    n_val = int(args.val_frac * N)
    val_idx = perm[:n_val]

    # ---- pose-discriminative hard negatives (spec B): OTHER skills' terminal
    # states as label-0, so this classifier learns to REJECT other poses. The
    # v1 classifiers cross-fired (handstand and scorpion both read "inverted,
    # hands down") because each was trained only on its own falls as negatives.
    # Added to TRAIN only (fit_classifier keeps the val split within-skill, so
    # the section-4 calibration vs the raw critic remains honest).
    hard_neg, hard_neg_skills = [], []
    if (not args.no_hard_negatives):
        others = [s for s in sio.list_skills(args.out_root) if s != args.skill_id]
        if (args.hard_negatives):
            others = [s for s in args.hard_negatives if s != args.skill_id]
        for s in others:
            tp = os.path.join(sio.skill_dir(s, args.out_root), "terminal_states.pt")
            if (not os.path.exists(tp)):
                continue
            tf = torch.load(tp).get("features", None)
            if (tf is None or tf.shape[0] == 0 or tf.shape[-1] != feats.shape[-1]):
                continue
            hard_neg.append(tf.float()); hard_neg_skills.append(s)
            print("  hard negatives from {:<12s}: {} states".format(s, tf.shape[0]))
    extra_neg = torch.cat(hard_neg, dim=0) if hard_neg else None

    model, info = ic.fit_classifier(feats, labels, hidden=tuple(args.hidden),
                                    epochs=args.epochs, lr=args.lr,
                                    val_frac=args.val_frac, seed=args.seed,
                                    extra_neg_features=extra_neg)
    # Recompute on the EXACT same val split for an apples-to-apples critic compare
    xva, yva = feats[val_idx], labels[val_idx].numpy()
    cva = critic[val_idx.numpy()]
    pva = model.prob(xva).numpy()

    clf_auc = ic.roc_auc(pva, yva)
    clf_ece = ic.expected_calibration_error(pva, yva)
    clf_brier = ic.brier(pva, yva)
    base = ic.critic_baseline_metrics(cva, yva)

    print("\n=== section-4 calibration: {} ===".format(args.skill_id))
    print("  classifier : AUC={:.3f}  ECE={:.3f}  Brier={:.3f}".format(clf_auc, clf_ece, clf_brier))
    print("  raw critic : AUC={:.3f}  ECE={:.3f}  Brier={:.3f}".format(base["auc"], base["ece"], base["brier"]))
    go = (clf_auc >= 0.75) and (clf_auc >= base["auc"])
    print("  GO/NO-GO (AUC>=0.75 and >= critic): {}".format("GO" if go else "NO-GO"))

    # ---- cross-pose rejection (spec B): C_skill should score OTHER poses' held
    # terminal states LOW. This is the property the v1 classifiers lacked and the
    # reason the transitions reward-hacked the gate.
    cross = {}
    for s, tf in zip(hard_neg_skills, hard_neg):
        pr = model.prob(tf).numpy()
        cross[s] = round(float(np.median(pr)), 3)
        print("  reject {:<12s}: median p_init={:.3f} (want << {:.2f})".format(s, cross[s], args.threshold))

    # --- reliability figure (the load-bearing section-4 plot) ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    cc = np.array(info["reliability"]["pred"]); ce = np.array(info["reliability"]["emp"])
    ax[0].plot(cc, ce, "o-", label="classifier (ECE={:.3f})".format(clf_ece))
    bc = np.array(base["reliability"]["pred"]); be = np.array(base["reliability"]["emp"])
    ax[0].plot(bc, be, "s-", label="raw critic (ECE={:.3f})".format(base["ece"]))
    ax[0].set_xlabel("predicted hold prob"); ax[0].set_ylabel("empirical hold rate")
    ax[0].set_title("{}: reliability".format(args.skill_id)); ax[0].legend(); ax[0].grid(alpha=0.3)

    # score histograms split by outcome
    ax[1].hist(pva[yva == 1], bins=30, alpha=0.5, density=True, label="classifier|held")
    ax[1].hist(pva[yva == 0], bins=30, alpha=0.5, density=True, label="classifier|fell")
    ax[1].set_xlabel("classifier p_init"); ax[1].set_ylabel("density")
    ax[1].set_title("AUC clf={:.3f} vs critic={:.3f}".format(clf_auc, base["auc"]))
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(sdir, "calibration.png")
    fig.savefig(fig_path, dpi=110); plt.close(fig)

    ic.save_classifier(model, os.path.join(sdir, "classifier.pt"),
                       meta={"feature_names": ds["feature_names"], "skill_id": args.skill_id,
                             "val_auc": clf_auc, "threshold": args.threshold})
    summary = {"skill_id": args.skill_id, "n_states": int(N), "pos_frac": float((labels == 1).float().mean()),
               "classifier": {"auc": clf_auc, "ece": clf_ece, "brier": clf_brier},
               "raw_critic": {"auc": base["auc"], "ece": base["ece"], "brier": base["brier"]},
               "n_hard_neg": int(info.get("n_hard_neg", 0)),
               "cross_pose_reject_median": cross,
               "go_no_go": "GO" if go else "NO-GO"}
    with open(os.path.join(sdir, "calibration.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # update the card's competence threshold
    card = sio.read_card(args.skill_id, args.out_root) or {}
    card.setdefault("competence_region", {})
    card["competence_region"]["threshold"] = args.threshold
    card["competence_region"]["classifier_auc"] = round(clf_auc, 3)
    sio.write_card(card, args.out_root)
    print("  wrote {}  +  calibration.png/json".format(os.path.join(sdir, "classifier.pt")))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill_id", required=True)
    p.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "..", "..", "skills"))
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    # Pose-discriminative training (spec B). By default, every OTHER skill that
    # has terminal_states.pt contributes hard negatives. Pass an explicit list
    # to --hard_negatives to restrict, or --no_hard_negatives to reproduce the
    # old (cross-firing) within-skill-only classifier.
    p.add_argument("--hard_negatives", nargs="*", default=None)
    p.add_argument("--no_hard_negatives", action="store_true")
    args = p.parse_args()
    args.out_root = os.path.normpath(args.out_root)
    build(args)


if __name__ == "__main__":
    main()
