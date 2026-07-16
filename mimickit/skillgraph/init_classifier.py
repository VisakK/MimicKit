"""Initiation-set classifier + calibration tooling (spec section 4).

A PPO critic is on-policy and over-optimistic off-distribution -- exactly at the
terminal states of skill A, which are off skill B's policy distribution. So the
handoff gate is NOT a raw-critic threshold but a calibrated two-class classifier
C_B(s) -> p_init trained from states where B succeeds (holds its pose) vs.
fails (falls / times out). This module:

  * defines the small MLP C_B and a standardizing feature wrapper,
  * fits it from a labeled dataset with class balancing,
  * provides the load-bearing calibration metrics (reliability curve, AUC,
    Brier, ECE) so we can show classifier vs raw-critic vs ensemble (the
    section-4 figure), all with numpy/torch only (no sklearn dependency).
"""
import numpy as np
import torch
import torch.nn as nn


class InitiationClassifier(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        # feature standardization buffers (set at fit time)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

    def forward(self, x):
        x = (x - self.feat_mean) / torch.clamp(self.feat_std, min=1e-6)
        return self.net(x).squeeze(-1)

    def prob(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def roc_auc(scores, labels):
    """Rank-based ROC-AUC (Mann-Whitney U). scores/labels: 1-D np arrays."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if (n_pos == 0 or n_neg == 0):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank_per_group = (start + cum + 1) / 2.0
    ranks = avg_rank_per_group[inv]
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def reliability_curve(probs, labels, n_bins=10):
    """Predicted vs empirical success per probability bin. Returns
    (bin_centers, predicted_mean, empirical_mean, counts)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, pred, emp, cnt = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if (m.sum() == 0):
            continue
        centers.append(0.5 * (lo + hi))
        pred.append(probs[m].mean())
        emp.append(labels[m].mean())
        cnt.append(int(m.sum()))
    return np.array(centers), np.array(pred), np.array(emp), np.array(cnt)


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    _, pred, emp, cnt = reliability_curve(probs, labels, n_bins)
    if (len(cnt) == 0):
        return float("nan")
    w = cnt / cnt.sum()
    return float(np.sum(w * np.abs(pred - emp)))


def brier(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def fit_classifier(features, labels, hidden=(128, 128), epochs=300, lr=1e-3,
                   weight_decay=1e-4, val_frac=0.2, device="cpu", seed=0,
                   verbose=True, extra_neg_features=None):
    """Fit C_B on (features [N,D], labels [N] in {0,1}). Returns (model, info)
    where info has held-out AUC/Brier/ECE + the reliability curve.

    extra_neg_features [M,D] (optional): hard negatives added to the TRAINING
    set only (label 0), e.g. OTHER skills' terminal states so C_B learns to
    reject other poses (spec B, pose-discriminative classifier). They are kept
    OUT of the val split so the section-4 within-skill calibration vs the raw
    critic stays an honest apples-to-apples comparison."""
    g = torch.Generator().manual_seed(seed)
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    N, D = features.shape

    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(val_frac * N))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    xtr, ytr = features[tr_idx], labels[tr_idx]
    xva, yva = features[val_idx], labels[val_idx]

    # Hard negatives (other poses) augment the train set only.
    n_extra_neg = 0
    if (extra_neg_features is not None and len(extra_neg_features) > 0):
        extra = torch.as_tensor(extra_neg_features, dtype=torch.float32)
        assert extra.shape[-1] == D, "hard-negative feature dim {} != {}".format(extra.shape[-1], D)
        xtr = torch.cat([xtr, extra], dim=0)
        ytr = torch.cat([ytr, torch.zeros(extra.shape[0], dtype=torch.float32)], dim=0)
        n_extra_neg = int(extra.shape[0])

    model = InitiationClassifier(D, hidden).to(device)
    model.feat_mean.copy_(xtr.mean(0))
    model.feat_std.copy_(xtr.std(0))

    # class balancing via pos_weight
    n_pos = float((ytr == 1).sum())
    n_neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xtr, ytr = xtr.to(device), ytr.to(device)
    xva, yva = xva.to(device), yva.to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logit = model(xtr)
        loss = loss_fn(logit, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(xva)).cpu().numpy()
    yv = yva.cpu().numpy()
    info = {
        "n_train": int(len(tr_idx)) + n_extra_neg, "n_val": int(len(val_idx)),
        "n_hard_neg": n_extra_neg,
        "pos_frac": float((labels == 1).float().mean()),
        "val_auc": roc_auc(pva, yv),
        "val_brier": brier(pva, yv),
        "val_ece": expected_calibration_error(pva, yv),
    }
    centers, pred, emp, cnt = reliability_curve(pva, yv)
    info["reliability"] = {"centers": centers.tolist(), "pred": pred.tolist(),
                           "emp": emp.tolist(), "counts": cnt.tolist()}
    if (verbose):
        print("[classifier] N={} D={} pos_frac={:.2f} hard_neg={}  val AUC={:.3f} Brier={:.3f} ECE={:.3f}".format(
            N, D, info["pos_frac"], n_extra_neg, info["val_auc"], info["val_brier"], info["val_ece"]))
    return model, info


def critic_baseline_metrics(critic_values, labels, n_bins=10):
    """Section-4 baseline: treat the raw critic value as the success predictor.
    Map V -> [0,1] by min-max (monotone, so AUC is unaffected) for the
    reliability/Brier/ECE numbers, and report AUC on the raw V."""
    v = np.asarray(critic_values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    auc = roc_auc(v, y)
    lo, hi = np.nanmin(v), np.nanmax(v)
    p = (v - lo) / max(hi - lo, 1e-9)
    centers, pred, emp, cnt = reliability_curve(p, y, n_bins)
    return {
        "auc": auc, "brier": brier(p, y), "ece": expected_calibration_error(p, y, n_bins),
        "reliability": {"centers": centers.tolist(), "pred": pred.tolist(),
                        "emp": emp.tolist(), "counts": cnt.tolist()},
    }


def save_classifier(model, path, meta):
    torch.save({"state_dict": model.state_dict(), "in_dim": model.feat_mean.shape[0],
                "meta": meta}, path)


def load_classifier(path, device="cpu"):
    blob = torch.load(path, map_location=device)
    model = InitiationClassifier(blob["in_dim"])
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob.get("meta", {})
