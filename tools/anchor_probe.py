"""Stage-1 go/no-go: does the geometric fingerprint separate balance regimes?

Reads data/clip_annotations/index.json, builds a physical fingerprint vector per
clip, and runs the anchor validation from the clustering plan:

  1. crow ~= one-legged-crow (near) and BOTH far from handstand.
  2. {handstand, headstand, pincha, scorpion} land in distinct regions.
  3. Ablation: dropping the contact-set + inversion + height channels must
     COLLAPSE the inversions together -- proving the separation is bought by the
     engineered physical-contact features, not generic kinematics.

Also previews the coarse cluster structure (agglomerative) as a sanity check.

  env_isaaclab/bin/python tools/anchor_probe.py
"""
import os
import sys
import json

import numpy as np

ANNO = "data/clip_annotations/index.json"

# bodies used for the contact multi-hot (the support-relevant ones)
CONTACT_VOCAB = ["Head", "L_Hand", "R_Hand", "L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow",
                 "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe",
                 "Pelvis", "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder"]

# feature groups -> (names, is_physical_contact_channel). The ablation drops the
# channels marked True (contact set, inversion, key-body heights).
def build_features(rows, ablate=False):
    X, names = [], []
    for r in rows:
        v = []
        # continuous geometry (kept in both full and ablated)
        v += [r.get("com_height") or 0, r.get("support_span_m") or 0,
              r.get("support_area_m2") or 0, min(r.get("support_aspect") or 0, 20),
              r.get("com_margin_m") or 0, (r.get("num_supports") or 0) / 6.0,
              r.get("lr_symmetry") or 0, (r.get("n_body_on_body") or 0) / 6.0]
        if not ablate:
            # inversion + vertical structure
            v += [r.get("up_z") or 0, r.get("head_height") or 0, r.get("hips_height") or 0,
                  r.get("feet_height") or 0, r.get("hands_height") or 0]
            # contact-set multi-hot
            cb = set(r.get("contact_bodies") or [])
            v += [1.0 if b in cb else 0.0 for b in CONTACT_VOCAB]
        X.append(v)
    return np.array(X, dtype=float)


def zscore(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def find(rows, *subs):
    """indices of clips whose family/clip matches any substring group (AND within
    a tuple, OR across args). Returns list of (idx, clip)."""
    out = []
    for i, r in enumerate(rows):
        key = (r["clip"] + " " + (r.get("family") or "")).lower()
        for grp in subs:
            grp = grp if isinstance(grp, tuple) else (grp,)
            if all(s.lower() in key for s in grp):
                out.append((i, r["clip"])); break
    return out


def centroid(D, idxs):
    return D[idxs].mean(0) if idxs else None


def main():
    rows = json.load(open(ANNO))
    print("clips: {}".format(len(rows)))

    anchors = {
        "crow":       find(rows, ("bakasana", "-a"), ("bakasana", "-b"), ("bakasana", "hold"), "crow_pose"),
        "one_leg_crow": find(rows, "one_legged_crow"),
        "handstand":  find(rows, "adho_mukha_vrksasana", ("kound",)),   # handstand + koundi(hand-inv)
        "headstand":  find(rows, "sirsasana_or_salamba", "salamba_sirsasana"),
        "pincha":     find(rows, "pincha_mayurasana"),
        "scorpion":   find(rows, "vrischikasana", "scorpion"),
    }
    anchors = {k: [i for i, _ in v] for k, v in anchors.items() if v}

    for full, tag in [(True, "FULL fingerprint"), (False, "ABLATED (no contact/inversion/heights)")]:
        X = zscore(build_features(rows, ablate=not full))
        # cosine distance between family centroids
        cent = {k: centroid(X, idxs) for k, idxs in anchors.items()}
        def d(a, b):
            ca, cb = cent[a], cent[b]
            return float(np.linalg.norm(ca - cb))
        print("\n================= {} (dim={}) =================".format(tag, X.shape[1]))
        keys = [k for k in ["crow", "one_leg_crow", "handstand", "headstand", "pincha", "scorpion"] if k in cent]
        print("centroid L2 distances:")
        print("        " + "".join("{:>12s}".format(k[:11]) for k in keys))
        for a in keys:
            print("{:>8s}".format(a[:8]) + "".join("{:12.2f}".format(d(a, b)) for b in keys))
        if "crow" in cent and "one_leg_crow" in cent and "handstand" in cent:
            near = d("crow", "one_leg_crow"); far = d("crow", "handstand")
            print("  crow<->one_leg_crow = {:.2f}   crow<->handstand = {:.2f}   ratio far/near = {:.2f}"
                  .format(near, far, far / max(near, 1e-6)))

    # cluster preview on the full fingerprint
    print("\n================= cluster preview (agglomerative, full fingerprint) =================")
    X = zscore(build_features(rows, ablate=False))
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        Z = linkage(X, method="ward")
        for k in (6, 10):
            lab = fcluster(Z, t=k, criterion="maxclust")
            print("\n-- {} clusters --".format(k))
            for c in sorted(set(lab)):
                members = [rows[i]["family"] or rows[i]["clip"] for i in range(len(rows)) if lab[i] == c]
                modes = [rows[i].get("support_mode") for i in range(len(rows)) if lab[i] == c]
                top = max(set(modes), key=modes.count) if modes else "?"
                uniq = sorted(set(m[:22] for m in members))
                print("  c{:<2d} n={:<3d} [{:16s}] {}".format(
                    c, len(members), str(top), ", ".join(uniq[:8]) + (" ..." if len(uniq) > 8 else "")))
    except Exception as e:
        print("scipy hierarchy unavailable:", e)


if __name__ == "__main__":
    main()
