"""Hand-base geometry forensics from a hold-node telemetry.npz.

Measures the metrics that separated the natural pure-AMP handstand from the
deformed-base hybrids (see Yoga_skill_nodes.MD, stagger forensics 2026-07-03):

  - fore-aft hand STAGGER in the shoulder-anatomical frame (+ = LEFT hand
    leads).  Axis = cross(L_Sh - R_Sh, Chest - Pelvis) projected to the ground
    plane -- uses no world-up, so the sign stays correct while inverted.
    Reference (handstand_hold): +0.9 cm.  Deformed hybrid: -16.5 cm.
  - LATERAL hand separation (reference handstand_hold: 35.7 cm).
  - wrist->hand pitch per side (reference approx -20 deg; fingertip stilts
    read -70..-90 deg).
  - mean hand height.

Usage:
  python tools/hand_base_forensics.py --node_dir output/yoga_nodes_v2/<run>
  python tools/hand_base_forensics.py --telemetry path/to/telemetry.npz \
      [--warmup_steady 15]

Telemetry format: tools/collect_crow_telemetry.py (body_names, body_pos [T,24,3]).
NOTE: check_hold_node.py collects with --warmup 10, so frame 0 here is ~0.33 s
after the RSI reset, not the raw init state.
"""

import argparse
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", help="run dir containing telemetry.npz")
    p.add_argument("--telemetry", help="explicit path to telemetry.npz")
    p.add_argument("--warmup_steady", type=int, default=15,
                   help="first N frames treated as settle; steady stats start here")
    args = p.parse_args()

    path = args.telemetry or os.path.join(args.node_dir, "telemetry.npz")
    d = np.load(path, allow_pickle=True)
    names = list(d["body_names"])
    P = d["body_pos"]
    i = {n: names.index(n) for n in
         ["Pelvis", "Chest", "L_Shoulder", "R_Shoulder",
          "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]}

    lat = P[:, i["L_Shoulder"]] - P[:, i["R_Shoulder"]]
    spine = P[:, i["Chest"]] - P[:, i["Pelvis"]]
    fore = np.cross(lat, spine)
    fore[:, 2] = 0.0
    fore /= np.linalg.norm(fore, axis=1, keepdims=True)
    latg = lat.copy()
    latg[:, 2] = 0.0
    latg /= np.linalg.norm(latg, axis=1, keepdims=True)

    dh = P[:, i["L_Hand"]] - P[:, i["R_Hand"]]
    stag = np.einsum("ij,ij->i", dh, fore) * 100.0
    sep = np.einsum("ij,ij->i", dh, latg) * 100.0

    def pitch(side):
        v = P[:, i[f"{side}_Hand"]] - P[:, i[f"{side}_Wrist"]]
        return np.degrees(np.arctan2(v[:, 2], np.linalg.norm(v[:, :2], axis=1)))

    pl, pr = pitch("L"), pitch("R")
    S = slice(args.warmup_steady, None)

    print(f"telemetry: {path}  ({P.shape[0]} frames)")
    print(f"fore-aft stagger cm (+=L leads): frame0 {stag[0]:+.1f}  "
          f"settle(0-{args.warmup_steady - 1}) {stag[:args.warmup_steady].mean():+.1f}  "
          f"steady {stag[S].mean():+.1f} +/- {stag[S].std():.1f}")
    print(f"lateral separation cm:           steady {sep[S].mean():.1f} +/- {sep[S].std():.1f}")
    print(f"wrist->hand pitch deg:           L {pl[S].mean():+.1f}   R {pr[S].mean():+.1f}")
    print(f"hand height cm:                  L {P[S, i['L_Hand'], 2].mean() * 100:.1f}  "
          f"R {P[S, i['R_Hand'], 2].mean() * 100:.1f}")


if __name__ == "__main__":
    main()
