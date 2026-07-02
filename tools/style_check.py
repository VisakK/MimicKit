"""Quick crow-STYLE check from a telemetry npz: head height and arm 'flare'
(elbow / hand spread), over the held one-legged-crow segment. Compares the
arms-flare / head-drop the user flagged across policies.

  PY=/home/visakii/Documents/moves/MimicKit/../env_isaaclab/bin/python
  $PY tools/style_check.py output/one_legged_crow_slow_telemetry.npz output/one_legged_crow_ft2_telemetry.npz
"""
import sys
import numpy as np

for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    names = list(d["body_names"]); bi = {n: i for i, n in enumerate(names)}
    bp = d["body_pos"]
    def P(n): return bp[:, bi[n]]
    # held segment = right toe lifted (the one-legged hold)
    sel = P("R_Toe")[:, 2] > 0.5
    if sel.sum() < 5: sel = np.ones(bp.shape[0], bool)
    def m(a): return float(np.mean(a[sel]))
    head_above_pelvis = m(P("Head")[:, 2] - P("Pelvis")[:, 2])   # +ve = head ABOVE hips (crow), -ve = dropped
    head_z = m(P("Head")[:, 2])
    elbow_spread = m(np.linalg.norm(P("L_Elbow") - P("R_Elbow"), axis=-1))
    hand_spread = m(np.linalg.norm(P("L_Hand") - P("R_Hand"), axis=-1))
    shoulder_w = m(np.linalg.norm(P("L_Shoulder") - P("R_Shoulder"), axis=-1))
    # flare ratio: elbows wider than shoulders => arms splayed out
    flare = elbow_spread / (shoulder_w + 1e-6)
    print("{}".format(path.split("/")[-1]))
    print("  head_z={:+.3f}  head-above-pelvis={:+.3f} (>0 = head up, crow)".format(head_z, head_above_pelvis))
    print("  elbow_spread={:.3f}  shoulder_w={:.3f}  flare(elbow/shoulder)={:.2f} (>1 = elbows splayed out)".format(
        elbow_spread, shoulder_w, flare))
    print("  hand_spread={:.3f}".format(hand_spread))
