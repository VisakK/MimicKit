"""Reference-clip baseline for the on-body positive-contact metric.

tools/onbody_contact_check.py reports policy body-origin distances between the
knee_support movers and targets — but "seated/pressed" does NOT read as 0 cm on
body origins (limb geometry sits between the origins). This tool prints the
REFERENCE clip's own distances on the identical metric, so a policy verdict is
"within X cm of the reference press", not an absolute number.

Example (tree): reference reads L_Ankle->R_Hip 11.9 cm although the annotation
sphere-distance is -0.146 (deep press) — a policy at 13.6 cm is 1.7 cm off the
reference, i.e. seated.

Usage:
  python tools/ref_onbody_baseline.py --env data/envs/amp_smpl_<node>_hold_lowtorque_ampft_env.yaml
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))

import numpy as np
import yaml

import anim.kin_char_model as kcm
import anim.motion_lib as mlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="ampft env yaml (motion_file, ground_offset, knee_support pairs)")
    args = p.parse_args()

    with open(args.env) as f:
        cfg = yaml.safe_load(f)["env"]
    movers = cfg.get("knee_support_bodies") or []
    targets = cfg.get("knee_support_target_bodies") or []
    w = cfg.get("reward_knee_support_w", 0.0)
    if not movers or not targets or w <= 0:
        print(f"no active knee_support pairs in {args.env} (w={w}) — nothing to baseline")
        return

    char = kcm.KinCharModel("cpu")
    char.load_char_file(os.path.join(_REPO, cfg["char_file"]))
    ml = mlib.MotionLib(os.path.join(_REPO, cfg["motion_file"]), char, "cpu",
                        auto_ground_offset=cfg.get("auto_ground_offset", False),
                        ground_offset=cfg.get("ground_offset", 0.0))
    body_pos, _ = char.forward_kinematics(ml._frame_root_pos, ml._frame_root_rot, ml._frame_joint_rot)
    names = [char.get_body_name(i) for i in range(char.get_num_joints())]
    P = body_pos.numpy()
    idx = {n: names.index(n) for n in set(movers + targets)}

    print(f"ref baseline: {cfg['motion_file']}  ({P.shape[0]} frames)  targets={targets}")
    for m in movers:
        d = {t: np.linalg.norm(P[:, idx[m]] - P[:, idx[t]], axis=1) * 100.0 for t in targets}
        best = min(d, key=lambda t: d[t].mean())
        print(f"REF {m:10s} nearest {best:10s} mean {d[best].mean():6.1f} cm  min {d[best].min():6.1f}  std {d[best].std():4.1f}")


if __name__ == "__main__":
    main()
