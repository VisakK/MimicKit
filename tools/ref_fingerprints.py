"""Compute REFERENCE fingerprints for every trimmed hold clip, so the node
verifier compares the policy against numbers derived from the exact reference
it tracks (same proxy, same FK path) rather than annotation values computed
with a different up-vector definition (chest-pelvis proxy != root quat up for
tucked poses like crow: -0.54 vs +0.13).

Writes tools/hold_nodes_ref_fingerprints.json:
  node -> {up_z_proxy, root_up_z, root_h, heights: {body: mean_z}}
computed as means over all frames of data/motions/smpl_holds/<node>_hold with
the node's measured ground_offset applied (read from the env yaml).

Run: /home/visakii/Documents/moves/env_isaaclab/bin/python tools/ref_fingerprints.py
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))

import torch
import yaml

import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import util.torch_util as torch_util

CHAR = "data/assets/smpl/smpl_boxhands.xml"
HOLD_DIR = os.path.join(_REPO, "data/motions/smpl_holds")
ENV_DIR = os.path.join(_REPO, "data/envs")

# Nodes whose trimmed hold clip lives OUTSIDE smpl_holds (ground_offset read
# from the named env yaml instead of deepmimic_smpl_<node>_hold_env.yaml).
# crow: the v2 crow node was built directly on the pre-trimmed Bakasana_hold
# clip; adding it here closes the "crow has no calibrated fingerprint" gap.
EXTRA = {
    "crow": ("data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_hold",
             "data/envs/amp_smpl_crow_hold_env.yaml"),
}
KEY = ["L_Toe", "R_Toe", "L_Hand", "R_Hand", "Head", "L_Ankle", "R_Ankle",
       "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder", "L_Knee", "R_Knee"]


def main():
    device = "cpu"
    kin = kin_char_model.KinCharModel(device)
    kin.load_char_file(os.path.join(_REPO, CHAR))
    names = kin.get_body_names()
    bi = {n: i for i, n in enumerate(names)}

    out = {}
    jobs = []
    for fn in sorted(os.listdir(HOLD_DIR)):
        if not fn.endswith("_hold"):
            continue
        node = fn[:-len("_hold")]
        jobs.append((node, os.path.join(HOLD_DIR, fn),
                     os.path.join(ENV_DIR, f"deepmimic_smpl_{node}_hold_env.yaml")))
    for node, (mpath, env_rel) in EXTRA.items():
        jobs.append((node, os.path.join(_REPO, mpath),
                     os.path.join(_REPO, env_rel)))
    for node, motion_path, env_yaml in sorted(jobs):
        offset = 0.0
        if os.path.exists(env_yaml):
            with open(env_yaml) as f:
                offset = float(yaml.safe_load(f)["env"].get("ground_offset", 0.0))
        mlib = motion_lib.MotionLib(motion_file=motion_path,
                                    kin_char_model=kin, device=device,
                                    auto_ground_offset=False, ground_offset=offset,
                                    char_file=os.path.join(_REPO, CHAR))
        body_pos, _ = kin.forward_kinematics(
            mlib._frame_root_pos, mlib._frame_root_rot, mlib._frame_joint_rot)
        n = body_pos.shape[0]
        up = body_pos[:, bi["Chest"], :] - body_pos[:, bi["Pelvis"], :]
        up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
        rup = torch_util.quat_rotate(
            mlib._frame_root_rot, torch.tensor([0.0, 0.0, 1.0]).expand(n, 3))
        out[node] = dict(
            up_z_proxy=round(float(up[:, 2].mean()), 3),
            root_up_z=round(float(rup[:, 2].mean()), 3),
            root_h=round(float(body_pos[:, bi["Pelvis"], 2].mean()), 3),
            heights={b: round(float(body_pos[:, bi[b], 2].mean()), 3)
                     for b in KEY if b in bi},
        )
        print(f"{node:<15s} up_z_proxy={out[node]['up_z_proxy']:+.2f} "
              f"root_up_z={out[node]['root_up_z']:+.2f} root_h={out[node]['root_h']:.2f}")

    path = os.path.join(_REPO, "tools/hold_nodes_ref_fingerprints.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
