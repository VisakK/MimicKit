"""Generate per-pose DeepMimic env configs for the 11-node yoga skill graph.

Clones the proven deepmimic_smpl_handstand_orient_env.yaml and applies only the
data-grounded per-pose overrides from tools/analyze_pose_clips.py, so every node
shares ONE obs layout (global obs + contact obs + key bodies) -> the initiation
classifiers are cross-comparable (needed for pose-discriminative hard negatives
and the feasibility matrix). Per-pose differences:

  * motion_file + episode_length (cover the clip's hold window)
  * contact_bodies (the FALL set): inverted head-loaded poses (headstand, pincha)
    must NOT count head/neck contact as a fall.
  * reward stack: hand-supported inversions keep the hand com_support /
    force_balance / orient aux; foot/standing poses disable them (pure DeepMimic
    tracking, which trains arbitrary mocap fine) since hand aux is meaningless
    and orient-on-hands is wrong there.

handstand + scorpion keep their existing hand-tuned configs (already trained);
this writes the other nodes (incl. crow on the new -a take).

Run:  env_isaaclab/bin/python tools/make_node_configs.py
"""
import copy
import os
import sys

import yaml

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(REPO, "data", "envs", "deepmimic_smpl_handstand_orient_env.yaml")
OUT_DIR = os.path.join(REPO, "data", "envs")
MOT = "data/motions/smpl"

# Fall (contact_bodies) sets.
TRUNK_HEAD = ["Pelvis", "Torso", "Spine", "Chest", "Head", "Neck",
              "R_Shoulder", "L_Shoulder", "R_Thorax", "L_Thorax"]
TRUNK = ["Pelvis", "Torso", "Spine", "Chest",
         "R_Shoulder", "L_Shoulder", "R_Thorax", "L_Thorax"]  # head/neck load-bearing

# support: "hands" keeps the handstand hand-aux stack; "feet" disables hand aux
# (pure DeepMimic tracking) for foot/standing poses.
NODES = {
    # --- hand-supported inversions (hand aux stack) ---
    "crow":      dict(motion="220923_Crane_Crow_Pose_or_Bakasana_-a", episode=28.0, support="hands",
                      fall=TRUNK_HEAD, tag="smpl_crow"),  # the new -a take (converter strips parens)
    "headstand": dict(motion="220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b",
                      episode=44.0, support="hands", fall=TRUNK, tag="smpl_headstand"),
    "pincha":    dict(motion="220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a",
                      episode=60.0, support="hands", fall=TRUNK, tag="smpl_pincha"),
    # --- foot/forearm-supported & standing (pure tracking, no hand aux) ---
    "downdog":   dict(motion="220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
                      episode=28.0, support="feet", fall=TRUNK_HEAD, tag="smpl_downdog"),
    "dolphin_plank": dict(motion="220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a",
                      episode=28.0, support="feet", fall=TRUNK_HEAD, tag="smpl_dolphin_plank"),
    "side_plank": dict(motion="220923_Side_Plank_Pose_or_Vasisthasana_-a",
                      episode=38.0, support="feet", fall=TRUNK_HEAD, tag="smpl_side_plank"),
    "warrior2":  dict(motion="220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
                      episode=28.0, support="feet", fall=TRUNK_HEAD, tag="smpl_warrior2"),
    "warrior3":  dict(motion="220926_Warrior_III_Pose_or_Virabhadrasana_III_-a",
                      episode=40.0, support="feet", fall=TRUNK_HEAD, tag="smpl_warrior3"),
    "natarajasana": dict(motion="220926_Lord_of_the_Dance_Pose_or_Natarajasana_-a",
                      episode=40.0, support="feet", fall=TRUNK_HEAD, tag="smpl_natarajasana"),
}


def main():
    with open(BASE, "r") as f:
        base = yaml.safe_load(f)

    for node, spec in NODES.items():
        cfg = copy.deepcopy(base)
        env = cfg["env"]
        env["motion_file"] = os.path.join(MOT, spec["motion"])
        env["episode_length"] = spec["episode"]
        env["log_tag"] = spec["tag"]
        env["contact_bodies"] = spec["fall"]

        if (spec["support"] == "feet"):
            # disable the hand-specific aux terms; pure DeepMimic tracking + energy
            for w in ["reward_com_support_w", "reward_force_balance_w", "reward_orient_w"]:
                env[w] = 0.0
            env.pop("com_support_bodies", None)
            env.pop("force_balance_bodies", None)
            env.pop("orient_bodies", None)
        # support == "hands": keep the base hand aux stack unchanged.

        out = os.path.join(OUT_DIR, "deepmimic_smpl_{}_env.yaml".format(node))
        with open(out, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print("wrote {}  (motion={}, episode={}s, support={}, fall={})".format(
            os.path.relpath(out, REPO), spec["motion"], spec["episode"],
            spec["support"], "TRUNK" if spec["fall"] is TRUNK else "TRUNK+HEAD"))
    return


if __name__ == "__main__":
    main()
