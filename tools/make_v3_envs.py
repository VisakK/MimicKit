"""Generate the v3 AMP-first ladder env pairs (amponly + ampft) for the 23-node
hold campaign, plus the queue manifest tools/v3_ladder_queue.txt.

Recipe: Yoga_skill_nodes.MD "THE VALIDATED RECIPE v3" (handstand/scorpion
proven). Exemplars followed EXACTLY:
  data/envs/amp_smpl_handstand_hold_lowtorque_{amponly,ampft}_env.yaml
  data/envs/amp_smpl_scorpion_hold_lowtorque_{amponly,ampft}_env.yaml

Per node, the v2 hold env (deepmimic_smpl_<node>_hold_env.yaml; crow ->
amp_smpl_crow_hold_env.yaml) supplies: motion_file, ground_offset,
init_time_range (already loop-1.0), contact_bodies fall set,
obs_contact_bodies, key_bodies, joint_err_w and the curated aux body lists.

Stage 1 (amponly): pure disc reward -- enable_task_tracking False,
pose_termination False, ALL reward weights 0.0 (aux lists kept
present-but-zeroed), char_file = smpl_boxhands_lowtorque.xml.
Stage 2 (ampft): tracking 0.5/0.1/0.5/0.2/0.5 + GENTLE aux tier:
cop_support 0.25/10 (v2 lists; +0.04 forward lean iff hands-only support),
toe_force_pen 0.25/cap30 iff the v2 env ran it, and the POSITIVE-CONTACT
knee_support 0.25/20 (v2 pairs kept; else derived from the clip annotation
body_on_body[_ext]; else omitted -- never invented). Inversion/energy/toe_lift
stay 0 (v3 finding).

Run (CPU only, idempotent):
  /home/visakii/Documents/moves/env_isaaclab/bin/python tools/make_v3_envs.py

Emits a full validation report (yaml loads, key names vs
mimickit/envs/deepmimic_env.py -- the silent-no-op gotcha, body names vs the
24 SMPL bodies, weight>0 => non-empty lists, ampft/amponly sibling identity)
and exits nonzero if any hard gate fails.
"""

import json
import os
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DATE = "2026-07-04"

# Queue order (campaign order). handstand + scorpion already done (v3).
NODES = [
    "tree", "natarajasana", "headstand", "pincha", "eagle", "big_toe",
    "half_moon", "firefly", "side_crow", "eight_angle", "koundinyasana",
    "warrior2", "triangle", "chair", "boat", "side_plank", "camel", "wheel",
    "warrior3", "downdog", "shoulderstand", "plow", "crow",
    # v3.1 connector additions (2026-07-05 flow-first recalibration)
    "plank", "malasana", "dolphin_plank", "tadasana",
]

# Explicit seed overrides for the v3.1 additions (index-derived 117-124 would
# collide with the reserved pincha/tree ft2 escalation seeds 118/120).
SEED_OVERRIDE = {"plank": (121, 122), "malasana": (123, 124),
                 "dolphin_plank": (125, 126), "tadasana": (127, 128)}


def seeds_ports(node, i):
    s1, s2 = SEED_OVERRIDE.get(node, (71 + 2 * i, 72 + 2 * i))
    return s1, s2, 29530 + 2 * i, 29531 + 2 * i

SMPL_BODIES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee",
    "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

# POSITIVE-CONTACT (knee_support) wiring for nodes WITHOUT v2 pairs.
# Derived from data/clip_annotations/<clip>.yaml hold_signature.body_on_body
# (+ _ext). Functional presses/wraps/grips only -- parallel-limb adjacency
# (knee-x-knee of legs held together, hanging-arm-beside-leg, folded-torso
# proximity) is NOT a support contact and is deliberately not wired.
ANNOT_KS = {
    "tree": dict(
        bodies=["L_Ankle", "L_Toe"], targets=["R_Hip", "R_Knee"],
        note="lifted-left-foot press into standing right thigh "
             "(ext: L_Ankle x R_Hip -0.146, L_Toe x R_Hip -0.118)"),
    "eagle": dict(
        bodies=["L_Hip", "L_Knee", "L_Ankle", "L_Elbow", "L_Wrist"],
        targets=["R_Knee", "R_Ankle", "R_Toe", "R_Elbow", "R_Wrist", "R_Hand"],
        note="BOTH wraps in one pair set (term = min-over-targets per mover): "
             "leg wrap L_Hip/L_Knee/L_Ankle -> R_Knee/R_Ankle/R_Toe "
             "(ext depths -0.248/-0.207/-0.171/-0.168/-0.014), arm wrap "
             "L_Elbow/L_Wrist -> R_Elbow/R_Wrist/R_Hand (-0.142/-0.089/-0.024). "
             "R_Hip target OMITTED (would let the L_Hip mover latch onto the "
             "adjacent thigh trivially); L_Hand x R_Hand (+0.006) omitted"),
    "triangle": dict(
        bodies=["R_Toe"], targets=["R_Hand"],
        note="hand-on-toe grip (body_on_body R_Toe x R_Hand -0.023; "
             "annotation hands_height 0.091 = hand at foot level)"),
    "side_plank": dict(
        bodies=["R_Ankle", "R_Toe"], targets=["L_Hip"],
        note="top (right) foot pressed into bottom-left thigh "
             "(ext: L_Hip x R_Ankle -0.120, L_Hip x R_Toe -0.003; annotation "
             "contact set is L hand/foot only, so the R side is the free side)"),
    "plow": dict(
        bodies=["L_Hand", "R_Hand"], targets=["L_Hip", "R_Hip"],
        note="supported-plow hands press the hips/back (body_on_body "
             "R_Hip x R_Hand -0.082, L_Hip x L_Hand -0.046; hands_height "
             "0.282 ~ hips_height 0.252 -> hands NOT on the floor)"),
}

# Nodes where knee_support stays OFF, with the audited reason.
KS_OMIT_REASON = {
    "headstand": "annotation ext is knee-x-knee parallel-leg adjacency "
                 "(-0.191) of legs held together overhead, not a press",
    "pincha": "annotation body_on_body and _ext both empty",
    "half_moon": "annotation body_on_body and _ext both empty",
    "koundinyasana": "annotation ext is the legs-glued-together stack "
                     "(knee-x-knee -0.281, ankle-x-ankle -0.125); the "
                     "leg-on-arm shelf is NOT in the annotation -- not wired "
                     "rather than invented",
    "warrior2": "annotation body_on_body and _ext both empty",
    "chair": "knees-together adjacency (knee-x-knee -0.299) + folded-torso "
             "hip-x-shoulder proximity, not a support press",
    "boat": "hands-near-knees pairs are asymmetric and shallow on the right "
            "(L_Knee x L_Hand -0.08 vs R -0.025) -- arms-extended-beside-"
            "knees adjacency, not a reliable grip",
    "camel": "no negative-depth pairs (deepest -0.000); shins parallel",
    "wheel": "annotation body_on_body and _ext both empty",
    "warrior3": "hip-x-wrist/hand pairs are the hanging/airplane arms beside "
                "the body, not a grip",
    "downdog": "knee-x-knee parallel-leg adjacency (-0.168)",
    "shoulderstand": "single idiosyncratic leg-cross (L_Knee x R_Hip -0.175) "
                     "in this reference; hands-on-back NOT in the annotation",
}

# Default (handstand-exemplar) aux lists used present-but-zeroed when the v2
# env never carried the term.
DEF_TFP = ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
DEF_COP = ["L_Ankle", "L_Hand", "L_Toe", "R_Ankle", "R_Hand", "R_Toe"]
DEF_COP_SUP = ["L_Hand", "R_Hand"]
DEF_INV = ["L_Hand", "R_Hand"]

FEET = {"L_Toe", "R_Toe", "L_Ankle", "R_Ankle"}


def fmt(v):
    return json.dumps(v)


def src_env_path(node):
    if node == "crow":
        return "data/envs/amp_smpl_crow_hold_env.yaml"
    return f"data/envs/deepmimic_smpl_{node}_hold_env.yaml"


def load_node(node):
    path = src_env_path(node)
    full = os.path.join(REPO, path)
    if not os.path.exists(full):
        return None
    txt = open(full).read()
    env = yaml.safe_load(txt)["env"]
    m = re.search(r"Source clip: (\S+)", txt)
    d = dict(
        src=path,
        clip=m.group(1) if m else os.path.basename(env["motion_file"]),
        motion_file=env["motion_file"],
        ground_offset=env["ground_offset"],
        init_time_range=env["init_time_range"],
        init_pose=env["init_pose"],
        key_bodies=env["key_bodies"],
        contact_bodies=env["contact_bodies"],
        obs_contact_bodies=env["obs_contact_bodies"],
        joint_err_w=env["joint_err_w"],
        tfp_w=env.get("reward_toe_force_pen_w", 0.0),
        tfp_bodies=env.get("toe_force_pen_bodies", DEF_TFP),
        cop_w=env.get("reward_cop_support_w", 0.0),
        cop_bodies=env.get("cop_bodies", DEF_COP),
        cop_support_bodies=env.get("cop_support_bodies", DEF_COP_SUP),
        inv_bodies=env.get("inversion_bonus_bodies", DEF_INV),
        v2_ks_w=env.get("reward_knee_support_w", 0.0),
        v2_ks_bodies=env.get("knee_support_bodies", []),
        v2_ks_targets=env.get("knee_support_target_bodies", []),
    )
    return d


def knee_support_for(node, d):
    """-> (bodies, targets, source, note). source in {v2, annotation, none}."""
    if d["v2_ks_w"] > 0.0 and d["v2_ks_bodies"] and d["v2_ks_targets"]:
        return (d["v2_ks_bodies"], d["v2_ks_targets"], "v2",
                "kept from the v2 env (curated grip/press pairs)")
    if node in ANNOT_KS:
        a = ANNOT_KS[node]
        return a["bodies"], a["targets"], "annotation", a["note"]
    return [], [], "none", KS_OMIT_REASON.get(node, "no pairs in v2 env or annotation")


def cop_for(node, d):
    """-> (w, cop_bodies, support_bodies, fwd_offset_or_None)."""
    if d["cop_w"] > 0.0 and d["cop_bodies"] and d["cop_support_bodies"]:
        sup = set(d["cop_support_bodies"])
        hands_only = {"L_Hand", "R_Hand"} <= sup and not (sup & FEET)
        return 0.25, d["cop_bodies"], d["cop_support_bodies"], (0.04 if hands_only else None)
    return 0.0, d["cop_bodies"], d["cop_support_bodies"], None


COMMON_TOP = """env_name: "amp"

env:
    char_file: "data/assets/smpl/smpl_boxhands_lowtorque.xml"
    camera_mode: "track"

    episode_length: 12.0 # seconds; several loop cycles -> sustained hold
    global_obs: True
    root_height_obs: True
{pt_block}
    enable_phase_obs: False
    num_disc_obs_steps: 10
    enable_task_tracking: {task_flag}
    enable_tar_obs: True
    num_phase_encoding: 4
    tar_obs_steps: [1, 2, 3]
    rand_reset: True
    init_time_range: {init_time_range}

    ref_char_offset: [2.0, 0.0, 0.0] # m
    init_pose: {init_pose}

    enable_early_termination: True
    key_bodies: {key_bodies}
    contact_bodies: {contact_bodies} # FALL set (v2-curated; load-bearing bodies excluded)

    motion_file: "{motion_file}"

    auto_ground_offset: False
    ground_offset: {ground_offset}   # seats the hold support at z=0 (measured; v2 value)

    enable_contact_obs: True
    obs_contact_bodies: {obs_contact_bodies}
    obs_contact_force_threshold: 1.0 # N
    obs_contact_margin_cap: 1.0 # m

{tracking_note}    joint_err_w: {joint_err_w}

    reward_pose_w: {w_pose}
    reward_vel_w: {w_vel}
    reward_root_pose_w: {w_root_pose}
    reward_root_vel_w: {w_root_vel}
    reward_key_pos_w: {w_key}
    reward_pose_scale: 0.25
    reward_vel_scale: 0.01
    reward_root_pose_scale: 5.0
    reward_root_vel_scale: 1.0
    reward_key_pos_scale: 10.0

"""

ENGINE = """

engine:
    engine_name: "isaac_lab"

    control_mode: "pos"
    control_freq: 30
    sim_freq: 120
    env_spacing: 5

    ground_contact_height: 0.3
"""


def env_body(d, stage, node):
    ks_bodies, ks_targets, ks_src, _ = knee_support_for(node, d)
    cop_w, cop_bodies, cop_sup, cop_fwd = cop_for(node, d)
    tfp_on = d["tfp_w"] > 0.0 and bool(d["tfp_bodies"])

    if stage == "amponly":
        pt_block = ("    pose_termination: False        "
                    "# no ref tracking under pure AMP; fall = contact_bodies")
        task_flag = ("False    # PURE AMP: reward = discriminator only")
        w = dict(w_pose="0.0", w_vel="0.0", w_root_pose="0.0",
                 w_root_vel="0.0", w_key="0.0")
        tracking_note = ("    # Tracking weights present but UNUSED "
                         "(task_reward_weight 0 in the agent, and\n"
                         "    # enable_task_tracking False). Kept at 0 to make "
                         '"pure AMP" unambiguous.\n')
        aux_hdr = ("    # --- ALL aux shaping OFF (stage-1 purity; lists kept "
                   "present-but-zeroed\n    # for the ampft sibling) ---\n")
        # exemplar placement: this note sits right above joint_err_w
        w_tfp, w_cop, w_ks = "0.0", "0.0", "0.0"
        fwd_line = ""
    else:
        pt_block = ("    pose_termination: True         "
                    "# CAMPAIGN AMENDMENT 1 (2026-07-05): anti-camp\n"
                    "    pose_termination_dist: 0.6 # m (v2 value); pincha ft "
                    "entrenched a survival camp without this")
        task_flag = ("True     # gentle DeepMimic channel (agent scales it by 0.25)")
        w = dict(w_pose="0.5", w_vel="0.1", w_root_pose="0.5",
                 w_root_vel="0.2", w_key="0.5")
        tracking_note = ""
        aux_hdr = ("    # --- GENTLE aux pack: crow ft-tier weights "
                   "(effective ~0.06 end-to-end via\n"
                   "    # the discfocus agent's task 0.25) ---\n")
        w_tfp = "0.25" if tfp_on else "0.0"
        w_cop = "0.25" if cop_w > 0.0 else "0.0"
        w_ks = "0.25" if ks_bodies else "0.0"
        fwd_line = ""
        if cop_w > 0.0 and cop_fwd is not None:
            fwd_line = ("    cop_support_forward_offset: 0.04   "
                        "# the crow lean-over-hands equilibrium (hand support)\n")

    top = COMMON_TOP.format(
        pt_block=pt_block, task_flag=task_flag, tracking_note=tracking_note,
        init_time_range=fmt(d["init_time_range"]),
        init_pose=fmt(d["init_pose"]),
        key_bodies=fmt(d["key_bodies"]), contact_bodies=fmt(d["contact_bodies"]),
        motion_file=d["motion_file"], ground_offset=repr(d["ground_offset"]),
        obs_contact_bodies=fmt(d["obs_contact_bodies"]),
        joint_err_w=fmt(d["joint_err_w"]), **w)

    ks_comment = ""
    if stage == "ampft" and ks_bodies:
        ks_comment = (f"    # positive-contact attractor ({ks_src}): seat "
                      f"{'/'.join(ks_bodies)} on {'/'.join(ks_targets)}\n")
    elif stage == "ampft":
        ks_comment = ("    # knee_support OFF: "
                      + KS_OMIT_REASON.get(node, "no pairs available")[:70] + "\n")

    aux = (
        aux_hdr
        + f"    reward_toe_force_pen_w: {w_tfp}\n"
        + "    toe_force_pen_cap: 30.0\n"
        + "    toe_force_pen_ref_h: 0.12\n"
        + f"    toe_force_pen_bodies: {fmt(d['tfp_bodies'])}\n"
        + f"    reward_cop_support_w: {w_cop}\n"
        + "    reward_cop_support_scale: 10.0\n"
        + fwd_line
        + "    cop_min_force: 10.0\n"
        + f"    cop_bodies: {fmt(cop_bodies)}\n"
        + f"    cop_support_bodies: {fmt(cop_sup)}\n"
        + ks_comment
        + f"    reward_knee_support_w: {w_ks}\n"
        + "    reward_knee_support_scale: 20.0\n"
        + f"    knee_support_bodies: {fmt(ks_bodies)}\n"
        + f"    knee_support_target_bodies: {fmt(ks_targets)}\n"
        + "\n"
        + "    # everything else stays OFF (v3 finding: no inversion bonus, no energy)\n"
        + "    reward_inversion_w: 0.0\n"
        + "    inversion_up_threshold: -0.5\n"
        + "    inversion_force_threshold: 5.0\n"
        + f"    inversion_bonus_bodies: {fmt(d['inv_bodies'])}\n"
        + "    reward_toe_lift_w: 0.0\n"
        + "    reward_com_support_w: 0.0\n"
        + "    reward_force_balance_w: 0.0\n"
        + "    reward_foot_clear_w: 0.0\n"
        + "    reward_orient_w: 0.0\n"
        + "    reward_energy_w: 0.0\n"
        + "    reward_energy_scale: 0.001\n"
    )
    return top + aux + ENGINE


def header(node, d, stage, i, out_path):
    title = node.upper().replace("_", " ")
    seed1, seed2, port1, port2 = seeds_ports(node, i)
    ks_bodies, ks_targets, ks_src, ks_note = knee_support_for(node, d)
    cop_w, _, cop_sup, cop_fwd = cop_for(node, d)
    tfp_on = d["tfp_w"] > 0.0 and bool(d["tfp_bodies"])

    if stage == "amponly":
        return f"""# {title} HOLD node -- PURE AMP base on the CORRECTED asset (v3 ladder stage 1).
#
# AUTO-GENERATED by tools/make_v3_envs.py ({GEN_DATE}) from {d['src']}
# (v2 hold node; source clip {d['clip']}).
# Recipe: Yoga_skill_nodes.MD "THE VALIDATED RECIPE v3" (handstand/scorpion
# proven): disc-only long-budget base first, gentle targeted finetune second
# (the _ampft_ sibling). Deltas from the v2 source env (clip, grounding, obs,
# init range and fall set copied VERBATIM):
#   char_file -> smpl_boxhands_lowtorque (fixed pi/180 gains + human distal caps)
#   enable_task_tracking False, ALL reward_*_w 0.0 (reward = 100% discriminator)
#   pose_termination False (no ref pose under pure AMP; fall = contact_bodies)
# GOTCHAS (handstand-proven): Test_Return/Reward_Total read 0 by design --
# watch Test_Episode_Length -> 360; balance can inflect VERY late (~300-330M
# of 400M) -- do not early-stop on a long plateau.
#
# Train FROM SCRATCH (queue row {i}: tools/v3_ladder_queue.txt):
#   run.py --mode train --num_envs 4096 \\
#     --env_config {out_path} \\
#     --agent_config data/agents/amp_smpl_agent.yaml \\
#     --max_samples 400000000 --rand_seed {seed1} --master_port {port1}
"""

    cop_line = "#   cop_support 0 (v2 env carried no CoP lists -- not invented)"
    if cop_w > 0.0:
        lean = " + forward_offset 0.04 (hand support)" if cop_fwd else ""
        cop_line = (f"#   cop_support 0 -> 0.25, scale 10 (v2 lists: support = "
                    f"{'/'.join(cop_sup)}){lean}")
    tfp_line = ("#   toe_force_pen 0 -> 0.25 (cap 30, v2 bodies)" if tfp_on
                else "#   toe_force_pen stays 0 (v2 env did not run it)")
    if ks_bodies:
        ks_line = (f"#   knee_support 0 -> 0.25, scale 20 ({ks_src} pairs): "
                   f"{'/'.join(ks_bodies)} -> {'/'.join(ks_targets)}")
    else:
        ks_line = "#   knee_support stays 0: " + KS_OMIT_REASON.get(node, "no pairs")
    note_lines = ""
    if ks_bodies:
        import textwrap
        note_lines = "\n".join("# " + l for l in textwrap.wrap(
            "knee_support rationale: " + ks_note, width=76)) + "\n"

    return f"""# {title} HOLD -- GENTLE FINETUNE of the pure-AMP base (v3 ladder stage 2).
#
# AUTO-GENERATED by tools/make_v3_envs.py ({GEN_DATE}) from {d['src']}
# (v2 hold node; source clip {d['clip']}).
# Warm-starts from output/yoga_nodes_v2/{node}_lt_amponly/model.pt and re-adds
# a SMALL task channel (discfocus agent: task 0.25 / disc 1.0, LR 1.5e-5).
# Deltas from the _amponly_ sibling (everything else identical, incl.
# pose_termination OFF):
#   enable_task_tracking True; tracking 0.5/0.1/0.5/0.2/0.5 (v2 joint_err_w)
{cop_line}
{tfp_line}
{ks_line}
# NOT enabled: inversion bonus, energy, toe_lift (v3 finding: keep the delta
# minimal and attributable).
{note_lines}#
# Finetune (weights-only resume; fresh 150M budget; queue row {i}):
#   run.py --mode train --num_envs 4096 \\
#     --env_config {out_path} \\
#     --agent_config data/agents/amp_smpl_hold_discfocus_agent.yaml \\
#     --model_file output/yoga_nodes_v2/{node}_lt_amponly/model.pt \\
#     --max_samples 150000000 --rand_seed {seed2} --master_port {port2}
"""


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def collect_quoted_strings(path):
    txt = open(path).read()
    return set(re.findall(r'"([A-Za-z_][A-Za-z_0-9]*)"', txt))


def validate(written, queue_path, check_queue=True):
    ok = True
    report = []
    envs_dir = os.path.join(REPO, "mimickit/envs")
    dm_keys = collect_quoted_strings(os.path.join(envs_dir, "deepmimic_env.py"))
    all_keys = set()
    for root, _, files in os.walk(os.path.join(REPO, "mimickit")):
        for f in files:
            if f.endswith(".py"):
                all_keys |= collect_quoted_strings(os.path.join(root, f))

    body_list_keys = [
        "key_bodies", "contact_bodies", "obs_contact_bodies",
        "toe_force_pen_bodies", "cop_bodies", "cop_support_bodies",
        "knee_support_bodies", "knee_support_target_bodies",
        "inversion_bonus_bodies",
    ]
    weight_to_lists = {
        "reward_toe_force_pen_w": ["toe_force_pen_bodies"],
        "reward_cop_support_w": ["cop_bodies", "cop_support_bodies"],
        "reward_knee_support_w": ["knee_support_bodies", "knee_support_target_bodies"],
        "reward_key_pos_w": ["key_bodies"],
        "reward_pose_w": ["joint_err_w"],
    }
    # pose_termination deliberately NOT a sibling field: CAMPAIGN AMENDMENT 1
    # (2026-07-05) sets it True in ampft envs while amponly stays False.
    sibling_fields = [
        "char_file", "episode_length", "motion_file", "ground_offset",
        "init_time_range", "init_pose", "key_bodies", "contact_bodies",
        "obs_contact_bodies", "joint_err_w",
        "auto_ground_offset",
    ]

    bad_keys, bad_reward_keys, bad_bodies, bad_weights = [], [], [], []
    parsed = {}
    n_loaded = 0
    for path in written:
        try:
            doc = yaml.safe_load(open(os.path.join(REPO, path)))
            parsed[path] = doc
            n_loaded += 1
        except Exception as e:  # noqa: BLE001
            ok = False
            report.append(f"YAML-LOAD FAIL {path}: {e}")
            continue
        env = doc["env"]
        eng = doc["engine"]
        for k in env:
            if k not in all_keys:
                bad_keys.append(f"{path}: env.{k}")
            if (k.startswith("reward_") or any(k.startswith(p) for p in (
                    "toe_", "cop_", "knee_", "inversion_", "com_support",
                    "force_balance", "foot_clear", "orient_", "joint_err")))\
                    and k not in dm_keys:
                bad_reward_keys.append(f"{path}: env.{k}")
        for k in eng:
            if k not in all_keys:
                bad_keys.append(f"{path}: engine.{k}")
        for lk in body_list_keys:
            for b in env.get(lk, []) or []:
                if b not in SMPL_BODIES:
                    bad_bodies.append(f"{path}: {lk} -> {b}")
        for wk, lks in weight_to_lists.items():
            if env.get(wk, 0.0) > 0.0:
                for lk in lks:
                    if not env.get(lk):
                        bad_weights.append(f"{path}: {wk}>0 but {lk} empty")

    report.append(f"yaml-load: {n_loaded}/{len(written)} files parse")
    report.append("reward/aux keys vs deepmimic_env.py (silent-no-op gate): "
                  + ("ALL FOUND" if not bad_reward_keys else "MISSING: " + "; ".join(bad_reward_keys)))
    report.append("all env+engine keys vs mimickit/ quoted strings: "
                  + ("ALL FOUND" if not bad_keys else "MISSING: " + "; ".join(bad_keys)))
    report.append("body names vs 24 SMPL bodies: "
                  + ("ALL VALID" if not bad_bodies else "INVALID: " + "; ".join(bad_bodies)))
    report.append("weight>0 => non-empty lists: "
                  + ("PASS" if not bad_weights else "FAIL: " + "; ".join(bad_weights)))
    if bad_reward_keys or bad_keys or bad_bodies or bad_weights:
        ok = False

    sib_bad = []
    for node in NODES:
        a = f"data/envs/amp_smpl_{node}_hold_lowtorque_amponly_env.yaml"
        f_ = f"data/envs/amp_smpl_{node}_hold_lowtorque_ampft_env.yaml"
        if a not in parsed or f_ not in parsed:
            continue
        ea, ef = parsed[a]["env"], parsed[f_]["env"]
        for fld in sibling_fields:
            if ea.get(fld) != ef.get(fld):
                sib_bad.append(f"{node}: {fld}")
    report.append("ampft init/clip fields identical to amponly sibling: "
                  + ("PASS" if not sib_bad else "FAIL: " + "; ".join(sib_bad)))
    if sib_bad:
        ok = False

    # queue manifest sanity
    if not check_queue:
        report.append("queue manifest: SKIPPED (--only run; queue not rewritten)")
        return ok, report
    fp = json.load(open(os.path.join(REPO, "tools/hold_nodes_ref_fingerprints.json")))
    rows = [l.split() for l in open(os.path.join(REPO, queue_path))
            if l.strip() and not l.startswith("#")]
    q_bad = []
    if len(rows) != len(NODES):
        q_bad.append(f"{len(rows)} rows != {len(NODES)}")
    for r in rows:
        if len(r) != 7:
            q_bad.append(f"row '{r[0]}' has {len(r)} cols")
        if not os.path.exists(os.path.join(REPO, r[1])):
            q_bad.append(f"{r[0]}: missing {r[1]}")
        if not os.path.exists(os.path.join(REPO, r[2])):
            q_bad.append(f"{r[0]}: missing {r[2]}")
        if r[0] not in fp and r[0] != "crow":
            q_bad.append(f"{r[0]}: not a fingerprint key")
    report.append("queue manifest (23 rows, 7 cols, env paths exist, node keys "
                  "in hold_nodes_ref_fingerprints.json; crow uses "
                  "check_hold_node.py's built-in EXPECT entry): "
                  + ("PASS" if not q_bad else "FAIL: " + "; ".join(q_bad)))
    if q_bad:
        ok = False
    return ok, report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="comma-separated node subset: emit only those env "
                         "pairs and do NOT rewrite the queue manifest (a full "
                         "run rewrites tools/v3_ladder_queue.txt, clobbering "
                         "any hand-trimmed campaign queue)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    written = []
    skipped = []
    rows = []
    summary = []
    for i, node in enumerate(NODES):
        if only is not None and node not in only:
            continue
        d = load_node(node)
        if d is None:
            skipped.append(f"{node}: v2 source env {src_env_path(node)} missing")
            continue
        pair = []
        for stage in ("amponly", "ampft"):
            out = f"data/envs/amp_smpl_{node}_hold_lowtorque_{stage}_env.yaml"
            txt = header(node, d, stage, i, out) + env_body(d, stage, node)
            with open(os.path.join(REPO, out), "w") as f:
                f.write(txt)
            written.append(out)
            pair.append(out)
        s1, s2, p1, p2 = seeds_ports(node, i)
        rows.append(f"{node} {pair[0]} {pair[1]} {s1} {s2} {p1} {p2}")
        ksb, kst, kss, _ = knee_support_for(node, d)
        cw, _, csup, cfwd = cop_for(node, d)
        summary.append(
            f"{node:15s} ks[{kss}]: "
            + (f"{ksb} -> {kst}" if ksb else "OFF")
            + f" | cop {'0.25' + ('+lean' if cfwd else '') if cw else '0'}"
            + f" | tfp {'0.25' if d['tfp_w'] > 0 else '0'}")

    queue_path = "tools/v3_ladder_queue.txt"
    if only is None:
        with open(os.path.join(REPO, queue_path), "w") as f:
            f.write(
                "# v3 AMP-first ladder queue (generated by tools/make_v3_envs.py "
                f"{GEN_DATE}).\n"
                "# Consumed by tools/run_v3_ladder.sh. Columns (whitespace-separated):\n"
                "#   node  env_amponly  env_ampft  seed1  seed2  port1  port2\n"
                "# handstand + scorpion are NOT here (v3 already done for both).\n"
                + "\n".join(rows) + "\n")
        print(f"wrote {len(written)} env files + {queue_path} ({len(rows)} rows)")
    else:
        print(f"wrote {len(written)} env files (--only; queue NOT rewritten). "
              "Candidate rows:")
        for r in rows:
            print("  " + r)
    for s in skipped:
        print("SKIPPED:", s)
    print("\nper-node aux summary:")
    for s in summary:
        print(" ", s)

    ok, report = validate(written, queue_path, check_queue=only is None)
    print("\nVALIDATION:")
    for r in report:
        print(" ", r)
    if not ok:
        print("\nHARD GATE FAILED")
        sys.exit(1)
    print("\nall gates PASS")


if __name__ == "__main__":
    main()
