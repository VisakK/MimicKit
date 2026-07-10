"""Generate v2 HOLD-node training configs for the yoga skill graph.

For each pose in the ROSTER (selected from data/clip_annotations/, see
Yoga_annotations.MD section 9 and Yoga_skill_nodes.MD):

  1. TRIM the annotated primary-hold window to a seamless WRAP-looping
     ping-pong clip:      data/motions/smpl_holds/<node>_hold
     (tools/trim_hold_clip.py -- the crow-hold recipe: the reference never
     advances into entry/exit, so neither tracking nor a disc can teach a
     foot-down or a stand-up.)
  2. MEASURE the hold-specific ground offset on the trimmed clip
     (tools/measure_ground_offset.py). Because the trimmed clip IS the hold,
     its global-min lift == the seat-the-hold-at-z=0 offset (the attempt-2
     post-mortem fix: never auto_ground_offset, never a whole-clip min).
  3. EMIT  data/envs/deepmimic_smpl_<node>_hold_env.yaml -- pure DeepMimic
     (2 fit on the A5000 concurrently) + the annotation-parameterized aux
     reward pack validated on crow (Crow_pose.MD):
       - upright arm balances: toe_lift + toe_force_pen + cop_support(hands)
         + knee_support(annotated leg-x-arm pairs)
       - inversions: cop_support(support set) + inversion bonus + toe_force_pen
       - standing balances: toe_lift + toe_force_pen on the LIFTED foot only
       - wide-stance / grounded: pure tracking (+ small energy term)
     Fall sets are per pose (load-bearing bodies removed, annotation caveats
     corrected: headstand +Head, wheel +feet, eight-angle +right hand).
  4. Write the training queue manifest tools/hold_nodes_queue.txt
     (node, env, agent, samples) in the intended launch order.

Run (CPU only, no Isaac):
  /home/visakii/Documents/moves/env_isaaclab/bin/python tools/make_hold_nodes.py
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/visakii/Documents/moves/env_isaaclab/bin/python"
SRC_DIR = "data/motions/smpl"
HOLD_DIR = "data/motions/smpl_holds"
ENV_DIR = "data/envs"
CHAR = "data/assets/smpl/smpl_boxhands.xml"

AGENT_STD = "data/agents/deepmimic_smpl_ppo_agent.yaml"
AGENT_LOWLR = "data/agents/deepmimic_smpl_lowlr_ppo_agent.yaml"

MAX_WIN_S = 8.0     # trimmed hold window cap (ping-pong doubles it)
MIN_WIN_S = 1.0
FPS = 30.0

# ---------------------------------------------------------------- fall sets
TRUNK_HEAD = ["Pelvis", "Torso", "Spine", "Chest", "Head", "Neck",
              "R_Shoulder", "L_Shoulder", "R_Thorax", "L_Thorax"]
TRUNK_HEAD_HIPS = TRUNK_HEAD + ["L_Hip", "R_Hip"]

# joint_err_w layouts (23 joints: L leg | R leg | spine | L arm | R arm)
JW_LIMB = [2.0, 2.5, 0.6, 0.3, 2.0, 2.5, 0.6, 0.3,
           1.0, 0.6, 0.6, 0.4, 0.4,
           1.5, 2.0, 2.5, 3.0, 3.0, 1.5, 2.0, 2.5, 3.0, 3.0]
JW_BACKBEND = [2.0, 2.5, 0.6, 0.3, 2.0, 2.5, 0.6, 0.3,
               2.0, 1.5, 1.5, 0.6, 0.6,
               1.5, 2.0, 2.5, 3.0, 3.0, 1.5, 2.0, 2.5, 3.0, 3.0]

TOES = ["L_Toe", "R_Toe"]
TOES_ANKLES = ["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
HANDS = ["L_Hand", "R_Hand"]

# Aux pack shorthand builders ------------------------------------------------
def crow_pack(knee_bodies, knee_targets, fwd=0.0, knee_w=0.5):
    """Upright arm balance: the full validated crow stack."""
    return dict(
        toe_lift=dict(bodies=TOES, w=0.45),
        toe_pen=dict(bodies=TOES_ANKLES, w=0.5),
        cop=dict(support=HANDS, w=0.5, scale=15.0, fwd=fwd, left=0.0),
        knee=dict(bodies=knee_bodies, targets=knee_targets, w=knee_w, scale=25.0),
    )

def inversion_pack(support, inv_bodies, cop_w=0.3):
    """Inverted holds: CoP over the support set + gated inversion bonus +
    anti-toe-tap penalty (ref toes are high, so the ref gate stays on)."""
    return dict(
        toe_pen=dict(bodies=TOES_ANKLES, w=0.5),
        cop=dict(support=support, w=cop_w, scale=10.0, fwd=0.0, left=0.0),
        inv=dict(bodies=inv_bodies, w=0.3),
    )

def standing_pack(lift_side, grip=None):
    """One-legged standing balance: keep the LIFTED foot up and unloaded.
    Optional grip term seats the lifted foot into the gripping hand."""
    p = dict(
        toe_lift=dict(bodies=[f"{lift_side}_Toe"], w=0.45),
        toe_pen=dict(bodies=[f"{lift_side}_Toe", f"{lift_side}_Ankle"], w=0.5),
    )
    if grip:
        p["knee"] = dict(bodies=grip[0], targets=grip[1], w=0.3, scale=20.0)
    return p

# ---------------------------------------------------------------- the roster
# window = annotated primary hold [start_s, end_s] (data/clip_annotations/).
# All aux body sets come from hold_signature.contact_bodies / body_on_body,
# with the manual corrections listed in Yoga_skill_nodes.MD.
ROSTER = [
    # ---------------- inversions ----------------
    dict(id="handstand", family="Handstand (Adho Mukha Vrksasana)",
         clip="220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
         window=(58.20, 60.37), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=100_000_000,
         aux=inversion_pack(HANDS, HANDS)),
    dict(id="headstand", family="Supported Headstand (Salamba Sirsasana)",
         clip="220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a",
         window=(8.77, 56.83),
         fall=["Pelvis", "Torso", "Spine", "Chest"],  # head/neck/shoulders load-bearing
         jw=JW_LIMB, agent="lowlr", samples=100_000_000,
         aux=inversion_pack(["L_Elbow", "R_Elbow", "L_Hand", "R_Hand"],
                            ["L_Elbow", "R_Elbow"])),
    dict(id="pincha", family="Forearm Stand (Pincha Mayurasana)",
         clip="220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c",
         window=(29.27, 65.30), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=100_000_000,
         aux=inversion_pack(["L_Elbow", "R_Elbow", "L_Hand", "R_Hand"],
                            ["L_Elbow", "R_Elbow"])),
    dict(id="scorpion", family="Scorpion (Vrischikasana, handstand variant)",
         clip="scorpion_pose", window=(1.73, 26.57), fall=TRUNK_HEAD,
         jw=JW_BACKBEND, term_dist=0.7, agent="lowlr", samples=100_000_000,
         aux=inversion_pack(HANDS, HANDS)),
    dict(id="shoulderstand", family="Shoulderstand (Salamba Sarvangasana)",
         clip="220923_Supported_Shoulderstand_pose_or_Salamba_Sarvangasana_-a",
         window=(10.50, 45.10),
         fall=["Pelvis", "L_Hip", "R_Hip", "Torso"],  # shoulder girdle/head/arms load-bearing
         jw=JW_LIMB, agent="lowlr", samples=60_000_000,
         aux=dict(toe_pen=dict(bodies=TOES_ANKLES, w=0.5),
                  inv=dict(bodies=["L_Shoulder", "R_Shoulder"], w=0.3))),
    dict(id="plow", family="Plow (Halasana)",
         clip="220926_Plow_Pose_or_Halasana_-b", window=(6.83, 64.17),
         fall=["Pelvis", "L_Hip", "R_Hip", "Torso"],
         jw=JW_LIMB, agent="lowlr", samples=60_000_000,
         aux=dict(inv=dict(bodies=["L_Shoulder", "R_Shoulder"], w=0.3))),
    # ---------------- arm balances ----------------
    dict(id="firefly", family="Firefly (Tittibhasana)",
         clip="220923_Firefly_Pose_or_Tittibhasana_-b", window=(6.03, 27.90),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="lowlr", samples=80_000_000,
         aux=crow_pack(["L_Knee", "R_Knee"], ["L_Shoulder", "R_Shoulder"])),
    dict(id="eight_angle", family="Eight-Angle (Astavakrasana)",
         clip="220926_Eight-Angle_Pose_or_Astavakrasana_-a", window=(6.97, 39.10),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="lowlr", samples=80_000_000,
         aux=crow_pack(["L_Knee", "R_Knee"], ["R_Shoulder", "R_Elbow"])),
    dict(id="side_crow", family="Side Crow (Parsva Bakasana)",
         clip="220923_Side_Crane_Crow_Pose_or_Parsva_Bakasana_-c",
         window=(3.67, 52.83), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=80_000_000,
         aux=crow_pack(["L_Knee"], ["L_Shoulder", "L_Elbow"])),
    dict(id="koundinyasana", family="Koundinyasana (press/inverted take)",
         clip="kound_a_pose", window=(4.33, 51.90), fall=TRUNK_HEAD,
         jw=JW_LIMB, agent="lowlr", samples=80_000_000,
         aux=inversion_pack(HANDS, HANDS)),
    # ---------------- standing balances ----------------
    dict(id="tree", family="Tree (Vrksasana)",
         clip="220923_Tree_Pose_or_Vrksasana_-a", window=(0.0, 40.30),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="lowlr", samples=60_000_000,
         aux=standing_pack("L")),
    dict(id="warrior3", family="Warrior III (Virabhadrasana III)",
         clip="220926_Warrior_III_Pose_or_Virabhadrasana_III_-b",
         window=(0.0, 39.03), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=60_000_000, aux=standing_pack("L")),
    dict(id="natarajasana", family="Dancer (Natarajasana)",
         clip="220926_Lord_of_the_Dance_Pose_or_Natarajasana_-c",
         window=(0.0, 63.63), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=60_000_000,
         aux=standing_pack("L", grip=(["L_Toe"], ["L_Hand"]))),
    dict(id="eagle", family="Eagle (Garudasana)",
         clip="220926_Eagle_Pose_or_Garudasana_-a", window=(0.0, 39.03),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="lowlr", samples=60_000_000,
         aux=standing_pack("R")),
    dict(id="half_moon", family="Half Moon (Ardha Chandrasana)",
         clip="220923_Half_Moon_Pose_or_Ardha_Chandrasana_-a",
         window=(7.50, 34.67), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=60_000_000, aux=standing_pack("L")),
    dict(id="big_toe", family="Standing Hand-to-Big-Toe (U. Padangusthasana)",
         clip="220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c",
         window=(0.0, 42.83), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="lowlr", samples=60_000_000,
         aux=standing_pack("R", grip=(["R_Toe"], ["L_Hand"]))),
    # ---------------- wide stance / grounded (pure tracking) ----------------
    dict(id="warrior2", family="Warrior II (Virabhadrasana II)",
         clip="220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
         window=(5.83, 21.13), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="std", samples=50_000_000, aux=dict()),
    dict(id="triangle", family="Triangle (Utthita Trikonasana)",
         clip="220923_Extended_Revolved_Triangle_Pose_or_Utthita_Trikonasana_-b",
         window=(4.87, 26.57), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="std", samples=50_000_000, aux=dict()),
    dict(id="chair", family="Chair (Utkatasana)",
         clip="220923_Chair_Pose_or_Utkatasana_-b", window=(6.00, 18.93),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="std", samples=50_000_000,
         aux=dict()),
    dict(id="boat", family="Boat (Paripurna Navasana)",
         clip="220923_Boat_Pose_or_Paripurna_Navasana_-a", window=(9.37, 20.93),
         fall=["Chest", "Neck", "Head"],  # sit bones/torso load-bearing
         jw=JW_LIMB, agent="lowlr", samples=50_000_000,
         aux=dict(toe_lift=dict(bodies=TOES, w=0.45),
                  toe_pen=dict(bodies=TOES_ANKLES, w=0.5))),
    dict(id="side_plank", family="Side Plank (Vasisthasana)",
         clip="220923_Side_Plank_Pose_or_Vasisthasana_-e", window=(8.40, 38.80),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="lowlr", samples=60_000_000,
         aux=dict(toe_pen=dict(bodies=["R_Toe", "R_Ankle"], w=0.3,
                               ref_h=0.08))),  # feet stack low; gentler gate
    dict(id="camel", family="Camel (Ustrasana)",
         clip="220923_Camel_Pose_or_Ustrasana_-c", window=(7.40, 24.17),
         fall=["Torso", "Spine", "Chest", "Neck", "Head"],  # kneeling base allowed
         jw=JW_BACKBEND, term_dist=0.7, agent="std", samples=50_000_000,
         aux=dict()),
    dict(id="wheel", family="Wheel (Urdhva Dhanurasana)",
         clip="220926_Upward_Bow_Wheel_Pose_or_Urdhva_Dhanurasana_-b",
         window=(15.07, 52.43), fall=TRUNK_HEAD_HIPS, jw=JW_BACKBEND,
         term_dist=0.7, agent="std", samples=50_000_000, aux=dict()),
    # downdog clip history (check ALL FOUR limbs when picking references!):
    #   -b (original): lifts the LEFT HAND for 22% of the hold -> 3-point policy
    #   -c (1st fix): hands clean but it is a THREE-LEGGED dog (L_Toe to 1.81 m)
    #     -> faithfully-tracked 3L policy archived as downdog3L (kick-up precursor)
    #   -a (current): 97% four-limb-planted, longest clean run 11.4 s
    dict(id="downdog", family="Downward Dog (Adho Mukha Svanasana) [hub]",
         clip="220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
         window=(7.57, 19.00), fall=TRUNK_HEAD, jw=JW_LIMB,
         agent="std", samples=50_000_000, aux=dict()),
    # ------- v3.1 connector additions (2026-07-05 flow-first recalibration) --
    dict(id="plank", family="Plank (Kumbhakasana) [connector]",
         clip="220923_Plank_Pose_or_Kumbhakasana_-a", window=(4.47, 25.20),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="std", samples=50_000_000,
         aux=dict()),
    dict(id="malasana", family="Garland / deep squat (Malasana) [crouch hub]",
         clip="220923_Garland_Pose_or_Malasana_-a", window=(5.93, 13.97),
         fall=TRUNK_HEAD, jw=JW_LIMB, agent="std", samples=50_000_000,
         aux=dict()),
    dict(id="dolphin_plank",
         family="Dolphin Plank (Makara Adho Mukha Svanasana) [connector]",
         clip="220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a",
         window=(5.33, 27.63), fall=TRUNK_HEAD, jw=JW_LIMB, agent="std",
         samples=50_000_000, aux=dict()),
    # tadasana has NO dedicated clip in the corpus; the hold is the donor
    # clip's opening standing rest (12.53 s, up_z 0.994, com_h 1.03, feet-only
    # contact; NOT its primary hold -- verified against the annotation).
    dict(id="tadasana", family="Mountain (Tadasana) [neutral hub, synthetic]",
         clip="220926_Intense_Side_Stretch_Pose_or_Parsvottanasana_-b",
         window=(0.0, 12.53), fall=TRUNK_HEAD, jw=JW_LIMB, agent="std",
         samples=50_000_000, aux=dict()),
]

# Launch order: hard inversions early, interleaved with quick wins so the
# roster fills breadth-first while the big runs bake.
QUEUE_ORDER = [
    "handstand", "tree", "headstand", "warrior3", "pincha", "downdog",
    "scorpion", "shoulderstand", "firefly", "plow", "side_crow",
    "natarajasana", "koundinyasana", "eagle", "eight_angle", "half_moon",
    "big_toe", "warrior2", "triangle", "chair", "boat", "side_plank",
    "camel", "wheel",
]


def compute_window(start, end):
    """Inset the hold boundaries (settling frames), then center-crop to cap."""
    dur = end - start
    inset = min(1.0, 0.1 * dur)
    s, e = start + inset, end - inset
    if e - s > MAX_WIN_S:
        mid = 0.5 * (s + e)
        s, e = mid - MAX_WIN_S / 2, mid + MAX_WIN_S / 2
    if e - s < MIN_WIN_S:
        s, e = start, end  # too short to inset; take the raw hold
    return round(s, 2), round(e, 2)


def trim_and_measure(node):
    src = os.path.join(SRC_DIR, node["clip"])
    out = os.path.join(HOLD_DIR, node["id"] + "_hold")
    s, e = compute_window(*node["window"])
    r = subprocess.run(
        [PY, "tools/trim_hold_clip.py", "--motion_file", src, "--out", out,
         "--start", str(s), "--end", str(e), "--pingpong"],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"trim failed for {node['id']}:\n{r.stdout}\n{r.stderr}")
    m = subprocess.run(
        [PY, "tools/measure_ground_offset.py", "--motion_file", out,
         "--char_file", CHAR],
        cwd=REPO, capture_output=True, text=True)
    if m.returncode != 0:
        raise RuntimeError(f"measure failed for {node['id']}:\n{m.stdout}\n{m.stderr}")
    mo = re.search(r"auto_ground_offset would lift the WHOLE clip by ([+-]?[\d.]+) m",
                   m.stdout)
    mb = re.search(r"global min  = ([+-]?[\d.]+) m .* body='([^']+)'", m.stdout)
    if not mo:
        raise RuntimeError(f"could not parse offset for {node['id']}:\n{m.stdout}")
    offset = max(0.0, float(mo.group(1)))
    body = mb.group(2) if mb else "?"
    return dict(win=(s, e), offset=round(offset, 4), deep_body=body,
                trim_log=r.stdout.strip().splitlines()[-3:])


def fmt(xs):
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


def emit_env(node, meta):
    s, e = meta["win"]
    loop = round(2 * (e - s) - 2 / FPS, 2)
    init_hi = max(0.5, round(loop - 1.0, 2))
    term_dist = node.get("term_dist", 0.6)
    aux = node["aux"]
    L = []
    A = L.append
    A(f"# {node['family']} -- v2 HOLD node (auto-generated by tools/make_hold_nodes.py)")
    A(f"# Source clip: {node['clip']}")
    A(f"# Primary hold [{node['window'][0]}, {node['window'][1]}]s -> trimmed ping-pong")
    A(f"# window [{s}, {e}]s, loop {loop}s (data/motions/smpl_holds/{node['id']}_hold).")
    A(f"# ground_offset {meta['offset']} seats the HOLD support at z=0 (deepest body:")
    A(f"# {meta['deep_body']}; measured on the trimmed clip -- never auto_ground_offset).")
    A(f"# Recipe: Crow_pose.MD hold-node + Yoga_annotations.MD section 9.")
    A('env_name: "deepmimic"')
    A("")
    A("env:")
    A(f'    char_file: "{CHAR}"')
    A('    camera_mode: "track"')
    A("")
    A("    episode_length: 12.0 # seconds; several loop cycles -> sustained hold")
    A("    global_obs: True")
    A("    root_height_obs: True")
    A("    pose_termination: True")
    A(f"    pose_termination_dist: {term_dist} # m")
    A("    enable_phase_obs: False")
    A("    enable_tar_obs: True")
    A("    num_phase_encoding: 4")
    A("    tar_obs_steps: [1, 2, 3]")
    A("    rand_reset: True")
    A(f"    init_time_range: [0.0, {init_hi}]")
    A("")
    A("    ref_char_offset: [2.0, 0.0, 0.0] # m")
    A("    init_pose: [0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]")
    A("")
    A("    enable_early_termination: True")
    A('    key_bodies: ["L_Toe", "R_Toe", "Head", "L_Hand", "R_Hand"]')
    A(f"    contact_bodies: {fmt(node['fall'])} # FALL set (load-bearing bodies excluded)")
    A("")
    A(f'    motion_file: "{HOLD_DIR}/{node["id"]}_hold"')
    A("")
    A("    auto_ground_offset: False")
    A(f"    ground_offset: {meta['offset']}   # seats the hold support at z=0 (measured)")
    A("")
    A("    enable_contact_obs: True")
    A('    obs_contact_bodies: ["L_Wrist", "L_Hand", "R_Wrist", "R_Hand", "L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]')
    A("    obs_contact_force_threshold: 1.0 # N")
    A("    obs_contact_margin_cap: 1.0 # m")
    A("")
    A(f"    joint_err_w: {node['jw']}")
    A("")
    A("    reward_pose_w: 0.5")
    A("    reward_vel_w: 0.1")
    A("    reward_root_pose_w: 0.5")
    A("    reward_root_vel_w: 0.2")
    A("    reward_key_pos_w: 0.5")
    A("    reward_pose_scale: 0.25")
    A("    reward_vel_scale: 0.01")
    A("    reward_root_pose_scale: 5.0")
    A("    reward_root_vel_scale: 1.0")
    A("    reward_key_pos_scale: 10.0")
    A("")
    A("    # --- annotation-driven aux pack (weights 0 = term fully off) ---")
    if "toe_lift" in aux:
        t = aux["toe_lift"]
        A(f"    reward_toe_lift_w: {t['w']}")
        A("    reward_toe_lift_scale: 100.0")
        A("    toe_lift_min_h: 0.10")
        A(f"    toe_lift_ref_h: {t.get('ref_h', 0.12)}")
        A(f"    toe_lift_bodies: {fmt(t['bodies'])}")
    if "toe_pen" in aux:
        t = aux["toe_pen"]
        A(f"    reward_toe_force_pen_w: {t['w']}")
        A("    toe_force_pen_cap: 30.0")
        A(f"    toe_force_pen_ref_h: {t.get('ref_h', 0.12)}")
        A(f"    toe_force_pen_bodies: {fmt(t['bodies'])}")
    if "cop" in aux:
        c = aux["cop"]
        cop_bodies = sorted(set(c["support"]) | set(TOES_ANKLES))
        A(f"    reward_cop_support_w: {c['w']}")
        A(f"    reward_cop_support_scale: {c['scale']}")
        A("    cop_min_force: 10.0")
        A(f"    cop_bodies: {fmt(cop_bodies)}")
        A(f"    cop_support_bodies: {fmt(c['support'])}")
        if c.get("fwd", 0.0):
            A(f"    cop_support_forward_offset: {c['fwd']}")
        if c.get("left", 0.0):
            A(f"    cop_support_left_offset: {c['left']}")
    if "knee" in aux:
        k = aux["knee"]
        A(f"    reward_knee_support_w: {k['w']}")
        A(f"    reward_knee_support_scale: {k['scale']}")
        A(f"    knee_support_bodies: {fmt(k['bodies'])}")
        A(f"    knee_support_target_bodies: {fmt(k['targets'])}")
    if "inv" in aux:
        v = aux["inv"]
        A(f"    reward_inversion_w: {v['w']}")
        A("    inversion_up_threshold: -0.5")
        A("    inversion_force_threshold: 5.0")
        A(f"    inversion_bonus_bodies: {fmt(v['bodies'])}")
    A("")
    A("    # everything else off (force_balance rewards the toe-rest; keep 0)")
    A("    reward_com_support_w: 0.0")
    A("    reward_force_balance_w: 0.0")
    A("    reward_foot_clear_w: 0.0")
    A("    reward_orient_w: 0.0")
    A("    reward_energy_w: 0.05")
    A("    reward_energy_scale: 0.001")
    A("")
    A("")
    A("engine:")
    A('    engine_name: "isaac_lab"')
    A("")
    A('    control_mode: "pos"')
    A("    control_freq: 30")
    A("    sim_freq: 120")
    A("    env_spacing: 5")
    A("")
    A("    ground_contact_height: 0.3")
    A("")
    path = os.path.join(ENV_DIR, f"deepmimic_smpl_{node['id']}_hold_env.yaml")
    with open(os.path.join(REPO, path), "w") as f:
        f.write("\n".join(L))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma-separated node ids subset")
    args = p.parse_args()
    os.makedirs(os.path.join(REPO, HOLD_DIR), exist_ok=True)
    roster = ROSTER
    if args.only:
        want = set(args.only.split(","))
        roster = [n for n in ROSTER if n["id"] in want]

    by_id = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(trim_and_measure, n): n for n in roster}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                meta = fut.result()
            except Exception as exn:
                print(f"[FAIL] {n['id']}: {exn}", file=sys.stderr)
                continue
            path = emit_env(n, meta)
            by_id[n["id"]] = (n, meta, path)
            print(f"[ok] {n['id']:<14s} win={meta['win']} offset={meta['offset']:+.4f} "
                  f"(deepest: {meta['deep_body']}) -> {path}")

    if not args.only:
        qpath = os.path.join(REPO, "tools/hold_nodes_queue.txt")
        with open(qpath, "w") as f:
            f.write("# node env_config agent_config max_samples\n")
            for nid in QUEUE_ORDER:
                if nid not in by_id:
                    continue
                n, meta, path = by_id[nid]
                agent = AGENT_LOWLR if n["agent"] == "lowlr" else AGENT_STD
                f.write(f"{nid} {path} {agent} {n['samples']}\n")
        print(f"\nqueue manifest -> {qpath} ({len(by_id)} nodes)")
    missing = [n["id"] for n in roster if n["id"] not in by_id]
    if missing:
        print(f"[WARN] failed nodes: {missing}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
