"""Mine the annotated yoga corpus for IN-CORPUS transition demonstrations.

For every clip: re-run the Stage-1 annotator with return_trace=True, compute a
rich fingerprint for EVERY hold window (not just the primary), label each hold
against the signed-off node inventory's reference fingerprints, and emit:

  * chains.txt        per-clip hold chain  (label / contacts / up_z / com / dur)
  * edges.json        every inter-hold segment = a real transition demo
  * edge_summary.txt  aggregated (from_label -> to_label) demo inventory

Outputs land in data/clip_annotations/transition_mining/. This is the demo
inventory behind Yoga_edge_framework_v3.md §1 (first run 2026-07-09).
Run: env_isaaclab/bin/python tools/mine_transitions.py   (repo root, ~3 min CPU)
"""
import os, sys, json, glob, fnmatch, collections

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "mimickit"))

import numpy as np
import torch

import annotate_clips as AC
import anim.kin_char_model as kin_char_model
import anim.char_geoms as char_geoms

OUT_DIR = os.path.join(REPO, "data", "clip_annotations", "transition_mining")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- node -> clip
# Signed-off inventory (Yoga_skill_nodes.MD HANDOFF §1) + parked/future nodes.
# Patterns are matched as substrings against clip basenames.
NODE_CLIPS = {
    # node            (clip substring pattern,        hold selector)
    "tadasana":       ("220926_Intense_Side_Stretch_Pose_or_Parsvottanasana_-b", "t:2.26-10.27"),
    "tree":           ("220923_Tree_Pose_or_Vrksasana_-a", "primary"),
    "eagle":          ("220926_Eagle_Pose_or_Garudasana_-a", "primary"),
    "plank":          ("220923_Plank_Pose_or_Kumbhakasana_-a", "primary"),
    "malasana":       ("Garland_Pose_or_Malasana_-a", "primary"),
    "side_plank":     ("220923_Side_Plank_Pose_or_Vasisthasana_-e", "primary"),
    "warrior2":       ("Virabhadrasana_II_-a", "primary"),
    "warrior3":       ("Virabhadrasana_III_-b", "primary"),
    "natarajasana":   ("220926_Lord_of_the_Dance_Pose_or_Natarajasana_-c", "primary"),
    "headstand":      ("220926_Supported_Headstand_pose_or_Salamba_Sirsasana_-a", "primary"),
    "handstand":      ("220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a", "primary"),
    "scorpion":       ("220923_Scorpion_pose_or_vrischikasana-b", "primary"),
    "downdog":        ("220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a", "primary"),
    "downdog3L":      ("220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-c", "primary"),
    "crow":           ("220923_Crane_Crow_Pose_or_Bakasana_-b", "primary"),
    "triangle":       ("Utthita_Trikonasana_-b", "primary"),
    "camel":          ("220923_Camel_Pose_or_Ustrasana_-c", "primary"),
    "half_moon":      ("220923_Half_Moon_Pose_or_Ardha_Chandrasana_-a", "primary"),
    "wheel":          ("220926_Upward_Bow_Wheel_Pose_or_Urdhva_Dhanurasana_-b", "primary"),
    # parked / deferred / future gateway nodes -- useful waypoint labels
    "pincha":         ("220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-c", "primary"),
    "dolphin_plank":  ("220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a", "primary"),
    "dolphin":        ("220923_Dolphin_Pose_or_Ardha_Pincha_Mayurasana_-a", "primary"),
    "chair":          ("220923_Chair_Pose_or_Utkatasana_-b", "primary"),
    "big_toe":        ("220923_Standing_big_toe_hold_pose_or_Utthita_Padangusthasana-c", "primary"),
    "uttanasana":     ("220923_Standing_Forward_Bend_pose_or_Uttanasana_-a", "primary"),
}

GROUP = {}
for s in "LR":
    GROUP[f"{s}_Wrist"] = f"hand:{s}"; GROUP[f"{s}_Hand"] = f"hand:{s}"
    GROUP[f"{s}_Ankle"] = f"foot:{s}"; GROUP[f"{s}_Toe"] = f"foot:{s}"
    GROUP[f"{s}_Elbow"] = f"elbow:{s}"; GROUP[f"{s}_Knee"] = f"knee:{s}"
    GROUP[f"{s}_Hip"] = "pelvis"; GROUP[f"{s}_Shoulder"] = f"shoulder:{s}"
    GROUP[f"{s}_Thorax"] = f"shoulder:{s}"
GROUP.update({"Pelvis": "pelvis", "Torso": "torso", "Spine": "torso",
              "Chest": "torso", "Neck": "head", "Head": "head"})

def canon(contact_bodies):
    return frozenset(GROUP.get(b, b) for b in contact_bodies)

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

# ---------------------------------------------------------------- fingerprints
def hold_features(trace, body_names, i0, i1, contact_bodies):
    """Rich per-hold fingerprint from the trace arrays."""
    bp = trace["body_pos"]            # (nf, nb, 3)
    com = trace["com"]                # (nf, 3) or (nf,)
    up = trace["up_z"]                # (nf,)
    floor = trace["floor"]
    sl = slice(i0, i1 + 1)
    idx = {n: k for k, n in enumerate(body_names)}
    def med_z(names):
        cols = [idx[n] for n in names]
        return float(np.median(bp[sl][:, cols, 2].numpy())) - floor
    com_z = com[sl][:, 2] if com.ndim == 2 else com[sl]
    return {
        "up_z": float(np.median(up[sl].numpy())),
        "com_z": float(np.median(com_z.numpy())) - floor,
        "head_z": med_z(["Head"]),
        "hands_z": med_z(["L_Hand", "R_Hand"]),
        "feet_z": med_z(["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]),
        "contacts": sorted(contact_bodies),
        "cset": canon(contact_bodies),
    }

W = dict(jac=2.0, up_z=1.0, com_z=1.5, head_z=0.8, hands_z=0.5, feet_z=0.5)
LABEL_THRESH = 1.10   # best score must be below this
MARGIN = 0.10         # ... and beat 2nd best by this, else mark ambiguous

def score(f, ref):
    s = W["jac"] * (1.0 - jaccard(f["cset"], ref["cset"]))
    for k in ("up_z", "com_z", "head_z", "hands_z", "feet_z"):
        s += W[k] * abs(f[k] - ref[k])
    return s

def describe(f):
    c = "+".join(sorted(f["cset"])) or "none"
    return f"[{c} | up_z {f['up_z']:+.2f} com {f['com_z']:.2f} head {f['head_z']:.2f}]"

# ------------------------------------------------------------------------ main
def main():
    kcm = kin_char_model.KinCharModel(AC.DEVICE)
    kcm.load_char_file(AC.CHAR_FILE)
    body_names = kcm.get_body_names()
    parents = [int(kcm.get_parent_id(i)) for i in range(len(body_names))]
    geoms = char_geoms.load_char_geoms(AC.CHAR_FILE, body_names, AC.DEVICE)

    clips = AC.list_clips()
    clips = [(n, p) for n, p in clips if not n.endswith("_hold")]
    print(f"mining {len(clips)} clips", flush=True)

    # pass 1: annotate everything once, keep per-hold features
    per_clip = {}
    for k, (name, path) in enumerate(clips):
        try:
            ann, trace = AC.annotate_clip(name, path, kcm, body_names, parents,
                                          geoms, return_trace=True)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True)
            continue
        holds = ann["segments"]["holds"]
        feats = [hold_features(trace, body_names, h["start_frame"],
                               h["end_frame"], h["contact_bodies"])
                 for h in holds]
        per_clip[name] = dict(ann=ann, feats=feats)
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(clips)}]", flush=True)
        del trace

    # node reference fingerprints (from each node clip's own hold)
    refs = {}
    for node, (pat, sel) in NODE_CLIPS.items():
        match = [n for n in per_clip if pat in n]
        if not match:
            print(f"  !! no clip for node {node} (pattern {pat})")
            continue
        name = sorted(match, key=len)[0]
        ann, feats = per_clip[name]["ann"], per_clip[name]["feats"]
        holds = ann["segments"]["holds"]
        if sel == "primary":
            hi = ann["segments"]["primary_hold_index"]
        else:  # "t:a-b" -> hold overlapping that time window most
            a, b = map(float, sel[2:].split("-"))
            hi = max(range(len(holds)), key=lambda i: min(holds[i]["end_s"], b)
                     - max(holds[i]["start_s"], a))
        refs[node] = dict(feats[hi], clip=name, hold=hi)
        print(f"  ref {node:14s} <- {name} hold#{hi} {describe(feats[hi])}")

    # pass 2: label every hold in every clip
    def label(f):
        scored = sorted((score(f, r), n) for n, r in refs.items())
        (s1, n1), (s2, n2) = scored[0], scored[1]
        if s1 < LABEL_THRESH and (s2 - s1) >= MARGIN:
            return n1, s1, f"{n2}:{s2:.2f}"
        if s1 < LABEL_THRESH:
            return f"~{n1}|{n2}", s1, f"{n2}:{s2:.2f}"   # ambiguous
        return "?", s1, f"{n1}:{s1:.2f}"

    edges = []
    chains_lines = []
    for name, d in sorted(per_clip.items()):
        ann, feats = d["ann"], d["feats"]
        seg = ann["segments"]
        holds, prim = seg["holds"], seg["primary_hold_index"]
        labels = []
        for i, (h, f) in enumerate(zip(holds, feats)):
            lab, s, alt = label(f)
            labels.append(lab)
            star = " *PRIMARY*" if i == prim else ""
            chains_lines.append(
                f"  hold{i:2d} {h['start_s']:7.2f}-{h['end_s']:7.2f}s "
                f"({h['dur_s']:5.2f}s) {lab:22s} s={s:.2f} alt={alt} "
                f"{describe(f)}{star}")
        chains_lines.insert(len(chains_lines) - len(holds),
                            f"\n{name}  ({ann['length_s']:.1f}s, "
                            f"{len(holds)} holds, primary #{prim})")
        for ih in seg["inter_hold"]:
            a, b = ih["from_hold"], ih["to_hold"]
            if a >= len(labels) or b >= len(labels):
                continue
            edges.append(dict(
                clip=name, from_hold=a, to_hold=b,
                from_label=labels[a], to_label=labels[b],
                from_desc=describe(feats[a]), to_desc=describe(feats[b]),
                start_frame=ih["start_frame"], end_frame=ih["end_frame"],
                dur_s=ih["dur_s"],
                from_dwell_s=holds[a]["dur_s"], to_dwell_s=holds[b]["dur_s"],
                involves_primary=(a == prim or b == prim)))

    with open(os.path.join(OUT_DIR, "chains.txt"), "w") as f:
        f.write("\n".join(chains_lines))
    with open(os.path.join(OUT_DIR, "edges.json"), "w") as f:
        json.dump(edges, f, indent=1)

    # aggregate: labeled-endpoint transition inventory
    agg = collections.defaultdict(list)
    for e in edges:
        if e["from_label"] not in ("?",) and e["to_label"] not in ("?",):
            agg[(e["from_label"], e["to_label"])].append(e)
    lines = ["IN-CORPUS TRANSITION DEMO INVENTORY (labeled endpoints only)",
             "=" * 70]
    for (a, b), es in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"\n{a:>22s} -> {b:<22s}  {len(es)} demo(s)")
        for e in sorted(es, key=lambda e: e["dur_s"]):
            lines.append(f"    {e['dur_s']:5.2f}s  frames {e['start_frame']:5d}-"
                         f"{e['end_frame']:5d}  {e['clip']}")
    with open(os.path.join(OUT_DIR, "edge_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote chains.txt, edges.json, edge_summary.txt -> {OUT_DIR}")
    print(f"{len(edges)} inter-hold transition segments, "
          f"{len(agg)} labeled directed pairs")

if __name__ == "__main__":
    main()
