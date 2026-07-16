"""Locate the cleanest tadasana->pose ENTRY transition in a source clip.

Reuses annotate_clips.annotate_clip(return_trace=True) so the per-frame signals
are IDENTICAL to the Stage-1 annotation (zero drift). Prints a phase timeline
(up_z / com / per-foot & per-hand ground clearance / horizontal foot & hand
speed / contact set) and auto-detects candidate ENTRY windows: the span from a
2-foot standing rest into the target hold. For unsegmented single-leg poses
(warrior3) the annotator lumps the whole clip into one hold, so we detect the
weight-shift breakpoint ourselves (contact-set change + foot lift + up_z drop).

Usage:
  env_isaaclab/bin/python tools/investigate_entry.py --clip <substr> [--fps_report 0.5]
"""
import argparse, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import anim.kin_char_model as kin_char_model
import anim.char_geoms as char_geoms
import annotate_clips as AC

CHAR_FILE = "data/assets/smpl/smpl_boxhands.xml"
MOTION_DIR = "data/motions/smpl"


def find_clip(substr):
    hits = [f for f in os.listdir(MOTION_DIR)
            if substr.lower() in f.lower() and not f.endswith(".png")
            and not f.endswith(".yaml")]
    # motion files have no extension in this corpus
    hits = [f for f in hits if "." not in f]
    if len(hits) != 1:
        print(f"substr '{substr}' -> {hits}")
        assert len(hits) == 1, "ambiguous or no match"
    return hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--stride_s", type=float, default=0.5, help="timeline print stride")
    args = p.parse_args()

    name = find_clip(args.clip)
    clip_file = os.path.join(MOTION_DIR, name)
    print(f"=== {name} ===")

    kcm = kin_char_model.KinCharModel("cpu")
    kcm.load_char_file(CHAR_FILE)
    body_names = kcm.get_body_names()
    parents = [int(kcm.get_parent_id(i)) for i in range(len(body_names))]
    geoms = char_geoms.load_char_geoms(CHAR_FILE, body_names, "cpu")
    idx = {n: i for i, n in enumerate(body_names)}

    ann, tr = AC.annotate_clip(name, clip_file, kcm, body_names, parents, geoms,
                               return_trace=True)
    fps = float(tr["fps"]); nf = int(tr["nf"])
    up_z = np.asarray(tr["up_z"]); com = np.asarray(tr["com"])
    bp = np.asarray(tr["body_pos"]); minz = np.asarray(tr["minz"])
    in_contact = np.asarray(tr["in_contact"]); floor = float(tr["floor"])

    LT, LA, RT, RA = idx["L_Toe"], idx["L_Ankle"], idx["R_Toe"], idx["R_Ankle"]
    LH, RH = idx["L_Hand"], idx["R_Hand"]
    LW, RW = idx["L_Wrist"], idx["R_Wrist"]
    PEL = idx["Pelvis"]

    # per-foot / per-hand ground clearance = min collision-geom z above floor
    Lz = np.minimum(minz[:, LT], minz[:, LA]) - floor
    Rz = np.minimum(minz[:, RT], minz[:, RA]) - floor
    LHz = np.minimum(minz[:, LH], minz[:, LW]) - floor
    RHz = np.minimum(minz[:, RH], minz[:, RW]) - floor
    pelz = bp[:, PEL, 2]

    # horizontal speeds (heading-agnostic, world xy) smoothed 5-frame
    def hspeed(b):
        v = np.gradient(bp[:, b, :2], axis=0) * fps
        s = np.linalg.norm(v, axis=-1)
        k = 5
        return np.convolve(s, np.ones(k) / k, mode="same")
    Lft_sp = np.maximum(hspeed(LT), hspeed(LA))
    Rft_sp = np.maximum(hspeed(RT), hspeed(RA))
    hand_sp = np.maximum(hspeed(LH), hspeed(RH))

    def cset(f):  # compact contact-set string
        feet = ("L" if in_contact[f, LT] or in_contact[f, LA] else "-") + \
               ("R" if in_contact[f, RT] or in_contact[f, RA] else "-")
        hands = ("l" if in_contact[f, LH] or in_contact[f, LW] else "-") + \
                ("r" if in_contact[f, RH] or in_contact[f, RW] else "-")
        return f"feet:{feet} hands:{hands}"

    holds = list(tr["holds"])          # list of (i0, i1) tuples
    prim = tr["primary"]               # (i0, i1) tuple
    print(f"frames={nf} fps={fps:.0f} len={nf/fps:.1f}s floor={floor:.3f}")
    print(f"holds (from annotation): primary={prim}")
    for (i0, i1) in sorted(holds, key=lambda h: h[0]):
        print(f"  hold f[{i0:>4}-{i1:>4}] "
              f"({i0/fps:5.1f}-{i1/fps:5.1f}s {(i1-i0)/fps:4.1f}s) upz~"
              f"{np.median(up_z[i0:i1+1]):+.2f}  {cset(i0)}->{cset(i1)}")

    print("\n t(s)   f   up_z  com_z  pelz  Lfoot Rfoot  Lhand Rhand  "
          "Lspd Rspd Hspd  contact")
    stride = max(1, int(args.stride_s * fps))
    for f in range(0, nf, stride):
        print(f"{f/fps:5.1f} {f:5d} {up_z[f]:+.2f} {com[f,2]:5.2f} {pelz[f]:5.2f}"
              f"  {Lz[f]*100:4.0f} {Rz[f]*100:4.0f}  {LHz[f]*100:5.0f}{RHz[f]*100:5.0f}"
              f"  {Lft_sp[f]:4.1f}{Rft_sp[f]:4.1f}{hand_sp[f]:4.1f}  {cset(f)}")

    # ---- auto-detect entry windows -------------------------------------------
    # standing = both feet in contact AND up_z>0.9 AND pelz>0.85
    standing = (in_contact[:, LT] | in_contact[:, LA]) & \
               (in_contact[:, RT] | in_contact[:, RA]) & (up_z > 0.9) & (pelz > 0.85)
    print("\n--- ENTRY DETECTION ---")
    # first sustained standing window at clip start
    if standing[:int(2 * fps)].any():
        s0 = 0
        # end of the opening standing rest = last standing frame before first
        # sustained non-standing (0.5s) block
        f = 0
        while f < nf and standing[f]:
            f += 1
        # allow brief dropouts: require 0.5s of non-standing to call it "left"
        while f < nf:
            if not standing[f:f + int(0.5 * fps)].any():
                break
            f += 1
        print(f"opening 2-foot standing rest: f[0-{f}] ({f/fps:.1f}s), "
              f"leaves standing ~f{f} (t={f/fps:.2f}s)")
    # target hold = the primary hold (or the most-inverted / most-single-support)
    if prim is not None:
        i0, i1 = prim
        print(f"primary hold starts f{i0} (t={i0/fps:.2f}s) "
              f"upz~{np.median(up_z[i0:i1+1]):+.2f} {cset(i0)}")
    # single-support onset: first frame where exactly one foot leaves contact
    # for >=0.5s while the other stays (warrior3 hinge)
    one_foot = ((in_contact[:, LT] | in_contact[:, LA]).astype(int) +
                (in_contact[:, RT] | in_contact[:, RA]).astype(int)) == 1
    on = None
    for f in range(int(1 * fps), nf - int(0.5 * fps)):
        if one_foot[f:f + int(0.5 * fps)].all():
            on = f; break
    if on is not None:
        lifted = "R" if (in_contact[on, LT] or in_contact[on, LA]) else "L"
        print(f"first sustained SINGLE-SUPPORT onset: f{on} (t={on/fps:.2f}s), "
              f"{lifted} foot lifts (stance={'L' if lifted=='R' else 'R'})")


if __name__ == "__main__":
    main()
