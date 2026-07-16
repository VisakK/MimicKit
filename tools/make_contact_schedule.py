"""Build a per-frame contact schedule for an EDGE clip — the target flags for
DeepMimicEnv's `reward_contact_schedule_w` term (edge framework v3 §2.1).

PHASE-BASED, not geometry-thresholded: mid-transition contact is inherently
ambiguous at the 1-3 cm scale (this corpus's crow reference hovers its tucked
toes 2-5 cm up — the toe-dab geometry — so both the annotator's relative rule
and absolute height rules misread the tail; measured 2026-07-10). Contact is
therefore constrained ONLY where the demo is unambiguous:

  frames <= lead_end   : A's hold contact set   (listed bodies=1, others=0)
  frames >= tail_start : B's hold contact set   (from B's calibrated gate)
  in between           : -1 (don't-care) for every body

Flag values: 1 = must be loaded, 0 = must NOT be loaded, -1 = unconstrained.
The env term scores mean agreement over the constrained entries only.

Usage (crow entry):
  env_isaaclab/bin/python tools/make_contact_schedule.py \
      --motion_file data/motions/smpl_edges/tadasana_to_crow \
      --bodies L_Wrist,L_Hand,R_Wrist,R_Hand,L_Ankle,L_Toe,R_Ankle,R_Toe \
      --lead_end 1.0 --tail_start 2.9 \
      --a_contacts L_Ankle,L_Toe,R_Ankle,R_Toe \
      --b_contacts L_Hand,R_Hand --b_free L_Wrist,R_Wrist
(--b_free = don't-care in the tail — e.g. wrists, which may or may not carry
load within a planted-hand unit; --a_free likewise for the lead-in.)
Output: <motion_file>.contact_schedule.npz {flags [T,B] int8, bodies, fps}
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import anim.motion as motion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion_file", required=True)
    p.add_argument("--bodies", required=True, help="comma-separated, defines flag column order")
    p.add_argument("--lead_end", type=float, required=True, help="s; A-hold constraint ends here")
    p.add_argument("--tail_start", type=float, required=True, help="s; B-hold constraint starts here")
    p.add_argument("--a_contacts", default="", help="bodies loaded during the lead-in")
    p.add_argument("--a_free", default="", help="lead-in don't-care bodies")
    p.add_argument("--b_contacts", default="", help="bodies loaded during the tail")
    p.add_argument("--b_free", default="", help="tail don't-care bodies")
    p.add_argument("--events", action="store_true",
                   help="EVENT-BASED flags for the support bodies: detect swing "
                        "windows from horizontal body speed (robust to retarget "
                        "float — the 5cm height rule reads a floated support "
                        "foot as airborne; measured on malasana->tadasana "
                        "2026-07-11) and emit planted=1 / swing=0 / transition "
                        "buffer=-1 PER FRAME across the whole clip. Non-event "
                        "bodies keep the phase logic.")
    p.add_argument("--event_bodies", default="",
                   help="bodies to event-detect (default: --a_contacts)")
    p.add_argument("--speed_thr", type=float, default=0.25,
                   help="m/s; horizontal body speed above this = swing")
    p.add_argument("--buffer_frames", type=int, default=2,
                   help="frames of don't-care around each contact transition")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    bodies = args.bodies.split(",")
    out = args.out or (args.motion_file + ".contact_schedule.npz")
    m = motion.load_motion(args.motion_file)
    fps = float(m.fps)
    T = np.asarray(m.frames).shape[0]

    def phase_flags(contacts, free):
        con = set(filter(None, contacts.split(",")))
        fre = set(filter(None, free.split(",")))
        unknown = (con | fre) - set(bodies)
        assert not unknown, f"bodies not in --bodies list: {unknown}"
        return np.array([1 if b in con else (-1 if b in fre else 0)
                         for b in bodies], dtype=np.int8)

    flags = np.full((T, len(bodies)), -1, dtype=np.int8)
    i_lead = int(round(args.lead_end * fps))
    i_tail = int(round(args.tail_start * fps))
    flags[:i_lead] = phase_flags(args.a_contacts, args.a_free)
    flags[i_tail:] = phase_flags(args.b_contacts, args.b_free)

    if args.events:
        import torch
        import anim.kin_char_model as kin_char_model
        import util.torch_util as torch_util
        ev_bodies = list(filter(None, (args.event_bodies or args.a_contacts).split(",")))
        unknown = set(ev_bodies) - set(bodies)
        assert not unknown, f"event bodies not in --bodies list: {unknown}"
        kcm = kin_char_model.KinCharModel("cpu")
        kcm.load_char_file("data/assets/smpl/smpl_boxhands.xml")
        names = kcm.get_body_names()
        fr = np.asarray(m.frames, dtype=np.float32)
        rp = torch.tensor(fr[:, 0:3])
        rr = torch_util.exp_map_to_quat(torch.tensor(fr[:, 3:6]))
        jr = torch_util.quat_pos(kcm.dof_to_rot(torch.tensor(fr[:, 6:])))
        bp, _ = kcm.forward_kinematics(rp, rr, jr)
        bp = bp.numpy()
        k = np.ones(5) / 5.0   # 5-frame boxcar: fd noise sits well below 0.25 m/s
        for b in ev_bodies:
            xy = bp[:, names.index(b), :2]
            spd = np.linalg.norm(np.gradient(xy, axis=0), axis=-1) * fps
            spd = np.convolve(spd, k, mode="same")
            col = np.where(spd > args.speed_thr, 0, 1).astype(np.int8)
            trans = np.where(np.diff(col) != 0)[0]
            for t0 in trans:
                lo = max(0, t0 - args.buffer_frames + 1)
                hi = min(T, t0 + args.buffer_frames + 1)
                col[lo:hi] = -1
            flags[:, bodies.index(b)] = col
        n_sw = int((flags[:, [bodies.index(b) for b in ev_bodies]] == 0).sum())
        print(f"[events] {len(ev_bodies)} bodies, speed_thr {args.speed_thr} m/s, "
              f"{n_sw} swing body-frames detected")

    np.savez(out, flags=flags, bodies=np.array(bodies), fps=fps)

    sym = {1: "#", 0: ".", -1: " "}
    step = max(1, T // 78)
    print(f"{args.motion_file}: {T} frames @ {fps:.0f}fps; lead<= {args.lead_end}s, "
          f"tail>= {args.tail_start}s (blank = don't-care)")
    for k, b in enumerate(bodies):
        row = "".join(sym[int(flags[t, k])] for t in range(0, T, step))
        print(f"  {b:8s} |{row}|")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
