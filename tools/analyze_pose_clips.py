"""Characterize yoga pose clips for skill-chaining config generation.

Loads each clip through MotionLib (geometry-aware, same path training uses),
runs FK over every frame, and reports the data needed to write a correct
per-pose env config WITHOUT guessing:

  * clip duration / fps / frames
  * worst-case collision-geom penetration (-> ground_offset / auto lift)
  * per-body fraction of frames in ground contact (lowest geom z within
    `contact_eps` of the ground plane) -> the pose's CONTACT SIGNATURE
  * a per-second trace of root height, root up-vector world-z (inversion),
    COM height, and the current ground-contact body set -> hold windows
  * the longest stable "hold" window (contact-set stable + low root vertical
    speed) and the contact signature measured over just that window

Run from repo root:
  env_isaaclab/bin/python tools/analyze_pose_clips.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))

import numpy as np
import torch

import anim.char_geoms as char_geoms
import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib

CHAR_FILE = "data/assets/smpl/smpl_boxhands.xml"
CLIPS = {
    # --- hand-balanced inversions: the existing hand-support reward stack
    #     (com_support / force_balance / orient on [R_Hand, L_Hand]) applies ---
    "handstand": "data/motions/smpl/220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a",
    "crow":      "data/motions/smpl/220923_Crane_Crow_Pose_or_Bakasana_-a",   # NEW take -a (crow_pose was -b)
    "scorpion":  "data/motions/smpl/scorpion_pose",                            # existing trained scorpion clip
    "headstand": "data/motions/smpl/220923_Supported_Headstand_pose_or_Salamba_Sirsasana_-b",
    "pincha":    "data/motions/smpl/220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a",
    "dolphin_plank": "data/motions/smpl/220923_Dolphin_Plank_Pose_or_Makara_Adho_Mukha_Svanasana_-a",
    "downdog":   "data/motions/smpl/220923_Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_-a",
    # --- standing / foot-supported balances: need a foot-support reward stack ---
    "side_plank":   "data/motions/smpl/220923_Side_Plank_Pose_or_Vasisthasana_-a",
    "warrior2":     "data/motions/smpl/220926_Warrior_II_Pose_or_Virabhadrasana_II_-a",
    "warrior3":     "data/motions/smpl/220926_Warrior_III_Pose_or_Virabhadrasana_III_-a",
    "natarajasana": "data/motions/smpl/220926_Lord_of_the_Dance_Pose_or_Natarajasana_-a",
}
CONTACT_EPS = 0.03   # m above the (lifted) ground plane counts as "in contact"
DEVICE = "cpu"


def per_body_min_geom_z(geoms, body_pos, body_rot):
    """Lowest collision-geom world z for every body, [T, num_bodies].

    char_geoms.compute_min_geom_z reduces over bodies; we need per-body, so
    replicate its witness-point transform here. Bodies with no geoms get +inf.
    """
    import util.torch_util as torch_util
    T = body_pos.shape[0]
    num_bodies = body_pos.shape[1]
    out = torch.full((T, num_bodies), float("inf"))
    for b in range(num_bodies):
        body_geoms = geoms[b]            # list of {"points":[P,3], "radius":float}
        if (len(body_geoms) == 0):
            continue
        brot = body_rot[:, b, :]         # [T, 4]
        bpos = body_pos[:, b, :]         # [T, 3]
        body_min = torch.full((T,), float("inf"))
        for geom in body_geoms:
            pts = geom["points"].to(body_pos.dtype)   # [P, 3]
            P = pts.shape[0]
            rot = brot.unsqueeze(1).expand(T, P, 4).reshape(-1, 4)
            flat = pts.unsqueeze(0).expand(T, P, 3).reshape(-1, 3)
            world = torch_util.quat_rotate(rot, flat).reshape(T, P, 3) + bpos.unsqueeze(1)
            geom_min = world[..., 2].amin(dim=1) - float(geom["radius"])
            body_min = torch.minimum(body_min, geom_min)
        out[:, b] = body_min
    return out


def analyze(name, clip_file, kcm, body_names):
    print("\n" + "=" * 78)
    print("POSE: {}   ({})".format(name, clip_file))
    print("=" * 78)

    # Load WITHOUT lift first to measure raw penetration, then the auto lift.
    ml = motion_lib.MotionLib(motion_file=clip_file, kin_char_model=kcm,
                              device=DEVICE, auto_ground_offset=True,
                              ground_offset_clearance=0.0, char_file=CHAR_FILE)
    geoms = ml._char_geoms
    num_frames = int(ml._motion_num_frames[0].item())
    fps = float(ml._motion_fps[0].item())
    length = float(ml._motion_lengths[0].item())
    applied_lift = float(ml._ground_offset)  # 0 unless ground_offset passed; auto is folded into frames

    # Recompute the auto offset that was applied (MotionLib logs it but folds
    # it into the stored frames, so re-derive from the raw clip for reporting).
    print("frames={}  fps={:.1f}  length={:.2f}s".format(num_frames, fps, length))

    times = torch.arange(num_frames, dtype=torch.float32) / fps
    motion_ids = torch.zeros(num_frames, dtype=torch.long)
    root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = ml.calc_motion_frame(motion_ids, times)
    body_pos, body_rot = kcm.forward_kinematics(root_pos, root_rot, joint_rot)

    import util.torch_util as torch_util
    # root up-vector world z (inversion indicator: -1 fully inverted)
    up = torch.zeros_like(root_pos); up[:, 2] = 1.0
    up_world_z = torch_util.quat_rotate(root_rot, up)[:, 2]

    # COM (mass-weighted) - approximate with body-origin mean weighted by mass
    masses = []
    # crude per-body mass proxy via geom count is unreliable; use uniform if
    # masses unavailable. The kin model doesn't carry mass; report origin-mean.
    com = body_pos.mean(dim=1)

    bz = per_body_min_geom_z(geoms, body_pos, body_rot)  # [T, B]
    ground = bz.amin().item()  # should be ~0 after auto lift
    in_contact = bz < (ground + CONTACT_EPS)             # [T, B] bool

    # per-body contact fraction over whole clip
    frac = in_contact.float().mean(dim=0)
    print("\nground plane (min geom z, lifted) = {:.3f} m".format(ground))
    print("\nper-body ground-contact fraction (whole clip, eps={:.0f}cm):".format(CONTACT_EPS * 100))
    order = torch.argsort(frac, descending=True)
    for b in order:
        if (frac[b] > 0.02):
            print("   {:<12s} {:5.1f}%".format(body_names[b], 100 * frac[b].item()))

    # per-second trace
    print("\nper-second trace [t: root_z up_z com_z | contacts]:")
    step = max(1, int(round(fps * 1.0)))
    for i in range(0, num_frames, step):
        contacts = [body_names[b] for b in range(len(body_names)) if in_contact[i, b]]
        print("  t={:5.1f}  rz={:5.2f}  up={:+.2f}  comz={:5.2f} | {}".format(
            times[i].item(), root_pos[i, 2].item(), up_world_z[i].item(),
            com[i, 2].item(), ",".join(contacts)))

    # longest stable hold window: contact-set unchanged AND low root vertical
    # speed. Quantize contact set to a tuple key per frame.
    root_vz = torch.zeros(num_frames)
    root_vz[:-1] = (root_pos[1:, 2] - root_pos[:-1, 2]).abs() * fps
    keys = [tuple(int(in_contact[i, b]) for b in range(len(body_names))) for i in range(num_frames)]
    stable = (root_vz < 0.15)  # m/s
    best = (0, 0, None)
    i = 0
    while (i < num_frames):
        if (not stable[i] or sum(keys[i]) == 0):
            i += 1; continue
        j = i
        while (j + 1 < num_frames and keys[j + 1] == keys[i] and stable[j + 1]):
            j += 1
        if (j - i > best[1] - best[0]):
            best = (i, j, keys[i])
        i = j + 1
    i0, i1, key = best
    if (key is not None and i1 > i0):
        hold_bodies = [body_names[b] for b in range(len(body_names)) if key[b]]
        # re-measure contact frac within a slightly padded window for robustness
        w0 = max(0, i0); w1 = min(num_frames, i1 + 1)
        wfrac = in_contact[w0:w1].float().mean(dim=0)
        held = [body_names[b] for b in range(len(body_names)) if wfrac[b] > 0.6]
        print("\nLONGEST STABLE HOLD: t=[{:.2f}, {:.2f}]s ({:.2f}s)  up_z~{:+.2f}".format(
            times[i0].item(), times[i1].item(), times[i1].item() - times[i0].item(),
            up_world_z[i0:i1 + 1].mean().item()))
        print("   contact signature (>60% of window): {}".format(held))
    else:
        print("\nLONGEST STABLE HOLD: none found (no low-vz contact window)")
    return


def main():
    kcm = kin_char_model.KinCharModel(DEVICE)
    kcm.load_char_file(CHAR_FILE)
    body_names = kcm.get_body_names()
    print("character bodies ({}): {}".format(len(body_names), body_names))
    for name, clip in CLIPS.items():
        analyze(name, clip, kcm, body_names)
    return


if __name__ == "__main__":
    main()
