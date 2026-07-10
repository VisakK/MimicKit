"""Stage-1 kinematic annotation of the yoga motion corpus (no GPU, no learning).

For every clip in data/motions/smpl/ this runs geometry-aware FK (the same
MotionLib + KinCharModel path training uses) and writes a per-clip annotation
config that later training scripts (and the skill-graph clustering) consume.

Per clip we annotate (all derived from FK geometry, NOT from a policy or the
physics solver -- teleported poses have ill-defined solver forces, but the
contact SET, support geometry and penetration are deterministic):

  * ground penetration  -> the ground_offset / auto_ground_offset a training
    config needs (worst pre-lift collision-geom dip below the floor, per body)
  * contact signature   -> which bodies are load-bearing during the hold
  * support polygon      -> base-of-support span / convex-hull area / aspect and
    the signed COM-over-support margin (the balance state variable)
  * COM height           -> mass-weighted (anthropometric segment fractions; see
    MASS_FRACTIONS -- a kinematic estimate, swap for sim masses if exactness
    matters)
  * body-on-body proximity -> non-adjacent collision geoms resting on each other
    (defines eight-angle / Koundinyasana / firefly arm-balances the ground
    sensor is blind to). body_on_body is leg x arm (unchanged, consumed by
    index.json / anchor_probe); body_on_body_ext adds cross-side leg x leg and
    arm x arm pairs (tree foot-on-thigh press, eagle leg + arm wraps)
  * hold segments        -> every stable held-pose window (multi-attempt clips
    have several) with frame ranges + per-window contact signature
  * entry / exit / inter-hold segments -> frame ranges. Each single-pose clip is
    two transition demonstrations: hub->pose (entry) and pose->hub (exit). These
    become directed transition edges in the skill graph; the entry-start and
    exit-end poses are fingerprinted too so endpoint clustering can discover the
    shared hubs (standing / squat / quadruped / supine / seated).
  * L/R symmetry         -> heading-local mirror discrepancy in [0,1]

Outputs:
  data/clip_annotations/<clip>.yaml     one config per clip (the deliverable)
  data/clip_annotations/index.json      flat per-clip fingerprint table for
                                        clustering + the anchor validation probe

Run from repo root:
  env_isaaclab/bin/python tools/annotate_clips.py            # all clips
  env_isaaclab/bin/python tools/annotate_clips.py --clips handstand,crow  # subset by substring
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))

import numpy as np
import torch
import yaml

import anim.kin_char_model as kin_char_model
import anim.motion_lib as motion_lib
import anim.char_geoms as char_geoms
import util.torch_util as torch_util

CHAR_FILE = "data/assets/smpl/smpl_boxhands.xml"
MOTION_DIR = "data/motions/smpl"
OUT_DIR = "data/clip_annotations"
DEVICE = "cpu"

CONTACT_EPS = 0.05       # m: lowest geom within this of the floor -> in contact.
                         # 5cm (not 3) absorbs this mocap's ~2-3cm L/R stagger and
                         # float/skate, so symmetric supports register on both sides.
HOLD_SPEED = 0.20        # m/s: mean body speed below this -> quasi-static
MIN_HOLD_S = 0.8         # s: shortest window we call a "hold"
ONBODY_EPS = 0.03        # m: non-adjacent geoms closer than this -> on-body support
ONBODY_EXT_EPS = 0.05    # m: looser eps for body_on_body_ext (cross-side leg x leg /
                         # arm x arm). Measured: every true press/wrap pair on
                         # tree + eagle sits at <= +0.039 (eagle wrist x wrist
                         # +0.019, ankle wrap +0.039) while the first NOT-touching
                         # pairs on unrelated clips appear at >= +0.054 (boat
                         # ankle x toe, crow hip x ankle). 0.05 clears the real
                         # contacts by the corpus's ~2-3cm mocap L/R stagger
                         # without admitting that noise band; 0.03 would clip the
                         # eagle wraps to a 1cm margin.
ENDPOINT_WIN_S = 0.5     # s: averaging window at entry-start / exit-end
CONTACT_FRAC = 0.6       # fraction of a window a body must contact to be "load-bearing"

# Clips that are not single-subject yoga captures -> skip.
SKIP_SUBSTR = ("smpl_",)  # locomotion .pkl clips (smpl_walk, smpl_jog, ...)
SKIP_EXACT = {"yoga.pkl", "yoga_conversion_log.csv"}

# Anthropometric segment-mass fractions mapped onto the 24 SMPL bodies (relative;
# normalized at load). A kinematic COM estimate -- the XML carries no masses.
# Replace with measured sim rigid-body masses if COM must match training exactly.
MASS_FRACTIONS = {
    "Pelvis": 0.120, "Torso": 0.100, "Spine": 0.100, "Chest": 0.100,
    "Neck": 0.020, "Head": 0.073,
    "L_Hip": 0.100, "L_Knee": 0.047, "L_Ankle": 0.013, "L_Toe": 0.002,
    "R_Hip": 0.100, "R_Knee": 0.047, "R_Ankle": 0.013, "R_Toe": 0.002,
    "L_Thorax": 0.008, "L_Shoulder": 0.027, "L_Elbow": 0.016, "L_Wrist": 0.005, "L_Hand": 0.006,
    "R_Thorax": 0.008, "R_Shoulder": 0.027, "R_Elbow": 0.016, "R_Wrist": 0.005, "R_Hand": 0.006,
}

MIDLINE_BODIES = ("Pelvis", "Torso", "Spine", "Chest", "Neck", "Head")


def parse_family(clip_name):
    """Strip session prefix + variant suffix -> (family, variant)."""
    import re
    stem = clip_name
    m = re.match(r"^(220923|220926)_(.*)$", stem)
    if m:
        stem = m.group(2)
    variant = None
    m = re.match(r"^(.*?)[_-]([a-z])$", stem)     # trailing _-a / -a / _a
    if m:
        stem, variant = m.group(1), m.group(2)
    stem = stem.rstrip("_-")
    if stem.endswith("_hold"):
        stem = stem[:-5]
        variant = "hold"
    return stem, variant


def per_body_witness(geoms, body_pos, body_rot):
    """Per body: lowest collision-geom world z [T,B] and the xy of that witness
    point [T,B,2]. Bodies with no geoms get +inf z / nan xy."""
    T, B = body_pos.shape[0], body_pos.shape[1]
    min_z = torch.full((T, B), float("inf"))
    wit_xy = torch.full((T, B, 2), float("nan"))
    for b in range(B):
        bg = geoms[b]
        if len(bg) == 0:
            continue
        brot = body_rot[:, b, :]
        bpos = body_pos[:, b, :]
        # stack all geom corner points (radius folded into an effective z drop)
        allpts = []
        allrad = []
        for g in bg:
            pts = g["points"].to(body_pos.dtype)   # [P,3]
            allpts.append(pts)
            allrad.append(torch.full((pts.shape[0],), float(g["radius"]), dtype=body_pos.dtype))
        pts = torch.cat(allpts, dim=0)             # [P,3]
        rad = torch.cat(allrad, dim=0)             # [P]
        P = pts.shape[0]
        rot = brot.unsqueeze(1).expand(T, P, 4).reshape(-1, 4)
        flat = pts.unsqueeze(0).expand(T, P, 3).reshape(-1, 3)
        world = torch_util.quat_rotate(rot, flat).reshape(T, P, 3) + bpos.unsqueeze(1)
        eff_z = world[..., 2] - rad.unsqueeze(0)   # [T,P] lowest surface point
        z_min, z_arg = eff_z.min(dim=1)            # [T]
        min_z[:, b] = z_min
        wit_xy[:, b, :] = torch.gather(
            world[..., :2], 1, z_arg.view(T, 1, 1).expand(T, 1, 2)).squeeze(1)
    return min_z, wit_xy


def body_spheres(geoms, body_pos_f, body_rot_f):
    """One bounding sphere per body at a single frame: (center[B,3], radius[B]).
    Used for cheap non-adjacent body-on-body proximity. inf radius where empty."""
    B = body_pos_f.shape[0]
    ctr = body_pos_f.clone()
    rad = torch.full((B,), float("-inf"))
    for b in range(B):
        bg = geoms[b]
        if len(bg) == 0:
            rad[b] = float("-inf"); continue
        pts = torch.cat([g["points"].to(body_pos_f.dtype) for g in bg], dim=0)
        gr = max(float(g["radius"]) for g in bg)
        world = torch_util.quat_rotate(
            body_rot_f[b].unsqueeze(0).expand(pts.shape[0], 4), pts) + body_pos_f[b]
        c = world.mean(dim=0)
        spread = (world - c).norm(dim=-1).max().item()
        ctr[b] = c
        rad[b] = spread + gr
    return ctr, rad


def support_metrics(pts_xy, com_xy, num_dirs=64):
    """pts_xy [K,2] contact-vertex positions; com_xy [2]. Returns
    (span, area, aspect, margin). margin>0 => COM inside the support hull."""
    K = pts_xy.shape[0]
    if K == 0:
        return 0.0, 0.0, 1.0, -1.0
    if K == 1:
        d = float(np.linalg.norm(pts_xy[0] - com_xy))
        return 0.0, 0.0, 1.0, -d
    # span = max pairwise distance
    dif = pts_xy[:, None, :] - pts_xy[None, :, :]
    span = float(np.sqrt((dif ** 2).sum(-1)).max())
    # aspect from PCA of the point set
    c = pts_xy - pts_xy.mean(0)
    cov = c.T @ c / max(K - 1, 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-9, None)
    aspect = float(min(np.sqrt(ev[-1] / ev[0]), 20.0))   # cap near-collinear blowup
    # area via convex hull (0 for collinear / <3)
    area = 0.0
    if K >= 3:
        try:
            from scipy.spatial import ConvexHull
            area = float(ConvexHull(pts_xy).volume)  # 2D "volume" == area
        except Exception:
            area = 0.0
    # signed support-function margin (positive inside), works for any K>=2
    ang = np.linspace(0, 2 * np.pi, num_dirs, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], -1)      # [D,2]
    hull_sup = (pts_xy @ dirs.T).max(0)                  # [D]
    com_sup = com_xy @ dirs.T                            # [D]
    margin = float((hull_sup - com_sup).min())
    return span, area, aspect, margin


def lr_symmetry(body_pos_local, body_names):
    """Heading-local mirror discrepancy -> symmetry in [0,1] (1 = symmetric).
    Mirror = negate y (left axis) then swap L_/R_ bodies."""
    name2i = {n: i for i, n in enumerate(body_names)}
    mirrored = body_pos_local.clone()
    mirrored[:, 1] = -mirrored[:, 1]
    dists = []
    for n, i in name2i.items():
        if n.startswith("L_"):
            j = name2i.get("R_" + n[2:])
            if j is not None:
                dists.append((body_pos_local[i] - mirrored[j]).norm().item())
        elif n in MIDLINE_BODIES:
            dists.append((body_pos_local[i] - mirrored[i]).norm().item())
    if not dists:
        return 1.0
    return float(np.exp(-np.mean(dists) / 0.25))


def find_holds(body_speed, fps, num_frames):
    """All maximal quasi-static (low mean body speed) windows >= MIN_HOLD_S.
    Segmenting on speed alone -- NOT on a constant per-frame contact set -- is
    robust to a support flickering across the contact threshold (which otherwise
    shatters one hold into many). The window's contact signature is then the set
    of bodies contacting for >CONTACT_FRAC of it. Returns (i0,i1) sorted by start."""
    stable = body_speed < HOLD_SPEED
    min_len = max(1, int(round(MIN_HOLD_S * fps)))
    holds = []
    i = 0
    while i < num_frames:
        if not stable[i]:
            i += 1; continue
        j = i
        while j + 1 < num_frames and stable[j + 1]:
            j += 1
        if (j - i + 1) >= min_len:
            holds.append((i, j))
        i = j + 1
    return holds


def contact_bodies_in(in_contact, i0, i1, body_names):
    frac = in_contact[i0:i1 + 1].float().mean(dim=0)
    return [body_names[b] for b in range(len(body_names)) if frac[b] > CONTACT_FRAC]


FOOT_BODIES = {"L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "L_Knee", "R_Knee"}
STANDING_COM_H = 0.9   # approx neutral-standing COM height (m), for salience


def hold_salience(i0, i1, up_z, com, in_contact, body_names):
    """How non-neutral (asana-like) a held window is. The characteristic asana
    hold is the MOST salient, not merely the longest -- an incidental standing
    rest scores ~0, an inversion / arm-balance / low pose scores high. Combines
    inversion, non-foot support, and low COM."""
    uz = float(up_z[i0:i1 + 1].mean())
    cm = float(com[i0:i1 + 1, 2].mean())
    cbs = contact_bodies_in(in_contact, i0, i1, body_names)
    nonfoot = sum(1 for b in cbs if b not in FOOT_BODIES)
    nonfoot_frac = nonfoot / max(len(cbs), 1)
    return 0.5 * (1.0 - uz) + nonfoot_frac + max(0.0, (STANDING_COM_H - cm) / STANDING_COM_H)


def window_fingerprint(i0, i1, ctx):
    """Support-mode fingerprint over frames [i0,i1] (inclusive)."""
    bn = ctx["body_names"]
    in_contact = ctx["in_contact"]
    wit_xy = ctx["wit_xy"]
    body_pos = ctx["body_pos"]
    up_z = ctx["up_z"]
    com = ctx["com"]
    geoms = ctx["geoms"]
    body_rot = ctx["body_rot"]
    parents = ctx["parents"]

    cbodies = contact_bodies_in(in_contact, i0, i1, bn)
    cids = [bn.index(b) for b in cbodies]

    # support vertices = mean witness xy of each contacting body over the window
    pts = []
    for b in cids:
        m = in_contact[i0:i1 + 1, b]
        xy = wit_xy[i0:i1 + 1, b, :][m]
        if xy.shape[0] > 0 and not torch.isnan(xy).any():
            pts.append(xy.mean(dim=0).numpy())
    pts = np.array(pts) if pts else np.zeros((0, 2))
    com_xy = com[i0:i1 + 1, :2].mean(dim=0).numpy()
    span, area, aspect, margin = support_metrics(pts, com_xy)

    # body-on-body: cheap sphere test on the window midpoint pose. Restricted to
    # LEG x ARM proximity = a leg resting on an arm (crow shin-on-triceps,
    # eight-angle / Koundinyasana thigh-on-arm, firefly) -- the on-body support
    # that defines arm-balances and that the ground sensor is blind to. Skips
    # trunk-internal compactness (hips-near-spine etc.), which is not support.
    mid = (i0 + i1) // 2
    ctr, rad = body_spheres(geoms, body_pos[mid], body_rot[mid])
    onbody = []
    onbody_ext = []
    ground = set(cids)
    B = len(bn)
    is_leg = lambda n: any(k in n for k in ("Hip", "Knee", "Ankle", "Toe"))
    is_arm = lambda n: any(k in n for k in ("Shoulder", "Elbow", "Wrist", "Hand"))
    # ext pair classes = the positive contacts a finetune stage should REWARD
    # (contacts that SHOULD exist), which the leg x arm class misses:
    #   LEG x LEG, cross-side only -- tree's lifted foot pressed into the standing
    #   inner thigh, eagle's leg wrap. Same-side leg pairs are chain-adjacent and
    #   trivially near, and Hip x Hip is pelvis anatomy, not contact -> excluded.
    #   ARM x ARM (Elbow/Wrist/Hand, cross-side) -- the eagle arm wrap.
    is_wrap_arm = lambda n: any(k in n for k in ("Elbow", "Wrist", "Hand"))
    cross_side = lambda x, y: x[:2] != y[:2]            # L_ vs R_ prefixes
    for a in range(B):
        for b in range(a + 1, B):
            if rad[a] == float("-inf") or rad[b] == float("-inf"):
                continue
            na, nb = bn[a], bn[b]
            legarm = (is_leg(na) and is_arm(nb)) or (is_arm(na) and is_leg(nb))
            legleg = (is_leg(na) and is_leg(nb) and cross_side(na, nb)
                      and not ("Hip" in na and "Hip" in nb))
            armarm = is_wrap_arm(na) and is_wrap_arm(nb) and cross_side(na, nb)
            if not (legarm or legleg or armarm):
                continue
            if a in ground and b in ground:             # both on floor -> not on-body
                continue
            d = (ctr[a] - ctr[b]).norm().item() - rad[a].item() - rad[b].item()
            if legarm and d < ONBODY_EPS:
                onbody.append([na, nb, round(d, 3)])
            elif (legleg or armarm) and d < ONBODY_EXT_EPS:
                onbody_ext.append([na, nb, round(d, 3)])

    # heading-local pose at window mid for symmetry
    hinv = torch_util.calc_heading_quat_inv(ctx["root_rot"][mid:mid + 1])
    loc = body_pos[mid] - ctx["root_pos"][mid]
    loc = torch_util.quat_rotate(hinv.expand(B, 4), loc)
    sym = lr_symmetry(loc, bn)

    # vertical structure above the floor: separates poses that share a contact
    # set (headstand head is low, pincha head is high; handstand hips high).
    minz = ctx["minz"]
    floor = ctx["floor"]

    def h(names):
        ids = [bn.index(n) for n in names if n in bn]
        return round(float(minz[i0:i1 + 1, ids].amin(dim=1).mean() - floor), 3) if ids else None

    return {
        "up_z": round(float(up_z[i0:i1 + 1].mean()), 3),
        "com_height": round(float(com[i0:i1 + 1, 2].mean()), 3),
        "head_height": h(["Head"]),
        "hips_height": h(["Pelvis"]),
        "feet_height": h(["L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]),
        "hands_height": h(["L_Hand", "R_Hand", "L_Wrist", "R_Wrist"]),
        "contact_bodies": cbodies,
        "num_supports": len(cbodies),
        "support_span_m": round(span, 3),
        "support_area_m2": round(area, 4),
        "support_aspect": round(aspect, 2),
        "com_margin_m": round(margin, 3),
        "body_on_body": onbody,
        "body_on_body_ext": onbody_ext,
        "lr_symmetry": round(sym, 3),
        "is_inverted": bool(up_z[i0:i1 + 1].mean() < -0.3),
    }


def coarse_support_mode(fp):
    """Heuristic regime label from the primary-hold fingerprint (a sanity readout
    only; the real partition comes from CLUSTERING the full fingerprint vector).
    Uses which limbs actually bear load -- the lowest ones -- not just the
    contact set, so a grazing toe doesn't turn an arm-balance into standing."""
    cb = fp["contact_bodies"]
    forearms = any("Elbow" in b for b in cb)
    hands_ct = any(("Hand" in b or "Wrist" in b) for b in cb)
    feet_ct = any(("Toe" in b or "Ankle" in b) for b in cb)
    uz = fp["up_z"]
    hh = fp["hands_height"] if fp["hands_height"] is not None else 9.9
    fh = fp["feet_height"] if fp["feet_height"] is not None else 9.9
    hd = fp["head_height"] if fp["head_height"] is not None else 9.9
    inv = uz < -0.5
    hands_bearing = hands_ct and hh < 0.10 and hh <= fh + 0.03   # hands on/near floor
    if inv:
        if hd < 0.10:
            return "head_inversion"        # crown near floor (tripod/sirsasana)
        if forearms:
            return "forearm_inversion"     # pincha / supported headstand on forearms
        return "hands_inverted"            # handstand
    if hands_bearing and fh > hh + 0.05:   # hands loaded, feet lifted off -> arm balance
        return "hands_arm_balance"         # crow, firefly, eight-angle, side-crow
    if feet_ct:
        return "standing_balance" if fp["num_supports"] <= 2 else "standing_wide"
    return "grounded_or_supine"


def annotate_clip(name, clip_file, kcm, body_names, parents, geoms, return_trace=False):
    # raw (no lift) pass -> penetration / ground_offset
    ml_raw = motion_lib.MotionLib(motion_file=clip_file, kin_char_model=kcm,
                                  device=DEVICE, auto_ground_offset=False,
                                  char_file=CHAR_FILE)
    nf = int(ml_raw._motion_num_frames[0].item())
    fps = float(ml_raw._motion_fps[0].item())
    times = torch.arange(nf, dtype=torch.float32) / fps
    mids = torch.zeros(nf, dtype=torch.long)
    rp, rr, rv, rav, jr, dv = ml_raw.calc_motion_frame(mids, times)
    bp, br = kcm.forward_kinematics(rp, rr, jr)
    raw_minz, _ = per_body_witness(geoms, bp, br)
    floor_raw = 0.0
    pen = (floor_raw - raw_minz.amin(dim=0)).clamp(min=0.0)   # per-body dip below floor
    worst_pen = float(pen.max())
    per_body_pen = {body_names[b]: round(float(pen[b]), 4)
                    for b in range(len(body_names)) if pen[b] > 0.005}

    # lifted pass -> everything else on a clean floor at ~0
    ml = motion_lib.MotionLib(motion_file=clip_file, kin_char_model=kcm,
                              device=DEVICE, auto_ground_offset=True,
                              ground_offset_clearance=0.0, char_file=CHAR_FILE)
    rp, rr, rv, rav, jr, dv = ml.calc_motion_frame(mids, times)
    bp, br = kcm.forward_kinematics(rp, rr, jr)
    minz, wit_xy = per_body_witness(geoms, bp, br)
    floor = float(minz.amin())                       # ~0 after auto lift; for heights
    # Contact is per-frame relative to the LOWEST body that frame, not the global
    # lift point: robust to the auto-offset over-lifting the hold when some other
    # frame/limb dips deeper, and to L/R stagger. Something is always grounded in
    # a yoga hold, so the lowest body defining the floor each frame is correct.
    floor_t = minz.amin(dim=1, keepdim=True)
    in_contact = (minz - floor_t) < CONTACT_EPS

    up = torch.zeros_like(rp); up[:, 2] = 1.0
    up_z = torch_util.quat_rotate(rr, up)[:, 2]

    mass = torch.tensor([MASS_FRACTIONS.get(n, 0.0) for n in body_names])
    mass = mass / mass.sum()
    com = torch.einsum("tbk,b->tk", bp, mass)

    body_speed = torch.zeros(nf)
    if nf > 1:
        body_speed[:-1] = (bp[1:] - bp[:-1]).norm(dim=-1).mean(dim=1) * fps
        body_speed[-1] = body_speed[-2]

    ctx = dict(body_names=body_names, in_contact=in_contact, wit_xy=wit_xy,
               body_pos=bp, body_rot=br, up_z=up_z, com=com, geoms=geoms,
               parents=parents, root_rot=rr, root_pos=rp, minz=minz, floor=floor)

    # quasi-static windows, keep only those with a real ground contact
    holds = [h for h in find_holds(body_speed, fps, nf)
             if contact_bodies_in(in_contact, h[0], h[1], body_names)]
    # primary = the most salient (asana-like) hold, not merely the longest.
    primary = max(holds, key=lambda h: hold_salience(h[0], h[1], up_z, com,
                                                      in_contact, body_names),
                  default=None)

    # Sharpen the fingerprint window: a long hold can sweep through the pose
    # (headstand legs move -> up_z swings [-1,+0.5], mean washes out). Describe a
    # short window around the PEAK-salience frame inside the primary hold instead.
    nonfoot_row = torch.tensor([0.0 if n in FOOT_BODIES else 1.0 for n in body_names])
    ncon = in_contact.sum(dim=1).clamp(min=1)
    frame_sal = (0.5 * (1.0 - up_z)
                 + (in_contact.float() * nonfoot_row).sum(dim=1) / ncon
                 + ((STANDING_COM_H - com[:, 2]) / STANDING_COM_H).clamp(min=0.0))
    sig_win = primary
    if primary is not None:
        i0, i1 = primary
        peak = i0 + int(torch.argmax(frame_sal[i0:i1 + 1]))
        w = max(1, int(round(0.75 * fps)))
        sig_win = (max(i0, peak - w), min(i1, peak + w))

    def seg(i0, i1):
        return {"start_frame": int(i0), "end_frame": int(i1),
                "start_s": round(i0 / fps, 2), "end_s": round(i1 / fps, 2),
                "dur_s": round((i1 - i0) / fps, 2)}

    hold_dicts = []
    for (i0, i1) in holds:
        d = seg(i0, i1)
        d["contact_bodies"] = contact_bodies_in(in_contact, i0, i1, body_names)
        d["up_z"] = round(float(up_z[i0:i1 + 1].mean()), 3)
        hold_dicts.append(d)

    holds_time = sorted(holds, key=lambda h: h[0])
    inter = []
    for k in range(len(holds_time) - 1):
        g0, g1 = holds_time[k][1], holds_time[k + 1][0]
        if g1 - g0 > 1:
            s = seg(g0, g1); s["from_hold"] = k; s["to_hold"] = k + 1
            inter.append(s)

    def endpoint_fp(i0, i1):
        fp = window_fingerprint(i0, i1, ctx)
        return {"up_z": fp["up_z"], "com_height": fp["com_height"],
                "contact_bodies": fp["contact_bodies"], "support_mode": coarse_support_mode(fp)}

    # entry/exit MOVEMENTS bracketing the asana (primary) hold: the mocap of
    # getting INTO and OUT OF the pose. Each becomes a directed transition edge
    # (hub->pose, pose->hub) in the skill graph; the immediately-bracketing holds
    # are the poses it connects. rest_start/rest_end (the clip's first/last hold)
    # are the reliable rest hubs for endpoint clustering -- clips begin and end in
    # a standing/seated rest even when the entry field itself is empty.
    entry_tr = exit_tr = None
    endpoints = {}
    if primary is not None and holds_time:
        p = holds_time.index(primary)
        prev_h = holds_time[p - 1] if p > 0 else None
        next_h = holds_time[p + 1] if p < len(holds_time) - 1 else None
        entry_tr = seg(prev_h[1] if prev_h else 0, primary[0])
        exit_tr = seg(primary[1], next_h[0] if next_h else nf - 1)
        endpoints["rest_start"] = endpoint_fp(*holds_time[0])
        endpoints["rest_end"] = endpoint_fp(*holds_time[-1])
        if prev_h is not None:
            endpoints["entry_from"] = endpoint_fp(*prev_h)
        if next_h is not None:
            endpoints["exit_to"] = endpoint_fp(*next_h)

    hold_fp = window_fingerprint(*sig_win, ctx) if primary is not None else None

    family, variant = parse_family(name)
    ann = {
        "clip": name,
        "motion_file": clip_file,
        "family": family,
        "variant": variant,
        "char_file": CHAR_FILE,
        "fps": round(fps, 3),
        "num_frames": nf,
        "length_s": round(nf / fps, 2),
        "ground": {
            "worst_penetration_m": round(worst_pen, 4),
            "recommended_ground_offset_m": round(worst_pen, 4),
            "auto_ground_offset_applied_m": round(float(ml._ground_offset), 4),
            "per_body_penetration_m": per_body_pen,
        },
        "segments": {
            "holds": hold_dicts,
            "primary_hold_index": (holds.index(primary) if primary is not None else None),
            "entry_transition": entry_tr,
            "exit_transition": exit_tr,
            "inter_hold": inter,
        },
        "hold_signature": hold_fp,
        "endpoints": endpoints,
        "suggested": {
            "contact_bodies": (hold_fp["contact_bodies"] if hold_fp else []),
            "is_inverted": (hold_fp["is_inverted"] if hold_fp else None),
            "support_mode": (coarse_support_mode(hold_fp) if hold_fp else None),
        },
    }
    if return_trace:
        # per-frame arrays + derived structures for visualization -- exactly what
        # the annotation was computed from, so the viz can't drift from the YAML.
        trace = dict(nf=nf, fps=fps, body_pos=bp, body_rot=br, minz=minz,
                     in_contact=in_contact, wit_xy=wit_xy, up_z=up_z, com=com,
                     body_speed=body_speed, frame_sal=frame_sal, floor=floor,
                     holds=holds, primary=primary, sig_win=sig_win)
        return ann, trace
    return ann


def flat_row(ann):
    """One flat record for the clustering index / anchor probe."""
    fp = ann["hold_signature"] or {}
    return {
        "clip": ann["clip"], "family": ann["family"], "variant": ann["variant"],
        "length_s": ann["length_s"], "num_holds": len(ann["segments"]["holds"]),
        "primary_hold_dur_s": (ann["segments"]["holds"][ann["segments"]["primary_hold_index"]]["dur_s"]
                               if ann["segments"]["primary_hold_index"] is not None else 0.0),
        "up_z": fp.get("up_z"), "com_height": fp.get("com_height"),
        "head_height": fp.get("head_height"), "hips_height": fp.get("hips_height"),
        "feet_height": fp.get("feet_height"), "hands_height": fp.get("hands_height"),
        "num_supports": fp.get("num_supports"), "support_span_m": fp.get("support_span_m"),
        "support_area_m2": fp.get("support_area_m2"), "support_aspect": fp.get("support_aspect"),
        "com_margin_m": fp.get("com_margin_m"), "lr_symmetry": fp.get("lr_symmetry"),
        "is_inverted": fp.get("is_inverted"),
        "contact_bodies": fp.get("contact_bodies"),
        "n_body_on_body": len(fp.get("body_on_body", [])),
        "support_mode": ann["suggested"]["support_mode"],
        "ground_offset_m": ann["ground"]["recommended_ground_offset_m"],
    }


def list_clips(filter_substr=None):
    out = []
    for p in sorted(glob.glob(os.path.join(MOTION_DIR, "*"))):
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        if base in SKIP_EXACT or any(s in base for s in SKIP_SUBSTR):
            continue
        if base.endswith((".pkl", ".csv", ".npz", ".txt")):
            continue
        if filter_substr and not any(f.lower() in base.lower() for f in filter_substr):
            continue
        out.append((base, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None, help="comma-separated substrings to filter")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kcm = kin_char_model.KinCharModel(DEVICE)
    kcm.load_char_file(CHAR_FILE)
    body_names = kcm.get_body_names()
    parents = [int(kcm.get_parent_id(i)) for i in range(len(body_names))]
    geoms = char_geoms.load_char_geoms(CHAR_FILE, body_names, DEVICE)

    flt = args.clips.split(",") if args.clips else None
    clips = list_clips(flt)
    print("annotating {} clips -> {}".format(len(clips), args.out_dir))

    index = []
    for k, (name, path) in enumerate(clips):
        try:
            ann = annotate_clip(name, path, kcm, body_names, parents, geoms)
        except Exception as e:
            print("  [{:3d}/{}] SKIP {}: {}".format(k + 1, len(clips), name, e))
            continue
        with open(os.path.join(args.out_dir, name + ".yaml"), "w") as f:
            yaml.safe_dump(ann, f, sort_keys=False, default_flow_style=False)
        index.append(flat_row(ann))
        fp = ann["hold_signature"] or {}
        print("  [{:3d}/{}] {:52s} mode={:16s} up_z={:+.2f} sup={} {}".format(
            k + 1, len(clips), name[:52],
            str(ann["suggested"]["support_mode"]), fp.get("up_z", 0.0) or 0.0,
            fp.get("num_supports", 0), fp.get("contact_bodies", [])))

    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print("\nwrote {} annotations + index.json".format(len(index)))


if __name__ == "__main__":
    main()
