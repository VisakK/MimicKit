import os, sys, math
import numpy as np
import torch
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))

from mimickit.anim.kin_char_model import KinCharModel
from mimickit.anim import motion as motion_mod
import mimickit.util.torch_util as torch_util

DEV = "cpu"
XML = "/home/visakii/Documents/moves/MimicKit/data/assets/smpl/smpl.xml"

def parse_geoms(xml_path, body_names):
    """Return dict body_name -> list of geom dicts with local-frame lowest-z helper data."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    geoms = {bn: [] for bn in body_names}

    def quat_xyzw_from_wxyz(q):
        # mujoco quat is w x y z; return x y z w
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    def recurse(node):
        bname = node.attrib.get("name")
        for g in node.findall("geom"):
            gtype = g.attrib.get("type", "sphere")
            posd = g.attrib.get("pos")
            pos = np.fromstring(posd, sep=" ") if posd else np.zeros(3)
            quatd = g.attrib.get("quat")
            if quatd:
                qw = np.fromstring(quatd, sep=" ")
                gq = quat_xyzw_from_wxyz(qw)
            else:
                gq = np.array([0.0, 0.0, 0.0, 1.0])
            sized = g.attrib.get("size")
            size = np.fromstring(sized, sep=" ") if sized else np.array([0.0])
            fromtod = g.attrib.get("fromto")
            fromto = np.fromstring(fromtod, sep=" ") if fromtod else None
            geoms[bname].append(dict(type=gtype, pos=pos, quat=gq, size=size, fromto=fromto))
        for c in node.findall("body"):
            recurse(c)

    recurse(wb.find("body"))
    return geoms

def box_corners(half):
    sx, sy, sz = half
    c = []
    for ax in (-1,1):
        for ay in (-1,1):
            for az in (-1,1):
                c.append([ax*sx, ay*sy, az*sz])
    return np.array(c)  # (8,3)

def geom_lowest_z(geom, body_pos_w, body_rot_w):
    """body_pos_w (N,3), body_rot_w (N,4 xyzw) torch tensors. Returns (N,) lowest world z of this geom."""
    N = body_pos_w.shape[0]
    bpos = body_pos_w
    brot = body_rot_w
    gtype = geom["type"]
    if gtype == "sphere":
        gp = torch.tensor(geom["pos"], dtype=torch.float32).expand(N,3)
        cw = bpos + torch_util.quat_rotate(brot, gp)
        r = float(geom["size"][0])
        return cw[:,2] - r
    elif gtype == "capsule":
        r = float(geom["size"][0])
        ft = geom["fromto"]
        p1 = torch.tensor(ft[0:3], dtype=torch.float32).expand(N,3)
        p2 = torch.tensor(ft[3:6], dtype=torch.float32).expand(N,3)
        w1 = bpos + torch_util.quat_rotate(brot, p1)
        w2 = bpos + torch_util.quat_rotate(brot, p2)
        return torch.minimum(w1[:,2], w2[:,2]) - r
    elif gtype == "box":
        half = geom["size"][:3]
        corners = box_corners(half)  # (8,3)
        gq = torch.tensor(geom["quat"], dtype=torch.float32).expand(N,4)
        gpos = torch.tensor(geom["pos"], dtype=torch.float32).expand(N,3)
        zmins = None
        for cc in corners:
            cpt = torch.tensor(cc, dtype=torch.float32).expand(N,3)
            # geom-local -> body-local
            blocal = gpos + torch_util.quat_rotate(gq, cpt)
            cw = bpos + torch_util.quat_rotate(brot, blocal)
            z = cw[:,2]
            zmins = z if zmins is None else torch.minimum(zmins, z)
        return zmins
    else:
        gp = torch.tensor(geom["pos"], dtype=torch.float32).expand(N,3)
        cw = bpos + torch_util.quat_rotate(brot, gp)
        return cw[:,2]

def run(motion_file, label):
    kcm = KinCharModel(DEV)
    kcm.load_char_file(XML)
    body_names = kcm.get_body_names()
    geoms = parse_geoms(XML, body_names)

    m = motion_mod.load_motion(motion_file)
    frames = torch.tensor(np.array(m.frames, dtype=np.float32), device=DEV)
    root_pos = frames[:, 0:3]
    root_rot_exp = frames[:, 3:6]
    joint_dof = frames[:, 6:]
    root_rot = torch_util.exp_map_to_quat(root_rot_exp)
    joint_rot = kcm.dof_to_rot(joint_dof)
    joint_rot = torch_util.quat_pos(joint_rot)

    body_pos, body_rot = kcm.forward_kinematics(root_pos, root_rot, joint_rot)
    # body_pos (F, J, 3)
    F, J, _ = body_pos.shape

    # ---- origin-only min z (what readers computed) ----
    origin_minz_perframe = body_pos[:,:,2].min(dim=1).values  # (F,)
    o_min = origin_minz_perframe.min().item()
    o_arg = origin_minz_perframe.argmin().item()

    # ---- geometry-aware min z ----
    geom_minz_perframe = torch.full((F,), 1e9)
    geom_minz_bodyname = [None]*F
    for j, bn in enumerate(body_names):
        bp = body_pos[:, j, :]
        br = body_rot[:, j, :]
        for g in geoms[bn]:
            lz = geom_lowest_z(g, bp, br)  # (F,)
            upd = lz < geom_minz_perframe
            geom_minz_perframe = torch.minimum(geom_minz_perframe, lz)
    g_min = geom_minz_perframe.min().item()
    g_arg = geom_minz_perframe.argmin().item()

    # which body/geom is lowest at the global worst frame
    worst = g_arg
    best_body = None; best_z = 1e9; best_type=None
    for j, bn in enumerate(body_names):
        bp = body_pos[worst:worst+1, j, :]
        br = body_rot[worst:worst+1, j, :]
        for g in geoms[bn]:
            lz = geom_lowest_z(g, bp, br).item()
            if lz < best_z:
                best_z = lz; best_body = bn; best_type = g["type"]

    print(f"\n==== {label} : {os.path.basename(motion_file)} ====")
    print(f"frames={F}  fps={m.fps}")
    print(f"[origin]   min body-ORIGIN z = {o_min:.4f} m at frame {o_arg}")
    print(f"[geom]     min GEOM-extent z = {g_min:.4f} m at frame {g_arg}")
    print(f"           worst body/geom @frame{worst}: {best_body} ({best_type}) z={best_z:.4f}")
    print(f"           geom-vs-origin extra penetration = {(o_min - g_min):.4f} m")
    print(f"           => constant z-offset to lift clip clear of ground = {max(0.0,-g_min):.4f} m")
    # per-frame offset stats
    pen = torch.clamp(-geom_minz_perframe, min=0.0)
    print(f"           per-frame penetration: max={pen.max().item():.4f} mean={pen.mean().item():.4f} "
          f"frac_frames_penetrating={ (pen>1e-4).float().mean().item():.3f}")
    return

if __name__ == "__main__":
    mdir = "/home/visakii/Documents/moves/MimicKit/data/motions/smpl"
    targets = [
        (os.path.join(mdir, "220923_Handstand_pose_or_Adho_Mukha_Vrksasana_-a"), "HANDSTAND(configured)"),
        (os.path.join(mdir, "smpl_run.pkl"), "RUN"),
        (os.path.join(mdir, "smpl_jog.pkl"), "JOG"),
    ]
    for f, lab in targets:
        if os.path.exists(f):
            try:
                run(f, lab)
            except Exception as e:
                print(f"\n==== {lab} FAILED: {e}")
                import traceback; traceback.print_exc()
