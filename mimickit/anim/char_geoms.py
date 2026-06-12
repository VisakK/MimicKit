import numpy as np
import torch
import xml.etree.ElementTree as ET

from util.logger import Logger
import util.torch_util as torch_util

# Collision-geometry parsing for MJCF character files. Body origins alone
# underestimate the character's true ground extent (e.g. the SMPL heel box
# bottom sits ~5 cm below the ankle origin), so anything that needs to reason
# about ground clearance has to look at the geoms. Each geom is reduced to a
# set of body-local witness points plus a radius:
#   sphere  -> 1 center point,  r = radius
#   capsule -> 2 endpoints,     r = radius
#   box     -> 8 corners,       r = 0
# The lowest world z of a geom is then min_z(points rotated into world) - r,
# which is exact for spheres/capsules/boxes on a flat ground plane.


def _mjcf_quat_to_xyzw(quat_wxyz):
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                    dtype=np.float32)

def _rotate_np(quat_xyzw, points):
    q = torch.tensor(quat_xyzw, dtype=torch.float32).expand(points.shape[0], 4)
    p = torch.tensor(points, dtype=torch.float32)
    return torch_util.quat_rotate(q, p).numpy()

def _box_corners(half_extents):
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append([sx * half_extents[0],
                                sy * half_extents[1],
                                sz * half_extents[2]])
    return np.array(corners, dtype=np.float32)

def _parse_geom(geom_xml):
    geom_type = geom_xml.attrib.get("type", "sphere")

    pos = geom_xml.attrib.get("pos")
    pos = np.fromstring(pos, dtype=np.float32, sep=" ") if pos else np.zeros(3, dtype=np.float32)

    quat = geom_xml.attrib.get("quat")
    if (quat):
        quat = _mjcf_quat_to_xyzw(np.fromstring(quat, dtype=np.float32, sep=" "))
    else:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    size = geom_xml.attrib.get("size")
    size = np.fromstring(size, dtype=np.float32, sep=" ") if size else np.zeros(1, dtype=np.float32)

    fromto = geom_xml.attrib.get("fromto")
    fromto = np.fromstring(fromto, dtype=np.float32, sep=" ") if fromto else None

    if (geom_type == "sphere"):
        points = pos[np.newaxis, :]
        radius = float(size[0])
    elif (geom_type == "capsule"):
        radius = float(size[0])
        if (fromto is not None):
            points = np.stack([fromto[0:3], fromto[3:6]], axis=0)
        else:
            half_len = float(size[1])
            axis = _rotate_np(quat, np.array([[0.0, 0.0, half_len]], dtype=np.float32))[0]
            points = np.stack([pos + axis, pos - axis], axis=0)
    elif (geom_type == "box"):
        radius = 0.0
        corners = _box_corners(size[:3])
        points = pos[np.newaxis, :] + _rotate_np(quat, corners)
    else:
        # Fallback: treat unsupported geom types as a point at the geom
        # origin. This underestimates the geom's extent, so warn.
        Logger.print("char_geoms: unsupported geom type '{:s}', treating as a point".format(geom_type))
        points = pos[np.newaxis, :]
        radius = 0.0

    return {"points": points.astype(np.float32), "radius": radius}

def load_char_geoms(char_file, body_names, device):
    """Parses the collision geoms of every body in a MJCF character file.

    Returns a list aligned with body_names; entry b is a list of geom dicts
    with body-local witness "points" ([P, 3] tensor) and a scalar "radius".
    """
    tree = ET.parse(char_file)
    worldbody = tree.getroot().find("worldbody")
    geoms_by_name = {name: [] for name in body_names}

    def _recurse(body_xml):
        body_name = body_xml.attrib.get("name")
        if (body_name in geoms_by_name):
            for geom_xml in body_xml.findall("geom"):
                geoms_by_name[body_name].append(_parse_geom(geom_xml))
        for child_xml in body_xml.findall("body"):
            _recurse(child_xml)

    for root_body in worldbody.findall("body"):
        _recurse(root_body)

    char_geoms = []
    for name in body_names:
        body_geoms = []
        for geom in geoms_by_name[name]:
            body_geoms.append({
                "points": torch.tensor(geom["points"], dtype=torch.float32, device=device),
                "radius": geom["radius"],
            })
        char_geoms.append(body_geoms)
    return char_geoms

def compute_min_geom_z(char_geoms, body_pos, body_rot):
    """Lowest world z over all collision geoms, per frame.

    body_pos: [N, B, 3], body_rot: [N, B, 4] (xyzw). Returns [N].
    """
    n = body_pos.shape[0]
    min_z = torch.full([n], np.inf, dtype=body_pos.dtype, device=body_pos.device)

    for b, body_geoms in enumerate(char_geoms):
        if (len(body_geoms) == 0):
            continue
        b_pos = body_pos[:, b, :]
        b_rot = body_rot[:, b, :]
        for geom in body_geoms:
            points = geom["points"]
            num_points = points.shape[0]
            rot_flat = b_rot.unsqueeze(1).expand(n, num_points, 4).reshape(-1, 4)
            points_flat = points.unsqueeze(0).expand(n, num_points, 3).reshape(-1, 3)
            world_points = torch_util.quat_rotate(rot_flat, points_flat).reshape(n, num_points, 3)
            world_points = world_points + b_pos.unsqueeze(1)
            geom_min_z = torch.min(world_points[..., 2], dim=1)[0] - geom["radius"]
            min_z = torch.minimum(min_z, geom_min_z)

    return min_z
