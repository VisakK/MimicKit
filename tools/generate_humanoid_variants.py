"""Generate humanoid.xml variants for the running-mechanics experiment.

Creates five variants in data/assets/humanoid/ that differ in limb length, mass
density, or actuator strength. Run from the repo root:

    python tools/generate_humanoid_variants.py

Each variant is a strict transform of humanoid.xml so the joint layout (and
therefore the kin_char_model / motion file mapping) is preserved. Only `pos`,
`fromto`, `density`, `actuatorfrcrange`, and motor `gear` attributes are
rewritten. The init_pose root height for each variant is also printed so the
matching env yaml can be set correctly.
"""

import copy
import os
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BASE_PATH = os.path.join(REPO_ROOT, "data", "assets", "humanoid", "humanoid.xml")
OUT_DIR = os.path.join(REPO_ROOT, "data", "assets", "humanoid")

# Bodies whose `pos` is a leg-segment offset; scaling these (with the matching
# `fromto` capsule endpoints) makes a taller / shorter character.
LEG_BODIES = {"right_thigh", "left_thigh", "right_shin", "left_shin",
              "right_foot", "left_foot"}


def _scale_vec(text, scale_xyz):
    nums = [float(x) for x in text.split()]
    out = []
    for i, v in enumerate(nums):
        out.append(v * scale_xyz[i % 3])
    return " ".join("{:.6f}".format(v) for v in out)


def _scale_leg_lengths(root, scale):
    """Scale leg-segment offsets along the body z axis by `scale`.

    The `pos` of each leg child body is the offset from its parent (along -Z for
    the legs in this asset). Scaling this stretches the limb. We also scale the
    matching capsule `fromto` so the visual/collision geometry matches.
    """
    for body in root.iter("body"):
        name = body.get("name")
        if name in LEG_BODIES:
            pos = body.get("pos")
            if pos is not None:
                body.set("pos", _scale_vec(pos, (1.0, 1.0, scale)))
            for geom in body.findall("geom"):
                fromto = geom.get("fromto")
                if fromto is not None:
                    geom.set("fromto", _scale_vec(fromto, (1.0, 1.0, scale)))


def _scale_density(root, scale):
    for geom in root.iter("geom"):
        d = geom.get("density")
        if d is not None:
            geom.set("density", "{:.1f}".format(float(d) * scale))


def _scale_torque(root, scale):
    # actuatorfrcrange limits live on the joints; gears live on the actuators.
    for joint in root.iter("joint"):
        rng = joint.get("actuatorfrcrange")
        if rng is not None:
            lo, hi = [float(x) for x in rng.split()]
            joint.set("actuatorfrcrange", "{:.1f} {:.1f}".format(lo * scale, hi * scale))
    for motor in root.iter("motor"):
        g = motor.get("gear")
        if g is not None:
            motor.set("gear", "{:.1f}".format(float(g) * scale))


def _estimate_root_height(scale_legs):
    # Default pelvis-to-floor height in humanoid.xml is ~0.8824 m, which is
    # roughly thigh + shin + foot vertical extent. Scale the leg portion only.
    base = 0.882416
    leg_extent = 0.421546 + 0.409870 + 0.05  # thigh + shin + foot offset
    pelvis_above_legs = base - leg_extent
    return pelvis_above_legs + leg_extent * scale_legs


VARIANTS = [
    # name, leg_scale, density_scale, torque_scale
    ("humanoid_tall.xml",  1.15, 1.00, 1.00),
    ("humanoid_short.xml", 0.85, 1.00, 1.00),
    ("humanoid_heavy.xml", 1.00, 1.30, 1.00),
    ("humanoid_light.xml", 1.00, 0.75, 1.00),
    ("humanoid_weak.xml",  1.00, 1.00, 0.70),
]


def main():
    if not os.path.exists(BASE_PATH):
        print("Base humanoid not found: {}".format(BASE_PATH), file=sys.stderr)
        sys.exit(1)

    base_tree = ET.parse(BASE_PATH)
    print("name                   leg_scale  density_scale  torque_scale  init_root_z")
    for name, leg_s, dens_s, tq_s in VARIANTS:
        tree = copy.deepcopy(base_tree)
        root = tree.getroot()
        if leg_s != 1.0:
            _scale_leg_lengths(root, leg_s)
        if dens_s != 1.0:
            _scale_density(root, dens_s)
        if tq_s != 1.0:
            _scale_torque(root, tq_s)

        out_path = os.path.join(OUT_DIR, name)
        tree.write(out_path)
        print("{:22s} {:>9.2f}  {:>13.2f}  {:>12.2f}  {:>10.4f}".format(
            name, leg_s, dens_s, tq_s, _estimate_root_height(leg_s)))


if __name__ == "__main__":
    main()
