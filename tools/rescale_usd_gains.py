"""Fix the MJCF->USD angular-drive gain convention (the previously UN-SCRIPTED
step 3 of the asset pipeline — its absence is how the 2026-06-15 smpl_boxhands
regen shipped with per-radian gains in USD's per-degree drive slots, i.e. a
~57.3x-too-stiff character; verified 2026-07-03 with usd-core: hip stiffness
500 in the live asset vs 8.7266 = 500*pi/180 in the shipped smpl.usd).

Multiplies every drive:rot{X,Y,Z}:physics:{stiffness,damping} by pi/180 in the
given USD layers. maxForce is a plain torque and is NOT touched. Idempotence
guard: refuses to rescale a layer whose max stiffness is already < THRESH
(i.e. looks already converted) unless --force.

Usage (any python with usd-core, NOT the Kit app):
  python3 tools/rescale_usd_gains.py data/assets/smpl/<asset>.usd \
      data/assets/smpl/configuration/<asset>_physics.usd
"""
import argparse
import math

from pxr import Usd

THRESH = 50.0  # per-degree stiffness above this ~= raw per-radian values


def rescale(path, force):
    stage = Usd.Stage.Open(path)
    attrs = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if n.startswith("drive:rot") and (n.endswith(":physics:stiffness")
                                              or n.endswith(":physics:damping")):
                v = attr.Get()
                if v is not None:
                    attrs.append((attr, float(v)))
    if not attrs:
        print(f"{path}: no drive gain attrs authored in this layer, skipping")
        return
    mx = max(v for _, v in attrs if v > 0)
    if mx < THRESH and not force:
        print(f"{path}: max stiffness/damping {mx:.3f} < {THRESH} -> looks "
              f"already per-degree; NOT rescaling (use --force to override)")
        return
    k = math.pi / 180.0
    for attr, v in attrs:
        attr.Set(v * k)
    stage.GetRootLayer().Save()
    print(f"{path}: rescaled {len(attrs)} drive gain attrs by pi/180 "
          f"(max was {mx:.1f}, now {mx * k:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("layers", nargs="+")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for path in args.layers:
        rescale(path, args.force)


if __name__ == "__main__":
    main()
