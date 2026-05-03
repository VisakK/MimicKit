"""Strip PhysicsArticulationRootAPI / PhysxArticulationAPI from the synthetic
`worldBody` prim authored by Isaac Lab's MJCF→USD converter for each humanoid
variant. The legitimate articulation root on `pelvis/pelvis` is left intact.

Run with the pxr Python bindings on PYTHONPATH/LD_LIBRARY_PATH (see Bash invocation).
"""

import os
import sys

from pxr import Usd, UsdPhysics

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "humanoid")
ASSET_DIR = os.path.abspath(ASSET_DIR)
NAMES = ["humanoid", "humanoid_tall", "humanoid_short",
         "humanoid_heavy", "humanoid_light", "humanoid_weak"]


def layers_for(name: str):
    yield os.path.join(ASSET_DIR, f"{name}.usd")
    yield os.path.join(ASSET_DIR, "configuration", f"{name}_physics.usd")


def strip(layer_path: str) -> int:
    if not os.path.isfile(layer_path):
        return 0
    stage = Usd.Stage.Open(layer_path)
    changed = 0
    for prim in stage.Traverse():
        if prim.GetName() != "worldBody":
            continue
        for api in (UsdPhysics.ArticulationRootAPI,
                    # PhysxArticulationAPI lives in PhysxSchema, drop by name:
                    None):
            if api is None:
                # Drop PhysxArticulationAPI by editing apiSchemas metadata directly.
                meta = prim.GetMetadata("apiSchemas")
                if not meta:
                    continue
                items = list(meta.GetAddedOrExplicitItems())
                kept = [s for s in items if s != "PhysxArticulationAPI"]
                if len(kept) != len(items):
                    from pxr import Sdf
                    new_meta = Sdf.TokenListOp.CreateExplicit(kept)
                    prim.SetMetadata("apiSchemas", new_meta)
                    changed += 1
            else:
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
                    changed += 1
    if changed:
        stage.GetRootLayer().Save()
    return changed


def main():
    total = 0
    for name in NAMES:
        for layer in layers_for(name):
            n = strip(layer)
            rel = os.path.relpath(layer, ASSET_DIR)
            print(f"  [{rel}]  removed {n} API(s)")
            total += n
    print(f"\nTotal API removals: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
