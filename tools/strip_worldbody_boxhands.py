"""Strip the spurious PhysicsArticulationRootAPI / PhysxArticulationAPI that
Isaac Lab's MJCF->USD converter authors on the synthetic `worldBody` prim, which
makes the asset have TWO articulation roots (worldBody + Pelvis/Pelvis) and fail
to load ("Failed to find a single articulation"). Generalised, in-app version of
strip_worldbody_articulation.py (pxr is only importable inside the Kit app).

Usage:
  python tools/strip_worldbody_boxhands.py <layer.usd> [<layer.usd> ...] --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("layers", nargs="+", help="USD layer files to strip")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Sdf


def strip(layer_path):
    if (not os.path.isfile(layer_path)):
        print("  [skip missing] {}".format(layer_path)); return 0
    stage = Usd.Stage.Open(layer_path)
    changed = 0
    for prim in stage.Traverse():
        if (prim.GetName() != "worldBody"):
            continue
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI); changed += 1
        meta = prim.GetMetadata("apiSchemas")
        if (meta):
            items = list(meta.GetAddedOrExplicitItems())
            kept = [s for s in items if s != "PhysxArticulationAPI"]
            if (len(kept) != len(items)):
                prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(kept))
                changed += 1
    if (changed):
        stage.GetRootLayer().Save()
    print("  [{}] worldBody APIs removed: {}".format(os.path.basename(layer_path), changed))
    return changed


def main():
    total = sum(strip(p) for p in args_cli.layers)
    print("Total removals: {}".format(total))


if __name__ == "__main__":
    main()
    simulation_app.close()
