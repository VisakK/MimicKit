"""Inspect and (optionally) fix ArticulationRootAPI on converter-generated humanoid USDs.

Phase 1 (--inspect, default): walks every USD layer for the original backup and
each converter-generated humanoid set, prints which prims carry
ArticulationRootAPI. This is the smoking-gun check.

Phase 2 (--fix): for each layer where worldBody (or any non-pelvis prim under
the articulation chain) carries ArticulationRootAPI, removes that single API
application so only the floating-base pelvis remains the articulation root.

Run from repo root with the Isaac Lab venv:

    python tools/fix_humanoid_usd_articulation_root.py            # inspect only
    python tools/fix_humanoid_usd_articulation_root.py --fix      # inspect + fix
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--fix", action="store_true",
                    help="If set, strip ArticulationRootAPI from non-pelvis prims after inspecting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ASSET_DIR = os.path.join(REPO_ROOT, "data", "assets", "humanoid")

# Top-level USDs to inspect (and the layer files they pull in via the
# instanceable converter). The set was chosen so we cover every place
# ArticulationRootAPI could be authored.
NAMES = ["humanoid", "humanoid_tall", "humanoid_short",
         "humanoid_heavy", "humanoid_light", "humanoid_weak"]


def all_layers_for(name: str) -> list[str]:
    """Top-level + the four configuration sub-layers."""
    out = [os.path.join(ASSET_DIR, f"{name}.usd")]
    for suffix in ("base", "physics", "robot", "sensor"):
        p = os.path.join(ASSET_DIR, "configuration", f"{name}_{suffix}.usd")
        if os.path.isfile(p):
            out.append(p)
    return out


def inspect_layer(layer_path: str) -> list[tuple[str, list[str]]]:
    """Open a single USD layer (not composed) and return (prim_path, applied_schemas)
    for every prim with an Articulation* schema applied locally on this layer."""
    stage = Usd.Stage.Open(layer_path)
    hits: list[tuple[str, list[str]]] = []
    for prim in stage.Traverse():
        # GetAppliedSchemas reflects the composed result; for layer-local check
        # use prim metadata 'apiSchemas'. We want to know what *this* layer adds.
        meta = prim.GetMetadata("apiSchemas")
        applied = list(meta.GetAddedOrExplicitItems()) if meta else []
        if any("Articulation" in s for s in applied):
            hits.append((str(prim.GetPath()), applied))
    return hits


def remove_articulation_root_on(layer_path: str, prim_path: str) -> bool:
    """Edit `layer_path` in place so that `prim_path` no longer has
    ArticulationRootAPI applied locally. Returns True if a change was made."""
    stage = Usd.Stage.Open(layer_path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    api = UsdPhysics.ArticulationRootAPI(prim)
    if not api:
        return False
    # RemoveAPI strips the Apply on the local edit target.
    ok = prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if ok:
        stage.GetRootLayer().Save()
    return bool(ok)


def main():
    backup = "/tmp/humanoid_orig.usd"
    print("=" * 88)
    print(f"BACKUP: {backup}")
    print("=" * 88)
    if os.path.isfile(backup):
        for prim_path, schemas in inspect_layer(backup):
            print(f"  {prim_path}  schemas={schemas}")
    else:
        print("  (missing)")

    print()
    converter_findings: dict[str, list[tuple[str, str, list[str]]]] = {}
    for name in NAMES:
        print("=" * 88)
        print(f"GENERATED: {name}")
        print("=" * 88)
        rows: list[tuple[str, str, list[str]]] = []
        for layer in all_layers_for(name):
            hits = inspect_layer(layer)
            for prim_path, schemas in hits:
                rel = os.path.relpath(layer, ASSET_DIR)
                print(f"  [{rel}]  {prim_path}  schemas={schemas}")
                rows.append((layer, prim_path, schemas))
        converter_findings[name] = rows

    if not args_cli.fix:
        print("\nInspection only. Re-run with --fix to strip the rogue root(s).")
        return

    print("\n" + "=" * 88)
    print("FIX PASS")
    print("=" * 88)
    for name, rows in converter_findings.items():
        for layer, prim_path, schemas in rows:
            # Keep ArticulationRootAPI only on the pelvis articulation.
            # Anything containing 'worldBody' is the synthetic MuJoCo world we want gone.
            if "worldBody" in prim_path:
                changed = remove_articulation_root_on(layer, prim_path)
                rel = os.path.relpath(layer, ASSET_DIR)
                print(f"  {'STRIPPED' if changed else 'NO-OP  '}  [{rel}]  {prim_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
