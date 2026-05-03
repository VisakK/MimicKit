"""Batch-convert humanoid.xml + variants to USD in a single Isaac Sim session.

Usage (from repo root, with Isaac Lab venv active):

    python tools/convert_humanoid_variants_to_usd.py

Converts data/assets/humanoid/humanoid.xml and humanoid_{tall,short,heavy,light,weak}.xml
to .usd files alongside them. A single AppLauncher is started so the six conversions
share one Isaac Sim startup cost.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ASSET_DIR = os.path.join(REPO_ROOT, "data", "assets", "humanoid")
NAMES = ["humanoid", "humanoid_tall", "humanoid_short",
         "humanoid_heavy", "humanoid_light", "humanoid_weak"]


def convert_one(name: str) -> str:
    mjcf_path = os.path.join(ASSET_DIR, f"{name}.xml")
    usd_name = f"{name}.usd"
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(mjcf_path)
    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=ASSET_DIR,
        usd_file_name=usd_name,
        fix_base=False,
        import_sites=True,
        force_usd_conversion=True,
        make_instanceable=True,
    )
    converter = MjcfConverter(cfg)
    return converter.usd_path


def main():
    print("=" * 80)
    print(f"Converting {len(NAMES)} MJCF assets in {ASSET_DIR}")
    print("=" * 80)
    for name in NAMES:
        out = convert_one(name)
        print(f"[OK] {name}.xml -> {out}")
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
