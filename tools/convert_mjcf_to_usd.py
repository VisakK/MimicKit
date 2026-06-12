"""Convert a single MJCF character file to USD with Isaac Lab's MjcfConverter.

Generic sibling of convert_humanoid_variants_to_usd.py (which is hardcoded to
the humanoid variant set). The USD lands next to the XML (or at --out), which
is where the isaac_lab engine looks for it (it swaps the char_file extension).

Usage (from repo root, with the Isaac Lab venv active):

    python tools/convert_mjcf_to_usd.py data/assets/smpl/smpl_boxhands.xml --headless

After converting, check the ArticulationRootAPI placement (the converter has
authored it on the synthetic worldBody prim in the past; only the floating
base body, e.g. Pelvis, should carry it - see strip_worldbody_articulation.py).
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("xml", help="Path to the MJCF .xml character file")
parser.add_argument("--out", default=None,
                    help="Output .usd path (default: alongside the XML, same stem)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def main():
    mjcf_path = os.path.abspath(args_cli.xml)
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(mjcf_path)
    if args_cli.out is not None:
        out_path = os.path.abspath(args_cli.out)
    else:
        out_path = os.path.splitext(mjcf_path)[0] + ".usd"

    cfg = MjcfConverterCfg(
        asset_path=mjcf_path,
        usd_dir=os.path.dirname(out_path),
        usd_file_name=os.path.basename(out_path),
        fix_base=False,
        import_sites=True,
        force_usd_conversion=True,
        make_instanceable=True,
    )
    converter = MjcfConverter(cfg)
    print(f"[OK] {mjcf_path} -> {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
