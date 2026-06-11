"""
Batch-convert every SMPL/AMASS .npz in an input folder into MimicKit motion
files, mirroring the documented single-file command:

    python tools/smpl_to_mimickit/smpl_to_mimickit.py \
        --input_file <npz> --output_file <name> --z_correction calibrate

Each output gets a unique, readable name derived from its source filename.
A CSV log records, per pose: source file, output name, frame count, FPS,
SMPL pose-param count, DoFs, motion length, and success/error.

Run from the repo root:
    ../env_isaaclab/bin/python tools/smpl_to_mimickit/batch_convert_yoga.py \
        --input_dir yoga_poses --out_dir data/motions/smpl \
        --csv data/motions/smpl/yoga_conversion_log.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import traceback

sys.path.append(".")

from tools.smpl_to_mimickit.smpl_to_mimickit import (  # noqa: E402
    convert_smpl_to_mimickit,
    load_smpl_motion,
)

# Tokens that are constant across the yoga dataset and add no information.
_STRIP_TOKENS = ["yogi_body_hands_03596_", "_stageii"]


def make_name(basename: str) -> str:
    name = basename[:-4] if basename.lower().endswith(".npz") else basename
    for tok in _STRIP_TOKENS:
        name = name.replace(tok, "")
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)  # sanitize spaces/parens/etc.
    name = re.sub(r"_+", "_", name).strip("_")
    return name


CSV_FIELDS = [
    "input_file", "output_name", "num_frames", "fps", "pose_params",
    "dofs", "frame_dim", "length_s", "loop_mode", "z_correction",
    "convert_ok", "error",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="yoga_poses")
    parser.add_argument("--out_dir", default="data/motions/smpl")
    parser.add_argument("--csv", default="data/motions/smpl/yoga_conversion_log.csv")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"])
    parser.add_argument("--z_correction", default="calibrate",
                        choices=["none", "calibrate", "full"])
    parser.add_argument("--limit", type=int, default=-1,
                        help="convert only the first N files (smoke test)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if args.limit > 0:
        files = files[:args.limit]

    rows = []
    used = {}
    n_ok = n_fail = 0
    for i, f in enumerate(files):
        base = os.path.basename(f)
        name = make_name(base)
        if name in used:
            used[name] += 1
            name = f"{name}_dup{used[name]}"
        else:
            used[name] = 0
        out_path = os.path.join(args.out_dir, name)

        row = {k: "" for k in CSV_FIELDS}
        row.update(input_file=base, output_name=name,
                   loop_mode=args.loop, z_correction=args.z_correction)
        try:
            poses, trans, fps = load_smpl_motion(f)
            row["pose_params"] = int(poses.shape[1])
            motion = convert_smpl_to_mimickit(
                f, out_path, loop_mode=args.loop, z_correction=args.z_correction)
            nframes, frame_dim = motion.frames.shape
            row.update(
                num_frames=int(nframes),
                fps=int(fps),
                dofs=int(frame_dim - 6),  # frame = root_pos(3)+root_rot(3)+dof
                frame_dim=int(frame_dim),
                length_s=round(float(nframes - 1) / fps, 4),
                convert_ok=True,
                error="",
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            row["convert_ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"
            n_fail += 1
            sys.stderr.write(f"[FAIL {base}] {row['error']}\n")
            sys.stderr.write(traceback.format_exc())
        rows.append(row)
        print(f"[{i + 1}/{len(files)}] {'OK ' if row['convert_ok'] is True else 'ERR'} "
              f"{name}", flush=True)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nDONE: {n_ok} converted, {n_fail} failed, CSV -> {args.csv}",
          flush=True)


if __name__ == "__main__":
    main()
