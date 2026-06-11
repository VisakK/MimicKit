"""
Load-test converted MimicKit motion files through the exact code path the
environment uses: KinCharModel(smpl.xml) + MotionLib(motion_file, ...).

This validates that each motion file "can be loaded into the MimicKit
environment" without booting Isaac Lab (run.py --mode test never terminates,
since test_episodes defaults to int-max). MotionLib construction exercises
dof_to_rot / FK / velocity computation -- i.e. everything the env does at
load time -- so a clean construction is a faithful pass.

Usage:
    ../env_isaaclab/bin/python tools/smpl_to_mimickit/batch_test_motions.py \
        --motion_dir data/motions/smpl --char_file data/assets/smpl/smpl.xml \
        --pattern '<glob>' [--device cpu]
Prints one JSON line per file: {"file","ok","num_motions","length_s","error"}.
"""
import argparse
import csv
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, "mimickit")  # internal modules use `import anim.x` / `import util.x`

import torch  # noqa: E402,F401  (torch is imported transitively; kept for device check)
import anim.kin_char_model as kin_char_model  # noqa: E402
import anim.motion_lib as motion_lib  # noqa: E402

LOAD_FIELDS = ["load_ok", "load_num_motions", "load_length_s", "load_error"]


def load_one(char, path, device):
    rec = {"load_ok": False, "load_num_motions": "", "load_length_s": "",
           "load_error": ""}
    try:
        ml = motion_lib.MotionLib(motion_file=path, kin_char_model=char,
                                  device=device)
        rec["load_ok"] = True
        rec["load_num_motions"] = int(ml.get_num_motions())
        rec["load_length_s"] = round(float(ml.get_total_length()), 4)
        del ml
    except Exception as e:  # noqa: BLE001
        rec["load_error"] = f"{type(e).__name__}: {e}"
        sys.stderr.write(f"[LOAD FAIL {os.path.basename(path)}] {rec['load_error']}\n")
        sys.stderr.write(traceback.format_exc())
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", default="data/motions/smpl")
    parser.add_argument("--char_file", default="data/assets/smpl/smpl.xml")
    parser.add_argument("--pattern", default="*", help="glob within motion_dir")
    parser.add_argument("--files", nargs="*", default=None,
                        help="explicit list of files (overrides pattern)")
    parser.add_argument("--augment_csv", default=None,
                        help="CSV with an 'output_name' column to load-test and "
                             "rewrite in place with load_* columns appended")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    char = kin_char_model.KinCharModel(device)
    char.load_char_file(args.char_file)
    dof_size = char.get_dof_size()
    print(json.dumps({"event": "char_loaded", "dof_size": int(dof_size),
                      "device": device}), flush=True)

    if args.augment_csv:
        with open(args.augment_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            in_fields = reader.fieldnames or []
        out_fields = in_fields + [f for f in LOAD_FIELDS if f not in in_fields]
        n_ok = n_fail = n_skip = 0
        for i, row in enumerate(rows):
            name = row.get("output_name", "").strip()
            converted = str(row.get("convert_ok", "")).strip().lower() == "true"
            if not name or not converted:
                row.update({k: "" for k in LOAD_FIELDS})
                row["load_ok"] = "" if not name else False
                n_skip += 1
            else:
                rec = load_one(char, os.path.join(args.motion_dir, name), device)
                row.update(rec)
                n_ok += int(rec["load_ok"] is True)
                n_fail += int(rec["load_ok"] is False)
            print(f"[{i + 1}/{len(rows)}] load {row.get('load_ok')} {name}",
                  flush=True)
        with open(args.augment_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nLOAD-TEST DONE: {n_ok} ok, {n_fail} failed, {n_skip} skipped. "
              f"CSV -> {args.augment_csv}", flush=True)
        return

    if args.files:
        files = list(args.files)
    else:
        files = sorted(glob.glob(os.path.join(args.motion_dir, args.pattern)))
    for path in files:
        rec = {"file": os.path.basename(path)}
        rec.update(load_one(char, path, device))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
