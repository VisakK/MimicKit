"""Print the latest training metrics for yoga run dir(s) by parsing the
fixed-width (25-char) log.txt table, plus a health flag (NaN-guard abort scanned
from the tail of stdout.txt without loading the whole — often ~1 GB — file).

Usage:
  env_isaaclab/bin/python tools/yoga_monitor.py output/yoga_transitions/scorpion_to_handstand
  env_isaaclab/bin/python tools/yoga_monitor.py output/yoga_transitions/*  output/yoga_skills/*
"""
import os
import sys

KEYS = ["Iteration", "Samples", "Test_Return", "Critic_Loss", "Disc_Loss",
        "Disc_Agent_Acc", "Disc_Demo_Acc", "Goal_Frac", "Pose_Dist", "Up_Z_Err",
        "Reward_Total"]
WIDTH = 25


def parse_log(path):
    with open(path, "r", errors="ignore") as f:
        txt = f.read()
    lines = [l for l in txt.replace("\r", "\n").split("\n") if l.strip()]
    if (len(lines) < 2):
        return None
    def fields(l):
        return [l[i:i + WIDTH].strip() for i in range(0, len(l), WIDTH)]
    hdr = fields(lines[0])
    row = fields(lines[-1])
    return dict(zip(hdr, row))


def tail_has(path, needle, nbytes=200000):
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            if (sz > nbytes):
                f.seek(sz - nbytes)
            chunk = f.read().decode("utf-8", "ignore")
        return needle in chunk
    except OSError:
        return False


def main():
    runs = sys.argv[1:]
    if (not runs):
        print("usage: yoga_monitor.py <run_dir> [run_dir ...]"); return
    for run in runs:
        run = run.rstrip("/")
        name = os.path.basename(run)
        log = os.path.join(run, "log.txt")
        so = os.path.join(run, "stdout.txt")
        diverged = tail_has(so, "[NaN-guard]") if os.path.exists(so) else False
        d = parse_log(log) if os.path.exists(log) else None
        if (d is None):
            print("{:<32s} no log yet      diverged={}".format(name, diverged))
            continue
        vals = "  ".join("{}={}".format(k, d.get(k, "-")) for k in KEYS if k in d)
        print("{:<32s} {}  diverged={}".format(name, vals, diverged))
    return


if __name__ == "__main__":
    main()
