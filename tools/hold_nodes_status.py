"""One-line-per-node status of the v2 hold-node training queue.

Parses each output/yoga_nodes_v2/<node>/log.txt (\r-delimited tabular log)
and prints: samples progress vs budget, Test_Return, Test_Episode_Length
(survival, /12 s max), the anti-tap Toe_Force where present, and run state
(RUNNING / DONE / DEAD / QUEUED).

Usage: python tools/hold_nodes_status.py   (plain CPU, any python with nothing
       but the stdlib)
"""

import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.path.join(REPO, "tools/hold_nodes_queue.txt")
OUT = os.path.join(REPO, "output/yoga_nodes_v2")


def parse_log(path):
    with open(path, "rb") as f:
        raw = f.read().decode(errors="replace")
    lines = [l for l in raw.replace("\r", "\n").splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].split()
    last = lines[-1].split()
    if len(last) != len(header):  # partial line mid-write
        for cand in reversed(lines[1:]):
            if len(cand.split()) == len(header):
                last = cand.split()
                break
        else:
            return None
    return dict(zip(header, last))


def main():
    rows = []
    with open(QUEUE) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            node, env, agent, samples = line.split()
            budget = int(samples)
            d = os.path.join(OUT, node)
            log = os.path.join(d, "log.txt")
            if not os.path.exists(log):
                rows.append((node, "QUEUED", "", "", "", ""))
                continue
            info = parse_log(log)
            if info is None:
                rows.append((node, "BOOTING", "", "", "", ""))
                continue
            got = int(float(info.get("Samples", 0)))
            # liveness from model.pt (written every iter); the text log only
            # flushes on output iters (~every 100), so its age lies.
            model = os.path.join(d, "model.pt")
            age = time.time() - os.path.getmtime(model if os.path.exists(model) else log)
            state = ("DONE" if got >= budget * 0.99
                     else "RUNNING" if age < 300 else "DEAD?")
            prog = f"{got/1e6:.0f}/{budget/1e6:.0f}M"
            tr = info.get("Test_Return", "?")
            tel = info.get("Test_Episode_Length", "?")
            toe = info.get("Toe_Force", "-")
            rows.append((node, state, prog, tr, tel, toe))
    print(f"{'node':<15s} {'state':<8s} {'samples':<10s} {'Test_Ret':<10s} "
          f"{'Test_EpLen':<11s} {'Toe_Force':<9s}")
    for r in rows:
        print(f"{r[0]:<15s} {r[1]:<8s} {r[2]:<10s} {str(r[3])[:9]:<10s} "
              f"{str(r[4])[:10]:<11s} {str(r[5])[:8]:<9s}")


if __name__ == "__main__":
    main()
