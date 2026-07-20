"""Job board — the graph's typed state machine + append-only decision ledger
(Yoga_orchestration_protocol.md §2.1: single source of truth, single writer).

State is written ONLY through this module's transition functions (used by
tools/bank_edge.py and tools/repoint_node.py) — never hand-edited, never
duplicated into prose. Every transition appends one line to
skills/decisions.jsonl {ts, object, from, to, tool, verdict, diagnosis} —
the campaign's audit trail.

Edge job states (the protocol's ladder):

  planned -> demo_mined -> trained -> certified_local -> certified_chained -> banked
                              ^ (router rungs demote certified_*/banked back here)
  terminal: parked(diagnosis: balance_limited|camped|demo_quality|node_invalid|
                   unreachable)   [revivable only to planned = re-plan]
            retired(superseded_by: <route>)

Legacy statuses from the hand-built v3 graph are read-compatible aliases
(needs_train/synth/hard ~ planned; solved/needs_amp_retrain ~ trained); the
first tool transition normalizes them.

CLI (read-only views; writes go through bank_edge/repoint_node):
  python tools/job_board.py show [--drift]   # the board (+ disk-drift audit)
  python tools/job_board.py log [-n 20]      # ledger tail
"""
import argparse
import json
import os
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(REPO, "skills", "graph.yaml")
LEDGER_PATH = os.path.join(REPO, "skills", "decisions.jsonl")

EDGE_STATES = ["planned", "demo_mined", "trained", "certified_local",
               "certified_chained", "banked", "parked", "retired"]
LEGACY = {"needs_train": "planned", "synth": "planned", "hard": "planned",
          "solved": "trained", "needs_amp_retrain": "trained"}
PARK_DIAGNOSES = {"balance_limited", "camped", "demo_quality", "node_invalid",
                  "unreachable"}
CERTIFIED = {"certified_local", "certified_chained", "banked"}

# forward ladder + demotions; parked/retired handled specially (any -> terminal)
ALLOWED = {
    "planned": {"demo_mined", "trained"},
    "demo_mined": {"trained"},
    "trained": {"certified_local"},
    "certified_local": {"certified_chained", "trained"},
    "certified_chained": {"banked", "trained"},
    "banked": {"trained"},
    "parked": {"planned"},
    "retired": set(),
}


def canon(status):
    return LEGACY.get(status, status)


def load_graph(path=GRAPH_PATH):
    with open(path) as fh:
        return yaml.safe_load(fh)


def save_graph(g, path=GRAPH_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        yaml.safe_dump(g, fh, sort_keys=False, default_flow_style=False, width=100)
    os.replace(tmp, path)


def get_edge(g, edge_id):
    return next((e for e in g.get("edges", []) if e["id"] == edge_id), None)


def get_node(g, node_id):
    return next((n for n in g.get("nodes", []) if n["id"] == node_id), None)


def inbound_edges(g, node_id):
    return [e for e in g.get("edges", []) if e.get("to") == node_id]


def ledger_append(obj, frm, to, tool, verdict=None, diagnosis=None,
                  path=LEDGER_PATH):
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "object": obj,
           "from": frm, "to": to, "tool": tool}
    if verdict is not None:
        row["verdict"] = verdict
    if diagnosis is not None:
        row["diagnosis"] = diagnosis
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def transition_edge(edge, to_status, tool, verdict=None, diagnosis=None,
                    force=False, ledger_path=LEDGER_PATH):
    """Mutate edge status with legality enforcement + ledger line. The caller
    owns load/save of the graph (one save per tool invocation)."""
    frm_raw = edge.get("status", "planned")
    frm = canon(frm_raw)
    if to_status not in EDGE_STATES:
        raise SystemExit(f"unknown target state '{to_status}'")
    legal = (to_status in ALLOWED.get(frm, set())
             or (to_status == "parked" and diagnosis in PARK_DIAGNOSES)
             or (to_status == "retired" and diagnosis))
    if not legal and not force:
        hint = (f" (parked requires diagnosis in {sorted(PARK_DIAGNOSES)})"
                if to_status == "parked" else "")
        raise SystemExit(
            f"ILLEGAL transition edge:{edge['id']} {frm_raw}({frm}) -> "
            f"{to_status}{hint}; allowed: {sorted(ALLOWED.get(frm, set()))} "
            "or terminal parked/retired with a diagnosis. Use --force only "
            "with a logged diagnosis.")
    edge["status"] = to_status
    if diagnosis is not None:
        edge["diagnosis"] = diagnosis
    elif to_status not in ("parked", "retired"):
        edge.pop("diagnosis", None)
    return ledger_append(f"edge:{edge['id']}", frm_raw, to_status, tool,
                         verdict, diagnosis, path=ledger_path)


def record_verdict(edge, tool, verdict, diagnosis=None, ledger_path=LEDGER_PATH):
    """Ledger a measured verdict that does NOT move the state (from == to) —
    negative results are first-class state."""
    s = edge.get("status", "planned")
    return ledger_append(f"edge:{edge['id']}", s, s, tool, verdict, diagnosis,
                         path=ledger_path)


def edge_on_disk(edge):
    d, m = edge.get("dir"), edge.get("model")
    return bool(d and m and os.path.exists(os.path.join(REPO, d, m)))


# --------------------------------------------------------------------------- #
# read-only CLI
# --------------------------------------------------------------------------- #
def cmd_show(args):
    g = load_graph(args.graph)
    print(f"== nodes ({len(g.get('nodes', []))}) ==")
    for n in g.get("nodes", []):
        ver = f" v{n['version']}" if n.get("version") else ""
        gate = n.get("gate_calibrated_on", "-")
        needs = f" NEEDS:{','.join(n['needs'])}" if n.get("needs") else ""
        print(f"  {n['id']:<14}{ver:<4} {n.get('status', '-'):<22} "
              f"gate={gate:<14} {n.get('dir', '-')}{needs}")
    print(f"\n== edges ({len(g.get('edges', []))}) ==")
    for e in g.get("edges", []):
        s_raw = e.get("status", "planned")
        s = canon(s_raw)
        tag = s if s == s_raw else f"{s}[{s_raw}]"
        extra = ""
        if e.get("diagnosis"):
            extra += f" diag={e['diagnosis']}"
        if e.get("superseded_by"):
            extra += f" superseded_by={e['superseded_by']}"
        if e.get("last_cert"):
            extra += f" cert={e['last_cert'].get('rung')}" + \
                     ("/chained" if e['last_cert'].get('chained') else "/local")
        print(f"  {e['id']:<28} {tag:<28} {extra}")
        if args.drift:
            actual = edge_on_disk(e)
            if actual != bool(e.get("on_disk")):
                print(f"      DRIFT: on_disk field {e.get('on_disk')} but "
                      f"filesystem says {actual} ({e.get('dir')}/{e.get('model')})")
            if not e.get("dir"):
                for cand in (e["id"], e["id"] + "_deepmimic"):
                    p = os.path.join(REPO, "output", "yoga_edges_v3", cand)
                    if os.path.exists(p):
                        print(f"      DRIFT: dir is null but candidate run dir "
                              f"exists: output/yoga_edges_v3/{cand}")
                        break


def cmd_log(args):
    if not os.path.exists(args.ledger):
        print("(empty ledger)")
        return
    with open(args.ledger) as fh:
        lines = fh.readlines()
    for ln in lines[-args.n:]:
        r = json.loads(ln)
        v = f" verdict={r['verdict']}" if "verdict" in r else ""
        d = f" diag={r['diagnosis']}" if "diagnosis" in r else ""
        print(f"  {r['ts']} {r['object']:<32} {r['from']} -> {r['to']:<20} "
              f"[{r['tool']}]{v}{d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default=GRAPH_PATH)
    p.add_argument("--ledger", default=LEDGER_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("show", help="print the board")
    ps.add_argument("--drift", action="store_true",
                    help="audit board vs filesystem")
    pl = sub.add_parser("log", help="ledger tail")
    pl.add_argument("-n", type=int, default=20)
    args = p.parse_args()
    {"show": cmd_show, "log": cmd_log}[args.cmd](args)


if __name__ == "__main__":
    main()
