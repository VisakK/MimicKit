"""repoint_node — version-bump a node and demote its inbound certifications.

The protocol's node-versioning rule (Yoga_orchestration_protocol.md §2.1):
repointing a node (e.g. after a B1 basin finetune) bumps <node>@vN, records
basin provenance (+ harvest blob hash), marks the gate/hold-state machinery
STALE, and **auto-demotes every inbound edge** from certified_*/banked back to
trained with reason stale_target — their certs were bought against the OLD
node and must be cheaply re-bought (oracle re-cert, no GPU training).

This tool moves BOARD STATE only; the follow-up GPU/eval work it mandates is
recorded on the node as `needs: [collect_hold_states, calibrate_gate]` and in
the ledger. It does not run Isaac.

Usage (repo root, plain python):
  python tools/repoint_node.py --node_id tadasana \
      --new_dir output/yoga_nodes_v2/tadasana_coadapt --model model.pt \
      --harvest output/yoga_nodes_v2/tadasana_coadapt/edge_arrivals.pt \
      --note "B1 on warrior2/warrior3 arrivals"
"""
import argparse
import hashlib
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "tools"))
import job_board as jb

TOOL = "repoint_node"


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_id", required=True)
    p.add_argument("--new_dir", required=True, help="repo-relative run dir")
    p.add_argument("--model", default="model.pt")
    p.add_argument("--harvest", default=None,
                   help="the arrivals blob the basin was finetuned on")
    p.add_argument("--note", default=None)
    p.add_argument("--graph", default=jb.GRAPH_PATH)
    p.add_argument("--ledger", default=jb.LEDGER_PATH)
    args = p.parse_args()

    g = jb.load_graph(args.graph)
    n = jb.get_node(g, args.node_id)
    if n is None:
        raise SystemExit(f"node '{args.node_id}' not on the board")
    if not os.path.exists(os.path.join(REPO, args.new_dir, args.model)):
        raise SystemExit(f"no model at {args.new_dir}/{args.model} — "
                         "repointing to a dir without a model is illegal")

    old_ver = int(n.get("version", 1))
    new_ver = old_ver + 1
    prov = {"prev_dir": n.get("dir"), "dir": args.new_dir,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.harvest:
        prov["harvest"] = args.harvest
        prov["harvest_sha"] = sha12(os.path.join(REPO, args.harvest)
                                    if not os.path.isabs(args.harvest)
                                    else args.harvest)
    if args.note:
        prov["note"] = args.note

    n["version"] = new_ver
    n["dir"] = args.new_dir
    n["model"] = args.model
    n["on_disk"] = True
    n.setdefault("basin_finetuned_from", []).append(prov)
    # the old gate/hold_states no longer describe this policy's basin
    n["gate_calibrated_on"] = "stale"
    n["needs"] = ["collect_hold_states", "calibrate_gate"]
    jb.ledger_append(f"node:{args.node_id}", f"v{old_ver}", f"v{new_ver}", TOOL,
                     verdict=f"repointed to {args.new_dir}"
                             + (f" (harvest {prov.get('harvest_sha')})"
                                if args.harvest else ""),
                     diagnosis=args.note, path=args.ledger)
    print(f"  node:{args.node_id}: v{old_ver} -> v{new_ver} @ {args.new_dir} "
          "(gate STALE; needs collect_hold_states + calibrate_gate)")

    demoted = []
    for e in jb.inbound_edges(g, args.node_id):
        if jb.canon(e.get("status", "planned")) in jb.CERTIFIED:
            jb.transition_edge(
                e, "trained", TOOL,
                verdict=f"inbound cert invalidated by "
                        f"{args.node_id}@v{new_ver}",
                diagnosis=f"stale_target:{args.node_id}@v{new_ver}",
                ledger_path=args.ledger)
            demoted.append(e["id"])
    jb.save_graph(g, args.graph)
    if demoted:
        print(f"  demoted inbound edges -> trained (stale_target): "
              f"{', '.join(demoted)}")
    else:
        print("  no certified inbound edges to demote")
    print(f"board -> {args.graph}")


if __name__ == "__main__":
    main()
