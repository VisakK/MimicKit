"""bank_edge — artifact-driven edge state transitions on the job board.

The ONLY way an edge's status moves (with tools/repoint_node.py's demotions).
Every claim is backed by an on-disk artifact; every move is ledgered
(skills/decisions.jsonl). No artifact, no transition.

What the artifacts buy (applied stepwise, each step ledgered):
  --dir/--model, model on disk        planned/demo_mined -> trained
  --cert bridge_cert.json, R10, local trained -> certified_local
  --cert ..., R10, CHAINED            ... -> certified_chained (the chained
                                      exam subsumes the local one: it is the
                                      same cert run FROM the delivered
                                      distribution — strictly harder)
  --cert ..., non-R10 on a certified/banked edge -> DEMOTED to trained with
                                      the routed rung as the diagnosis
  --cert ..., non-R10 otherwise      -> verdict LEDGERED, state unchanged
                                      (negative results are first-class)
  --bank --note "<visual sign-off>"   certified_chained -> banked
  --park --diagnosis <cls>            any -> parked (terminal)
  --retire --superseded_by <route>    any -> retired (terminal)

Usage examples (repo root, plain python, no Isaac):
  python tools/bank_edge.py --edge_id downdog_to_tadasana \
      --dir output/yoga_edges_v3/downdog_to_tadasana_deepmimic --model model.pt \
      --cert output/yoga_edges_v3/downdog_to_tadasana_deepmimic/bridge_cert.json
  python tools/bank_edge.py --edge_id downdog_to_plank --register \
      --from_node downdog --to_node plank --routing_class deepmimic \
      --dir output/yoga_edges_v3/downdog_to_plank_deepmimic --model model.pt \
      --cert output/yoga_flows/plank_vinyasa/chain_cert/stage1_downdog_to_plank/bridge_cert.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "tools"))
import job_board as jb

TOOL = "bank_edge"
PASS_RUNG = "R10_certified_local"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edge_id", required=True)
    p.add_argument("--graph", default=jb.GRAPH_PATH)
    p.add_argument("--ledger", default=jb.LEDGER_PATH)
    p.add_argument("--register", action="store_true",
                   help="add the edge to the board if missing")
    p.add_argument("--from_node", default=None)
    p.add_argument("--to_node", default=None)
    p.add_argument("--routing_class", default="deepmimic")
    p.add_argument("--dir", default=None, help="run dir (repo-relative)")
    p.add_argument("--model", default=None)
    p.add_argument("--cert", default=None, help="a bridge_cert.json")
    p.add_argument("--bank", action="store_true")
    p.add_argument("--park", action="store_true")
    p.add_argument("--retire", action="store_true")
    p.add_argument("--diagnosis", default=None)
    p.add_argument("--superseded_by", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--force", action="store_true",
                   help="override transition legality (diagnosis required)")
    args = p.parse_args()
    if args.force and not (args.diagnosis or args.note):
        raise SystemExit("--force requires --diagnosis or --note (logged)")

    g = jb.load_graph(args.graph)
    e = jb.get_edge(g, args.edge_id)
    moves = []

    if e is None:
        if not args.register:
            raise SystemExit(f"edge '{args.edge_id}' not on the board "
                             "(pass --register with --from_node/--to_node)")
        if not (args.from_node and args.to_node):
            raise SystemExit("--register needs --from_node and --to_node")
        e = {"id": args.edge_id, "from": args.from_node, "to": args.to_node,
             "routing_class": args.routing_class, "dir": None, "model": None,
             "on_disk": False, "status": "planned"}
        if args.note:
            e["note"] = args.note
        g["edges"].append(e)
        moves.append(jb.ledger_append(f"edge:{args.edge_id}", None, "planned",
                                      TOOL, verdict="registered",
                                      path=args.ledger))

    # ---- run artifacts: dir/model -> trained
    if args.dir:
        e["dir"] = args.dir
        e["model"] = args.model or e.get("model") or "model.pt"
    e["on_disk"] = jb.edge_on_disk(e)
    if e["on_disk"] and jb.canon(e.get("status", "planned")) in ("planned",
                                                                "demo_mined"):
        moves.append(jb.transition_edge(
            e, "trained", TOOL,
            verdict=f"model on disk: {e['dir']}/{e['model']}",
            ledger_path=args.ledger))

    # ---- certificate: the routed rung IS the verdict
    if args.cert:
        cert = json.load(open(args.cert))
        rung = cert["verdict"]["rung"]
        chained = bool(cert["channels"]["harness"].get("chained_from"))
        kind = "chained" if chained else "local"
        rel_cert = os.path.relpath(os.path.abspath(args.cert), REPO)
        e["last_cert"] = {"path": rel_cert, "rung": rung, "chained": chained}
        rec = cert["channels"]["recovery"]["recovery_rate"]
        if rec >= 0:
            e["recovery_oracle"] = rec
        verdict = f"{rung} ({kind} cert: {rel_cert})"
        st = jb.canon(e.get("status", "planned"))
        if rung == PASS_RUNG:
            if st == "trained":
                moves.append(jb.transition_edge(e, "certified_local", TOOL,
                                                verdict, ledger_path=args.ledger))
                st = "certified_local"
            if chained and st == "certified_local":
                moves.append(jb.transition_edge(e, "certified_chained", TOOL,
                                                verdict, ledger_path=args.ledger))
        else:
            if st in jb.CERTIFIED:
                moves.append(jb.transition_edge(
                    e, "trained", TOOL, verdict,
                    diagnosis=f"demoted:{rung}", ledger_path=args.ledger))
            else:
                moves.append(jb.record_verdict(e, TOOL, verdict,
                                               ledger_path=args.ledger))

    # ---- terminals / bank
    if args.bank:
        if not args.note and not args.force:
            raise SystemExit("--bank requires --note (the visual sign-off "
                             "reference — certs never bank alone)")
        moves.append(jb.transition_edge(e, "banked", TOOL,
                                        verdict=f"visual sign-off: {args.note}",
                                        force=args.force,
                                        ledger_path=args.ledger))
    if args.park:
        moves.append(jb.transition_edge(e, "parked", TOOL,
                                        diagnosis=args.diagnosis,
                                        force=args.force,
                                        ledger_path=args.ledger))
    if args.retire:
        if not args.superseded_by:
            raise SystemExit("--retire requires --superseded_by <route>")
        e["superseded_by"] = args.superseded_by
        moves.append(jb.transition_edge(e, "retired", TOOL,
                                        diagnosis=args.superseded_by,
                                        force=args.force,
                                        ledger_path=args.ledger))

    jb.save_graph(g, args.graph)
    if moves:
        for m in moves:
            print(f"  {m['object']}: {m['from']} -> {m['to']}"
                  + (f"  [{m.get('verdict', '')}]" if m.get("verdict") else ""))
    else:
        print(f"  edge:{args.edge_id}: no state change "
              f"(status={e.get('status')})")
    print(f"board -> {args.graph}")


if __name__ == "__main__":
    main()
