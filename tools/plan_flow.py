"""plan_flow — turn a flow request into a per-seam PLAN (no Isaac, no GPU).

The planning layer of the orchestration protocol (Yoga_orchestration_protocol.md
§2.2): for every consecutive waypoint pair A->B it emits

  - the BOARD action (from skills/graph.yaml job states: reuse a certified
    edge / certify a trained one / train / decompose around parked/retired),
  - the ROUTING KNOB — a lookup on the destination node's balance_class
    (Recipe v5: stable -> pure-DeepMimic task-only; precarious -> destination-
    hold disc co-train; control_limited -> DeepMimic + R9 park risk),
  - node-judge PREREQUISITES (gate.yaml / hold_states.pt / repoint `needs`),
  - DEMO CANDIDATES mined from data/clip_annotations/ via the hold roster
    (the Annotator doctrine: the destination node's OWN entry transition
    first, then the source node's own exit, then a corpus inter-hold scan,
    then the synth fallback command),
  - a REGIME-BOUNDARY feasibility flag (disjoint support + large up_z gap =
    the tadasana<->plank / tadasana->crow class) + DECOMPOSE candidates
    (BFS over non-terminal board edges through hubs/gateways).

The plan lists commands; it never runs them (GPU spend stays with the
orchestrator loop + runbooks). LLM judgment is confined to choosing among
the mined demo candidates and queue order.

Usage (repo root, plain python):
  python tools/plan_flow.py --flow_id downdog_to_handstand
  python tools/plan_flow.py --waypoints tadasana,malasana,crow,tadasana --name squat_to_crow
Output: output/yoga_flows/<name>/flow_plan.yaml + stdout summary.
"""
import argparse
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "tools"))
os.chdir(REPO)
import job_board as jb
import make_hold_nodes as MH

PY = "/home/visakii/Documents/moves/env_isaaclab/bin/python"
ANN_DIR = "data/clip_annotations"
SRC_DIR = "data/motions/smpl"
ROSTER = {n["id"]: n for n in MH.ROSTER}

# Recipe v5 disc-tail knob: a LOOKUP on the destination's balance_class
ROUTING = {
    "stable": "deepmimic_task_only (start with pure DeepMimic — the "
              "stance-step / entry-edge lesson)",
    "control_limited": "deepmimic_task_only, but expect R9 balance-control "
                       "risk on the hold (park candidate; big_toe class)",
    "precarious": "deepmimic + destination-hold disc co-train (Recipe v5 "
                  "disc-tail — the downdog3L->handstand fingertip lesson: "
                  "kinematics-task solves the transit, AMP holds the "
                  "precarious pose)",
}


def load_ann(clip):
    p = os.path.join(ANN_DIR, clip + ".yaml")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return yaml.safe_load(fh)


def node_sig(g, node_id):
    """({clip, src}, annotation, hold_signature) for a node's source clip.
    ROSTER first (the hold-extraction provenance); fall back to the board
    node's own env_config motion_file (crow-class nodes trained pre-roster)."""
    r = ROSTER.get(node_id)
    if r is not None:
        clip, src = r["clip"], os.path.join(SRC_DIR, r["clip"])
    else:
        n = jb.get_node(g, node_id)
        clip = src = None
        if n and n.get("dir"):
            envp = os.path.join(n["dir"], "env_config.yaml")
            if os.path.exists(envp):
                mf = (yaml.safe_load(open(envp)).get("env") or {}).get("motion_file")
                if mf:
                    src, clip = mf, os.path.basename(mf)
        if clip is None:
            return None, None, None
    ann = load_ann(clip)
    return {"clip": clip, "src": src}, ann, (ann or {}).get("hold_signature")


def sig_match(sig, hold, jaccard_min=0.5, up_z_tol=0.30):
    """Match a node hold_signature against an annotated hold/endpoint."""
    if sig is None or hold is None:
        return 0.0, 1.0, False
    ca = set(sig.get("contact_bodies", []))
    cb = set(hold.get("contact_bodies", []))
    j = len(ca & cb) / max(len(ca | cb), 1)
    dz = abs(float(sig.get("up_z", 0.0)) - float(hold.get("up_z", 0.0)))
    return round(j, 2), round(dz, 2), (j >= jaccard_min and dz <= up_z_tol)


def mine_demos(g, a_id, b_id):
    """The Annotator doctrine, in order. Returns (candidates, synth_cmd)."""
    ra, ann_a, sig_a = node_sig(g, a_id)
    rb, ann_b, sig_b = node_sig(g, b_id)
    cands = []

    # 1) destination's OWN source clip: does its entry come from an A-like hold?
    if ann_b is not None:
        seg = ann_b.get("segments", {})
        entry = (ann_b.get("endpoints") or {}).get("entry_from")
        et = seg.get("entry_transition")
        j, dz, ok = sig_match(sig_a, entry)
        if ok and et:
            holds = seg.get("holds", [])
            ph = holds[seg.get("primary_hold_index", 0)]
            cands.append({
                "source": "destination_own_entry", "clip": rb["clip"],
                "window_s": [et["start_s"], ph["end_s"]],
                "match": {"jaccard": j, "up_z_gap": dz},
                "note": f"{b_id}'s own corpus entry (highest-trust demo)"})

    # 2) source's OWN clip: does its exit land in a B-like hold?
    if ann_a is not None:
        seg = ann_a.get("segments", {})
        exit_to = (ann_a.get("endpoints") or {}).get("exit_to")
        xt = seg.get("exit_transition")
        j, dz, ok = sig_match(sig_b, exit_to)
        if ok and xt:
            holds = seg.get("holds", [])
            ph = holds[seg.get("primary_hold_index", 0)]
            cands.append({
                "source": "source_own_exit", "clip": ra["clip"],
                "window_s": [ph["start_s"], xt["end_s"]],
                "match": {"jaccard": j, "up_z_gap": dz},
                "note": f"{a_id}'s own corpus exit"})

    # 3) corpus scan: any clip holding A then B across one inter-hold transition
    if sig_a is not None and sig_b is not None:
        for f in sorted(os.listdir(ANN_DIR)):
            if not f.endswith(".yaml"):
                continue
            ann = load_ann(f[:-5])
            if ann is None:
                continue
            seg = ann.get("segments", {})
            holds = seg.get("holds", [])
            for ih in seg.get("inter_hold", []):
                hi, hj = ih.get("from_hold"), ih.get("to_hold")
                if hi is None or hj is None or hi >= len(holds) or hj >= len(holds):
                    continue
                ja, dza, oka = sig_match(sig_a, holds[hi])
                jb_, dzb, okb = sig_match(sig_b, holds[hj])
                if oka and okb:
                    cands.append({
                        "source": "corpus_scan", "clip": ann["clip"],
                        "window_s": [holds[hi]["start_s"], holds[hj]["end_s"]],
                        "match": {"jaccard_a": ja, "jaccard_b": jb_,
                                  "up_z_gap_a": dza, "up_z_gap_b": dzb},
                        "note": "in-corpus A->B transition"})

    # 4) synth fallback (always emitted; pay-for-it expectation logged)
    synth = None
    if ra is not None and rb is not None and ann_a and ann_b:
        ha = ann_a["segments"]["holds"][ann_a["segments"].get("primary_hold_index", 0)]
        hb = ann_b["segments"]["holds"][ann_b["segments"].get("primary_hold_index", 0)]
        a_end = ha["end_frame"]
        b_start = hb["start_frame"]
        synth = (f"env -u DISPLAY {PY} tools/make_synth_edge_clip.py "
                 f"--src_a_file {ra['src']} "
                 f"--src_a_win {max(ha['start_frame'], a_end - 60)} {a_end} "
                 f"--src_b_file {rb['src']} "
                 f"--src_b_win {b_start} {min(hb['end_frame'], b_start + 90)} "
                 f"--bridge_s 1.2 --out data/motions/smpl_edges/{a_id}_to_{b_id}")
    return cands, synth


def node_prereqs(g, node_id):
    """Judge-machinery prerequisites for certifying arrivals INTO this node."""
    n = jb.get_node(g, node_id)
    if n is None:
        return {"on_board": False,
                "todo": [f"node '{node_id}' is not on the board — register it"]}
    d = n.get("dir")
    todo = []
    if not d or not os.path.exists(os.path.join(d, "model.pt")):
        todo.append("no hold policy on disk — train the node first")
    if not d or not os.path.exists(os.path.join(d, "hold_states.pt")):
        todo.append("collect hold_states (tools/collect_hold_states)")
    if not d or not os.path.exists(os.path.join(d, "gate.yaml")):
        todo.append("calibrate arrival gate (tools/calibrate_gate.py) — no "
                    "judge, no certification")
    if n.get("gate_calibrated_on") == "stale":
        todo.append(f"gate STALE (repointed @v{n.get('version')}) — "
                    "re-collect + recalibrate")
    for x in n.get("needs", []) or []:
        t = f"node.needs: {x}"
        if t not in todo:
            todo.append(t)
    return {"on_board": True, "balance_class": n.get("balance_class", "stable"),
            "dir": d, "todo": todo}


def board_action(edge):
    if edge is None:
        return "TRAIN_NEW", "no edge on the board — full pipeline: demo -> " \
                            "train (train_edge_monitored.sh) -> register " \
                            "(bank_edge) -> certify"
    st = jb.canon(edge.get("status", "planned"))
    return {
        "planned": ("TRAIN", "demo -> train -> certify"),
        "demo_mined": ("TRAIN", "train on the mined demo -> certify"),
        "trained": ("CERTIFY", "model on disk — run certify_bridge (t=0) and "
                               "route the verdict"),
        "certified_local": ("CHAIN_CERT", "locally certified — needs the "
                                          "chained seam pass (certify_chain)"),
        "certified_chained": ("REUSE", "chained-certified — reuse; visual "
                                       "sign-off pending unless banked"),
        "banked": ("REUSE", "banked"),
        "parked": ("DECOMPOSE", f"parked ({edge.get('diagnosis')}) — do NOT "
                                "retry; route around"),
        "retired": ("REROUTE", f"retired — use the superseding route: "
                               f"{edge.get('superseded_by')}"),
    }[st]


def decompose_candidates(g, a_id, b_id, max_hops=2):
    """Routes A->B over non-terminal board edges (the graph's own answer to an
    infeasible direct edge). Returns up to 3 paths with per-hop statuses."""
    adj = {}
    for e in g.get("edges", []):
        if jb.canon(e.get("status", "planned")) in ("parked", "retired"):
            continue
        adj.setdefault(e["from"], []).append(e)
    paths, frontier = [], [(a_id, [])]
    for _ in range(max_hops):
        nxt = []
        for cur, path in frontier:
            for e in adj.get(cur, []):
                if any(h["id"] == e["id"] for h in path):
                    continue
                np_ = path + [e]
                if e["to"] == b_id:
                    paths.append(np_)
                else:
                    nxt.append((e["to"], np_))
        frontier = nxt
    # a 1-hop path is the direct edge itself, not a decomposition
    paths = [p for p in paths if len(p) > 1]
    out = []
    for p in sorted(paths, key=len)[:3]:
        out.append(" -> ".join([a_id] + [e["to"] for e in p]) + "  [" +
                   ", ".join(f"{e['id']}:{jb.canon(e.get('status', 'planned'))}"
                             for e in p) + "]")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow_id", help="a flow from skills/graph.yaml")
    p.add_argument("--waypoints", help="comma-separated node ids (ad-hoc flow)")
    p.add_argument("--name", default=None)
    p.add_argument("--graph", default=jb.GRAPH_PATH)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    g = jb.load_graph(args.graph)
    if args.flow_id:
        flow = next((f for f in g.get("flows", []) if f["id"] == args.flow_id),
                    None)
        if flow is None:
            raise SystemExit(f"flow '{args.flow_id}' not in {args.graph}")
        wps = [(w["node"], float(w.get("dwell_s", 0))) for w in flow["waypoints"]]
        name = args.name or args.flow_id
    elif args.waypoints:
        wps = [(w.strip(), 0.0) for w in args.waypoints.split(",")]
        name = args.name or "adhoc_flow"
    else:
        raise SystemExit("need --flow_id or --waypoints")

    plan = {"flow": name, "graph": args.graph,
            "waypoints": [{"node": n, "dwell_s": d} for n, d in wps],
            "seams": [], "ready": True}
    print(f"[plan:{name}] {' -> '.join(n for n, _ in wps)}")

    for (a, _), (b, dwell) in zip(wps, wps[1:]):
        edge = next((e for e in g.get("edges", [])
                     if e.get("from") == a and e.get("to") == b), None)
        act, why = board_action(edge)
        prereq = node_prereqs(g, b)
        seam = {"edge": f"{a}_to_{b}",
                "edge_id": edge["id"] if edge else None,
                "board_status": (edge or {}).get("status"),
                "action": act, "why": why,
                "routing": ROUTING.get(prereq.get("balance_class", "stable"),
                                       ROUTING["stable"]),
                "destination_prereqs": prereq}
        if dwell > 0:
            seam["dwell_s"] = dwell

        if act in ("TRAIN", "TRAIN_NEW", "DECOMPOSE", "REROUTE"):
            cands, synth = mine_demos(g, a, b)
            seam["demo_candidates"] = cands or "NONE in corpus"
            if synth:
                seam["synth_fallback_cmd"] = synth
            # regime-boundary feasibility (disjoint support + big posture gap)
            _, _, sa = node_sig(g, a)
            _, _, sb = node_sig(g, b)
            if sa and sb:
                j, dz, _ = sig_match(sa, sb, jaccard_min=0.0, up_z_tol=99)
                if j == 0.0 and dz > 0.5:
                    seam["feasibility_flag"] = (
                        f"REGIME BOUNDARY (support Jaccard 0, up_z gap {dz}) — "
                        "the tadasana<->plank / tadasana->crow class. Run the "
                        "t=0 probe on a SHORT budget before committing; "
                        "prefer decompose.")
            dec = decompose_candidates(g, a, b)
            if dec:
                seam["decompose_candidates"] = dec
            seam["post_train"] = [
                f"register + certify: python tools/bank_edge.py --edge_id "
                f"{a}_to_{b} [--register --from_node {a} --to_node {b}] "
                f"--dir <run_dir> --cert <run_dir>/bridge_cert.json",
                f"cert: env -u DISPLAY {PY} tools/certify_bridge.py --edge_dir "
                f"<run_dir> --target_node_dir {prereq.get('dir') or '<node_dir>'}",
            ]
        if act == "CERTIFY" and edge is not None:
            seam["cert_cmd"] = (f"env -u DISPLAY {PY} tools/certify_bridge.py "
                                f"--edge_dir {edge.get('dir')} --edge_model "
                                f"{edge.get('model', 'model.pt')} "
                                f"--target_node_dir {prereq.get('dir')}")
        if act not in ("REUSE",) or prereq["todo"]:
            plan["ready"] = False
        plan["seams"].append(seam)

        flag = " !" + seam["feasibility_flag"].split("(")[0].strip() \
            if seam.get("feasibility_flag") else ""
        todo = f" prereqs:{len(prereq['todo'])}" if prereq["todo"] else ""
        print(f"  {a} -> {b:<14} {act:<10} "
              f"(board: {(edge or {}).get('status')}){todo}{flag}")

    out = args.out or os.path.join("output", "yoga_flows", name,
                                   "flow_plan.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        yaml.safe_dump(plan, fh, sort_keys=False, width=100)
    print(f"[plan:{name}] ready={plan['ready']} -> {out}")


if __name__ == "__main__":
    main()
