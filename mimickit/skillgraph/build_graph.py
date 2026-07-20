"""Generate the v3 skill-graph interface spec (yoga_flow_paper_plan.md §5.1, §handoff).

Emits skills/graph.yaml = routing table + interface/repair spec. Encodes the locked
handoff model (user 2026-07-15): B1 (finetune node basin on edge arrivals) + E1
(recovery-calibrated arrival gate) + D1 (arm-agnostic hubs), with A1 (canonical pose
port) as the fallback contract. Per edge, `handoff.recovery_ladder` is the ORDERED
list of local repairs the LLM orchestrator tries when a flow seam fails.

Regenerate whenever nodes/edges are trained: it reads on-disk status from the run dirs.
Plain Python (no Isaac); backs up any existing graph.yaml.

MERGE RULE (Yoga_orchestration_protocol.md §2.1): the job board (tools/job_board.py,
bank_edge.py, repoint_node.py) is the single writer of live state — regeneration must
never clobber it. For ids present in the existing graph, persisted board fields
(status, dir/model, version, basin provenance, gate staleness, certs, terminals)
OVERLAY the spec defaults; board-registered nodes/edges/flows unknown to the spec are
carried over wholesale; on_disk is recomputed from the merged dir/model.

  env_isaaclab/bin/python mimickit/skillgraph/build_graph.py
"""
import os
import shutil

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES_ROOT = "output/yoga_nodes_v2"
EDGES_ROOT = "output/yoga_edges_v3"
CALIBRATED = {"downdog", "crow", "warrior3", "tadasana", "malasana"}  # have gate.yaml
PRECARIOUS = {"crow", "handstand", "scorpion", "headstand", "pincha"}  # inverted/small base
# control-limited single-leg standing balances: hold their own states but the basin is
# narrow (CoM drifts out dynamically; positional terms can't damp it) -> the deferred
# balance-controller class. Their arrival ladder needs a balance_curriculum rung too.
NEEDS_BALANCE = {"warrior3", "natarajasana", "tree", "eagle", "big_toe", "half_moon"}

# id -> (run-dir under NODES_ROOT, role, regime).  role: hub|gateway|pose
NODE_SPEC = [
    ("tadasana",     "tadasana_lt_ampft",     "hub",     "standing"),
    ("malasana",     "malasana_lt_ampft",     "hub",     "squat"),
    ("plank",        "plank_lt_ampft",        "gateway", "floor"),
    ("downdog",      "downdog_lt_ampft",      "gateway", "floor_pike"),
    ("downdog3L",    "downdog3L_lt_ampft",    "pose",    "floor_pike"),
    ("warrior2",     "warrior2_lt_ampft",     "pose",    "standing_wide"),
    ("warrior3",     "warrior3_coadapt",      "pose",    "standing_balance"),  # B1-finetuned
    ("triangle",     "triangle_lt_ampft",     "pose",    "standing_wide"),
    ("half_moon",    "half_moon_lt_ampft",    "pose",    "standing_balance"),
    ("tree",         "tree_lt_ampft2",        "pose",    "standing_balance"),
    ("eagle",        "eagle_lt_ampft",        "pose",    "standing_balance"),
    ("natarajasana", "natarajasana_lt_ampft", "pose",    "standing_balance"),
    ("side_plank",   "side_plank_lt_ampft",   "pose",    "floor"),
    ("camel",        "camel_lt_ampft",        "pose",    "backbend"),
    ("wheel",        "wheel_lt_ampft",        "pose",    "backbend"),
    ("crow",         "crow_lt_comfwd",        "pose",    "arm_balance"),
    ("headstand",    "headstand_lt_ampft",    "pose",    "inversion"),
    ("handstand",    "handstand_lt_ampft",    "pose",    "inversion"),
    ("scorpion",     "scorpion_lt_ampft",     "pose",    "inversion"),
]

# id, from, to, routing_class, run-dir (under EDGES_ROOT or None), model, status, recovery, note
EDGE_SPEC = [
    ("crow_to_tadasana", "crow", "tadasana", "deepmimic", "crow_to_tadasana", "model.pt",
     "banked", 0.77, "AMP-hybrid warm-start; zero-seam (crow node's own clip)"),
    ("malasana_to_tadasana", "malasana", "tadasana", "deepmimic", "malasana_to_tadasana_deepmimic",
     "model.pt", "banked", 1.0, "functionally solved; GAIT slides (malasana = corpus-worst step)"),
    ("warrior2_to_tadasana", "warrior2", "tadasana", "deepmimic", "warrior2_to_tadasana_deepmimic",
     "model.pt", "banked", 1.0, "signed-off; cleanest lateral step; seam arms-gap ~2.2 rad into tadasana"),
    ("tadasana_to_warrior2", "tadasana", "warrior2", "deepmimic", "tadasana_to_warrior2_deepmimic",
     "model_solved.pt", "solved", None, "pure DeepMimic; pending visual sign-off"),
    ("tadasana_to_warrior3", "tadasana", "warrior3", "deepmimic", "tadasana_to_warrior3_deepmimic",
     "model_solved.pt", "solved", None, "single-leg entry solved by pure DeepMimic; pending visual"),
    ("tadasana_to_downdog", "tadasana", "downdog", "deepmimic", "tadasana_to_downdog_deepmimic",
     "model_solved.pt", "solved", None, "pure DeepMimic; pending visual"),
    ("downdog3L_to_handstand", "downdog3L", "handstand", "mixed", "downdog3L_to_handstand_deepmimic",
     "model_solved.pt", "needs_amp_retrain", None,
     "kick-up traverses but HOLD fingertips under DeepMimic; retrain disc-dominant AMP"),
    ("tadasana_to_crow", "tadasana", "crow", "amp", "tadasana_to_crow", "model.pt",
     "hard", 0.008, "camped x3; DECOMPOSE via malasana (tadasana->malasana->crow)"),
    # planned / synthesis (no model on disk yet)
    ("tadasana_to_malasana", "tadasana", "malasana", "deepmimic", None, None,
     "needs_train", None, "cheap (reverse of banked); inherits malasana slide risk"),
    ("downdog_to_tadasana", "downdog", "tadasana", "deepmimic", None, None,
     "needs_train", None, "cheap (reverse of banked tadasana->downdog)"),
    ("downdog_to_downdog3L", "downdog", "downdog3L", "deepmimic", None, None,
     "synth", None, "trivial leg-lift; mirror handstand->downdog3L reverse-kick"),
    ("warrior3_to_tadasana", "warrior3", "tadasana", "deepmimic", None, None,
     "synth", None, "single-leg step-down; mirror the banked entry; control-limited class"),
    ("handstand_to_tadasana", "handstand", "tadasana", "mixed", None, None,
     "hard", None, "controlled inverted descent (1 demo); or stepped exit via downdog3L->downdog"),
    ("malasana_to_crow", "malasana", "crow", "amp", None, None,
     "needs_train", None, "THE crux; synth from crow standing-entry + disc-dominant AMP; reuse crow gate"),
]

# flow id -> [(node, dwell_s), ...]  ; dwell_s>0 only at named/held poses
FLOW_SPEC = {
    "warrior_roundtrip": [("tadasana", 0), ("warrior2", 0), ("tadasana", 0)],  # plumbing smoke
    "warrior_hub":       [("tadasana", 0), ("warrior2", 2.0), ("tadasana", 0),
                          ("warrior3", 2.0), ("tadasana", 2.0)],
    "downdog_to_handstand": [("tadasana", 0), ("downdog", 1.0), ("downdog3L", 0),
                             ("handstand", 4.0), ("tadasana", 2.0)],   # hold handstand = show the skill
    "squat_to_crow":     [("tadasana", 0), ("malasana", 1.0), ("crow", 4.0), ("tadasana", 2.0)],
    "warrior3_roundtrip": [("tadasana", 0), ("warrior3", 3.0), ("tadasana", 2.0)],  # single-leg hold
}

ROLE_OF = {n[0]: n[2] for n in NODE_SPEC}

# board-owned fields that survive regeneration (single-writer rule)
PERSIST_EDGE = ("status", "dir", "model", "recovery_oracle", "note", "diagnosis",
                "superseded_by", "last_cert")
PERSIST_NODE = ("status", "dir", "model", "version", "basin_finetuned_from",
                "gate_calibrated_on", "oracle_validity", "needs", "arrival_gate")


def merge_persisted(fresh, existing_path):
    """Overlay live board state from the existing graph onto the spec-built one."""
    if not os.path.exists(existing_path):
        return fresh
    with open(existing_path) as fh:
        old = yaml.safe_load(fh) or {}
    carried, kept = [], 0
    for kind, keys in (("nodes", PERSIST_NODE), ("edges", PERSIST_EDGE)):
        old_by = {o["id"]: o for o in old.get(kind, []) if isinstance(o, dict)}
        fresh_ids = set()
        for f in fresh.get(kind, []):
            fresh_ids.add(f["id"])
            o = old_by.get(f["id"])
            if not o:
                continue
            for k in keys:
                if k in o and o[k] is not None:
                    f[k] = o[k]
                    kept += 1
        for oid, o in old_by.items():
            if oid not in fresh_ids:
                fresh[kind].append(o)
                carried.append(f"{kind[:-1]}:{oid}")
    fresh_flows = {fl["id"] for fl in fresh.get("flows", [])}
    for fl in old.get("flows", []):
        if fl["id"] not in fresh_flows:
            fresh["flows"].append(fl)
            carried.append(f"flow:{fl['id']}")
    # on_disk is DERIVED — recompute from the merged dir/model
    for n in fresh["nodes"]:
        n["on_disk"] = bool(n.get("dir") and os.path.exists(
            os.path.join(n["dir"], n.get("model", "model.pt"))))
    for e in fresh["edges"]:
        e["on_disk"] = bool(e.get("dir") and e.get("model") and os.path.exists(
            os.path.join(e["dir"], e["model"])))
    print(f"merge: overlaid {kept} persisted board fields"
          + (f"; carried board-only objects: {', '.join(carried)}"
             if carried else ""))
    return fresh


def recovery_ladder(to_id):
    """The ORDERED local repairs the orchestrator tries when this arrival seam fails."""
    lad = [
        {"action": "recalibrate_gate", "on": "edge_arrivals", "status": "pending"},        # E1
        {"action": "finetune_node_basin", "node": to_id, "on": "edge_arrivals", "status": "pending"},  # B1
    ]
    if ROLE_OF.get(to_id) == "hub":
        lad.append({"action": "make_arm_agnostic", "node": to_id, "status": "pending"})    # D1
    if to_id in PRECARIOUS or to_id in NEEDS_BALANCE:
        lad.append({"action": "balance_curriculum", "node": to_id, "status": "pending"})   # D2
    lad.append({"action": "retarget_edge_to_port", "target": to_id, "status": "pending"})  # A1 fallback
    return lad


def build():
    os.chdir(REPO)
    nodes = []
    for nid, d, role, regime in NODE_SPEC:
        run = os.path.join(NODES_ROOT, d)
        on_disk = os.path.exists(os.path.join(run, "model.pt"))
        node = {
            "id": nid, "role": role, "regime": regime, "recipe": "amp_first_finetune",
            "balance_class": ("precarious" if nid in PRECARIOUS else
                              "control_limited" if nid in NEEDS_BALANCE else "stable"),
            "dir": run, "model": "model.pt", "on_disk": on_disk,
            "port": {"hold_ref": "hold_states.pt#hold_ref",
                     "style_agnostic": ["arms"] if nid == "tadasana" else []},
            "status": "signed_off",
        }
        if nid in CALIBRATED:
            node["arrival_gate"] = os.path.join(run, "gate.yaml")
            node["gate_calibrated_on"] = "hold_states"   # -> edge_arrivals after E1
            node["oracle_validity"] = 1.0
        node["basin_finetuned_from"] = []                # B1 provenance (grows as edges land)
        nodes.append(node)

    edges = []
    for eid, frm, to, rc, d, model, status, rec, note in EDGE_SPEC:
        run = os.path.join(EDGES_ROOT, d) if d else None
        on_disk = bool(run and model and os.path.exists(os.path.join(run, model)))
        edge = {
            "id": eid, "from": frm, "to": to, "routing_class": rc,
            "dir": run, "model": model, "on_disk": on_disk, "status": status,
            "note": note,
            "handoff": {
                "lands_on": to, "departs_from": frm,
                "switch": "recovery_gate", "gate_calibrated_on": "hold_states",
                "recovery_ladder": recovery_ladder(to),
            },
        }
        if rec is not None:
            edge["recovery_oracle"] = rec
        edges.append(edge)

    flows = [{"id": fid, "waypoints": [{"node": n, "dwell_s": w} for n, w in wp]}
             for fid, wp in FLOW_SPEC.items()]

    graph = {
        "version": 3,
        "asset": "smpl_boxhands_lowtorque",
        "handoff_model": {
            "primary": ["E1_recovery_calibrated_gate", "B1_finetune_node_basin_on_arrivals"],
            "hub_fix": "D1_arm_agnostic_hubs",
            "port_contract": "A1_canonical_pose_port",
            "note": "Per-edge handoff.recovery_ladder = the orchestrator's ordered local "
                    "repairs on seam failure. Dwell only at named/held poses (flow waypoints).",
        },
        "nodes": nodes, "edges": edges, "flows": flows,
    }

    out = os.path.join("skills", "graph.yaml")
    os.makedirs("skills", exist_ok=True)
    if os.path.exists(out):
        bak = os.path.join("skills", "graph_v2_backup.yaml")
        if not os.path.exists(bak):
            shutil.copy(out, bak)
            print(f"backed up existing graph -> {bak}")
    graph = merge_persisted(graph, out)
    with open(out, "w") as fh:
        yaml.safe_dump(graph, fh, sort_keys=False, default_flow_style=False, width=100)

    n_disk = sum(n["on_disk"] for n in nodes)
    e_disk = sum(e["on_disk"] for e in edges)
    print(f"wrote {out}: {len(nodes)} nodes ({n_disk} on-disk), "
          f"{len(edges)} edges ({e_disk} on-disk), {len(flows)} flows")
    missing = [n["id"] for n in nodes if not n["on_disk"]] + \
              [e["id"] for e in edges if not e["on_disk"] and e["status"] not in
               ("needs_train", "synth", "hard")]
    if missing:
        print("WARNING missing-on-disk (expected trained):", missing)


if __name__ == "__main__":
    build()
