"""On-disk world model for the yoga skill graph (spec section 6).

Everything is human-readable yaml/torch so the (eventual) LLM orchestrator and a
human can both read and edit it. Layout under <root>/ (default: repo-level
`skills/`):

    skills/
      graph.yaml                      # transition matrix (edges + status)
      <skill_id>/
        card.yaml                     # skill card
        terminal_states.pt            # stable held-pose states for RSI
        classifier.pt                 # initiation classifier weights + meta
        dataset.pt                    # labeled (features, obs, label) for calib

`card.yaml` schema (a superset of spec section 6, grounded in this codebase):

    skill_id: handstand
    algo: deepmimic
    env_config: data/envs/deepmimic_smpl_handstand_orient_env.yaml
    agent_config: data/agents/deepmimic_smpl_ppo_agent.yaml
    policy_ckpt: output/yoga_orient/model_seed42.pt
    motion_file: data/motions/smpl/220923_Handstand_...
    hold_window: [75.1, 89.8]         # seconds, the longest stable hold
    contact_signature:
      ground_contacts: [L_Hand, R_Hand]
      inverted: true                  # root up-z < -0.5 during the hold
    competence_region:
      classifier_ckpt: skills/handstand/classifier.pt
      threshold: 0.5
      feature_names: [...]
    value_stats:
      hold_phase_time: 82.0           # representative held reference time
    terminal_state_distribution:
      states_file: skills/handstand/terminal_states.pt
      count: 1234
      held_seconds_mean: 12.7
      fall_rate: 0.08
"""
import os
import yaml
import torch

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "skills")


def skill_dir(skill_id, root=DEFAULT_ROOT):
    return os.path.normpath(os.path.join(root, skill_id))


def card_path(skill_id, root=DEFAULT_ROOT):
    return os.path.join(skill_dir(skill_id, root), "card.yaml")


def graph_path(root=DEFAULT_ROOT):
    return os.path.normpath(os.path.join(root, "graph.yaml"))


def ensure_skill_dir(skill_id, root=DEFAULT_ROOT):
    d = skill_dir(skill_id, root)
    os.makedirs(d, exist_ok=True)
    return d


def write_card(card, root=DEFAULT_ROOT):
    skill_id = card["skill_id"]
    ensure_skill_dir(skill_id, root)
    with open(card_path(skill_id, root), "w") as f:
        yaml.safe_dump(card, f, sort_keys=False, default_flow_style=False)
    return card_path(skill_id, root)


def read_card(skill_id, root=DEFAULT_ROOT):
    p = card_path(skill_id, root)
    if (not os.path.exists(p)):
        return None
    with open(p, "r") as f:
        return yaml.safe_load(f)


def list_skills(root=DEFAULT_ROOT):
    if (not os.path.isdir(root)):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if (os.path.exists(card_path(name, root))):
            out.append(name)
    return out


def read_graph(root=DEFAULT_ROOT):
    p = graph_path(root)
    if (not os.path.exists(p)):
        return {"edges": []}
    with open(p, "r") as f:
        g = yaml.safe_load(f)
    return g if (g is not None) else {"edges": []}


def write_graph(graph, root=DEFAULT_ROOT):
    os.makedirs(root, exist_ok=True)
    with open(graph_path(root), "w") as f:
        yaml.safe_dump(graph, f, sort_keys=False, default_flow_style=False)
    return graph_path(root)


def update_edge(frm, to, meta, root=DEFAULT_ROOT):
    """Insert/replace the A->B edge with `meta` (status, success_rate, ...)."""
    graph = read_graph(root)
    edges = graph.get("edges", [])
    found = False
    for e in edges:
        if (e.get("from") == frm and e.get("to") == to):
            e.update(meta)
            found = True
            break
    if (not found):
        entry = {"from": frm, "to": to}
        entry.update(meta)
        edges.append(entry)
    graph["edges"] = edges
    write_graph(graph, root)
    return graph


def get_edge(frm, to, root=DEFAULT_ROOT):
    for e in read_graph(root).get("edges", []):
        if (e.get("from") == frm and e.get("to") == to):
            return e
    return None


def init_graph(skill_ids, root=DEFAULT_ROOT):
    """Initialise a fully-connected, all-`untried` directed graph (Algorithm
    step 2: nodes=skills, edges=untried)."""
    edges = []
    for a in skill_ids:
        for b in skill_ids:
            if (a == b):
                continue
            edges.append({"from": a, "to": b, "status": "untried"})
    graph = {"nodes": list(skill_ids), "edges": edges}
    write_graph(graph, root)
    return graph
