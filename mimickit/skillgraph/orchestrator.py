"""Orchestrator tool interface for the yoga skill graph (spec section 5).

These are the meta-controller's *action space*, each grounded in a concrete
MimicKit call so the (eventual) LLM orchestrator -- or a human -- drives the loop
without re-implementing anything. The simulator disposes; the orchestrator only
proposes (spec section 13). Read-only / config-gen tools run inline; training and
rollout tools shell out to the standalone scripts (run.py, collect_skill.py) so
they inherit the exact, tested launch path and can run non-blocking.

Tools implemented:
  read_skill_graph / update_skill_graph / read_skill_card / list_skills
  query_initiation_classifier(skill_id, features)   -> p_init  (section 4 gate)
  make_transition_config(A, B)                       -> writes env+agent yaml (sec 8)
  launch_training(env, agent, out_dir, ...)          -> run_id  (non-blocking, sec 5)
  launch_transition_training(A, B, ...)              -> run_id  (RSI=A_term -> B, sec 8)
  monitor_training(run_dir)                          -> curves  (parse log, sec 5)
  collect_skill_states(skill_id, ...)                -> shells collect_skill.py

query_value_function / evaluate_transition need a live Isaac env (obs space +
rollout); they are thin shell-outs documented here and realised by the
standalone scripts, not duplicated inline.
"""
import os
import re
import subprocess
import sys
import yaml

import torch

THIS = os.path.dirname(__file__)
REPO = os.path.normpath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mimickit"))

import skillgraph.skill_io as sio
import skillgraph.init_classifier as ic

PYEXE = os.environ.get("MIMICKIT_PY",
                       "/home/visakii/Documents/moves/env_isaaclab/bin/python")
AGENT_DEFAULT = "data/agents/deepmimic_smpl_ppo_agent.yaml"          # pose-node / collect (PPO)
TRANSITION_AGENT = "data/agents/amp_smpl_transition_hybrid_agent.yaml"  # transitions (hybrid AMP)


# ----------------------------- world-model IO -------------------------------
def read_skill_graph(root=sio.DEFAULT_ROOT):
    return sio.read_graph(root)


def update_skill_graph(frm, to, meta, root=sio.DEFAULT_ROOT):
    return sio.update_edge(frm, to, meta, root)


def read_skill_card(skill_id, root=sio.DEFAULT_ROOT):
    return sio.read_card(skill_id, root)


def list_skills(root=sio.DEFAULT_ROOT):
    return sio.list_skills(root)


# --------------------------- competence queries -----------------------------
def query_initiation_classifier(skill_id, features, root=sio.DEFAULT_ROOT):
    """p_init = C_skill(features). `features` is [N, D] in the canonical layout
    of skillgraph.state_features (the same layout every skill is collected in)."""
    clf_path = os.path.join(sio.skill_dir(skill_id, root), "classifier.pt")
    model, _ = ic.load_classifier(clf_path)
    feats = torch.as_tensor(features, dtype=torch.float32)
    return model.prob(feats)


# --------------------------- transition config gen --------------------------
def make_transition_config(A, B, root=sio.DEFAULT_ROOT, reward_goal_w=1.0,
                           goal_threshold=0.5, goal_pose_dist=0.18, goal_up_z_tol=0.4,
                           transition_ramp_seconds=5.0, episode_length=8.0,
                           num_disc_obs_steps=10, keep_aux=False,
                           curriculum_init_frac=0.5, curriculum_phase_max=1.0,
                           reward_goal_shape_w=2.0, reward_goal_shape_scale=3.0,
                           env_out=None, agent_out=None):
    """Write a runnable HYBRID transition env config (spec sections 8, C.4, G):
      RSI       = A's terminal states,
      reference = a per-env MOVING A->B interpolation (TransitionEnv, spec C.1),
      style     = an AMP discriminator on target B's hold clip (non-farmable),
      goal obs  = B's hold pose appended to the obs (#2 goal-conditioning),
      handoff   = B's POSE-DISCRIMINATIVE classifier AND a pose-distance check
                  (spec D).
    Clones B's env config so the online classifier features + obs layout + tracking
    match B. Pairs with the hybrid AMP transition agent. Returns (env, agent)."""
    cardA = sio.read_card(A, root)
    cardB = sio.read_card(B, root)
    assert cardA is not None and cardB is not None, "both skills must have cards"

    with open(os.path.join(REPO, cardB["env_config"]), "r") as f:
        cfg = yaml.safe_load(f)
    env = cfg["env"]
    cfg["env_name"] = "transition"

    hpt = float(cardB["value_stats"]["hold_phase_time"])

    # hybrid AMP switches (TransitionEnv subclasses AMPEnv)
    env["num_disc_obs_steps"] = num_disc_obs_steps
    env["enable_task_tracking"] = True

    # goal-conditioning + moving reference + pose-distance gate
    env["enable_goal_obs"] = True
    env["enable_tar_obs"] = False             # goal obs replaces the clip tar frames
    env["init_states_file"] = os.path.join("skills", A, "terminal_states.pt")
    env["goal_classifier"] = os.path.join("skills", B, "classifier.pt")
    env["goal_phase_time"] = hpt
    env["goal_demo_time_range"] = [max(0.0, hpt - 0.5), hpt + 0.5]  # disc demos over B's hold
    env["goal_threshold"] = goal_threshold
    env["goal_pose_dist"] = goal_pose_dist
    env["goal_up_z_tol"] = goal_up_z_tol
    env["reward_goal_w"] = reward_goal_w
    env["reward_goal_shape_w"] = reward_goal_shape_w      # spec section 8 dense shaping
    env["reward_goal_shape_scale"] = reward_goal_shape_scale
    env["transition_ramp_seconds"] = transition_ramp_seconds
    env["curriculum_init_frac"] = curriculum_init_frac    # spec C.3 RSI-along-path
    env["curriculum_phase_max"] = curriculum_phase_max
    env["pose_termination"] = False           # init is far from B
    env["episode_length"] = episode_length     # a transition is short
    env["log_tag"] = "transition_{}_to_{}".format(A, B)

    # Drop the farmable aux shaping (com_support / force_balance / energy /
    # orient-on-hands) cloned from B's NODE config. In a transition these are an
    # escape hatch the policy farms by LEAVING the pose (it stands up, banks
    # orient/com/energy, and abandons tracking) -- exactly the v2-attempt-1
    # failure (notes section 7.2 / C.2: scorpion->handstand collapsed upright,
    # end shape dist 0.87 m, Orient_R 0.48 while Pose_R ~0). With them off, the
    # ONLY reward is DeepMimic tracking of the MOVING reference + the goal bonus
    # + the AMP disc style, so reaching B's pose is the only way to score.
    if (not keep_aux):
        for w in ("reward_com_support_w", "reward_force_balance_w",
                  "reward_energy_w", "reward_orient_w"):
            env[w] = 0.0

    env_out = env_out or os.path.join(REPO, "data", "envs",
                                      "transition_{}_to_{}_env.yaml".format(A, B))
    with open(env_out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    agent_out = agent_out or TRANSITION_AGENT
    print("[orchestrator] wrote hybrid transition config {} (goal_phase_time={:.2f}, ramp={}s)".format(
        env_out, hpt, transition_ramp_seconds))
    return env_out, agent_out


# ------------------------------- training -----------------------------------
def launch_training(env_config, agent_config, out_dir, max_samples=200_000_000,
                    num_envs=4096, seed=42, master_port=6700, visualize=False):
    """Non-blocking train launch (wraps run.py). Returns the Popen handle; the
    run_id is out_dir. int_output_dir is always set (a NaN divergence otherwise
    overwrites the last healthy model -- a hard-won lesson from the hybrid runs)."""
    os.makedirs(os.path.join(out_dir, "int"), exist_ok=True)
    cmd = [PYEXE, "mimickit/run.py", "--mode", "train", "--num_envs", str(num_envs),
           "--env_config", env_config, "--agent_config", agent_config,
           "--visualize", "true" if visualize else "false",
           "--max_samples", str(max_samples), "--rand_seed", str(seed),
           "--master_port", str(master_port),
           "--log_file", os.path.join(out_dir, "log.txt"),
           "--out_model_file", os.path.join(out_dir, "model.pt"),
           "--int_output_dir", os.path.join(out_dir, "int")]
    stdout = open(os.path.join(out_dir, "stdout.txt"), "w")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=stdout, stderr=subprocess.STDOUT)
    print("[orchestrator] launched train pid={} -> {}".format(proc.pid, out_dir))
    return proc


def launch_transition_training(A, B, out_dir=None, root=sio.DEFAULT_ROOT, **kw):
    env_cfg, agent_cfg = make_transition_config(A, B, root)
    out_dir = out_dir or os.path.join(REPO, "output", "yoga_transitions", "{}_to_{}".format(A, B))
    # config paths are repo-relative for run.py's cwd=REPO
    env_rel = os.path.relpath(env_cfg, REPO)
    # The hybrid disc + ~200k replay buffer is ~15.5 GB, so transitions run one
    # at a time and with fewer envs than the pure-DeepMimic pose skills.
    kw.setdefault("num_envs", 2048)
    kw.setdefault("max_samples", 150_000_000)
    return launch_training(env_rel, agent_cfg, out_dir, **kw)


def monitor_training(run_dir):
    """Latest console metrics + health for a run dir. The agent's print_log emits
    'Key | value' lines to stdout; we keep the last value seen per key and also
    flag a NaN-guard abort (the divergence the NaN-guard now catches). Returns
    {'latest': {...}, 'diverged': bool, 'has_log': bool}."""
    out = {"latest": {}, "diverged": False, "has_log": False}
    path = os.path.join(run_dir, "stdout.txt")
    if (not os.path.exists(path)):
        return out
    out["has_log"] = True
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    if ("[NaN-guard]" in text):
        out["diverged"] = True
    text = text.replace("\r", "\n")
    for line in text.splitlines():
        if ("|" not in line):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if (len(parts) == 2):
            k, v = parts
            try:
                out["latest"][k] = float(v)
            except ValueError:
                pass
    return out


def mark_edge_diverged(frm, to, run_dir, root=sio.DEFAULT_ROOT):
    """Record a NaN-diverged transition training as a failed edge (spec: log
    failed edges). eval_transition records below-floor edges as 'retry'; this
    handles the runs that never finish because they diverged."""
    return sio.update_edge(frm, to, {
        "status": "retry",
        "diagnosis": "training diverged (NaN-guard abort); see {}".format(
            os.path.join(run_dir, "stdout.txt"))}, root)


def collect_skill_states(skill_id, env_config, model_file, agent_config=AGENT_DEFAULT,
                         master_port=6900, extra=None):
    cmd = [PYEXE, "mimickit/skillgraph/collect_skill.py", "--skill_id", skill_id,
           "--env_config", env_config, "--agent_config", agent_config,
           "--model_file", model_file, "--master_port", str(master_port)]
    if (extra):
        cmd += extra
    print("[orchestrator] collecting states for {}".format(skill_id))
    return subprocess.run(cmd, cwd=REPO)
