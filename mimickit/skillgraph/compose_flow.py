"""Compose and execute a multi-pose yoga flow over the skill graph
(spec section 11 inference-time composition, section 12 claim 3).

A flow is a sequence of poses, e.g. [scorpion, handstand, scorpion], that exists
in NO single mocap clip. It is executed stage by stage: stage i runs the trained
transition policy P_i -> P_{i+1}, RSI'd from the states the PREVIOUS stage
actually REACHED -- not from P_i's clean terminal-state set. Feeding the realised
states forward exposes compounding error (section 13): a stage's input is the
previous stage's output, warts and all. Per stage we report the success rate
(reach & hold the next pose's classifier-gated region); end-to-end we report the
surviving fraction vs flow length (the compounding-error curve).

Each stage runs in its OWN process (Isaac Lab allows one sim app per process), so
the orchestrator subprocesses `--run_stage` invocations of this same file and
threads a reached-states file between them.

Usage (after the needed edges are trained):
  env_isaaclab/bin/python mimickit/skillgraph/compose_flow.py \
     --flow scorpion handstand scorpion
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import skillgraph.skill_io as sio
from skillgraph import orchestrator as orch

REPO = orch.REPO
PYEXE = orch.PYEXE


# ----------------------------- single stage (subprocess) --------------------
def run_stage(args):
    import envs.env_builder as env_builder
    import envs.base_env as base_env
    import learning.agent_builder as agent_builder
    import util.mp_util as mp_util
    import util.util as util

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    env_cfg = os.path.join("data", "envs", "transition_{}_to_{}_env.yaml".format(args.frm, args.to))
    if (args.init_states and args.init_states != "default"):
        import yaml
        with open(os.path.join(REPO, env_cfg), "r") as f:
            cfg = yaml.safe_load(f)
        cfg["env"]["init_states_file"] = args.init_states
        tmp = tempfile.NamedTemporaryFile("w", suffix="_flowstage_env.yaml",
                                          dir=os.path.join(REPO, "data", "envs"), delete=False)
        yaml.safe_dump(cfg, tmp); tmp.close()
        env_cfg = os.path.relpath(tmp.name, REPO)

    env = env_builder.build_env(env_cfg, args.num_envs, device, visualize=False)
    # Flow stages start from the PREVIOUS stage's reached states (real handoff),
    # never the training-time curriculum starts partway toward B.
    if (hasattr(env, "_curriculum_init_frac")):
        env._curriculum_init_frac = 0.0
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file); agent.eval(); agent.set_mode(agent._mode.__class__.TEST)

    control_freq = int(round(1.0 / env._engine.get_timestep()))
    hold_steps = int(round(args.success_hold_seconds * control_freq))
    NULL = base_env.DoneFlags.NULL.value
    N = args.num_envs

    def get_state():
        cid = env._get_char_id(); e = env._engine
        return {"root_pos": e.get_root_pos(cid).clone(), "root_rot": e.get_root_rot(cid).clone(),
                "root_vel": e.get_root_vel(cid).clone(), "root_ang_vel": e.get_root_ang_vel(cid).clone(),
                "dof_pos": e.get_dof_pos(cid).clone(), "dof_vel": e.get_dof_vel(cid).clone()}

    ever_held = torch.zeros(N, dtype=torch.bool, device=device)
    n_eps = 0; n_succ = 0
    held = {k: [] for k in ["root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]}
    obs, info = agent._reset_envs()
    for step in range(args.num_steps):
        with torch.no_grad():
            action = agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        st = get_state()
        obs, r, done, info = env.step(action)
        holding = env._goal_hold_count >= hold_steps
        ever_held |= holding
        for e in holding.nonzero(as_tuple=False).flatten().tolist():
            for k in held:
                held[k].append(st[k][e].clone())
        done_ids = (done != NULL).nonzero(as_tuple=False).flatten()
        if (len(done_ids) > 0):
            for e in done_ids.tolist():
                n_eps += 1; n_succ += 1 if ever_held[e] else 0
            ever_held[done_ids] = False
            obs, info = agent._reset_envs(done_ids)

    reached = {k: (torch.stack(v) if len(v) else torch.zeros(0, *([3] if "pos" in k or "vel" in k else [4]))) for k, v in held.items()}
    n_reached = reached["root_pos"].shape[0]
    if (n_reached > args.max_reached):
        sel = torch.randperm(n_reached)[:args.max_reached]
        reached = {k: v[sel] for k, v in reached.items()}
        n_reached = args.max_reached
    if (n_reached > 0):
        reached["features"] = torch.zeros(n_reached, 1)
        reached["motion_time"] = torch.zeros(n_reached)
    torch.save(reached, args.out_reached)
    with open(args.result_json, "w") as f:
        json.dump({"success_rate": n_succ / max(n_eps, 1), "n_eps": n_eps,
                   "n_reached": int(n_reached)}, f)
    print("[stage {}->{}] success={:.3f} ({} eps) reached={}".format(
        args.frm, args.to, n_succ / max(n_eps, 1), n_eps, n_reached))
    return


# ----------------------------- orchestrator ---------------------------------
def orchestrate(args):
    flow = args.flow
    assert len(flow) >= 2, "a flow needs >= 2 poses"
    print("=== composing flow: {} ===".format(" -> ".join(flow)))
    tmp_dir = tempfile.mkdtemp()
    stage_rates = []
    init_states = "default"
    for i in range(len(flow) - 1):
        A, B = flow[i], flow[i + 1]
        edge = sio.get_edge(A, B)
        assert edge is not None and edge.get("status") == "trained", \
            "edge {}->{} not trained (status={})".format(A, B, edge and edge.get("status"))
        model = os.path.join(REPO, "output", "yoga_transitions", "{}_to_{}".format(A, B), "model.pt")
        assert os.path.exists(model), "missing trained policy {}".format(model)
        out_reached = os.path.join(tmp_dir, "reached_{}.pt".format(i))
        result_json = os.path.join(tmp_dir, "result_{}.json".format(i))
        cmd = [PYEXE, "mimickit/skillgraph/compose_flow.py", "--run_stage",
               "--from", A, "--to", B, "--model_file", model,
               "--init_states", init_states, "--out_reached", out_reached,
               "--result_json", result_json, "--num_envs", str(args.num_envs),
               "--num_steps", str(args.num_steps),
               "--success_hold_seconds", str(args.success_hold_seconds),
               "--master_port", str(args.master_port + i), "--agent_config", args.agent_config]
        subprocess.run(cmd, cwd=REPO, check=True)
        res = json.load(open(result_json))
        stage_rates.append(res["success_rate"])
        print("  stage {}: {} -> {}  success={:.3f}  reached-states={}".format(
            i, A, B, res["success_rate"], res["n_reached"]))
        if (res["n_reached"] == 0):
            print("  flow BREAKS at stage {} (nothing reached {})".format(i, B)); break
        init_states = out_reached

    print("\n=== flow result: {} ===".format(" -> ".join(flow)))
    cum = 1.0
    for i, r in enumerate(stage_rates):
        cum *= r
        print("  stage {} ({}->{}): stage={:.3f}  cumulative={:.3f}".format(
            i, flow[i], flow[i + 1], r, cum))
    print("  END-TO-END (compounded) : {:.3f}".format(cum))
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", nargs="+")
    p.add_argument("--agent_config", default="data/agents/amp_smpl_transition_hybrid_agent.yaml")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--success_hold_seconds", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=6980)
    p.add_argument("--rand_seed", type=int, default=321)
    p.add_argument("--max_reached", type=int, default=2000)
    # internal single-stage mode
    p.add_argument("--run_stage", action="store_true")
    p.add_argument("--from", dest="frm")
    p.add_argument("--to")
    p.add_argument("--model_file")
    p.add_argument("--init_states", default="default")
    p.add_argument("--out_reached")
    p.add_argument("--result_json")
    args = p.parse_args()
    if (args.run_stage):
        run_stage(args)
    else:
        assert args.flow, "--flow required"
        orchestrate(args)


if __name__ == "__main__":
    main()
