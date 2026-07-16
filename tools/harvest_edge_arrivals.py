"""Harvest an edge's DELIVERED-arrival states for co-adaptation (B1).

Rolls a trained EDGE policy from the true handoff (RSI at the lead-in) with a
per-env ACTION-NOISE spectrum (for variety), and captures each surviving env's
full dynamic state at a settle time. States are SE(2)-aligned to the target
NODE's hold_ref and stamped with the hold-phase motion_time, so they can be fed
straight into the node env's config-gated `init_states_file` reset (a fraction of
node-finetune resets then start from these off-manifold deliveries, teaching the
node to HOLD what the edge actually delivers).

Usage (repo root, env python; ONE Isaac job at a time):
  env_isaaclab/bin/python tools/harvest_edge_arrivals.py \
      --edge_dir output/yoga_edges_v3/tadasana_to_warrior3_deepmimic --edge_model model_solved.pt \
      --target_node_dir output/yoga_nodes_v2/warrior3_lt_ampft \
      --num_envs 256 --settle_at 11.5 --noise 0.12 \
      --out output/yoga_nodes_v2/warrior3_lt_ampft/edge_arrivals.pt
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.util as util

STATE_KEYS = ["root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]


def heading_align(state, ref_root_pos, ref_root_rot):
    """SE(2)-align a batch into the reference frame (root xy->ref, heading->ref; z kept)."""
    q_s = torch_util.calc_heading_quat(state["root_rot"])
    q_r = torch_util.calc_heading_quat(ref_root_rot.expand_as(state["root_rot"]))
    q = torch_util.quat_mul(q_r, torch_util.quat_conjugate(q_s))
    out = {k: v.clone() for k, v in state.items()}
    out["root_rot"] = torch_util.quat_mul(q, state["root_rot"])
    for k in ("root_vel", "root_ang_vel"):
        out[k] = torch_util.quat_rotate(q, state[k])
    rp = state["root_pos"].clone()
    rp[:, 0:2] = ref_root_pos[0, 0:2]
    out["root_pos"] = rp
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edge_dir", required=True)
    p.add_argument("--edge_model", default="model.pt")
    p.add_argument("--target_node_dir", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--init_at", type=float, default=0.0, help="RSI time (0 = true handoff)")
    p.add_argument("--settle_at", type=float, default=None,
                   help="capture time (s); default = edge episode_length - 0.5")
    p.add_argument("--noise", type=float, default=0.12, help="max per-env action-noise std")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29840)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    env_cfg = os.path.join(args.edge_dir, "env_config.yaml")
    agent_cfg = os.path.join(args.edge_dir, "agent_config.yaml")
    model = os.path.join(args.edge_dir, args.edge_model)
    env = env_builder.build_env(env_cfg, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(agent_cfg, env, device)
    agent.load(model)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)
    env._init_time_range = [args.init_at, args.init_at]

    hold_ref = torch.load(os.path.join(args.target_node_dir, "hold_states.pt"),
                          map_location=device)["hold_ref"]
    cid = env._get_char_id()
    e = env._engine
    control_freq = int(round(1.0 / e.get_timestep()))
    import yaml
    ep_len = float(yaml.safe_load(open(env_cfg))["env"].get("episode_length", 12.0))
    settle = args.settle_at if args.settle_at is not None else ep_len - 0.5
    settle_step = int(round(settle * control_freq))
    N = args.num_envs
    FAIL = base_env.DoneFlags.FAIL.value

    # per-env action-noise scale (0 .. noise), for a spread of delivered poses
    scale = torch.linspace(0.0, args.noise, N, device=device).unsqueeze(-1)

    obs, info = agent._reset_envs()
    fell = torch.zeros(N, dtype=torch.bool, device=device)
    captured = None
    for step in range(settle_step + 1):
        with torch.no_grad():
            a = agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        a = a + scale * torch.randn_like(a)
        if step == settle_step:
            captured = {k: getattr(e, "get_" + k)(cid).clone() for k in STATE_KEYS}
        obs, r, done, info = env.step(a)
        fell |= (done == FAIL)
        if step >= settle_step:
            break

    keep = (~fell).nonzero(as_tuple=False).flatten()
    state = {k: captured[k][keep] for k in STATE_KEYS}
    aligned = heading_align(state, hold_ref["root_pos"], hold_ref["root_rot"])
    aligned["motion_time"] = hold_ref["motion_time"].expand(len(keep)).clone()
    torch.save(aligned, args.out)
    # provenance / sanity
    upl = torch.zeros(len(keep), 3, device=device); upl[:, 2] = 1.0
    upz = torch_util.quat_rotate(aligned["root_rot"], upl)[:, 2]
    print(f"[harvest] kept {len(keep)}/{N} (fell {int(fell.sum())}) @ t={settle:.1f}s")
    print(f"[harvest] delivered up_z: mean={float(upz.mean()):.3f} "
          f"min={float(upz.min()):.3f} max={float(upz.max()):.3f} "
          f"(node hold_ref up_z={float(hold_ref['up_z'][0]):.3f})")
    print(f"[harvest] wrote {args.out}")


if __name__ == "__main__":
    main()
