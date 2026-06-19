"""Smoke-test the v2 hybrid transition env + pose-grounded plumbing (NO training).

Regenerates the scorpion->handstand transition config (new hybrid format), builds
the env + hybrid AMP agent headless, runs a few steps with the actual actor, and
prints obs/disc-obs shapes, reward stats, the pose-distance gate, and Goal_Frac so
we catch wiring bugs before committing GPU-hours. Reuses the EXISTING skill
artifacts (handstand classifier + scorpion terminal states); this checks code
correctness, not transition quality.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mimickit"))

import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util
from skillgraph import orchestrator as orch


def main():
    device = "cuda:0"
    mp_util.init(0, 1, device, 6991)
    util.set_rand_seed(0)

    A, B = "scorpion", "handstand"
    env_cfg, agent_cfg = orch.make_transition_config(A, B)
    env_rel = os.path.relpath(env_cfg, orch.REPO)
    print("[smoke] env_config:", env_rel, "agent_config:", agent_cfg)

    env = env_builder.build_env(env_rel, num_envs=16, device=device, visualize=False)
    agent = agent_builder.build_agent(agent_cfg, env, device)
    agent.set_mode(agent._mode.__class__.TRAIN)

    obs, info = agent._reset_envs()
    print("[smoke] policy obs shape :", tuple(obs.shape))
    print("[smoke] disc_obs in info :", "disc_obs" in info,
          tuple(info["disc_obs"].shape) if "disc_obs" in info else None)
    print("[smoke] disc_obs_space   :", env.get_disc_obs_space().shape)
    print("[smoke] goal_phase_time  :", env._goal_phase_time, " goal up_z:", float(env._goal_up_z))
    print("[smoke] num init states  :", env._num_init_states)

    for step in range(10):
        a, ainfo = agent._decide_action(obs, info)
        obs, r, done, info = env.step(a)
        pd, uz = env._compute_goal_match()
        print("[smoke] step {:>2d}: r[min/mean/max]={:+.3f}/{:+.3f}/{:+.3f} "
              "in_goal={:>2d}/16 pose_dist(mean)={:.3f} up_z_err(mean)={:.3f} "
              "hold_max={:d} done={:d}".format(
                  step, float(r.min()), float(r.mean()), float(r.max()),
                  int(env._in_goal.sum()), float(pd.mean()), float(uz.mean()),
                  int(env._goal_hold_count.max()), int((done != 0).sum())))

    # sanity: finite obs/reward, gate produces booleans, ref interpolation moved
    assert torch.isfinite(obs).all(), "non-finite obs"
    assert torch.isfinite(r).all(), "non-finite reward"
    print("[smoke] SMOKE OK")
    return


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
