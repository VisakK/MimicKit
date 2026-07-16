"""Pose-grounded evaluation of a trained transition policy A->B (spec section 5
evaluate_transition, section 12 claim 2; the spec-8.A fix for the broken v1
metric).

The v1 metric declared success if B's initiation classifier fired for >= 1 s at
ANY point in the episode. That was farmable (the classifier was also the reward)
and not a pose check: it banked transient firing from the inverted START state
and exploited classifier blind spots, reporting 0.67 / 0.93 for transitions that
visibly collapsed to upright (notes section 6). This rewrite scores the TRUE,
classifier-independent question:

    success = the character ENDS within a root-relative per-body shape-error
    threshold of B's hold pose AND held it for >= T seconds
    (optionally AND root up_z within tolerance of B's).

The shape distance is heading-removed but inversion-sensitive (so a stand-up cheat
reads as far from an inverted target), computed by env._compute_goal_match against
B's hold pose. This is exactly the probe that exposed the v1 failures; promoting it
here makes every future number trustworthy and visually faithful.

Usage:
  env_isaaclab/bin/python mimickit/skillgraph/eval_transition.py \
     --env_config data/envs/transition_scorpion_to_handstand_env.yaml \
     --model_file output/yoga_transitions/scorpion_to_handstand/model.pt \
     --from scorpion --to handstand --master_port 6975 --update_graph
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util
import skillgraph.skill_io as sio


def evaluate(args):
    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    env = env_builder.build_env(args.env_config, args.num_envs, device, visualize=False)
    # Eval ONLY the true handoff: force RSI at phase 0 (A's terminus), never the
    # training-time curriculum starts partway up the A->B path (which would
    # fake-inflate success by beginning near B).
    if (hasattr(env, "_curriculum_init_frac")):
        env._curriculum_init_frac = 0.0
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    control_freq = int(round(1.0 / env._engine.get_timestep()))
    hold_steps = int(round(args.success_hold_seconds * control_freq))
    NULL = base_env.DoneFlags.NULL.value
    FAIL = base_env.DoneFlags.FAIL.value
    N = args.num_envs

    # per-env running state
    pose_hold_count = torch.zeros(N, device=device, dtype=torch.long)  # consecutive in-pose steps
    in_pose_steps = torch.zeros(N, device=device)
    in_goal_steps = torch.zeros(N, device=device)                      # env gate (classifier+dist), diagnostic
    ep_steps = torch.zeros(N, device=device)
    ep_results = []  # (success, fail, ep_len_steps, in_pose_frac, in_goal_frac, end_pose_dist, end_up_z_err)

    obs, info = agent._reset_envs()
    print("[eval {}->{}] {} envs x {} steps; success = END within {:.2f} m of B's hold (up_z_err<{}) held >= {:.1f}s ({} steps)".format(
        args.frm, args.to, N, args.num_steps, args.eval_pose_dist,
        args.eval_up_z_tol, args.success_hold_seconds, hold_steps))
    for step in range(args.num_steps):
        with torch.no_grad():
            norm_obs = agent._obs_norm.normalize(obs)
            action = agent._a_norm.unnormalize(agent._model.eval_actor(norm_obs).mode)
        obs, r, done, info = env.step(action)

        # Pose-grounded membership against B's actual hold (classifier-independent).
        pose_dist, up_z_err = env._compute_goal_match()
        in_pose = pose_dist < args.eval_pose_dist
        if (args.eval_up_z_tol is not None):
            in_pose = in_pose & (up_z_err < args.eval_up_z_tol)
        pose_hold_count = torch.where(in_pose, pose_hold_count + 1,
                                      torch.zeros_like(pose_hold_count))
        in_pose_steps += in_pose.float()
        in_goal_steps += env._in_goal.float()
        ep_steps += 1.0

        done_ids = (done != NULL).nonzero(as_tuple=False).flatten()
        if (len(done_ids) > 0):
            for e in done_ids.tolist():
                # "ends in B's pose and held >= T s" = the final T s were in-pose.
                succ = bool(pose_hold_count[e] >= hold_steps)
                ep_results.append((
                    succ, bool(done[e].item() == FAIL), int(ep_steps[e]),
                    float(in_pose_steps[e] / max(ep_steps[e], 1)),
                    float(in_goal_steps[e] / max(ep_steps[e], 1)),
                    float(pose_dist[e]), float(up_z_err[e])))
            pose_hold_count[done_ids] = 0
            in_pose_steps[done_ids] = 0.0
            in_goal_steps[done_ids] = 0.0
            ep_steps[done_ids] = 0.0
            obs, info = agent._reset_envs(done_ids)

    if (len(ep_results) == 0):
        print("[eval] no completed episodes; raise --num_steps"); return
    n = len(ep_results)
    succ = sum(1 for r in ep_results if r[0])
    fail = sum(1 for r in ep_results if r[1])
    mean_len = sum(r[2] for r in ep_results) / n / control_freq
    mean_in_pose = sum(r[3] for r in ep_results) / n
    mean_in_goal = sum(r[4] for r in ep_results) / n
    mean_end_dist = sum(r[5] for r in ep_results) / n
    mean_end_upz_err = sum(r[6] for r in ep_results) / n
    succ_rate = succ / n

    print("\n=== transition {} -> {} (pose-grounded) ===".format(args.frm, args.to))
    print("  episodes               : {}".format(n))
    print("  SUCCESS rate           : {:.3f}  (ENDS in B's pose, held >= {:.1f}s)".format(
        succ_rate, args.success_hold_seconds))
    print("  fall rate              : {:.3f}".format(fail / n))
    print("  mean in-pose frac      : {:.3f}".format(mean_in_pose))
    print("  mean in-goal frac      : {:.3f}  (env gate: classifier AND dist)".format(mean_in_goal))
    print("  mean END shape dist    : {:.3f} m  (B's hold; lower is better)".format(mean_end_dist))
    print("  mean END up_z err      : {:.3f}".format(mean_end_upz_err))
    print("  mean episode length    : {:.2f}s".format(mean_len))
    gate = "PASS" if succ_rate >= args.success_floor else "BELOW FLOOR"
    print("  gate (>= {:.2f})          : {}".format(args.success_floor, gate))

    if (args.update_graph):
        sio.update_edge(args.frm, args.to, {
            "status": "trained" if succ_rate >= args.success_floor else "retry",
            "transition_policy_ckpt": args.model_file,
            "success_rate": round(succ_rate, 3),
            "fall_rate": round(fail / n, 3),
            "mean_in_pose_frac": round(mean_in_pose, 3),
            "mean_end_shape_dist": round(mean_end_dist, 3),
            "eval_metric": "pose_grounded",
            "n_eval_eps": n,
        }, root=args.out_root)
        print("  -> graph edge updated (status={})".format("trained" if succ_rate >= args.success_floor else "retry"))
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_config", required=True)
    p.add_argument("--model_file", required=True)
    p.add_argument("--agent_config", default="data/agents/amp_smpl_transition_hybrid_agent.yaml")
    p.add_argument("--from", dest="frm", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=500)
    p.add_argument("--success_hold_seconds", type=float, default=1.0)
    # Pose-grounded thresholds (spec 8.A). 0.12-0.18 m mean per-body shape error
    # is "on the pose"; up_z tol keeps a stand-up cheat from counting toward an
    # inverted target.
    p.add_argument("--eval_pose_dist", type=float, default=0.15)
    p.add_argument("--eval_up_z_tol", type=float, default=0.4)
    p.add_argument("--success_floor", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=6975)
    p.add_argument("--rand_seed", type=int, default=123)
    p.add_argument("--update_graph", action="store_true")
    p.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "..", "..", "skills"))
    args = p.parse_args()
    args.out_root = os.path.normpath(args.out_root)
    evaluate(args)


if __name__ == "__main__":
    main()
