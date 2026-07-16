"""Collect a trained skill's terminal states + initiation dataset + skill card.

This is the data-generation step behind spec sections 4, 6, and 8. It builds the
skill's env+agent exactly as run.py does, loads the policy, and rolls it with a
little action noise (to broaden the visited distribution and induce the falls we
need as negatives). For every visited state it records the full character state,
the policy observation, the raw critic value, the canonical balance features, the
clip phase, and the episode termination type. Then:

  * INITIATION DATASET (classifier training, section 4): each state is labeled
    positive iff the policy holds for >= `hold_seconds` from it without a FAIL
    (look-ahead within the episode), negative otherwise. TIME/SUCC endings are
    not failures. -> {features, obs, critic_value, label}.
  * TERMINAL STATES (RSI for transitions INTO/OUT of this skill, section 8):
    the quasi-static, contact-signature-matching, positively-labeled held states
    -> full kinematic state for set_init_state_distribution.
  * CONTACT SIGNATURE + hold stats -> the skill card.

Usage (run from repo root, like run.py):
  env_isaaclab/bin/python mimickit/skillgraph/collect_skill.py \
     --skill_id handstand \
     --env_config data/envs/deepmimic_smpl_handstand_orient_env.yaml \
     --agent_config data/agents/deepmimic_smpl_ppo_agent.yaml \
     --model_file output/yoga_orient/model_seed42.pt \
     --num_envs 256 --num_steps 400 --master_port 6900
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util

import skillgraph.state_features as sf
import skillgraph.skill_io as sio


def _get_state(env):
    """Full kinematic state + contact forces of the live character, all [N,*]."""
    cid = env._get_char_id()
    e = env._engine
    return {
        "root_pos": e.get_root_pos(cid).clone(),
        "root_rot": e.get_root_rot(cid).clone(),
        "root_vel": e.get_root_vel(cid).clone(),
        "root_ang_vel": e.get_root_ang_vel(cid).clone(),
        "dof_pos": e.get_dof_pos(cid).clone(),
        "dof_vel": e.get_dof_vel(cid).clone(),
        "body_pos": e.get_body_pos(cid).clone(),
        "contact_forces": e.get_ground_contact_forces(cid).clone(),
    }


def collect(args):
    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    env = env_builder.build_env(args.env_config, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    kcm = env._kin_char_model
    body_names = kcm.get_body_names()
    contact_ids = env._obs_contact_body_ids
    key_ids = env._key_body_ids
    com_w = env._com_body_weights
    contact_names = [body_names[i] for i in contact_ids.tolist()]
    key_names = [body_names[i] for i in key_ids.tolist()]
    support_dirs = getattr(env, "_support_polygon_dirs", None)
    fthresh = getattr(env, "_obs_contact_force_threshold", 1.0)
    # get_timestep() is the control timestep (1/control_freq); same value the
    # env uses to advance tar_obs steps.
    control_freq = int(round(1.0 / env._engine.get_timestep()))
    hold_steps = int(round(args.hold_seconds * control_freq))

    N = args.num_envs
    NULL = base_env.DoneFlags.NULL.value
    FAIL = base_env.DoneFlags.FAIL.value

    # per-(step) records, kept on CPU
    rec_feat, rec_obs, rec_val, rec_done = [], [], [], []
    rec_env, rec_time, rec_steps_since_reset = [], [], []
    rec_state = {k: [] for k in ["root_pos", "root_rot", "root_vel", "root_ang_vel",
                                 "dof_pos", "dof_vel", "contact_forces", "body_pos"]}
    rec_upz = []
    rec_noise = []

    obs, info = agent._reset_envs()
    steps_since_reset = torch.zeros(N, dtype=torch.long, device=device)

    # Per-env action-noise SPECTRUM (resampled per episode): a stable policy
    # rarely falls at noise 0 (no negatives) and always falls at large noise (no
    # positives); neither trains a classifier. Spreading noise across envs makes
    # low-noise envs hold (positive / clean terminal states) and high-noise envs
    # wobble and fall (negative), spanning B's competence boundary in one rollout.
    def _sample_noise(n):
        return args.action_noise_min + (args.action_noise_max - args.action_noise_min) \
               * torch.rand(n, device=device)
    noise_scale = _sample_noise(N)

    print("[collect:{}] rolling {} envs x {} steps (hold_seconds={} -> {} steps), "
          "noise in [{:.2f},{:.2f}]".format(args.skill_id, N, args.num_steps,
          args.hold_seconds, hold_steps, args.action_noise_min, args.action_noise_max))
    for step in range(args.num_steps):
        st = _get_state(env)
        feats = sf.compute_state_features(
            root_pos=st["root_pos"], root_rot=st["root_rot"], root_vel=st["root_vel"],
            root_ang_vel=st["root_ang_vel"], body_pos=st["body_pos"],
            contact_forces=st["contact_forces"], contact_body_ids=contact_ids,
            key_body_ids=key_ids, com_weights=com_w, force_threshold=fthresh,
            support_dirs=support_dirs)
        # raw critic value on the real policy obs (section-4 baseline predictor)
        with torch.no_grad():
            norm_obs = agent._obs_norm.normalize(obs)
            val = agent._model.eval_critic(norm_obs).squeeze(-1)
            a_dist = agent._model.eval_actor(norm_obs)
            a_norm = a_dist.mode + noise_scale.unsqueeze(-1) * torch.randn_like(a_dist.mode)
            action = agent._a_norm.unnormalize(a_norm)

        mtime = env._get_motion_times()
        up_local = torch.zeros_like(st["root_pos"]); up_local[..., 2] = 1.0
        import util.torch_util as tu
        upz = tu.quat_rotate(st["root_rot"], up_local)[..., 2]

        next_obs, r, done, info = env.step(action)

        rec_feat.append(feats.cpu()); rec_obs.append(obs.cpu())
        rec_val.append(val.cpu()); rec_done.append(done.cpu().clone())
        rec_env.append(torch.arange(N)); rec_time.append(mtime.cpu())
        rec_steps_since_reset.append(steps_since_reset.cpu().clone())
        rec_upz.append(upz.cpu())
        rec_noise.append(noise_scale.cpu().clone())
        for k in rec_state:
            rec_state[k].append(st[k].cpu())

        # reset finished envs
        done_ids = (done != NULL).nonzero(as_tuple=False).flatten()
        steps_since_reset += 1
        if (len(done_ids) > 0):
            steps_since_reset[done_ids] = 0
            noise_scale[done_ids] = _sample_noise(len(done_ids))
            # env.reset returns the full-batch obs buffer with the reset envs
            # recomputed and the survivors left at their post-step obs.
            obs, info = agent._reset_envs(done_ids)
        else:
            obs = next_obs

    T = len(rec_feat)
    feat = torch.stack(rec_feat)        # [T,N,D]
    obsb = torch.stack(rec_obs)         # [T,N,O]
    valb = torch.stack(rec_val)         # [T,N]
    doneb = torch.stack(rec_done)       # [T,N]
    timeb = torch.stack(rec_time)       # [T,N]
    ssr = torch.stack(rec_steps_since_reset)  # [T,N]
    upzb = torch.stack(rec_upz)         # [T,N]

    # ---- label each (t, env) by hold_seconds look-ahead within its episode ----
    # negative iff a FAIL occurs within hold_steps AFTER t in the same episode.
    label = torch.ones(T, N, dtype=torch.float32)
    for n in range(N):
        # episode boundaries: steps where done!=NULL end an episode
        ends = (doneb[:, n] != NULL).nonzero(as_tuple=False).flatten().tolist()
        start = 0
        for end in ends:
            is_fail = (doneb[end, n].item() == FAIL)
            if (is_fail):
                lo = max(start, end - hold_steps + 1)
                label[lo:end + 1, n] = 0.0
            start = end + 1
        # trailing partial episode (no terminal yet): leave as positive (held so
        # far); states within hold_steps of the rollout end are ambiguous -> drop
        if (start < T):
            label[T - hold_steps + 1:T, n] = -1.0  # sentinel: drop (unknown)

    valid = (ssr > 0) & (label >= 0.0)   # drop stale post-reset + ambiguous tail
    vmask = valid.reshape(-1)
    feat_flat = feat.reshape(T * N, -1)[vmask]
    obs_flat = obsb.reshape(T * N, -1)[vmask]
    val_flat = valb.reshape(-1)[vmask]
    label_flat = label.reshape(-1)[vmask]
    time_flat = timeb.reshape(-1)[vmask]
    upz_flat = upzb.reshape(-1)[vmask]

    pos_frac = float((label_flat == 1).float().mean())
    print("[collect:{}] {} valid states, pos_frac={:.3f}, n_fail_neg={}".format(
        args.skill_id, int(vmask.sum()), pos_frac, int((label_flat == 0).sum())))

    # ---- canonical held POSE = quasi-static held states in the pose's
    # inversion band. With full-clip RSI, "held >= hold_seconds" also includes
    # the upright kick-up / exit phases (up_z ~ 0), which would pollute the
    # contact signature and the terminal-state set used for handoff RSI. The
    # up_z gate (from tools/analyze_pose_clips.py: handstand/scorpion inverted,
    # crow upright) isolates the actual pose. Default gate is wide-open.
    contact_flag_cols = feat_flat[:, 8:8 + len(contact_names)]
    held = label_flat == 1
    root_lin_speed = feat_flat[:, 4]
    quiet = root_lin_speed < args.terminal_speed
    upz_gate = torch.ones_like(held)
    if (args.hold_up_z_max is not None):
        upz_gate &= upz_flat <= args.hold_up_z_max
    if (args.hold_up_z_min is not None):
        upz_gate &= upz_flat >= args.hold_up_z_min
    pose_mask = held & quiet & upz_gate
    if (pose_mask.sum() < 50):   # gate too tight -> fall back to all quiet holds
        print("[collect:{}] WARN up_z gate left {} pose states; ignoring gate".format(
            args.skill_id, int(pose_mask.sum())))
        pose_mask = held & quiet

    contact_frac = contact_flag_cols[pose_mask].mean(dim=0)
    ground_contacts = [contact_names[i] for i in range(len(contact_names))
                       if contact_frac[i].item() > 0.5]
    pose_up_z = upz_flat[pose_mask].mean().item()
    inverted = bool(pose_up_z < -0.5)
    print("[collect:{}] pose signature: contacts={} inverted={} (pose up_z={:.2f}, "
          "{} pose states)".format(args.skill_id, ground_contacts, inverted,
          pose_up_z, int(pose_mask.sum())))

    # ---- terminal states: signature-matching canonical-pose states ------------
    sig_ids = [contact_names.index(b) for b in ground_contacts]
    sig_ok = torch.ones(feat_flat.shape[0], dtype=torch.bool)
    for i in sig_ids:
        sig_ok &= contact_flag_cols[:, i] > 0.5
    term_mask = pose_mask & sig_ok
    term_idx_full = valid.reshape(-1).nonzero(as_tuple=False).flatten()[term_mask.nonzero(as_tuple=False).flatten()]
    n_term_avail = int(term_mask.sum())
    sel = term_idx_full
    if (n_term_avail > args.max_terminal):
        perm = torch.randperm(n_term_avail)[:args.max_terminal]
        sel = term_idx_full[perm]
    # gather full state for the selected flat indices
    state_T_N = {k: torch.stack(rec_state[k]) for k in rec_state}  # each [T,N,*]
    def gather_state(key):
        flat = state_T_N[key].reshape(T * N, *state_T_N[key].shape[2:])
        return flat[sel]
    terminal = {k: gather_state(k) for k in rec_state}
    terminal["features"] = feat.reshape(T * N, -1)[sel]
    terminal["motion_time"] = timeb.reshape(-1)[sel]
    print("[collect:{}] terminal states: {} available, saved {}".format(
        args.skill_id, n_term_avail, terminal["root_pos"].shape[0]))

    # ---- write artifacts ------------------------------------------------------
    sdir = sio.ensure_skill_dir(args.skill_id, args.out_root)
    dataset = {
        "features": feat_flat, "obs": obs_flat, "critic_value": val_flat,
        "label": label_flat, "motion_time": time_flat, "up_z": upz_flat,
        "feature_names": sf.feature_names(contact_names, key_names),
        "contact_names": contact_names, "key_names": key_names,
    }
    torch.save(dataset, os.path.join(sdir, "dataset.pt"))
    torch.save(terminal, os.path.join(sdir, "terminal_states.pt"))

    # CLEAN-policy stability: the per-env noise spectrum confounds an aggregate
    # fall_rate (high-noise envs always fall). Report stability over only the
    # near-zero-noise episodes (bottom of the spectrum) so the card reflects the
    # actual deterministic policy, not the perturbed exploration distribution.
    noiseb = torch.stack(rec_noise)   # [T,N], constant within an episode
    clean_thresh = 0.1 * args.action_noise_max + 1e-6
    n_eps = 0; n_fail = 0; held_lens = []                 # clean-only
    n_eps_all = 0
    for n in range(N):
        ends = (doneb[:, n] != NULL).nonzero(as_tuple=False).flatten().tolist()
        start = 0
        for end in ends:
            n_eps_all += 1
            ep_noise = noiseb[start, n].item()
            if (ep_noise <= clean_thresh):
                n_eps += 1
                if (doneb[end, n].item() == FAIL):
                    n_fail += 1
                held_lens.append((end - start + 1) / control_freq)
            start = end + 1
    card = sio.read_card(args.skill_id, args.out_root) or {}
    card.update({
        "skill_id": args.skill_id,
        "algo": "deepmimic",
        "env_config": args.env_config,
        "agent_config": args.agent_config,
        "policy_ckpt": args.model_file,
        "contact_signature": {"ground_contacts": ground_contacts, "inverted": inverted,
                              "pose_up_z": round(pose_up_z, 3)},
        "value_stats": {"hold_phase_time": float(time_flat[pose_mask].median().item()) if pose_mask.any() else None},
        "competence_region": {
            "classifier_ckpt": os.path.join("skills", args.skill_id, "classifier.pt"),
            "threshold": 0.5, "feature_names": dataset["feature_names"]},
        "terminal_state_distribution": {
            "states_file": os.path.join("skills", args.skill_id, "terminal_states.pt"),
            "count": int(terminal["root_pos"].shape[0]),
            "state_hold_prob": round(pos_frac, 3),
            "clean_held_seconds_mean": round(float(np.mean(held_lens)), 2) if held_lens else None,
            "clean_fall_rate": round(float(n_fail / n_eps), 3) if n_eps > 0 else None,
            "clean_eps": n_eps,
            "note": "clean_* over near-zero-noise episodes; state_hold_prob over the noise spectrum"},
    })
    sio.write_card(card, args.out_root)
    print("[collect:{}] wrote card + dataset ({} states) + terminal_states ({}). "
          "clean_eps={} clean_fall_rate={}".format(args.skill_id, int(vmask.sum()),
          terminal["root_pos"].shape[0], n_eps,
          round(n_fail / n_eps, 3) if n_eps > 0 else "n/a"))
    return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill_id", required=True)
    p.add_argument("--env_config", required=True)
    p.add_argument("--agent_config", required=True)
    p.add_argument("--model_file", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--action_noise_min", type=float, default=0.0)
    p.add_argument("--action_noise_max", type=float, default=0.2)
    p.add_argument("--hold_seconds", type=float, default=3.0)
    p.add_argument("--terminal_speed", type=float, default=0.4)
    p.add_argument("--max_terminal", type=int, default=2000)
    # Pose inversion gate (from tools/analyze_pose_clips.py): isolates the held
    # pose from the upright kick-up/exit phases that full-clip RSI also visits.
    # Inverted poses: --hold_up_z_max -0.5 ; upright poses: --hold_up_z_min 0.0
    p.add_argument("--hold_up_z_max", type=float, default=None)
    p.add_argument("--hold_up_z_min", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=6900)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "..", "..", "skills"))
    args = p.parse_args()
    args.out_root = os.path.normpath(args.out_root)
    collect(args)
    return


if __name__ == "__main__":
    main()
