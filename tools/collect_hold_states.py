"""Harvest hold states from a v3 node policy (edge framework v3, doc §4/§7).

Rolls a signed-off node policy from its RUN-DIR SNAPSHOT configs (env_config.yaml
+ agent_config.yaml + model.pt — the v3-native pairing, never `skills/`) with a
per-env action-noise spectrum, and records for every visited state the full
dynamic character state + balance diagnostics + a deterministic hold label
(no FAIL within ±hold_window seconds inside the episode — look-BEHIND and
look-ahead, unlike collect_skill's ahead-only).

Consumers: tools/takeover_oracle.py (oracle labeling / validity precheck),
tools/calibrate_gate.py (arrival-certificate threshold fit), and later the
edge envs' A-state RSI tier (hold_states are the harvested init distribution).

Usage (repo root, env python; ONE Isaac job at a time):
  env_isaaclab/bin/python tools/collect_hold_states.py \
      --node_dir output/yoga_nodes_v2/downdog_lt_ampft \
      --num_envs 256 --num_steps 400 --master_port 29610
Output: <node_dir>/hold_states.pt
"""
import argparse
import os
import sys

import numpy as np
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

sys.path.insert(0, os.path.join(REPO, "mimickit", "skillgraph"))
import state_features as sf


STATE_KEYS = ["root_pos", "root_rot", "root_vel", "root_ang_vel",
              "dof_pos", "dof_vel", "body_pos", "body_vel", "contact_forces"]


def get_state(env):
    cid = env._get_char_id()
    e = env._engine
    st = {
        "root_pos": e.get_root_pos(cid).clone(),
        "root_rot": e.get_root_rot(cid).clone(),
        "root_vel": e.get_root_vel(cid).clone(),
        "root_ang_vel": e.get_root_ang_vel(cid).clone(),
        "dof_pos": e.get_dof_pos(cid).clone(),
        "dof_vel": e.get_dof_vel(cid).clone(),
        "body_pos": e.get_body_pos(cid).clone(),
        "body_vel": e.get_body_vel(cid).clone(),
        "contact_forces": e.get_ground_contact_forces(cid).clone(),
    }
    return st


def pose_dist_to(body_pos, root_rot, ref_body_pos, ref_root_rot):
    """Heading-removed root-relative per-body distance (the _compute_goal_match
    metric): returns (mean [N], max [N])."""
    import envs.char_env as char_env
    cur_rel = body_pos[:, 1:, :] - body_pos[:, 0:1, :]
    cur_local = char_env.convert_to_local_body_pos(root_rot, cur_rel)
    ref_rel = ref_body_pos[:, 1:, :] - ref_body_pos[:, 0:1, :]
    ref_local = char_env.convert_to_local_body_pos(ref_root_rot, ref_rel)
    d = torch.linalg.vector_norm(cur_local - ref_local, dim=-1)   # [N,B-1]
    return d.mean(dim=-1), d.max(dim=-1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=400)
    p.add_argument("--action_noise_min", type=float, default=0.0)
    p.add_argument("--action_noise_max", type=float, default=0.2)
    p.add_argument("--hold_window", type=float, default=2.0,
                   help="s; hold label = no FAIL within +/- this window")
    p.add_argument("--max_states", type=int, default=16000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29610)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    node_dir = args.node_dir
    env_cfg = os.path.join(node_dir, "env_config.yaml")
    agent_cfg = os.path.join(node_dir, "agent_config.yaml")
    model = os.path.join(node_dir, "model.pt")
    for f in (env_cfg, agent_cfg, model):
        assert os.path.exists(f), f"missing snapshot artifact: {f}"
    out_path = args.out or os.path.join(node_dir, "hold_states.pt")

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    env = env_builder.build_env(env_cfg, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(agent_cfg, env, device)
    agent.load(model)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    kcm = env._kin_char_model
    body_names = kcm.get_body_names()
    contact_ids = env._obs_contact_body_ids
    key_ids = env._key_body_ids
    com_w = env._com_body_weights
    fthresh = getattr(env, "_obs_contact_force_threshold", 1.0)
    support_dirs = getattr(env, "_support_polygon_dirs", None)
    control_freq = int(round(1.0 / env._engine.get_timestep()))
    win_steps = int(round(args.hold_window * control_freq))
    N = args.num_envs
    NULL = base_env.DoneFlags.NULL.value
    FAIL = base_env.DoneFlags.FAIL.value

    # hold reference frame at the middle of the RSI window (ping-pong hold loop
    # -> any in-window phase is the held pose). Stored so downstream tools can
    # compute pose_dist-to-hold for ANY state without reloading the motion lib.
    motion_len = float(env._motion_lib._motion_lengths[0].item())
    init_range = getattr(env, "_init_time_range", None)
    if init_range is not None:
        t_lo, t_hi = float(init_range[0]), min(float(init_range[1]), motion_len)
    else:
        t_lo, t_hi = 0.0, motion_len
    t_hold = torch.tensor([(t_lo + t_hi) / 2.0], device=device)
    mid0 = torch.zeros(1, dtype=torch.long, device=device)
    h_rp, h_rr, h_rv, h_rav, h_jr, h_dv = env._motion_lib.calc_motion_frame(mid0, t_hold)
    h_bp, h_br = kcm.forward_kinematics(h_rp, h_rr, h_jr)
    up_l = torch.zeros_like(h_rp); up_l[..., 2] = 1.0
    hold_ref = {"root_pos": h_rp.cpu(), "root_rot": h_rr.cpu(),
                "body_pos": h_bp.cpu(),
                "up_z": torch_util.quat_rotate(h_rr, up_l)[..., 2].cpu(),
                "motion_time": t_hold.cpu()}

    def sample_noise(n):
        return args.action_noise_min + (args.action_noise_max - args.action_noise_min) \
               * torch.rand(n, device=device)

    rec = {k: [] for k in STATE_KEYS}
    rec_meta = {k: [] for k in ["features", "motion_time", "up_z", "noise",
                                "done", "ssr", "com_vel", "dof_speed",
                                "pose_dist_mean", "pose_dist_max"]}

    obs, info = agent._reset_envs()
    ssr = torch.zeros(N, dtype=torch.long, device=device)
    noise = sample_noise(N)

    print(f"[harvest:{os.path.basename(node_dir)}] {N} envs x {args.num_steps} steps, "
          f"noise [{args.action_noise_min},{args.action_noise_max}], "
          f"hold window ±{args.hold_window}s ({win_steps} steps)", flush=True)

    hr_bp = h_bp.expand(N, -1, -1)
    hr_rr = h_rr.expand(N, -1)
    for step in range(args.num_steps):
        st = get_state(env)
        feats = sf.compute_state_features(
            root_pos=st["root_pos"], root_rot=st["root_rot"], root_vel=st["root_vel"],
            root_ang_vel=st["root_ang_vel"], body_pos=st["body_pos"],
            contact_forces=st["contact_forces"], contact_body_ids=contact_ids,
            key_body_ids=key_ids, com_weights=com_w, force_threshold=fthresh,
            support_dirs=support_dirs)
        com_vel = torch.einsum("nbk,b->nk", st["body_vel"], com_w)
        dof_speed = st["dof_vel"].abs().mean(dim=-1)
        up_local = torch.zeros_like(st["root_pos"]); up_local[..., 2] = 1.0
        upz = torch_util.quat_rotate(st["root_rot"], up_local)[..., 2]
        pd_mean, pd_max = pose_dist_to(st["body_pos"], st["root_rot"], hr_bp, hr_rr)
        mtime = env._get_motion_times()

        with torch.no_grad():
            norm_obs = agent._obs_norm.normalize(obs)
            a_dist = agent._model.eval_actor(norm_obs)
            a_norm = a_dist.mode + noise.unsqueeze(-1) * torch.randn_like(a_dist.mode)
            action = agent._a_norm.unnormalize(a_norm)

        next_obs, r, done, info = env.step(action)

        for k in STATE_KEYS:
            rec[k].append(st[k].cpu())
        rec_meta["features"].append(feats.cpu())
        rec_meta["motion_time"].append(mtime.cpu())
        rec_meta["up_z"].append(upz.cpu())
        rec_meta["noise"].append(noise.cpu().clone())
        rec_meta["done"].append(done.cpu().clone())
        rec_meta["ssr"].append(ssr.cpu().clone())
        rec_meta["com_vel"].append(com_vel.cpu())
        rec_meta["dof_speed"].append(dof_speed.cpu())
        rec_meta["pose_dist_mean"].append(pd_mean.cpu())
        rec_meta["pose_dist_max"].append(pd_max.cpu())

        done_ids = (done != NULL).nonzero(as_tuple=False).flatten()
        ssr += 1
        if len(done_ids) > 0:
            ssr[done_ids] = 0
            noise[done_ids] = sample_noise(len(done_ids))
            obs, info = agent._reset_envs(done_ids)
        else:
            obs = next_obs

    T = args.num_steps
    doneb = torch.stack(rec_meta["done"])          # [T,N]
    ssrb = torch.stack(rec_meta["ssr"])            # [T,N]

    # hold label: within the episode, no FAIL within win_steps BEFORE or AFTER.
    label = torch.ones(T, N, dtype=torch.float32)
    for n in range(N):
        ends = (doneb[:, n] != NULL).nonzero(as_tuple=False).flatten().tolist()
        start = 0
        for end in ends:
            if doneb[end, n].item() == FAIL:
                lo = max(start, end - win_steps + 1)
                label[lo:end + 1, n] = 0.0
            start = end + 1
        if start < T:   # trailing episode: tail is ambiguous
            label[max(start, T - win_steps + 1):T, n] = -1.0
    # look-behind: also require the state to be >= win_steps into its episode
    seasoned = ssrb >= win_steps
    valid = (ssrb > 0) & (label >= 0.0)

    flat = lambda x: x.reshape(T * N, *x.shape[2:])
    vmask = valid.reshape(-1)
    idx_all = vmask.nonzero(as_tuple=False).flatten()

    # stratified subsample: keep held+seasoned (positives) and everything near
    # the boundary (fails + unseasoned holds) up to max_states, spread evenly.
    lab_f = label.reshape(-1)[idx_all]
    sea_f = seasoned.reshape(-1)[idx_all]
    pos_sel = ((lab_f == 1.0) & sea_f).nonzero(as_tuple=False).flatten()
    neg_sel = (lab_f == 0.0).nonzero(as_tuple=False).flatten()
    oth_sel = ((lab_f == 1.0) & ~sea_f).nonzero(as_tuple=False).flatten()
    g = torch.Generator().manual_seed(args.rand_seed)
    def take(sel, k):
        if len(sel) <= k:
            return sel
        return sel[torch.randperm(len(sel), generator=g)[:k]]
    n_pos = min(len(pos_sel), args.max_states // 2)
    n_neg = min(len(neg_sel), args.max_states // 4)
    n_oth = min(len(oth_sel), args.max_states - n_pos - n_neg)
    keep_local = torch.cat([take(pos_sel, n_pos), take(neg_sel, n_neg), take(oth_sel, n_oth)])
    keep = idx_all[keep_local]

    blob = {k: flat(torch.stack(rec[k]))[keep] for k in STATE_KEYS}
    for k in ["features", "motion_time", "up_z", "noise", "com_vel",
              "dof_speed", "pose_dist_mean", "pose_dist_max"]:
        blob[k] = flat(torch.stack(rec_meta[k]))[keep]
    blob["hold_label"] = lab_f[keep_local]
    blob["seasoned"] = sea_f[keep_local]
    blob["hold_ref"] = hold_ref
    blob["meta"] = {
        "node_dir": node_dir, "env_config": env_cfg, "agent_config": agent_cfg,
        "model": model, "body_names": body_names,
        "contact_body_names": [body_names[i] for i in contact_ids.tolist()],
        "contact_body_ids": contact_ids.cpu(),
        "com_weights": com_w.cpu(), "char_weight": float(env._char_weight),
        "control_freq": control_freq, "force_threshold": float(fthresh),
        "noise_range": [args.action_noise_min, args.action_noise_max],
        "hold_window_s": args.hold_window,
        "feature_names": sf.feature_names(
            [body_names[i] for i in contact_ids.tolist()],
            [body_names[i] for i in key_ids.tolist()]),
    }
    torch.save(blob, out_path)

    n_kept = len(keep)
    n_pos_k = int((blob["hold_label"] == 1.0).sum())
    print(f"[harvest:{os.path.basename(node_dir)}] kept {n_kept} states "
          f"({n_pos_k} held+seasoned positives, {int((blob['hold_label']==0).sum())} "
          f"fail-window negatives) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
