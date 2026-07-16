"""Takeover oracle for the edge framework (Yoga_edge_framework_v3.md §3.2).

SUCCESS(s) := inject state s into node B's env (run-dir snapshot pairing),
run B's signed-off policy deterministically for --hold_seconds; success iff no
FAIL done fires (fall-set contact OR pose divergence — B's own trained band).

This is the operational ground truth used to (a) validity-precheck each node's
oracle (B must pass from its own clean hold states), (b) label harvested states
for the arrival-certificate calibration (tools/calibrate_gate.py), and later
(c) score edge policies offline (arrival states -> takeover pass rate).

State injection = an instance-level override of env._reset_char: the motion
clock is set to each state's stored motion_time (tar-obs / pose-termination
stay coherent — node envs run track_global_root, so states must be world-frame
consistent with the reference, which same-env harvested states are), then the
engine state is written from the blob. No repo env-code changes.

Usage (repo root, env python; ONE Isaac job at a time):
  env_isaaclab/bin/python tools/takeover_oracle.py \
      --node_dir output/yoga_nodes_v2/downdog_lt_ampft \
      --states output/yoga_nodes_v2/downdog_lt_ampft/hold_states.pt \
      --hold_seconds 5 --num_envs 256 --master_port 29611
Output: <states>.oracle.pt  {verdict [K] bool, t_fail [K] s (-1 = survived)}
"""
import argparse
import os
import sys
import types

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


def make_inject_reset(env, batch):
    """Instance override of _reset_char: motion clock from the stored
    motion_time, engine state from the stored batch. Mirrors the stock
    _reset_ref_motion + _ref_state_init flow (incl. the stale-contact and
    impulse-accumulator guards of DeepMimicEnv._reset_char)."""
    cid = env._get_char_id()
    batch = {k: v.to(env._device) for k, v in batch.items()}

    def _reset_char(self, env_ids):
        n = len(env_ids)
        t = batch["motion_time"][env_ids]
        mids = torch.zeros(n, dtype=torch.long, device=self._device)
        self._motion_ids[env_ids] = mids
        self._motion_time_offsets[env_ids] = t

        rp, rr, rv, rav, jr, dv = self._motion_lib.calc_motion_frame(mids, t)
        self._ref_root_pos[env_ids] = rp
        self._ref_root_rot[env_ids] = rr
        self._ref_root_vel[env_ids] = rv
        self._ref_root_ang_vel[env_ids] = rav
        self._ref_joint_rot[env_ids] = jr
        self._ref_dof_vel[env_ids] = dv
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot
        self._ref_dof_pos[env_ids] = self._motion_lib.joint_rot_to_dof(jr)

        e = self._engine
        for setter, key in ((e.set_root_pos, "root_pos"), (e.set_root_rot, "root_rot"),
                            (e.set_root_vel, "root_vel"), (e.set_root_ang_vel, "root_ang_vel"),
                            (e.set_dof_pos, "dof_pos"), (e.set_dof_vel, "dof_vel")):
            setter(env_ids, cid, batch[key][env_ids].to(self._device))
        e.set_body_vel(env_ids, cid, 0.0)
        e.set_body_ang_vel(env_ids, cid, 0.0)

        if hasattr(self, "_prev_contact_force_z") and self._impulse_body_ids.shape[0] > 0:
            self._prev_contact_force_z[env_ids] = 1.0e6
        if hasattr(self, "_curr_bout_vert_impulse") and self._vert_impulse_body_ids.shape[0] > 0:
            self._curr_bout_vert_impulse[env_ids] = 0.0
            self._prev_vert_impulse_in_contact[env_ids] = True
        if self._enable_contact_obs:
            self._contact_obs_stale[env_ids] = True
        return

    return types.MethodType(_reset_char, env)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", required=True, help="judging node B's run dir")
    p.add_argument("--states", required=True, help="hold_states.pt-format blob to label")
    p.add_argument("--hold_seconds", type=float, default=5.0)
    p.add_argument("--strict", action="store_true",
                   help="keep B's pose_termination in the verdict (old behavior)."
                        " Default = RECOVERY oracle: pose_termination OFF, "
                        "verdict = no fall-set contact AND the END state (last "
                        "1s mean) is back inside B's pose band — an off-band "
                        "arrival is allowed to recover instead of being "
                        "executed at injection by B's tracking radius.")
    p.add_argument("--end_pose_mean", type=float, default=None,
                   help="end-state pose band; default = gate.yaml theta_pose_mean"
                        " (recovery mode only)")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--max_states", type=int, default=6000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29611)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    node_dir = args.node_dir
    env_cfg = os.path.join(node_dir, "env_config.yaml")
    agent_cfg = os.path.join(node_dir, "agent_config.yaml")
    model = os.path.join(node_dir, "model.pt")
    out_path = args.out or (args.states + ".oracle.pt")

    blob = torch.load(args.states, map_location="cpu")
    K_all = blob["root_pos"].shape[0]
    K = min(K_all, args.max_states)
    g = torch.Generator().manual_seed(args.rand_seed)
    sel = torch.arange(K_all) if K_all <= K else torch.randperm(K_all, generator=g)[:K]
    cand = {k: blob[k][sel] for k in ["root_pos", "root_rot", "root_vel",
                                      "root_ang_vel", "dof_pos", "dof_vel",
                                      "motion_time"]}

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    hold_ref = None
    theta = args.end_pose_mean
    up_z_tol = 0.15
    if not args.strict:
        # RECOVERY oracle: disable pose_termination via a derived env config.
        import yaml
        with open(env_cfg) as fh:
            cfg = yaml.safe_load(fh)
        cfg["env"]["pose_termination"] = False
        env_cfg_derived = out_path + ".oracle_env.yaml"
        with open(env_cfg_derived, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        env_cfg = env_cfg_derived
        b_blob = torch.load(os.path.join(node_dir, "hold_states.pt"), map_location="cpu")
        hold_ref = b_blob["hold_ref"]
        if theta is None:
            gate_path = os.path.join(node_dir, "gate.yaml")
            if os.path.exists(gate_path):
                g = yaml.safe_load(open(gate_path))
                theta = float(g["theta_pose_mean"])
                up_z_tol = max(up_z_tol, 2.0 * float(g["up_z_tol"]))
            else:
                theta = 0.25

    env = env_builder.build_env(env_cfg, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(agent_cfg, env, device)
    agent.load(model)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    control_freq = int(round(1.0 / env._engine.get_timestep()))
    hold_steps = int(round(args.hold_seconds * control_freq))
    N = args.num_envs
    NULL = base_env.DoneFlags.NULL.value
    FAIL = base_env.DoneFlags.FAIL.value

    verdict = torch.zeros(K, dtype=torch.bool)
    t_fail = torch.full((K,), -1.0)

    n_batches = (K + N - 1) // N
    print(f"[oracle:{os.path.basename(node_dir)}] labeling {K} states "
          f"({n_batches} batches x {N} envs x {hold_steps} steps)", flush=True)

    # recovery-mode end-state machinery
    cid = env._get_char_id()
    if hold_ref is not None:
        import envs.char_env as char_env
        ref_bp = hold_ref["body_pos"].to(device)
        ref_rr = hold_ref["root_rot"].to(device)
        hold_up = float(hold_ref["up_z"][0])
        ref_rel = ref_bp[:, 1:, :] - ref_bp[:, 0:1, :]
        ref_local = char_env.convert_to_local_body_pos(ref_rr, ref_rel)

        def end_pose(n_):
            bp = env._engine.get_body_pos(cid); rr = env._engine.get_root_rot(cid)
            cur = char_env.convert_to_local_body_pos(rr, bp[:, 1:, :] - bp[:, 0:1, :])
            pd = torch.linalg.vector_norm(cur - ref_local, dim=-1).mean(dim=-1)
            up_l = torch.zeros(n_, 3, device=device); up_l[:, 2] = 1.0
            upz = torch_util.quat_rotate(rr, up_l)[:, 2]
            return pd, upz

    for b in range(n_batches):
        lo, hi = b * N, min((b + 1) * N, K)
        nb = hi - lo
        # env-slot i judges candidate lo+i; surplus slots replay slot 0.
        pad = torch.arange(N) % nb
        batch = {k: cand[k][lo + pad] for k in cand}
        env._reset_char = make_inject_reset(env, batch)

        obs, info = agent._reset_envs()
        failed = torch.zeros(N, dtype=torch.bool, device=device)
        fail_t = torch.full((N,), -1.0, device=device)
        end_pd = torch.zeros(N, device=device); end_upz = torch.zeros(N, device=device)
        end_n = 0
        for step in range(hold_steps):
            with torch.no_grad():
                norm_obs = agent._obs_norm.normalize(obs)
                a_norm = agent._model.eval_actor(norm_obs).mode
                action = agent._a_norm.unnormalize(a_norm)
            obs, r, done, info = env.step(action)
            new_fail = (done == FAIL) & ~failed
            fail_t[new_fail] = (step + 1) / control_freq
            failed |= new_fail
            if hold_ref is not None and step >= hold_steps - control_freq:
                pd_now, upz_now = end_pose(N)
                end_pd += pd_now; end_upz += upz_now; end_n += 1
            # any premature TIME/SUCC done would re-reset the env into a fresh
            # injected state — with 12 s episodes and 5 s holds it can't fire,
            # but freeze bookkeeping if it ever does.
            if bool((done == FAIL).all()):
                break
        ok = ~failed
        if hold_ref is not None and end_n > 0:
            end_pd /= end_n; end_upz /= end_n
            ok &= (end_pd < theta) & ((end_upz - hold_up).abs() < up_z_tol)
        verdict[lo:hi] = ok[:nb].cpu()
        t_fail[lo:hi] = fail_t[:nb].cpu()
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"  batch {b+1}/{n_batches}: cumulative pass "
                  f"{float(verdict[:hi].float().mean()):.3f}", flush=True)

    out = {"verdict": verdict, "t_fail": t_fail, "sel": sel,
           "hold_seconds": args.hold_seconds, "node_dir": node_dir,
           "states_file": args.states}
    torch.save(out, out_path)

    # summary splits (if the blob carries harvest labels)
    msg = f"[oracle:{os.path.basename(node_dir)}] overall pass {float(verdict.float().mean()):.3f}"
    if "hold_label" in blob:
        lab = blob["hold_label"][sel]
        sea = blob["seasoned"][sel] if "seasoned" in blob else torch.ones_like(lab, dtype=torch.bool)
        clean = (lab == 1.0) & sea
        if "noise" in blob:
            clean &= blob["noise"][sel] <= 0.02
        if clean.any():
            msg += f" | VALIDITY (clean held states): {float(verdict[clean].float().mean()):.3f} ({int(clean.sum())} states)"
        fail_lab = lab == 0.0
        if fail_lab.any():
            msg += f" | harvest-negatives pass rate: {float(verdict[fail_lab].float().mean()):.3f}"
    print(msg, flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
