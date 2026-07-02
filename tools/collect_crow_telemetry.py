"""Roll out a trained policy for one episode and record full physical telemetry
to an .npz, for offline visualization (no Isaac needed for the plots).

Records, per control step:
  * body_pos [B,3], body_rot [B,4]       (env-local world frame)
  * net_force [B,3]                        (all contacts per body)
  * ground_force [B,3]                     (ground-only per body)
  * (net - ground) -> body-to-body force is derived by the plotter
  * contact_pt [B,3]                       (avg ground contact point per body;
                                            NaN where not in contact) -- needs
                                            enable_contact_points (injected here)
  * dof_pos [D], dof_vel [D], dof_torque [D]

Default target = the solved crow (ft4_lowlr). Override with --env/--agent/--model.

Run:
  env_isaaclab/bin/python tools/collect_crow_telemetry.py
  # then: env_isaaclab/bin/python tools/plot_contact_forces.py   (etc.)
"""
import sys, os, argparse, tempfile
sys.path.insert(0, "mimickit")
import numpy as np
import yaml
import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="data/envs/amp_smpl_crow_hybrid_ft4_env.yaml")
    p.add_argument("--agent", default="data/agents/amp_smpl_transition_lowlr_agent.yaml")
    p.add_argument("--model", default="output/model_yoga_amp_crow_hybrid_ft4_lowlr.pt")
    p.add_argument("--out", default="output/crow_telemetry.npz")
    p.add_argument("--steps", type=int, default=200, help="max control steps to record")
    p.add_argument("--warmup", type=int, default=15, help="settle steps before recording")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    # Inject enable_contact_points into the engine block via a temp config so the
    # ground contact sensors also report the per-body avg contact point. Training
    # configs are left untouched (the flag defaults OFF).
    cfg = env_builder.load_env_file(args.env)
    cfg.setdefault("engine", {})["enable_contact_points"] = True
    tmp = tempfile.NamedTemporaryFile("w", suffix="_analysis_env.yaml", delete=False)
    yaml.safe_dump(cfg, tmp); tmp.close()

    mp_util.init(0, 1, args.device, 6993)
    env = env_builder.build_env(tmp.name, 1, args.device, visualize=False)
    agent = agent_builder.build_agent(args.agent, env, args.device)
    agent.load(args.model)
    agent.eval(); agent.set_mode(agent._mode.__class__.TEST)

    kcm = env._kin_char_model
    body_names = list(kcm.get_body_names())
    B = len(body_names)
    dof_size = int(kcm.get_dof_size())
    cid = env._get_char_id()
    e = env._engine

    # DoF labels: SMPL non-root bodies each carry 3 spherical DoFs (x/y/z).
    if (dof_size == 3 * (B - 1)):
        axes = ["x", "y", "z"]
        dof_names = ["{}_{}".format(body_names[1 + d // 3], axes[d % 3]) for d in range(dof_size)]
    else:
        dof_names = ["dof_{}".format(d) for d in range(dof_size)]

    char_weight = float(getattr(env, "_char_weight", np.nan))
    try:
        env_off = e._env_offsets[0].detach().cpu().numpy()  # [2]
    except Exception:
        env_off = np.zeros(2, dtype=np.float32)

    def deterministic_action(obs):
        with torch.no_grad():
            return agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)

    obs, info = agent._reset_envs()
    for _ in range(args.warmup):
        a = deterministic_action(obs)
        obs, r, done, info = env.step(a)
        obs, info = agent._reset_done_envs(done)

    rec = {k: [] for k in ["body_pos", "body_rot", "net_force", "ground_force",
                            "contact_pt", "dof_pos", "dof_vel", "dof_torque"]}

    def snap(t):
        bp = e.get_body_pos(cid)[0]                         # [B,3] env-local
        br = e.get_body_rot(cid)[0]                         # [B,4] xyzw
        net = e.get_contact_forces(cid)[0]                  # [B,3] all contacts
        grd = e.get_ground_contact_forces(cid)[0]          # [B,3] ground only
        cpts = e.get_ground_contact_points(cid)            # [N,B,3] world or None
        if (cpts is None):
            cp = torch.full((B, 3), float("nan"), device=bp.device)
        else:
            cp = cpts[0].clone()
            cp[:, 0] -= float(env_off[0]); cp[:, 1] -= float(env_off[1])  # -> env-local
        rec["body_pos"].append(bp.cpu().numpy())
        rec["body_rot"].append(br.cpu().numpy())
        rec["net_force"].append(net.cpu().numpy())
        rec["ground_force"].append(grd.cpu().numpy())
        rec["contact_pt"].append(cp.cpu().numpy())
        rec["dof_pos"].append(e.get_dof_pos(cid)[0].cpu().numpy())
        rec["dof_vel"].append(e.get_dof_vel(cid)[0].cpu().numpy())
        rec["dof_torque"].append(e.get_dof_forces(cid)[0].cpu().numpy())

    n = 0
    for t in range(args.steps):
        snap(t)
        a = deterministic_action(obs)
        obs, r, done, info = env.step(a)
        n += 1
        if bool(done[0]):
            break
        obs, info = agent._reset_done_envs(done)

    out = {k: np.stack(v) for k, v in rec.items()}  # each [T, ...]
    np.savez(args.out,
             body_names=np.array(body_names), dof_names=np.array(dof_names),
             dt=float(env._engine.get_timestep()), char_weight=char_weight,
             **out)
    os.unlink(tmp.name)
    print("Recorded {} steps -> {}".format(n, args.out))
    print("bodies={}  dofs={}  char_weight={:.1f} N".format(B, dof_size, char_weight))
    print("contact_pt available: {}".format(not np.all(np.isnan(out["contact_pt"]))))


if __name__ == "__main__":
    main()
