"""Decisive sim test for the handstand fingertip issue. Rolls out the DeepMimic
handstand-edge policy to the handstand hold and measures, frame-invariant (angles
from body positions -> no quaternion-convention pitfalls):
  (1) CONVENTION: physical arm angles (get_body_pos) vs kcm-FK of the policy's OWN
      dofs (get_dof_pos). If they DIVERGE -> reward/obs (kcm exp-map) != physics
      (Euler hinges) = the representation bug. If they MATCH -> physics==kcm.
  (2) DEVIATION: policy physical arm angles vs the REFERENCE (kcm-FK of clip tail).
  (3) FINGERTIP SIGNAL: physical wrist height vs reference wrist height.
"""
import os, sys
import numpy as np, torch
REPO = '/home/visakii/Documents/moves/MimicKit'
sys.path.insert(0, os.path.join(REPO, 'tools')); sys.path.insert(0, os.path.join(REPO, 'mimickit')); os.chdir(REPO)
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util
import anim.motion as motion, util.torch_util as tu

RUN = 'output/yoga_edges_v3/downdog3L_to_handstand_deepmimic'
CLIP = 'data/motions/smpl_edges/downdog3L_to_handstand'
N = 8; STEPS = 213; HOLD = 25   # last HOLD steps = handstand hold
device = 'cuda:0'; mp_util.init(0, 1, device, 29770); util.set_rand_seed(7)
env = env_builder.build_env(f'{RUN}/env_config.yaml', N, device, visualize=False)
agent = agent_builder.build_agent(f'{RUN}/agent_config.yaml', env, device)
agent.load(f'{RUN}/model_solved.pt'); agent.eval(); agent.set_mode(agent._mode.__class__.TEST)
kcm = env._kin_char_model; names = kcm.get_body_names(); idx = {n: i for i, n in enumerate(names)}
cid = env._get_char_id(); e = env._engine

def ang(a, b, c):  # angle at b (deg), from positions a,b,c  [...,3]
    u = a - b; v = c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))

def arm_angles(bp):  # bp [...,B,3] -> dict of L/R elbow & wrist bend angles
    out = {}
    for s in ['L', 'R']:
        Sh, El, Wr, Hd = idx[f'{s}_Shoulder'], idx[f'{s}_Elbow'], idx[f'{s}_Wrist'], idx[f'{s}_Hand']
        out[f'{s}_elbow'] = ang(bp[..., Sh, :], bp[..., El, :], bp[..., Wr, :])
        out[f'{s}_wrist'] = ang(bp[..., El, :], bp[..., Wr, :], bp[..., Hd, :])
    return out

env._init_time_range = [0.0, 0.0001]
obs, info = agent._reset_envs()
PHYS_BP = []; FK_BP = []; alive = torch.ones(N, dtype=torch.bool, device=device); death = torch.full((N,), STEPS, dtype=torch.long, device=device)
for step in range(STEPS):
    with torch.no_grad():
        a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
    obs, r, done, info = env.step(a)
    phys_bp = e.get_body_pos(cid)                      # [N,B,3] physical world
    dof = e.get_dof_pos(cid)                           # [N,dof] kcm order
    jrot = kcm.dof_to_rot(dof)
    r0 = torch.zeros((N, 3), device=device); rq = torch.zeros((N, 4), device=device); rq[:, 3] = 1
    fk_bp, _ = kcm.forward_kinematics(r0, rq, jrot)    # [N,B,3] kcm view (root frame)
    PHYS_BP.append(phys_bp.cpu().numpy()); FK_BP.append(fk_bp.cpu().numpy())
    death[done.bool() & alive] = step; alive &= ~done.bool()
PHYS = np.stack(PHYS_BP); FK = np.stack(FK_BP)   # [STEPS,N,B,3]
surv = (death.cpu().numpy() >= STEPS - HOLD - 2)
ns = surv.sum()
print(f"=== handstand physics-vs-FK forensic (N={N}, survived-to-hold={ns}) ===")
hold = slice(STEPS - HOLD, STEPS)
pa = arm_angles(PHYS[hold][:, surv]); fa = arm_angles(FK[hold][:, surv])

# reference (kcm view = exp-map FK of clip tail)
m = motion.load_motion(CLIP); rf = np.asarray(m.frames, np.float32)
rfr = torch.tensor(rf[-HOLD:], dtype=torch.float32, device=device)
rbp, _ = kcm.forward_kinematics(torch.zeros((HOLD, 3), device=device),
                                torch.tensor([[0,0,0,1.]]*HOLD, device=device),
                                kcm.dof_to_rot(rfr[:, 6:]))
ra = arm_angles(rbp.cpu().numpy())

print(f"\n(1) CONVENTION test — physical vs kcm-FK of SAME dofs (>~10 deg gap => physics != reward/obs):")
print(f"    {'joint':<10} {'physical':>9} {'kcm-FK':>9} {'|gap|':>7}")
for k in ['L_elbow','R_elbow','L_wrist','R_wrist']:
    p = np.mean(pa[k]); f = np.mean(fa[k])
    print(f"    {k:<10} {p:9.1f} {f:9.1f} {abs(p-f):7.1f}")

print(f"\n(2) DEVIATION — policy physical vs REFERENCE arm angles:")
print(f"    {'joint':<10} {'policy':>9} {'ref':>9} {'|dev|':>7}")
for k in ['L_elbow','R_elbow','L_wrist','R_wrist']:
    p = np.mean(pa[k]); rr = np.mean(ra[k])
    print(f"    {k:<10} {p:9.1f} {rr:9.1f} {abs(p-rr):7.1f}")

print(f"\n(3) FINGERTIP signal — wrist HEIGHT (world z, cm) during hold:")
for s in ['L', 'R']:
    wi = idx[f'{s}_Wrist']; hi = idx[f'{s}_Hand']
    wz = PHYS[hold][:, surv, wi, 2].mean() * 100
    hz = PHYS[hold][:, surv, hi, 2].mean() * 100
    # reference wrist/hand height (world z from full FK with real root)
    rbp_w, _ = kcm.forward_kinematics(torch.tensor(rf[-HOLD:, :3], device=device), tu.exp_map_to_quat(torch.tensor(rf[-HOLD:, 3:6], device=device)), kcm.dof_to_rot(rfr[:, 6:]))
    rwz = rbp_w[:, wi, 2].mean().item() * 100; rhz = rbp_w[:, hi, 2].mean().item() * 100
    print(f"    {s}: policy wrist {wz:5.1f} hand {hz:5.1f}  |  ref wrist {rwz:5.1f} hand {rhz:5.1f}   (fingertip => wrist >> hand)")
