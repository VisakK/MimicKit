"""Probe WHY the trained crow rests on its toes instead of a feet-up Bakasana.

Rolls the trained crow policy, lets it settle into the held pose, and measures
on the held frames:
  * toe/ankle GROUND contact force + toe world height  -> are the feet planted?
  * min distance from each shin (L_Knee/R_Knee body) to the upper-arm / forearm
    bodies (Shoulder/Elbow) -> is the policy USING knee-on-arm support at all, or
    keeping the knees away from the arms?
  * whether ANY body-body (self) contact registers near the arms.

This disambiguates the toe-rest cause: (a) knees NEAR arms + feet down = over-
supported, a reward issue (just push feet up); (b) knees FAR from arms = the
policy never uses the knee-arm shelf (frictionless self-contact / it fell into the
toe-rest basin) -> needs friction and/or a feet-up incentive.

Run: env_isaaclab/bin/python tools/probe_crow_contact.py
"""
import sys, os
sys.path.insert(0, "mimickit")
import numpy as np
import torch
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util

ENV = "data/envs/deepmimic_smpl_crow_env.yaml"
AGENT = "data/agents/deepmimic_smpl_ppo_agent.yaml"
MODEL = "output/yoga_skills/crow/model.pt"

mp_util.init(0, 1, "cuda:0", 6991)
N = 64
env = env_builder.build_env(ENV, N, "cuda:0", visualize=False)
agent = agent_builder.build_agent(AGENT, env, "cuda:0")
agent.load(MODEL)
agent.eval(); agent.set_mode(agent._mode.__class__.TEST)

kcm = env._kin_char_model
names = list(kcm.get_body_names())
def idx(n): return names.index(n)
print("BODY NAMES:", names)

cid = env._get_char_id()
e = env._engine

# knee bodies (shins) and arm bodies; tolerate naming variants
def first(*cands):
    for c in cands:
        if c in names: return c
    return None
knees = [b for b in [first("L_Knee"), first("R_Knee")] if b]
arms  = [b for b in [first("L_Shoulder"), first("R_Shoulder"),
                     first("L_Elbow"), first("R_Elbow")] if b]
toes  = [b for b in [first("L_Toe"), first("R_Toe")] if b]
ankles= [b for b in [first("L_Ankle"), first("R_Ankle")] if b]
print("knees:", knees, "arms:", arms, "toes:", toes)

obs, info = agent._reset_envs()
# settle into the held pose
for _ in range(250):
    with torch.no_grad():
        a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
    obs, r, done, info = env.step(a)
    obs, info = agent._reset_done_envs(done)

# measure over the next 120 held steps, only on envs that are upright-ish (held)
knee_arm_d, toe_h, toe_f, ankle_f = [], [], [], []
contact_ids = env._obs_contact_body_ids.tolist()
cname = [names[i] for i in contact_ids]
for _ in range(120):
    with torch.no_grad():
        a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
    bp = e.get_body_pos(cid)                      # [N, B, 3]
    cf = e.get_ground_contact_forces(cid)         # [N, B', 3] over obs_contact bodies
    cfn = torch.linalg.vector_norm(cf, dim=-1)    # [N, B']
    # min knee->arm distance (body origins)
    kd = []
    for k in knees:
        for ar in arms:
            kd.append(torch.linalg.vector_norm(bp[:, idx(k)] - bp[:, idx(ar)], dim=-1))
    knee_arm_d.append(torch.stack(kd, -1).min(-1).values)   # [N]
    toe_h.append(torch.stack([bp[:, idx(t), 2] for t in toes], -1).mean(-1))  # [N]
    # toe / ankle ground contact force
    tf = [cfn[:, cname.index(t)] for t in toes if t in cname]
    af = [cfn[:, cname.index(a_)] for a_ in ankles if a_ in cname]
    if tf: toe_f.append(torch.stack(tf, -1).mean(-1))
    if af: ankle_f.append(torch.stack(af, -1).mean(-1))
    obs, r, done, info = env.step(a)
    obs, info = agent._reset_done_envs(done)

def stat(x):
    x = torch.stack(x).flatten(); return float(x.mean()), float(x.median())
kam, kamed = stat(knee_arm_d); th, thmed = stat(toe_h)
tf_m = stat(toe_f) if toe_f else (None,None); af_m = stat(ankle_f) if ankle_f else (None,None)
print("\n===== CROW TOE-REST PROBE =====")
print(f"  min knee->arm body distance : mean {kam:.3f} m  median {kamed:.3f} m")
print(f"  toe world height            : mean {th:.3f} m  median {thmed:.3f} m")
print(f"  toe GROUND contact force    : mean {tf_m[0]:.1f} N (median {tf_m[1]:.1f})" if tf_m[0] is not None else "  toe force: n/a")
print(f"  ankle GROUND contact force  : mean {af_m[0]:.1f} N (median {af_m[1]:.1f})" if af_m[0] is not None else "  ankle force: n/a")
# rough geom radii for interpretation: upper-arm capsule ~0.045, shin capsule ~0.05
print("  (knee->arm < ~0.10 m would mean the shins are ON the upper arms; "
      ">0.15 m means the policy is NOT using the arm shelf.)")
print("PROBE_DONE")
