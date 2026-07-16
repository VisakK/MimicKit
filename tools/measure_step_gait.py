"""Step-vs-drag gait test (PROMOTED from session scratchpad 2026-07-12).
Peak swing-foot lift vs a STANDING-phase baseline + horizontal speed, over
128 deterministic rollouts. Edit the env/agent/model paths + the window/
baseline times for the arm under test (see Yoga_edge_framework_v3.md §11
gotchas: baseline on STANDING frames, forced windows before any SUCC end)."""
import sys, os
import numpy as np
import torch

REPO = '/home/visakii/Documents/moves/MimicKit'
sys.path.insert(0, os.path.join(REPO, 'mimickit'))
os.chdir(REPO)

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util

device = 'cuda:0'
mp_util.init(0, 1, device, 29692)
util.set_rand_seed(42)
env = env_builder.build_env('data/envs/edge_malasana_to_tadasana_ft_env.yaml', 128, device, visualize=False)
agent = agent_builder.build_agent('data/agents/amp_smpl_hold_discfocus_agent.yaml', env, device)
agent.load('output/yoga_edges_v3/malasana_to_tadasana_ft2/model.pt')
agent.eval()
agent.set_mode(agent._mode.__class__.TEST)
env._init_time_range = [0.0, 0.0001]

kcm = env._kin_char_model
names = kcm.get_body_names()
LT, RT = names.index('L_Toe'), names.index('R_Toe')
cid = env._get_char_id()
e = env._engine
dt = e.get_timestep()
steps = int(9.5 / dt)

obs, info = agent._reset_envs()
tz = {LT: [], RT: []}
txy = {LT: [], RT: []}
for step in range(steps):
    with torch.no_grad():
        a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
    obs, r, done, info = env.step(a)
    bp = e.get_body_pos(cid)
    for b in (LT, RT):
        tz[b].append(bp[:, b, 2].cpu().numpy().copy())
        txy[b].append(bp[:, b, :2].cpu().numpy().copy())

fps = 1.0 / dt
res = {}
for b, nm in ((LT, 'L_Toe'), (RT, 'R_Toe')):
    z = np.stack(tz[b])          # [T, N]
    xy = np.stack(txy[b])        # [T, N, 2]
    spd = np.linalg.norm(np.gradient(xy, axis=0), axis=-1) * fps   # [T, N]
    w0, w1 = int(2.8 * fps), int(8.0 * fps)     # narrowing window
    base = np.median(z[int(8.5 * fps):int(9.4 * fps)], axis=0)  # STANDING baseline (squat toe z is higher; old baseline hid lifts)
    clear = (z[w0:w1] - base[None, :])
    # per-env peak clearance and peak speed within the window
    res[nm] = (clear.max(axis=0), spd[w0:w1].max(axis=0))
    print(f'{nm}: peak lift vs STANDING baseline cm  mean {clear.max(axis=0).mean()*100:.1f}  '
          f'p25 {np.percentile(clear.max(axis=0),25)*100:.1f}  p75 {np.percentile(clear.max(axis=0),75)*100:.1f} | '
          f'abs z: standing {base.mean()*100:.1f}cm  window-peak {z[w0:w1].max(axis=0).mean()*100:.1f}cm | '
          f'peak horiz speed {spd[w0:w1].max(axis=0).mean():.2f} m/s')

# reference comparison (toe BODY z, same signal class as the policy measurement)
import anim.motion as motion
import util.torch_util as torch_util
m = motion.load_motion('data/motions/smpl_edges/malasana_to_tadasana_v2')
fr = np.asarray(m.frames, dtype=np.float32)
rp = torch.tensor(fr[:, 0:3], device=device)
rr = torch_util.exp_map_to_quat(torch.tensor(fr[:, 3:6], device=device))
jr = torch_util.quat_pos(kcm.dof_to_rot(torch.tensor(fr[:, 6:], device=device)))
bp, _ = kcm.forward_kinematics(rp, rr, jr); bp = bp.cpu().numpy()
for b, nm, (s0, s1) in ((RT, 'R_Toe', (86, 100)), (LT, 'L_Toe', (129, 142))):
    zb = bp[:30, b, 2].mean()
    print(f'REF {nm}: planted body-z {zb*100:.1f}cm; swing-window peak clearance '
          f'{(bp[s0:s1, b, 2].max()-zb)*100:.1f}cm')
