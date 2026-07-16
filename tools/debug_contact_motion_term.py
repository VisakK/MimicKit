"""Per-frame instrumentation of the contact_motion reward term over a live
rollout (PROMOTED from scratchpad 2026-07-12). Use before concluding any
window-gated term is satisfied — training means hide gated channels."""
import sys, os
import torch

REPO = '/home/visakii/Documents/moves/MimicKit'
sys.path.insert(0, os.path.join(REPO, 'mimickit'))
os.chdir(REPO)

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util

device = 'cuda:0'
mp_util.init(0, 1, device, 29714)
util.set_rand_seed(7)
env = env_builder.build_env('data/envs/edge_malasana_to_tadasana_ft_env.yaml', 8, device, visualize=False)
agent = agent_builder.build_agent('data/agents/amp_smpl_hold_discfocus_agent.yaml', env, device)
agent.load('output/yoga_edges_v3/malasana_to_tadasana_ft2/model.pt')
agent.eval(); agent.set_mode(agent._mode.__class__.TEST)
env._init_time_range = [0.0, 0.0001]

env._ensure_contact_motion_masks()
swing_mask, make_mask = env._cm_masks
print('swing frames per col:', swing_mask.sum(dim=0).tolist())
print('make  frames per col:', make_mask.sum(dim=0).tolist())
print('cm cols:', env._cm_cols, 'anchor cols:', env._cm_anchor_cols)
ids_all = env._contact_schedule_body_ids
foot_ids = ids_all[env._cm_cols]
names = env._kin_char_model.get_body_names()
print('cm bodies:', [names[i] for i in foot_ids.tolist()])

cid = env._get_char_id(); e = env._engine
obs, info = agent._reset_envs()
fps = env._contact_schedule_fps
for step in range(int(9.0 * 30)):
    with torch.no_grad():
        a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
    obs, r, done, info = env.step(a)
    t = env._get_motion_times()[0]
    idx = int(min(round(float(t) * fps), swing_mask.shape[0] - 1))
    sw = swing_mask[idx]; mk = make_mask[idx]
    if bool(sw.any()) or bool(mk.any()):
        bp = e.get_body_pos(cid)
        z = bp[0, foot_ids, 2]
        ref_z = env._ref_body_pos[0, foot_ids, 2]
        short = torch.clamp(ref_z - env._contact_motion_z_slack - z, min=0.0) * sw.float()
        _, err_m, cm_r = env._apply_contact_motion_reward(
            torch.zeros(8, device=device), e.get_body_pos(cid))
        print(f't={float(t):.2f} sw={sw.int().tolist()} mk={mk.int().tolist()} '
              f'z={[round(float(v),3) for v in z]} ref_z={[round(float(v),3) for v in ref_z]} '
              f'short={[round(float(v),3) for v in short]} r0={float(cm_r[0]):.3f}')
