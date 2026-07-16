"""Honest per-start-tier eval of an ENTRY-edge policy (tadasana -> pose).
Env-agnostic and pose-agnostic: derives the TARGET signature (torso up-vector,
per-key-body height) from the edge clip's OWN tail, so the same script judges
warrior2 (wide upright), warrior3 (single-leg torso-horizontal), and handstand
(inverted) without hardcoded thresholds.

Success = TRAVERSED from a given RSI start to the target hold (survive to ~clip
end in the DeepMimic env, which SUCCs at motion end) AND final pose matches the
reference tail. The crucial tier is t=0 (the true edge start); wide-RSI training
curves are curriculum-optimistic (framework lesson).

  RUN / ENV_YAML / AGENT_YAML / CLIP / TIERS / PORT / TAG via env vars."""
import sys, os
import numpy as np
import torch
REPO = '/home/visakii/Documents/moves/MimicKit'
sys.path.insert(0, os.path.join(REPO, 'tools')); sys.path.insert(0, os.path.join(REPO, 'mimickit'))
os.chdir(REPO)
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util
import anim.motion as motion
import util.torch_util as tu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUN = os.environ['RUN']
ENV_YAML = os.environ.get('ENV_YAML', f'{RUN}/env_config.yaml')
AGENT_YAML = os.environ.get('AGENT_YAML', f'{RUN}/agent_config.yaml')
CLIP = os.environ['CLIP']
TAG = os.environ.get('TAG', os.path.basename(RUN))
MODEL = os.environ.get('MODEL', f'{RUN}/model.pt')
TIERS = [float(x) for x in os.environ.get('TIERS', '0.0,1.0,2.0').split(',')]
N = int(os.environ.get('N', '64'))
device = 'cuda:0'; mp_util.init(0, 1, device, int(os.environ.get('PORT', '29760'))); util.set_rand_seed(7)

env = env_builder.build_env(ENV_YAML, N, device, visualize=False)
agent = agent_builder.build_agent(AGENT_YAML, env, device)
agent.load(MODEL); agent.eval(); agent.set_mode(agent._mode.__class__.TEST)
kcm = env._kin_char_model; names = kcm.get_body_names(); idx = {n: i for i, n in enumerate(names)}
KEY = ['L_Toe', 'R_Toe', 'Head', 'L_Hand', 'R_Hand']
KI = [idx[k] for k in KEY]
PEL, CH = idx['Pelvis'], idx['Chest']
LA, RA = idx['L_Ankle'], idx['R_Ankle']
cid = env._get_char_id(); e = env._engine; dt = e.get_timestep(); fps = 1.0 / dt
parents = [int(kcm.get_parent_id(i)) for i in range(len(names))]

# ---- target signature from the reference clip tail (last ~0.5s) --------------
m = motion.load_motion(CLIP); rf = np.asarray(m.frames, np.float32); rfps = float(m.fps)
clip_len_s = (rf.shape[0] - 1) / rfps
rbp, _ = kcm.forward_kinematics(
    torch.tensor(rf[:, :3], device=device),
    tu.exp_map_to_quat(torch.tensor(rf[:, 3:6], device=device)),
    tu.quat_pos(kcm.dof_to_rot(torch.tensor(rf[:, 6:], device=device))))
rbp = rbp.cpu().numpy()  # [T,B,3]
def up_proxy(bp):  # torso up-vector z (chest-above-pelvis), works inverted too
    v = bp[..., CH, :] - bp[..., PEL, :]
    return v[..., 2] / (np.linalg.norm(v, axis=-1) + 1e-6)
tail0 = int(rf.shape[0] - 0.5 * rfps)
tgt_up = float(np.mean(up_proxy(rbp[tail0:])))
tgt_keyz = rbp[tail0:][:, KI, 2].mean(0)         # [5] key-body target heights
tgt_pelz = float(rbp[tail0:, PEL, 2].mean())
tgt_width = float(np.linalg.norm(rbp[tail0:, LA, :2] - rbp[tail0:, RA, :2], axis=-1).mean())
STEPS = min(int(clip_len_s * fps) + 25, 640)
clip_end_step = int(clip_len_s * fps)
print(f'=== ENTRY EDGE eval: {TAG} (N={N}) ===')
print(f'clip {os.path.basename(CLIP)} len {clip_len_s:.2f}s -> STEPS {STEPS}, clip_end_step {clip_end_step}')
print(f'TARGET (ref tail): up_proxy {tgt_up:+.2f}  pelvis_z {tgt_pelz:.2f}  stance_w {tgt_width*100:.0f}cm')
print(f'  key heights (cm) ' + '  '.join(f'{k}:{tgt_keyz[i]*100:.0f}' for i, k in enumerate(KEY)))

def rollout(t0):
    env._init_time_range = [float(t0), float(t0) + 0.0001]
    obs, info = agent._reset_envs()
    BP = []; alive = torch.ones(N, dtype=torch.bool, device=device)
    death = torch.full((N,), STEPS, dtype=torch.long, device=device)
    for step in range(STEPS):
        with torch.no_grad():
            a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        obs, r, done, info = env.step(a)
        BP.append(e.get_body_pos(cid).cpu().numpy().copy())
        death[done.bool() & alive] = step; alive &= ~done.bool()
    return np.stack(BP), death.cpu().numpy()   # BP [STEPS,N,B,3]

print(f'\n{"start t":>7} {"traversed":>10} {"fell<end":>9} {"camp/wrong":>11} {"survive(s)":>11} {"upErr":>6} {"keyErr":>7}   note')
RES = {}
for t0 in TIERS:
    BP, death = rollout(t0)
    trav = np.zeros(N, bool); fell = np.zeros(N, bool); camp = np.zeros(N, bool)
    upE = np.zeros(N); keyE = np.zeros(N)
    # a start at t0 reaches the CLAMP clip end after fewer steps (motion_len_term)
    steps_to_end = clip_end_step - int(round(t0 * fps))
    for n in range(N):
        f = death[n] - 1
        up = up_proxy(BP[f, n]); kz = BP[f, n, KI, 2]
        upE[n] = abs(up - tgt_up); keyE[n] = np.abs(kz - tgt_keyz).mean()
        survived = death[n] >= steps_to_end - 15
        pose_ok = (upE[n] < 0.25) and (keyE[n] < 0.20)
        if survived and pose_ok: trav[n] = True
        elif not survived: fell[n] = True
        else: camp[n] = True   # survived but wrong pose (camped / diverged-in-band)
    note = 'TRUE edge start' if t0 == 0 else ('mid-transition' if t0 < clip_len_s - 3 else 'target tail')
    print(f'{t0:7.1f} {100*trav.mean():9.0f}% {100*fell.mean():8.0f}% {100*camp.mean():10.0f}% '
          f'{death.mean()/fps:10.1f} {upE.mean():6.2f} {keyE.mean():7.2f}   {note}')
    RES[t0] = (BP, death, trav, fell, camp)

# ---- t=0 detail + render -----------------------------------------------------
BP, death, trav, fell, camp = RES[0.0]
print(f'\n=== t=0 detail (target up_proxy {tgt_up:+.2f}, key-z {np.round(tgt_keyz*100).astype(int)}cm) ===')
fz = np.array([death[n] - 1 for n in range(N)])
finals_up = np.array([up_proxy(BP[fz[n], n]) for n in range(N)])
finals_keyz = np.array([BP[fz[n], n, KI, 2] for n in range(N)])
finals_pelz = np.array([BP[fz[n], n, PEL, 2] for n in range(N)])
print(f'  final up_proxy: mean {finals_up.mean():+.2f} (tgt {tgt_up:+.2f})   final pelvis_z: mean {finals_pelz.mean():.2f} (tgt {tgt_pelz:.2f})')
for i, k in enumerate(KEY):
    print(f'  {k:>7} final z mean {finals_keyz[:,i].mean()*100:5.0f}cm  (tgt {tgt_keyz[i]*100:.0f})')
verdict = ('TRAVERSES to target' if trav.mean() > 0.5 else
           ('FALLS early' if fell.mean() > 0.5 else 'CAMPS / wrong-pose (survives, off-target)'))
print(f'\n>>> t=0 VERDICT: {verdict};  traversed {100*trav.mean():.0f}%  fell {100*fell.mean():.0f}%  camp {100*camp.mean():.0f}%')

# strip: rep env (median survival) front view + reference tail
rep = int(np.argsort(death)[N // 2]); De = death[rep]
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.3)
ax0 = fig.add_subplot(gs[0])
ax0.bar([str(t) for t in TIERS], [100 * RES[t][2].mean() for t in TIERS], color='tab:green', alpha=0.6, label='traversed')
ax0.bar([str(t) for t in TIERS], [100 * RES[t][3].mean() for t in TIERS], bottom=[100 * RES[t][2].mean() for t in TIERS], color='tab:red', alpha=0.4, label='fell')
ax0.set_ylabel('% envs'); ax0.set_ylim(0, 105); ax0.legend(fontsize=8)
ax0.set_title(f'{TAG} — per-start-tier ENTRY success (t=0 verdict: {verdict})', fontsize=11)
ax1 = fig.add_subplot(gs[1]); kfs = np.linspace(0, De - 1, 7).astype(int)
span = np.nanmax(np.abs(BP[:De, rep, :, 1] - BP[:De, rep, PEL, 1][:, None])) * 2 + 0.3
for k, f in enumerate(kfs):
    off = k * (span + 0.3); H = (BP[f, rep, :, 1] - BP[f, rep, PEL, 1]) + off; Z = BP[f, rep, :, 2]
    for b in range(len(names)):
        p = parents[b]
        if p >= 0: ax1.plot([H[b], H[p]], [Z[b], Z[p]], color='0.6', lw=1.3)
    ax1.scatter(H, Z, s=10, c='steelblue', zorder=2)
    ax1.text(off, -0.15, f't={f/fps:.2f}s', ha='center', fontsize=8)
ax1.axhline(0, color='saddlebrown', lw=2, alpha=0.6); ax1.set_aspect('equal'); ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title(f'POLICY from t=0 (front view) — rep env, ends {De/fps:.1f}s  up_proxy {up_proxy(BP[De-1,rep]):+.2f}', fontsize=10)
out = f'{RUN}/entry_eval_{TAG}.png'
fig.savefig(out, dpi=100, bbox_inches='tight'); print(f'\nwrote {out}')
np.savez(f'{RUN}/entry_rollout_t0.npz', BP=BP.astype(np.float32), death=death, rep=rep, fps=fps,
         tgt_up=tgt_up, tgt_keyz=tgt_keyz)
