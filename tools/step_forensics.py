"""Reference-clip step/CoM anatomy (PROMOTED from scratchpad 2026-07-12):
per-frame stance width, foot-unit witness z, horizontal foot speed (the
float-robust contact indicator), CoM share along the inter-ankle axis.
Produced step_forensics.png for malasana_to_tadasana."""
import sys, os
import numpy as np
import torch

REPO = '/home/visakii/Documents/moves/MimicKit'
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'mimickit'))
os.chdir(REPO)

import anim.motion as motion
import anim.kin_char_model as kcm_mod
import anim.char_geoms as char_geoms
import util.torch_util as torch_util
import annotate_clips as AC

kcm = kcm_mod.KinCharModel('cpu'); kcm.load_char_file('data/assets/smpl/smpl_boxhands.xml')
names = kcm.get_body_names()
geoms = char_geoms.load_char_geoms('data/assets/smpl/smpl_boxhands.xml', names, 'cpu')

m = motion.load_motion('data/motions/smpl_edges/malasana_to_tadasana')
fps = float(m.fps)
fr = np.asarray(m.frames, dtype=np.float32)
T = fr.shape[0]
rp = torch.tensor(fr[:, 0:3]); rr = torch_util.exp_map_to_quat(torch.tensor(fr[:, 3:6]))
jr = torch_util.quat_pos(kcm.dof_to_rot(torch.tensor(fr[:, 6:])))
bp, br = kcm.forward_kinematics(rp, rr, jr)

minz, wit_xy = AC.per_body_witness(geoms, bp, br)   # [T,B], [T,B,2]

idx = {n: i for i, n in enumerate(names)}
LT, LA, RT, RA = idx['L_Toe'], idx['L_Ankle'], idx['R_Toe'], idx['R_Ankle']

# foot-unit signals (min over toe+ankle per side = the support unit)
foot_z = {'L': torch.minimum(minz[:, LT], minz[:, LA]).numpy(),
          'R': torch.minimum(minz[:, RT], minz[:, RA]).numpy()}
# foot horizontal position = toe body xy; speed via central fd
foot_xy = {'L': bp[:, LT, :2].numpy(), 'R': bp[:, RT, :2].numpy()}
foot_speed = {}
for s in 'LR':
    v = np.gradient(foot_xy[s], axis=0) * fps
    foot_speed[s] = np.linalg.norm(v, axis=-1)

width = np.linalg.norm(foot_xy['L'] - foot_xy['R'], axis=-1)

# stock annotator contact rule (floor = 0, baked clip)
stock_contact = {s: foot_z[s] < AC.CONTACT_EPS for s in 'LR'}

# kinematic CoM (annotator mass fractions, corpus-standard)
mass = np.zeros(len(names))
for n, f in AC.MASS_FRACTIONS.items():
    if n in idx: mass[idx[n]] = f
mass = mass / mass.sum()
com = (bp.numpy() * mass[None, :, None]).sum(axis=1)   # [T,3]
com_v = np.gradient(com[:, :2], axis=0) * fps
com_speed = np.linalg.norm(com_v, axis=-1)

# CoM share along the inter-ankle axis: 0 = over R foot, 1 = over L foot
aL, aR = bp[:, LA, :2].numpy(), bp[:, RA, :2].numpy()
axis = aL - aR
alpha = ((com[:, :2] - aR) * axis).sum(-1) / np.maximum((axis * axis).sum(-1), 1e-9)

# swing detection: horizontal speed OR relative lift above own 10-frame min
REL = {s: foot_z[s] - np.minimum.accumulate(foot_z[s][::-1])[::-1] * 0 for s in 'LR'}
base = {s: np.percentile(foot_z[s][:30], 50) for s in 'LR'}   # lead-in seat level
swing = {s: (foot_speed[s] > 0.25) for s in 'LR'}

tt = np.arange(T) / fps
print('=== per-0.1s timeline (transition ends 3.07s; step window 3.0-5.07s) ===')
print(f"{'t':>5} {'width':>6} | {'Lz':>5} {'Lspd':>5} {'Lsw':>3} {'Lstock':>6} | "
      f"{'Rz':>5} {'Rspd':>5} {'Rsw':>3} {'Rstock':>6} | {'alpha':>6} {'comspd':>6}")
for f in range(75, T, 3):
    print(f"{tt[f]:5.2f} {width[f]*100:5.0f}cm | {foot_z['L'][f]*100:5.1f} {foot_speed['L'][f]:5.2f} "
          f"{'*' if swing['L'][f] else '.':>3} {'C' if stock_contact['L'][f] else '-':>6} | "
          f"{foot_z['R'][f]*100:5.1f} {foot_speed['R'][f]:5.2f} {'*' if swing['R'][f] else '.':>3} "
          f"{'C' if stock_contact['R'][f] else '-':>6} | {alpha[f]:6.2f} {com_speed[f]:6.2f}")

# contiguous swing windows
print('\n=== swing windows (foot speed > 0.25 m/s, >= 3 frames) ===')
for s in 'LR':
    on = swing[s].astype(int)
    d = np.diff(np.concatenate([[0], on, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(starts, ends):
        if b - a >= 3:
            print(f"  {s} foot: {tt[a]:.2f}-{tt[b-1]:.2f}s  peak spd {foot_speed[s][a:b].max():.2f} m/s  "
                  f"lift {(foot_z[s][a:b].max()-base[s])*100:+.1f}cm  width {width[a]*100:.0f}->{width[min(b,T-1)]*100:.0f}cm  "
                  f"alpha(mid) {alpha[(a+b)//2]:.2f}  stock-contact frames {int(stock_contact[s][a:b].sum())}/{b-a}")

# CoM shift preceding each lift: alpha at window start vs 0.3s before
print('\n=== weight transfer check (alpha: 0=over R foot, 1=over L foot) ===')
for f0 in (90, 100, 110, 120, 130, 140, 150):
    if f0 < T:
        print(f"  t={tt[f0]:.2f}s alpha={alpha[f0]:.2f} com_speed={com_speed[f0]:.2f} width={width[f0]*100:.0f}cm")

# figure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
axes[0].plot(tt, width * 100); axes[0].set_ylabel('stance width (cm)')
axes[0].axvspan(1.0, 3.07, color='orange', alpha=0.12)
for s, c in zip('LR', ('tab:blue', 'tab:red')):
    axes[1].plot(tt, foot_z[s] * 100, c, label=f'{s} foot unit min-z')
axes[1].axhline(AC.CONTACT_EPS * 100, color='gray', ls='--', lw=0.8, label='stock 5cm rule')
axes[1].legend(); axes[1].set_ylabel('foot z (cm)')
for s, c in zip('LR', ('tab:blue', 'tab:red')):
    axes[2].plot(tt, foot_speed[s], c, label=f'{s} foot speed')
axes[2].axhline(0.25, color='gray', ls='--', lw=0.8)
axes[2].legend(); axes[2].set_ylabel('foot horiz speed (m/s)')
axes[3].plot(tt, alpha, 'k', label='CoM share (0=R foot, 1=L foot)')
axes[3].axhline(0.5, color='gray', ls=':', lw=0.8)
ax2 = axes[3].twinx(); ax2.plot(tt, com_speed, 'g', alpha=0.5); ax2.set_ylabel('CoM speed (m/s)', color='g')
axes[3].legend(loc='upper left'); axes[3].set_ylabel('alpha'); axes[3].set_xlabel('s')
for ax in axes: ax.axvspan(3.07, 5.07, color='purple', alpha=0.06)
fig.suptitle('malasana->tadasana reference: step-together forensics (orange=transition, purple=step window)')
fig.tight_layout()
fig.savefig('output/yoga_edges_v3/malasana_to_tadasana/step_forensics.png', dpi=110)
print('\nfig -> output/yoga_edges_v3/malasana_to_tadasana/step_forensics.png')
