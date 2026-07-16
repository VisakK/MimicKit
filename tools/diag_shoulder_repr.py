"""Test the exp-map (kcm SPHERICAL / reward+obs) vs Euler-xyz (3 physics hinges)
representation gap on the SHOULDER dof, per pose. If the gap is large for the
handstand (overhead shoulder) and small for warrior2/downdog (moderate arm), then
even perfect DeepMimic joint-tracking lands the physical arm wrong -> fingertips.
Pure math on the reference clip dofs — no sim."""
import os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit")); os.chdir(REPO)
import anim.kin_char_model as kcm_mod
import anim.motion as motion
import util.torch_util as tu

kcm = kcm_mod.KinCharModel("cpu")
kcm.load_char_file("data/assets/smpl/smpl_boxhands.xml")
names = kcm.get_body_names()
jid = {n: i for i, n in enumerate(names)}

def euler_xyz_quat(dof3):
    # 3 sequential hinges about local x,y,z (XML tree order) = Rx * Ry * Rz
    ax = torch.tensor([1.,0,0]); ay = torch.tensor([0,1.,0]); az = torch.tensor([0,0,1.])
    qx = tu.axis_angle_to_quat(ax, dof3[0]); qy = tu.axis_angle_to_quat(ay, dof3[1]); qz = tu.axis_angle_to_quat(az, dof3[2])
    return tu.quat_mul(qx, tu.quat_mul(qy, qz))

def quat_angle_between(q1, q2):
    d = tu.quat_mul(q1, tu.quat_conjugate(q2))
    w = torch.clamp(torch.abs(d[..., 3]), 0, 1)
    return float(torch.rad2deg(2 * torch.acos(w)))

CASES = [
    ("handstand", "data/motions/smpl_edges/downdog3L_to_handstand", -20),   # tail = handstand
    ("warrior2",  "data/motions/smpl_edges/tadasana_to_warrior2",   -20),   # tail = warrior2
    ("warrior3",  "data/motions/smpl_edges/tadasana_to_warrior3",   -20),
    ("downdog",   "data/motions/smpl_edges/tadasana_to_downdog",    -20),
]
print(f"{'pose':<10} {'joint':<11} {'|dof|(deg)':>10} {'expmap-vs-Euler gap(deg)':>26}")
for nm, clip, tail_off in CASES:
    m = motion.load_motion(clip); frames = np.asarray(m.frames, np.float32)
    fr = torch.tensor(frames[tail_off], dtype=torch.float32)   # a tail (target-hold) frame
    dof = fr[6:]  # joint dofs (kcm order)
    for jn in ["L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow"]:
        j = jid[jn]; i0 = kcm.get_joint_dof_idx(j); dim = kcm.get_joint_dof_dim(j)
        d3 = dof[i0:i0+dim]
        if dim != 3:  # elbow may be spherical too here; handle 3-dof only
            print(f"{nm:<10} {jn:<11}  (dof_dim={dim})"); continue
        q_expmap = tu.exp_map_to_quat(d3)     # what kcm/reward/obs use
        q_euler = euler_xyz_quat(d3)          # what 3 physics hinges produce
        mag = float(torch.rad2deg(torch.norm(d3)))
        gap = quat_angle_between(q_expmap, q_euler)
        print(f"{nm:<10} {jn:<11} {mag:10.0f} {gap:26.1f}")
    print()
