"""Verify the regenerated+stripped smpl_boxhands asset: build a boxhands env,
print the per-DOF effort caps (arm joints should now be Shoulder 150 / Elbow 100
/ Wrist 75 / Hand 50; legs unchanged), and step a few times to confirm a single
articulation loads and physics is finite."""
import sys
sys.path.insert(0, "mimickit")
import torch
import envs.env_builder as env_builder
import util.mp_util as mp_util

mp_util.init(0, 1, "cuda:0", 6995)
env = env_builder.build_env("data/envs/deepmimic_smpl_handstand_orient_env.yaml",
                            1, "cuda:0", visualize=False)
kcm = env._kin_char_model
cid = env._get_char_id()
dof_names = []
for j in range(1, kcm.get_num_joints()):
    joint = kcm.get_joint(j); dim = joint.get_dof_dim()
    dof_names += ([joint.name] if dim == 1 else [f"{joint.name}_{a}" for a in range(dim)])
tl = env._engine.get_obj_torque_lim(0, cid)
import collections
print("VERIFY_CAPS_START n_dofs={} n_names={}".format(int(tl.shape[0]), len(dof_names)))
hist = collections.Counter(int(round(float(tl[i]))) for i in range(int(tl.shape[0])))
print("MAXF_HIST", dict(sorted(hist.items())))
for i, n in enumerate(dof_names):
    if i < int(tl.shape[0]) and any(k in n for k in ["Shoulder", "Elbow", "Wrist", "Hand"]):
        print("  {:<16} maxF={:.0f}".format(n, float(tl[i])))
a = torch.zeros(1, int(env.get_action_space().shape[0]), device="cuda:0")
ok = True
for _ in range(5):
    obs, r, done, info = env.step(a)
    ok = ok and bool(torch.isfinite(obs).all())
print("VERIFY_STEP_OK", ok)
print("VERIFY_DONE")
