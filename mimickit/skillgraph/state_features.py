"""Canonical state featurizer for the yoga skill graph.

The initiation classifier (Deep Skill Chaining, Bagaria & Konidaris ICLR 2020;
spec section 4) must answer "from state s, can skill B reach and hold its pose?"
We deliberately featurize the *physical state* (balance + contact descriptors),
NOT a policy's reference-conditioned observation: the policy obs depends on the
reference clip phase (tar_obs), so it is ill-defined off the policy's own
distribution -- exactly where the handoff question lives. The features here are
well-defined for any character state regardless of which skill produced it,
which is what lets one skill's classifier be queried at another skill's
terminal states (the zero-shot feasibility test).

Feature layout (heading-invariant where it should be):
    [ root_h(1), root_up(3), root_lin_speed(1), root_ang_speed(1),
      com_h(1), support_margin(1), contact_flags(K), key_local(3*M) ]
with K = len(contact_body_ids), M = len(key_body_ids). All position-like
quantities are expressed root-heading-local so the descriptor is invariant to
where on the floor / which way the character faces.
"""
import torch

import util.torch_util as torch_util


def feature_names(contact_body_names, key_body_names):
    names = ["root_h", "root_up_x", "root_up_y", "root_up_z",
             "root_lin_speed", "root_ang_speed", "com_h", "support_margin"]
    names += ["contact_{}".format(b) for b in contact_body_names]
    for b in key_body_names:
        names += ["key_{}_x".format(b), "key_{}_y".format(b), "key_{}_z".format(b)]
    return names


def feature_dim(num_contact_bodies, num_key_bodies):
    return 8 + num_contact_bodies + 3 * num_key_bodies


def compute_state_features(root_pos, root_rot, root_vel, root_ang_vel,
                           body_pos, contact_forces, contact_body_ids,
                           key_body_ids, com_weights, force_threshold=1.0,
                           support_dirs=None, margin_cap=1.0):
    """Batched canonical features. All inputs [N, ...]; returns [N, D].

    root_pos      [N,3]      root world position
    root_rot      [N,4]      root world orientation (xyzw)
    root_vel      [N,3]      root linear velocity (world)
    root_ang_vel  [N,3]      root angular velocity (world)
    body_pos      [N,B,3]    all body world positions (FK output)
    contact_forces[N,B,3]    per-body ground contact force (world)
    contact_body_ids [K]     bodies whose contact flag we expose
    key_body_ids     [M]     bodies whose root-local position we expose
    com_weights      [B]     normalized per-body mass weights for the COM
    support_dirs   [Sd,2]    horizontal fan for the support-polygon margin
                             (default: 16 directions, built here if None)
    """
    device = root_pos.device
    N = root_pos.shape[0]

    # root up-vector in the world (z-component is the inversion indicator).
    up_local = torch.zeros_like(root_pos)
    up_local[..., 2] = 1.0
    root_up = torch_util.quat_rotate(root_rot, up_local)              # [N,3]

    root_lin_speed = torch.linalg.vector_norm(root_vel, dim=-1, keepdim=True)
    root_ang_speed = torch.linalg.vector_norm(root_ang_vel, dim=-1, keepdim=True)

    com = torch.einsum("nbk,b->nk", body_pos, com_weights)           # [N,3]
    com_h = com[..., 2:3]

    # contact flags over the candidate bodies
    cand_forces = contact_forces[:, contact_body_ids, :]
    flags = (torch.linalg.vector_norm(cand_forces, dim=-1) > force_threshold).float()  # [N,K]

    # support-polygon margin: signed horizontal distance from COM to the convex
    # hull of the contacting candidate bodies (positive inside). Same support-
    # function estimator as the contact obs in deepmimic_env.compute_contact_obs.
    if (support_dirs is None):
        num_dirs = 16
        ang = (2.0 * torch.pi / num_dirs) * torch.arange(num_dirs, device=device, dtype=torch.float32)
        support_dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
    cand_pos = body_pos[:, contact_body_ids, :]
    com_xy = com[..., :2]
    point_proj = torch.matmul(cand_pos[..., :2], support_dirs.t())   # [N,K,Sd]
    masked = torch.where(flags.unsqueeze(-1) > 0.5, point_proj,
                         torch.full_like(point_proj, -1e9))
    hull_support = torch.max(masked, dim=1)[0]                       # [N,Sd]
    com_proj = torch.matmul(com_xy, support_dirs.t())               # [N,Sd]
    margin = torch.min(hull_support - com_proj, dim=-1)[0]
    margin = torch.clamp(margin, min=-margin_cap, max=margin_cap)
    any_contact = torch.any(flags > 0.5, dim=-1)
    margin = torch.where(any_contact, margin, torch.full_like(margin, -margin_cap)).unsqueeze(-1)

    # key bodies, root-heading-local (invariant to floor position / facing).
    heading_inv = torch_util.calc_heading_quat_inv(root_rot)         # [N,4]
    key = body_pos[:, key_body_ids, :] - root_pos.unsqueeze(1)       # [N,M,3]
    M = key.shape[1]
    hexp = heading_inv.unsqueeze(1).repeat(1, M, 1).reshape(-1, 4)
    key_local = torch_util.quat_rotate(hexp, key.reshape(-1, 3)).reshape(N, M * 3)

    feats = torch.cat([root_pos[..., 2:3], root_up, root_lin_speed, root_ang_speed,
                       com_h, margin, flags, key_local], dim=-1)
    return feats
