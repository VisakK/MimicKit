"""Transition-policy environment for the yoga skill graph (spec sections 8, C, G).

Carries the character from skill A's terminal states into skill B's pose. This
is the v2 redesign that fixes the failures documented in the experiment notes
(sections 6/7): the v1 fixed far-away target gave the dense tracking reward no
gradient, and the (non-pose-specific) initiation classifier used as the reward
bonus was farmable by simply standing up. The new design:

  * INIT (RSI) = skill A's saved terminal states (skills/A/terminal_states.pt),
    set directly env-local (the cheap, unambiguous grounding of section 8).
  * REFERENCE  = a per-env MOVING interpolation from each env's RSI start pose
    (A's terminus) to B's fixed hold pose, advanced by an episode-phase ramp
    (spec C.1). The inherited DeepMimic tracking reward against this moving
    reference is dense and climbable the WHOLE way (unlike the v1 fixed target,
    whose error saturated the exp kernel to a flat zero). After the ramp the
    reference rests at B's hold, so tracking keeps driving the policy to STAY in
    B (contractive, section 13).
  * STYLE     = an AMP discriminator on B's hold clip. This env subclasses
    AMPEnv with enable_task_tracking, so the agent mixes
    task_reward_weight * (tracking + aux + goal bonus)
    + disc_reward_weight * disc_style. The discriminator is the NON-FARMABLE
    realism gate: a collapse-to-upright pose is obviously off-distribution to
    B's hold discriminator no matter what a classifier says, so it directly
    penalizes the v1 cheat (spec C.4 / G).
  * GOAL      = B's (now pose-discriminative) initiation classifier AND a true
    pose-distance check to B's hold (spec D). The per-step goal bonus
    reward_goal_w * 1[in_goal] is paid only when BOTH fire, so it is no longer
    farmable. `Goal_Frac` (mean in-goal) is the live training scoreboard.
  * OBS       = goal-conditioned: enable_goal_obs appends B's hold pose to the
    policy obs (#2), so the same architecture generalises across targets.

pose_termination MUST be off (the character starts far from B). The episode ends
only on a fall (trunk contact) or timeout. The transition config reuses B's
obs_contact_bodies + key_bodies so the online classifier features match what B's
classifier was trained on.
"""
import os
import torch

import envs.amp_env as amp_env
import envs.base_env as base_env
import skillgraph.init_classifier as ic
import skillgraph.state_features as sf


def quat_slerp(q0, q1, t):
    """Shortest-arc slerp between unit quaternions (xyzw). q0,q1: [...,4];
    t broadcastable to [...,1]. Falls back to nlerp for near-parallel pairs."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)          # take the shorter arc
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-5
    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / torch.clamp(sin_theta, min=1e-8))
    w1 = torch.where(small, t, torch.sin(t * theta) / torch.clamp(sin_theta, min=1e-8))
    q = w0 * q0 + w1 * q1
    q = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=1e-8)
    return q


class TransitionEnv(amp_env.AMPEnv):
    def __init__(self, config, num_envs, device, visualize):
        env_config = config["env"]
        self._init_states_file = env_config["init_states_file"]
        self._goal_classifier_file = env_config["goal_classifier"]
        self._goal_threshold = env_config.get("goal_threshold", 0.5)
        # pose-distance handoff gate (spec D): mean root-relative per-body shape
        # error to B's hold, plus an optional up-vector tolerance.
        self._goal_pose_dist = env_config.get("goal_pose_dist", 0.18)
        self._goal_up_z_tol = env_config.get("goal_up_z_tol", None)
        self._reward_goal_w = env_config.get("reward_goal_w", 1.0)
        # Dense goal-shape reward (spec section 8: w_shape * exp(-scale * pose_err_to_B)).
        # Pointed at B's ACTUAL hold pose (the eval metric), in cartesian body
        # space with a GENTLE scale so it never saturates flat -- this is the
        # continuous, climbable, non-farmable pull that closes (and holds) the
        # last gap the joint-space moving-ref tracking kernel saturates on.
        self._reward_goal_shape_w = env_config.get("reward_goal_shape_w", 2.0)
        self._reward_goal_shape_scale = env_config.get("reward_goal_shape_scale", 3.0)
        # seconds over which the moving reference ramps from A's terminus to B's
        # hold (then rests at B). The dense, climbable path of spec C.1.
        self._transition_ramp_seconds = env_config.get("transition_ramp_seconds", 3.0)
        # Curriculum RSI along the A->B interpolation (spec section 8 / C.3): a
        # fraction of envs reset at a random phase in (0, phase_max] toward B so
        # the policy gets RSI coverage of the near-B segment (learns to COMPLETE
        # and HOLD the pose), not only the far A terminus. The rest reset at
        # phase 0 (the true handoff the eval scores).
        self._curriculum_init_frac = env_config.get("curriculum_init_frac", 0.5)
        self._curriculum_phase_max = env_config.get("curriculum_phase_max", 1.0)
        # disc-demo sampling window (seconds); defaults to the single hold frame.
        # A small window captures the hold's natural micro-variation.
        gpt = float(env_config["goal_phase_time"])
        self._goal_demo_time_range = env_config.get("goal_demo_time_range", [gpt, gpt])

        assert (not env_config.get("pose_termination", False)), \
            "transition env must run with pose_termination: False (init is far from B)"
        assert env_config.get("enable_goal_obs", False), \
            "transition env requires enable_goal_obs: True (goal frame = B's hold)"
        assert env_config.get("enable_task_tracking", False), \
            "transition env requires enable_task_tracking: True (hybrid AMP task channel)"

        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize)
        return

    # ---- motion + init-state + classifier loading ----------------------------
    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)   # B's clip -> reference target + disc demos
        blob = torch.load(self._init_states_file, map_location=self._device)
        self._init_set = {k: blob[k].to(self._device).float()
                          for k in ["root_pos", "root_rot", "root_vel",
                                    "root_ang_vel", "dof_pos", "dof_vel"]}
        self._num_init_states = self._init_set["root_pos"].shape[0]
        assert self._num_init_states > 0, "empty init-state set: {}".format(self._init_states_file)
        self._goal_clf, _ = ic.load_classifier(self._goal_classifier_file, self._device)
        print("[transition] {} init states from {}; goal clf {} (pose_dist<{}, p_init>{})".format(
            self._num_init_states, os.path.basename(self._init_states_file),
            os.path.basename(self._goal_classifier_file), self._goal_pose_dist, self._goal_threshold))
        return

    # ---- B is the disc-demo / reference-seed source: sample its hold window ---
    def _sample_motion_times(self, n):
        motion_ids = torch.zeros(n, dtype=torch.int64, device=self._device)
        t0, t1 = self._goal_demo_time_range
        motion_times = t0 + (t1 - t0) * torch.rand(n, device=self._device)
        return motion_ids, motion_times

    def _get_motion_times(self, env_ids=None):
        ref = self._time_buf if env_ids is None else self._time_buf[env_ids]
        return torch.full_like(ref, float(self._goal_phase_time))

    def _build_data_buffers(self):
        super()._build_data_buffers()
        n = self.get_num_envs()
        self._goal_hold_count = torch.zeros(n, device=self._device, dtype=torch.long)
        self._in_goal = torch.zeros(n, device=self._device, dtype=torch.bool)
        self._last_pose_dist = torch.zeros(n, device=self._device)
        self._last_up_z_err = torch.zeros(n, device=self._device)
        self._interp_phase0 = torch.zeros(n, device=self._device)  # per-env curriculum start phase
        return

    def _ensure_interp_buffers(self):
        if (hasattr(self, "_interp_start_root_pos")):
            return
        n = self.get_num_envs()
        num_joints = self._kin_char_model.get_num_joints()
        self._interp_start_root_pos = torch.zeros(n, 3, device=self._device)
        self._interp_start_root_rot = torch.zeros(n, 4, device=self._device)
        self._interp_start_root_rot[:, 3] = 1.0
        self._interp_start_joint_rot = torch.zeros(n, num_joints - 1, 4, device=self._device)
        self._interp_start_joint_rot[..., 3] = 1.0
        return

    # ---- RSI from A's terminal states, with curriculum along the A->B path ----
    def _ref_state_init(self, env_ids):
        n = len(env_ids)
        device = self._device
        cid = self._get_char_id()
        self._ensure_goal_frame()
        self._ensure_interp_buffers()

        idx = torch.randint(0, self._num_init_states, (n,), device=device)
        s = self._init_set
        s_root_pos = s["root_pos"][idx]
        s_root_rot = s["root_rot"][idx]
        s_joint_rot = self._kin_char_model.dof_to_rot(s["dof_pos"][idx])   # [n,J,4]

        # curriculum start phase: phi0=0 for true-handoff envs, phi0~U(0,max] for
        # the curriculum fraction (reset partway up the A->B interpolation).
        phi0 = torch.zeros(n, device=device)
        curri = torch.rand(n, device=device) < self._curriculum_init_frac
        n_cur = int(curri.sum())
        if (n_cur > 0):
            phi0[curri] = self._curriculum_phase_max * torch.rand(n_cur, device=device)

        g_root_pos = self._goal_root_pos.expand(n, -1)
        g_root_rot = self._goal_root_rot.expand(n, -1)
        g_joint_rot = self._goal_joint_rot.expand(n, -1, -1)

        # initial pose = interp(sampled A terminus, B hold, phi0)
        p = phi0.unsqueeze(-1)
        init_root_pos = (1.0 - p) * s_root_pos + p * g_root_pos
        init_root_rot = quat_slerp(s_root_rot, g_root_rot, p)
        init_joint_rot = quat_slerp(s_joint_rot, g_joint_rot, p.unsqueeze(1))
        init_dof = self._motion_lib.joint_rot_to_dof(init_joint_rot)

        # keep the terminal-state velocities for true-handoff envs; synthetic
        # interpolated poses (phi0>0) start at rest.
        rv = s["root_vel"][idx].clone(); rav = s["root_ang_vel"][idx].clone(); dv = s["dof_vel"][idx].clone()
        rv[curri] = 0.0; rav[curri] = 0.0; dv[curri] = 0.0

        self._engine.set_root_pos(env_ids, cid, init_root_pos)
        self._engine.set_root_rot(env_ids, cid, init_root_rot)
        self._engine.set_root_vel(env_ids, cid, rv)
        self._engine.set_root_ang_vel(env_ids, cid, rav)
        self._engine.set_dof_pos(env_ids, cid, init_dof)
        self._engine.set_dof_vel(env_ids, cid, dv)
        self._engine.set_body_vel(env_ids, cid, 0.0)
        self._engine.set_body_ang_vel(env_ids, cid, 0.0)

        # the moving reference for these envs ramps from the SAMPLED A terminus
        # (s_A), starting at phase phi0 (so it equals the init pose at t=0).
        self._interp_start_root_pos[env_ids] = s_root_pos
        self._interp_start_root_rot[env_ids] = s_root_rot
        self._interp_start_joint_rot[env_ids] = s_joint_rot
        self._interp_phase0[env_ids] = phi0

        if (hasattr(self, "_goal_hold_count")):
            self._goal_hold_count[env_ids] = 0
            self._in_goal[env_ids] = False
        return

    # ---- moving A->B reference replaces the clip reference (spec C.1) ---------
    def _update_ref_motion(self):
        self._ensure_goal_frame()
        self._ensure_interp_buffers()
        n = self.get_num_envs()
        # phase advances from each env's curriculum start phi0 toward B.
        phase = torch.clamp(self._interp_phase0
                            + self._time_buf / max(self._transition_ramp_seconds, 1e-6),
                            0.0, 1.0).unsqueeze(-1)                       # [n,1]

        g_root_pos = self._goal_root_pos.expand(n, -1)                    # [n,3]
        g_root_rot = self._goal_root_rot.expand(n, -1)                    # [n,4]
        g_joint_rot = self._goal_joint_rot.expand(n, -1, -1)             # [n,J,4]

        self._ref_root_pos[:] = (1.0 - phase) * self._interp_start_root_pos + phase * g_root_pos
        self._ref_root_rot[:] = quat_slerp(self._interp_start_root_rot, g_root_rot, phase)
        self._ref_joint_rot[:] = quat_slerp(self._interp_start_joint_rot, g_joint_rot, phase.unsqueeze(1))
        # quasi-static target schedule: zero reference velocities (the small vel
        # reward weights then ask only "don't thrash"; at phase=1 this becomes a
        # proper hold-still target).
        self._ref_root_vel[:] = 0.0
        self._ref_root_ang_vel[:] = 0.0
        self._ref_dof_vel[:] = 0.0

        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        if (self._enable_ref_char()):
            self._ref_dof_pos[:] = self._motion_lib.joint_rot_to_dof(self._ref_joint_rot)
        return

    # ---- goal membership = pose-discriminative classifier AND pose distance ---
    def _update_misc(self):
        super()._update_misc()   # AMPEnv: moving ref (our override) + disc hist + contact
        pose_dist, up_z_err = self._compute_goal_match()
        with torch.no_grad():
            p = self._goal_clf.prob(self._compute_goal_features())
        in_goal = (p > self._goal_threshold) & (pose_dist < self._goal_pose_dist)
        if (self._goal_up_z_tol is not None):
            in_goal = in_goal & (up_z_err < self._goal_up_z_tol)
        self._in_goal = in_goal
        self._goal_hold_count = torch.where(
            self._in_goal, self._goal_hold_count + 1,
            torch.zeros_like(self._goal_hold_count))
        self._last_pose_dist = pose_dist
        self._last_up_z_err = up_z_err
        return

    def _compute_goal_features(self):
        cid = self._get_char_id()
        e = self._engine
        return sf.compute_state_features(
            root_pos=e.get_root_pos(cid), root_rot=e.get_root_rot(cid),
            root_vel=e.get_root_vel(cid), root_ang_vel=e.get_root_ang_vel(cid),
            body_pos=e.get_body_pos(cid), contact_forces=e.get_ground_contact_forces(cid),
            contact_body_ids=self._obs_contact_body_ids, key_body_ids=self._key_body_ids,
            com_weights=self._com_body_weights,
            force_threshold=self._obs_contact_force_threshold,
            support_dirs=getattr(self, "_support_polygon_dirs", None))

    def _update_reward(self):
        # AMPEnv hybrid: task reward = DeepMimic tracking of the MOVING reference
        # + aux (enable_task_tracking). The agent then mixes the disc style on
        # top. We add the (now non-farmable) goal bonus into the task channel.
        super()._update_reward()
        # dense, non-saturating pull toward B's actual hold pose (spec section 8)
        goal_shape_r = torch.exp(-self._reward_goal_shape_scale * self._last_pose_dist)
        self._reward_buf[:] = self._reward_buf \
            + self._reward_goal_shape_w * goal_shape_r \
            + self._reward_goal_w * self._in_goal.float()
        if (self._mode == base_env.EnvMode.TRAIN and hasattr(self, "_reward_term_tracker")):
            self._reward_term_tracker.update({
                "goal_frac": self._in_goal.float(),
                "goal_bonus": self._reward_goal_w * self._in_goal.float(),
                "goal_shape_r": goal_shape_r,
                "pose_dist": self._last_pose_dist,
                "up_z_err": self._last_up_z_err})
        return
