import atexit
import numpy as np
import os
import torch
import anim.motion as motion
import anim.motion_lib as motion_lib
import envs.base_env as base_env
import envs.char_env as char_env
import engines.engine as engine
import util.stats_tracker as stats_tracker
import util.torch_util as torch_util

class DeepMimicEnv(char_env.CharEnv):
    def __init__(self, config, num_envs, device, visualize):
        env_config = config["env"]
        self._enable_early_termination = env_config["enable_early_termination"]
        self._num_phase_encoding = env_config.get("num_phase_encoding", 0)

        self._pose_termination = env_config.get("pose_termination", False)
        self._pose_termination_dist = env_config.get("pose_termination_dist", 1.0)
        self._enable_phase_obs = env_config.get("enable_phase_obs", True)
        self._enable_tar_obs = env_config.get("enable_tar_obs", False)
        self._tar_obs_steps = env_config.get("tar_obs_steps", [1])
        self._tar_obs_steps = torch.tensor(self._tar_obs_steps, device=device, dtype=torch.int)
        self._rand_reset = env_config.get("rand_reset", True)

        # Optional window (in seconds) for reference-state-init time sampling.
        # Default None keeps the standard uniform sampling over the whole
        # clip. Useful for focusing training on a sub-segment of a long clip
        # (e.g. the held portion of a pose) without re-cutting the motion.
        self._init_time_range = env_config.get("init_time_range", None)
        if (self._init_time_range is not None):
            assert(len(self._init_time_range) == 2
                   and 0.0 <= self._init_time_range[0] < self._init_time_range[1]), \
                "init_time_range must be [t0, t1] seconds with 0 <= t0 < t1"

        # Whole-clip ground offset for the reference motion (see MotionLib).
        # Off by default; lifts clips whose collision geometry penetrates the
        # ground so reference-state-init does not fire depenetration impulses.
        self._auto_ground_offset = env_config.get("auto_ground_offset", False)
        self._ground_offset_clearance = env_config.get("ground_offset_clearance", 0.0)
        # Constant whole-clip lift (meters, default 0). Manual alternative to
        # auto_ground_offset for softening reset depenetration impulses
        # without lifting by the clip's full worst-case penetration.
        self._ground_offset = env_config.get("ground_offset", 0.0)

        # Contact-aware observations (disabled by default). When enabled,
        # appends to the policy observation:
        #   1) per-body binary contact flags over obs_contact_bodies
        #      (||ground contact force|| > obs_contact_force_threshold),
        #   2) a signed support-polygon margin: horizontal distance from the
        #      mass-weighted COM to the convex hull of the contacting bodies
        #      (positive = inside the hull; -margin_cap sentinel when
        #      airborne), and
        #   3) a contact graph of pairwise body-position deltas, zeroed
        #      unless both bodies of a pair are in contact.
        self._enable_contact_obs = env_config.get("enable_contact_obs", False)
        self._obs_contact_bodies = env_config.get("obs_contact_bodies", [])
        self._obs_contact_force_threshold = env_config.get("obs_contact_force_threshold", 1.0)
        self._obs_contact_margin_cap = env_config.get("obs_contact_margin_cap", 1.0)

        self._ref_char_offset = torch.tensor(env_config["ref_char_offset"], device=device, dtype=torch.float)
        self._log_tracking_error = env_config.get("log_tracking_error", False)
        self._log_ground_contact_forces = env_config.get("log_ground_contact_forces", False)
        self._log_joint_torques = env_config.get("log_joint_torques", False)
        self._log_dir = env_config.get("log_dir", "output/diagnostics")
        self._log_tag = env_config.get("log_tag", "run")
        self._log_save_interval = env_config.get("log_save_interval", 100)

        self._reward_pose_w = env_config.get("reward_pose_w")
        self._reward_vel_w = env_config.get("reward_vel_w")
        self._reward_root_pose_w = env_config.get("reward_root_pose_w")
        self._reward_root_vel_w = env_config.get("reward_root_vel_w")
        self._reward_key_pos_w = env_config.get("reward_key_pos_w")

        self._reward_pose_scale = env_config.get("reward_pose_scale")
        self._reward_vel_scale = env_config.get("reward_vel_scale")
        self._reward_root_pose_scale = env_config.get("reward_root_pose_scale")
        self._reward_root_vel_scale = env_config.get("reward_root_vel_scale")
        self._reward_key_pos_scale = env_config.get("reward_key_pos_scale")

        # Foot-strike impulse penalty (set weight to 0 to disable).
        self._reward_impulse_w = env_config.get("reward_impulse_w", 0.0)
        self._reward_impulse_scale = env_config.get("reward_impulse_scale", 1e-5)
        self._reward_impulse_threshold = env_config.get("reward_impulse_threshold", 50.0)
        self._impulse_penalty_bodies = env_config.get("impulse_penalty_bodies",
                                                      ["right_foot", "left_foot"])

        # Metabolic cost-of-transport bonus (set weight to 0 to disable). Adds
        # an additive term reward_cot_w * exp(-reward_cot_scale * cot) to the
        # tracking reward, where cot = sum(|tau_i * dq_i|) / (m * g * v).
        # `v` is the horizontal speed of the root, clamped from below by
        # reward_cot_min_speed to avoid blowups when the agent is near rest.
        self._reward_cot_w = env_config.get("reward_cot_w", 0.0)
        self._reward_cot_scale = env_config.get("reward_cot_scale", 0.5)
        self._reward_cot_min_speed = env_config.get("reward_cot_min_speed", 0.5)

        # Vertical-impulse penalty (set weight to 0 to disable). Integrates
        # the per-foot vertical GRF over each ground-contact bout (heel-strike
        # to toe-off; accumulator resets at the next heel-strike). The peak
        # across feet is mapped through exp(-scale * I) and added to the
        # tracking reward with weight reward_vert_impulse_w.
        self._reward_vert_impulse_w = env_config.get("reward_vert_impulse_w", 0.0)
        self._reward_vert_impulse_scale = env_config.get("reward_vert_impulse_scale", 0.01)
        self._reward_vert_impulse_threshold = env_config.get("reward_vert_impulse_threshold", 50.0)
        self._vert_impulse_penalty_bodies = env_config.get("vert_impulse_penalty_bodies",
                                                            ["right_foot", "left_foot"])

        # Center-of-mass over support reward (set weight to 0 to disable). Adds
        # reward_com_support_w * exp(-reward_com_support_scale * d^2) to the
        # tracking reward, where d is the horizontal distance from the
        # whole-body COM to the centroid of `com_support_bodies`. For a
        # quasi-static, mirror-symmetric hold, static moment balance ties the
        # per-contact load asymmetry to this COM offset, so driving it to zero
        # equalizes the contact forces — using kinematics only (no force sensing).
        self._reward_com_support_w = env_config.get("reward_com_support_w", 0.0)
        self._reward_com_support_scale = env_config.get("reward_com_support_scale", 10.0)
        self._com_support_bodies = env_config.get("com_support_bodies", [])

        # Contact-force balance reward (set weight to 0 to disable). Adds
        # reward_force_balance_w * exp(-reward_force_balance_scale * c), where
        # c = sum_i ||F_i||^2 / W^2 over `force_balance_bodies` (W = body
        # weight). This convex cost is minimized, under the weight-support
        # constraint the tracking reward imposes, by the minimum-norm force
        # distribution: equal load sharing across symmetric contacts and no
        # wasteful internal/shear forces.
        self._reward_force_balance_w = env_config.get("reward_force_balance_w", 0.0)
        self._reward_force_balance_scale = env_config.get("reward_force_balance_scale", 1.0)
        self._force_balance_bodies = env_config.get("force_balance_bodies", [])

        # Energy penalty (set weight to 0 to disable). Adds
        # reward_energy_w * exp(-reward_energy_scale * e), where
        # e = sum_i |tau_i * dq_i| is the total mechanical power (W). This
        # penalizes torque-thrashing control; quasi-static torque abuse is
        # bounded separately by the actuator effort limits.
        self._reward_energy_w = env_config.get("reward_energy_w", 0.0)
        self._reward_energy_scale = env_config.get("reward_energy_scale", 1e-3)

        self._visualize_ref_char = env_config.get("visualize_ref_char", True)
        
        super().__init__(config=config, num_envs=num_envs, device=device,
                         visualize=visualize)
        
        return
    
    def get_reward_succ(self):
        # setting the done flag flat to fail at the end of the motion avoids the
        # local minimal of a character just standing still until the end of the motion
        return 0.0
    
    def get_reward_fail(self):
        return 0.0
    
    def set_mode(self, mode):
        super().set_mode(mode)

        if (self._mode == base_env.EnvMode.TRAIN):
            # Reset the per-iter reward-term tracker at the start of each
            # train rollout so its averages reflect only this iteration's
            # data (which is what the policy update actually consumed).
            if (hasattr(self, "_reward_term_tracker")):
                self._reward_term_tracker.reset()

        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):
                self._error_tracker.reset()

            # Reset accumulated logs every time we re-enter test mode and
            # arrange to flush them on process exit so users can Ctrl-C safely.
            self._ground_contact_forces_log = []
            self._joint_torques_log = []
            self._dof_pos_log = []
            self._dof_vel_log = []
            self._body_pos_log = []
            self._time_log = []
            self._log_step_count = 0
            if (self._log_ground_contact_forces or self._log_joint_torques) \
                    and not self._log_atexit_registered:
                atexit.register(self._save_logs)
                self._log_atexit_registered = True

        return

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        
        num_envs = self.get_num_envs()
        self._motion_ids = torch.zeros(num_envs, device=self._device, dtype=torch.int64)
        self._motion_time_offsets = torch.zeros(num_envs, device=self._device, dtype=torch.float32)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        body_rot = self._engine.get_body_rot(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)

        self._ref_root_pos = torch.zeros_like(root_pos)
        self._ref_root_rot = torch.zeros_like(root_rot)
        self._ref_root_vel = torch.zeros_like(root_vel)
        self._ref_root_ang_vel = torch.zeros_like(root_ang_vel)
        self._ref_body_pos = torch.zeros_like(body_pos)
        self._ref_body_rot = torch.zeros_like(body_rot)
        self._ref_joint_rot = torch.zeros_like(body_rot[..., 1:, :])
        self._ref_dof_pos = torch.zeros_like(dof_pos) 
        self._ref_dof_vel = torch.zeros_like(dof_vel)
  
        env_config = config["env"]
        contact_bodies = env_config.get("contact_bodies", [])
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        # Bodies whose foot-strike impulse we penalize when reward_impulse_w > 0.
        # When the penalty is disabled, skip the body-name lookup entirely so
        # configs that don't define impulse_penalty_bodies (and run on skeletons
        # without the default body names) still load. Downstream reward/reset
        # code already guards on _impulse_body_ids.shape[0] > 0.
        impulse_bodies = self._impulse_penalty_bodies if self._reward_impulse_w > 0.0 else []
        self._impulse_body_ids = self._build_body_ids_tensor(impulse_bodies)
        num_envs = self.get_num_envs()
        self._prev_contact_force_z = torch.zeros([num_envs, len(self._impulse_body_ids)],
                                                 device=self._device, dtype=torch.float32)
        self._step_impulse_penalty = torch.zeros([num_envs], device=self._device, dtype=torch.float32)

        # Per-bout vertical-impulse accumulator. Each entry integrates F_z*dt
        # while the foot is in contact and is reset on the next heel-strike.
        # Disabled => no body lookup (see note above).
        vert_impulse_bodies = (self._vert_impulse_penalty_bodies
                               if self._reward_vert_impulse_w > 0.0 else [])
        self._vert_impulse_body_ids = self._build_body_ids_tensor(vert_impulse_bodies)
        self._curr_bout_vert_impulse = torch.zeros(
            [num_envs, len(self._vert_impulse_body_ids)],
            device=self._device, dtype=torch.float32)
        self._prev_vert_impulse_in_contact = torch.zeros(
            [num_envs, len(self._vert_impulse_body_ids)],
            device=self._device, dtype=torch.bool)

        # Support bodies for the COM-over-support reward and contact bodies for
        # the force-balance reward. Both skip the name lookup when disabled;
        # downstream reward code guards on *_body_ids.shape[0] > 0. Stateless
        # terms (computed each step), so no reset bookkeeping is needed.
        com_support_bodies = self._com_support_bodies if self._reward_com_support_w > 0.0 else []
        self._com_support_body_ids = self._build_body_ids_tensor(com_support_bodies)
        force_balance_bodies = self._force_balance_bodies if self._reward_force_balance_w > 0.0 else []
        self._force_balance_body_ids = self._build_body_ids_tensor(force_balance_bodies)

        # Cache m*g for the cost-of-transport reward. Assumes all envs share
        # the same character (true today since char_file is global), so a
        # single scalar covers every env.
        char_mass = float(self._engine.calc_obj_mass(0, char_id))
        self._char_weight = char_mass * 9.81

        # Per-body masses (common body order) cached for the offline
        # mass-weighted center-of-mass computation in the diagnostics plotter.
        self._body_masses = self._engine.get_body_masses(0, char_id)
        # Normalized on-device per-body mass weights for the online
        # COM-over-support reward (einsum against body_pos each step).
        com_masses = self._body_masses.to(self._device).float()
        self._com_body_weights = com_masses / torch.clamp(com_masses.sum(), min=1e-8)

        joint_err_w = env_config.get("joint_err_w", None)
        self._parse_joint_err_weights(joint_err_w)

        # Contact-observation buffers (only built when the feature is on).
        if (self._enable_contact_obs):
            assert(len(self._obs_contact_bodies) > 0), \
                "enable_contact_obs requires a non-empty obs_contact_bodies list"
            self._obs_contact_body_ids = self._build_body_ids_tensor(self._obs_contact_bodies)
            num_contact_bodies = self._obs_contact_body_ids.shape[0]

            # Fixed fan of horizontal unit directions for the batched
            # support-function evaluation of the support-polygon margin.
            num_dirs = 16
            angles = (2.0 * np.pi / num_dirs) * torch.arange(num_dirs, device=self._device,
                                                             dtype=torch.float32)
            self._support_polygon_dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

            # All unordered body pairs for the contact graph.
            pair_ids = torch.triu_indices(num_contact_bodies, num_contact_bodies, offset=1,
                                          device=self._device)
            self._contact_pair_ids_i = pair_ids[0]
            self._contact_pair_ids_j = pair_ids[1]

            # Contact sensor data is not refreshed when envs are reset without
            # stepping physics, so the first observation after a reset would
            # otherwise contain the previous episode's contact forces. Envs
            # flagged stale get their contact observations zeroed until the
            # next physics step.
            self._contact_obs_stale = torch.zeros(num_envs, device=self._device, dtype=torch.bool)

        return

    def _load_motions(self, motion_file):
        self._motion_lib = motion_lib.MotionLib(motion_file=motion_file,
                                                kin_char_model=self._kin_char_model,
                                                device=self._device,
                                                auto_ground_offset=self._auto_ground_offset,
                                                ground_offset_clearance=self._ground_offset_clearance,
                                                ground_offset=self._ground_offset,
                                                char_file=self._char_file)

        if (self._init_time_range is not None):
            min_len = self._motion_lib.get_motion_lengths().min().item()
            assert(self._init_time_range[0] < min_len), \
                ("init_time_range[0] ({:.3f}s) >= shortest clip length ({:.3f}s); "
                 "reference-state init would collapse to the clip end").format(
                    self._init_time_range[0], min_len)

        return
    
    def _parse_joint_err_weights(self, joint_err_w):
        num_joints = self._kin_char_model.get_num_joints()

        if (joint_err_w is None):
            self._joint_err_w = torch.ones(num_joints - 1, device=self._device, dtype=torch.float32)
        else:
            self._joint_err_w = torch.tensor(joint_err_w, device=self._device, dtype=torch.float32)

        assert(self._joint_err_w.shape[-1] == num_joints - 1)
        
        dof_size = self._kin_char_model.get_dof_size()
        self._dof_err_w = torch.zeros(dof_size, device=self._device, dtype=torch.float32)

        for j in range(1, num_joints):
            dof_dim = self._kin_char_model.get_joint_dof_dim(j)
            if (dof_dim > 0):
                curr_w = self._joint_err_w[j - 1]
                dof_idx = self._kin_char_model.get_joint_dof_idx(j)
                self._dof_err_w[dof_idx:dof_idx + dof_dim] = curr_w

        return
    
    def _enable_ref_char(self):
        return self._visualize and self._visualize_ref_char

    def _get_ref_char_color(self):
        engine_name = self._engine.get_name()
        if (engine_name == "isaac_lab"):
            col = np.array([0.25, 0.4, 0.1])
        else:
            col = np.array([0.5, 0.9, 0.1])
        return col

    def _reset_char(self, env_ids):
        self._reset_ref_motion(env_ids)
        self._ref_state_init(env_ids)

        if (self._enable_ref_char()):
            self._reset_ref_char(env_ids)

        # Treat post-reset feet as already in contact so the first step does
        # not fire a spurious strike event in the impulse penalty.
        if (hasattr(self, "_prev_contact_force_z") and self._impulse_body_ids.shape[0] > 0):
            self._prev_contact_force_z[env_ids] = 1.0e6

        # Reset the per-bout vertical-impulse accumulator. Marking the foot
        # as already in contact suppresses a spurious heel-strike event on
        # the first post-reset step.
        if (hasattr(self, "_curr_bout_vert_impulse")
                and self._vert_impulse_body_ids.shape[0] > 0):
            self._curr_bout_vert_impulse[env_ids] = 0.0
            self._prev_vert_impulse_in_contact[env_ids] = True

        # The contact sensor still holds the pre-reset pose's forces until the
        # next physics step, so contact observations for these envs are zeroed
        # until then.
        if (self._enable_contact_obs):
            self._contact_obs_stale[env_ids] = True

        return

    def _reset_ref_char(self, env_ids):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos[env_ids] + self._ref_char_offset
        self._engine.set_root_pos(env_ids, ref_char_id, root_pos)
        self._engine.set_root_rot(env_ids, ref_char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, ref_char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, ref_char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, ref_char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, ref_char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, ref_char_id, 0.0)
        return

    def _reset_ref_motion(self, env_ids):
        n = len(env_ids)
        motion_ids, motion_times = self._sample_motion_times(n)
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)

        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_joint_rot[env_ids] = joint_rot
        self._ref_dof_vel[env_ids] = dof_vel
        
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
        self._ref_dof_pos[env_ids] = dof_pos

        return

    def _get_ref_char_id(self):
        return self._ref_char_ids[0]

    def _ref_state_init(self, env_ids):
        char_id = self._get_char_id()
        
        self._engine.set_root_pos(env_ids, char_id, self._ref_root_pos[env_ids])
        self._engine.set_root_rot(env_ids, char_id, self._ref_root_rot[env_ids])
        self._engine.set_root_vel(env_ids, char_id, self._ref_root_vel[env_ids])
        self._engine.set_root_ang_vel(env_ids, char_id, self._ref_root_ang_vel[env_ids])
        
        self._engine.set_dof_pos(env_ids, char_id, self._ref_dof_pos[env_ids])
        self._engine.set_dof_vel(env_ids, char_id, self._ref_dof_vel[env_ids])
        
        self._engine.set_body_vel(env_ids, char_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, char_id, 0.0)

        return

    def _get_motion_times(self, env_ids=None):
        if (env_ids is None):
            motion_times = self._time_buf + self._motion_time_offsets
        else:
            motion_times = self._time_buf[env_ids] + self._motion_time_offsets[env_ids]
        return motion_times

    def _update_misc(self):
        super()._update_misc()
        self._update_ref_motion()

        if (self._enable_ref_char()):
            self._update_ref_char()

        # _update_misc only runs post-physics, so contact forces are fresh
        # again for every env.
        if (self._enable_contact_obs):
            self._contact_obs_stale[:] = False

        return
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_joint_rot[:] = joint_rot
        self._ref_dof_vel[:] = dof_vel

        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(self._ref_root_pos, self._ref_root_rot,
                                                                             self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot

        if (self._enable_ref_char()):
            dof_pos = self._motion_lib.joint_rot_to_dof(joint_rot)
            self._ref_dof_pos[:] = dof_pos

        return

    def _update_ref_char(self):
        ref_char_id = self._get_ref_char_id()

        root_pos = self._ref_root_pos + self._ref_char_offset
        body_pos = self._ref_body_pos + self._ref_char_offset

        self._engine.set_root_pos(None, ref_char_id, root_pos)
        self._engine.set_root_rot(None, ref_char_id, self._ref_root_rot)
        self._engine.set_root_vel(None, ref_char_id, 0.0)
        self._engine.set_root_ang_vel(None, ref_char_id, 0.0)
        
        self._engine.set_dof_pos(None, ref_char_id, self._ref_dof_pos)
        self._engine.set_dof_vel(None, ref_char_id, 0.0)

        self._engine.set_body_pos(None, ref_char_id, body_pos)
        self._engine.set_body_rot(None, ref_char_id, self._ref_body_rot)
        self._engine.set_body_vel(None, ref_char_id, 0.0)
        self._engine.set_body_ang_vel(None, ref_char_id, 0.0)
        return
    
    def _track_global_root(self):
        return self._enable_tar_obs and self._global_obs

    def _sample_motion_times(self, n):
        motion_ids = self._motion_lib.sample_motions(n)

        if (self._rand_reset):
            if (self._init_time_range is not None):
                motion_len = self._motion_lib.get_motion_length(motion_ids)
                lo = torch.clamp(torch.full_like(motion_len, self._init_time_range[0]),
                                 max=motion_len)
                hi = torch.clamp(torch.full_like(motion_len, self._init_time_range[1]),
                                 max=motion_len)
                rand_phase = torch.rand(n, dtype=motion_len.dtype, device=self._device)
                motion_times = lo + rand_phase * (hi - lo)
            else:
                motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(n, dtype=torch.float, device=self._device)

        return motion_ids, motion_times

    def _build_data_buffers(self):
        super()._build_data_buffers()

        if (self._log_tracking_error):
            num_track_errors = 7
            self._error_tracker = stats_tracker.StatsTracker(num_track_errors, device=self._device)

        # Per-step diagnostic logs. Each is a list of CPU tensors that gets
        # stacked along time when flushed to disk.
        self._ground_contact_forces_log = []
        self._joint_torques_log = []
        self._dof_pos_log = []
        self._dof_vel_log = []
        self._body_pos_log = []
        self._time_log = []
        self._log_step_count = 0
        self._log_atexit_registered = False

        # Iter-level reward-term means. Accumulates per-step means during
        # train rollouts and is published into _diagnostics by
        # get_diagnostics() (which the agent calls once per iter at log
        # time). Lets wandb show reward_term/pose_r, reward_term/cot, ...
        self._reward_term_tracker = _RewardTermTracker(self._device)

        return
    
    def _build_envs(self, config, num_envs):
        self._ref_char_ids = []

        super()._build_envs(config, num_envs)

        self._char_file = config["env"]["char_file"]
        motion_file = config["env"]["motion_file"]
        self._load_motions(motion_file)
        return
    
    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        if (self._enable_ref_char()):
            ref_char_col = self._get_ref_char_color()
            ref_char_id = self._build_ref_character(env_id, config, color=ref_char_col)
            self._ref_char_ids.append(ref_char_id)
            
            if (env_id == 0):
                self._ref_char_ids.append(ref_char_id)
            else:
                ref_char_id0 = self._ref_char_ids[0]
                assert(ref_char_id0 == ref_char_id)
        
        return 
    
    def _build_ref_character(self, env_id, config, color):
        char_file = config["env"]["char_file"]
        char_id = self._engine.create_obj(env_id=env_id, 
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file, 
                                          name="ref_character",
                                          is_visual=True,
                                          enable_self_collisions=False,
                                          disable_motors=True,
                                          color=color)
        return char_id

    def _compute_obs(self, env_ids=None):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times(env_ids)
        
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

            motion_ids = motion_ids[env_ids]
            
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        
        if (self._enable_phase_obs):
            motion_phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        else:
            motion_phase = torch.zeros([0], device=self._device)

        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)

        if (self._enable_tar_obs):
            tar_root_pos, tar_root_rot, tar_joint_rot = self._fetch_tar_obs_data(motion_ids, motion_times)
            tar_root_pos_flat = torch.reshape(tar_root_pos, [tar_root_pos.shape[0] * tar_root_pos.shape[1], 
                                                             tar_root_pos.shape[-1]])
            tar_root_rot_flat = torch.reshape(tar_root_rot, [tar_root_rot.shape[0] * tar_root_rot.shape[1], 
                                                             tar_root_rot.shape[-1]])
            tar_joint_rot_flat = torch.reshape(tar_joint_rot, [tar_joint_rot.shape[0] * tar_joint_rot.shape[1], 
                                                               tar_joint_rot.shape[-2], tar_joint_rot.shape[-1]])
            tar_body_pos_flat, _ = self._kin_char_model.forward_kinematics(tar_root_pos_flat, tar_root_rot_flat,
                                                                           tar_joint_rot_flat)
            tar_body_pos = torch.reshape(tar_body_pos_flat, [tar_root_pos.shape[0], tar_root_pos.shape[1], 
                                                             tar_body_pos_flat.shape[-2], tar_body_pos_flat.shape[-1]])

            if (self._has_key_bodies()):
                tar_key_pos = tar_body_pos[..., self._key_body_ids, :]
            else:
                tar_key_pos = torch.zeros([0], device=self._device)
        else:
            tar_root_pos = torch.zeros([0], device=self._device)
            tar_root_rot = tar_root_pos
            tar_joint_rot = tar_root_pos
            tar_key_pos = tar_root_pos

        obs = compute_deepmimic_obs(root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel,
                                    joint_rot=joint_rot,
                                    dof_vel=dof_vel,
                                    key_pos=key_pos,
                                    global_obs=self._global_obs,
                                    root_height_obs=self._root_height_obs,
                                    phase=motion_phase,
                                    num_phase_encoding=self._num_phase_encoding,
                                    enable_phase_obs=self._enable_phase_obs,
                                    enable_tar_obs=self._enable_tar_obs,
                                    tar_root_pos=tar_root_pos,
                                    tar_root_rot=tar_root_rot,
                                    tar_joint_rot=tar_joint_rot,
                                    tar_key_pos=tar_key_pos)

        if (self._enable_contact_obs):
            contact_forces = self._engine.get_ground_contact_forces(char_id)
            contact_obs_stale = self._contact_obs_stale
            if (env_ids is not None):
                contact_forces = contact_forces[env_ids]
                contact_obs_stale = contact_obs_stale[env_ids]

            contact_obs = compute_contact_obs(body_pos=body_pos,
                                              root_rot=root_rot,
                                              contact_forces=contact_forces,
                                              contact_obs_stale=contact_obs_stale,
                                              contact_body_ids=self._obs_contact_body_ids,
                                              com_weights=self._com_body_weights,
                                              force_threshold=self._obs_contact_force_threshold,
                                              support_dirs=self._support_polygon_dirs,
                                              pair_ids_i=self._contact_pair_ids_i,
                                              pair_ids_j=self._contact_pair_ids_j,
                                              global_obs=self._global_obs,
                                              margin_cap=self._obs_contact_margin_cap)
            obs = torch.cat([obs, contact_obs], dim=-1)

        return obs
    
    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        if (self._has_key_bodies()):
            key_pos = body_pos[..., self._key_body_ids, :]
            ref_key_pos = self._ref_body_pos[..., self._key_body_ids, :]
        else:
            key_pos = torch.zeros([0], device=self._device)
            ref_key_pos = key_pos

        track_root_h = self._root_height_obs
        track_root = self._track_global_root()

        r, pose_r, vel_r, root_pose_r, root_vel_r, key_pos_r = compute_reward(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            key_pos=key_pos,

            tar_root_pos=self._ref_root_pos,
            tar_root_rot=self._ref_root_rot,
            tar_root_vel=self._ref_root_vel,
            tar_root_ang_vel=self._ref_root_ang_vel,
            tar_joint_rot=self._ref_joint_rot,
            tar_dof_vel=self._ref_dof_vel,
            tar_key_pos=ref_key_pos,

            joint_rot_err_w=self._joint_err_w,
            dof_err_w=self._dof_err_w,
            track_root_h=track_root_h,
            track_root=track_root,

            pose_w=self._reward_pose_w,
            vel_w=self._reward_vel_w,
            root_pose_w=self._reward_root_pose_w,
            root_vel_w=self._reward_root_vel_w,
            key_pos_w=self._reward_key_pos_w,

            pose_scale=self._reward_pose_scale,
            vel_scale=self._reward_vel_scale,
            root_pose_scale=self._reward_root_pose_scale,
            root_vel_scale=self._reward_root_vel_scale,
            key_pos_scale=self._reward_key_pos_scale)
        self._reward_buf[:] = r

        # Per-step reward components for the iter-average tracker. Optional
        # add-on terms (cot_r, vert_impulse_r) are appended below if active.
        reward_components = {
            "pose_r": pose_r,
            "vel_r": vel_r,
            "root_pose_r": root_pose_r,
            "root_vel_r": root_vel_r,
            "key_pos_r": key_pos_r,
        }

        self._apply_aux_rewards(char_id, root_vel, dof_vel, body_pos, reward_components)
        return

    def _apply_aux_rewards(self, char_id, root_vel, dof_vel, body_pos, reward_components):
        # Additive shaping terms (all yaml-gated, default off), applied on top
        # of whatever is already in self._reward_buf. Shared with AMPEnv,
        # whose task reward is exactly these terms over a zeroed buffer.
        # Publishes reward_total and feeds the iter-average reward tracker.

        # Cache GRF once if any GRF-dependent term is active.
        ground_contact_forces = None
        need_gcf = (self._reward_vert_impulse_w > 0.0 and self._vert_impulse_body_ids.shape[0] > 0) \
                   or (self._reward_impulse_w > 0.0 and self._impulse_body_ids.shape[0] > 0) \
                   or (self._reward_force_balance_w > 0.0 and self._force_balance_body_ids.shape[0] > 0)
        if (need_gcf):
            ground_contact_forces = self._engine.get_ground_contact_forces(char_id)

        if (self._reward_cot_w > 0.0):
            self._reward_buf[:], cot, cot_r = self._apply_cot_reward(self._reward_buf,
                                                                      char_id, root_vel, dof_vel)
            reward_components["cot"] = cot
            reward_components["cot_r"] = cot_r

        if (self._reward_vert_impulse_w > 0.0 and self._vert_impulse_body_ids.shape[0] > 0):
            self._reward_buf[:], peak_imp, vert_impulse_r = self._apply_vert_impulse_penalty(
                self._reward_buf, ground_contact_forces)
            reward_components["vert_impulse_peak"] = peak_imp
            reward_components["vert_impulse_r"] = vert_impulse_r

        if (self._reward_impulse_w > 0.0 and self._impulse_body_ids.shape[0] > 0):
            self._reward_buf[:], strike_imp = self._apply_impulse_penalty(
                self._reward_buf, ground_contact_forces)
            reward_components["foot_strike_impulse"] = strike_imp

        if (self._reward_com_support_w > 0.0 and self._com_support_body_ids.shape[0] > 0):
            self._reward_buf[:], com_offset, com_support_r = self._apply_com_support_reward(
                self._reward_buf, body_pos)
            reward_components["com_support_offset"] = com_offset
            reward_components["com_support_r"] = com_support_r

        if (self._reward_force_balance_w > 0.0 and self._force_balance_body_ids.shape[0] > 0):
            self._reward_buf[:], fb_cost, force_balance_r = self._apply_force_balance_reward(
                self._reward_buf, ground_contact_forces)
            reward_components["force_balance_cost"] = fb_cost
            reward_components["force_balance_r"] = force_balance_r

        if (self._reward_energy_w > 0.0):
            self._reward_buf[:], energy, energy_r = self._apply_energy_reward(
                self._reward_buf, char_id, dof_vel)
            reward_components["energy"] = energy
            reward_components["energy_r"] = energy_r

        # Total weighted reward (post all transformations) for inspection.
        reward_components["reward_total"] = self._reward_buf

        # Accumulate iter-level reward-term means across the train rollout.
        # The tracker is reset on set_mode(TRAIN), and its means are pulled
        # by get_diagnostics at log time. Test-rollout steps are skipped so
        # the published numbers reflect the data the policy was trained on.
        if (self._mode == base_env.EnvMode.TRAIN):
            self._reward_term_tracker.update(reward_components)

        return

    def _apply_cot_reward(self, reward_buf, char_id, root_vel, dof_vel):
        # Approximate metabolic Cost of Transport as
        #   cot = sum(|tau_i * dq_i|) / (m * g * v),
        # with v the horizontal-plane root speed clamped from below to avoid
        # divide-by-zero when the character is near rest. The reward is
        # exp(-scale * cot) (in [0, 1]) added to the existing tracking reward.
        # Returns (new_reward, cot, cot_r) so the caller can feed `cot` and
        # `cot_r` into the iter-average reward-term tracker.
        dof_torques = self._engine.get_dof_forces(char_id)
        mech_power = torch.sum(torch.abs(dof_torques * dof_vel), dim=-1)

        horiz_speed = torch.linalg.vector_norm(root_vel[..., :2], dim=-1)
        horiz_speed = torch.clamp(horiz_speed, min=self._reward_cot_min_speed)

        cot = mech_power / (self._char_weight * horiz_speed)
        cot_r = torch.exp(-self._reward_cot_scale * cot)

        return reward_buf + self._reward_cot_w * cot_r, cot, cot_r

    def _apply_energy_reward(self, reward_buf, char_id, dof_vel):
        # Total mechanical power e = sum(|tau_i * dq_i|) (W), rewarded as
        # exp(-scale * e) in [0, 1]. The torques are the actuator-model
        # values (post effort-limit clipping). Returns (new_reward, energy,
        # energy_r) so the caller can feed both into the iter-average
        # reward-term tracker.
        dof_torques = self._engine.get_dof_forces(char_id)
        energy = torch.sum(torch.abs(dof_torques * dof_vel), dim=-1)
        energy_r = torch.exp(-self._reward_energy_scale * energy)
        return reward_buf + self._reward_energy_w * energy_r, energy, energy_r

    def _apply_vert_impulse_penalty(self, reward_buf, ground_contact_forces):
        # Maintain a per-foot accumulator for the integral of F_z * dt across
        # the current ground-contact bout. The accumulator resets at heel-
        # strike (false->true contact transition) and stays put while the
        # foot is in flight, so the most recent bout's impulse is held until
        # the next heel-strike. Penalty signal at every step is the peak
        # across feet, mapped through exp(-scale * I). Returns (new_reward,
        # peak_impulse, vert_impulse_r) for the iter-average tracker.
        foot_force_z = ground_contact_forces[:, self._vert_impulse_body_ids, 2]
        in_contact = foot_force_z >= self._reward_vert_impulse_threshold

        heel_strike = torch.logical_and(~self._prev_vert_impulse_in_contact, in_contact)
        self._curr_bout_vert_impulse = torch.where(
            heel_strike,
            torch.zeros_like(self._curr_bout_vert_impulse),
            self._curr_bout_vert_impulse)

        dt = self._engine.get_timestep()
        increment = torch.where(in_contact, foot_force_z * dt,
                                torch.zeros_like(foot_force_z))
        self._curr_bout_vert_impulse = self._curr_bout_vert_impulse + increment

        self._prev_vert_impulse_in_contact = in_contact.detach().clone()

        peak_impulse = torch.max(self._curr_bout_vert_impulse, dim=-1)[0]
        vert_impulse_r = torch.exp(-self._reward_vert_impulse_scale * peak_impulse)

        return (reward_buf + self._reward_vert_impulse_w * vert_impulse_r,
                peak_impulse, vert_impulse_r)

    def _apply_impulse_penalty(self, reward_buf, ground_contact_forces):
        # Detect a foot strike as a low->high transition in vertical GRF, then
        # penalize the magnitude of the GRF spike at the transition. Keeping the
        # detection event-based (rather than a per-step force penalty) keeps the
        # signal aligned with the biomechanical notion of impulse at impact.
        foot_forces = ground_contact_forces[:, self._impulse_body_ids, :]
        foot_force_z = foot_forces[..., 2]

        thresh = self._reward_impulse_threshold
        prev_below = self._prev_contact_force_z < thresh
        curr_above = foot_force_z >= thresh
        strike = torch.logical_and(prev_below, curr_above)

        force_mag = torch.linalg.vector_norm(foot_forces, dim=-1)
        impulse_mag = torch.where(strike, force_mag, torch.zeros_like(force_mag))
        # Sum across feet so a double-strike still gets penalized.
        per_env_impulse = impulse_mag.sum(dim=-1)

        impulse_r = torch.exp(-self._reward_impulse_scale * per_env_impulse)
        # Convex blend: shave off some of the tracking reward proportional to
        # how violent the strike was, normalized by reward_impulse_w.
        blended = (1.0 - self._reward_impulse_w) * reward_buf \
                  + self._reward_impulse_w * impulse_r * reward_buf

        self._prev_contact_force_z = foot_force_z.detach().clone()
        self._step_impulse_penalty = per_env_impulse.detach().clone()
        return blended, per_env_impulse

    def _apply_com_support_reward(self, reward_buf, body_pos):
        # Reward keeping the horizontal whole-body COM projection over the
        # centroid of the support bodies. For a quasi-static, mirror-symmetric
        # hold, static moment balance makes the per-contact load asymmetry
        # proportional to this COM offset, so minimizing it equalizes the
        # contact forces with no force sensing. Returns (new_reward,
        # offset_sq, com_support_r) for the iter-average reward-term tracker.
        com = torch.einsum("nbk,b->nk", body_pos, self._com_body_weights)
        support_xy = body_pos[:, self._com_support_body_ids, :2].mean(dim=1)
        offset_sq = torch.sum((com[..., :2] - support_xy) ** 2, dim=-1)
        com_support_r = torch.exp(-self._reward_com_support_scale * offset_sq)
        return reward_buf + self._reward_com_support_w * com_support_r, offset_sq, com_support_r

    def _apply_force_balance_reward(self, reward_buf, ground_contact_forces):
        # Penalize the sum of squared contact-force magnitudes over the listed
        # bodies, normalized by body weight squared so the cost is
        # dimensionless. Under the weight-support constraint the tracking
        # reward imposes, this convex cost is minimized by the minimum-norm
        # force distribution: equal load sharing across symmetric contacts and
        # no wasteful internal/shear forces. Returns (new_reward, cost,
        # force_balance_r) for the iter-average reward-term tracker.
        forces = ground_contact_forces[:, self._force_balance_body_ids, :]
        cost = torch.sum(forces * forces, dim=(-2, -1)) / (self._char_weight ** 2)
        force_balance_r = torch.exp(-self._reward_force_balance_scale * cost)
        return reward_buf + self._reward_force_balance_w * force_balance_r, cost, force_balance_r

    def _update_done(self):
        motion_times = self._get_motion_times()
        motion_len = self._motion_lib.get_motion_length(self._motion_ids)
        motion_loop_mode = self._motion_lib.get_motion_loop_mode(self._motion_ids)
        motion_len_term = motion_loop_mode != motion.LoopMode.WRAP.value

        track_root = self._track_global_root()
        
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        ground_contact_forces = self._engine.get_ground_contact_forces(char_id)

        body_names = self._kin_char_model.get_body_names()
        # if self._contact_body_ids.shape[0] > 0:
        #     contact_body_names = [body_names[i] for i in self._contact_body_ids]
        #     masked_contact_buf = ground_contact_forces.detach().clone()
        #     masked_contact_buf[:, self._contact_body_ids, :] = 0
        #     print(f"contact_body_names: {contact_body_names}")

        self._done_buf[:] = compute_done(done_buf=self._done_buf,
                                         time=self._time_buf, 
                                         ep_len=self._episode_length, 
                                         root_rot=root_rot,
                                         body_pos=body_pos,
                                         tar_root_rot=self._ref_root_rot,
                                         tar_body_pos=self._ref_body_pos,
                                         ground_contact_force=ground_contact_forces,
                                         contact_body_ids=self._contact_body_ids,
                                         pose_termination=self._pose_termination,
                                         pose_termination_dist=self._pose_termination_dist,
                                         global_obs=self._global_obs,
                                         enable_early_termination=self._enable_early_termination,
                                         motion_times=motion_times,
                                         motion_len=motion_len,
                                         motion_len_term=motion_len_term,
                                         track_root=track_root,)
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)

        if (self._mode == base_env.EnvMode.TEST):
            if (self._log_tracking_error):
                self._record_tracking_error(env_ids)
            if (self._log_ground_contact_forces):
                self._record_ground_contact_forces(env_ids)
            if (self._log_joint_torques):
                self._record_joint_torques(env_ids)

            if (self._log_ground_contact_forces or self._log_joint_torques):
                # Match the env_ids gate used inside _record_*. _update_info
                # is called both from step() (env_ids=None) and from reset()
                # with env_ids that may be an empty tensor when no env
                # terminated; the GRF/torque recorders skip that case, so
                # the time log must skip it too or it drifts to ~2x.
                if (env_ids is None or len(env_ids) > 0):
                    self._time_log.append(self._time_buf.detach().clone().cpu())
                    # Body positions are recorded here (paired with time)
                    # so derived metrics (COM velocity, stride length,
                    # cost of transport) can be reconstructed offline
                    # whenever either logging flag is on.
                    char_id = self._get_char_id()
                    body_pos = self._engine.get_body_pos(char_id)
                    self._body_pos_log.append(body_pos.detach().clone().cpu())
                    self._log_step_count += 1
                    if (self._log_save_interval > 0
                            and self._log_step_count % self._log_save_interval == 0):
                        self._save_logs()

        return

    def get_diagnostics(self):
        # Publish per-iter reward-term means (then reset the tracker so the
        # next iter accumulates afresh). The agent calls this once per iter
        # at log time, so this is a natural read-with-reset boundary.
        if (hasattr(self, "_reward_term_tracker")
                and self._reward_term_tracker.has_data()):
            for k, v in self._reward_term_tracker.get_means().items():
                self._diagnostics["reward_term/{}".format(k)] = v
            self._reward_term_tracker.reset()
        return self._diagnostics
    
    def _record_tracking_error(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            char_id = self._get_char_id()
            root_pos = self._engine.get_root_pos(char_id)
            root_rot = self._engine.get_root_rot(char_id)
            root_vel = self._engine.get_root_vel(char_id)
            root_ang_vel = self._engine.get_root_ang_vel(char_id)
            dof_pos = self._engine.get_dof_pos(char_id)
            dof_vel = self._engine.get_dof_vel(char_id)
            body_pos = self._engine.get_body_pos(char_id)
            body_rot = self._engine.get_body_rot(char_id)

            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

            ref_root_pos = self._ref_root_pos
            ref_root_rot = self._ref_root_rot
            ref_joint_rot = self._ref_joint_rot
            ref_root_vel = self._ref_root_vel
            ref_root_ang_vel = self._ref_root_ang_vel
            ref_dof_vel = self._ref_dof_vel

            if env_ids is not None:
                root_pos = root_pos[env_ids]
                root_rot = root_rot[env_ids]
                joint_rot = joint_rot[env_ids]
                root_vel = root_vel[env_ids]
                root_ang_vel = root_ang_vel[env_ids]
                dof_vel = dof_vel[env_ids]
                body_pos = body_pos[env_ids]
                body_rot = body_rot[env_ids]

                ref_root_pos = ref_root_pos[env_ids]
                ref_root_rot = ref_root_rot[env_ids]
                ref_joint_rot = ref_joint_rot[env_ids]
                ref_root_vel = ref_root_vel[env_ids]
                ref_root_ang_vel = ref_root_ang_vel[env_ids]
                ref_dof_vel = ref_dof_vel[env_ids]
            
            ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(ref_root_pos, ref_root_rot, ref_joint_rot)

            tracking_error = compute_tracking_error(root_pos=root_pos,
                                                    root_rot=root_rot,
                                                    body_rot=body_rot,
                                                    body_pos=body_pos,

                                                    tar_root_pos=ref_root_pos,
                                                    tar_root_rot=ref_root_rot,
                                                    tar_body_rot=ref_body_rot,
                                                    tar_body_pos=ref_body_pos,

                                                    root_vel=root_vel,
                                                    root_ang_vel=root_ang_vel,
                                                    dof_vel=dof_vel,
                                                    tar_dof_vel=ref_dof_vel,
                                                    tar_root_vel=ref_root_vel,
                                                    tar_root_ang_vel=ref_root_ang_vel)

            self._error_tracker.update(tracking_error)

            err_stats = self._error_tracker.get_mean()
            self._diagnostics["root_pos_err"] = err_stats[0]
            self._diagnostics["root_rot_err"] = err_stats[1]
            self._diagnostics["body_pos_err"] = err_stats[2]
            self._diagnostics["body_rot_err"] = err_stats[3]
            self._diagnostics["dof_vel_err"] = err_stats[4]
            self._diagnostics["root_vel_err"] = err_stats[5]
            self._diagnostics["root_ang_vel_err"] = err_stats[6]

        return
    
    def _record_ground_contact_forces(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            char_id = self._get_char_id()
            # Per-body ground contact forces in [num_envs, num_bodies, 3].
            ground_contact_forces = self._engine.get_ground_contact_forces(char_id)
            self._ground_contact_forces_log.append(ground_contact_forces.detach().clone().cpu())
            self._diagnostics["ground_contact_forces"] = ground_contact_forces

        return

    def _record_joint_torques(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            char_id = self._get_char_id()
            dof_torques = self._engine.get_dof_forces(char_id)
            dof_pos = self._engine.get_dof_pos(char_id)
            dof_vel = self._engine.get_dof_vel(char_id)
            self._joint_torques_log.append(dof_torques.detach().clone().cpu())
            self._dof_pos_log.append(dof_pos.detach().clone().cpu())
            self._dof_vel_log.append(dof_vel.detach().clone().cpu())
            self._diagnostics["dof_torques"] = dof_torques
        return

    def _save_logs(self):
        # Best-effort flush of accumulated diagnostic logs to disk. Safe to call
        # repeatedly (it overwrites the same file).
        if (len(self._ground_contact_forces_log) == 0
                and len(self._joint_torques_log) == 0):
            return

        try:
            os.makedirs(self._log_dir, exist_ok=True)
        except Exception:
            return

        body_names = self._kin_char_model.get_body_names()
        dof_names = []
        for j in range(1, self._kin_char_model.get_num_joints()):
            joint = self._kin_char_model.get_joint(j)
            dim = joint.get_dof_dim()
            if (dim == 1):
                dof_names.append(joint.name)
            else:
                for axis in range(dim):
                    dof_names.append("{}_{}".format(joint.name, axis))

        payload = {
            "body_names": body_names,
            "dof_names": dof_names,
            "control_freq": self._engine.get_control_freq() if hasattr(self._engine, "get_control_freq") else None,
            "timestep": self._engine.get_timestep() if hasattr(self._engine, "get_timestep") else None,
        }
        if (len(self._ground_contact_forces_log) > 0):
            payload["ground_contact_forces"] = torch.stack(self._ground_contact_forces_log, dim=0)
        if (len(self._joint_torques_log) > 0):
            payload["joint_torques"] = torch.stack(self._joint_torques_log, dim=0)
        if (len(self._dof_pos_log) > 0):
            payload["dof_pos"] = torch.stack(self._dof_pos_log, dim=0)
        if (len(self._dof_vel_log) > 0):
            payload["dof_vel"] = torch.stack(self._dof_vel_log, dim=0)
        if (len(self._body_pos_log) > 0):
            payload["body_pos"] = torch.stack(self._body_pos_log, dim=0)
        # Total character mass (kg). Saved so the offline plotter can
        # compute cost of transport without re-querying the engine.
        if (hasattr(self, "_char_weight")):
            payload["char_mass"] = float(self._char_weight / 9.81)
        # Per-body masses (kg) in body_names order, for the mass-weighted COM.
        if (hasattr(self, "_body_masses")):
            payload["body_masses"] = self._body_masses.detach().cpu()
        if (len(self._time_log) > 0):
            payload["time"] = torch.stack(self._time_log, dim=0)

        out_path = os.path.join(self._log_dir, "diagnostics_{}.pt".format(self._log_tag))
        torch.save(payload, out_path)
        return
    
    def _fetch_tar_obs_data(self, motion_ids, motion_times):
        n = motion_ids.shape[0]
        num_steps = self._tar_obs_steps.shape[0]
        assert(num_steps > 0)
        
        motion_times = motion_times.unsqueeze(-1)
        time_steps = self._engine.get_timestep() * self._tar_obs_steps
        motion_times = motion_times + time_steps
        motion_ids_tiled = torch.broadcast_to(motion_ids.unsqueeze(-1), motion_times.shape)

        motion_ids_tiled = motion_ids_tiled.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self._motion_lib.calc_motion_frame(motion_ids_tiled, motion_times)

        root_pos = root_pos.reshape([n, num_steps, root_pos.shape[-1]])
        root_rot = root_rot.reshape([n, num_steps, root_rot.shape[-1]])
        joint_rot = joint_rot.reshape([n, num_steps, joint_rot.shape[-2], joint_rot.shape[-1]])
        return root_pos, root_rot, joint_rot


class _RewardTermTracker:
    """Accumulates per-step batch means of reward components across one
    training rollout, then exposes the across-step means at iter end. Each
    update call takes a dict of {name: tensor[num_envs]} and folds in that
    step's batch mean. Reset clears all accumulators back to zero/empty.
    """

    def __init__(self, device):
        self._device = device
        self._sums = {}
        self._count = 0
        return

    def reset(self):
        # Clear the dict (not just zero it) so the next iter publishes only
        # the keys that were actually updated, even if the active reward
        # weights changed mid-run.
        self._sums = {}
        self._count = 0
        return

    def has_data(self):
        return self._count > 0

    def update(self, components):
        for k, v in components.items():
            if (not torch.is_tensor(v)):
                continue
            mean_step = v.detach().float().mean()
            if (k not in self._sums):
                self._sums[k] = torch.zeros((), device=self._device, dtype=torch.float32)
            self._sums[k] = self._sums[k] + mean_step
        self._count += 1
        return

    def get_means(self):
        if (self._count == 0):
            return {}
        denom = float(self._count)
        return {k: (s / denom).item() for k, s in self._sums.items()}


def compute_contact_obs(body_pos, root_rot, contact_forces, contact_obs_stale,
                        contact_body_ids, com_weights, force_threshold,
                        support_dirs, pair_ids_i, pair_ids_j, global_obs, margin_cap):
    """Contact-aware observation block: [flags (K), margin (1), graph (3*K*(K-1)/2)].

    flags:  1.0 for each candidate body whose ground-contact force norm
            exceeds force_threshold (0.0 otherwise).
    margin: signed horizontal distance from the mass-weighted COM to the
            convex hull of the contacting candidate bodies, evaluated with a
            fixed fan of support directions (positive = COM inside the hull).
            Clamped to +/- margin_cap; -margin_cap sentinel when nothing is
            in contact. Only approximately rotation-invariant: the fan is
            world-fixed, so the worst-case error is ~(hull half-span) *
            sin(pi/num_dirs) (~4 cm for a 0.4 m two-contact segment with 16
            directions). Monotone in the COM offset at any fixed heading.
    graph:  pairwise position deltas between candidate bodies, zeroed unless
            both bodies of the pair are in contact. World axes when
            global_obs, heading-local otherwise (matching compute_char_obs).

    Envs flagged in contact_obs_stale (just reset, sensor data is from the
    previous episode) get zero flags, which propagates to the sentinel margin
    and a zero graph.
    """
    candidate_forces = contact_forces[:, contact_body_ids, :]
    flags = (torch.linalg.vector_norm(candidate_forces, dim=-1) > force_threshold).float()
    flags = flags * torch.logical_not(contact_obs_stale).float().unsqueeze(-1)

    candidate_pos = body_pos[:, contact_body_ids, :]

    # Support-polygon margin via the support function of the contact hull:
    # margin = min_d [ max_{i in contact}(p_i . d) - com . d ] over the
    # direction fan. Exact in the dense-direction limit; degrades gracefully
    # to (negative) point/segment distance for 1-2 contacts.
    com_xy = torch.einsum("nbk,b->nk", body_pos, com_weights)[..., :2]
    point_proj = torch.matmul(candidate_pos[..., :2], support_dirs.t())
    masked_proj = torch.where(flags.unsqueeze(-1) > 0.5, point_proj,
                              torch.full_like(point_proj, -1e9))
    hull_support = torch.max(masked_proj, dim=1)[0]
    com_proj = torch.matmul(com_xy, support_dirs.t())
    margin = torch.min(hull_support - com_proj, dim=-1)[0]
    margin = torch.clamp(margin, min=-margin_cap, max=margin_cap)
    any_contact = torch.any(flags > 0.5, dim=-1)
    margin = torch.where(any_contact, margin, torch.full_like(margin, -margin_cap))

    # Contact graph: relative geometry of the active support points.
    pair_deltas = candidate_pos[:, pair_ids_j, :] - candidate_pos[:, pair_ids_i, :]
    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        heading_inv_expand = heading_inv_rot.unsqueeze(-2).repeat((1, pair_deltas.shape[1], 1))
        pair_deltas_flat = torch_util.quat_rotate(heading_inv_expand.reshape(-1, 4),
                                                  pair_deltas.reshape(-1, 3))
        pair_deltas = pair_deltas_flat.reshape(pair_deltas.shape)
    pair_mask = flags[:, pair_ids_i] * flags[:, pair_ids_j]
    graph = pair_deltas * pair_mask.unsqueeze(-1)
    graph_flat = graph.reshape(graph.shape[0], -1)

    contact_obs = torch.cat([flags, margin.unsqueeze(-1), graph_flat], dim=-1)
    return contact_obs

@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (Tensor, int) -> Tensor
    phase_obs = phase.unsqueeze(-1)

    # positional embedding of phase
    if (num_phase_encoding > 0):
        pe_exp = torch.arange(num_phase_encoding, device=phase.device, dtype=phase.dtype)
        pe_scale = 2.0 * np.pi * torch.pow(2.0, pe_exp)
        pe_scale = pe_scale.unsqueeze(0)
        pe_val = phase.unsqueeze(-1) * pe_scale
        pe_sin = torch.sin(pe_val)
        pe_cos = torch.cos(pe_val)

        phase_obs = torch.cat((phase_obs, pe_sin, pe_cos), dim=-1)

    return phase_obs

@torch.jit.script
def convert_to_local(root_rot, root_vel, root_ang_vel, key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_root_rot = torch_util.quat_mul(heading_inv_rot, root_rot)
    local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
    local_root_ang_vel = torch_util.quat_rotate(heading_inv_rot, root_ang_vel)
    
    if (len(key_pos) > 0):
        heading_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, key_pos.shape[1], 1))
        flat_heading_rot_expand = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                                heading_rot_expand.shape[2])
        flat_key_pos = key_pos.reshape(key_pos.shape[0] * key_pos.shape[1], key_pos.shape[2])
        flat_local_key_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_key_pos)
        local_key_pos = flat_local_key_pos.reshape(key_pos.shape[0], key_pos.shape[1], key_pos.shape[2])
    else:
        local_key_pos = key_pos

    return local_root_rot, local_root_vel, local_root_ang_vel, local_key_pos

@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos,
                    global_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if (not global_obs):
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                               heading_inv_rot_expand.shape[2]))
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)
        
        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if (len(key_pos) > 0):
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape((heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1] * heading_inv_rot_expand.shape[2],
                                                                   heading_inv_rot_expand.shape[3]))
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2],
                                            key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if (root_height_obs):
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])
    
    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if (len(key_pos) > 0):
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs

@torch.jit.script
def compute_deepmimic_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs, 
                          phase, num_phase_encoding, enable_phase_obs, 
                          enable_tar_obs, tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, int, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
    char_obs = char_env.compute_char_obs(root_pos=root_pos,
                                            root_rot=root_rot,
                                            root_vel=root_vel,
                                            root_ang_vel=root_ang_vel,
                                            joint_rot=joint_rot,
                                            dof_vel=dof_vel,
                                            key_pos=key_pos,
                                            global_obs=global_obs,
                                            root_height_obs=root_height_obs)
    obs = [char_obs]

    if (enable_phase_obs):
        phase_obs = compute_phase_obs(phase=phase, num_phase_encoding=num_phase_encoding)
        obs.append(phase_obs)

    if (enable_tar_obs):
        if (global_obs):
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(ref_root_pos=ref_root_pos,
                                  ref_root_rot=ref_root_rot,
                                  root_pos=tar_root_pos, 
                                  root_rot=tar_root_rot, 
                                  joint_rot=tar_joint_rot,
                                  key_pos=tar_key_pos,
                                  global_obs=global_obs,
                                  root_height_obs=root_height_obs)
        
        tar_obs = torch.reshape(tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]])
        obs.append(tar_obs)

    obs = torch.cat(obs, dim=-1)
    
    return obs

@torch.jit.script
def compute_done(done_buf, time, ep_len, root_rot, body_pos, tar_root_rot, tar_body_pos, 
                 ground_contact_force, contact_body_ids,
                 pose_termination, pose_termination_dist, 
                 global_obs, enable_early_termination,
                 motion_times, motion_len, motion_len_term,
                 track_root):
    # type: (Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, float, bool, bool, Tensor, Tensor, Tensor, bool) -> Tensor
    done = torch.full_like(done_buf, base_env.DoneFlags.NULL.value)
    
    timeout = time >= ep_len
    done[timeout] = base_env.DoneFlags.TIME.value
    
    motion_end = motion_times >= motion_len
    motion_end = torch.logical_and(motion_end, motion_len_term)
    done[motion_end] = base_env.DoneFlags.SUCC.value

    if (enable_early_termination):
        failed = torch.zeros(done.shape, device=done.device, dtype=torch.bool)
        if (contact_body_ids.shape[0] > 0):
            fall_contact = torch.any(torch.abs(ground_contact_force[:, contact_body_ids, :]) > 0.1, dim=-1)

            has_fallen = torch.any(fall_contact, dim=-1)
            failed = torch.logical_or(failed, has_fallen)

        if (pose_termination):
            root_pos = body_pos[..., 0:1, :]
            tar_root_pos = tar_body_pos[..., 0:1, :]

            if (not global_obs):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos
                body_pos = char_env.convert_to_local_root_body_pos(root_rot, body_pos)
                tar_body_pos = char_env.convert_to_local_root_body_pos(tar_root_rot, tar_body_pos)

            elif (not track_root):
                body_pos = body_pos[..., 1:, :] - root_pos
                tar_body_pos = tar_body_pos[..., 1:, :] - tar_root_pos

            body_pos_diff = tar_body_pos - body_pos
            body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
            body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
            pose_fail = body_pos_dist > pose_termination_dist * pose_termination_dist

            if (track_root):
                root_pos_diff = tar_root_pos - root_pos
                root_pos_dist = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
                root_pos_fail = root_pos_dist > pose_termination_dist * pose_termination_dist
                root_pos_fail = root_pos_fail.squeeze(-1)
                pose_fail = torch.logical_or(pose_fail, root_pos_fail)

            failed = torch.logical_or(failed, pose_fail)
            
        # only fail after first timestep
        not_first_step = (time > 0.0)
        failed = torch.logical_and(failed, not_first_step)
        done[failed] = base_env.DoneFlags.FAIL.value
    
    return done

@torch.jit.script
def compute_reward(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos,
                   tar_root_pos, tar_root_rot, tar_root_vel, tar_root_ang_vel,
                   tar_joint_rot, tar_dof_vel, tar_key_pos,
                   joint_rot_err_w, dof_err_w, track_root_h, track_root,
                   pose_w, vel_w, root_pose_w, root_vel_w, key_pos_w,
                   pose_scale, vel_scale, root_pose_scale, root_vel_scale, key_pos_scale):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, float, float, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    pose_diff = torch_util.quat_diff_angle(joint_rot, tar_joint_rot)
    pose_err = torch.sum(joint_rot_err_w * pose_diff * pose_diff, dim=-1)

    vel_diff = tar_dof_vel - dof_vel
    vel_err = torch.sum(dof_err_w * vel_diff * vel_diff, dim=-1)

    root_pos_diff = tar_root_pos - root_pos

    if (not track_root):
        root_pos_diff[..., 0:2] = 0

    if (not track_root_h):
        root_pos_diff[..., 2] = 0

    root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
    
    if (len(key_pos) > 0):
        key_pos = key_pos - root_pos.unsqueeze(-2)
        tar_key_pos = tar_key_pos - tar_root_pos.unsqueeze(-2)

    if (not track_root):
        root_rot, root_vel, root_ang_vel, key_pos = convert_to_local(root_rot, root_vel, root_ang_vel, key_pos)
        tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos = convert_to_local(tar_root_rot, tar_root_vel, tar_root_ang_vel, tar_key_pos)
        
    root_rot_err = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err *= root_rot_err

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)

    if (len(key_pos) > 0):
        key_pos_diff = tar_key_pos - key_pos
        key_pos_err = torch.sum(key_pos_diff * key_pos_diff, dim=-1)
        key_pos_err = torch.sum(key_pos_err, dim=-1)
    else:
        key_pos_err = torch.zeros([0], device=key_pos.device)

    pose_r = torch.exp(-pose_scale * pose_err)
    vel_r = torch.exp(-vel_scale * vel_err)
    root_pose_r = torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
    root_vel_r = torch.exp(-root_vel_scale * (root_vel_err + 0.1 * root_ang_vel_err))
    key_pos_r = torch.exp(-key_pos_scale * key_pos_err)

    r = pose_w * pose_r \
        + vel_w * vel_r \
        + root_pose_w * root_pose_r \
        + root_vel_w * root_vel_r \
        + key_pos_w * key_pos_r

    return r, pose_r, vel_r, root_pose_r, root_vel_r, key_pos_r

@torch.jit.script
def compute_tracking_error(root_pos, root_rot, body_rot, body_pos,
                            tar_root_pos, tar_root_rot,
                            tar_body_rot, tar_body_pos,
                            root_vel, root_ang_vel, dof_vel,
                            tar_root_vel, tar_root_ang_vel, tar_dof_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    body_pos = body_pos - root_pos.unsqueeze(-2)
    tar_body_pos = tar_body_pos - tar_root_pos.unsqueeze(-2)

    root_pos_diff = tar_root_pos - root_pos
    root_pos_err = torch.linalg.vector_norm(root_pos_diff, dim=-1)

    body_rot_diff = torch_util.quat_diff_angle(body_rot, tar_body_rot)
    body_rot_err = torch.abs(body_rot_diff)
    body_rot_err = torch.mean(body_rot_err, dim=-1)

    body_pos_diff = tar_body_pos - body_pos
    body_pos_diff_l2 = torch.linalg.vector_norm(body_pos_diff, dim=-1)
    body_pos_err = torch.mean(body_pos_diff_l2, dim=-1)

    root_rot_diff = torch_util.quat_diff_angle(root_rot, tar_root_rot)
    root_rot_err = torch.abs(root_rot_diff)

    dof_vel_diff = tar_dof_vel - dof_vel
    dof_vel_err = torch.mean(torch.abs(dof_vel_diff), dim=-1)

    root_vel_diff = tar_root_vel - root_vel
    root_vel_err = torch.mean(torch.abs(root_vel_diff), dim=-1)

    root_ang_vel_diff = tar_root_ang_vel - root_ang_vel
    root_ang_vel_err = torch.mean(torch.abs(root_ang_vel_diff), dim=-1)

    tracking_error = torch.stack([root_pos_err, root_rot_err, body_pos_err, body_rot_err, dof_vel_err, root_vel_err, root_ang_vel_err], dim=-1)
    return tracking_error