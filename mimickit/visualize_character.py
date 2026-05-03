import argparse
import os
import time

import numpy as np
import torch

import envs.char_env as char_env
import engines.engine as engine


DEFAULT_XML = "data/assets/humanoid/fullbody_humanoid.xml"


class CharacterViewerEnv(char_env.CharEnv):
    def __init__(self, config, num_envs, device, visualize, fix_root, exercise_joints, time_per_dof):
        self._fix_root = fix_root
        self._exercise_joints = exercise_joints
        self._time_per_dof = time_per_dof

        super().__init__(config=config, num_envs=num_envs, device=device, visualize=visualize)

        if self._exercise_joints:
            num_dofs = int(self._pd_low.shape[0])
            self._episode_length = max(self._episode_length, self._time_per_dof * max(1, num_dofs))

        return

    def _build_character(self, env_id, config, color=None):
        char_file = config["env"]["char_file"]
        char_id = self._engine.create_obj(env_id=env_id,
                                          obj_type=engine.ObjType.articulated,
                                          asset_file=char_file,
                                          name="character",
                                          start_pos=self._init_root_pos.cpu().numpy(),
                                          start_rot=self._init_root_rot.cpu().numpy(),
                                          enable_self_collisions=False,
                                          fix_root=self._fix_root,
                                          color=color)
        return char_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        pd_low = self._action_space.low
        pd_high = self._action_space.high
        self._pd_low = torch.tensor(pd_low, device=self._device, dtype=torch.float32)
        self._pd_high = torch.tensor(pd_high, device=self._device, dtype=torch.float32)
        return

    def _apply_action(self, actions):
        if self._exercise_joints:
            actions = self._calc_test_action(actions)
        super()._apply_action(actions)
        return

    def _calc_test_action(self, actions):
        test_actions = torch.zeros_like(actions)

        num_envs = self.get_num_envs()
        num_dofs = self._pd_low.shape[0]
        env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)

        phase = self._time_buf / self._time_per_dof
        dof_id = phase.type(torch.long)
        dof_id = dof_id + env_ids
        dof_id = torch.remainder(dof_id, num_dofs)

        curr_low = self._pd_low[dof_id]
        curr_high = self._pd_high[dof_id]

        joint_phase = phase - torch.floor(phase)
        lerp = torch.sin(2 * np.pi * joint_phase)
        lim_val = torch.where(lerp < 0.0, curr_low, curr_high)
        abs_lerp = torch.abs(lerp)
        dof_val = abs_lerp * lim_val

        test_actions[torch.arange(actions.shape[0]), dof_id] = dof_val
        return test_actions


def parse_args():
    parser = argparse.ArgumentParser(description="Load a character XML into MimicKit and visualize it without training.")
    parser.add_argument("--char-file", default=DEFAULT_XML,
                        help="Path to the MuJoCo XML asset to visualize.")
    parser.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="Torch device to use.")
    parser.add_argument("--engine", default="isaac_lab", choices=["isaac_lab", "isaac_gym"],
                        help="Simulation backend.")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments to create.")
    parser.add_argument("--height", type=float, default=1.2,
                        help="Initial root height in meters.")
    parser.add_argument("--camera-mode", default="still", choices=["still", "track"],
                        help="Viewer camera behavior.")
    parser.add_argument("--control-freq", type=int, default=30,
                        help="Control frequency in Hz.")
    parser.add_argument("--sim-freq", type=int, default=120,
                        help="Physics frequency in Hz.")
    parser.add_argument("--env-spacing", type=float, default=3.0,
                        help="Spacing between environments.")
    parser.add_argument("--exercise-joints", action="store_true",
                        help="Sweep one DOF at a time to verify joint limits.")
    parser.add_argument("--time-per-dof", type=float, default=2.0,
                        help="Seconds spent on each DOF when --exercise-joints is enabled.")
    parser.add_argument("--free-root", dest="fix_root", action="store_false",
                        help="Allow the root to move freely instead of pinning the character in place.")
    # parser.set_defaults(fix_root=True)
    parser.add_argument("--steps", type=int, default=0,
                        help="Number of simulation steps to run. Use 0 to run until interrupted.")
    return parser.parse_args()


def resolve_path(path):
    if os.path.isabs(path):
        return path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd_path = os.path.abspath(path)
    repo_path = os.path.join(repo_root, path)

    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(repo_path)


def build_config(args):
    char_file = resolve_path(args.char_file)
    if not os.path.exists(char_file):
        raise FileNotFoundError(f"Character file not found: {char_file}")

    config = {
        "env": {
            "char_file": char_file,
            "camera_mode": args.camera_mode,
            "episode_length": 3600.0,
            "global_obs": False,
            "root_height_obs": True,
            "init_pose": [0.0, 0.0, args.height],
        },
        "engine": {
            "engine_name": args.engine,
            "control_mode": "pos",
            "control_freq": args.control_freq,
            "sim_freq": args.sim_freq,
            "env_spacing": args.env_spacing,
            "ground_contact_height": 0.3,
        },
    }
    return config


def reset_done_envs(env, done):
    done_env_ids = torch.nonzero(done != 0, as_tuple=False).flatten()
    if len(done_env_ids) > 0:
        env.reset(done_env_ids)
    return


def main():
    args = parse_args()
    config = build_config(args)

    env = CharacterViewerEnv(config=config,
                             num_envs=args.num_envs,
                             device=args.device,
                             visualize=True,
                             fix_root=args.fix_root,
                             exercise_joints=args.exercise_joints,
                             time_per_dof=args.time_per_dof)

    env.reset()

    action_size = int(np.prod(env.get_action_space().shape))
    actions = torch.zeros((args.num_envs, action_size), device=args.device, dtype=torch.float32)

    char_id = env._get_char_id()
    num_bodies = env._engine.get_obj_num_bodies(char_id)
    num_dofs = env._engine.get_obj_num_dofs(char_id)

    print(f"Loaded character: {config['env']['char_file']}")
    print(f"Bodies: {num_bodies}, DoFs: {num_dofs}, Action size: {action_size}")
    if args.exercise_joints:
        print("Joint sweep enabled. Press Ctrl+C to stop.")
    else:
        print("Static visualization enabled. Press Ctrl+C to stop.")

    step_count = 0
    try:
        while args.steps <= 0 or step_count < args.steps:
            _, _, done, _ = env.step(actions)
            reset_done_envs(env, done)
            step_count += 1
    except KeyboardInterrupt:
        print("\nStopped visualization.")

    time.sleep(0.1)
    return


if __name__ == "__main__":
    main()
