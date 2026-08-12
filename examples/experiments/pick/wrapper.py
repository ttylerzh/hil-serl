"""Pick-task behavior layered on the reusable UR7e environment."""

import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ur7e_env.envs.ur7e_env import UR7eEnv


class PickEnv(UR7eEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_episode_length = int(self.config.MAX_EPISODE_LENGTH)
        self.reset_lift_distance = float(self.config.RESET_LIFT_DISTANCE)
        self.reset_timeout = float(self.config.RESET_TIMEOUT)
        if (
            self.max_episode_length <= 0
            or not 0.0 <= self.reset_lift_distance <= 0.2
            or self.reset_timeout <= 0
        ):
            raise ValueError("invalid Pick episode or reset configuration")
        self.curr_path_length = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.curr_path_length += 1
        terminated = terminated or self.curr_path_length >= self.max_episode_length
        info["succeed"] = bool(reward)
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.curr_path_length = 0
        if self.fake_env:
            return self.observation_space.sample(), {"succeed": False}

        self._update_state()
        self._post("open_gripper")
        time.sleep(self.gripper_sleep)
        self._post(
            "jointreset",
            timeout=self.reset_timeout,
            json={"lift_distance": self.reset_lift_distance},
        )
        self._update_state()
        self.workspace_center = self.currpos.copy()
        self.workspace_rotation = Rotation.from_quat(self.workspace_center[3:])
        self.last_gripper_act = time.monotonic()
        return self._get_obs(), {"succeed": False}


class TranslationOnlyWrapper(gym.ActionWrapper):
    """Keep 7D actions for Hybrid SAC while forcing UR rotation commands to zero."""

    def __init__(self, env):
        super().__init__(env)
        if env.action_space.shape != (7,):
            raise ValueError("translation-only Pick environment requires 7D actions")

    def action(self, action):
        action = np.asarray(action)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain 7 finite values")
        if not action.flags.writeable:
            raise ValueError("action must be writable so replay data can be projected")
        # The actor stores this same array after env.step; keep replay actions honest.
        action[3:6] = 0.0
        return action


class SpacemouseIntervention(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        import pyspacemouse

        self.device = pyspacemouse.open()
        if self.device is None:
            raise RuntimeError("未检测到 SpaceMouse")

    def step(self, action):
        from ur7e_env.spacemouse.teleop import latest_spacemouse_action

        expert_action = latest_spacemouse_action(
            self.device,
            gripper=True,
            translation_only=False,
        )
        expert_action[3:6] = 0.0
        intervened = np.any(expert_action)
        observation, reward, terminated, truncated, info = self.env.step(
            expert_action if intervened else action
        )
        if intervened:
            info["intervene_action"] = expert_action
        return observation, reward, terminated, truncated, info

    def close(self):
        self.device.close()
        self.env.close()


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        if env.action_space.shape != (7,):
            raise ValueError("Pick gripper penalty requires 7D actions")
        self.penalty = float(penalty)
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_gripper_pos = float(observation["state"]["gripper_pose"][0])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(info.get("intervene_action", action))
        gripper_action = action[-1]
        info["grasp_penalty"] = self.penalty if (
            (gripper_action < -0.5 and self.last_gripper_pos > 0.85)
            or (gripper_action > 0.5 and self.last_gripper_pos < 0.85)
        ) else 0.0
        self.last_gripper_pos = float(observation["state"]["gripper_pose"][0])
        return observation, reward, terminated, truncated, info


class BinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier):
        super().__init__(env)
        self.reward_classifier = reward_classifier

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        reward = int(self.reward_classifier(observation))
        terminated = terminated or bool(reward)
        info["succeed"] = bool(reward)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return observation, info
