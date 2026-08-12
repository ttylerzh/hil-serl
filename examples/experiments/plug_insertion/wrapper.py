"""Pick-task behavior layered on the reusable UR7e environment."""

import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from ur7e_env.envs.ur7e_env import UR7eEnv


class InsertPlugEnv(UR7eEnv):
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
        self._post("close_gripper")
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
    """Expose XYZ actions while keeping rotation and gripper fixed."""

    def __init__(self, env):
        super().__init__(env)
        if env.action_space.shape != (7,):
            raise ValueError("InsertPlug environment requires 7D internal actions")
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain 3 finite XYZ values")
        full_action = np.zeros(7, dtype=np.float32)
        full_action[:3] = action
        return full_action


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
            gripper=False,
            translation_only=True,
        )
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
