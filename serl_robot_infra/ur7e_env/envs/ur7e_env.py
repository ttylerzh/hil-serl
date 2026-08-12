"""Minimal Gym environment for Cartesian UR7e control."""

import queue
import time

import cv2
import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from ur7e_env.camera.basler_capture import BaslerCapture


class DefaultEnvConfig:
    SERVER_URL = "http://127.0.0.1:5000/"
    BASLER_CAMERAS = {}
    REALSENSE_CAMERAS = {}
    IMAGE_CROP = {}
    IMAGE_SIZE = (128, 128)
    ACTION_SCALE = (0.005, 0.05)  # metres, radians per control step
    TRANSLATION_LIMIT = np.array([0.2, 0.2, 0.2])
    WORKSPACE_LOW = None
    WORKSPACE_HIGH = None
    ROTATION_LIMIT = np.deg2rad(90.0)
    REQUEST_TIMEOUT = 5.0
    GRIPPER_ENABLED = False
    GRIPPER_SLEEP = 0.6


class UR7eEnv(gym.Env):
    """Apply normalized 6-DoF actions around the pose seen at startup."""

    def __init__(self, hz=10.0, config=None, fake_env=False):
        config = config or DefaultEnvConfig()
        if hz <= 0:
            raise ValueError("hz must be positive")

        self.hz = float(hz)
        self.fake_env = bool(fake_env)
        self.config = config
        self.url = config.SERVER_URL.rstrip("/") + "/"
        self.basler_camera_configs = config.BASLER_CAMERAS
        self.realsense_camera_configs = config.REALSENSE_CAMERAS
        duplicate_names = (
            self.basler_camera_configs.keys() & self.realsense_camera_configs.keys()
        )
        if duplicate_names:
            raise ValueError(
                f"camera names must be unique: {', '.join(sorted(duplicate_names))}"
            )
        self.camera_configs = {
            **self.basler_camera_configs,
            **self.realsense_camera_configs,
        }
        self.image_crop = config.IMAGE_CROP
        self.image_size = tuple(config.IMAGE_SIZE)
        self.action_scale = np.asarray(config.ACTION_SCALE, dtype=float)
        self.translation_limit = np.asarray(config.TRANSLATION_LIMIT, dtype=float)
        workspace_low = getattr(config, "WORKSPACE_LOW", None)
        workspace_high = getattr(config, "WORKSPACE_HIGH", None)
        if (workspace_low is None) != (workspace_high is None):
            raise ValueError("WORKSPACE_LOW and WORKSPACE_HIGH must be set together")
        self.workspace_low = (
            None if workspace_low is None else np.asarray(workspace_low, dtype=float)
        )
        self.workspace_high = (
            None if workspace_high is None else np.asarray(workspace_high, dtype=float)
        )
        self.rotation_limit = float(config.ROTATION_LIMIT)
        self.request_timeout = float(config.REQUEST_TIMEOUT)
        self.gripper_enabled = bool(config.GRIPPER_ENABLED)
        self.gripper_sleep = float(config.GRIPPER_SLEEP)
        if self.action_scale.shape != (2,) or np.any(self.action_scale <= 0):
            raise ValueError("ACTION_SCALE must contain two positive values")
        if self.translation_limit.shape != (3,) or np.any(self.translation_limit <= 0):
            raise ValueError("TRANSLATION_LIMIT must contain three positive values")
        if self.workspace_low is not None and (
            self.workspace_low.shape != (3,)
            or self.workspace_high.shape != (3,)
            or not np.all(np.isfinite(self.workspace_low))
            or not np.all(np.isfinite(self.workspace_high))
            or np.any(self.workspace_low >= self.workspace_high)
        ):
            raise ValueError("workspace bounds must contain three increasing finite values")
        if (
            self.rotation_limit <= 0
            or self.request_timeout <= 0
            or self.gripper_sleep < 0
        ):
            raise ValueError(
                "ROTATION_LIMIT and REQUEST_TIMEOUT must be positive and "
                "GRIPPER_SLEEP must be non-negative"
            )
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("IMAGE_SIZE must contain a positive width and height")

        self.cameras = {}
        self.last_gripper_act = 0.0
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(7 if self.gripper_enabled else 6,),
            dtype=np.float32,
        )
        state_space = {
            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
            "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
            "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
        }
        if self.gripper_enabled:
            state_space["gripper_pose"] = gym.spaces.Box(
                0.0, 1.0, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_space),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(*self.image_size[::-1], 3),
                            dtype=np.uint8,
                        )
                        for key in self.camera_configs
                    }
                ),
            }
        )

        self.session = None
        if self.fake_env:
            return
        self.session = requests.Session()
        self._update_state()
        self.workspace_center = self.currpos.copy()
        self.workspace_rotation = Rotation.from_quat(self.workspace_center[3:])
        self.init_cameras()

    def _post(self, route, timeout=None, **kwargs):
        response = self.session.post(
            self.url + route,
            timeout=self.request_timeout if timeout is None else timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def _update_state(self):
        state = self._post("getstate").json()
        self.currpos = np.asarray(state["pose"], dtype=float)
        self.currvel = np.asarray(state["vel"], dtype=float)
        self.currforce = np.asarray(state["force"], dtype=float)
        self.currtorque = np.asarray(state["torque"], dtype=float)
        self.q = np.asarray(state["q"], dtype=float)
        self.dq = np.asarray(state["dq"], dtype=float)
        if self.gripper_enabled:
            self.curr_gripper_pos = np.asarray(state["gripper_pos"], dtype=np.float32)
            if self.curr_gripper_pos.shape != (1,):
                raise ValueError("server gripper_pos must contain one value")

    def _clip_workspace(self, pose):
        if self.workspace_low is None:
            low = self.workspace_center[:3] - self.translation_limit
            high = self.workspace_center[:3] + self.translation_limit
        else:
            low, high = self.workspace_low, self.workspace_high
        pose[:3] = np.clip(pose[:3], low, high)
        relative = Rotation.from_quat(pose[3:]) * self.workspace_rotation.inv()
        rotvec = relative.as_rotvec()
        angle = np.linalg.norm(rotvec)
        if angle > self.rotation_limit:
            relative = Rotation.from_rotvec(rotvec * self.rotation_limit / angle)
            pose[3:] = (relative * self.workspace_rotation).as_quat()
        return pose

    def step(self, action):
        started_at = time.perf_counter()
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(
                f"action must contain {self.action_space.shape[0]} finite values"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_action = action[:6]

        if np.any(arm_action):
            self._update_state()
        target = self.currpos.copy()
        target[:3] += arm_action[:3] * self.action_scale[0]
        target[3:] = (
            Rotation.from_rotvec(arm_action[3:] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        target = self._clip_workspace(target)
        if np.any(arm_action):
            self._post("pose", json={"arr": target.tolist()})
        if self.gripper_enabled:
            self._send_gripper_command(action[6])

        time.sleep(max(0.0, 1.0 / self.hz - (time.perf_counter() - started_at)))
        self._update_state()
        return self._get_obs(), 0, False, False, {}

    def _get_obs(self):
        state = {
                "tcp_pose": self.currpos.copy(),
                "tcp_vel": self.currvel.copy(),
                "tcp_force": self.currforce.copy(),
                "tcp_torque": self.currtorque.copy(),
                "joint_pos": self.q.copy(),
                "joint_vel": self.dq.copy(),
            }
        if self.gripper_enabled:
            state["gripper_pose"] = self.curr_gripper_pos.copy()
        return {
            "state": state,
            "images": self.get_images(),
        }

    def _send_gripper_command(self, action):
        if time.monotonic() - self.last_gripper_act < self.gripper_sleep:
            return
        if action <= -0.5 and self.curr_gripper_pos[0] > 0.85:
            self._post("close_gripper")
        elif action >= 0.5 and self.curr_gripper_pos[0] < 0.85:
            self._post("open_gripper")
        else:
            return
        self.last_gripper_act = time.monotonic()

    def init_cameras(self):
        self.close_cameras()
        try:
            for name, kwargs in self.basler_camera_configs.items():
                self.cameras[name] = VideoCapture(
                    BaslerCapture(name=name, **kwargs)
                )
            for name, kwargs in self.realsense_camera_configs.items():
                self.cameras[name] = VideoCapture(RSCapture(name=name, **kwargs))
        except Exception:
            self.close_cameras()
            raise

    def get_images(self):
        images = {}
        for name, camera in self.cameras.items():
            try:
                frame = camera.read()
            except queue.Empty as error:
                raise RuntimeError(f"Basler camera {name} timed out") from error
            cropped = self.image_crop[name](frame) if name in self.image_crop else frame
            if cropped.size == 0:
                raise ValueError(f"IMAGE_CROP for {name} produced an empty image")
            resized = cv2.resize(cropped, self.image_size, interpolation=cv2.INTER_AREA)
            images[name] = np.ascontiguousarray(resized[..., ::-1])
        return images

    def close_cameras(self):
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.fake_env:
            return self.observation_space.sample(), {}
        self._update_state()
        return self._get_obs(), {}

    def close(self):
        self.close_cameras()
        if self.session is not None:
            self.session.close()
