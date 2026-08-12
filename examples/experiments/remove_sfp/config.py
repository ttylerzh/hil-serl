import numpy as np

from experiments.config import DefaultTrainingConfig
from ur7e_env.envs.ur7e_env import DefaultEnvConfig


class RemoveSFPEnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    BASLER_SETTINGS = {
        "dim": (2464, 2056),
        "fps": 10,
        "exposure": 20_000,
        "gain": 100,
        "gamma": True,
        "black_level": 0,
        "white_balance": (1.578125, 1.0, 2.28125),
    }
    BASLER_CAMERAS = {
        "left": {
            "serial_number": "23132348",
            **BASLER_SETTINGS,
        },
        "center": {
            "serial_number": "23399785",
            **BASLER_SETTINGS,
        },
        "right": {
            "serial_number": "23915699",
            **BASLER_SETTINGS,
        },
    }
    REALSENSE_CAMERAS = {
        "world": {
            "serial_number": "235422301854",
            "dim": (640, 480),
            "fps": 15,
            "exposure": 200,
        }
    }
    IMAGE_CROP = {
        "left": lambda img: img[100:1956, 100:2364],
        "center": lambda img: img[100:1956, 100:2364],
        "right": lambda img: img[100:1956, 100:2364],
        "world": lambda img: img[180:300, 200:360],

    }
    IMAGE_SIZE = (128, 128)

    ACTION_SCALE = (0.01, 0.1)
    TRANSLATION_LIMIT = np.array([0.05, 0.05, 0.05])
    WORKSPACE_LOW = np.array([0.35, -0.16, 0.015])
    WORKSPACE_HIGH = np.array([0.54, 0.06, 0.31])
    ROTATION_LIMIT = np.deg2rad(60.0)
    REQUEST_TIMEOUT = 5.0
    GRIPPER_ENABLED = True
    GRIPPER_SLEEP = 0.6
    MAX_EPISODE_LENGTH = 300
    RESET_LIFT_DISTANCE = 0.10
    RESET_TIMEOUT = 30.0


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["left", "center", "right"]
    classifier_keys = ["left", "world"]
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",
    ]
    max_traj_length = RemoveSFPEnvConfig.MAX_EPISODE_LENGTH
    buffer_period = 1000
    checkpoint_period = 1000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        from experiments.remove_sfp.wrapper import (
            BinaryRewardClassifierWrapper,
            GripperPenaltyWrapper,
            RemoveSFPEnv,
            SpacemouseIntervention,
            TranslationOnlyWrapper,
        )
        from serl_launcher.wrappers.chunking import ChunkingWrapper
        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

        env = RemoveSFPEnv(config=RemoveSFPEnvConfig(), fake_env=fake_env)
        env = TranslationOnlyWrapper(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = GripperPenaltyWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            import jax
            import jax.numpy as jnp

            from serl_launcher.networks.reward_classifier import load_classifier_func

            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path="/home/pksun/zzh/hil-serl/classifier_ckpt/",
            )

            def reward_func(observation):
                return int(
                    (1 / (1 + jnp.exp(-classifier_func(observation))) > 0.8).item()
                )

            env = BinaryRewardClassifierWrapper(env, reward_func)
        return env
