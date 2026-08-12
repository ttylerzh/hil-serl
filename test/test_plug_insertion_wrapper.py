import gymnasium as gym
import numpy as np

from experiments.plug_insertion.wrapper import TranslationOnlyWrapper


def test_translation_only_action():
    env = gym.Env()
    env.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
    wrapped = TranslationOnlyWrapper(env)

    action = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert wrapped.action_space.shape == (3,)
    np.testing.assert_array_equal(
        wrapped.action(action),
        np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


if __name__ == "__main__":
    test_translation_only_action()
