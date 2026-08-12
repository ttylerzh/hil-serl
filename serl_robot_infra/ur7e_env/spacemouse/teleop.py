"""Control the UR7e environment with one SpaceMouse."""

import argparse
import time

import numpy as np

DEADZONE = 0.05
INPUT_TIMEOUT = 0.25


def spacemouse_action(state, gripper=False, translation_only=False):
    action = np.array(
        [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
        dtype=np.float32,
    )
    action[np.abs(action) < DEADZONE] = 0.0
    if gripper:
        buttons = state.buttons
        gripper_action = -1.0 if buttons[0] else 1.0 if buttons[1] else 0.0
        action = np.append(action, gripper_action).astype(np.float32)
    if translation_only:
        action = np.concatenate((action[:3], action[6:])) if gripper else action[:3]
    return action


def latest_spacemouse_action(device, gripper=False, translation_only=False):
    state = device.read()
    while True:
        timestamp = state.t
        state = device.read()
        if state.t == timestamp:
            break
    if time.perf_counter() - state.t > INPUT_TIMEOUT:
        size = (3 if translation_only else 6) + int(gripper)
        return np.zeros(size, dtype=np.float32)
    return spacemouse_action(
        state, gripper=gripper, translation_only=translation_only
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="plug_insertion")
    return parser, parser.parse_args(argv)


def main():
    parser, args = parse_args()
    from experiments.mappings import CONFIG_MAPPING

    if args.task not in CONFIG_MAPPING:
        parser.error(
            f"unknown task {args.task!r}; choose from {', '.join(sorted(CONFIG_MAPPING))}"
        )

    env = CONFIG_MAPPING[args.task]().get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    print("当前 TCP 位姿:", env.unwrapped.currpos)
    input("按 Enter 启用 SpaceMouse，Ctrl+C 退出...")
    try:
        while True:
            env.step(action.copy())
    except KeyboardInterrupt:
        print("SpaceMouse 控制已停止")
    finally:
        env.close()


if __name__ == "__main__":
    main()
