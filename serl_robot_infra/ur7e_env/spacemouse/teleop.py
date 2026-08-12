"""Control the UR7e environment with one SpaceMouse."""

import time

import numpy as np

CONTROL_HZ = 10.0
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


def main():
    import pyspacemouse
    from experiments.pick.config import PickEnvConfig
    from experiments.pick.wrapper import PickEnv, TranslationOnlyWrapper

    env = TranslationOnlyWrapper(PickEnv(hz=CONTROL_HZ, config=PickEnvConfig()))
    device = pyspacemouse.open()
    if device is None:
        env.close()
        raise RuntimeError("未检测到 SpaceMouse")

    print("当前 TCP 位姿:", env.unwrapped.currpos)
    input("按 Enter 启用 SpaceMouse，Ctrl+C 退出...")
    try:
        while True:
            env.step(
                latest_spacemouse_action(
                    device, gripper=True, translation_only=False
                )
            )
    except KeyboardInterrupt:
        print("SpaceMouse 控制已停止")
    finally:
        device.close()
        env.close()


if __name__ == "__main__":
    main()
