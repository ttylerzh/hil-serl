"""Flask control server for a UR7e using ur_rtde."""

import argparse
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


SERVO_LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300.0
SERVO_COMMAND_TIMEOUT = 0.25
SERVO_SMOOTHING_TIME = 0.05
MAX_TARGET_TRANSLATION = 0.02
MAX_TARGET_ROTATION = 0.2


def _vector(values, size, name):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return vector


def _rtde_to_serl_pose(pose):
    pose = _vector(pose, 6, "RTDE pose")
    return np.concatenate((pose[:3], Rotation.from_rotvec(pose[3:]).as_quat()))


def _serl_to_rtde_pose(pose):
    pose = _vector(pose, 7, "pose")
    if np.linalg.norm(pose[3:]) == 0:
        raise ValueError("pose quaternion must be non-zero")
    return np.concatenate((pose[:3], Rotation.from_quat(pose[3:]).as_rotvec()))


def _blend_pose(current, target, fraction):
    """Move an RTDE pose towards a target along the shortest rotation."""
    blended = current.copy()
    blended[:3] += (target[:3] - current[:3]) * fraction
    current_rotation = Rotation.from_rotvec(current[3:])
    rotation_delta = Rotation.from_rotvec(target[3:]) * current_rotation.inv()
    blended[3:] = (
        Rotation.from_rotvec(rotation_delta.as_rotvec() * fraction) * current_rotation
    ).as_rotvec()
    return blended


def _positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _byte(value):
    value = int(value)
    if not 0 <= value <= 255:
        raise ValueError("must be between 0 and 255")
    return value


class UR7eServer:
    def __init__(
        self,
        robot_ip,
        frequency=500.0,
        linear_speed=0.1,
        linear_acceleration=0.05,
        joint_speed=1,
        joint_acceleration=0.5,
        servo_lookahead_time=SERVO_LOOKAHEAD_TIME,
        servo_gain=SERVO_GAIN,
        servo_command_timeout=SERVO_COMMAND_TIMEOUT,
        servo_smoothing_time=SERVO_SMOOTHING_TIME,
        max_target_translation=MAX_TARGET_TRANSLATION,
        max_target_rotation=MAX_TARGET_ROTATION,
        reset_joint_target=None,
        gripper=None,
        gripper_open_position=0,
        gripper_closed_position=255,
        gripper_speed=100,
        gripper_force=50,
    ):
        import rtde_control
        import rtde_receive

        if frequency <= 0:
            raise ValueError("frequency must be positive")
        if not 0.03 <= servo_lookahead_time <= 0.2:
            raise ValueError("servo lookahead time must be between 0.03 and 0.2")
        if not 100 <= servo_gain <= 2000:
            raise ValueError("servo gain must be between 100 and 2000")

        self.receiver = None
        self.controller = None
        self._servo_thread = None
        self._target_lock = threading.Lock()
        self._target_event = threading.Event()
        self._stop_event = threading.Event()
        self._servo_idle = threading.Event()
        self._servo_idle.set()
        self._target = None
        self._target_updated_at = 0.0
        self._manual_motion = False
        self._servo_error = None
        self.gripper = gripper
        self.gripper_open_position = _byte(gripper_open_position)
        self.gripper_closed_position = _byte(gripper_closed_position)
        self.gripper_speed = _byte(gripper_speed)
        self.gripper_force = _byte(gripper_force)
        if self.gripper_open_position >= self.gripper_closed_position:
            raise ValueError("gripper open position must be below closed position")

        self.frequency = float(frequency)
        self.servo_dt = 1.0 / self.frequency
        self.linear_speed = float(linear_speed)
        self.linear_acceleration = float(linear_acceleration)
        self.joint_speed = float(joint_speed)
        self.joint_acceleration = float(joint_acceleration)
        self.servo_lookahead_time = float(servo_lookahead_time)
        self.servo_gain = float(servo_gain)
        self.servo_command_timeout = float(servo_command_timeout)
        self.servo_smoothing_time = float(servo_smoothing_time)
        self.max_target_translation = float(max_target_translation)
        self.max_target_rotation = float(max_target_rotation)
        if min(
            self.linear_speed,
            self.linear_acceleration,
            self.joint_speed,
            self.joint_acceleration,
            self.servo_command_timeout,
            self.servo_smoothing_time,
            self.max_target_translation,
            self.max_target_rotation,
        ) <= 0:
            raise ValueError("motion and servo parameters must be positive")
        self.servo_smoothing_fraction = -np.expm1(
            -self.servo_dt / self.servo_smoothing_time
        )

        self.receiver = rtde_receive.RTDEReceiveInterface(robot_ip, frequency)
        try:
            self.controller = rtde_control.RTDEControlInterface(robot_ip, frequency)
            if not self.receiver.isConnected() or not self.controller.isConnected():
                raise RuntimeError(f"Failed to connect to UR7e at {robot_ip}")
            self.reset_joint_target = (
                None
                if reset_joint_target is None
                else _vector(reset_joint_target, 6, "reset joint target").tolist()
            )
            self._servo_thread = threading.Thread(
                target=self._servo_loop,
                name="ur7e-servo",
                daemon=True,
            )
            self._servo_thread.start()
        except Exception:
            self.close()
            raise

    def get_state(self):
        wrench = _vector(self.receiver.getActualTCPForce(), 6, "TCP wrench")
        state = {
            "pose": _rtde_to_serl_pose(self.receiver.getActualTCPPose()).tolist(),
            "vel": _vector(self.receiver.getActualTCPSpeed(), 6, "TCP velocity").tolist(),
            "force": wrench[:3].tolist(),
            "torque": wrench[3:].tolist(),
            "q": _vector(self.receiver.getActualQ(), 6, "joint positions").tolist(),
            "dq": _vector(self.receiver.getActualQd(), 6, "joint velocities").tolist(),
            "safety_mode": self.receiver.getSafetyMode(),
        }
        if self.gripper is not None:
            position = self.gripper.get_position()
            normalized = 1.0 - (
                (position - self.gripper_open_position)
                / (self.gripper_closed_position - self.gripper_open_position)
            )
            state["gripper_pos"] = [float(np.clip(normalized, 0.0, 1.0))]
        return state

    def move_gripper(self, position):
        if self.gripper is None:
            raise RuntimeError("Robotiq gripper is not configured")
        self.gripper.move(position, self.gripper_speed, self.gripper_force)

    def open_gripper(self):
        self.move_gripper(self.gripper_open_position)

    def close_gripper(self):
        self.move_gripper(self.gripper_closed_position)

    def set_servo_target(self, pose):
        target = _serl_to_rtde_pose(pose)
        actual = _vector(self.receiver.getActualTCPPose(), 6, "actual TCP pose")
        translation_delta = np.linalg.norm(target[:3] - actual[:3])
        rotation_delta = (
            Rotation.from_rotvec(target[3:])
            * Rotation.from_rotvec(actual[3:]).inv()
        ).magnitude()
        if translation_delta > self.max_target_translation:
            raise ValueError(
                f"target translation step exceeds {self.max_target_translation} m"
            )
        if rotation_delta > self.max_target_rotation:
            raise ValueError(
                f"target rotation step exceeds {self.max_target_rotation} rad"
            )

        with self._target_lock:
            if self._servo_error is not None:
                raise RuntimeError("UR7e servo loop failed") from self._servo_error
            if self._manual_motion:
                raise RuntimeError("UR7e joint reset is in progress")
            self._target = target
            self._target_updated_at = time.monotonic()
            self._servo_idle.clear()
        self._target_event.set()

    def _servo_loop(self):
        active = False
        commanded_pose = None
        try:
            while not self._stop_event.is_set():
                with self._target_lock:
                    target = None if self._target is None else self._target.copy()
                    target_updated_at = self._target_updated_at
                    manual_motion = self._manual_motion

                stale = (
                    target is None
                    or manual_motion
                    or time.monotonic() - target_updated_at
                    > self.servo_command_timeout
                )
                if stale:
                    if active:
                        self.controller.servoStop()
                        active = False
                    commanded_pose = None
                    self._servo_idle.set()
                    self._target_event.wait(timeout=0.05)
                    self._target_event.clear()
                    continue

                self._servo_idle.clear()
                cycle_start = self.controller.initPeriod()
                if commanded_pose is None:
                    commanded_pose = _vector(
                        self.receiver.getActualTCPPose(), 6, "actual TCP pose"
                    )
                commanded_pose = _blend_pose(
                    commanded_pose, target, self.servo_smoothing_fraction
                )
                active = True
                if not self.controller.servoL(
                    commanded_pose.tolist(),
                    self.linear_speed,
                    self.linear_acceleration,
                    self.servo_dt,
                    self.servo_lookahead_time,
                    self.servo_gain,
                ):
                    raise RuntimeError("UR7e servoL failed")
                self.controller.waitPeriod(cycle_start)
        except Exception as error:
            with self._target_lock:
                self._servo_error = error
            self._stop_event.set()
        finally:
            if active:
                try:
                    self.controller.servoStop()
                except Exception:
                    pass
            self._servo_idle.set()

    def reset_joint(self, lift_distance=0.0, joint_target=None):
        joint_target = (
            self.reset_joint_target
            if joint_target is None
            else _vector(joint_target, 6, "reset joint target").tolist()
        )
        if joint_target is None:
            raise ValueError("reset joint target is not configured")
        lift_distance = float(lift_distance)
        if not np.isfinite(lift_distance) or not 0.0 <= lift_distance <= 0.2:
            raise ValueError("lift distance must be between 0 and 0.2 m")
        with self._target_lock:
            if self._servo_error is not None:
                raise RuntimeError("UR7e servo loop failed") from self._servo_error
            self._manual_motion = True
            self._target = None
        self._target_event.set()
        if not self._servo_idle.wait(timeout=1.0):
            with self._target_lock:
                self._manual_motion = False
            raise RuntimeError("UR7e servo loop did not stop")
        try:
            if lift_distance:
                lift_target = _vector(
                    self.receiver.getActualTCPPose(), 6, "actual TCP pose"
                )
                lift_target[2] += lift_distance
                if not self.controller.moveL(
                    lift_target.tolist(),
                    self.linear_speed,
                    self.linear_acceleration,
                ):
                    raise RuntimeError("UR7e reset lift moveL failed")
            if not self.controller.moveJ(
                joint_target,
                self.joint_speed,
                self.joint_acceleration,
            ):
                raise RuntimeError("UR7e moveJ failed")
        finally:
            with self._target_lock:
                self._manual_motion = False
                self._target = None
                self._target_updated_at = 0.0

    def close(self):
        if self._servo_thread is not None:
            self._stop_event.set()
            self._target_event.set()
            self._servo_thread.join()
            self._servo_thread = None
        if self.controller is not None:
            try:
                self.controller.stopScript()
            except Exception:
                pass
            try:
                self.controller.disconnect()
            except Exception:
                pass
            self.controller = None
        if self.receiver is not None:
            try:
                self.receiver.disconnect()
            except Exception:
                pass
            self.receiver = None
        if self.gripper is not None:
            self.gripper.close()
            self.gripper = None


def create_app(robot):
    from flask import Flask, jsonify, request

    webapp = Flask(__name__)

    @webapp.errorhandler(ValueError)
    def invalid_request(error):
        return jsonify({"error": str(error)}), 400

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": robot.get_state()["pose"]})

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        pose = robot.get_state()["pose"]
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        return jsonify({"pose": pose[:3] + euler.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": robot.get_state()["vel"]})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": robot.get_state()["force"]})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": robot.get_state()["torque"]})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": robot.get_state()["q"]})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": robot.get_state()["dq"]})

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(robot.get_state())

    @webapp.route("/pose", methods=["POST"])
    def pose():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "arr" not in data:
            raise ValueError("request JSON must contain 'arr'")
        robot.set_servo_target(data["arr"])
        return "Moved"

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            raise ValueError("request JSON must be an object")
        robot.reset_joint(
            data.get("lift_distance", 0.0),
            data.get("joint_target"),
        )
        return "Reset Joint"

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        robot.open_gripper()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        robot.close_gripper()
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "gripper_pos" not in data:
            raise ValueError("request JSON must contain 'gripper_pos'")
        robot.move_gripper(_byte(data["gripper_pos"]))
        return "Moved Gripper"

    return webapp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_ip", default="192.168.15.1")
    parser.add_argument("--flask_url", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--frequency", type=_positive_float, default=500.0)
    parser.add_argument("--linear_speed", type=_positive_float, default=0.03)
    parser.add_argument("--linear_acceleration", type=_positive_float, default=0.05)
    parser.add_argument("--joint_speed", type=_positive_float, default=0.15)
    parser.add_argument("--joint_acceleration", type=_positive_float, default=0.2)
    parser.add_argument(
        "--servo_lookahead_time", type=_positive_float, default=SERVO_LOOKAHEAD_TIME
    )
    parser.add_argument("--servo_gain", type=_positive_float, default=SERVO_GAIN)
    parser.add_argument(
        "--servo_command_timeout", type=_positive_float, default=SERVO_COMMAND_TIMEOUT
    )
    parser.add_argument(
        "--servo_smoothing_time", type=_positive_float, default=SERVO_SMOOTHING_TIME
    )
    parser.add_argument(
        "--max_target_translation",
        type=_positive_float,
        default=MAX_TARGET_TRANSLATION,
    )
    parser.add_argument(
        "--max_target_rotation", type=_positive_float, default=MAX_TARGET_ROTATION
    )
    parser.add_argument("--reset_joint_target", type=float, nargs=6)
    parser.add_argument("--gripper_host")
    parser.add_argument("--gripper_port", type=int, default=63352)
    parser.add_argument("--gripper_open_position", type=_byte, default=0)
    parser.add_argument("--gripper_closed_position", type=_byte, default=255)
    parser.add_argument("--gripper_speed", type=_byte, default=100)
    parser.add_argument("--gripper_force", type=_byte, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    gripper = None
    if args.gripper_host:
        from robot_servers.ur7e_robotiq_server import RobotiqGripper

        gripper = RobotiqGripper()
        gripper.connect(args.gripper_host, args.gripper_port)
        gripper.activate()
    robot = UR7eServer(
        args.robot_ip,
        frequency=args.frequency,
        linear_speed=args.linear_speed,
        linear_acceleration=args.linear_acceleration,
        joint_speed=args.joint_speed,
        joint_acceleration=args.joint_acceleration,
        servo_lookahead_time=args.servo_lookahead_time,
        servo_gain=args.servo_gain,
        servo_command_timeout=args.servo_command_timeout,
        servo_smoothing_time=args.servo_smoothing_time,
        max_target_translation=args.max_target_translation,
        max_target_rotation=args.max_target_rotation,
        reset_joint_target=args.reset_joint_target,
        gripper=gripper,
        gripper_open_position=args.gripper_open_position,
        gripper_closed_position=args.gripper_closed_position,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )
    try:
        create_app(robot).run(host=args.flask_url, port=args.port, threaded=False)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
