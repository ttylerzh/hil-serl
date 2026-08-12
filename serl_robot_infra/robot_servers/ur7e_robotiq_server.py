"""Minimal Robotiq gripper client for the UR controller's port 63352 server."""

import socket
import threading
import time


class RobotiqGripper:
    def __init__(self):
        self.socket = None
        self._lock = threading.Lock()

    def connect(self, host, port=63352, timeout=5.0):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)

    def _set(self, **values):
        command = "SET " + " ".join(f"{key} {value}" for key, value in values.items())
        with self._lock:
            self.socket.sendall((command + "\n").encode())
            if self.socket.recv(1024) != b"ack":
                raise RuntimeError(f"Robotiq rejected command: {command}")

    def _get(self, variable):
        with self._lock:
            self.socket.sendall(f"GET {variable}\n".encode())
            response = self.socket.recv(1024).decode().split()
        if len(response) != 2 or response[0] != variable:
            raise RuntimeError(f"Unexpected Robotiq response: {' '.join(response)}")
        return int(response[1])

    def activate(self, timeout=10.0):
        if self._get("STA") == 3:
            return
        deadline = time.monotonic() + timeout
        self._set(ACT=0, ATR=0)
        while self._get("ACT") != 0 or self._get("STA") != 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("Robotiq reset timed out")
            time.sleep(0.05)
        self._set(ACT=1)
        while self._get("ACT") != 1 or self._get("STA") != 3:
            if time.monotonic() >= deadline:
                raise TimeoutError("Robotiq activation timed out")
            time.sleep(0.05)

    def get_position(self):
        return self._get("POS")

    def move(self, position, speed, force):
        position, speed, force = (
            max(0, min(255, int(value))) for value in (position, speed, force)
        )
        self._set(POS=position, SPE=speed, FOR=force, GTO=1)

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
