"""Newline-delimited JSON client for the in-game Bridge (harness/MoSimRL/Bridge.cs)."""

from __future__ import annotations

import json
import socket

# Button order must match Bridge.cs / ButtonInjector.cs.
INTAKE, SHOOT, PASS, MANUAL, SPECIAL = range(5)


class BridgeError(RuntimeError):
    pass


class MoSimClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 47414, timeout: float = 180.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self._buf = b""
        self.last_cmd = ""

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def _call(self, msg: dict) -> dict:
        self.last_cmd = msg.get("cmd", "")
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise BridgeError("bridge closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise BridgeError(reply.get("error", "unknown bridge error"))
        return reply

    def hello(self) -> dict:
        return self._call({"cmd": "hello"})

    def config(self, time_scale: float | None = None, decision_steps: int | None = None,
               cameras: bool | None = None, fps: int | None = None, steps_per_frame: int | None = None) -> dict:
        msg: dict = {"cmd": "config"}
        if time_scale is not None:
            msg["timeScale"] = time_scale
        if decision_steps is not None:
            msg["decisionSteps"] = decision_steps
        if cameras is not None:
            msg["cameras"] = 1 if cameras else 0
        if fps is not None:
            msg["fps"] = fps
        if steps_per_frame is not None:
            msg["stepsPerFrame"] = steps_per_frame
        return self._call(msg)

    def reset(self) -> dict:
        return self._call({"cmd": "reset"})

    def act(self, vx: float, vz: float, rot: float, buttons, rff: bool = False, lite: bool | None = None) -> dict:
        msg = {"cmd": "act", "v": [float(vx), float(vz)], "rot": float(rot),
               "b": [1 if b else 0 for b in buttons], "rff": 1 if rff else 0}
        if lite is not None:
            msg["lite"] = 1 if lite else 0
        return self._call(msg)

    # lookahead planning (Snapshot.cs): only at a decision point, the instance keeps waiting for its next act
    def snapshot(self) -> dict:
        """{"blob": base64 match state, "t": timer, "ms": capture time, "summary": sizes}"""
        return self._call({"cmd": "snapshot"})

    def restore(self, blob: str) -> dict:
        """Continue the match held in `blob` (from any instance); returns the state right after the restore."""
        return self._call({"cmd": "restore", "blob": blob})

    def keys(self) -> dict:
        return self._call({"cmd": "keys"})

    def release(self) -> dict:
        return self._call({"cmd": "release"})
