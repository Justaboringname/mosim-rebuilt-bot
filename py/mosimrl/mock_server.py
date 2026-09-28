"""A crude stand-in for the in-game Bridge, speaking the same protocol.

PURPOSE: test the client / policy / CMA-ES plumbing without the game. It is NOT a faithful simulator —
robot kinematics, intake and shooting are toy models with guessed rates, so scores here say nothing
about MoSim. Replace every guessed rate with calibrated values before using anything like this as a surrogate.
"""

from __future__ import annotations

import json
import math
import socket
import threading

import numpy as np

from . import field as F
from .shifts import blue_active

DT = 22 * 0.0045          # one decision = 22 physics steps
VMAX = 3.0                # m/s   (guess)
SHOT_PERIOD = 0.20        # s/ball (guess)
PICK_RADIUS = 0.35        # m     (guess)
CAPACITY = 60             # balls (guess)


class MockMatch:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        fz = np.arange(-2.54, 2.55, 0.152)
        fz = fz[np.abs(fz) > 0.1]
        fx = np.linspace(-0.84, 0.84, 12)
        grid = np.array([(x, z) for x in fx for z in fz])[:400]
        depot = np.array([(x, z) for x in np.linspace(7.74, 8.19, 4) for z in np.linspace(-2.31, -1.55, 6)])
        spill = np.column_stack([self.rng.uniform(6.8, 7.9, 24), self.rng.uniform(2.9, 3.9, 24)])
        self.balls = np.vstack([grid, depot, spill])
        self.t = 160.0
        self.x, self.z, self.yaw = 3.7855, -3.6288, 0.0
        self.vx = self.vz = 0.0
        self.held = 8
        self.blue = 0
        self.blue_auto = 0
        self.shot_clock = 0.0
        self.end_since = None

    def step(self, vx: float, vz: float, rot: float, buttons) -> None:
        enabled = not (137.0 < self.t <= 140.0) and self.t > 0
        shoot = bool(buttons[1])
        mult = 0.55 if shoot else 1.0
        if enabled:
            n = math.hypot(vx, vz)
            if n > 1:
                vx, vz = vx / n, vz / n
            tvx, tvz = vx * VMAX * mult, vz * VMAX * mult
        else:
            tvx = tvz = 0.0
        a = min(1.0, DT / 0.25)
        self.vx += (tvx - self.vx) * a
        self.vz += (tvz - self.vz) * a
        nx, nz = self.x + self.vx * DT, self.z + self.vz * DT
        # the band between neutral zone and blue zone is passable only on bump / trench lanes
        if F.NEUTRAL_FACE < nx < F.ZONE_LINE:
            az = abs(nz)
            if not (0.60 <= az <= 2.45 or 2.75 <= az <= 4.03):
                nx = self.x
        self.x = float(np.clip(nx, F.X_WALL_RED + 0.5, F.X_WALL - 0.5))
        self.z = float(np.clip(nz, -F.Z_WALL + 0.4, F.Z_WALL - 0.4))
        if enabled:
            self.yaw = (self.yaw - rot * 180.0 * DT) % 360.0

        if enabled and buttons[0] and self.held < CAPACITY and len(self.balls):
            h = math.radians(self.yaw)
            fx, fz = self.x + 0.45 * math.sin(h), self.z + 0.45 * math.cos(h)
            d = np.hypot(self.balls[:, 0] - fx, self.balls[:, 1] - fz)
            take = np.where(d < PICK_RADIUS)[0][: CAPACITY - self.held]
            if len(take):
                self.held += len(take)
                self.balls = np.delete(self.balls, take, axis=0)

        self.shot_clock += DT
        if enabled and shoot and self.held > 0 and F.feedable(self.x, self.z):
            while self.shot_clock >= SHOT_PERIOD and self.held > 0:
                self.shot_clock -= SHOT_PERIOD
                self.held -= 1
                won = self.blue_auto > 0
                if blue_active(self.t, won) or blue_active(self.t + 3.0, won):
                    self.blue += 1
                    if self.t > 140.0:
                        self.blue_auto += 1
                land = (self.rng.uniform(0.5, 2.8), self.rng.uniform(-2.5, 2.5))
                self.balls = np.vstack([self.balls, land])
        else:
            self.shot_clock = min(self.shot_clock, SHOT_PERIOD)

        self.t -= DT
        if self.t <= 0 and self.end_since is None:
            self.end_since = self.t

    @property
    def done(self) -> bool:
        return self.end_since is not None and (self.end_since - self.t) >= 3.5

    def state(self) -> dict:
        won = 1 if (self.t <= 130.0 and self.blue_auto > 0) else 0
        return {
            "ok": True, "done": self.done, "t": round(self.t, 3), "gs": 0 if self.t > 140 else (3 if self.t <= 0 else 1),
            "rs": 1 if (137.0 < self.t <= 140.0 or self.t <= 0) else 0,
            "blue": self.blue, "red": 0, "blueAuto": self.blue_auto, "hub": 2, "shift": 0, "toShift": 0,
            "wonAuto": won, "gameTime": round(160.0 - self.t, 3),
            "robot": {"x": self.x, "y": 0.0, "z": self.z, "yaw": self.yaw, "pitch": 0, "roll": 0,
                      "vx": self.vx, "vz": self.vz, "wy": 0, "inZone": F.in_blue_zone(self.x)},
            "fuel": [[float(b[0]), float(b[1]), 0.077] for b in self.balls],
            "held": self.held,
        }


def serve(port: int, seed: int = 0) -> threading.Thread:
    """Start a mock bridge on 127.0.0.1:port in a daemon thread."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    def run():
        episode = 0
        while True:
            conn, _ = srv.accept()
            f = conn.makefile("rwb")
            match = None
            for raw in f:
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd in ("hello", "config"):
                    reply = {"ok": True, "mock": True, "fixedDeltaTime": 0.0045, "timeScale": 1.0}
                elif cmd == "reset":
                    episode += 1
                    match = MockMatch(seed=seed * 100003 + episode)
                    reply = match.state()
                elif cmd == "act" and match is not None:
                    match.step(*msg.get("v", [0, 0]), msg.get("rot", 0.0), msg.get("b", [0] * 5))
                    reply = match.state()
                elif cmd == "release":
                    reply = {"ok": True}
                else:
                    reply = {"ok": False, "error": f"bad cmd {cmd}"}
                f.write((json.dumps(reply) + "\n").encode())
                f.flush()
            conn.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th
