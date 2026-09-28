"""A recorded human match (run/demos/*.jsonl) as a time-indexed "ghost" the bot can follow.

The fuel layout and the robot's start pose are identical every match (checked across the 1085 and 1118 demos:
496 balls, max per-ball offset 1.3 cm), so "be where the human was, doing what the human did, at the same match
clock" is a meaningful target. Consumers: GhostPolicy (tracker), the ghost-tracking reward, the BC dataset.

Indexing: by the match clock t (160 → 0), not by record count. The clock is what the hub shifts are tied to, and
it is what the Bridge reports. The 3 s auto→teleop pause freezes t at 140 with the robot disabled; those records
collapse onto t = 140 and the last one (the pose teleop starts from) wins.

Driver input → field frame (verified by correlating sticks with the next record's velocity, both demos):
field-centric, camera yaw 270 → world vx ∝ -ty, world vz ∝ +tx (r ≈ 0.93); the rotate axis already follows the
Bridge convention (rot > 0 → yaw decreases, r(rot, wy) ≈ -0.9).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUTO, TELEOP, ENDGAME, END = range(4)      # MoSimCore.Enums.GameState
ENABLED, DISABLED = 0, 1                   # MoSimCore.Enums.RobotState
N_BUTTONS = 5                              # intake, autoshoot, autopass, manualshoot, special (client.py order)


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                       # partially flushed last line
    return rows


def match_slice(rows: list[dict]) -> list[dict]:
    """The one real match inside a recording: drop quick restarts at the start (the timer jumps back up by < 1 s,
    which the Recorder does not split on) and the t = 0 rows it keeps writing after the buzzer."""
    t = [r["t"] for r in rows]
    start = 0
    for i in range(1, len(rows)):
        if t[i] > 140.0 and t[i] > t[i - 1] + 0.05:
            start = i
    while start < len(rows) and not (rows[start]["gs"] == AUTO and rows[start]["rs"] == ENABLED):
        start += 1
    end = next((i for i in range(start + 1, len(rows)) if t[i] <= 0.01), len(rows) - 1)
    return rows[start:end + 50]            # 5 s past the buzzer: balls in flight still score (1094 → 1118 in one demo)


def field_command(r: dict) -> tuple[float, float, float]:
    """The driver's stick input as a Bridge action (field vx, field vz, rot), each in [-1, 1]."""
    i = r.get("in") or {}
    tx, ty = i.get("tx", 0.0), i.get("ty", 0.0)
    cam = i.get("cam")
    if i.get("fc", 1) == 1 and (cam is None or abs(cam - 270.0) < 1.0):
        return -ty, tx, i.get("rot", 0.0)
    # general field-centric case: rotate the stick by the camera yaw (unused so far — both demos are cam 270)
    a = math.radians(cam)
    return tx * math.cos(a) - ty * math.sin(a), tx * math.sin(a) + ty * math.cos(a), i.get("rot", 0.0)


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


@dataclass
class GhostFrame:
    t: float
    x: float
    z: float
    yaw: float
    vx: float
    vz: float
    held: int
    blue: int
    cmd: tuple[float, float, float]    # (field vx, field vz, rot) stick command, [-1, 1]
    buttons: tuple[int, ...]


class Ghost:
    def __init__(self, rows: list[dict], name: str = ""):
        rows = match_slice(rows)
        # one row per distinct clock value, keeping the last (post-pause pose at t = 140)
        by_t: dict[float, dict] = {}
        for r in rows:
            by_t[round(r["t"], 3)] = r
        keep = sorted(by_t.items(), key=lambda kv: -kv[0])
        self.name = name
        self.rows = [r for _, r in keep]
        self.t = np.array([k for k, _ in keep])                      # descending
        self.x = np.array([r["x"] for r in self.rows]); self.z = np.array([r["z"] for r in self.rows])
        self.yaw_unwrapped = np.degrees(np.unwrap(np.radians([r["yaw"] for r in self.rows])))
        self.vx = np.array([r["vx"] for r in self.rows]); self.vz = np.array([r["vz"] for r in self.rows])
        self.held = np.array([r["held"] for r in self.rows]); self.blue = np.array([r["blue"] for r in self.rows])
        self.cmd = np.array([field_command(r) for r in self.rows])
        self.buttons = np.array([(r.get("in") or {}).get("b", [0] * N_BUTTONS) for r in self.rows], dtype=int)
        self.final = int(self.blue[-1])

    @classmethod
    def from_file(cls, path: str | Path) -> "Ghost":
        return cls(load_rows(path), Path(path).stem)

    def _interp(self, arr: np.ndarray, t: float) -> float:
        # self.t is descending; np.interp needs ascending
        return float(np.interp(-t, -self.t, arr))

    def index(self, t: float) -> int:
        """Most recent record at or before match clock t (clock counts down)."""
        i = int(np.searchsorted(-self.t, -t, side="right")) - 1
        return max(0, min(i, len(self.t) - 1))

    def at(self, t: float) -> GhostFrame:
        i = self.index(t)
        return GhostFrame(t=t, x=self._interp(self.x, t), z=self._interp(self.z, t),
                          yaw=float(wrap_deg(self._interp(self.yaw_unwrapped, t))),
                          vx=self._interp(self.vx, t), vz=self._interp(self.vz, t),
                          held=int(self.held[i]), blue=int(self.blue[i]),
                          cmd=tuple(float(c) for c in self.cmd[i]), buttons=tuple(int(b) for b in self.buttons[i]))

    def score_at(self, t: float) -> int:
        return int(self.blue[self.index(t)])


def mirror_ghost(g: Ghost) -> Ghost:
    """The same demo reflected across the field centreline (z → −z). The neutral zone, hubs, bumps and trenches are
    symmetric in z, so this is a feasible route that sweeps the other side; the blue-zone depot / outpost / tower are
    not symmetric (use with care). Heading: yaw → 180° − yaw; field stick: vz → −vz; rotate stick → −rot."""
    import copy
    m = copy.copy(g)
    m.name = g.name + "-mirror"
    m.z = -g.z
    m.vz = -g.vz
    m.yaw_unwrapped = 180.0 - g.yaw_unwrapped
    m.cmd = g.cmd * np.array([1.0, -1.0, -1.0])
    return m
