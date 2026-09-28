"""Free ball-seeking collector for the neutral-zone phases, on top of the ghost's macro schedule.

In the neutral / conveyor phases (phases.PhaseMap) the robot stops following the human's path and instead goes
where the balls are: every `replan_s` it scores straight intake-first sweeps from its current position (32 headings
× 4 lengths) by the floor balls inside the intake swath, divided by the time the sweep takes (drive + turn), and
drives the best one at full speed, intake-first. Before the ghost's next crossing it drives back to where the ghost
is 1.5 s before that crossing (collecting on the way) and hands control back to GhostPolicy, which then does the
crossing, the zone work and the shooting exactly as before. Buttons in seek mode follow the ghost's (Intake held,
AutoPass when the human was passing), so the dead-window conveyor keeps feeding the zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from .ghost import Ghost, wrap_deg
from .ghost_policy import GhostPolicy
from .phases import PhaseMap

HEADINGS = np.radians(np.arange(0, 360, 11.25))          # 32 candidate directions (field frame, yaw convention)
LENGTHS = np.array([1.0, 1.75, 2.5, 3.5])
X_MIN, X_MAX, Z_ABS = -2.45, 2.6, 3.6                    # stay clear of the red structures / blue hub band / walls


def floor_xz(fuel) -> np.ndarray:
    a = np.asarray(fuel if fuel is not None else [], dtype=np.float32).reshape(-1, 3)
    return a[a[:, 2] < 0.25, :2] if len(a) else np.zeros((0, 2), np.float32)


@dataclass
class SeekParams:
    swath: float = 0.33          # half-width of the intake swath (m)
    speed: float = 2.3           # assumed sweep speed (m/s) for the time estimate
    turn_rate: float = 4.0       # rad/s for the turn-time estimate
    replan_s: float = 0.3
    hysteresis: float = 1.25     # switch target only if the new utility is this much better
    back_margin: float = 0.8     # s of slack when heading back to the ghost's pre-crossing point
    speed_cmd: float = 1.0       # stick magnitude while seeking
    kp_yaw: float = 1.2          # rot per 45° heading error while seeking
    min_balls: float = 2.0       # below this, a sweep is not worth leaving the ghost for


def best_sweep(p_xz, yaw_deg, balls, sp: SeekParams, prev=None):
    """Return (heading_rad, length, utility, balls) of the best straight sweep from p_xz."""
    if not len(balls):
        return None
    dx = balls[:, 0] - p_xz[0]
    dz = balls[:, 1] - p_xz[1]
    best = None
    yaw = math.radians(yaw_deg)
    for h in HEADINGS:
        fx, fz = math.sin(h), math.cos(h)                  # yaw convention: 0 = +z, 90° = +x
        along = dx * fx + dz * fz
        lat = np.abs(dx * fz - dz * fx)
        inlane = lat < sp.swath
        dturn = abs(math.atan2(math.sin(h - yaw), math.cos(h - yaw)))
        for L in LENGTHS:
            ex, ez = p_xz[0] + fx * L, p_xz[1] + fz * L
            if not (X_MIN <= ex <= X_MAX and abs(ez) <= Z_ABS):
                continue
            n = float((inlane & (along > 0.15) & (along < L + 0.35)).sum())
            u = n / (L / sp.speed + dturn / sp.turn_rate + 0.3)
            if best is None or u > best[2]:
                best = (float(h), float(L), u, n)
    if best and prev is not None and prev[2] > 0 and best[2] < prev[2] * sp.hysteresis:
        return prev
    return best


class SeekerPolicy:
    """GhostPolicy everywhere, except neutral/conveyor phases where a greedy ball-seeker drives."""

    def __init__(self, ghost: Ghost, params: dict | None = None, seek: SeekParams | None = None,
                 phases=("neutral", "conveyor")):
        self.tracker = GhostPolicy(ghost, {**GhostPolicy(ghost).p, **(params or {})})
        self.ghost = ghost
        self.pm = PhaseMap(ghost)
        self.sp = seek or SeekParams()
        self.phases = set(phases)
        self.reset()

    # interface used by episode.run_episode / run_ghost
    @property
    def unsticks(self):
        return self.tracker.unsticks

    @property
    def max_err(self):
        return self.tracker.max_err

    @property
    def deploy_waits(self):
        return self.tracker.deploy_waits

    @property
    def resyncs(self):
        return self.tracker.resyncs

    def reset(self) -> None:
        self.tracker.reset()
        self.plan = None               # (heading, length, utility, n, origin_xz)
        self.plan_age = 0.0
        self.mode = "track"
        self.seek_steps = 0
        self.stall_t = 0.0

    def _handback_point(self, t: float):
        """Ghost position 1.5 s before the next crossing, and the clock at which it is there."""
        c = self.pm.next_crossing(t)
        if c is None:
            return None, None
        t_m = c.t_start + 1.5
        g = self.ghost.at(t_m)
        return (g.x, g.z), t_m

    def act(self, s: dict):
        t = float(s["t"])
        rob = s["robot"]
        pos = (rob["x"], rob["z"])
        ph = self.pm.phase(t)
        disabled = s.get("rs", 0) == 1
        seekable = (ph in self.phases) and not disabled and t < 139.0

        if seekable:
            hb, t_m = self._handback_point(t)
            if hb is not None:
                d_back = math.hypot(hb[0] - pos[0], hb[1] - pos[1])
                t_left = t - t_m
                need = d_back / self.sp.speed + self.sp.back_margin
                if t_left <= need:
                    return self._drive_to(s, hb, t, mode="back")
            return self._seek(s, t)

        if self.mode != "track":
            # re-entering tracking: the robot is near the ghost's pre-crossing point by construction
            self.tracker.lost_t = 0.0
        self.mode = "track"
        self.plan = None
        return self.tracker.act(s)

    def _buttons(self, t: float):
        b = [bool(x) for x in self.ghost.at(t).buttons]
        b[0] = True                     # always intake while in the neutral zone
        b[1] = False                    # never AutoShoot out here (it would pass anyway); keep the ghost's AutoPass
        return b

    def _steer(self, rob, heading_rad, speed):
        vx, vz = math.sin(heading_rad) * speed, math.cos(heading_rad) * speed
        yaw_err = float(wrap_deg(rob["yaw"] - math.degrees(heading_rad)))       # rot > 0 decreases yaw
        rot = float(np.clip(self.sp.kp_yaw * yaw_err / 45.0, -1.0, 1.0))
        return vx, vz, rot

    def _seek(self, s, t):
        rob = s["robot"]
        pos = (rob["x"], rob["z"])
        self.plan_age += 0.099
        balls = floor_xz(s.get("fuel"))
        if self.plan is None or self.plan_age >= self.sp.replan_s:
            prev = self.plan[:4] if self.plan else None
            b = best_sweep(pos, rob["yaw"], balls, self.sp, prev)
            if b is None or b[3] < self.sp.min_balls:
                self.mode = "seek-idle"
                self.plan = None
                return self.tracker.act(s)          # nothing worth chasing: follow the human's path
            self.plan = (*b, pos) if (prev is None or b is not prev) else self.plan
            self.plan_age = 0.0
        h, L, u, n, origin = self.plan
        done = math.hypot(pos[0] - origin[0], pos[1] - origin[1]) >= L
        if done:
            self.plan = None
        # stall: pushing but not moving → drop the plan (a new one is chosen next step)
        speed = math.hypot(rob.get("vx", 0.0), rob.get("vz", 0.0))
        self.stall_t = self.stall_t + 0.099 if speed < 0.25 else 0.0
        if self.stall_t > 0.6:
            self.plan, self.stall_t = None, 0.0
        vx, vz, rot = self._steer(rob, h, self.sp.speed_cmd)
        self.mode = "seek"
        self.seek_steps += 1
        return vx, vz, rot, self._buttons(t), {"mode": "seek", "plan_n": n, "err": 0.0, "ghost_blue": self.ghost.at(t).blue,
                                              "ghost_held": self.ghost.at(t).held, "phase": "seek"}

    def _drive_to(self, s, target, t, mode):
        rob = s["robot"]
        dx, dz = target[0] - rob["x"], target[1] - rob["z"]
        d = math.hypot(dx, dz)
        h = math.atan2(dx, dz)
        spd = 1.0 if d > 0.6 else max(0.3, d / 0.6)
        vx, vz, rot = self._steer(rob, h, spd)
        self.mode = mode
        return vx, vz, rot, self._buttons(t), {"mode": mode, "err": round(d, 2), "ghost_blue": self.ghost.at(t).blue,
                                               "ghost_held": self.ghost.at(t).held, "phase": mode}
