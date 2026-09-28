"""Parameterised scripted policy for 4414 HighTide (blue) — the thing CMA-ES tunes.

Two modes:
  COLLECT  drive intake-first to the best floor-ball target (utility = density − travel cost − x preference)
  SCORE    drive to the shooting point, holding AutoShoot whenever feeding is possible and the hub is (about to be)
           active for blue — including a lead before a window opens and a tail into the 3 s post-deactivation grace.

Crossing between the neutral zone and the blue zone uses the bump lanes at full throttle (4414 does not fit under the trench bar).
All geometry comes from field.py; all behaviour knobs are in PARAM_SPEC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import field as F
from .client import INTAKE, SHOOT
from .shifts import BLUE, blue_active, seconds_until_blue_active

# name, low, high, default — CMA-ES searches the unit cube and maps through these ranges.
PARAM_SPEC = [
    ("shoot_x",            4.10, 6.80, 5.20),
    ("shoot_z",           -3.30, 3.30, 0.00),
    ("fill_active",        3.0, 60.0, 16.0),
    ("fill_inactive",      8.0, 100.0, 40.0),
    ("max_collect_s",      3.0, 25.0, 10.0),
    ("density_r",          0.30, 1.50, 0.60),
    ("w_density",          0.00, 3.00, 1.00),
    ("w_dist",             0.20, 3.00, 1.00),
    ("x_pref",            -3.00, 7.50, 1.50),
    ("w_xpref",            0.00, 2.00, 0.30),
    ("collect_speed",      0.15, 0.70, 0.35),
    ("drive_speed",        0.50, 1.00, 1.00),
    ("lead_s",             0.00, 3.00, 1.80),
    ("tail_s",             0.00, 3.00, 1.00),
    ("return_margin_s",    0.00, 6.00, 2.00),
    ("intake_while_shoot", 0.00, 1.00, 0.30),
    ("shoot_on_move",      0.00, 1.00, 0.80),
]
PARAM_NAMES = [p[0] for p in PARAM_SPEC]


def default_params() -> dict:
    return {n: d for n, _, _, d in PARAM_SPEC}


def from_unit(u) -> dict:
    return {n: lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo) for (n, lo, hi, _), x in zip(PARAM_SPEC, u)}


def to_unit(params: dict) -> np.ndarray:
    return np.array([(params[n] - lo) / (hi - lo) for n, lo, hi, _ in PARAM_SPEC])


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def region(x: float) -> str:
    if x >= F.ZONE_LINE:
        return "Z"
    if x <= F.NEUTRAL_FACE:
        return "N"
    return "B"


STALL_S = 1.2    # s of feeding with no progress before leftover balls are written off
STUCK_MAX = 6    # at most this many leftover balls may be written off as stuck
VMAX_EST = 2.7   # m/s, calibrated top speed (runs/calib/calibration.json)


@dataclass
class ScriptedPolicy:
    p: dict = dc_field(default_factory=default_params)
    mode: str = "SCORE"
    lane: float | None = None
    target: tuple | None = None
    target_u: float = -1e9
    collect_start: float = 0.0
    empty_steps: int = 0
    shoot_since: float | None = None
    held_at_shoot: int = 0
    hist: list = dc_field(default_factory=list)          # [(t, x, z)] recent positions
    blacklist: list = dc_field(default_factory=list)     # [(x, z, expires_at_timer)]
    backoff_until: float | None = None

    def reset(self) -> None:
        self.mode = "SCORE"          # 8 preloads: score first
        self.lane = None
        self.target = None
        self.target_u = -1e9
        self.collect_start = 0.0
        self.empty_steps = 0
        self.shoot_since = None
        self.held_at_shoot = 0
        self.hist = []
        self.blacklist = []
        self.backoff_until = None

    # ------------------------------------------------------------------ geometry helpers

    def lanes(self):
        return list(F.BUMP_LANES)          # 4414 cannot pass under the trench bar (calibrated)

    def best_lane(self, z_from: float, z_to: float) -> float:
        return min(self.lanes(), key=lambda lz: abs(z_from - lz) + abs(z_to - lz))

    def path_len(self, a, b) -> float:
        ra, rb = region(a[0]), region(b[0])
        if ra == rb or "B" in (ra, rb):
            return math.dist(a, b)
        lz = self.best_lane(a[1], b[1])
        xa = F.CROSS_X_NEUTRAL if ra == "N" else F.CROSS_X_ZONE
        xb = F.CROSS_X_ZONE if ra == "N" else F.CROSS_X_NEUTRAL
        return math.dist(a, (xa, lz)) + abs(xb - xa) + math.dist((xb, lz), b)

    def next_waypoint(self, pos, goal):
        """Returns (waypoint, speed_scale). Handles neutral<->zone crossings through a locked lane."""
        rp, rg = region(pos[0]), region(goal[0])
        if rp == rg and rp != "B":
            self.lane = None
            return goal, 1.0
        toward_zone = goal[0] > pos[0]
        if self.lane is None:
            self.lane = self.best_lane(pos[1], goal[1]) if rp != "B" else min(self.lanes(), key=lambda lz: abs(pos[1] - lz))
        lz = self.lane
        entry_x = F.CROSS_X_NEUTRAL if toward_zone else F.CROSS_X_ZONE
        exit_x = F.CROSS_X_ZONE if toward_zone else F.CROSS_X_NEUTRAL
        on_lane = abs(pos[1] - lz) < 0.30
        before_entry = pos[0] < entry_x - 0.15 if toward_zone else pos[0] > entry_x + 0.15
        if rp != "B" and (before_entry or not on_lane):
            return (entry_x, lz), 1.0
        past_exit = pos[0] >= exit_x - 0.10 if toward_zone else pos[0] <= exit_x + 0.10
        if past_exit:
            self.lane = None
            return goal, 1.0
        return (exit_x, lz), 1.0            # full throttle: 0.6 stalls on the bump ramp

    def moved(self, t: float, window: float) -> float | None:
        """Distance travelled over the last `window` timer-seconds (None if not enough history)."""
        old = [h for h in self.hist if h[0] - t >= window]
        if not old:
            return None
        _, x0, z0 = old[-1]
        _, x1, z1 = self.hist[-1]
        return math.hypot(x1 - x0, z1 - z0)

    # ------------------------------------------------------------------ target selection

    def pick_target(self, pos, fuel: np.ndarray, shoot_pt):
        if fuel.size == 0:
            return (1.5, 0.0), -1e9
        x, z, y = fuel[:, 0], fuel[:, 1], fuel[:, 2]
        floor = y < 0.25
        ok = floor & (np.abs(z) < F.Z_WALL - 0.45) & (x < F.X_WALL - 0.35) & (x > -3.2)
        ok &= ~((x > F.NEUTRAL_FACE - 0.10) & (x < F.ZONE_LINE + 0.25))            # hub / bump / trench band
        tx0, tx1, tz0, tz1 = F.TOWER_KEEPOUT
        ok &= ~((x > tx0 - 0.3) & (x < tx1) & (z > tz0 - 0.3) & (z < tz1 + 0.3))
        for bx, bz, _ in self.blacklist:
            ok &= (x - bx) ** 2 + (z - bz) ** 2 > 0.6 ** 2
        cand = fuel[ok][:, :2]
        if len(cand) == 0:
            return (1.5, 0.0), -1e9
        if len(cand) > 250:
            cand = cand[:: int(math.ceil(len(cand) / 250))]
        floor_xz = fuel[floor][:, :2]
        r = self.p["density_r"]
        d2 = ((cand[:, None, :] - floor_xz[None, :, :]) ** 2).sum(-1)
        density = (d2 < r * r).sum(1).astype(float)
        travel = np.array([self.path_len(pos, tuple(c)) for c in cand])
        ret = np.array([self.path_len(tuple(c), shoot_pt) for c in cand])
        u = (self.p["w_density"] * density - self.p["w_dist"] * (travel + 0.5 * ret)
             - self.p["w_xpref"] * np.abs(cand[:, 0] - self.p["x_pref"]))
        i = int(np.argmax(u))
        best, best_u = (float(cand[i, 0]), float(cand[i, 1])), float(u[i])
        # hysteresis: keep the previous target unless the new one is clearly better
        if self.target is not None and self.target_u > best_u - 1.0:
            dprev = np.min(((cand - np.array(self.target)) ** 2).sum(1)) if len(cand) else 1e9
            if dprev < 0.3 ** 2:
                return self.target, self.target_u
        return best, best_u

    # ------------------------------------------------------------------ main step

    def act(self, s: dict):
        p = self.p
        rob = s["robot"]
        pos = (rob["x"], rob["z"])
        yaw = rob["yaw"]
        t = s["t"]
        held = s.get("held", 0)
        fuel = np.asarray(s.get("fuel", []), dtype=float).reshape(-1, 3)

        won = (s.get("wonAuto", -1) == BLUE) if t <= 130.0 else (s.get("blueAuto", 0) > 0)
        active_now = blue_active(t, won)
        to_active = seconds_until_blue_active(t, won)
        shoot_window = active_now or to_active <= p["lead_s"] or blue_active(t + p["tail_s"], won)

        shoot_pt = (p["shoot_x"], p["shoot_z"])
        if F.in_box(*shoot_pt, F.TOWER_KEEPOUT, 0.4):
            shoot_pt = (min(shoot_pt[0], F.TOWER_KEEPOUT[0] - 0.5), shoot_pt[1])
        est_return = self.path_len(pos, shoot_pt) / (VMAX_EST * p["drive_speed"]) + 0.5

        # A few balls can sit in the slide/hopper where they never feed (calibration: 1-3 balls). If we have
        # been feeding for STALL_S without the count dropping and only a handful remain, treat it as empty.
        stalled = (self.shoot_since is not None and held <= STUCK_MAX and held >= self.held_at_shoot
                   and (self.shoot_since - t) >= STALL_S)
        self.empty_steps = self.empty_steps + 1 if (held <= 0 or stalled) else 0
        self.hist.append((t, pos[0], pos[1]))
        self.hist = [h for h in self.hist if h[0] - t <= 3.0]
        self.blacklist = [b for b in self.blacklist if b[2] < t]

        # -------- mode transitions
        if self.mode == "SCORE":
            if self.empty_steps >= 3 and t > 1.0:
                self._enter_collect(t)
            elif not shoot_window and held < p["fill_inactive"] and to_active > est_return + p["return_margin_s"] + 4.0:
                self._enter_collect(t)
        else:  # COLLECT
            fill = p["fill_active"] if (active_now or to_active < est_return + 3.0) else p["fill_inactive"]
            timed_out = (self.collect_start - t) > p["max_collect_s"] and held > 0   # timer counts down
            must_return = (not active_now) and to_active <= est_return + p["return_margin_s"] and held > 0
            endgame = t < est_return + 2.0 and held > 0
            if held >= fill or timed_out or must_return or endgame:
                self.mode = "SCORE"
                self.target = None

        # -------- motion
        buttons = [False] * 5
        rot = 0.0
        if self.mode == "COLLECT":
            self.target, self.target_u = self.pick_target(pos, fuel, shoot_pt)
            goal = self.target
            buttons[INTAKE] = True
        else:
            goal = shoot_pt
            buttons[INTAKE] = p["intake_while_shoot"] > 0.5

        wp, speed_scale = self.next_waypoint(pos, goal)
        crossing_run = self.lane is not None and wp[0] in (F.CROSS_X_ZONE, F.CROSS_X_NEUTRAL) and \
            abs(pos[1] - self.lane) < 0.30 and region(pos[0]) != region(wp[0])
        approaching_lane = self.lane is not None and not crossing_run
        dx, dz = wp[0] - pos[0], wp[1] - pos[1]
        d = math.hypot(dx, dz)
        speed = p["drive_speed"] * speed_scale
        if wp == goal:
            if self.mode == "COLLECT":
                # drive in at full speed, then creep through the balls (fast passes bulldoze them away)
                speed = p["drive_speed"] if d > 1.2 else p["collect_speed"]
            else:
                speed *= min(1.0, d / 0.6)
        vx, vz = (dx / d * speed, dz / d * speed) if d > 1e-3 else (0.0, 0.0)

        if approaching_lane:
            # square up to yaw 0/180 before the bump run (the orientation that crossed in calibration)
            want = 0.0 if abs(wrap_deg(yaw)) < 90 else 180.0
            rot = float(np.clip(wrap_deg(yaw - want) / 45.0, -1.0, 1.0))
        elif self.mode == "COLLECT" and d > 0.2 and not crossing_run:
            want = math.degrees(math.atan2(dx, dz))           # yaw 0 = +z, 90 = +x
            rot = float(np.clip(wrap_deg(yaw - want) / 45.0, -1.0, 1.0))   # rot > 0 decreases yaw

        # ---- unstick
        moved = self.moved(t, 1.2)
        if crossing_run or region(pos[0]) == "B":
            if self.backoff_until is not None and t > self.backoff_until:
                vx, vz, rot = -vx, -vz, 0.0                     # back off, then retry the run
            elif self.backoff_until is not None:
                self.backoff_until = None
                self.hist = []
            elif moved is not None and moved < 0.10:
                self.backoff_until = t - 0.6
                vx, vz, rot = -vx, -vz, 0.0
        elif self.mode == "COLLECT" and moved is not None and moved < 0.15 and self.target is not None \
                and math.dist(pos, self.target) > 0.5 and (self.collect_start - t) > 1.5:
            self.blacklist.append((self.target[0], self.target[1], t - 20.0))
            self.target, self.target_u = None, -1e9
            self.hist = []

        feed_ok = F.feedable(*pos) and held > 0 and shoot_window
        if feed_ok and (self.mode == "SCORE" or p["shoot_on_move"] > 0.5):
            if p["shoot_on_move"] > 0.5 or math.dist(pos, shoot_pt) < 0.5:
                buttons[SHOOT] = True

        if buttons[SHOOT]:
            if self.shoot_since is None or held < self.held_at_shoot:
                self.shoot_since, self.held_at_shoot = t, held       # (re)start the stall clock on progress
        else:
            self.shoot_since = None

        info = {"mode": self.mode, "goal": goal, "wp": wp, "active": active_now, "win": shoot_window,
                "held": held, "to_active": round(to_active, 2)}
        return vx, vz, rot, buttons, info

    def _enter_collect(self, t: float) -> None:
        self.mode = "COLLECT"
        self.collect_start = t
        self.target = None
        self.target_u = -1e9
