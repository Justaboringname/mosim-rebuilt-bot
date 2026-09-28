"""Residual layer between a (learned or heuristic) decision u ∈ [-1, 1]^k and GhostPolicy.

u[0] lateral: desired sideways offset from the ghost's path, as a fraction of the phase leash L
u[1] warp:    rate at which the ghost clock runs ahead (+) or behind (−), as a fraction of 0.3 s/s, |warp| ≤ W

The decoder filters and rate-limits the offset, shrinks the leash smoothly before crossings (so the robot is back on
the ghost's lane for bump/trench runs), clamps the displaced target out of places the ghost itself is not (feed
gates, walls, tower, the far side), and hands GhostPolicy.act its offset/warp/off_vel arguments. u = 0 reproduces
the plain tracker exactly.

Policies here: ZeroResidual (control arm), GreedySwath (non-learned "look at the balls": shift toward the lane
with the most floor balls on the ghost's next 1.5 s of path), OUResidual (exploration probe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import field as F
from .ghost import Ghost
from .ghost_policy import GhostPolicy
from .phases import PhaseMap, leash

DT = 0.099                 # Bridge decision period (s)
CLEAR_SAFE = 1.2           # m: the displaced target may not be closer to a structure than the ghost, capped at this


_CLEAR = None


def structure_boxes() -> np.ndarray:
    """(M, 4) x0 x1 z0 z1 footprints the residual must keep clear of: every field collider at robot height
    (minisim/robot_obstacles.json: walls, hubs, trench dividers, towers) plus the four bumps. Pushing balls against a
    bump or hub face jams the real robot (ab_c50: 21 of 32 episodes), and MiniSim does not reproduce the jam."""
    global _CLEAR
    if _CLEAR is None:
        import json
        from pathlib import Path
        from minisim.field import BUMPS
        rows = json.load(open(Path(__file__).resolve().parents[1] / "minisim/robot_obstacles.json"))
        _CLEAR = np.array([[r["x"][0], r["x"][1], r["z"][0], r["z"][1]] for r in rows] +
                          [[x0, x1, z0, z1] for (x0, x1), (z0, z1) in BUMPS], np.float64)
    return _CLEAR


def structure_dist(x: float, z: float) -> float:
    B = structure_boxes()
    dx = np.maximum(np.maximum(B[:, 0] - x, 0.0), x - B[:, 1])
    dz = np.maximum(np.maximum(B[:, 2] - z, 0.0), z - B[:, 3])
    return float(np.sqrt(dx * dx + dz * dz).min())
SWATH_OFFSETS = np.linspace(-3.0, 3.0, 17)
SWATH_R = 0.36             # half-width of the intake swath (m)
SWATH_HORIZON = np.arange(0.0, 1.51, 0.25)


def floor_balls(fuel) -> np.ndarray:
    a = np.asarray(fuel if fuel is not None else [], dtype=np.float32).reshape(-1, 3)
    return a[a[:, 2] < 0.25, :2] if len(a) else np.zeros((0, 2), np.float32)


def path_normal(g: Ghost, tau: float, last=(0.0, 1.0)) -> tuple[float, float]:
    """Left normal of the ghost's direction of travel over the next 0.5 s (field frame, x/z)."""
    a, b = g.at(tau), g.at(tau - 0.5)
    dx, dz = b.x - a.x, b.z - a.z
    n = math.hypot(dx, dz)
    if n < 0.2:                      # nearly stopped: keep the previous normal
        return last
    return (-dz / n, dx / n)


def swath_yield(g: Ghost, tau: float, balls: np.ndarray, normal) -> np.ndarray:
    """Floor balls within SWATH_R of the ghost's next-1.5 s path shifted sideways by each of SWATH_OFFSETS."""
    if not len(balls):
        return np.zeros(len(SWATH_OFFSETS), np.float32)
    pts = np.array([[g.at(tau - h).x, g.at(tau - h).z] for h in SWATH_HORIZON], np.float32)   # H x 2
    nrm = np.asarray(normal, np.float32)
    shifted = pts[None, :, :] + SWATH_OFFSETS[:, None, None].astype(np.float32) * nrm[None, None, :]  # O x H x 2
    d2 = ((shifted[:, :, None, :] - balls[None, None, :, :]) ** 2).sum(-1)                    # O x H x N
    near = (d2.min(1) < SWATH_R ** 2)                                                            # O x N
    return near.sum(1).astype(np.float32)


@dataclass
class ResidualDecoder:
    ghost: Ghost
    phases: PhaseMap
    leash_scale: dict = dc_field(default_factory=dict)   # phase → fraction of the final leash unlocked
    ramp_s: float = 2.5                                   # leash shrinks to 0 over this long before a crossing mask
    heading_max: float = 30.0                             # max intake-first heading correction (deg); 0 = off
    clearance: bool = False                               # keep the displaced target as clear of structures as the ghost
    lateral: float = 0.0
    warp: float = 0.0
    normal: tuple = (0.0, 1.0)
    off_prev: tuple = (0.0, 0.0)

    def reset(self) -> None:
        self.lateral, self.warp, self.normal, self.off_prev = 0.0, 0.0, (0.0, 1.0), (0.0, 0.0)

    def limits(self, tau: float) -> tuple[str, float, float]:
        ph = self.phases.phase(tau)
        L, W = leash(ph, self.leash_scale)
        # shrink ahead of the next crossing mask so we are back on the lane when the run-up starts
        c = self.phases.next_crossing(tau)
        if c is not None:
            to_mask = tau - (c.t_start + 1.5)
            f = float(np.clip(to_mask / self.ramp_s, 0.0, 1.0))
            L, W = L * f, W * f
        return ph, L, W

    def decode(self, s: dict, u) -> dict:
        """u → GhostPolicy.act kwargs, plus diagnostics."""
        t = float(s["t"])
        disabled = s.get("rs", 0) == 1
        tau = t - self.warp
        ph, L, W = self.limits(tau)
        if disabled:
            L, W = 0.0, 0.0
        u_l = float(u[0]) if len(u) > 0 else 0.0
        u_w = float(u[1]) if len(u) > 1 else 0.0

        # warp: integrate the requested rate, then pull back inside |warp| ≤ W at ≤ 1 s/s
        self.warp += 0.3 * u_w * DT
        if abs(self.warp) > W:
            self.warp = math.copysign(max(W, abs(self.warp) - 1.0 * DT), self.warp)
        tau = t - self.warp

        g = self.ghost.at(tau)
        self.normal = path_normal(self.ghost, tau, self.normal)
        shooting = bool(g.buttons[1])
        rate = (0.6 if shooting else 1.5) * DT
        target = float(np.clip(u_l * L, -L, L))
        self.lateral += float(np.clip(target - self.lateral, -rate, rate))
        if abs(self.lateral) > L:                         # leash shrank: come back at the rate limit
            self.lateral = math.copysign(max(L, abs(self.lateral) - rate), self.lateral)

        ox, oz = self.lateral * self.normal[0], self.lateral * self.normal[1]
        # rate-limit the offset VECTOR too: at ghost U-turns the left normal flips sides and a scalar-limited offset
        # would jump across the path (up to 6 m/s of target motion)
        dx_, dz_ = ox - self.off_prev[0], oz - self.off_prev[1]
        dn = math.hypot(dx_, dz_)
        if dn > rate:
            ox, oz = self.off_prev[0] + dx_ * rate / dn, self.off_prev[1] + dz_ * rate / dn
        # clamp the displaced target to where the ghost itself may be (never tighter than the ghost's own position)
        tx, tz = g.x + ox, g.z + oz
        if g.x <= 3.05:
            tx = float(np.clip(tx, min(-2.9, g.x), max(2.9, g.x)))
        elif g.x >= 4.25:
            tx = max(tx, min(4.4, g.x))
        tx = float(np.clip(tx, min(-7.8, g.x), max(7.8, g.x)))
        tz = float(np.clip(tz, min(-3.6, g.z), max(3.6, g.z)))
        if F.in_box(tx, tz, F.TOWER_KEEPOUT, 0.3) and not F.in_box(g.x, g.z, F.TOWER_KEEPOUT, 0.3):
            tx = min(tx, F.TOWER_KEEPOUT[0] - 0.3)
        ox, oz = tx - g.x, tz - g.z
        if self.clearance and (ox or oz):
            # never closer to a structure than the ghost itself is (capped at CLEAR_SAFE): shrink the offset along
            # its own direction until the displaced target is at least that far out
            need = min(structure_dist(g.x, g.z), CLEAR_SAFE)
            if structure_dist(tx, tz) < need - 1e-3:
                lo, hi = 0.0, 1.0
                for _ in range(10):
                    mid = 0.5 * (lo + hi)
                    if structure_dist(g.x + mid * ox, g.z + mid * oz) >= need - 1e-3:
                        lo = mid
                    else:
                        hi = mid
                ox, oz = lo * ox, lo * oz
                tx, tz = g.x + ox, g.z + oz
        ovx, ovz = (ox - self.off_prev[0]) / DT, (oz - self.off_prev[1]) / DT
        self.off_prev = (ox, oz)
        # Intake-first while displaced: the human drives intake-first while collecting; when the residual moves the
        # target sideways, turn the heading toward the displaced direction of travel (≤ heading_max°), otherwise
        # the robot crabs sideways and plows balls with its side bumper instead of taking them in.
        dyaw = 0.0
        vx_, vz_ = g.vx + ovx, g.vz + ovz
        if self.heading_max > 0 and math.hypot(g.vx, g.vz) > 0.8 and math.hypot(vx_, vz_) > 0.5 and g.buttons[0]:
            ghost_motion = math.degrees(math.atan2(g.vx, g.vz))
            if abs(((g.yaw - ghost_motion) + 180) % 360 - 180) < 35:          # human is driving intake-first here
                want = math.degrees(math.atan2(vx_, vz_))
                dyaw = float(np.clip(((want - ghost_motion) + 180) % 360 - 180, -self.heading_max, self.heading_max))
        return {"kwargs": {"offset": (ox, oz), "warp": self.warp, "off_vel": (ovx, ovz), "dyaw": dyaw},
                "phase": ph, "L": L, "W": W, "tau": tau, "lateral": self.lateral, "normal": self.normal}


class ResidualAgent:
    """Wraps GhostPolicy + ResidualDecoder behind the episode.py policy interface; `chooser(s, dec) -> u`."""

    def __init__(self, ghost: Ghost, chooser, params: dict | None = None, leash_scale: dict | None = None,
                 decide_every: int = 2, heading_max: float = 30.0, clearance: bool = False):
        self.ghost = ghost
        self.tracker = GhostPolicy(ghost, {**GhostPolicy(ghost).p, **(params or {})})
        self.phases = PhaseMap(ghost)
        self.dec = ResidualDecoder(ghost, self.phases, leash_scale or {}, heading_max=heading_max, clearance=clearance)
        self.chooser = chooser
        self.decide_every = decide_every
        self.k = 0
        self.u = np.zeros(2, np.float32)
        self.last = {}

    @property
    def unsticks(self):
        return self.tracker.unsticks

    @property
    def max_err(self):
        return self.tracker.max_err

    @property
    def deploy_waits(self):
        return self.tracker.deploy_waits

    def reset(self) -> None:
        self.tracker.reset(); self.dec.reset(); self.k = 0; self.u = np.zeros(2, np.float32)
        if hasattr(self.chooser, "reset"):
            self.chooser.reset()

    def act(self, s: dict):
        if self.k % self.decide_every == 0:
            self.u = np.asarray(self.chooser(s, self), np.float32)
        self.k += 1
        d = self.dec.decode(s, self.u)
        ux, uz, rot, buttons, info = self.tracker.act(s, **d["kwargs"])
        info.update({"phase": d["phase"], "lat": round(d["lateral"], 2), "warp": round(self.dec.warp, 2),
                     "L": round(d["L"], 2), "u": [round(float(v), 3) for v in self.u]})
        self.last = d
        return ux, uz, rot, buttons, info


def zero_chooser(s, agent) -> np.ndarray:
    return np.zeros(2, np.float32)


class GreedySwath:
    """Non-learned ball-seeker: pick the lateral offset (within the current leash) whose shifted 1.5 s path sweeps the
    most floor balls; prefer staying put unless another lane is clearly better (hysteresis)."""

    def __init__(self, margin: float = 3.0):
        self.margin = margin
        self.current = 0.0

    def reset(self):
        self.current = 0.0

    def __call__(self, s, agent: ResidualAgent) -> np.ndarray:
        dec = agent.dec
        tau = float(s["t"]) - dec.warp
        ph, L, W = dec.limits(tau)
        if L < 0.05:
            self.current = 0.0
            return np.zeros(2, np.float32)
        y = swath_yield(agent.ghost, tau, floor_balls(s.get("fuel")), path_normal(agent.ghost, tau, dec.normal))
        ok = np.abs(SWATH_OFFSETS) <= L + 1e-6
        best = float(SWATH_OFFSETS[ok][np.argmax(y[ok])])
        cur_idx = int(np.argmin(np.abs(SWATH_OFFSETS - self.current)))
        if y[ok].max() >= y[cur_idx] + self.margin or not ok[cur_idx]:
            self.current = best
        return np.array([self.current / max(L, 1e-6), 0.0], np.float32)


class OUResidual:
    """Ornstein–Uhlenbeck lateral noise (exploration / sensitivity probe)."""

    def __init__(self, sigma: float = 0.5, tau_s: float = 1.5, seed: int | None = None):
        self.sigma, self.tau_s = sigma, tau_s
        self.rng = np.random.default_rng(seed)
        self.x = 0.0

    def reset(self):
        self.x = 0.0

    def __call__(self, s, agent: ResidualAgent) -> np.ndarray:
        dt = DT * agent.decide_every
        self.x += -self.x * dt / self.tau_s + self.sigma * math.sqrt(2 * dt / self.tau_s) * self.rng.normal()
        return np.array([float(np.clip(self.x, -1, 1)), 0.0], np.float32)


class ScheduleChooser:
    """Static residual: lateral u (fraction of the leash) as a piecewise-linear function of the match clock, given as
    {clock: u} knots — lets CMA-ES search whole-segment route shifts that are scored end to end."""

    def __init__(self, knots: dict):
        items = sorted(((float(k), float(v)) for k, v in knots.items()), key=lambda kv: -kv[0])
        self.t = np.array([-k for k, _ in items])
        self.u = np.array([v for _, v in items])

    def reset(self):
        pass

    def __call__(self, s, agent) -> np.ndarray:
        if not len(self.t):
            return np.zeros(2, np.float32)
        return np.array([float(np.clip(np.interp(-float(s["t"]), self.t, self.u), -1, 1)), 0.0], np.float32)
