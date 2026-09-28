"""Phase labelling of a Ghost's timeline, and how much a residual policy may deviate from the ghost in each phase.

Phases (by the ghost's state at clock tau):
  opening      t > 140: deterministic first cycle, the proven tracker owns it
  crossing     1.5 s before → 0.5 s after each ghost bump / trench crossing (band 3.05 < x < 4.25)
  pause        the disabled auto→teleop pause (handled by the caller: robot state disabled)
  neutral      ghost in the neutral zone (x ≤ 3.05) — ball collection; widest leash
  conveyor     ghost in neutral during a blue-inactive window (collect + AutoPass to stock the zone)
  harvest      ghost in the blue zone with Intake held and not shooting
  shoot        ghost in the blue zone holding AutoShoot

The leash (max lateral offset L in metres, max clock warp W in seconds) per phase is a curriculum: every phase
starts at `start` and is widened by the trainer up to `final` (docs/design/verdict-2026-09-26.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ghost import Ghost
from .shifts import blue_active

BAND = (3.05, 4.25)
PHASES = ("opening", "crossing", "pause", "neutral", "conveyor", "harvest", "shoot")
FINAL_LEASH = {            # (L metres, W seconds)
    "opening": (0.0, 0.0), "crossing": (0.0, 0.0), "pause": (0.0, 0.0),
    "neutral": (3.0, 2.0), "conveyor": (3.0, 2.0), "harvest": (1.0, 1.0), "shoot": (0.6, 0.5),
}


@dataclass
class Crossing:
    t_start: float      # clock when the ghost enters the band (clock counts down)
    t_end: float
    z: float
    kind: str           # "trench" | "bump"
    inbound: bool       # neutral → blue zone
    held: int


def crossings(g: Ghost) -> list[Crossing]:
    x, t = g.x, g.t
    inband = (x > BAND[0]) & (x < BAND[1])
    out, start = [], None
    for i, b in enumerate(inband):
        if b and start is None:
            start = i
        elif not b and start is not None:
            z = float(np.mean(g.z[start:i]))
            before = x[start - 1] if start > 0 else x[start]
            out.append(Crossing(float(t[start]), float(t[i - 1]), z, "trench" if abs(z) > 2.6 else "bump",
                                bool(x[i] > before), int(g.held[start])))
            start = None
    return out


class PhaseMap:
    def __init__(self, g: Ghost, pre_s: float = 1.5, post_s: float = 0.5):
        self.g = g
        self.cross = crossings(g)
        self.masks = [(c.t_start + pre_s, c.t_end - post_s) for c in self.cross]   # (hi, lo) in clock terms
        self.won = True

    def phase(self, tau: float) -> str:
        if tau > 140.0:
            return "opening"
        for hi, lo in self.masks:
            if lo <= tau <= hi:
                return "crossing"
        f = self.g.at(tau)
        if f.x <= BAND[0]:
            return "neutral" if blue_active(tau, self.won) else "conveyor"
        if f.x >= BAND[1]:
            return "shoot" if f.buttons[1] else "harvest"
        return "crossing"

    def next_crossing(self, tau: float) -> Crossing | None:
        return next((c for c in self.cross if c.t_start <= tau), None)

    def seconds_to_next_crossing(self, tau: float) -> float:
        c = self.next_crossing(tau)
        return tau - c.t_start if c else 99.0


def leash(phase: str, scale: dict | None = None) -> tuple[float, float]:
    """Current (L, W) for a phase; `scale` maps phase → fraction of FINAL_LEASH unlocked by the curriculum."""
    L, W = FINAL_LEASH[phase]
    f = (scale or {}).get(phase, 0.0)        # phases must be opted in explicitly (default: no deviation)
    return L * f, W * f
