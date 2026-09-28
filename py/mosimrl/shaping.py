"""Potential-based reward shaping for the residual policy (docs/design/verdict-2026-09-26.md, "reward").

r_k = Δblue_k + γ·Φ(s_{k+1}) − Φ(s_k), Φ(terminal) = 0 — policy-invariant (Ng et al. 1999), it only speeds learning.

Φ(s) = λ_h(t)·min(held, 110) + λ_s(t)·min(STOCK, cap(t)) + 0.8·AIR_hub(s)
  held     balls carried (worth ~0.5: they still have to be driven home and fired)
  STOCK    loose floor balls inside the blue alliance zone (x > 4.25) — cheap to harvest later (~0.35)
  AIR_hub  balls in flight toward the hub while it counts (~0.8)
The λ ramps take the value away near the buzzer, when a held/stocked ball can no longer become a point; cap(t)
stops the stock term from paying for more balls than the next active window can fire (≈12 balls/s).
"""

from __future__ import annotations

import numpy as np

from .shifts import blue_active

GAMMA = 0.995
ZONE_X = 4.25
ACTIVE_WINDOWS = [(160.0, 130.0), (105.0, 80.0), (55.0, 0.0)]   # blue won auto (always true in solo)


def active_seconds_ahead(t: float) -> float:
    """Active seconds left in the current blue window, or the length of the next one."""
    for hi, lo in ACTIVE_WINDOWS:
        if lo < t <= hi:
            return t - lo
    for hi, lo in ACTIVE_WINDOWS:
        if hi < t:
            return hi - lo
    return 0.0


def fuel_counts(fuel) -> tuple[int, int]:
    """(STOCK, AIR_hub-candidates) from a Bridge fuel list [[x, z, y], ...]."""
    a = np.asarray(fuel if fuel is not None else [], dtype=np.float32).reshape(-1, 3)
    if not len(a):
        return 0, 0
    x, z, y = a[:, 0], a[:, 1], a[:, 2]
    stock = int(((x > ZONE_X) & (x < 8.27) & (y < 0.25)).sum())
    air = int(((y > 1.2) & (x > 3.4) & (x < 7.5) & (np.abs(z) < 3.5)).sum())
    return stock, air


def potential(s: dict) -> float:
    t = float(s["t"])
    held = float(s.get("held", 0))
    stock, air = fuel_counts(s.get("fuel"))
    lam_h = 0.5 * float(np.clip((t - 1.0) / 4.0, 0.0, 1.0))
    lam_s = 0.35 * float(np.clip((t - 6.0) / 8.0, 0.0, 1.0))
    cap = max(0.0, 12.0 * active_seconds_ahead(t) - held)
    counts = blue_active(t, True) or blue_active(t - 2.0, True)
    return lam_h * min(held, 110.0) + lam_s * min(float(stock), cap) + (0.8 * air if counts else 0.0)


def shaped_reward(s0: dict, s1: dict | None, final_blue: float | None = None, gamma: float = GAMMA) -> float:
    """Reward for the decision taken at s0 and ending at s1 (None = terminal, then final_blue is the last score)."""
    if s1 is None:
        return float(final_blue - s0.get("blue", 0)) - potential(s0)
    return float(s1.get("blue", 0) - s0.get("blue", 0)) + gamma * potential(s1) - potential(s0)
