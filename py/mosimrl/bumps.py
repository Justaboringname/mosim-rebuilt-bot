"""The ghost route with smooth lateral bumps on its collection trips.

A trip (a, b) of the match clock gets offset(t) = A * sin(pi * (a - t) / (a - b)) * n(t) added to the ghost's
position, n being the (smoothed) left normal of the ghost's path. The bump is zero at both ends, so crossings,
the zone work and all button timing stay the human's; only the line through the neutral zone moves. The ghost's
velocity, stick feed-forward and heading are bent to match the new line (intake-first stays intake-first).
headroom.py estimates, from the balls the bot actually saw, which amplitude sweeps the most per trip.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .ghost import Ghost
from .headroom import clamp_to_field, ghost_path

VMAX = 2.7


def _items(bumps):
    out = []
    for k, v in bumps.items():
        a, b = (float(x) for x in k.split(","))
        f0, f1, A = (0.0, 1.0, float(v)) if not isinstance(v, (list, tuple)) else (float(v[0]), float(v[1]), float(v[2]))
        out.append((a, b, f0, f1, A))
    return out


def bumped_ghost(g: Ghost, bumps: dict) -> Ghost:
    """bumps: {"a,b": A} (bump over the whole trip) or {"a,b": [f0, f1, A]} (bump over the fraction f0..f1 of the
    trip a → b, match clock, a > b); A metres along the path's left normal."""
    items = _items(bumps)
    m = copy.copy(g)
    m.x, m.z = g.x.copy(), g.z.copy()
    m.vx, m.vz = g.vx.copy(), g.vz.copy()
    m.cmd = g.cmd.copy()
    m.yaw_unwrapped = g.yaw_unwrapped.copy()
    m.name = g.name + "-bumped"
    for a, b, f0, f1, A in items:
        if A == 0.0:
            continue
        ts, p, n = ghost_path(g, a, b)
        idx = np.where((g.t <= a) & (g.t >= b))[0]
        if len(idx) < 3:
            continue
        t = g.t[idx]
        nx = np.interp(-t, -ts, n[:, 0]); nz = np.interp(-t, -ts, n[:, 1])
        u = (a - t) / (a - b)
        w = np.clip((u - f0) / (f1 - f0), 0.0, 1.0)
        s = np.where((u >= f0) & (u <= f1), np.sin(np.pi * w), 0.0)
        p0 = np.stack([g.x[idx], g.z[idx]], 1)
        q = clamp_to_field(p0 + A * s[:, None] * np.stack([nx, nz], 1), p0)
        off = q - p0
        dt = -np.gradient(t)                                  # clock counts down: elapsed time per record
        dt = np.where(dt > 1e-3, dt, 0.1)
        ov = np.gradient(off, axis=0) / dt[:, None]
        m.x[idx], m.z[idx] = q[:, 0], q[:, 1]
        m.vx[idx] = g.vx[idx] + ov[:, 0]
        m.vz[idx] = g.vz[idx] + ov[:, 1]
        c = g.cmd[idx].copy()
        c[:, 0] += ov[:, 0] / VMAX
        c[:, 1] += ov[:, 1] / VMAX
        nn = np.maximum(1.0, np.hypot(c[:, 0], c[:, 1]))
        c[:, 0] /= nn; c[:, 1] /= nn
        m.cmd[idx] = c
        # turn the heading with the path so the intake still leads (only where the robot is really moving)
        sp0 = np.hypot(g.vx[idx], g.vz[idx])
        a0 = np.arctan2(g.vx[idx], g.vz[idx]); a1 = np.arctan2(m.vx[idx], m.vz[idx])
        d = np.degrees(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))
        d = np.where(sp0 > 0.6, np.clip(d, -45, 45), 0.0)
        m.yaw_unwrapped[idx] = g.yaw_unwrapped[idx] + d
    return m
