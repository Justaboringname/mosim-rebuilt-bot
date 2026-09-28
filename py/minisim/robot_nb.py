"""Numba kernels for sim2's per-step robot work: footprint-vs-structure collision and intake roller contact."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _in_rect(cx, cz, x, z, f0, f1, r0, r1, hx, lz0, lz1):
    dx, dz = cx - x, cz - z
    lx = dx * r0 + dz * r1
    lzz = dx * f0 + dz * f1
    return -hx <= lx <= hx and lz0 <= lzz <= lz1


@njit(cache=True)
def robot_collide(x, z, yaw, vx, vz, boxes, rects, mu_div, climb_pen):
    """Same algorithm as MiniSim2._robot_collide (2 passes, SAT against axis-aligned footprints, linear response with
    Coulomb friction). boxes: (M, 5) x0 x1 z0 z1 mu. rects: (R, 3) hx lz0 lz1. Returns x, z, vx, vz, climbed."""
    a = math.radians(yaw)
    f0, f1 = math.sin(a), math.cos(a)
    r0, r1 = math.cos(a), -math.sin(a)
    climbed = False
    cx4 = np.empty(4); cz4 = np.empty(4); bx4 = np.empty(4); bz4 = np.empty(4)
    axes = np.empty((4, 2))
    for _ in range(2):
        for m in range(boxes.shape[0]):
            x0, x1, z0, z1, mu = boxes[m, 0], boxes[m, 1], boxes[m, 2], boxes[m, 3], boxes[m, 4]
            if x1 < x - 0.8 or x0 > x + 0.8 or z1 < z - 0.8 or z0 > z + 0.8:
                continue
            for q in range(rects.shape[0]):
                hx, lz0, lz1 = rects[q, 0], rects[q, 1], rects[q, 2]
                sxs = (-hx, hx, hx, -hx); lzs = (lz0, lz0, lz1, lz1)
                for c in range(4):
                    cx4[c] = x + sxs[c] * r0 + lzs[c] * f0
                    cz4[c] = z + sxs[c] * r1 + lzs[c] * f1
                bx4[0] = x0; bz4[0] = z0; bx4[1] = x1; bz4[1] = z0; bx4[2] = x1; bz4[2] = z1; bx4[3] = x0; bz4[3] = z1
                axes[0, 0] = 1.0; axes[0, 1] = 0.0; axes[1, 0] = 0.0; axes[1, 1] = 1.0
                axes[2, 0] = r0; axes[2, 1] = r1; axes[3, 0] = f0; axes[3, 1] = f1
                best = -1.0; bn0 = 0.0; bn1 = 0.0; sep = False
                for k in range(4):
                    a0, a1 = axes[k, 0], axes[k, 1]
                    pmin = 1e9; pmax = -1e9; qmin = 1e9; qmax = -1e9
                    for c in range(4):
                        pr = cx4[c] * a0 + cz4[c] * a1
                        pb = bx4[c] * a0 + bz4[c] * a1
                        pmin = min(pmin, pr); pmax = max(pmax, pr); qmin = min(qmin, pb); qmax = max(qmax, pb)
                    ov = min(pmax, qmax) - max(pmin, qmin)
                    if ov <= 0:
                        sep = True
                        break
                    crr = x * a0 + z * a1; cb = (x0 + x1) / 2 * a0 + (z0 + z1) / 2 * a1
                    sg = 1.0 if crr >= cb else -1.0
                    if best < 0 or ov < best:
                        best = ov; bn0 = a0 * sg; bn1 = a1 * sg
                if sep:
                    continue
                if mu >= mu_div and best > climb_pen:
                    climbed = True
                x += bn0 * best; z += bn1 * best
                vn = vx * bn0 + vz * bn1
                if vn >= 0:
                    continue
                t0, t1 = -bn1, bn0
                vt = vx * t0 + vz * t1
                dvt = max(-mu * -vn, min(mu * -vn, -vt))
                vx += -vn * bn0 + dvt * t0
                vz += -vn * bn1 + dvt * t1
    return x, z, vx, vz, climbed


@njit(cache=True)
def roller_touch(p, live, x, z, yaw, roller_half, roller_z, roller_y, reach):
    """Indices of live balls touching the intake roller (a cylinder along robot x at robot-frame z = roller_z)."""
    a = math.radians(yaw)
    sa, ca = math.sin(a), math.cos(a)
    out = np.empty(p.shape[0], np.int64); k = 0
    r2 = reach * reach
    for i in range(p.shape[0]):
        if not live[i]:
            continue
        dx, dz = p[i, 0] - x, p[i, 2] - z
        if dx * dx + dz * dz > 1.0:
            continue
        lz = dx * sa + dz * ca
        lx = dx * ca - dz * sa
        dy = p[i, 1] - roller_y
        if abs(lx) <= roller_half and (lz - roller_z) ** 2 + dy * dy <= r2:
            out[k] = i; k += 1
    return out[:k]
