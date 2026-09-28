"""Numba-compiled version of phys.Balls.step (same model, same constants, same solver order).

Drop-in for phys.Balls in sim2: NBBalls(n, world) with p / v / w / live / awake / low_t, wake(idx) and
step(dt, robot) -> impulse on the robot. The robot dict must carry "center", "reach", "boxes" and "vel" = (vx, vz, w).
Ball-ball neighbours come from a uniform grid (cell 0.15 m) rebuilt every step.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .phys import (ANG_DRAG, BOUNCE_V, E_BALL, E_CARPET, E_ROBOT, G, INERTIA, MASS, MU_BALL, MU_CARPET, MU_ROBOT,
                   R_BALL, R_FIELD, SLEEP_E, SLEEP_T, WAKE_V, StaticWorld)

GX0, GZ0, CELL, NX, NZ = -10.0, -5.0, 0.15, 134, 67
MAXC = 16384


@njit(cache=True)
def _cross(a0, a1, a2, b0, b1, b2):
    return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0


@njit(cache=True)
def _sphere_obb(px, py, pz, c, h, R, rad):
    """Returns (hit, nx, ny, nz, depth) for a sphere vs one oriented box; the y-face is never used for a centre inside
    when floor_mode (robot boxes) — handled by caller via h[1] large."""
    dx, dy, dz = px - c[0], py - c[1], pz - c[2]
    l0 = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
    l1 = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
    l2 = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz
    c0 = min(max(l0, -h[0]), h[0]); c1 = min(max(l1, -h[1]), h[1]); c2 = min(max(l2, -h[2]), h[2])
    d0, d1, d2 = l0 - c0, l1 - c1, l2 - c2
    dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if dist >= rad:
        return False, 0.0, 0.0, 0.0, 0.0
    if dist > 1e-9:
        n0, n1, n2 = d0 / dist, d1 / dist, d2 / dist
        depth = rad - dist
    else:
        g0, g1, g2 = h[0] - abs(l0), h[1] - abs(l1), h[2] - abs(l2)
        n0 = n1 = n2 = 0.0
        if g0 <= g1 and g0 <= g2:
            n0 = 1.0 if l0 >= 0 else -1.0; depth = rad + g0
        elif g1 <= g2:
            n1 = 1.0 if l1 >= 0 else -1.0; depth = rad + g1
        else:
            n2 = 1.0 if l2 >= 0 else -1.0; depth = rad + g2
    wx = n0 * R[0, 0] + n1 * R[1, 0] + n2 * R[2, 0]
    wy = n0 * R[0, 1] + n1 * R[1, 1] + n2 * R[2, 1]
    wz = n0 * R[0, 2] + n1 * R[1, 2] + n2 * R[2, 2]
    return True, wx, wy, wz, depth


@njit(cache=True)
def _robot_obb(px, py, pz, c, h, R, rad):
    """Like _sphere_obb but a centre inside the box never leaves through the top/bottom (robot vs floor ball)."""
    dx, dy, dz = px - c[0], py - c[1], pz - c[2]
    l0 = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
    l1 = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
    l2 = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz
    c0 = min(max(l0, -h[0]), h[0]); c1 = min(max(l1, -h[1]), h[1]); c2 = min(max(l2, -h[2]), h[2])
    d0, d1, d2 = l0 - c0, l1 - c1, l2 - c2
    dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    if dist >= rad:
        return False, 0.0, 0.0, 0.0, 0.0
    if dist > 1e-9:
        n0, n1, n2 = d0 / dist, d1 / dist, d2 / dist
        depth = rad - dist
    else:
        g0, g2 = h[0] - abs(l0), h[2] - abs(l2)
        n0 = n1 = n2 = 0.0
        if g0 <= g2:
            n0 = 1.0 if l0 >= 0 else -1.0; depth = rad + g0
        else:
            n2 = 1.0 if l2 >= 0 else -1.0; depth = rad + g2
    wx = n0 * R[0, 0] + n1 * R[1, 0] + n2 * R[2, 0]
    wy = n0 * R[0, 1] + n1 * R[1, 1] + n2 * R[2, 1]
    wz = n0 * R[0, 2] + n1 * R[1, 2] + n2 * R[2, 2]
    return True, wx, wy, wz, depth


@njit(cache=True)
def step_nb(p, v, w, live, awake, low_t, dt,
            sc, sh, sR, slo, shi, sen, sramp,
            has_robot, rc, reach, rbc, rbh, rbR, nrb, rvx, rvz, rw,
            grid_head, grid_next, ci, cj, cn, cd, cmu, ce, cvo, ck, dv, dw, cnt, cand, near):
    n = p.shape[0]
    J = np.zeros(3)
    # forces on awake live balls
    for i in range(n):
        if live[i] and awake[i]:
            v[i, 1] -= G * dt
            f = 1.0 / (1.0 + ANG_DRAG * dt)
            w[i, 0] *= f; w[i, 1] *= f; w[i, 2] *= f
    # candidates
    cand[:] = False
    near[:] = False
    ncand = 0
    for i in range(n):
        if not live[i]:
            continue
        if awake[i]:
            cand[i] = True
        if has_robot:
            ddx, ddz = p[i, 0] - rc[0], p[i, 2] - rc[2]
            if ddx * ddx + ddz * ddz < reach * reach:
                near[i] = True
                cand[i] = True
        if cand[i]:
            ncand += 1
    if ncand == 0:
        return J
    # grid of live balls
    grid_head[:] = -1
    for i in range(n):
        if live[i]:
            gx = int((p[i, 0] - GX0) / CELL); gz = int((p[i, 2] - GZ0) / CELL)
            if 0 <= gx < NX and 0 <= gz < NZ:
                k = gx * NZ + gz
                grid_next[i] = grid_head[k]; grid_head[k] = i
            else:
                grid_next[i] = -1
    # contacts: (i, j, nx, ny, nz, depth, mu, e, vox, voy, voz, kind) kind 0 static/carpet, 1 robot, 2 ball pair
    nc = 0
    K = sc.shape[0]
    for i in range(n):
        if not cand[i] or nc >= MAXC - 64:
            continue
        px, py, pz = p[i, 0], p[i, 1], p[i, 2]
        if py < R_FIELD:
            ci[nc] = i; cj[nc] = -1; cn[nc, 0] = 0.0; cn[nc, 1] = 1.0; cn[nc, 2] = 0.0
            cd[nc] = R_FIELD - py; cmu[nc] = MU_CARPET; ce[nc] = E_CARPET; cvo[nc, :] = 0.0; ck[nc] = 0; nc += 1
        for k in range(K):
            if not sen[k]:
                continue
            if px < slo[k, 0] or px > shi[k, 0] or py < slo[k, 1] or py > shi[k, 1] or pz < slo[k, 2] or pz > shi[k, 2]:
                continue
            hit, nx, ny, nz, dep = _sphere_obb(px, py, pz, sc[k], sh[k], sR[k], R_FIELD)
            if hit and dep > 0:
                ci[nc] = i; cj[nc] = -1; cn[nc, 0] = nx; cn[nc, 1] = ny; cn[nc, 2] = nz; cd[nc] = dep
                cmu[nc] = 0.0 if sramp[k] else MU_CARPET; ce[nc] = 0.0 if sramp[k] else E_CARPET
                cvo[nc, :] = 0.0; ck[nc] = 0; nc += 1
        if has_robot and near[i]:
            for b in range(nrb):
                hit, nx, ny, nz, dep = _robot_obb(px, py, pz, rbc[b], rbh[b], rbR[b], R_FIELD)
                if hit and dep > 0:
                    ci[nc] = i; cj[nc] = -2; cn[nc, 0] = nx; cn[nc, 1] = ny; cn[nc, 2] = nz; cd[nc] = dep
                    cmu[nc] = MU_ROBOT; ce[nc] = E_ROBOT
                    rx, rz = px - rc[0], pz - rc[2]
                    cvo[nc, 0] = rvx + rw * rz; cvo[nc, 1] = 0.0; cvo[nc, 2] = rvz - rw * rx
                    ck[nc] = 1; nc += 1
                    awake[i] = True; low_t[i] = 0.0
        # ball pairs (i < j when both candidates; otherwise the candidate is i)
        gx = int((px - GX0) / CELL); gz = int((pz - GZ0) / CELL)
        for ax in range(gx - 1, gx + 2):
            if ax < 0 or ax >= NX:
                continue
            for az in range(gz - 1, gz + 2):
                if az < 0 or az >= NZ:
                    continue
                j = grid_head[ax * NZ + az]
                while j >= 0:
                    if j != i and (not cand[j] or j > i):
                        ddx, ddy, ddz = p[j, 0] - px, p[j, 1] - py, p[j, 2] - pz
                        dist = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                        if dist < 2 * R_BALL and nc < MAXC:
                            dist = max(dist, 1e-9)
                            ci[nc] = i; cj[nc] = j
                            cn[nc, 0] = ddx / dist; cn[nc, 1] = ddy / dist; cn[nc, 2] = ddz / dist
                            cd[nc] = 2 * R_BALL - dist; ck[nc] = 2; nc += 1
                            if awake[i] and not awake[j]:
                                awake[j] = True; low_t[j] = 0.0
                            elif awake[j] and not awake[i]:
                                awake[i] = True; low_t[i] = 0.0
                    j = grid_next[j]
    # solve: 2 Jacobi passes relaxed by contact count
    kt_s = 1.0 / MASS + R_FIELD * R_FIELD / INERTIA
    kt_b = 2.0 / MASS + 2.0 * R_BALL * R_BALL / INERTIA
    for it in range(2):
        dv[:] = 0.0; dw[:] = 0.0; cnt[:] = 0.0
        for c in range(nc):
            i = ci[c]; nx, ny, nz = cn[c, 0], cn[c, 1], cn[c, 2]
            if ck[c] != 2:
                ax_, ay_, az_ = -nx * R_FIELD, -ny * R_FIELD, -nz * R_FIELD
                cx, cy, cz = _cross(w[i, 0], w[i, 1], w[i, 2], ax_, ay_, az_)
                vcx = v[i, 0] + cx - cvo[c, 0]; vcy = v[i, 1] + cy - cvo[c, 1]; vcz = v[i, 2] + cz - cvo[c, 2]
                vn = vcx * nx + vcy * ny + vcz * nz
                tgt = -ce[c] * vn if (it == 0 and vn < -BOUNCE_V) else 0.0
                jn = max(0.0, tgt - vn) * MASS
                tx, ty, tz = vcx - vn * nx, vcy - vn * ny, vcz - vn * nz
                vts = np.sqrt(tx * tx + ty * ty + tz * tz)
                jt = min(vts / kt_s, cmu[c] * jn)
                if vts > 1e-9:
                    tx /= vts; ty /= vts; tz /= vts
                else:
                    tx = ty = tz = 0.0
                Jx, Jy, Jz = jn * nx - jt * tx, jn * ny - jt * ty, jn * nz - jt * tz
                dv[i, 0] += Jx / MASS; dv[i, 1] += Jy / MASS; dv[i, 2] += Jz / MASS
                qx, qy, qz = _cross(ax_, ay_, az_, Jx, Jy, Jz)
                dw[i, 0] += qx / INERTIA; dw[i, 1] += qy / INERTIA; dw[i, 2] += qz / INERTIA
                cnt[i] += 1.0
                if ck[c] == 1 and it == 0:
                    J[0] -= Jx; J[1] -= Jy; J[2] -= Jz
            else:
                j = cj[c]
                aax, aay, aaz = nx * R_BALL, ny * R_BALL, nz * R_BALL
                c1x, c1y, c1z = _cross(w[i, 0], w[i, 1], w[i, 2], aax, aay, aaz)
                c2x, c2y, c2z = _cross(w[j, 0], w[j, 1], w[j, 2], -aax, -aay, -aaz)
                vrx = v[i, 0] + c1x - v[j, 0] - c2x; vry = v[i, 1] + c1y - v[j, 1] - c2y
                vrz = v[i, 2] + c1z - v[j, 2] - c2z
                vn = vrx * nx + vry * ny + vrz * nz
                tgt = E_BALL * vn if (it == 0 and vn > BOUNCE_V) else 0.0
                jn = max(0.0, vn + tgt) * MASS / 2.0
                tx, ty, tz = vrx - vn * nx, vry - vn * ny, vrz - vn * nz
                vts = np.sqrt(tx * tx + ty * ty + tz * tz)
                jt = min(vts / kt_b, MU_BALL * jn)
                if vts > 1e-9:
                    tx /= vts; ty /= vts; tz /= vts
                else:
                    tx = ty = tz = 0.0
                Jx, Jy, Jz = -jn * nx - jt * tx, -jn * ny - jt * ty, -jn * nz - jt * tz
                dv[i, 0] += Jx / MASS; dv[i, 1] += Jy / MASS; dv[i, 2] += Jz / MASS
                dv[j, 0] -= Jx / MASS; dv[j, 1] -= Jy / MASS; dv[j, 2] -= Jz / MASS
                qx, qy, qz = _cross(aax, aay, aaz, Jx, Jy, Jz)
                dw[i, 0] += qx / INERTIA; dw[i, 1] += qy / INERTIA; dw[i, 2] += qz / INERTIA
                qx, qy, qz = _cross(-aax, -aay, -aaz, -Jx, -Jy, -Jz)
                dw[j, 0] += qx / INERTIA; dw[j, 1] += qy / INERTIA; dw[j, 2] += qz / INERTIA
                cnt[i] += 1.0; cnt[j] += 1.0
                if it == 0 and np.sqrt(vrx * vrx + vry * vry + vrz * vrz) > WAKE_V:
                    awake[i] = True; awake[j] = True; low_t[i] = 0.0; low_t[j] = 0.0
        for i in range(n):
            if cnt[i] > 0:
                s = 1.0 / np.sqrt(max(cnt[i], 1.0))
                v[i, 0] += dv[i, 0] * s; v[i, 1] += dv[i, 1] * s; v[i, 2] += dv[i, 2] * s
                w[i, 0] += dw[i, 0] * s; w[i, 1] += dw[i, 1] * s; w[i, 2] += dw[i, 2] * s
    # positional correction
    for c in range(nc):
        i = ci[c]
        corr = max(cd[c] - 0.001, 0.0)
        if ck[c] != 2:
            p[i, 0] += cn[c, 0] * corr * 0.8; p[i, 1] += cn[c, 1] * corr * 0.8; p[i, 2] += cn[c, 2] * corr * 0.8
        else:
            j = cj[c]
            p[i, 0] -= cn[c, 0] * corr * 0.4; p[i, 1] -= cn[c, 1] * corr * 0.4; p[i, 2] -= cn[c, 2] * corr * 0.4
            p[j, 0] += cn[c, 0] * corr * 0.4; p[j, 1] += cn[c, 1] * corr * 0.4; p[j, 2] += cn[c, 2] * corr * 0.4
    # integrate and sleep
    for i in range(n):
        if live[i] and awake[i]:
            p[i, 0] += v[i, 0] * dt; p[i, 1] += v[i, 1] * dt; p[i, 2] += v[i, 2] * dt
            E = 0.5 * (v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2) + 0.5 * (INERTIA / MASS) * (w[i, 0] ** 2 + w[i, 1] ** 2 + w[i, 2] ** 2)
            if E < SLEEP_E:
                low_t[i] += dt
                if low_t[i] >= SLEEP_T:
                    awake[i] = False; v[i, :] = 0.0; w[i, :] = 0.0
            else:
                low_t[i] = 0.0
    return J


class NBBalls:
    def __init__(self, n: int, world: StaticWorld | None = None):
        self.world = world or StaticWorld()
        self.p = np.zeros((n, 3)); self.v = np.zeros((n, 3)); self.w = np.zeros((n, 3))
        self.live = np.zeros(n, np.bool_); self.awake = np.zeros(n, np.bool_); self.low_t = np.zeros(n)
        self._head = np.full(NX * NZ, -1, np.int64); self._next = np.full(n, -1, np.int64)
        self._ramp = (self.world.kind == "ramp")
        self._buf = (np.empty(MAXC, np.int64), np.empty(MAXC, np.int64), np.empty((MAXC, 3)), np.empty(MAXC),
                     np.empty(MAXC), np.empty(MAXC), np.zeros((MAXC, 3)), np.empty(MAXC, np.int64),
                     np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n, np.bool_), np.zeros(n, np.bool_))
        self._empty_boxes = (np.zeros((1, 3)), np.ones((1, 3)), np.tile(np.eye(3), (1, 1, 1)))

    def wake(self, idx):
        self.awake[idx] = True
        self.low_t[idx] = 0.0

    def step(self, dt: float, robot=None):
        W = self.world
        if robot is not None:
            bs = robot["boxes"]
            rbc = np.array([b[0] for b in bs]); rbh = np.array([b[1] for b in bs]); rbR = np.array([b[2] for b in bs])
            vx, vz, wr = robot["vel"]
            return step_nb(self.p, self.v, self.w, self.live, self.awake, self.low_t, dt,
                           W.c, W.h, W.R, W.lo, W.hi, W.enabled, self._ramp,
                           True, np.asarray(robot["center"], float), float(robot["reach"]), rbc, rbh, rbR, len(bs),
                           float(vx), float(vz), float(wr), self._head, self._next, *self._buf)
        c, h, R = self._empty_boxes
        return step_nb(self.p, self.v, self.w, self.live, self.awake, self.low_t, dt,
                       W.c, W.h, W.R, W.lo, W.hi, W.enabled, self._ramp,
                       False, np.zeros(3), 0.0, c, h, R, 0, 0.0, 0.0, 0.0, self._head, self._next, *self._buf)
