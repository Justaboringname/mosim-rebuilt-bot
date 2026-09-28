"""Rigid-body ball physics with MoSim's PhysX parameters (docs/policy-facts/fuel_flow.md section 6):

    fuel: sphere r 0.075 (vs field and robot), r 0.070 (vs other fuel), mass 1 kg, I = 0.4 m r^2, drag 0,
          angular drag 0.95, material Fuel (dynamic mu 0.15, static 0.17, bounciness 0.5; friction Multiply,
          bounce Average)
    field / carpet / bumps / hub base: default material (mu 0.6, e 0, Average)
    ball-carpet mu 0.09, e 0.25; ball-ball mu 0.0225, e 0.5
    gravity 9.81, bounce threshold 2 m/s, sleep threshold 0.05 (mass-normalised energy) for 0.4 s,
    fixed dt 0.0045 s

Static geometry = every enabled, non-trigger BoxCollider of the level2 scene a floor ball can reach
(field_obbs.json, extracted from docs/policy-facts/field.json), as oriented boxes; the carpet is the plane y = 0.

The solver is a small impulse solver (contact normal with restitution above the bounce threshold, Coulomb friction
with spin, positional de-penetration), vectorised over contacts. Sleeping balls cost nothing until something
touches them. The robot is a kinematic set of oriented boxes whose contact impulses are returned to the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

G = 9.81
R_FIELD, R_BALL = 0.075, 0.070
MASS = 1.0
INERTIA = 0.4 * MASS * R_FIELD ** 2
ANG_DRAG = 0.95
MU_CARPET, E_CARPET = 0.15 * 0.6, 0.25
MU_BALL, E_BALL = 0.15 * 0.15, 0.5
MU_ROBOT, E_ROBOT = 0.15 * 0.6, 0.25
BOUNCE_V = 2.0
SLEEP_E, SLEEP_T = 0.05, 0.4
WAKE_V = 0.05

OBB_FILE = Path(__file__).with_name("field_obbs.json")


class StaticWorld:
    def __init__(self, path=OBB_FILE):
        rows = json.load(open(path))
        self.rows = rows
        self.c = np.array([r["center"] for r in rows], float)
        self.h = np.array([r["half"] for r in rows], float)
        self.R = np.array([r["axes"] for r in rows], float)                  # (K, 3 local axes, 3 world)
        self.lo = np.array([r["aabb_min"] for r in rows], float) - R_FIELD - 0.01
        self.hi = np.array([r["aabb_max"] for r in rows], float) + R_FIELD + 0.01
        self.kind = np.array([r["kind"] for r in rows])
        self.enabled = np.ones(len(rows), bool)
        self.blue_door = np.array(["BlueOutpost/ChuteDoor" in r["path"] for r in rows])

    def contacts(self, P):
        """Sphere (r = R_FIELD) vs oriented boxes. Returns ball idx, box idx, world normal (box -> ball), depth."""
        inside = np.all((P[:, None, :] >= self.lo[None]) & (P[:, None, :] <= self.hi[None]), axis=2) & self.enabled[None]
        bi, ki = np.nonzero(inside)
        if not len(bi):
            return bi, ki, np.zeros((0, 3)), np.zeros(0)
        d = P[bi] - self.c[ki]
        R = self.R[ki]
        loc = np.einsum("nij,nj->ni", R, d)
        h = self.h[ki]
        cl = np.clip(loc, -h, h)
        diff = loc - cl
        dist = np.linalg.norm(diff, axis=1)
        out = dist > 1e-9
        n_loc = np.zeros_like(loc)
        n_loc[out] = diff[out] / dist[out, None]
        depth = np.where(out, R_FIELD - dist, 0.0)
        if (~out).any():                                   # centre inside the box: push out through the nearest face
            m = ~out
            gap = h[m] - np.abs(loc[m])
            k = gap.argmin(1)
            n_in = np.zeros((m.sum(), 3))
            n_in[np.arange(m.sum()), k] = np.sign(loc[m][np.arange(m.sum()), k] + 1e-12)
            n_loc[m] = n_in
            depth[m] = R_FIELD + gap[np.arange(m.sum()), k]
        keep = depth > 0
        n_w = np.einsum("nij,ni->nj", R[keep], n_loc[keep])
        return bi[keep], ki[keep], n_w, depth[keep]


class Balls:
    """All fuel of one match. `live` marks balls simulated on the field (not in a hopper / hub / scheduled flight)."""

    def __init__(self, n: int, world: StaticWorld | None = None):
        self.world = world or StaticWorld()
        self.p = np.zeros((n, 3)); self.v = np.zeros((n, 3)); self.w = np.zeros((n, 3))
        self.live = np.zeros(n, bool)
        self.awake = np.zeros(n, bool)
        self.low_t = np.zeros(n)

    def wake(self, idx):
        self.awake[idx] = True
        self.low_t[idx] = 0.0

    # ------------------------------------------------------------------
    def step(self, dt: float, robot=None):
        """Advance one physics step. robot: None or dict(boxes=[(center(3), half(3), R(3,3))], vel_at(points)->(n,3)).
        Returns the total impulse the balls applied to the robot (3,)."""
        live = np.nonzero(self.live)[0]
        if not len(live):
            return np.zeros(3)
        aw = live[self.awake[live]]
        # integrate forces on awake balls
        if len(aw):
            self.v[aw, 1] -= G * dt
            self.w[aw] *= 1.0 / (1.0 + ANG_DRAG * dt)
        # candidate set: awake balls, sleeping balls near awake balls or near the robot
        P = self.p
        cand = aw
        near_robot = np.zeros(0, int)
        if robot is not None:
            rc = robot["center"]
            dd = (P[live, 0] - rc[0]) ** 2 + (P[live, 2] - rc[2]) ** 2
            near_robot = live[dd < robot["reach"] ** 2]
            cand = np.union1d(cand, near_robot)
        if not len(cand):
            return np.zeros(3)
        J_robot = np.zeros(3)
        # ------------------------------------------------ contacts
        rows = []                                            # (i, j(-1 static,-2 robot), n, depth, mu, e, vel_other)
        # carpet
        pc = P[cand]
        m = pc[:, 1] < R_FIELD
        if m.any():
            i = cand[m]
            rows.append((i, np.full(len(i), -1), np.tile([0.0, 1.0, 0.0], (len(i), 1)), R_FIELD - pc[m, 1],
                         MU_CARPET, E_CARPET, np.zeros((len(i), 3))))
        # static boxes
        bi, ki, n_w, depth = self.world.contacts(pc)
        if len(bi):
            ramp = self.world.kind[ki] == "ramp"
            mu = np.where(ramp, 0.0, MU_CARPET); e = np.where(ramp, 0.0, E_CARPET)
            rows.append((cand[bi], np.full(len(bi), -1), n_w, depth, mu, e, np.zeros((len(bi), 3))))
        # ball-ball among live balls near the candidates
        pairs = self._pairs(live, cand)
        # robot boxes
        if robot is not None and len(near_robot):
            rb_i, rb_n, rb_d = robot_contacts(P[near_robot], robot["boxes"])
            if len(rb_i):
                i = near_robot[rb_i]
                vo = robot["vel_at"](P[i])
                rows.append((i, np.full(len(i), -2), rb_n, rb_d, MU_ROBOT, E_ROBOT, vo))
                self.wake(i)
        # ------------------------------------------------ solve (2 Jacobi passes, relaxed by contact count)
        for it in range(2):
            dv = np.zeros_like(self.v); dw = np.zeros_like(self.w); cnt = np.zeros(len(self.v))
            for (i, j, n, depth, mu, e, vo) in rows:
                arm = -n * R_FIELD
                vc = self.v[i] + np.cross(self.w[i], arm) - vo
                vn = (vc * n).sum(1)
                tgt = np.where(vn < -BOUNCE_V, -np.broadcast_to(e, vn.shape) * vn, 0.0) if it == 0 else np.zeros_like(vn)
                jn = np.maximum(0.0, (tgt - vn)) * MASS
                vt = vc - vn[:, None] * n
                vts = np.linalg.norm(vt, axis=1)
                kt = 1.0 / MASS + R_FIELD ** 2 / INERTIA
                jt_mag = np.minimum(vts / kt, np.broadcast_to(mu, vts.shape) * jn)
                t = np.where(vts[:, None] > 1e-9, vt / np.maximum(vts, 1e-9)[:, None], 0.0)
                J = jn[:, None] * n - jt_mag[:, None] * t
                np.add.at(dv, i, J / MASS)
                np.add.at(dw, i, np.cross(arm, J) / INERTIA)
                np.add.at(cnt, i, 1.0)
                if (j == -2).any():
                    J_robot -= J[j == -2].sum(0) if it == 0 else 0.0
            if pairs is not None:
                a, b, n, depth = pairs
                arm_a, arm_b = n * R_BALL, -n * R_BALL           # n points a -> b
                va = self.v[a] + np.cross(self.w[a], arm_a); vb = self.v[b] + np.cross(self.w[b], arm_b)
                vrel = va - vb
                vn = (vrel * n).sum(1)                           # > 0: approaching
                tgt = np.where(vn > BOUNCE_V, E_BALL * vn, 0.0) if it == 0 else np.zeros_like(vn)
                jn = np.maximum(0.0, vn + tgt) * MASS / 2.0
                vt = vrel - vn[:, None] * n
                vts = np.linalg.norm(vt, axis=1)
                kt = 2.0 / MASS + 2.0 * R_BALL ** 2 / INERTIA
                jt = np.minimum(vts / kt, MU_BALL * jn)
                t = np.where(vts[:, None] > 1e-9, vt / np.maximum(vts, 1e-9)[:, None], 0.0)
                J = -jn[:, None] * n - jt[:, None] * t             # impulse on a
                np.add.at(dv, a, J / MASS); np.add.at(dv, b, -J / MASS)
                np.add.at(dw, a, np.cross(arm_a, J) / INERTIA); np.add.at(dw, b, np.cross(arm_b, -J) / INERTIA)
                np.add.at(cnt, a, 1.0); np.add.at(cnt, b, 1.0)
                moving = np.linalg.norm(vrel, axis=1) > WAKE_V
                self.wake(np.concatenate([a[moving], b[moving]]))
            touched = np.nonzero(cnt)[0]
            s = 1.0 / np.maximum(cnt[touched], 1.0) ** 0.5
            self.v[touched] += dv[touched] * s[:, None]
            self.w[touched] += dw[touched] * s[:, None]
        # ------------------------------------------------ positional correction
        for (i, j, n, depth, mu, e, vo) in rows:
            np.add.at(self.p, i, n * np.maximum(depth - 0.001, 0.0)[:, None] * 0.8)
        if pairs is not None:
            a, b, n, depth = pairs
            c = n * (np.maximum(depth - 0.001, 0.0) * 0.4)[:, None]
            np.add.at(self.p, a, -c); np.add.at(self.p, b, c)
        # ------------------------------------------------ integrate and sleep
        aw = live[self.awake[live]]
        self.p[aw] += self.v[aw] * dt
        E = 0.5 * (self.v[aw] ** 2).sum(1) + 0.5 * (INERTIA / MASS) * (self.w[aw] ** 2).sum(1)
        low = E < SLEEP_E
        self.low_t[aw] = np.where(low, self.low_t[aw] + dt, 0.0)
        sl = aw[self.low_t[aw] >= SLEEP_T]
        if len(sl):
            self.awake[sl] = False; self.v[sl] = 0.0; self.w[sl] = 0.0
        return J_robot

    def _pairs(self, live, cand):
        if len(cand) == 0:
            return None
        P = self.p
        tree = cKDTree(P[live])
        lists = tree.query_ball_point(P[cand], 2 * R_BALL)
        a, b = [], []
        for ci, lst in zip(cand, lists):
            for k in lst:
                j = live[k]
                if j != ci:
                    a.append(ci); b.append(j)
        if not a:
            return None
        a = np.array(a); b = np.array(b)
        key = np.minimum(a, b) * 100000 + np.maximum(a, b)
        _, u = np.unique(key, return_index=True)
        a, b = a[u], b[u]
        d = P[b] - P[a]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n = d / dist[:, None]
        depth = 2 * R_BALL - dist
        m = depth > 0
        if not m.any():
            return None
        # a sleeping ball hit by an awake one wakes up
        self.wake(np.concatenate([a[m], b[m]])[np.concatenate([self.awake[b[m]], self.awake[a[m]]])])
        return a[m], b[m], n[m], depth[m]


def robot_contacts(P, boxes):
    """Sphere vs the robot's oriented boxes. boxes: list of (center(3), half(3), R(3,3) rows = local axes in world).
    Returns ball index, world normal (robot -> ball), depth."""
    idx, nrm, dep = [], [], []
    for c, h, R in boxes:
        d = P - c
        loc = d @ R.T
        cl = np.clip(loc, -h, h)
        diff = loc - cl
        dist = np.linalg.norm(diff, axis=1)
        hit = dist < R_FIELD
        if not hit.any():
            continue
        ii = np.nonzero(hit)[0]
        out = dist[ii] > 1e-9
        n_loc = np.zeros((len(ii), 3)); dp = np.zeros(len(ii))
        n_loc[out] = diff[ii][out] / dist[ii][out, None]
        dp[out] = R_FIELD - dist[ii][out]
        if (~out).any():
            m = ~out
            lm = loc[ii][m]
            gap = h - np.abs(lm)
            gap[:, 1] = np.inf                              # never push a floor ball out through the top/bottom
            k = gap.argmin(1)
            nn = np.zeros((m.sum(), 3)); nn[np.arange(m.sum()), k] = np.sign(lm[np.arange(m.sum()), k] + 1e-12)
            n_loc[m] = nn
            dp[m] = R_FIELD + gap[np.arange(m.sum()), k]
        idx.append(ii); nrm.append(n_loc @ R); dep.append(dp)
    if not idx:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(idx), np.concatenate(nrm), np.concatenate(dep)
