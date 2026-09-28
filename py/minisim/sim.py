"""MiniSim: a small, fast REBUILT simulator for team 4414 HighTide (batch-1 numpy reference implementation).

Game logic is re-implemented from an analysis of the game's behaviour (robot HighTide, drive controller, hub scoring,
hub shifts, outpost); constants are measured or read from the game. No game code is copied. the physics MoSim leaves to PhysX (ball rolling, robot pushing balls, hub exits, pass landings, capture
by the intake roller, fire rate) is fitted from real-game traces (minisim/calib/*, runs/ghost/cal1 dense data).

The interface mirrors the Bridge: reset() / act(vx, vz, rot, buttons) return the same state dict (t, gs, rs, robot,
held, mech, gate, fuel, fid, blue, blueAuto, hub, gameTime, done), so GhostPolicy and episode.run_episode run
unchanged. One decision = 0.099 s (22 MoSim physics steps), simulated in SUB substeps.
"""

from __future__ import annotations

import math

import numpy as np

from . import field as F

FLOOR, AIR, HUB, HOPPER, CORRAL, CHUTE = range(6)

DEFAULT = dict(
    # drivetrain (calib/kin.py fit_force_clean on cal1): dv/dt = K u f(|v|) - (D + bd * nearby balls) v
    K=19.62, D=5.872, ball_drag=0.0514, shoot_mult=0.55, w_max=-4.575, w_tau=0.1445,
    robot_r=0.45,
    # bumps: ramps of 16 deg; gravity along the slope pushes away from the peak
    bump_acc=2.87, bump_drive=0.75, bump_latch_speed=1.75, bump_latch_s=1.0, bump_clear_crest_p=0.5,
    trench_vmax=1.9,
    trench_bar_held=100,
    # intake (calib/intake.py): capture window in robot frame, per-decision capture 0.88 centre / 0.64 edge
    in_z0=0.40, in_z1=0.68, in_x=0.40, q_dec_center=0.88, q_dec_edge=0.64, q_dec_slow=0.06,
    deploy_s=0.3, cap_soft=95, cap_mid=105, cap_hard=112,
    body_x=0.44, body_z0=-0.43, body_z1=0.40, push_e=0.3,
    # shooter / passer (calib/shots.py): rate(held) = r_max (1 - exp(-held / r_k)); gate turret rate <= 1.2 rad/s
    r_max=17.5, r_k=15.0, turret_rate_max=1.2, som_tof=1.15, hit_p=0.975,
    hub_t_mu=2.35, hub_t_sd=0.53, exit_x=2.45, exit_z_sd=0.25, exit_v0=1.35, exit_v_scale=0.45, exit_vz_sd=0.3,
    pass_t_mu=1.29, pass_t_sd=0.15, pass_x_mu=5.5, pass_x_sd=0.95, pass_z_mu=1.7, pass_z_sd=0.4,
    pass_land_v=0.8,
    # ball rolling (calib/roll.py): exponential decay, tau 1.8 s; stop below v_stop
    # rolling: decel(v) = roll_k * max(0, v - roll_v0) + roll_c (calib/roll.py: ~0 below 0.5 m/s, 1 m/s^2 at 2 m/s)
    roll_k=0.75, roll_v0=0.55, roll_c=0.02, v_stop=0.03, wall_e=0.3, ball_d=0.14, ball_e=0.3,
    # outpost
    chute_release_rate=12.0, corral_teleport_s=0.5,
    zone_x=3.738,            # InAllianceZone: x + maxWidth(0.512) > 4.25
)


class MiniSim:
    SUB = 6
    DEC = 0.099

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.p = {**DEFAULT, **(params or {})}
        self.rng = np.random.default_rng(seed)
        self.fuel0 = F.initial_fuel()
        self.n = len(self.fuel0) + 8

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> dict:
        p, n = self.p, self.n
        self.pos = np.zeros((n, 2)); self.vel = np.zeros((n, 2)); self.y = np.zeros(n)
        self.state = np.full(n, FLOOR, int)
        m = len(self.fuel0)
        self.pos[:m] = self.fuel0[:, :2]; self.y[:m] = self.fuel0[:, 2]
        self.state[m:] = HOPPER                                          # 8 preloads
        chute = (self.pos[:, 0] > F.X_WALL) & (self.pos[:, 1] > 0) & (np.arange(n) < m)
        self.state[chute] = CHUTE
        # flight / hub bookkeeping
        self.t_land = np.zeros(n); self.p_land = np.zeros((n, 2)); self.v_land = np.zeros((n, 2))
        self.p_from = np.zeros((n, 2)); self.t_from = np.zeros(n)
        self.t_score = np.zeros(n); self.t_exit = np.zeros(n); self.scored = np.zeros(n, bool)
        self.exit_z = np.zeros(n); self.exit_v = np.zeros((n, 2))
        # robot (user-saved spawn, as in the demos)
        self.x, self.z, self.yaw = 3.786, -3.628, 359.6
        self.vx = self.vz = self.w = 0.0
        self.deployed, self.deploy_g = False, None
        self.latch = 0.0; self.latch_hold = 0.0; self.latch_crest = False
        self.shot_acc = 0.0
        self.prev_bearing = None
        self.cmd = (0.0, 0.0, 0.0, [False] * 5)
        # clock
        self.g = 0.0                  # elapsed game time
        self.t = 160.0
        self.gs, self.rs = 0, 0
        self.pause_left = 3.0
        self.end_g = None
        self.blue_auto = self.blue_tele = 0
        self.door_open_until = 2.3    # OpenChuteOnStart: one 2 s door sequence at the start
        self.corral_timer = 0.0
        self.release_acc = 0.0
        self.last_err = 0.0
        return self.state_dict()

    @property
    def held(self) -> int:
        return int((self.state == HOPPER).sum())

    # ------------------------------------------------------------------ one decision
    def act(self, vx: float, vz: float, rot: float, buttons, rff: bool = False) -> dict:
        b = [bool(x) for x in (list(buttons) + [False] * 5)[:5]]
        self.cmd = (float(np.clip(vx, -1, 1)), float(np.clip(vz, -1, 1)), float(np.clip(rot, -1, 1)), b)
        dt = self.DEC / self.SUB
        nb = self._near_balls(0.9)
        for _ in range(self.SUB):
            self._clock(dt)
            self._robot(dt, nb)
            self._outpost(dt)
            self._balls(dt)
            self._shoot(dt)
            self.g += dt
        return self.state_dict()

    # ------------------------------------------------------------------ clock / shifts
    def _clock(self, dt):
        if self.end_g is not None:
            return
        if self.gs == 0:
            self.t -= dt
            if self.t <= 140.0:
                self.t, self.gs, self.rs = 140.0, 1, 1
        elif self.rs == 1 and self.pause_left > 0:
            self.pause_left -= dt
            if self.pause_left <= 0:
                self.rs = 0
        else:
            self.t -= dt
            if self.t <= 30.0 and self.gs == 1:
                self.gs = 2
            if self.t <= 0.0:
                self.t, self.gs, self.rs, self.end_g = 0.0, 3, 1, self.g

    def scoring_open(self) -> bool:
        """HubScoring: pools the scorer while the hub is active, for 3 s after it goes inactive, always in auto, and
        for 3 s after the match ends."""
        if self.end_g is not None:
            return self.g - self.end_g < 3.0
        if F.blue_hub_active(self.t, self.gs):
            return True
        edge = F.last_deactivation(self.t)
        return edge is not None and edge - self.t < 3.0

    # ------------------------------------------------------------------ robot
    def _robot(self, dt, nb):
        p = self.p
        ux, uz, rot, b = self.cmd
        enabled = self.rs == 0
        wants = b[1] or b[2] or b[3]
        if not enabled:
            ux = uz = rot = 0.0
        mult = p["shoot_mult"] if (wants and enabled) else 1.0
        ux, uz = ux * mult, uz * mult
        nrm = math.hypot(ux, uz)
        if nrm > 1.0:
            ux, uz = ux / nrm, uz / nrm
        s = math.hypot(self.vx, self.vz)
        f = (1.0 - min(s / 5.18, 1.0) * 0.075) ** 10
        on_bump = self._on_bump(self.x, self.z)
        drive = p["bump_drive"] if on_bump else 1.0          # wheels partly off the carpet on the ramps
        Deff = p["D"] + p["ball_drag"] * nb
        tx, tz = drive * p["K"] * ux * f / Deff, drive * p["K"] * uz * f / Deff
        e = math.exp(-Deff * dt)
        self.vx = tx + (self.vx - tx) * e
        self.vz = tz + (self.vz - tz) * e
        # bump ramps: gravity along the slope pushes away from the peak. Hightide.UpdateTiltState: the crossing latch
        # sets at tilt >= 12 deg with speed > 1.75 m/s; it clears at the crest if tilt <= 2 deg happens to be sampled
        # (about half the crossings in cal1), otherwise 1 s after the set condition ends.
        if on_bump:
            peak = F.BUMP_PEAK_X if self.x > 0 else -F.BUMP_PEAK_X
            if abs(self.x - peak) < F.BUMP_HALF:
                self.vx += p["bump_acc"] * math.copysign(1.0, self.x - peak) * dt
            if math.hypot(self.vx, self.vz) > p["bump_latch_speed"] and abs(self.x - peak) > 0.15:
                if self.latch <= 0 or self.latch_hold > 0:
                    self.latch, self.latch_hold = p["bump_latch_s"], 0.0
                    self.latch_crest = self.rng.random() < p["bump_clear_crest_p"]
            if self.latch > 0 and self.latch_crest and abs(self.x - peak) < 0.08:
                self.latch = 0.0
        if self.latch > 0 and not (on_bump and math.hypot(self.vx, self.vz) > p["bump_latch_speed"]):
            self.latch = max(0.0, self.latch - dt)
        # trench lanes: under the overhead bar the robot does not reach full speed (cal1 crossings ~1.9 m/s)
        if 3.07 < abs(self.x) < 4.21 and 2.76 < abs(self.z) < 4.03:
            sp = math.hypot(self.vx, self.vz)
            if sp > p["trench_vmax"]:
                self.vx *= p["trench_vmax"] / sp; self.vz *= p["trench_vmax"] / sp
        wt = p["w_max"] * rot
        self.w = wt + (self.w - wt) * math.exp(-dt / p["w_tau"])
        self.yaw = (self.yaw + math.degrees(self.w) * dt) % 360.0
        self.x += self.vx * dt
        self.z += self.vz * dt
        self._robot_collide()
        # intake deploy latch
        if enabled and b[0] and not self.deployed:
            self.deployed, self.deploy_g = True, self.g

    @staticmethod
    def _on_bump(x, z):
        for (x0, x1), (z0, z1) in F.BUMPS:
            if x0 <= x <= x1 and z0 <= z <= z1:
                return True
        return False

    def _robot_collide(self):
        R = self.p["robot_r"]
        boxes = list(F.ROBOT_BOXES)
        if self.held > self.p["trench_bar_held"]:
            boxes += F.TRENCH_BARS
        for (x0, x1), (z0, z1) in boxes:
            cx, cz = min(max(self.x, x0), x1), min(max(self.z, z0), z1)
            dx, dz = self.x - cx, self.z - cz
            d = math.hypot(dx, dz)
            if d < R:
                if d < 1e-6:                                            # centre inside: push out along x
                    dx, dz, d = (1.0 if self.x > (x0 + x1) / 2 else -1.0), 0.0, 1.0
                    self.x = (x1 if dx > 0 else x0) + dx * R
                else:
                    self.x, self.z = cx + dx / d * R, cz + dz / d * R
                nx, nz = dx / d, dz / d
                vn = self.vx * nx + self.vz * nz
                if vn < 0:
                    self.vx -= vn * nx; self.vz -= vn * nz
        lim_x, lim_z = F.X_WALL - R, F.Z_WALL - R
        if abs(self.x) > lim_x:
            self.x = math.copysign(lim_x, self.x); self.vx = 0.0
        if abs(self.z) > lim_z:
            self.z = math.copysign(lim_z, self.z); self.vz = 0.0

    # ------------------------------------------------------------------ balls
    def _near_balls(self, r):
        fl = self.state == FLOOR
        d2 = (self.pos[fl, 0] - self.x) ** 2 + (self.pos[fl, 1] - self.z) ** 2
        return int((d2 < r * r).sum())

    def _balls(self, dt):
        p = self.p
        st = self.state
        # landings and hub exits
        land = (st == AIR) & (self.g >= self.t_land)
        if land.any():
            self.pos[land] = self.p_land[land]; self.vel[land] = self.v_land[land]; self.y[land] = F.BALL_R
            st[land] = FLOOR
        hub = st == HUB
        if hub.any():
            due = hub & ~self.scored & (self.g >= self.t_score)
            if due.any():
                k = int(due.sum())
                self.scored[due] = True
                if self.scoring_open():
                    if self.gs == 0:
                        self.blue_auto += k
                    else:
                        self.blue_tele += k
            ex = hub & (self.g >= self.t_exit)
            if ex.any():
                self.pos[ex, 0] = p["exit_x"]; self.pos[ex, 1] = self.exit_z[ex]
                self.vel[ex] = self.exit_v[ex]; self.y[ex] = F.BALL_R
                st[ex] = FLOOR
        # rolling
        fl = st == FLOOR
        if not fl.any():
            return
        idx = np.nonzero(fl)[0]
        P = self.pos[idx]; V = self.vel[idx]
        P += V * dt
        sp = np.hypot(V[:, 0], V[:, 1])
        dec = (p["roll_k"] * np.maximum(0.0, sp - p["roll_v0"]) + p["roll_c"]) * dt
        scale = np.where(sp > 1e-9, np.maximum(0.0, sp - dec) / np.maximum(sp, 1e-9), 0.0)
        V *= scale[:, None]
        V[sp < p["v_stop"]] = 0.0
        # walls (side walls everywhere; alliance walls except the corral openings)
        r = F.BALL_R
        hit = np.abs(P[:, 1]) > F.Z_WALL - r
        P[hit, 1] = np.sign(P[hit, 1]) * (F.Z_WALL - r); V[hit, 1] *= -p["wall_e"]
        over = np.abs(P[:, 0]) > F.X_WALL - r
        if over.any():
            zz = P[:, 1] * np.sign(P[:, 0])                            # blue-side frame (red mirrored)
            opening = np.zeros(len(P), bool)
            for z0, z1 in F.CORRAL_OPEN_Z:
                opening |= (zz > z0) & (zz < z1)
            into = over & opening & (P[:, 0] > 0)                       # blue corral (red corral: static sink)
            back = over & ~opening
            P[back, 0] = np.sign(P[back, 0]) * (F.X_WALL - r); V[back, 0] *= -p["wall_e"]
            if into.any():
                self.state[idx[into]] = CORRAL
                V[into] = 0.0
        for boxes in (F.BALL_BOXES, F.BUMPS):
            for (x0, x1), (z0, z1) in boxes:
                inside = (P[:, 0] > x0 - r) & (P[:, 0] < x1 + r) & (P[:, 1] > z0 - r) & (P[:, 1] < z1 + r)
                if not inside.any():
                    continue
                pen = np.stack([P[inside, 0] - (x0 - r), (x1 + r) - P[inside, 0],
                                P[inside, 1] - (z0 - r), (z1 + r) - P[inside, 1]], 1)
                k = pen.argmin(1)
                ii = np.nonzero(inside)[0]
                for j, side in enumerate(k):
                    i = ii[j]
                    if side == 0:
                        P[i, 0] = x0 - r; V[i, 0] = -abs(V[i, 0]) * p["wall_e"]
                    elif side == 1:
                        P[i, 0] = x1 + r; V[i, 0] = abs(V[i, 0]) * p["wall_e"]
                    elif side == 2:
                        P[i, 1] = z0 - r; V[i, 1] = -abs(V[i, 1]) * p["wall_e"]
                    else:
                        P[i, 1] = z1 + r; V[i, 1] = abs(V[i, 1]) * p["wall_e"]
        self._ball_contacts(P, V)
        # robot: intake capture and body push
        self._robot_balls(idx, P, V, dt)
        self.pos[idx] = P; self.vel[idx] = V

    def _ball_contacts(self, P, V):
        """Ball-ball contacts (contact diameter 0.14 m): separate overlaps and exchange the normal velocity with
        restitution ball_e, so a push travels through a packed grid instead of balls sliding through each other."""
        p = self.p
        if len(P) < 2:
            return
        from scipy.spatial import cKDTree
        pairs = cKDTree(P).query_pairs(p["ball_d"], output_type="ndarray")
        if not len(pairs):
            return
        i, j = pairs[:, 0], pairs[:, 1]
        d = P[j] - P[i]
        dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        n = d / dist[:, None]
        over = (p["ball_d"] - dist) * 0.5
        corr = n * over[:, None]
        np.add.at(P, i, -corr); np.add.at(P, j, corr)
        vi = (V[i] * n).sum(1); vj = (V[j] * n).sum(1)
        closing = vi - vj
        m = closing > 0
        if m.any():
            e = p["ball_e"]
            dvi = -(1 + e) * 0.5 * closing[m]
            np.add.at(V, i[m], dvi[:, None] * n[m]); np.add.at(V, j[m], -dvi[:, None] * n[m])

    def _robot_balls(self, idx, P, V, dt):
        p = self.p
        a = math.radians(self.yaw)
        sa, ca = math.sin(a), math.cos(a)
        dx, dz = P[:, 0] - self.x, P[:, 1] - self.z
        near = dx * dx + dz * dz < 1.2 ** 2
        if not near.any():
            return
        lz = dx * sa + dz * ca                     # forward
        lx = dx * ca - dz * sa                     # right
        r = F.BALL_R
        _, _, _, b = self.cmd
        enabled = self.rs == 0
        ready = self.deployed and self.deploy_g is not None and self.g - self.deploy_g >= p["deploy_s"]
        front = p["in_z1"] if self.deployed else p["body_z1"]
        # intake capture
        if ready and enabled:
            win = near & (lz >= p["in_z0"]) & (lz <= p["in_z1"]) & (np.abs(lx) <= p["in_x"])
            if win.any():
                held = self.held
                capf = 1.0 if held < p["cap_soft"] else (0.55 if held < p["cap_mid"] else (0.3 if held < p["cap_hard"] else 0.0))
                qd = np.where(np.abs(lx[win]) < 0.2, p["q_dec_center"], p["q_dec_edge"]) if b[0] else \
                    np.full(int(win.sum()), p["q_dec_slow"])
                q = 1.0 - (1.0 - qd * capf) ** (1.0 / self.SUB)
                got = self.rng.random(len(q)) < q
                wi = np.nonzero(win)[0][got]
                room = max(0, p["cap_hard"] - held)
                wi = wi[:room]
                if len(wi):
                    self.state[idx[wi]] = HOPPER
                    near[wi] = False
                    V[wi] = 0.0
        # body push (balls not captured): out of the footprint along the shallowest side, with the robot's velocity
        body = near & (lx > -p["body_x"] - r) & (lx < p["body_x"] + r) & (lz > p["body_z0"] - r) & (lz < front + r)
        body &= self.state[idx] == FLOOR
        if self.deployed:                          # balls in the intake mouth sit against the roller, not the bumper
            mouth = (lz >= p["in_z0"]) & (np.abs(lx) <= p["in_x"])
            body &= ~mouth
        if not body.any():
            return
        bi = np.nonzero(body)[0]
        pen = np.stack([lx[bi] + p["body_x"] + r, p["body_x"] + r - lx[bi], lz[bi] - (p["body_z0"] - r),
                        front + r - lz[bi]], 1)
        k = pen.argmin(1)
        if self.deployed:
            # the deployed intake is a scoop: a ball that reaches the roller line inside the mouth width is held
            # against it (pushed straight ahead) until the roller takes it, instead of being overrun
            scoop = (np.abs(lx[bi]) <= p["in_x"]) & (lz[bi] < p["in_z0"]) & (lz[bi] > p["body_z0"])
            pen[scoop, 3] = p["in_z0"] - lz[bi][scoop]
            k = np.where(scoop, 3, k)
        nlx = np.where(k == 0, -1.0, np.where(k == 1, 1.0, 0.0))
        nlz = np.where(k == 2, -1.0, np.where(k == 3, 1.0, 0.0))
        dpen = pen[np.arange(len(bi)), k]
        # robot frame -> world: forward (sa, ca), right (ca, -sa)
        nwx = nlz * sa + nlx * ca
        nwz = nlz * ca - nlx * sa
        P[bi, 0] += nwx * dpen; P[bi, 1] += nwz * dpen
        # velocity of the robot at the contact point (planar), normal component, pushed with restitution
        vpx, vpz = self.vx, self.vz                 # chassis rotation neglected at the bumper
        vn = vpx * nwx + vpz * nwz
        vn = np.maximum(vn, 0.0) * (1.0 + p["push_e"])
        vb_n = V[bi, 0] * nwx + V[bi, 1] * nwz
        dv = np.maximum(vn - vb_n, 0.0)
        V[bi, 0] += dv * nwx; V[bi, 1] += dv * nwz

    # ------------------------------------------------------------------ outpost
    def _outpost(self, dt):
        p = self.p
        near_door = math.hypot(self.x - F.CHUTE_DOOR[0], self.z - F.CHUTE_DOOR[1]) < F.OPEN_DISTANCE
        chute = np.nonzero(self.state == CHUTE)[0]
        corral = np.nonzero(self.state == CORRAL)[0]
        if len(chute) >= F.CHUTE_CAP and len(corral) >= 12 and self.g >= self.door_open_until:
            self.door_open_until = self.g + 2.0
        is_open = near_door or self.g < self.door_open_until
        if is_open and len(chute):
            self.release_acc += p["chute_release_rate"] * dt
            k = min(int(self.release_acc), len(chute))
            if k:
                self.release_acc -= k
                ii = chute[:k]
                self.state[ii] = FLOOR
                self.pos[ii, 0] = 8.10
                self.pos[ii, 1] = self.rng.uniform(3.1, 3.7, k)
                self.vel[ii, 0] = -self.rng.uniform(0.3, 2.6, k)
                self.vel[ii, 1] = self.rng.normal(0.0, 0.3, k)
                self.y[ii] = F.BALL_R
        elif not is_open:
            self.release_acc = 0.0
            self.corral_timer += dt
            if len(corral) and len(chute) < F.CHUTE_CAP and self.corral_timer >= p["corral_teleport_s"]:
                self.corral_timer = 0.0
                self.state[corral[0]] = CHUTE
                self.pos[corral[0]] = (8.9, 3.3)

    # ------------------------------------------------------------------ shooter
    def _shoot(self, dt):
        p = self.p
        _, _, _, b = self.cmd
        enabled = self.rs == 0
        wants = b[1] or b[2] or b[3]
        x, z = self.x, self.z
        in_zone = x > p["zone_x"]
        bump_strip = 3.5 < x <= 3.8
        is_pass = b[2] or (not in_zone and not bump_strip)
        under_trench = 3.418 < abs(x) < 3.872 and abs(z) > 2.7
        # turret: target bearing rate relative to the chassis (shoot-on-move aim point = target - v * tof)
        if is_pass:
            tx, tz = F.PASS_NODES[0] if z < 0 else F.PASS_NODES[1]
        else:
            tx, tz = F.HUB_NODE
        if wants:
            tx -= self.vx * p["som_tof"]; tz -= self.vz * p["som_tof"]
        bearing = math.atan2(tx - x, tz - z)
        rate = 0.0
        if self.prev_bearing is not None:
            db = (bearing - self.prev_bearing + math.pi) % (2 * math.pi) - math.pi
            rate = db / dt - self.w
        self.prev_bearing = bearing
        self.last_err = rate / 2.0                 # steady turret lag in degrees ~ bearing rate (rad/s) / 2
        turret_ready = abs(rate) <= p["turret_rate_max"]
        gate = turret_ready and self.latch <= 0 and not under_trench and not bump_strip
        self.gate_now = {"err": round(self.last_err, 3), "wrap": 0, "bump": int(self.latch > 0),
                         "trench": int(under_trench), "bz": int(bump_strip)}
        if not (enabled and wants and gate):
            self.shot_acc = 0.0 if not wants else self.shot_acc
            return
        held = self.held
        if held <= 0:
            return
        self.shot_acc += p["r_max"] * (1.0 - math.exp(-held / p["r_k"])) * dt
        k = min(int(self.shot_acc), held)
        if k <= 0:
            return
        self.shot_acc -= k
        hop = np.nonzero(self.state == HOPPER)[0][:k]
        rng = self.rng
        for i in hop:
            self.p_from[i] = (x, z); self.t_from[i] = self.g
            if not is_pass and b[1]:
                if rng.random() < p["hit_p"]:
                    self.state[i] = HUB
                    self.scored[i] = False
                    th = float(np.clip(rng.normal(p["hub_t_mu"], p["hub_t_sd"]), 1.4, 3.6))
                    self.t_score[i] = self.g + th - 0.1
                    self.t_exit[i] = self.g + th
                    self.exit_z[i] = float(np.clip(rng.normal(0.0, p["exit_z_sd"]), -0.5, 0.5))
                    self.exit_v[i] = (-(p["exit_v0"] + rng.exponential(p["exit_v_scale"])), rng.normal(0, p["exit_vz_sd"]))
                    self.pos[i] = F.HUB_NODE; self.y[i] = 1.9
                    continue
                lx, lz = 4.4 + rng.uniform(0, 0.8), rng.normal(0.0, 0.8)          # miss: drops beside the hub
                tf = 1.3
            else:
                sgn = -1.0 if z < 0 else 1.0
                lx = rng.normal(p["pass_x_mu"], p["pass_x_sd"])
                lz = sgn * rng.normal(p["pass_z_mu"], p["pass_z_sd"])
                tf = float(np.clip(rng.normal(p["pass_t_mu"], p["pass_t_sd"]), 0.9, 1.8))
            lx = float(np.clip(lx, 3.9, F.X_WALL - 0.1))
            lz = float(np.clip(lz, -F.Z_WALL + 0.1, F.Z_WALL - 0.1))
            self.state[i] = AIR
            self.t_land[i] = self.g + tf
            self.p_land[i] = (lx, lz)
            d = np.array([lx - x, lz - z]); d /= max(np.linalg.norm(d), 1e-6)
            self.v_land[i] = d * p["pass_land_v"] * rng.uniform(0.3, 1.7)

    # ------------------------------------------------------------------ Bridge-shaped state
    def state_dict(self) -> dict:
        vis = np.nonzero(self.state != HOPPER)[0]
        pos = self.pos[vis]; y = self.y[vis].copy()
        air = self.state[vis] == AIR
        if air.any():
            ii = vis[air]
            u = np.clip((self.g - self.t_from[ii]) / np.maximum(self.t_land[ii] - self.t_from[ii], 1e-3), 0, 1)
            pos = pos.copy()
            pos[air] = self.p_from[ii] + (self.p_land[ii] - self.p_from[ii]) * u[:, None]
            y[air] = 0.5 + 4.0 * u * (1 - u) * 2.5
        fuel = np.concatenate([np.round(pos, 3), np.round(y, 3)[:, None]], 1).tolist()
        done = self.end_g is not None and self.g - self.end_g >= 3.5
        slide = 0.0
        if self.deployed:
            slide = 0.305 * min(1.0, (self.g - self.deploy_g) / self.p["deploy_s"])
        return {
            "ok": True, "done": done, "t": round(self.t, 3), "gs": self.gs, "rs": self.rs,
            "blue": self.blue_auto + self.blue_tele, "blueAuto": self.blue_auto, "red": 0,
            "hub": 2 if F.blue_hub_active(self.t, self.gs) else 0, "gameTime": round(self.g, 4),
            "robot": {"x": self.x, "z": self.z, "y": 0.0, "yaw": self.yaw, "pitch": 0.0, "roll": 0.0,
                      "vx": self.vx, "vz": self.vz, "wy": self.w, "inZone": self.x > self.p["zone_x"]},
            "mech": {"dep": int(self.deployed), "kick": int(self.deployed), "slide": round(slide, 3)},
            "gate": getattr(self, "gate_now", None),
            "fuel": fuel, "fid": vis.tolist(), "held": self.held,
        }


class MiniSimClient:
    """Drop-in for MoSimClient in episode.run_episode."""

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.sim = MiniSim(params, seed)
        self.last_cmd = ""

    def reset(self) -> dict:
        return self.sim.reset()

    def act(self, vx, vz, rot, buttons, rff: bool = False) -> dict:
        return self.sim.act(vx, vz, rot, buttons)
