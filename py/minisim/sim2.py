"""MiniSim v2: MoSim game logic on top of a PhysX-parameter rigid-body ball world (phys.py).

Modelled on MoSim's behaviour (re-implemented, no game code copied; geometry from the field scene's colliders):
  - ball physics: PhysX parameters and the level2 static colliders (phys.py)
  - match clock, 3 s auto->teleop pause, hub shifts and HubScoring's 3 s grace (RebuiltShifts.cs, HubScoring.cs)
  - Hightide: shooting multiplier 0.55, AutoShoot/AutoPass targets, feed gates (turret error, bump-crossing latch,
    trench, bump strip), intake deploy latch (Hightide.cs)
  - outpost: chute capacity 24, corral teleport, door opens within 1 m of the robot or at match start (Outpost.cs)
  - drivetrain: DriveController force law structure (falloff (1 - 0.075 v/5.18)^10), rigidbody drag
Measured where MoSim leaves it to PhysX internals we do not simulate (calib/*, cal1 dense data):
  - effective drive gain / drag, yaw response, bump ramp slow-down
  - roller capture rate, fire rate vs hopper count, hub transit time, hub exit velocity, pass landing spread
The interface is the Bridge's (reset / act -> state dict), so GhostPolicy and run_episode work unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from . import field as F
from .phys import R_FIELD, Balls, StaticWorld
from .phys_nb import NBBalls
from .robot_nb import robot_collide, roller_touch

LIVE, HOPPER, HUB, AIRHUB = 0, 1, 2, 3          # AIRHUB: hub-bound shot in flight (not in the ball world)

DEFAULT = dict(
    # drive: dv/dt = K*u*falloff - D*v - C*v/|v| (calib/kin.fit_force_coulomb on r4+hr1b+cal1; the C term is what makes
    # top speed ~proportional to the stick: 2.49 m/s full, 1.37 at the 0.55 shooting multiplier, as measured)
    # C_drive: constant drivetrain resistance. Real one-step data wants it (K=19.01, D=4.079, C=2.983; present even with
    # no balls near), but with the contact multipliers tuned for the old law the closed loop collapses (route arms
    # -90..-330 even with C scaled by the contact multiplier). Off until the contact models are refitted with it.
    K=19.62, D=5.872, C_drive=0.0, shoot_mult=0.55, w_max=-4.575, w_tau=0.1445, robot_mass=118.0,
    bump_acc=2.87, bump_drive=0.75, bump_latch_speed=1.75, bump_latch_s=1.0, bump_clear_crest_p=0.5,
    trench_vmax=1.9, trench_bar_held=100, robot_r=0.45,
    # robot footprint (robot frame, x right / z forward) and contact with field structures (default material mu 0.6;
    # the trench divider triangles are ramps the bumper rides up on: treated as sticky)
    body_x=0.44, body_z0=-0.37, body_z1=0.36, slide_x=0.40, slide_z1=0.64, robot_I=12.7, mu_field=0.6, mu_divider=1.2,
    bump_up_decel=4.4, bump_stuck_v=0.2, bump_stuck_s=(1.0, 3.0),
    # a bumper corner pushed into a trench-divider triangle rides up its 30 deg ramp (cal2: pitch -10 deg, y +4 cm,
    # wheels slip): drive drops until the robot is backed off it
    climb_pen=0.02, climb_drive=0.15,
    # robot colliders (robot frame: x right, y up, z forward)
    chassis_c=(0.0, 0.11, -0.015), chassis_h=(0.44, 0.09, 0.345),
    plate_x=0.405, plate_hx=0.035, plate_h=0.10,
    roller_z_dep=0.57, roller_z_stow=0.27, roller_y=0.185, roller_half=0.38, roller_reach=0.125,
    p_cap_full=0.12, p_cap_slow=0.004, deploy_s=0.3, cap_soft=95, cap_mid=105, cap_hard=112,
    # shooter / hub
    # hub shots (cal1/cal2 dense data): 94% of the balls that reach the hub top score; the other 6% bounce off the rim
    # and drop in front of the hub on the neutral side (score / hub passages = 0.94 in every episode, independent of
    # frame rate); another 2.2% miss beside the hub into the zone
    # rim_scale: sysid knob on the hit table's rim share (the difference goes to / comes from misses; P(score) fixed)
    r_max=17.5, r_k=15.0, turret_rate_max=1.2, turret_gain=2.0, turret_wmax=10.0, turret_tol_deg=0.6, feed_hold_s=0.1, som_tof=1.15, hit_p=0.92, rim_p=0.058, rim_scale=1.0,
    hub_t_mu=2.2, hub_t_sd=0.45, exit_x=3.0, exit_y=0.80, exit_z_sd=0.25, exit_v0=1.35, exit_v_scale=0.45,
    exit_vz_sd=0.3,
    pass_t_mu=1.29, pass_t_sd=0.15, pass_x_mu=5.5, pass_x_sd=0.95, pass_z_mu=1.7, pass_z_sd=0.4, turret_y=0.60,
    # launches (calib/launch.py, launch_table.json): speed / elevation keyed on the 3-D turret-to-hub distance, aimed at
    # the shoot-on-move-adjusted target; pass shots fly in the ball world (they fall short from deep in the neutral
    # zone), hub-bound shots fly visibly for hub_top_t before entering the hub model
    physical_pass=True, hub_top_t=0.79, launch_v_scale=1.0,
    zone_x=3.738, corral_teleport_s=0.5,
)

CHUTE_BOX = ((8.318, 9.06), (0.72, 1.10), (2.91, 3.83))       # Outpost chute trigger (blue)
CORRAL_BOX = ((8.80, 9.22), (0.0, 0.38), (2.91, 3.83))        # corral trigger
FUEL_SPAWN = (8.933, 0.99)


class MiniSim2:
    DEC_STEPS = 22
    DT = 0.0045

    def __init__(self, params: dict | None = None, seed: int = 0, world: StaticWorld | None = None):
        self.p = {**DEFAULT, **(params or {})}
        self.contact = contact_arrays(self.p)
        self.rng = np.random.default_rng(seed)
        self.world = world or StaticWorld()
        self.fuel0 = F.initial_fuel()
        self.n = len(self.fuel0) + 8

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> dict:
        n, m = self.n, len(self.fuel0)
        self.world.enabled[:] = True
        self.b = (NBBalls if self.p.get("numba", True) else Balls)(n, self.world)
        self.b.p[:m] = np.stack([self.fuel0[:, 0], self.fuel0[:, 2], self.fuel0[:, 1]], 1)   # x, y, z
        self.b.live[:m] = True
        self.state = np.full(n, LIVE, int); self.state[m:] = HOPPER
        self.t_cap = np.zeros(n)                    # hopper queue order: the ball captured first leaves first
        self.t_score = np.zeros(n); self.t_exit = np.zeros(n); self.scored = np.zeros(n, bool)
        self.exit_z = np.zeros(n); self.exit_v = np.zeros((n, 2))
        self.fly = np.zeros((n, 7)); self.fly_rim = np.zeros(n, bool)          # t0, p0 (x y z), v0 (x y z)
        self.x, self.z, self.yaw = 3.786, -3.628, 359.6
        self.vx = self.vz = self.w = 0.0
        self.deployed, self.deploy_g = False, None
        self.latch = 0.0; self.latch_crest = False; self.stuck_left = 0.0; self.climbed = False
        self.shot_acc = 0.0; self.prev_bearing = None; self.turret = None; self.gate_open_g = -1e9
        self.cmd = (0.0, 0.0, 0.0, [False] * 5)
        self.g, self.t, self.gs, self.rs = 0.0, 160.0, 0, 0
        self.pause_left, self.end_g = 3.0, None
        self.blue_auto = self.blue_tele = 0
        self.door_open_until = 2.3
        self.corral_timer = 0.0
        self.gate_now = None
        return self.state_dict()

    @property
    def held(self) -> int:
        return int((self.state == HOPPER).sum())

    def act(self, vx, vz, rot, buttons, rff: bool = False) -> dict:
        b = [bool(x) for x in (list(buttons) + [False] * 5)[:5]]
        self.cmd = (float(np.clip(vx, -1, 1)), float(np.clip(vz, -1, 1)), float(np.clip(rot, -1, 1)), b)
        for _ in range(self.DEC_STEPS):
            self._clock(self.DT)
            self._robot(self.DT)
            self._outpost(self.DT)
            self._hub(self.DT)
            J = self.b.step(self.DT, self._robot_body())
            self.vx += J[0] / self.p["robot_mass"]; self.vz += J[2] / self.p["robot_mass"]
            self._intake(self.DT)
            self._shoot(self.DT)
            self.g += self.DT
        fl = np.nonzero(self.state == AIRHUB)[0]
        if len(fl):
            tau = (self.g - self.fly[fl, 0])[:, None]
            self.b.p[fl] = self.fly[fl, 1:4] + self.fly[fl, 4:7] * tau + np.array([0.0, -4.905, 0.0]) * tau ** 2
        return self.state_dict()

    # ------------------------------------------------------------------ clock
    def _clock(self, dt):
        if self.end_g is not None:
            return
        if self.gs == 0 and self.rs == 0:
            self.t -= dt
            if self.t <= 140.0:
                self.t, self.rs = 140.0, 1            # 3 s pause: robot disabled, GameState still Auto
        elif self.gs == 0 and self.rs == 1:
            self.pause_left -= dt
            if self.pause_left <= 0:
                self.gs, self.rs = 1, 0
        else:
            self.t -= dt
            if self.t <= 30.0 and self.gs == 1:
                self.gs = 2
            if self.t <= 0.0:
                self.t, self.gs, self.rs, self.end_g = 0.0, 3, 1, self.g

    def scoring_open(self) -> bool:
        if self.end_g is not None:
            return self.g - self.end_g < 3.0
        if F.blue_hub_active(self.t, self.gs):
            return True
        edge = F.last_deactivation(self.t)
        return edge is not None and edge - self.t < 3.0

    # ------------------------------------------------------------------ robot
    def _robot(self, dt):
        p = self.p
        ux, uz, rot, b = self.cmd
        enabled = self.rs == 0
        if not enabled:
            ux = uz = rot = 0.0
        wants = (b[1] or b[2] or b[3]) and enabled
        mult = p["shoot_mult"] if wants else 1.0
        ux, uz = ux * mult, uz * mult
        nrm = math.hypot(ux, uz)
        if nrm > 1.0:
            ux, uz = ux / nrm, uz / nrm
        s = math.hypot(self.vx, self.vz)
        f = (1.0 - min(s / 5.18, 1.0) * 0.075) ** 10
        on_bump = _on_bump(self.x, self.z)
        drive = (p["bump_drive"] if on_bump else 1.0) * (p["climb_drive"] if self.climbed else 1.0)
        D = p["D"]
        tx, tz = drive * p["K"] * ux * f / D, drive * p["K"] * uz * f / D
        e = math.exp(-D * dt)
        self.vx = tx + (self.vx - tx) * e
        self.vz = tz + (self.vz - tz) * e
        s1 = math.hypot(self.vx, self.vz)
        cf = p["C_drive"] * drive * dt              # contact multipliers (bump, divider climb) scale net drive
        if s1 > cf:
            self.vx -= cf * self.vx / s1; self.vz -= cf * self.vz / s1
        else:
            self.vx = self.vz = 0.0
        if self.stuck_left > 0:                        # high-centred on a bump crest: wheels off the carpet
            self.stuck_left -= dt
            self.vx = self.vz = 0.0
        if on_bump:
            peak = F.BUMP_PEAK_X if self.x > 0 else -F.BUMP_PEAK_X
            if abs(self.x - peak) < F.BUMP_HALF:
                self.vx += p["bump_acc"] * math.copysign(1.0, self.x - peak) * dt
            if math.hypot(self.vx, self.vz) > p["bump_latch_speed"] and abs(self.x - peak) > 0.15:
                self.latch = p["bump_latch_s"]
                self.latch_crest = self.rng.random() < p["bump_clear_crest_p"]
            if self.latch > 0 and self.latch_crest and abs(self.x - peak) < 0.08:
                self.latch = 0.0
        if self.latch > 0 and not (on_bump and math.hypot(self.vx, self.vz) > p["bump_latch_speed"]):
            self.latch = max(0.0, self.latch - dt)
        if 3.07 < abs(self.x) < 4.21 and 2.76 < abs(self.z) < 4.03:
            sp = math.hypot(self.vx, self.vz)
            if sp > p["trench_vmax"]:
                self.vx *= p["trench_vmax"] / sp; self.vz *= p["trench_vmax"] / sp
        wt = p["w_max"] * rot
        self.w = wt + (self.w - wt) * math.exp(-dt / p["w_tau"])
        self.yaw = (self.yaw + math.degrees(self.w) * dt) % 360.0
        self.x += self.vx * dt; self.z += self.vz * dt
        self._robot_collide()
        if enabled and b[0] and not self.deployed:
            self.deployed, self.deploy_g = True, self.g

    def _robot_rects(self):
        p = self.p
        rects = [(p["body_x"], p["body_z0"], p["body_z1"])]
        if self.deployed:
            rects.append((p["slide_x"], p["body_z1"], p["slide_z1"]))
        return rects

    def _robot_collide(self):
        """Robot footprint (bumper rectangle + deployed intake) against the field structures' footprints (2D AABBs
        of the colliders at robot height), linear response with Coulomb friction (robot_nb.robot_collide)."""
        p = self.p
        obst, obst_bar, rect_stow, rect_dep = self.contact
        boxes = obst if self.held <= p["trench_bar_held"] else obst_bar
        rects = rect_dep if self.deployed else rect_stow
        self.x, self.z, self.vx, self.vz, self.climbed = robot_collide(
            self.x, self.z, self.yaw, self.vx, self.vz, boxes, rects, p["mu_divider"], p["climb_pen"])

    def _frame(self):
        a = math.radians(self.yaw)
        sa, ca = math.sin(a), math.cos(a)
        # rows = world directions of robot-local x (right), y (up), z (forward)
        return np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])

    def _robot_body(self):
        p = self.p
        Rm = self._frame()
        c0 = np.array([self.x, 0.0, self.z])
        boxes = [(c0 + np.array(p["chassis_c"]) @ Rm, np.array(p["chassis_h"]), Rm)]
        if self.deployed:
            rz = p["roller_z_dep"]
            zc, zh = (0.33 + rz) / 2, (rz - 0.33) / 2 + 0.03
            for sx in (-1, 1):
                c = np.array([sx * p["plate_x"], p["plate_h"], zc]) @ Rm
                boxes.append((c0 + c, np.array([p["plate_hx"], p["plate_h"], zh]), Rm))
        vx, vz, w = self.vx, self.vz, self.w

        def vel_at(P):
            rx, rz = P[:, 0] - self.x, P[:, 2] - self.z
            # yaw rate w (rad/s, yaw increasing clockwise from above = rotation about +y in Unity's left-handed frame)
            return np.stack([vx + w * rz, np.zeros(len(P)), vz - w * rx], 1)
        return {"center": c0, "reach": 1.1, "boxes": boxes, "vel_at": vel_at, "vel": (vx, vz, w)}

    # ------------------------------------------------------------------ intake
    def _intake(self, dt):
        p = self.p
        _, _, _, b = self.cmd
        if not (self.deployed and self.rs == 0 and self.g - self.deploy_g >= p["deploy_s"]):
            return
        touch = roller_touch(self.b.p, self.b.live, self.x, self.z, self.yaw, p["roller_half"], p["roller_z_dep"],
                             p["roller_y"], p["roller_reach"])
        if not len(touch):
            return
        held = self.held
        capf = 1.0 if held < p["cap_soft"] else (0.55 if held < p["cap_mid"] else (0.3 if held < p["cap_hard"] else 0.0))
        q = (p["p_cap_full"] if b[0] else p["p_cap_slow"]) * capf
        got = touch[self.rng.random(len(touch)) < q]
        got = got[:max(0, p["cap_hard"] - held)]
        if len(got):
            self.b.live[got] = False; self.b.awake[got] = False
            self.state[got] = HOPPER
            self.t_cap[got] = self.g

    # ------------------------------------------------------------------ outpost
    def _outpost(self, dt):
        near = math.hypot(self.x - F.CHUTE_DOOR[0], self.z - F.CHUTE_DOOR[1]) < F.OPEN_DISTANCE
        P = self.b.p
        live = self.b.live

        def count(box):
            (x0, x1), (y0, y1), (z0, z1) = box
            m = live & (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)
            return np.nonzero(m)[0]
        self.count_timer = getattr(self, "count_timer", 0.1) + dt
        if self.count_timer >= 0.1 or not hasattr(self, "_chute"):
            self.count_timer = 0.0
            self._chute, self._corral = count(CHUTE_BOX), count(CORRAL_BOX)
        chute, corral = self._chute, self._corral
        if len(chute) >= F.CHUTE_CAP and len(corral) >= 12 and self.g >= self.door_open_until:
            self.door_open_until = self.g + 2.0
        is_open = near or self.g < self.door_open_until
        was = not self.world.enabled[self.world.blue_door].all()
        self.world.enabled[self.world.blue_door] = not is_open
        if is_open and not was:
            self.b.wake(chute)
        if not is_open:
            self.corral_timer += dt
            if len(corral) and len(chute) < F.CHUTE_CAP and self.corral_timer >= self.p["corral_teleport_s"]:
                self.corral_timer = 0.0
                i = corral[0]
                self.b.p[i] = (FUEL_SPAWN[0], FUEL_SPAWN[1], self.b.p[i, 2]); self.b.v[i] = 0; self.b.w[i] = 0
                self.b.wake([i])

    # ------------------------------------------------------------------ hub
    def _hub(self, dt):
        fl = np.nonzero(self.state == AIRHUB)[0]
        if len(fl):
            done = fl[self.g - self.fly[fl, 0] >= self.p["hub_top_t"]]
            for i in done:
                if self.fly_rim[i]:                       # off the rim: falls out in front of the hub, unscored
                    lx, lz, tf = self.rng.uniform(2.3, 2.95), self.rng.normal(0.0, 0.35), 1.0
                    y0 = 1.9
                    vy = (R_FIELD - y0 + 0.5 * 9.81 * tf * tf) / tf
                    self.state[i] = LIVE; self.b.live[i] = True
                    self.b.p[i] = (F.HUB_NODE[0] - 0.4, y0, F.HUB_NODE[1] + lz * 0.3)
                    self.b.v[i] = ((lx - (F.HUB_NODE[0] - 0.4)) / tf, vy, (lz - lz * 0.3) / tf)
                    self.b.w[i] = 0.0; self.b.wake([i])
                else:
                    self.state[i] = HUB
                    self.b.p[i] = (F.HUB_NODE[0], 1.9, F.HUB_NODE[1])
        hub = np.nonzero(self.state == HUB)[0]
        if not len(hub):
            return
        due = hub[~self.scored[hub] & (self.g >= self.t_score[hub])]
        if len(due):
            self.scored[due] = True
            if self.scoring_open():
                if self.gs == 0:
                    self.blue_auto += len(due)
                else:
                    self.blue_tele += len(due)
        ex = hub[self.g >= self.t_exit[hub]]
        if len(ex):
            p = self.p
            self.state[ex] = LIVE
            self.b.live[ex] = True
            self.b.p[ex] = np.stack([np.full(len(ex), p["exit_x"]), np.full(len(ex), p["exit_y"]), self.exit_z[ex]], 1)
            self.b.v[ex] = np.stack([self.exit_v[ex, 0], np.zeros(len(ex)), self.exit_v[ex, 1]], 1)
            self.b.w[ex] = 0.0
            self.b.wake(ex)

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
        tx, tz = (F.PASS_NODES[0] if z < 0 else F.PASS_NODES[1]) if is_pass else F.HUB_NODE
        tx0, tz0 = tx, tz
        if wants:
            tx -= self.vx * p["som_tof"]; tz -= self.vz * p["som_tof"]
        bearing = math.atan2(tx - x, tz - z)
        # turret (hightide_contract.md §3): the joint slews at clamp(2 * error_deg, ±10) rad/s on top of the chassis
        # rotation it rides on; the feed gate is |error| <= 0.6 deg. Gating on the error (not on the per-step rate of
        # the aim bearing) matters: ball contacts jolt the chassis velocity, the shoot-on-move aim point jumps by
        # dv * 1.15 s, and a 4.5 ms finite difference turns that into rate spikes the real turret never sees.
        if self.turret is None:
            self.turret = bearing
        err = (bearing - self.turret + math.pi) % (2 * math.pi) - math.pi
        err_deg = math.degrees(err)
        omega = max(-p["turret_wmax"], min(p["turret_wmax"], p["turret_gain"] * err_deg))
        self.turret = (self.turret + (self.w + omega) * dt + math.pi) % (2 * math.pi) - math.pi
        gate = abs(err_deg) <= p["turret_tol_deg"] and self.latch <= 0 and not under_trench and not bump_strip
        # balls already in the uptake / flywheel still leave after the gate shuts (real per-decision fire rate stays
        # ~9/s at 0.6-1.0 deg turret error): the feed keeps going for feed_hold_s after the last open step
        if gate:
            self.gate_open_g = self.g
        elif self.g - self.gate_open_g <= p["feed_hold_s"] and self.latch <= 0 and not under_trench and not bump_strip:
            gate = True
        self.gate_now = {"err": round(err_deg, 3), "wrap": 0, "bump": int(self.latch > 0),
                         "trench": int(under_trench), "bz": int(bump_strip)}
        if not (enabled and wants and gate):
            if not wants:
                self.shot_acc = 0.0
            return
        held = self.held
        if held <= 0:
            return
        self.shot_acc += p["r_max"] * (1.0 - math.exp(-held / p["r_k"])) * dt
        k = min(int(self.shot_acc), held)
        if k <= 0:
            return
        self.shot_acc -= k
        rng = self.rng
        hop = np.nonzero(self.state == HOPPER)[0]
        for i in hop[np.argsort(self.t_cap[hop], kind="stable")][:k]:
            u = rng.random()
            hub_shot = not is_pass and b[1]
            if hub_shot:
                ps, pr, _ = hit_probs(math.hypot(x - F.HUB_NODE[0], z - F.HUB_NODE[1]), math.hypot(self.vx, self.vz))
                pr = min(pr * p["rim_scale"], 1.0 - ps)
            else:
                ps, pr = p["hit_p"], p["rim_p"]
            if hub_shot and u < ps + pr:
                # hub-bound (scores, or bounces off the rim): visible ballistic flight, then the hub model
                p0, v0 = self._launch(True, F.HUB_NODE[0], F.HUB_NODE[1])
                self.state[i] = AIRHUB; self.fly_rim[i] = u >= ps
                self.fly[i, 0] = self.g; self.fly[i, 1:4] = p0; self.fly[i, 4:7] = v0
                self.b.p[i] = p0
                if u < ps:
                    self.scored[i] = False
                    th = float(np.clip(rng.normal(p["hub_t_mu"], p["hub_t_sd"]), 1.3, 3.5))
                    self.t_score[i] = self.g + th
                    self.t_exit[i] = self.g + th + 0.2
                    self.exit_z[i] = float(np.clip(rng.normal(0.0, p["exit_z_sd"]), -0.5, 0.5))
                    self.exit_v[i] = (-(p["exit_v0"] + rng.exponential(p["exit_v_scale"])), rng.normal(0.0, p["exit_vz_sd"]))
                continue
            if hub_shot:                                      # miss: drops beside the hub
                lx, lz, tf = 4.4 + rng.uniform(0, 0.8), rng.normal(0.0, 0.8), 1.3
            elif p["physical_pass"]:
                tx, tz = tx0, tz0
                p0, v0 = self._launch(False, tx, tz)
                self.state[i] = LIVE; self.b.live[i] = True
                self.b.p[i] = p0; self.b.v[i] = v0; self.b.w[i] = 0.0
                self.b.wake([i])
                continue
            else:
                sgn = -1.0 if z < 0 else 1.0
                lx = rng.normal(p["pass_x_mu"], p["pass_x_sd"])
                lz = sgn * rng.normal(p["pass_z_mu"], p["pass_z_sd"])
                tf = float(np.clip(rng.normal(p["pass_t_mu"], p["pass_t_sd"]), 0.9, 1.8))
            lx = float(np.clip(lx, 3.9, F.X_WALL - 0.2)); lz = float(np.clip(lz, -F.Z_WALL + 0.2, F.Z_WALL - 0.2))
            y0 = p["turret_y"]
            vy = (R_FIELD - y0 + 0.5 * 9.81 * tf * tf) / tf
            self.state[i] = LIVE
            self.b.live[i] = True
            self.b.p[i] = (x, y0, z)
            self.b.v[i] = ((lx - x) / tf, vy, (lz - z) / tf)
            self.b.w[i] = 0.0
            self.b.wake([i])

    def _launch(self, hub: bool, tx: float, tz: float):
        """Ball launch from the turret: speed and elevation from the measured tables at the 3-D distance between the
        turret and the shoot-on-move-adjusted hub node (Hightide keys pass power on the HUB distance too); azimuth at
        the shoot-on-move-adjusted target; the ball also carries the robot's velocity. Returns p0, v0 as (x, y, z)."""
        p, rng = self.p, self.rng
        som = p["som_tof"]
        hx, hz = F.HUB_NODE[0] - self.vx * som, F.HUB_NODE[1] - self.vz * som
        d = math.sqrt((hx - self.x) ** 2 + (hz - self.z) ** 2 + (1.938 - 0.476) ** 2)
        v, v_sd, el, el_sd, az_mu, az_sd = launch_params(hub, d)
        v = max(0.5, (v + rng.normal(0.0, v_sd)) * p["launch_v_scale"])
        el = math.radians(el + rng.normal(0.0, el_sd))
        az = math.atan2(tz - self.vz * som - self.z, tx - self.vx * som - self.x) + math.radians(az_mu + rng.normal(0.0, az_sd))
        vh = v * math.cos(el)
        v0 = (self.vx + vh * math.cos(az), v * math.sin(el), self.vz + vh * math.sin(az))
        return (self.x, p["turret_y"], self.z), v0

    # ------------------------------------------------------------------ Bridge-shaped state
    def state_dict(self) -> dict:
        vis = np.nonzero(self.state != HOPPER)[0]
        P = self.b.p[vis]
        fuel = np.stack([P[:, 0], P[:, 2], P[:, 1]], 1).round(3).tolist()
        done = self.end_g is not None and self.g - self.end_g >= 3.5
        slide = 0.305 * min(1.0, (self.g - self.deploy_g) / self.p["deploy_s"]) if self.deployed else 0.0
        return {
            "ok": True, "done": done, "t": round(self.t, 3), "gs": self.gs, "rs": self.rs,
            "blue": self.blue_auto + self.blue_tele, "blueAuto": self.blue_auto, "red": 0,
            "hub": 2 if F.blue_hub_active(self.t, self.gs) else 0, "gameTime": round(self.g, 4),
            "robot": {"x": self.x, "z": self.z, "y": 0.0, "yaw": self.yaw, "pitch": 0.0, "roll": 0.0,
                      "vx": self.vx, "vz": self.vz, "wy": self.w, "inZone": self.x > self.p["zone_x"]},
            "mech": {"dep": int(self.deployed), "kick": int(self.deployed), "slide": round(slide, 3)},
            "gate": self.gate_now, "fuel": fuel, "fid": vis.tolist(), "held": self.held,
        }


_HIT = None


def hit_probs(dist, speed):
    """P(score), P(rim fall-out), P(miss) of an AutoShoot hub shot from the robot's distance to the hub node and its
    speed at launch (calib/hitmodel.py, 84k real launches)."""
    global _HIT
    if _HIT is None:
        import json
        from pathlib import Path
        _HIT = json.load(open(Path(__file__).with_name("hit_table.json")))
    i = min(max(np.searchsorted(_HIT["d_bins"], dist, side="right") - 1, 0), len(_HIT["d_bins"]) - 2)
    j = min(max(np.searchsorted(_HIT["v_bins"], speed, side="right") - 1, 0), len(_HIT["v_bins"]) - 2)
    return _HIT["p"][i][j]


_LAUNCH = None


def launch_params(hub: bool, d: float):
    """(v, v_sd, elevation deg, el_sd, az_mu deg, az_sd) of a hub shot / pass shot at turret-to-hub distance d
    (calib/launch.py; linear in d between bins, clamped at the ends)."""
    global _LAUNCH
    if _LAUNCH is None:
        import json
        from pathlib import Path
        T = json.load(open(Path(__file__).with_name("launch_table.json")))
        _LAUNCH = {}
        for name in ("hub", "pass"):
            rows = [r for r in T[name]["bins"] if r and r["n"] >= 300]
            _LAUNCH[name] = (np.array([[r["d"], r["v"], r["v_sd"], r["el"], r["el_sd"]] for r in rows]),
                             T[name]["az_mu"], T[name]["az_sd"])
    A, az_mu, az_sd = _LAUNCH["hub" if hub else "pass"]
    return (float(np.interp(d, A[:, 0], A[:, 1])), float(np.interp(d, A[:, 0], A[:, 2])),
            float(np.interp(d, A[:, 0], A[:, 3])), float(np.interp(d, A[:, 0], A[:, 4])), az_mu, az_sd)


def _load_obstacles(p=DEFAULT):
    import json
    from pathlib import Path
    rows = json.load(open(Path(__file__).with_name("robot_obstacles.json")))
    obs = [(r["x"][0], r["x"][1], r["z"][0], r["z"][1], p["mu_divider"] if r["divider"] else p["mu_field"])
           for r in rows]
    bars = [(x0, x1, z0, z1, p["mu_field"]) for (x0, x1), (z0, z1) in F.TRENCH_BARS]
    return obs, bars


def contact_arrays(p=DEFAULT):
    """Robot-vs-structure contact inputs built from a param dict (so friction and footprint are sysid-able)."""
    obs, bars = _load_obstacles(p)
    return (np.array(obs, float), np.array(obs + bars, float),
            np.array([[p["body_x"], p["body_z0"], p["body_z1"]]]),
            np.array([[p["body_x"], p["body_z0"], p["body_z1"]], [p["slide_x"], p["body_z1"], p["slide_z1"]]]))


OBST_ARR, OBST_BAR_ARR, RECT_STOW, RECT_DEP = contact_arrays()


def _in_rect(c, x, z, fwd, right, hx, lz0, lz1):
    dx, dz = c[0] - x, c[1] - z
    lx = dx * right[0] + dz * right[1]; lzz = dx * fwd[0] + dz * fwd[1]
    return -hx <= lx <= hx and lz0 <= lzz <= lz1


def _on_bump(x, z):
    for (x0, x1), (z0, z1) in F.BUMPS:
        if x0 <= x <= x1 and z0 <= z <= z1:
            return True
    return False


class MiniSim2Client:
    def __init__(self, params: dict | None = None, seed: int = 0):
        self.sim = MiniSim2(params, seed)
        self.last_cmd = ""

    def reset(self) -> dict:
        return self.sim.reset()

    def act(self, vx, vz, rot, buttons, rff: bool = False) -> dict:
        return self.sim.act(vx, vz, rot, buttons)
