"""Follow a human demo (Ghost) closed-loop: the demo's stick command as feed-forward, a PD correction toward where
the ghost is, and the ghost's buttons copied through.

Feed-forward uses the recorded stick, not recorded velocity / vmax: on the bump ramp the human holds full throttle
while the robot moves slowly, and anything less than full throttle stalls on the crest (calibration), so a
velocity-derived feed-forward would under-drive exactly where it matters.

Tunables live in GHOST_SPEC so CMA-ES (or an RL residual on top) can adjust them; defaults are hand-set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from .ghost import Ghost, wrap_deg

# name, low, high, default — defaults from CMA-ES run c1 (runs/cma/c1, validated in runs/ghost/v1:
# 1004 ± 23 over 32 matches vs 919 ± 27 for the hand-set values)
GHOST_SPEC = [
    ("kp_pos",     0.0, 3.0, 1.826),    # stick units per metre of position error
    ("kd_vel",     0.0, 1.0, 0.296),   # stick units per m/s of velocity error (ghost − robot) / vmax
    ("lookahead",  0.0, 0.6, 0.001),   # s: chase the ghost slightly ahead to cover the one-step actuation lag
    ("kp_yaw",     0.0, 3.0, 0.478),    # rot per 45° of heading error
    ("ff_gain",    0.5, 1.2, 0.884),    # scale on the stick feed-forward
    ("time_shift", -1.5, 1.5, 0.102),   # s: follow the ghost this much earlier (+) or later (−) than the clock
    ("stall_s",    0.3, 2.0, 0.6),    # s of pushing without moving (while the ghost moves) before backing off
    ("backoff_s",  0.2, 1.0, 0.4),    # s of reversing away from the obstacle
    ("deploy_x",   1.5, 4.0, 3.354),    # m: first deploy once x < deploy_x. The spawn is under the trench bar and the
                                      # kicker bar swinging up can catch on it (slide stuck ~0.13 m, kick never locks):
                                      # ~5/88 jams deploying at spawn, 0/56 once clear of the bar; 3.5 costs ~3 pts.
    ("resync_err", 0.8, 99.0, 2.381),   # m: lost this far from the ghost for resync_s → rewind to the nearest path point
    ("resync_s",   0.1, 2.0, 0.5),
    ("catchup",    0.0, 0.5, 0.24),   # s/s: a rewound ghost plays this much faster until the lag is gone
    ("wall_push",  0.0, 0.6, 0.161),    # stick: extra push into the end wall while the ghost slides along it
    ("jam_s",      0.3, 99.0, 99.0),  # s: slide stuck short this long → agitate. OFF: agitation never freed a real
                                      # (kicker-on-trench-bar) jam and it passed the preloaded balls away
    ("agit_s",     0.1, 0.6, 0.3),    # s of AutoPass with Intake released (agitation) to free it; auto period only
    ("rot_ff",     0.0, 1.0, 0.0),    # 1: feed the rotate command to RotateAction too (turret feed-forward, as a human's stick does)
    ("yaw_dt",     0.0, 0.6, 0.2),    # s: heading target this much later than the position target (runs/ghost/yaw1+yaw2:
                                      # 0.2 → 1022 vs 998 for 0, 32 v 32; 0.3 → 1013, 0.4 → 992)
    ("btn_dt",    -0.4, 0.4, 0.0),    # s: buttons taken this much later (+) / earlier (−) than the position target
] + [(f"ts{i}", -1.5, 1.5, 0.0) for i in range(8)]   # piecewise-linear schedule shift (s) at TS_KNOTS: follow the
                                                     # ghost this much earlier (+) / later (−) in each phase of the match
TS_KNOTS = [160.0, 140.0, 130.0, 105.0, 80.0, 55.0, 30.0, 0.0]
_REG_CACHE: dict = {}          # (demo name, horizon) -> the balls the human's mouth swept, per demo fuel frame
VMAX = 2.7


def default_ghost_params() -> dict:
    return {n: d for n, _, _, d in GHOST_SPEC}


@dataclass
class GhostPolicy:
    ghost: Ghost
    p: dict = dc_field(default_factory=default_ghost_params)
    max_err: float = 0.0
    stall_t: float = 0.0
    backoff_left: float = 0.0
    backoff_dir: tuple = (0.0, 0.0)
    unsticks: int = 0
    deploy_waits: int = 0
    lag: float = 0.0
    lost_t: float = 0.0
    resyncs: int = 0
    jam_t: float = 0.0
    agit_left: float = 0.0
    unjams: int = 0
    # route splicing: other demos by name, and [(clock, name), ...] switch requests (clock counts down)
    alts: dict = dc_field(default_factory=dict)
    schedule: list = dc_field(default_factory=list)
    route: str = "base"
    switches: list = dc_field(default_factory=list)

    def reset(self) -> None:
        self.max_err = 0.0
        self.stall_t = 0.0
        self.backoff_left = 0.0
        self.unsticks = 0
        self.deploy_waits = 0
        self.lag = 0.0
        self.lost_t = 0.0
        self.resyncs = 0
        self.jam_t = 0.0
        self.agit_left = 0.0
        self.unjams = 0
        if "base" not in self.alts:
            self.alts = {"base": self.ghost, **self.alts}
        self.ghost = self.alts["base"]
        self.route = "base"
        self.switches = []
        self._sched = sorted(self.schedule, key=lambda e: -e[0])
        self._cr_done, self.cr_log = frozenset(), []

    def act(self, s: dict, offset=(0.0, 0.0), warp: float = 0.0, dyaw: float = 0.0, off_vel=(0.0, 0.0),
            btn: dict | None = None):
        """Track the ghost. The keyword arguments are the residual interface (all zero = plain tracking):
        offset (x, z) metres added to the ghost target; warp seconds the ghost clock runs ahead (+) of the match
        clock; dyaw degrees added to the target heading; off_vel = d(offset)/dt (m/s, feed-forward);
        btn = {button index: bool} overrides of the ghost's buttons."""
        p = self.p
        rob = s["robot"]
        self._maybe_switch(s, warp)
        t = s["t"] - p["time_shift"] - self._ts(s["t"]) - warp + self.lag   # lag > 0: the ghost is rewound
        if p.get("cr_on", 0.0) >= 0.5 and s.get("fuel") is not None:
            self._cr(s, t)
        g = self.ghost.at(t)
        ga = self.ghost.at(t - p["lookahead"])            # clock counts down: "ahead" = smaller t
        if p.get("reg_on", 0.0) >= 0.5:
            ro, rv = self._reg(s, t, g)
            offset = (offset[0] + ro[0], offset[1] + ro[1])
            off_vel = (off_vel[0] + rv[0], off_vel[1] + rv[1])
        if p.get("lbs_on", 0.0) >= 0.5:
            lo, lv, ldy = self._lbs(s, t, g)
            offset = (offset[0] + lo[0], offset[1] + lo[1])
            off_vel = (off_vel[0] + lv[0], off_vel[1] + lv[1])
            dyaw = dyaw + ldy
        tx, tz = ga.x + offset[0], ga.z + offset[1]

        ex, ez = tx - rob["x"], tz - rob["z"]
        evx, evz = g.vx + off_vel[0] - rob.get("vx", 0.0), g.vz + off_vel[1] - rob.get("vz", 0.0)
        ffx, ffz, ffr = (c * p["ff_gain"] for c in g.cmd)
        ffx += off_vel[0] / VMAX
        ffz += off_vel[1] / VMAX
        ux = ffx + p["kp_pos"] * ex + p["kd_vel"] * evx / VMAX
        uz = ffz + p["kp_pos"] * ez + p["kd_vel"] * evz / VMAX
        if g.x > 7.65 and abs(g.vz) > 0.4:                 # sliding along the end wall past the tower: stay flush
            ux += p["wall_push"]
        n = math.hypot(ux, uz)
        if n > 1.0:
            ux, uz = ux / n, uz / n

        # Heading on its own clock: time_shift makes the position track ~0.1 s early, which in a fast turn
        # (~200°/s) puts the intake 10-20° ahead of where the human pointed it (fewer balls through the turn at
        # 137 s: 11 vs 22). yaw_dt s later (clock counts down: +) re-times the heading target and its stick FF.
        yaw_dt = p.get("yaw_dt", 0.0)
        gy = self.ghost.at(t - p["lookahead"] + yaw_dt) if yaw_dt else ga
        if yaw_dt:
            ffr = gy.cmd[2] * p["ff_gain"]
        yaw_err = float(wrap_deg(rob["yaw"] - (gy.yaw + dyaw)))   # rot > 0 decreases yaw
        rot = float(np.clip(ffr + p["kp_yaw"] * yaw_err / 45.0, -1.0, 1.0))

        # Opening replay (ol_until > 0): the start state is identical every match (496 balls within 1.3 cm), so until the
        # clock reaches ol_until the human's own stick is replayed open-loop at the exact clock (no PD, no time shift).
        # Tracking instead lagged 0.6-0.8 m at 158.5 s and cut the corner into the first ball block: the layout was
        # only 60% the human's by 156 s and 25% by 140 s, so the rest of the route swept where the balls no longer were.
        ol = p.get("ol_until", 0.0)
        if ol > 0.0 and s["t"] > ol:
            go = self.ghost.at(s["t"])
            ux, uz = go.cmd[0] * p.get("ol_gain", 1.0), go.cmd[1] * p.get("ol_gain", 1.0)
            rot = float(np.clip(go.cmd[2], -1.0, 1.0))
            n = math.hypot(ux, uz)
            if n > 1.0:
                ux, uz = ux / n, uz / n

        # Unstick: pushing hard but not moving while the ghost is moving (e.g. a bumper corner caught on the tower
        # while sliding along the end wall) → reverse away from the push for backoff_s, then resume tracking.
        dt = 0.099
        speed = math.hypot(rob.get("vx", 0.0), rob.get("vz", 0.0))
        # not while the robot is disabled (the 3 s auto->teleop pause): the demo's stick is often still pushed there, and
        # a "stall" detected in the pause fired a backoff right as teleop began (1085 route: ~1 s and 0.6 m lost)
        pushing = (math.hypot(ux, uz) > 0.5 and speed < 0.25 and math.hypot(g.vx, g.vz) > 0.5
                   and s.get("rs", 0) == 0)
        self.stall_t = self.stall_t + dt if pushing else 0.0
        if self.backoff_left <= 0.0 and self.stall_t >= p["stall_s"]:
            self.backoff_left, self.backoff_dir = p["backoff_s"], (-ux, -uz)
            self.stall_t = 0.0
            self.unsticks += 1
        if self.backoff_left > 0.0:
            self.backoff_left -= dt
            ux, uz = self.backoff_dir

        # Opening intake deploy: the robot spawns inside the trench/bump block (x 3.07..4.22) beside the divider, and
        # a slide deployed there can catch on the structure at ~0.1 m and stay jammed through auto (auto ~0-70
        # instead of ~160; worse under CPU load; holding still or agitating made it worse). So the first deploy is
        # delayed until the robot has driven out of the block into the open neutral zone (x < deploy_x); the first
        # balls are ~3 m further on, and the slide extends fully in ~0.3 s.
        btn_dt = p.get("btn_dt", 0.0)
        buttons = [bool(b) for b in (self.ghost.at(t + btn_dt).buttons if btn_dt else g.buttons)]
        mech = s.get("mech") or {}
        if mech.get("dep", 1) == 0 and rob["x"] > p["deploy_x"] and s["t"] > 150.0:
            buttons[0] = False
            self.deploy_waits += 1
        for k, v in (btn or {}).items():
            buttons[k] = bool(v)
        # Opening slide jam (robot spawns beside the trench/bump divider; ~1 in 10-16 matches the slide catches at
        # ~0.13 m and stays there through auto). Only a slide that has been stuck short for jam_s counts (a normal
        # extension takes ~0.4 s); then AutoPass with Intake released for agit_s makes the slide agitate in/out,
        # which is what freed it in the logs. At most 3 tries, auto period only. AutoPass (not AutoShoot) so any
        # preloaded balls it feeds land in our own zone as stock.
        if s["t"] > 140.0 and self.unjams < 3:
            short = mech.get("dep") == 1 and mech.get("slide", 1.0) < 0.2 and buttons[0]
            self.jam_t = self.jam_t + 0.099 if (short and self.agit_left <= 0.0) else 0.0
            if self.jam_t >= p["jam_s"]:
                self.agit_left, self.jam_t = p["agit_s"], 0.0
                self.unjams += 1
        if self.agit_left > 0.0:
            self.agit_left -= 0.099
            buttons[0], buttons[1], buttons[2] = False, False, True

        # Intake pulsing while shooting (user tip, 2026-09-27): with Intake released, Hightide agitates the slide in/out
        # (GetIntakeTargetMeters), which pushes the hopper's balls to the feeder; pressed, the slide extends and the
        # rollers run full. With few balls aboard, alternating the two feeds better. pulse_held > 0 enables it below
        # that many held balls: pulse_on s pressed, pulse_off s released.
        # ManualShoot instead of AutoShoot inside the zone (manual=1): fixed 2600 rpm / hood 25°, but no turret-ready,
        # bump or trench feed gate (Hightide.Shooter), so it feeds whenever pressed.
        if p.get("manual", 0.0) >= 0.5 and buttons[1] and rob["x"] > 3.9:
            buttons[1], buttons[3] = False, True
        ph = p.get("pulse_held", 0.0)
        if ph > 0.0 and buttons[1] and s.get("held", 999) < ph and s["t"] < 139.0:
            period = p.get("pulse_on", 0.3) + p.get("pulse_off", 0.3)
            self.pulse_t = (getattr(self, "pulse_t", 0.0) + 0.099) % period
            buttons[0] = self.pulse_t < p.get("pulse_on", 0.3)

        # Lost recovery: far from the ghost for a while (wedged, pushed off by balls) → do not cut straight across the
        # field (hub / bumps are in the way); rewind the ghost to the nearest point of its own (obstacle-free) path in
        # the last 8 s and follow it from there. The rewound ghost then plays slightly fast until the lag is gone.
        # While rewound, AutoShoot is only allowed when a shot can still count (hub active / about to be / grace).
        dist = math.hypot(g.x - rob["x"], g.z - rob["z"])
        disabled = s.get("rs", 0) == 1
        self.lost_t = self.lost_t + 0.099 if (dist > p["resync_err"] and not disabled and s["t"] < 139.0) else 0.0
        if self.lost_t >= p["resync_s"] and self.backoff_left <= 0.0:
            best, best_d = 0.0, dist
            for back in np.arange(0.2, 8.01, 0.2):
                f = self.ghost.at(t + back)
                d = math.hypot(f.x - rob["x"], f.z - rob["z"])
                if d < best_d:
                    best, best_d = float(back), d
            if best > 0.0 and best_d < dist - 0.5 and self.lag + best <= 10.0:
                self.lag += best
                self.resyncs += 1
            self.lost_t = 0.0
        elif self.lag > 0.0 and dist < 0.6:
            self.lag = max(0.0, self.lag - p["catchup"] * 0.099)
        elif self.lag < 0.0 and dist < 0.6:                 # a spliced route runs ahead of the clock: let it wait
            self.lag = min(0.0, self.lag + p["catchup"] * 0.099)
        if abs(self.lag) > 0.5 and buttons[1]:
            from .shifts import blue_active
            tt = float(s["t"])
            if not (blue_active(tt, True) or blue_active(tt - 1.5, True) or blue_active(tt + 2.5, True)):
                buttons[1] = False
        # Hub-edge shooting (HubScoring keeps counting for 3 s after the hub turns off; press→score ≈ 1.8 s): the human
        # holds AutoShoot ~1.28 s past each off edge (+44-46 pts each in the 1118 run). tail_x extends that; lead_x
        # starts shooting that many s before an on edge. Only inside the zone (outside, AutoShoot would pass instead).
        tail_x, lead_x = p.get("tail_x", 0.0), p.get("lead_x", 0.0)
        if (tail_x > 0.0 or lead_x > 0.0) and rob["x"] > 3.9:
            tt = float(s["t"])
            if any(e - 1.28 - tail_x < tt <= e for e in (130.0, 80.0)) or any(e < tt <= e + lead_x for e in (105.0, 55.0)):
                buttons[1] = True
        # Collection aids (docs/research-log/STATUS.zh.md 2026-09-27 intake analysis; all off by default). The 4414 mouth captures only
        # balls within ±0.30 m of the centreline with the slide fully out; the front corners (|lat| 0.37-0.55) and
        # sides push. IF: point the intake along the path tangent while collecting (the human crabs ~26° in dead
        # windows). YR: cap the yaw rate while collecting (push rate 0.14 below 1 rad/s, 0.24 above 3). SG: while
        # shooting/passing on the move, keep Intake pressed so the slide stays out (agitation retracts it to
        # 0.19-0.25 m, closing the channel: ~70% of the balls met then are pushed).
        if p.get("if_on", 0.0) >= 0.5 or p.get("yr_max", 0.0) > 0.0:
            a_, b_ = self.ghost.at(t - p["lookahead"] + 0.05), self.ghost.at(t - p["lookahead"] - 0.05)
            vtx, vtz = (b_.x - a_.x) / 0.1 + off_vel[0], (b_.z - a_.z) / 0.1 + off_vel[1]
            tx_, tz_ = getattr(self, "_tan", (vtx, vtz))
            k = min(1.0, 0.099 / p.get("if_tau", 0.2))
            tx_, tz_ = tx_ + k * (vtx - tx_), tz_ + k * (vtz - tz_)
            self._tan = (tx_, tz_)
            vt = math.hypot(tx_, tz_)
            collecting = buttons[0] and s.get("rs", 0) == 0 and mech.get("dep", 1) == 1
            if p.get("if_on", 0.0) >= 0.5:
                psi = math.degrees(math.atan2(tx_, tz_))
                d = float(wrap_deg(psi - gy.yaw))
                elig = (collecting and vt > p.get("if_vmin", 0.8) and abs(d) <= p.get("if_rev", 120.0)
                        and self.backoff_left <= 0.0 and self.lag == 0.0 and g.x <= 7.65
                        and not (2.9 < abs(rob["x"]) < 4.4))            # bump / trench band: cross as the human does
                feeding = (buttons[1] or buttons[2] or buttons[3]) and s.get("held", 0) > 0
                rate = (p.get("if_slew_feed", 40.0) if feeding else p.get("if_slew", 120.0)) * 0.099
                target = float(np.clip(d, -p.get("if_max", 90.0), p.get("if_max", 90.0))) if elig else 0.0
                ifd = getattr(self, "_ifd", 0.0)
                ifd += float(np.clip(target - ifd, -rate, rate))
                self._ifd = ifd
                dpsi = float(wrap_deg(psi - getattr(self, "_psi_prev", psi))) / 0.099
                self._psi_prev = psi
                if abs(ifd) > 1.0:
                    yaw_err = float(wrap_deg(rob["yaw"] - (gy.yaw + dyaw + ifd)))
                    w = 1.0 if elig else 0.0
                    ffr2 = (1.0 - w) * ffr + w * (-math.radians(dpsi) / 3.5)
                    rot = float(np.clip(ffr2 + p["kp_yaw"] * yaw_err / 45.0, -1.0, 1.0))
            yr = p.get("yr_max", 0.0)
            if yr > 0.0 and collecting and vt > 0.8:
                rot = float(np.clip(rot, -yr / 3.5, yr / 3.5))
        if (p.get("sg_on", 0.0) >= 0.5 and mech.get("dep", 1) == 1 and not buttons[0] and (buttons[1] or buttons[2])
                and self.agit_left <= 0.0 and speed > p.get("sg_v", 0.5) and s["t"] < 139.0):
            buttons[0] = True
        # Shooting yaw-rate cap (ys_max deg/s, off by default; docs/research-log/proposals.md 2026-09-27): the turret's aim error
        # grows with the chassis yaw rate and the feed gate holds shots meanwhile (control, AutoShoot, held >= 20: 16.7
        # launches/s below 20 deg/s, 13.7 at 60-120, 4.0 above 120). Cap the rotation while shooting in the zone.
        ys = p.get("ys_max", 0.0)
        if ys > 0.0 and buttons[1] and rob["x"] > 3.9 and s.get("held", 0) >= p.get("ys_held", 20.0) and s["t"] < 139.0:
            lim = math.radians(ys) / 3.5
            rot = float(np.clip(rot, -lim, lim))
        err = math.hypot(g.x - rob["x"], g.z - rob["z"])
        self.max_err = max(self.max_err, math.hypot(tx - rob["x"], tz - rob["z"]))
        info = {"route": self.route, "mode": "backoff" if self.backoff_left > 0.0 else ("resync" if self.lag > 0.0 else "ghost"), "lag": round(self.lag, 2), "err": round(err, 2), "yaw_err": round(yaw_err, 1),
                "ghost_held": g.held, "ghost_blue": g.blue, "win": None}
        return ux, uz, rot, buttons, info

    @property
    def needs_fuel(self) -> bool:
        return self.p.get("lbs_on", 0.0) >= 0.5 or self.p.get("reg_on", 0.0) >= 0.5 or self.p.get("cr_on", 0.0) >= 0.5

    # ------------------------------------------------------------------ whole-window coverage re-planning (cr_on)
    # Once per dead window, on entering the ghost's neutral-zone stretch: re-plan the rest of the stretch against the
    # balls actually there (mosimrl.coverage) and splice it into the ghost, if its estimated intake beats the ghost's own
    # stretch by at least cr_min. Entry/exit points and times, buttons and crossings stay the human's.
    def _cr(self, s: dict, t: float) -> None:
        from . import coverage as cv
        segs = getattr(self, "_cr_segs", None)
        if segs is None:
            segs = self._cr_segs = [cv.segment(self.ghost, *w) for w in cv.DEAD]
        done = getattr(self, "_cr_done", frozenset())
        for i, sg in enumerate(segs):
            if sg is None or i in done:
                continue
            ta, tb = sg
            if tb + 8.0 < t <= ta and s.get("rs", 0) == 0 and self.lag == 0.0:
                self._cr_done = done | {i}
                pl = cv.plan(self.ghost, s["fuel"], int(s.get("held", 0)), t, tb, W=int(self.p.get("cr_w", 1024)))
                use = pl["e_plan"] - pl["e_ghost"] >= self.p.get("cr_min", 0.0)
                self.cr_log = list(getattr(self, "cr_log", [])) + [
                    {"win": i, "t": round(t, 2), "e_plan": round(pl["e_plan"], 1), "e_ghost": round(pl["e_ghost"], 1),
                     "n": pl["n"], "used": bool(use)}]
                if use:
                    self.ghost = cv.splice(self.ghost, pl)
                    self.route = self.route + "+cr"

    # ------------------------------------------------------------------ pile registration (reg_on)
    # The human's route carries intent: which pile to take next and from which side (user: "我都是走到球的左下角开始吸
    # 球的"). The piles themselves end up elsewhere every match (chaos), so a fixed path sweeps air. For the next reg_h s
    # of the route, take the balls the human's mouth actually swept in the demo, find where that set of balls lies now
    # (best translation within ±reg_dmax), and follow the human's path translated onto it: same entry side and sweep
    # direction, relative to where the pile really is.
    def _human_targets(self):
        key = (self.ghost.name, self.p.get("reg_h", 2.0))
        if key in _REG_CACHE:
            return _REG_CACHE[key]
        rows = [r for r in getattr(self.ghost, "rows", []) if "fuel" in r]
        H = self.p.get("reg_h", 2.0)
        ks = np.arange(int(round(H / 0.1)) + 1)
        taus, sets = [], []
        for r in rows:
            B = np.asarray(r["fuel"], np.float64).reshape(-1, 3)
            B = B[B[:, 2] < 0.25][:, :2]
            fr = [self.ghost.at(r["t"] - k * 0.1) for k in ks]
            px = np.array([f.x for f in fr]); pz = np.array([f.z for f in fr]); yw = np.radians([f.yaw for f in fr])
            relx = B[None, :, 0] - px[:, None]; relz = B[None, :, 1] - pz[:, None]
            fwd = relx * np.sin(yw)[:, None] + relz * np.cos(yw)[:, None]
            lat = np.abs(relx * np.cos(yw)[:, None] - relz * np.sin(yw)[:, None])
            swept = ((fwd >= 0.30) & (fwd <= 0.75) & (lat <= 0.35)).any(0)
            taus.append(r["t"]); sets.append(B[swept])
        _REG_CACHE[key] = (np.array(taus), sets)
        return _REG_CACHE[key]

    def _reg(self, s: dict, t: float, g):
        p = self.p
        st = getattr(self, "_reg_st", None)
        if st is None:
            st = self._reg_st = {"dx": 0.0, "dz": 0.0, "tx": 0.0, "tz": 0.0, "n": 0}
        rob = s["robot"]
        mech = s.get("mech") or {}
        fuel = s.get("fuel")
        px0, pz0 = st["dx"], st["dz"]
        elig = (fuel is not None and g.buttons[0] and s.get("rs", 0) == 0 and mech.get("dep", 1) == 1
                and not (2.9 < abs(rob["x"]) < 4.4) and self.lag == 0.0 and self.backoff_left <= 0.0)
        st["n"] += 1
        if not elig:
            st["tx"], st["tz"] = 0.0, 0.0
        elif st["n"] % 3 == 1:
            taus, sets = self._human_targets()
            i = int(np.argmin(np.abs(taus - t))) if len(taus) else -1
            T = sets[i] if i >= 0 and abs(taus[i] - t) < 0.35 else np.zeros((0, 2))
            if len(T) < p.get("reg_min", 5):
                st["tx"], st["tz"] = 0.0, 0.0
            else:
                from scipy.spatial import cKDTree
                a = np.asarray(fuel, np.float64).reshape(-1, 3)
                a = a[(a[:, 2] < 0.25) & (np.hypot(a[:, 0] - rob["x"], a[:, 1] - rob["z"]) < 4.5)][:, :2]
                if len(a) < 3:
                    st["tx"], st["tz"] = 0.0, 0.0
                else:
                    tree = cKDTree(a)
                    dmax = p.get("reg_dmax", 0.8)
                    grid = np.round(np.arange(-dmax, dmax + 1e-6, 0.1), 3)
                    DX, DZ = np.meshgrid(grid, grid, indexing="ij")
                    sh = np.stack([DX.ravel(), DZ.ravel()], 1)                      # (S, 2)
                    pts = (T[None, :, :] + sh[:, None, :]).reshape(-1, 2)
                    d, _ = tree.query(pts)
                    sig = 0.08
                    score = np.exp(-(d.reshape(len(sh), len(T)) ** 2) / (2 * sig * sig)).sum(1)
                    score -= p.get("reg_kappa", 1.0) * np.hypot(sh[:, 0], sh[:, 1])   # prefer small shifts
                    # keep clear of structures / walls along the next seconds of the shifted route
                    from .residual import structure_dist
                    fr = [self.ghost.at(t - k) for k in (0.5, 1.0, 2.0)]
                    order = np.argsort(-score)
                    cur = int(np.argmin(np.hypot(sh[:, 0] - st["tx"], sh[:, 1] - st["tz"])))
                    pick = None
                    for j in order[:40]:
                        okk = True
                        for f in fr:
                            x2, z2 = f.x + sh[j, 0], f.z + sh[j, 1]
                            if (structure_dist(x2, z2) < min(structure_dist(f.x, f.z), 1.2) - 0.05 or
                                    (abs(z2) > 3.6 and abs(z2) > abs(f.z)) or (abs(x2) > 7.75 and abs(x2) > abs(f.x))):
                                okk = False
                                break
                        if okk:
                            pick = j
                            break
                    matched = score[pick] + p.get("reg_kappa", 1.0) * math.hypot(*sh[pick]) if pick is not None else 0.0
                    if pick is None or matched < p.get("reg_frac", 0.4) * len(T):
                        st["tx"], st["tz"] = 0.0, 0.0
                    elif score[pick] > score[cur] + p.get("reg_hyst", 1.0):
                        st["tx"], st["tz"] = float(sh[pick, 0]), float(sh[pick, 1])
        r = p.get("reg_rate", 0.05)
        st["dx"] += float(np.clip(st["tx"] - st["dx"], -r, r))
        st["dz"] += float(np.clip(st["tz"] - st["dz"], -r, r))
        return (st["dx"], st["dz"]), ((st["dx"] - px0) / 0.099, (st["dz"] - pz0) / 0.099)

    # ------------------------------------------------------------------ ball-aware local steering (lbs_on)
    # The route is the human's, but ball piles never end up where they were in the human's match: two bot matches with
    # the same policy match only ~38% of floor balls (within 5 cm) 10 s after the first sweep, so a fixed path sweeps
    # air (user, 2026-09-27: "你在扫空气"). Each decision this looks at the floor balls the next ~1 s of the route would
    # meet and picks a lateral shift of the route (±lbs_dmax) and a heading tweak (±lbs_psi) that put the most balls
    # through the mouth and the fewest into the bumper corners / sides (4414 geometry: mouth |lat| <= 0.30 m captures
    # ~0.95-0.98; 0.30-0.555 is bumper corner; sides/back push). That makes it eat piles from their edges instead of
    # ramming them, the way the human starts a block at its corner. Macro timing, buttons and crossings stay the human's.
    _PCAP_X = np.array([0.0, 0.2, 0.3, 0.35, 0.40, 0.45, 0.555])
    _PCAP_Y = np.array([0.98, 0.98, 0.95, 0.80, 0.60, 0.40, 0.15])

    def _lbs(self, s: dict, t: float, g):
        p = self.p
        st = getattr(self, "_lbs_st", None)
        if st is None:
            st = self._lbs_st = {"d": 0.0, "psi": 0.0}
        rob = s["robot"]
        mech = s.get("mech") or {}
        fuel = s.get("fuel")
        d_prev = st["d"]
        # path normal at the current ghost point (left of travel)
        g2 = self.ghost.at(t - 0.1)
        vx_, vz_ = g2.x - g.x, g2.z - g.z
        vn = math.hypot(vx_, vz_)
        elig = (fuel is not None and g.buttons[0] and s.get("rs", 0) == 0 and mech.get("dep", 1) == 1
                and not (2.9 < abs(rob["x"]) < 4.4) and self.lag == 0.0 and self.backoff_left <= 0.0
                and s["t"] < 159.0 and vn > 0.02)
        if elig:
            nx, nz = -vz_ / vn, vx_ / vn
            a = np.asarray(fuel, np.float64).reshape(-1, 3)
            a = a[(a[:, 2] < 0.25) & (np.hypot(a[:, 0] - rob["x"], a[:, 1] - rob["z"]) < 3.5)][:, :2]
            dmax, psim = p.get("lbs_dmax", 0.8), p.get("lbs_psi", 30.0)
            D = np.round(np.arange(-dmax, dmax + 1e-6, 0.1), 3)
            PSI = np.array([-psim, -psim / 2, 0.0, psim / 2, psim]) if psim > 0 else np.array([0.0])
            K = int(round(p.get("lbs_h", 1.0) / 0.1))
            ks = np.arange(K + 1)
            gp = [self.ghost.at(t - k * 0.1) for k in ks]
            gyw = np.radians([self.ghost.at(t - k * 0.1 + p.get("yaw_dt", 0.0)).yaw for k in ks])
            gx = np.array([f.x for f in gp]); gz = np.array([f.z for f in gp])
            w = np.minimum(1.0, ks * 0.1 / 0.4)
            # candidate positions (C_d, K): shifted route, blended in from the robot's actual position
            px = w[None, :] * (gx[None, :] + D[:, None] * nx) + (1 - w[None, :]) * (rob["x"] + gx - gx[0])[None, :]
            pz = w[None, :] * (gz[None, :] + D[:, None] * nz) + (1 - w[None, :]) * (rob["z"] + gz - gz[0])[None, :]
            ok = np.ones(len(D), bool)
            from .residual import structure_dist
            for i, dd in enumerate(D):
                if dd == 0.0:
                    continue
                for k in (K // 2, K):
                    if structure_dist(px[i, k], pz[i, k]) < min(structure_dist(gx[k], gz[k]), 1.2) - 0.05 or \
                            (abs(pz[i, k]) > 3.6 and abs(pz[i, k]) > abs(gz[k])) or (abs(px[i, k]) > 7.75 and abs(px[i, k]) > abs(gx[k])):
                        ok[i] = False
                        break
            held = s.get("held", 0)
            hf = 1.0 if held < 80 else 0.9 if held < 95 else 0.63 if held < 104 else 0.3
            best, bestJ, curJ = (0.0, 0.0), -1e9, None
            if len(a):
                for psi in PSI:
                    yaw = gyw[None, :] + math.radians(psi)                    # (1, K)
                    hx, hz = np.sin(yaw), np.cos(yaw)                         # heading (x, z)
                    rx, rz = np.cos(yaw), -np.sin(yaw)                        # lateral axis
                    relx = a[None, None, :, 0] - px[:, :, None]               # (C, K, N)
                    relz = a[None, None, :, 1] - pz[:, :, None]
                    fwd = relx * hx[:, :, None] + relz * hz[:, :, None]
                    lat = np.abs(relx * rx[:, :, None] + relz * rz[:, :, None])
                    inside = (fwd >= -0.45) & (fwd <= 0.62) & (lat <= 0.555)
                    hit = inside.any(1)                                       # (C, N)
                    first = inside.argmax(1)
                    fe = np.take_along_axis(fwd, first[:, None, :], 1)[:, 0, :]
                    le = np.take_along_axis(lat, first[:, None, :], 1)[:, 0, :]
                    pc = np.where(fe >= 0.35, np.interp(le, self._PCAP_X, self._PCAP_Y), 0.0) * hf
                    cap = (pc * hit).sum(1)
                    push = ((1.0 - pc) * hit).sum(1)
                    J = (cap - p.get("lbs_mu", 0.5) * push - p.get("lbs_kappa", 2.0) * np.abs(D)
                         - p.get("lbs_rho", 4.0) * np.abs(D - d_prev) - p.get("lbs_zeta", 1.0) * abs(psi) / 30.0)
                    J = np.where(ok, J, -1e9)
                    i = int(np.argmax(J))
                    if J[i] > bestJ:
                        bestJ, best = float(J[i]), (float(D[i]), float(psi))
                    if abs(psi - st["psi"]) < 1e-6 or (curJ is None and psi == 0.0):
                        ic = int(np.argmin(np.abs(D - d_prev)))
                        curJ = float(J[ic])
            if curJ is not None and bestJ < curJ + p.get("lbs_hyst", 0.5):
                best = (d_prev, st["psi"])
            tgt_d, tgt_psi = best
        else:
            tgt_d, tgt_psi = 0.0, 0.0
            nx, nz = (-vz_ / vn, vx_ / vn) if vn > 0.02 else (0.0, 0.0)
        st["d"] += float(np.clip(tgt_d - st["d"], -p.get("lbs_rate", 0.05), p.get("lbs_rate", 0.05)))
        st["psi"] += float(np.clip(tgt_psi - st["psi"], -6.0, 6.0))
        dv = (st["d"] - d_prev) / 0.099
        return (st["d"] * nx, st["d"] * nz), (dv * nx, dv * nz), st["psi"]

    def _maybe_switch(self, s: dict, warp: float) -> None:
        """Pending switch request whose clock has come: move to the requested demo at the point of its path nearest
        the robot within ±2 s of the current ghost clock (so the splice needs no detour). Retried for up to 3 s,
        then dropped."""
        sched = getattr(self, "_sched", None)
        if not sched or s["t"] > sched[0][0] or s.get("rs", 0) == 1:
            return
        t_req, name = sched[0]
        if name == self.route or name not in self.alts:
            sched.pop(0)
            return
        rob = s["robot"]
        tau = s["t"] - self.p["time_shift"] - self._ts(s["t"]) - warp + self.lag
        g2 = self.alts[name]
        best, best_d = None, 1e9
        for dt in np.arange(-2.0, 2.01, 0.1):
            f = g2.at(tau + dt)
            d = math.hypot(f.x - rob["x"], f.z - rob["z"])
            if d < best_d:
                best, best_d = float(dt), d
        if best_d < 0.6:
            self.ghost = g2
            self.lag += best
            self.route = name
            self.switches.append((round(float(s["t"]), 1), name, round(best, 2), round(best_d, 2)))
            sched.pop(0)
        elif t_req - s["t"] > 3.0:
            self.switches.append((round(float(s["t"]), 1), name + "-skipped", None, round(best_d, 2)))
            sched.pop(0)

    @property
    def rot_ff(self) -> bool:
        return self.p.get("rot_ff", 0.0) >= 0.5

    def _ts(self, t: float) -> float:
        """Schedule shift at clock t (piecewise-linear over TS_KNOTS; all zero by default)."""
        v = [self.p.get(f"ts{i}", 0.0) for i in range(len(TS_KNOTS))]
        if not any(v):
            return 0.0
        return float(np.interp(-t, [-k for k in TS_KNOTS], v))

