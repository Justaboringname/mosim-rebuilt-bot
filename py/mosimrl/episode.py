"""Run one match with a policy through a MoSimClient and report the score."""

from __future__ import annotations

import time

import numpy as np

from .client import MoSimClient


class MatchClock:
    """Physics-rate match clock.

    The Bridge's `t` is BaseTimerManager.Timer, which ticks in Update (once per rendered frame), while the Bridge
    samples it from FixedUpdate. When the machine stalls for a moment, Unity runs many physics steps inside one frame:
    several decisions in a row then report the same `t` while the robot keeps moving, the ghost target freezes and
    the stick feed-forward drives the robot past it (rf batch, 2026-09-26: all 8 instances overshot to x≈-4.2 at
    t≈75 in the same stalled second and got stuck, -40 each). Between Timer updates, extrapolate with `gameTime`
    (Time.time inside FixedUpdate = fixedTime, which advances per physics step); resync whenever the Timer moves.
    """

    def __init__(self):
        self.t_last = self.g_last = None

    def __call__(self, s: dict) -> float:
        t, g = s.get("t"), s.get("gameTime")
        running = s.get("rs", 0) == 0 and s.get("gs", -1) in (0, 1, 2)
        if t is None or g is None or not running or self.t_last is None or t != self.t_last:
            self.t_last, self.g_last = t, g
            return t
        return max(0.0, t - (g - self.g_last))


class BallTracker:
    """Per-ball events from the Bridge's stable ball ids (`fid`; balls inside the hopper are left out of `fuel`).

    intake: a ball that was listed at the previous decision and is not now (it entered the hopper), with the last
            position where it lay on the floor (y < 0.2) and when it was there, plus the robot pose.
    launch: a ball that reappears after having been in the hopper (shot, pass, or spat out), with where it appeared.
    Rows are [t, id, x, z, t_floor, robot_x, robot_z, robot_yaw] and [t, id, x, z, y].
    """

    N = 1024

    def __init__(self):
        self.present = np.zeros(self.N, bool)
        self.seen = np.zeros(self.N, bool)
        self.fx = np.full(self.N, np.nan); self.fz = np.full(self.N, np.nan); self.ft = np.full(self.N, np.nan)
        self.intakes, self.launches = [], []

    def update(self, s: dict) -> None:
        fid, fuel = s.get("fid"), s.get("fuel")
        if fid is None or fuel is None or len(fid) != len(fuel):
            return
        ids = np.asarray(fid, int)
        a = np.asarray(fuel, float).reshape(-1, 3)
        now = np.zeros(self.N, bool); now[ids] = True
        t, r = s["t"], s.get("robot") or {}
        for i in np.nonzero(self.present & ~now)[0]:
            self.intakes.append([round(t, 3), int(i), round(self.fx[i], 3), round(self.fz[i], 3), round(self.ft[i], 3),
                                 round(r.get("x", 0.0), 3), round(r.get("z", 0.0), 3), round(r.get("yaw", 0.0), 1)])
        back = ~self.present & now & self.seen
        if back.any():
            pos = {int(i): k for k, i in enumerate(ids)}
            for i in np.nonzero(back)[0]:
                p = a[pos[int(i)]]
                self.launches.append([round(t, 3), int(i), round(p[0], 3), round(p[1], 3), round(p[2], 3)])
        fl = a[:, 2] < 0.2
        self.fx[ids[fl]] = a[fl, 0]; self.fz[ids[fl]] = a[fl, 1]; self.ft[ids[fl]] = t
        self.present = now
        self.seen |= now


class DenseFuel:
    """Every ball's position at every decision, indexed by its stable id (NaN while it is in the hopper), as a
    compact array for fitting MiniSim physics (rolling, pushing, exits): positions (T, N, 3) in cm as int16."""

    def __init__(self, n: int = 504):
        self.n, self.rows, self.t, self.gt = n, [], [], []

    def update(self, s: dict) -> None:
        fid, fuel = s.get("fid"), s.get("fuel")
        if fid is None or fuel is None or len(fid) != len(fuel):
            return
        a = np.full((self.n, 3), -32768, np.int16)
        ids = np.asarray(fid, int)
        ok = ids < self.n
        a[ids[ok]] = np.clip(np.round(np.asarray(fuel, float)[ok] * 100.0), -32767, 32767).astype(np.int16)
        self.rows.append(a); self.t.append(s["t"]); self.gt.append(s.get("gameTime", np.nan))

    def arrays(self) -> dict:
        return {"pos_cm": np.stack(self.rows) if self.rows else np.zeros((0, self.n, 3), np.int16),
                "t": np.asarray(self.t, np.float32), "gameTime": np.asarray(self.gt, np.float64)}


def run_episode(client: MoSimClient, policy, trace: bool = False, max_steps: int = 5000, fuel_every: int = 5,
                dense: bool = False) -> dict:
    policy.reset()
    wall0 = time.time()
    clock = MatchClock()
    balls = BallTracker()
    dfuel = DenseFuel() if dense else None
    s = client.reset()
    s["t_raw"] = s.get("t"); s["t"] = clock(s)
    first = s
    steps, log = 0, []
    while not s.get("done") and steps < max_steps:
        if trace:
            balls.update(s)
        if dfuel is not None:
            dfuel.update(s)
        vx, vz, rot, buttons, info = policy.act(s)
        if trace:
            r = s["robot"]
            log.append({"t": s["t"], "t_raw": s.get("t_raw"), "x": r["x"], "z": r["z"], "yaw": r["yaw"], "held": s.get("held"),
                        "blue": s["blue"], "bIn": s.get("bIn"), "mech": s.get("mech"), "frame": s.get("frame"), "rotIn": s.get("rotIn"), "vx": vx, "vz": vz, "rot": rot, "b": [int(b) for b in buttons],
                        "mode": info.get("mode"), "win": info.get("win"),
                        "gate": s.get("gate"), "rvx": r.get("vx"), "rvz": r.get("vz"), "wy": r.get("wy"),
                        "pitch": r.get("pitch"), "roll": r.get("roll"), "y": r.get("y"), "hub": s.get("hub"),
                        "gameTime": s.get("gameTime"),
                        **({"fuel": [[round(b[0], 2), round(b[1], 2), round(b[2], 2)] for b in s.get("fuel") or []],
                            "fid": s.get("fid")}
                           if fuel_every and steps % fuel_every == 0 else {}),
                        **{k: info[k] for k in ("err", "yaw_err", "ghost_blue", "ghost_held", "lag", "phase", "lat", "warp", "u") if k in info}})
        s = client.act(vx, vz, rot, buttons, rff=bool(getattr(policy, "rot_ff", False)))
        s["t_raw"] = s.get("t"); s["t"] = clock(s)
        steps += 1
    return {"score": s.get("blue", 0), "auto": s.get("blueAuto", 0), "steps": steps,
            "wall_s": round(time.time() - wall0, 1), "done": bool(s.get("done")), "trace": log,
            "intakes": balls.intakes, "launches": balls.launches, "first_state": first,
            **({"dense": dfuel.arrays()} if dfuel is not None else {})}
