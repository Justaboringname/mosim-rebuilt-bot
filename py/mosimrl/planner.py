"""Lookahead planning with the real game as the model.

The main match only ever snapshots (Snapshot.cs); planner instances restore the snapshot and roll out candidate
continuations to the end of the current window, and the main match executes the best one for a while. Because the
rollouts run in MoSim itself there is no simulator gap; the price is noise (a restored match does not repeat
exactly: ball piles are chaotic), which the candidate scoring has to average over.

    python -m mosimrl.planner gate4 --t0 127 --until 105 --reps 8
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .episode import MatchClock
from .gamectl import Game
from .ghost import Ghost
from .ghost_policy import default_ghost_params
from .run_ghost import make_policy

DEMO = Path(__file__).resolve().parents[2] / "run/demos/demo-20260926-015440-m1.jsonl"
DT = 0.099

# blue-owned balls: in the hopper, or loose on the floor in the blue alliance zone (flow.py's STOCK, x > 4.25)
STOCK_X = 4.25


def owned(s: dict) -> dict:
    a = np.asarray(s.get("fuel") or [], float).reshape(-1, 3)
    x, z, y = (a[:, 0], a[:, 1], a[:, 2]) if len(a) else (np.zeros(0),) * 3
    floor = y < 0.25
    stock = int((floor & (x > STOCK_X)).sum())
    air_in = int((~floor & (x > STOCK_X)).sum())            # passes about to land in the zone
    hubair = int(((y > 0.8) & (np.abs(x - 3.655) < 2.6) & (np.abs(z) < 2.6)).sum())   # shots in flight
    return {"held": int(s.get("held", 0)), "stock": stock, "air": air_in, "blue": int(s.get("blue", 0)),
            "owned": int(s.get("held", 0)) + stock + air_in, "hubair": hubair}


# ---------------------------------------------------------------------------------------------- candidates

def clone(pol):
    """Independent copy of a tracking policy (the ghost path itself is shared, it is never modified)."""
    p = copy.copy(pol)
    for k in ("switches", "_sched"):
        if hasattr(pol, k):
            setattr(p, k, list(getattr(pol, k)))
    if hasattr(pol, "p"):
        p.p = dict(pol.p)
    if hasattr(pol, "tracker"):                         # ResidualAgent wraps a GhostPolicy
        p.tracker = clone(pol.tracker)
    return p


class Candidate:
    """A continuation of the base tracker: `kind` and its parameters; act() is the tracker's act with the
    candidate's modification for the first `dur` seconds after it starts, then plain tracking."""

    def __init__(self, base, kind: str = "ghost", dur: float = 1e9, **kw):
        self.base, self.kind, self.dur, self.kw = base, kind, dur, kw
        self.t_start = None

    def describe(self) -> str:
        return self.kind + ("" if not self.kw else " " + json.dumps(self.kw)) + ("" if self.dur > 1e8 else f" for {self.dur}s")

    def act(self, s: dict):
        if self.t_start is None:
            self.t_start = s["t"]
        on = self.t_start - s["t"] < self.dur
        if not on or self.kind == "ghost":
            return self.base.act(s)
        if self.kind == "offset":
            return self.base.act(s, offset=(self.kw.get("dx", 0.0), self.kw.get("dz", 0.0)))
        if self.kind == "warp":
            return self.base.act(s, warp=self.kw["w"])
        if self.kind == "sweep":
            return self._sweep(s)
        if self.kind == "route":
            if not getattr(self, "switched", False):
                self.switched = True
                switch_route(self.base, ROUTES[self.kw["route"]], self.kw["route"], s, self.kw.get("tol", 1.0))
            return self.base.act(s)
        raise ValueError(self.kind)

    def _sweep(self, s: dict):
        """Intake-first straight sweep of length L along heading h (yaw convention: 0 = +z, 90° = +x), keeping the
        ghost's AutoPass; the tracker runs underneath so it notices it is off-path and rewinds to rejoin afterwards."""
        vx, vz, rot, b, info = self.base.act(s)
        rob = s["robot"]
        if getattr(self, "origin", None) is None:
            self.origin, self.stall = (rob["x"], rob["z"]), 0.0
        h, L = self.kw["h"], self.kw["L"]
        gone = math.hypot(rob["x"] - self.origin[0], rob["z"] - self.origin[1])
        speed = math.hypot(rob.get("vx", 0.0), rob.get("vz", 0.0))
        self.stall = self.stall + DT if speed < 0.2 and self.t_start - s["t"] > 0.5 else 0.0
        if gone >= L or self.stall > 0.5:
            self.dur = 0.0                                   # done (or stuck): back to the tracker for good
            return vx, vz, rot, b, info
        sp = self.kw.get("speed", 1.0)
        yaw_err = float((rob["yaw"] - math.degrees(h) + 180.0) % 360.0 - 180.0)   # rot > 0 decreases yaw
        rot = float(np.clip(1.2 * yaw_err / 45.0, -1.0, 1.0))
        b = list(b); b[0] = True
        if not self.kw.get("shoot"):
            b[1] = False
        return math.sin(h) * sp, math.cos(h) * sp, rot, b, {**info, "mode": "sweep"}


# route library: every full human demo and its z-mirror, switchable where two paths meet
ROUTES: dict = {}


def load_routes() -> None:
    from .ghost import mirror_ghost
    from .run_ghost import DEMO_KEYS
    for key, path in DEMO_KEYS.items():
        g = Ghost.from_file(path)
        ROUTES[key] = g
        ROUTES[key + "m"] = mirror_ghost(g)


def route_join(tracker, g2, s: dict, dt_range: float = 3.0):
    """(clock shift, distance) of the point of route g2 nearest the robot within ±dt_range of the tracker's clock."""
    rob = s["robot"]
    tau = s["t"] - tracker.p["time_shift"] - tracker._ts(s["t"]) + tracker.lag
    best, best_d = 0.0, 1e9
    for dt in np.arange(-dt_range, dt_range + 1e-6, 0.1):
        f = g2.at(tau + dt)
        d = math.hypot(f.x - rob["x"], f.z - rob["z"])
        if d < best_d:
            best, best_d = float(dt), d
    return best, best_d


def switch_route(tracker, g2, name: str, s: dict, tol: float = 1.0) -> bool:
    shift, d = route_join(tracker, g2, s)
    if d > tol:
        return False
    tracker.ghost, tracker.route = g2, name
    tracker.lag += shift
    return True


# neutral-zone sweeps (seeker.best_sweep generalised to the K best distinct headings)
SWEEP_HEADINGS = np.radians(np.arange(0, 360, 15.0))
SWEEP_LENGTHS = np.array([1.0, 1.75, 2.5, 3.5])
X_MIN, X_MAX, Z_ABS = -2.45, 2.6, 3.6
SWATH, SWEEP_SPEED, TURN_RATE = 0.33, 1.6, 4.0


# blue alliance zone (active windows: harvest the stock while shooting; the hub band x < 4.4 is avoided)
ZX_MIN, ZX_MAX, ZZ_ABS, ZONE_SPEED = 4.45, 7.9, 3.75, 1.5


def top_sweeps(s: dict, k: int = 6, min_sep_deg: float = 40.0, zone: bool = False) -> list[dict]:
    rob = s["robot"]
    px, pz, yaw = rob["x"], rob["z"], math.radians(rob["yaw"])
    a = np.asarray(s.get("fuel") or [], float).reshape(-1, 3)
    xmin, xmax, zabs, speed = (ZX_MIN, ZX_MAX, ZZ_ABS, ZONE_SPEED) if zone else (X_MIN, X_MAX, Z_ABS, SWEEP_SPEED)
    if not len(a) or not (xmin - 0.5 <= px <= xmax + 0.5):
        return []
    a = a[a[:, 2] < 0.25]
    dx, dz = a[:, 0] - px, a[:, 1] - pz
    rows = []
    for h in SWEEP_HEADINGS:
        fx, fz = math.sin(h), math.cos(h)
        along = dx * fx + dz * fz
        inlane = np.abs(dx * fz - dz * fx) < SWATH
        dturn = abs(math.atan2(math.sin(h - yaw), math.cos(h - yaw)))
        for L in SWEEP_LENGTHS:
            ex, ez = px + fx * L, pz + fz * L
            if not (xmin <= ex <= xmax and abs(ez) <= zabs):
                continue
            n = float((inlane & (along > 0.15) & (along < L + 0.35)).sum())
            rows.append((n / (L / speed + dturn / TURN_RATE + 0.3), float(h), float(L), n))
    rows.sort(reverse=True)
    out = []
    if k <= 0:
        return out
    for u, h, L, n in rows:
        if n < 3:
            break
        if all(abs(math.degrees(math.atan2(math.sin(h - o["h"]), math.cos(h - o["h"])))) >= min_sep_deg for o in out):
            out.append({"h": round(h, 4), "L": L, "n": n})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------------------------- instances

class Pool:
    def __init__(self, ports: list[int], steps_per_frame: int = 2, extra_args: list[str] | None = None):
        self.games = [Game(port=p, time_scale=1.0, steps_per_frame=steps_per_frame, extra_args=extra_args) for p in ports]
        with ThreadPoolExecutor(len(self.games)) as ex:
            list(ex.map(lambda g: g.attach_or_launch(), self.games))
        with ThreadPoolExecutor(len(self.games)) as ex:            # every planner sits inside a match
            list(ex.map(lambda g: g.client.reset(), self.games))
        self.free = queue.Queue()
        for g in self.games:
            self.free.put(g)

    def close(self, quit_game: bool = True) -> None:
        for g in self.games:
            g.close(quit_game=quit_game)

    def rollout(self, blob: str, pol, until_t: float) -> dict:
        """One rollout on the next free planner. A planner that dies (socket reset / hang) is relaunched, put back in
        a match, and the rollout is redone from a fresh copy of the policy (the old copy may have advanced)."""
        import socket
        from .client import BridgeError
        pol0 = clone(pol)
        for attempt in range(3):
            g = self.free.get()
            try:
                return rollout(g, blob, clone(pol0), until_t)
            except (socket.timeout, TimeoutError, ConnectionError, OSError, BridgeError) as e:
                if attempt == 2:
                    raise
                try:
                    g.recover(f"rollout: {type(e).__name__}: {e}")
                    g.client.reset()
                except Exception as e2:
                    print(f"[planner] could not recover port {g.port}: {e2}", flush=True)
            finally:
                self.free.put(g)

    def map(self, blob: str, jobs: list[tuple], until_t: float) -> list[dict]:
        """jobs: [(tag, policy)] -> [{"tag", **result}] (each job on the next free planner)."""
        with ThreadPoolExecutor(len(self.games)) as ex:
            futs = [(tag, ex.submit(self.rollout, blob, pol, until_t)) for tag, pol in jobs]
            return [{"tag": tag, **f.result()} for tag, f in futs]


def rollout(g: Game, blob: str, pol, until_t: float) -> dict:
    w0 = time.time()
    s = g.client.restore(blob)
    clock = MatchClock()
    s["t"] = clock(s)
    t0, n = s["t"], 0
    needs_fuel = bool(getattr(pol, "needs_fuel", False))
    base = getattr(pol, "base", pol)
    u0 = getattr(base, "unsticks", 0)
    while s["t"] > until_t and not s.get("done"):
        vx, vz, rot, b, _ = pol.act(s)
        last = s["t"] - DT <= until_t + 1e-3
        s = g.client.act(vx, vz, rot, b, lite=(not last) and not needs_fuel)
        s["t"] = clock(s)
        n += 1
    if "fuel" not in s:                     # ended early (done) in lite mode: nothing more to read without acting
        return {"t0": t0, "t": s["t"], "steps": n, "wall": time.time() - w0, "blue": s.get("blue", 0),
                "held": s.get("held", 0), "owned": None}
    return {"t0": t0, "t": s["t"], "steps": n, "wall": time.time() - w0, **owned(s),
            "x": s["robot"]["x"], "z": s["robot"]["z"], "unsticks": getattr(base, "unsticks", 0) - u0,
            "cr": list(getattr(base, "cr_log", []))}


# ---------------------------------------------------------------------------------------------- gate 4

def play_to(g: Game, pol, t_stop: float, clock: MatchClock, s: dict) -> dict:
    # teleop targets wait out the auto->teleop pause (t stays 140 for 3 s); auto targets stop in auto
    while s["t"] > t_stop or (t_stop < 140.0 and s.get("gs") not in (1, 2)):
        vx, vz, rot, b, _ = pol.act(s)
        s = g.client.act(vx, vz, rot, b)
        s["t"] = clock(s)
    return s


def gate4(args) -> None:
    ports = list(range(args.port0, args.port0 + args.planners + 1))
    main_g = Game(port=ports[0], time_scale=1.0, steps_per_frame=2)
    main_g.attach_or_launch()
    pool = Pool(ports[1:])
    out = Path(__file__).resolve().parents[2] / "runs/planner"
    out.mkdir(parents=True, exist_ok=True)
    try:
        ghost = Ghost.from_file(str(DEMO))
        pol = make_policy(ghost, default_ghost_params(), {})
        pol.reset()
        clock = MatchClock()
        s = main_g.client.reset(); s["t"] = clock(s)
        results = []
        for t0 in args.t0:
            s = play_to(main_g, pol, t0, clock, s)
            snap = main_g.client.snapshot()
            print(f"snapshot t={snap['t']}  {snap['ms']} ms  {snap['summary']}", flush=True)
            cands = [("ghost", {}), ("offset", {"dz": 0.4}), ("offset", {"dz": -0.4}), ("warp", {"w": 1.0}), ("warp", {"w": -1.0})]
            jobs = []
            for kind, kw in cands:
                for r in range(args.reps if kind != "ghost" else 2 * args.reps):
                    jobs.append((f"{kind} {json.dumps(kw)}", Candidate(clone(pol), kind, **kw)))
            w0 = time.time()
            res = pool.map(snap["blob"], jobs, args.until)
            wall = time.time() - w0
            print(f"t0 {t0}: {len(jobs)} rollouts to t={args.until} in {wall:.1f}s "
                  f"(mean rollout {np.mean([r['wall'] for r in res]):.1f}s, {np.mean([r['steps'] for r in res]):.0f} steps)")
            for tag in dict.fromkeys(r["tag"] for r in res):
                rs = [r for r in res if r["tag"] == tag]
                ow = np.array([r["owned"] for r in rs], float)
                print(f"  {tag:24s} n={len(rs):2d} owned {ow.mean():6.1f} ± {ow.std(ddof=1):5.1f} (se {ow.std(ddof=1) / math.sqrt(len(ow)):4.1f})"
                      f"  held {np.mean([r['held'] for r in rs]):5.1f} stock {np.mean([r['stock'] for r in rs]):6.1f} "
                      f"air {np.mean([r['air'] for r in rs]):4.1f} blue {np.mean([r['blue'] for r in rs]):6.1f}")
            results.append({"t0": t0, "until": args.until, "res": res})
            # the main match carries on as the ghost (its own continuation is one more sample)
            s = play_to(main_g, pol, args.until, clock, s)
            print(f"  main match itself: {owned(s)}", flush=True)
        (out / f"gate4-{int(time.time())}.json").write_text(json.dumps(results))
    finally:
        main_g.close()
        pool.close()


# ---------------------------------------------------------------------------------------------- MPC

DEAD_WINDOWS = [(130.0, 105.0), (80.0, 55.0)]      # blue won auto (solo): its hub is off here
ACTIVE_WINDOWS = [(140.0, 130.0), (105.0, 80.0), (55.0, 0.0)]   # 140-130 both; 55-30 blue + 30-0 both: one stretch


def plan_window(t: float, stop_before: float, kinds=("dead",)):
    """(hi, lo, kind) of the planning window containing t, or None."""
    wins = ([(hi, lo, "dead") for hi, lo in DEAD_WINDOWS] if "dead" in kinds else []) + \
           ([(hi, lo, "active") for hi, lo in ACTIVE_WINDOWS] if "active" in kinds else [])
    for hi, lo, kind in wins:
        if lo + stop_before < t <= hi:
            return hi, lo, kind
    return None


def horizon(t: float, win: tuple, h_active: float) -> float:
    """Rollout end: a dead window's end; an active window's end + 3 s grace, at most h_active s ahead."""
    hi, lo, kind = win
    if kind == "dead":
        return lo
    return max(lo - 3.0 if lo > 0 else 0.3, t - h_active)


def value(r: dict, kind: str, lam: float) -> float | None:
    """Rollout score: dead windows -> owned balls at the window end (≈ next window's score); active windows ->
    points scored so far plus lam per ball still owned (worth less the nearer the buzzer)."""
    if r.get("owned") is None:
        return None
    if kind == "dead":
        return r["owned"]
    l = lam * min(1.0, max(0.0, (r["t"] - 1.0) / 12.0)) if r["t"] < 25.0 else lam
    return r["blue"] + l * (r["held"] + r["stock"]) + r.get("hubair", 0)


def make_candidates(s: dict, args, wkind: str = "dead", pol=None) -> list[tuple]:
    """[(tag, kind, kw, dur, reps)] around the current state; ghost always first."""
    out = [("ghost", "ghost", {}, 1e9, args.ghost_reps)]
    zone = wkind == "active"
    for sw in top_sweeps(s, k=args.sweeps, zone=zone):
        out.append((f"{'zsweep' if zone else 'sweep'} h{math.degrees(sw['h']):.0f} L{sw['L']} n{sw['n']:.0f}", "sweep",
                    {"h": sw["h"], "L": sw["L"], "shoot": zone}, 1e9, args.reps))
    if args.routes:
        if not ROUTES:
            load_routes()
        cur = getattr(pol, "route", "base")
        for name, g2 in ROUTES.items():
            if g2.name == pol.ghost.name:
                continue
            shift, d = route_join(pol, g2, s)
            if d <= args.route_tol:
                out.append((f"route {name} ({d:.2f}m {shift:+.1f}s)", "route", {"route": name, "tol": args.route_tol}, 1e9, args.reps))
    rob = s["robot"]
    v = math.hypot(rob.get("vx", 0.0), rob.get("vz", 0.0))
    if v > 0.3:
        px, pz = -rob["vz"] / v, rob["vx"] / v                  # left of the direction of travel
        for side in (1, -1):
            out.append((f"lateral {side * args.lateral:+.2f}", "offset",
                        {"dx": side * args.lateral * px, "dz": side * args.lateral * pz}, args.offset_s, args.reps))
    return out


def select(res: list[dict], cands: list[tuple], z: float, margin: float, wkind: str = "dead", lam: float = 0.8):
    stats = {}
    for tag, *_ in cands:
        v = np.array([value(r, wkind, lam) for r in res if r["tag"] == tag and r.get("owned") is not None], float)
        if len(v):
            stats[tag] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 10.0, len(v))
    g = stats.get("ghost", (0.0, 10.0, 0))
    best = max(stats, key=lambda k: stats[k][0])
    gain = stats[best][0] - g[0]
    need = max(margin, z * math.hypot(stats[best][1], g[1]))
    return (best if best != "ghost" and gain > need else "ghost"), stats


def mpc_match(main_g: Game, pool: Pool, pol, args, ep: int) -> dict:
    clock = MatchClock()
    pol.reset()
    s = main_g.client.reset(); s["t"] = clock(s)
    active, next_plan, decisions, win_end = None, None, [], {}
    trace, pending = [], []            # pending: decisions waiting for the main match to reach their horizon
    w0 = time.time()
    while not s.get("done"):
        win = plan_window(s["t"], args.stop_before, args.kinds) if s.get("gs") in (1, 2) and s.get("rs", 0) == 0 else None
        if win is None:
            active, next_plan = None, None
        elif next_plan is None or s["t"] <= next_plan:
            pw = time.time()
            snap = main_g.client.snapshot()
            cands = make_candidates(s, args, win[2], pol)
            jobs = [(tag, Candidate(clone(pol), kind, dur=dur, **kw)) for tag, kind, kw, dur, reps in cands for _ in range(reps)]
            h_end = horizon(s["t"], win, args.h_active)
            res = pool.map(snap["blob"], jobs, h_end)
            choice, stats = select(res, cands, args.z, args.margin, win[2], args.lam)
            tag, kind, kw, dur, _ = next(c for c in cands if c[0] == choice)
            active = None if kind == "ghost" else Candidate(pol, kind, dur=dur, **kw)
            next_plan = s["t"] - args.delta
            d = {"t": round(s["t"], 2), "win": win, "choice": choice, "ghost": stats.get("ghost"), "best": stats.get(choice),
                 "n_cands": len(cands), "wall": round(time.time() - pw, 1),
                 "stats": {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in stats.items()}}
            d.update({"h_end": round(h_end, 2), "kind": win[2]})
            decisions.append(d)
            pending.append(d)
            alt = max((k for k in stats if k != "ghost"), key=lambda k: stats[k][0], default=None)
            print(f"  ep{ep} t={d['t']:6.2f} {win[2]:6s} {choice:26s} gain {stats[choice][0] - stats['ghost'][0]:+5.1f} "
                  f"(ghost {stats['ghost'][0]:.1f}±{stats['ghost'][1]:.1f}; best alt {alt} "
                  f"{(stats[alt][0] - stats['ghost'][0]) if alt else 0:+.1f})  {d['wall']}s", flush=True)
        if active is not None:
            vx, vz, rot, b, _ = active.act(s)
            if active.kind != "sweep" and active.t_start - s["t"] >= active.dur:
                active = None
        else:
            vx, vz, rot, b, _ = pol.act(s)
        prev_t = s["t"]
        s = main_g.client.act(vx, vz, rot, b)
        s["t"] = clock(s)
        rob = s["robot"]
        trace.append([round(s["t"], 2), round(rob["x"], 2), round(rob["z"], 2), round(rob["yaw"]), s.get("held", 0), s["blue"],
                      getattr(active, "kind", "ghost") if active is not None else "ghost", round(getattr(pol, "lag", 0.0), 2)])
        for d in [d for d in pending if prev_t > d["h_end"] >= s["t"]]:
            # the same value the rollouts were scored with, now on the real match: calibration of the planner
            d["realized"] = round(value({**owned(s), "t": s["t"]}, d["kind"], args.lam), 1)
            pending.remove(d)
        for hi, lo in DEAD_WINDOWS:
            if prev_t > lo >= s["t"]:
                win_end[f"{hi:.0f}-{lo:.0f}"] = owned(s)
    return {"ep": ep, "score": s.get("blue", 0), "wall": round(time.time() - w0, 1), "win_end": win_end,
            "decisions": decisions, "n_switch": sum(d["choice"] != "ghost" for d in decisions), "trace": trace}


def mpc(args) -> None:
    ports = list(range(args.port0, args.port0 + args.planners + 1))
    extra = args.game_args.split() if args.game_args else None
    main_g = Game(port=ports[0], time_scale=1.0, steps_per_frame=2, extra_args=extra)
    main_g.attach_or_launch()
    pool = Pool(ports[1:], extra_args=extra)
    out = Path(__file__).resolve().parents[2] / "runs/planner" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args)))
    try:
        ghost = Ghost.from_file(str(DEMO))
        for ep in range(args.episodes):
            pol = make_policy(ghost, default_ghost_params(), {})
            r = mpc_match(main_g, pool, pol, args, ep)
            (out / f"ep{ep}.json").write_text(json.dumps(r))
            print(json.dumps({k: r[k] for k in ("ep", "score", "wall", "n_switch", "win_end")}), flush=True)
            cal = [(d["best"][0], d["realized"]) for d in r["decisions"] if "realized" in d and d["best"]]
            if cal:
                err = np.array([b - a for a, b in cal])
                print(f"  calibration: realized - predicted (chosen) mean {err.mean():+.1f} sd {err.std():.1f} over {len(cal)}", flush=True)
    finally:
        main_g.close()
        pool.close()


# ---------------------------------------------------------------------------------------------- screening

def screen(args) -> None:
    """Local A/B of tracker variants at one moment of the match: M main matches (different ball layouts) are played to
    t0 and snapshotted; from each snapshot every arm is rolled out `reps` times for `dur` s. Metric: balls acquired =
    held + scored 1.8 s later... approximated as the value of the window kind at the rollout end. Much less noise
    than whole matches (spread ~6 balls vs ~28 points), so a fix can be checked where it acts."""
    ports = list(range(args.port0, args.port0 + args.planners + args.mains))
    mains = [Game(port=p, time_scale=1.0, steps_per_frame=2) for p in ports[:args.mains]]
    with ThreadPoolExecutor(len(mains)) as ex:
        list(ex.map(lambda g: g.attach_or_launch(), mains))
    pool = Pool(ports[args.mains:])
    arms = [json.loads(a) for a in args.arms.split(";")]
    ghost = Ghost.from_file(str(DEMO))
    try:
        def to_t0(g):
            from .gamectl import robust_episode
            def run(client):
                pol = make_policy(ghost, default_ghost_params(), {})
                pol.reset(); clock = MatchClock()
                s = client.reset(); s["t"] = clock(s)
                s = play_to(g, pol, args.t0, clock, s)
                return pol, g.client.snapshot()
            return robust_episode(g, run)
        with ThreadPoolExecutor(len(mains)) as ex:
            snaps = list(ex.map(to_t0, mains))
        until = args.t0 - args.dur
        win = plan_window(args.t0, 0.0, ("dead", "active"))
        wkind = win[2] if win else "active"
        rows = {i: [] for i in range(len(arms))}
        for m, (pol, snap) in enumerate(snaps):
            jobs = []
            for _ in range(args.reps):                      # round-robin over arms: no arm always runs first/last
                for i, arm in enumerate(arms):
                    c = clone(pol); c.p.update(arm)
                    jobs.append((i, c))
            res = pool.map(snap["blob"], jobs, until)
            for r in res:
                rows[r["tag"]].append((m, value(r, wkind, 1.0), r["owned"], r["blue"], r.get("unsticks", 0)))
                for c in r.get("cr") or []:
                    print(f"      cr arm {r['tag']} main {m}: {json.dumps(c)}", flush=True)
        base = {m: np.mean([v for mm, v, *_ in rows[0] if mm == m]) for m in range(len(snaps))}
        for i, arm in enumerate(arms):
            d = np.array([v - base[m] for m, v, *_ in rows[i]])
            # paired by main match: per-main mean difference, SE over mains (includes arm 0's own noise)
            dm = np.array([np.mean([v for mm, v, *_ in rows[i] if mm == m]) - base[m] for m in range(len(snaps))])
            print(f"      paired-by-main Δ {dm.mean():+6.2f} ± {dm.std(ddof=1) / math.sqrt(len(dm)):.2f} (M={len(dm)})", flush=True)
            print(f"arm {i} {json.dumps(arm):40s} {wkind}-value vs arm0 {d.mean():+6.2f} ± {d.std(ddof=1) / math.sqrt(len(d)):.2f}  "
                  f"(n={len(d)}; owned {np.mean([r[2] for r in rows[i]]):.1f} blue {np.mean([r[3] for r in rows[i]]):.1f} "
                  f"unsticks {np.mean([r[4] for r in rows[i]]):.2f})", flush=True)
    finally:
        for g in mains:
            g.close()
        pool.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g4 = sub.add_parser("gate4")
    g4.add_argument("--t0", type=float, nargs="+", default=[127.0])
    g4.add_argument("--until", type=float, default=105.0)
    g4.add_argument("--reps", type=int, default=8)
    g4.add_argument("--planners", type=int, default=8)
    g4.add_argument("--port0", type=int, default=47520)
    m = sub.add_parser("mpc")
    m.add_argument("--name", default="m1")
    m.add_argument("--episodes", type=int, default=1)
    m.add_argument("--planners", type=int, default=12)
    m.add_argument("--port0", type=int, default=47520)
    m.add_argument("--delta", type=float, default=2.0, help="s of game time between plans")
    m.add_argument("--stop-before", type=float, default=3.0, help="no planning in the last s of a window")
    m.add_argument("--sweeps", type=int, default=6)
    m.add_argument("--lateral", type=float, default=0.4)
    m.add_argument("--offset-s", type=float, default=3.0)
    m.add_argument("--reps", type=int, default=3)
    m.add_argument("--ghost-reps", type=int, default=6)
    m.add_argument("--z", type=float, default=2.0)
    m.add_argument("--margin", type=float, default=3.0)
    m.add_argument("--kinds", default="dead", help="comma list: dead,active")
    m.add_argument("--routes", type=int, default=0, help="1: add switches to other demo routes / mirrors")
    m.add_argument("--route-tol", type=float, default=1.0, help="m: a route is joinable if it passes this close")
    m.add_argument("--game-args", default="", help="extra MoSimulator args, e.g. '-job-worker-count 1'")
    m.add_argument("--h-active", type=float, default=12.0, help="max rollout length in active windows (s)")
    m.add_argument("--lam", type=float, default=1.0, help="value of a still-owned ball in active windows")
    sc = sub.add_parser("screen")
    sc.add_argument("--t0", type=float, required=True)
    sc.add_argument("--dur", type=float, default=5.0)
    sc.add_argument("--arms", required=True, help="';'-separated GhostPolicy param overrides; arm 0 is the control")
    sc.add_argument("--mains", type=int, default=4)
    sc.add_argument("--planners", type=int, default=8)
    sc.add_argument("--reps", type=int, default=4)
    sc.add_argument("--port0", type=int, default=47560)
    args = ap.parse_args()
    if args.cmd == "screen":
        screen(args)
    if args.cmd == "gate4":
        gate4(args)
    elif args.cmd == "mpc":
        args.kinds = tuple(args.kinds.split(","))
        mpc(args)


if __name__ == "__main__":
    main()
