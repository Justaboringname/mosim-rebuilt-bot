"""Real-engine option learning (docs/research-log/proposals.md 2026-09-27 "OL"): randomized options at events, credited by
snapshot rollouts, one base-anchored improvement step. Route untouched (no position, offset or warp changes).

Event A, a turn while shooting: ghost AutoShoot pressed, robot x > 3.9, held >= 20, the ghost heading turns > 45 deg
over the next 1.0 s (yaw_dt clock), >= 1 s since the last trigger, >= 1.5 s after the rollout start.
  options: base 0.5 / placebo 0.1 / cap 0.4. cap = |rot| <= 0.30 until |yaw_err| < 10 deg (after it first exceeded it),
  3 s, AutoShoot released or held < 20; then the limit ramps back to 1 at 2 units/s.
Event C, hopper hold before 55 (once, at t <= 60): base 0.5 / placebo 0.1 / hold 0.4. hold = AutoPass released
  until t = 55; aborted if held >= 90.

Collection: M mains play the default tracker and snapshot at 106 (W2) and 62 (W3). Each snapshot gets 10 rollouts on the
planner pool: 2 pure base (options drawn but forced to base) and 8 randomized. W2 runs to 77 (value = blue + beta*owned,
beta 0.35; 2 of the 10 continue to the buzzer for an unbiased check); W3 runs to the buzzer (value = final blue).

    python -m mosimrl.optlearn collect --run ol1 --mains-total 400 --planners 14 --mains 2
    python -m mosimrl.optlearn analyze --run ol1
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .client import BridgeError
from .episode import MatchClock
from .gamectl import Game, robust_episode
from .ghost import Ghost, wrap_deg
from .ghost_policy import default_ghost_params
from .planner import DT, clone, owned, play_to
from .run_ghost import DEMO_KEYS, make_policy

ROOT = Path(__file__).resolve().parents[2]
PROPS_A = (("base", 0.5), ("placebo", 0.1), ("cap", 0.4))
PROPS_C = (("base", 0.5), ("placebo", 0.1), ("hold", 0.4))
CAP = 0.30
FEATS = ["t_to_end", "held", "held_over70", "dheld_1s", "dblue_1s", "turn_05", "turn_10", "turn_20", "abs_wy",
         "abs_yaw_err", "gate_err", "speed", "x", "z", "lagging", "ghost_minus_held"]


class OptionAgent:
    def __init__(self, base, seed: int, mode: str = "random"):
        self.base = base                       # a GhostPolicy (planner.rollout reads .base.unsticks)
        self.rng = np.random.default_rng(seed)
        self.mode = mode                       # "random" | "base"
        self.events: list[dict] = []
        self.cur: dict | None = None           # active Event A option
        self.lim = 1.0
        self.last_trig = 1e9
        self.t0: float | None = None
        self.c_state = "pending"               # Event C: pending | done | holding
        self.hist: deque = deque(maxlen=12)

    needs_fuel = False

    @property
    def unsticks(self) -> int:
        return getattr(self.base, "unsticks", 0)

    def _draw(self, props) -> tuple[str, float]:
        names, ps = zip(*props)
        k = int(self.rng.choice(len(names), p=ps))
        return (names[k] if self.mode == "random" else "base"), float(ps[k])

    def _turn(self, T0: float, dt: float) -> float:
        g = self.base.ghost
        return abs(g._interp(g.yaw_unwrapped, T0 - dt) - g._interp(g.yaw_unwrapped, T0))

    def _phi(self, s: dict, tg: float, yd: float, info: dict, win_end: float) -> list[float]:
        r = s["robot"]; held = s.get("held", 0)
        old = self.hist[0] if self.hist else (s["t"], held, s.get("blue", 0))
        return [s["t"] - win_end, held, max(0, held - 70), held - old[1], s.get("blue", 0) - old[2],
                self._turn(tg + yd, 0.5), self._turn(tg + yd, 1.0), self._turn(tg + yd, 2.0),
                abs(math.degrees(r.get("wy", 0.0))), abs(info.get("yaw_err", 0.0)),
                abs((s.get("gate") or {}).get("err", 0.0)), math.hypot(r.get("vx", 0.0), r.get("vz", 0.0)),
                r["x"], r["z"], float(self.base.lag > 0), info.get("ghost_held", held) - held]

    def act(self, s: dict):
        t = s["t"]
        if self.t0 is None:
            self.t0 = t
        vx, vz, rot, b, info = self.base.act(s)
        p = self.base.p
        tg = t - p["time_shift"] - self.base._ts(t) + self.base.lag
        yd = p.get("yaw_dt", 0.0)
        held = s.get("held", 0)
        r = s["robot"]
        wy = abs(math.degrees(r.get("wy", 0.0)))
        # --- Event A: a turn while shooting
        gb = self.base.ghost.at(tg).buttons
        shooting = bool(gb[1]) and r["x"] > 3.9 and held >= 20
        if (self.cur is None and self.lim >= 1.0 and shooting and self.t0 - t >= 1.5 and self.last_trig - t >= 1.0
                and self._turn(tg + yd, 1.0) > 45.0 and (105.0 >= t > 77.0 or 55.0 >= t > 0.0 or 139.0 >= t > 127.0)):
            opt, pr = self._draw(PROPS_A)
            win_end = 77.0 if t > 60 else 0.0
            ev = {"ev": "A", "t": round(t, 2), "opt": opt, "p": pr, "phi": [round(v, 3) for v in self._phi(s, tg, yd, info, win_end)],
                  "hi_s": 0.0, "rec_s": None, "big": False}
            self.events.append(ev)
            self.last_trig = t
            self.cur = {"ev": ev, "t": t} if opt == "cap" else {"ev": ev, "t": t, "watch_only": True}
            if opt == "cap":
                self.lim = CAP
        if self.cur is not None:
            ev, el = self.cur["ev"], self.cur["t"] - t
            ev["hi_s"] += 0.099 if wy > 120.0 else 0.0
            ye = abs(info.get("yaw_err", 0.0))
            if ye >= 10.0:
                ev["big"] = True
            elif ev["big"] and ev["rec_s"] is None:
                ev["rec_s"] = round(el, 2)
            ended = el >= 3.0 or (ev["big"] and ye < 10.0) or not b[1] or held < 20
            if ended:
                self.cur = None
        if self.lim < 1.0:
            rot = float(np.clip(rot, -self.lim, self.lim))
            if self.cur is None:
                self.lim = min(1.0, self.lim + 2.0 * 0.099)
        # --- Event C: hopper hold before 55
        if self.c_state == "pending" and 60.0 >= t > 56.0:
            opt, pr = self._draw(PROPS_C)
            self.events.append({"ev": "C", "t": round(t, 2), "opt": opt, "p": pr,
                                "phi": [round(v, 3) for v in self._phi(s, tg, yd, info, 55.0)], "held_at": held})
            self.c_state = "holding" if opt == "hold" else "done"
        if self.c_state == "holding":
            if t <= 55.0 or held >= 90:
                self.c_state = "done"
            else:
                b = list(b); b[2] = False
        self.hist.append((t, held, s.get("blue", 0)))
        return vx, vz, rot, b, info


def roll(g: Game, blob: str, agent: OptionAgent, until_t: float | None, cont: bool) -> dict:
    """Restore, run the agent to until_t (full state there) and optionally on to the buzzer."""
    w0 = time.time()
    s = g.client.restore(blob)
    clock = MatchClock()
    s["t"] = clock(s)
    t0 = s["t"]
    out: dict = {"t0": t0}
    while not s.get("done"):
        vx, vz, rot, b, _ = agent.act(s)
        cross = until_t is not None and "at" not in out and s["t"] - DT <= until_t + 1e-3
        s = g.client.act(vx, vz, rot, b, lite=not cross)
        s["t"] = clock(s)
        if cross:
            out["at"] = {"t": s["t"], **owned(s)}
            if not cont:
                break
    out["final_blue"] = s.get("blue", 0) if s.get("done") else None
    out["blue_end"] = s.get("blue", 0)
    out["unsticks"] = agent.unsticks
    out["events"] = agent.events
    out["wall"] = round(time.time() - w0, 1)
    return out


class Collector:
    def __init__(self, args):
        self.args = args
        self.dir = ROOT / "runs/rl_opt" / args.run
        self.dir.mkdir(parents=True, exist_ok=True)
        extra = args.game_args.split() if args.game_args else []
        ports = list(range(args.port0, args.port0 + args.planners + args.mains))
        self.mains = [Game(port=p, time_scale=1.0, steps_per_frame=2, extra_args=extra) for p in ports[:args.mains]]
        self.planners = [Game(port=p, time_scale=1.0, steps_per_frame=2, extra_args=extra) for p in ports[args.mains:]]
        with ThreadPoolExecutor(len(ports)) as ex:
            list(ex.map(lambda g: g.attach_or_launch(), self.mains + self.planners))
        with ThreadPoolExecutor(len(self.planners)) as ex:
            list(ex.map(lambda g: g.client.reset(), self.planners))
        self.free: queue.Queue = queue.Queue()
        for g in self.planners:
            self.free.put(g)
        self.ex = ThreadPoolExecutor(len(self.planners))
        self.slots = threading.Semaphore(args.backlog)
        self.lock = threading.Lock()
        self.ghost = Ghost.from_file(DEMO_KEYS["1118"])
        self.n_main = 0                         # continue main ids after an earlier (restarted) collection
        for name in ("mains.jsonl", "rollouts.jsonl", "failed.jsonl"):
            if (self.dir / name).exists():
                for l in open(self.dir / name):
                    try:
                        self.n_main = max(self.n_main, int(json.loads(l)["main"]))
                    except Exception:
                        pass
        self.mains_start = self.n_main
        self.stop = threading.Event()

    def write(self, name: str, rec: dict) -> None:
        with self.lock:
            with open(self.dir / name, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def job(self, blob, pol, meta, until_t, cont, mode, seed):
        try:
            for attempt in range(3):
                g = self.free.get()
                try:
                    res = roll(g, blob, OptionAgent(clone(pol), seed, mode), until_t, cont)
                    self.write("rollouts.jsonl", {**meta, "mode": mode, "seed": seed, "cont": cont, **res})
                    if res.get("final_blue") is not None:
                        # at the buzzer the Bridge releases control and would answer the next restore with 'unknown
                        # cmd' (-> a full relaunch); a reset puts the instance back into a match, awaiting an action
                        g.client.reset()
                    return
                except (socket.timeout, TimeoutError, ConnectionError, OSError, BridgeError) as e:
                    try:
                        g.recover(f"optlearn: {type(e).__name__}: {e}"); g.client.reset()
                    except Exception as e2:
                        print(f"[optlearn] could not recover {g.port}: {e2}", flush=True)
                finally:
                    self.free.put(g)
            self.write("failed.jsonl", meta)
        finally:
            self.slots.release()

    def dispatch(self, blob, pol, meta, until_t, n_base, n_rand, n_cont):
        modes = ["base"] * n_base + ["random"] * n_rand
        cont_idx = set(np.random.default_rng(hash((meta["main"], meta["win"])) % 2**32).choice(len(modes), n_cont, replace=False)) if n_cont else set()
        for k, mode in enumerate(modes):
            self.slots.acquire()
            seed = (meta["main"] * 1000 + (0 if meta["win"] == "W2" else 500) + k) % 2**31
            self.ex.submit(self.job, blob, pol, meta, until_t, k in cont_idx, mode, seed)

    def main_loop(self, g: Game) -> None:
        while not self.stop.is_set():
            with self.lock:
                if self.n_main - self.mains_start >= self.args.mains_total:
                    return
                self.n_main += 1
                mid = self.n_main
            def run(client):
                pol = make_policy(self.ghost, default_ghost_params(), {})
                pol.reset(); clock = MatchClock()
                s = client.reset(); s["t"] = clock(s)
                s = play_to(g, pol, 106.0, clock, s)
                snap = client.snapshot()
                self.dispatch(snap["blob"], clone(pol), {"main": mid, "win": "W2"}, 77.0, 2, 8, 2)
                s = play_to(g, pol, 62.0, clock, s)
                snap = client.snapshot()
                self.dispatch(snap["blob"], clone(pol), {"main": mid, "win": "W3"}, None, 2, 8, 0)
                while not s.get("done"):
                    vx, vz, rot, b, _ = pol.act(s)
                    s = client.act(vx, vz, rot, b, lite=True)
                    s["t"] = clock(s)
                return s.get("blue", 0)
            try:
                score = robust_episode(g, run)
                self.write("mains.jsonl", {"main": mid, "score": score, "wall": time.time()})
                print(f"[optlearn] main {mid} score {score}", flush=True)
            except Exception as e:
                print(f"[optlearn] main {mid} failed: {e}", flush=True)

    def run(self) -> None:
        try:
            ths = [threading.Thread(target=self.main_loop, args=(g,)) for g in self.mains]
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            self.ex.shutdown(wait=True)
        finally:
            for g in self.mains + self.planners:
                g.close(quit_game=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--run", required=True)
    c.add_argument("--mains-total", type=int, default=400)
    c.add_argument("--mains", type=int, default=2)
    c.add_argument("--planners", type=int, default=14)
    c.add_argument("--port0", type=int, default=47700)
    c.add_argument("--backlog", type=int, default=40)
    c.add_argument("--game-args", default="")
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    args = ap.parse_args()
    if args.cmd == "collect":
        Collector(args).run()
    else:
        from .optfit import analyze
        analyze(ROOT / "runs/rl_opt" / args.run)


if __name__ == "__main__":
    main()
