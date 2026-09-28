"""Throughput benchmark: N headless instances stepping the ghost tracker in parallel (the run_ghost per-step work: full
state, BallTracker, trace), reporting decisions/s and where the wall time goes on the Python side.

    python -m mosimrl.bench --n 16 --game-args "-job-worker-count 1" --port0 47600

Each run launches fresh instances on its own ports and quits them afterwards (attach_or_launch would otherwise reuse an
instance started with other args).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .episode import BallTracker, MatchClock
from .gamectl import Game
from .ghost import Ghost
from .ghost_policy import default_ghost_params
from .run_ghost import DEMO_KEYS, make_policy


def run_one(g: Game, ghost: Ghost, steps: int, lite: bool) -> dict:
    pol = make_policy(ghost, default_ghost_params(), {})
    pol.reset()
    clock = MatchClock()
    balls = BallTracker()
    s = g.client.reset()
    s["t"] = clock(s)
    rt = py = 0.0
    t_start = time.perf_counter()
    for k in range(steps):
        a = time.perf_counter()
        if not lite:
            balls.update(s)
        vx, vz, rot, b, info = pol.act(s)
        _ = {"t": s["t"], "x": s["robot"]["x"], "fuel": [[round(q[0], 2), round(q[1], 2), round(q[2], 2)] for q in s.get("fuel") or []]
             } if (not lite and k % 5 == 0) else None
        m = time.perf_counter()
        s = g.client.act(vx, vz, rot, b, lite=lite)
        s["t"] = clock(s)
        e = time.perf_counter()
        py += m - a; rt += e - m
        if s.get("done"):
            break
    wall = time.perf_counter() - t_start
    return {"steps": k + 1, "wall": wall, "py": py, "rt": rt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--port0", type=int, default=47600)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--game-args", default="")
    ap.add_argument("--lite", action="store_true")
    args = ap.parse_args()
    ghost = Ghost.from_file(DEMO_KEYS["1118"])
    extra = args.game_args.split() if args.game_args else []
    games = [Game(port=args.port0 + i, time_scale=1.0, steps_per_frame=2, extra_args=extra) for i in range(args.n)]
    with ThreadPoolExecutor(args.n) as ex:
        list(ex.map(lambda g: g.attach_or_launch(), games))
    try:
        with ThreadPoolExecutor(args.n) as ex:              # warm-up: first reset loads the scene
            list(ex.map(lambda g: g.client.reset(), games))
        cpu = {}

        def sample_cpu():
            time.sleep(8)
            out = subprocess.run(["ps", "-A", "-o", "%cpu=,rss=,comm="], capture_output=True, text=True).stdout
            game = [l.split(None, 2) for l in out.splitlines() if "MoSimulator" in l]
            cpu["game_cpu"] = round(sum(float(c) for c, _, _ in game), 0)
            cpu["game_rss_gb"] = round(sum(int(r) for _, r, _ in game) / 1e6, 1)
            cpu["python_cpu"] = round(sum(float(l.split(None, 2)[0]) for l in out.splitlines() if "Python" in l), 0)
        th = threading.Thread(target=sample_cpu); th.start()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.n) as ex:
            res = list(ex.map(lambda g: run_one(g, ghost, args.steps, args.lite), games))
        wall = time.perf_counter() - t0
        th.join()
        steps = sum(r["steps"] for r in res)
        out = {"n": args.n, "game_args": args.game_args, "lite": args.lite, "decisions_per_s": round(steps / wall, 1),
               "per_instance_per_s": round(steps / wall / args.n, 2),
               "py_frac": round(sum(r["py"] for r in res) / sum(r["wall"] for r in res), 3),
               "roundtrip_frac": round(sum(r["rt"] for r in res) / sum(r["wall"] for r in res), 3),
               "ms_per_step_py": round(1000 * sum(r["py"] for r in res) / steps, 2),
               "ms_per_step_roundtrip": round(1000 * sum(r["rt"] for r in res) / steps, 2), **cpu}
        print(json.dumps(out), flush=True)
    finally:
        for g in games:
            g.close(quit_game=True)


if __name__ == "__main__":
    main()
