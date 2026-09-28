"""CMA-ES over GhostPolicy parameters on N headless instances (noisy fitness = mean score over k matches).

    python -m mosimrl.cma_ghost --run c1 --ports 47500-47507 --k 3 --gens 30

Every generation also plays `--control` matches with the current incumbent (best-so-far mean) and the defaults, so
load/instance drift is visible. State is pickled each generation (resume with --resume).
"""

from __future__ import annotations

import argparse
import json
import pickle
import queue
import threading
import time
from pathlib import Path

import cma
import numpy as np

from .episode import run_episode
from .gamectl import Game, robust_episode
from .ghost import Ghost
from .ghost_policy import GhostPolicy, default_ghost_params
from .rl_ppo import parse_ports

ROOT = Path(__file__).resolve().parents[2]
# name, low, high (search box; defaults from GHOST_SPEC)
BASE_SPACE = [
    ("kp_pos", 0.5, 2.5), ("kd_vel", 0.0, 0.8), ("lookahead", 0.0, 0.5), ("kp_yaw", 0.3, 2.5),
    ("ff_gain", 0.8, 1.15), ("time_shift", -0.6, 0.6), ("wall_push", 0.0, 0.6), ("deploy_x", 3.0, 3.8),
    ("resync_err", 1.0, 3.0), ("catchup", 0.05, 0.4),
]
EXTRA_SPACE = [("stall_s", 0.3, 1.5), ("backoff_s", 0.2, 0.8), ("resync_s", 0.2, 1.5)]
# per-channel timing (heading / buttons on their own clocks, runs/ghost/yaw1+yaw2)
TIMING_SPACE = [("yaw_dt", 0.0, 0.5), ("btn_dt", -0.4, 0.4)]
TS_SPACE = [(f"ts{i}", -1.2, 1.2) for i in range(1, 8)]      # ts0 (t=160) stays 0: the start is fixed
# lateral route-shift knots over the two dead-window conveyor segments (u = fraction of the conveyor leash)
KNOT_TIMES = [128, 124, 120, 116, 112, 108, 78, 74, 70, 66, 62, 58]
KNOT_SPACE = [(f"k{t}", -1.0, 1.0) for t in KNOT_TIMES]
SPACE = list(BASE_SPACE)


def to_unit(p: dict) -> np.ndarray:
    return np.array([(p.get(n, 0.0) - lo) / (hi - lo) for n, lo, hi in SPACE])


def from_unit(u) -> dict:
    return {n: float(lo + np.clip(x, 0, 1) * (hi - lo)) for (n, lo, hi), x in zip(SPACE, u)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--demo", default=str(ROOT / "run/demos/demo-20260926-015440-m1.jsonl"))
    ap.add_argument("--ports", default="47500-47507")
    ap.add_argument("--k", type=int, default=3, help="matches per candidate")
    ap.add_argument("--control", type=int, default=2, help="default-param matches per generation")
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=0)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--space", default="base", choices=["base", "ts", "all", "knots", "base+", "timing"])
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--steps-per-frame", type=int, default=None, help="fixed frame step (use with --time-scale 1)")
    ap.add_argument("--conveyor-leash", type=float, default=0.667, help="fraction of the 3 m final leash (knots space)")
    ap.add_argument("--x0", default="", help="json file of starting params (e.g. a previous favorite.json)")
    args = ap.parse_args()
    global SPACE
    SPACE = {"base": BASE_SPACE, "ts": TS_SPACE, "all": BASE_SPACE + TS_SPACE, "knots": KNOT_SPACE,
             "base+": BASE_SPACE + EXTRA_SPACE, "timing": BASE_SPACE + TIMING_SPACE}[args.space]

    out = ROOT / "runs/cma" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ghost = Ghost.from_file(args.demo)
    base = default_ghost_params()
    if args.x0:
        base.update(json.loads(Path(args.x0).read_text()))
    games = [Game(port=p, time_scale=args.time_scale, steps_per_frame=args.steps_per_frame) for p in parse_ports(args.ports)]
    ths = [threading.Thread(target=g.attach_or_launch) for g in games]
    [t.start() for t in ths]; [t.join() for t in ths]

    jobs: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()

    def worker(game: Game):
        while True:
            job = jobs.get()
            if job is None:
                return
            if args.space == "knots":
                from .residual import ResidualAgent, ScheduleChooser
                kn = {t: job["params"].get(f"k{t}", 0.0) for t in KNOT_TIMES}
                kn.update({131: 0.0, 105: 0.0, 81: 0.0, 55: 0.0})      # zero outside the conveyor segments
                pol = ResidualAgent(ghost, ScheduleChooser(kn), base, leash_scale={"conveyor": args.conveyor_leash})
            else:
                pol = GhostPolicy(ghost, {**base, **job["params"]})   # control jobs (params={}) play `base`
            try:
                r = robust_episode(game, lambda c: run_episode(c, pol, trace=False))
                results.put({**job, "score": r["score"], "auto": r["auto"], "unsticks": pol.unsticks, "port": game.port})
            except Exception as e:
                results.put({**job, "error": repr(e), "port": game.port})

    for g in games:
        threading.Thread(target=worker, args=(g,), daemon=True).start()

    es_path = out / "es.pkl"
    if args.resume and es_path.exists():
        es, gen, hist = pickle.loads(es_path.read_bytes())
    else:
        opts = {"bounds": [0.0, 1.0], "seed": 7, "verbose": -9}
        if args.popsize:
            opts["popsize"] = args.popsize
        es = cma.CMAEvolutionStrategy(to_unit(base).tolist(), args.sigma0, opts)
        gen, hist = 0, []
    log = (out / "evals.jsonl").open("a")
    while gen < args.gens and not es.stop():
        t0 = time.time()
        X = es.ask()
        n = 0
        for i, x in enumerate(X):
            for _ in range(args.k):
                jobs.put({"gen": gen, "cand": i, "params": from_unit(x)}); n += 1
        for _ in range(args.control):
            jobs.put({"gen": gen, "cand": -1, "params": {}}); n += 1
        got = [results.get() for _ in range(n)]
        for g_ in got:
            log.write(json.dumps(g_) + "\n")
        log.flush()
        fit = []
        for i in range(len(X)):
            s = [g_["score"] for g_ in got if g_["cand"] == i and "score" in g_]
            fit.append(-float(np.mean(s)) if s else 0.0)
        es.tell(X, fit)
        ctrl = [g_["score"] for g_ in got if g_["cand"] == -1 and "score" in g_]
        row = {"gen": gen, "time": time.strftime("%H:%M:%S"), "wall_s": round(time.time() - t0),
               "cand_means": sorted([-f for f in fit], reverse=True), "control": ctrl,
               "mean_of_means": float(-np.mean(fit)), "sigma": float(es.sigma),
               "xfav": from_unit(es.result.xfavorite)}
        hist.append(row)
        print(json.dumps(row), flush=True)
        gen += 1
        es_path.write_bytes(pickle.dumps((es, gen, hist)))
        (out / "favorite.json").write_text(json.dumps(from_unit(es.result.xfavorite), indent=1))
    for _ in games:
        jobs.put(None)


if __name__ == "__main__":
    main()
