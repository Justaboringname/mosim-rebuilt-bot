"""CMA-ES over the scripted policy's parameters.

    python -m mosimrl.cmaes_run --ports 47414 --out runs/cma1            # real game(s)
    python -m mosimrl.cmaes_run --mock --ports 48001,48002 --out runs/mock  # plumbing test

Each candidate plays `--episodes` matches (mean score is the fitness). Several game instances on
different ports evaluate candidates in parallel. Everything is logged to <out>/evals.jsonl, the
optimizer state is pickled every generation, and <out>/best.json holds the best mean so far.
"""

from __future__ import annotations

import argparse
import json
import pickle
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cma
import numpy as np

from .client import MoSimClient
from .episode import run_episode
from .policy import PARAM_NAMES, ScriptedPolicy, default_params, from_unit, to_unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="47414")
    ap.add_argument("--mock", action="store_true", help="start mock bridges on the ports (plumbing test only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--popsize", type=int, default=0, help="0 = CMA default for the dimension")
    ap.add_argument("--maxgen", type=int, default=60)
    ap.add_argument("--sigma0", type=float, default=0.20)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--x0", default="", help="json file with starting params (default: policy defaults)")
    ap.add_argument("--time-scale", type=float, default=0.0, help="configure bridges' timeScale (0 = leave)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mock:
        from .mock_server import serve
        for i, p in enumerate(ports):
            serve(p, seed=args.seed * 1000 + i)
        time.sleep(0.2)

    clients: queue.Queue = queue.Queue()
    for p in ports:
        c = MoSimClient(port=p)
        info = c.hello()
        if args.time_scale > 0 or args.no_cameras:
            info = c.config(time_scale=args.time_scale or None, cameras=False if args.no_cameras else None)
        print(f"port {p}: {info}")
        clients.put(c)

    def evaluate(u) -> dict:
        c = clients.get()
        try:
            params = from_unit(u)
            scores, autos, walls = [], [], []
            for _ in range(args.episodes):
                r = run_episode(c, ScriptedPolicy(params))
                scores.append(r["score"]); autos.append(r["auto"]); walls.append(r["wall_s"])
            return {"params": params, "scores": scores, "mean": float(np.mean(scores)),
                    "auto": autos, "wall_s": float(np.sum(walls))}
        finally:
            clients.put(c)

    es_path = out / "es.pkl"
    if args.resume and es_path.exists():
        es, gen, best = pickle.loads(es_path.read_bytes())
        print(f"resumed at generation {gen}, best {best['mean'] if best else None}")
    else:
        x0 = default_params()
        if args.x0:
            x0.update(json.loads(Path(args.x0).read_text()))
        opts = {"bounds": [0.0, 1.0], "seed": args.seed, "verbose": -9}
        if args.popsize:
            opts["popsize"] = args.popsize
        es = cma.CMAEvolutionStrategy(to_unit(x0).tolist(), args.sigma0, opts)
        gen, best = 0, None

    log = (out / "evals.jsonl").open("a")
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        while gen < args.maxgen and not es.stop():
            t0 = time.time()
            X = es.ask()
            results = list(pool.map(evaluate, X))
            es.tell(X, [-r["mean"] for r in results])
            for i, r in enumerate(results):
                log.write(json.dumps({"gen": gen, "i": i, **r}) + "\n")
                if best is None or r["mean"] > best["mean"]:
                    best = {"gen": gen, **r}
            log.flush()
            (out / "best.json").write_text(json.dumps(best, indent=1))
            gen += 1
            es_path.write_bytes(pickle.dumps((es, gen, best)))
            means = [r["mean"] for r in results]
            print(f"gen {gen:3d}  mean {np.mean(means):7.1f}  max {np.max(means):7.1f}  "
                  f"best-ever {best['mean']:7.1f}  sigma {es.sigma:.3f}  {time.time() - t0:6.1f}s", flush=True)

    mean_params = from_unit(es.result.xfavorite)
    (out / "cma_mean.json").write_text(json.dumps(mean_params, indent=1))
    print("best:", json.dumps({k: round(best["params"][k], 3) for k in PARAM_NAMES}))


if __name__ == "__main__":
    main()
