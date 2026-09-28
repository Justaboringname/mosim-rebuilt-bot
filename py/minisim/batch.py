"""Write MiniSim episodes in the same layout as a real run_ghost batch (runs/ghost/<name>/summary.json, *-epN.json,
*.dense.npz), so every calibration script can be pointed at a simulated batch and compared line by line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mosimrl.episode import run_episode
from mosimrl.ghost import Ghost
from mosimrl.ghost_policy import default_ghost_params
from mosimrl.run_ghost import make_policy, summarise

from .sim import MiniSimClient
from .sim2 import MiniSim2Client

CLIENTS = {"v1": MiniSimClient, "v2": MiniSim2Client}

DEMO = "../run/demos/demo-20260926-015440-m1.jsonl"


def _one(job):
    name, ep, arm_i, arm, params, seed, sim = job
    out = Path("../runs/ghost") / name
    g = Ghost.from_file(DEMO)
    pol = make_policy(g, default_ghost_params(), arm, seed=ep)
    r = run_episode(CLIENTS[sim](params, seed=seed), pol, trace=True, dense=True)
    r.pop("first_state")
    dense = r.pop("dense")
    np.savez_compressed(out / f"ghost-{g.name}-ep{ep}.dense.npz", **dense)
    summ = summarise(r["trace"], g)
    (out / f"ghost-{g.name}-ep{ep}.json").write_text(json.dumps({**r, **summ, "port": 0}))
    return {"ep": ep, "arm": arm_i, "score": r["score"], "auto": r["auto"], "unsticks": pol.unsticks, **summ}


def write_batch(name: str, arms: list[dict], n: int, params: dict | None = None, seed0: int = 0,
                sim: str = "v2", procs: int = 16) -> Path:
    out = Path("../runs/ghost") / name
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, ep, ep % len(arms), arms[ep % len(arms)], params, seed0 + ep, sim) for ep in range(n)]
    from multiprocessing import get_context
    with get_context("spawn").Pool(min(procs, n)) as pool:
        rows = pool.map(_one, jobs)
    g = Ghost.from_file(DEMO)
    (out / "summary.json").write_text(json.dumps({"ghost": g.name, "episodes": rows, "sim": sim, "params": params}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("-n", type=int, default=8); ap.add_argument("--arms", default="{}")
    ap.add_argument("--sim", default="v2"); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    p = write_batch(a.name, [json.loads(x) for x in a.arms.split(";")], a.n, params=json.loads(a.params),
                    sim=a.sim, procs=a.procs)
    S = json.load(open(p / "summary.json"))
    print(p, [e["score"] for e in S["episodes"]])
