"""Run GhostPolicy (optionally with bumps) in MiniSim, many episodes, and print the same summary as run_ghost."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mosimrl.episode import run_episode
from mosimrl.ghost import Ghost
from mosimrl.ghost_policy import default_ghost_params
from mosimrl.run_ghost import make_policy

from .sim import MiniSimClient

DEMO = "../run/demos/demo-20260926-015440-m1.jsonl"
WINDOWS = [(160, 130), (130, 105), (105, 80), (80, 55), (55, 30), (30, -5)]


def per_window(trace):
    t = np.array([s["t"] for s in trace]); blue = np.array([s["blue"] for s in trace])
    out = []
    for hi, lo in WINDOWS:
        m = np.nonzero((t <= hi) & (t > lo))[0]
        out.append(int(blue[m[-1]] - blue[m[0]]) if len(m) else 0)
    out[-1] += int(blue[-1] - blue[np.nonzero(t > -5)[0][-1]])
    return out


def run(arm: dict, n: int, seed0: int = 0, params: dict | None = None):
    g = Ghost.from_file(DEMO)
    gp = default_ghost_params()
    res = []
    for i in range(n):
        c = MiniSimClient(params, seed=seed0 + i)
        pol = make_policy(g, gp, arm, seed=i)
        r = run_episode(c, pol, trace=True, fuel_every=0)
        res.append({"score": r["score"], "auto": r["auto"], "win": per_window(r["trace"]),
                    "unsticks": pol.unsticks, "trace": r["trace"]})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="{}")
    ap.add_argument("-n", type=int, default=16)
    a = ap.parse_args()
    for k, arm in enumerate(a.arms.split(";")):
        t0 = time.time()
        R = run(json.loads(arm), a.n)
        s = np.array([r["score"] for r in R]); w = np.array([r["win"] for r in R])
        print(f"arm {k}: mean {s.mean():.1f} sd {s.std():.1f}  auto {np.mean([r['auto'] for r in R]):.1f}  "
              f"windows {w.mean(0).round(1).tolist()}  unst {np.mean([r['unsticks'] for r in R]):.2f}  "
              f"({time.time() - t0:.0f}s)")
