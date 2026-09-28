"""Joint (whole-window) headroom: the trips of one window share one ball pool.

headroom.py scored every trip on its own and summed the gains. bump1 (2026-09-26) showed why that is wrong: a bump
that takes a cluster early steals it from the ghost's next trip, which was going to sweep it anyway (124-118 +18
intakes, then 118-112 -60). Here a window's whole route (all its trips, bumped or not) sweeps a single robot-free
pool, each ball counted once, with the hopper cap and pass drain.

    python -m mosimrl.joint ../runs/ghost/ck '{"124,118": [0.5, 1.0, -0.9], ...}'
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

from .bumps import bumped_ghost
from .ghost import Ghost
from .headroom import DEMO, PASS_RATE, hopper_count, robot_free, snapshots, sweep_times

WINDOWS = {"transition": (139.2, 135.5), "dead1": (129.0, 106.0), "dead2": (78.0, 58.0), "endgame": (31.0, 6.0)}


def window_path(g: Ghost, a: float, b: float, dt: float = 0.05):
    ts = np.arange(a, b - 1e-9, -dt)
    return ts, np.array([(g._interp(g.x, t), g._interp(g.z, t)) for t in ts])


class Evaluator:
    def __init__(self, run_dir: str, max_eps: int | None = None):
        self.g = Ghost.from_file(DEMO)
        self.eps = []
        for f in sorted(glob.glob(f"{run_dir}/*-ep*.json"))[:max_eps]:
            tr = json.load(open(f))["trace"]
            if not any(r.get("fid") for r in tr):
                continue
            S = list(snapshots(tr))
            per = {}
            for w, (a, b) in WINDOWS.items():
                sn = [x for x in S if b - 0.3 <= x[0] <= a + 0.3]
                held0 = next(r["held"] for r in tr if r["t"] <= a)
                per[w] = (robot_free(sn), held0)
            self.eps.append(per)

    def drain(self, t):
        b = self.g.at(t).buttons
        return PASS_RATE if (b[1] or b[2] or b[3]) else 0.0

    def score(self, bumps: dict | None, windows=None) -> dict:
        m = bumped_ghost(self.g, bumps) if bumps else self.g
        out = {}
        for w, (a, b) in WINDOWS.items():
            if windows and w not in windows:
                continue
            ts, q = window_path(m, a, b)
            vals = []
            for per in self.eps:
                free, held0 = per[w]
                vals.append(hopper_count(sweep_times(q, ts, free), ts, held0, self.drain))
            out[w] = float(np.mean(vals))
        return out


if __name__ == "__main__":
    ev = Evaluator(sys.argv[1])
    base = ev.score(None)
    alt = ev.score(json.loads(sys.argv[2])) if len(sys.argv) > 2 else None
    print(len(ev.eps), "episodes")
    for w in WINDOWS:
        print(f"{w:10s} ghost {base[w]:6.1f}" + (f"  bumped {alt[w]:6.1f}  diff {alt[w] - base[w]:+6.1f}" if alt else ""))
