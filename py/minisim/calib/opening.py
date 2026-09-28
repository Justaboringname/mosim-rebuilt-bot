"""Fast check of the first auto sweep: intakes by t = 157.0 / 156.5 / 155.5 in MiniSim under parameter variants
(real cal1 arm 0: 75.8 / 102.8 / 117.8)."""

from __future__ import annotations

import json
import sys

import numpy as np

from mosimrl.episode import BallTracker, MatchClock
from mosimrl.ghost import Ghost
from mosimrl.ghost_policy import GhostPolicy, default_ghost_params

from ..sim import MiniSimClient

DEMO = "../run/demos/demo-20260926-015440-m1.jsonl"


def first_sweep(params, n=6, until=155.4):
    g = Ghost.from_file(DEMO)
    out = []
    for i in range(n):
        c = MiniSimClient(params, seed=100 + i)
        pol = GhostPolicy(g, default_ghost_params()); pol.reset()
        bt = BallTracker(); s = c.reset()
        while s["t"] > until:
            bt.update(s)
            vx, vz, rot, b, _ = pol.act(s)
            s = c.act(vx, vz, rot, b)
        I = np.array(bt.intakes) if bt.intakes else np.zeros((0, 8))
        out.append([(I[:, 0] >= T).sum() for T in (157.0, 156.5, 155.5)])
    return np.array(out).mean(0)


if __name__ == "__main__":
    for spec in sys.argv[1:] or ["{}"]:
        print(spec, first_sweep(json.loads(spec)).round(1), " real [75.8 102.8 117.8]")
