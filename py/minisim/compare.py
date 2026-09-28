"""Side-by-side timeline: MiniSim (GhostPolicy) vs the real game (cal1 arm 0) — held and score each second."""

from __future__ import annotations

import sys

import numpy as np

from minisim.calib.common import episodes
from minisim.ghost_run import run


def at(trace, T, key):
    ts = np.array([s["t"] for s in trace])
    i = int(np.searchsorted(-ts, -T, side="right")) - 1
    return trace[max(0, i)][key]


def main(t0=160, t1=0, step=1.0, n=6, arm="{}", params=None):
    import json
    sim = run(json.loads(arm), n, params=params)
    real = [d["trace"] for d in episodes("cal1", arm=0)]
    print("   t | real held  sim held | real blue sim blue | real x,z        sim x,z")
    for T in np.arange(t0, t1 - 1e-9, -step):
        rh = np.mean([at(tr, T, "held") for tr in real]); sh = np.mean([at(r["trace"], T, "held") for r in sim])
        rb = np.mean([at(tr, T, "blue") for tr in real]); sb = np.mean([at(r["trace"], T, "blue") for r in sim])
        rx = np.mean([at(tr, T, "x") for tr in real]); rz = np.mean([at(tr, T, "z") for tr in real])
        sx = np.mean([at(r["trace"], T, "x") for r in sim]); sz = np.mean([at(r["trace"], T, "z") for r in sim])
        print(f"{T:5.1f} | {rh:8.1f} {sh:8.1f} | {rb:8.1f} {sb:8.1f} | {rx:6.2f},{rz:5.2f}  {sx:6.2f},{sz:5.2f}")


if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else []
    main(*a) if a else main()
