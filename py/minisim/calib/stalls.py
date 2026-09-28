"""Stall statistics by location (same definition on real and simulated batches): the robot is commanded
(|stick| > 0.5) but moves < 0.25 m/s for >= 0.3 s; the stall lasts until it moves again at > 0.35 m/s."""

from __future__ import annotations

import collections
import sys

import numpy as np

from .common import episodes


def where(x, z):
    x, z = abs(x), abs(z)
    if 2.3 < x < 4.5 and 2.1 < z < 3.5:
        return "trench divider"
    if 3.0 < x < 4.3 and 0.55 < z < 2.5:
        return "bump"
    if x > 7.0:
        return "end wall"
    if z > 3.45:
        return "side wall"
    if 2.4 < x < 3.3 and z < 2.5:
        return "hub/bump face"
    return "open"


def stalls(batch, arm=None):
    ev = collections.defaultdict(list); n = 0
    for d in episodes(batch, arm=arm):
        n += 1; tr = d["trace"]; k = 0
        while k < len(tr):
            r = tr[k]
            if r.get("rvx") is None:
                break
            u = np.hypot(r["vx"], r["vz"]); sp = np.hypot(r["rvx"], r["rvz"])
            if u > 0.5 and sp < 0.25 and 0.5 < r["t"] < 139.5:
                j = k
                while j < len(tr) and np.hypot(tr[j]["rvx"], tr[j]["rvz"]) < 0.35:
                    j += 1
                if j - k >= 3:
                    ev[where(r["x"], r["z"])].append((j - k) * 0.099)
                k = max(j, k + 1)
            else:
                k += 1
    return n, ev


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        b, a = (spec.split(":") + [None])[:2]
        n, ev = stalls(b, None if a is None else int(a))
        tot = sum(sum(v) for v in ev.values()) / max(n, 1)
        print(f"{spec:12s} eps {n:3d}  stall s/ep {tot:5.1f}  " + "  ".join(
            f"{k}: {len(v) / n:.2f}/ep {np.mean(v):.2f}s (max {max(v):.1f})" for k, v in sorted(ev.items())))
