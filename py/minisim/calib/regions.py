"""Ball conservation by region over time, real vs sim (dense batches): held, in hub, airborne, blue-zone floor,
neutral floor, red-zone floor, blue corral/chute (x > 8.27), red outpost (x < -8.27)."""

from __future__ import annotations

import sys

import numpy as np

from .common import episodes

NAMES = ["held", "hub", "air", "blue", "neutral", "red", "blue_out", "red_out"]


def counts(P, gone):
    x, z, y = P[..., 0], P[..., 1], P[..., 2]
    ok = ~gone
    hub = ok & (x > 3.0) & (x < 4.35) & (np.abs(z) < 0.7) & (y > 0.75)
    blue_out = ok & (x > 8.27); red_out = ok & (x < -8.27)
    air = ok & (y > 0.3) & ~hub & ~blue_out & ~red_out
    fl = ok & ~hub & ~air & ~blue_out & ~red_out
    return np.stack([gone.sum(-1), hub.sum(-1), air.sum(-1), (fl & (x > 3.8)).sum(-1), (fl & (np.abs(x) <= 3.8)).sum(-1),
                     (fl & (x < -3.8)).sum(-1), blue_out.sum(-1), red_out.sum(-1)], -1)


def timeline(batch, arm, times, maxn=8):
    out = []
    for n, d in enumerate(episodes(batch, arm=arm, dense=True)):
        if n >= maxn:
            break
        raw = d["dense"]["pos_cm"]; P = raw.astype(float) / 100; gone = raw[:, :, 0] == -32768; t = d["dense"]["t"]
        ks = [int(np.argmin(np.abs(t - T))) for T in times]
        out.append(counts(P[ks], gone[ks]))
    return np.mean(out, 0)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    arm = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    times = [158, 150, 140, 130, 118, 106, 92, 80, 68, 56, 42, 31, 21, 10, 1]
    A, B = timeline(a, arm, times), timeline(b, arm, times)
    print("t    | " + " ".join(f"{n:>11s}" for n in NAMES))
    for i, T in enumerate(times):
        print(f"{T:4d} | " + " ".join(f"{A[i, j]:5.0f}/{B[i, j]:<5.0f}" for j in range(len(NAMES))))
