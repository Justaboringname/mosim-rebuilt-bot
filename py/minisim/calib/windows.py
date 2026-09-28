"""Per-window intakes / launches / score, real batch vs simulated batch (same trace format)."""

from __future__ import annotations

import sys

import numpy as np

from .common import episodes

W = [(160, 140), (140, 130), (130, 105), (105, 80), (80, 55), (55, 30), (30, -5)]


def table(batch, arm=0):
    rows = []
    for d in episodes(batch, arm=arm):
        I = np.array(d["intakes"]) if d["intakes"] else np.zeros((0, 8))
        L = np.array(d["launches"]) if d["launches"] else np.zeros((0, 5))
        tr = d["trace"]; t = np.array([r["t"] for r in tr]); b = np.array([r["blue"] for r in tr])
        row = []
        for hi, lo in W:
            m = np.nonzero((t <= hi) & (t > lo))[0]
            sc = b[m[-1]] - b[max(m[0] - 1, 0)] if len(m) else 0
            if lo < 0:
                sc += b[-1] - b[m[-1]]
            row += [((I[:, 0] <= hi) & (I[:, 0] > lo)).sum(), ((L[:, 0] <= hi) & (L[:, 0] > lo)).sum(), sc]
        rows.append(row)
    return np.array(rows, float).mean(0)


if __name__ == "__main__":
    A = table(sys.argv[1]); B = table(sys.argv[2])
    print(f"{'window':10s} | {'intake':>13s} | {'launch':>13s} | {'score':>13s}")
    for k, (hi, lo) in enumerate(W):
        print(f"{hi:4d}-{max(lo, 0):<4d}  | {A[3 * k]:6.0f} {B[3 * k]:6.0f} | {A[3 * k + 1]:6.0f} {B[3 * k + 1]:6.0f} | "
              f"{A[3 * k + 2]:6.0f} {B[3 * k + 2]:6.0f}")
    print(f"{'total':10s} | {A[0::3].sum():6.0f} {B[0::3].sum():6.0f} | {A[1::3].sum():6.0f} {B[1::3].sum():6.0f} | "
          f"{A[2::3].sum():6.0f} {B[2::3].sum():6.0f}")
