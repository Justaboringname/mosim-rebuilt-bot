"""Rolling of floor balls that nobody touches: speed decay per decision (0.099 s) for balls on the floor moving
> 0.05 m/s and more than 1.2 m from the robot."""

from __future__ import annotations

import numpy as np

from .common import episodes


def samples(batch="cal1"):
    V0, V1, X, Z = [], [], [], []
    for d in episodes(batch, dense=True):
        tr = d["trace"]; raw = d["dense"]["pos_cm"]; T = min(len(tr), len(raw))
        gone = raw[:, :, 0] == -32768
        P = raw.astype(float) / 100
        for k in range(1, T - 1):
            a = tr[k]
            ok = ~gone[k - 1] & ~gone[k] & ~gone[k + 1] & (P[k, :, 2] < 0.1) & (P[k - 1, :, 2] < 0.1) & (P[k + 1, :, 2] < 0.1)
            far = np.hypot(P[k, :, 0] - a["x"], P[k, :, 1] - a["z"]) > 1.2
            m = ok & far
            v0 = (P[k, m, :2] - P[k - 1, m, :2]) / 0.099
            v1 = (P[k + 1, m, :2] - P[k, m, :2]) / 0.099
            s0 = np.linalg.norm(v0, axis=1)
            sel = s0 > 0.05
            V0.append(v0[sel]); V1.append(v1[sel]); X.append(P[k, m, 0][sel]); Z.append(P[k, m, 1][sel])
    return np.concatenate(V0), np.concatenate(V1), np.concatenate(X), np.concatenate(Z)


if __name__ == "__main__":
    v0, v1, x, z = samples()
    s0, s1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
    print("samples", len(s0))
    for lo, hi in [(0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 5)]:
        m = (s0 >= lo) & (s0 < hi)
        if m.sum() > 50:
            ds = s1[m] - s0[m]
            print(f"speed {lo:.2f}-{hi:.2f}: n={m.sum():6d}  mean dv/decision {ds.mean():+.3f}  median {np.median(ds):+.3f}"
                  f"  -> decel {-np.median(ds) / 0.099:.2f} m/s^2   ratio s1/s0 median {np.median(s1[m] / s0[m]):.3f}")
