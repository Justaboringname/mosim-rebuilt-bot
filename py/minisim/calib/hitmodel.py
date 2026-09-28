"""Hub-shot outcome table from real dense data: P(score / rim-fall / miss) by distance to the hub node and robot
speed at launch (AutoShoot inside the zone). Written to minisim/hit_table.json for sim2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import episodes

D_BINS = [0, 2.5, 3.5, 4.0, 4.5, 5.0, 9.0]
V_BINS = [0, 0.3, 0.8, 1.2, 1.6, 4.0]


def collect(batches=("cal1", "r4", "cal2")):
    rows = []
    for bat in batches:
        for d in episodes(bat, dense=True):
            tr = d["trace"]; raw = d["dense"]["pos_cm"]; T = min(len(tr), len(raw))
            gone = raw[:, :, 0] == -32768; P = raw.astype(float) / 100
            for k in range(1, T - 40):
                a = tr[k - 1]
                if not a["b"][1] or a["x"] < 3.8 or a.get("rvx") is None:
                    continue
                for i in np.nonzero(gone[k - 1] & ~gone[k])[0]:
                    path = P[k:k + 40, i]; g = gone[k:k + 40, i]
                    if g.any():
                        path = path[:np.argmax(g)]
                    x, z, y = path[:, 0], path[:, 1], path[:, 2]
                    if len(y) == 0 or y.max() < 1.2:          # never flew: hopper-top pop / box flicker, not a shot
                        continue
                    inhub = (x > 3.0) & (x < 4.35) & (np.abs(z) < 0.7) & (y > 0.75)
                    if not inhub.any():
                        c = 2
                    else:
                        j0 = np.argmax(inhub)
                        out = np.nonzero((np.arange(len(x)) > j0) & (x < 3.05))[0]
                        if not len(out):
                            continue
                        c = 0 if y[out[0]] < 1.05 else 1
                    rows.append((np.hypot(a["x"] - 3.655, a["z"] + 0.006), np.hypot(a["rvx"], a["rvz"]), c))
    return np.array(rows, float)


if __name__ == "__main__":
    R = collect()
    tab = np.zeros((len(D_BINS) - 1, len(V_BINS) - 1, 3))
    for i in range(len(D_BINS) - 1):
        for j in range(len(V_BINS) - 1):
            m = (R[:, 0] >= D_BINS[i]) & (R[:, 0] < D_BINS[i + 1]) & (R[:, 1] >= V_BINS[j]) & (R[:, 1] < V_BINS[j + 1])
            cnt = np.bincount(R[m, 2].astype(int), minlength=3) + np.array([20.0, 1.0, 1.0])   # prior toward scoring
            tab[i, j] = cnt / cnt.sum()
            print(f"dist {D_BINS[i]}-{D_BINS[i+1]} speed {V_BINS[j]}-{V_BINS[j+1]}: n={int(m.sum()):6d}  "
                  f"score {tab[i, j, 0]:.3f} rim {tab[i, j, 1]:.3f} miss {tab[i, j, 2]:.3f}")
    json.dump({"d_bins": D_BINS, "v_bins": V_BINS, "p": tab.tolist(), "n": len(R)},
              open(Path(__file__).resolve().parents[1] / "hit_table.json", "w"))
