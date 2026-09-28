"""Launch tracking from dense ball arrays: every ball that leaves the hopper, its flight, and where it ends up.

Classes: 'hub' = entered the hub (passes the scorer region x 3.05..4.3, |z| < 0.65 above y 0.7 and then leaves it
toward -x through the exits), 'land' = came down elsewhere (pass or miss). Records launch pose/buttons, time to
hub exit, exit velocity, landing point."""

from __future__ import annotations

import numpy as np

from .common import episodes


def track(batch="cal1", horizon=45):
    out = []
    for d in episodes(batch, dense=True):
        tr = d["trace"]; raw = d["dense"]["pos_cm"]; T = min(len(tr), len(raw))
        gone = raw[:, :, 0] == -32768
        P = raw.astype(float) / 100
        for k in range(1, T - 1):
            new = np.nonzero(gone[k - 1] & ~gone[k])[0]
            for i in new:
                a = tr[k - 1]
                path = P[k:min(T, k + horizon), i]
                g = gone[k:min(T, k + horizon), i]
                if g.any():
                    path = path[:np.argmax(g)]
                if len(path) < 2:
                    continue
                x, z, y = path[:, 0], path[:, 1], path[:, 2]
                inhub = (x > 3.0) & (x < 4.35) & (np.abs(z) < 0.7) & (y > 0.6)
                cls, t_exit, vexit, land = "land", None, None, None
                if inhub.any():
                    j0 = np.argmax(inhub)
                    after = np.nonzero((np.arange(len(x)) > j0) & (x < 3.0) & (y < 0.9))[0]
                    if len(after):
                        cls = "hub"
                        j = after[0]
                        t_exit = j * 0.099
                        if j + 1 < len(x):
                            vexit = ((x[j + 1] - x[j]) / 0.099, (z[j + 1] - z[j]) / 0.099)
                    else:
                        cls = "hub?"
                air = np.nonzero(y > 0.5)[0]
                if len(air):
                    dn = np.nonzero((np.arange(len(y)) > air[0]) & (y < 0.2))[0]
                    if len(dn):
                        land = (x[dn[0]], z[dn[0]], dn[0] * 0.099)
                out.append({"ep": d["ep"], "arm": d["arm"], "t": tr[k]["t"], "id": int(i), "rx": a["x"], "rz": a["z"],
                            "ryaw": a["yaw"], "rv": (a.get("rvx"), a.get("rvz")), "b": a["b"], "held": a["held"],
                            "x0": x[0], "z0": z[0], "y0": y[0], "cls": cls, "t_exit": t_exit, "vexit": vexit,
                            "land": land, "zone": a["x"] > 3.8})
    return out


if __name__ == "__main__":
    L = track()
    import collections
    print("launches", len(L), collections.Counter((l["cls"], "shoot" if l["b"][1] else ("pass" if l["b"][2] else "other"),
                                                  "zone" if l["zone"] else "out") for l in L).most_common())
    hub = [l for l in L if l["cls"] == "hub"]
    te = np.array([l["t_exit"] for l in hub])
    print("launch->hub exit s: p10/50/90", np.percentile(te, [10, 50, 90]).round(2))
    ve = np.array([l["vexit"] for l in hub if l["vexit"]])
    print("exit velocity vx p10/50/90", np.percentile(ve[:, 0], [10, 50, 90]).round(2), " vz p10/50/90",
          np.percentile(ve[:, 1], [10, 50, 90]).round(2))
    z0 = np.array([l["x0"] for l in hub]); print("exit x", np.percentile([l["land"][0] for l in hub if l["land"]], [10, 50, 90]))
