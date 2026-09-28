"""Shooter launch kinematics from real dense data -> minisim/launch_table.json.

Hightide sets flywheel rpm and hood from the 3-D distance d_hub between the turret (robot centre, y 0.476) and the
shoot-on-move-adjusted hub node (hub - v_robot * 1.15 s, y 1.938) -- for pass shots too (hightide_contract.md §4:
pass power is keyed on the HUB distance, not the pass node). The ball leaves the shooter at ~0.6 m with the robot's
velocity plus a launch velocity aimed (±2.5°) at the shoot-on-move-adjusted target. This measures, per d_hub bin, the
launch speed relative to the robot and the elevation, for hub shots (in zone) and pass shots (out of zone).

    python -m minisim.calib.launch cal2 r4 hr1b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .common import episodes

HUB = np.array([3.655, 1.938, -0.006])
PASS = [(5.91, -1.675), (5.91, 1.675)]
G, DT = 9.81, 0.099
D_EDGES = np.arange(1.5, 9.51, 0.5)


def launches(batch, maxn=None):
    out = []
    for n, d in enumerate(episodes(batch, dense=True)):
        if (maxn and n >= maxn) or d.get("dense") is None:
            continue
        tr = d["trace"]; raw = d["dense"]["pos_cm"]; T = min(len(tr), len(raw))
        gone = raw[:, :, 0] == -32768
        P = raw.astype(float) / 100
        t3 = np.arange(3) * DT
        for k in range(1, T - 4):
            for i in np.nonzero(gone[k - 1] & ~gone[k])[0]:
                if gone[k:k + 4, i].any():
                    continue
                x, z, y = P[k:k + 3, i, 0], P[k:k + 3, i, 1], P[k:k + 3, i, 2]
                if P[k:k + 4, i, 2].max() < 0.9:            # hopper pop, not a shot
                    continue
                a = tr[k - 1]
                rvx, rvz = a.get("rvx", 0.0), a.get("rvz", 0.0)
                vx = np.polyfit(t3, x, 1)[0] - rvx
                vz = np.polyfit(t3, z, 1)[0] - rvz
                vy = np.polyfit(t3, y + 0.5 * G * t3 ** 2, 1)[0]
                aim = HUB - np.array([rvx, 0.0, rvz]) * 1.15
                dh = float(np.linalg.norm(aim - np.array([a["x"], 0.476, a["z"]])))
                zone = a["x"] > 3.738
                if zone and not a["b"][2]:
                    tx, tz = HUB[0], HUB[2]
                else:
                    tx, tz = PASS[0] if a["z"] < 0 else PASS[1]
                az_aim = np.arctan2(tz - rvz * 1.15 - a["z"], tx - rvx * 1.15 - a["x"])
                az_err = (np.degrees(np.arctan2(vz, vx) - az_aim) + 180) % 360 - 180
                out.append((int(zone and not a["b"][2]), dh, float(np.sqrt(vx * vx + vz * vz + vy * vy)),
                            float(np.degrees(np.arctan2(vy, np.hypot(vx, vz)))), float(az_err)))
    return np.array(out, float)


def table(batches):
    A = np.concatenate([launches(b) for b in batches])
    res = {"d_edges": D_EDGES.tolist(), "batches": list(batches)}
    for kind, name in ((1, "hub"), (0, "pass")):
        m = A[:, 0] == kind
        rows = []
        for lo, hi in zip(D_EDGES[:-1], D_EDGES[1:]):
            mm = m & (A[:, 1] >= lo) & (A[:, 1] < hi)
            if mm.sum() < 30:
                rows.append(None); continue
            sp, el = A[mm, 2], A[mm, 3]
            rows.append({"d": float(np.median(A[mm, 1])), "n": int(mm.sum()),
                         "v": float(np.median(sp)), "v_sd": float((np.percentile(sp, 84) - np.percentile(sp, 16)) / 2),
                         "el": float(np.median(el)), "el_sd": float((np.percentile(el, 84) - np.percentile(el, 16)) / 2)})
        az = A[m, 4]
        res[name] = {"bins": rows, "az_sd": float((np.percentile(az, 84) - np.percentile(az, 16)) / 2),
                     "az_mu": float(np.median(az)), "n": int(m.sum())}
    return res


if __name__ == "__main__":
    res = table(sys.argv[1:] or ["cal2", "r4", "hr1b"])
    Path(__file__).resolve().parents[1].joinpath("launch_table.json").write_text(json.dumps(res, indent=1))
    for name in ("hub", "pass"):
        print(f"{name}: n {res[name]['n']}  az mu {res[name]['az_mu']:+.2f} sd {res[name]['az_sd']:.2f}")
        for r in res[name]["bins"]:
            if r:
                print(f"  d {r['d']:4.2f} n {r['n']:5d}  v {r['v']:5.2f} ± {r['v_sd']:4.2f}   el {r['el']:5.1f} ± {r['el_sd']:4.1f}")
