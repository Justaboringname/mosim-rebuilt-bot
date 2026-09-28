"""Ball-flow comparison: where the balls are over the match, human demos vs bot traces (needs fuel snapshots).

    python -m mosimrl.flow ../runs/ghost/fl8

For each sampled clock time: STOCK (loose floor balls in the blue zone, x > 4.25), NEUTRAL (floor balls with
|x| < 3.05), RED side floor balls (x < -4.25), airborne balls, and held — median over bot episodes vs each demo.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

from .ghost import load_rows, match_slice

TIMES = [150, 140, 135, 128, 120, 112, 106, 100, 95, 90, 85, 80, 75, 68, 62, 56, 50, 45, 40, 35, 30, 20, 10, 3]
DEMOS = ["demo-20260926-015440-m1", "demo-20260926-014353-m7", "demo-20260926-020926-m1"]


def counts(fuel) -> dict:
    a = np.asarray(fuel or [], np.float32).reshape(-1, 3)
    if not len(a):
        return {"stock": 0, "neutral": 0, "red": 0, "air": 0}
    x, y = a[:, 0], a[:, 2]
    floor = y < 0.25
    return {"stock": int((floor & (x > 4.25)).sum()), "neutral": int((floor & (np.abs(x) < 3.05)).sum()),
            "red": int((floor & (x < -4.25)).sum()), "air": int((~floor).sum())}


def series(rows, key_t="t"):
    snaps = [r for r in rows if "fuel" in r]
    out = {}
    for tt in TIMES:
        r = min(snaps, key=lambda r: abs(r[key_t] - tt))
        out[tt] = {**counts(r["fuel"]), "held": r.get("held", 0)}
    return out


def main() -> None:
    d = Path(sys.argv[1])
    bots = []
    for f in glob.glob(str(d / "*.json")):
        if f.endswith("summary.json"):
            continue
        tr = json.load(open(f))["trace"]
        if any("fuel" in s for s in tr):
            bots.append(series(tr))
    humans = {n: series(match_slice(load_rows(Path(__file__).resolve().parents[2] / f"run/demos/{n}.jsonl"))) for n in DEMOS}
    print(f"bot episodes with fuel: {len(bots)}")
    for key in ("stock", "neutral", "held", "air", "red"):
        print(f"\n{key}:  t " + " ".join(f"{t:>4}" for t in TIMES))
        med = [int(np.median([b[t][key] for b in bots])) for t in TIMES] if bots else []
        print(f"   bot(med) " + " ".join(f"{v:>4}" for v in med))
        for n, h in humans.items():
            print(f"   {n[-6:]:>8} " + " ".join(f"{h[t][key]:>4}" for t in TIMES))


if __name__ == "__main__":
    main()
