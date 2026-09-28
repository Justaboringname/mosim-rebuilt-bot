"""In-game calibration through the bridge — run this before any CMA-ES run on the real game.

    python -m mosimrl.calibrate --port 47414 --out runs/calib

Measures what the static analysis could not (docs/policy-facts, reconcile list):
  drive     overideInput mapping, top speed, acceleration, yaw-rate sign
  fire      AutoShoot feed rate and scoring with the 8 preloads, stationary at a few distances
  trench    whether 4414 passes under the trench bar
  bump      bump crossing time / success
  intake    pickup rate driving intake-first through the centre grid
Each test is its own fresh match (reset). Results -> <out>/calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from .client import INTAKE, SHOOT, MoSimClient

NONE = [False] * 5


def buttons(*idx):
    b = [False] * 5
    for i in idx:
        b[i] = True
    return b


def goto(c, s, x, z, speed=1.0, b=NONE, tol=0.15, timeout=12.0):
    t0 = s["gameTime"]
    while s["gameTime"] - t0 < timeout:
        r = s["robot"]
        dx, dz = x - r["x"], z - r["z"]
        d = math.hypot(dx, dz)
        if d < tol:
            break
        k = speed * min(1.0, d / 0.6)
        s = c.act(dx / d * k, dz / d * k, 0.0, b)
    return s


def hold(c, s, seconds, vx=0.0, vz=0.0, rot=0.0, b=NONE, every=None):
    t0 = s["gameTime"]
    rows = []
    while s["gameTime"] - t0 < seconds and not s.get("done"):
        s = c.act(vx, vz, rot, b)
        if every:
            rows.append(every(s))
    return s, rows


def test_drive(c):
    s = c.reset()
    out = {}
    runs = {"plus_x": ((5.0, -3.0), (1.0, 0.0), 1.0), "plus_z": ((6.0, -3.0), (0.0, 1.0), 1.2)}
    for name, (start, (vx, vz), secs) in runs.items():
        s = goto(c, s, *start, speed=0.6)
        s, _ = hold(c, s, 1.0)
        s, rows = hold(c, s, secs, vx=vx, vz=vz,
                       every=lambda st: (st["gameTime"], st["robot"]["x"], st["robot"]["z"], st["robot"]["vx"], st["robot"]["vz"]))
        speeds = [math.hypot(r[3], r[4]) for r in rows]
        vmax = max(speeds)
        out[name] = {"cmd": [vx, vz], "dx": rows[-1][1] - rows[0][1], "dz": rows[-1][2] - rows[0][2], "vmax": vmax,
                     "t_to_90pct": next((r[0] - rows[0][0] for r, v in zip(rows, speeds) if v > 0.9 * vmax), None)}
        s, _ = hold(c, s, 0.8)
    yaw0 = s["robot"]["yaw"]
    s, _ = hold(c, s, 1.0, rot=0.5)
    out["rot_plus_half_1s"] = {"yaw_before": yaw0, "yaw_after": s["robot"]["yaw"],
                               "note": "expected: yaw decreases for rot > 0"}
    return out


def test_fire(c, spots=((4.6, 0.0), (5.5, 0.0), (6.5, 0.0), (5.5, 2.0))):
    out = []
    for x, z in spots:
        s = c.reset()
        s = goto(c, s, x, z, 0.8)
        s, _ = hold(c, s, 0.5)
        held0, blue0, t0 = s["held"], s["blue"], s["gameTime"]
        s, rows = hold(c, s, 6.0, b=buttons(SHOOT), every=lambda st: (st["gameTime"], st["held"], st["blue"]))
        empty_t = next((r[0] - t0 for r in rows if r[1] == 0), None)
        s, _ = hold(c, s, 3.0)   # let the last balls land
        out.append({"spot": [x, z], "hub_dist": math.hypot(x - 3.655, z), "held0": held0,
                    "seconds_to_empty": empty_t, "scored": s["blue"] - blue0,
                    "first_score_after_s": next((r[0] - t0 for r in rows if r[2] > blue0), None)})
    return out


def test_crossing(c, lane_z, label):
    s = c.reset()
    s = goto(c, s, 4.8, lane_z, 0.8)
    x0, t0 = s["robot"]["x"], s["gameTime"]
    s, rows = hold(c, s, 2.0, vx=-0.8, every=lambda st: (st["gameTime"], st["robot"]["x"], st["robot"]["y"],
                                                          st["robot"]["pitch"], st["robot"]["roll"]))
    crossed = next((r[0] - t0 for r in rows if r[1] < 2.9), None)
    return {"lane": label, "z": lane_z, "x_start": x0, "x_end": s["robot"]["x"], "crossed_after_s": crossed,
            "max_pitch": max(abs(((r[3] + 180) % 360) - 180) for r in rows),
            "min_y": min(r[2] for r in rows)}


def test_intake(c):
    s = c.reset()
    s = goto(c, s, 4.8, 1.5, 0.8)
    s = goto(c, s, 2.8, 1.5, 0.8)          # over the bump into the neutral zone
    s = goto(c, s, 1.0, 2.4, 0.6)
    held0, t0 = s["held"], s["gameTime"]
    out = []
    for speed in (0.3, 0.6, 1.0):
        s, rows = hold(c, s, 1.6, vz=-speed, b=buttons(INTAKE),
                       every=lambda st: (st["gameTime"], st["held"]))
        out.append({"speed_cmd": speed, "held_start": rows[0][1], "held_end": rows[-1][1],
                    "balls_per_s": (rows[-1][1] - rows[0][1]) / max(1e-6, rows[-1][0] - rows[0][0])})
        s = goto(c, s, 1.0 - 0.3 * len(out), 2.4, 0.6, b=buttons(INTAKE))
    return {"held_before": held0, "runs": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47414)
    ap.add_argument("--out", default="../runs/calib")
    ap.add_argument("--only", default="", help="comma list of: drive,fire,trench,bump,intake")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = MoSimClient(port=args.port)
    info = c.hello()
    print("bridge:", info)
    only = set(args.only.split(",")) if args.only else {"drive", "fire", "trench", "bump", "intake"}
    res = {"bridge": info, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    if "drive" in only:
        res["drive"] = test_drive(c); print("drive", res["drive"])
    if "fire" in only:
        res["fire"] = test_fire(c); print("fire", res["fire"])
    if "trench" in only:
        res["trench"] = [test_crossing(c, -3.39, "trench-"), test_crossing(c, 3.39, "trench+")]; print("trench", res["trench"])
    if "bump" in only:
        res["bump"] = [test_crossing(c, -1.52, "bump-"), test_crossing(c, 1.52, "bump+")]; print("bump", res["bump"])
    if "intake" in only:
        res["intake"] = test_intake(c); print("intake", res["intake"])
    (out / "calibration.json").write_text(json.dumps(res, indent=1))
    c.release()
    print("wrote", out / "calibration.json")


if __name__ == "__main__":
    main()
