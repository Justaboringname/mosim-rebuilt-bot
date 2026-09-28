"""Break a human demo (run/demos/*.jsonl from Recorder.cs) into the strategy a scripted policy can copy.

    python -m mosimrl.analyze_demos [files...] --out ../runs/demos

Per match: score timeline vs hub shifts, time split by zone, bump crossings (lane, direction, duration),
collection episodes (where, how many balls, how fast), shooting episodes (where, distance to hub, balls
fired, points scored), and cycle statistics. Writes one PNG per match plus summary.json / summary.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from . import field as F
from .ghost import load_rows, match_slice
from .shifts import blue_active

INTAKE, SHOOT, PASS, MANUAL, SPECIAL = range(5)


def load(path: str) -> list[dict]:
    """The one real match in a recording (quick restarts and post-buzzer rows trimmed; see ghost.match_slice)."""
    return match_slice(load_rows(path))


def zone_of(x: float) -> str:
    if x >= F.ZONE_LINE:
        return "blue_zone"
    if x > F.NEUTRAL_FACE:
        return "band"
    if x > -F.NEUTRAL_FACE:
        return "neutral"
    return "red_side"


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def analyse(rows: list[dict]) -> dict:
    t = np.array([r["t"] for r in rows])
    x = np.array([r["x"] for r in rows]); z = np.array([r["z"] for r in rows])
    held = np.array([r["held"] for r in rows]); blue = np.array([r["blue"] for r in rows])
    b = np.array([r.get("in", {}).get("b", [0] * 5) for r in rows])
    speed = np.hypot([r["vx"] for r in rows], [r["vz"] for r in rows])
    dt = 0.1
    won = any(r.get("wonAuto") == 1 for r in rows if r["t"] <= 130)

    zones = [zone_of(v) for v in x]
    zone_time = {k: round(zones.count(k) * dt, 1) for k in ("blue_zone", "band", "neutral", "red_side")}

    # crossings: contiguous stretches in the band that exit on the other side
    crossings = []
    for s, e in segments(np.array([zn == "band" for zn in zones])):
        before = zones[s - 1] if s > 0 else None
        after = zones[e] if e < len(zones) else None
        if before and after and before != after:
            crossings.append({"t": round(float(t[s]), 1), "from": before, "to": after,
                              "z": round(float(np.mean(z[s:e])), 2), "secs": round((e - s) * dt, 1),
                              "entry_speed": round(float(speed[max(0, s - 1)]), 2)})

    # collection: intake held; shooting: AutoShoot/ManualShoot/AutoPass held
    collect = []
    for s, e in segments(b[:, INTAKE] == 1):
        gained = int(held[min(e, len(held) - 1)] - held[s])
        collect.append({"t": round(float(t[s]), 1), "secs": round((e - s) * dt, 1), "gained": gained,
                        "x": round(float(np.mean(x[s:e])), 2), "z": round(float(np.mean(z[s:e])), 2),
                        "zone": zone_of(float(np.mean(x[s:e]))), "mean_speed": round(float(np.mean(speed[s:e])), 2)})
    shots = []
    shoot_mask = (b[:, SHOOT] == 1) | (b[:, MANUAL] == 1) | (b[:, PASS] == 1)
    for s, e in segments(shoot_mask):
        e2 = min(len(rows) - 1, e + 20)                       # score lands ~1.8 s later
        shots.append({"t": round(float(t[s]), 1), "secs": round((e - s) * dt, 1),
                      "fired": int(held[s] - held[min(e, len(held) - 1)]), "scored": int(blue[e2] - blue[s]),
                      "x": round(float(np.mean(x[s:e])), 2), "z": round(float(np.mean(z[s:e])), 2),
                      "hub_dist": round(float(np.mean(np.hypot(x[s:e] - F.HUB_NODE[0], z[s:e] - F.HUB_NODE[1]))), 2),
                      "moving": round(float(np.mean(speed[s:e])), 2),
                      "button": "manual" if b[s:e, MANUAL].any() else ("pass" if b[s:e, PASS].any() else "auto"),
                      "hub_active": bool(blue_active(float(t[s]), won))})

    # points per window
    windows = [(160, 130), (130, 105), (105, 80), (80, 55), (55, 30), (30, 0)]
    per_window = []
    for hi, lo in windows:
        m = (t <= hi) & (t > lo)
        if m.any():
            idx = np.where(m)[0]
            per_window.append({"window": f"{hi}-{lo}", "blue_active": blue_active((hi + lo) / 2, won),
                               "points": int(blue[idx[-1]] - blue[idx[0]]),
                               "time_in_zone": round(float(np.mean([zones[i] == 'blue_zone' for i in idx])), 2)})

    return {"final": int(blue[-1]), "auto": int(rows[-1].get("blueAuto", 0)), "won_auto": won,
            "duration_s": round(len(rows) * dt, 1), "zone_time_s": zone_time, "per_window": per_window,
            "crossings": crossings, "collect": collect, "shots": shots,
            "max_held": int(held.max()), "mean_speed_moving": round(float(speed[speed > 0.3].mean()) if (speed > 0.3).any() else 0.0, 2)}


def plot(rows: list[dict], a: dict, png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    x = np.array([r["x"] for r in rows]); z = np.array([r["z"] for r in rows])
    held = np.array([r["held"] for r in rows]); t = np.array([r["t"] for r in rows]); blue = np.array([r["blue"] for r in rows])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [1.1, 0.6]})

    ax.add_patch(Rectangle((-8.27, -4.03), 16.54, 8.06, fill=False, lw=1.5, color="#444"))
    for (x0, x1, z0, z1), c in [(F.HUB_BOX, "#1f5fbf"), ((-4.25, -3.05, -0.62, 0.62), "#bf1f1f"),
                                (F.TOWER_KEEPOUT, "#888"), (F.DEPOT_BOX, "#bbb")]:
        ax.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, color=c, alpha=0.35))
    for sgn in (1, -1):
        ax.add_patch(Rectangle((3.07, sgn * 0.6 if sgn > 0 else -2.45), 1.15, 1.85, color="#e0a030", alpha=0.3))
        ax.add_patch(Rectangle((-4.22, sgn * 0.6 if sgn > 0 else -2.45), 1.15, 1.85, color="#e0a030", alpha=0.3))
    ax.axvline(F.ZONE_LINE, color="#1f5fbf", ls="--", lw=0.8); ax.axvline(-F.ZONE_LINE, color="#bf1f1f", ls="--", lw=0.8)
    sc = ax.scatter(x, z, c=held, s=4, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="balls carried")
    for s in a["shots"]:
        ax.plot(s["x"], s["z"], "r*", ms=8)
    for c in a["collect"]:
        ax.plot(c["x"], c["z"], "g^", ms=6)
    ax.set_xlim(-8.6, 8.6); ax.set_ylim(-4.3, 4.3); ax.set_aspect("equal")
    ax.set_title(f"{title}  —  final {a['final']}  (red * = shooting, green ^ = intake)")
    ax.set_xlabel("x (+x = blue end)"); ax.set_ylabel("z")

    won = a["won_auto"]
    for hi in np.arange(160, 0, -0.5):
        if not blue_active(hi, won):
            ax2.axvspan(160 - hi, 160 - hi + 0.5, color="#ddd", lw=0)
    ax2.plot(160 - t, blue, label="blue score")
    ax2.plot(160 - t, held, label="balls carried")
    ax2.set_xlabel("match time elapsed (s)  — grey = blue hub inactive"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default="../runs/demos")
    ap.add_argument("--min-records", type=int, default=1200, help="skip aborted matches shorter than this (0.1 s each)")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(str(Path(__file__).resolve().parents[2] / "run/demos/*.jsonl")))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in files:
        rows = load(f)
        if len(rows) < args.min_records:
            print(f"skip {Path(f).name}: {len(rows)} records (aborted match?)")
            continue
        a = analyse(rows)
        png = out / (Path(f).stem + ".png")
        plot(rows, a, png, Path(f).stem)
        a["file"] = Path(f).name
        summary.append(a)
        print(f"{Path(f).name}: final {a['final']}  zones {a['zone_time_s']}  crossings {len(a['crossings'])}  "
              f"intake bouts {len(a['collect'])}  shot bouts {len(a['shots'])}  -> {png.name}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
