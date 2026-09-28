"""Side-by-side top-down replay: a human demo vs bot matches, synced on the match clock (data for replay.html).

    python -m mosimrl.replay --out /tmp/replay_data.js ../runs/ghost/show1/ghost-...-ep0.json [...]

Bot files need their .dense.npz (run_ghost --dense). Balls are stored every `--every` decisions as int16 cm (x, z, y),
gzip + base64; robot pose / held / score / buttons every decision.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from .ghost import load_rows, match_slice

ROOT = Path(__file__).resolve().parents[2]


def field_shapes() -> list[dict]:
    obbs = json.loads((ROOT / "py/minisim/field_obbs.json").read_text())
    out = []
    for o in obbs:
        c, h, ax = np.array(o["center"]), np.array(o["half"]), np.array(o["axes"])
        corners = [c + sx * h[0] * ax[0] + sy * h[1] * ax[1] + sz * h[2] * ax[2]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        pts = np.array([[p[0], p[2]] for p in corners])
        hull = _hull(pts)
        path = o["path"]
        kind = ("bump" if o["kind"] == "bump" else "hub" if "Hub" in path else "trench" if "Trench" in path
                else "tower" if "Tower" in path else "depot" if "Depot" in path else "outpost" if "Outpost" in path
                else "wall" if "Perimeter" in path else "other")
        if kind == "wall":
            continue
        out.append({"k": kind, "p": np.round(hull, 3).tolist(), "top": round(float(o["aabb_max"][1]), 2)})
    return out


def _hull(p: np.ndarray) -> np.ndarray:
    p = np.unique(np.round(p, 4), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in p[::-1]:
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    return np.array(lo[:-1] + hi[:-1])


def pack_frames(frames: list[np.ndarray]) -> dict:
    """frames: list of (n_i, 3) float arrays in metres -> gzip+b64 int16 blob plus per-frame counts."""
    counts = [len(f) for f in frames]
    flat = np.concatenate([np.clip(np.round(f * 100), -32767, 32767).astype("<i2").reshape(-1)
                           for f in frames]) if frames else np.zeros(0, "<i2")
    return {"counts": counts, "b64": base64.b64encode(gzip.compress(flat.tobytes(), 6)).decode()}


def human(path: str, every: int) -> dict:
    rows = match_slice(load_rows(path))
    t = [round(r["t"], 3) for r in rows]
    robot = [[round(r["x"], 3), round(r["z"], 3), round(r["yaw"], 1), int(r.get("held", 0)), int(r["blue"]),
              int("".join(str(int(b)) for b in r["in"]["b"]), 2)] for r in rows]
    fi, frames = [], []
    for i, r in enumerate(rows):
        if "fuel" in r:
            fi.append(i)
            frames.append(np.asarray(r["fuel"], float).reshape(-1, 3))
    return {"name": "你 · 1118", "final": int(rows[-1]["blue"]), "t": t, "robot": robot, "fi": fi, **pack_frames(frames)}


def bot(path: str, every: int, label: str) -> dict:
    d = json.loads(Path(path).read_text())
    tr = d["trace"]
    dense = Path(path.replace(".json", ".dense.npz"))
    pos = np.load(dense)["pos_cm"].astype(float) / 100.0 if dense.exists() else None
    n = min(len(tr), len(pos)) if pos is not None else len(tr)
    t = [round(s["t"], 3) for s in tr[:n]]
    robot = [[round(s["x"], 3), round(s["z"], 3), round(s["yaw"], 1), int(s["held"]), int(s["blue"]),
              int("".join(str(int(b)) for b in s["b"]), 2)] for s in tr[:n]]
    fi, frames = [], []
    if pos is not None:
        for i in range(0, n, every):
            a = pos[i]
            a = a[a[:, 0] > -327]
            fi.append(i); frames.append(a)
    else:                                   # no dense file: the trace keeps the balls every 5th decision
        for i, s in enumerate(tr[:n]):
            if s.get("fuel"):
                fi.append(i); frames.append(np.asarray(s["fuel"], float).reshape(-1, 3))
    return {"name": label, "final": int(d["score"]), "t": t, "robot": robot, "fi": fi, **pack_frames(frames)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--labels", default="")
    ap.add_argument("--demo", default=str(ROOT / "run/demos/demo-20260926-015440-m1.jsonl"))
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels = args.labels.split(",") if args.labels else [f"bot {i}" for i in range(len(args.bots))]
    data = {"field": field_shapes(), "human": human(args.demo, args.every),
            "bots": [bot(p, args.every, l) for p, l in zip(args.bots, labels)]}
    Path(args.out).write_text("const REPLAY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
