"""Shared loaders for MiniSim calibration: real-game traces (runs/ghost/<batch>/*-ep*.json) and dense ball arrays."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[3] / "runs/ghost"


def episodes(batch: str, arm: int | None = None, dense: bool = False):
    S = json.load(open(RUNS / batch / "summary.json"))
    arms = {e["ep"]: e["arm"] for e in S["episodes"]}
    for f in sorted(glob.glob(str(RUNS / batch / "*-ep*.json"))):
        ep = int(f.split("-ep")[1].split(".")[0])
        if arm is not None and arms.get(ep) != arm:
            continue
        d = json.load(open(f))
        if dense:
            p = Path(f).with_suffix("").as_posix() + ".dense.npz"
            d["dense"] = dict(np.load(p)) if Path(p).exists() else None
        d["ep"], d["arm"] = ep, arms.get(ep)
        yield d


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0
