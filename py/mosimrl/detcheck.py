"""Determinism check: restore the same snapshot into two instances, drive both with identical actions, and compare
their states step by step (robot pose, held, score, every ball).

    python -m mosimrl.detcheck --game-args "-job-worker-count 0" --port0 47800

A drives with the ghost tracker to --t-snap and snapshots. B restores and drives with the tracker (actions recorded);
C restores and replays B's actions open loop. Then B restores again and replays too (same-instance repeat).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .episode import MatchClock
from .gamectl import Game
from .ghost import Ghost
from .ghost_policy import default_ghost_params
from .planner import play_to
from .run_ghost import DEMO_KEYS, make_policy


def vec(s: dict) -> tuple[np.ndarray, np.ndarray]:
    r = s["robot"]
    head = np.array([r["x"], r["z"], r["yaw"], s.get("held", 0), s.get("blue", 0)], float)
    fuel = np.asarray(s.get("fuel") or [], float).reshape(-1, 3)
    fid = np.asarray(s.get("fid") or [], int)
    full = np.full((600, 3), np.nan)
    if len(fid):
        full[fid] = fuel
    return head, full


def drive(g: Game, blob: str, steps: int, pol=None, actions=None):
    s = g.client.restore(blob)
    clock = MatchClock(); s["t"] = clock(s)
    acts, traj = [], [vec(s)]
    for k in range(steps):
        if actions is None:
            a = pol.act(s)[:4]
        else:
            a = actions[k]
        acts.append(a)
        s = g.client.act(*a, lite=False)
        s["t"] = clock(s)
        traj.append(vec(s))
        if s.get("done"):
            break
    return acts, traj


def compare(t1, t2, label):
    n = min(len(t1), len(t2))
    first = None
    for k in range(n):
        dh = np.abs(t1[k][0] - t2[k][0]).max()
        db = np.nanmax(np.abs(t1[k][1] - t2[k][1])) if np.isfinite(t1[k][1]).any() else 0.0
        if first is None and (dh > 0 or db > 0):
            first = (k, dh, db)
    k = n - 1
    dh = np.abs(t1[k][0] - t2[k][0])
    db = np.nanmax(np.abs(t1[k][1] - t2[k][1]))
    print(f"{label}: first difference at step {first[0] if first else None}"
          + (f" (robot {first[1]:.2e}, ball {first[2]:.2e})" if first else "")
          + f"; after {n - 1} steps: robot |d| {np.round(dh[:3], 4).tolist()} held/blue {dh[3]:.0f}/{dh[4]:.0f}, max ball |d| {db:.3f} m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port0", type=int, default=47800)
    ap.add_argument("--game-args", default="")
    ap.add_argument("--t-snap", type=float, default=120.0)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    extra = args.game_args.split() if args.game_args else []
    games = [Game(port=args.port0 + i, time_scale=1.0, steps_per_frame=2, extra_args=extra) for i in range(3)]
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda g: g.attach_or_launch(), games))
    try:
        A, B, C = games
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda g: g.client.reset(), [B, C]))
        ghost = Ghost.from_file(DEMO_KEYS["1118"])
        pol = make_policy(ghost, default_ghost_params(), {})
        pol.reset(); clock = MatchClock()
        s = A.client.reset(); s["t"] = clock(s)
        s = play_to(A, pol, args.t_snap, clock, s)
        blob = A.client.snapshot()["blob"]
        import copy
        pb = copy.deepcopy(pol)
        acts, tB = drive(B, blob, args.steps, pol=pb)
        _, tC = drive(C, blob, args.steps, actions=acts)
        compare(tB, tC, f"cross-instance restore+replay ({args.game_args or 'default threads'})")
        _, tB2 = drive(B, blob, args.steps, actions=acts)
        compare(tB, tB2, f"same-instance restore+replay ({args.game_args or 'default threads'})")
        # A continues natively from the snapshot point with the same actions (warm physics caches)
        tA = [vec(s)]
        for a in acts:
            s = A.client.act(*a, lite=False); s["t"] = clock(s); tA.append(vec(s))
        compare(tB, tA, "native continuation vs restore")
    finally:
        for g in games:
            g.close(quit_game=True)


if __name__ == "__main__":
    main()
