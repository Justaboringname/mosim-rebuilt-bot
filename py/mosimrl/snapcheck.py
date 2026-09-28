"""Gate 3 of the lookahead planner: does a snapshot restored in another instance continue the match the same way?

    python -m mosimrl.snapcheck --t0 100 --n 100

S plays the ghost to t0, snapshots, and keeps playing N decisions (actions recorded). Then, replaying those
actions open-loop:  T  = another instance after restore;  T2 = T restored again;  S2 = S restored in place.
Per decision it compares every ball (by stable fid), the robot, held and score. S vs S2 / T vs T2 measure how
exactly a restore reproduces itself; S vs T is what a planner sees relative to the real match.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .episode import MatchClock
from .gamectl import Game
from .ghost import Ghost
from .ghost_policy import default_ghost_params
from .run_ghost import make_policy

DEMO = Path(__file__).resolve().parents[2] / "run/demos/demo-20260926-015440-m1.jsonl"


def fuel_arr(s: dict, n: int = 512) -> np.ndarray:
    a = np.full((n, 3), np.nan)
    if s.get("fid") is not None:
        a[np.asarray(s["fid"], int)] = np.asarray(s["fuel"], float)
    return a


def compare(a: list[dict], b: list[dict]) -> list[dict]:
    out = []
    for sa, sb in zip(a, b):
        fa, fb = fuel_arr(sa), fuel_arr(sb)
        both = ~np.isnan(fa[:, 0]) & ~np.isnan(fb[:, 0])
        d = np.linalg.norm(fa[both] - fb[both], axis=1) if both.any() else np.zeros(1)
        ra, rb = sa["robot"], sb["robot"]
        ma, mb = sa.get("mech") or {}, sb.get("mech") or {}
        out.append({"t": sa["t"], "robot_d": float(np.hypot(ra["x"] - rb["x"], ra["z"] - rb["z"])),
                    "mech": ((ma.get("slide"), ma.get("kang"), ma.get("kick")), (mb.get("slide"), mb.get("kang"), mb.get("kick"))),
                    "held": (sa.get("held"), sb.get("held")), "blue": (sa["blue"], sb["blue"]),
                    "presence_mismatch": int((np.isnan(fa[:, 0]) != np.isnan(fb[:, 0])).sum()),
                    "ball_max": float(d.max()), "ball_mean": float(d.mean()), "moved_1cm": int((d > 0.01).sum())})
    return out


def show(name: str, rows: list[dict]) -> None:
    print(f"--- {name}")
    for k in (0, 1, 2, 5, 10, 20, 50, len(rows) - 1):
        if k < len(rows):
            r = rows[k]
            print(f"  step {k:3d} t {r['t']:6.2f}  robot {r['robot_d']:.4f} m  held {r['held']}  blue {r['blue']}  "
                  f"presence! {r['presence_mismatch']:3d}  balls max {r['ball_max']:.3f} mean {r['ball_mean']:.4f} >1cm {r['moved_1cm']}  mech {r['mech']}")


def replay(g: Game, blob: str, actions: list) -> tuple[dict, list[dict]]:
    s0 = g.client.restore(blob)
    states = []
    for a in actions:
        states.append(g.client.act(*a))
    return s0, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ports", default="47510,47511")
    ap.add_argument("--keep-games", action="store_true")
    args = ap.parse_args()
    ports = [int(p) for p in args.ports.split(",")]
    games = [Game(port=p, time_scale=1.0, steps_per_frame=2) for p in ports]
    with ThreadPoolExecutor(len(games)) as ex:
        list(ex.map(lambda g: g.attach_or_launch(), games))
    S, T = games
    try:
        s_first, t_first = S.client.reset(), T.client.reset()
        ks, kt = S.client.keys(), T.client.keys()
        bs = {k: (x, z, y) for k, x, z, y in ks["balls"]}; bt = {k: (x, z, y) for k, x, z, y in kt["balls"]}
        common = set(bs) & set(bt)
        dpos = [np.hypot(bs[k][0] - bt[k][0], bs[k][1] - bt[k][1]) for k in common]
        print(f"keys: S {ks['keys']} dups {ks['dups']} balls {len(bs)} | T {kt['keys']} dups {kt['dups']} balls {len(bt)} | "
              f"common balls {len(common)}  start-position diff max {max(dpos):.3f} m")

        ghost = Ghost.from_file(str(DEMO))
        pol = make_policy(ghost, default_ghost_params(), {})
        pol.reset()
        clock = MatchClock()
        s = s_first; s["t"] = clock(s)
        t_wall = time.time()
        while s["t"] > args.t0 or s.get("gs") not in (1, 2):
            vx, vz, rot, b, _ = pol.act(s)
            s = S.client.act(vx, vz, rot, b); s["t"] = clock(s)
        snap = S.client.snapshot()
        print(f"snapshot at t={snap['t']} ({time.time() - t_wall:.1f}s to get there): {snap['ms']} ms, {snap['summary']}, "
              f"b64 {len(snap['blob']) / 1024:.0f} KB")
        actions, s_states = [], []
        for _ in range(args.n):
            vx, vz, rot, b, _ = pol.act(s)
            actions.append((vx, vz, rot, b))
            s = S.client.act(vx, vz, rot, b); s["t"] = clock(s)
            s_states.append(s)

        t0 = time.time()
        r0, t_states = replay(T, snap["blob"], actions)
        print(f"T restore: {r0.get('restored')}  t={r0['t']} (replay {time.time() - t0:.1f}s)")
        _, t2_states = replay(T, snap["blob"], actions)
        r0s, s2_states = replay(S, snap["blob"], actions)
        print(f"S restore: {r0s.get('restored')}")
        show("S vs T   (real match vs planner instance)", compare(s_states, t_states))
        show("T vs T2  (same instance, restored twice)", compare(t_states, t2_states))
        show("S vs S2  (in-place restore on the source)", compare(s_states, s2_states))
        out = Path(__file__).resolve().parents[2] / "runs/snapcheck"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"check-t{int(args.t0)}.json").write_text(json.dumps({
            "st": compare(s_states, t_states), "tt": compare(t_states, t2_states), "ss": compare(s_states, s2_states)}))
    finally:
        for g in games:
            g.close(quit_game=not args.keep_games)


if __name__ == "__main__":
    main()
