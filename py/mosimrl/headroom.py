"""Offline headroom of trip-level route choice.

For each collection trip of the ghost route, take a family of smooth lateral deformations of the ghost path (a
sine bump that is zero at the trip's ends, so crossings and timing are untouched) and count, on the ball positions
the bot actually saw (0.5 s snapshots with stable ball ids), how many balls each variant would have swept.
Two numbers matter:
  oracle  = mean over episodes of the best variant, scored on the real ball motion during the trip
  chooser = the variant picked from the snapshot at trip start only (what an online planner could see)
both against the unmodified ghost path. The robot's effect on the balls is ignored (it only removes the balls it
sweeps), so this is an upper-bound style estimate, not a promise.

    python -m mosimrl.headroom ../runs/ghost/ck
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np

from .ghost import Ghost

DEMO = "../run/demos/demo-20260926-015440-m1.jsonl"
TRIPS = [(139.2, 135.5), (129, 124), (124, 118), (118, 112), (112, 106), (78, 74), (74, 70), (70, 66), (66, 62),
         (62, 58), (31, 27), (21, 17), (10, 6)]
AMPS = np.array([-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
X_LIM, Z_LIM = (-3.0, 2.95), 3.55
W = 0.40


def ghost_path(g: Ghost, a: float, b: float, dt: float = 0.05):
    ts = np.arange(a, b - 1e-9, -dt)
    p = np.array([(g._interp(g.x, t), g._interp(g.z, t)) for t in ts])
    v = np.gradient(p, axis=0)
    n = np.stack([-v[:, 1], v[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    # smooth the normal so U-turns do not flip the offset direction abruptly
    k = np.ones(15) / 15
    n = np.stack([np.convolve(n[:, 0], k, "same"), np.convolve(n[:, 1], k, "same")], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    return ts, p, n


def clamp_to_field(q, p):
    """Keep a displaced target out of the hub/bump fronts and off the side walls, but never tighter than where the
    ghost itself is (the human's route does enter the trench lanes and hugs the red bump face)."""
    q = q.copy()
    lanes = np.abs(q[:, 1]) >= 2.5                         # trench lanes: the robot may go deeper in x there
    xmax = np.where(lanes, 3.0, 2.6); xmin = -xmax
    neutral = np.abs(p[:, 0]) < 3.0
    q[:, 0] = np.where(neutral, np.clip(q[:, 0], np.minimum(xmin, p[:, 0]), np.maximum(xmax, p[:, 0])), q[:, 0])
    zl = np.maximum(Z_LIM, np.abs(p[:, 1]))
    q[:, 1] = np.clip(q[:, 1], -zl, zl)
    return q


SUBS = [(0.0, 1.0), (0.0, 0.5), (0.25, 0.75), (0.5, 1.0)]   # bump support as fractions of the trip


def variants(ts, p, n):
    """Bumps over the whole trip or half of it, every amplitude; index 0 is always the unmodified ghost path."""
    out, specs = [p.copy()], [(0.0, 1.0, 0.0)]
    u = (ts[0] - ts) / (ts[0] - ts[-1])
    for f0, f1 in SUBS:
        w = np.clip((u - f0) / (f1 - f0), 0.0, 1.0)
        s = np.where((u >= f0) & (u <= f1), np.sin(np.pi * w), 0.0)[:, None]
        for A in AMPS:
            if A == 0.0:
                continue
            out.append(clamp_to_field(p + A * s * n, p))
            specs.append((f0, f1, float(A)))
    return out, specs


def feasible(q, p, ts, slack=1.12, add=0.25, max_extra=0.06):
    """A variant the robot can actually drive on the ghost's clock: nowhere faster than the ghost itself (+12 %
    +0.25 m/s; the human is at full stick almost everywhere), total length at most +6 %."""
    dt = np.abs(np.diff(ts)).mean()
    vq = np.linalg.norm(np.diff(q, axis=0), axis=1) / dt
    vp = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    k = np.ones(5) / 5
    vq, vp = np.convolve(vq, k, "same"), np.convolve(vp, k, "same")
    return bool(np.all(vq <= slack * vp + add) and path_len(q) <= (1 + max_extra) * path_len(p) + 0.2)


def path_len(q):
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def snapshots(trace):
    for r in trace:
        if r.get("fid") is not None and r.get("fuel") is not None:
            a = np.asarray(r["fuel"], float).reshape(-1, 3)
            yield r["t"], np.asarray(r["fid"], int), a


FWD = 0.75        # s of path ahead of each snapshot that can reach a ball before the next one (0.5 s + intake lead)
CAP = True        # hopper capacity 100, drained at PASS_RATE while the ghost holds a shoot/pass button
PASS_RATE = 15.0


def robot_free(snaps):
    """Snapshots with the robot's removals undone: a floor ball that vanished (went into the real robot's hopper,
    or was shot straight out) is kept at its last floor position for the rest of the trip."""
    last = {}
    out = []
    for t, ids, a in snaps:
        fl = a[:, 2] < 0.2
        cur = {int(i): (x, z) for i, (x, z) in zip(ids[fl], a[fl, :2])}
        listed = set(int(i) for i in ids)
        for i, p in list(last.items()):
            if i not in listed:               # in the real hopper now: pretend it is still lying there
                cur.setdefault(i, p)
        last.update({i: p for i, p in cur.items()})
        for i in [i for i in last if i in listed and i not in cur]:
            del last[i]                       # airborne / raised now: not a floor ball any more
        out.append((t, np.array(list(cur.keys()), int), np.array(list(cur.values()), float).reshape(-1, 2)))
    return out


def actual_intakes(snaps, rob):
    """Balls the real robot took during the trip: on the floor at one snapshot, then gone (hopper) or airborne
    close to the robot at the next one."""
    got = set()
    for (t0, ids0, a0), (t1, ids1, a1) in zip(snaps, snaps[1:]):
        fl0 = set(int(i) for i in ids0[a0[:, 2] < 0.2])
        pos1 = {int(i): p for i, p in zip(ids1, a1)}
        for i in fl0:
            p = pos1.get(i)
            if p is None:
                got.add(i)
            elif p[2] > 0.3 and rob(t1) is not None and np.hypot(p[0] - rob(t1)[0], p[1] - rob(t1)[1]) < 2.5:
                got.add(i)
    return len(got)


def sweep_times(q, ts, snaps, start_only=False):
    """{ball id: match clock when path q (time-indexed by ts) first sweeps it}, from robot-free floor snapshots."""
    first = {}
    snaps = snaps[:1] if start_only else snaps
    for t, ids, xz in snaps:
        if not len(ids):
            continue
        m = np.ones(len(ts), bool) if start_only else (ts <= t + 0.05) & (ts >= t - FWD)
        if not m.any():
            continue
        D = np.linalg.norm(xz[:, None, :] - q[m][None, :, :], axis=2)
        hit = D.min(1) < W
        tt = ts[m][np.argmax(D < W, axis=1)]
        for i, th in zip(ids[hit], tt[hit]):
            i = int(i)
            if i not in first or th > first[i]:
                first[i] = float(th)
    return first


def sweep_count(q, ts, snaps, start_only=False):
    return len(sweep_times(q, ts, snaps, start_only))


def hopper_count(times: dict, ts, held0: float, drain, cap: float = 100.0, dt: float = 0.05) -> int:
    """Balls actually kept when the hopper holds at most `cap` and drains at drain(t) balls/s (passing/shooting)."""
    arr = np.sort(-np.array(list(times.values()))) if times else np.array([])   # ascending in elapsed time
    arr = -arr
    h, got, k = held0, 0, 0
    for t in ts:
        h = max(0.0, h - drain(t) * dt)
        while k < len(arr) and arr[k] >= t:
            if h < cap:
                h += 1; got += 1
            k += 1
    return got


def main(run_dir: str) -> None:
    g = Ghost.from_file(DEMO)
    fam = {tr: (ghost_path(g, *tr)) for tr in TRIPS}
    rows = []
    fams = {}
    for trip, (ts, p, n) in fam.items():
        V, specs = variants(ts, p, n)
        keep = [k for k, q in enumerate(V) if k == 0 or feasible(q, p, ts)]
        fams[trip] = ([V[k] for k in keep], [specs[k] for k in keep])

    def drain(t):
        b = g.at(t).buttons
        return PASS_RATE if (b[1] or b[2] or b[3]) else 0.0
    for f in sorted(glob.glob(f"{run_dir}/*-ep*.json")):
        d = json.load(open(f))
        tr = d["trace"]
        if not any(r.get("fid") for r in tr):
            continue
        S = list(snapshots(tr))
        tt = np.array([r["t"] for r in tr]); rx = np.array([r["x"] for r in tr]); rz = np.array([r["z"] for r in tr])

        def rob(t):
            i = int(np.searchsorted(-tt, -t, side="right")) - 1
            return (rx[i], rz[i]) if 0 <= i < len(tt) else None
        for trip in TRIPS:
            a, b = trip
            ts, p, n = fam[trip]
            V, specs = fams[trip]
            sn = [x for x in S if b - 0.3 <= x[0] <= a + 0.3]
            if len(sn) < 2:
                continue
            free = robot_free(sn)
            held0 = next(r["held"] for r in tr if r["t"] <= a)
            real = [hopper_count(sweep_times(q, ts, free), ts, held0, drain) for q in V]
            start = [hopper_count(sweep_times(q, ts, free, True), ts, held0, drain) for q in V]
            rows.append((trip, real, start, actual_intakes(sn, rob)))
    if not rows:
        print("no traces with ball ids in", run_dir)
        return
    print(f"{len(rows) // len(TRIPS)} episodes; feasible variants per trip "
          f"{[len(fams[t][0]) for t in TRIPS]}")
    print("trip            actual  ghost  oracle  chooser  fixed-best (f0,f1,A)")
    tot = np.zeros(5)
    best_fixed = {}
    for trip in TRIPS:
        R = [(np.array(r), np.array(s), k) for t, r, s, k in rows if t == trip]
        real = np.array([r for r, _, _ in R]); start = np.array([s for _, s, _ in R])
        act = np.mean([k for _, _, k in R])
        gh = real[:, 0].mean()
        orc = real.max(1).mean()
        pick = start.argmax(1)
        ch = real[np.arange(len(real)), pick].mean()
        fixed_k = int(real.mean(0).argmax())
        fx = real[:, fixed_k].mean()
        tot += (act, gh, orc, ch, fx)
        best_fixed[f"{trip[0]},{trip[1]}"] = fams[trip][1][fixed_k]
        print(f"{str(trip):15s} {act:6.1f} {gh:6.1f} {orc:7.1f} {ch:8.1f}  {fx:6.1f} {fams[trip][1][fixed_k]}")
    print(f"{'TOTAL':15s} {tot[0]:6.1f} {tot[1]:6.1f} {tot[2]:7.1f} {tot[3]:8.1f}  {tot[4]:6.1f}")
    print("BEST_FIXED", json.dumps(best_fixed))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../runs/ghost/ck")
