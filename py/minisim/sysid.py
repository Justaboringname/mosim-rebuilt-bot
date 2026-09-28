"""Joint system identification of MiniSim's measured (non-PhysX) parameters against several real batches.

Targets per real arm: mean score in each window (160-140, 140-130, 130-105, 105-80, 80-55, 55-30, 30-end) and, where
dense ball data exists, ball counts by region (held, blue-zone floor, neutral floor, red-zone floor) at t = 106/56/31.
Candidates are scored on common random numbers (same seeds for every candidate) with a persistent worker pool.

    python -m minisim.sysid --gens 30
The fresh holdout batch (runs/ghost/hr1) is never used here.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

WINDOWS = [(160, 140), (140, 130), (130, 105), (105, 80), (80, 55), (55, 30), (30, -5)]
REGION_T = [106.0, 56.0, 31.0]
DEMO = "../run/demos/demo-20260926-015440-m1.jsonl"
OUT = Path("../runs/sysid")

# (real batch, arm index, arm spec used in the sim)
CAL = [
    ("r4", 0, {}),
    ("r4", 1, {"ghost": "1097"}),
    ("r4", 2, {"ghost": "1085"}),
    ("cal2", 0, {"chooser": "seeker"}),
    ("cal2", 1, {"chooser": "greedy", "leash": {"neutral": 0.667, "conveyor": 0.667}}),
    ("bump1", 1, "BUMP1"),
    ("ck", 0, {}),
]

# name, low, high, default (sim2.DEFAULT) — the measured / phenomenological parameters only
SPACE = [
    ("p_cap_full", 0.04, 0.40, 0.12),
    ("p_cap_slow", 0.0, 0.03, 0.004),
    ("r_max", 13.0, 22.0, 17.5),
    ("rim_scale", 0.5, 1.6, 1.0),
    ("hub_t_mu", 1.7, 2.7, 2.2),
    ("exit_v0", 0.9, 1.9, 1.35),
    ("exit_v_scale", 0.15, 0.9, 0.45),
    ("pass_x_mu", 4.9, 6.1, 5.5),
    ("pass_x_sd", 0.5, 1.5, 0.95),
    ("pass_z_mu", 1.2, 2.1, 1.7),
    ("cap_soft", 85.0, 104.0, 95.0),                     # must stay below cap_mid (105)
    ("bump_drive", 0.5, 1.0, 0.75),
    ("turret_rate_max", 0.8, 2.2, 1.2),
]


def arm_spec(a):
    if a == "BUMP1":
        return json.load(open("../runs/ghost/bump1-arm.json"))
    return a


def real_targets():
    from .calib.common import episodes
    from .calib.regions import counts
    T = []
    for batch, arm, _ in CAL:
        wins, regs = [], []
        for d in episodes(batch, arm=arm, dense=True):
            tr = d["trace"]
            wins.append(window_scores([(r["t"], r["blue"]) for r in tr]))
            if d.get("dense") is not None:
                raw = d["dense"]["pos_cm"]; P = raw.astype(float) / 100; gone = raw[:, :, 0] == -32768
                t = d["dense"]["t"]
                ks = [int(np.argmin(np.abs(t - x))) for x in REGION_T]
                c = counts(P[ks], gone[ks])                      # held hub air blue neutral red ...
                regs.append(c[:, [0, 3, 4, 5]])
        T.append({"batch": batch, "arm": arm, "n": len(wins), "win": np.mean(wins, 0).tolist(),
                  "win_se": (np.std(wins, 0) / max(1, len(wins) - 1) ** 0.5).tolist(),
                  "reg": np.mean(regs, 0).tolist() if regs else None})
    return T


def window_scores(tb):
    t = np.array([x[0] for x in tb]); b = np.array([x[1] for x in tb], float)
    out = []
    for hi, lo in WINDOWS:
        m = np.nonzero((t <= hi) & (t > lo))[0]
        if not len(m):
            out.append(0.0); continue
        s = b[m[-1]] - b[max(m[0] - 1, 0)]
        if lo < 0:
            s += b[-1] - b[m[-1]]
        out.append(float(s))
    return out


_GHOST = None


def _job(args):
    params, arm, seed = args
    from mosimrl.ghost import Ghost
    from mosimrl.ghost_policy import default_ghost_params
    from mosimrl.run_ghost import make_policy
    from .calib.regions import counts
    from .sim2 import MiniSim2Client, HOPPER
    global _GHOST
    if _GHOST is None:
        _GHOST = Ghost.from_file(DEMO)
    pol = make_policy(_GHOST, default_ghost_params(), arm_spec(arm), seed=seed)
    c = MiniSim2Client(params, seed=seed)
    s = c.reset(); pol.reset()
    tb, regs, want = [], [], list(REGION_T)
    from mosimrl.episode import MatchClock
    clock = MatchClock()
    s["t"] = clock(s)
    steps = 0
    while not s.get("done") and steps < 5000:
        vx, vz, rot, b, _ = pol.act(s)
        s = c.act(vx, vz, rot, b)
        s["t"] = clock(s)
        tb.append((s["t"], s["blue"]))
        if want and s["t"] <= want[0]:
            sim = c.sim
            P = np.stack([sim.b.p[:, 0], sim.b.p[:, 2], sim.b.p[:, 1]], 1)[None]
            gone = (sim.state == HOPPER)[None]
            regs.append(counts(P, gone)[0][[0, 3, 4, 5]].tolist())
            want.pop(0)
        steps += 1
    return window_scores(tb), regs


ARMS = []                                   # unique sim arm specs (r4 arm 0, bump1 arm 0 and ck are the same policy)
for _, _, a in CAL:
    if a not in ARMS:
        ARMS.append(a)
CAL_ARM = [ARMS.index(a) for _, _, a in CAL]


class Evaluator:
    """Runs every (candidate, arm, seed) job of a whole CMA generation in one pool map, so no core idles between
    candidates. Seeds are common random numbers: seed0 + e for every candidate and arm."""

    def __init__(self, targets, eps_per_arm=8, procs=16, seed0=1000, arms=None):
        self.T = targets
        self.k = eps_per_arm
        self.seed0 = seed0
        self.arms = arms if arms is not None else ARMS
        self.pool = get_context("spawn").Pool(procs)

    def run_many(self, plist):
        jobs = [(p, arm, self.seed0 + e) for p in plist for arm in self.arms for e in range(self.k)]
        res = self.pool.map(_job, jobs, chunksize=1)
        out, i = [], 0
        for _ in plist:
            per_arm = []
            for _ in self.arms:
                chunk = res[i:i + self.k]; i += self.k
                per_arm.append({"win": np.mean([r[0] for r in chunk], 0), "reg": np.mean([r[1] for r in chunk], 0),
                                "tot": [float(sum(r[0])) for r in chunk]})
            out.append(per_arm)
        return out

    def run(self, params):
        return self.run_many([params])[0]

    def loss(self, sim_arms):
        L = 0.0; parts = []
        for tgt, ai in zip(self.T, CAL_ARM):
            s = sim_arms[ai]
            w = np.array(tgt["win"]); sw = s["win"]
            e_w = float((((sw - w) / 12.0) ** 2).sum())
            e_tot = float(((sw.sum() - w.sum()) / 15.0) ** 2) * 2.0
            e_r = 0.0
            if tgt["reg"] is not None:
                e_r = float((((np.array(s["reg"]) - np.array(tgt["reg"])) / 20.0) ** 2).sum()) * 0.5
            L += e_w + e_tot + e_r
            parts.append(round(e_w + e_tot + e_r, 1))
        return L, parts


def decode(x):
    return {name: float(lo + (hi - lo) * min(max(v, 0.0), 1.0)) for (name, lo, hi, _), v in zip(SPACE, x)}


def encode_default():
    return [(d - lo) / (hi - lo) for (_, lo, hi, d) in SPACE]


HOLD_ARMS = Path("../runs/ghost/hold_real_arms.txt")
HOLD_BATCHES = ("hr1", "hr1b")


def holdout(params_list, names, eps=16, batches=HOLD_BATCHES, seed0=5000):
    """Sim vs real on the fresh holdout arms (never used in fitting; seeds disjoint from the fitting seeds).
    Gate: every arm within ~25 points, and no ranking reversal larger than the real gap — a reversal only
    counts when the real gap is > 2x its combined standard error (sim + real), else it is noise between near-ties.
    Arms whose real SE is itself >= ~15 are low-information for the 25-point test; errors are also given in SE units."""
    from .calib.common import episodes
    arms = [json.loads(a) for a in HOLD_ARMS.read_text().strip().split(";")]
    real = []
    for k in range(len(arms)):
        sc = [d["score"] for b in batches for d in episodes(b, arm=k)]
        real.append((float(np.mean(sc)), float(np.std(sc, ddof=1) / len(sc) ** 0.5), len(sc)))
    ev = Evaluator(None, eps_per_arm=eps, arms=arms, seed0=seed0)
    t0 = time.time()
    sims = ev.run_many(params_list)
    print(f"({len(params_list) * len(arms) * eps} sim matches in {time.time() - t0:.0f}s)", flush=True)
    rows = []
    for name, sim in zip(names, sims):
        tot = [float(np.mean(a["tot"])) for a in sim]
        se_s = [float(np.std(a["tot"], ddof=1) / len(a["tot"]) ** 0.5) for a in sim]
        err = [s_ - r[0] for s_, r in zip(tot, real)]
        z = [e / (r[1] ** 2 + q ** 2) ** 0.5 for e, r, q in zip(err, real, se_s)]
        rev = []
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                gap_r, gap_s = real[i][0] - real[j][0], tot[i] - tot[j]
                se_gap = (real[i][1] ** 2 + real[j][1] ** 2 + se_s[i] ** 2 + se_s[j] ** 2) ** 0.5
                if gap_r * gap_s < 0 and abs(gap_r) > 2 * se_gap:
                    rev.append((i, j, round(gap_r), round(gap_s), round(se_gap, 1)))
        informative = [k for k in range(len(arms)) if real[k][1] < 15]
        ok = all(abs(err[k]) <= 25 for k in informative) and not rev
        rows.append({"name": name, "sim": tot, "sim_se": se_s, "err": err, "z": z, "reversals": rev, "pass": ok,
                     "informative": informative})
        print(f"== {name}: {'PASS' if ok else 'FAIL'}   (25-pt test on arms {informative})")
        for k in range(len(arms)):
            print(f"  arm {k}: real {real[k][0]:7.1f} ± {real[k][1]:4.1f} (n={real[k][2]:2d})  sim {tot[k]:7.1f} ± "
                  f"{se_s[k]:4.1f}  err {err[k]:+6.1f} ({z[k]:+5.1f} SE)   {json.dumps(arms[k])[:60]}")
        print("  significant reversals (i, j, real gap, sim gap, SE of gap):", rev)
    return {"real": real, "rows": rows, "batches": list(batches)}


def main():
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=14)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.18)
    ap.add_argument("--holdout", action="store_true", help="evaluate default and best params on the hr1 holdout")
    ap.add_argument("--hold-batches", default=",".join(HOLD_BATCHES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.holdout:
        ps, names = [decode(encode_default())], ["default"]
        if (OUT / "best.json").exists():
            ps.append(json.loads((OUT / "best.json").read_text())["params"]); names.append("sysid best")
        res = holdout(ps, names, batches=tuple(a.hold_batches.split(",")))
        (OUT / "holdout.json").write_text(json.dumps(res, indent=1))
        return
    tpath = OUT / "targets.json"
    if not tpath.exists():
        tpath.write_text(json.dumps(real_targets()))
    T = json.loads(tpath.read_text())
    ev = Evaluator(T, eps_per_arm=a.eps)
    x0 = encode_default()
    t0 = time.time()
    base = ev.run(decode(x0)); L0, P0 = ev.loss(base)
    print(f"default loss {L0:.1f} parts {P0}  ({time.time() - t0:.0f}s)", flush=True)
    log = open(OUT / "evals.jsonl", "a")
    es = cma.CMAEvolutionStrategy(x0, a.sigma, {"bounds": [0, 1], "popsize": a.pop, "seed": 7})
    best = (L0, decode(x0))
    for gen in range(a.gens):
        t0 = time.time()
        X = es.ask()
        P = [decode(x) for x in X]
        sims = ev.run_many(P)
        Ls = []
        for p, sim in zip(P, sims):
            L, parts = ev.loss(sim)
            Ls.append(L)
            log.write(json.dumps({"gen": gen, "loss": L, "parts": parts, "params": p,
                                  "win": [s["win"].tolist() for s in sim]}) + "\n"); log.flush()
            if L < best[0]:
                best = (L, p)
                (OUT / "best.json").write_text(json.dumps({"loss": L, "params": p, "parts": parts}, indent=1))
        es.tell(X, Ls)
        print(f"gen {gen}: min {min(Ls):.1f} median {np.median(Ls):.1f} best {best[0]:.1f}  ({time.time() - t0:.0f}s)",
              flush=True)
    print("best", json.dumps(best[1]))


if __name__ == "__main__":
    main()
