"""Analysis for mosimrl.optlearn (pre-registered gates K0-K2 in docs/research-log/proposals.md, "OL").

Per rollout r at snapshot s: Y_r = alpha_s + sum_o theta_o * X_ro + eps, X_ro = sum over the rollout's events of
(T_eo - p_o) (propensity-centred treatment counts), randomized rollouts only, snapshot fixed effects (within-snapshot
demeaning). Heterogeneity: X_roj = sum_e (T_eo - p_o) * phi_ej with standardized phi, ridge, CV grouped by main.
SEs: cluster bootstrap over mains.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .optlearn import FEATS

BETAS = (0.35, 0.2, 0.7)
OPTS = {"A": ("placebo", "cap"), "C": ("placebo", "hold")}
PROP = {"A": {"base": 0.5, "placebo": 0.1, "cap": 0.4}, "C": {"base": 0.5, "placebo": 0.1, "hold": 0.4}}


def load(d: Path):
    R = [json.loads(l) for l in open(d / "rollouts.jsonl")]
    M = [json.loads(l) for l in open(d / "mains.jsonl")] if (d / "mains.jsonl").exists() else []
    return R, M


def value(r: dict, beta: float = 0.35, full: bool = False):
    if r["win"] == "W3" or full:
        return r.get("final_blue")
    a = r.get("at")
    return None if not a else a["blue"] + beta * a["owned"]


def design(R, win, ev, beta=0.35, full=False):
    rows = []
    for r in R:
        if r["win"] != win or r["mode"] != "random":
            continue
        y = value(r, beta, full)
        if y is None:
            continue
        x = []
        for o in OPTS[ev]:
            x.append(sum((e["opt"] == o) - PROP[ev][o] for e in r["events"] if e["ev"] == ev))
        rows.append((r["main"], y, x))
    return rows


def fe_ols(rows):
    """within-snapshot (main x window) demeaned OLS: returns theta (n_opt,)"""
    by = defaultdict(list)
    for m, y, x in rows:
        by[m].append((y, x))
    Y, X = [], []
    for m, L in by.items():
        if len(L) < 2:
            continue
        y = np.array([a for a, _ in L], float); x = np.array([b for _, b in L], float)
        Y.append(y - y.mean()); X.append(x - x.mean(0))
    if not Y:
        return None
    Y = np.concatenate(Y); X = np.concatenate(X)
    th, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return th


def boot(rows, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    mains = sorted({m for m, _, _ in rows})
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    est = fn(rows)
    bs = []
    for _ in range(B):
        pick = rng.choice(mains, len(mains), replace=True)
        rr = []
        for k, m in enumerate(pick):
            rr += [(f"{m}#{k}", y, x) for _, y, x in by[m]]
        v = fn(rr)
        if v is not None:
            bs.append(v)
    return est, np.std(bs, axis=0, ddof=1) if bs else None


def events_per(R, win, ev):
    n = [len([e for e in r["events"] if e["ev"] == ev]) for r in R if r["win"] == win and r["mode"] == "random"]
    return float(np.mean(n)) if n else 0.0


def analyze(d: Path) -> None:
    R, M = load(d)
    mains = sorted({r["main"] for r in R})
    print(f"rollouts {len(R)} from {len(mains)} mains; main matches finished {len(M)}"
          + (f" (mean score {np.mean([m['score'] for m in M]):.1f})" if M else ""))
    if len(M) >= 2:
        span = (M[-1]["wall"] - M[0]["wall"]) / 3600
        print(f"throughput: {len(M) / max(span, 1e-6):.0f} mains/h, {len(R) / max(span, 1e-6):.0f} rollouts/h")
    # K0 (a1): pure-base pair difference per window (slot/order bias)
    for win in ("W2", "W3"):
        diffs, sig = [], []
        by = defaultdict(list)
        for r in R:
            if r["win"] == win and r["mode"] == "base" and value(r) is not None:
                by[r["main"]].append(value(r))
        for m, v in by.items():
            if len(v) == 2:
                diffs.append(v[0] - v[1]); sig.append(abs(v[0] - v[1]) / np.sqrt(2))
        if diffs:
            print(f"K0 {win} base-pair diff {np.mean(diffs):+.2f} ± {np.std(diffs, ddof=1) / np.sqrt(len(diffs)):.2f} "
                  f"(n={len(diffs)}); per-rollout sigma ≈ {np.sqrt(np.mean(np.square(diffs)) / 2):.1f}")
    # effects: theta per option (per event), implied per-window effect = theta * events per window
    for win in ("W2", "W3"):
        for ev in ("A", "C"):
            if win == "W2" and ev == "C":
                continue
            rows = design(R, win, ev)
            if len(rows) < 20:
                continue
            th, se = boot(rows, fe_ols, B=500)
            if th is None or se is None:
                continue
            k = events_per(R, win, ev)
            s = "  ".join(f"{o}: {th[i]:+.2f} ± {se[i]:.2f}/event" for i, o in enumerate(OPTS[ev]))
            print(f"{win} event {ev} ({k:.1f} events/rollout): {s}   implied all-{OPTS[ev][1]} per window {th[1] * k:+.1f} ± {se[1] * k:.1f}")
            if win == "W2":
                for beta in BETAS[1:]:
                    th2, se2 = boot(design(R, win, ev, beta), fe_ols, B=300)
                    print(f"      beta {beta}: {OPTS[ev][1]} {th2[1]:+.2f} ± {se2[1]:.2f}/event")
                rf = design(R, win, ev, full=True)
                if len(rf) >= 20:
                    th3, se3 = boot(rf, fe_ols, B=300)
                    print(f"      full-horizon subset (n={len(rf)}): {OPTS[ev][1]} {th3[1]:+.2f} ± {se3[1]:.2f}/event")
    # K0 (b): manipulation check for Event A
    hi = defaultdict(list); rec = defaultdict(list)
    for r in R:
        for e in r["events"]:
            if e["ev"] == "A" and r["mode"] == "random":
                hi[e["opt"]].append(e.get("hi_s", 0.0))
                rec[e["opt"]].append(e.get("rec_s") is not None and e["rec_s"] <= 3.0 or not e.get("big"))
    if hi:
        print("K0 manipulation: time >120°/s per event " + ", ".join(f"{o} {np.mean(v):.3f}s (n={len(v)})" for o, v in hi.items())
              + " | yaw_err back <10° within 3 s: " + ", ".join(f"{o} {np.mean(v):.2f}" for o, v in rec.items()))
    unst = defaultdict(list)
    for r in R:
        unst[(r["win"], r["mode"])].append(r.get("unsticks", 0))
    print("unsticks per rollout: " + ", ".join(f"{k[0]}/{k[1]} {np.mean(v):.2f}" for k, v in sorted(unst.items())))


# ---------------------------------------------------------------------------------------------- K1 heterogeneity
def het_rows(R, win, ev, beta=0.35, full=False):
    """per randomized rollout: (main, y, X) with X = [sum(T_opt - p) * [1, phi...] for the treatment option,
    sum(T_placebo - p)]; phi standardized over all events of this window/event type."""
    opt = OPTS[ev][1]
    allphi = np.array([e["phi"] for r in R if r["win"] == win and r["mode"] == "random"
                       for e in r["events"] if e["ev"] == ev], float)
    if len(allphi) < 50:
        return None, None
    mu, sd = allphi.mean(0), allphi.std(0) + 1e-9
    rows = []
    for r in R:
        if r["win"] != win or r["mode"] != "random":
            continue
        y = value(r, beta, full)
        if y is None:
            continue
        x = np.zeros(2 + allphi.shape[1])
        for e in r["events"]:
            if e["ev"] != ev:
                continue
            ph = (np.array(e["phi"], float) - mu) / sd
            c = (e["opt"] == opt) - PROP[ev][opt]
            x[0] += c; x[1:-1] += c * ph
            x[-1] += (e["opt"] == "placebo") - PROP[ev]["placebo"]
        rows.append((r["main"], y, x))
    return rows, (mu, sd)


def _demean(rows):
    by = defaultdict(list)
    for m, y, x in rows:
        by[m].append((y, x))
    mains, Y, X = [], [], []
    for m, L in by.items():
        if len(L) < 2:
            continue
        y = np.array([a for a, _ in L]); x = np.array([b for _, b in L])
        Y.append(y - y.mean()); X.append(x - x.mean(0)); mains += [m] * len(L)
    return np.array(mains), np.concatenate(Y), np.concatenate(X)


def _ridge(X, Y, lam, free):
    P = np.eye(X.shape[1]) * lam
    for j in free:
        P[j, j] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ Y)


def cv_gain(mains, Y, X, lam, folds=5, seed=0):
    """grouped-CV MSE(base: intercept + placebo only) − MSE(full ridge). > 0: phi predicts the effect."""
    um = np.unique(mains); rng = np.random.default_rng(seed); rng.shuffle(um)
    fold = {m: i % folds for i, m in enumerate(um)}
    f = np.array([fold[m] for m in mains])
    base_cols = [0, X.shape[1] - 1]
    e_full = e_base = 0.0
    for k in range(folds):
        tr, te = f != k, f == k
        wf = _ridge(X[tr], Y[tr], lam, base_cols)
        wb = np.linalg.lstsq(X[tr][:, base_cols], Y[tr], rcond=None)[0]
        e_full += ((Y[te] - X[te] @ wf) ** 2).sum()
        e_base += ((Y[te] - X[te][:, base_cols] @ wb) ** 2).sum()
    return (e_base - e_full) / len(Y)


def k1(d: Path, n_perm: int = 200) -> None:
    R, _ = load(d)
    for win, ev, full in (("W2", "A", False), ("W2", "A", True), ("W3", "A", False), ("W3", "C", False)):
        rows, _ = het_rows(R, win, ev, full=full)
        if rows is None:
            continue
        mains, Y, X = _demean(rows)
        lams = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
        gains = {lam: cv_gain(mains, Y, X, lam) for lam in lams}
        lam = max(gains, key=gains.get); g = gains[lam]
        rng = np.random.default_rng(1)
        null = []
        for _ in range(n_perm):                       # permute the phi part across rollouts (keeps treatment counts)
            Xp = X.copy(); idx = rng.permutation(len(X)); Xp[:, 1:-1] = X[idx, 1:-1]
            null.append(max(cv_gain(mains, Y, Xp, l) for l in lams))
        p = (1 + sum(n >= g for n in null)) / (1 + n_perm)
        print(f"K1 heterogeneity {win}{' full' if full else ''} event {ev}: CV gain {g:+.3f} (lam {lam:g}), "
              f"permutation p = {p:.3f}  (n rollouts {len(Y)})")


# ---------------------------------------------------------------------------------------------- K0 re-check, K2
def base_pairs_by_seed(R):
    for win in ("W2", "W3"):
        by = defaultdict(dict)
        for r in R:
            if r["win"] == win and r["mode"] == "base" and value(r) is not None:
                by[r["main"]][r["seed"]] = value(r)
        d = [v[min(v)] - v[max(v)] for v in by.values() if len(v) == 2]
        print(f"K0 {win} base-pair diff by submission order {np.mean(d):+.2f} ± {np.std(d, ddof=1) / np.sqrt(len(d)):.2f} (n={len(d)})")


def k2(d: Path, folds: int = 5, B: int = 1000, lcb_z: float = 1.28) -> None:
    """Cross-fitted value of the LCB policy 'cap where the predicted per-event effect's lower bound > 0'.
    Per fold: ridge on the training mains (lam by inner CV), predicted effect tau(phi) = w0 + w^T phi with a bootstrap
    SE; on the held-out mains, Z_r = sum over events with pi(phi_e) = cap of (T_e,cap - p). Pooled held-out FE
    regression of Y on Z gives the mean effect of capping among selected events; value per window = that effect x
    selected events per rollout. SE by cluster bootstrap over mains."""
    R, _ = load(d)
    tot, tot_var = 0.0, 0.0
    for win, full in (("W2", False), ("W3", False)):
        rows, (mu, sd) = het_rows(R, win, "A", full=full)
        mains, Y, X = _demean(rows)
        # raw rollout records aligned with rows (randomized, value not None, snapshot with >= 2 rollouts)
        recs = [r for r in R if r["win"] == win and r["mode"] == "random" and value(r, 0.35, full) is not None]
        cnt = defaultdict(int)
        for r in recs:
            cnt[r["main"]] += 1
        recs = [r for r in recs if cnt[r["main"]] >= 2]
        order = defaultdict(list)
        for r in recs:
            order[r["main"]].append(r)
        recs = [r for m in dict.fromkeys(mains) for r in order[m]]
        assert len(recs) == len(Y)
        um = np.unique(mains); rng = np.random.default_rng(0); rng.shuffle(um)
        fold = {m: i % folds for i, m in enumerate(um)}
        f = np.array([fold[m] for m in mains])
        Z = np.zeros(len(Y)); nsel = np.zeros(len(Y)); nev = np.zeros(len(Y))
        base_cols = [0, X.shape[1] - 1]
        for k in range(folds):
            tr = f != k
            lams = [100.0, 1000.0, 1e4, 1e5]
            lam = max(lams, key=lambda l: cv_gain(mains[tr], Y[tr], X[tr], l))
            w = _ridge(X[tr], Y[tr], lam, base_cols)
            # bootstrap SE of tau(phi) over training mains
            trm = np.unique(mains[tr]); W = []
            for b in range(100):
                pick = rng.choice(trm, len(trm)); idx = np.concatenate([np.where(mains == m)[0] for m in pick])
                W.append(_ridge(X[idx], Y[idx], lam, base_cols))
            W = np.array(W)
            for i in np.where(f == k)[0]:
                for e in recs[i]["events"]:
                    if e["ev"] != "A":
                        continue
                    ph = np.concatenate([[1.0], (np.array(e["phi"], float) - mu) / sd])
                    tau = ph @ w[:-1]; se = (W[:, :-1] @ ph).std()
                    nev[i] += 1
                    if tau - lcb_z * se > 0:
                        nsel[i] += 1
                        Z[i] += (e["opt"] == "cap") - PROP["A"]["cap"]
        # FE regression of Y on Z (Y, Z demeaned within snapshot)
        Zd = Z.copy()
        for m in np.unique(mains):
            ii = mains == m; Zd[ii] -= Z[ii].mean()

        def est(idx):
            z, y = Zd[idx], Y[idx]
            return (z @ y) / max(z @ z, 1e-9) * nsel[idx].mean()
        v = est(np.arange(len(Y)))
        bs = []
        for b in range(B):
            pick = rng.choice(um, len(um)); idx = np.concatenate([np.where(mains == m)[0] for m in pick])
            bs.append(est(idx))
        s = np.std(bs, ddof=1)
        print(f"K2 {win}: LCB policy caps {nsel.mean():.2f} of {nev.mean():.2f} events/rollout; value vs base {v:+.2f} ± {s:.2f} per window")
        tot += v; tot_var += s ** 2
    print(f"K2 total per match (W2 in blue+0.35·owned units, W3 in points): {tot:+.2f} ± {np.sqrt(tot_var):.2f} "
          f"-> {'PASS' if tot >= 5 and tot - 1.96 * np.sqrt(tot_var) > 1 else 'FAIL'} (needs >= +5 and lower bound > +1)")
