"""Play matches by following a recorded human demo (GhostPolicy) and report how close the bot stays to it.

    python -m mosimrl.run_ghost --demo ../run/demos/demo-20260926-015440-m1.jsonl --episodes 3 \
        --time-scale 2 --out ../runs/ghost

Per episode: final score vs the ghost's, tracking error by 30 s window, where the bot fell behind (first time the
score gap exceeds 30), and a PNG of both paths. Scores are never uploaded (the Bridge forces wasCheated).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .episode import run_episode
from .gamectl import Game, robust_episode
from .ghost import Ghost, mirror_ghost
from .ghost_policy import GhostPolicy, default_ghost_params
from .residual import GreedySwath, OUResidual, ResidualAgent, ScheduleChooser, zero_chooser

WINDOWS = [(160, 130), (130, 105), (105, 80), (80, 55), (55, 30), (30, 0)]


def summarise(trace: list[dict], ghost: Ghost) -> dict:
    t = np.array([s["t"] for s in trace]); err = np.array([s.get("err", np.nan) for s in trace])
    blue = np.array([s["blue"] for s in trace]); gblue = np.array([s.get("ghost_blue", 0) for s in trace])
    gap = gblue - blue
    behind = next((round(float(t[i]), 1) for i in range(len(t)) if gap[i] > 30), None)
    per = []
    for hi, lo in WINDOWS:
        m = (t <= hi) & (t > lo)
        if m.any():
            idx = np.where(m)[0]
            per.append({"window": f"{hi}-{lo}", "bot": int(blue[idx[-1]] - blue[idx[0]]),
                        "ghost": int(gblue[idx[-1]] - gblue[idx[0]]),
                        "err_med": round(float(np.nanmedian(err[m])), 2), "err_max": round(float(np.nanmax(err[m])), 2)})
    return {"behind_from_t": behind, "per_window": per}


def plot(trace: list[dict], ghost: Ghost, png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.1, 0.6]})
    ax.plot(ghost.x, ghost.z, "-", color="#bbb", lw=1, label=f"ghost ({ghost.name})")
    x = [s["x"] for s in trace]; z = [s["z"] for s in trace]
    sc = ax.scatter(x, z, c=[s.get("err", 0) for s in trace], s=4, cmap="inferno_r", vmin=0, vmax=2)
    fig.colorbar(sc, ax=ax, label="distance to ghost (m)")
    ax.set_xlim(-8.6, 8.6); ax.set_ylim(-4.3, 4.3); ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_title(title)
    t = np.array([s["t"] for s in trace])
    ax2.plot(160 - t, [s["blue"] for s in trace], label="bot score")
    ax2.plot(160 - t, [s.get("ghost_blue", 0) for s in trace], label="ghost score", color="#999")
    ax2.plot(160 - t, [s.get("held", 0) for s in trace], label="bot held", lw=0.8)
    ax2.plot(160 - t, [s.get("ghost_held", 0) for s in trace], label="ghost held", lw=0.8, color="#ccc")
    ax2.set_xlabel("match time elapsed (s)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=100); plt.close(fig)


_ghost_cache: dict = {}
_DEMOS = Path(__file__).resolve().parents[2] / "run/demos"
DEMO_KEYS = {"1118": str(_DEMOS / "demo-20260926-015440-m1.jsonl"), "1085": str(_DEMOS / "demo-20260926-014353-m7.jsonl"),
             "1097": str(_DEMOS / "demo-20260926-020926-m1.jsonl")}


def make_policy(ghost: Ghost, params: dict, arm: dict, seed: int = 0):
    """Arm spec: plain GhostPolicy param overrides, or {"chooser": "zero"|"greedy"|"ou", "leash": frac or
    {phase: frac}, "margin": .., "sigma": ..} for a ResidualAgent on top of the tracker."""
    arm = dict(arm)
    gpath = arm.pop("ghost", None)
    gpath = DEMO_KEYS.get(gpath, gpath)
    if gpath:
        if gpath not in _ghost_cache:
            _ghost_cache[gpath] = Ghost.from_file(gpath)
        ghost = _ghost_cache[gpath]
    bumps = arm.pop("bumps", None)              # {"a,b": amplitude m} smooth lateral bumps on collection trips
    if bumps:
        from .bumps import bumped_ghost
        key = ("bumps", ghost.name, json.dumps(bumps, sort_keys=True))
        if key not in _ghost_cache:
            _ghost_cache[key] = bumped_ghost(ghost, bumps)
        ghost = _ghost_cache[key]
    kind = arm.pop("chooser", None)
    if kind == "seeker":
        from .seeker import SeekerPolicy, SeekParams
        sp_over = arm.pop("seek", {})
        phases = tuple(arm.pop("phases", ["neutral", "conveyor"]))
        return SeekerPolicy(ghost, {**params, **arm}, SeekParams(**sp_over), phases=phases)
    splice = arm.pop("splice", None)            # [[clock, demo-key], ...], demo keys from DEMO_KEYS
    if kind is None:
        pol = GhostPolicy(ghost, {**params, **arm})
        if splice:
            for _, key in splice:
                if key not in _ghost_cache:
                    base_key = key[:-1] if key.endswith("m") else key
                    if DEMO_KEYS[base_key] not in _ghost_cache:
                        _ghost_cache[DEMO_KEYS[base_key]] = Ghost.from_file(DEMO_KEYS[base_key])
                    gb = _ghost_cache[DEMO_KEYS[base_key]]
                    _ghost_cache[key] = mirror_ghost(gb) if key.endswith("m") else gb
            pol.alts = {key: _ghost_cache[key] for _, key in splice}
            pol.schedule = [(float(c), key) for c, key in splice]
        return pol
    lz = arm.pop("leash", 1.0)
    scale = lz if isinstance(lz, dict) else {ph: lz for ph in ("neutral", "conveyor", "harvest", "shoot")}
    if kind == "zero":
        chooser = zero_chooser
    elif kind == "greedy":
        chooser = GreedySwath(margin=arm.pop("margin", 3.0))
    elif kind == "schedule":
        chooser = ScheduleChooser(arm.pop("knots"))
    elif kind == "ou":
        chooser = OUResidual(sigma=arm.pop("sigma", 0.5), seed=seed)
    elif kind == "net":                            # trained residual policy (minisim/rl.py), deterministic mean action
        import torch
        from .nets import ResidualNet
        from .rl_ppo import NetChooser
        ck = torch.load(arm.pop("ckpt"), map_location="cpu")
        net = ResidualNet(k=ck.get("k", 2)); net.load_state_dict(ck["net"]); net.eval()
        torch.set_num_threads(1)
        chooser = NetChooser(net, arm.pop("act_dims", 2), deterministic=True, seed=seed)
    else:
        raise ValueError(kind)
    hmax = arm.pop("heading_max", 30.0)
    every = arm.pop("decide_every", 2)
    clear = bool(arm.pop("clearance", False))
    return ResidualAgent(ghost, chooser, {**params, **arm}, leash_scale=scale, heading_max=hmax, decide_every=every,
                         clearance=clear)


def start_alignment(s: dict, ghost: Ghost) -> dict:
    """How well a fresh Bridge reset matches the demo's first frame: pose, clock, and fuel layout (nearest-neighbour
    distance from each demo ball to the closest live ball). If this is off, a time-indexed ghost is misaligned
    from step one and every score is noise."""
    g = ghost.rows[0]
    rob = s["robot"]
    out = {"bridge_t": s["t"], "ghost_t": g["t"], "pose_err_m": round(float(np.hypot(rob["x"] - g["x"], rob["z"] - g["z"])), 3),
           "yaw_err_deg": round(float((rob["yaw"] - g["yaw"] + 180) % 360 - 180), 2)}
    gf = next((r["fuel"] for r in ghost.rows if "fuel" in r), None)
    bf = s.get("fuel")
    if gf and bf:
        a = np.asarray(gf, dtype=float)[:, :2]; b = np.asarray(bf, dtype=float)[:, :2]
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
        out.update({"fuel_n": [len(a), len(b)], "fuel_nn_max_m": round(float(d.max()), 3),
                    "fuel_nn_p95_m": round(float(np.percentile(d, 95)), 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--ports", default="47500", help="comma list; one headless instance per port, run in parallel")
    ap.add_argument("--episodes", type=int, default=3, help="total episodes across all instances")
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--steps-per-frame", type=int, default=None, help="fixed frame step (use with --time-scale 1)")
    ap.add_argument("--render", action="store_true", help="windowed with cameras (default: headless, no GPU)")
    ap.add_argument("--params", default="", help="json file overriding GHOST_SPEC defaults")
    ap.add_argument("--arms", default="", help="A/B: semicolon-separated JSON param overrides, episodes alternate arms")
    ap.add_argument("--out", default="../runs/ghost")
    ap.add_argument("--game-args", default="", help="extra MoSimulator command-line args when this script launches it")
    ap.add_argument("--keep-games", action="store_true", help="leave the instances running afterwards")
    ap.add_argument("--dense", action="store_true", help="also save every ball's position at every decision (.dense.npz)")
    args = ap.parse_args()

    ghost = Ghost.from_file(args.demo)
    params = default_ghost_params()
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arms = [json.loads(a) for a in args.arms.split(";")] if args.arms else [{}]
    from .rl_ppo import parse_ports
    ports = parse_ports(args.ports)
    games = [Game(port=p, time_scale=args.time_scale, headless=not args.render, cameras=args.render,
                  extra_args=args.game_args.split() if args.game_args else [], steps_per_frame=args.steps_per_frame)
             for p in ports]
    with ThreadPoolExecutor(len(games)) as pool:
        list(pool.map(lambda g: g.attach_or_launch(), games))

    todo = queue.Queue()
    for ep in range(args.episodes):
        todo.put(ep)
    results, lock, t0 = [], threading.Lock(), time.time()

    def worker(game: Game) -> None:
        while True:
            try:
                ep = todo.get_nowait()
            except queue.Empty:
                return
            arm = ep % len(arms)
            pol = make_policy(ghost, params, arms[arm], seed=ep)
            r = robust_episode(game, lambda c: run_episode(c, pol, trace=True, dense=args.dense))
            first = r.pop("first_state")
            dense = r.pop("dense", None)
            if dense is not None:
                np.savez_compressed(out / f"ghost-{ghost.name}-ep{ep}.dense.npz", **dense)
            align = start_alignment(first, ghost)
            summ = summarise(r["trace"], ghost)
            name = f"ghost-{ghost.name}-ep{ep}"
            with lock:                                     # pyplot is not thread-safe
                plot(r["trace"], ghost, out / f"{name}.png", f"{name}: bot {r['score']} vs ghost {ghost.final}")
            (out / f"{name}.json").write_text(json.dumps({**r, **summ, "align": align, "params": params,
                                                          "port": game.port}))
            row = {"ep": ep, "arm": arm, "port": game.port, "score": r["score"], "auto": r["auto"], "wall_s": r["wall_s"],
                   "max_err": round(pol.max_err, 2), "unsticks": pol.unsticks, "deploy_waits": pol.deploy_waits, "switches": getattr(pol, "switches", None), "resyncs": getattr(pol, "resyncs", getattr(getattr(pol, "tracker", None), "resyncs", 0)), "hangs": game.hangs,
                   "align_fuel_p95": align.get("fuel_nn_p95_m"), **summ}
            with lock:
                results.append(row)
                print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(len(games)) as pool:
        list(pool.map(worker, games))
    for g in games:
        g.close(quit_game=not args.keep_games)
    scores = [r["score"] for r in results]
    wall = time.time() - t0
    for a in range(len(arms)):
        sa = [r["score"] for r in results if r["arm"] == a]
        au = [r["auto"] for r in results if r["arm"] == a]
        print(f"arm {a} {json.dumps(arms[a])}: n={len(sa)} mean {np.mean(sa):.1f} sd {np.std(sa):.1f} "
              f"se {np.std(sa) / max(1, len(sa) - 1) ** 0.5:.1f}  auto mean {np.mean(au):.1f} min {min(au)}")
    print(f"ghost {ghost.final}  bot scores {sorted(scores)}  mean {np.mean(scores):.1f}  sd {np.std(scores):.1f}  "
          f"{len(scores)} eps in {wall:.0f}s = {len(scores) / wall * 3600:.1f} eps/h on {len(games)} instances")
    (out / "summary.json").write_text(json.dumps({"ghost": ghost.name, "ghost_final": ghost.final, "params": params,
                                                  "ports": ports, "wall_s": wall, "episodes": results}, indent=1))


if __name__ == "__main__":
    main()
