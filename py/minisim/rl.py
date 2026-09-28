"""PPO for the ball-seeing ghost-residual policy, trained in MiniSim (proposal generator; real A/B decides).

    python -m minisim.rl --run s1 --iters 300

What is learned: a residual on the 1118 ghost tracker (mosimrl.residual): u[0] = lateral offset from the ghost's path
(fraction of the phase leash), u[1] = along-track time warp, decided every 0.5 s. u = 0 is the plain tracker, and the
actor's mean layer starts at zero, so training starts from the baseline. The sim cannot rank whole routes (1118 vs 1085
is reversed), so the action space is local to one ghost by construction.

What it sees (mosimrl.obs.residual_obs, identical in MiniSim and in the real game): every loose ball as counts on a
0.5 m field grid and a 0.25 m robot-frame grid (floor and airborne channels, hub interior masked on both sides), the
ghost's next seconds of path, robot / match / phase features and the 17-lane swath yield.

Workers: one process per core, each with its own MiniSim2 and a CPU copy of the network (weights broadcast through a
file each iteration). Every training episode draws domain-randomised sim parameters (DR); evaluation uses the
defaults on fixed seeds, paired against the u = 0 control on the same seeds (common random numbers).
Reward: Δscore + potential shaping (mosimrl.shaping), γ per 0.5 s decision.
Outputs: runs/rl_sim/<run>/ (metrics.jsonl, eval.jsonl, ckpt-*.pt).
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "run/demos/demo-20260926-015440-m1.jsonl")

# domain randomisation: name -> (kind, a, b); "mul" draws U(a, b) x default, "add" adds U(a, b)
DR = {
    "p_cap_full": ("mul", 0.8, 1.2), "r_max": ("mul", 0.93, 1.07), "launch_v_scale": ("mul", 0.97, 1.03),
    "K": ("mul", 0.95, 1.05), "D": ("mul", 0.95, 1.05), "hub_t_mu": ("add", -0.2, 0.2),
    "exit_v0": ("mul", 0.85, 1.15), "climb_drive": ("mul", 0.7, 1.3), "mu_field": ("mul", 0.8, 1.2),
    "bump_drive": ("mul", 0.9, 1.1),
}

_W = {}


def _init(run_dir, leash, decide_every, act_dims, clearance=True):
    import torch
    torch.set_num_threads(1)
    from mosimrl.ghost import Ghost
    _W.update(run_dir=Path(run_dir), leash=leash, every=decide_every, k=act_dims, clearance=clearance,
              ghost=Ghost.from_file(DEMO),
              net=None, version=-1)


def _load_net(version):
    import torch
    from mosimrl.nets import ResidualNet
    if _W["version"] != version:
        ck = torch.load(_W["run_dir"] / "cur.pt", map_location="cpu")
        if _W["net"] is None:
            _W["net"] = ResidualNet(k=2)
            _W["net"].eval()
        _W["net"].load_state_dict(ck["net"])
        _W["version"] = version
    return _W["net"]


def dr_params(rng):
    from .sim2 import DEFAULT
    p = {}
    for name, (kind, a, b) in DR.items():
        u = rng.uniform(a, b)
        p[name] = DEFAULT[name] * u if kind == "mul" else DEFAULT[name] + u
    return p


def _episode(job):
    """job: kind in {train, eval, control}, seed, version, gamma, lam, dr."""
    from mosimrl.episode import MatchClock
    from mosimrl.residual import ResidualAgent, zero_chooser
    from mosimrl.rl_ppo import NetChooser, episode_arrays
    from .sim2 import MiniSim2Client
    kind, seed = job["kind"], job["seed"]
    rng = np.random.default_rng(seed + 7_000_000)
    params = dr_params(rng) if job.get("dr") else {}
    if kind == "control":
        chooser = zero_chooser
    else:
        chooser = NetChooser(_load_net(job["version"]), _W["k"], deterministic=(kind == "eval"), seed=seed)
    agent = ResidualAgent(_W["ghost"], chooser, leash_scale=_W["leash"], decide_every=_W["every"],
                          clearance=_W["clearance"])
    c = MiniSim2Client(params, seed=seed)
    agent.reset()
    clock = MatchClock()
    s = c.reset(); s["t"] = clock(s)
    n = 0
    while not s.get("done") and n < 5000:
        vx, vz, rot, b, _ = agent.act(s)
        s = c.act(vx, vz, rot, b); s["t"] = clock(s)
        n += 1
    out = {"kind": kind, "seed": seed, "score": int(s["blue"]), "unsticks": agent.tracker.unsticks}
    if kind == "train":
        out["arrays"] = episode_arrays(chooser.buf, s["blue"], gamma=job["gamma"], lam=job["lam"])
    if kind != "control":
        act_u = np.array([bb["u"] for bb in chooser.buf if bb["active"]])
        out["mean_abs_u"] = np.abs(act_u).mean(0).round(3).tolist() if len(act_u) else [0.0, 0.0]
    return out


def main():
    import torch
    from mosimrl.nets import ResidualNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=64, help="training episodes per iteration")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--leash", default='{"neutral":0.333,"conveyor":0.333,"harvest":0.5,"shoot":0.5}')
    ap.add_argument("--decide-every", type=int, default=5)
    ap.add_argument("--act-dims", type=int, default=1, help="1 = lateral only (s2: the warp lost in real, ab_c50)")
    ap.add_argument("--no-clearance", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--mu-l2", type=float, default=0.02)
    ap.add_argument("--target-kl", type=float, default=0.015)
    ap.add_argument("--value-warmup-iters", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=48)
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log-std0", type=float, default=-2.0, help="initial exploration log-std (std 0.135 = 0.13 m lateral)")
    a = ap.parse_args()

    out = ROOT / "runs/rl_sim" / a.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = ResidualNet(k=2, log_std0=a.log_std0)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    it0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); it0 = ck["it"] + 1
    net.to(dev)
    leash = json.loads(a.leash)
    pool = get_context("spawn").Pool(a.procs, initializer=_init,
                                     initargs=(str(out), leash, a.decide_every, a.act_dims, not a.no_clearance))

    eval_seeds = [900_000 + i for i in range(a.eval_seeds)]
    cpath = out / "control.json"
    if cpath.exists():
        control = {int(k): v for k, v in json.loads(cpath.read_text()).items()}
    else:
        t0 = time.time()
        res = pool.map(_episode, [{"kind": "control", "seed": s} for s in eval_seeds], chunksize=1)
        control = {r["seed"]: r["score"] for r in res}
        cpath.write_text(json.dumps(control))
        dr_ctrl = pool.map(_episode, [{"kind": "control", "seed": 10_000_000 + i, "dr": True} for i in range(64)], chunksize=1)
        (out / "control_dr.json").write_text(json.dumps([r["score"] for r in dr_ctrl]))
        print(f"control (u=0) on {len(eval_seeds)} eval seeds: {np.mean(list(control.values())):.1f}; with DR "
              f"{np.mean([r['score'] for r in dr_ctrl]):.1f} ± {np.std([r['score'] for r in dr_ctrl]):.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    metrics = (out / "metrics.jsonl").open("a")
    evlog = (out / "eval.jsonl").open("a")
    k = a.act_dims
    for it in range(it0, it0 + a.iters):
        t0 = time.time()
        torch.save({"net": {kk: v.detach().cpu() for kk, v in net.state_dict().items()}}, out / "cur.pt")
        jobs = [{"kind": "train", "seed": it * 10_000 + i, "version": it, "gamma": a.gamma, "lam": a.lam,
                 "dr": not a.no_dr} for i in range(a.episodes)]
        do_eval = it % a.eval_every == 0
        if do_eval:
            jobs += [{"kind": "eval", "seed": s, "version": it} for s in eval_seeds]
        res = pool.map(_episode, jobs, chunksize=1)
        train = [r for r in res if r["kind"] == "train"]
        collect_s = time.time() - t0

        B = {kk: np.concatenate([r["arrays"][kk] for r in train]) for kk in train[0]["arrays"]}
        tb = {kk: torch.from_numpy(np.asarray(v, dtype=np.float32 if kk != "active" else bool)).to(dev)
              for kk, v in B.items() if kk != "rew"}
        N = len(B["adv"])
        act = tb["active"]
        adv = tb["adv"]
        if act.any():
            aa = adv[act]
            adv = (adv - aa.mean()) / (aa.std() + 1e-8)
        policy_on = it - it0 >= a.value_warmup_iters or bool(a.resume)
        kls, clipf, vls, pls = [], [], [], []
        stop = False
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for i0 in range(0, N, a.minibatch):
                idx = perm[i0:i0 + a.minibatch]
                mu, std, v = net(tb["field"][idx], tb["ego"][idx], tb["vec"][idx])
                dist = torch.distributions.Normal(mu[:, :k], std[:, :k])
                logp = dist.log_prob(tb["u"][idx][:, :k]).sum(-1)
                ratio = (logp - tb["logp"][idx]).exp()
                m = tb["active"][idx].float()
                ad = adv[idx]
                pl = -(torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad) * m).sum() / m.sum().clamp(min=1)
                vl = ((v - tb["ret"][idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                l2 = (mu[:, :k] ** 2 * m[:, None]).sum() / m.sum().clamp(min=1)
                loss = 0.5 * vl + ((pl - a.ent * ent + a.mu_l2 * l2) if policy_on else 0.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - tb["logp"][idx])).mul(m).sum() / m.sum().clamp(min=1)
                    kls.append(float(kl)); vls.append(float(vl)); pls.append(float(pl))
                    clipf.append(float(((ratio - 1).abs() > a.clip).float().mul(m).sum() / m.sum().clamp(min=1)))
            if policy_on and np.mean(kls[-max(1, N // a.minibatch):]) > a.target_kl:
                stop = True
                break
        with torch.no_grad():
            v_old = torch.from_numpy(np.concatenate([r["arrays"]["ret"] - r["arrays"]["adv"] for r in train])).to(dev)
            ev = 1 - float(((tb["ret"] - v_old) ** 2).mean() / (tb["ret"].var() + 1e-8))
        ts = [r["score"] for r in train]
        row = {"it": it, "time": time.strftime("%H:%M:%S"), "collect_s": round(collect_s, 1),
               "update_s": round(time.time() - t0 - collect_s, 1), "train_mean": round(float(np.mean(ts)), 1),
               "train_sd": round(float(np.std(ts)), 1), "N": N, "active_frac": round(float(B["active"].mean()), 3),
               "mean_abs_u": np.mean([r["mean_abs_u"] for r in train], 0).round(3).tolist(),
               "unsticks": round(float(np.mean([r["unsticks"] for r in train])), 2),
               "kl": round(float(np.mean(kls)), 4), "clipfrac": round(float(np.mean(clipf)), 3),
               "vloss": round(float(np.mean(vls)), 3), "value_ev": round(ev, 3), "early_stop": stop,
               "policy_on": policy_on, "std": [round(float(x), 3) for x in net.log_std.detach().exp().cpu()]}
        if do_eval:
            evs = {r["seed"]: r["score"] for r in res if r["kind"] == "eval"}
            d = np.array([evs[s] - control[s] for s in eval_seeds], float)
            erow = {"it": it, "eval": round(float(np.mean(list(evs.values()))), 1),
                    "control": round(float(np.mean([control[s] for s in eval_seeds])), 1),
                    "delta": round(float(d.mean()), 1), "delta_se": round(float(d.std(ddof=1) / len(d) ** 0.5), 1),
                    "mean_abs_u": np.mean([r["mean_abs_u"] for r in res if r["kind"] == "eval"], 0).round(3).tolist()}
            evlog.write(json.dumps(erow) + "\n"); evlog.flush()
            row["eval_delta"] = erow["delta"]; row["eval_delta_se"] = erow["delta_se"]
        metrics.write(json.dumps(row) + "\n"); metrics.flush()
        print(json.dumps(row), flush=True)
        ck = {"net": {kk: v.cpu() for kk, v in net.state_dict().items()}, "opt": opt.state_dict(), "it": it, "k": 2,
              "args": vars(a)}
        torch.save(ck, out / "ckpt-last.pt")
        if do_eval:
            torch.save({"net": ck["net"], "it": it, "k": 2, "args": vars(a)}, out / f"ckpt-{it:04d}.pt")
    pool.close()


if __name__ == "__main__":
    main()
