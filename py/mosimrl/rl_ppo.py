"""PPO for the ball-seeing residual policy on N headless MoSim instances.

    python -m mosimrl.rl_ppo --run r1 --ports 47500-47507 --episodes-per-iter 24 --iters 200 \
        --leash '{"neutral":0.333,"conveyor":0.333}' --act-dims 1

Each iteration: every worker (one per game instance, own thread, own CPU copy of the network) plays episodes with
the current policy; one episode per iteration is a u = 0 control (the plain tracker, same code path) and one is a
deterministic evaluation of the mean policy — both logged, neither trained on. The learner (MPS) then runs PPO on
the stochastic episodes. Everything goes to runs/rl/<run>/ (metrics.jsonl, ckpt-*.pt, episodes.jsonl).

Reward per decision: Δblue + γΦ(s') − Φ(s) (shaping.py), scaled by REWARD_SCALE. Policy loss only on decisions
where the residual can act (leash > 0); value loss everywhere. Hangs: the partial episode is dropped (gamectl).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .gamectl import Game, robust_episode
from .ghost import Ghost
from .nets import ResidualNet
from .obs import residual_obs
from .residual import ResidualAgent, zero_chooser
from .shaping import GAMMA, potential

REWARD_SCALE = 0.05
ROOT = Path(__file__).resolve().parents[2]


class NetChooser:
    """Chooser for ResidualAgent: runs the network on the Bridge state and records the transition data."""

    def __init__(self, net: ResidualNet, act_dims: int, deterministic: bool, seed: int):
        self.net, self.act_dims, self.det = net, act_dims, deterministic
        self.rng = np.random.default_rng(seed)
        self.buf: list[dict] = []

    def reset(self):
        self.buf = []

    def __call__(self, s, agent):
        o = residual_obs(s, agent)
        with torch.no_grad():
            f = torch.from_numpy(o["field"])[None]
            e = torch.from_numpy(o["ego"])[None]
            v_ = torch.from_numpy(o["vec"])[None]
            mu, std, v = self.net(f, e, v_)
        mu, std, v = mu[0].numpy(), std[0].numpy(), float(v[0])
        u = mu.copy()
        if not self.det:
            u[:self.act_dims] += std[:self.act_dims] * self.rng.standard_normal(self.act_dims)
        u[self.act_dims:] = 0.0
        d = self.act_dims
        logp = float(-0.5 * (((u[:d] - mu[:d]) / std[:d]) ** 2).sum() - np.log(std[:d]).sum() - 0.5 * d * math.log(2 * math.pi))
        active = (o["L"] > 0.05) or (o["W"] > 0.05 and d > 1)
        self.buf.append({"field": o["field"].astype(np.float16), "ego": o["ego"].astype(np.float16), "vec": o["vec"],
                         "u": u.astype(np.float32), "logp": logp, "v": v, "active": active,
                         "pot": potential(s), "blue": float(s.get("blue", 0)), "phase": o["phase"]})
        return np.clip(u, -1.0, 1.0)


def run_episode(client, agent: ResidualAgent) -> dict:
    agent.reset()
    s = client.reset()
    wall0 = time.time()
    while not s.get("done"):
        vx, vz, rot, b, info = agent.act(s)
        s = client.act(vx, vz, rot, b)
    return {"score": int(s.get("blue", 0)), "auto": int(s.get("blueAuto", 0)), "wall_s": round(time.time() - wall0, 1),
            "unsticks": agent.tracker.unsticks, "resyncs": agent.tracker.resyncs}


def episode_arrays(buf: list[dict], final_blue: float, gamma: float = GAMMA, lam: float = 0.95) -> dict:
    n = len(buf)
    rew = np.zeros(n, np.float32)
    for k in range(n):
        if k + 1 < n:
            rew[k] = buf[k + 1]["blue"] - buf[k]["blue"] + gamma * buf[k + 1]["pot"] - buf[k]["pot"]
        else:
            rew[k] = final_blue - buf[k]["blue"] - buf[k]["pot"]
    rew *= REWARD_SCALE
    val = np.array([b["v"] for b in buf], np.float32)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for k in reversed(range(n)):
        nv = val[k + 1] if k + 1 < n else 0.0
        delta = rew[k] + gamma * nv - val[k]
        last = delta + gamma * lam * last
        adv[k] = last
    return {"field": np.stack([b["field"] for b in buf]), "ego": np.stack([b["ego"] for b in buf]),
            "vec": np.stack([b["vec"] for b in buf]), "u": np.stack([b["u"] for b in buf]),
            "logp": np.array([b["logp"] for b in buf], np.float32), "adv": adv, "ret": adv + val,
            "active": np.array([b["active"] for b in buf], bool), "rew": rew}


def parse_ports(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--demo", default=str(ROOT / "run/demos/demo-20260926-015440-m1.jsonl"))
    ap.add_argument("--ports", default="47500-47507")
    ap.add_argument("--episodes-per-iter", type=int, default=24)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--leash", default='{"neutral":0.333,"conveyor":0.333}')
    ap.add_argument("--act-dims", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--ent", type=float, default=0.001)
    ap.add_argument("--mu-l2", type=float, default=0.05)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--value-warmup-iters", type=int, default=2)
    ap.add_argument("--resume", default="", help="checkpoint to start from")
    ap.add_argument("--time-scale", type=float, default=2.0)
    ap.add_argument("--decide-every", type=int, default=10, help="Bridge steps (0.099 s) per residual decision")
    ap.add_argument("--lam", type=float, default=0.95)
    args = ap.parse_args()

    out = ROOT / "runs/rl" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.set_num_threads(2)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ghost = Ghost.from_file(args.demo)
    leash = json.loads(args.leash)
    gamma_dec = 0.9975 ** args.decide_every          # ≈ 0.995 per 0.2 s, half-life ~27 s whatever the decision rate

    net = ResidualNet(k=2)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); start_it = ck.get("it", 0) + 1
    net.to(dev)

    ports = parse_ports(args.ports)
    games = [Game(port=p, time_scale=args.time_scale) for p in ports]
    threads_up = []
    for g in games:
        th = threading.Thread(target=g.attach_or_launch); th.start(); threads_up.append(th)
    for th in threads_up:
        th.join()

    jobs: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    state = {"cpu_sd": None, "version": -1}
    lock = threading.Lock()

    def worker(game: Game, wid: int):
        local = ResidualNet(k=2); local.eval()
        have = -2
        torch.set_num_threads(1)
        while True:
            job = jobs.get()
            if job is None:
                return
            with lock:
                if have != state["version"]:
                    local.load_state_dict(state["cpu_sd"]); have = state["version"]
            kind = job["kind"]
            if kind == "control":
                chooser = zero_chooser
            else:
                chooser = NetChooser(local, args.act_dims, deterministic=(kind == "eval"), seed=job["seed"])
            agent = ResidualAgent(ghost, chooser, leash_scale=leash, decide_every=args.decide_every)
            try:
                r = robust_episode(game, lambda c: run_episode(c, agent))
                item = {"job": job, "port": game.port, **r}
                if kind != "control":
                    buf = chooser.buf
                    if kind == "train":
                        item["arrays"] = episode_arrays(buf, r["score"], gamma=gamma_dec, lam=args.lam)
                    act_u = [abs(b["u"][0]) for b in buf if b["active"]]
                    item["mean_abs_u"] = float(np.mean(act_u)) if act_u else 0.0
            except Exception as e:                                  # give up on this episode, keep the fleet alive
                import traceback
                item = {"job": job, "error": repr(e), "tb": traceback.format_exc()[-800:], "port": game.port}
            results.put(item)

    ws = [threading.Thread(target=worker, args=(g, i), daemon=True) for i, g in enumerate(games)]
    for w in ws:
        w.start()

    metrics = (out / "metrics.jsonl").open("a")
    epilog = (out / "episodes.jsonl").open("a")
    for it in range(start_it, args.iters):
        t0 = time.time()
        with lock:
            state["cpu_sd"] = {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}
            state["version"] = it
        n = args.episodes_per_iter
        kinds = ["control", "eval"] + ["train"] * (n - 2)
        for i, k in enumerate(kinds):
            jobs.put({"kind": k, "it": it, "seed": it * 1000 + i})
        got = [results.get() for _ in kinds]
        errs = [g for g in got if "error" in g]
        train = [g for g in got if g["job"]["kind"] == "train" and "arrays" in g]
        ctrl = [g["score"] for g in got if g["job"]["kind"] == "control" and "score" in g]
        ev = [g["score"] for g in got if g["job"]["kind"] == "eval" and "score" in g]
        for g in got:
            epilog.write(json.dumps({k: v for k, v in g.items() if k != "arrays"}) + "\n")
        epilog.flush()
        collect_s = time.time() - t0

        # ---------------------------------------------------------------- PPO update
        B = {k: np.concatenate([g["arrays"][k] for g in train]) for k in train[0]["arrays"]} if train else None
        stats = {}
        if B is not None:
            tb = {k: torch.from_numpy(np.asarray(v, dtype=np.float32 if k != "active" else bool)).to(dev)
                  for k, v in B.items() if k != "rew"}
            N = len(B["adv"])
            act = tb["active"]
            adv = tb["adv"]
            if act.any():
                a_act = adv[act]
                adv = (adv - a_act.mean()) / (a_act.std() + 1e-8)
            d = args.act_dims
            policy_on = it - start_it >= args.value_warmup_iters
            kls, clipf, vls, pls = [], [], [], []
            stop = False
            for ep_i in range(args.epochs):
                perm = torch.randperm(N, device=dev)
                for i0 in range(0, N, args.minibatch):
                    idx = perm[i0:i0 + args.minibatch]
                    mu, std, v = net(tb["field"][idx], tb["ego"][idx], tb["vec"][idx])
                    dist = torch.distributions.Normal(mu[:, :d], std[:, :d])
                    logp = dist.log_prob(tb["u"][idx][:, :d]).sum(-1)
                    ratio = (logp - tb["logp"][idx]).exp()
                    m = tb["active"][idx].float()
                    a = adv[idx]
                    pl = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * m).sum() / m.sum().clamp(min=1)
                    vl = ((v - tb["ret"][idx]) ** 2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    l2 = (mu[:, :d] ** 2 * m[:, None]).sum() / m.sum().clamp(min=1)
                    loss = 0.5 * vl + (pl - args.ent * ent + args.mu_l2 * l2 if policy_on else 0.0)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - (logp - tb["logp"][idx])).mul(m).sum() / m.sum().clamp(min=1)
                        kls.append(float(kl)); clipf.append(float(((ratio - 1).abs() > args.clip).float().mul(m).sum() / m.sum().clamp(min=1)))
                        vls.append(float(vl)); pls.append(float(pl))
                if policy_on and np.mean(kls[-max(1, N // args.minibatch):]) > args.target_kl:
                    stop = True
                    break
            with torch.no_grad():
                ev_ = 1 - float(((tb["ret"] - torch.from_numpy(np.concatenate([g["arrays"]["ret"] - g["arrays"]["adv"] for g in train])).to(dev)) ** 2).mean()
                                / (tb["ret"].var() + 1e-8))
            stats = {"kl": float(np.mean(kls)), "clipfrac": float(np.mean(clipf)), "vloss": float(np.mean(vls)),
                     "ploss": float(np.mean(pls)), "early_stop": stop, "policy_on": policy_on, "N": N,
                     "active_frac": float(B["active"].mean()), "value_ev": ev_,
                     "std": [round(float(x), 3) for x in net.log_std.detach().exp().cpu()]}
        train_scores = [g["score"] for g in train]
        row = {"it": it, "time": time.strftime("%H:%M:%S"), "collect_s": round(collect_s, 1), "update_s": round(time.time() - t0 - collect_s, 1),
               "train_mean": float(np.mean(train_scores)) if train_scores else None, "train_sd": float(np.std(train_scores)) if train_scores else None,
               "n_train": len(train_scores), "control": ctrl, "eval": ev, "errors": len(errs),
               "mean_abs_u": float(np.mean([g.get("mean_abs_u", 0) for g in train])) if train else None,
               "unsticks": float(np.mean([g["unsticks"] for g in got if "unsticks" in g])),
               "hangs": sum(g.hangs for g in games), **stats}
        metrics.write(json.dumps(row) + "\n"); metrics.flush()
        print(json.dumps(row), flush=True)
        torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "opt": opt.state_dict(), "it": it,
                    "args": vars(args)}, out / "ckpt-last.pt")
        if it % 10 == 0:
            torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "it": it}, out / f"ckpt-{it:04d}.pt")

    for _ in ws:
        jobs.put(None)
    for g in games:
        g.close(quit_game=True)


if __name__ == "__main__":
    main()
