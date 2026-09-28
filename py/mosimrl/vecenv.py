"""N headless MoSim instances stepped in lock-step from one Python process (for RL data collection).

    env = VecMoSim(ports=range(47500, 47512))
    states = env.reset_all()                 # list of Bridge state dicts
    states, rewards, dones, infos = env.step(actions)   # actions: list of (vx, vz, rot, buttons)

Each instance is driven by its own thread (the Bridge calls block ~50 ms at timeScale 2). An instance that hangs
or dies is recovered by gamectl (sample → SIGKILL → relaunch) and its episode is marked `aborted` in infos; the
caller should drop that partial episode. Finished episodes are reset automatically on the next step
(`dones[i]` is True on the step that returned the terminal state; the following state is a fresh reset).
Reward = change in blue score since the previous state (points), the only game reward.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor

from .client import BridgeError
from .gamectl import Game

_RECOVERABLE = (socket.timeout, TimeoutError, ConnectionError, BridgeError, OSError)


class VecMoSim:
    def __init__(self, ports, time_scale: float = 2.0, call_timeout: float = 60.0):
        self.games = [Game(port=p, time_scale=time_scale, call_timeout=call_timeout) for p in ports]
        self.n = len(self.games)
        self.pool = ThreadPoolExecutor(self.n)
        list(self.pool.map(lambda g: g.attach_or_launch(), self.games))
        self.last = [None] * self.n
        self.episode_ids = [0] * self.n
        self.need_reset = [True] * self.n

    def _reset_one(self, i: int) -> dict:
        g = self.games[i]
        for attempt in range(4):
            try:
                s = g.client.reset()
                self.last[i] = s
                self.need_reset[i] = False
                self.episode_ids[i] += 1
                return s
            except _RECOVERABLE as e:
                if isinstance(e, BridgeError) and "closed" not in str(e):
                    raise
                g.recover(f"reset: {type(e).__name__}: {e}")
        raise RuntimeError(f"port {g.port}: reset failed repeatedly")

    def reset_all(self) -> list[dict]:
        return list(self.pool.map(self._reset_one, range(self.n)))

    def _step_one(self, i: int, action) -> tuple[dict, float, bool, dict]:
        g = self.games[i]
        if self.need_reset[i]:
            s = self._reset_one(i)
            return s, 0.0, False, {"reset": True}
        vx, vz, rot, buttons = action
        try:
            s = g.client.act(vx, vz, rot, buttons)
        except _RECOVERABLE as e:
            if isinstance(e, BridgeError) and "closed" not in str(e):
                raise
            g.recover(f"act: {type(e).__name__}: {e}")
            s = self._reset_one(i)
            return s, 0.0, False, {"aborted": True, "reset": True}
        r = float(s.get("blue", 0) - self.last[i].get("blue", 0))
        self.last[i] = s
        done = bool(s.get("done"))
        if done:
            self.need_reset[i] = True
        return s, r, done, {"score": s.get("blue", 0)} if done else {}

    def step(self, actions) -> tuple[list[dict], list[float], list[bool], list[dict]]:
        out = list(self.pool.map(lambda ia: self._step_one(*ia), enumerate(actions)))
        states, rewards, dones, infos = (list(x) for x in zip(*out))
        return states, rewards, dones, infos

    def close(self, quit_games: bool = True) -> None:
        for g in self.games:
            g.close(quit_game=quit_games)
        self.pool.shutdown(wait=False)
