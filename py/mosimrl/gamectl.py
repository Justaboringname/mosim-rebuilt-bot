"""Launch / watchdog / relaunch MoSimulator instances for bot runs (several at once, headless by default).

Each instance gets its own port (-mosimrl-port) and its own Unity log (-logFile), and by default runs with
-batchmode -nographics: no window, no graphics device, no GPU use — the user asked that bot runs never render.

MoSim sometimes hangs natively (every thread idle-waiting, main thread blocked inside a Unity frame, no managed
code on any stack; cause unknown — see runs/hangs/). Long runs therefore treat an instance as something that can
die: every Bridge call has a wall timeout, and on a timeout the watchdog samples the process into runs/hangs/,
SIGKILLs it, relaunches, and the caller retries the episode.

Only ever kills instances this module launched: they are identified by their -mosimrl-port argument, which
human-launched sessions never carry.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .client import BridgeError, MoSimClient

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "run"
APP = Path.home() / "Library/Application Support/Steam/steamapps/common/MoSimulator/MoSimulator.app"
EXE = APP / "Contents/MacOS/MoSimulator"
LOGS = ROOT / "runs/logs"
HANGS = ROOT / "runs/hangs"
HEADLESS = ["-batchmode", "-nographics"]


def _port_pids(port: int) -> list[int]:
    out = subprocess.run(["pgrep", "-f", f"{EXE}.*-mosimrl-port {port}( |$)"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_flags() -> None:
    (RUN / "ENABLE").touch(); (RUN / "BRIDGE").touch()
    for f in ("RECORD", "PROBE", "AUTOSTART", "AUTOQUIT"):
        (RUN / f).unlink(missing_ok=True)


FLAGS = ("RECORD", "BRIDGE", "PROBE", "AUTOSTART", "AUTOQUIT")
_flag_lock = threading.Lock()
_flag_users = 0
_flag_saved: dict | None = None


def _flags_acquire() -> None:
    """Switch the harness flags to bot mode for a launch. Launches run in parallel threads: the first one saves the
    user's flags (e.g. ENABLE + RECORD for their own Steam sessions) and the last one to finish puts them back, so
    a thread never 'restores' another thread's bot-mode flags (that race once left BRIDGE on and RECORD off)."""
    global _flag_users, _flag_saved
    with _flag_lock:
        if _flag_users == 0:
            _flag_saved = {f: (RUN / f).exists() for f in FLAGS}
            ensure_flags()
        _flag_users += 1


def _flags_release() -> None:
    global _flag_users, _flag_saved
    with _flag_lock:
        _flag_users -= 1
        if _flag_users == 0 and _flag_saved is not None:
            for f, present in _flag_saved.items():
                if present:
                    (RUN / f).touch()
                else:
                    (RUN / f).unlink(missing_ok=True)
            _flag_saved = None


class Game:
    def __init__(self, port: int = 47500, time_scale: float = 2.0, headless: bool = True, cameras: bool = False,
                 call_timeout: float = 60.0, extra_args: list[str] | None = None, steps_per_frame: int | None = None):
        self.port, self.time_scale, self.headless, self.cameras = port, time_scale, headless, cameras
        # steps_per_frame N > 0: fixed frame step (N physics steps per frame, CPU-bound, deterministic interleave)
        self.steps_per_frame = steps_per_frame
        self.call_timeout = call_timeout
        self.extra_args = extra_args or []
        self.pid: int | None = None
        self.client: MoSimClient | None = None
        self.hangs = 0
        self.last_cmd = ""
        self.log = LOGS / f"player-{port}.log"

    # ------------------------------------------------------------------ lifecycle
    def launch(self, wait_s: float = 180.0) -> None:
        if _port_pids(self.port):
            raise RuntimeError(f"an instance on port {self.port} is already running")
        _flags_acquire()
        try:
            LOGS.mkdir(parents=True, exist_ok=True)
            # -mosimrl-owner: the instance quits by itself once this process is gone (Bridge.CheckOwner); `open`
            # detaches it from us, and orphaned headless players once blocked a macOS restart
            args = ["open", "-n", "-g", str(APP), "--args", "-mosimrl-port", str(self.port), "-logFile", str(self.log),
                    "-mosimrl-owner", str(os.getpid()), "-mosimrl-run", str(RUN), *(HEADLESS if self.headless else []), *self.extra_args]
            subprocess.run(args, check=True)
            t0 = time.time()
            while time.time() - t0 < wait_s:
                pids = _port_pids(self.port)
                if pids:
                    self.pid = max(pids)
                if self.pid and _listening(self.port):
                    break
                time.sleep(1.0)
            else:
                raise RuntimeError(f"port {self.port}: bridge never started listening")
        finally:
            # the instance read its flags at Awake; put back what was there (e.g. RECORD for the human's own sessions)
            _flags_release()
        self.connect()

    def connect(self) -> None:
        self.client = MoSimClient(port=self.port, timeout=self.call_timeout)
        self.client.hello()
        info = self.client.config(time_scale=self.time_scale, cameras=self.cameras, steps_per_frame=self.steps_per_frame)
        print(f"[gamectl:{self.port}] connected pid={self.pid} {info}", flush=True)

    def attach_or_launch(self) -> None:
        pids = _port_pids(self.port)
        if pids and _listening(self.port):
            self.pid = max(pids)
            self.connect()
        else:
            for p in pids:                                    # half-dead instance of ours on this port
                self._kill(p)
            self.launch()

    def _kill(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        for _ in range(40):
            if pid not in _port_pids(self.port):
                return
            time.sleep(0.25)

    def recover(self, reason: str) -> None:
        """Called after a Bridge call timed out or the socket died."""
        self.hangs += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = HANGS / f"{stamp}-p{self.port}"
        d.mkdir(parents=True, exist_ok=True)
        if self.client:
            self.last_cmd = self.client.last_cmd
            self.client.close()
            self.client = None
        (d / "reason.txt").write_text(f"{reason}\nlast_cmd={self.last_cmd}\nheadless={self.headless} "
                                      f"cameras={self.cameras} time_scale={self.time_scale} extra={self.extra_args}\n")
        for pid in _port_pids(self.port):
            subprocess.run(["sample", str(pid), "2", "-file", str(d / "sample.txt")], capture_output=True, timeout=60)
            if self.log.exists():
                shutil.copy(self.log, d / "Player.log")
            self._kill(pid)
        print(f"[gamectl:{self.port}] hang #{self.hangs} ({reason}); evidence in {d}; relaunching", flush=True)
        self.pid = None
        self.launch()

    def close(self, quit_game: bool = False) -> None:
        if self.client:
            try:
                self.client.release()
            except Exception:
                pass
            self.client.close()
            self.client = None
        if quit_game:
            for pid in _port_pids(self.port):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


def robust_episode(game: Game, run, max_retries: int = 3):
    """run(client) -> result. Retries on hang/socket death after recovering the game; partial episodes are dropped."""
    for attempt in range(max_retries + 1):
        try:
            return run(game.client)
        except (socket.timeout, TimeoutError, ConnectionError, BridgeError, OSError) as e:
            if isinstance(e, BridgeError) and "closed" not in str(e):
                raise
            if attempt == max_retries:
                raise
            game.recover(f"{type(e).__name__}: {e}")
