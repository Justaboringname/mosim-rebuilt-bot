"""Observation encoding shared by BC, RL and the ghost-residual policy: how a network "sees" the balls.

Two ball views, both counts of loose balls per cell (carried balls are already excluded by the Bridge/Recorder):
  field_grid  whole field in the world frame (x across, z down), fixed 0.5 m cells → 34 x 17
  ego_grid    a square window centred on the robot and rotated into its frame (+z forward = intake side),
              0.25 m cells → 32 x 32 over 8 m x 8 m
plus a flat vector of robot / match / ghost features. All functions are pure numpy and work on either a Bridge
state dict or a demo record (same field names: t, x, z, yaw, vx, vz, held, blue, fuel).
"""

from __future__ import annotations

import math

import numpy as np

from . import field as F
from .shifts import blue_active, seconds_until_blue_active, seconds_until_blue_inactive

FIELD_RES = 0.5
FIELD_W = int(math.ceil(2 * F.X_WALL / FIELD_RES))      # 34 cells along x
FIELD_H = int(math.ceil(2 * F.Z_WALL / FIELD_RES))      # 17 cells along z
EGO_RES = 0.25
EGO_N = 32                                               # 8 m window
COUNT_SCALE = 4.0                                        # cell count / 4 → ~[0, 1+]


def fuel_array(fuel) -> np.ndarray:
    a = np.asarray(fuel if fuel is not None else [], dtype=np.float32)
    return a.reshape(-1, 3)[:, :2] if a.size else np.zeros((0, 2), np.float32)


def field_grid(fuel) -> np.ndarray:
    xz = fuel_array(fuel)
    g = np.zeros((FIELD_H, FIELD_W), np.float32)
    if len(xz):
        ix = np.clip(((xz[:, 0] + F.X_WALL) / FIELD_RES).astype(int), 0, FIELD_W - 1)
        iz = np.clip(((xz[:, 1] + F.Z_WALL) / FIELD_RES).astype(int), 0, FIELD_H - 1)
        np.add.at(g, (iz, ix), 1.0)
    return g / COUNT_SCALE


def ego_grid(fuel, x: float, z: float, yaw_deg: float) -> np.ndarray:
    """Balls around the robot in its own frame: row 0 = far ahead, column 0 = robot's left.

    Unity yaw 0 faces +z, yaw 90 faces +x (clockwise seen from above). Forward f = (sin y, cos y) in (x, z);
    right r = (cos y, -sin y).
    """
    xz = fuel_array(fuel)
    g = np.zeros((EGO_N, EGO_N), np.float32)
    if not len(xz):
        return g
    y = math.radians(yaw_deg)
    dx, dz = xz[:, 0] - x, xz[:, 1] - z
    fwd = dx * math.sin(y) + dz * math.cos(y)
    right = dx * math.cos(y) - dz * math.sin(y)
    half = EGO_N * EGO_RES / 2
    col = ((right + half) / EGO_RES).astype(int)
    row = ((half - fwd) / EGO_RES).astype(int)
    ok = (col >= 0) & (col < EGO_N) & (row >= 0) & (row < EGO_N)
    np.add.at(g, (row[ok], col[ok]), 1.0)
    return g / COUNT_SCALE


def robot_vec(s: dict) -> np.ndarray:
    """Robot + match features (normalised). Works on Bridge states (s['robot'][...]) and demo records."""
    r = s.get("robot", s)
    t = float(s["t"])
    won = s.get("wonAuto", 1) == 1 if t <= 130 else True
    yaw = math.radians(r["yaw"])
    active = blue_active(t, won)
    return np.array([
        r["x"] / F.X_WALL, r["z"] / F.Z_WALL, math.sin(yaw), math.cos(yaw),
        r.get("vx", 0.0) / 2.7, r.get("vz", 0.0) / 2.7, r.get("wy", 0.0) / 6.0,
        s.get("held", 0) / 100.0,
        t / 160.0, float(active),
        min(seconds_until_blue_active(t, won), 30.0) / 30.0 if not active else 0.0,
        min(seconds_until_blue_inactive(t, won), 30.0) / 30.0 if active else 0.0,
        float(F.feedable(r["x"], r["z"])),
    ], dtype=np.float32)


ROBOT_DIM = 13


def ghost_vec(ghost, s: dict, horizons=(0.0, 0.5, 1.0, 2.0, 3.0)) -> np.ndarray:
    """Where the demo robot is at the same clock and a few seconds ahead, relative to us, in the robot frame,
    plus the demo's stick command, buttons and ball count now. Empty (zeros) if ghost is None."""
    out = np.zeros(GHOST_DIM, np.float32)
    if ghost is None:
        return out
    r = s.get("robot", s)
    t = float(s["t"])
    y = math.radians(r["yaw"])
    k = 0
    for h in horizons:
        g = ghost.at(t - h)
        dx, dz = g.x - r["x"], g.z - r["z"]
        out[k] = (dx * math.sin(y) + dz * math.cos(y)) / 4.0          # forward
        out[k + 1] = (dx * math.cos(y) - dz * math.sin(y)) / 4.0      # right
        dyaw = math.radians((g.yaw - r["yaw"] + 180.0) % 360.0 - 180.0)
        out[k + 2], out[k + 3] = math.sin(dyaw), math.cos(dyaw)
        k += 4
    g = ghost.at(t)
    out[k:k + 3] = g.cmd
    out[k + 3:k + 8] = g.buttons
    out[k + 8] = g.held / 100.0
    return out


GHOST_DIM = 5 * 4 + 3 + 5 + 1


# ---------------------------------------------------------------------------------------------------------------
# Residual-policy observation (docs/design/verdict-2026-09-26.md): what the ball-seeing residual network sees.

RES_FIELD_C = 4        # floor, airborne, ghost path next 5 s, robot footprint (+ heading cell)
RES_EGO_C = 2          # floor, ghost path next 3 s (robot frame)
PHASE_NAMES = ("opening", "crossing", "pause", "neutral", "conveyor", "harvest", "shoot")


def _field_cell(x, z):
    ix = int(np.clip((x + F.X_WALL) / FIELD_RES, 0, FIELD_W - 1))
    iz = int(np.clip((z + F.Z_WALL) / FIELD_RES, 0, FIELD_H - 1))
    return iz, ix


def _ego_cell(dx, dz, yaw_rad):
    fwd = dx * math.sin(yaw_rad) + dz * math.cos(yaw_rad)
    right = dx * math.cos(yaw_rad) - dz * math.sin(yaw_rad)
    half = EGO_N * EGO_RES / 2
    col = int((right + half) / EGO_RES)
    row = int((half - fwd) / EGO_RES)
    return row, col


def residual_obs(s: dict, agent) -> dict:
    """agent: residual.ResidualAgent (ghost, dec, phases, tracker). Returns float32 arrays field/ego/vec."""
    from .residual import SWATH_OFFSETS, floor_balls, path_normal, swath_yield
    from .shaping import fuel_counts

    r = s["robot"]
    t = float(s["t"])
    g = agent.ghost
    dec = agent.dec
    lag = getattr(agent.tracker, "lag", 0.0)
    tau = t - dec.warp + lag
    fuel = np.asarray(s.get("fuel") or [], np.float32).reshape(-1, 3)
    if len(fuel):
        # the hub interior is masked on both sides: MiniSim parks hub-state balls at the node while real ones roll
        # through the hub's ramps, so what a network would see there differs between the two worlds
        fuel = fuel[~((fuel[:, 0] > 3.0) & (fuel[:, 0] < 4.35) & (np.abs(fuel[:, 1]) < 0.7) & (fuel[:, 2] > 0.6))]
    floor = fuel[fuel[:, 2] < 0.25] if len(fuel) else fuel
    air = fuel[fuel[:, 2] >= 0.25] if len(fuel) else fuel

    field = np.zeros((RES_FIELD_C, FIELD_H, FIELD_W), np.float32)
    field[0] = field_grid(floor)
    field[1] = field_grid(air)
    for h in np.arange(0.0, 5.01, 0.25):
        f = g.at(tau - h)
        iz, ix = _field_cell(f.x, f.z)
        field[2, iz, ix] = max(field[2, iz, ix], 1.0 - h / 5.0)
    iz, ix = _field_cell(r["x"], r["z"])
    field[3, iz, ix] = 1.0
    y = math.radians(r["yaw"])
    iz, ix = _field_cell(r["x"] + 0.5 * math.sin(y), r["z"] + 0.5 * math.cos(y))
    field[3, iz, ix] = max(field[3, iz, ix], 0.5)

    ego = np.zeros((RES_EGO_C, EGO_N, EGO_N), np.float32)
    ego[0] = ego_grid(floor, r["x"], r["z"], r["yaw"])
    for h in np.arange(0.0, 3.01, 0.1):
        f = g.at(tau - h)
        row, col = _ego_cell(f.x - r["x"], f.z - r["z"], y)
        if 0 <= row < EGO_N and 0 <= col < EGO_N:
            ego[1, row, col] = max(ego[1, row, col], 1.0 - h / 3.0)

    ph, L, W = dec.limits(t - dec.warp)
    sw = swath_yield(g, tau, floor_balls(s.get("fuel")), path_normal(g, tau, dec.normal)) / 20.0
    stock, air_n = fuel_counts(s.get("fuel"))
    gf = g.at(tau)
    nxt = agent.phases.seconds_to_next_crossing(tau)
    c = agent.phases.next_crossing(tau)
    vec = np.concatenate([
        robot_vec(s),
        ghost_vec(g, {**s, "t": tau}),
        np.array([dec.lateral / 3.0, dec.warp / 2.0, L / 3.0, W / 2.0, lag / 5.0,
                  (s.get("held", 0) - gf.held) / 50.0, (s.get("blue", 0) - gf.blue) / 100.0,
                  min(nxt, 20.0) / 10.0, float(c is not None and c.kind == "trench"), float(c is not None and c.inbound),
                  stock / 300.0, air_n / 50.0, float(s.get("rs", 0) == 1)], np.float32),
        np.array([float(ph == p) for p in PHASE_NAMES], np.float32),
        sw.astype(np.float32),
    ]).astype(np.float32)
    return {"field": field, "ego": ego, "vec": vec, "phase": ph, "L": L, "W": W}


RES_VEC_DIM = ROBOT_DIM + GHOST_DIM + 13 + len(PHASE_NAMES) + 17
