"""REBUILT field for MiniSim: static geometry (collider boxes read from the game's field scene, see field_obbs.json),
the hub shift schedule and the starting fuel layout.

World frame = Unity world = MoSim field frame: x is the field length (+x = BLUE alliance end), z the width, yaw in
degrees with 0 = +z and 90 = +x. The field is point-symmetric (x, z) -> (-x, -z) between the alliances.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parents[2] / "run/demos/demo-20260926-015440-m1.jsonl"

X_WALL, Z_WALL = 8.2746, 4.031          # inner faces of the alliance walls / side walls
BALL_R = 0.075


def _mirror(b):
    (x0, x1), (z0, z1) = b
    return ((-x1, -x0), (-z1, -z0))


# Robot obstacles (2D AABBs, (x0, x1), (z0, z1)) on the blue side; the red side is the point mirror.
BLUE_ROBOT_BOXES = [
    ((3.050, 4.250), (-0.598, 0.602)),     # hub base
    ((3.050, 4.242), (2.453, 2.753)),      # left trench divider (triangles + groundbox)
    ((3.050, 4.242), (-2.759, -2.459)),    # right trench divider
    ((7.163, 7.253), (-0.159, -0.120)),    # tower uprights
    ((7.163, 7.253), (0.698, 0.737)),
]
# Trench overhead bars: only block a robot whose hopper is over-full (balls stacked above the 0.564 m bar).
BLUE_TRENCH_BARS = [((3.574, 3.724), (2.453, 4.253)), ((3.600, 3.700), (-4.116, -2.816))]
# Bump plates: drivable ramps (peak at x = 3.643, height 0.167 m); balls treat them as a low wall.
BLUE_BUMPS = [((3.074, 4.212), (0.598, 2.448)), ((3.080, 4.218), (-2.449, -0.599))]
BUMP_PEAK_X, BUMP_HALF, BUMP_H = 3.643, 0.57, 0.167

ROBOT_BOXES = BLUE_ROBOT_BOXES + [_mirror(b) for b in BLUE_ROBOT_BOXES]
TRENCH_BARS = BLUE_TRENCH_BARS + [_mirror(b) for b in BLUE_TRENCH_BARS]
BUMPS = BLUE_BUMPS + [_mirror(b) for b in BLUE_BUMPS]

# Balls: hub bases and trench dividers are walls; so are the side walls and the alliance walls except the corral
# openings under the outpost (balls roll under the outpost wall into the corral).
BALL_BOXES = [BLUE_ROBOT_BOXES[0], BLUE_ROBOT_BOXES[1], BLUE_ROBOT_BOXES[2]]
BALL_BOXES = BALL_BOXES + [_mirror(b) for b in BALL_BOXES]
CORRAL_OPEN_Z = [(2.985, 3.364), (3.404, 3.792)]          # blue side (z > 0); red is mirrored

# Outpost (Outpost.cs): chute door hinge, chute capacity 24, corral region behind the wall.
CHUTE_DOOR = (8.173, 3.7425)
OPEN_DISTANCE = 1.0
CHUTE_CAP = 24

# Pass nodes and hub aim node (AutoAngleNodes)
HUB_NODE = (3.655, -0.006)
PASS_NODES = [(5.91, -1.675), (5.91, 1.675)]


def blue_hub_active(t: float, gs: int) -> bool:
    """RebuiltShifts.Update for a solo blue robot (red never scores in auto, so blue wins auto)."""
    if gs == 0 or t >= 130.0 or t < 30.0:
        return True
    if t >= 105.0:
        return False           # LostAuto = red
    if t >= 80.0:
        return True
    if t >= 55.0:
        return False
    return True


def last_deactivation(t: float) -> float | None:
    """Match clock at which the blue hub last went inactive before t (for HubScoring's 3 s disableDelay)."""
    for edge in (130.0, 80.0):
        if t < edge and t >= edge - 25.0:
            return edge
    return None


def initial_fuel() -> np.ndarray:
    """Starting ball positions (x, z, y): the 496 balls on the field at the first recorded moment of the published
    human demo (504 in total; the robot holds the other 8 as preload). The match start is the same every time
    (within ~1.3 cm)."""
    with open(DEMO) as f:
        for line in f:
            r = json.loads(line)
            if "fuel" in r:
                return np.asarray(r["fuel"], float).reshape(-1, 3)
    raise RuntimeError(f"no fuel record in {DEMO}")
