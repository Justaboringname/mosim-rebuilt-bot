"""REBUILT field constants in the MoSim world frame (docs/policy-facts/field.json).

Frame: x = field length, +x = BLUE end; z = field width; yaw 0 faces +z, yaw 90 faces +x
(Unity, left-handed; positive yaw is clockwise seen from above). 4414 plays BLUE.
"""

import math

X_WALL = 8.27          # blue end wall (inner face 8.2746)
X_WALL_RED = -8.27
Z_WALL = 4.03          # side walls at ±4.03 (inner)

HUB_NODE = (3.655, -0.006)          # blue hub aim node (x, z)
HUB_BOX = (3.05, 4.25, -0.62, 0.62)  # blue hub solid footprint x0, x1, z0, z1
SCORER_BOX_X = 3.19                  # scoring trigger sits at the neutral-zone face

ZONE_LINE = 4.25        # alliance-zone line; InAllianceZone: x + maxWidth(~0.51) > 4.25
NEUTRAL_FACE = 3.05     # blue hub / bump band starts here on the neutral side
BUMP_STRIP = (3.5, 3.8)  # Hightide refuses to feed while 3.5 < x <= 3.8
TRENCH_Z = 2.75         # trench channel |z| 2.75..4.03; bar at x 3.574..3.724, underside y 0.564
FEED_X_MIN = 3.9        # safe x for AutoShoot feeding anywhere in z

# Crossing lanes between the blue zone and the neutral zone (z of lane centre, kind).
BUMP_LANES = (-1.52, 1.52)          # bumps span |z| 0.60..2.45
TRENCH_LANES = (-3.39, 3.39)        # trench channels span |z| 2.75..4.03 — 4414 fits under the bar only with a (near-)empty
                                    # hopper (user, 2026-09-26; the human goes OUT through the trench empty, IN over the bump)
# Bump crossing needs full throttle and a run-up (0.6 stalls on the ramp at x~3.75; 1.0 crosses in ~1.5 s).
CROSS_X_NEUTRAL = 1.90              # run-up start on the neutral side (ramp begins at x 3.07)
CROSS_X_ZONE = 5.60                 # run-up start on the alliance side (ramp begins at x 4.22)

TOWER_KEEPOUT = (7.05, 8.30, -0.45, 1.00)   # blue tower (uprights + base), inflated a little
DEPOT_BOX = (7.58, 8.27, -2.47, -1.40)      # 3 cm fence; balls inside
DEPOT_CENTRE = (7.96, -1.93)
CHUTE_DOOR = (8.173, 3.743)                 # outpost chute opens when robot centre within 1.0 m

ROBOT_HALF = 0.5        # bumper half-length (1.0 x 0.8 box)


def in_blue_zone(x: float) -> bool:
    return x > ZONE_LINE - 0.51


def in_neutral(x: float) -> bool:
    return x < NEUTRAL_FACE


def feedable(x: float, z: float) -> bool:
    """Where Hightide's AutoShoot will actually feed (bump strip / trench band / alliance zone gates)."""
    if not in_blue_zone(x):
        return False
    if BUMP_STRIP[0] < x <= BUMP_STRIP[1]:
        return False
    if abs(z) > 2.7 and 3.418 < x < 3.872:
        return False
    return x > 3.8


def hub_distance(x: float, z: float) -> float:
    return math.hypot(x - HUB_NODE[0], z - HUB_NODE[1])


def in_box(x: float, z: float, box, margin: float = 0.0) -> bool:
    x0, x1, z0, z1 = box
    return x0 - margin <= x <= x1 + margin and z0 - margin <= z <= z1 + margin
