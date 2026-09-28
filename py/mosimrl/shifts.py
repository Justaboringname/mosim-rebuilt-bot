"""REBUILT hub-activity schedule (as the game applies it), timer counting down from 160.

>=140 both (auto) · 140-130 both · 130-105 LostAuto · 105-80 WonAuto · 80-55 LostAuto · 55-30 WonAuto · <30 both.
Hub scoring keeps counting for 3 s after a hub deactivates.
"""

RED, BLUE, BOTH = 0, 1, 2
GRACE_S = 3.0

# (upper timer bound, lower bound, owner) where owner is "both", "won" or "lost"
_WINDOWS = [
    (160.0, 130.0, "both"),
    (130.0, 105.0, "lost"),
    (105.0, 80.0, "won"),
    (80.0, 55.0, "lost"),
    (55.0, 30.0, "won"),
    (30.0, -1e9, "both"),
]


def blue_active(t: float, blue_won_auto: bool) -> bool:
    for hi, lo, owner in _WINDOWS:
        if lo < t <= hi or (hi == 160.0 and t > 160.0):
            if owner == "both":
                return True
            return (owner == "won") == blue_won_auto
    return True


def seconds_until_blue_active(t: float, blue_won_auto: bool, step: float = 0.05) -> float:
    """Timer-seconds until blue's hub is next active (0 if active now)."""
    s = 0.0
    while s < 60.0:
        if blue_active(t - s, blue_won_auto):
            return s
        s += step
    return 60.0


def seconds_until_blue_inactive(t: float, blue_won_auto: bool, step: float = 0.05) -> float:
    s = 0.0
    while s < 160.0:
        if not blue_active(t - s, blue_won_auto):
            return s
        s += step
    return 160.0
