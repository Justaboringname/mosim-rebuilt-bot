"""Whole-window coverage re-planning (CR) for the dead windows (docs/research-log/proposals.md, 2026-09-27).

The human's dead-window route is a coverage pattern (adjacent strips) laid over the human's own ball layout; in a bot
match the layout has diverged, so the same strips leave ~25% of the neutral balls instead of ~15%. Local shifts make
it worse (they open gaps). CR re-plans the whole neutral-zone segment of the window against the balls actually there:
beam search over 0.5 s heading moves, scored with the static-field intake estimator E (4414 footprint, capture table
by lateral offset, pushed balls counted as lost, held-dependent capture, pass drain). The plan is spliced into the
ghost (same entry/exit points and times, the human's buttons) so the proven tracker drives it.
"""

from __future__ import annotations

import copy
import math

import numpy as np

PX = np.array([0.0, 0.2, 0.3, 0.35, 0.40, 0.45, 0.555])
PY = np.array([0.98, 0.98, 0.95, 0.80, 0.60, 0.40, 0.15])
STOCK_X = 4.25
XLIM, ZLIM = 2.9, 3.45
DA = np.radians([-75, -50, -25, 0, 25, 50, 75])
DEAD = ((130.0, 105.0), (80.0, 55.0))
DRAIN = 10.9          # v1: constant launches/s while held > 0 (lbsdiag traces)
# v2 (crdiag + lbsdiag, dead windows, AutoPass pressed, held >= 15): the turret cannot hold a pass while the robot turns
W_TAB = np.array([10.0, 40.0, 90.0, 150.0]); R_TAB = np.array([19.4, 16.3, 11.5, 5.7])


def pass_rate(omega_deg):
    return np.interp(omega_deg, W_TAB, R_TAB)
VMAX = 2.7
V_SWEEP = 1.40        # achieved speed at full stick while plowing through balls (human, dead windows)
STICK = 1.96         # human stick / (speed / VMAX) while sweeping
K_ROT = 3.5           # rad/s per unit rot command


def hfac(h):
    return np.where(h < 80, 1.0, np.where(h < 95, 0.9, np.where(h < 104, 0.63, 0.3)))


def field(fuel) -> np.ndarray:
    a = np.asarray(fuel, float).reshape(-1, 3)
    return a[(a[:, 2] < 0.25) & (a[:, 0] <= STOCK_X)][:, :2]


def estimate(balls, px, pz, pyaw, on, held0, drain=DRAIN, dt=0.1, pass_on=None) -> float:
    """E: expected captures of non-stock floor balls along a path (yaw deg; heading (sin, cos) in (x, z)).
    pass_on (per sample, AutoPass pressed) switches to the v2 drain: pass_rate(|yaw rate|) while pressed, else 0."""
    alive = np.ones(len(balls))
    if pass_on is not None:
        om = np.abs(np.diff(np.degrees(np.unwrap(np.radians(pyaw))), prepend=pyaw[0])) / dt
        om = np.convolve(om, np.ones(3) / 3, mode="same")
        drains = pass_rate(om) * np.asarray(pass_on, float)
    held, cap = float(held0), 0.0
    bx, bz = balls[:, 0], balls[:, 1]
    for k in range(len(px)):
        y = math.radians(pyaw[k]); hx, hz = math.sin(y), math.cos(y)
        rx, rz = bx - px[k], bz - pz[k]
        fwd = rx * hx + rz * hz; lat = np.abs(rx * hz - rz * hx)
        ins = (fwd >= -0.45) & (fwd <= 0.62) & (lat <= 0.555) & (alive > 0)
        if ins.any():
            pc = np.where(fwd[ins] >= 0.35, np.interp(lat[ins], PX, PY), 0.0) * float(hfac(held)) * float(on[k])
            c = float((alive[ins] * pc).sum())
            cap += c; held += c
            alive[ins] = 0.0
        held = max(0.0, held - (drains[k] if pass_on is not None else drain) * dt)
    return cap


def beam(balls, start, exitp, D, v, held0, drain=DRAIN, W=1024, dt=0.1, sub=5, pass_on=None):
    """Best path (K, 3: x, z, yaw deg) at dt spacing from start (x, z, heading deg) that ends within reach of exitp
    after D s at speed v, maximising E. Returns (path, E)."""
    x0, z0, yaw0 = start
    N = len(balls); bx, bz = balls[:, 0], balls[:, 1]
    X = np.array([x0]); Z = np.array([z0]); H = np.array([math.radians(yaw0)])
    A = np.ones((1, N)); HL = np.array([float(held0)]); S = np.zeros(1)
    paths: list[list] = [[]]
    nsteps = int(D / (dt * sub))
    for st in range(nsteps):
        rem = D - (st + 1) * dt * sub
        nb, na = len(X), len(DA)
        x = np.repeat(X, na); z = np.repeat(Z, na); h0 = np.repeat(H, na); dA = np.tile(DA, nb)
        a = np.repeat(A, na, 0); hl = np.repeat(HL, na); s = np.repeat(S, na); par = np.repeat(np.arange(nb), na)
        ok = np.ones(len(x), bool); pts = []
        for k in range(sub):
            h = h0 + dA * (k + 1) / sub
            xn = x + v * dt * np.sin(h); zn = z + v * dt * np.cos(h)
            ok &= ((np.abs(xn) <= XLIM) | (np.abs(xn) < np.abs(x))) & ((np.abs(zn) <= ZLIM) | (np.abs(zn) < np.abs(z)))
            x, z = xn, zn
            pts.append((x.copy(), z.copy(), np.degrees(h)))
            hx, hz = np.sin(h)[:, None], np.cos(h)[:, None]
            rx, rz = bx[None] - x[:, None], bz[None] - z[:, None]
            fwd = rx * hx + rz * hz; lat = np.abs(rx * hz - rz * hx)
            ins = (fwd >= -0.45) & (fwd <= 0.62) & (lat <= 0.555) & (a > 0)
            pc = np.where(fwd >= 0.35, np.interp(lat, PX, PY), 0.0) * hfac(hl)[:, None]
            c = (a * pc * ins).sum(1)
            if pass_on is not None:
                kk = min(st * sub + k, len(pass_on) - 1)
                dr = pass_rate(np.degrees(np.abs(dA)) / (sub * dt)) * float(pass_on[kk])
            else:
                dr = drain
            s = s + c; hl = np.maximum(0, hl + c - dr * dt)
            a = np.where(ins, 0.0, a)
        h = h0 + dA
        ok &= np.hypot(exitp[0] - x, exitp[1] - z) <= v * rem + 0.35
        if not ok.any():
            break
        key = np.round(x / 0.3) * 1000 + np.round(z / 0.3) * 10 + np.round(np.degrees(h) % 360 / 45)
        order = np.argsort(-(s + np.where(ok, 0, -1e9)))
        seen, keep = set(), []
        for i in order:
            if not ok[i] or key[i] in seen:
                continue
            seen.add(key[i]); keep.append(i)
            if len(keep) >= W:
                break
        keep = np.array(keep)
        paths = [paths[par[i]] + [(p[0][i], p[1][i], p[2][i]) for p in pts] for i in keep]
        X, Z, H, A, HL, S = x[keep], z[keep], h[keep], a[keep], hl[keep], s[keep]
    b = int(np.argmax(S))
    return np.array(paths[b], float), float(S[b])


def segment(ghost, t0: float, t1: float) -> tuple[float, float] | None:
    """The ghost's neutral-interior stretch inside a dead window: first / last clock with |x| <= XLIM."""
    inn = [t for t in np.arange(t0, t1, -0.1) if abs(ghost.at(t).x) <= XLIM and abs(ghost.at(t).z) <= 3.6]
    return (float(inn[0]), float(inn[-1])) if inn else None


def ghost_path(ghost, ta: float, tb: float, dt=0.1):
    fr = [ghost.at(t) for t in np.arange(ta, tb, -dt)]
    return (np.array([f.x for f in fr]), np.array([f.z for f in fr]), np.array([f.yaw for f in fr]),
            np.array([f.buttons[0] for f in fr], float), np.array([f.buttons[2] for f in fr], float))


def plan(ghost, fuel, held: int, t_now: float, tb: float, W: int = 1024, v2: bool = True) -> dict:
    """Plan the rest of the segment [t_now, tb] on the current field; also E of the ghost's own segment."""
    F = field(fuel)
    gx, gz, gyaw, gon, gpass = ghost_path(ghost, t_now, tb)
    po = gpass if v2 else None
    D = t_now - tb
    L = float(np.hypot(np.diff(gx), np.diff(gz)).sum())
    v = min(V_SWEEP, L / max(D, 1e-3))                  # the human sweeps piles at full stick, ~1.35 m/s
    e_ghost = estimate(F, gx, gz, gyaw, gon, held, pass_on=po)
    h0 = math.degrees(math.atan2(gx[min(3, len(gx) - 1)] - gx[0], gz[min(3, len(gz) - 1)] - gz[0]))
    path, _ = beam(F, (gx[0], gz[0], h0), (gx[-1], gz[-1]), D, v, held, W=W, pass_on=po)
    e_plan = (estimate(F, path[:, 0], path[:, 1], path[:, 2], np.ones(len(path)), held,
                       pass_on=None if po is None else po[:len(path)]) if len(path) else -1.0)
    return {"path": path, "t_now": t_now, "tb": tb, "v": v, "e_plan": e_plan, "e_ghost": e_ghost, "n": len(F)}


def splice(ghost, pl: dict, blend_in: float = 0.6, blend_out: float = 1.0):
    """A copy of the ghost whose rows in (tb, t_now) follow the plan: position blended in over blend_in s and back
    onto the ghost over the last blend_out s; heading = plan heading (as an offset from the ghost's, faded in/out the
    same way, so the unwrapped yaw stays continuous); velocity and stick feed-forward re-derived. Buttons unchanged."""
    g2 = copy.copy(ghost)
    for k in ("x", "z", "yaw_unwrapped", "vx", "vz", "cmd"):
        setattr(g2, k, getattr(ghost, k).copy())
    path, t_now, tb = pl["path"], pl["t_now"], pl["tb"]
    if not len(path):
        return ghost
    pt = t_now - 0.1 * (np.arange(len(path)) + 1)          # clock of each plan sample (descending)
    m = (g2.t < t_now) & (g2.t > tb)
    tr = g2.t[m]
    px = np.interp(-tr, -pt, path[:, 0]); pz = np.interp(-tr, -pt, path[:, 1])
    pyu = np.degrees(np.unwrap(np.radians(path[:, 2])))
    py = np.interp(-tr, -pt, pyu)
    w_in, w_out = np.clip((t_now - tr) / blend_in, 0, 1), np.clip((tr - tb) / blend_out, 0, 1)
    w = w_in * w_out
    gxr, gzr, gyr = ghost.x[m], ghost.z[m], ghost.yaw_unwrapped[m]
    dy = np.degrees(np.unwrap(np.radians(((py - gyr) + 180) % 360 - 180)))
    # the plan may have turned whole circles relative to the ghost: fade out to the nearest multiple of 360 (not to 0,
    # which would unwind them in one spin) and shift the rest of the ghost's unwrapped yaw by that multiple
    j = np.where(w_out < 1.0)[0]
    k360 = 360.0 * round(float(dy[j[0]]) / 360.0) if len(j) else 0.0
    g2.x[m] = gxr + w * (px - gxr)
    g2.z[m] = gzr + w * (pz - gzr)
    g2.yaw_unwrapped[m] = gyr + w_in * ((1.0 - w_out) * k360 + w_out * dy)
    g2.yaw_unwrapped[g2.t <= tb] += k360
    idx = np.where(m)[0]
    lo, hi = max(0, idx[0] - 1), min(len(g2.t), idx[-1] + 2)
    tt = -g2.t[lo:hi]                                        # ascending real time
    vx = np.gradient(g2.x[lo:hi], tt); vz = np.gradient(g2.z[lo:hi], tt)
    wy = np.gradient(np.radians(g2.yaw_unwrapped[lo:hi]), tt)
    sel = m[lo:hi]
    g2.vx[lo:hi][sel] = vx[sel]; g2.vz[lo:hi][sel] = vz[sel]
    sp = np.hypot(vx, vz); gain = np.minimum(STICK / VMAX, 1.0 / np.maximum(sp, 1e-6))   # |stick| <= 1
    g2.cmd[lo:hi][sel, 0] = (vx * gain)[sel]
    g2.cmd[lo:hi][sel, 1] = (vz * gain)[sel]
    g2.cmd[lo:hi][sel, 2] = np.clip(-wy[sel] / K_ROT, -1, 1)
    g2.name = ghost.name + "+cr"
    return g2
