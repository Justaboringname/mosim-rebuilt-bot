"""Drivetrain fit: field-frame stick command -> next velocity, rotate command -> next yaw rate (per 0.099 s decision)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .common import episodes, wrap180


def rows(batch="cal1"):
    out = []
    for d in episodes(batch):
        tr = d["trace"]
        for k in range(len(tr) - 1):
            a, b = tr[k], tr[k + 1]
            if a.get("rvx") is None or b.get("rvx") is None:
                continue
            mult = 0.55 if (a["b"][1] or a["b"][2] or a["b"][3]) else 1.0
            tilt = max(abs(wrap180(a["pitch"])), abs(wrap180(a["roll"])))
            out.append((a["vx"], a["vz"], a["rot"], a["rvx"], a["rvz"], a["wy"], b["rvx"], b["rvz"], b["wy"], mult,
                        a["held"], tilt, a["x"], a["z"], a["t"] - b["t"], a["mode"] == "backoff", a["yaw"], b["yaw"]))
    return np.array(out, float)


def fit(batch="cal1"):
    R = rows(batch)
    ux, uz, rot, vx, vz, w, vx1, vz1, w1, mult, held, tilt, x, z, dt, back, yaw, yaw1 = R.T
    ok = (tilt < 3) & (np.abs(z) < 3.6) & (np.abs(x) < 7.9) & (np.abs(dt - 0.099) < 0.01) & (back == 0) & \
         ~((np.abs(x) > 3.0) & (np.abs(x) < 4.3))

    def res(p):
        vmax, a, c = p
        aa = a / (1 + c * held)
        return np.r_[(vx + aa * (vmax * mult * ux - vx) - vx1)[ok], (vz + aa * (vmax * mult * uz - vz) - vz1)[ok]]
    r = least_squares(res, [2.7, 0.5, 0.005])
    print(f"translation: vmax {r.x[0]:.3f} m/s, alpha {r.x[1]:.3f}/decision, mass c {r.x[2]:.4f}/ball, "
          f"rmse {np.sqrt(np.mean(r.fun ** 2)):.3f} m/s  (n={ok.sum()})")
    sp = np.hypot(vx1, vz1); u = np.hypot(ux, uz)
    for m in (1.0, 0.55):
        s = ok & (mult == m) & (u > 0.95)
        print(f"  mult {m}: speed at full stick p50/p90/p99 {np.percentile(sp[s], [50, 90, 99]).round(2)}")
    dyaw = np.radians(wrap180(yaw1 - yaw)) / 0.099        # yaw rate from pose (deg->rad), unity yaw

    def resw(p):
        wm, b = p
        return (w + b * (wm * rot - w) - w1)[ok]
    rw = least_squares(resw, [-5.0, 0.5])
    print(f"yaw: wmax {rw.x[0]:.3f} rad/s per unit rot, beta {rw.x[1]:.3f}, rmse {np.sqrt(np.mean(rw.fun ** 2)):.3f};"
          f" corr(wy, dyaw/dt) {np.corrcoef(w1[ok], dyaw[ok])[0, 1]:.3f}")
    return {"vmax": r.x[0], "alpha": r.x[1], "mass_c": r.x[2], "wmax": rw.x[0], "beta": rw.x[1]}


if __name__ == "__main__":
    print(fit())


def simulate_decision(v, u, mult, held, p, steps=22, dt=0.0045):
    """MoSim DriveController force law, 22 physics steps: impulse K*|u|*falloff(|v|/5.18) along u, mass m0+mb*held,
    Unity linear drag D (dv = -D v dt). v, u: (n,2) field frame; mult: shooting multiplier on the stick."""
    K, mb, D = p
    uu = u * mult[:, None]
    n = np.linalg.norm(uu, axis=1, keepdims=True)
    uu = np.where(n > 1, uu / np.maximum(n, 1e-9), uu)
    m = 1.0 + mb * held[:, None]
    for _ in range(steps):
        s = np.linalg.norm(v, axis=1, keepdims=True)
        f = (1 - np.clip(s / 5.18, 0, 1) * 0.075) ** 10
        v = v + dt * (K * uu * f / m - D * v)
    return v


def fit_force(batch="cal1"):
    R = rows(batch)
    ux, uz, rot, vx, vz, w, vx1, vz1, w1, mult, held, tilt, x, z, dt, back, yaw, yaw1 = R.T
    ok = (tilt < 3) & (np.abs(z) < 3.6) & (np.abs(x) < 7.9) & (np.abs(dt - 0.099) < 0.01) & (back == 0) & \
         ~((np.abs(x) > 3.0) & (np.abs(x) < 4.3)) & (np.abs(rot) < 0.3)
    v0 = np.stack([vx, vz], 1)[ok]; u = np.stack([ux, uz], 1)[ok]; v1 = np.stack([vx1, vz1], 1)[ok]

    def res(p):
        return (simulate_decision(v0, u, mult[ok], held[ok], p) - v1).ravel()
    r = least_squares(res, [17.0, 0.005, 4.0], bounds=([1, 0, 0], [100, 0.1, 30]))
    print(f"force law: K {r.x[0]:.2f} m/s^2 (per unit mass), mb {r.x[1]:.4f}/ball, D {r.x[2]:.3f}/s, "
          f"rmse {np.sqrt(np.mean(r.fun ** 2)):.3f} m/s (n={ok.sum()})")
    for h in (0, 50, 100):
        v = np.zeros((1, 2))
        for _ in range(40):
            v = simulate_decision(v, np.array([[1.0, 0.0]]), np.array([1.0]), np.array([float(h)]), r.x)
        vs = np.zeros((1, 2))
        for _ in range(40):
            vs = simulate_decision(vs, np.array([[1.0, 0.0]]), np.array([0.55]), np.array([float(h)]), r.x)
        print(f"  held {h:3d}: top speed {v[0, 0]:.2f} m/s, shooting {vs[0, 0]:.2f} m/s")
    return r.x


def near_balls(batch="cal1", r=0.9):
    """Number of floor balls within r of the robot centre at every decision, aligned with rows(batch)."""
    out = []
    for d in episodes(batch, dense=True):
        tr = d["trace"]; raw = d["dense"]["pos_cm"]
        P = raw.astype(float) / 100; P[raw[:, :, 0] == -32768] = np.nan
        for k in range(len(tr) - 1):
            a, b = tr[k], tr[k + 1]
            if a.get("rvx") is None or b.get("rvx") is None:
                continue
            p = P[min(k, len(P) - 1)]
            dd = np.hypot(p[:, 0] - a["x"], p[:, 1] - a["z"])
            out.append(np.nansum((dd < r) & (p[:, 2] < 0.2)))
    return np.array(out, float)


def fit_force_clean(batch="cal1"):
    """Force law on decisions with few balls around (free dynamics), then a ball-drag term on the rest."""
    R = rows(batch); nb = near_balls(batch)
    ux, uz, rot, vx, vz, w, vx1, vz1, w1, mult, held, tilt, x, z, dt, back, yaw, yaw1 = R.T
    base = (tilt < 3) & (np.abs(z) < 3.6) & (np.abs(x) < 7.9) & (np.abs(dt - 0.099) < 0.01) & (back == 0) & \
           ~((np.abs(x) > 3.0) & (np.abs(x) < 4.3)) & (np.abs(rot) < 0.3)
    ok = base & (nb < 4)
    v0 = np.stack([vx, vz], 1); u = np.stack([ux, uz], 1); v1 = np.stack([vx1, vz1], 1)

    def res(p):
        return (simulate_decision(v0[ok], u[ok], mult[ok], held[ok] * 0, p) - v1[ok]).ravel()
    r = least_squares(res, [17.0, 0.0, 4.0], bounds=([1, 0, 0], [100, 1e-6, 30]))
    K, _, D = r.x
    print(f"free force law: K {K:.2f}, D {D:.3f}, rmse {np.sqrt(np.mean(r.fun ** 2)):.3f} (n={ok.sum()})")
    # ball drag: extra linear drag per nearby floor ball
    ok2 = base & (nb >= 4)

    def res2(q):
        return (simulate_decision_drag(v0[ok2], u[ok2], mult[ok2], nb[ok2], (K, D, q[0])) - v1[ok2]).ravel()
    r2 = least_squares(res2, [0.05], bounds=([0], [2]))
    print(f"ball drag: {r2.x[0]:.4f}/s per floor ball within 0.9 m, rmse {np.sqrt(np.mean(r2.fun ** 2)):.3f} (n={ok2.sum()})")
    for m in (1.0, 0.55):
        v = np.zeros((1, 2))
        for _ in range(40):
            v = simulate_decision_drag(v, np.array([[1.0, 0.0]]), np.array([m]), np.array([0.0]), (K, D, r2.x[0]))
        print(f"  top speed mult {m}: {v[0, 0]:.2f} m/s")
    return {"K": K, "D": D, "ball_drag": r2.x[0]}


def simulate_decision_drag(v, u, mult, nb, p, steps=22, dt=0.0045):
    K, D, bd = p
    uu = u * mult[:, None]
    n = np.linalg.norm(uu, axis=1, keepdims=True)
    uu = np.where(n > 1, uu / np.maximum(n, 1e-9), uu)
    DD = D + bd * nb[:, None]
    for _ in range(steps):
        s = np.linalg.norm(v, axis=1, keepdims=True)
        f = (1 - np.clip(s / 5.18, 0, 1) * 0.075) ** 10
        v = v + dt * (K * uu * f - DD * v)
    return v


def simulate_decision_coulomb(v, u, mult, p, steps=22, dt=0.0045):
    """Force law + linear drag D + constant (Coulomb-like) resistance C along -v (wheel scrub / carpet): dv/dt =
    K*u*falloff(|v|) - D*v - C*v/|v|. The pure linear-drag law makes top speed too concave in the stick (sim 2.30 m/s at
    full stick vs 1.45 m/s at the 0.55 shooting multiplier; real 2.46 vs 1.39)."""
    K, D, C = p
    uu = u * mult[:, None]
    n = np.linalg.norm(uu, axis=1, keepdims=True)
    uu = np.where(n > 1, uu / np.maximum(n, 1e-9), uu)
    for _ in range(steps):
        s = np.linalg.norm(v, axis=1, keepdims=True)
        f = (1 - np.clip(s / 5.18, 0, 1) * 0.075) ** 10
        dv = dt * (K * uu * f - D * v)
        fric = dt * C * v / np.maximum(s, 1e-6)
        # friction cannot reverse the velocity within a step
        fric = np.where(s > dt * C, fric, v + dv * 0)
        v = v + dv - fric
    return v


def fit_force_coulomb(batches=("r4", "hr1b", "cal1")):
    Rs = [rows(b) for b in batches]
    R = np.concatenate(Rs)
    ux, uz, rot, vx, vz, w, vx1, vz1, w1, mult, held, tilt, x, z, dt, back, yaw, yaw1 = R.T
    ok = (tilt < 3) & (np.abs(z) < 3.4) & (np.abs(x) < 7.6) & (np.abs(dt - 0.099) < 0.01) & (back == 0) & \
         ~((np.abs(x) > 3.0) & (np.abs(x) < 4.3)) & (np.abs(rot) < 0.3)
    v0 = np.stack([vx, vz], 1)[ok]; u = np.stack([ux, uz], 1)[ok]; v1 = np.stack([vx1, vz1], 1)[ok]; m = mult[ok]

    def res(p):
        e = simulate_decision_coulomb(v0, u, m, p) - v1
        return np.clip(e, -0.5, 0.5).ravel()                 # collisions are outliers
    r0 = least_squares(lambda p: res([p[0], p[1], 0.0]), [19.6, 5.9], bounds=([1, 0], [100, 30]))
    r = least_squares(res, [19.6, 5.0, 1.0], bounds=([1, 0, 0], [100, 30, 20]))
    print(f"no C : K {r0.x[0]:.2f} D {r0.x[1]:.3f}  rmse {np.sqrt(np.mean(r0.fun ** 2)):.4f}")
    print(f"with C: K {r.x[0]:.2f} D {r.x[1]:.3f} C {r.x[2]:.3f}  rmse {np.sqrt(np.mean(r.fun ** 2)):.4f} (n={ok.sum()})")
    for p, name in ((list(r0.x) + [0.0], "no C"), (r.x, "with C")):
        out = []
        for mm, uu in ((1.0, 1.0), (0.55, 1.0), (1.0, 0.44)):
            v = np.zeros((1, 2))
            for _ in range(60):
                v = simulate_decision_coulomb(v, np.array([[uu, 0.0]]), np.array([mm]), p)
            out.append(round(float(v[0, 0]), 2))
        print(f"  {name}: steady speed full / shooting / u=0.44: {out}   (real 2.46 / 1.39 / ~1.1)")
    return r.x
