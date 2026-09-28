"""Intake capture zone: for floor balls near the robot, P(in the hopper at the next decision) by robot-frame
position (lx = right, lz = forward), with Intake held vs not; plus hopper capacity."""

from __future__ import annotations

import numpy as np

from .common import episodes


def robot_frame(px, pz, x, z, yaw_deg):
    a = np.radians(yaw_deg)
    dx, dz = px - x, pz - z
    fwd = dx * np.sin(a) + dz * np.cos(a)
    right = dx * np.cos(a) - dz * np.sin(a)
    return right, fwd


def collect(batch="cal1"):
    L, F, C, I, SP, HE = [], [], [], [], [], []
    maxheld = 0
    for d in episodes(batch, dense=True):
        tr = d["trace"]; raw = d["dense"]["pos_cm"]
        gone = raw[:, :, 0] == -32768
        P = raw.astype(float) / 100
        for k in range(len(tr) - 1):
            if k + 1 >= len(P):
                break
            a = tr[k]
            maxheld = max(maxheld, a["held"] or 0)
            p = P[k]
            fl = (~gone[k]) & (p[:, 2] < 0.2)
            lx, lz = robot_frame(p[:, 0], p[:, 1], a["x"], a["z"], a["yaw"])
            near = fl & (np.abs(lx) < 1.0) & (lz > -1.0) & (lz < 1.5)
            idx = np.nonzero(near)[0]
            cap = gone[k + 1, idx]
            L.append(lx[idx]); F.append(lz[idx]); C.append(cap)
            I.append(np.full(len(idx), a["b"][0])); SP.append(np.full(len(idx), np.hypot(a["rvx"] or 0, a["rvz"] or 0)))
            HE.append(np.full(len(idx), a["held"] or 0))
    return (np.concatenate(L), np.concatenate(F), np.concatenate(C), np.concatenate(I), np.concatenate(SP),
            np.concatenate(HE), maxheld)


if __name__ == "__main__":
    lx, lz, cap, intk, sp, held, mh = collect()
    print("max held observed", mh, " samples", len(lx), " captures", cap.sum())
    for on in (1, 0):
        m = intk == on
        print(f"\nIntake held={on}: P(capture next decision) by robot-frame cell (rows lz forward, cols lx right)")
        xb = np.arange(-0.8, 0.81, 0.2); zb = np.arange(1.4, -0.81, -0.2)
        print("lz\\lx " + " ".join(f"{x:+.1f}" for x in (xb[:-1] + 0.1)))
        for z1, z0 in zip(zb[:-1], zb[1:]):
            row = []
            for x0, x1 in zip(xb[:-1], xb[1:]):
                s = m & (lx >= x0) & (lx < x1) & (lz >= z0) & (lz < z1)
                row.append(f"{cap[s].mean():.2f}" if s.sum() > 30 else "  . ")
            print(f"{(z0 + z1) / 2:+.1f}  " + " ".join(row))
    # held dependence in the core zone
    core = (intk == 1) & (np.abs(lx) < 0.3) & (lz > 0.3) & (lz < 0.8)
    for lo, hi in [(0, 40), (40, 80), (80, 95), (95, 105), (105, 200)]:
        s = core & (held >= lo) & (held < hi)
        if s.sum() > 30:
            print(f"core zone, held {lo}-{hi}: P(capture) {cap[s].mean():.2f} (n={s.sum()})")
