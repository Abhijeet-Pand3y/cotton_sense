"""
leakage_audit.py — does information from the ball being scored leak into the
constants used to score it?

FOUR CONSTANTS GO INTO EVERY SIZE ESTIMATE
    D = 2 * r * (z/a)^(1/gamma) / f          and   Z_pupil = Z_tape + c

For each one, ask: was it fitted using the very row it is now being used to
score? If yes, the reported error is optimistic.

    f, c    fitted on ALL 116 ball-frames, assuming D = 40 mm      -> LEAKS
    gamma   median of the OTHER sets                               -> clean
    a (global)   fitted on all rows                                -> LEAKS
    a (per-frame) fitted on every ball in the frame, INCLUDING the
                  one being scored                                 -> LEAKS

The last one is the subtle and important one. The honest version of "a known
reference object is in the frame" uses a DIFFERENT object as the reference and
predicts the target. Using the target's own true distance to set the scale, then
measuring the target, is circular.

STRICT PROTOCOL, for comparison
    f, c    leave-one-SET-out
    gamma   leave-one-SET-out (already the case)
    a       leave-one-BALL-out inside the frame
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "scripts/depth")
from size_regression import load, solve_pinhole                   # noqa: E402

BALL = 40.0


def fit_fc(rows):
    r = np.array([x["r"] for x in rows]); Z = np.array([x["Ztape"] for x in rows])
    s, i = np.polyfit(Z, 1 / r, 1)
    return 2 / (s * BALL), i / s


def main():
    rows = load()
    f_all, c_all = fit_fc(rows)
    for x in rows:
        x["Zp"] = x["Ztape"] + c_all
    sets = sorted({x["set"] for x in rows})
    byframe = defaultdict(list)
    for x in rows:
        byframe[(x["set"], x["frame"])].append(x)

    # gamma per set, then leave-one-set-out
    gset = {}
    for s in sets:
        gs = []
        for k, rr in byframe.items():
            if k[0] != s:
                continue
            Z = np.array([y["Zp"] for y in rr]); z = np.array([y["z"] for y in rr])
            if np.unique(np.round(Z, 1)).size < 2:
                continue
            gs.append(np.polyfit(np.log(Z), np.log(z), 1)[0])
        if gs:
            gset[s] = float(np.median(gs))
    gcross = {s: float(np.median([v for t, v in gset.items() if t != s])) for s in gset}

    print("=" * 76)
    print("1. HOW MUCH DOES f MOVE IF THE SCORED SET IS EXCLUDED?")
    print("=" * 76)
    print(f"  f on all 116 rows: {f_all:.1f} px,  c = {c_all:+.1f} mm")
    print(f"  {'held-out set':<36}{'f (px)':>10}{'c (mm)':>9}{'f shift':>10}")
    fc_loso = {}
    for s in sets:
        tr = [x for x in rows if x["set"] != s]
        f_s, c_s = fit_fc(tr)
        fc_loso[s] = (f_s, c_s)
        print(f"  {s:<36}{f_s:>10.1f}{c_s:>+9.1f}{100*(f_s-f_all)/f_all:>9.2f}%")
    sh = [100 * (fc_loso[s][0] - f_all) / f_all for s in sets]
    print(f"\n  worst shift {max(abs(min(sh)), abs(max(sh))):.2f}% -> f is stable, so its "
          f"leakage is small but NOT zero.")

    print("\n" + "=" * 76)
    print("2. THE BIG ONE — does 'a' use the ball it is scoring?")
    print("=" * 76)

    def score(strict):
        out = []
        for k, rr in byframe.items():
            Zt = np.array([y["Zp"] for y in rr]); zt = np.array([y["z"] for y in rr])
            if np.unique(np.round(Zt, 1)).size < 2:
                continue
            g = gcross.get(k[0])
            if g is None:
                continue
            f_use, c_use = fc_loso[k[0]] if strict else (f_all, c_all)
            for j, y in enumerate(rr):
                if strict:
                    # scale from the OTHER balls in this frame only
                    m = [i for i in range(len(rr)) if i != j]
                    if not m:
                        continue
                    Zo = np.array([rr[i]["Ztape"] + c_use for i in m])
                    zo = np.array([rr[i]["z"] for i in m])
                    if np.unique(np.round(Zo, 1)).size < 1:
                        continue
                    a = float(np.exp(np.mean(np.log(zo) - g * np.log(Zo))))
                    Ztrue = y["Ztape"] + c_use
                else:
                    a = float(np.exp(np.mean(np.log(zt) - g * np.log(Zt))))
                    Ztrue = y["Zp"]
                Z = (y["z"] / a) ** (1 / g)
                D = 2 * y["r"] * Z / f_use
                out.append((D, 100 * (D - BALL) / BALL, len(rr)))
        return out

    for strict, name in ((False, "AS SHIPPED  (a uses every ball incl. target)"),
                         (True, "STRICT      (a uses only the OTHER balls, f LOSO)")):
        v = score(strict)
        D = np.array([x[0] for x in v]); e = np.abs([x[1] for x in v])
        print(f"\n  {name}")
        print(f"    n={len(D):>3}   mean {D.mean():6.2f} mm   sd {D.std(ddof=1):5.2f}   "
              f"mean |err| {e.mean():5.2f}%   range {D.min():.1f}-{D.max():.1f}")
        for nb in (2, 3, 6):
            s2 = [x for x in v if x[2] == nb]
            if s2:
                e2 = np.abs([x[1] for x in s2])
                print(f"      frames with {nb} balls: n={len(s2):>3}  "
                      f"mean |err| {e2.mean():5.2f}%")

    print("\n" + "=" * 76)
    print("3. IS THE MEAN OF 40 EVIDENCE?")
    print("=" * 76)
    print("""  No. Two separate fits both push it there:
    - f was chosen so that 2*r*Z_tape/f averages 40 on these balls;
    - a is chosen so DA3's depths match the true distances in the frame.
  D/40 = Z_est/Z_true, and both fits drive that ratio to 1 on average.
  So a mean near 40 is built in. Only the SPREAD carries information.""")


if __name__ == "__main__":
    main()
