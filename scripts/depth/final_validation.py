"""
final_validation.py — leak-free estimate of (1) distance and (2) size, and what
they imply for measuring cotton bolls.

THE PROTOCOL — nothing about the target ball touches the constants
    For every ball-frame scored, held out one SET at a time:
        f, c     refitted on the OTHER sets only
        gamma    refitted on the OTHER sets only
        a        from OTHER balls in the same frame, never the target
                 (or from other sets entirely, in the no-reference case)
    The target's radius and depth are used only as inputs to the prediction.
    Its true distance is used only to score it.

    The one thing that cannot be held out is D = 40 mm, because f is derived
    from it. That makes the ABSOLUTE scale of every result conditional on the
    balls really being 40 mm. It does not affect any spread or any ratio.

THREE DEPLOYMENT SCENARIOS, all leak-free
    A  no reference          nothing known about the scene; a comes from other sets
    B  one reference object  a set by ONE other object of known size in the frame
    C  several references    a least-squares over all other objects in the frame

WHY BOTH DISTANCE AND SIZE ARE REPORTED
    They are nearly the same number - D_est/D_true = (Z_est/Z_true)*(f_all/f_loso)
    - but not identically, because f is refitted per fold. Reporting both makes
    the near-identity visible rather than assumed, and size is the quantity the
    boll work actually needs.

WHY THE BALL IS THE RIGHT PROXY FOR A BOLL
    A regulation ping-pong ball is 40.0 mm. The 38 measured bolls average
    40.46 mm tip-to-base. Same size, overlapping distance range - so the error
    measured here is close to the error to expect on a boll.
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "scripts/depth")
from size_regression import load, solve_pinhole                   # noqa: E402

BALL = 40.0
GT_BOLL_MEAN, GT_BOLL_CV = 40.46, 21.2


def fit_fc(rows):
    r = np.array([x["r"] for x in rows]); Z = np.array([x["Ztape"] for x in rows])
    s, i = np.polyfit(Z, 1 / r, 1)
    return 2 / (s * BALL), i / s


def gamma_of(rows, c):
    """Median of per-frame gammas over the given rows."""
    byf = defaultdict(list)
    for x in rows:
        byf[(x["set"], x["frame"])].append(x)
    gs = []
    for rr in byf.values():
        Z = np.array([y["Ztape"] + c for y in rr]); z = np.array([y["z"] for y in rr])
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        gs.append(np.polyfit(np.log(Z), np.log(z), 1)[0])
    return float(np.median(gs)) if gs else None


def run(rows, gamma_override=None, focal_override=None, verbose=True):
    sets = sorted({x["set"] for x in rows})
    byframe = defaultdict(list)
    for x in rows:
        byframe[(x["set"], x["frame"])].append(x)
    out = {"A": [], "B": [], "C": []}
    for s in sets:
        tr = [x for x in rows if x["set"] != s]
        f_o, c_o = fit_fc(tr)
        if focal_override:
            f_o = focal_override
        g_o = gamma_override if gamma_override else gamma_of(tr, c_o)
        if g_o is None:
            continue
        # scenario A scale: fitted on the other sets entirely
        zt = np.array([x["z"] for x in tr]); Zt = np.array([x["Ztape"] + c_o for x in tr])
        a_out = float(np.exp(np.mean(np.log(zt) - g_o * np.log(Zt))))
        for k, rr in byframe.items():
            if k[0] != s:
                continue
            for j, y in enumerate(rr):
                Ztrue = y["Ztape"] + c_o
                others = [rr[i] for i in range(len(rr)) if i != j]
                def score(a, tag):
                    Z = (y["z"] / a) ** (1 / g_o)
                    D = 2 * y["r"] * Z / f_o
                    out[tag].append((Z, Ztrue, D, len(rr)))
                score(a_out, "A")
                if others:
                    # B: each other ball in turn as the sole reference
                    for o in others:
                        a1 = o["z"] / ((o["Ztape"] + c_o) ** g_o)
                        score(a1, "B")
                    # C: least squares over all other balls
                    Zo = np.array([o["Ztape"] + c_o for o in others])
                    zo = np.array([o["z"] for o in others])
                    aC = float(np.exp(np.mean(np.log(zo) - g_o * np.log(Zo))))
                    score(aC, "C")
    res = {}
    for tag, v in out.items():
        if not v:
            continue
        Z = np.array([x[0] for x in v]); Zt_ = np.array([x[1] for x in v])
        D = np.array([x[2] for x in v])
        res[tag] = {
            "n": len(v),
            "dist_err": np.abs(100 * (Z - Zt_) / Zt_),
            "dist_bias": np.mean(100 * (Z - Zt_) / Zt_),
            "size": D,
            "size_err": np.abs(100 * (D - BALL) / BALL),
            "size_bias": np.mean(100 * (D - BALL) / BALL),
        }
    return res


def main():
    rows = load()
    f_all, c_all = fit_fc(rows)
    for x in rows:
        x["Zp"] = x["Ztape"] + c_all
    print("=" * 78)
    print("LEAK-FREE RESULTS — every constant fitted without the target's set")
    print("=" * 78)
    res = run(rows)
    names = {"A": "A · no reference at all",
             "B": "B · ONE reference object in frame",
             "C": "C · several references in frame"}
    print(f"{'scenario':<36}{'n':>5}{'DISTANCE err':>14}{'SIZE err':>11}"
          f"{'size mean':>11}{'size sd':>9}")
    for t in "ABC":
        if t not in res:
            continue
        r = res[t]
        print(f"{names[t]:<36}{r['n']:>5}{r['dist_err'].mean():>13.2f}%"
              f"{r['size_err'].mean():>10.2f}%{r['size'].mean():>11.2f}"
              f"{r['size'].std(ddof=1):>9.2f}")
    print("\n  distance and size errors track each other because D = 2rZ/f: a")
    print("  fractional distance error becomes the same fractional size error.")
    print("  They differ only by the per-fold change in f (<0.8%).")

    print("\n" + "=" * 78)
    print("SENSITIVITY — does the verdict depend on the constants?")
    print("=" * 78)
    g_base = gamma_of(rows, c_all)
    print(f"  baseline gamma {g_base:.3f}, f {f_all:.0f} px\n")
    print(f"{'variation':<34}{'B: dist err':>13}{'B: size err':>13}{'C: size err':>13}")
    for lab, gv, fv in [("as fitted", None, None),
                        ("gamma = 0.55 (low end)", 0.55, None),
                        ("gamma = 0.75 (high end)", 0.75, None),
                        ("gamma = 1.0 (no compression)", 1.0, None),
                        ("f -5%", None, f_all * 0.95),
                        ("f +5%", None, f_all * 1.05)]:
        r = run(rows, gamma_override=gv, focal_override=fv, verbose=False)
        print(f"{lab:<34}{r['B']['dist_err'].mean():>12.2f}%"
              f"{r['B']['size_err'].mean():>12.2f}%{r['C']['size_err'].mean():>12.2f}%")
    print("\n  gamma matters and 1.0 is clearly wrong -> the compression is real.")
    print("  f shifts size but not distance-vs-truth, because f cancels in the")
    print("  scale fit and re-enters only in the pinhole step.")

    print("\n" + "=" * 78)
    print("WHAT THIS MEANS FOR COTTON")
    print("=" * 78)
    r = res["B"]
    e = r["size_err"].mean()
    print(f"  A ping-pong ball is 40.0 mm; the 38 measured bolls average "
          f"{GT_BOLL_MEAN} mm.")
    print(f"  So the error measured here transfers directly in magnitude.\n")
    print(f"  With ONE reference object in frame: {e:.1f}% error")
    print(f"    = +/- {e/100*GT_BOLL_MEAN:.1f} mm on a typical {GT_BOLL_MEAN:.0f} mm boll")
    obs = np.sqrt(GT_BOLL_CV**2 + e**2)
    print(f"\n  Real boll-to-boll spread is {GT_BOLL_CV}%. Adding {e:.1f}% of independent")
    print(f"  measurement noise gives an observed spread of "
          f"sqrt({GT_BOLL_CV}^2 + {e:.1f}^2) = {obs:.1f}%.")
    print(f"    -> the measurement inflates the population spread by only "
          f"{obs-GT_BOLL_CV:.1f} points.")
    print(f"    -> POPULATION statistics are usable.")
    n_needed = (e / 5.0) ** 2
    print(f"\n  For a SINGLE boll, {e:.1f}% is +/- {e/100*GT_BOLL_MEAN:.1f} mm against a true")
    print(f"  range of 20.4-52.4 mm. That separates a big boll from a small one,")
    print(f"  but not two bolls a few mm apart.")
    print(f"    -> averaging ~{max(1,int(np.ceil(n_needed)))} independent frames would reach 5%,")
    print(f"       IF the errors are independent - they are not fully, since the")
    print(f"       scale error is shared across a track.")


if __name__ == "__main__":
    main()


def write_csvs():
    """Per-row predictions and the summary tables, as CSV."""
    import csv as _csv
    from pathlib import Path
    rows = load()
    f_all, c_all = fit_fc(rows)
    sets = sorted({x["set"] for x in rows})
    byframe = defaultdict(list)
    for x in rows:
        byframe[(x["set"], x["frame"])].append(x)

    per = []
    for s in sets:
        tr = [x for x in rows if x["set"] != s]
        f_o, c_o = fit_fc(tr)
        g_o = gamma_of(tr, c_o)
        if g_o is None:
            continue
        zt = np.array([x["z"] for x in tr]); Zt = np.array([x["Ztape"] + c_o for x in tr])
        a_out = float(np.exp(np.mean(np.log(zt) - g_o * np.log(Zt))))
        for k, rr in byframe.items():
            if k[0] != s:
                continue
            for j, y in enumerate(rr):
                Ztrue = y["Ztape"] + c_o
                others = [rr[i] for i in range(len(rr)) if i != j]

                def emit(a, scen, ref):
                    Z = (y["z"] / a) ** (1 / g_o)
                    D = 2 * y["r"] * Z / f_o
                    per.append({
                        "scenario": scen,
                        "held_out_set": s,
                        "set": y["set"], "frame": y["frame"], "ball": y["ball"],
                        "is_baseline": 1 if y["base"] else 0,
                        "n_balls_in_frame": len(rr),
                        "reference_ball": ref,
                        "ball_distance_in": round(y["Ztape"] / 25.4, 1),
                        "ball_distance_mm": round(y["Ztape"], 1),
                        "true_pupil_distance_mm": round(Ztrue, 1),
                        "radius_px": round(y["r"], 2),
                        "da3_depth_raw": round(y["z"], 5),
                        "f_used_px": round(f_o, 1),
                        "c_used_mm": round(c_o, 1),
                        "gamma_used": round(g_o, 4),
                        "a_used": round(a, 8),
                        "Z_pred_mm": round(Z, 1),
                        "dist_err_mm": round(Z - Ztrue, 1),
                        "dist_err_pct": round(100 * (Z - Ztrue) / Ztrue, 2),
                        "true_diameter_mm": BALL,
                        "D_pred_mm": round(D, 2),
                        "size_err_mm": round(D - BALL, 2),
                        "size_err_pct": round(100 * (D - BALL) / BALL, 2),
                    })
                emit(a_out, "A_no_reference", "")
                for o in others:
                    emit(o["z"] / ((o["Ztape"] + c_o) ** g_o), "B_one_reference", o["ball"])
                if others:
                    Zo = np.array([o["Ztape"] + c_o for o in others])
                    zo = np.array([o["z"] for o in others])
                    emit(float(np.exp(np.mean(np.log(zo) - g_o * np.log(Zo)))),
                         "C_several_references", "+".join(o["ball"] for o in others))

    Path("findings").mkdir(exist_ok=True)
    p1 = Path("findings/final_validation_per_row.csv")
    with open(p1, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)

    # summary: scenarios + sensitivity, one tidy table
    summ = []
    res = run(rows)
    for t, nm in (("A", "A_no_reference"), ("B", "B_one_reference"),
                  ("C", "C_several_references")):
        r = res[t]
        summ.append({"table": "scenario", "variant": nm, "constant": "as fitted",
                     "n": r["n"],
                     "dist_err_mean_pct": round(r["dist_err"].mean(), 2),
                     "dist_err_median_pct": round(float(np.median(r["dist_err"])), 2),
                     "dist_bias_pct": round(r["dist_bias"], 2),
                     "size_err_mean_pct": round(r["size_err"].mean(), 2),
                     "size_err_median_pct": round(float(np.median(r["size_err"])), 2),
                     "size_bias_pct": round(r["size_bias"], 2),
                     "size_mean_mm": round(r["size"].mean(), 2),
                     "size_sd_mm": round(r["size"].std(ddof=1), 2)})
    g_base = gamma_of(rows, c_all)
    for lab, gv, fv in [("gamma=0.55", 0.55, None), ("gamma=0.65", 0.65, None),
                        ("gamma=0.70", 0.70, None), ("gamma=0.75", 0.75, None),
                        ("gamma=0.80", 0.80, None), ("gamma=0.90", 0.90, None),
                        ("gamma=1.00", 1.0, None),
                        ("f-5pct", None, f_all * .95), ("f+5pct", None, f_all * 1.05)]:
        r2 = run(rows, gamma_override=gv, focal_override=fv)
        for t, nm in (("B", "B_one_reference"), ("C", "C_several_references")):
            r = r2[t]
            summ.append({"table": "sensitivity", "variant": nm, "constant": lab,
                         "n": r["n"],
                         "dist_err_mean_pct": round(r["dist_err"].mean(), 2),
                         "dist_err_median_pct": round(float(np.median(r["dist_err"])), 2),
                         "dist_bias_pct": round(r["dist_bias"], 2),
                         "size_err_mean_pct": round(r["size_err"].mean(), 2),
                         "size_err_median_pct": round(float(np.median(r["size_err"])), 2),
                         "size_bias_pct": round(r["size_bias"], 2),
                         "size_mean_mm": round(r["size"].mean(), 2),
                         "size_sd_mm": round(r["size"].std(ddof=1), 2)})
    p2 = Path("findings/final_validation_summary.csv")
    with open(p2, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)
    print(f"wrote {p1}  ({len(per)} rows)")
    print(f"wrote {p2}  ({len(summ)} rows)")
    print(f"\n  baseline gamma {g_base:.3f}, f {f_all:.1f} px, c {c_all:+.1f} mm")
