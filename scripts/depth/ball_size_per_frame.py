"""
ball_size_per_frame.py — measured ping-pong ball DIAMETER in every frame,
against the known 40 mm.

    D = 2 * r_px * Z / f          Z from DA3 depth through the fitted mapping

Three distance sources are reported side by side so the error can be attributed:

  D_from_tape       Z = the tape reading. This is a CLOSURE CHECK only: f and
                    the tape offset were solved from these same balls, so it
                    must land on 40. If it does not, the arithmetic is wrong.
  D_perframe        Z from DA3, with ONLY the scale constant fitted inside that
                    frame; gamma is held fixed at the value from the OTHER sets.
                    That distinction matters. Fitting gamma AND the scale per
                    frame spends two parameters, so a two-ball frame is fitted
                    exactly and D comes back equal to the tape answer - forced,
                    not measured. Holding gamma fixed spends one, leaving a real
                    residual even with two balls. Still blank for single-ball
                    frames, where the one parameter absorbs everything.
  D_globalscale     Z from DA3 with ONE scale constant for every set. This is
                    what you get with no per-scene calibration, and it is the
                    honest worst case.
  D_strict          Same as D_perframe but with every trace of this ball removed
                    from the constants used to score it: the scale comes from the
                    OTHER balls in the frame only, and f/c are refitted with this
                    whole SET held out. This is the leak-free number and it is
                    the one to quote. D_perframe runs 1.8 points optimistic
                    against it, because letting the target ball help set the
                    scale and then measuring that same ball is circular.

Distance is from the entrance pupil: Z_pupil = tape + c, with c solved jointly
with f. Leaving c out makes the ball read 37 mm instead of 40.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_regression import load, solve_pinhole            # noqa: E402

BALL_MM = 40.0


def main():
    rows = load()
    f_px, c_mm = solve_pinhole(rows)
    for x in rows:
        x["Zp"] = x["Ztape"] + c_mm

    byframe = defaultdict(list)
    for x in rows:
        byframe[(x["set"], x["frame"])].append(x)
    # gamma per SET first (median of its per-frame fits), so each set can then be
    # given a gamma that never saw it.
    perframe_g = {}
    for k, rr in byframe.items():
        Z = np.array([y["Zp"] for y in rr]); z = np.array([y["z"] for y in rr])
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        g, _ = np.polyfit(np.log(Z), np.log(z), 1)
        perframe_g[k] = float(g)
    gset = {}
    for s in {k[0] for k in perframe_g}:
        gset[s] = float(np.median([perframe_g[k] for k in perframe_g if k[0] == s]))
    # cross-set gamma: for each set, the median gamma of every OTHER set
    gcross = {s: float(np.median([v for t, v in gset.items() if t != s]))
              for s in gset}
    # scale a fitted per frame at that fixed gamma -> one parameter, real residual
    fit = {}
    for k, rr in byframe.items():
        Z = np.array([y["Zp"] for y in rr]); z = np.array([y["z"] for y in rr])
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        g = gcross.get(k[0], float(np.median(list(gset.values()))))
        a = float(np.exp(np.mean(np.log(z) - g * np.log(Z))))
        fit[k] = (g, a)
    g_glob = float(np.median(list(gset.values())))
    zz = np.array([x["z"] for x in rows]); ZZ = np.array([x["Zp"] for x in rows])
    a_glob = float(np.exp(np.mean(np.log(zz) - g_glob * np.log(ZZ))))
    # f and c refitted with each set held out, for the strict column
    fc_loso = {}
    for s_ in {x["set"] for x in rows}:
        tr = [y for y in rows if y["set"] != s_]
        rr = np.array([y["r"] for y in tr]); ZZ = np.array([y["Ztape"] for y in tr])
        sl, ic = np.polyfit(ZZ, 1 / rr, 1)
        fc_loso[s_] = (2 / (sl * BALL_MM), ic / sl)

    print(f"f = {f_px:.1f} px, tape offset c = {c_mm:+.1f} mm")
    print(f"global gamma = {g_glob:.3f}, global scale a = {a_glob:.5g}")

    out = []
    for x in sorted(rows, key=lambda v: (v["set"], v["ball"], v["frame"])):
        k = (x["set"], x["frame"])
        d_tape = 2 * x["r"] * x["Zp"] / f_px
        rec = {
            "set": x["set"], "frame": x["frame"],
            "is_baseline": 1 if x["base"] else 0,
            "ball": x["ball"],
            "ball_distance_in": round(x["Ztape"] / 25.4, 1),
            "ball_distance_mm": round(x["Ztape"], 1),
            "ball_pupil_distance_mm": round(x["Zp"], 1),
            # The distractor object moved through the scene. Blank on baseline
            # frames, where nothing was added - that is the clean reference.
            "object_distance_in": (round(float(x["object_distance_in"]), 1)
                                   if x.get("object_distance_in") else ""),
            "object_distance_mm": (round(float(x["object_distance_in"]) * 25.4, 1)
                                   if x.get("object_distance_in") else ""),
            # GX010118 only: a 7th ball held by hand, so its distance is
            # approximate and it is not one of the fixed measured balls.
            "handheld_ball_distance_in": (x["handheld_ball_distance_in"]
                                          if x.get("handheld_ball_distance_in") else ""),
            "handheld_ball_distance_mm": (
                round(float(x["handheld_ball_distance_in"]) * 25.4, 1)
                if x.get("handheld_ball_distance_in", "").replace(".", "", 1).isdigit()
                else ("unknown" if x.get("handheld_ball_distance_in") else "")),
            # Signed gap: how much nearer (-) or farther (+) the object sat
            # than the ball being measured. This is the quantity that matters
            # for whether the object perturbs this ball's depth.
            "object_minus_ball_mm": (
                round(float(x["object_distance_in"]) * 25.4 - x["Ztape"], 1)
                if x.get("object_distance_in") else ""),
            "radius_px": round(x["r"], 2),
            "diameter_px": round(2 * x["r"], 2),
            "da3_depth_raw": round(x["z"], 5),
            "true_diameter_mm": BALL_MM,
            "D_from_tape_mm": round(d_tape, 2),
            "err_from_tape_pct": round(100 * (d_tape - BALL_MM) / BALL_MM, 2),
        }
        if k in fit:
            g, a = fit[k]
            Z = (x["z"] / a) ** (1 / g)
            D = 2 * x["r"] * Z / f_px
            rec.update(n_balls_in_frame=len(byframe[k]),
                       gamma_used=round(g, 3),
                       Z_perframe_mm=round(Z, 1),
                       D_perframe_mm=round(D, 2),
                       err_perframe_pct=round(100 * (D - BALL_MM) / BALL_MM, 2))
        else:
            rec.update(n_balls_in_frame=len(byframe[k]), gamma_used="",
                       Z_perframe_mm="", D_perframe_mm="", err_perframe_pct="")
        # strict: scale from the OTHER balls in this frame, f/c without this set
        if k in fit and len(byframe[k]) > 1:
            g_s = fit[k][0]
            f_s, c_s = fc_loso[x["set"]]
            others = [y for y in byframe[k] if y is not x]
            Zo = np.array([y["Ztape"] + c_s for y in others])
            zo = np.array([y["z"] for y in others])
            a_s = float(np.exp(np.mean(np.log(zo) - g_s * np.log(Zo))))
            Zst = (x["z"] / a_s) ** (1 / g_s)
            Dst = 2 * x["r"] * Zst / f_s
            rec.update(Z_strict_mm=round(Zst, 1), D_strict_mm=round(Dst, 2),
                       err_strict_pct=round(100 * (Dst - BALL_MM) / BALL_MM, 2))
        else:
            rec.update(Z_strict_mm="", D_strict_mm="", err_strict_pct="")

        Zg = (x["z"] / a_glob) ** (1 / g_glob)
        Dg = 2 * x["r"] * Zg / f_px
        rec.update(Z_globalscale_mm=round(Zg, 1),
                   D_globalscale_mm=round(Dg, 2),
                   err_globalscale_pct=round(100 * (Dg - BALL_MM) / BALL_MM, 2))
        out.append(rec)

    p = Path("findings/pingpong_ball_size_per_frame_v2.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nwrote {p}  ({len(out)} rows)")

    def stat(key):
        v = np.array([r[key] for r in out if r[key] != ""], float)
        return v
    print(f"\n{'source':<22}{'n':>5}{'mean mm':>10}{'sd':>8}{'mean |err|':>12}{'range mm':>16}")
    for key, name in (("D_from_tape_mm", "tape (closure)"),
                      ("D_perframe_mm", "DA3, per-frame (leaky)"),
                      ("D_strict_mm", "DA3, per-frame STRICT"),
                      ("D_globalscale_mm", "DA3, one global scale")):
        v = stat(key)
        e = np.abs(100 * (v - BALL_MM) / BALL_MM)
        print(f"{name:<22}{v.size:>5}{v.mean():>10.2f}{v.std(ddof=1):>8.2f}"
              f"{e.mean():>11.2f}%{f'{v.min():.1f}-{v.max():.1f}':>16}")


if __name__ == "__main__":
    main()
