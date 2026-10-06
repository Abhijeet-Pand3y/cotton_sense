"""
build_team_csv.py — the shareable measurement table.

Every number here was produced by predicting a ball the calculation had never
seen: the camera constants, the depth curve and the scale were all worked out
from OTHER recordings, then applied. No ball helped set the constants used to
measure it.

Columns that could not be validated that way are deliberately absent.

Three setups, described by what is physically in the picture:
    setup1  nothing of known size in the picture
    setup2  one object of known size in the picture
    setup3  several objects of known size in the picture
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/depth")
from final_validation import fit_fc, gamma_of, BALL          # noqa: E402
from size_regression import load                             # noqa: E402


def main():
    rows = load()
    sets = sorted({x["set"] for x in rows})
    byframe = defaultdict(list)
    for x in rows:
        byframe[(x["set"], x["frame"])].append(x)

    out = []
    for s in sets:
        tr = [x for x in rows if x["set"] != s]
        f_o, c_o = fit_fc(tr)
        g_o = gamma_of(tr, c_o)
        if g_o is None:
            continue
        zt = np.array([x["z"] for x in tr]); Zt = np.array([x["Ztape"] + c_o for x in tr])
        a_none = float(np.exp(np.mean(np.log(zt) - g_o * np.log(Zt))))
        for k, rr in byframe.items():
            if k[0] != s:
                continue
            for j, y in enumerate(rr):
                Ztrue = y["Ztape"] + c_o
                others = [rr[i] for i in range(len(rr)) if i != j]

                def pred(a):
                    Z = (y["z"] / a) ** (1 / g_o)
                    return Z, 2 * y["r"] * Z / f_o

                Z1, D1 = pred(a_none)
                rec = {
                    "recording": y["set"],
                    "photo": y["frame"],
                    "scene": "balls only" if y["base"] else "extra object present",
                    "ball": y["ball"],
                    "balls_in_photo": len(rr),
                    "true_distance_in": round(y["Ztape"] / 25.4, 1),
                    "true_distance_mm": round(Ztrue, 1),
                    "true_size_mm": BALL,
                    "measured_radius_px": round(y["r"], 2),
                    "depth_model_output": round(y["z"], 5),
                    "setup1_distance_mm": round(Z1, 1),
                    "setup1_size_mm": round(D1, 2),
                    "setup1_size_error_pct": round(100 * (D1 - BALL) / BALL, 2),
                }
                if others:
                    # setup2: each possible guide in turn, then the typical
                    # outcome. A user picks one guide; the median says what a
                    # typical pick delivers, and the range says how much the
                    # choice matters.
                    ds, Zs = [], []
                    for o in others:
                        a1 = o["z"] / ((o["Ztape"] + c_o) ** g_o)
                        Zi, Di = pred(a1)
                        ds.append(Di); Zs.append(Zi)
                    ds = np.array(ds)
                    rec.update(
                        setup2_distance_mm=round(float(np.median(Zs)), 1),
                        setup2_size_mm=round(float(np.median(ds)), 2),
                        setup2_size_error_pct=round(
                            100 * (float(np.median(ds)) - BALL) / BALL, 2),
                        setup2_best_case_size_mm=round(
                            float(ds[np.argmin(np.abs(ds - BALL))]), 2),
                        setup2_worst_case_size_mm=round(
                            float(ds[np.argmax(np.abs(ds - BALL))]), 2),
                    )
                    Zo = np.array([o["Ztape"] + c_o for o in others])
                    zo = np.array([o["z"] for o in others])
                    a3 = float(np.exp(np.mean(np.log(zo) - g_o * np.log(Zo))))
                    Z3, D3 = pred(a3)
                    rec.update(setup3_distance_mm=round(Z3, 1),
                               setup3_size_mm=round(D3, 2),
                               setup3_size_error_pct=round(100 * (D3 - BALL) / BALL, 2))
                else:
                    for c_ in ("setup2_distance_mm", "setup2_size_mm",
                               "setup2_size_error_pct", "setup2_best_case_size_mm",
                               "setup2_worst_case_size_mm", "setup3_distance_mm",
                               "setup3_size_mm", "setup3_size_error_pct"):
                        rec[c_] = ""
                out.append(rec)

    p = Path("findings/team_share/ball_measurements.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"wrote {p}  ({len(out)} rows, {len(out[0])} columns)")

    for tag in ("setup1", "setup2", "setup3"):
        v = np.array([r[f"{tag}_size_mm"] for r in out if r[f"{tag}_size_mm"] != ""])
        e = np.abs(100 * (v - BALL) / BALL)
        print(f"  {tag}: n={v.size:>3}  mean {v.mean():.2f} mm  sd {v.std(ddof=1):.2f}  "
              f"typical error {e.mean():.1f}%")


if __name__ == "__main__":
    main()
