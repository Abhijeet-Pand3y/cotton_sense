"""
verify_noref_set.py — re-derive every number quoted about GX010120_test from the
shipped files alone.

The claims in findings/per_set/GX010120_test.md and section 6 of
findings/team_share/README.md were computed in an analysis session. This script
recomputes each one from findings/team_share/noref_new_data.csv,
findings/team_share/ball_measurements.csv and
frames/GX010120_test/noref_summary.json, and fails loudly on any that no longer
match. Run it after anything that could move those files.

  python scripts/depth/verify_noref_set.py
"""
import csv
import json
import sys
from math import sqrt

import numpy as np

BALL = 40.0
NEW = "findings/team_share/noref_new_data.csv"
OLD = "findings/team_share/ball_measurements.csv"
SUM = "frames/GX010120_test/noref_summary.json"


def main():
    new = list(csv.DictReader(open(NEW)))
    olds = np.array([float(r["setup1_size_error_pct"])
                     for r in csv.DictReader(open(OLD))])
    old = np.abs(olds)

    D = np.array([float(r["noref_size_mm"]) for r in new])
    e = np.array([float(r["noref_size_error_pct"]) for r in new])
    z = np.array([float(r["z"]) for r in new])
    rp = np.array([float(r["r_px"]) for r in new])
    Zg = np.array([float(r["geom_distance_mm"]) for r in new])
    Zd = np.array([float(r["noref_distance_mm"]) for r in new])
    dvg = np.array([float(r["distance_vs_geom_pct"]) for r in new])
    b = np.array([r["scene"] == "ball only" for r in new])
    o = np.array([float(r["object_distance_in"]) for r in new if r["object_distance_in"]])
    d = np.array([float(r["noref_size_mm"]) for r in new if r["object_distance_in"]])
    s = json.load(open(SUM))

    va, vb = old.var(ddof=1) / len(old), np.abs(e).var(ddof=1) / len(e)
    t = (np.abs(e).mean() - old.mean()) / sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(old) - 1) + vb ** 2 / (len(e) - 1))

    checks = [
        ("size mean mm", D.mean(), 44.43), ("size sd mm", D.std(ddof=1), 3.13),
        ("mean abs err %", np.abs(e).mean(), 11.08),
        ("mean abs err mm", np.abs(D - BALL).mean(), 4.43),
        ("signed mean err %", e.mean(), 11.08),
        ("fraction positive", (e > 0).mean(), 1.0),
        ("size min mm", D.min(), 40.08), ("size max mm", D.max(), 49.13),
        ("size span, % of true", 100 * (D.max() - D.min()) / BALL, 22.6),
        ("r_px spread %", 100 * (rp.max() - rp.min()) / rp.mean(), 1.38),
        ("z spread %", 100 * (z.max() - z.min()) / z.mean(), 14.7),
        ("geometry Z mean mm", Zg.mean(), 497.8),
        ("geometry Z sd mm", Zg.std(ddof=1), 2.7),
        ("geometry Z in", Zg.mean() / 25.4, 19.60),
        ("DA3 Z mean mm", Zd.mean(), 588.4),
        ("DA3 vs geometry %", dvg.mean(), 18.22),
        ("base mean mm", D[b].mean(), 42.86),
        ("bottle mean mm", D[~b].mean(), 44.96),
        ("bottle sd mm", D[~b].std(ddof=1), 3.41),
        ("bottle - base mm", D[~b].mean() - D[b].mean(), 2.10),
        ("base pair spread mm", abs(D[b][0] - D[b][1]), 2.74),
        ("r^2, size vs bottle distance", np.corrcoef(o, d)[0, 1] ** 2, 0.07),
        ("baseline mean abs %", old.mean(), 16.99),
        ("baseline mean abs mm", BALL * old.mean() / 100, 6.80),
        ("baseline signed %", olds.mean(), -2.09),
        ("baseline fraction positive", (olds > 0).mean(), 0.54),
        ("Welch t", t, -2.03), ("Welch df", df, 8.6),
        ("focal px", s["focal_px"], 2905.6),
        ("tape offset mm", s["tape_offset_mm"], 32.01),
        ("gamma", s["gamma"], 0.7303), ("scale a", s["scale_a"], 0.0053),
        ("training ball-frames", s["trained_on_ballframes"], 116),
        ("training recordings", s["trained_on_sets"], 10),
    ]

    bad = 0
    for label, got, want in checks:
        ok = abs(got - want) <= 0.01 * max(1.0, abs(want))
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {got:.4g} (documented {want})")
    print(f"\n{len(checks) - bad}/{len(checks)} checks pass")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
