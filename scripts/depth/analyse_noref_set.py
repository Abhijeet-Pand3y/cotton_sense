"""
analyse_noref_set.py — measure a set that has NO tape-measured ball.

WHY THIS EXISTS
  analyse_ball_sets.py needs a tape distance for every ball: it fits gamma and
  the focal length against real distance and scores the result in millimetres.
  A set recorded without measuring where the ball was cannot enter that path at
  all.

  GX010120 is such a set. One 40 mm ball sits on a stand at an unrecorded
  distance while a water bottle is placed at tape-measured distances (5, 11, 15,
  19, 25, 29 in). So the ball's SIZE is known and its DISTANCE is not, which is
  exactly the deployment case the team document calls "nothing of known size in
  the picture": every constant has to come from other recordings, and the frame
  itself contributes nothing but pixels.

WHAT IS PREDICTED, AND FROM WHAT
  Focal length f, tape offset c, compression exponent gamma and the global scale
  a are all fitted on the OTHER recordings in findings/combined_ball_frames.csv
  and then applied here unchanged. This set is never in its own training data,
  so the numbers below are held-out predictions, not fits.

      distance   Z = (z / a) ** (1 / gamma)
      size       D = 2 * r * Z / f

  Both come out of the same z, so they are one measurement reported two ways —
  a distance error and a size error of the same magnitude and opposite sign.
  They are not independent evidence.

THE TWO THINGS THIS SET CAN ACTUALLY SETTLE
  1. SIZE, scored.  The ball is a 40 mm sphere, so the predicted diameter has a
     true value to be wrong against. This is the headline number.

  2. DISTANCE, cross-checked but NOT scored.  With no tape there is no ground
     truth for the ball's distance. What there is instead is a second, fully
     independent estimate from the ball's pixel radius and the fitted focal
     alone, using no depth model at all:  Z_geom = f * 40 / (2 * r) - c.
     Agreement between the two bounds how far off DA3 can be here; it is not an
     accuracy measurement, and is not reported as one.

  3. THE BASELINE COMPARISON.  Two frames hold the ball alone and six hold the
     ball with the bottle at a known distance. Since the ball never changes,
     every difference between those frames is the bottle's doing. That makes the
     bottle's tape distance the experimental variable and the ball's measured
     size the response — the context effect, measured against a clean base.

USAGE
  python scripts/depth/analyse_noref_set.py --set frames/GX010120_test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_validation import fit_fc, gamma_of, BALL          # noqa: E402

COMBINED = "findings/combined_ball_frames.csv"


def sample_ball_depth(img_path, depth_path, cx, cy, r, inner=0.6):
    """Median DA3 depth over the inner disc of the ball, in depth-map pixels.

    Same sampling as ball_video_test.cmd_measure — inner 60% so the silhouette
    edge, where depth bleeds onto the background, never enters the median. The
    circle is this frame's own fit, not a static one, because the ball moves.
    """
    import cv2
    depth = np.load(depth_path).astype(np.float32)
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    sx, sy = depth.shape[1] / W, depth.shape[0] / H
    yy, xx = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    rr = r * inner * (sx + sy) / 2
    m = ((xx - cx * sx) ** 2 + (yy - cy * sy) ** 2) <= rr * rr
    v = depth[m & np.isfinite(depth) & (depth > 0)]
    if v.size < 5:
        return None, None, int(v.size)
    return (float(np.median(v)),
            float(np.percentile(v, 75) - np.percentile(v, 25)), int(v.size))


def load_training(exclude_set):
    """Rows from every OTHER recording, in the shape final_validation expects."""
    rows = []
    for x in csv.DictReader(open(COMBINED)):
        if x["status"] != "ok" or not x["z"] or not x["tape_mm"]:
            continue
        if x["set"] == exclude_set:
            continue
        rows.append({"set": x["set"], "frame": x["frame"], "ball": x["ball"],
                     "z": float(x["z"]), "Ztape": float(x["tape_mm"]),
                     "r": float(x["r_px"]), "base": x["is_base"] == "1"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set folder, e.g. frames/GX010120_test")
    ap.add_argument("--out_csv", help="default <set>/noref_analysis.csv")
    ap.add_argument("--out_md", help="default findings/per_set/<name>.md")
    args = ap.parse_args()

    sdir = Path(args.set)
    name = sdir.name
    out_csv = Path(args.out_csv) if args.out_csv else sdir / "noref_analysis.csv"
    out_md = Path(args.out_md) if args.out_md else Path("findings/per_set") / f"{name}.md"

    per = list(csv.DictReader(open(sdir / "locate_out" / "per_frame_balls.csv")))
    dep = {p.stem: p for p in (sdir / "depth" / "depth").glob("*.npy")}

    # ---- constants, fitted on every OTHER recording -------------------------
    tr = load_training(name)
    f, c = fit_fc(tr)
    g = gamma_of(tr, c)
    zt = np.array([x["z"] for x in tr])
    Zt = np.array([x["Ztape"] + c for x in tr])
    a = float(np.exp(np.mean(np.log(zt) - g * np.log(Zt))))
    n_sets = len({x["set"] for x in tr})
    print(f"trained on {len(tr)} ball-frames from {n_sets} other recordings")
    print(f"  focal f = {f:.0f} px   tape offset c = {c:+.1f} mm"
          f"   gamma = {g:.3f}   scale a = {a:.4f}")

    rows = []
    for x in per:
        if x["status"] != "ok" or not x["cx"]:
            print(f"  [skip] {x['frame']}: status={x['status']}")
            continue
        stem = x["frame"]
        img = (sdir / "baseline" / f"{stem}.jpg" if x["is_base"] == "1"
               else sdir / f"{stem}.jpg")
        if stem not in dep:
            print(f"  [skip] {stem}: no depth map")
            continue
        cx, cy, r = float(x["cx"]), float(x["cy"]), float(x["r"])
        z, iqr, n = sample_ball_depth(img, dep[stem], cx, cy, r)
        if z is None:
            print(f"  [skip] {stem}: only {n} depth pixels on the ball")
            continue

        Z = (z / a) ** (1.0 / g)          # DA3 distance, no reference
        D = 2 * r * Z / f                 # size that implies
        Zgeom = f * BALL / (2 * r) - c    # distance from pixel size alone
        obj = "" if x["is_base"] == "1" else stem.split("in_")[0][3:]
        rows.append({
            "set": name, "frame": stem,
            "scene": "ball only" if x["is_base"] == "1" else "bottle present",
            "object_distance_in": obj,
            "r_px": round(r, 2), "cx": round(cx, 1), "cy": round(cy, 1),
            "move_px": x["move_px"], "match": x["match"],
            "z": round(z, 5), "z_iqr": round(iqr, 5), "z_npix": n,
            "noref_distance_mm": round(Z, 1),
            "noref_size_mm": round(D, 2),
            "noref_size_error_pct": round(100 * (D - BALL) / BALL, 2),
            "geom_distance_mm": round(Zgeom, 1),
            "distance_vs_geom_pct": round(100 * (Z - Zgeom) / Zgeom, 2),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} frames)")

    # ---- summary ------------------------------------------------------------
    err = np.array([x["noref_size_error_pct"] for x in rows])
    D = np.array([x["noref_size_mm"] for x in rows])
    base = np.array([x["scene"] == "ball only" for x in rows])
    summary = {
        "set": name, "n_frames": len(rows),
        "focal_px": round(f, 1), "tape_offset_mm": round(c, 2),
        "gamma": round(g, 4), "scale_a": round(a, 5),
        "trained_on_sets": n_sets, "trained_on_ballframes": len(tr),
        "size_mm_mean": round(float(D.mean()), 2),
        "size_mm_sd": round(float(D.std(ddof=1)), 2),
        "size_err_pct_mean_signed": round(float(err.mean()), 2),
        "size_err_pct_mean_abs": round(float(np.abs(err).mean()), 2),
        "size_err_mm_mean_abs": round(float(np.abs(D - BALL).mean()), 2),
        "base_size_mm_mean": round(float(D[base].mean()), 2),
        "object_size_mm_mean": round(float(D[~base].mean()), 2),
        "object_minus_base_mm": round(float(D[~base].mean() - D[base].mean()), 2),
    }
    (sdir / "noref_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return rows, summary


if __name__ == "__main__":
    main()
