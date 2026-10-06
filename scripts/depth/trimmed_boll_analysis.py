"""
trimmed_boll_analysis.py — boll tip-to-base length on the trimmed LeftDown set,
against the 38 physically measured bolls.

WHY THE TRIM MATTERS
    DA3 has no absolute scale; it reads it off scene content. With the
    greenhouse roof in frame the depth range ran to ~11.9 and the ball
    calibration was extrapolated for 86 % of measurements. Cropping the top 15 %
    removes the roof and the range falls to 0.47-1.34, against 0.25-1.03 for the
    ball calibration scene. The two scenes are now comparable, so applying the
    ball mapping here is a real out-of-sample prediction rather than an
    extrapolation.

TWO ANALYSES, AND ONLY ONE OF THEM IS A TEST
    A. ball calibration, untouched      -> is the predicted mean near 40.46 mm?
                                           A genuine prediction. Nothing here saw
                                           the ground truth.
    B. scale fitted to the GT mean      -> the mean then agrees BY CONSTRUCTION
                                           and proves nothing. What it does test
                                           is the SPREAD and the shape of the
                                           distribution, which one scale factor
                                           cannot fake.

GROUND TRUTH
    "Height" is the tip-to-base axis: its CV of 21.2 % is the figure
    depth_findings.md quotes as the true spread. n = 38, mean 40.46 mm.

    Track ids are not matched to boll ids, so this compares DISTRIBUTIONS, not
    boll against boll.
"""
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
RUN = Path("inference_results/left_down_v5_trimmed")
GT = "ground_truth/greenhouse_day_1/Cotton boll dimensions_Greenhouse 27Mar2026 (1).xlsx"
F_PX, GAMMA = 2905.6, 0.663


def ground_truth():
    z = zipfile.ZipFile(GT)
    s = z.read("xl/worksheets/sheet1.xml").decode("utf8")
    out = []
    for rn, body in re.findall(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', s, re.S):
        if rn == "1":
            continue
        c = {}
        for m in re.finditer(r'<c r="([A-Z]+)\d+"[^>]*>(?:<v>(.*?)</v>)?', body):
            if m.group(2) is not None:
                c[m.group(1)] = float(m.group(2))
        if "C" in c:
            out.append({"pot": c.get("A"), "boll": c.get("B"), "height": c["C"],
                        "length": c.get("D"), "width": c.get("E")})
    return out


def track(dets_by_frame, iou_thr=0.3):
    """Greedy IoU tracker. The scene is near-static, so this only has to survive
    small motion; it exists to give one measurement per physical boll rather
    than one per frame."""
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        i = max(0, x1 - x0) * max(0, y1 - y0)
        return i / (aw * ah + bw * bh - i + 1e-9)
    tracks, active = [], []
    for fi, (fname, dets) in enumerate(dets_by_frame):
        used = set()
        for t in active:
            best, bi = iou_thr, None
            for j, d in enumerate(dets):
                if j in used:
                    continue
                v = iou(t["bbox"], d["bbox"])
                if v > best:
                    best, bi = v, j
            if bi is not None:
                used.add(bi)
                t["bbox"] = dets[bi]["bbox"]
                t["dets"].append((fname, dets[bi]))
                t["miss"] = 0
            else:
                t["miss"] += 1
        for j, d in enumerate(dets):
            if j not in used:
                t = {"bbox": d["bbox"], "dets": [(fname, d)], "miss": 0}
                tracks.append(t); active.append(t)
        active = [t for t in active if t["miss"] <= 5]
    return tracks


def main():
    import cv2
    ci = json.loads((RUN / "depth" / "crop_info.json").read_text())
    y_off = ci["crop_top_px"]
    print(f"crop: top {y_off}px removed for depth inference "
          f"({ci['orig_height']} -> {ci['cropped_height']})")

    dets_by_frame = []
    for p in sorted((RUN / "tracked_json").glob("*.json")):
        dets_by_frame.append((p.stem, json.load(open(p))))
    tracks = track(dets_by_frame)
    tracks = [t for t in tracks if len(t["dets"]) >= 5]
    print(f"{sum(len(f[1]) for f in dets_by_frame)} detections -> "
          f"{len(tracks)} tracks with >=5 frames")

    dd = RUN / "depth" / "depth"
    rows = []
    for ti, t in enumerate(tracks):
        for fname, d in t["dets"]:
            npy = dd / f"{fname}.npy"
            if not npy.exists():
                continue
            k = d.get("kpts")
            if not k or len(k) < 2 or d.get("det_conf", 0) < 0.5:
                continue
            if k[0][2] < 0.5 or k[1][2] < 0.5:
                continue
            dep = np.load(npy).astype(np.float32)
            # detections are in ORIGINAL frame coords; depth is of the crop
            sx = dep.shape[1] / ci["cropped_width"]
            sy = dep.shape[0] / ci["cropped_height"]
            poly = np.asarray(d["mask_poly"], np.float32).copy()
            poly[:, 1] -= y_off
            c = poly.mean(0)
            q = np.stack([(c[0] + (poly[:, 0] - c[0]) * .55) * sx,
                          (c[1] + (poly[:, 1] - c[1]) * .55) * sy], 1).astype(np.int32)
            m = np.zeros(dep.shape, np.uint8)
            cv2.fillPoly(m, [q], 1)
            v = dep[(m > 0) & np.isfinite(dep) & (dep > 0)]
            if v.size < 4:
                continue
            px = float(np.hypot(k[0][0] - k[1][0], k[0][1] - k[1][1]))
            rows.append({"track": ti, "frame": fname, "px": px,
                         "z": float(np.median(v)),
                         "det_conf": d.get("det_conf"),
                         "visfrac": d.get("visfrac", 1),
                         "occ": d.get("occ_class")})
    print(f"{len(rows)} measurable boll-frames")

    gt = ground_truth()
    H = np.array([g["height"] for g in gt])
    print(f"\nGROUND TRUTH: n={len(H)}  mean {H.mean():.2f} mm  sd {H.std(ddof=1):.2f}  "
          f"CV {100*H.std(ddof=1)/H.mean():.1f}%  range {H.min():.1f}-{H.max():.1f}")

    # ---- A. ball calibration, untouched -> a real prediction ---------------
    from size_regression import load, solve_pinhole
    ball = load(); f_b, c_b = solve_pinhole(ball)
    zb = np.array([x["z"] for x in ball]); Zb = np.array([x["Ztape"] + c_b for x in ball])
    a_ball = float(np.exp(np.mean(np.log(zb) - GAMMA * np.log(Zb))))

    def lengths(a):
        per = {}
        for r in rows:
            Z = (r["z"] / a) ** (1.0 / GAMMA)
            per.setdefault(r["track"], []).append(r["px"] * Z / F_PX)
        return np.array([np.median(v) for v in per.values()]), per

    LA, perA = lengths(a_ball)
    zs = np.array([r["z"] for r in rows])
    print(f"\ndepth z on this trimmed set: {zs.min():.3f}-{zs.max():.3f}   "
          f"(ball calibration covered {zb.min():.3f}-{zb.max():.3f})")
    inb = ((zs >= zb.min()) & (zs <= zb.max())).mean()
    print(f"  {100*inb:.1f}% of readings now fall INSIDE the calibrated range "
          f"(was 13.8% before the trim)")

    print(f"\n=== A. BALL CALIBRATION APPLIED BLIND (a prediction) ===")
    print(f"  n={len(LA)} bolls   mean {LA.mean():.2f} mm   sd {LA.std(ddof=1):.2f}   "
          f"CV {100*LA.std(ddof=1)/LA.mean():.1f}%   range {LA.min():.1f}-{LA.max():.1f}")
    print(f"  ground truth mean is {H.mean():.2f} mm -> predicted mean is "
          f"{100*(LA.mean()-H.mean())/H.mean():+.1f}% off")

    # ---- B. scale fitted to the GT mean -> tests spread only ---------------
    k = H.mean() / LA.mean()
    a_gt = a_ball / (k ** GAMMA)
    LB, perB = lengths(a_gt)
    print(f"\n=== B. SCALE FITTED TO THE GT MEAN (mean agrees by construction) ===")
    print(f"  n={len(LB)} bolls   mean {LB.mean():.2f} mm   sd {LB.std(ddof=1):.2f}   "
          f"CV {100*LB.std(ddof=1)/LB.mean():.1f}%   range {LB.min():.1f}-{LB.max():.1f}")
    print(f"  GT CV is {100*H.std(ddof=1)/H.mean():.1f}% -> measured spread is the test")

    # raw pixels, for contrast: what depth actually buys
    perpx = {}
    for r in rows:
        perpx.setdefault(r["track"], []).append(r["px"])
    P = np.array([np.median(v) for v in perpx.values()])
    Pm = P / P.mean() * H.mean()
    print(f"\n  raw pixels only, rescaled to the same mean: CV "
          f"{100*Pm.std(ddof=1)/Pm.mean():.1f}%  range {Pm.min():.1f}-{Pm.max():.1f}")

    out = Path("findings/trimmed_boll")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out / "per_boll.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "n_frames", "median_px", "median_z",
                    "length_ballcal_mm", "length_gtcal_mm"])
        for t in sorted(perA):
            w.writerow([t, len(perA[t]), round(float(np.median(perpx[t])), 1),
                        round(float(np.median([r["z"] for r in rows if r["track"] == t])), 4),
                        round(float(np.median(perA[t])), 2),
                        round(float(np.median(perB[t])), 2)])
    json.dump({"a_ball": a_ball, "a_gt": a_gt, "gamma": GAMMA, "focal_px": F_PX,
               "scale_ratio_gt_over_ball": k},
              open(out / "calibration.json", "w"), indent=2)
    print(f"\nwrote {out}/per_frame.csv, per_boll.csv, calibration.json")


if __name__ == "__main__":
    main()
