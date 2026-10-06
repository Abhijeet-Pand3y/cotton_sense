"""
boll_length.py — real tip-to-base boll length from DA3 depth, on LeftDown_v5.

THE CHAIN
    px          |tip - base| in pixels, from the pose head
    z           DA3 depth sampled over the BOLL MASK, not at the keypoints
    Z           real distance, from the mapping fitted on the ping-pong balls
    length      px * Z / f

WHY DEPTH COMES FROM THE MASK, NOT THE KEYPOINTS
    The tip and base are predicted as heatmap peaks. They sit on the boll's
    silhouette edge, where the depth map transitions from boll to background
    over a pixel or two - and DA3 runs at 504x322, so one depth pixel spans
    roughly nine image pixels. Sampling there mixes boll depth with whatever is
    behind it. The mask interior is the only place the depth is reliably the
    boll's own, so an eroded mask is used and its median taken.

PROJECTED LENGTH IS A LOWER BOUND
    px*Z/f measures the tip-base separation PROJECTED onto the image plane. A
    boll tilted towards or away from the camera has part of its length along the
    view axis, which the projection does not see. So this number is biased
    LOW by 1/cos(tilt), and never high.

    Two estimates are therefore reported:
      length_2d   projection only. Certain, but a lower bound.
      length_3d   adds the tip-base depth difference back in, sampled from the
                  depth map near each keypoint but INSIDE the mask. Recovers the
                  tilt, at the cost of depending on a depth difference of a few
                  millimetres measured by a network whose noise is larger than
                  that. Reported for comparison, and not trusted by default.

NO REFERENCE OBJECT IS PRESENT
    Nothing of known size is in these frames, so this is the "no reference"
    regime from findings/size_from_depth.md - roughly 14 % distance error, and
    that carries one-for-one into length.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

F_PX_DEFAULT = 2905.6      # solved from the 40 mm balls (size_regression.py)
C_MM_DEFAULT = 32.0        # tape offset, entrance pupil


def fit_mapping(model="proportional"):
    """Refit the depth->distance mapping on the ball data, in pupil units."""
    from size_regression import load, solve_pinhole, fit_model
    rows = load()
    f_px, c_mm = solve_pinhole(rows)
    for x in rows:
        x["Zp"] = x["Ztape"] + c_mm
    fn = fit_model(model, np.array([x["z"] for x in rows]),
                   np.array([x["Zp"] for x in rows]))
    return fn, f_px, c_mm


def mask_depth(depth, poly, W, H, erode=0.55, min_px=4):
    """Median depth inside a shrunk copy of the mask polygon.

    Shrinking towards the centroid keeps the sample off the silhouette edge,
    where a depth pixel straddles boll and background.
    """
    import cv2
    p = np.asarray(poly, np.float32)
    if p.ndim != 2 or len(p) < 3:
        return None, 0
    c = p.mean(0)
    p = (c + (p - c) * erode)
    sx, sy = depth.shape[1] / W, depth.shape[0] / H
    q = np.stack([p[:, 0] * sx, p[:, 1] * sy], 1).astype(np.int32)
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(m, [q], 1)
    v = depth[(m > 0) & np.isfinite(depth) & (depth > 0)]
    if v.size < min_px:                       # tiny boll: fall back to centroid
        yy, xx = int(round(c[1] * sy)), int(round(c[0] * sx))
        if 0 <= yy < depth.shape[0] and 0 <= xx < depth.shape[1]:
            d = depth[max(0, yy-1):yy+2, max(0, xx-1):xx+2]
            d = d[np.isfinite(d) & (d > 0)]
            return (float(np.median(d)), int(d.size)) if d.size else (None, 0)
        return None, 0
    return float(np.median(v)), int(v.size)


def kp_depth(depth, poly, xy, W, H, rad_frac=0.18):
    """Depth near a keypoint but restricted to the mask, per the brief.

    The keypoint itself sits on the edge, so a disc around it is intersected
    with the mask and only the inside part is used.
    """
    import cv2
    p = np.asarray(poly, np.float32)
    if p.ndim != 2 or len(p) < 3:
        return None
    sx, sy = depth.shape[1] / W, depth.shape[0] / H
    q = np.stack([p[:, 0] * sx, p[:, 1] * sy], 1).astype(np.int32)
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(m, [q], 1)
    ext = np.hypot(p[:, 0].ptp() * sx, p[:, 1].ptp() * sy)
    r = max(1.5, rad_frac * ext)
    yy, xx = np.ogrid[0:depth.shape[0], 0:depth.shape[1]]
    disc = ((xx - xy[0] * sx) ** 2 + (yy - xy[1] * sy) ** 2) <= r * r
    v = depth[(m > 0) & disc & np.isfinite(depth) & (depth > 0)]
    return float(np.median(v)) if v.size >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="inference_results/LeftDown_v5")
    ap.add_argument("--model", default="proportional",
                    help="depth->distance mapping (no reference object here, so "
                         "proportional is the cross-validated best)")
    ap.add_argument("--focal_px", type=float, default=None)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--min_kp_conf", type=float, default=0.5)
    ap.add_argument("--out", default="findings/boll_length")
    args = ap.parse_args()

    import cv2
    run = Path(args.run)
    fn, f_solved, c_mm = fit_mapping(args.model)
    f_px = args.focal_px or f_solved
    print(f"mapping = {args.model} (fitted on the ball sets), f = {f_px:.1f} px")

    tracked = json.load(open(run / "tracked.json"))
    dep_dir = run / "depth" / "depth"
    rows, missing = [], 0
    for fr in tracked:
        stem = Path(fr["src"]).stem
        npy = dep_dir / f"{stem}.npy"
        if not npy.exists():
            missing += 1
            continue
        if not fr["bolls"]:
            continue
        img = cv2.imread(str(run / "frames" / fr["src"]))
        if img is None:
            continue
        H, W = img.shape[:2]
        depth = np.load(npy).astype(np.float32)
        for b in fr["bolls"]:
            k = b.get("kpts")
            if not k or len(k) < 2 or b.get("det_conf", 0) < args.min_conf:
                continue
            (tx, ty, tc), (bx, by, bc) = k[0], k[1]
            if tc < args.min_kp_conf or bc < args.min_kp_conf:
                continue
            poly = b.get("mask_poly")
            z, npx = mask_depth(depth, poly, W, H)
            if z is None:
                continue
            Z = float(fn(z))
            px = float(np.hypot(tx - bx, ty - by))
            l2d = px * Z / f_px
            zt = kp_depth(depth, poly, (tx, ty), W, H)
            zb = kp_depth(depth, poly, (bx, by), W, H)
            l3d = ""
            if zt and zb:
                Zt, Zb = float(fn(zt)), float(fn(zb))
                dx = (tx - bx) * Z / f_px
                dy = (ty - by) * Z / f_px
                l3d = round(float(np.sqrt(dx * dx + dy * dy + (Zt - Zb) ** 2)), 2)
            rows.append({
                "frame": stem, "track_id": b.get("track_id"),
                "det_conf": round(b.get("det_conf", 0), 4),
                "tip_conf": round(tc, 4), "base_conf": round(bc, 4),
                "occ_class": b.get("occ_class"),
                "is_occluded": b.get("is_occluded"),
                "visfrac": round(b.get("visfrac", 1), 4),
                "px_tip_base": round(px, 2), "mask_px_in_depth": npx,
                "z_boll": round(z, 5), "Z_mm": round(Z, 1),
                "length_2d_mm": round(l2d, 2), "length_3d_mm": l3d,
                "z_tip": round(zt, 5) if zt else "",
                "z_base": round(zb, 5) if zb else "",
            })
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "boll_length_per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} boll-frames measured ({missing} frames had no depth map)")
    print(f"wrote {out/'boll_length_per_frame.csv'}")


if __name__ == "__main__":
    main()
