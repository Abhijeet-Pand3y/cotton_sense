"""
sample_boll_depth.py — per-boll depth from a detection JSON + a depth .npy.

Pure numpy/opencv: no torch, no Detectron2, no DA3. Runs in any env, so the
detector, the depth network and this can each live wherever they like.

WHAT IT COMPUTES, PER BOLL
  boll depth   median over an ERODED mask. Eroding first drops the silhouette
               ring, where a pixel straddles boll and background and the depth
               network blends the two — the usual source of wild outliers.
  tip / base   median over a window around each keypoint, keeping only pixels
               within --tol_pct of the boll depth. The gate is what separates
               "this pixel is on the boll" from "this is the leaf behind it".
  dz           tip depth minus base depth: how much the boll leans toward or
               away from the camera.
  kept_pct     share of window pixels that passed the gate. This is the number
               to watch. Near 0 means the gate is too tight (or the keypoint is
               off the boll) and the result falls back to the boll depth, which
               is not a measurement. Near 100 means the gate is too loose to be
               rejecting anything.

RESOLUTION
  The depth map need not match the image. Detection coordinates are image pixels
  and get scaled into depth indices, with x and y handled separately since the
  aspect ratios rarely match exactly.

WHAT THIS IS NOT
  The tip/base region is a square window around the keypoint, not the thresholded
  heatmap region — inference does not currently save the heatmaps. A window is
  the honest approximation until it does.

USAGE
  python sample_boll_depth.py \
      --det   inference_results/RightDown_30fps/tracked_json/frame_000565_t18p85s.json \
      --depth depth_out/gopro/depth/frame_000565_t18p85s.npy \
      --erode_frac 0.15 --tol_pct 5 --kpt_win 25

  # whole folder, one row per boll, written to CSV
  python sample_boll_depth.py --det_dir <dir> --depth_dir <dir> --out depths.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_dets(path):
    data = json.load(open(path))
    return data.get("bolls", data) if isinstance(data, dict) else data


def scale_of(depth_shape, img_wh):
    """(sx, sy) mapping image px -> depth indices."""
    dh, dw = depth_shape
    iw, ih = img_wh
    return dw / iw, dh / ih


def eroded_mask(poly, depth_shape, img_wh, erode_frac):
    """Rasterise the polygon into depth-index space, then erode inward.

    Erosion is a fraction of the mask's smaller dimension, so it scales with the
    boll instead of removing a fixed ring that would swallow small bolls whole.
    Falls back to the un-eroded mask if erosion leaves too little.
    """
    import cv2
    sx, sy = scale_of(depth_shape, img_wh)
    pts = np.asarray(poly, dtype=np.float64) * np.array([sx, sy])
    m = np.zeros(depth_shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    full = int(m.sum())
    if full == 0:
        return None, 0, 0

    x, y, w, h = cv2.boundingRect(np.round(pts).astype(np.int32))
    k = int(round(min(w, h) * erode_frac))
    if k < 1:
        return m.astype(bool), full, full
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    e = cv2.erode(m, ker, iterations=1)
    if e.sum() < 10:                       # eroded away — keep the original
        return m.astype(bool), full, full
    return e.astype(bool), full, int(e.sum())


def kpt_depth(depth, kx, ky, img_wh, z_ref, tol_pct, win_px):
    """Median depth in a window around a keypoint, gated to the boll's depth.

    Returns (z, kept, total, est). est=True means nothing passed the gate and the
    boll depth was substituted — a fallback, not a measurement.
    """
    dh, dw = depth.shape
    sx, sy = scale_of(depth.shape, img_wh)
    cx, cy = int(round(kx * sx)), int(round(ky * sy))
    r = max(1, int(round(win_px * (sx + sy) / 2)))    # window px -> depth px
    x0, x1 = max(0, cx - r), min(dw, cx + r + 1)
    y0, y1 = max(0, cy - r), min(dh, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0, 0, True

    patch = depth[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    total = int(valid.sum())
    if total == 0:
        return (z_ref, 0, 0, True) if z_ref is not None else (None, 0, 0, True)

    if z_ref is None:
        v = patch[valid]
        return float(np.median(v)), int(v.size), total, False

    keep = valid & (np.abs(patch - z_ref) <= abs(z_ref) * tol_pct / 100.0)
    n = int(keep.sum())
    if n < 5:
        return z_ref, n, total, True
    return float(np.median(patch[keep])), n, total, False


def load_crop_info(depth_dir, explicit=None):
    """Find crop_info.json written by run_da3_depth.py --crop_top_frac.

    Detections are in ORIGINAL image coordinates; if the depth was computed on a
    cropped frame, every y must be shifted by the same offset or all sampling
    lands in the wrong place. Reading the sidecar keeps the two in sync
    automatically instead of relying on the numbers being retyped correctly.
    """
    if explicit:
        cands = [Path(explicit)]
    else:
        d = Path(depth_dir)
        cands = [d.parent / "crop_info.json", d / "crop_info.json"]
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        return 0, None
    info = json.loads(p.read_text())
    y_off = int(info.get("crop_top_px", 0))
    wh = (info.get("cropped_width"), info.get("cropped_height"))
    print(f"[crop_info] {p}: top {y_off}px removed, "
          f"cropped frame {wh[0]}x{wh[1]}")
    return y_off, (wh if all(wh) else None)


def px_lengths(poly, tip, base):
    """Pixel-space lengths, in IMAGE px.

    kpt  : straight tip->base distance. Direct, but inherits keypoint jitter.
    axis : the mask's extent along the tip->base DIRECTION. Only the direction
           comes from the keypoints, so it is far steadier than the kpt distance
           while still measuring along the boll rather than across it.
    width: mask extent perpendicular to that axis.
    """
    out = {"px_kpt": None, "px_axis": None, "px_width": None}
    if not (tip and base and tip[0] is not None and base[0] is not None):
        return out
    ux, uy = base[0] - tip[0], base[1] - tip[1]
    n = (ux * ux + uy * uy) ** 0.5
    out["px_kpt"] = float(n)
    if n < 1e-6 or not poly:
        return out
    ux, uy = ux / n, uy / n
    p = np.asarray(poly, dtype=np.float64)
    t = p[:, 0] * ux + p[:, 1] * uy          # along the axis
    w = -p[:, 0] * uy + p[:, 1] * ux         # across it
    out["px_axis"] = float(t.max() - t.min())
    out["px_width"] = float(w.max() - w.min())
    return out


def process(det_path, depth_path, args, y_off=0, crop_wh=None):
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim != 2:
        raise SystemExit(f"{depth_path}: expected 2-D depth, got {depth.shape}")
    dets = load_dets(det_path)

    # Image size the detections should be measured against. With a top crop this
    # is the CROPPED frame, since that is what the depth map covers.
    img_wh = args.img_size or crop_wh
    if img_wh is None:
        mx = my = 0
        for d in dets:
            for x, y in (d.get("mask_poly") or []):
                mx, my = max(mx, x), max(my, y)
        if mx <= 0 or my <= 0:
            raise SystemExit("could not infer image size; pass --img_size W H")
        img_wh = (int(np.ceil(mx)), int(np.ceil(my)))

    rows = []
    for i, d in enumerate(dets):
        poly = d.get("mask_poly")
        if not poly or len(poly) < 3:
            continue
        if y_off:
            poly = [[px, py - y_off] for px, py in poly]
            if max(p[1] for p in poly) <= 0:      # entirely in the removed band
                continue
        em, n_full, n_er = eroded_mask(poly, depth.shape, img_wh, args.erode_frac)
        if em is None:
            continue
        v = depth[em & np.isfinite(depth) & (depth > 0)]
        if v.size < args.min_px:
            continue
        z_boll = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))

        kp = d.get("kpts") or []
        tip = kp[0] if len(kp) > 0 else None
        base = kp[1] if len(kp) > 1 else None
        zt = zb = None
        tk = tt = bk = bt = 0
        te = be = True
        if tip and tip[0] is not None:
            zt, tk, tt, te = kpt_depth(depth, tip[0], tip[1] - y_off, img_wh,
                                       z_boll, args.tol_pct, args.kpt_win)
        if base and base[0] is not None:
            zb, bk, bt, be = kpt_depth(depth, base[0], base[1] - y_off, img_wh,
                                       z_boll, args.tol_pct, args.kpt_win)

        # pixel lengths use the SAME shifted coords as the depth sampling
        tip_s = [tip[0], tip[1] - y_off, tip[2]] if (tip and tip[0] is not None) else None
        base_s = [base[0], base[1] - y_off, base[2]] if (base and base[0] is not None) else None
        pl = px_lengths(poly, tip_s, base_s)

        # Depth-corrected size proxy. True length = px * Z / f, so px * Z is the
        # true length times the constant f: proportional to real size, in
        # arbitrary units. A consistent scale error in Z only changes that
        # constant, so the proxy stays proportional even if Z is not metric.
        proxy_kpt = pl["px_kpt"] * z_boll if pl["px_kpt"] is not None else None
        proxy_axis = pl["px_axis"] * z_boll if pl["px_axis"] is not None else None

        rows.append({
            "frame": Path(det_path).stem, "boll": i, "track": d.get("track_id"),
            "occ_class": d.get("occ_class"), "det_conf": d.get("det_conf"),
            "z_boll": z_boll, "boll_iqr": iqr, "boll_iqr_pct": 100 * iqr / z_boll,
            "n_px": int(v.size), "erode_kept_pct": 100 * n_er / max(1, n_full),
            "z_tip": zt, "tip_kept_pct": 100 * tk / max(1, tt), "tip_est": te,
            "z_base": zb, "base_kept_pct": 100 * bk / max(1, bt), "base_est": be,
            "dz": (zt - zb) if (zt is not None and zb is not None) else None,
            "px_kpt": pl["px_kpt"], "px_axis": pl["px_axis"],
            "px_width": pl["px_width"],
            "size_proxy_kpt": proxy_kpt, "size_proxy_axis": proxy_axis,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det"), ap.add_argument("--depth")
    ap.add_argument("--det_dir"), ap.add_argument("--depth_dir")
    ap.add_argument("--erode_frac", type=float, default=0.15,
                    help="erode the mask inward by this fraction of its smaller side")
    ap.add_argument("--tol_pct", type=float, default=5.0,
                    help="keypoint pixels must be within this %% of the boll depth")
    ap.add_argument("--kpt_win", type=int, default=25,
                    help="half-size of the keypoint window, in IMAGE px")
    ap.add_argument("--min_px", type=int, default=20,
                    help="skip bolls with fewer eroded depth pixels than this")
    ap.add_argument("--img_size", nargs=2, type=int, default=None,
                    metavar=("W", "H"), help="image size if it cannot be inferred")
    ap.add_argument("--crop_info", default=None,
                    help="path to crop_info.json (default: next to --depth_dir)")
    ap.add_argument("--out", help="write all rows to this CSV")
    args = ap.parse_args()

    pairs = []
    if args.det and args.depth:
        pairs = [(Path(args.det), Path(args.depth))]
    elif args.det_dir and args.depth_dir:
        dmap = {p.stem: p for p in Path(args.depth_dir).glob("*.npy")}
        for j in sorted(Path(args.det_dir).glob("*.json")):
            if j.stem in dmap:
                pairs.append((j, dmap[j.stem]))
        print(f"{len(pairs)} frame(s) with both detections and depth")
    else:
        ap.error("need --det + --depth, or --det_dir + --depth_dir")

    depth_ref = args.depth_dir or str(Path(args.depth).parent)
    y_off, crop_wh = load_crop_info(depth_ref, args.crop_info)

    all_rows = []
    for det_p, dep_p in pairs:
        rows = process(det_p, dep_p, args, y_off, crop_wh)
        all_rows += rows
        if len(pairs) == 1:
            print(f"\n{det_p.stem}   depth {np.load(dep_p).shape}   "
                  f"erode {args.erode_frac:.0%}  tol {args.tol_pct}%  "
                  f"win {args.kpt_win}px")
            print("-" * 92)
            print(f"{'boll':>5} {'trk':>4} {'occ':>10} {'z_boll':>8} {'iqr%':>6} "
                  f"{'px':>5} {'z_tip':>8} {'t%':>5} {'z_base':>8} {'b%':>5} "
                  f"{'dz(mm)':>8}")
            print("-" * 92)
            for r in rows:
                def f(v, p=3):
                    return "  —  " if v is None else "%.*f" % (p, v)
                dz_s = "—" if r["dz"] is None else "%.1f" % (r["dz"] * 1000)
                print("%5d %4s %10s %8.3f %6.2f %5d %8s%s%5.0f %8s%s%5.0f %8s"
                      % (r["boll"], str(r["track"]), str(r["occ_class"])[:10],
                         r["z_boll"], r["boll_iqr_pct"], r["n_px"],
                         f(r["z_tip"]), "~" if r["tip_est"] else " ",
                         r["tip_kept_pct"],
                         f(r["z_base"]), "~" if r["base_est"] else " ",
                         r["base_kept_pct"], dz_s))
            print("-" * 92)
            print("~ = nothing passed the gate, boll depth substituted (not a measurement)")
            print("t% / b% = share of window pixels that passed the gate")

    if all_rows:
        ests = sum(1 for r in all_rows if r["tip_est"] or r["base_est"])
        kept = [r["tip_kept_pct"] for r in all_rows if not r["tip_est"]]
        print(f"\n{len(all_rows)} boll(s); {ests} fell back to the boll depth "
              f"({100*ests/len(all_rows):.0f}%)")
        if kept:
            print(f"median tip gate pass rate: {np.median(kept):.0f}%")
        print("  pass rate very low  -> tol_pct too tight, or keypoints off the boll")
        print("  pass rate near 100% -> tol_pct too loose to reject anything")

    if args.out and all_rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
