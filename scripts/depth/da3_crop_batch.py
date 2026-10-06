"""
da3_crop_batch.py — replicate the crop/scale test across many frames, and
measure frame-to-frame depth stability.

TWO QUESTIONS AT ONCE
  1. CROP SCALE.  Does the boll/full depth ratio behave the same way on every
     frame? The single-frame test showed absolute scale collapsing on tight crops
     and recovering as context grows. If that curve is consistent across frames,
     it is a property of the method (correctable); if it scatters, cropping is
     unpredictable and cannot be rescaled reliably.

  2. FRAME-TO-FRAME STABILITY.  Near-identical frames can yield different depth.
     For every track_id seen in several frames, this reports the spread of the
     full-frame boll depth across those frames. That spread is the noise floor:
     no crop trick can measure better than it, and it says directly how many
     frames need averaging to reach a target precision.

USAGE
  python da3_crop_batch.py \
      --images_dir data/test_data/frames \
      --det_dir    data/test_data/tracked_json \
      --sizes 600 1000 1500 2000 \
      --max_bolls 3 \
      --out crop_batch/

  # crop test is the expensive part; skip it to only measure stability
  python da3_crop_batch.py --images_dir ... --det_dir ... --no_crops --out ...

OUTPUT
  <out>/results.csv   one row per (frame, boll, window)
  console             per-size aggregates + per-track stability table

NOTES
  * Frames and detections are paired by filename stem.
  * Bolls are ranked by mask area and the largest --max_bolls are used, since
    small bolls give too few depth pixels to measure.
  * Bolls whose window would be clipped by the image edge are skipped for that
    window size, so sizes stay comparable.
  * Written against the DA3 README API; not executed here.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_dets(path):
    data = json.load(open(path))
    return data.get("bolls", data) if isinstance(data, dict) else data


def poly_area(poly):
    p = np.asarray(poly, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def boll_center(d):
    if d.get("mask_poly"):
        p = np.asarray(d["mask_poly"], dtype=np.float64)
        return p[:, 0].mean(), p[:, 1].mean()
    if d.get("bbox"):
        bx, by, bw, bh = d["bbox"]
        return bx + bw / 2, by + bh / 2
    return None


def poly_mask(poly, x0, y0, w, h):
    import cv2
    pts = np.asarray(poly, dtype=np.float64) - np.array([x0, y0])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def stats(depth, mask):
    import cv2
    dh, dw = depth.shape
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (dw, dh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    v = depth[mask & np.isfinite(depth) & (depth > 0)]
    if v.size < 10:
        return None
    return {"med": float(np.median(v)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "n": int(v.size)}


def spread(vals):
    """Robust spread summary of a list of values."""
    a = np.asarray(vals, dtype=np.float64)
    med = float(np.median(a))
    iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
    return {"n": a.size, "med": med, "iqr": iqr,
            "cv_pct": 100 * float(a.std()) / med if med else float("nan"),
            "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--det_dir", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[600, 1000, 1500, 2000])
    ap.add_argument("--max_bolls", type=int, default=3,
                    help="largest N bolls per frame")
    ap.add_argument("--min_area_px", type=float, default=5000,
                    help="skip bolls with mask area below this")
    ap.add_argument("--no_crops", action="store_true",
                    help="full frame only (stability test, much faster)")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--limit_frames", type=int, default=0)
    ap.add_argument("--crop_top_frac", type=float, default=0.0,
                    help="Remove this fraction of the image from the TOP before "
                         "any processing (e.g. 0.22 to cut the greenhouse roof "
                         "above the backdrop). Detection coords are shifted to "
                         "match. Run with and without to compare stability.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3

    frames = sorted(p for p in Path(args.images_dir).iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    if args.limit_frames:
        frames = frames[:args.limit_frames]
    if not frames:
        raise SystemExit(f"no images in {args.images_dir}")

    det_by_stem = {p.stem: p for p in Path(args.det_dir).glob("*.json")}
    paired = [(f, det_by_stem[f.stem]) for f in frames if f.stem in det_by_stem]
    print(f"{len(frames)} frames, {len(paired)} with detections")
    if not paired:
        raise SystemExit("no frame/detection stem matches")

    out = Path(args.out)
    (out / "tmp").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    def depth_of(bgr, tag):
        p = out / "tmp" / f"{tag}.jpg"
        cv2.imwrite(str(p), bgr)
        pred = model.inference([str(p)])
        return np.asarray(pred.depth)[0].astype(np.float32)

    rows = []
    # track_id -> list of full-frame boll depths (frame-to-frame stability)
    by_track = defaultdict(list)

    for fi, (img_path, det_path) in enumerate(paired):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Optionally drop the top band (roof / sky above the backdrop). Detection
        # coords live in ORIGINAL image space, so shift them by the same offset.
        y_off = int(round(img.shape[0] * args.crop_top_frac)) if args.crop_top_frac else 0
        if y_off:
            img = img[y_off:]
        H, W = img.shape[:2]
        dets = load_dets(det_path)

        cand = []
        for i, d in enumerate(dets):
            poly = d.get("mask_poly")
            if not poly or len(poly) < 3:
                continue
            a = poly_area(poly)
            if a < args.min_area_px:
                continue
            c = boll_center(d)
            if not c:
                continue
            if y_off:
                poly = [[px, py - y_off] for px, py in poly]
                c = (c[0], c[1] - y_off)
                if c[1] < 0:          # boll was inside the removed band
                    continue
            cand.append((a, i, d, poly, c))
        cand.sort(key=lambda t: -t[0])
        cand = cand[:args.max_bolls]
        if not cand:
            continue

        dfull = depth_of(img, "full")
        print(f"\n[{fi+1}/{len(paired)}] {img_path.stem}: {len(cand)} boll(s)")

        for area, idx, d, poly, (cx, cy) in cand:
            sfull = stats(dfull, poly_mask(poly, 0, 0, W, H))
            if not sfull:
                continue
            tid = d.get("track_id")
            if tid is not None:
                by_track[tid].append(sfull["med"])
            rows.append({"frame": img_path.stem, "boll": idx, "track": tid,
                         "window": "full", "crop_w": W, "crop_h": H,
                         "med": sfull["med"], "iqr": sfull["iqr"],
                         "n": sfull["n"], "rel": 1.0,
                         "iqr_pct": 100 * sfull["iqr"] / sfull["med"]})
            print(f"   boll#{idx} track={tid} area={area:.0f}px  "
                  f"full med={sfull['med']:.3f} iqr={sfull['iqr']:.4f} "
                  f"n={sfull['n']}")

            if args.no_crops:
                continue
            for s in args.sizes:
                half = s / 2
                x0, y0 = round(cx - half), round(cy - half)
                x1, y1 = round(cx + half), round(cy + half)
                if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                    print(f"      crop{s}: skipped (clipped by image edge)")
                    continue
                sub = img[y0:y1, x0:x1]
                dc = depth_of(sub, f"c{s}")
                sc = stats(dc, poly_mask(poly, x0, y0, sub.shape[1], sub.shape[0]))
                if not sc:
                    continue
                rel = sc["med"] / sfull["med"]
                rows.append({"frame": img_path.stem, "boll": idx, "track": tid,
                             "window": f"crop{s}", "crop_w": sub.shape[1],
                             "crop_h": sub.shape[0], "med": sc["med"],
                             "iqr": sc["iqr"], "n": sc["n"], "rel": rel,
                             "iqr_pct": 100 * sc["iqr"] / sc["med"]})
                print(f"      crop{s}: med={sc['med']:.3f} rel={rel:.3f} "
                      f"iqr%={100*sc['iqr']/sc['med']:.2f} n={sc['n']}")

    if not rows:
        raise SystemExit("no measurements produced")

    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    # ---- aggregate 1: crop scale behaviour --------------------------------
    if not args.no_crops:
        print("\n" + "=" * 74)
        print("CROP SCALE — boll depth relative to the full frame (1.0 = agrees)")
        print("=" * 74)
        print(f"{'window':>10} {'n':>4} {'rel med':>9} {'rel IQR':>9} "
              f"{'min':>7} {'max':>7} {'boll IQR%':>10}")
        for w_ in ["full"] + [f"crop{s}" for s in args.sizes]:
            sel = [r for r in rows if r["window"] == w_]
            if not sel:
                continue
            sp = spread([r["rel"] for r in sel])
            ip = float(np.median([r["iqr_pct"] for r in sel]))
            print(f"{w_:>10} {sp['n']:>4} {sp['med']:>9.3f} {sp['iqr']:>9.3f} "
                  f"{sp['min']:>7.3f} {sp['max']:>7.3f} {ip:>10.2f}")
        print("-" * 74)
        print("rel med far from 1.0  -> that window size distorts absolute scale")
        print("rel IQR small         -> distortion is CONSISTENT, so correctable")
        print("rel IQR large         -> unpredictable per frame; cropping unsafe")
        print("boll IQR% rising      -> finer 3-D structure resolved on the boll")

    # ---- aggregate 2: frame-to-frame stability ---------------------------
    multi = {t: v for t, v in by_track.items() if len(v) >= 3}
    print("\n" + "=" * 74)
    print("FRAME-TO-FRAME STABILITY — full-frame boll depth for the same track")
    print("=" * 74)
    if not multi:
        print("no track_id appeared in 3+ frames; cannot assess "
              "(use tracked_json and frames from one clip)")
    else:
        print(f"{'track':>7} {'frames':>7} {'med (m)':>9} {'IQR (m)':>9} "
              f"{'CV %':>7} {'range (m)':>11}")
        cvs = []
        for t, vals in sorted(multi.items()):
            sp = spread(vals)
            cvs.append(sp["cv_pct"])
            print(f"{t:>7} {sp['n']:>7} {sp['med']:>9.3f} {sp['iqr']:>9.4f} "
                  f"{sp['cv_pct']:>7.2f} {sp['max']-sp['min']:>11.4f}")
        mcv = float(np.median(cvs))
        print("-" * 74)
        print(f"median CV across tracks: {mcv:.2f}%")
        print("This is the per-frame noise floor. Averaging k frames shrinks it")
        print(f"by ~sqrt(k) if the error is independent: 4 frames -> ~{mcv/2:.2f}%, "
              f"9 -> ~{mcv/3:.2f}%.")
        print("If it does NOT shrink that way, the error is systematic (bias per")
        print("viewpoint), and averaging will not fix it.")


if __name__ == "__main__":
    main()
