"""
da3_crop_compare.py — does cropping change what DA3 reports?

QUESTION
  Monocular depth is only defined up to an unknown scale, and the network infers
  that scale from scene context. Cropping changes the context, so absolute depth
  may shift between crops of the same boll. But the black backdrop sits at a
  roughly fixed distance and appears in most crops, so it may serve as an anchor:
  if backdrop depth is stable across crops, crops can be rescaled to agree; and
  if the boll/backdrop RATIO is stable even when absolutes drift, relative
  structure survives cropping and only the scale needs fixing.

WHAT IT DOES
  For one boll: cut fixed square windows of several sizes, boll centred in each,
  run DA3 on every crop and on the full frame, and report per crop
      boll depth     — median over the mask polygon (or a central disc)
      backdrop depth — median over the darkest pixels (the black cloth)
      ratio          — backdrop / boll
  Stable ratio across sizes => croppable with rescaling. Unstable => cropping
  changes the geometry the network sees, not just the scale.

BACKDROP DETECTION
  The cloth is identified by RGB darkness, not by depth: pixels below the
  --dark_pct luminance percentile. Chosen because the cloth is genuinely black,
  so this is independent of the depth values being tested. Note that DA3 has few
  cues on a textureless dark region, so its backdrop depth may be unreliable —
  that is part of what this script measures.

USAGE
  # centre from a detection JSON (uses its mask for the boll region)
  python da3_crop_compare.py \
      --image inference_results/LeftDown_v3/frames/frame_000500_t16p67s.jpg \
      --det   inference_results/LeftDown_v3/json/frame_000500_t16p67s.json \
      --boll_idx 0 \
      --sizes 600 1000 1500 2000 --out crop_cmp/

  # or an explicit centre, no detections needed
  python da3_crop_compare.py --image <frame.jpg> --center 1400 2600 \
      --sizes 600 1000 1500 2000 --out crop_cmp/

CAVEAT
  Written against the DA3 README API and not executed here. Crops are square and
  clamped to the image, so a window near an edge may be smaller than requested;
  the actual size is printed.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_boll(det_path, idx):
    """Return (center_xy, mask_poly) for one detection."""
    data = json.load(open(det_path))
    dets = data.get("bolls", data) if isinstance(data, dict) else data
    if not dets:
        raise SystemExit(f"no detections in {det_path}")
    if idx >= len(dets):
        raise SystemExit(f"--boll_idx {idx} but only {len(dets)} detections")
    d = dets[idx]
    poly = d.get("mask_poly")
    if poly:
        pts = np.asarray(poly, dtype=np.float64)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    elif d.get("bbox"):
        bx, by, bw, bh = d["bbox"]
        cx, cy = bx + bw / 2, by + bh / 2
    else:
        raise SystemExit("detection has neither mask_poly nor bbox")
    print(f"boll #{idx}: centre ({cx:.0f}, {cy:.0f}), "
          f"{len(poly) if poly else 0} mask verts, "
          f"det_conf={d.get('det_conf')}")
    return (cx, cy), poly


def crop_box(cx, cy, size, W, H):
    """Square window of `size` centred on (cx,cy), clamped to the image."""
    half = size / 2
    x0 = int(max(0, round(cx - half)))
    y0 = int(max(0, round(cy - half)))
    x1 = int(min(W, round(cx + half)))
    y1 = int(min(H, round(cy + half)))
    return x0, y0, x1, y1


def poly_mask(poly, x0, y0, w, h):
    """Rasterise a polygon (image coords) into a crop-local boolean mask."""
    import cv2
    pts = np.asarray(poly, dtype=np.float64) - np.array([x0, y0])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def central_disc(h, w, frac=0.12):
    """Fallback boll region: a small disc at the crop centre."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = min(h, w) * frac
    return ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= r * r


def stats(depth, mask):
    """Median/IQR/count of depth inside a boolean mask, resized to depth grid."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--det", help="per-frame detection JSON (for centre + mask)")
    ap.add_argument("--boll_idx", type=int, default=0)
    ap.add_argument("--center", nargs=2, type=float,
                    help="explicit centre x y (instead of --det)")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[600, 1000, 1500, 2000],
                    help="square window sizes in px")
    ap.add_argument("--dark_pct", type=float, default=20.0,
                    help="luminance percentile defining backdrop pixels")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--out", required=True, help="folder for crops + depth npys")
    args = ap.parse_args()

    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    H, W = img.shape[:2]
    print(f"image {Path(args.image).name}: {W} x {H}")

    if args.det:
        (cx, cy), poly = load_boll(args.det, args.boll_idx)
    elif args.center:
        cx, cy = args.center
        poly = None
        print(f"centre ({cx:.0f}, {cy:.0f}) from --center, no mask")
    else:
        raise SystemExit("need --det or --center")

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    # full frame first, as the reference
    jobs = [("full", 0, 0, W, H)]
    for s in args.sizes:
        jobs.append((f"crop{s}", *crop_box(cx, cy, s, W, H)))

    rows = []
    for name, x0, y0, x1, y1 in jobs:
        sub = img[y0:y1, x0:x1]
        h, w = sub.shape[:2]
        cpath = out / "crops" / f"{name}.jpg"
        cv2.imwrite(str(cpath), sub)

        pred = model.inference([str(cpath)])
        depth = np.asarray(pred.depth)[0].astype(np.float32)
        np.save(out / "depth" / f"{name}.npy", depth)

        # boll region
        if poly:
            bm = poly_mask(poly, x0, y0, w, h)
            if bm.sum() < 20:
                bm = central_disc(h, w)
        else:
            bm = central_disc(h, w)

        # backdrop = darkest pixels in the crop
        lum = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        thr = np.percentile(lum, args.dark_pct)
        dm = lum <= thr

        b = stats(depth, bm)
        k = stats(depth, dm)
        ratio = (k["med"] / b["med"]) if (b and k and b["med"] > 0) else None
        rows.append((name, w, h, depth.shape, b, k, ratio))

        def fmt(s):
            if not s:
                return "—"
            return "med=%.3f iqr=%.3f n=%d" % (s["med"], s["iqr"], s["n"])

        print("\n[%s] crop %dx%d -> depth %dx%d"
              % (name, w, h, depth.shape[1], depth.shape[0]))
        print("    boll     : %s" % fmt(b))
        print("    backdrop : %s" % fmt(k))
        print("    backdrop/boll ratio : %s"
              % ("—" if ratio is None else "%.3f" % ratio))

    # summary
    print("\n" + "=" * 68)
    print(f"{'window':>10} {'boll med':>10} {'backdrop':>10} {'ratio':>8} {'boll/full':>10}")
    print("=" * 68)
    full_boll = next((r[4]["med"] for r in rows if r[0] == "full" and r[4]), None)
    for name, w, h, dshape, b, k, ratio in rows:
        rel = (b["med"] / full_boll) if (b and full_boll) else None
        print(f"{name:>10} "
              f"{('%.3f' % b['med']) if b else '—':>10} "
              f"{('%.3f' % k['med']) if k else '—':>10} "
              f"{('%.3f' % ratio) if ratio is not None else '—':>8} "
              f"{('%.3f' % rel) if rel is not None else '—':>10}")
    print("=" * 68)
    print("boll/full ~1 across sizes  -> absolute depth survives cropping")
    print("boll/full drifts, ratio stable -> only the SCALE changes; rescale via")
    print("                                  the backdrop or the full frame")
    print("ratio also drifts          -> cropping changes the inferred geometry;")
    print("                                  crops are not directly comparable")


if __name__ == "__main__":
    main()
