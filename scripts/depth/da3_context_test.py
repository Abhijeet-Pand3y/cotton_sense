"""
da3_context_test.py — does what else is in frame change DA3's depth?

THE QUESTION
  Monocular depth is scale-ambiguous: the network infers absolute distance from
  context. So the same object, shot from the same place, may be reported at
  different distances depending on what else is visible. This measures that
  directly, and says how field images should be framed.

THE REFERENCE
  Every shot contains a ping-pong ball (40 mm regulation diameter). Because a
  sphere looks the same from any angle, its pixel diameter gives an independent
  distance via the pinhole relation:
        Z_true = f_px * 40 mm / diameter_px
  That is the yardstick DA3 is scored against. Without it you would only learn
  that two shots disagree, not which one is right.

  Note this needs a focal length. If --focal_px is not given, the ball still
  works as a RELATIVE reference: shots are compared by the ratio
  (DA3 depth) / (1 / diameter_px), which is constant if DA3 is behaving.

WHAT YOU PROVIDE
  A folder of images and, for each, where the ball is. Either:
    - a circles.json  { "shot1.jpg": {"cx": 812, "cy": 640, "r": 47}, ... }
    - or --auto to detect the ball by Hough circle (verify the overlay before
      trusting it)

USAGE
  python da3_context_test.py --images_dir data/context_test \
      --circles data/context_test/circles.json --focal_px 2450 \
      --out context_out/

  python da3_context_test.py --images_dir data/context_test --auto --out context_out/

OUTPUT
  per shot : ball depth from DA3, ball depth from its pixel size, ratio,
             frame depth range, near/far percentiles
  summary  : how much the DA3 ball depth moves between shots, which is the
             number that answers the question
"""

import argparse
import json
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BALL_MM = 40.0            # regulation ping-pong ball diameter


def detect_ball(bgr, hp):
    """Hough-circle guess at the ball. VERIFY the overlay before trusting it.

    Hough votes on edge geometry alone and ignores colour, so any round edge
    pattern competes with the ball — a pot rim, a leaf curl, a highlight. All
    candidates are returned ranked so the caller can pick, and the chosen radius
    feeds straight into the distance estimate, where a few pixels of error is a
    few percent of distance.
    """
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, hp["blur"])
    h, w = g.shape
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=hp["dp"],
                               minDist=max(8, min(h, w) // 4),
                               param1=hp["p1"], param2=hp["p2"],
                               minRadius=int(min(h, w) * hp["rmin"]),
                               maxRadius=int(min(h, w) * hp["rmax"]))
    if circles is None:
        return None, []
    c = np.round(circles[0]).astype(int)
    if hp["pick"] == "brightest":
        # The ball is white; score candidates by mean intensity inside them.
        def score(t):
            x, y, r = t
            y0, y1 = max(0, y - r), min(h, y + r)
            x0, x1 = max(0, x - r), min(w, x + r)
            patch = g[y0:y1, x0:x1]
            return float(patch.mean()) if patch.size else 0.0
        ranked = sorted(c, key=score, reverse=True)
    elif hp["pick"] == "center":
        ranked = sorted(c, key=lambda t: (t[0] - w / 2) ** 2 + (t[1] - h / 2) ** 2)
    else:
        ranked = sorted(c, key=lambda t: -t[2])          # largest
    b = ranked[hp["nth"]] if hp["nth"] < len(ranked) else ranked[0]
    alts = [{"cx": int(t[0]), "cy": int(t[1]), "r": int(t[2])} for t in ranked[:5]]
    return {"cx": int(b[0]), "cy": int(b[1]), "r": int(b[2])}, alts


def disc_mask(shape, cx, cy, r, frac=0.6):
    """Inner disc of the ball, shrunk so the silhouette edge is excluded."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = max(1.0, r * frac)
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr


def stats(depth, mask):
    import cv2
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    v = depth[mask & np.isfinite(depth) & (depth > 0)]
    if v.size < 5:
        return None
    return {"med": float(np.median(v)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--circles", help="JSON of {filename: {cx, cy, r}} in image px")
    ap.add_argument("--auto", action="store_true",
                    help="detect the ball with Hough circles instead")
    ap.add_argument("--focal_px", type=float, default=None,
                    help="focal length in px; enables an absolute check")
    ap.add_argument("--ball_mm", type=float, default=BALL_MM)
    ap.add_argument("--crop_top_frac", type=float, default=0.0,
                    help="drop this fraction from the top before inference")
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    # Hough tuning — the defaults are a starting point, not a calibration.
    ap.add_argument("--hough_dp", type=float, default=1.5)
    ap.add_argument("--hough_p1", type=float, default=120)
    ap.add_argument("--hough_p2", type=float, default=40,
                    help="lower = more circles found, more false ones")
    ap.add_argument("--hough_blur", type=int, default=5, help="median blur, odd")
    ap.add_argument("--rmin_frac", type=float, default=0.01)
    ap.add_argument("--rmax_frac", type=float, default=0.20)
    ap.add_argument("--pick", default="brightest",
                    choices=["brightest", "largest", "center"],
                    help="which candidate to take; the ball is white, so "
                         "'brightest' usually beats 'largest'")
    ap.add_argument("--nth", type=int, default=0,
                    help="take the Nth ranked candidate (0 = first)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hp = {"dp": args.hough_dp, "p1": args.hough_p1, "p2": args.hough_p2,
          "blur": args.hough_blur | 1, "rmin": args.rmin_frac,
          "rmax": args.rmax_frac, "pick": args.pick, "nth": args.nth}

    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3

    paths = sorted(p for p in Path(args.images_dir).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images in {args.images_dir}")

    circles = {}
    if args.circles:
        circles = json.loads(Path(args.circles).read_text())
    elif not args.auto:
        raise SystemExit("give --circles or --auto")

    out = Path(args.out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "check").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] unreadable: {p.name}")
            continue
        y_off = int(round(img.shape[0] * args.crop_top_frac)) if args.crop_top_frac else 0
        if y_off:
            img = img[y_off:]
        H, W = img.shape[:2]

        c = circles.get(p.name)
        alts = []
        if c is None and args.auto:
            c, alts = detect_ball(img, hp)
            if alts and len(alts) > 1:
                print(f"    {p.name}: {len(alts)} candidates, radii "
                      + ", ".join(str(a["r"]) for a in alts))
        if c is None:
            print(f"  [warn] no ball for {p.name}; skipping")
            continue
        cx, cy, r = float(c["cx"]), float(c["cy"]) - y_off, float(c["r"])

        tmp = out / "depth" / f"_{p.stem}.jpg"
        cv2.imwrite(str(tmp), img)
        pred = model.inference([str(tmp)])
        depth = np.asarray(pred.depth)[0].astype(np.float32)
        np.save(out / "depth" / f"{p.stem}.npy", depth)

        # ball depth from DA3
        sx, sy = depth.shape[1] / W, depth.shape[0] / H
        dm = disc_mask(depth.shape, cx * sx, cy * sy, r * (sx + sy) / 2)
        sb = stats(depth, dm)

        # independent distance from the ball's pixel size
        z_geo = (args.focal_px * args.ball_mm / 1000.0 / (2 * r)) if args.focal_px else None

        fin = depth[np.isfinite(depth) & (depth > 0)]
        rows.append({
            "shot": p.stem, "w": W, "h": H,
            "cx": cx, "cy": cy, "r": r,
            "z_da3": sb["med"] if sb else None,
            "iqr": sb["iqr"] if sb else None,
            "z_geo": z_geo,
            "d_min": float(np.percentile(fin, 1)) if fin.size else None,
            "d_med": float(np.median(fin)) if fin.size else None,
            "d_max": float(np.percentile(fin, 99)) if fin.size else None,
        })

        # overlay so the ball placement can be eyeballed
        vis = img.copy()
        th = max(2, W // 500)
        for a in alts[1:]:
            cv2.circle(vis, (int(a["cx"]), int(a["cy"]) - y_off), int(a["r"]),
                       (160, 160, 160), max(1, th - 1))    # rejected, grey
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), th)
        cv2.circle(vis, (int(cx), int(cy)), int(r * 0.6), (0, 255, 0), max(1, th - 1))
        cv2.imwrite(str(out / "check" / f"{p.stem}.jpg"), vis)

        print(f"  {p.stem:28s} ball r={r:5.1f}px  z_da3="
              f"{(sb['med'] if sb else float('nan')):.3f}"
              + (f"  z_geo={z_geo:.3f}" if z_geo else "")
              + f"  range {rows[-1]['d_min']:.2f}-{rows[-1]['d_max']:.2f}")

    if not rows:
        raise SystemExit("nothing measured")

    print("\n" + "=" * 92)
    print(f"{'shot':<26}{'ball r px':>10}{'z_da3':>9}{'z_geo':>9}"
          f"{'ratio':>8}{'frame range':>16}{'ball IQR':>10}")
    print("=" * 92)
    for r_ in rows:
        ratio = (r_["z_da3"] / r_["z_geo"]) if (r_["z_da3"] and r_["z_geo"]) else None
        z_geo_s = "%.3f" % r_["z_geo"] if r_["z_geo"] else "—"
        ratio_s = "%.3f" % ratio if ratio else "—"
        rng_s = "%.2f-%.2f" % (r_["d_min"], r_["d_max"])
        print("%-26s%10.1f%9.3f%9s%8s%16s%10.4f"
              % (r_["shot"][:26], r_["r"], r_["z_da3"] or 0, z_geo_s,
                 ratio_s, rng_s, r_["iqr"] or 0))

    z = np.array([r_["z_da3"] for r_ in rows if r_["z_da3"]], dtype=float)
    rp = np.array([r_["r"] for r_ in rows if r_["z_da3"]], dtype=float)
    print("=" * 92)
    print("\nDID THE CAMERA MOVE?")
    print(f"  ball radius across shots : {rp.min():.0f}-{rp.max():.0f} px "
          f"(x{rp.max()/rp.min():.2f})")
    if rp.max() / rp.min() > 1.10:
        print("  The ball changed apparent size, so the camera DID move. Raw depth")
        print("  spread is therefore NOT the context effect — distance changed too.")
        print("  Use the invariant below instead.")
    else:
        print("  Apparent size is near constant, so the camera held still and the")
        print("  raw depth spread below is attributable to context.")

    print("\nSCALE INVARIANT  z_da3 x r_px")
    print("  True distance goes as 1/r, so if DA3 tracked distance correctly this")
    print("  product would be the same in every shot whatever the framing. Its")
    print("  spread is the context effect, and it needs no focal length.")
    inv = z * rp
    print(f"  values : " + "  ".join(f"{v:.1f}" for v in inv))
    print(f"  spread : {100*(inv.max()-inv.min())/inv.mean():.1f}% of mean")
    print("  Under ~10% => framing barely matters. Over ~30% => framing dominates,")
    print("  and field capture must match whatever conditions were calibrated.")
    print(f"\n  (raw DA3 depth alone: min {z.min():.3f} max {z.max():.3f}, "
          f"spread {100*(z.max()-z.min())/z.mean():.1f}% — confounded if the camera moved)")

    geo = [r_ for r_ in rows if r_["z_geo"]]
    if geo:
        ratios = np.array([r_["z_da3"] / r_["z_geo"] for r_ in geo])
        print(f"\nABSOLUTE ACCURACY (ball size as truth)")
        print(f"  z_da3 / z_geo : min {ratios.min():.3f}  med "
              f"{np.median(ratios):.3f}  max {ratios.max():.3f}")
        print("  A ratio far from 1.0 but CONSISTENT is fine — one calibration")
        print("  constant fixes it. A ratio that VARIES between shots is not.")
    else:
        print("\n(no --focal_px, so no absolute check; relative comparison only)")

    with open(out / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out/'results.json'}")
    print(f"check the ball outlines in {out/'check'} before trusting any of this")


if __name__ == "__main__":
    main()
