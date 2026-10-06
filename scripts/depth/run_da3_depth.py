"""
run_da3_depth.py — Depth Anything 3 (metric) runner, output tailored for the
Inference_Viewer's depth overlay.

PURPOSE
  (1) Verify the DA3 install works in a dedicated env.
  (2) Produce per-frame depth maps the viewer can load directly, plus a colored
      visualization for eyeballing. Also validates DA3 metric depth against
      RealSense depth on rgbd frames (RealSense = reference).

OUTPUT FORMAT (matches the viewer's .npy parser)
  The viewer reads 2-D .npy maps, and treats float32 ('<f4') maps as ALREADY IN
  METERS. So we save <stem>.npy as float32 meters, shape (H, W), C-order.
  numpy.save produces exactly this. Without a focal length there is no true metric
  scale, so metric output is only meaningful WITH --focal_px.

MODEL
  DA3METRIC-LARGE (Apache-2.0). Metric meters via the DA3 FAQ formula:
        metric_depth_m = focal_px * net_output / 300.0

MONOCULAR NOTE
  One image per inference call, so frames are never fused as multi-view.

USAGE
  # relative test first (no true scale, viewer 'cm' will be arbitrary):
  python run_da3_depth.py --images_dir <gopro_frames> --out depth_out/gopro --limit 10

  # metric (viewer cm/heights become real) once you know the GoPro focal in px:
  python run_da3_depth.py --images_dir <gopro_frames> --out depth_out/gopro \
      --focal_px 1800 --limit 10

OUTPUTS (per input <stem>)
  <out>/depth/<stem>.npy   -> float32 meters, (H,W)   [load this folder in viewer]
  <out>/viz/<stem>.png     -> colored depth for eyeballing
  prints per-image depth min/median/max (meters if --focal_px, else raw units)

CAVEAT
  Written against the DA3 README API; NOT run here. If a field name differs, the
  error points at the exact call to fix.
"""

import argparse
import glob
from pathlib import Path

import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def colorize(depth, conf=None):
    """Map a depth array to an 8-bit RGB visualization (near=bright)."""
    d = depth.astype(np.float32).copy()
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo = np.percentile(d[finite], 2)
    hi = np.percentile(d[finite], 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    dn = np.clip((d - lo) / (hi - lo), 0, 1)
    # invert so near (small depth) is bright
    dn = 1.0 - dn
    try:
        import cv2
        vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    except Exception:
        g = (dn * 255).astype(np.uint8)
        vis = np.stack([g, g, g], axis=-1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE",
                    help="HF repo or local dir for the DA3 model")
    ap.add_argument("--focal_px", type=float, default=None,
                    help="Camera focal length in px; enables metric meters "
                         "(metric = focal * net_output / 300)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N images (0 = all). Use a small "
                         "number for the first install test.")
    ap.add_argument("--range", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="Process only frames [START:END] by position in the "
                         "sorted list. Use it to drop the background-only lead-in "
                         "and tail of a clip.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip frames whose .npy already exists, so an "
                         "interrupted run can be resumed.")
    ap.add_argument("--crop_bottom_frac", type=float, default=0.0,
                    help="Remove this fraction from the BOTTOM of every frame "
                         "before inference (e.g. 0.10 to cut foreground ground "
                         "that anchors DA3's near depth range). Recorded in "
                         "crop_info.json; it does not shift coordinates, but it "
                         "does change the cropped height downstream sampling "
                         "must scale by.")
    ap.add_argument("--crop_top_frac", type=float, default=0.0,
                    help="Remove this fraction from the TOP of every frame "
                         "before inference (e.g. 0.22 to cut a greenhouse roof "
                         "that field data will not have). The offset is written "
                         "to crop_info.json so downstream sampling can shift "
                         "detection coordinates to match.")
    ap.add_argument("--no_viz", action="store_true",
                    help="Skip writing the colored depth PNGs (saves time/disk)")
    args = ap.parse_args()

    import torch
    try:
        from depth_anything_3.api import DepthAnything3
    except Exception as e:
        print("[FATAL] could not import depth_anything_3. Is DA3 installed in "
              "this env? See https://github.com/ByteDance-Seed/Depth-Anything-3")
        raise

    paths = sorted(q for q in glob.glob(f"{args.images_dir}/**/*", recursive=True)
                   if Path(q).suffix.lower() in IMG_EXTS)
    n_all = len(paths)
    if args.range:
        lo, hi = args.range
        paths = paths[lo:hi]
        print(f"range [{lo}:{hi}] -> {len(paths)} of {n_all} frames")
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"No images under {args.images_dir}")
        return
    print(f"DA3: {len(paths)} images from {args.images_dir}")

    out = Path(args.out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        (out / "viz").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} on {device} ...")
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    try:
        import cv2
        have_cv2 = True
    except Exception:
        have_cv2 = False

    if args.focal_px is None:
        print("[warn] no --focal_px: saving RAW network output as float32. The "
              "viewer will read it as 'meters' but the scale is arbitrary until "
              "you pass a real focal length.")

    n_skipped = 0
    crop_info_written = False
    for i, p in enumerate(paths):
        stem = Path(p).stem
        if args.skip_existing and (out / "depth" / f"{stem}.npy").exists():
            n_skipped += 1
            continue

        infer_path = p
        if args.crop_top_frac > 0 or args.crop_bottom_frac > 0:
            if not have_cv2:
                raise SystemExit("--crop_top_frac / --crop_bottom_frac need "
                                 "opencv (pip install opencv-python)")
            im = cv2.imread(p)
            if im is None:
                print(f"  [warn] could not read {p}")
                continue
            oh, ow = im.shape[:2]
            y_off = int(round(oh * args.crop_top_frac))
            y_bot = int(round(oh * args.crop_bottom_frac))
            im = im[y_off:oh - y_bot] if y_bot else im[y_off:]
            tmp = out / "_cropped"
            tmp.mkdir(exist_ok=True)
            infer_path = str(tmp / f"{stem}.jpg")
            cv2.imwrite(infer_path, im)
            if not crop_info_written:
                import json as _json
                blob = _json.dumps({
                    "crop_top_frac": args.crop_top_frac,
                    "crop_top_px": y_off,
                    "crop_bottom_frac": args.crop_bottom_frac,
                    "crop_bottom_px": y_bot,
                    "orig_width": ow, "orig_height": oh,
                    "cropped_width": im.shape[1], "cropped_height": im.shape[0],
                }, indent=2)
                # Written in BOTH places on purpose: the run folder is where
                # sample_boll_depth.py looks, and inside depth/ so that selecting
                # just that folder in the viewer still carries the offset.
                (out / "crop_info.json").write_text(blob)
                (out / "depth" / "crop_info.json").write_text(blob)
                print(f"[crop] removing top {y_off}px + bottom {y_bot}px of {oh} "
                      f"-> {im.shape[1]}x{im.shape[0]}; wrote crop_info.json "
                      f"(run folder + depth/)")
                crop_info_written = True

        # ONE image per call -> true monocular, no multi-view fusion
        pred = model.inference([infer_path])

        depth = np.asarray(pred.depth)[0].astype(np.float32)   # (H,W) raw net output

        # metric scaling per DA3 FAQ; without focal this stays raw units
        if args.focal_px is not None:
            depth_out = (args.focal_px * depth / 300.0).astype(np.float32)
            units = "m"
        else:
            depth_out = depth.astype(np.float32)
            units = "raw"

        # report focal if the model estimated intrinsics (informational)
        est_focal = None
        if getattr(pred, "intrinsics", None) is not None:
            try:
                K = np.asarray(pred.intrinsics)[0]
                est_focal = float((K[0, 0] + K[1, 1]) / 2)
            except Exception:
                est_focal = None

        # viewer expects 2-D float32 ('<f4') meters, C-order -> np.save does this
        depth_out = np.ascontiguousarray(depth_out)
        np.save(out / "depth" / f"{stem}.npy", depth_out)

        if not args.no_viz and have_cv2:
            vis = colorize(depth_out)
            cv2.imwrite(str(out / "viz" / f"{stem}.png"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        fin = depth_out[np.isfinite(depth_out)]
        stats = (f"min={fin.min():.3f} med={np.median(fin):.3f} "
                 f"max={fin.max():.3f} {units}") if fin.size else "no finite depth"
        extra = f" est_focal~{est_focal:.0f}px" if est_focal else ""
        print(f"  [{i+1}/{len(paths)}] {stem}: {stats}{extra}")

    print(f"\nDone. Depth maps (.npy float32): {out/'depth'}")
    if n_skipped:
        print(f"skipped {n_skipped} frame(s) that already had a .npy")
    print("Load the depth/ folder in the Inference Viewer's 'Load depth' button.")
    if args.focal_px is None:
        print("REMINDER: pass --focal_px for the viewer's cm/height numbers to be "
              "real meters.")


if __name__ == "__main__":
    main()
