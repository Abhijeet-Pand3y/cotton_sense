#!/usr/bin/env python
"""Apple Depth Pro over a frame folder, written in the same layout as run_da3_depth.py.

Drop-in alternative to DA3 for the calibration experiments: same --crop_top_frac /
--crop_bottom_frac semantics, same out/depth/*.npy + crop_info.json, so
fit_gamma_from_annotations.py and the viewer read it with no changes.

The difference that matters: Depth Pro claims METRIC depth in metres and
estimates its own focal length, so if that holds there is nothing to fit --
    size_mm = tb_px * Z_m * 1000 / f_px
which is the general model with gamma = 1.0 and exp(C) = 1000/f_px. Fitting it
with fit_gamma_from_annotations.py is therefore a direct test: a truly metric
depth should come back gamma ~= 1.0, and exp(C) should match 1000/focal.

Per-frame focal estimates are recorded in focal_px.json -- Depth Pro infers focal
from field of view, and a CROPPED image has a smaller FOV at the same true focal,
so watch for a systematic shift there before trusting the metric scale.

Needs the depthpro env:
  /home/abhpande/miniconda3/envs/depthpro/bin/python scripts/depth/run_depthpro_depth.py \
      --images_dir data/images/right_down --out depth_out/right_down_dp_crop25 \
      --crop_top_frac 0.25
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

import cv2
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint",
                    default="third_party/ml-depth-pro/checkpoints/depth_pro.pt")
    ap.add_argument("--crop_top_frac", type=float, default=0.0)
    ap.add_argument("--crop_bottom_frac", type=float, default=0.0)
    ap.add_argument("--max_height", type=int, default=504,
                    help="downsample saved maps to this height (504 = what DA3 "
                         "writes, so the two are compared at equal resolution); "
                         "0 keeps full resolution")
    ap.add_argument("--focal_px", type=float, default=None,
                    help="override Depth Pro's own focal estimate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join("third_party", "ml-depth-pro", "src"))
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    import dataclasses
    cfg = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT,
                              checkpoint_uri=os.path.abspath(a.checkpoint))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(
        config=cfg, device=dev, precision=torch.half)
    model.eval()
    print(f"Depth Pro loaded on {dev}")

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(a.images_dir).iterdir() if p.suffix.lower() in exts)
    if a.limit:
        paths = paths[:a.limit]
    out = Path(a.out); (out / "depth").mkdir(parents=True, exist_ok=True)
    print(f"Found {len(paths)} images in {a.images_dir}")

    crop_written = False
    focals = {}
    for i, p in enumerate(paths, 1):
        stem = p.stem
        dst = out / "depth" / f"{stem}.npy"
        if a.skip_existing and dst.exists():
            continue
        # cv2, not PIL: it applies the EXIF orientation, so these frames come in
        # portrait exactly as run_da3_depth.py sees them. PIL leaves them
        # landscape and the crop then takes a slice off the wrong axis.
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  [warn] could not read {p}")
            continue
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        oh, ow = im.shape[:2]
        y_off = int(round(oh * a.crop_top_frac))
        y_bot = int(round(oh * a.crop_bottom_frac))
        if y_off or y_bot:
            im = im[y_off:oh - y_bot] if y_bot else im[y_off:]
            if not crop_written:
                blob = json.dumps({
                    "crop_top_frac": a.crop_top_frac, "crop_top_px": y_off,
                    "crop_bottom_frac": a.crop_bottom_frac, "crop_bottom_px": y_bot,
                    "orig_width": ow, "orig_height": oh,
                    "cropped_width": im.shape[1], "cropped_height": im.shape[0],
                }, indent=2)
                (out / "crop_info.json").write_text(blob)
                (out / "depth" / "crop_info.json").write_text(blob)
                print(f"[crop] removing top {y_off}px + bottom {y_bot}px of {oh} "
                      f"-> {im.shape[1]}x{im.shape[0]}; wrote crop_info.json")
                crop_written = True

        with torch.no_grad():
            pred = model.infer(transform(im), f_px=a.focal_px)
        depth = pred["depth"].detach().cpu().numpy().astype(np.float32)   # metres
        f_px = float(pred["focallength_px"].detach().cpu())
        focals[stem] = f_px

        if a.max_height and depth.shape[0] > a.max_height:
            s = a.max_height / depth.shape[0]
            depth = cv2.resize(depth, (max(1, int(round(depth.shape[1] * s))), a.max_height),
                               interpolation=cv2.INTER_AREA)
        np.save(dst, depth)
        if i % 20 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {stem}: min={depth.min():.3f} "
                  f"med={np.median(depth):.3f} max={depth.max():.3f} m  f={f_px:.0f}px")

    if focals:
        v = np.array(list(focals.values()))
        json.dump({"per_frame": focals,
                   "median_px": float(np.median(v)), "std_px": float(v.std())},
                  open(out / "focal_px.json", "w"), indent=1)
        print(f"\nDepth Pro focal estimate: median {np.median(v):.0f}px "
              f"(sd {v.std():.0f}) over {len(v)} frames -> focal_px.json")
    print(f"\nDone. Metric depth (.npy float32, METRES): {out/'depth'}")


if __name__ == "__main__":
    main()
