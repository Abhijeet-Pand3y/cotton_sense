#!/usr/bin/env python
"""Convert raw DA3 maps into viewer-ready metric depth using a fitted calibration.

The viewer reads float32 .npy as METRES and applies no gamma of its own, so the
correction has to be baked into the maps:

    size_mm = tb_px * z_raw**(1/gamma) * exp(C)        (fitted)
    size_mm = tb_px * Z_mm / f                         (pinhole)
 => Z_mm    = f * exp(C) * z_raw**(1/gamma)

f cancels between the two, so any consistent focal works as long as the SAME
value goes into metadata.json -- which is what the viewer divides by.

    python scripts/depth/make_viewer_metric_depth.py \
        --src inference_results/LeftDown_v3_soft/depth_trimmed_crop15 \
        --dst inference_results/LeftDown_v3_soft/depth_trimmed_metric
"""
import argparse, json, os, shutil, glob
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding depth/ of raw DA3 maps")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--calib", default="findings/cotton_tag_calibration/calibration.json")
    ap.add_argument("--focal_px", type=float, default=2905.6)
    ap.add_argument("--width", type=int, default=2988)
    ap.add_argument("--height", type=int, default=5312)
    a = ap.parse_args()

    cal = json.load(open(a.calib))
    gamma, scale = cal["gamma"], cal["scale"]
    k = a.focal_px * scale / 1000.0          # metres per z_raw**(1/gamma)

    src_depth = os.path.join(a.src, "depth")
    dst_depth = os.path.join(a.dst, "depth")
    os.makedirs(dst_depth, exist_ok=True)

    n = 0
    for p in sorted(glob.glob(os.path.join(src_depth, "*.npy"))):
        z = np.load(p).astype(np.float64)
        Z = (k * np.power(np.clip(z, 1e-6, None), 1.0 / gamma)).astype(np.float32)
        np.save(os.path.join(dst_depth, os.path.basename(p)), Z)
        n += 1

    for side in ("crop_info.json",):
        for base in (a.src, src_depth):
            s = os.path.join(base, side)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(a.dst, side))
                shutil.copy2(s, os.path.join(dst_depth, side))
                break

    meta = {
        "intrinsics": {"fx": a.focal_px, "fy": a.focal_px,
                       "ppx": a.width / 2.0, "ppy": a.height / 2.0,
                       "width": a.width, "height": a.height},
        "depth_scale": 1.0,
        "units": "meters",
        "note": ("float32 metres, already gamma-corrected: "
                 f"Z_m = {k:.6f} * z_raw**(1/{gamma}). The viewer must NOT apply "
                 "any further correction. focal_px cancels between this file and "
                 "the maps, so sizes are unaffected by its exact value."),
        "calibration": dict(cal, focal_px=a.focal_px,
                            method=f"Z_m = focal_px * exp(C) * z_raw**(1/gamma) / 1000"),
    }
    for d in (a.dst, dst_depth):
        json.dump(meta, open(os.path.join(d, "metadata.json"), "w"), indent=1)

    print(f"converted {n} maps -> {dst_depth}")
    print(f"Z_m = {k:.6f} * z_raw**(1/{gamma})   (gamma={gamma}, scale={scale:.6f})")


if __name__ == "__main__":
    main()
