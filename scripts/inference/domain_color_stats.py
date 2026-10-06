"""
domain_color_stats.py — Aggregate color statistics across a folder of frames.

Computes per-channel mean/std in both RGB and LAB, averaged over a sample of
images. Use it to characterize a domain (gopro training frames vs rgbd frames)
so a color transfer can map one domain's statistics onto the other.

LAB is included because mean/std color transfer (Reinhard et al.) is most robust
in LAB: L=lightness, a=green-red, b=blue-yellow. Matching mean/std per LAB
channel shifts tone/brightness without the channel-coupling artifacts you get
doing it naively in RGB.

USAGE:
    python domain_color_stats.py --dir data/rgbd/frames --sample 200 --out rgbd_stats.json
    python domain_color_stats.py --dir datasets/combined_all_v3/right_down --sample 200 --out gopro_stats.json
"""

import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="Folder of frames (recursive)")
    p.add_argument("--sample", type=int, default=200,
                   help="Randomly sample this many images (0 = all)")
    p.add_argument("--range", nargs=2, type=int, default=None,
                   metavar=("START", "END"),
                   help="Use frames [START:END] by sorted order, BEFORE sampling. "
                        "Use this to skip empty lead-in frames.")
    p.add_argument("--out", default=None, help="Write stats JSON here")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = [q for q in glob.glob(f"{args.dir}/**/*", recursive=True)
             if Path(q).suffix.lower() in IMG_EXTS]
    if not paths:
        print(f"No images found under {args.dir}")
        return
    paths.sort()  # sortable frame names -> capture order

    if args.range:
        lo, hi = args.range
        paths = paths[lo:hi]
        print(f"Range [{lo}:{hi}] -> {len(paths)} frames before sampling")

    random.seed(args.seed)
    if args.sample and len(paths) > args.sample:
        paths = random.sample(paths, args.sample)
    print(f"Sampling {len(paths)} images from {args.dir}")

    # accumulate per-channel sum and sum-of-squares over ALL pixels
    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_sqsum = np.zeros(3, dtype=np.float64)
    lab_sum = np.zeros(3, dtype=np.float64)
    lab_sqsum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    n_ok = 0

    for pth in paths:
        img = cv2.imread(pth)  # BGR
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
        px = rgb.reshape(-1, 3)
        lx = lab.reshape(-1, 3)
        rgb_sum += px.sum(0); rgb_sqsum += (px * px).sum(0)
        lab_sum += lx.sum(0); lab_sqsum += (lx * lx).sum(0)
        n_pixels += px.shape[0]
        n_ok += 1

    if n_pixels == 0:
        print("No readable images.")
        return

    rgb_mean = rgb_sum / n_pixels
    rgb_std = np.sqrt(np.maximum(0, rgb_sqsum / n_pixels - rgb_mean ** 2))
    lab_mean = lab_sum / n_pixels
    lab_std = np.sqrt(np.maximum(0, lab_sqsum / n_pixels - lab_mean ** 2))

    stats = {
        "dir": args.dir,
        "n_images": n_ok,
        "rgb_mean": rgb_mean.tolist(),
        "rgb_std": rgb_std.tolist(),
        "lab_mean": lab_mean.tolist(),
        "lab_std": lab_std.tolist(),
    }

    print(f"  images read : {n_ok}")
    print(f"  RGB mean    : ({rgb_mean[0]:.1f}, {rgb_mean[1]:.1f}, {rgb_mean[2]:.1f})")
    print(f"  RGB std     : ({rgb_std[0]:.1f}, {rgb_std[1]:.1f}, {rgb_std[2]:.1f})")
    print(f"  LAB mean    : ({lab_mean[0]:.1f}, {lab_mean[1]:.1f}, {lab_mean[2]:.1f})")
    print(f"  LAB std     : ({lab_std[0]:.1f}, {lab_std[1]:.1f}, {lab_std[2]:.1f})")
    print(f"  green-vs-(R+B)/2 = {rgb_mean[1] - (rgb_mean[0]+rgb_mean[2])/2:+.1f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
