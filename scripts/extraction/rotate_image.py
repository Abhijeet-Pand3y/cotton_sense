#!/usr/bin/env python3
"""
rotate_images.py
Rotate all images in a folder by 90°, 180°, or 270° and overwrite the originals.

Usage:
    python rotate_images.py <folder_path> <angle>

Arguments:
    folder_path  - Path to the folder containing images
    angle        - Rotation angle: 90 (clockwise), 180, or 270 (counter-clockwise)

Example:
    python rotate_image.py ./my_photos 90
    python rotate_image.py ./my_photos 180
    python rotate_image.py ./my_photos 270
"""

import sys
from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def rotate_images(folder_path: str, angle: int):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    if angle not in (90, 180, 270):
        print(f"Error: Angle must be 90, 180, or 270. Got {angle}.")
        sys.exit(1)

    # PIL's rotate is counter-clockwise, so we invert for intuitive clockwise input.
    # 180 is symmetric so no change needed.
    pil_angle = 360 - angle if angle != 180 else 180

    image_files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"No supported image files found in '{folder_path}'.")
        return

    print(f"Found {len(image_files)} image(s). Rotating {angle}°...\n")

    success, failed = 0, 0

    for img_path in sorted(image_files):
        try:
            with Image.open(img_path) as img:
                rotated = img.rotate(pil_angle, expand=True)
                rotated.save(img_path)
            print(f"  ✓ {img_path.name}")
            success += 1
        except Exception as e:
            print(f"  ✗ {img_path.name} — {e}")
            failed += 1

    print(f"\nDone! {success} rotated, {failed} failed.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    folder_arg = sys.argv[1]
    try:
        angle_arg = int(sys.argv[2])
    except ValueError:
        print("Error: Angle must be an integer (90, 180, or 270).")
        sys.exit(1)

    rotate_images(folder_arg, angle_arg)