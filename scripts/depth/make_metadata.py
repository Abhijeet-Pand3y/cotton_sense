"""
make_metadata.py — write the metadata.json the Inference Viewer needs to compute
3-D tip<->base distances from a depth map.

WHY
  The viewer deprojects tip/base pixels to 3-D with a pinhole model:
      X = (u - ppx) * Z / fx      Y = (v - ppy) * Z / fy      Z = depth
  Without intrinsics it cannot do this and the 3-D size stays "-". RealSense runs
  ship intrinsics; monocular depth (Depth Anything, Depth Pro) does not, so you
  supply them here.

UNITS THAT MATTER
  fx/fy/ppx/ppy must be in the pixel units of the IMAGE (the frames you loaded),
  NOT of the depth map. Detection coords (kpts, mask_poly) are image pixels, and
  that is what the viewer feeds the pinhole equations. If the depth map is a
  different resolution the viewer rescales lookups itself.

FOCAL LENGTH
  fx_px = (image_width_px / 2) / tan(hfov_rad / 2)
  or, from a sensor spec:  fx_px = focal_mm * image_width_px / sensor_width_mm
  Pass whichever you have: --focal_px, --hfov_deg, or --focal_mm + --sensor_mm.

USAGE
  # from a known horizontal FOV
  python make_metadata.py --width 1080 --height 1920 --hfov_deg 90 \
      --out depth_out/gopro_test/metadata.json

  # from a focal length already in pixels
  python make_metadata.py --width 1080 --height 1920 --focal_px 540 \
      --out depth_out/gopro_test/metadata.json

Put metadata.json next to the depth/ folder (or anywhere in the run folder) and
load the run folder in the viewer — it auto-detects the file.

CAVEATS
  * A pinhole model ignores lens distortion. Action-cam lenses are strongly
    barrel-distorted, so off-centre measurements carry extra error unless the
    frames were undistorted first.
  * If the frames were cropped or resized after capture, the focal length in
    pixels scales by the same factor. Compute fx for the frames you actually
    loaded, not the original sensor output.
  * These intrinsics only fix the geometry. If the depth values themselves are
    not true metres, the distances will be wrong by that same scale factor.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True,
                    help="Image width in px (of the frames you load in the viewer)")
    ap.add_argument("--height", type=int, required=True, help="Image height in px")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--focal_px", type=float, help="Focal length in pixels")
    g.add_argument("--hfov_deg", type=float, help="Horizontal field of view, degrees")
    g.add_argument("--focal_mm", type=float, help="Focal length in mm (needs --sensor_mm)")
    ap.add_argument("--sensor_mm", type=float,
                    help="Sensor width in mm (with --focal_mm)")
    ap.add_argument("--ppx", type=float, default=None,
                    help="Principal point x (default: image centre)")
    ap.add_argument("--ppy", type=float, default=None,
                    help="Principal point y (default: image centre)")
    ap.add_argument("--depth_scale", type=float, default=1.0,
                    help="Metres per raw depth unit. Ignored for float32 (<f4) "
                         "maps, which the viewer already treats as metres, but "
                         "the field must be present for auto-detection.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.focal_px is not None:
        fx = args.focal_px
    elif args.hfov_deg is not None:
        fx = (args.width / 2.0) / math.tan(math.radians(args.hfov_deg) / 2.0)
    else:
        if not args.sensor_mm:
            ap.error("--focal_mm requires --sensor_mm")
        fx = args.focal_mm * args.width / args.sensor_mm
    fy = fx  # square pixels assumed

    ppx = args.ppx if args.ppx is not None else args.width / 2.0
    ppy = args.ppy if args.ppy is not None else args.height / 2.0

    meta = {
        "intrinsics": {"fx": fx, "fy": fy, "ppx": ppx, "ppy": ppy,
                       "width": args.width, "height": args.height},
        "depth_scale": args.depth_scale,
        "note": "intrinsics are in IMAGE pixel units; pinhole model, no distortion",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))

    hfov = 2 * math.degrees(math.atan((args.width / 2.0) / fx))
    vfov = 2 * math.degrees(math.atan((args.height / 2.0) / fy))
    print(f"wrote {out}")
    print(f"  image     : {args.width} x {args.height}")
    print(f"  fx = fy   : {fx:.2f} px")
    print(f"  ppx, ppy  : {ppx:.1f}, {ppy:.1f}")
    print(f"  implied   : HFOV {hfov:.1f} deg, VFOV {vfov:.1f} deg")
    print("  sanity-check those FOV numbers against the lens before trusting "
          "any measurement.")


if __name__ == "__main__":
    main()
