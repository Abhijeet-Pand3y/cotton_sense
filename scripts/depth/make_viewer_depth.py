"""
make_viewer_depth.py — turn raw DA3 depth into METRIC depth the Inference Viewer
can measure with, plus the metadata.json it needs.

WHY A CONVERSION STEP AT ALL
    Inference_Viewer.html already does the right geometry: it deprojects tip and
    base through the pinhole model and reports a 3-D length. It needs two things
    it cannot invent - camera intrinsics, and depth in METRES.

    Reading the viewer's own rule (line ~682): a float32 .npy is taken as already
    metric and depth_scale is NOT applied. So the cleanest integration is to bake
    the calibration into the maps and write float32 metres. No HTML is modified.

THE SCALE IS PER-SCENE AND MUST BE SET PER-SCENE
    DA3 has no absolute scale; it infers depth from scene content. Measured here:

        ball calibration scene   full-frame depth p99 ~ 1.03
        cotton greenhouse scene  full-frame depth p99 ~ 9.07

    Nearly an order of magnitude apart, because one is a cloth backdrop about a
    metre away and the other is a greenhouse with a roof many metres off. So the
    constant fitted on the balls is NOT transferable to cotton, and pretending
    otherwise is what produced 68-76 mm bolls in findings/boll_length/.

    What DOES transfer:
      - the intrinsics (focal length is a property of the camera, not the scene)
      - the exponent gamma (the SHAPE of the compression)
    What does not:
      - the scale constant a

    So gamma is carried over and a is solved from ONE known length in this
    scene:
        L_true = px * Z / f     and     Z = (z_raw / a)^(1/gamma)
        =>  Z_needed = L_true * f / px
        =>  a = z_raw / Z_needed^gamma

    One measured boll, or one object of known size, calibrates the whole run.

USAGE
    # calibrate from one boll you have physically measured
    python make_viewer_depth.py --run inference_results/LeftDown_v5 \
        --ref_track 77 --ref_length_mm 42

    # or from a known camera-to-object distance
    python make_viewer_depth.py --run ... --ref_track 77 --ref_distance_mm 600

    # or fall back to the ball-scene constant (NOT recommended - see above)
    python make_viewer_depth.py --run ... --use_ball_scale

Then in the viewer: Load frames -> Load detections -> Load depth (pick the
depth_metric/depth folder; metadata.json sits inside it and is auto-detected).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

F_PX = 2905.6          # solved from the 40 mm balls; a camera property
GAMMA = 0.663          # median across ball sets; the shape, assumed transferable


def track_reference(run: Path, track_id: int):
    """Median pixel tip-base length and median raw boll depth for one track."""
    import csv
    p = run.parent.parent / "findings/boll_length/boll_length_per_frame.csv"
    p = Path("findings/boll_length/boll_length_per_frame.csv")
    if not p.exists():
        raise SystemExit("run scripts/depth/boll_length.py first (it supplies the "
                         "per-frame pixel length and boll depth used to calibrate)")
    rows = [x for x in csv.DictReader(open(p)) if str(x["track_id"]) == str(track_id)]
    if not rows:
        raise SystemExit(f"track {track_id} not found in boll_length_per_frame.csv")
    px = float(np.median([float(x["px_tip_base"]) for x in rows]))
    z = float(np.median([float(x["z_boll"]) for x in rows]))
    return px, z, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default <run>/depth_metric")
    ap.add_argument("--focal_px", type=float, default=F_PX)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--ref_track", type=int)
    ap.add_argument("--ref_length_mm", type=float,
                    help="physically measured tip-base length of --ref_track")
    ap.add_argument("--ref_distance_mm", type=float,
                    help="measured camera-to-boll distance for --ref_track")
    ap.add_argument("--scale_a", type=float,
                    help="set the scale constant directly (e.g. one solved "
                         "against measured ground truth)")
    ap.add_argument("--use_ball_scale", action="store_true",
                    help="use the ball-scene constant; wrong scene, see the header")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad_crop", action="store_true",
                    help="depth was inferred on a top-cropped image (crop_info.json). "
                         "Pad the removed strip back so the map lines up with the "
                         "FULL frame the viewer loads, keeping detection "
                         "coordinates valid. The padded strip is 0 = no data.")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out) if args.out else run / "depth_metric"
    src = run / "depth" / "depth"
    if not src.exists():
        raise SystemExit(f"no raw depth maps at {src}")

    g = args.gamma
    if args.scale_a:
        a = args.scale_a
        how = f"scale a supplied directly = {a:.6g}"
    elif args.use_ball_scale:
        from size_regression import load, solve_pinhole
        ball = load(); f_b, c_b = solve_pinhole(ball)
        z = np.array([x["z"] for x in ball]); Z = np.array([x["Ztape"] + c_b for x in ball])
        a = float(np.exp(np.mean(np.log(z) - g * np.log(Z))))
        how = ("ball-scene constant — WRONG SCENE. Absolute millimetres from this "
               "are extrapolated and were shown to read ~1.5x high.")
    elif True:
        if args.ref_track is None or (args.ref_length_mm is None
                                      and args.ref_distance_mm is None):
            raise SystemExit(
                "give --ref_track with either --ref_length_mm (a boll you have "
                "measured) or --ref_distance_mm (a distance you have measured); "
                "or --use_ball_scale to accept the untransferable constant.")
        px, z_ref, n = track_reference(run, args.ref_track)
        if args.ref_distance_mm:
            Z_need = args.ref_distance_mm
            how = (f"track {args.ref_track}: measured distance "
                   f"{args.ref_distance_mm:.0f} mm over {n} frames")
        else:
            Z_need = args.ref_length_mm * args.focal_px / px
            how = (f"track {args.ref_track}: measured length "
                   f"{args.ref_length_mm:.1f} mm, median {px:.1f} px over {n} "
                   f"frames -> implies {Z_need:.0f} mm distance")
        a = z_ref / (Z_need ** g)

    # Z_mm = (z/a)^(1/gamma); metres for the viewer
    print(f"gamma = {g:.3f}   scale a = {a:.6g}")
    print(f"calibration: {how}")

    crop = None
    ci = run / "depth" / "crop_info.json"
    if args.pad_crop:
        if not ci.exists():
            raise SystemExit(f"--pad_crop needs {ci}")
        crop = json.loads(ci.read_text())
        print(f"padding back the top {crop['crop_top_px']}px removed before inference")

    dst = out / "depth"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npy"))
    if args.limit:
        files = files[:args.limit]
    lo, hi = [], []
    for i, p in enumerate(files):
        zr = np.load(p).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            Zm = ((np.maximum(zr, 1e-6) / a) ** (1.0 / g)) / 1000.0   # mm -> m
        Zm = np.where(np.isfinite(Zm), Zm, 0).astype(np.float32)
        if args.pad_crop and crop:
            # Rows removed from the top of the IMAGE must come back as rows at
            # the top of the DEPTH MAP, in the same proportion, or every
            # detection y-coordinate lands on the wrong depth pixel.
            npad = int(round(Zm.shape[0] * crop["crop_top_px"] / crop["cropped_height"]))
            Zm = np.vstack([np.zeros((npad, Zm.shape[1]), np.float32), Zm])
        np.save(dst / p.name, Zm)
        v = Zm[Zm > 0]
        if v.size:
            lo.append(float(np.percentile(v, 1))); hi.append(float(np.percentile(v, 99)))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}")

    # intrinsics from a real frame, so width/height and the principal point match
    import cv2
    fr = sorted((run / "frames").glob("*.jpg"))
    im = cv2.imread(str(fr[0]))
    H, W = im.shape[:2]
    meta = {
        "intrinsics": {"fx": args.focal_px, "fy": args.focal_px,
                       "ppx": W / 2.0, "ppy": H / 2.0, "width": W, "height": H},
        "depth_scale": 1.0,
        "units": "meters",
        "note": ("float32 metres; the viewer uses float32 maps directly and does "
                 "not apply depth_scale. Calibration: " + how),
        "calibration": {"gamma": g, "scale_a": a, "focal_px": args.focal_px,
                        "method": "Z_mm = (z_raw/a)^(1/gamma)"},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(out / "metadata.json", dst / "metadata.json")
    print(f"\nwrote {len(files)} metric maps -> {dst}")
    print(f"wrote metadata.json (fx=fy={args.focal_px:.1f}, "
          f"ppx={W/2:.0f}, ppy={H/2:.0f}) in {out} and {dst}")
    if lo:
        print(f"metric depth p1-p99 across frames: {np.mean(lo):.2f} - "
              f"{np.mean(hi):.2f} m")


if __name__ == "__main__":
    main()
