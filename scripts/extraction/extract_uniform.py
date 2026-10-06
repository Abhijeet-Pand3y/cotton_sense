"""
extract_uniform.py — take every Nth frame from a video, in order, no selection.

Deliberately dumb. The sharpness/diversity extractor picks frames it judges
"good", which biases which object positions survive — fatal when the object's
position IS the variable under study. This keeps a regular sample of everything.

Frame names carry the ORIGINAL frame index and timestamp, so a frame can always
be traced back to a moment in the clip.

USAGE
  python extract_uniform.py --video clip.MP4 --out frames/clip --every 5
  python extract_uniform.py --video clip.MP4 --out frames/clip --every 5 \
      --start 2 --end 40            # seconds, to trim the ends
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--start", type=float, default=0.0, help="skip before this (s)")
    ap.add_argument("--end", type=float, default=-1.0, help="stop after this (s)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0)
    args = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0 = int(args.start * fps)
    f1 = int(args.end * fps) if args.end > 0 else total
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{Path(args.video).name}: {w}x{h}  {fps:.2f} fps  {total} frames")
    print(f"every {args.every}, frames {f0}-{f1}  -> ~{max(0, f1 - f0)//args.every}")

    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(args.rotate)
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f0 <= i < f1 and (i - f0) % args.every == 0:
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            t = i / fps
            name = f"frame_{i:06d}_t{int(t)}p{int(round((t % 1) * 100)):02d}s.jpg"
            cv2.imwrite(str(out / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            n += 1
            if n % 50 == 0:
                print(f"  {n} written (at frame {i})")
        i += 1
    cap.release()
    print(f"wrote {n} frames -> {out}")


if __name__ == "__main__":
    main()
