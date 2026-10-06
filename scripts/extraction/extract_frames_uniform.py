"""
Cotton Boll Frame Extraction - Uniform Interval (for Track Mode)
=================================================================
Extracts frames at uniform time intervals for CVAT Track annotation.
Unlike the diversity-based extractor, this produces evenly spaced frames
so polygon interpolation between keyframes is accurate.

Usage:
    python extract_frames_uniform.py --video gopro.MP4 -o ./frames -n 200 \
        --start 21 --end 43 --rotate 270

    # Or specify interval directly instead of frame count:
    python extract_frames_uniform.py --video gopro.MP4 -o ./frames \
        --start 21 --end 43 --interval 0.1 --rotate 270
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path


def extract_uniform_frames(video_path, output_dir, num_frames=None,
                           interval_sec=None, start_sec=0, end_sec=None,
                           rotate=None, min_sharpness=0):
    """
    Extract frames at uniform time intervals.

    Either num_frames or interval_sec must be provided.
    If both given, interval_sec takes priority.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    # Time range
    start_frame = int(start_sec * fps)
    if end_sec is not None:
        end_frame = min(int(end_sec * fps), total_frames)
    else:
        end_frame = total_frames

    usable_duration = (end_frame - start_frame) / fps

    # Compute interval
    if interval_sec is not None:
        actual_interval = interval_sec
        actual_num_frames = int(usable_duration / interval_sec) + 1
    elif num_frames is not None:
        actual_interval = usable_duration / max(num_frames - 1, 1)
        actual_num_frames = num_frames
    else:
        raise ValueError("Must provide either --num_frames or --interval")

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total duration: {duration:.1f}s")
    print(f"  Using: {start_sec:.1f}s - {end_sec if end_sec else duration:.1f}s "
          f"({usable_duration:.1f}s)")
    print(f"  Frames to extract: {actual_num_frames}")
    print(f"  Interval: {actual_interval:.3f}s ({1/actual_interval:.1f} fps)")
    if rotate:
        print(f"  Rotation: {rotate} degrees")
    print()

    # Generate uniform timestamps
    timestamps = [start_sec + i * actual_interval for i in range(actual_num_frames)]
    timestamps = [t for t in timestamps if t <= (end_sec if end_sec else duration)]

    # Rotation setup
    rotate_code = None
    if rotate == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate == 180:
        rotate_code = cv2.ROTATE_180
    elif rotate == 270 or rotate == -90:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    # Extract and save
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    skipped = 0

    print(f"Extracting {len(timestamps)} frames...")

    for i, t in enumerate(timestamps):
        frame_idx = int(t * fps)
        frame_idx = min(frame_idx, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            skipped += 1
            continue

        # Optional sharpness filter
        if min_sharpness > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < min_sharpness:
                skipped += 1
                continue
        else:
            sharpness = 0

        # Rotate
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # Save
        timestamp_str = f"{t:.2f}".replace(".", "p")
        filename = f"frame_{frame_idx:06d}_t{timestamp_str}s.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved.append({
            "filename": filename,
            "frame_idx": frame_idx,
            "timestamp": round(t, 3),
            "sharpness": round(sharpness, 1)
        })

        if (i + 1) % 50 == 0:
            print(f"  Saved {i+1}/{len(timestamps)}...")

    cap.release()

    # Save metadata
    metadata = {
        "video": os.path.basename(video_path),
        "video_resolution": f"{width}x{height}",
        "video_fps": fps,
        "video_total_frames": total_frames,
        "video_duration_sec": round(duration, 2),
        "time_range": f"{start_sec}s - {end_sec if end_sec else duration}s",
        "interval_sec": round(actual_interval, 4),
        "effective_fps": round(1 / actual_interval, 2),
        "rotation": rotate if rotate else 0,
        "output_resolution": f"{height}x{width}" if rotate in (90, 270, -90) else f"{width}x{height}",
        "num_extracted": len(saved),
        "num_skipped": skipped,
        "frames": saved
    }

    meta_path = os.path.join(output_dir, "extraction_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Frames saved: {len(saved)} (skipped: {skipped})")
    print(f"  Directory: {output_dir}/")
    print(f"  Metadata: {meta_path}")
    print(f"  Interval: {actual_interval:.3f}s (every {actual_interval*fps:.1f} video frames)")
    print(f"  Timestamp range: {saved[0]['timestamp']:.2f}s - {saved[-1]['timestamp']:.2f}s")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames at uniform intervals for Track annotation"
    )
    parser.add_argument("--video", "-v", required=True,
                        help="Path to input video file")
    parser.add_argument("--output", "-o", default="./frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--num_frames", "-n", type=int, default=None,
                        help="Number of frames to extract (evenly spaced)")
    parser.add_argument("--interval", "-i", type=float, default=None,
                        help="Time interval between frames in seconds "
                             "(overrides --num_frames)")
    parser.add_argument("--start", "-s", type=float, default=0,
                        help="Start time in seconds")
    parser.add_argument("--end", "-e", type=float, default=None,
                        help="End time in seconds")
    parser.add_argument("--rotate", "-r", type=int, default=None,
                        choices=[90, 180, 270, -90],
                        help="Rotate frames: 90 (CW), 270/-90 (CCW), 180")
    parser.add_argument("--min_sharpness", type=float, default=0,
                        help="Minimum sharpness threshold (default: 0, disabled)")

    args = parser.parse_args()

    if args.num_frames is None and args.interval is None:
        parser.error("Must provide either --num_frames or --interval")

    extract_uniform_frames(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num_frames,
        interval_sec=args.interval,
        start_sec=args.start,
        end_sec=args.end,
        rotate=args.rotate,
        min_sharpness=args.min_sharpness
    )


if __name__ == "__main__":
    main()
