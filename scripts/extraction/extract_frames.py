"""
Cotton Boll Annotation Pipeline - Frame Extraction
====================================================
Extracts diverse frames from GoPro video for annotation.
Uses multiple strategies to ensure diversity:
  - Scene change detection (content diversity)
  - Uniform sampling (temporal coverage)
  - Brightness/blur filtering (quality control)

Usage:
    python extract_frames.py --video <path_to_video> --output ./frames --num_frames 200

    # For 2-min out-and-back video, portrait mode, with gap control:
    python extract_frames.py --video gopro.MP4 -o ./frames -n 200 \
        --start 5 --end 65 --rotate 270 --max_gap 3.0
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from collections import defaultdict


def compute_frame_metrics(frame):
    """Compute quality and content metrics for a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = np.mean(gray)

    # Blur score (higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color histogram (for diversity comparison)
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return {
        "brightness": float(brightness),
        "sharpness": float(laplacian_var),
        "histogram": hist
    }


def histogram_distance(hist1, hist2):
    """Chi-squared distance between two color histograms."""
    return cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CHISQR
    )


def extract_diverse_frames(video_path, output_dir, num_frames=200,
                           min_brightness=30, max_brightness=240,
                           min_sharpness=50, diversity_weight=0.5,
                           start_sec=0, end_sec=None,
                           max_gap_sec=3.0, rotate=None):
    """
    Extract diverse, high-quality frames from video.

    Strategy:
    1. Sample candidate frames at regular intervals (3x target count)
    2. Filter out low-quality frames (too dark, too bright, too blurry)
    3. Greedily select diverse frames based on visual content
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    # Apply time range (e.g., first half only for out-and-back videos)
    start_frame = int(start_sec * fps)
    if end_sec is not None:
        end_frame = min(int(end_sec * fps), total_frames)
    else:
        end_frame = total_frames

    usable_frames = end_frame - start_frame
    usable_duration = usable_frames / fps

    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total duration: {duration:.1f}s ({duration/60:.1f}min)")
    print(f"  Using: {start_sec:.1f}s - {end_sec if end_sec else duration:.1f}s "
          f"({usable_duration:.1f}s, {usable_frames} frames)")
    print(f"  Target frames: {num_frames}")
    print()

    # Step 1: Sample candidate frames
    # For short clips, sample more densely to ensure full temporal coverage
    num_candidates = min(num_frames * 5, usable_frames)
    candidate_indices = np.linspace(start_frame, end_frame - 1, num_candidates, dtype=int)
    # Remove duplicate indices (can happen with very short clips)
    candidate_indices = np.unique(candidate_indices)

    print(f"Step 1: Sampling {num_candidates} candidate frames...")
    candidates = []

    for i, frame_idx in enumerate(candidate_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        metrics = compute_frame_metrics(frame)
        metrics["frame_idx"] = int(frame_idx)
        metrics["timestamp"] = float(frame_idx / fps)
        candidates.append(metrics)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{num_candidates} candidates...")

    print(f"  Got {len(candidates)} valid candidates")

    # Step 2: Quality filtering
    print(f"\nStep 2: Quality filtering...")
    quality_candidates = [
        c for c in candidates
        if (min_brightness <= c["brightness"] <= max_brightness
            and c["sharpness"] >= min_sharpness)
    ]
    print(f"  {len(quality_candidates)} passed quality filter "
          f"(removed {len(candidates) - len(quality_candidates)} low-quality)")

    if len(quality_candidates) < num_frames:
        print(f"  WARNING: Only {len(quality_candidates)} quality frames available, "
              f"lowering sharpness threshold...")
        # Relax sharpness filter
        sharpness_values = sorted([c["sharpness"] for c in candidates], reverse=True)
        if len(sharpness_values) >= num_frames:
            relaxed_threshold = sharpness_values[num_frames - 1]
        else:
            relaxed_threshold = 0
        quality_candidates = [
            c for c in candidates
            if (min_brightness <= c["brightness"] <= max_brightness
                and c["sharpness"] >= relaxed_threshold)
        ]
        print(f"  After relaxation: {len(quality_candidates)} candidates")

    # Step 3: Greedy diversity selection
    print(f"\nStep 3: Selecting {num_frames} diverse frames...")
    selected = []

    if not quality_candidates:
        print("ERROR: No quality candidates found. Check video quality.")
        cap.release()
        return

    # Start with the frame that has highest sharpness
    best_start = max(quality_candidates, key=lambda c: c["sharpness"])
    selected.append(best_start)
    remaining = [c for c in quality_candidates if c is not best_start]

    while len(selected) < num_frames and remaining:
        best_score = -1
        best_idx = 0

        for i, cand in enumerate(remaining):
            # Visual diversity: min histogram distance to any selected frame
            min_hist_dist = min(
                histogram_distance(cand["histogram"], s["histogram"])
                for s in selected
            )

            # Temporal diversity: min time distance to any selected frame
            min_time_dist = min(
                abs(cand["timestamp"] - s["timestamp"])
                for s in selected
            )
            # Normalize time distance (0-1 range)
            norm_time_dist = min_time_dist / max(usable_duration, 1)

            # Combined score
            score = (diversity_weight * min_hist_dist +
                     (1 - diversity_weight) * norm_time_dist)

            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(remaining.pop(best_idx))

        if len(selected) % 50 == 0:
            print(f"  Selected {len(selected)}/{num_frames}...")

    # Step 3b: Fill large temporal gaps
    # Sort selected by timestamp, find gaps > max_gap_sec, insert best candidate
    # Uses two pools: first try quality-filtered 'remaining', then fall back to
    # ALL candidates (even low-quality) to guarantee no huge gaps.
    selected.sort(key=lambda c: c["timestamp"])
    selected_indices = {c["frame_idx"] for c in selected}

    # Build fallback pool from ALL candidates (unfiltered) not already selected
    all_remaining = [c for c in candidates if c["frame_idx"] not in selected_indices]

    gap_fills = 0
    gap_fills_fallback = 0
    max_iterations = 500  # safety limit

    for _ in range(max_iterations):
        # Recalculate gaps each iteration
        timestamps_sel = [s["timestamp"] for s in selected]
        if len(timestamps_sel) < 2:
            break
        gaps = [(timestamps_sel[i+1] - timestamps_sel[i], i)
                for i in range(len(timestamps_sel)-1)]
        largest_gap, gap_idx = max(gaps, key=lambda x: x[0])

        if largest_gap <= max_gap_sec:
            break  # All gaps are acceptable

        gap_start = timestamps_sel[gap_idx]
        gap_end = timestamps_sel[gap_idx + 1]
        gap_mid = (gap_start + gap_end) / 2

        # Try quality-filtered pool first
        best_cand = None
        best_dist = float('inf')
        best_cand_idx = -1
        best_pool = None

        for pool_name, pool in [("remaining", remaining), ("all_remaining", all_remaining)]:
            for i, cand in enumerate(pool):
                if gap_start < cand["timestamp"] < gap_end:
                    dist = abs(cand["timestamp"] - gap_mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_cand = cand
                        best_cand_idx = i
                        best_pool = pool_name
            if best_cand is not None and pool_name == "remaining":
                break  # Prefer quality-filtered candidates

        if best_cand is None:
            # No candidates at all in this gap — nothing we can do
            print(f"  WARNING: Cannot fill gap at {gap_start:.1f}s-{gap_end:.1f}s "
                  f"({largest_gap:.1f}s) — no candidates available")
            break

        # Remove from the pool it was found in
        if best_pool == "remaining":
            remaining.pop(best_cand_idx)
            # Also remove from all_remaining
            all_remaining = [c for c in all_remaining if c["frame_idx"] != best_cand["frame_idx"]]
            gap_fills += 1
        else:
            all_remaining.pop(best_cand_idx)
            gap_fills_fallback += 1

        selected.append(best_cand)
        selected.sort(key=lambda c: c["timestamp"])
        selected_indices.add(best_cand["frame_idx"])

    if gap_fills > 0 or gap_fills_fallback > 0:
        print(f"  Filled {gap_fills} gaps from quality pool, "
              f"{gap_fills_fallback} from fallback pool (>{max_gap_sec}s threshold)")
    print(f"  Final frame count: {len(selected)}")

    # Final gap report
    timestamps_final = [s["timestamp"] for s in selected]
    if len(timestamps_final) >= 2:
        final_gaps = [timestamps_final[i+1] - timestamps_final[i]
                      for i in range(len(timestamps_final)-1)]
        print(f"  Max gap after fill: {max(final_gaps):.2f}s")

    # Determine rotation
    rotate_code = None
    if rotate == 90:
        rotate_code = cv2.ROTATE_90_CLOCKWISE
    elif rotate == 180:
        rotate_code = cv2.ROTATE_180
    elif rotate == 270 or rotate == -90:
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

    # Step 4: Save frames
    print(f"\nStep 4: Saving {len(selected)} frames to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Sort by frame index for consistent naming
    selected.sort(key=lambda c: c["frame_idx"])

    metadata = {
        "video": os.path.basename(video_path),
        "video_resolution": f"{width}x{height}",
        "video_fps": fps,
        "video_total_frames": total_frames,
        "video_duration_sec": round(duration, 2),
        "time_range": f"{start_sec}s - {end_sec if end_sec else duration}s",
        "rotation": rotate if rotate else 0,
        "output_resolution": f"{height}x{width}" if rotate in (90, 270, -90) else f"{width}x{height}",
        "max_gap_sec": max_gap_sec,
        "num_extracted": len(selected),
        "frames": []
    }

    for i, sel in enumerate(selected):
        frame_idx = sel["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Rotate if needed (portrait GoPro footage)
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        # Name format: frame_XXXXXX_tXXXXs.jpg
        # Includes frame index and timestamp for easy reference
        timestamp_str = f"{sel['timestamp']:.2f}".replace(".", "p")
        filename = f"frame_{frame_idx:06d}_t{timestamp_str}s.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        metadata["frames"].append({
            "filename": filename,
            "frame_idx": frame_idx,
            "timestamp": round(sel["timestamp"], 3),
            "brightness": round(sel["brightness"], 1),
            "sharpness": round(sel["sharpness"], 1)
        })

    # Save metadata
    meta_path = os.path.join(output_dir, "extraction_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cap.release()

    print(f"\nDone!")
    print(f"  Frames saved to: {output_dir}/")
    print(f"  Metadata saved to: {meta_path}")
    print(f"  Timestamp range: {selected[0]['timestamp']:.1f}s - {selected[-1]['timestamp']:.1f}s")

    # Print distribution stats
    timestamps = [s["timestamp"] for s in selected]
    gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    print(f"  Avg gap between frames: {np.mean(gaps):.2f}s")
    print(f"  Min gap: {np.min(gaps):.2f}s | Max gap: {np.max(gaps):.2f}s")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract diverse frames from GoPro video for annotation"
    )
    parser.add_argument("--video", "-v", required=True,
                        help="Path to input video file")
    parser.add_argument("--output", "-o", default="./frames",
                        help="Output directory for extracted frames (default: ./frames)")
    parser.add_argument("--num_frames", "-n", type=int, default=200,
                        help="Number of frames to extract (default: 200)")
    parser.add_argument("--min_sharpness", type=float, default=50,
                        help="Minimum sharpness threshold (default: 50)")
    parser.add_argument("--diversity_weight", type=float, default=0.5,
                        help="Weight for visual vs temporal diversity 0-1 (default: 0.5)")
    parser.add_argument("--start", "-s", type=float, default=0,
                        help="Start time in seconds (default: 0). Skip initial useless footage.")
    parser.add_argument("--end", "-e", type=float, default=None,
                        help="End time in seconds (default: full video). "
                             "For 2min out-and-back, use half e.g. --start 5 --end 65")
    parser.add_argument("--rotate", "-r", type=int, default=None, choices=[90, 180, 270, -90],
                        help="Rotate frames: 90 (CW), 270/-90 (CCW), 180. "
                             "Use for portrait GoPro footage.")
    parser.add_argument("--max_gap", type=float, default=3.0,
                        help="Maximum allowed gap in seconds between consecutive frames "
                             "(default: 3.0). Fills gaps after diversity selection.")

    args = parser.parse_args()

    extract_diverse_frames(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num_frames,
        min_sharpness=args.min_sharpness,
        diversity_weight=args.diversity_weight,
        start_sec=args.start,
        end_sec=args.end,
        max_gap_sec=args.max_gap,
        rotate=args.rotate
    )


if __name__ == "__main__":
    main()