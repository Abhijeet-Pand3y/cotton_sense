"""
track_detections.py — Assign stable track_id across frames (IoU SORT-lite) and
smooth tip/base keypoints along each track to remove per-frame pose jitter.

WHY
---
Per-frame argmax pose estimation jitters: a boll's tip/base can flip between two
near-equal heatmap peaks on almost-identical adjacent frames. Tracking links a
boll across frames; then we smooth each track's keypoints OFFLINE with a CENTERED
median filter (uses past AND future frames) which rejects isolated flips that no
online/per-frame method can fix.

INPUT
-----
--input  either:
    * a combined video-mode JSON: [{frame, src, bolls:[...]}]   (detections_all.json), OR
    * a directory of per-frame JSONs (image-mode flat lists), matched/ordered by name.
Each boll record carries: bbox [x,y,w,h], det_conf, kpts [[tx,ty,tc],[bx,by,bc]],
mask_poly, occ_class, visfrac, ... (see inference_viz.py).

OUTPUT
------
--output           combined video-mode tracked JSON (same shape + track_id, smoothed kpts)
--output_json_dir  per-frame tracked JSONs (image-mode + track_id), named by frame stem.
                   Load THIS folder + the frames/ folder in Inference_Viewer to get
                   track trails and stabilized pose.

TRACKER
-------
SORT-lite: match each frame's detections to active tracks by IoU of bboxes
(Hungarian if scipy is present, else greedy). Unmatched detections start new
tracks; tracks unseen for > max_age frames are retired. No Kalman — adjacent-frame
motion is small at video frame rates.

SMOOTHING
---------
Per track, per keypoint (tip, base), independently on x and y:
  --smooth_method median       centered median (default; best at killing flips)
                  median_ema    median then a light EMA (extra smoothness)
                  ema           EMA only
                  none          tracking only, no smoothing
  --smooth_window N             centered window size (default 5)
Missing keypoints (conf<=0 or null) are treated as gaps and skipped, not zero-filled.
orientation_deg is recomputed from the smoothed tip/base.

USAGE
-----
  python track_detections.py \
      --input inference_results/RightDown_30fps/detections_all.json \
      --output inference_results/RightDown_30fps/tracked.json \
      --output_json_dir inference_results/RightDown_30fps/tracked_json \
      --iou_thresh 0.3 --max_age 5 --min_score 0.3 \
      --smooth_method median --smooth_window 5
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

IMG_STEM_FROM_SRC = lambda s: Path(s).stem if s else None


def load_frames(input_path: Path):
    """Return an ordered list of frame dicts: [{frame, src, bolls:[...]}].

    Accepts a combined video-mode JSON or a directory of per-frame image-mode
    JSONs. For a directory, frames are ordered by filename and src/frame are
    synthesized from the file stem.
    """
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir()
                       if p.suffix.lower() == ".json")
        frames = []
        for i, f in enumerate(files):
            bolls = json.load(open(f))
            if not isinstance(bolls, list):
                raise ValueError(f"{f} is not an image-mode flat list")
            frames.append({"frame": i, "src": f.stem + ".jpg", "bolls": bolls})
        return frames

    data = json.load(open(input_path))
    if isinstance(data, list) and data and isinstance(data[0], dict) \
            and "bolls" in data[0]:
        # video-mode; ensure ordered by frame index
        return sorted(data, key=lambda fr: fr.get("frame", 0))
    raise ValueError(
        f"{input_path} is neither a directory nor a video-mode "
        f"[{{frame,src,bolls}}] JSON. (A flat image-mode list has no frame order.)")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections(iou_mat: np.ndarray, thresh: float):
    """Return list of (track_idx, det_idx) pairs with IoU >= thresh."""
    if iou_mat.size == 0:
        return []
    if SCIPY_OK:
        ti, di = linear_sum_assignment(-iou_mat)
        return [(int(t), int(d)) for t, d in zip(ti, di)
                if iou_mat[t, d] >= thresh]
    # greedy fallback
    pairs, used_t, used_d = [], set(), set()
    order = np.dstack(
        np.unravel_index(np.argsort(-iou_mat, axis=None), iou_mat.shape))[0]
    for t, d in order:
        t, d = int(t), int(d)
        if t in used_t or d in used_d:
            continue
        if iou_mat[t, d] < thresh:
            break
        pairs.append((t, d))
        used_t.add(t)
        used_d.add(d)
    return pairs


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Track:
    __slots__ = ("id", "bbox", "age", "hits", "last_frame")

    def __init__(self, tid, bbox, frame_idx):
        self.id = tid
        self.bbox = bbox
        self.age = 0          # frames since last match
        self.hits = 1
        self.last_frame = frame_idx

    def update(self, bbox, frame_idx):
        self.bbox = bbox
        self.age = 0
        self.hits += 1
        self.last_frame = frame_idx


def run_tracker(frames, iou_thresh, max_age, min_score):
    """Assign a track_id (or None) to every boll, in place. Returns # tracks."""
    tracks = []
    next_id = 0

    for fi, frame in enumerate(frames):
        bolls = frame["bolls"]
        # everything starts unassigned
        for b in bolls:
            b["track_id"] = None
        # candidate detections gated by score
        dets = [b for b in bolls
                if (b.get("det_conf") is None or b["det_conf"] >= min_score)]

        if tracks and dets:
            iou_mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
            for ti, t in enumerate(tracks):
                for di, d in enumerate(dets):
                    iou_mat[ti, di] = iou_xywh(t.bbox, d["bbox"])
            matches = match_detections(iou_mat, iou_thresh)
        else:
            matches = []

        matched_t, matched_d = set(), set()
        for ti, di in matches:
            tracks[ti].update(dets[di]["bbox"], fi)
            dets[di]["track_id"] = tracks[ti].id
            matched_t.add(ti)
            matched_d.add(di)

        # new tracks for unmatched detections
        for di, d in enumerate(dets):
            if di not in matched_d:
                t = Track(next_id, d["bbox"], fi)
                d["track_id"] = t.id
                tracks.append(t)
                next_id += 1

        # age unmatched tracks, retire the stale ones
        for ti, t in enumerate(tracks):
            if ti not in matched_t:
                t.age += 1
        tracks = [t for t in tracks if t.age <= max_age]

    return next_id


# ---------------------------------------------------------------------------
# Keypoint smoothing
# ---------------------------------------------------------------------------

def smooth_seq(vals: np.ndarray, window: int, method: str) -> np.ndarray:
    """Smooth a 1-D sequence with NaN gaps. Centered median and/or EMA."""
    n = len(vals)
    out = vals.astype(np.float64).copy()
    half = max(1, window // 2)

    if method in ("median", "median_ema"):
        med = out.copy()
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            w = out[lo:hi]
            w = w[~np.isnan(w)]
            if len(w):
                med[i] = np.median(w)
        out = med

    if method in ("ema", "median_ema"):
        alpha = 2.0 / (window + 1)
        prev = None
        for i in range(n):
            if np.isnan(out[i]):
                continue
            prev = out[i] if prev is None else alpha * out[i] + (1 - alpha) * prev
            out[i] = prev

    return out


def smooth_keypoints(frames, window, method):
    """Group bolls by track_id and smooth tip/base x,y along each track."""
    if method == "none":
        return 0

    # track_id -> ordered list of (frame_idx, boll_ref)
    by_track = defaultdict(list)
    for fi, frame in enumerate(frames):
        for b in frame["bolls"]:
            tid = b.get("track_id")
            if tid is not None:
                by_track[tid].append((fi, b))

    n_smoothed = 0
    for tid, seq in by_track.items():
        seq.sort(key=lambda s: s[0])
        if len(seq) < 2:
            continue

        for kp in (0, 1):  # 0 = tip, 1 = base
            def get(b):
                k = b.get("kpts")
                if not k or kp >= len(k):
                    return (np.nan, np.nan)
                pt = k[kp]
                if pt is None or pt[0] is None or (len(pt) > 2 and pt[2] <= 0):
                    return (np.nan, np.nan)
                return (float(pt[0]), float(pt[1]))

            xs = np.array([get(b)[0] for _, b in seq])
            ys = np.array([get(b)[1] for _, b in seq])
            if np.all(np.isnan(xs)):
                continue
            xs_s = smooth_seq(xs, window, method)
            ys_s = smooth_seq(ys, window, method)

            for i, (_, b) in enumerate(seq):
                k = b.get("kpts")
                if not k or kp >= len(k) or k[kp] is None or k[kp][0] is None:
                    continue
                if not np.isnan(xs_s[i]):
                    k[kp][0] = float(xs_s[i])
                    k[kp][1] = float(ys_s[i])
        n_smoothed += 1

        # recompute orientation from smoothed tip/base
        for _, b in seq:
            k = b.get("kpts")
            if k and len(k) >= 2 and k[0] and k[1] and \
                    k[0][0] is not None and k[1][0] is not None:
                b["orientation_deg"] = math.degrees(
                    math.atan2(k[0][1] - k[1][1], k[0][0] - k[1][0]))

    return n_smoothed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="IoU tracking + keypoint smoothing")
    p.add_argument("--input", required=True, type=Path,
                   help="detections_all.json (video-mode) or a per-frame JSON dir")
    p.add_argument("--output", type=Path, default=None,
                   help="combined video-mode tracked JSON")
    p.add_argument("--output_json_dir", type=Path, default=None,
                   help="per-frame tracked JSONs (load these + frames/ in the viewer)")
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--max_age", type=int, default=5,
                   help="retire a track after this many unmatched frames")
    p.add_argument("--min_score", type=float, default=0.3,
                   help="ignore detections below this det_conf for tracking")
    p.add_argument("--min_track_len", type=int, default=1,
                   help="drop track_ids seen in fewer than this many frames")
    p.add_argument("--smooth_method", default="median",
                   choices=["median", "median_ema", "ema", "none"])
    p.add_argument("--smooth_window", type=int, default=5)
    args = p.parse_args()

    if not args.output and not args.output_json_dir:
        p.error("Specify at least one of --output or --output_json_dir")

    frames = load_frames(args.input)
    n_frames = len(frames)
    n_dets = sum(len(f["bolls"]) for f in frames)
    print(f"Loaded {n_frames} frames, {n_dets} detections")
    if not SCIPY_OK:
        print("  [note] scipy not found — using greedy IoU matching")

    n_tracks = run_tracker(frames, args.iou_thresh, args.max_age, args.min_score)
    print(f"Tracker: {n_tracks} raw track ids "
          f"(iou>={args.iou_thresh}, max_age={args.max_age}, min_score={args.min_score})")

    # optional: drop very short tracks (likely false positives)
    if args.min_track_len > 1:
        counts = defaultdict(int)
        for f in frames:
            for b in f["bolls"]:
                if b.get("track_id") is not None:
                    counts[b["track_id"]] += 1
        drop = {tid for tid, c in counts.items() if c < args.min_track_len}
        for f in frames:
            for b in f["bolls"]:
                if b.get("track_id") in drop:
                    b["track_id"] = None
        print(f"  dropped {len(drop)} tracks shorter than {args.min_track_len} frames")

    n_sm = smooth_keypoints(frames, args.smooth_window, args.smooth_method)
    print(f"Smoothing: {args.smooth_method} (window={args.smooth_window}) "
          f"applied to {n_sm} tracks")

    # final track stats
    final_ids = set()
    for f in frames:
        for b in f["bolls"]:
            if b.get("track_id") is not None:
                final_ids.add(b["track_id"])
    print(f"Final: {len(final_ids)} tracks across {n_frames} frames")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(frames, fh)
        print(f"Wrote {args.output}  (video-mode, tracked)")

    if args.output_json_dir:
        args.output_json_dir.mkdir(parents=True, exist_ok=True)
        for f in frames:
            stem = IMG_STEM_FROM_SRC(f.get("src")) or f"frame_{f['frame']:06d}"
            with open(args.output_json_dir / f"{stem}.json", "w") as fh:
                json.dump(f["bolls"], fh)
        print(f"Wrote {n_frames} per-frame JSONs -> {args.output_json_dir}/")
        print("  Load this folder + the frames/ folder in Inference_Viewer.")


if __name__ == "__main__":
    main()
