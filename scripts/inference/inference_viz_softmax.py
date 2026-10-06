"""
inference_viz.py — Run the trained cotton-boll Mask R-CNN on a FOLDER OF IMAGES
or a VIDEO, and emit both (a) annotated JPGs and (b) detection JSON consumable
by Inference_Viewer AND by track_detections.py.

INPUT: either --images_dir <folder> OR --video <file.mp4> (mutually exclusive).

OUTPUTS
-------
Writes to <output>/:
  json/<stem>.json        flat list of per-boll records (IMAGE-MODE).
                          The viewer matches each JSON to its frame by stem.
  viz/<stem>.jpg          annotated frame (bbox, mask, tip/base, pose axis, label)
  frames/<stem>.jpg       (VIDEO ONLY) the raw sampled frame, so the viewer has
                          images to display. Drag this folder into the viewer.
  detections_all.json     ONE combined VIDEO-MODE file: [{frame, src, bolls:[...]}]
                          Directly viewable AND the input track_detections.py
                          consumes (it adds track_id and re-emits the same shape).

The viewer matches frames to detections by FILENAME STEM, so the staged frame
<stem>.jpg and its json/<stem>.json line up automatically.

VIDEO SAMPLING
--------------
Pick ONE (priority: --every_n_frames > --fps > --interval):
  --every_n_frames N   emit every Nth source frame (1=every frame). Exact.
  --fps F              target output fps, converted to nearest integer step.
  --interval S         seconds between samples (default 0.5).
  --start / --end      clip a time window (seconds; --end -1 = to end)
You can never sample faster than the source fps (step floors at 1 = every frame).
Frame names follow the extract_frames_uniform.py convention
(frame_<idx>_t<sec>p<centi>s) so video output is naming-compatible with the
existing extracted-frame datasets.

PER-BOLL RECORD (matches the viewer + tracker schema)
-----------------------------------------------------
  det_conf        float   detection score
  bbox            [x,y,w,h]   (xywh, original frame pixels)
  mask_poly       [[x,y],...] largest mask contour, simplified
  kpts            [[tx,ty,tc],[bx,by,bc]]   tip then base; tc/bc = heatmap peak
  occ_class       str     one of none|leaf|stem|other_boll|other|frame_edge
  occ_class_conf  float   softmax confidence of the predicted class
  is_occluded     0|1     binary occlusion head (>0.5)
  occlusion_pct   float   regressed occlusion fraction in [0,1]
  visfrac         float   1 - occlusion_pct   (visible fraction)
  tip_conf/base_conf    float  heatmap peak (== kpts[*][2], duplicated)
  tip_conc/base_conc    float  peak / mean  (sharpness)
  orientation_deg float   atan2(ty-by, tx-bx) in degrees, or null if a kpt missing

USAGE
-----
  # image folder
  python inference_viz.py --weights models/combined_all_v2/model_final.pth \
      --images_dir data/images/extracted_frames/left_down \
      --output inference_results/left_down_v2 --score_thresh 0.5

  # video, sampled every 0.5s
  python inference_viz.py --weights models/combined_all_v2/model_final.pth \
      --video data/raw_video/left_down.mp4 \
      --output inference_results/left_down_video --interval 0.5

Run with scripts/training on PYTHONPATH so the custom ROI head registers:
  PYTHONPATH=scripts/training python scripts/inference/inference_viz.py ...
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Importing the training module registers CottonROIHeads in the ROI_HEADS
# registry (via its @ROI_HEADS_REGISTRY.register() decorator) and gives us the
# heatmap size constant. Without this import the custom head can't be built.
from train_maskrcnn_pose import CottonROIHeads, HEATMAP_SIZE  # noqa: F401


# Canonical occlusion-class order. MUST match cvat_to_coco_v4.py and the model's
# occ_class_pred output ordering, and the viewer's class keys.
OCC_CLASSES = ["none", "leaf", "stem", "other_boll", "other", "frame_edge"]

# Pose heatmap spans the box expanded by this factor (see train_maskrcnn_pose.py).
# MUST match POSE_BOX_SCALE in the trainer, or inference keypoints land wrong.
POSE_BOX_SCALE = 1.5


def expand_box_xyxy(x1, y1, x2, y2, scale):
    """Expand an xyxy box about its center by `scale` per-dimension."""
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    hw = (x2 - x1) * 0.5 * scale
    hh = (y2 - y1) * 0.5 * scale
    return cx - hw, cy - hh, cx + hw, cy + hh


# ---------------------------------------------------------------------------
# Color transfer (Reinhard mean/std in LAB) — optional inference preprocessing
# ---------------------------------------------------------------------------

class ColorTransfer:
    """Reinhard et al. mean/std color transfer in LAB space.

    Maps a source-domain frame's LAB statistics onto a target domain, per channel:
        out = (in - src_mean) * (tgt_std / src_std) + tgt_mean
    then clips to valid LAB range and converts back to BGR.

    Built from two stats JSONs produced by domain_color_stats.py:
      src = the domain of the INPUT frames (e.g. rgbd_stats.json)
      tgt = the domain to match TOWARD (e.g. gopro_stats.json)

    NOTE: this shifts aggregate tone/brightness/warmth. It does NOT fix
    resolution differences, and because the stats blend camera + scene content,
    the match is approximate. Judge by detections, not by how it looks.
    """

    def __init__(self, src_stats: dict, tgt_stats: dict):
        self.src_mean = np.array(src_stats["lab_mean"], dtype=np.float32)
        self.src_std = np.array(src_stats["lab_std"], dtype=np.float32)
        self.tgt_mean = np.array(tgt_stats["lab_mean"], dtype=np.float32)
        self.tgt_std = np.array(tgt_stats["lab_std"], dtype=np.float32)
        # guard against divide-by-zero on a flat channel
        self.src_std = np.maximum(self.src_std, 1e-3)
        self.scale = self.tgt_std / self.src_std

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab = (lab - self.src_mean) * self.scale + self.tgt_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_color_transfer(src_json, tgt_json):
    if not src_json or not tgt_json:
        return None
    with open(src_json) as f:
        src = json.load(f)
    with open(tgt_json) as f:
        tgt = json.load(f)
    print(f"[color_transfer] src {src['dir']} LAB mean={src['lab_mean']}")
    print(f"[color_transfer] tgt {tgt['dir']} LAB mean={tgt['lab_mean']}")
    return ColorTransfer(src, tgt)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

def build_predictor(weights: str, score_thresh: float = 0.5) -> DefaultPredictor:
    """Build a DefaultPredictor wired to the custom CottonROIHeads.

    DefaultPredictor handles resize + normalization, runs the model, and
    rescales pred_boxes / pred_masks back to ORIGINAL image coordinates.
    Custom per-RoI fields (pred_tip_heatmap 56x56, occlusion outputs) pass
    through postprocessing unchanged.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NAME = "CottonROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.MASK_ON = True
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)


# ---------------------------------------------------------------------------
# Geometry / heatmap helpers
# ---------------------------------------------------------------------------

def mask_to_polygon(mask: np.ndarray, epsilon_px: float = 1.5,
                    min_pts: int = 3) -> list:
    """Largest external contour of a binary mask, simplified, as [[x,y],...].

    Returns [] if no contour with >= min_pts is found.
    """
    m = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, epsilon_px, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < min_pts:
        return []
    return [[float(x), float(y)] for x, y in pts]


def heatmap_argmax_to_image(heatmap: np.ndarray, box) -> tuple:
    """Map a per-RoI heatmap argmax to original-image coords.

    heatmap: (H, W) in [0,1] (already sigmoid'd by the model).
    box: (x1, y1, x2, y2) tight detection box in original image pixels. The pose
         heatmap spans this box EXPANDED by POSE_BOX_SCALE, so we map the argmax
         through the expanded box (matching the trainer).
    Returns (x, y, peak, concentration).
      peak          = heatmap.max()              (confidence)
      concentration = peak / (mean + eps)        (sharpness)
    """
    h, w = heatmap.shape
    yy, xx = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    x1, y1, x2, y2 = expand_box_xyxy(*box, POSE_BOX_SCALE)
    x = x1 + (xx + 0.5) / w * (x2 - x1)
    y = y1 + (yy + 0.5) / h * (y2 - y1)
    peak = float(heatmap.max())
    mean = float(heatmap.mean())
    conc = peak / (mean + 1e-6)
    return float(x), float(y), peak, float(conc)


def heatmap_softargmax_to_image(heatmap: np.ndarray, box, beta: float = 8.0) -> tuple:
    """Soft-argmax: softmax-weighted expected position over the heatmap.

    Continuous in the input, so it damps the frame-to-frame flipping that hard
    argmax produces on ambiguous/bimodal heatmaps. beta controls sharpness:
      low beta  -> smoother, but pulls toward the heatmap CENTER on flat maps
      high beta -> approaches hard argmax
    The heatmap is normalized by its own max before the softmax so beta behaves
    consistently regardless of absolute peak height.

    peak/concentration are still reported from the RAW heatmap (same confidence
    signal as hard argmax), so downstream confidence gating is unchanged.
    """
    h, w = heatmap.shape
    hm = heatmap.astype(np.float64)
    peak = float(hm.max())
    mean = float(hm.mean())
    conc = peak / (mean + 1e-6)

    # normalize by max so beta is scale-invariant, then softmax
    norm = hm / (peak + 1e-9)
    z = beta * norm
    z -= z.max()
    e = np.exp(z)
    p = e / (e.sum() + 1e-12)

    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    ex = float((p.sum(axis=0) * xs).sum())  # expected column
    ey = float((p.sum(axis=1) * ys).sum())  # expected row

    x1, y1, x2, y2 = expand_box_xyxy(*box, POSE_BOX_SCALE)
    x = x1 + (ex + 0.5) / w * (x2 - x1)
    y = y1 + (ey + 0.5) / h * (y2 - y1)
    return float(x), float(y), peak, float(conc)


def keypoint_from_heatmap(heatmap, box, soft, beta):
    """Dispatch to hard or soft argmax based on the `soft` flag."""
    if soft:
        return heatmap_softargmax_to_image(heatmap, box, beta)
    return heatmap_argmax_to_image(heatmap, box)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(predictor, img: np.ndarray, soft=False, beta=8.0) -> list:
    """Run the model on one BGR image; return a list of per-boll records.

    soft/beta control keypoint extraction: hard argmax (default) or soft-argmax
    (continuous, damps frame-to-frame flipping). See keypoint_from_heatmap.
    """
    outputs = predictor(img)
    inst = outputs["instances"].to("cpu")
    n = len(inst)
    if n == 0:
        return []

    boxes = inst.pred_boxes.tensor.numpy()           # (N,4) xyxy, original coords
    scores = inst.scores.numpy()                     # (N,)
    masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
    tip_hms = inst.pred_tip_heatmap.numpy() if inst.has("pred_tip_heatmap") else None
    base_hms = inst.pred_base_heatmap.numpy() if inst.has("pred_base_heatmap") else None
    is_occ = inst.pred_is_occluded.numpy() if inst.has("pred_is_occluded") else None
    occ_pct = inst.pred_occlusion_pct.numpy() if inst.has("pred_occlusion_pct") else None
    occ_cls = inst.pred_occ_class.numpy() if inst.has("pred_occ_class") else None
    occ_cls_conf = inst.pred_occ_class_conf.numpy() if inst.has("pred_occ_class_conf") else None

    bolls = []
    for i in range(n):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        rec = {
            "det_conf": float(scores[i]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],   # xywh
        }

        # mask polygon
        if masks is not None:
            rec["mask_poly"] = mask_to_polygon(masks[i])
        else:
            rec["mask_poly"] = []

        # tip / base keypoints from heatmaps
        tip = [None, None, 0.0]
        base = [None, None, 0.0]
        if tip_hms is not None:
            tx, ty, tc, tconc = keypoint_from_heatmap(tip_hms[i, 0], (x1, y1, x2, y2), soft, beta)
            tip = [tx, ty, tc]
            rec["tip_conf"] = tc
            rec["tip_conc"] = tconc
        if base_hms is not None:
            bx, by, bc, bconc = keypoint_from_heatmap(base_hms[i, 0], (x1, y1, x2, y2), soft, beta)
            base = [bx, by, bc]
            rec["base_conf"] = bc
            rec["base_conc"] = bconc
        rec["kpts"] = [tip, base]

        # orientation (tip relative to base), only if both kpts present
        if tip[0] is not None and base[0] is not None:
            rec["orientation_deg"] = math.degrees(
                math.atan2(tip[1] - base[1], tip[0] - base[0]))
        else:
            rec["orientation_deg"] = None

        # occlusion outputs
        if occ_cls is not None:
            cid = int(occ_cls[i])
            rec["occ_class"] = OCC_CLASSES[cid] if 0 <= cid < len(OCC_CLASSES) else "none"
        if occ_cls_conf is not None:
            rec["occ_class_conf"] = float(occ_cls_conf[i])
        if is_occ is not None:
            rec["is_occluded"] = int(is_occ[i] > 0.5)
        if occ_pct is not None:
            pct = float(occ_pct[i])
            rec["occlusion_pct"] = pct
            rec["visfrac"] = float(max(0.0, min(1.0, 1.0 - pct)))

        bolls.append(rec)

    return bolls


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# BGR colors per class (cv2). Mirrors the viewer palette.
VIZ_COLOR = {
    "none":       (60, 220, 60),
    "leaf":       (60, 220, 220),
    "stem":       (60, 165, 255),
    "other_boll": (60, 60, 220),
    "other":      (255, 90, 190),
    "frame_edge": (255, 200, 60),
}
COLOR_TIP = (255, 220, 60)    # cyan-ish (BGR)
COLOR_BASE = (220, 60, 220)   # magenta
COLOR_AXIS = (255, 255, 255)


def draw_annotations(img: np.ndarray, bolls: list, score_thresh: float) -> np.ndarray:
    viz = img.copy()
    H, W = viz.shape[:2]
    lw = max(2, min(H, W) // 600)
    for d in bolls:
        if d["det_conf"] < score_thresh:
            continue
        cls = d.get("occ_class", "none")
        color = VIZ_COLOR.get(cls, (200, 200, 200))

        # mask outline
        poly = d.get("mask_poly")
        if poly and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(viz, [pts], True, color, lw)

        # bbox
        x, y, w, h = d["bbox"]
        cv2.rectangle(viz, (int(x), int(y)), (int(x + w), int(y + h)), color, lw)

        # tip / base + axis
        kpts = d.get("kpts", [[None]*3, [None]*3])
        tip, base = kpts[0], kpts[1]
        kr = max(4, lw * 2)
        if tip[0] is not None and base[0] is not None:
            cv2.line(viz, (int(tip[0]), int(tip[1])),
                     (int(base[0]), int(base[1])), COLOR_AXIS, max(1, lw // 2))
        if tip[0] is not None:
            cv2.circle(viz, (int(tip[0]), int(tip[1])), kr, COLOR_TIP, -1)
        if base[0] is not None:
            cv2.circle(viz, (int(base[0]), int(base[1])), kr, COLOR_BASE, -1)

        # label
        parts = [f"{cls} {d.get('occ_class_conf', 0):.2f}",
                 f"d={d['det_conf']:.2f}"]
        if "visfrac" in d:
            parts.append(f"v={d['visfrac']:.2f}")
        label = " | ".join(parts)
        cv2.putText(viz, label, (int(x), max(15, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, lw * 0.25), color, max(1, lw // 2))
    return viz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def frame_stem(idx: int, t_sec: float) -> str:
    """Stable, sortable frame name shared by the JPG and its JSON.

    Example: frame_000123_t40p99s  ->  frame_000123_t40p99s.jpg / .json
    Mirrors the extract_frames_uniform.py convention (t<sec>p<centi>s) so video
    output is naming-compatible with the existing extracted-frame datasets.
    """
    whole = int(t_sec)
    centi = int(round((t_sec - whole) * 100))
    if centi == 100:
        whole += 1
        centi = 0
    return f"frame_{idx:06d}_t{whole}p{centi:02d}s"


def resolve_step_frames(src_fps: float, every_n_frames, target_fps, interval_sec):
    """Decide the integer frame step from whichever sampling flag was given.

    Priority: --every_n_frames > --fps > --interval.
    Returns (step_frames, description_str). step_frames is always >= 1; you can
    never sample faster than the source (step 1 = every frame).
    """
    if every_n_frames is not None:
        step = max(1, int(every_n_frames))
        return step, f"every {step} frame(s)  (~{src_fps/step:.2f} fps out)"
    if target_fps is not None:
        step = max(1, int(round(src_fps / float(target_fps))))
        eff = src_fps / step
        note = "" if abs(eff - target_fps) < 0.5 else \
            f"  [note: source is {src_fps:.1f} fps, closest achievable is {eff:.2f}]"
        return step, f"target {target_fps} fps -> every {step} frame(s) (~{eff:.2f} fps out){note}"
    # fall back to interval (seconds between samples)
    step = max(1, int(round(interval_sec * src_fps)))
    return step, f"interval {interval_sec}s -> every {step} frame(s) (~{src_fps/step:.2f} fps out)"


def iter_video_frames(video_path: Path, step_frames: int,
                      start_sec: float, end_sec: float):
    """Yield (out_index, t_sec, bgr_frame), emitting every step_frames-th source
    frame within [start_sec, end_sec].

    out_index counts emitted (sampled) frames so names sort in capture order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total else None
    print(f"  video fps={fps:.2f} frames={total} duration="
          f"{duration:.1f}s" if duration else f"  video fps={fps:.2f}")

    step_frames = max(1, int(step_frames))
    out_idx = 0
    src_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        t = src_idx / fps
        in_window = (t >= start_sec) and (end_sec < 0 or t <= end_sec)
        if in_window and (src_idx % step_frames == 0):
            ok, frame = cap.retrieve()
            if ok:
                yield out_idx, t, frame
                out_idx += 1
        src_idx += 1
        if end_sec >= 0 and t > end_sec:
            break
    cap.release()


def write_frame_outputs(stem, src_name, frame_idx, img, bolls, args,
                        json_dir, viz_dir, frames_dir, combined):
    """Write per-frame JSON, optional annotated JPG, optional staged frame image,
    and append to the combined video-mode list."""
    if not args.no_json:
        with open(json_dir / f"{stem}.json", "w") as f:
            json.dump(bolls, f)

    if not args.no_viz:
        viz = draw_annotations(img, bolls, args.score_thresh)
        cv2.imwrite(str(viz_dir / f"{stem}.jpg"), viz)

    # For video input, also save the raw frame so the viewer has images to load
    if frames_dir is not None:
        cv2.imwrite(str(frames_dir / f"{stem}.jpg"), img)

    if not args.no_combined:
        combined.append({"frame": frame_idx, "src": src_name, "bolls": bolls})


def main():
    p = argparse.ArgumentParser(description="Cotton boll inference + viewer JSON")
    p.add_argument("--weights", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images_dir", help="Folder of frame images")
    src.add_argument("--video", help="Video file (mp4/mov/avi/...)")
    p.add_argument("--output", required=True)
    p.add_argument("--score_thresh", type=float, default=0.5)
    p.add_argument("--no_viz", action="store_true", help="Skip annotated JPGs")
    p.add_argument("--no_json", action="store_true", help="Skip per-frame JSONs")
    p.add_argument("--no_combined", action="store_true",
                   help="Skip the combined video-mode detections_all.json")
    # video sampling — pick ONE; priority: --every_n_frames > --fps > --interval
    p.add_argument("--every_n_frames", type=int, default=None,
                   help="(video) emit every Nth source frame. 1=every frame, "
                        "2=every other, etc. Exact, no rounding.")
    p.add_argument("--fps", type=float, default=None,
                   help="(video) target output fps; converted to the nearest "
                        "integer frame step. Can't exceed source fps.")
    p.add_argument("--interval", type=float, default=0.5,
                   help="(video) seconds between sampled frames (default 0.5). "
                        "Used only if --every_n_frames and --fps are unset.")
    p.add_argument("--start", type=float, default=0.0,
                   help="(video) start time in seconds")
    p.add_argument("--end", type=float, default=-1.0,
                   help="(video) end time in seconds, -1 = to end")
    # optional color-domain transfer (Reinhard LAB mean/std)
    p.add_argument("--color_src", default=None,
                   help="stats JSON for the INPUT domain (e.g. rgbd_stats.json)")
    p.add_argument("--color_tgt", default=None,
                   help="stats JSON for the TARGET domain to match toward "
                        "(e.g. gopro_stats.json). Both required to enable transfer.")
    p.add_argument("--color_debug_n", type=int, default=0,
                   help="Save this many before/after transfer pairs to "
                        "<output>/color_debug/ then continue.")
    p.add_argument("--soft_argmax", action="store_true",
                   help="Use soft-argmax (weighted expected position) for tip/base "
                        "instead of hard argmax. Damps frame-to-frame flipping.")
    p.add_argument("--softarg_beta", type=float, default=8.0,
                   help="Soft-argmax sharpness. Lower=smoother (pulls to center on "
                        "flat maps), higher=closer to hard argmax. Default 8.")
    args = p.parse_args()

    out_dir = Path(args.output)
    json_dir = out_dir / "json"
    viz_dir = out_dir / "viz"
    frames_dir = out_dir / "frames"   # only used for video input
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_json:
        json_dir.mkdir(exist_ok=True)
    if not args.no_viz:
        viz_dir.mkdir(exist_ok=True)

    predictor = build_predictor(args.weights, args.score_thresh)
    color = load_color_transfer(args.color_src, args.color_tgt)
    debug_left = args.color_debug_n if color else 0
    debug_dir = out_dir / "color_debug"
    if debug_left:
        debug_dir.mkdir(exist_ok=True)
    combined = []
    total_bolls = 0
    n_frames = 0

    if args.video:
        video_path = Path(args.video)
        if video_path.suffix.lower() not in VIDEO_EXTS:
            print(f"[warn] {video_path.suffix} not a known video ext; trying anyway")
        frames_dir.mkdir(exist_ok=True)   # video frames need staging for the viewer
        print(f"Processing video: {video_path.name}")
        # Probe fps once to resolve the sampling step from whichever flag was given
        _probe = cv2.VideoCapture(str(video_path))
        src_fps = (_probe.get(cv2.CAP_PROP_FPS) or 30.0) if _probe.isOpened() else 30.0
        _probe.release()
        step_frames, step_desc = resolve_step_frames(
            src_fps, args.every_n_frames, args.fps, args.interval)
        print(f"  sampling: {step_desc}")
        for frame_idx, t_sec, frame in iter_video_frames(
                video_path, step_frames, args.start, args.end):
            stem = frame_stem(frame_idx, t_sec)
            if color is not None:
                if debug_left > 0:
                    cv2.imwrite(str(debug_dir / f"{stem}_before.jpg"), frame)
                frame = color(frame)
                if debug_left > 0:
                    cv2.imwrite(str(debug_dir / f"{stem}_after.jpg"), frame)
                    debug_left -= 1
            bolls = process_image(predictor, frame, args.soft_argmax, args.softarg_beta)
            total_bolls += len(bolls)
            write_frame_outputs(stem, f"{stem}.jpg", frame_idx, frame, bolls,
                                args, json_dir, viz_dir, frames_dir, combined)
            n_frames += 1
            if n_frames % 25 == 0:
                print(f"  {n_frames} frames processed (t={t_sec:.1f}s)")
        print(f"  Staged {n_frames} frames -> {frames_dir}/")

    else:
        img_paths = sorted(
            [q for q in Path(args.images_dir).iterdir()
             if q.suffix.lower() in IMG_EXTS],
            key=lambda q: q.name)
        print(f"Found {len(img_paths)} images in {args.images_dir}")
        for frame_idx, img_path in enumerate(img_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [warn] could not read {img_path.name}")
                continue
            stage_dir = None
            if color is not None:
                if debug_left > 0:
                    cv2.imwrite(str(debug_dir / f"{img_path.stem}_before.jpg"), img)
                img = color(img)
                if debug_left > 0:
                    cv2.imwrite(str(debug_dir / f"{img_path.stem}_after.jpg"), img)
                    debug_left -= 1
                # stage the TRANSFERRED frame so the viewer shows what the model saw
                frames_dir.mkdir(exist_ok=True)
                stage_dir = frames_dir
            bolls = process_image(predictor, img, args.soft_argmax, args.softarg_beta)
            total_bolls += len(bolls)
            write_frame_outputs(img_path.stem, img_path.name, frame_idx, img,
                                bolls, args, json_dir, viz_dir, stage_dir, combined)
            n_frames += 1
            if n_frames % 25 == 0:
                print(f"  {n_frames}/{len(img_paths)} processed")

    if not args.no_combined:
        with open(out_dir / "detections_all.json", "w") as f:
            json.dump(combined, f)
        print(f"Wrote {out_dir / 'detections_all.json'}  "
              f"({len(combined)} frames, video-mode)")

    print(f"\nDone. {n_frames} frames, {total_bolls} bolls total.")
    if args.video:
        print(f"  Frames:    {frames_dir}/   (drag this into the viewer)")
    if not args.no_json:
        print(f"  Per-frame JSON: {json_dir}/")
    if not args.no_viz:
        print(f"  Annotated JPGs: {viz_dir}/")
    print("\nIn the viewer, load BOTH the frames folder and the json folder "
          "(or just detections_all.json + frames).")


if __name__ == "__main__":
    main()
