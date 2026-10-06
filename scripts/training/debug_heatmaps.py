"""
debug_heatmaps.py — Verify the GT heatmap pipeline is producing usable targets.

Pulls one batch through the dataset mapper, then for each instance:
  1. Saves the full-image GT tip heatmap (sanity: Gaussian inside polygon?)
  2. Saves the full-image GT base heatmap
  3. Saves the GT bbox + tip polygon + base rectangle overlay on the image
  4. Saves the per-RoI cropped target (56x56) — what the loss actually sees
  5. Prints stats: peak value, mean, % nonzero, visibility flag

If per-RoI crops are mostly zero, the pose loss can't learn — proposal/GT mismatch.

Usage:
    python debug_heatmaps.py \
        --train_json ./dataset/train.json \
        --images_dir ./right_down/images \
        --output ./debug_out \
        --n_images 3
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2 import model_zoo

# Reuse pieces from the training script
from train_maskrcnn_pose import (
    register_cotton_dataset,
    CottonDatasetMapper,
    make_masked_gaussian_heatmap,
    HEATMAP_SIZE,
)


def overlay_polygons(img, anns, save_path):
    """Draw bbox, tip polygon, base rectangle for each instance."""
    out = img.copy()
    for ann in anns:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if ann.get("tip_polygon"):
            pts = np.array(ann["tip_polygon"], dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        if ann.get("base_rectangle"):
            pts = np.array(ann["base_rectangle"], dtype=np.int32)
            cv2.polylines(out, [pts], True, (255, 0, 0), 2)
        if ann.get("tip_point"):
            pt = (int(ann["tip_point"][0]), int(ann["tip_point"][1]))
            cv2.circle(out, pt, 8, (0, 255, 255), -1)
        if ann.get("base_point"):
            pt = (int(ann["base_point"][0]), int(ann["base_point"][1]))
            cv2.circle(out, pt, 8, (255, 255, 0), -1)
    cv2.imwrite(str(save_path), out)


def heatmap_to_color(hm):
    """Float32 [0,1] → uint8 RGB jet colormap."""
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_u8 = (hm * 255).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_images", type=int, default=3)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    register_cotton_dataset("cotton_train", args.train_json, args.images_dir)

    # Build a config and mapper just like training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.FORMAT = "BGR"

    mapper = CottonDatasetMapper(cfg, is_train=True)

    from detectron2.data import DatasetCatalog
    dataset = DatasetCatalog.get("cotton_train")

    print(f"\n=== Diagnostic: inspecting {args.n_images} images ===\n")

    overall_stats = {
        "total_instances": 0,
        "tip_shape_count": 0,
        "tip_point_count": 0,
        "tip_none_count": 0,
        "base_shape_count": 0,
        "base_point_count": 0,
        "base_none_count": 0,
        "empty_tip_hm": 0,
        "empty_base_hm": 0,
        "empty_tip_crop": 0,
        "empty_base_crop": 0,
    }

    for img_idx in range(min(args.n_images, len(dataset))):
        record = dataset[img_idx]
        fname = Path(record["file_name"]).stem
        print(f"--- Image {img_idx}: {fname} ---")

        # First: visualize raw annotations on the original image
        raw_img = cv2.imread(record["file_name"])
        overlay_polygons(raw_img, record["annotations"],
                         out_dir / f"{img_idx:02d}_{fname}_anns.jpg")

        # Now run through the mapper (with augmentation + heatmap build)
        mapped = mapper(dict(record))  # dict() to avoid mutation
        image = mapped["image"]  # (3, H, W) tensor, post-augmentation
        instances = mapped["instances"]

        if not instances.has("gt_tip_heatmaps"):
            print(f"  No instances after filter, skipping")
            continue

        # Convert image back to displayable
        img_disp = image.permute(1, 2, 0).numpy().astype(np.uint8)
        H, W = img_disp.shape[:2]
        print(f"  Post-aug image shape: {img_disp.shape}")
        print(f"  Instances: {len(instances)}")

        boxes = instances.gt_boxes.tensor.numpy()
        tip_hms = instances.gt_tip_heatmaps.numpy()
        base_hms = instances.gt_base_heatmaps.numpy()
        tip_vis = instances.gt_tip_visibility.numpy()
        base_vis = instances.gt_base_visibility.numpy()
        occ = instances.gt_is_occluded.numpy()
        pct = instances.gt_occlusion_pct.numpy()

        # Save image with bboxes
        img_with_boxes = img_disp.copy()
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / f"{img_idx:02d}_{fname}_postaug.jpg"), img_with_boxes)

        # Aggregate heatmaps over all instances for one visual
        agg_tip = tip_hms.max(axis=0) if len(tip_hms) > 0 else np.zeros((H, W))
        agg_base = base_hms.max(axis=0) if len(base_hms) > 0 else np.zeros((H, W))

        tip_overlay = cv2.addWeighted(img_disp, 0.5, heatmap_to_color(agg_tip), 0.5, 0)
        base_overlay = cv2.addWeighted(img_disp, 0.5, heatmap_to_color(agg_base), 0.5, 0)
        cv2.imwrite(str(out_dir / f"{img_idx:02d}_{fname}_tip_hm.jpg"), tip_overlay)
        cv2.imwrite(str(out_dir / f"{img_idx:02d}_{fname}_base_hm.jpg"), base_overlay)

        # Per-instance crop diagnostic — mimic what the loss does
        for i in range(len(instances)):
            overall_stats["total_instances"] += 1
            if tip_vis[i] == 2:
                overall_stats["tip_shape_count"] += 1
            elif tip_vis[i] == 1:
                overall_stats["tip_point_count"] += 1
            else:
                overall_stats["tip_none_count"] += 1
            if base_vis[i] == 2:
                overall_stats["base_shape_count"] += 1
            elif base_vis[i] == 1:
                overall_stats["base_point_count"] += 1
            else:
                overall_stats["base_none_count"] += 1

            if tip_hms[i].max() < 1e-6:
                overall_stats["empty_tip_hm"] += 1
            if base_hms[i].max() < 1e-6:
                overall_stats["empty_base_hm"] += 1

            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            tip_crop = tip_hms[i, y1:y2, x1:x2]
            base_crop = base_hms[i, y1:y2, x1:x2]

            # Resize to per-RoI 56x56 (what the loss sees)
            tip_t = torch.from_numpy(tip_crop).unsqueeze(0).unsqueeze(0).float()
            base_t = torch.from_numpy(base_crop).unsqueeze(0).unsqueeze(0).float()
            tip_roi = F.interpolate(tip_t, size=(HEATMAP_SIZE, HEATMAP_SIZE),
                                    mode="bilinear", align_corners=False).squeeze().numpy()
            base_roi = F.interpolate(base_t, size=(HEATMAP_SIZE, HEATMAP_SIZE),
                                     mode="bilinear", align_corners=False).squeeze().numpy()

            if tip_roi.max() < 1e-6:
                overall_stats["empty_tip_crop"] += 1
            if base_roi.max() < 1e-6:
                overall_stats["empty_base_crop"] += 1

            # Save first 3 instances per image as detailed visuals
            if i < 3:
                # Composite: bbox crop | tip heatmap crop | base heatmap crop
                # all at 56x56
                bbox_img = img_disp[y1:y2, x1:x2]
                bbox_56 = cv2.resize(bbox_img, (HEATMAP_SIZE, HEATMAP_SIZE))
                tip_color = heatmap_to_color(tip_roi)
                base_color = heatmap_to_color(base_roi)

                # 4-panel: bbox, tip overlay, base overlay, tip+base alone
                tip_blend = cv2.addWeighted(bbox_56, 0.5, tip_color, 0.5, 0)
                base_blend = cv2.addWeighted(bbox_56, 0.5, base_color, 0.5, 0)
                panel = np.hstack([bbox_56, tip_blend, base_blend])
                save = out_dir / f"{img_idx:02d}_{fname}_inst{i:02d}_roi.jpg"
                cv2.imwrite(str(save), panel)

            print(f"  Inst {i}: bbox=[{x1},{y1},{x2},{y2}] "
                  f"tip_vis={tip_vis[i]} base_vis={base_vis[i]} "
                  f"occ={int(occ[i])} pct={pct[i]:.2f} "
                  f"tip_hm_max={tip_hms[i].max():.3f} base_hm_max={base_hms[i].max():.3f} "
                  f"tip_roi_max={tip_roi.max():.3f} base_roi_max={base_roi.max():.3f}")

    print("\n=== OVERALL STATS ===")
    n = max(1, overall_stats["total_instances"])
    print(f"  Total instances:        {overall_stats['total_instances']}")
    print(f"  Tip shape (v=2):        {overall_stats['tip_shape_count']} ({100*overall_stats['tip_shape_count']/n:.1f}%)")
    print(f"  Tip point (v=1):        {overall_stats['tip_point_count']} ({100*overall_stats['tip_point_count']/n:.1f}%)")
    print(f"  Tip none (v=0):         {overall_stats['tip_none_count']} ({100*overall_stats['tip_none_count']/n:.1f}%)")
    print(f"  Base shape (v=2):       {overall_stats['base_shape_count']} ({100*overall_stats['base_shape_count']/n:.1f}%)")
    print(f"  Base point (v=1):       {overall_stats['base_point_count']} ({100*overall_stats['base_point_count']/n:.1f}%)")
    print(f"  Base none (v=0):        {overall_stats['base_none_count']} ({100*overall_stats['base_none_count']/n:.1f}%)")
    print()
    print(f"  Empty TIP full heatmap:  {overall_stats['empty_tip_hm']} (should be ~tip_none count)")
    print(f"  Empty BASE full heatmap: {overall_stats['empty_base_hm']} (should be ~base_none count)")
    print(f"  Empty TIP RoI crop:      {overall_stats['empty_tip_crop']}  <-- if high, pose loss is starved")
    print(f"  Empty BASE RoI crop:     {overall_stats['empty_base_crop']} <-- if high, pose loss is starved")
    print()
    print(f"Visual outputs written to: {out_dir}/")
    print(f"  *_anns.jpg   — original image with annotation overlays")
    print(f"  *_postaug.jpg — post-augmentation image with GT bboxes")
    print(f"  *_tip_hm.jpg / *_base_hm.jpg — heatmap overlays on full image")
    print(f"  *_inst##_roi.jpg — per-instance bbox | tip RoI | base RoI (the loss view)")


if __name__ == "__main__":
    main()
