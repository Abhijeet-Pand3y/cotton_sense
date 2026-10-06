"""
pose_eval.py — Quantitative evaluation of the cotton boll Mask R-CNN + pose model
on a validation set with full GT annotations.

Computes per-detection metrics by matching each prediction to its closest GT
instance (by bbox IoU), then aggregates:

  Detection / segmentation:
    - Mean IoU on matched instances
    - Recall / precision at IoU >= 0.5 (the matching threshold)

  Pose (only for instances with the corresponding visible polygon in GT):
    - Mean / median pixel error (tip and base separately)
    - Mean / median normalized error (error / bbox diagonal)
    - PCK at thresholds {5, 10, 20} px and {5, 10, 20}% of bbox diagonal
    - Confidence distribution
    - Calibration: error in top vs bottom confidence quartile

  Occlusion:
    - Binary accuracy, precision, recall
    - Confusion matrix
    - Pct MAE on occluded GT samples

Stratified by task: each metric is reported overall AND per task (track / front
/ back / ...). Task tag is read from the val.json annotations' "task" field
(written by combine_datasets.py).

Usage:
    python pose_eval.py \
        --weights ./output_combined/model_final.pth \
        --val_json ./dataset_combined/val.json \
        --images_dir ./combined_images \
        --output ./eval_combined
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Register custom heads so build_predictor can load the trained checkpoint
from train_maskrcnn_pose import (
    CottonROIHeads, PoseHeatmapHead, OcclusionHead, HEATMAP_SIZE,
)
from inference_viz import (
    heatmap_argmax_to_image, heatmap_confidence, build_predictor,
)


# ---------- helpers ----------

def bbox_iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def polygon_centroid(pts):
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(np.mean(xs)), float(np.mean(ys)))


def mask_iou_from_polygon(pred_mask_bool, gt_segm_polygons, h, w):
    """IoU between predicted boolean mask and GT polygon segmentation."""
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for poly_flat in gt_segm_polygons:
        if len(poly_flat) < 6:
            continue
        pts = np.array(poly_flat, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(gt_mask, [pts], 1)
    inter = np.logical_and(pred_mask_bool, gt_mask.astype(bool)).sum()
    union = np.logical_or(pred_mask_bool, gt_mask.astype(bool)).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes_xywh, gt_anns, iou_threshold=0.5):
    """Greedy matching by bbox IoU. Returns list of (gt_idx, pred_idx, iou)."""
    if not pred_boxes_xywh or not gt_anns:
        return []
    pairs = []
    for g_idx, g in enumerate(gt_anns):
        for p_idx, p_box in enumerate(pred_boxes_xywh):
            iou = bbox_iou(g["bbox"], p_box)
            if iou >= iou_threshold:
                pairs.append((iou, g_idx, p_idx))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    matches = []
    for iou, g_idx, p_idx in pairs:
        if g_idx in used_gt or p_idx in used_pred:
            continue
        used_gt.add(g_idx)
        used_pred.add(p_idx)
        matches.append((g_idx, p_idx, iou))
    return matches


# ---------- evaluation ----------

def evaluate(predictor, val_json_path, images_dir, out_dir):
    with open(val_json_path) as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    # Records keyed by task ("__all__" plus per-task)
    pose_records = defaultdict(list)
    occlusion_records = defaultdict(list)
    detection_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "matched": 0, "iou_sum": 0.0, "mask_iou_sum": 0.0, "mask_iou_n": 0})

    for img_id, img_info in images_by_id.items():
        img_path = Path(images_dir) / img_info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] couldn't read {img_path}")
            continue
        h, w = img.shape[:2]

        gt_anns = anns_by_image[img_id]
        task = gt_anns[0].get("task", "untagged") if gt_anns else img_info.get("task", "untagged")

        detection_stats["__all__"]["gt"] += len(gt_anns)
        detection_stats[task]["gt"] += len(gt_anns)

        outputs = predictor(img)
        inst = outputs["instances"].to("cpu")
        n_pred = len(inst)
        detection_stats["__all__"]["pred"] += n_pred
        detection_stats[task]["pred"] += n_pred

        if n_pred == 0:
            continue

        pred_boxes = inst.pred_boxes.tensor.numpy()
        pred_boxes_xywh = [
            [pred_boxes[i, 0], pred_boxes[i, 1],
             pred_boxes[i, 2] - pred_boxes[i, 0], pred_boxes[i, 3] - pred_boxes[i, 1]]
            for i in range(len(pred_boxes))
        ]

        tip_hms = inst.pred_tip_heatmap.numpy() if inst.has("pred_tip_heatmap") else None
        base_hms = inst.pred_base_heatmap.numpy() if inst.has("pred_base_heatmap") else None
        pred_masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        occ = inst.pred_is_occluded.numpy() if inst.has("pred_is_occluded") else None
        pct = inst.pred_occlusion_pct.numpy() if inst.has("pred_occlusion_pct") else None
        scores = inst.scores.numpy()

        matches = match_predictions_to_gt(pred_boxes_xywh, gt_anns, iou_threshold=0.5)

        for g_idx, p_idx, iou in matches:
            g = gt_anns[g_idx]
            g_task = g.get("task", task)
            for key in ("__all__", g_task):
                detection_stats[key]["matched"] += 1
                detection_stats[key]["iou_sum"] += iou

            # Mask IoU on this matched pair
            if pred_masks is not None and "segmentation" in g:
                m_iou = mask_iou_from_polygon(pred_masks[p_idx], g["segmentation"], h, w)
                for key in ("__all__", g_task):
                    detection_stats[key]["mask_iou_sum"] += m_iou
                    detection_stats[key]["mask_iou_n"] += 1

            x1, y1 = int(pred_boxes[p_idx, 0]), int(pred_boxes[p_idx, 1])
            x2, y2 = int(pred_boxes[p_idx, 2]), int(pred_boxes[p_idx, 3])
            bbox_diag = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

            if g.get("tip_polygon") is not None and tip_hms is not None:
                gt_tip = polygon_centroid(g["tip_polygon"])
                if gt_tip:
                    px, py, _ = heatmap_argmax_to_image(tip_hms[p_idx, 0], x1, y1, x2, y2)
                    err = float(np.hypot(px - gt_tip[0], py - gt_tip[1]))
                    conf, conc = heatmap_confidence(tip_hms[p_idx, 0])
                    rec = {"kind": "tip", "pixel_error": err,
                           "norm_error": err / max(1.0, bbox_diag),
                           "bbox_diag": bbox_diag, "confidence": conf, "concentration": conc,
                           "iou": iou, "det_score": float(scores[p_idx]), "task": g_task}
                    pose_records["__all__"].append(rec)
                    pose_records[g_task].append(rec)

            if g.get("base_rectangle") is not None and base_hms is not None:
                gt_base = polygon_centroid(g["base_rectangle"])
                if gt_base:
                    px, py, _ = heatmap_argmax_to_image(base_hms[p_idx, 0], x1, y1, x2, y2)
                    err = float(np.hypot(px - gt_base[0], py - gt_base[1]))
                    conf, conc = heatmap_confidence(base_hms[p_idx, 0])
                    rec = {"kind": "base", "pixel_error": err,
                           "norm_error": err / max(1.0, bbox_diag),
                           "bbox_diag": bbox_diag, "confidence": conf, "concentration": conc,
                           "iou": iou, "det_score": float(scores[p_idx]), "task": g_task}
                    pose_records["__all__"].append(rec)
                    pose_records[g_task].append(rec)

            if occ is not None:
                rec = {
                    "gt_is_occluded": int(g["is_occluded"]),
                    "pred_is_occluded": float(occ[p_idx]),
                    "gt_pct": float(g.get("occlusion_pct", 0.0)),
                    "pred_pct": float(pct[p_idx]) if pct is not None else None,
                }
                occlusion_records["__all__"].append(rec)
                occlusion_records[g_task].append(rec)

    # ---------- aggregate metrics per group ----------
    def summarize_pose(records, kind):
        sub = [r for r in records if r["kind"] == kind]
        if not sub:
            return None
        errs = np.array([r["pixel_error"] for r in sub])
        norm_errs = np.array([r["norm_error"] for r in sub])
        confs = np.array([r["confidence"] for r in sub])
        order = np.argsort(-confs)
        q = max(1, len(errs) // 4)
        return {
            "n": len(sub),
            "mean_pixel_error": float(errs.mean()),
            "median_pixel_error": float(np.median(errs)),
            "mean_normalized_error": float(norm_errs.mean()),
            "median_normalized_error": float(np.median(norm_errs)),
            "mean_confidence": float(confs.mean()),
            "median_confidence": float(np.median(confs)),
            "pck": {
                "pck@5px":  float((errs <= 5).mean()),
                "pck@10px": float((errs <= 10).mean()),
                "pck@20px": float((errs <= 20).mean()),
                "pck@5pct_diag":  float((norm_errs <= 0.05).mean()),
                "pck@10pct_diag": float((norm_errs <= 0.10).mean()),
                "pck@20pct_diag": float((norm_errs <= 0.20).mean()),
            },
            "calibration": {
                "err_top_quartile_conf":    float(errs[order[:q]].mean()),
                "err_bottom_quartile_conf": float(errs[order[3*q:]].mean()),
            },
        }

    def summarize_occlusion(records):
        if not records:
            return None
        gt_occ = np.array([r["gt_is_occluded"] for r in records])
        pred_occ = (np.array([r["pred_is_occluded"] for r in records]) > 0.5).astype(int)
        tp = int(((pred_occ == 1) & (gt_occ == 1)).sum())
        tn = int(((pred_occ == 0) & (gt_occ == 0)).sum())
        fp = int(((pred_occ == 1) & (gt_occ == 0)).sum())
        fn = int(((pred_occ == 0) & (gt_occ == 1)).sum())
        out = {
            "n": len(records),
            "binary_accuracy": (tp + tn) / max(1, len(records)),
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "precision": tp / max(1, tp + fp),
            "recall":    tp / max(1, tp + fn),
        }
        # Pct MAE on occluded GT
        gt_pcts = [r["gt_pct"] for r in records if r["gt_is_occluded"] == 1 and r["pred_pct"] is not None]
        pred_pcts = [r["pred_pct"] for r in records if r["gt_is_occluded"] == 1 and r["pred_pct"] is not None]
        if gt_pcts:
            out["pct_mae_on_occluded"] = float(np.mean(np.abs(np.array(gt_pcts) - np.array(pred_pcts))))
        return out

    report = {}
    for group, stats in detection_stats.items():
        gt = stats["gt"]; pred = stats["pred"]; matched = stats["matched"]
        det = {
            "gt_instances": gt,
            "predicted": pred,
            "matched_iou50": matched,
            "recall_iou50": matched / max(1, gt),
            "precision_iou50": matched / max(1, pred),
            "mean_box_iou_matched": stats["iou_sum"] / max(1, matched),
        }
        if stats["mask_iou_n"] > 0:
            det["mean_mask_iou_matched"] = stats["mask_iou_sum"] / stats["mask_iou_n"]
        group_report = {"detection": det}
        for kind in ("tip", "base"):
            s = summarize_pose(pose_records.get(group, []), kind)
            if s is not None:
                group_report[kind] = s
        occ_s = summarize_occlusion(occlusion_records.get(group, []))
        if occ_s is not None:
            group_report["occlusion"] = occ_s
        report[group] = group_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_report.json", "w") as f:
        def _coerce(o):
            if isinstance(o, dict):
                return {k: _coerce(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_coerce(x) for x in o]
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        json.dump(_coerce(report), f, indent=2)

    # ---------- pretty print ----------
    def fmt_group(name, r):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        d = r["detection"]
        print(f"\n  DETECTION")
        print(f"    GT instances:        {d['gt_instances']}")
        print(f"    Predicted:           {d['predicted']}")
        print(f"    Matched @ IoU>=0.5:  {d['matched_iou50']}")
        print(f"    Recall:              {d['recall_iou50']:.3f}")
        print(f"    Precision:           {d['precision_iou50']:.3f}")
        print(f"    Mean box IoU:        {d['mean_box_iou_matched']:.3f}")
        if "mean_mask_iou_matched" in d:
            print(f"    Mean mask IoU:       {d['mean_mask_iou_matched']:.3f}")
        for kind in ("tip", "base"):
            if kind not in r: continue
            k = r[kind]
            print(f"\n  {kind.upper()} POSE  (n={k['n']})")
            print(f"    Pixel error:    mean={k['mean_pixel_error']:.2f}  median={k['median_pixel_error']:.2f}")
            print(f"    Norm  error:    mean={k['mean_normalized_error']:.3f}  median={k['median_normalized_error']:.3f}")
            print(f"    Confidence:     mean={k['mean_confidence']:.3f}  median={k['median_confidence']:.3f}")
            p = k["pck"]
            print(f"    PCK@5px={p['pck@5px']:.2f}  PCK@10px={p['pck@10px']:.2f}  PCK@20px={p['pck@20px']:.2f}")
            print(f"    PCK@5%diag={p['pck@5pct_diag']:.2f}  PCK@10%={p['pck@10pct_diag']:.2f}  PCK@20%={p['pck@20pct_diag']:.2f}")
            c = k["calibration"]
            print(f"    Calibration:    top-conf err={c['err_top_quartile_conf']:.2f}  bottom-conf err={c['err_bottom_quartile_conf']:.2f}")
        if "occlusion" in r:
            o = r["occlusion"]; cm = o["confusion"]
            print(f"\n  OCCLUSION  (n={o['n']})")
            print(f"    Binary accuracy: {o['binary_accuracy']:.3f}")
            print(f"    Precision:       {o['precision']:.3f}")
            print(f"    Recall:          {o['recall']:.3f}")
            print(f"    TP={cm['tp']}  TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}")
            if "pct_mae_on_occluded" in o:
                print(f"    Pct MAE on occluded: {o['pct_mae_on_occluded']:.3f}")

    # Print overall first, then per-task
    if "__all__" in report:
        fmt_group("OVERALL", report["__all__"])
    for group in sorted([g for g in report.keys() if g != "__all__"]):
        fmt_group(f"TASK = {group}", report[group])

    # Plots for overall
    if pose_records.get("__all__"):
        for kind in ("tip", "base"):
            sub = [r for r in pose_records["__all__"] if r["kind"] == kind]
            if not sub:
                continue
            errs = np.array([r["pixel_error"] for r in sub])
            norm_errs = np.array([r["norm_error"] for r in sub])
            confs = np.array([r["confidence"] for r in sub])
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(errs, bins=30, color="steelblue")
            axes[0].set_title(f"{kind}: pixel error")
            axes[0].set_xlabel("pixels")
            axes[1].hist(norm_errs, bins=30, color="seagreen")
            axes[1].set_title(f"{kind}: normalized error")
            axes[1].set_xlabel("fraction of bbox diag")
            axes[2].hist(confs, bins=30, color="indianred")
            axes[2].set_title(f"{kind}: confidence")
            axes[2].set_xlabel("confidence")
            plt.tight_layout()
            plt.savefig(out_dir / f"{kind}_distributions.png", dpi=100)
            plt.close()

            plt.figure(figsize=(6, 5))
            plt.scatter(confs, errs, alpha=0.3, s=10)
            plt.xlabel(f"{kind} predicted confidence")
            plt.ylabel(f"{kind} pixel error")
            plt.title(f"{kind}: confidence vs error")
            plt.tight_layout()
            plt.savefig(out_dir / f"{kind}_conf_vs_err.png", dpi=100)
            plt.close()

    print(f"\nFull JSON: {out_dir/'eval_report.json'}")
    print(f"Plots: {out_dir}/*.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--val_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score_thresh", type=float, default=0.5)
    args = p.parse_args()

    predictor = build_predictor(args.weights, score_thresh=args.score_thresh)
    evaluate(predictor, args.val_json, args.images_dir, args.output)


if __name__ == "__main__":
    main()