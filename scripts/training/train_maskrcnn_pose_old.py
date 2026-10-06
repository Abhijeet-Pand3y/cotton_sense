"""
train_maskrcnn_pose.py — Mask R-CNN with:
  - Instance segmentation (visible mask only)
  - Tip/base pose estimation as masked Gaussian heatmaps (per RoI)
  - Binary occlusion classification
  - Occlusion percentage regression

Transfer learning from COCO-pretrained Mask R-CNN ResNet-50-FPN.
Freezes ResNet stages 1-2, trains the rest.

Usage:
  # Train
  python train_maskrcnn_pose.py --mode train \
      --train_json ./dataset/train.json --val_json ./dataset/val.json \
      --images_dir ./frames --output ./output --epochs 40

  # Inference
  python train_maskrcnn_pose.py --mode predict \
      --weights ./output/model_final.pth \
      --images_dir ./test_frames --output ./predictions
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes, Instances, BoxMode, polygons_to_bitmask
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer


# =============================================================================
# DATASET REGISTRATION
# =============================================================================

def register_cotton_dataset(name: str, json_path: str, images_dir: str):
    """Register a COCO-format dataset with our custom per-instance fields."""

    def loader():
        with open(json_path) as f:
            coco = json.load(f)

        # Index annotations by image_id
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        records = []
        for img in coco["images"]:
            rec = {
                "file_name": str(Path(images_dir) / img["file_name"]),
                "image_id": img["id"],
                "height": img["height"],
                "width": img["width"],
            }
            objs = []
            for a in anns_by_img.get(img["id"], []):
                obj = {
                    "bbox": a["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": 0,
                    "segmentation": a["segmentation"],
                    "iscrowd": a.get("iscrowd", 0),
                    # Custom per-instance fields preserved through the mapper
                    "is_occluded": a["is_occluded"],
                    "occlusion_pct": a["occlusion_pct"],
                    "tip_polygon": a.get("tip_polygon"),
                    "base_rectangle": a.get("base_rectangle"),
                    "tip_point": a.get("tip_point"),
                    "base_point": a.get("base_point"),
                    "tip_visibility": a.get("tip_visibility", "unknown"),
                    "base_visibility": a.get("base_visibility", "unknown"),
                }
                objs.append(obj)
            rec["annotations"] = objs
            records.append(rec)
        return records

    DatasetCatalog.register(name, loader)
    MetadataCatalog.get(name).set(thing_classes=["cotton_boll"])


# =============================================================================
# CUSTOM DATASET MAPPER
# Generates masked Gaussian heatmaps from polygon shapes at load time.
# =============================================================================

HEATMAP_SIZE = 56  # per-RoI heatmap resolution


def polygon_to_mask(pts: List[Tuple[float, float]],
                    h: int, w: int) -> np.ndarray:
    """Rasterize a polygon into a binary mask of shape (h, w)."""
    if not pts or len(pts) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    poly = np.array(pts, dtype=np.int32).reshape(-1, 2)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m


def make_masked_gaussian_heatmap(
    shape_pts: Optional[List[Tuple[float, float]]],
    point_xy: Optional[Tuple[float, float]],
    img_h: int, img_w: int,
    sigma_scale: float = 0.4,
) -> Tuple[Optional[np.ndarray], int]:
    """Build a full-image masked Gaussian heatmap centered at shape centroid
    (or point), with Gaussian decay clipped to the shape boundary.

    Returns (heatmap, visibility_flag):
      visibility_flag = 2 if shape provided, 1 if only point, 0 if neither.

    The heatmap is later resampled to per-RoI HEATMAP_SIZE by Detectron2's
    bilinear sampling; we keep it at image scale here so it transforms with
    the image augmentation pipeline.
    """
    if shape_pts and len(shape_pts) >= 3:
        # Shape-based: masked Gaussian
        xs = [p[0] for p in shape_pts]
        ys = [p[1] for p in shape_pts]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

        # σ from shape "radius" (mean distance from centroid)
        dists = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in shape_pts]
        radius = max(np.mean(dists), 2.0)
        sigma = max(1.5, radius * sigma_scale)

        # Bounding region around shape with margin
        xmin = max(0, int(min(xs) - 3 * sigma))
        xmax = min(img_w, int(max(xs) + 3 * sigma) + 1)
        ymin = max(0, int(min(ys) - 3 * sigma))
        ymax = min(img_h, int(max(ys) + 3 * sigma) + 1)

        if xmax <= xmin or ymax <= ymin:
            return None, 0

        hm = np.zeros((img_h, img_w), dtype=np.float32)
        ys_grid, xs_grid = np.mgrid[ymin:ymax, xmin:xmax]
        gauss = np.exp(-((xs_grid - cx) ** 2 + (ys_grid - cy) ** 2) / (2 * sigma ** 2))
        hm[ymin:ymax, xmin:xmax] = gauss

        # Hard clip to shape boundary
        shape_mask = polygon_to_mask(shape_pts, img_h, img_w)
        hm = hm * shape_mask.astype(np.float32)
        return hm, 2

    elif point_xy is not None:
        # Guessed point: small fixed-σ Gaussian, no mask clip
        cx, cy = float(point_xy[0]), float(point_xy[1])
        sigma = 4.0
        xmin = max(0, int(cx - 3 * sigma))
        xmax = min(img_w, int(cx + 3 * sigma) + 1)
        ymin = max(0, int(cy - 3 * sigma))
        ymax = min(img_h, int(cy + 3 * sigma) + 1)
        if xmax <= xmin or ymax <= ymin:
            return None, 0
        hm = np.zeros((img_h, img_w), dtype=np.float32)
        ys_grid, xs_grid = np.mgrid[ymin:ymax, xmin:xmax]
        hm[ymin:ymax, xmin:xmax] = np.exp(
            -((xs_grid - cx) ** 2 + (ys_grid - cy) ** 2) / (2 * sigma ** 2)
        )
        return hm, 1

    return None, 0


class CottonDatasetMapper:
    """Custom mapper: standard image augmentation + builds masked Gaussian
    heatmaps for tip and base per instance."""

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.use_instance_mask = cfg.MODEL.MASK_ON

        if is_train:
            self.augs = [
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),
                T.RandomFlip(horizontal=True, vertical=False),
            ]
        else:
            self.augs = [
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                ),
            ]

    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Build heatmaps at original image size before augmentation, then
        # apply same geometric transform via Detectron2's transform API.
        annos_in = dataset_dict.pop("annotations", [])
        orig_h, orig_w = image.shape[:2]

        # Pre-build tip/base heatmaps per instance at original scale
        tip_heatmaps, base_heatmaps = [], []
        tip_vis_flags, base_vis_flags = [], []
        is_occluded_list, occlusion_pct_list = [], []

        for ann in annos_in:
            tip_hm, tip_v = make_masked_gaussian_heatmap(
                ann.get("tip_polygon"), ann.get("tip_point"), orig_h, orig_w)
            base_hm, base_v = make_masked_gaussian_heatmap(
                ann.get("base_rectangle"), ann.get("base_point"), orig_h, orig_w)
            tip_heatmaps.append(tip_hm if tip_hm is not None else np.zeros((orig_h, orig_w), dtype=np.float32))
            base_heatmaps.append(base_hm if base_hm is not None else np.zeros((orig_h, orig_w), dtype=np.float32))
            tip_vis_flags.append(tip_v)
            base_vis_flags.append(base_v)
            is_occluded_list.append(ann["is_occluded"])
            occlusion_pct_list.append(ann["occlusion_pct"])

        # Apply augmentations
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augs)(aug_input)
        image = aug_input.image
        h, w = image.shape[:2]

        # Transform heatmaps with same geometry
        new_tip_hms, new_base_hms = [], []
        for thm, bhm in zip(tip_heatmaps, base_heatmaps):
            thm_t = transforms.apply_image(thm)
            bhm_t = transforms.apply_image(bhm)
            # Re-normalize peak to 1.0 after resize (bilinear can dim peaks)
            if thm_t.max() > 0:
                thm_t = thm_t / thm_t.max()
            if bhm_t.max() > 0:
                bhm_t = bhm_t / bhm_t.max()
            new_tip_hms.append(thm_t)
            new_base_hms.append(bhm_t)

        # Transform annotations (bbox + segmentation)
        annos_out = []
        for ann in annos_in:
            annos_out.append(
                utils.transform_instance_annotations(ann, transforms, image.shape[:2])
            )

        instances = utils.annotations_to_instances(
            annos_out, image.shape[:2], mask_format="polygon"
        )

        # Attach our custom fields
        if len(instances) > 0:
            instances.gt_tip_heatmaps = torch.stack(
                [torch.from_numpy(h.astype(np.float32)) for h in new_tip_hms]
            )
            instances.gt_base_heatmaps = torch.stack(
                [torch.from_numpy(h.astype(np.float32)) for h in new_base_hms]
            )
            instances.gt_tip_visibility = torch.tensor(tip_vis_flags, dtype=torch.long)
            instances.gt_base_visibility = torch.tensor(base_vis_flags, dtype=torch.long)
            instances.gt_is_occluded = torch.tensor(is_occluded_list, dtype=torch.float32)
            instances.gt_occlusion_pct = torch.tensor(occlusion_pct_list, dtype=torch.float32)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


# =============================================================================
# CUSTOM ROI HEADS
# Extends StandardROIHeads with pose heatmap heads + occlusion heads.
# =============================================================================


class PoseHeatmapHead(nn.Module):
    """Predicts a 56x56 heatmap per RoI (one channel per keypoint)."""

    def __init__(self, in_channels: int, num_keypoints: int = 1, hidden: int = 256):
        super().__init__()
        # 4 conv -> deconv -> upsample, similar to Detectron2 keypoint head
        layers = []
        c = in_channels
        for _ in range(4):
            layers += [nn.Conv2d(c, hidden, 3, padding=1), nn.ReLU(inplace=True)]
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.deconv = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1)
        self.predictor = nn.Conv2d(hidden, num_keypoints, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.deconv(x))
        x = self.predictor(x)
        # Upsample to HEATMAP_SIZE
        x = F.interpolate(x, size=(HEATMAP_SIZE, HEATMAP_SIZE), mode="bilinear", align_corners=False)
        return x  # (N, K, H, W)


class OcclusionHead(nn.Module):
    """Predicts is_occluded (binary) and occlusion_pct (regression)."""

    def __init__(self, in_channels: int, roi_size: int = 7, hidden: int = 256):
        super().__init__()
        self.flatten_dim = in_channels * roi_size * roi_size
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.occluded_pred = nn.Linear(hidden, 1)
        self.pct_pred = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        return self.occluded_pred(x).squeeze(-1), torch.sigmoid(self.pct_pred(x).squeeze(-1))


@ROI_HEADS_REGISTRY.register()
class CottonROIHeads(StandardROIHeads):
    """Standard heads + tip heatmap + base heatmap + occlusion heads."""

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        return ret

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        in_channels = [input_shape[f].channels for f in self.in_features][0]

        # Pose heads (one channel each — single keypoint per head)
        self.tip_head = PoseHeatmapHead(in_channels)
        self.base_head = PoseHeatmapHead(in_channels)

        # Pose pooler: 14x14, same scales as mask
        self.pose_pooler = ROIPooler(
            output_size=14,
            scales=tuple(1.0 / input_shape[f].stride for f in self.in_features),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        # Occlusion head off the box features (7x7)
        self.occlusion_head = OcclusionHead(in_channels, roi_size=7)

        # Loss weights
        self.loss_w = {
            "tip": 1.0,
            "base": 1.0,
            "occluded": 0.5,
            "pct": 0.5,
        }

    def _forward_pose(self, features, instances):
        """Compute pose heatmaps for tip + base.

        During training, restrict to POSITIVE proposals (those matched to a GT
        instance) — negatives have no GT pose target. This is the same
        convention Detectron2's keypoint head uses internally.
        """
        features = [features[f] for f in self.in_features]
        if self.training:
            proposals = [x for x in instances if len(x) > 0]
            # Filter to positives only — Detectron2 marks bg with class == num_classes
            bg_idx = self.num_classes
            pos_proposals = []
            pos_boxes = []
            for pp in proposals:
                pos_mask = pp.gt_classes != bg_idx
                if pos_mask.sum() == 0:
                    continue
                pos_pp = pp[pos_mask]
                pos_proposals.append(pos_pp)
                # Pool features at the GT box, not the proposal box — GT box is
                # where the pose target Gaussian actually lives.
                pos_boxes.append(pos_pp.gt_boxes)
            boxes = pos_boxes
            instances_for_pose = pos_proposals
        else:
            boxes = [x.pred_boxes for x in instances]
            instances_for_pose = instances

        if not boxes or sum(len(b) for b in boxes) == 0:
            return {} if self.training else instances

        pooled = self.pose_pooler(features, boxes)
        tip_pred = self.tip_head(pooled)
        base_pred = self.base_head(pooled)

        if self.training:
            tip_targets, tip_weights = self._build_pose_targets(
                instances_for_pose, "gt_tip_heatmaps", "gt_tip_visibility")
            base_targets, base_weights = self._build_pose_targets(
                instances_for_pose, "gt_base_heatmaps", "gt_base_visibility")

            losses = {}
            losses["loss_tip_pose"] = self.loss_w["tip"] * pose_heatmap_loss(tip_pred, tip_targets, tip_weights)
            losses["loss_base_pose"] = self.loss_w["base"] * pose_heatmap_loss(base_pred, base_targets, base_weights)
            return losses
        else:
            tip_prob = torch.sigmoid(tip_pred)
            base_prob = torch.sigmoid(base_pred)
            for inst, t, b in zip(instances, _split_per_instance(tip_prob, instances),
                                  _split_per_instance(base_prob, instances)):
                inst.pred_tip_heatmap = t
                inst.pred_base_heatmap = b
            return instances

    def _build_pose_targets(self, proposals, hm_field: str, vis_field: str):
        """Crop GT heatmap to each POSITIVE proposal box and resize to HEATMAP_SIZE.

        Detectron2 samples 512 proposals per image (mix of positives matched to GT
        and negatives matching background). For pose, only positive proposals have
        a meaningful target — negative proposals are random background regions
        where no GT keypoint exists, so cropping the GT heatmap there gives zeros
        and starves the loss. We filter to positives via gt_classes != bg_idx.
        """
        targets, weights = [], []
        bg_idx = self.num_classes  # background class index in Detectron2

        for pp in proposals:
            if len(pp) == 0:
                continue
            if not pp.has(hm_field):
                continue
            gt_hms = getattr(pp, hm_field)
            vis = getattr(pp, vis_field)
            gt_classes = pp.gt_classes  # (N,) — bg_idx for negatives
            boxes = pp.proposal_boxes.tensor if pp.has("proposal_boxes") else pp.gt_boxes.tensor

            # IMPORTANT: also align with gt_boxes for positives — the proposal box
            # is the *sampled* region, but pose lives on the matched GT instance,
            # so we crop using gt_boxes (where the GT actually is), not proposal_boxes.
            gt_boxes_t = pp.gt_boxes.tensor

            pos_mask = gt_classes != bg_idx
            pos_indices = pos_mask.nonzero(as_tuple=True)[0].tolist()

            for i in pos_indices:
                x1, y1, x2, y2 = gt_boxes_t[i].tolist()
                x1, y1 = int(max(0, x1)), int(max(0, y1))
                x2, y2 = int(min(gt_hms.shape[-1], x2)), int(min(gt_hms.shape[-2], y2))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    targets.append(torch.zeros(1, HEATMAP_SIZE, HEATMAP_SIZE, device=gt_hms.device))
                    weights.append(0.0)
                    continue
                crop = gt_hms[i, y1:y2, x1:x2].unsqueeze(0).unsqueeze(0)
                resized = F.interpolate(crop, size=(HEATMAP_SIZE, HEATMAP_SIZE),
                                        mode="bilinear", align_corners=False).squeeze(0)
                if resized.max() > 0:
                    resized = resized / resized.max()
                targets.append(resized)
                weights.append(1.0 if vis[i].item() == 2 else 0.0)

        if not targets:
            return None, None
        return torch.stack(targets), torch.tensor(weights, device=targets[0].device)

    def _forward_occlusion(self, box_features, instances):
        """Predict occlusion from box-head pooled features."""
        if box_features is None or box_features.numel() == 0:
            return {} if self.training else instances

        occ_logits, pct_pred = self.occlusion_head(box_features)

        if self.training:
            gt_occ = torch.cat([p.gt_is_occluded for p in instances if p.has("gt_is_occluded")])
            gt_pct = torch.cat([p.gt_occlusion_pct for p in instances if p.has("gt_occlusion_pct")])
            if len(gt_occ) != len(occ_logits):
                # Sampling mismatch — skip rather than crash
                return {}
            l_occ = F.binary_cross_entropy_with_logits(occ_logits, gt_occ)
            # Only regress pct on occluded samples
            mask = gt_occ > 0.5
            if mask.sum() > 0:
                l_pct = F.smooth_l1_loss(pct_pred[mask], gt_pct[mask])
            else:
                l_pct = torch.tensor(0.0, device=occ_logits.device)
            return {
                "loss_occluded": self.loss_w["occluded"] * l_occ,
                "loss_pct": self.loss_w["pct"] * l_pct,
            }
        else:
            offset = 0
            for inst in instances:
                n = len(inst)
                inst.pred_is_occluded = torch.sigmoid(occ_logits[offset:offset + n])
                inst.pred_occlusion_pct = pct_pred[offset:offset + n]
                offset += n
            return instances

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            losses.update(self._forward_pose(features, proposals))
            # Recompute box features for occlusion head
            box_features = self.box_pooler([features[f] for f in self.box_in_features],
                                           [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features) if hasattr(self.box_head, "forward") else box_features
            # If box_head returns flattened features already, occlusion head needs raw pooled
            pooled = self.box_pooler([features[f] for f in self.box_in_features],
                                     [x.proposal_boxes for x in proposals])
            losses.update(self._forward_occlusion(pooled, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            pred_instances = self._forward_pose(features, pred_instances)
            pooled = self.box_pooler([features[f] for f in self.box_in_features],
                                     [x.pred_boxes for x in pred_instances])
            pred_instances = self._forward_occlusion(pooled, pred_instances)
            return pred_instances, {}


def _split_per_instance(tensor, instances):
    """Split a flat (N, ...) prediction tensor back into per-image lists."""
    out = []
    offset = 0
    for inst in instances:
        n = len(inst)
        out.append(tensor[offset:offset + n])
        offset += n
    return out


def pose_heatmap_loss(pred_logits, target, weights, pos_weight: float = 30.0):
    """Weighted BCE on sigmoid'd heatmap predictions.

    Sparse Gaussian targets cause MSE to be dominated by the empty surround,
    starving the loss signal. BCE with a positive-pixel weight treats this as
    per-pixel binary classification and balances foreground vs background.

    Args:
        pred_logits: (N, 1, H, W) — raw logits, no activation
        target: (N, 1, H, W) — Gaussian in [0, 1]
        weights: (N,) — per-instance loss weight (1 for visible, 0 for skip)
        pos_weight: scalar — weight on positive (Gaussian peak) pixels
    """
    if target is None or len(target) == 0:
        return pred_logits.sum() * 0.0

    target = target.to(pred_logits.device)
    weights = weights.to(pred_logits.device)

    pw = torch.tensor(pos_weight, device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, pos_weight=pw, reduction="none"
    )  # (N, 1, H, W)
    per_instance = bce.mean(dim=(1, 2, 3))  # (N,)

    w_sum = weights.sum()
    if w_sum < 1e-6:
        return pred_logits.sum() * 0.0
    return (per_instance * weights).sum() / w_sum


# =============================================================================
# CONFIG + TRAINER
# =============================================================================

def build_cfg(args) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("cotton_train",)
    cfg.DATASETS.TEST = ("cotton_val",) if args.val_json else ()

    # Transfer learning from COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.NAME = "CottonROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Disable Detectron2's built-in keypoint head — we use our pose heads
    cfg.MODEL.KEYPOINT_ON = False
    cfg.MODEL.MASK_ON = True

    # Freeze early ResNet stages
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Input resolution — shorter edge 800, max 1333
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Training schedule
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))
    cfg.SOLVER.CHECKPOINT_PERIOD = max(500, args.max_iter // 10)
    cfg.SOLVER.WARMUP_ITERS = 500

    # Evaluate on val every checkpoint period (same cadence as checkpoint + viz)
    cfg.TEST.EVAL_PERIOD = max(500, args.max_iter // 10)

    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.OUTPUT_DIR = args.output

    return cfg


class CottonTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CottonDatasetMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=CottonDatasetMapper(cfg, is_train=False))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """COCO evaluator for box AP and mask AP. Pose heads aren't COCO-format
        so they're not evaluated by AP here — we visualize them separately."""
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder,
                             tasks=("bbox", "segm"))

    def build_hooks(self):
        """Default hooks + periodic visualization on val images."""
        hooks = super().build_hooks()
        # Insert viz hook just before the periodic writer
        if self.cfg.DATASETS.TEST:
            hooks.insert(-1, PeriodicVizHook(
                cfg=self.cfg,
                period=self.cfg.SOLVER.CHECKPOINT_PERIOD,
                dataset_name=self.cfg.DATASETS.TEST[0],
                n_images=4,
                output_dir=os.path.join(self.cfg.OUTPUT_DIR, "viz"),
                model=self.model,
            ))
        return hooks


from detectron2.engine import HookBase
import detectron2.data.transforms as Transforms


class PeriodicVizHook(HookBase):
    """Every `period` iters, runs inference on a few val images and saves
    visualizations of predicted masks + pose heatmaps. Lets us see whether
    predictions look reasonable without waiting for end of training.
    """

    def __init__(self, cfg, period, dataset_name, n_images, output_dir, model):
        self.cfg = cfg.clone()
        self.period = period
        self.dataset_name = dataset_name
        self.n_images = n_images
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._model = model

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.period != 0 and next_iter != self.trainer.max_iter:
            return
        self._run_viz(next_iter)

    def _run_viz(self, step):
        from detectron2.data import DatasetCatalog
        from detectron2.modeling import build_model
        dataset = DatasetCatalog.get(self.dataset_name)
        sample = dataset[:self.n_images]

        was_training = self._model.training
        self._model.eval()

        with torch.no_grad():
            for rec in sample:
                img = cv2.imread(rec["file_name"])
                if img is None:
                    continue
                # Resize to test resolution
                aug = Transforms.ResizeShortestEdge(
                    short_edge_length=self.cfg.INPUT.MIN_SIZE_TEST,
                    max_size=self.cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
                ai = Transforms.AugInput(img)
                Transforms.AugmentationList([aug])(ai)
                im = ai.image
                ten = torch.as_tensor(im.transpose(2, 0, 1).astype(np.float32))
                inputs = [{"image": ten, "height": im.shape[0], "width": im.shape[1]}]

                outputs = self._model(inputs)[0]["instances"].to("cpu")

                # Compose viz
                viz = im.copy()
                if len(outputs) > 0:
                    boxes = outputs.pred_boxes.tensor.numpy()
                    scores = outputs.scores.numpy()
                    masks = outputs.pred_masks.numpy() if outputs.has("pred_masks") else None
                    tip_hms = outputs.pred_tip_heatmap.numpy() if outputs.has("pred_tip_heatmap") else None
                    base_hms = outputs.pred_base_heatmap.numpy() if outputs.has("pred_base_heatmap") else None
                    occ = outputs.pred_is_occluded.numpy() if outputs.has("pred_is_occluded") else None

                    # Mask overlay
                    if masks is not None:
                        for m, s in zip(masks, scores):
                            if s < 0.5:
                                continue
                            color = np.array([0, 255, 0], dtype=np.uint8)
                            viz[m] = (0.5 * viz[m] + 0.5 * color).astype(np.uint8)

                    # Draw boxes + tip/base argmax points
                    for i, (b, s) in enumerate(zip(boxes, scores)):
                        if s < 0.5:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in b]
                        cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        # tip argmax (red) and base argmax (blue) in RoI space → image
                        if tip_hms is not None:
                            yy, xx = np.unravel_index(tip_hms[i, 0].argmax(), tip_hms[i, 0].shape)
                            tx = int(x1 + (xx + 0.5) / HEATMAP_SIZE * (x2 - x1))
                            ty = int(y1 + (yy + 0.5) / HEATMAP_SIZE * (y2 - y1))
                            cv2.circle(viz, (tx, ty), 5, (0, 0, 255), -1)
                        if base_hms is not None:
                            yy, xx = np.unravel_index(base_hms[i, 0].argmax(), base_hms[i, 0].shape)
                            bx = int(x1 + (xx + 0.5) / HEATMAP_SIZE * (x2 - x1))
                            by = int(y1 + (yy + 0.5) / HEATMAP_SIZE * (y2 - y1))
                            cv2.circle(viz, (bx, by), 5, (255, 0, 0), -1)
                        # Label with score + occlusion
                        label = f"{s:.2f}"
                        if occ is not None:
                            label += f" occ={occ[i]:.2f}"
                        cv2.putText(viz, label, (x1, max(15, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                fname = Path(rec["file_name"]).stem
                save_path = Path(self.output_dir) / f"iter{step:06d}_{fname}.jpg"
                cv2.imwrite(str(save_path), viz)

        if was_training:
            self._model.train()
        print(f"[viz] Wrote {self.n_images} val visualizations to {self.output_dir}/iter{step:06d}_*.jpg")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--train_json", type=str)
    p.add_argument("--val_json", type=str)
    p.add_argument("--images_dir", type=str)
    p.add_argument("--output", type=str, default="./output")
    p.add_argument("--weights", type=str, help="Path to model_final.pth for predict mode")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--max_iter", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, help="Approx epochs (overrides max_iter)")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "train":
        register_cotton_dataset("cotton_train", args.train_json, args.images_dir)
        if args.val_json:
            register_cotton_dataset("cotton_val", args.val_json, args.images_dir)

        # Compute max_iter from epochs if given
        if args.epochs:
            with open(args.train_json) as f:
                n_images = len(json.load(f)["images"])
            iters_per_epoch = max(1, n_images // args.batch_size)
            args.max_iter = iters_per_epoch * args.epochs
            print(f"[schedule] {n_images} images, batch {args.batch_size}, "
                  f"{iters_per_epoch} iters/epoch, {args.epochs} epochs -> "
                  f"{args.max_iter} total iters")

        cfg = build_cfg(args)
        trainer = CottonTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    else:
        # Inference mode (uses standard DefaultPredictor for now)
        if not args.weights:
            raise ValueError("--weights required for predict mode")
        cfg = build_cfg(args)
        cfg.MODEL.WEIGHTS = args.weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        predictor = DefaultPredictor(cfg)

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in Path(args.images_dir).glob("*.jpg"):
            img = cv2.imread(str(img_path))
            outputs = predictor(img)
            inst = outputs["instances"].to("cpu")
            np.savez(
                out_dir / f"{img_path.stem}.npz",
                boxes=inst.pred_boxes.tensor.numpy(),
                scores=inst.scores.numpy(),
                masks=inst.pred_masks.numpy() if inst.has("pred_masks") else np.zeros(0),
                tip_heatmaps=inst.pred_tip_heatmap.numpy() if inst.has("pred_tip_heatmap") else np.zeros(0),
                base_heatmaps=inst.pred_base_heatmap.numpy() if inst.has("pred_base_heatmap") else np.zeros(0),
                is_occluded=inst.pred_is_occluded.numpy() if inst.has("pred_is_occluded") else np.zeros(0),
                occlusion_pct=inst.pred_occlusion_pct.numpy() if inst.has("pred_occlusion_pct") else np.zeros(0),
            )
            print(f"Saved {img_path.stem}.npz ({len(inst)} bolls)")


if __name__ == "__main__":
    main()
