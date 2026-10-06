# How the pipeline works

Status as of 2026-10-06. Everything below was checked against the code in
`scripts/` and the files in `datasets/`, `models/`, `evaluation/` and `findings/`.
Line references are to the files as they are now. Where something is an inference
rather than read directly from code or data, it says so.

There are two subsystems:

1. **Detection, segmentation and pose**: a Mask R-CNN that finds each cotton boll
   and predicts its visible mask, tip and base points, and occlusion.
2. **Depth and size**: Depth Anything 3 (DA3) depth plus a fitted calibration turns
   a boll's tip-to-base pixel length into millimetres.

```
GoPro video
  └─ extract frames ──► CVAT annotation (XML) ──► COCO JSON ──► combined dataset
                                                                    │
                                                    train Mask R-CNN + pose/occlusion heads
                                                                    │
video / frames ──► inference_viz.py ──► per-frame JSON ──► track_detections.py ──► tracked JSON
                                                                                       │
frames ──► run_da3_depth.py ──► raw depth .npy ──────────────────────────► apply_boll_sizes.py ──► mm per boll
                       │                                                               ▲
                       └──► fit_gamma_from_annotations.py (tagged bolls + tape GT) ──► calibration.json
```

---

## 1. Frame extraction (`scripts/extraction/`)

There are two extractors, for the two CVAT annotation modes.

**`extract_frames.py`** picks diverse frames for shape-mode annotation:
1. Samples `5 × num_frames` evenly spaced candidates between `--start` and `--end`
   (line 103). The docstring says 3×; the code uses 5×.
2. Drops frames with mean grey brightness outside 30–240 or with Laplacian variance
   (sharpness) below `--min_sharpness` (default 50). If too few pass, it lowers the
   sharpness bar to the Nth-best value.
3. Selects greedily. It starts from the sharpest frame, then repeatedly adds the
   candidate that maximises
   `w·(min χ² colour-histogram distance to the chosen frames) + (1−w)·(min time distance / duration)`,
   with `w = --diversity_weight` (default 0.5).
4. Fills gaps: while any gap between chosen frames exceeds `--max_gap` (3 s), it adds
   the candidate nearest the gap's midpoint. It tries quality-passing frames first and
   falls back to any candidate. The final count can therefore exceed `--num_frames`.
5. Optionally rotates (`--rotate 90|180|270`) for portrait GoPro footage, saves
   JPEG q95 as `frame_<idx:06d>_t<sec>p<centi>s.jpg`, and writes
   `extraction_metadata.json`.

**`extract_frames_uniform.py`** samples at a fixed interval (`--interval` or
`--num_frames`) over `[--start, --end]`, with an optional sharpness filter. It uses
the same naming and rotation. Even spacing is what makes CVAT track-mode polygon
interpolation accurate.

The remaining extraction scripts handle other sources: `extract_ros2_bag.py` and
`extract_for_annotation_rgbd.py` for RealSense/ROS bags, plus small single-image
helpers.

---

## 2. Annotation schema (CVAT "for images 1.1" XML)

`cvat_to_coco_v4.py` reads these labels (lines 150–180). Every shape needs a
`boll_id` attribute. Shapes with no id or id `0` are **skipped** and counted.

| CVAT label | Shape | Meaning |
|---|---|---|
| `cotton_boll` | polygon | Boll outline. `annotation_type` ∈ `whole_visible`, `visible_parts`, `whole_estimate` |
| `tip_polygob` (sic) | 3-pt polygon | Visible tip region |
| `base_rectangle` | 4-pt polygon | Visible base region |
| `tip` / `base` | point | Guessed tip/base location when not visible |

`cotton_boll` attributes used: `tip_visibility`, `base_visibility` and `occluded_by` ∈
{none, leaf, stem, other_boll, other, frame_edge}. These are read only from the
`whole_visible` / `whole_estimate` polygon.

`boll_id` groups shapes into one physical boll within a frame. It is copied into the
COCO record as metadata. **It is never a training target.**

---

## 3. CVAT XML → COCO (`scripts/dataset/cvat_to_coco_v4.py`)

This produces one COCO annotation per `boll_id` per frame (lines 182–244):

- **Mask (`segmentation`)**: the polygons the model learns to segment. The model
  segments the *visible* boll only.
  - `whole_visible` present → those polygons. `is_occluded = (occluded_by != none)`,
    `occlusion_pct = 0`.
  - `visible_parts` + `whole_estimate` → mask is the visible parts, `is_occluded = 1`,
    `occlusion_pct = 1 − area(∪visible_parts)/area(∪whole_estimate)` (shapely union),
    clipped to [0, 1].
  - `visible_parts` only → `is_occluded = 1`, `occlusion_pct = 0` (unknown).
  - Anything else is skipped (`no_mask_polygon`).
- `bbox` is the box around the visible polygons. Boxes ≤ 1 px wide or tall are dropped.
- `occ_class_id` is the index into
  `OCC_CLASSES = [none, leaf, stem, other_boll, other, frame_edge]` (line 52). Unknown
  strings map to 0.
- Tip/base are carried through as `tip_polygon`, `base_rectangle`, `tip_point` and
  `base_point`, plus the visibility strings.
- Images that end up with no instances are dropped.
- `--val_split` makes a **random image-level split** (seed 42).

`category_id` is written as 1, and the training loader forces it to 0. There is one
class, `cotton_boll`.

---

## 4. Combining camera angles (`scripts/dataset/`)

Three combiners exist, and they produce differently-labelled datasets:

| Script | Angles | `task` / path prefix | Val split |
|---|---|---|---|
| `combine_datasets.py` | fixed: track (right_down), front, back | `track/`, `front/`, `back/` | random per task, `--val_split` |
| `combine_datasets_sequential.py` | same three | `track/`, `front/`, `back/` | last fraction of each task (temporal tail); `--val_random` per task |
| `combine_datasets_dynamic.py` | every subfolder of `--combine_set` | folder name, e.g. `left_down/` | `--val_count` images total (default 8), strided across angles |

All three call `convert_xml` from `cvat_to_coco_v4.py`, renumber image and annotation
ids globally, and prefix `file_name` with the angle. **They do not copy images**
(`_dynamic` copies only with `--stage_images`). At train time `--images_dir` must be a
folder containing one subfolder per prefix.

What the existing datasets actually contain (read from their JSON):

| Dataset | train imgs / anns | val imgs / anns | angles (`task`) | has `occ_class_id` |
|---|---|---|---|---|
| `datasets/combined_all/output_coco_dir` | 278 / 1991 | 68 / 285 | right_down, front_down, back_down | **no** |
| `datasets/combined_all_v2` | 539 / 2953 | 8 / 41 | + left_down | yes |
| `datasets/combined_all_v3` | 539 / 2975 | 8 / 41 | + left_down | yes |
| `datasets/combined_all_v4` | 688 / 3731 | 10 / 47 | + left_down, rgbd_1 | yes |

`combined_all` uses task names `right_down/front_down/back_down` and has no
`occ_class_id`. That is **not** what the current `combine_datasets.py` writes (it
writes `track/front/back` and includes `occ_class_id`), so `combined_all` was built by
an earlier version. The v2–v4 sets match `combine_datasets_dynamic.py` with
`--val_count` (8 or 10 val images).

---

## 5. Model (`scripts/training/train_maskrcnn_pose.py`)

### 5.1 Base network
`build_cfg` (line 690) loads Detectron2's `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x`:
- **Backbone**: ResNet-50 + FPN, COCO-pretrained weights, `FREEZE_AT = 2` (stem and
  res2 frozen).
- **RPN**: standard Detectron2 region proposals.
- **Box head**: standard. Classifies `cotton_boll` vs background and regresses boxes
  (**object detection**).
- **Mask head**: standard 28×28 per-RoI mask. Predicts the **visible-region
  segmentation**.
- `NUM_CLASSES = 1`, `KEYPOINT_ON = False` (Detectron2's own keypoint head is unused),
  `MASK_ON = True`.
- Input: train shortest edge randomly 640 or 800 (max 1333) with horizontal flip.
  Test shortest edge 800.

### 5.2 Custom heads: `CottonROIHeads(StandardROIHeads)`
Registered in `ROI_HEADS_REGISTRY` (line 393), selected via `cfg.MODEL.ROI_HEADS.NAME`.

**Tip head and base head** (`PoseHeatmapHead`, line 341): two independent heads with
the same structure. Each runs 14×14 ROIAlign features → 4× (3×3 conv 256 + ReLU) →
deconv ×2 → 1×1 conv → bilinear resize to **56×56**, giving one heatmap channel per
head.
- The RoI used for pose is the box **expanded 1.5× about its centre**
  (`POSE_BOX_SCALE`, line 123). This lets tip/base points just outside the tight
  visible-mask box be represented. The same scale must be used in training, GT
  cropping and inference.
- Training pools features at the **GT box** (expanded) of positive proposals only
  (line 476). Inference uses the predicted box (expanded).

**Occlusion head** (`OcclusionHead`, line 365): flattens the 7×7 box-pooler features →
FC 256 → FC 256 and produces three outputs:
- `is_occluded` logit (binary)
- `occlusion_pct` (sigmoid, regression)
- `occ_class` (6-way logits over `OCC_CLASSES`)

### 5.3 Targets: `CottonDatasetMapper` (line 227)
For each instance, built at full image resolution before augmentation, then
transformed with the image:
- If a tip/base **polygon** exists: a Gaussian centred on the polygon centroid,
  σ = max(1.5, 0.4 × mean vertex distance), **hard-clipped to the polygon**.
  Visibility flag = 2.
- Else, if only a guessed **point** exists: a Gaussian with σ = 4 px and no clip.
  Flag = 1.
- Else nothing. Flag = 0.

Peaks are renormalised to 1 after resizing. `_build_pose_targets` (line 511) crops
each heatmap to the expanded GT box, resizes it to 56×56, and gives it **weight 1
only if the flag is 2**.

> Consequence: guessed (point-only) tips and bases contribute **nothing** to the pose
> loss. The pose heads learn only from `tip_polygob` / `base_rectangle` shapes.

### 5.4 Losses
| Loss | Definition | Weight |
|---|---|---|
| `loss_cls`, `loss_box_reg`, `loss_rpn_*` | Detectron2 standard | 1 |
| `loss_mask` | Detectron2 per-pixel BCE | 1 |
| `loss_tip_pose`, `loss_base_pose` | per-pixel BCE with `pos_weight = 30`, mean per instance, weighted mean over instances (line 655) | 1.0 each |
| `loss_occluded` | BCE on `is_occluded` | 0.5 |
| `loss_pct` | smooth-L1 on `occlusion_pct`, occluded GT only | 0.5 |
| `loss_occ_class` | weighted CE, class weights ∝ √(total/count) normalised to mean 1, counts hard-coded `[2168, 500, 132, 108, 26, 60]` (line 426) | 0.3 |

The occlusion losses run on *all* sampled proposals, not only positives (line 630).
Detectron2's `label_and_sample_proposals` copies `gt_*` fields from each proposal's
best-matching GT, including for background proposals. Background RoIs are therefore
trained toward the occlusion labels of whatever boll they overlap most. This follows
from Detectron2's sampling code and is not shown in this repo.

### 5.5 Training loop
- `CottonTrainer(DefaultTrainer)` uses the custom mapper and a `COCOEvaluator`
  (bbox + segm AP) on the val set every `max(500, max_iter/10)` iterations.
- `PeriodicVizHook` writes 4 val images with masks, boxes and tip/base argmax to
  `<output>/viz/` at every checkpoint.
- `--epochs E` sets `max_iter = (n_train_images // batch) × E`. LR drops ×0.1 at 70%
  and 90%, with 500 warmup iterations. A checkpoint is saved every 500 iterations, and
  `--resume` continues from the last one.
- `--mode predict` writes raw `.npz` per image. In practice `inference_viz.py` is used
  instead (§6).

### 5.6 Training-script variants
| Script | Difference from canonical |
|---|---|
| `train_maskrcnn_pose_old.py` | no `POSE_BOX_SCALE` expansion, no `occ_class` head |
| `train_maskrcnn_pose_big_box.py` | same model as canonical; lacks `--resume` / `--checkpoint_period` |
| `train_maskrcnn_pose_gt_tipbase.py` | adds an optional geometric pose loss (`--geo_loss_weight`, soft-argmax direction/length vs mask), off by default |

### 5.7 Trained models (from `models/*/metrics.json`)
| Folder | Date | Iters | Has `loss_occ_class` | Last val bbox AP / segm AP |
|---|---|---|---|---|
| `combined_all_angles` | 2026-06-03 | 5560 | **no** | 62.1 / 67.6 (68 val imgs) |
| `combined_all_v2` | 2026-06-24 | 10760 | yes | 78.4 / 76.8 (8 val imgs) |
| `combined_all_v2_posebox` | 2026-06-25 | 10760 | yes | 79.8 / 77.6 |
| `combined_all_v3` | 2026-06-30 | 10760 | yes | 78.9 / 78.3 |
| `combined_all_v4` | 2026-07-08 | 13760 | yes | 79.3 / 78.4 (10 val imgs) |
| `combined_all_v5_geo` | 2026-07-09 | 13760 | yes | 80.2 / 79.2 |
| `right_down_300frames` | 2026-05-26 | 4800 | no | 91.4 / 86.6 |

**The AP values are not comparable across rows.** Each row is scored on a different
val set, and v2 onward on only 8–10 images.

---

## 6. Inference (`scripts/inference/inference_viz.py`)

Run with `PYTHONPATH=scripts/training` so that `CottonROIHeads` is registered.

1. `build_predictor` builds the same config with `SCORE_THRESH_TEST = --score_thresh`
   (default 0.5) and uses Detectron2 `DefaultPredictor`. Boxes and masks come back in
   original-image pixels.
2. Input is an image folder, or a video sampled by `--every_n_frames` > `--fps` >
   `--interval` (default 0.5 s). A video can be split across GPUs with `--shard i/N`.
   Video frames are saved to `frames/` so the viewer has images.
3. Optional **Reinhard LAB colour transfer** (`--color_src`/`--color_tgt` stats from
   `domain_color_stats.py`) is applied before the model runs. It is used to make
   RealSense frames look like GoPro frames.
4. Per detection (`process_image`, line 226):
   - `bbox` [x, y, w, h], `det_conf`
   - `mask_poly`: the **largest** external contour of the mask, simplified by
     `approxPolyDP` (ε = 1.5 px). Other mask fragments are discarded.
   - `kpts` = `[[tx, ty, tip_peak], [bx, by, base_peak]]`. This is the heatmap
     **argmax** mapped through the 1.5× expanded box. `*_conc` = peak / mean
     (sharpness).
   - `orientation_deg = atan2(ty−by, tx−bx)`
   - `occ_class` and its softmax confidence, `is_occluded` (> 0.5), `occlusion_pct`,
     and `visfrac = 1 − occlusion_pct`
5. Outputs: `json/<stem>.json` (per-frame list), `viz/<stem>.jpg`, and
   `detections_all.json` (`[{frame, src, bolls}]`).

`inference_viz_softmax.py` and `inference_viz_video.py` are variants of the same
script.

---

## 7. Tracking (`scripts/inference/track_detections.py`)

- **Matching**: each frame's detections with `det_conf ≥ --min_score` (0.3) are
  matched to live tracks by **bbox IoU**. It uses Hungarian assignment via scipy, or
  greedy matching if scipy is missing, and accepts pairs with IoU ≥ `--iou_thresh`
  (0.3). There is no motion model: a track's position is simply its last box.
- **Births and deaths**: an unmatched detection starts a new id from a counter. A
  track unmatched for more than `--max_age` (5) frames is dropped. Detections below
  `min_score` get `track_id = null`. `--min_track_len` can null out short tracks.
- **Smoothing**: per track, tip and base x/y are filtered separately with a centred
  median (default window 5, so past and future frames), optionally followed by an
  EMA. Missing keypoints are gaps rather than zeros. Orientation is then recomputed.
- Output has the same shape plus `track_id`, as a combined JSON and/or per-frame
  JSONs.

`track_id` has **no link to the physical tag (`boll_id`)**. A boll that is occluded
for more than 5 frames, or whose box moves enough to drop IoU below 0.3, gets a new id.

---

## 8. Evaluation (`scripts/inference/pose_eval.py`)

What it computes:
- **Matching**: greedy, highest bbox IoU first, IoU ≥ 0.5, one-to-one. This is not
  COCOeval.
- **Detection**: recall = matched / GT, precision = matched / predicted, mean box IoU
  of matched pairs.
- **Segmentation**: mean mask IoU of the predicted mask against the rasterised GT
  visible polygons.
- **Pose**: only for GT with a tip polygon or base rectangle (the centroid is the GT
  point). It reports pixel error, error / predicted-box diagonal, and PCK at
  5/10/20 px and 5/10/20% of the diagonal. Calibration compares the error in the top
  vs bottom confidence quartile.
- **Occlusion**: binary accuracy, precision, recall, confusion matrix, and the MAE of
  `occlusion_pct` on occluded GT.
- Everything is reported overall and per `task`, and written to `eval_report.json`
  plus histograms.

**The script does not run as it stands.** It imports `heatmap_confidence` from
`inference_viz`, which no longer defines it. It also calls
`heatmap_argmax_to_image(hm, x1, y1, x2, y2)` and unpacks 3 values, while the current
signature is `(heatmap, box)` and returns 4. It also needs both `scripts/training` and
`scripts/inference` on `PYTHONPATH`.

The one saved report, `evaluation/combined_all/eval_report.json` (2026-06-03), was
produced by an earlier version. Its 285 GT instances match the `combined_all` val set,
and its date matches `models/combined_all_angles`. Overall figures from that file:

| Metric | Value |
|---|---|
| Detection recall / precision @ IoU 0.5 | 280/285 = 0.982 / 280/331 = 0.846 |
| Mean mask IoU (matched) | 0.846 |
| Tip: n, median px error, PCK@10% diag | 272, 14.2 px, 0.735 |
| Base: n, median px error, PCK@10% diag | 205, 14.7 px, 0.946 |
| Occlusion binary accuracy (n = 280) | 0.929 |

Caveat: the `combined_all` val split is a *random* split of frames. 240 of its 285 val
instances come from the right_down track, which is consecutive uniformly-sampled video,
so val frames have near-identical neighbours in train. These numbers are likely
optimistic for new footage. `combine_datasets_sequential.py` exists to avoid this.

---

## 9. Depth and size (`scripts/depth/`, run in the `da3` env)

### 9.1 Depth maps: `run_da3_depth.py`
- Model `depth-anything/DA3METRIC-LARGE`, **one image per call** (monocular, no
  multi-view fusion).
- `--crop_top_frac` / `--crop_bottom_frac` crop the frame before inference. The pixel
  offsets go into `crop_info.json`, written to both the run folder and `depth/`. The
  standard is `--crop_top_frac 0.15`, which removes round(5312 × 0.15) = 797 px of
  greenhouse roof.
- Without `--focal_px` the saved `.npy` is the **raw network output** (float32,
  arbitrary units). That is what the calibration expects. With `--focal_px` it saves
  `focal × raw / 300`.

### 9.2 Per-boll depth sampling: `sample_boll_depth.py` → `analyze_depth_csv.py`
- Boll depth is the median over the detection mask after erosion
  (`--erode_frac 0.15`), which removes mixed boundary pixels.
- Tip/base depth is the median in a `--kpt_win` (25 px) window, keeping only pixels
  within `--tol_pct` (5%) of the boll depth. It reports `kept_pct` and `dz` (tip − base).
- Coordinates are scaled from image to depth resolution, with x and y handled
  separately, and shifted by the crop offset.
- `analyze_depth_csv.py` reports gate health, per-track depth spread, whether
  averaging k frames shrinks the spread by √k, and `dz` against the noise floor.

### 9.3 Calibration: `fit_gamma_from_annotations.py`
Model: DA3 depth is compressed, `z_raw ∝ Z^γ`. With pinhole `S = tb_px · Z / f`:

```
log S_true − log tb_px = (1/γ) · log z_raw + C
size_mm = tb_px · z_raw^(1/γ) · exp(C)
```

Procedure:
1. Parse `left_down.xml`. For every `cotton_boll` with a tip (`tip_polygob` centroid
   or `tip` point), a base, and a tape-measured height in `cotton_ground_truth.csv`
   (joined on `boll_id` = tag), compute `tb_px`.
2. `z_raw` = median raw depth inside the eroded *annotated* polygon. The crop offset
   is auto-read from `crop_info.json`.
3. Keep only clean instances: tip and base `visible`, `whole_visible`,
   `occluded_by = none`. Optionally keep only the middle N frames (`--middle`).
4. Aggregate to **one point per boll** (median over ≥ 3 instances) and fit the line
   by least squares.
5. γ is chosen by **leave-one-boll-out**: scan γ from 0.40 to 3.00, refit C without
   each boll in turn, and minimise the mean |size error|. C is then refit on all bolls.

Current `findings/cotton_tag_calibration/calibration.json`: γ = 0.70,
C = −1.2571, scale = exp(C) = 0.284477, **4 bolls**, 72 instances.
`calibration_mid30.json` holds identical numbers plus the `depth_variant` ("top 15%
cropped") and `frames` fields. The active `calibration.json` does **not** record
which depth variant it was fitted on.

### 9.4 Applying sizes: `apply_boll_sizes.py`
- For each tracked detection: `tb_px` = distance between the **predicted, smoothed**
  tip and base. `z_raw` = median raw depth in the eroded **predicted** mask polygon.
  `size_mm = tb_px · z_raw^(1/γ) · scale`.
- Per track it keeps frames with `det_conf ≥ 0.9`, `min(tip, base conf) ≥ 0.5` and
  `visfrac ≥ 0.8`, relaxing the visfrac filter if fewer than 5 remain. It reports the
  median size, SD% and quality (good ≥ 20 frames, fair ≥ 10, else weak).
- `--crop_top_px` is **not** auto-detected here, unlike the fitting script, and must be
  passed by hand (797 for crop15 maps). Image size defaults to 2988 × 5312.
- Writes `per_frame.csv` and `per_boll.csv`.

The calibration is fitted on *annotated* tip/base centroids and *annotated* polygons,
but applied to *predicted* keypoints and *predicted visible* masks. Any systematic
keypoint bias in the model goes straight into the size estimate.

### 9.5 Viewer depth: `make_viewer_metric_depth.py`
Writes `Z_m = focal · exp(C) · z_raw^(1/γ) / 1000` maps (default focal 2905.6 px) plus
`metadata.json` with matching intrinsics. The focal cancels in size computations. The
viewer reads `<f4` `.npy` as metres directly (`Inference_Viewer.html` line 683) and
multiplies integer formats by `depth_scale`.

---

## 10. Commands, end to end

```bash
# detection/pose  (conda activate segment; run from project root)
python scripts/extraction/extract_frames_uniform.py --video X.MP4 -o data/images/right_down -n 300 --rotate 270
python scripts/dataset/combine_datasets_dynamic.py --combine_set data/annotations/_combine_set \
    --images_root data/images --output datasets/combined_new
python scripts/training/train_maskrcnn_pose.py --mode train \
    --train_json datasets/combined_new/train.json --val_json datasets/combined_new/val.json \
    --images_dir data/images --output models/new --batch_size 2 --epochs 40 --lr 5e-4
PYTHONPATH=scripts/training python scripts/inference/inference_viz.py \
    --weights models/new/model_final.pth --video X.MP4 --output inference_results/run --every_n_frames 1
python scripts/inference/track_detections.py --input inference_results/run/detections_all.json \
    --output_json_dir inference_results/run/tracked_json

# depth/size  (conda activate da3)
python scripts/depth/run_da3_depth.py --images_dir inference_results/run/frames \
    --out inference_results/run/depth_crop15 --crop_top_frac 0.15 --no_viz
python scripts/depth/apply_boll_sizes.py --run inference_results/run --json_dir tracked_json \
    --depth_dir depth_crop15/depth --crop_top_px 797 --out_dir sizes
```

---

## 11. Known issues found while writing this

1. **`pose_eval.py` is broken**: it has a stale import and stale helper signatures
   (§8).
2. **The "canonical" model `combined_all_angles` predates the occlusion-class head.**
   Its `metrics.json` has no `loss_occ_class`. Loaded with today's `CottonROIHeads`,
   `occ_class_pred` is randomly initialised, so any `occ_class` / `occ_class_conf`
   it outputs is meaningless.
   - Its date (2026-06-03) and loss set also match `train_maskrcnn_pose_old.py`, which
     has **no 1.5× pose-box expansion**. If so, current inference maps its keypoints
     through the wrong box.
   - This part is inferred from dates and loss names, not proven. It can be checked by
     loading the model with each `POSE_BOX_SCALE` and comparing keypoint error.
   - The v2+ models were trained with both features.
3. **`combined_all` is not reproducible with the current `combine_datasets.py`**:
   task names and fields differ (§4).
4. **Random val splits on track-mode video leak near-duplicate frames** (§8). The
   v2–v4 val sets have only 8–10 images.
5. **Guessed tip/base points are ignored by the pose loss** (§5.3), and `pose_eval.py`
   evaluates only polygon GT. Pose quality on occluded tips and bases is unmeasured.
6. **The calibration rests on 4 tagged bolls** spanning 43–52 mm (see
   `findings/depth_findings.md`, and `CLAUDE.md` for the 43–52 mm range). The active
   `calibration.json` does not record its depth variant. `apply_boll_sizes.py` needs
   the crop offset passed by hand.
