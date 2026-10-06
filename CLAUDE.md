# CLAUDE.md

## File access
- Do not proactively explore or read files beyond what's needed for the current task.
- Only read files I explicitly reference or that are directly imported/called by files I reference.
- If you think another file is relevant, ask before reading it.

## What this is

Computer-vision research project — Texas Tech University, Applied Vision Lab. GoPro
greenhouse/field footage of cotton plants, two connected subsystems:

1. **Detection + pose** — custom Detectron2 Mask R-CNN (ResNet-50-FPN) with extra
   tip/base keypoint-heatmap and occlusion RoI heads, trained per-boll on annotated
   frames from three camera angles.
2. **Monocular depth** — Depth Anything 3 converts boll pixel size to real size, with a
   measured, correctable compression bias (`findings/depth_findings.md`). Lives in
   `scripts/depth/`.

No build system, no tests, no CI — standalone scripts run manually against local data.

## Environment

```bash
conda activate segment     # detection/pose
conda activate da3         # depth ONLY — depth_anything_3 is not importable under segment
```
Python 3.11, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, **Detectron2 0.6** (install
separately), shapely, opencv-python, pycocotools, numpy, pillow. `requirements.txt` is
UTF-16. Run scripts from the project root so relative paths resolve.

## Detection + pose pipeline

**1. Extract frames** — `scripts/extraction/extract_frames.py` (diverse selection) or
`extract_frames_uniform.py` (fixed interval, for CVAT track mode).

**2. Annotate in CVAT**, export "CVAT for images 1.1" XML. Two modes per angle:
- **track mode** (`right_down`, 300 frames) — args are `--track_xml`/`--track_images`.
- **shape mode** (`front_down`, `back_down`) — `--front_*`/`--back_*`.

Conventions (`annotation_guide/ANNOTATION_GUIDE.md`) are load-bearing: the `cotton_boll`
polygon covers the whole boll including bracts; `boll_id` is the physical tag number,
annotation metadata only — **never** a model prediction target; tip/base may be marked
not-visible. `left_down` additionally carries `tip_polygob` / `base_rectangle` polygons
(centroids give tip/base) — see the calibration section.

**3. CVAT XML → COCO:**
```bash
python scripts/dataset/cvat_to_coco_v4.py --xml <xml> --images_dir <dir> \
  --output datasets/right_down_only --val_split 0.2
```
`cvat_to_coco_v4.py` is current; `scripts/dataset/legacy/*` is superseded.

Combine all three angles (what the best model trains on):
```bash
python scripts/dataset/combine_datasets.py \
  --track_xml data/annotations/right_down/right_down.xml --track_images data/images/right_down \
  --front_xml ... --front_images ... --back_xml ... --back_images ... \
  --output datasets/combined_all/output_coco_dir --strategy all
```
`--strategy all` | `balanced` (subsample the large track set via `--track_sample`).
`combine_datasets_sequential.py` does a temporal val split; `_dynamic.py` is another
variant. Combined-dataset `file_name`s are angle-prefixed, so `--images_dir` must be the
COCO dir root, not a per-angle subfolder.

**4. Train** — `scripts/training/train_maskrcnn_pose.py` is canonical (`_big_box`,
`_gt_tipbase`, `_old` siblings are experiment variants — check which produced a result).

Architecture, all in that file:
- `CottonROIHeads(StandardROIHeads)`, registered in `ROI_HEADS_REGISTRY`, selected via
  `cfg.MODEL.ROI_HEADS.NAME`. Adds beside the box/mask heads:
  - `PoseHeatmapHead` — tip/base as a 2-channel Gaussian heatmap (`pose_heatmap_loss`,
    masked so invisible keypoints don't contribute).
  - `OcclusionHead` — per-instance binary classifier, `occ_class_weights` buffer.
- `CottonDatasetMapper` builds heatmap targets at load time.
- `NUM_CLASSES = 1` (`cotton_boll`); angle and tag are metadata, not classes.
- `CottonTrainer(DefaultTrainer)` + `PeriodicVizHook` (feeds `debug/heatmaps/`).

```bash
nohup python scripts/training/train_maskrcnn_pose.py --mode train \
  --train_json .../train.json --val_json .../val.json --images_dir .../output_coco_dir \
  --output models/combined_all_angles --batch_size 2 --epochs 40 --workers 2 --lr 5e-4 \
  > logs/train.log 2>&1 &
```
`--epochs` overrides `--max_iter`. Always background it.

**Evaluate** — `scripts/inference/pose_eval.py` gives detection recall / mask IoU (custom
greedy matching, not COCOeval), tip/base pixel error, PCK@10%-diagonal, occlusion accuracy.

**Inference** — `scripts/inference/inference_viz.py --weights ... --images_dir ... --output
... --score_thresh 0.5`; supports `--video` and optional histogram color transfer
(`--color_src`/`--color_tgt`). `inference_viz_softmax.py`, `inference_viz_video.py` are
variants.

**Tracking** — `scripts/inference/track_detections.py`, IoU SORT-lite + keypoint smoothing.
`track_id` is minted by a counter in that script — it is **not** the annotation `boll_id`
and carries no tag identity. It fragments across occlusions (one boll can get two ids).

### Model status
`models/combined_all_angles/model_final.pth` is canonical (multi-angle, 40 epochs). Other
folders under `models/` are earlier/single-angle runs. Every `.pth` is ~412 MB.

## Viewer

`viewer/Inference_Viewer.html` — self-contained browser viewer, open directly.
`viewer/Inference_Viewer_Export.html` — same plus video export. Load frames dir,
detections dir, and (optionally) a depth dir.

- **Export frame sequence** (use this): writes `f_%05d.jpg` to a folder you pick plus
  `encode_ffmpeg.txt`, then encode with the ffmpeg line it prints. Has a **Start at frame**
  field to resume a run that died.
- **Quick record (MP4)**: real-time canvas capture. MediaRecorder *silently drops* frames
  it cannot encode in time — one 2951-frame run kept 411 (13.9%) and came out
  variable-frame-rate, which is what makes VLC stutter and scrub badly. Preview only.
- Full-res rendering costs ~100–160 ms/frame, so long exports are minutes, and the tab can
  still OOM; resume rather than restart.
- The viewer reads float32 `.npy` **as metres and applies no gamma of its own**. Raw DA3
  maps therefore read in arbitrary units — see `make_viewer_metric_depth.py`.

## Depth pipeline (`scripts/depth/`)

Raw monocular depth is systematically *compressed*: `reported ∝ true^γ`. Applying
`depth^(1/γ)` corrects it; never use raw DA3 depth for size directly.

Flow: `run_da3_depth.py` (DA3 over a frame dir) → `sample_boll_depth.py` →
`analyze_depth_csv.py`. `ball_video_test.py`, `da3_crop_*.py`, `da3_context_test.py` are the
ping-pong-ball calibration experiments behind `findings/depth_findings.md` (γ ≈ 0.67 from
2–3 balls per setup).

### Crop convention — load-bearing
Run DA3 with `--crop_top_frac 0.15` (797 px on a 5312-tall frame) to keep the greenhouse
roof out of view; it writes `crop_info.json`, and downstream sampling must shift detection
y-coords by `crop_top_px`. Cropping moves raw depth ~23% (frame p95 1.53 → 1.13), so **a
scale constant fitted on cropped depth must not be applied to uncropped maps, or vice
versa**. Nothing enforces this yet — check `depth_variant` in the calibration JSON.

### Calibration against tagged bolls (`findings/cotton_tag_calibration/`)
`data/annotations/left_down/left_down.xml` has correct `boll_id` tags; joining them to
`findings/cotton_pending_calibration/cotton_ground_truth.csv` (38 bolls,
`height_tip_to_base_mm`) gives real sizes, so γ can be fitted with a residual:

```
log S_true − log tb_px = (1/γ)·log z_raw + C
size_mm = tb_px * z_raw**(1/γ) * exp(C)      # C absorbs depth scale and focal length
```

```bash
python scripts/depth/fit_gamma_from_annotations.py \
    --depth_dir depth_out/left_down_annot_crop15/depth --middle 30
python scripts/depth/apply_boll_sizes.py --run <run> --json_dir tracked_json_trimmed \
    --depth_dir depth_trimmed_crop15/depth --crop_top_px 797 --out_dir sizes_mid30
python scripts/depth/make_viewer_metric_depth.py --src <raw depth dir> --dst <metric dir>
```
`fit_gamma_from_annotations.py` auto-detects `crop_info.json`. `make_viewer_metric_depth.py`
bakes `Z_m = focal·exp(C)·z_raw^(1/γ)/1000` into new maps plus `metadata.json` so the viewer
shows real mm; the focal cancels and does not affect sizes.

**γ depends on what background is in frame.** Front/back frames of a pass look down the row
into open background and drag γ down; `--middle N` excludes them:

| band | bolls | γ (leave-one-boll-out) | held-out size error |
|---|---|---|---|
| all 187 frames | 7 | 0.53 | 8.3% |
| middle 100 | 6 | 0.67 | 5.1% |
| **middle 30** | **4** | **0.70** | **1.0%** |

Background-free, real cotton agrees with the ball value (0.663). Current
`calibration.json` is the middle-30 fit (γ=0.70, scale 0.284477).

**Limits:** only 4–7 tagged bolls, GT spanning just 43–52 mm, so bolls below ~40 mm are
extrapolation and 1.0% overstates real accuracy. Within-boll γ fitting does not work — the
camera pans along the row, so a boll's true distance barely changes while reported depth
wobbles (r² ≈ 0.05, wrong-sign slopes). All signal is between bolls; aggregate per boll,
never per instance, or precision is badly overstated.

### Ball calibration sets (`frames/<set>/`)
40 mm ping-pong-ball recordings behind the γ/focal calibration. Layout: `baseline/`,
`obj<D>in_*.jpg`, `metadata.csv`, `locate_out/`, `depth/depth/<stem>.npy`. Frames were cut
with `ffmpeg -ss <t> -i <VID>.MP4 -frames:v 1 -vf "crop=4780:2988:0:0" -q:v 2 <out>.jpg -y`.
Run DA3 on these with **no `--focal_px`** — existing `.npy` are raw network output.
`ball_video_test.py locate` fits its reference circle to a *median stack*, valid only if the
ball never moves; if `locate_per_frame.py` reports a large `move_px`, re-run `locate` on a
single baseline frame. Sets with no tape: use `analyse_noref_set.py`, then
`plot_noref_set.py` / `verify_noref_set.py`. `GX010120_test` is the worked example
(`findings/per_set/GX010120_test.md`).

## Known gaps / stale references

`DIRECTORY_MANIFEST.md` (2026-06-18) lists files as MISSING that have since reappeared —
verify against the actual tree. `tools/` is still empty. `_archive/` is superseded scratch.
Ground-truth tag↔track matching exists only for the annotated `left_down` frames; tracked
runs carry no tag identity, so per-boll validation outside those frames is not possible.
