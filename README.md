# Cotton Boll Detection & Pose Estimation

Computer-vision research project — **Texas Tech University, Applied Vision Lab**.
Custom Mask R-CNN (ResNet-50-FPN) with additional **pose** (tip/base heatmap) and
**occlusion** heads, trained on GoPro greenhouse footage of cotton plants across
three camera angles.

---

## What this project does

1. Extract frames from GoPro video.
2. Annotate cotton bolls in CVAT (polygon mask + tip/base quads/points + occlusion).
3. Convert CVAT XML → COCO JSON, then combine three camera angles into one dataset.
4. Train a Detectron2 Mask R-CNN with custom tip / base / occlusion RoI heads.
5. Evaluate pose accuracy and run inference + tracking + a browser viewer.

---

## Folder structure

```
cotton_boll_project/
├── README.md                  this file
├── DIRECTORY_MANIFEST.md      full file listing + exact run commands + flagged items
├── requirements.txt           pip freeze (UTF-16; Detectron2 installed separately)
│
├── data/
│   ├── raw_video/             GoPro MP4s (RightDown, FrontDown, BackDown, LeftDown, GX0100xx…)
│   ├── annotations/           CVAT XML exports
│   │   ├── right_down/        right_down_track_annotations.xml (orig) + right_down_fixed.xml (corrected)
│   │   ├── front_down/        front_down_shape_annotations.xml
│   │   ├── back_down/         back_down_shape_annotations.xml
│   │   └── _legacy/           earlier front/back exports (superseded by *_shape)
│   └── images/
│       ├── right_down/ (300)  front_down/ (29)  back_down/ (17)
│       ├── extracted_frames/  raw extracted frames per angle (~1.5k)
│       └── test_sets/         held-out test frames (front_down_test, 208)
│
├── datasets/                  COCO-format JSONs
│   ├── combined_all/          MULTI-ANGLE set used for the best model
│   │   ├── output_coco_dir/   train.json, val.json, right_down/ front_down/ back_down/ images
│   │   └── val_images_*/       relative symlinks to val images (rebuilt — see manifest)
│   ├── combined_all_sequential/  alt combined set, temporal val split
│   ├── right_down_only/  right_down_300/  right_down_root_split/   single-angle variants
│
├── scripts/
│   ├── extraction/   extract_frames.py, extract_frames_uniform.py, extract_image.py, rotate_image.py
│   ├── annotation/   annotate_image.py, predictions_to_cvat.py   (validate.py + fix_visibility_flags.py MISSING)
│   ├── dataset/      cvat_to_coco_v4.py, combine_datasets.py, combine_datasets_sequential.py
│   │   └── legacy/   cvat_to_coco_v1.py, cvat_xml_to_coco_v2.py  (superseded by v4)
│   ├── training/     train_maskrcnn_pose.py, debug_heatmaps.py
│   └── inference/    inference_viz.py, pose_eval.py              (track_detections.py MISSING)
│
├── models/
│   ├── combined_all_angles/   ★ BEST MODEL — model_final.pth + 10 checkpoints + metrics + viz
│   ├── right_down_300frames/  single-angle model, 300 frames
│   ├── intermediate_runs/     output_full, output_real (earlier real runs, kept)
│   └── test_runs/             output_debug, output_smoke1-4 (smoke tests, kept)
│
├── evaluation/combined_all/   eval_report.json + 4 PNG plots
├── inference_results/         per-run annotated JPGs + detections.json (7 runs)
├── debug/heatmaps/            dataset-mapper heatmap overlays
├── logs/                      training logs
├── docs/                      validation_review.xlsx, README_original.md
├── experiments/sam_finetuning/  earlier SAM fine-tuning experiment (separate from Mask R-CNN)
├── tools/                     (intended home for the CVAT tool clone — see manifest)
├── viewer/                    (Inference_Viewer_v2.html MISSING — see manifest)
└── _archive/                  zips, scratch (temp/tmp/pycache), stale symlinks & empty dirs
```

---

## Quick start (WSL Ubuntu)

```bash
conda activate segment
cd "/mnt/d/Segmentation/Depth Find/cotton_boll_project"
```

**Convert one angle (CVAT XML → COCO):**
```bash
python scripts/dataset/cvat_to_coco_v4.py \
  --xml data/annotations/right_down/right_down_fixed.xml \
  --images_dir data/images/right_down \
  --output datasets/right_down_only --val_split 0.2
```

**Combine all three angles:**
```bash
python scripts/dataset/combine_datasets.py \
  --track_xml data/annotations/right_down/right_down_fixed.xml --track_images data/images/right_down \
  --front_xml data/annotations/front_down/front_down_shape_annotations.xml --front_images data/images/front_down \
  --back_xml  data/annotations/back_down/back_down_shape_annotations.xml  --back_images  data/images/back_down \
  --output datasets/combined_all/output_coco_dir --strategy all
```
(Arg names are `--track_*` for the right_down set — not `--right_*`.)

**Train (best model recipe):**
```bash
nohup python scripts/training/train_maskrcnn_pose.py --mode train \
  --train_json datasets/combined_all/output_coco_dir/train.json \
  --val_json   datasets/combined_all/output_coco_dir/val.json \
  --images_dir datasets/combined_all/output_coco_dir \
  --output models/combined_all_angles \
  --batch_size 2 --epochs 40 --workers 2 --lr 5e-4 > logs/train.log 2>&1 &
```

**Evaluate:**
```bash
python scripts/inference/pose_eval.py \
  --weights models/combined_all_angles/model_final.pth \
  --val_json datasets/combined_all/output_coco_dir/val.json \
  --images_dir datasets/combined_all/output_coco_dir \
  --output evaluation/combined_all
```

**Inference + visualization:**
```bash
python scripts/inference/inference_viz.py \
  --weights models/combined_all_angles/model_final.pth \
  --images_dir data/images/test_sets/front_down_test \
  --output inference_results/my_run --score_thresh 0.5
```

---

## Results — multi-angle model (`models/combined_all_angles/model_final.pth`)

| Metric | Overall | right_down | front_down | back_down |
|---|---|---|---|---|
| Detection recall @IoU0.5 | 0.982 | 0.992 | 0.870 | 1.000 |
| Mean mask IoU | 0.846 | 0.845 | 0.824 | 0.879 |
| Tip median pixel error | 14.22 px | 14.93 | 9.98 | 10.32 |
| Base median pixel error | 14.69 px | 15.91 | 7.89 | 9.02 |
| Tip PCK@10% diag | 0.74 | 0.71 | 0.89 | 1.00 |
| Occlusion binary accuracy | 0.929 | 0.924 | 0.900 | 1.000 |

Camera angles: **right_down** (primary, 300 frames, CVAT track mode) · **front_down**
(29 frames, shape mode) · **back_down** (17 frames, shape mode).

---

## Dependencies

Python 3.11 · PyTorch 2.5.1+cu121 · torchvision 0.20.1+cu121 · **Detectron2 0.6**
(install separately — not in `requirements.txt`) · shapely 2.1.2 · opencv-python 4.13 ·
pycocotools 2.0.11 · numpy · pillow. SAM experiment also uses `segment-anything 1.0`.

```bash
pip install -r requirements.txt
# Detectron2 (match your torch/CUDA build):
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

See `DIRECTORY_MANIFEST.md` for the complete file listing, every run command with
correct paths, and a list of files that need your attention.
