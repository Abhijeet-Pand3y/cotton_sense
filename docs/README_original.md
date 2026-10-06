# Cotton Boll Segmentation + Keypoint Pipeline

## Overview

This pipeline detects cotton bolls in GoPro video and predicts:
- **Instance segmentation mask** (pixel-level boll boundary)
- **Tip keypoint** (x, y) — or "not visible"
- **Base keypoint** (x, y) — or "not visible"  
- **Orientation** (derived from tip→base angle)

Boll IDs (physical tags) are annotation metadata only — never a model prediction target.

---

## Pipeline Steps

### Step 1: Extract Frames from GoPro Video

```bash
pip install opencv-python numpy

python extract_frames.py \
    --video /path/to/gopro_video.MP4 \
    --output ./frames \
    --num_frames 200
```

The script selects **diverse** frames (not just every Nth) based on:
- Visual content diversity (color histogram distance)
- Temporal spread (covers full video duration)  
- Quality filtering (removes blurry/dark frames)

Output: `./frames/` directory with 200 JPGs + `extraction_metadata.json`

---

### Step 2: Annotate in CVAT

#### Install CVAT (Docker)
```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
# Open http://localhost:8080
```

#### Create Project
1. New Project → "Cotton Boll Segmentation"
2. Add labels:

| Label | Type | Attributes |
|-------|------|------------|
| `cotton_boll` | Polygon + Skeleton | `boll_id` (number), `tip_visible` (checkbox), `base_visible` (checkbox) |

For the skeleton, define 2 nodes: `tip` and `base` with one edge connecting them.

#### Create Task
1. New Task → upload your 200 extracted frames
2. Assign to project "Cotton Boll Segmentation"

#### Annotate Each Frame
For each boll:
1. Draw polygon around the boll (segmentation mask)
2. Place skeleton with `tip` and `base` keypoints
3. Set `boll_id` attribute (physical tag number)
4. Check/uncheck `tip_visible` and `base_visible`
5. If tip or base is not visible, leave the keypoint at (0,0) and uncheck visible

**Pro tip**: Enable SAM auto-segmentation in CVAT → click on boll → accept/refine mask → add keypoints manually. This makes mask annotation 3-5x faster.

#### Export
Menu → Export task → Format: **CVAT for images 1.1** → Download XML

---

### Step 3: Convert Annotations to COCO Format

```bash
python cvat_to_coco.py \
    --cvat ./exported_annotations.xml \
    --output ./dataset/annotations \
    --val_split 0.15
```

Then organize your dataset:
```bash
mkdir -p dataset/images
cp frames/*.jpg dataset/images/
```

Final structure:
```
dataset/
├── images/
│   ├── frame_000001_t0p50s.jpg
│   └── ...
└── annotations/
    ├── train.json    (170 images)
    └── val.json      (30 images)
```

---

### Step 4: Train Model

#### Install Detectron2
```bash
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

#### Train
```bash
python train_detectron2.py \
    --data_dir ./dataset \
    --output ./output \
    --num_gpus 1
```

Training runs for ~3000 iterations (~30 min on single GPU).  
Evaluates on val set every 500 iterations.

#### Key outputs:
- `output/model_final.pth` — trained weights
- `output/eval_results.json` — mAP for bbox, segm, keypoints

---

### Step 5: Predict on New Video / Frames

```bash
# On full video
python train_detectron2.py \
    --data_dir ./dataset \
    --output ./output \
    --predict /path/to/gopro_video.MP4 \
    --predict_output ./predictions \
    --confidence 0.5

# On directory of images
python train_detectron2.py \
    --data_dir ./dataset \
    --output ./output \
    --predict ./new_frames/ \
    --predict_output ./predictions \
    --confidence 0.5
```

Output:
- `predictions/predicted_video.mp4` — visualized predictions
- `predictions/frame_predictions/` — per-frame JSON with masks, keypoints, orientation

---

### Step 6: Model-Assisted Annotation (Iterate)

Convert predictions to CVAT import format:
```bash
python predictions_to_cvat.py \
    --pred_dir ./predictions/frame_predictions \
    --images_dir ./new_frames \
    --output ./cvat_import.xml \
    --confidence 0.5
```

In CVAT:
1. Create new task with the new frames
2. Upload annotations → CVAT 1.1 → `cvat_import.xml`
3. **Review and correct** pre-annotations (much faster than drawing from scratch)
4. Export corrected annotations
5. Add to dataset, retrain → repeat

Each iteration should improve significantly:
- Round 1: ~200 manual annotations → baseline model
- Round 2: ~500 total (300 corrected) → much better
- Round 3: ~1000 total → production quality

---

## Prediction JSON Format

Each detection includes:
```json
{
    "score": 0.95,
    "bbox": [x1, y1, x2, y2],
    "segmentation": [x1,y1, x2,y2, ...],
    "tip": {
        "x": 245.5,
        "y": 130.2,
        "confidence": 0.88,
        "visible": true
    },
    "base": {
        "x": 260.1,
        "y": 180.7,
        "confidence": 0.72,
        "visible": true
    },
    "orientation_deg": -62.3,
    "length_px": 52.1
}
```

---

## File Reference

| File | Purpose |
|------|---------|
| `extract_frames.py` | Smart frame extraction from GoPro video |
| `cvat_to_coco.py` | CVAT XML → COCO JSON converter |
| `train_detectron2.py` | Training + inference script |
| `predictions_to_cvat.py` | Model predictions → CVAT pre-annotations |

---

## Requirements

```
opencv-python>=4.8
numpy
torch>=2.0
torchvision
detectron2
```

## Hardware Notes

- Training: Single GPU with 8GB+ VRAM (RTX 3060 or better)
- Inference: GPU recommended but CPU works for individual images
- Frame extraction: CPU only, runs in ~1 min for typical GoPro video
