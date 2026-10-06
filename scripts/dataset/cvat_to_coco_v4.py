"""
cvat_to_coco_v4.py — Convert CVAT XML annotations (new schema with shape-based
tip/base) to a COCO-style JSON for Mask R-CNN training.

New schema labels handled:
  cotton_boll (polygon): annotation_type in {whole_visible, visible_parts, whole_estimate}
  tip_polygob (3-point polygon): shape for visible tip
  base_rectangle (4-point polygon): shape for visible base
  tip (point): guessed tip location
  base (point): guessed base location

Output: One COCO instance per physical boll (grouped by boll_id within a frame).
Each instance carries:
  - segmentation: visible region (whole_visible OR union of visible_parts)
  - bbox: from segmentation
  - tip_shape: 3-point polygon if present, else None
  - base_shape: 4-point polygon if present, else None
  - tip_point: (x, y) of guessed tip if no tip_shape
  - base_point: (x, y) of guessed base if no base_shape
  - is_occluded: 0 or 1
  - occlusion_pct: float in [0, 1] (0 if not occluded)
  - tip_visibility / base_visibility: visible | guessed | unknown

Usage:
    python cvat_to_coco_v4.py \
        --xml annotations.xml \
        --images_dir ./frames \
        --output ./dataset \
        --val_split 0.2
"""

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from shapely.geometry import Polygon
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False
    print("[warn] shapely not installed — occlusion_pct will be skipped. "
          "Install with: pip install shapely")


# Fixed occlusion-class label map. THIS ORDER IS CANONICAL and must match
# train_maskrcnn_pose.py, inference_viz.py, and the HTML viewer exactly.
# Index -> occluded_by string.
OCC_CLASSES = ["none", "leaf", "stem", "other_boll", "other", "frame_edge"]
OCC_CLASS_TO_ID = {name: i for i, name in enumerate(OCC_CLASSES)}


def occ_class_id(occluded_by: str) -> int:
    """Map an occluded_by string to its canonical class index.

    Unknown / unexpected values fall back to 0 (none) so training never crashes
    on a stray label; such cases should be caught by validate_annotations.py.
    """
    return OCC_CLASS_TO_ID.get((occluded_by or "none").strip(), 0)


def parse_points(s: str) -> List[Tuple[float, float]]:
    """Parse CVAT points string 'x1,y1;x2,y2;...' into list of (x,y)."""
    if not s:
        return []
    return [tuple(map(float, p.split(","))) for p in s.split(";") if p.strip()]


def polygon_area(pts: List[Tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def bbox_from_polygon(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return COCO bbox [x, y, w, h] from polygon points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


def bbox_from_polygons(polys: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    """Bbox enclosing multiple polygons."""
    xs, ys = [], []
    for p in polys:
        xs += [pt[0] for pt in p]
        ys += [pt[1] for pt in p]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


def shapely_union_area(polys: List[List[Tuple[float, float]]]) -> float:
    """Compute total area covered by union of polygons (handles overlap)."""
    if not SHAPELY_OK or not polys:
        return sum(polygon_area(p) for p in polys)
    sps = [Polygon(p).buffer(0) for p in polys if len(p) >= 3]
    if not sps:
        return 0.0
    union = sps[0]
    for s in sps[1:]:
        union = union.union(s)
    return float(union.area)


def flatten_segmentation(polys: List[List[Tuple[float, float]]]) -> List[List[float]]:
    """COCO segmentation: list of [x1, y1, x2, y2, ...] per polygon."""
    return [[c for pt in p for c in pt] for p in polys]


def convert_xml(xml_path: Path, images_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Parse one CVAT XML and return (images, annotations) in COCO style."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images, annotations = [], []
    image_id = 0
    ann_id = 0
    skipped = defaultdict(int)

    for image_el in root.findall(".//image"):
        fname = image_el.get("name", "")
        width = int(image_el.get("width", 0))
        height = int(image_el.get("height", 0))

        img_path = images_dir / fname
        if not img_path.exists():
            skipped["image_not_found"] += 1
            continue

        # Group shapes by boll_id
        grouped: Dict[str, dict] = defaultdict(lambda: {
            "whole_visible": [], "visible_parts": [], "whole_estimate": [],
            "tip_polygon": None, "base_rectangle": None,
            "tip_point": None, "base_point": None,
            "tip_visibility": "unknown", "base_visibility": "unknown",
            "occluded_by": "none",
        })

        for shape in image_el:
            label = shape.get("label", "")
            attrs = {a.get("name"): (a.text.strip() if a.text else "")
                     for a in shape.findall("attribute")}
            boll_id = attrs.get("boll_id", "")

            if not boll_id or boll_id == "0":
                if label == "cotton_boll":
                    skipped["cotton_boll_no_id"] += 1
                else:
                    skipped[f"{label}_no_id"] += 1
                continue

            pts = parse_points(shape.get("points", ""))

            if label == "cotton_boll":
                ann_type = attrs.get("annotation_type", "")
                grouped[boll_id][ann_type].append(pts)
                # Metadata only from whole_visible / whole_estimate
                if ann_type in ("whole_visible", "whole_estimate"):
                    grouped[boll_id]["tip_visibility"] = attrs.get("tip_visibility", "unknown")
                    grouped[boll_id]["base_visibility"] = attrs.get("base_visibility", "unknown")
                    grouped[boll_id]["occluded_by"] = attrs.get("occluded_by", "none")
            elif label == "tip_polygob":
                grouped[boll_id]["tip_polygon"] = pts
            elif label == "base_rectangle":
                grouped[boll_id]["base_rectangle"] = pts
            elif label == "tip":
                grouped[boll_id]["tip_point"] = pts[0] if pts else None
            elif label == "base":
                grouped[boll_id]["base_point"] = pts[0] if pts else None

        # Build one COCO annotation per boll_id
        instances_in_image = 0
        for boll_id, data in grouped.items():
            wv = data["whole_visible"]
            vp = data["visible_parts"]
            we = data["whole_estimate"]

            # Determine visible polygons (for mask training)
            if wv:
                visible_polys = wv
                is_occluded = 1 if data["occluded_by"] != "none" else 0
                occlusion_pct = 0.0
                if data["occluded_by"] == "frame_edge":
                    is_occluded = 1
                    occlusion_pct = 0.0
            elif vp and we:
                visible_polys = vp
                is_occluded = 1
                we_area = shapely_union_area(we)
                vp_area = shapely_union_area(vp)
                if we_area > 0:
                    occlusion_pct = max(0.0, min(1.0, 1.0 - vp_area / we_area))
                else:
                    occlusion_pct = 0.0
            elif vp and not we:
                visible_polys = vp
                is_occluded = 1
                occlusion_pct = 0.0  # unknown without whole_estimate
            else:
                skipped["no_mask_polygon"] += 1
                continue

            bbox = bbox_from_polygons(visible_polys)
            if bbox[2] <= 1 or bbox[3] <= 1:
                skipped["degenerate_bbox"] += 1
                continue

            seg_area = shapely_union_area(visible_polys)

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": bbox,
                "area": seg_area,
                "segmentation": flatten_segmentation(visible_polys),
                # Custom fields
                "boll_id": int(boll_id),
                "is_occluded": is_occluded,
                "occlusion_pct": float(occlusion_pct),
                "occluded_by": data["occluded_by"],
                "occ_class_id": occ_class_id(data["occluded_by"]),
                "tip_visibility": data["tip_visibility"],
                "base_visibility": data["base_visibility"],
                "tip_polygon": data["tip_polygon"],
                "base_rectangle": data["base_rectangle"],
                "tip_point": data["tip_point"],
                "base_point": data["base_point"],
            }
            annotations.append(ann)
            ann_id += 1
            instances_in_image += 1

        if instances_in_image > 0:
            images.append({
                "id": image_id,
                "file_name": fname,
                "width": width,
                "height": height,
            })
            image_id += 1

    print(f"\n[parse] {xml_path.name}")
    print(f"  Images: {len(images)}, Annotations: {len(annotations)}")
    if skipped:
        print(f"  Skipped:")
        for k, v in skipped.items():
            print(f"    {k}: {v}")

    return images, annotations


def write_coco(images, annotations, output_path: Path):
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cotton_boll"}],
        "info": {"description": "Cotton boll segmentation + pose dataset (v4 schema)"},
    }
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"  Wrote {output_path}")


def split_train_val(images, annotations, val_split: float, seed: int = 42):
    random.seed(seed)
    img_ids = [im["id"] for im in images]
    random.shuffle(img_ids)
    n_val = int(len(img_ids) * val_split)
    val_ids = set(img_ids[:n_val])

    train_imgs = [im for im in images if im["id"] not in val_ids]
    val_imgs = [im for im in images if im["id"] in val_ids]
    train_anns = [a for a in annotations if a["image_id"] not in val_ids]
    val_anns = [a for a in annotations if a["image_id"] in val_ids]

    return train_imgs, train_anns, val_imgs, val_anns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=Path, required=True, help="CVAT XML file")
    p.add_argument("--images_dir", type=Path, required=True, help="Directory with images")
    p.add_argument("--output", type=Path, required=True, help="Output directory")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    images, annotations = convert_xml(args.xml, args.images_dir)

    if args.val_split > 0:
        tr_i, tr_a, va_i, va_a = split_train_val(images, annotations, args.val_split, args.seed)
        write_coco(tr_i, tr_a, args.output / "train.json")
        write_coco(va_i, va_a, args.output / "val.json")
        print(f"\n[split] train: {len(tr_i)} images / {len(tr_a)} anns")
        print(f"        val:   {len(va_i)} images / {len(va_a)} anns")
    else:
        write_coco(images, annotations, args.output / "annotations.json")

    # Stats
    n_occluded = sum(1 for a in annotations if a["is_occluded"])
    n_with_tip_shape = sum(1 for a in annotations if a["tip_polygon"])
    n_with_base_shape = sum(1 for a in annotations if a["base_rectangle"])
    n_with_tip_point = sum(1 for a in annotations if a["tip_point"])
    n_with_base_point = sum(1 for a in annotations if a["base_point"])
    occluded_pcts = [a["occlusion_pct"] for a in annotations if a["is_occluded"] and a["occlusion_pct"] > 0]

    print(f"\n[stats]")
    print(f"  Total instances: {len(annotations)}")
    print(f"  Occluded: {n_occluded} ({100*n_occluded/max(1,len(annotations)):.1f}%)")
    print(f"  With tip polygon: {n_with_tip_shape}")
    print(f"  With base rectangle: {n_with_base_shape}")
    print(f"  With tip point (guessed): {n_with_tip_point}")
    print(f"  With base point (guessed): {n_with_base_point}")
    if occluded_pcts:
        print(f"  Occlusion pct: mean={sum(occluded_pcts)/len(occluded_pcts):.2f}, "
              f"max={max(occluded_pcts):.2f}")

    # Occlusion-class distribution (for the new classifier head)
    from collections import Counter
    occ_dist = Counter(a["occ_class_id"] for a in annotations)
    print(f"  Occ-class distribution:")
    for cid, name in enumerate(OCC_CLASSES):
        print(f"    {cid} {name:12s}: {occ_dist.get(cid, 0)}")


if __name__ == "__main__":
    main()
