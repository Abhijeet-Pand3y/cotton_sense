"""
CVAT → COCO Annotation Converter for Cotton Boll Pipeline
==========================================================
Converts CVAT XML exports (with polygon + skeleton + attributes)
into COCO JSON format compatible with Detectron2 keypoint training.

Also includes utilities to:
  - Validate annotations
  - Split into train/val sets
  - Generate annotation statistics

Usage:
    python cvat_to_coco.py --cvat export.xml --output ./annotations --val_split 0.15
"""

import xml.etree.ElementTree as ET
import json
import os
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ── COCO format template ──────────────────────────────────────────────

def create_coco_template():
    """Create empty COCO JSON structure with cotton boll category."""
    return {
        "info": {
            "description": "Cotton Boll Segmentation + Keypoint Dataset",
            "version": "1.0",
            "year": 2026,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "cotton_boll",
                "supercategory": "plant",
                "keypoints": ["tip", "base"],
                "skeleton": [[0, 1]]  # tip-base connection
            }
        ]
    }


# ── CVAT XML Parser ──────────────────────────────────────────────────

def parse_cvat_xml(xml_path):
    """
    Parse CVAT XML export into structured annotation data.

    Expects CVAT export with:
      - Polygon shapes (segmentation masks)
      - Skeleton/points (tip and base keypoints)
      - Attributes: boll_id, tip_visible, base_visible
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images_data = {}
    annotations = []
    ann_id = 1

    for image_elem in root.findall('.//image'):
        img_name = image_elem.get('name')
        img_width = int(image_elem.get('width'))
        img_height = int(image_elem.get('height'))
        img_id = int(image_elem.get('id'))

        images_data[img_id] = {
            "id": img_id,
            "file_name": img_name,
            "width": img_width,
            "height": img_height
        }

        # Group shapes by track/group for multi-part annotations
        # (polygon + keypoints belong to same boll instance)
        groups = defaultdict(list)
        for shape in image_elem:
            if shape.tag in ('polygon', 'points', 'skeleton'):
                group_id = shape.get('group_id', shape.get('z_order', str(ann_id)))
                groups[group_id].append(shape)

        # Process each group as one boll instance
        for group_id, shapes in groups.items():
            polygon = None
            tip = None
            base = None
            boll_id = None
            tip_visible = 2  # default: visible
            base_visible = 2

            for shape in shapes:
                # Extract attributes
                for attr in shape.findall('attribute'):
                    attr_name = attr.get('name', '').lower()
                    attr_val = attr.text.strip() if attr.text else ''

                    if attr_name == 'boll_id':
                        boll_id = int(attr_val) if attr_val.isdigit() else None
                    elif attr_name == 'tip_visible':
                        tip_visible = 2 if attr_val.lower() in ('true', '1', 'yes') else 0
                    elif attr_name == 'base_visible':
                        base_visible = 2 if attr_val.lower() in ('true', '1', 'yes') else 0

                if shape.tag == 'polygon':
                    points_str = shape.get('points', '')
                    points = []
                    for pt in points_str.split(';'):
                        x, y = pt.split(',')
                        points.extend([float(x), float(y)])
                    polygon = points

                elif shape.tag == 'points':
                    label = shape.get('label', '').lower()
                    points_str = shape.get('points', '')
                    x, y = points_str.split(',')
                    pt = (float(x), float(y))

                    if 'tip' in label:
                        tip = pt
                    elif 'base' in label:
                        base = pt

                elif shape.tag == 'skeleton':
                    # CVAT skeleton contains child elements for each point
                    for point_elem in shape.findall('.//points'):
                        label = point_elem.get('label', '').lower()
                        points_str = point_elem.get('points', '')
                        if points_str:
                            x, y = points_str.split(',')
                            pt = (float(x), float(y))
                            if 'tip' in label:
                                tip = pt
                            elif 'base' in label:
                                base = pt

            if polygon is None:
                continue  # Skip if no segmentation mask

            # Build COCO keypoints array: [x, y, v, x, y, v]
            # v: 0=not labeled, 1=labeled but occluded, 2=labeled and visible
            keypoints = []
            num_keypoints = 0

            if tip and tip_visible > 0:
                keypoints.extend([tip[0], tip[1], tip_visible])
                num_keypoints += 1
            else:
                keypoints.extend([0, 0, 0])

            if base and base_visible > 0:
                keypoints.extend([base[0], base[1], base_visible])
                num_keypoints += 1
            else:
                keypoints.extend([0, 0, 0])

            # Compute bbox from polygon
            poly_x = polygon[0::2]
            poly_y = polygon[1::2]
            x_min, x_max = min(poly_x), max(poly_x)
            y_min, y_max = min(poly_y), max(poly_y)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Compute area (shoelace formula)
            n = len(poly_x)
            area = 0.5 * abs(sum(
                poly_x[i] * poly_y[(i+1) % n] - poly_x[(i+1) % n] * poly_y[i]
                for i in range(n)
            ))

            annotation = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [polygon],
                "bbox": [round(b, 2) for b in bbox],
                "area": round(area, 2),
                "iscrowd": 0,
                "keypoints": [round(k, 2) for k in keypoints],
                "num_keypoints": num_keypoints
            }

            # Store boll_id as custom field (not used by model, just metadata)
            if boll_id is not None:
                annotation["attributes"] = {"boll_id": boll_id}

            annotations.append(annotation)
            ann_id += 1

    return images_data, annotations


# ── Dataset Split ─────────────────────────────────────────────────────

def split_dataset(images_data, annotations, val_split=0.15, seed=42):
    """Split into train and val sets by image."""
    random.seed(seed)
    image_ids = list(images_data.keys())
    random.shuffle(image_ids)

    val_count = max(1, int(len(image_ids) * val_split))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    train_coco = create_coco_template()
    val_coco = create_coco_template()

    for img_id, img_data in images_data.items():
        if img_id in train_ids:
            train_coco["images"].append(img_data)
        else:
            val_coco["images"].append(img_data)

    for ann in annotations:
        if ann["image_id"] in train_ids:
            train_coco["annotations"].append(ann)
        else:
            val_coco["annotations"].append(ann)

    return train_coco, val_coco


# ── Validation ────────────────────────────────────────────────────────

def validate_annotations(coco_data):
    """Check for common annotation issues."""
    issues = []
    stats = defaultdict(int)

    for ann in coco_data["annotations"]:
        stats["total_annotations"] += 1

        # Check segmentation
        if not ann.get("segmentation") or len(ann["segmentation"][0]) < 6:
            issues.append(f"Ann {ann['id']}: Segmentation has <3 points")

        # Check keypoints
        kp = ann.get("keypoints", [])
        tip_v = kp[2] if len(kp) >= 3 else 0
        base_v = kp[5] if len(kp) >= 6 else 0

        if tip_v > 0:
            stats["tip_visible"] += 1
        if base_v > 0:
            stats["base_visible"] += 1
        if tip_v > 0 and base_v > 0:
            stats["both_visible"] += 1
        if tip_v == 0 and base_v == 0:
            stats["neither_visible"] += 1

        # Check bbox validity
        bbox = ann.get("bbox", [])
        if len(bbox) == 4 and (bbox[2] <= 0 or bbox[3] <= 0):
            issues.append(f"Ann {ann['id']}: Invalid bbox dimensions")

        # Check area
        if ann.get("area", 0) <= 0:
            issues.append(f"Ann {ann['id']}: Zero or negative area")

    return issues, dict(stats)


# ── Statistics ────────────────────────────────────────────────────────

def print_stats(coco_data, split_name="Dataset"):
    """Print annotation statistics."""
    issues, stats = validate_annotations(coco_data)

    n_images = len(coco_data["images"])
    n_anns = stats.get("total_annotations", 0)

    print(f"\n{'='*50}")
    print(f"  {split_name} Statistics")
    print(f"{'='*50}")
    print(f"  Images:              {n_images}")
    print(f"  Annotations:         {n_anns}")
    print(f"  Avg bolls/image:     {n_anns/max(n_images,1):.1f}")
    print(f"  Tip visible:         {stats.get('tip_visible', 0)} ({100*stats.get('tip_visible',0)/max(n_anns,1):.0f}%)")
    print(f"  Base visible:        {stats.get('base_visible', 0)} ({100*stats.get('base_visible',0)/max(n_anns,1):.0f}%)")
    print(f"  Both visible:        {stats.get('both_visible', 0)} ({100*stats.get('both_visible',0)/max(n_anns,1):.0f}%)")
    print(f"  Neither visible:     {stats.get('neither_visible', 0)} ({100*stats.get('neither_visible',0)/max(n_anns,1):.0f}%)")

    if issues:
        print(f"\n  ⚠ Issues found: {len(issues)}")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")
    else:
        print(f"\n  ✓ No issues found")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML annotations to COCO JSON for Detectron2"
    )
    parser.add_argument("--cvat", required=True,
                        help="Path to CVAT XML export file")
    parser.add_argument("--output", "-o", default="./annotations",
                        help="Output directory for COCO JSON files")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--no_split", action="store_true",
                        help="Don't split, output single JSON")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Parsing CVAT XML: {args.cvat}")
    images_data, annotations = parse_cvat_xml(args.cvat)
    print(f"  Found {len(images_data)} images, {len(annotations)} annotations")

    if args.no_split:
        coco = create_coco_template()
        coco["images"] = list(images_data.values())
        coco["annotations"] = annotations

        out_path = os.path.join(args.output, "annotations.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        print_stats(coco, "Full Dataset")
        print(f"\nSaved to: {out_path}")

    else:
        train_coco, val_coco = split_dataset(
            images_data, annotations, val_split=args.val_split
        )

        train_path = os.path.join(args.output, "train.json")
        val_path = os.path.join(args.output, "val.json")

        with open(train_path, "w") as f:
            json.dump(train_coco, f, indent=2)
        with open(val_path, "w") as f:
            json.dump(val_coco, f, indent=2)

        print_stats(train_coco, "Train Split")
        print_stats(val_coco, "Val Split")
        print(f"\nSaved to:")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")


if __name__ == "__main__":
    main()
