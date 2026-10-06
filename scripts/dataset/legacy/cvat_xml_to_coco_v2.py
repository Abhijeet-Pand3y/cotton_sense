"""
CVAT XML → COCO Converter v3 (boll_id based grouping)
=======================================================
Groups all annotations of the same physical boll using the boll_id attribute
(present on cotton_boll polygons, tip points, and base points).

Handles:
  - Multiple polygons per boll (visible_parts + whole_estimate)
  - tip/base keypoints linked by boll_id
  - Occluded_by metadata
  - Auto-computes occlusion percentage per boll
  - Normalizes legacy occluder names (frame_edge → frame_edge_cutoff)

Output: COCO JSON where each boll instance is ONE annotation containing:
  - main polygon (whole_estimate if occluded, else whole_visible, else largest visible_part)
  - visible_parts list stored in attributes
  - whole_estimate polygon stored in attributes
  - tip/base keypoints
  - occlusion metadata

Usage:
    python cvat_xml_to_coco_v3.py \
        --xml_files task1.xml task2.xml \
        --images_dirs ./frames/task1 ./frames/task2 \
        --output ./merged_dataset
"""

import xml.etree.ElementTree as ET
import json
import os
import argparse
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict


VISIBILITY_MAP = {
    "visible": 2,
    "guessed": 1,
    "unknown": 0
}

# Legacy occluder name mapping
OCCLUDER_ALIASES = {
    "frame_edge": "frame_edge_cutoff",
}

OCCLUDER_CLASSES = [
    "none", "leaf", "stem", "other_boll",
    "flower", "frame_edge_cutoff", "other"
]


# ── Geometry helpers ──────────────────────────────────────────────────

def polygon_bbox(polygon):
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def polygon_area(polygon):
    xs = polygon[0::2]
    ys = polygon[1::2]
    n = len(xs)
    return 0.5 * abs(sum(
        xs[i] * ys[(i+1) % n] - xs[(i+1) % n] * ys[i]
        for i in range(n)
    ))


# ── XML parsing ───────────────────────────────────────────────────────

def parse_attrs(shape):
    attrs = {}
    for attr in shape.findall('attribute'):
        name = attr.get('name', '').lower()
        val = attr.text.strip() if attr.text else ''
        attrs[name] = val
    return attrs


def get_boll_id(attrs):
    """Parse boll_id as int, default 0."""
    bid = attrs.get('boll_id', '0')
    try:
        return int(float(bid))
    except (ValueError, TypeError):
        return 0


def normalize_occluder(occluder):
    """Map legacy occluder names to current schema."""
    occluder = (occluder or 'none').lower()
    return OCCLUDER_ALIASES.get(occluder, occluder)


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images_data = []

    for img_elem in root.findall('.//image'):
        img_name = img_elem.get('name')
        img_w = int(img_elem.get('width'))
        img_h = int(img_elem.get('height'))

        # Group annotations by boll_id
        bolls = defaultdict(lambda: {
            'polygons': [],  # list of dicts with keys: polygon, attrs
            'tip': None,
            'base': None
        })

        # Also track polygons with boll_id=0 (unlabeled) separately
        # These get unique virtual IDs
        zero_id_polys = []

        for shape in img_elem:
            attrs = parse_attrs(shape)
            bid = get_boll_id(attrs)

            if shape.tag == 'polygon':
                pts_str = shape.get('points', '')
                coords = []
                for pt in pts_str.split(';'):
                    if ',' in pt:
                        x, y = pt.split(',')
                        coords.extend([float(x), float(y)])

                entry = {'polygon': coords, 'attrs': attrs}
                if bid == 0:
                    zero_id_polys.append(entry)
                else:
                    bolls[bid]['polygons'].append(entry)

            elif shape.tag == 'points':
                label = shape.get('label', '').lower()
                pts_str = shape.get('points', '')
                if ',' not in pts_str:
                    continue
                x, y = pts_str.split(';')[0].split(',')
                pt = (float(x), float(y))
                if label == 'tip':
                    bolls[bid]['tip'] = pt
                elif label == 'base':
                    bolls[bid]['base'] = pt

        # Assign virtual IDs to zero-id polygons (treat as separate bolls)
        if zero_id_polys:
            next_id = max(bolls.keys(), default=0) + 10000
            for entry in zero_id_polys:
                bolls[next_id] = {'polygons': [entry], 'tip': None, 'base': None}
                next_id += 1

        images_data.append({
            'name': img_name,
            'width': img_w,
            'height': img_h,
            'bolls': dict(bolls)
        })

    return images_data


# ── Boll instance processing ──────────────────────────────────────────

def process_boll_instance(boll_id, boll_data):
    """
    Given grouped polygons + tip/base for one boll, produce a single
    annotation-ready dict.

    IMPORTANT: Main boll metadata (visibility, occluder) comes from
    whole_visible OR whole_estimate ONLY. visible_parts polygons are used
    ONLY for their segmentation masks (to compute occlusion %). Any
    attributes on visible_parts polygons are ignored.
    """
    whole_visible = None
    whole_estimate = None
    visible_parts = []

    for p in boll_data['polygons']:
        ann_type = p['attrs'].get('annotation_type', 'whole_visible').lower()
        if ann_type == 'whole_visible':
            whole_visible = p
        elif ann_type == 'whole_estimate':
            whole_estimate = p
        elif ann_type == 'visible_parts' or ann_type == 'visible_part':
            visible_parts.append(p)
        else:
            # Unknown type, treat as whole_visible
            whole_visible = p

    # Determine main polygon (source of truth for boll metadata)
    # Priority: whole_estimate > whole_visible > largest visible_part (fallback only)
    if whole_estimate is not None:
        main = whole_estimate
        is_occluded = True
        metadata_source = 'whole_estimate'
    elif whole_visible is not None:
        main = whole_visible
        occluded_by_norm = normalize_occluder(whole_visible['attrs'].get('occluded_by', 'none'))
        is_occluded = occluded_by_norm != 'none'
        metadata_source = 'whole_visible'
    elif visible_parts:
        # Fallback: no main annotation — use largest visible part for MASK ONLY
        # Don't trust its attributes for metadata
        main = max(visible_parts, key=lambda p: polygon_area(p['polygon']))
        is_occluded = len(visible_parts) > 1
        metadata_source = 'visible_parts_fallback'
    else:
        return None

    # Compute occlusion percentage
    occlusion_pct = 0.0
    if whole_estimate is not None:
        estimate_area = polygon_area(whole_estimate['polygon'])
        if visible_parts:
            visible_area = sum(polygon_area(p['polygon']) for p in visible_parts)
        elif whole_visible is not None:
            # Both present (unusual but possible): use whole_visible as visible area
            visible_area = polygon_area(whole_visible['polygon'])
        else:
            visible_area = 0
        occlusion_pct = max(0.0, min(1.0, 1.0 - (visible_area / max(estimate_area, 1))))

    # Use metadata ONLY from main annotation (whole_visible or whole_estimate)
    # Ignore attributes on visible_parts
    if metadata_source in ('whole_estimate', 'whole_visible'):
        attrs = main['attrs']
    else:
        # No trustworthy metadata source — use empty/default attrs
        attrs = {}

    return {
        'boll_id': boll_id,
        'main_polygon': main['polygon'],
        'main_attrs': attrs,
        'metadata_source': metadata_source,
        'visible_parts': [vp['polygon'] for vp in visible_parts],
        'whole_estimate': whole_estimate['polygon'] if whole_estimate else None,
        'is_occluded': is_occluded,
        'occluded_by': normalize_occluder(attrs.get('occluded_by', 'none')),
        'occlusion_pct': round(occlusion_pct, 3),
        'tip': boll_data['tip'],
        'base': boll_data['base']
    }


# ── COCO build ────────────────────────────────────────────────────────

def build_coco(all_sources):
    coco = {
        "info": {
            "description": "Cotton Boll Segmentation + Keypoints + Occlusion",
            "version": "3.0",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "cotton_boll",
            "supercategory": "plant",
            "keypoints": ["tip", "base"],
            "skeleton": [[1, 2]],
            "occluder_classes": OCCLUDER_CLASSES
        }]
    }

    image_id_counter = 1
    ann_id_counter = 1

    stats = {
        'total_images': 0,
        'total_bolls': 0,
        'whole_visible': 0,
        'occluded': 0,
        'occluders': defaultdict(int),
        'tips_matched': 0,
        'bases_matched': 0,
        'occlusion_pcts': [],
        'multi_polygon_bolls': 0
    }

    for source_idx, (images_data, source_name) in enumerate(all_sources):
        print(f"\n--- Source {source_idx+1}: {source_name} ---")
        print(f"  Images: {len(images_data)}")

        for img_data in images_data:
            # Process each boll instance
            boll_instances = []
            for bid, boll_data in img_data['bolls'].items():
                inst = process_boll_instance(bid, boll_data)
                if inst is not None:
                    boll_instances.append(inst)

            img_filename = f"src{source_idx+1}_{img_data['name']}"
            coco['images'].append({
                "id": image_id_counter,
                "file_name": img_filename,
                "width": img_data['width'],
                "height": img_data['height'],
                "source": source_name,
                "original_name": img_data['name']
            })
            stats['total_images'] += 1

            for boll in boll_instances:
                main_poly = boll['main_polygon']
                attrs = boll['main_attrs']

                # Count multi-polygon bolls
                if len(boll['visible_parts']) > 0 or boll['whole_estimate'] is not None:
                    stats['multi_polygon_bolls'] += 1

                # Keypoints
                tip_v = VISIBILITY_MAP.get(
                    attrs.get('tip_visibility', 'unknown').lower(), 0
                )
                base_v = VISIBILITY_MAP.get(
                    attrs.get('base_visibility', 'unknown').lower(), 0
                )
                kp = [0, 0, 0, 0, 0, 0]
                num_kp = 0
                if boll['tip'] is not None and tip_v > 0:
                    kp[0], kp[1], kp[2] = round(boll['tip'][0], 2), round(boll['tip'][1], 2), tip_v
                    num_kp += 1
                    stats['tips_matched'] += 1
                if boll['base'] is not None and base_v > 0:
                    kp[3], kp[4], kp[5] = round(boll['base'][0], 2), round(boll['base'][1], 2), base_v
                    num_kp += 1
                    stats['bases_matched'] += 1

                ann = {
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": 1,
                    "segmentation": [main_poly],
                    "bbox": [round(v, 2) for v in polygon_bbox(main_poly)],
                    "area": round(polygon_area(main_poly), 2),
                    "iscrowd": 0,
                    "keypoints": kp,
                    "num_keypoints": num_kp,
                    "attributes": {
                        "boll_id": boll['boll_id'],
                        "is_occluded": boll['is_occluded'],
                        "occluded_by": boll['occluded_by'],
                        "occlusion_pct": boll['occlusion_pct'],
                        "tip_visibility": attrs.get('tip_visibility', 'unknown'),
                        "base_visibility": attrs.get('base_visibility', 'unknown'),
                        "visible_parts": [
                            [round(v, 2) for v in vp] for vp in boll['visible_parts']
                        ],
                        "whole_estimate": (
                            [round(v, 2) for v in boll['whole_estimate']]
                            if boll['whole_estimate'] else None
                        )
                    }
                }
                coco['annotations'].append(ann)
                ann_id_counter += 1

                stats['total_bolls'] += 1
                if boll['is_occluded']:
                    stats['occluded'] += 1
                    stats['occluders'][boll['occluded_by']] += 1
                    if boll['occlusion_pct'] > 0:
                        stats['occlusion_pcts'].append(boll['occlusion_pct'])
                else:
                    stats['whole_visible'] += 1

            image_id_counter += 1

    print(f"\n{'='*60}\nDataset Summary\n{'='*60}")
    print(f"  Images: {stats['total_images']}")
    print(f"  Boll instances: {stats['total_bolls']}")
    print(f"    Fully visible: {stats['whole_visible']}")
    print(f"    Occluded: {stats['occluded']}")
    print(f"    Multi-polygon bolls (visible_parts or whole_estimate): {stats['multi_polygon_bolls']}")
    if stats['occluders']:
        print(f"  Occluders:")
        for k, v in sorted(stats['occluders'].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
    if stats['occlusion_pcts']:
        print(f"  Occlusion %: mean={np.mean(stats['occlusion_pcts']):.1%}, "
              f"max={max(stats['occlusion_pcts']):.1%}, "
              f"min={min(stats['occlusion_pcts']):.1%}")
    print(f"  Keypoints: tips={stats['tips_matched']}, bases={stats['bases_matched']}")

    return coco


# ── Split ─────────────────────────────────────────────────────────────

def split_dataset(coco, val_split=0.15, seed=42):
    import random
    random.seed(seed)
    ids = [img['id'] for img in coco['images']]
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * val_split))
    val_ids = set(ids[:n_val])

    train = {**coco, "images": [], "annotations": []}
    val = {**coco, "images": [], "annotations": []}

    for img in coco['images']:
        (val if img['id'] in val_ids else train)['images'].append(img)
    for ann in coco['annotations']:
        (val if ann['image_id'] in val_ids else train)['annotations'].append(ann)

    return train, val


def copy_images(sources_info, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for source_idx, (images_data, _, source_dir) in enumerate(sources_info):
        if source_dir is None:
            continue
        for img_data in images_data:
            src = os.path.join(source_dir, img_data['name'])
            dst = os.path.join(output_dir, f"src{source_idx+1}_{img_data['name']}")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
    print(f"Copied {copied} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_files", nargs='+', required=True)
    parser.add_argument("--images_dirs", nargs='*', default=[])
    parser.add_argument("--output", "-o", default="./merged_dataset_v3/backdown")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--no_split", action="store_true")
    args = parser.parse_args()

    all_sources = []
    sources_info = []
    for i, xml_path in enumerate(args.xml_files):
        print(f"Parsing {xml_path}...")
        data = parse_cvat_xml(xml_path)
        source_name = Path(xml_path).parent.name or f"source{i+1}"
        all_sources.append((data, source_name))
        img_dir = args.images_dirs[i] if i < len(args.images_dirs) else None
        sources_info.append((data, source_name, img_dir))

    coco = build_coco(all_sources)

    os.makedirs(args.output, exist_ok=True)
    ann_dir = os.path.join(args.output, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    if args.images_dirs:
        copy_images(sources_info, os.path.join(args.output, "images"))

    if args.no_split:
        out_path = os.path.join(ann_dir, "instances.json")
        with open(out_path, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f"\nSaved: {out_path}")
    else:
        train, val = split_dataset(coco, args.val_split)
        with open(os.path.join(ann_dir, "train.json"), 'w') as f:
            json.dump(train, f, indent=2)
        with open(os.path.join(ann_dir, "val.json"), 'w') as f:
            json.dump(val, f, indent=2)
        print(f"\nTrain: {len(train['images'])} imgs, {len(train['annotations'])} anns")
        print(f"Val:   {len(val['images'])} imgs, {len(val['annotations'])} anns")


if __name__ == "__main__":
    main()