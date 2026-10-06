"""
Predictions → CVAT Import Converter
=====================================
Converts model predictions (JSON) into CVAT-compatible format
for the model-assisted annotation loop.

After your model predicts on new frames, run this to create
pre-annotations that you can import into CVAT and just correct.

Usage:
    python predictions_to_cvat.py --pred_dir ./predictions/frame_predictions \
                                  --images_dir ./new_frames \
                                  --output ./cvat_import.xml \
                                  --confidence 0.5
"""

import os
import json
import argparse
import glob
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent
import cv2


def create_cvat_xml(pred_dir, images_dir, output_path, confidence=0.5):
    """
    Convert prediction JSONs to CVAT XML for import as pre-annotations.
    """
    root = Element("annotations")

    # Version info
    version = SubElement(root, "version")
    version.text = "1.1"

    # Meta
    meta = SubElement(root, "meta")
    task = SubElement(meta, "task")
    name_elem = SubElement(task, "name")
    name_elem.text = "cotton_boll_preannotation"

    # Collect prediction files
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))
    print(f"Found {len(pred_files)} prediction files")

    image_id = 0
    total_annotations = 0

    for pred_file in pred_files:
        with open(pred_file) as f:
            pred_data = json.load(f)

        # Find corresponding image
        source = pred_data.get("source", "")
        # Try to find image file
        img_name = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(images_dir, Path(source).stem + ext)
            if os.path.exists(candidate):
                img_name = Path(source).stem + ext
                img = cv2.imread(candidate)
                h, w = img.shape[:2]
                break

        if img_name is None:
            # Try the source name directly
            candidate = os.path.join(images_dir, source)
            if os.path.exists(candidate):
                img_name = source
                img = cv2.imread(candidate)
                h, w = img.shape[:2]
            else:
                print(f"  Skipping {pred_file}: no matching image found")
                continue

        # Create image element
        image_elem = Element("image")
        image_elem.set("id", str(image_id))
        image_elem.set("name", img_name)
        image_elem.set("width", str(w))
        image_elem.set("height", str(h))

        for det in pred_data.get("detections", []):
            if det.get("score", 0) < confidence:
                continue

            # Add polygon (segmentation)
            if "segmentation" in det:
                poly = det["segmentation"]
                # Convert flat list [x1,y1,x2,y2,...] to "x1,y1;x2,y2;..."
                points = []
                for i in range(0, len(poly), 2):
                    points.append(f"{poly[i]:.2f},{poly[i+1]:.2f}")
                points_str = ";".join(points)

                polygon_elem = SubElement(image_elem, "polygon")
                polygon_elem.set("label", "cotton_boll")
                polygon_elem.set("points", points_str)
                polygon_elem.set("occluded", "0")
                polygon_elem.set("z_order", str(total_annotations))

                # Add confidence as attribute
                attr = SubElement(polygon_elem, "attribute")
                attr.set("name", "confidence")
                attr.text = f"{det['score']:.3f}"

                # Add boll_id placeholder
                attr_id = SubElement(polygon_elem, "attribute")
                attr_id.set("name", "boll_id")
                attr_id.text = "-1"  # To be filled by annotator

                total_annotations += 1

            # Add tip keypoint
            if "tip" in det and det["tip"].get("visible", False):
                tip = det["tip"]
                point_elem = SubElement(image_elem, "points")
                point_elem.set("label", "tip")
                point_elem.set("points", f"{tip['x']:.2f},{tip['y']:.2f}")
                point_elem.set("occluded", "0")
                point_elem.set("z_order", str(total_annotations))
                total_annotations += 1

            # Add base keypoint
            if "base" in det and det["base"].get("visible", False):
                base = det["base"]
                point_elem = SubElement(image_elem, "points")
                point_elem.set("label", "base")
                point_elem.set("points", f"{base['x']:.2f},{base['y']:.2f}")
                point_elem.set("occluded", "0")
                point_elem.set("z_order", str(total_annotations))
                total_annotations += 1

        root.append(image_elem)
        image_id += 1

    # Write XML
    tree = ElementTree(root)
    indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"\nCreated CVAT import file: {output_path}")
    print(f"  Images: {image_id}")
    print(f"  Annotations: {total_annotations}")
    print(f"\nTo import into CVAT:")
    print(f"  1. Open your CVAT task")
    print(f"  2. Click 'Upload annotations'")
    print(f"  3. Select format: 'CVAT 1.1'")
    print(f"  4. Upload: {output_path}")
    print(f"  5. Review and correct the pre-annotations")


def main():
    parser = argparse.ArgumentParser(
        description="Convert model predictions to CVAT import format"
    )
    parser.add_argument("--pred_dir", required=True,
                        help="Directory with prediction JSON files")
    parser.add_argument("--images_dir", required=True,
                        help="Directory with corresponding images")
    parser.add_argument("--output", default="./cvat_import.xml",
                        help="Output CVAT XML file path")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum confidence threshold (default: 0.5)")

    args = parser.parse_args()
    create_cvat_xml(args.pred_dir, args.images_dir,
                    args.output, args.confidence)


if __name__ == "__main__":
    main()
