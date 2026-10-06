"""
copy_boll_id_track.py — Copies boll_id_track → boll_id for every shape in a CVAT XML.

boll_id_track is the immutable, reliable attribute set at track creation.
boll_id may have drifted due to being accidentally set as mutable.

This script overwrites boll_id with whatever boll_id_track says, on every shape
that has both attributes. Shapes with no boll_id_track are left untouched.

Writes a new XML file (default: <input>_fixed.xml). The original is never modified.

USAGE:
    python copy_boll_id_track.py --xml left_down_annotations.xml
    python copy_boll_id_track.py --xml left_down_annotations.xml --output left_down_fixed.xml
    python copy_boll_id_track.py --xml left_down_annotations.xml --dry_run
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def copy_boll_id_track(xml_path: Path, output_path: Path, dry_run: bool = False):
    ET.register_namespace("", "")  # preserve existing namespaces
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stats = {
        "shapes_visited": 0,
        "updated": 0,       # boll_id_track → boll_id written
        "already_match": 0, # both existed and were already equal
        "no_track_attr": 0, # shape had no boll_id_track, skipped
        "no_boll_id": 0,    # shape had boll_id_track but no boll_id attr node (created)
    }

    for image in root.findall(".//image"):
        for shape in image:
            stats["shapes_visited"] += 1

            attrs = {a.get("name"): a for a in shape.findall("attribute")}

            track_node = attrs.get("boll_id_track")
            id_node    = attrs.get("boll_id")

            if track_node is None:
                stats["no_track_attr"] += 1
                continue

            track_val = (track_node.text or "").strip()

            if id_node is None:
                # boll_id attribute element doesn't exist — create it
                stats["no_boll_id"] += 1
                new_attr = ET.SubElement(shape, "attribute")
                new_attr.set("name", "boll_id")
                new_attr.text = track_val
                stats["updated"] += 1
            else:
                id_val = (id_node.text or "").strip()
                if id_val == track_val:
                    stats["already_match"] += 1
                else:
                    id_node.text = track_val
                    stats["updated"] += 1

    # report
    print(f"Shapes visited   : {stats['shapes_visited']}")
    print(f"Updated          : {stats['updated']}  (boll_id_track → boll_id)")
    print(f"Already matched  : {stats['already_match']}")
    print(f"No boll_id_track : {stats['no_track_attr']}  (skipped)")
    print(f"boll_id created  : {stats['no_boll_id']}  (attr node didn't exist)")

    if dry_run:
        print("\n[DRY RUN] No file written.")
        return

    # ET.write preserves structure but re-serialises — make a backup just in case
    backup = output_path.with_suffix(".xml.bak") if output_path == xml_path else None
    if backup:
        shutil.copy2(xml_path, backup)
        print(f"\nBackup written : {backup}")

    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
    print(f"Output written : {output_path}")


def main():
    p = argparse.ArgumentParser(description="Copy boll_id_track → boll_id in CVAT XML")
    p.add_argument("--xml", required=True, help="Input CVAT XML file")
    p.add_argument("--output", default=None,
                   help="Output path (default: <input>_fixed.xml)")
    p.add_argument("--dry_run", action="store_true",
                   help="Parse and report counts without writing anything")
    args = p.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = xml_path.with_name(xml_path.stem + "_fixed.xml")

    copy_boll_id_track(xml_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
