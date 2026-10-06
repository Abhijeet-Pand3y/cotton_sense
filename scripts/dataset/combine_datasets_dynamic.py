# combine_datasets_dynamic.py
"""
Dynamic multi-angle COCO combiner for the cotton boll pose pipeline.

Instead of hardcoding three angles (track/front/back), this scans a "combine set"
folder and pulls in every angle it finds. Drop a new angle in and re-run — no code
edits needed.

------------------------------------------------------------------------------
DISCOVERY CONVENTION
------------------------------------------------------------------------------
Point --combine_set at a folder laid out like:

    data/annotations/_combine_set/
        left_down/      left_down_annotations_fixed.xml   (exactly one .xml)
        right_down/     right_down_fixed.xml
        front_down/     front_down_shape_annotations.xml
        back_down/      back_down_shape_annotations.xml

For each subfolder <angle>/ the script:
  - finds the single .xml inside it (errors if 0 or >1, unless --xml_glob narrows it)
  - pairs it with images at  <images_root>/<angle>/   (override map via --image_dir_for)
  - uses <angle> as the COCO file_name prefix, so frames never collide across angles

The merged dataset's file_name fields look like  left_down/frame_000123.jpg , so at
train/eval time pass  --images_dir <images_root>  (the parent of all angle folders),
OR copy each angle's frames under  <output>/<angle>/  (see --stage_images).

------------------------------------------------------------------------------
SPLIT (this run)
------------------------------------------------------------------------------
This run is inference-oriented: nearly everything goes to train. A small val slice
is still written so COCOEvaluator has something to run on.
  --val_count N   : hold out N images TOTAL for val (default 8), taken evenly across
                    angles by stride (temporally spread, not random adjacent frames).
  --val_split 0   : alias — if set >0, use a fraction instead of a fixed count.

NOTE: this is deliberately crude. A future version should do per-angle, mode-aware
splits (sequential tail for track-mode angles, random for shape-mode). Not now.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python combine_datasets_dynamic.py \
        --combine_set data/annotations/_combine_set \
        --images_root data/images \
        --output datasets/combined_all_v2 \
        --val_count 8

    # If an angle's images live somewhere off-convention:
    python combine_datasets_dynamic.py ... \
        --image_dir_for left_down=data/images/extracted_frames/left_down
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from cvat_to_coco_v4 import convert_xml, write_coco


# ---------------------------------------------------------------------------

def stride_sample(items, n):
    """Pick n items evenly spaced from a list (preserves order)."""
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]  # single item -> take the middle
    idx = sorted(set(round(i * (len(items) - 1) / (n - 1)) for i in range(n)))
    return [items[i] for i in idx]


def parse_override_map(pairs):
    """--image_dir_for left_down=path/to/imgs  ->  {'left_down': Path(...)}"""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--image_dir_for expects angle=path, got: {pair}")
        angle, path = pair.split("=", 1)
        out[angle.strip()] = Path(path.strip())
    return out


def find_xml(angle_dir: Path, xml_glob: str):
    """Find the single annotation XML inside an angle folder."""
    candidates = sorted(angle_dir.glob(xml_glob))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No XML matching '{xml_glob}' in {angle_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple XMLs in {angle_dir}: {[c.name for c in candidates]}. "
            f"Narrow with --xml_glob or keep one per folder.")
    return candidates[0]


def discover_angles(combine_set: Path, images_root: Path,
                    overrides: dict, xml_glob: str):
    """Return list of (angle, xml_path, images_dir), sorted by angle name."""
    if not combine_set.is_dir():
        raise FileNotFoundError(f"--combine_set not a directory: {combine_set}")

    angles = []
    for sub in sorted(p for p in combine_set.iterdir() if p.is_dir()):
        angle = sub.name
        xml_path = find_xml(sub, xml_glob)
        images_dir = overrides.get(angle, images_root / angle)
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Image dir for angle '{angle}' not found: {images_dir}\n"
                f"  Fix with --image_dir_for {angle}=<path>")
        angles.append((angle, xml_path, images_dir))

    if not angles:
        raise RuntimeError(f"No angle subfolders found in {combine_set}")
    return angles


def stage_images(all_images, angles, out_dir: Path, copy: bool):
    """
    Materialise referenced frames under <out_dir>/<angle>/ so the dataset is
    self-contained. file_name is already 'angle/frame.jpg', so the staged path
    is <out_dir>/<file_name> and training uses --images_dir <out_dir>.
    """
    # angle -> source images_dir
    src_for = {angle: images_dir for angle, _, images_dir in angles}

    staged, missing, existed = 0, 0, 0
    for im in all_images:
        angle = im["task"]
        # file_name is 'angle/basename' — strip the prefix to get the basename
        basename = Path(im["file_name"]).name
        src = src_for[angle] / basename
        dst_dir = out_dir / angle
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / basename

        if dst.exists() or dst.is_symlink():
            existed += 1
            continue
        if not src.exists():
            missing += 1
            print(f"  [warn] source frame missing: {src}")
            continue

        if copy:
            shutil.copy2(src, dst)
        else:
            # relative symlink so the dataset survives a move
            rel = os.path.relpath(src.resolve(), dst_dir.resolve())
            os.symlink(rel, dst)
        staged += 1

    mode = "copied" if copy else "symlinked"
    print(f"\n[stage] {mode} {staged} frames, {existed} already present, {missing} missing")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Dynamic multi-angle COCO combiner")
    p.add_argument("--combine_set", required=True, type=Path,
                   help="Folder containing one subfolder per angle")
    p.add_argument("--images_root", required=True, type=Path,
                   help="Parent of per-angle image folders (default convention)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--image_dir_for", nargs="*", default=[],
                   help="Per-angle image dir overrides: angle=path (repeatable)")
    p.add_argument("--xml_glob", default="*.xml",
                   help="Glob to locate the XML in each angle folder (default *.xml)")
    p.add_argument("--val_count", type=int, default=8,
                   help="Total images held out for val (stride-sampled across angles)")
    p.add_argument("--val_split", type=float, default=0.0,
                   help="If >0, use this fraction per angle instead of --val_count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage_images", action="store_true",
                   help="Symlink each angle's frames under <output>/<angle>/ so the "
                        "dataset is self-contained. Train with --images_dir <output>.")
    p.add_argument("--stage_copy", action="store_true",
                   help="With --stage_images, copy files instead of symlinking "
                        "(use if symlinks don't survive your filesystem/move).")
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = parse_override_map(args.image_dir_for)
    angles = discover_angles(args.combine_set, args.images_root, overrides, args.xml_glob)

    print("Discovered angles:")
    for angle, xml_path, images_dir in angles:
        print(f"  {angle:14s} xml={xml_path.name}  images={images_dir}")
    print()

    # --- Convert each angle, prefix file_names by angle ---
    per_angle = []
    for angle, xml_path, images_dir in angles:
        images, anns = convert_xml(xml_path, images_dir)
        for im in images:
            im["file_name"] = f"{angle}/{im['file_name']}"
        per_angle.append((angle, images, anns))
        print(f"[{angle}] {len(images)} images, {len(anns)} annotations")

    # --- Merge with globally unique ids ---
    all_images, all_anns = [], []
    next_img_id, next_ann_id = 0, 0
    for angle, images, anns in per_angle:
        id_map = {}
        for im in images:
            new_id = next_img_id
            next_img_id += 1
            id_map[im["id"]] = new_id
            new_im = dict(im)
            new_im["id"] = new_id
            new_im["task"] = angle
            all_images.append(new_im)
        for a in anns:
            new_a = dict(a)
            new_a["id"] = next_ann_id
            next_ann_id += 1
            new_a["image_id"] = id_map[a["image_id"]]
            new_a["task"] = angle
            all_anns.append(new_a)

    print(f"\n[merged] {len(all_images)} images, {len(all_anns)} annotations")

    # --- Pick val slice ---
    val_ids = set()
    if args.val_split and args.val_split > 0:
        # fraction per angle, random
        for angle, *_ in per_angle:
            imgs = [im for im in all_images if im["task"] == angle]
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs) * args.val_split))
            val_ids.update(im["id"] for im in imgs[:n_val])
    else:
        # fixed total count, stride-sampled across angles (temporally spread)
        n_angles = len(per_angle)
        per_angle_val = max(1, args.val_count // n_angles) if args.val_count > 0 else 0
        for angle, *_ in per_angle:
            imgs = sorted((im for im in all_images if im["task"] == angle),
                          key=lambda im: im["id"])
            for im in stride_sample(imgs, per_angle_val):
                val_ids.add(im["id"])

    train_imgs = [im for im in all_images if im["id"] not in val_ids]
    val_imgs   = [im for im in all_images if im["id"] in val_ids]
    train_anns = [a for a in all_anns if a["image_id"] not in val_ids]
    val_anns   = [a for a in all_anns if a["image_id"] in val_ids]

    write_coco(train_imgs, train_anns, out_dir / "train.json")
    write_coco(val_imgs, val_anns, out_dir / "val.json")

    print(f"\n[split] train: {len(train_imgs)} images / {len(train_anns)} anns")
    print(f"        val:   {len(val_imgs)} images / {len(val_anns)} anns")
    print(f"\nPer-angle counts:")
    for angle, *_ in per_angle:
        tr = sum(1 for im in train_imgs if im["task"] == angle)
        va = sum(1 for im in val_imgs if im["task"] == angle)
        print(f"  {angle:14s} train={tr:4d}  val={va:3d}")

    # Optionally materialise frames into the dataset folder
    if args.stage_images:
        stage_images(all_images, angles, out_dir, copy=args.stage_copy)
        images_dir_for_training = str(out_dir)
    else:
        images_dir_for_training = str(args.images_root)

    # Write a small manifest of what went in
    manifest = {
        "angles": [
            {"angle": a, "xml": str(x), "images_dir": str(i)}
            for a, x, i in angles
        ],
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "train_anns": len(train_anns),
        "val_anns": len(val_anns),
        "staged": bool(args.stage_images),
        "images_dir_for_training": images_dir_for_training,
        "note": ("Staged: frames live under <output>/<angle>/, train with "
                 "--images_dir <output>." if args.stage_images else
                 "Not staged: file_name is angle-prefixed, train with "
                 "--images_dir <images_root> (each angle must resolve there)."),
    }
    with open(out_dir / "combine_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out_dir / 'combine_manifest.json'}")
    print(f"\n>>> Train with:  --images_dir {images_dir_for_training}")


if __name__ == "__main__":
    main()
