# combine_datasets.py
"""
Combine right_down_fixed.xml + back_down_shape + front_down_shape
into one COCO-format dataset for joint training.

Two strategies:
  --strategy all          : use every frame from every task
  --strategy balanced     : sample N from track, all from shape tasks
"""

import argparse
import json
import random
from pathlib import Path

from cvat_to_coco_v4 import convert_xml, write_coco, split_train_val


def stride_sample(items, n):
    """Pick n items evenly spaced from a list."""
    if n >= len(items):
        return items
    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[i] for i in sorted(set(indices))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track_xml", required=True)
    p.add_argument("--track_images", required=True)
    p.add_argument("--front_xml", required=True)
    p.add_argument("--front_images", required=True)
    p.add_argument("--back_xml", required=True)
    p.add_argument("--back_images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strategy", choices=["all", "balanced"], default="balanced")
    p.add_argument("--track_sample", type=int, default=50,
                   help="(balanced strategy) how many frames to keep from track")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert each task to COCO records (in memory)
    tasks = [
        ("track", args.track_xml, args.track_images),
        ("front", args.front_xml, args.front_images),
        ("back",  args.back_xml,  args.back_images),
    ]
    per_task = []
    for name, xml, imgs in tasks:
        images, anns = convert_xml(Path(xml), Path(imgs))
        # Prefix file names so identical names across tasks don't collide
        for im in images:
            im["file_name"] = f"{name}/{im['file_name']}"
        per_task.append((name, images, anns))
        print(f"[{name}] {len(images)} images, {len(anns)} annotations")

    # Subsample track if balanced strategy
    if args.strategy == "balanced":
        track_name, track_imgs, track_anns = per_task[0]
        # Sort by id so stride is temporally even
        track_imgs_sorted = sorted(track_imgs, key=lambda im: im["id"])
        kept = stride_sample(track_imgs_sorted, args.track_sample)
        kept_ids = {im["id"] for im in kept}
        track_anns = [a for a in track_anns if a["image_id"] in kept_ids]
        per_task[0] = (track_name, kept, track_anns)
        print(f"[track] sampled down to {len(kept)} images, {len(track_anns)} annotations")

    # Merge with globally unique image and annotation ids
    all_images, all_anns = [], []
    next_img_id = 0
    next_ann_id = 0
    for name, images, anns in per_task:
        id_map = {}
        for im in images:
            new_id = next_img_id
            next_img_id += 1
            id_map[im["id"]] = new_id
            new_im = dict(im)
            new_im["id"] = new_id
            new_im["task"] = name
            all_images.append(new_im)
        for a in anns:
            new_a = dict(a)
            new_a["id"] = next_ann_id
            next_ann_id += 1
            new_a["image_id"] = id_map[a["image_id"]]
            new_a["task"] = name
            all_anns.append(new_a)

    print(f"\n[merged] {len(all_images)} images, {len(all_anns)} annotations")

    # Stratified train/val split: hold out the same fraction from each task
    train_imgs, train_anns, val_imgs, val_anns = [], [], [], []
    for name, *_ in tasks:
        task_imgs = [im for im in all_images if im["task"] == name]
        random.shuffle(task_imgs)
        n_val = max(1, int(len(task_imgs) * args.val_split))
        v_ids = {im["id"] for im in task_imgs[:n_val]}
        train_imgs += [im for im in task_imgs if im["id"] not in v_ids]
        val_imgs   += [im for im in task_imgs if im["id"] in v_ids]
        train_anns += [a for a in all_anns if a["image_id"] in {im["id"] for im in task_imgs} - v_ids]
        val_anns   += [a for a in all_anns if a["image_id"] in v_ids]

    write_coco(train_imgs, train_anns, out_dir / "train.json")
    write_coco(val_imgs, val_anns, out_dir / "val.json")
    print(f"\n[split] train: {len(train_imgs)} images / {len(train_anns)} anns")
    print(f"        val:   {len(val_imgs)} images / {len(val_anns)} anns")
    print(f"\nPer-task in val:")
    for name, *_ in tasks:
        n = sum(1 for im in val_imgs if im["task"] == name)
        print(f"  {name}: {n} images")


if __name__ == "__main__":
    main()