"""
combine_datasets.py — Merge multiple CVAT cotton-boll tasks into one COCO-format
dataset for joint training, with a proper temporal (sequential) val split.

Key design points:
  - Tasks are NOT synchronized in time across cameras. Each task's image IDs
    are local 0..N-1 indices. We treat each task independently.
  - Val split is the LAST `val_split` fraction of each task's frames in
    image-ID order. For video tasks this is a held-out temporal tail — a real
    generalization test, not a random scatter.
  - --val_random flag for tasks that are NOT consecutive video (e.g. shape
    tasks with manually selected frames). Per-task control via comma list.
  - --strategy balanced keeps all frames from shape tasks but stride-subsamples
    the track task's TRAIN portion to `--track_sample` frames so its 300-frame
    bulk doesn't drown out the 17 / 29 from the other angles. Val tail of
    track is left alone (held out in full).
  - Output JSON file_names are prefixed by task name (track/, front/, back/)
    so identical CVAT image names across tasks don't collide. Your images
    directory at train time must mirror this layout.

Usage:
    python combine_datasets.py \
      --track_xml ./right_down_fixed.xml --track_images ./right_down/images \
      --front_xml ./front_down.xml       --front_images ./front_down/images \
      --back_xml  ./back_down.xml        --back_images  ./back_down/images \
      --output ./dataset_combined \
      --strategy balanced --track_sample 50 \
      --val_random front,back     # shape tasks: random val. track stays sequential.
"""

import argparse
import random
from pathlib import Path

from cvat_to_coco_v4 import convert_xml, write_coco


def stride_sample(items, n):
    """Pick n items evenly spaced across a list."""
    if n >= len(items):
        return items
    indices = sorted(set(round(i * (len(items) - 1) / (n - 1)) for i in range(n)))
    return [items[i] for i in indices]


def split_task(images_sorted, val_split, random_val, seed):
    """Return (train_images, val_images) for one task.

    images_sorted: list of image dicts sorted by image_id (chronological for video tasks).
    random_val: True -> random shuffle then take tail. False -> sequential tail.
    """
    n = len(images_sorted)
    n_val = max(1, int(n * val_split))

    if random_val:
        rng = random.Random(seed)
        shuffled = list(images_sorted)
        rng.shuffle(shuffled)
        return shuffled[n_val:], shuffled[:n_val]
    else:
        return images_sorted[:n - n_val], images_sorted[n - n_val:]


def parse_args():
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
                   help="(balanced) how many track TRAIN frames to keep "
                        "(after val tail is held out)")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--val_random", default="",
                   help="Comma-separated task names that should use random "
                        "(not sequential) val split.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_val_tasks = {t.strip() for t in args.val_random.split(",") if t.strip()}

    task_specs = [
        ("track", args.track_xml, args.track_images),
        ("front", args.front_xml, args.front_images),
        ("back",  args.back_xml,  args.back_images),
    ]

    per_task = {}
    for name, xml, imgs in task_specs:
        images, anns = convert_xml(Path(xml), Path(imgs))
        for im in images:
            im["file_name"] = f"{name}/{im['file_name']}"
        anns_by_img = {}
        for a in anns:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        images_sorted = sorted(images, key=lambda im: im["id"])

        is_random = name in random_val_tasks
        train_imgs, val_imgs = split_task(images_sorted, args.val_split,
                                          is_random, args.seed)

        split_kind = "random" if is_random else "sequential tail"
        print(f"[{name}] {len(images)} total -> split ({split_kind}): "
              f"{len(train_imgs)} train + {len(val_imgs)} val")

        if args.strategy == "balanced" and name == "track":
            before = len(train_imgs)
            train_imgs = stride_sample(train_imgs, args.track_sample)
            print(f"        [balanced] track train subsampled: {before} -> {len(train_imgs)}")

        per_task[name] = {
            "train_imgs": train_imgs,
            "val_imgs": val_imgs,
            "anns_by_img": anns_by_img,
        }

    train_imgs_out, train_anns_out = [], []
    val_imgs_out, val_anns_out = [], []
    next_img_id = 0
    next_ann_id = 0

    def emit(imgs, anns_by_img, name, target_imgs, target_anns):
        nonlocal next_img_id, next_ann_id
        for im in imgs:
            new_id = next_img_id
            next_img_id += 1
            new_im = dict(im)
            old_id = im["id"]
            new_im["id"] = new_id
            new_im["task"] = name
            target_imgs.append(new_im)
            for a in anns_by_img.get(old_id, []):
                new_a = dict(a)
                new_a["id"] = next_ann_id
                next_ann_id += 1
                new_a["image_id"] = new_id
                new_a["task"] = name
                target_anns.append(new_a)

    for name in ["track", "front", "back"]:
        d = per_task[name]
        emit(d["train_imgs"], d["anns_by_img"], name, train_imgs_out, train_anns_out)
        emit(d["val_imgs"],   d["anns_by_img"], name, val_imgs_out,   val_anns_out)

    write_coco(train_imgs_out, train_anns_out, out_dir / "train.json")
    write_coco(val_imgs_out,   val_anns_out,   out_dir / "val.json")

    print(f"\n[merged] train: {len(train_imgs_out)} images / {len(train_anns_out)} anns")
    print(f"          val:   {len(val_imgs_out)} images / {len(val_anns_out)} anns")
    print(f"\nPer-task in val (for stratified evaluation):")
    for name in ["track", "front", "back"]:
        n = sum(1 for im in val_imgs_out if im["task"] == name)
        print(f"  {name}: {n} images")


if __name__ == "__main__":
    main()