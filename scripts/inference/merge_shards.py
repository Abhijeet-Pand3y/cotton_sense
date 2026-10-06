"""
merge_shards.py — Merge the output dirs of a sharded inference_viz.py run
(--shard i/N) back into ONE directory that looks exactly like a single-process
run, then (optionally) it is ready for track_detections.py.

WHY
---
inference_viz.py --shard i/N keeps the frame stem (frame_<idx>_t<sec>p<centi>s)
globally consistent across shards, so merging is just a union: no renumbering,
and name-sorted order is true temporal order. Only detections_all.json has to be
concatenated and re-sorted, because each shard wrote its own.

USAGE
  python scripts/inference/merge_shards.py --shards out/sh0 out/sh1 --output out/merged
"""
import argparse, json, shutil
from pathlib import Path

SUBDIRS = ("json", "viz", "frames", "tracked_json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--move", action="store_true",
                   help="Move files instead of copying (faster, consumes shards)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    op = shutil.move if args.move else shutil.copy2

    counts = {}
    for sub in SUBDIRS:
        srcs = [s / sub for s in args.shards if (s / sub).is_dir()]
        if not srcs:
            continue
        dst = args.output / sub
        dst.mkdir(exist_ok=True)
        seen = set()
        for sd in srcs:
            for f in sd.iterdir():
                if f.name in seen:
                    raise SystemExit(
                        f"COLLISION: {f.name} appears in more than one shard's {sub}/ "
                        "— shards overlap, refusing to merge")
                seen.add(f.name)
                op(str(f), str(dst / f.name))
        counts[sub] = len(seen)

    # detections_all.json: concatenate, dedupe by frame index, sort temporally
    combined, frames_seen = [], set()
    for s in args.shards:
        f = s / "detections_all.json"
        if not f.is_file():
            continue
        for rec in json.load(open(f)):
            if rec["frame"] in frames_seen:
                raise SystemExit(f"COLLISION: frame {rec['frame']} in >1 shard")
            frames_seen.add(rec["frame"])
            combined.append(rec)
    if combined:
        combined.sort(key=lambda r: r["frame"])
        out = args.output / "detections_all.json"
        json.dump(combined, open(out, "w"))
        gaps = [i for i, r in enumerate(combined) if r["frame"] != i]
        print(f"Wrote {out}  ({len(combined)} frames)"
              + (f"  [WARN {len(gaps)} index gaps — a shard may be missing]" if gaps else "  [contiguous]"))

    for k, v in counts.items():
        print(f"  {k}/: {v} files")


if __name__ == "__main__":
    main()
