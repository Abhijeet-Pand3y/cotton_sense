"""
extract_for_annotation.py — Pick ~N evenly-spaced frames from a range [x:y] of a
folder of numbered frames (0000.jpg ... 2000.jpg) and copy them into a new folder
ready for CVAT annotation.

Handles one or many source folders. For each source folder it selects frames whose
numeric index falls in [--start, --end], picks --count of them evenly spaced, and
copies them to the output (optionally into a per-source subfolder).

Selection is by the NUMBER in the filename (0000.jpg -> 0), not by position in the
directory listing, so missing frame numbers don't shift the selection.

USAGE
-----
  # one folder, frames 400..1600, ~300 evenly spaced
  python extract_for_annotation.py \
      --src data/rgbd_images/rgbd_1 \
      --start 400 --end 1600 --count 300 \
      --out data/rgbd_images/annotation_set

  # several folders at once, same range each, kept in per-source subfolders
  python extract_for_annotation.py \
      --src data/rgbd_images/rgbd_1 data/rgbd_images/rgbd_2 data/rgbd_images/rgbd_3 \
      --start 400 --end 1600 --count 300 \
      --out data/rgbd_images/annotation_set --subfolders

NOTES
-----
- --count is a target; you get at most that many (fewer if the range has fewer
  frames). Spacing is as even as integer rounding allows.
- Without --subfolders, all selected frames go flat into --out. If two source
  folders contain the same filename (e.g. both have 0500.jpg) they would collide;
  the script prefixes with the source folder name in that case to avoid overwrites.
- Files are COPIED, never moved. Sources are left untouched.
"""

import argparse
import re
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NUM_RE = re.compile(r"(\d+)")


def frame_number(path: Path):
    """Extract the first integer in the filename, or None."""
    m = NUM_RE.search(path.stem)
    return int(m.group(1)) if m else None


def evenly_spaced(items, n):
    """Pick n evenly-spaced items from a list (preserves order)."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    idx = sorted(set(round(i * (len(items) - 1) / (n - 1)) for i in range(n)))
    return [items[i] for i in idx]


def select_from_folder(src: Path, start: int, end: int, count: int):
    """Return the chosen Paths from one source folder, in frame order."""
    frames = []
    for p in src.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        num = frame_number(p)
        if num is None:
            continue
        if start <= num <= end:
            frames.append((num, p))
    frames.sort(key=lambda t: t[0])
    ordered = [p for _, p in frames]
    return evenly_spaced(ordered, count)


def parse_src_spec(spec, default_start, default_end):
    """Parse a --src entry that may carry its own range as PATH:START:END.

    Returns (Path, start, end). Bare paths fall back to the global --start/--end.
    Windows drive letters (C:\\...) are handled by splitting from the RIGHT and
    only treating trailing tokens as ints.
    """
    parts = spec.split(":")
    # peel up to two trailing integer tokens as end, start
    nums = []
    while len(parts) > 1 and len(nums) < 2 and parts[-1].lstrip("-").isdigit():
        nums.insert(0, int(parts.pop()))
    path = ":".join(parts)
    if len(nums) == 2:
        return Path(path), nums[0], nums[1]
    if len(nums) == 1:
        # only one trailing int is ambiguous; treat as end, keep default start
        return Path(path), default_start, nums[0]
    return Path(path), default_start, default_end


def main():
    p = argparse.ArgumentParser(description="Extract evenly-spaced frames for CVAT")
    p.add_argument("--src", nargs="+", required=True,
                   help="Source folders. Each may carry its own range as "
                        "PATH:START:END (e.g. data/rgbd_images/rgbd_1:350:1700). "
                        "Bare paths use the global --start/--end.")
    p.add_argument("--start", type=int, default=0,
                   help="Default first frame number (used when a src has no range)")
    p.add_argument("--end", type=int, default=10**9,
                   help="Default last frame number (used when a src has no range)")
    p.add_argument("--count", type=int, default=300,
                   help="Approx number of frames to pick per source (default 300)")
    p.add_argument("--out", required=True, help="Output folder for the annotation set")
    p.add_argument("--subfolders", action="store_true",
                   help="Keep each source's frames in its own subfolder under --out")
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for src_str in args.src:
        src, start, end = parse_src_spec(src_str, args.start, args.end)
        if start > end:
            print(f"[warn] {src.name}: start ({start}) > end ({end}), skipping")
            continue
        if not src.is_dir():
            print(f"[warn] not a directory, skipping: {src}")
            continue

        chosen = select_from_folder(src, start, end, args.count)
        if not chosen:
            print(f"[{src.name}] no frames in [{start},{end}]")
            continue

        if args.subfolders:
            dst_dir = out_root / src.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            prefix = ""
        else:
            dst_dir = out_root
            prefix = f"{src.name}_"

        copied = 0
        for pth in chosen:
            dst = dst_dir / f"{prefix}{pth.name}"
            if not dst.exists():
                shutil.copy2(pth, dst)
                copied += 1

        first_n = frame_number(chosen[0])
        last_n = frame_number(chosen[-1])
        print(f"[{src.name}] range [{start},{end}] -> selected {len(chosen)} "
              f"frames (#{first_n}..#{last_n}) -> {dst_dir}  (copied {copied})")
        grand_total += len(chosen)

    print(f"\nTotal frames selected: {grand_total}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
