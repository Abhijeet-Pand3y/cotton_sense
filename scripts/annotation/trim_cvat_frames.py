"""
trim_cvat_frames.py — Remove annotations past a frame cutoff from a CVAT
"CVAT for images 1.1" XML export.

WHY THIS EXISTS
---------------
In CVAT track mode, a track propagates/interpolates forward across frames. If you
only meant to annotate frames 0..150 but drew tracks, the export can contain
phantom shapes on frames 151+. This strips everything with frame > cutoff while
keeping the shapes on frames 0..cutoff.

Handles BOTH export layouts:
  - track-based:  <track><box frame="N">...</box>...</track>   (shapes per frame)
  - image-based:  <image id="N" name="..."><box.../></image>   (one node per frame)

For track-based: drops child shapes with frame > cutoff; removes tracks left empty.
For image-based: removes <image> elements whose frame index > cutoff.

Frame index source:
  - track children: the `frame` attribute on each shape [authoritative]
  - <image>: the `id` attribute (CVAT image id == frame index) [usually correct];
    falls back to a number parsed from `name` if id is missing.

USAGE
-----
  python trim_cvat_frames.py \
      --xml data/annotations/rgbd_1/rgbd_1.xml \
      --keep_through 150 \
      --out data/annotations/rgbd_1/rgbd_1_trimmed.xml
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SHAPE_TAGS = ("box", "polygon", "polyline", "points", "mask", "ellipse", "cuboid")
NUM_RE = re.compile(r"(\d+)")


def frame_of_image(img_el):
    """Frame index for an <image> element: prefer id, else first int in name."""
    if img_el.get("id") is not None and img_el.get("id").isdigit():
        return int(img_el.get("id"))
    name = img_el.get("name", "")
    m = NUM_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True, help="Input CVAT XML")
    p.add_argument("--keep_through", type=int, required=True,
                   help="Highest frame index to KEEP (inclusive). e.g. 150 keeps 0..150")
    p.add_argument("--out", required=True, help="Output XML path")
    args = p.parse_args()

    cutoff = args.keep_through
    tree = ET.parse(args.xml)
    root = tree.getroot()

    # ---- track-based layout -------------------------------------------------
    tracks = root.findall(".//track")
    n_track_shapes_removed = 0
    n_tracks_removed = 0
    n_track_shapes_kept = 0

    for track in list(tracks):
        # collect and filter child shapes by their own frame attr
        for child in list(track):
            if child.tag in SHAPE_TAGS:
                fr = child.get("frame")
                if fr is not None and fr.isdigit():
                    if int(fr) > cutoff:
                        track.remove(child)
                        n_track_shapes_removed += 1
                    else:
                        n_track_shapes_kept += 1
        # remove track if it now has no shape children
        remaining = [c for c in track if c.tag in SHAPE_TAGS]
        if not remaining:
            root.remove(track)
            n_tracks_removed += 1

    # ---- image-based layout -------------------------------------------------
    images = root.findall(".//image")
    n_images_removed = 0
    n_images_kept = 0
    for img in list(images):
        fr = frame_of_image(img)
        if fr is None:
            print(f"[warn] <image> with unparseable frame: "
                  f"id={img.get('id')} name={img.get('name')} — KEEPING it")
            n_images_kept += 1
            continue
        if fr > cutoff:
            root.remove(img)
            n_images_removed += 1
        else:
            n_images_kept += 1

    # ---- report -------------------------------------------------------------
    print(f"Input : {args.xml}")
    print(f"Cutoff: keep frames 0..{cutoff} (inclusive)")
    if tracks:
        print(f"Track-based layout detected ({len(tracks)} tracks):")
        print(f"  track shapes kept   : {n_track_shapes_kept}")
        print(f"  track shapes removed: {n_track_shapes_removed}")
        print(f"  tracks emptied+removed: {n_tracks_removed}")
    if images:
        print(f"Image-based layout detected ({len(images)} images):")
        print(f"  images kept   : {n_images_kept}")
        print(f"  images removed: {n_images_removed}")
    if not tracks and not images:
        print("[warn] no <track> or <image> elements found — is this a CVAT XML?")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"Wrote : {args.out}")


if __name__ == "__main__":
    main()
