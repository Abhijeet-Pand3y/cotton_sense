"""
validate_annotations.py — Comprehensive validator for cotton boll CVAT XML exports.

Enforces the project's annotation rules and reports every deviation, grouped by
category and boll, with compressed frame ranges. The data is treated as the thing
being checked, NOT as ground truth — a high count for a category means there is a
lot to review, not that the check is wrong.

================================================================================
RULES ENFORCED
================================================================================

1. boll_id REQUIRED
   Every cotton_boll, tip_polygob, and base_rectangle must carry a boll_id != 0.
   Without it the converter cannot group a boll with its tip/base.
   → [MISSING_BOLL_ID]

2. VISIBILITY MUST MATCH THE DRAWING
       tip/base_visibility = visible  →  a quad polygon must be present
       tip/base_visibility = guessed  →  a point must be present
       tip/base_visibility = unknown  →  neither present
   → [TIP_VIS_MISMATCH] / [BASE_VIS_MISMATCH]

3. OCCLUSION STRUCTURE
   If a boll is occluded by a REAL object (leaf / stem / other_boll) in a frame:
       - it MUST have a whole_estimate polygon
       - it MUST have at least one visible_parts polygon
       - it MUST NOT have a whole_visible polygon
   EXCEPTION: frame_edge is NOT a real occluder. A boll cut off by the image
   border is drawn as whole_visible (that polygon acts as the visible part),
   so whole_visible + occluded_by=frame_edge is legal.
   → [OCCLUSION_STRUCTURE]

   Separately, whole_visible and whole_estimate must never coexist on the same
   boll in the same frame (a boll cannot be both fully visible and estimated).
   → [VISIBLE_ESTIMATE_CONFLICT]

4. QUAD VERTEX COUNTS
   tip_polygob and base_rectangle are 4-point quads.
   → [TIP_VERTICES] / [BASE_VERTICES]

5. MASK INTEGRITY
   whole_estimate with occluded_by=none is contradictory.   → [BAD_OCCLUDED_BY]
   whole_estimate with no visible_parts / whole_visible.     → [NO_MASK]
   duplicate whole_estimate / whole_visible per boll/frame.  → [DUP_ESTIMATE]/[DUP_VISIBLE]

6. LINKAGE
   tip/base with no parent cotton_boll.                      → [ORPHAN_TIP]/[ORPHAN_BASE]
   more than one tip/base annotation per boll/frame.         → [DUP_TIP]/[DUP_BASE]

7. GEOMETRY SANITY
   vertex outside image bounds (CVAT shouldn't allow it, kept as a tripwire for
   rotation/orientation mismatches between XML and images). → [OUT_OF_BOUNDS]
   near-zero-area polygon (collinear / duplicate points).    → [DEGENERATE_POLY]
   suspiciously small cotton_boll polygon.                   → [TINY_POLY]

NOTE: tip/base-location-relative-to-boll is intentionally NOT checked. The quad
annotations are drawn with ~2 vertices inside the boll and ~2 outside (to give the
heatmap some background context), so the centroid legitimately sits near or beyond
the boll edge. A location check there would produce false positives.

================================================================================
USAGE
================================================================================
    python validate_annotations.py --xml annotations.xml
    python validate_annotations.py --xml annotations.xml --by_frame
    python validate_annotations.py --xml annotations.xml --max_frame 229
    python validate_annotations.py --xml annotations.xml --no_geometry
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict


# ---- configuration -----------------------------------------------------------
REAL_OCCLUDERS = {"leaf", "stem", "other_boll"}   # frame_edge is the exception
MIN_BOLL_AREA  = 200          # px^2 — below this a cotton_boll is "tiny" (warn)
MIN_POLY_AREA  = 1.0          # px^2 — below this any polygon is "degenerate"


# ---- parsing helpers ---------------------------------------------------------

def parse_points(points_str):
    pts = []
    for chunk in points_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = chunk.split(",")
        if len(xy) != 2:
            continue
        try:
            pts.append((float(xy[0]), float(xy[1])))
        except ValueError:
            continue
    return pts


def polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def compress(frames):
    if not frames:
        return ""
    frames = sorted(set(frames))
    out = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            out.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = f
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(out)


def group_by_boll(items):
    by = defaultdict(list)
    for tup in items:
        by[tup[1]].append(tup[0])
    out = []
    for bid in sorted(by.keys(), key=lambda x: int(x) if str(x).isdigit() else 1_000_000):
        out.append((bid, len(by[bid]), compress(by[bid])))
    return out


# ---- main validation ---------------------------------------------------------

def validate(xml_path, max_frame=None, do_geometry=True):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    issues = defaultdict(list)
    image_count = 0

    frame_bolls = defaultdict(lambda: defaultdict(lambda: {
        "shapes": defaultdict(int),         # annotation_type -> count
        "occluders": set(),                 # occluded_by values seen (non-none)
        "tip_polys": 0, "base_rects": 0,
        "tip_points": 0, "base_points": 0,
        "tip_visibility": None,
        "base_visibility": None,
    }))

    for image in root.findall(".//image"):
        try:
            frame = int(image.get("id", -1))
        except (TypeError, ValueError):
            continue
        if frame < 0:
            continue
        if max_frame is not None and frame > max_frame:
            continue
        image_count += 1

        img_w = float(image.get("width", 0) or 0)
        img_h = float(image.get("height", 0) or 0)

        for shape in image:
            label = shape.get("label", "")
            attrs = {a.get("name"): (a.text.strip() if a.text else "")
                     for a in shape.findall("attribute")}
            boll_id = attrs.get("boll_id", "")
            ann_type = attrs.get("annotation_type", "")
            occluded_by = attrs.get("occluded_by", "")
            pts = parse_points(shape.get("points", ""))
            valid_id = bool(boll_id) and boll_id != "0"

            # geometry: out of bounds (any shape)
            if do_geometry and img_w > 0 and img_h > 0:
                for (px, py) in pts:
                    if px < -1 or py < -1 or px > img_w + 1 or py > img_h + 1:
                        issues["OUT_OF_BOUNDS"].append(
                            (frame, boll_id or "(none)",
                             f"{label} pt ({px:.0f},{py:.0f}) vs img {img_w:.0f}x{img_h:.0f}"))
                        break

            if label == "cotton_boll":
                if not valid_id:
                    issues["MISSING_BOLL_ID"].append((frame, "(unset)", f"cotton_boll/{ann_type}"))
                if ann_type == "whole_estimate" and occluded_by == "none":
                    issues["BAD_OCCLUDED_BY"].append((frame, boll_id))
                if do_geometry and pts:
                    area = polygon_area(pts)
                    if area < MIN_POLY_AREA:
                        issues["DEGENERATE_POLY"].append((frame, boll_id, f"cotton_boll area={area:.1f}"))
                    elif area < MIN_BOLL_AREA:
                        issues["TINY_POLY"].append((frame, boll_id, f"area={area:.0f}px2"))
                if valid_id:
                    d = frame_bolls[frame][boll_id]
                    d["shapes"][ann_type] += 1
                    if occluded_by and occluded_by != "none":
                        d["occluders"].add(occluded_by)
                    if ann_type in ("whole_visible", "whole_estimate"):
                        d["tip_visibility"] = attrs.get("tip_visibility", "unknown")
                        d["base_visibility"] = attrs.get("base_visibility", "unknown")

            elif label == "tip_polygob":
                if not valid_id:
                    issues["MISSING_BOLL_ID"].append((frame, "(unset)", "tip_polygob"))
                elif len(pts) != 4:
                    issues["TIP_VERTICES"].append((frame, boll_id, len(pts)))
                if do_geometry and pts and polygon_area(pts) < MIN_POLY_AREA:
                    issues["DEGENERATE_POLY"].append((frame, boll_id, "tip_polygob ~0 area"))
                if valid_id:
                    frame_bolls[frame][boll_id]["tip_polys"] += 1

            elif label == "base_rectangle":
                if not valid_id:
                    issues["MISSING_BOLL_ID"].append((frame, "(unset)", "base_rectangle"))
                elif len(pts) != 4:
                    issues["BASE_VERTICES"].append((frame, boll_id, len(pts)))
                if do_geometry and pts and polygon_area(pts) < MIN_POLY_AREA:
                    issues["DEGENERATE_POLY"].append((frame, boll_id, "base_rectangle ~0 area"))
                if valid_id:
                    frame_bolls[frame][boll_id]["base_rects"] += 1

            elif label == "tip":
                if valid_id:
                    frame_bolls[frame][boll_id]["tip_points"] += 1

            elif label == "base":
                if valid_id:
                    frame_bolls[frame][boll_id]["base_points"] += 1

    # ---- aggregated per-boll checks ----
    for frame, bolls in frame_bolls.items():
        for boll_id, d in bolls.items():
            s = d["shapes"]
            wv = s.get("whole_visible", 0)
            vp = s.get("visible_parts", 0)
            we = s.get("whole_estimate", 0)

            # mask integrity
            if we and not vp and not wv:
                issues["NO_MASK"].append((frame, boll_id))
            if we > 1:
                issues["DUP_ESTIMATE"].append((frame, boll_id))
            if wv > 1:
                issues["DUP_VISIBLE"].append((frame, boll_id))

            # ── OCCLUSION STRUCTURE ──
            # Real occluder present (leaf/stem/other_boll; frame_edge excluded)?
            has_real_occluder = bool(d["occluders"] & REAL_OCCLUDERS)
            if has_real_occluder:
                if wv:
                    issues["OCCLUSION_STRUCTURE"].append(
                        (frame, boll_id, "occluded but has whole_visible (should be estimate+parts)"))
                if not we:
                    issues["OCCLUSION_STRUCTURE"].append(
                        (frame, boll_id, "occluded but missing whole_estimate"))
                if not vp:
                    issues["OCCLUSION_STRUCTURE"].append(
                        (frame, boll_id, "occluded but missing visible_parts"))

            # whole_visible + whole_estimate can never coexist
            if wv and we:
                issues["VISIBLE_ESTIMATE_CONFLICT"].append((frame, boll_id))

            n_tp = d["tip_polys"]; n_tpt = d["tip_points"]
            n_br = d["base_rects"]; n_bpt = d["base_points"]
            has_parent = bool(s)

            # linkage
            if (n_tp or n_tpt) and not has_parent:
                issues["ORPHAN_TIP"].append((frame, boll_id))
            if (n_br or n_bpt) and not has_parent:
                issues["ORPHAN_BASE"].append((frame, boll_id))
            if n_tp + n_tpt > 1:
                issues["DUP_TIP"].append((frame, boll_id, n_tp, n_tpt))
            if n_br + n_bpt > 1:
                issues["DUP_BASE"].append((frame, boll_id, n_br, n_bpt))

            # visibility consistency — tip
            tv = d["tip_visibility"]
            if tv is not None:
                if tv == "visible" and (n_tp != 1 or n_tpt > 0):
                    issues["TIP_VIS_MISMATCH"].append((frame, boll_id, f"vis=visible tp={n_tp} tpt={n_tpt}"))
                elif tv == "guessed" and (n_tpt != 1 or n_tp > 0):
                    issues["TIP_VIS_MISMATCH"].append((frame, boll_id, f"vis=guessed tp={n_tp} tpt={n_tpt}"))
                elif tv == "unknown" and (n_tp > 0 or n_tpt > 0):
                    issues["TIP_VIS_MISMATCH"].append((frame, boll_id, f"vis=unknown tp={n_tp} tpt={n_tpt}"))

            # visibility consistency — base
            bv = d["base_visibility"]
            if bv is not None:
                if bv == "visible" and (n_br != 1 or n_bpt > 0):
                    issues["BASE_VIS_MISMATCH"].append((frame, boll_id, f"vis=visible br={n_br} bpt={n_bpt}"))
                elif bv == "guessed" and (n_bpt != 1 or n_br > 0):
                    issues["BASE_VIS_MISMATCH"].append((frame, boll_id, f"vis=guessed br={n_br} bpt={n_bpt}"))
                elif bv == "unknown" and (n_br > 0 or n_bpt > 0):
                    issues["BASE_VIS_MISMATCH"].append((frame, boll_id, f"vis=unknown br={n_br} bpt={n_bpt}"))

    return issues, image_count


# ---- reporting ---------------------------------------------------------------

CATEGORY_ORDER = [
    "MISSING_BOLL_ID",
    "TIP_VIS_MISMATCH", "BASE_VIS_MISMATCH",
    "OCCLUSION_STRUCTURE", "VISIBLE_ESTIMATE_CONFLICT",
    "TIP_VERTICES", "BASE_VERTICES",
    "BAD_OCCLUDED_BY", "NO_MASK", "DUP_ESTIMATE", "DUP_VISIBLE",
    "ORPHAN_TIP", "ORPHAN_BASE", "DUP_TIP", "DUP_BASE",
    "OUT_OF_BOUNDS", "DEGENERATE_POLY", "TINY_POLY",
]

GROUP_BY_BOLL = {"NO_MASK", "DUP_ESTIMATE", "DUP_VISIBLE",
                 "ORPHAN_TIP", "ORPHAN_BASE", "BAD_OCCLUDED_BY",
                 "TINY_POLY", "VISIBLE_ESTIMATE_CONFLICT"}

SEVERITY = {
    "MISSING_BOLL_ID": "ERROR", "TIP_VIS_MISMATCH": "ERROR", "BASE_VIS_MISMATCH": "ERROR",
    "OCCLUSION_STRUCTURE": "ERROR", "VISIBLE_ESTIMATE_CONFLICT": "ERROR",
    "TIP_VERTICES": "ERROR", "BASE_VERTICES": "ERROR", "BAD_OCCLUDED_BY": "ERROR",
    "NO_MASK": "ERROR", "DUP_ESTIMATE": "ERROR", "DUP_VISIBLE": "ERROR",
    "ORPHAN_TIP": "ERROR", "ORPHAN_BASE": "ERROR", "DUP_TIP": "ERROR", "DUP_BASE": "ERROR",
    "OUT_OF_BOUNDS": "WARN", "DEGENERATE_POLY": "ERROR", "TINY_POLY": "WARN",
}


def print_category(cat, items):
    print(f"### [{SEVERITY.get(cat,'')}] {cat} — {len(items)}")
    if cat in GROUP_BY_BOLL:
        for bid, n, rng in group_by_boll(items):
            print(f"  boll_id={bid}: {n} frame(s) — {rng}")
    elif cat in ("TIP_VERTICES", "BASE_VERTICES"):
        by = defaultdict(list)
        for f, b, n in items:
            by[b].append((f, n))
        for bid in sorted(by.keys(), key=lambda x: int(x) if str(x).isdigit() else 1_000_000):
            frames = [f for f, _ in by[bid]]
            npts = sorted({n for _, n in by[bid]})
            print(f"  boll_id={bid}: {len(frames)} frame(s) with {npts} pts — {compress(frames)}")
    elif cat == "MISSING_BOLL_ID":
        for frame, _, ctx in items:
            print(f"  Frame {frame}: {ctx}")
    elif cat in ("DUP_TIP", "DUP_BASE"):
        for frame, bid, np_, npt in items:
            print(f"  Frame {frame} boll_id={bid}: poly={np_} point={npt}")
    elif cat == "OCCLUSION_STRUCTURE":
        # group by violation kind then by boll
        by_kind = defaultdict(list)
        for frame, bid, detail in items:
            by_kind[detail].append((frame, bid))
        for kind, entries in by_kind.items():
            print(f"  — {kind}: {len(entries)}")
            for bid, n, rng in group_by_boll([(f, b) for f, b in entries]):
                print(f"      boll_id={bid}: {n} frame(s) — {rng}")
    elif cat in ("TIP_VIS_MISMATCH", "BASE_VIS_MISMATCH", "OUT_OF_BOUNDS", "DEGENERATE_POLY"):
        for tup in items[:40]:
            frame, bid = tup[0], tup[1]
            detail = tup[2] if len(tup) > 2 else ""
            print(f"  Frame {frame} boll_id={bid}: {detail}")
        if len(items) > 40:
            print(f"  ... and {len(items)-40} more")
    else:
        for tup in items[:40]:
            print(f"  {tup}")
        if len(items) > 40:
            print(f"  ... and {len(items)-40} more")
    print()


def report(issues, image_count, by_frame=False):
    total = sum(len(v) for v in issues.values())
    n_err = sum(len(v) for k, v in issues.items() if SEVERITY.get(k) == "ERROR")
    n_warn = sum(len(v) for k, v in issues.items() if SEVERITY.get(k) == "WARN")

    print(f"Images checked: {image_count}")
    print(f"Total issues:   {total}  ({n_err} errors, {n_warn} warnings)\n")

    if total == 0:
        print("ALL CLEAN — no issues found.")
        return

    for cat in CATEGORY_ORDER:
        if issues.get(cat):
            print_category(cat, issues[cat])

    if by_frame:
        per_frame = defaultdict(set)
        for cat, items in issues.items():
            for tup in items:
                per_frame[tup[0]].add(cat)
        print("### PER-FRAME SUMMARY")
        for frame in sorted(per_frame.keys()):
            cats = sorted(per_frame[frame])
            print(f"  Frame {frame}: {len(cats)} type(s) — {', '.join(cats)}")
        print()


def main():
    p = argparse.ArgumentParser(description="Validate cotton boll CVAT annotations")
    p.add_argument("--xml", required=True)
    p.add_argument("--max_frame", type=int, default=None)
    p.add_argument("--by_frame", action="store_true")
    p.add_argument("--no_geometry", action="store_true",
                   help="Skip geometry checks (bounds, area)")
    args = p.parse_args()

    issues, n = validate(args.xml, max_frame=args.max_frame, do_geometry=not args.no_geometry)
    report(issues, n, by_frame=args.by_frame)


if __name__ == "__main__":
    main()
