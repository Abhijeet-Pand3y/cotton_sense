#!/usr/bin/env python
"""Worksheet for mapping viewer track ids to physical boll tags (real_id).

The Inference Viewer's boll report exports track ids, not tags -- matching a
track to the tag on the plant is a human, per-boll lookup. This writes one row
per track with the frame to jump to and where in it to look, so the lookup is
"open best_frame, look at (cx, cy), read the tag, type it in real_id".

    python scripts/depth/make_id_worksheet.py --run inference_results/rdbd_depth_inference

Fill in the real_id column, then join it to the viewer's exported
boll_size_report.csv on track_id (a merged row's track_ids are "a+b").
"""
import argparse, json, os, glob, csv, collections
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json_dir", default="tracked_json")
    ap.add_argument("--sizes", default="sizes/per_boll.csv", help="relative to --run")
    ap.add_argument("--out", default="id_worksheet.csv", help="relative to --run")
    ap.add_argument("--include_weak", action="store_true",
                    help="also list tracks the sizing step marked weak")
    a = ap.parse_args()

    tracks = collections.defaultdict(list)
    for jf in sorted(glob.glob(os.path.join(a.run, a.json_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jf))[0]
        for d in json.load(open(jf)):
            if d.get("track_id") is None:
                continue
            x, y, w, h = d["bbox"]
            tracks[d["track_id"]].append(dict(
                stem=stem, cx=x + w / 2, cy=y + h / 2, area=w * h,
                kpt=min(d["kpts"][0][2], d["kpts"][1][2]),
                visfrac=d.get("visfrac", 1.0)))

    sizes = {}
    sp = os.path.join(a.run, a.sizes)
    if os.path.exists(sp):
        sizes = {int(r["track_id"]): r for r in csv.DictReader(open(sp))}

    rows = []
    for t, v in sorted(tracks.items()):
        s = sizes.get(t)
        if s and s["quality"] == "weak" and not a.include_weak:
            continue
        # the frame to look at: biggest, least occluded, most confident pose
        best = max(v, key=lambda r: r["area"] * r["visfrac"] * (0.5 + r["kpt"]))
        rows.append(dict(
            real_id="", track_id=t, frames=len(v),
            first_frame=v[0]["stem"], last_frame=v[-1]["stem"],
            best_frame=best["stem"], cx=round(best["cx"]), cy=round(best["cy"]),
            size_mm=(s or {}).get("size_mm", ""), sd_pct=(s or {}).get("sd_pct", ""),
            z_mm=(s or {}).get("z_mm", ""), quality=(s or {}).get("quality", ""),
            notes=""))

    out = os.path.join(a.run, a.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} tracks -> {out}   (fill in real_id; blank = not a tagged boll)")


if __name__ == "__main__":
    main()
