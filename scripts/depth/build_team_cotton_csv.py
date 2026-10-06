"""build_team_cotton_csv.py — plain-language cotton files for findings/team_share/.

Rewrites the two internal cotton artefacts (findings/trimmed_boll/per_frame.csv
and the Plant & Soil measurement sheet) into shareable form:

    cotton_boll_measurements.csv   one row per tracked boll  (18)
    cotton_boll_frames.csv         one row per boll per frame (542)
    cotton_ground_truth.csv        one row per measured boll (38)
    cotton_results.png             copy of the trimmed-run figure

Nothing is recomputed from video here; the depth run is
scripts/depth/trimmed_boll_analysis.py.
"""
import csv
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

PER_FRAME = Path("findings/trimmed_boll/per_frame.csv")
GT_XLSX = Path("ground_truth/greenhouse_day_1/"
               "Cotton boll dimensions_Greenhouse 27Mar2026 (1).xlsx")
OUT = Path("findings/team_share")
GAMMA, F_PX = 0.663, 2905.6
A_BALL = 0.008030184315703093          # findings/trimmed_boll/calibration.json


def ground_truth():
    s = zipfile.ZipFile(GT_XLSX).read("xl/worksheets/sheet1.xml").decode("utf8")
    out = []
    for rn, body in re.findall(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', s, re.S):
        if rn == "1":
            continue
        c = {}
        for m in re.finditer(r'<c r="([A-Z]+)\d+"[^>]*>(?:<v>(.*?)</v>)?', body):
            if m.group(2) is not None:
                c[m.group(1)] = float(m.group(2))
        if "C" in c:
            out.append({"pot": c.get("A"), "boll": c.get("B"), "height": c["C"],
                        "length": c.get("D"), "width": c.get("E")})
    return out


def main():
    rows = list(csv.DictReader(open(PER_FRAME)))
    for r in rows:
        r["track"] = int(r["track"])
        r["px"] = float(r["px"])
        r["z"] = float(r["z"])
        r["visfrac"] = float(r["visfrac"])
        r["det_conf"] = float(r["det_conf"])

    gt = ground_truth()
    H = np.array([g["height"] for g in gt])

    by = {}
    for r in rows:
        by.setdefault(r["track"], []).append(r)

    # blind ball-calibration length, per frame then median over the track
    blind = {t: float(np.median([x["px"] * (x["z"] / A_BALL) ** (1 / GAMMA) / F_PX
                                 for x in v])) for t, v in by.items()}
    k = H.mean() / np.mean(list(blind.values()))   # one scale factor, all 18 tracks

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cotton_boll_measurements.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["boll_track", "frames_measured", "used_in_headline",
                    "median_visible_fraction", "median_detection_confidence",
                    "occluded_frame_fraction", "apparent_length_px",
                    "depth_model_output", "length_ball_calibration_mm",
                    "length_scaled_to_ground_truth_mm"])
        for t in sorted(by):
            v = by[t]
            w.writerow([
                t, len(v), int(len(v) >= 10),
                round(float(np.median([x["visfrac"] for x in v])), 3),
                round(float(np.median([x["det_conf"] for x in v])), 3),
                round(sum(x["occ"] != "none" for x in v) / len(v), 3),
                round(float(np.median([x["px"] for x in v])), 1),
                round(float(np.median([x["z"] for x in v])), 4),
                round(blind[t], 2),
                round(blind[t] * k, 2),
            ])

    with open(OUT / "cotton_boll_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["boll_track", "frame", "apparent_length_px",
                    "depth_model_output", "detection_confidence",
                    "visible_fraction", "occluded_by"])
        for r in rows:
            w.writerow([r["track"], r["frame"], round(r["px"], 1),
                        round(r["z"], 4), round(r["det_conf"], 3),
                        round(r["visfrac"], 3), r["occ"]])

    with open(OUT / "cotton_ground_truth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pot", "boll", "height_tip_to_base_mm", "length_mm", "width_mm"])
        for g in gt:
            w.writerow([int(g["pot"]) if g["pot"] else "",
                        int(g["boll"]) if g["boll"] else "",
                        g["height"], g["length"], g["width"]])

    fig = Path("findings/trimmed_boll/trimmed_boll.png")
    if fig.exists():
        shutil.copyfile(fig, OUT / "cotton_results.png")
        print(f"copied {fig} -> {OUT}/cotton_results.png")

    print(f"wrote {OUT}/cotton_boll_measurements.csv  ({len(by)} bolls)")
    print(f"wrote {OUT}/cotton_boll_frames.csv        ({len(rows)} rows)")
    print(f"wrote {OUT}/cotton_ground_truth.csv       ({len(gt)} bolls)")
    print(f"  scale factor applied to the blind lengths: {k:.4f}")


if __name__ == "__main__":
    main()
