#!/usr/bin/env python
"""Turn tracked detections + raw DA3 depth into millimetre boll sizes.

Uses the constants from fit_gamma_from_annotations.py:

    size_mm = tb_px * z_raw**(1/gamma) * exp(C)

tb_px is the predicted tip-to-base pixel distance, z_raw the median raw DA3
depth inside the eroded instance mask. gamma and C were fitted against
tape-measured, tag-matched bolls, so C carries the depth scale and the focal
length together -- it is only valid for the same camera and capture setup.

    python scripts/depth/apply_boll_sizes.py \
        --run inference_results/LeftDown_v3_soft \
        --json_dir tracked_json_trimmed --depth_dir depth_trimmed/depth
"""
import argparse, json, os, glob, math, csv, collections
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder")
    ap.add_argument("--json_dir", default="tracked_json")
    ap.add_argument("--depth_dir", default="depth/depth")
    ap.add_argument("--calib", default="findings/cotton_tag_calibration/calibration.json")
    ap.add_argument("--out_dir", default="sizes", help="relative to --run")
    ap.add_argument("--img_width", type=int, default=2988)
    ap.add_argument("--img_height", type=int, default=5312)
    ap.add_argument("--crop_top_px", type=int, default=0,
                    help="set if the depth maps were made from top-cropped images")
    ap.add_argument("--min_det_conf", type=float, default=0.9)
    ap.add_argument("--min_kpt_conf", type=float, default=0.5)
    ap.add_argument("--min_visfrac", type=float, default=0.80)
    a = ap.parse_args()

    cal = json.load(open(a.calib))
    gamma, scale = cal["gamma"], cal["scale"]
    W, H = a.img_width, a.img_height
    ch = H - a.crop_top_px

    per_frame = []
    for jf in sorted(glob.glob(os.path.join(a.run, a.json_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jf))[0]
        npy = os.path.join(a.run, a.depth_dir, stem + ".npy")
        if not os.path.exists(npy):
            continue
        D = np.load(npy).astype(np.float32)
        dh, dw = D.shape
        sx, sy = dw / W, dh / ch
        for d in json.load(open(jf)):
            if d.get("track_id") is None:
                continue
            q = np.array(d["mask_poly"], np.float32).copy()
            q[:, 0] *= sx
            q[:, 1] = (q[:, 1] - a.crop_top_px) * sy
            m = np.zeros((dh, dw), np.uint8)
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
            me = cv2.erode(m, np.ones((3, 3), np.uint8), 1)
            use = me if me.sum() >= 6 else m
            if use.sum() < 4:
                continue
            z = float(np.median(D[use > 0]))
            (x1, y1, c1), (x2, y2, c2) = d["kpts"][0], d["kpts"][1]
            tb = math.hypot(x1 - x2, y1 - y2)
            per_frame.append(dict(
                track_id=d["track_id"], frame=stem, tb_px=round(tb, 1), z_raw=round(z, 4),
                size_mm=round(tb * z ** (1 / gamma) * scale, 2),
                det_conf=round(d["det_conf"], 3), kpt_conf=round(min(c1, c2), 3),
                visfrac=round(d.get("visfrac", 1.0), 3),
                occlusion_pct=round(d.get("occlusion_pct", 0.0), 3)))

    if not per_frame:
        raise SystemExit("no detections matched a depth map -- check --depth_dir / --json_dir")

    out = os.path.join(a.run, a.out_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        w.writeheader(); w.writerows(per_frame)

    by = collections.defaultdict(list)
    for r in per_frame:
        by[r["track_id"]].append(r)

    rows = []
    for t, v in sorted(by.items()):
        good = [x for x in v if x["det_conf"] >= a.min_det_conf
                and x["kpt_conf"] >= a.min_kpt_conf and x["visfrac"] >= a.min_visfrac]
        if len(good) < 5:
            good = [x for x in v if x["det_conf"] >= a.min_det_conf
                    and x["kpt_conf"] >= a.min_kpt_conf] or v
        sz = np.array([x["size_mm"] for x in good])
        rows.append(dict(track_id=t, frames_total=len(v), frames_used=len(good),
                         size_mm=round(float(np.median(sz)), 1),
                         sd_pct=round(float(100 * sz.std(ddof=1) / sz.mean()), 1) if len(sz) > 1 else 0.0,
                         tb_px=round(float(np.median([x["tb_px"] for x in good])), 1),
                         z_raw=round(float(np.median([x["z_raw"] for x in good])), 3),
                         visfrac=round(float(np.median([x["visfrac"] for x in good])), 3),
                         quality="good" if len(good) >= 20 else ("fair" if len(good) >= 10 else "weak")))
    with open(os.path.join(out, "per_boll.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    keep = [r for r in rows if r["quality"] != "weak"]
    sz = np.array([r["size_mm"] for r in keep])
    print(f"{len(per_frame)} detections -> {len(rows)} tracks "
          f"({len(keep)} good/fair)   gamma={gamma}")
    print(f"size over good+fair bolls: median {np.median(sz):.1f} mm  mean {sz.mean():.1f}  "
          f"sd {sz.std(ddof=1):.1f} ({100*sz.std(ddof=1)/sz.mean():.1f}%)  "
          f"range {sz.min():.1f}-{sz.max():.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
