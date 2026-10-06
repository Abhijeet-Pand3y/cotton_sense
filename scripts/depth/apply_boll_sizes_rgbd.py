#!/usr/bin/env python
"""Tip-to-base boll size from tracked detections + RealSense RGB-D depth.

Pinhole, nothing else:

    size_mm = tb_px * Z_mm / f_px

tb_px is the predicted tip-to-base pixel distance, Z the median valid depth
inside the eroded instance mask, f = sqrt(fx*fy) from metadata.json. The depth
is already metric (uint16 * depth_scale), so unlike the monocular DA3 path
(apply_boll_sizes.py) there is no gamma and no fitted scale constant -- and the
calibration.json constants must NOT be applied here.

Output columns mirror apply_boll_sizes.py so the run drops straight beside a
GoPro run, with z_mm (metres*1000, real) in place of its z_raw (arbitrary units).

    python scripts/depth/apply_boll_sizes_rgbd.py \
        --run inference_results/rdbd_depth_inference \
        --json_dir tracked_json --depth_dir depth --out_dir sizes
"""
import argparse, json, os, glob, math, csv, collections
import numpy as np, cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder")
    ap.add_argument("--json_dir", default="tracked_json")
    ap.add_argument("--depth_dir", default="depth")
    ap.add_argument("--meta", default="metadata.json", help="relative to --run")
    ap.add_argument("--out_dir", default="sizes", help="relative to --run")
    ap.add_argument("--min_valid_frac", type=float, default=0.5,
                    help="min fraction of mask pixels carrying a depth reading")
    ap.add_argument("--z_min_m", type=float, default=0.1)
    ap.add_argument("--z_max_m", type=float, default=10.0)
    ap.add_argument("--min_tb_px", type=float, default=5.0,
                    help="tip/base closer than this is a collapsed pose, not a boll")
    ap.add_argument("--min_det_conf", type=float, default=0.9)
    ap.add_argument("--min_kpt_conf", type=float, default=0.0,
                    help="0 by default: the pose head's confidences are not calibrated on "
                         "this camera (median ~0.09 vs ~0.99 on GoPro) though its positions "
                         "still track; raise it for GoPro-style strictness")
    ap.add_argument("--min_visfrac", type=float, default=0.80)
    a = ap.parse_args()

    md = json.load(open(os.path.join(a.run, a.meta)))
    K = md["intrinsics"]
    f = math.sqrt(K["fx"] * K["fy"])
    scale_mm = md["depth_scale"] * 1000.0          # raw depth units -> mm
    W, H = K["width"], K["height"]

    skipped = collections.Counter()
    per_frame = []
    for jf in sorted(glob.glob(os.path.join(a.run, a.json_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jf))[0]
        npy = os.path.join(a.run, a.depth_dir, stem + ".npy")
        if not os.path.exists(npy):
            skipped["no_depth_map"] += 1
            continue
        D = np.load(npy).astype(np.float32) * scale_mm
        dh, dw = D.shape
        sx, sy = dw / W, dh / H
        for d in json.load(open(jf)):
            if d.get("track_id") is None:
                skipped["untracked"] += 1
                continue
            q = np.array(d["mask_poly"], np.float32).copy()
            q[:, 0] *= sx
            q[:, 1] *= sy
            m = np.zeros((dh, dw), np.uint8)
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
            me = cv2.erode(m, np.ones((3, 3), np.uint8), 1)
            use = me if me.sum() >= 6 else m
            n = int(use.sum())
            if n < 4:
                skipped["tiny_mask"] += 1
                continue
            z_ok = D[use > 0]
            z_ok = z_ok[z_ok > 0]
            vfrac = len(z_ok) / n
            if vfrac < a.min_valid_frac:
                skipped["depth_holes"] += 1
                continue
            z = float(np.median(z_ok))
            if not (a.z_min_m * 1000 <= z <= a.z_max_m * 1000):
                skipped["z_out_of_range"] += 1
                continue

            (x1, y1, c1), (x2, y2, c2) = d["kpts"][0], d["kpts"][1]
            tb = math.hypot(x1 - x2, y1 - y2)
            per_frame.append(dict(
                track_id=d["track_id"], frame=stem, tb_px=round(tb, 1), z_mm=round(z, 1),
                size_mm=round(tb * z / f, 2), degenerate=int(tb < a.min_tb_px),
                det_conf=round(d["det_conf"], 3), kpt_conf=round(min(c1, c2), 3),
                visfrac=round(d.get("visfrac", 1.0), 3),
                occlusion_pct=round(d.get("occlusion_pct", 0.0), 3)))

    if not per_frame:
        raise SystemExit("no tracked detections matched a depth map -- check --depth_dir / --json_dir")

    out = os.path.join(a.run, a.out_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "per_frame.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_frame[0].keys()))
        w.writeheader(); w.writerows(per_frame)

    by = collections.defaultdict(list)
    for r in per_frame:
        by[r["track_id"]].append(r)

    rows = []
    for t, v in sorted(by.items()):
        ok = [x for x in v if not x["degenerate"]]
        good = [x for x in ok if x["det_conf"] >= a.min_det_conf
                and x["kpt_conf"] >= a.min_kpt_conf and x["visfrac"] >= a.min_visfrac]
        if len(good) < 5:
            good = [x for x in ok if x["det_conf"] >= a.min_det_conf
                    and x["kpt_conf"] >= a.min_kpt_conf] or ok
        if not good:
            skipped["all_frames_degenerate"] += 1
            continue
        sz = np.array([x["size_mm"] for x in good])
        rows.append(dict(track_id=t, frames_total=len(v), frames_used=len(good),
                         degenerate_frames=len(v) - len(ok),
                         size_mm=round(float(np.median(sz)), 1),
                         sd_pct=round(float(100 * sz.std(ddof=1) / sz.mean()), 1) if len(sz) > 1 else 0.0,
                         tb_px=round(float(np.median([x["tb_px"] for x in good])), 1),
                         z_mm=round(float(np.median([x["z_mm"] for x in good])), 1),
                         visfrac=round(float(np.median([x["visfrac"] for x in good])), 3),
                         quality="good" if len(good) >= 20 else ("fair" if len(good) >= 10 else "weak")))
    with open(os.path.join(out, "per_boll.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    keep = [r for r in rows if r["quality"] != "weak"]
    sz = np.array([r["size_mm"] for r in keep])
    ndeg = sum(r["degenerate"] for r in per_frame)
    print(f"{len(per_frame)} detections -> {len(rows)} tracks ({len(keep)} good/fair)   "
          f"f={f:.1f}px  depth_scale={md['depth_scale']:.6f}  (metric: no gamma, no scale const)")
    print(f"median kpt_conf {np.median([r['kpt_conf'] for r in per_frame]):.3f}   "
          f"collapsed poses (tb<{a.min_tb_px:g}px) {ndeg} ({100*ndeg/len(per_frame):.1f}%), excluded")
    if skipped:
        print("skipped:", dict(skipped))
    print(f"size over good+fair bolls: median {np.median(sz):.1f} mm  mean {sz.mean():.1f}  "
          f"sd {sz.std(ddof=1):.1f} ({100*sz.std(ddof=1)/sz.mean():.1f}%)  "
          f"range {sz.min():.1f}-{sz.max():.1f}")
    wsd = np.array([r["sd_pct"] for r in keep])
    print(f"within-track sd of size_mm: median {np.median(wsd):.1f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
