#!/usr/bin/env python
"""Fit the DA3 power law from TRACKED DETECTIONS plus tag ground truth.

fit_gamma_from_annotations.py needs CVAT tip_polygob/base_rectangle annotations,
which only left_down has. This fits the same model

    log S_true - log tb_px = (1/gamma) * log z_raw + C
    size_mm = tb_px * z_raw**(1/gamma) * exp(C)

from the MODEL's predicted tip/base instead, joined to ground truth through a
viewer boll-report export whose real_id column has been filled in by hand. That
makes it usable on any angle that has tagged bolls, at the cost of inheriting
whatever bias the pose head has -- gamma and C then absorb part of it, so a fit
from here is not interchangeable with one from annotations.

All signal is BETWEEN bolls (a boll's true distance barely changes as the camera
pans), so every instance is aggregated to one point per boll before fitting.

    python scripts/depth/fit_gamma_from_tracks.py \
        --run inference_results/RightDown_v5 --json_dir tracked_json_trimmed \
        --depth_dir depth_trimmed_crop15/depth \
        --map inference_results/RightDown_v5/right_down_size_da3.csv
"""
import argparse, csv, glob, json, math, os, collections
import numpy as np, cv2


def load_gt(p):
    return {int(r["boll"]): float(r["height_tip_to_base_mm"]) for r in csv.DictReader(open(p))}


def load_map(p):
    """track_id -> real_id, from a viewer export with real_id filled in."""
    m = {}
    for r in csv.DictReader(open(p)):
        rid = (r.get("real_id") or "").strip()
        if not rid:
            continue
        for t in r["track_ids"].split("+"):
            m[int(t.strip())] = int(rid)
    return m


def sample(run, json_dir, depth_dir, tmap, crop_top_px, crop_bottom_px,
           min_det, min_kpt, min_vis, min_tb):
    per = collections.defaultdict(list)
    skip = collections.Counter()
    for jf in sorted(glob.glob(os.path.join(run, json_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(jf))[0]
        npy = os.path.join(run, depth_dir, stem + ".npy")
        if not os.path.exists(npy):
            continue
        D = None
        for d in json.load(open(jf)):
            t = d.get("track_id")
            if t is None or t not in tmap:
                continue
            (x1, y1, c1), (x2, y2, c2) = d["kpts"][0], d["kpts"][1]
            tb = math.hypot(x1 - x2, y1 - y2)
            if tb < min_tb:
                skip["degenerate_pose"] += 1; continue
            if d["det_conf"] < min_det or min(c1, c2) < min_kpt \
               or d.get("visfrac", 1.0) < min_vis:
                skip["low_conf"] += 1; continue
            if D is None:
                D = np.load(npy).astype(np.float32)
            dh, dw = D.shape
            ch = ORIG_H - crop_top_px - crop_bottom_px
            q = np.array(d["mask_poly"], np.float32).copy()
            q[:, 0] *= dw / ORIG_W
            q[:, 1] = (q[:, 1] - crop_top_px) * (dh / ch)
            if q[:, 1].max() >= dh or q[:, 1].min() < 0:
                skip["outside_crop"] += 1; continue
            m = np.zeros((dh, dw), np.uint8)
            cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
            me = cv2.erode(m, np.ones((3, 3), np.uint8), 1)
            use = me if me.sum() >= 6 else m
            if use.sum() < 4:
                skip["tiny_mask"] += 1; continue
            per[tmap[t]].append((tb, float(np.median(D[use > 0]))))
    return per, skip


def fit(B):
    x = np.log([b["z"] for b in B])
    y = np.log([b["S"] for b in B]) - np.log([b["tb"] for b in B])
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, C), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, C])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return 1.0 / slope, C, r2


def loo(B, gamma):
    errs = []
    for i in range(len(B)):
        tr = [b for j, b in enumerate(B) if j != i]
        te = B[i]
        C = float(np.mean([np.log(b["S"]) - np.log(b["tb"]) - (1 / gamma) * np.log(b["z"])
                           for b in tr]))
        errs.append(abs(te["tb"] * te["z"] ** (1 / gamma) * np.exp(C) - te["S"]) / te["S"])
    return 100 * float(np.mean(errs))


ORIG_W = ORIG_H = None

def main():
    global ORIG_W, ORIG_H
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json_dir", default="tracked_json_trimmed")
    ap.add_argument("--depth_dir", default="depth_trimmed_crop15/depth")
    ap.add_argument("--map", required=True, help="viewer export CSV with real_id filled in")
    ap.add_argument("--gt", default="findings/cotton_pending_calibration/cotton_ground_truth.csv")
    ap.add_argument("--out", default=None, help="write calibration JSON here")
    ap.add_argument("--img_width", type=int, default=2988)
    ap.add_argument("--img_height", type=int, default=5312)
    ap.add_argument("--crop_top_px", type=int, default=0, help="0 = auto from crop_info.json")
    ap.add_argument("--min_det_conf", type=float, default=0.9)
    ap.add_argument("--min_kpt_conf", type=float, default=0.5)
    ap.add_argument("--min_visfrac", type=float, default=0.80)
    ap.add_argument("--min_tb_px", type=float, default=5.0)
    ap.add_argument("--min_instances", type=int, default=5,
                    help="drop a boll measured in fewer frames than this")
    a = ap.parse_args()
    ORIG_W, ORIG_H = a.img_width, a.img_height

    gt = load_gt(a.gt)
    tmap = load_map(a.map)
    ct, cb = a.crop_top_px, 0
    ci = os.path.join(a.run, a.depth_dir, "crop_info.json")
    if os.path.exists(ci):
        info = json.load(open(ci))
        if ct == 0:
            ct = int(info["crop_top_px"])
        cb = int(info.get("crop_bottom_px", 0))
        print(f"crop_info.json: top {ct}px, bottom {cb}px")

    per, skip = sample(a.run, a.json_dir, a.depth_dir, tmap, ct, cb,
                       a.min_det_conf, a.min_kpt_conf, a.min_visfrac, a.min_tb_px)
    B = []
    for rid, v in sorted(per.items()):
        if rid not in gt or len(v) < a.min_instances:
            continue
        B.append(dict(boll_id=rid, n=len(v), S=gt[rid],
                      tb=float(np.median([x[0] for x in v])),
                      z=float(np.median([x[1] for x in v]))))
    if len(B) < 3:
        raise SystemExit(f"only {len(B)} usable bolls -- need at least 3 to fit")
    zs = [b["z"] for b in B]
    print(f"instances {sum(b['n'] for b in B)}   bolls {len(B)}   "
          f"depth spread {min(zs):.3f}-{max(zs):.3f}")
    if skip:
        print("skipped instances:", dict(skip))

    g_free, C_free, r2 = fit(B)
    print(f"free fit: gamma = {g_free:.3f}   C = {C_free:.3f}   r2 = {r2:.3f}")

    grid = np.arange(0.30, 1.51, 0.01)
    scores = [(loo(B, g), g) for g in grid]
    best_err, best_g = min(scores)
    C = float(np.mean([np.log(b["S"]) - np.log(b["tb"]) - (1 / best_g) * np.log(b["z"])
                       for b in B]))
    print(f"leave-one-boll-out optimum: gamma = {best_g:.2f}  (mean |err| {best_err:.1f}%)")
    for ref in (0.70, 0.663, 1.0):
        print(f"   for reference, gamma={ref}: LOO mean |err| {loo(B, ref):.1f}%")
    print(f"\nsize_mm = tb_px * z_raw**(1/{best_g:.2f}) * {math.exp(C):.6f}")
    print(f"{'boll':>5}{'n':>5}{'GT':>8}{'tb_px':>9}{'z_raw':>8}{'pred':>8}{'err%':>8}")
    for b in B:
        p = b["tb"] * b["z"] ** (1 / best_g) * math.exp(C)
        print(f"{b['boll_id']:>5}{b['n']:>5}{b['S']:>8.2f}{b['tb']:>9.1f}{b['z']:>8.3f}"
              f"{p:>8.1f}{100*(p-b['S'])/b['S']:>+7.1f}%")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(gamma=round(best_g, 3), C=C, scale=math.exp(C),
                       n_bolls=len(B), n_instances=sum(b["n"] for b in B),
                       loo_mean_abs_err_pct=round(best_err, 2),
                       note="size_mm = tb_px * z_raw**(1/gamma) * exp(C); z_raw is raw DA3, no focal",
                       fitted_on=a.map, source="predicted tip/base (fit_gamma_from_tracks.py)",
                       depth_variant=f"top {ct}px + bottom {cb}px cropped"),
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
