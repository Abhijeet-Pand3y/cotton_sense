#!/usr/bin/env python
"""Fit the DA3 depth power law against tape-measured, tag-matched cotton bolls.

Uses the CVAT annotations (where boll_id is the physical tag) plus
cotton_ground_truth.csv, so every boll's true tip-to-base height is known:

    S_true = tb_px * Z / f      with   Z = (z_raw / a)^(1/gamma)
 => log S_true - log tb_px = (1/gamma) * log z_raw + C

One line through the per-boll points gives gamma from the slope, and C absorbs
both the depth scale a and the focal length f. Unlike the ping-pong-ball fits,
this has enough points to leave a residual.

Depth maps must already exist (raw DA3, NO --focal_px):
    python scripts/depth/run_da3_depth.py --images_dir data/images/left_down \
        --out depth_out/left_down_annot --no_viz
"""
import argparse, xml.etree.ElementTree as ET, csv, math, os, json, collections
import numpy as np, cv2


def pts(el):
    return [tuple(float(v) for v in p.split(",")) for p in el.get("points").split(";")]


def centroid(P):
    return (sum(x for x, _ in P) / len(P), sum(y for _, y in P) / len(P))


def load_gt(path):
    gt = {}
    for r in csv.DictReader(open(path)):
        gt[int(r["boll"])] = float(r["height_tip_to_base_mm"])
    return gt


def parse_xml(xml_path, gt):
    """One row per annotated boll instance that has a tip, a base and a GT size."""
    root = ET.parse(xml_path).getroot()
    rows = []
    for im in root.findall("image"):
        stem = im.get("name").rsplit(".", 1)[0]
        W, H = int(im.get("width")), int(im.get("height"))
        bolls, tips, bases = {}, {}, {}
        for ch in im:
            a = {x.get("name"): x.text for x in ch.findall("attribute")}
            bid = a.get("boll_id")
            bid = int(bid) if bid not in (None, "") else None
            if bid is None:
                continue
            lab = ch.get("label")
            if lab == "cotton_boll":
                bolls[bid] = (pts(ch), a)
            elif lab == "tip_polygob" or (ch.tag == "points" and lab == "tip"):
                tips.setdefault(bid, centroid(pts(ch)))
            elif lab == "base_rectangle" or (ch.tag == "points" and lab == "base"):
                bases.setdefault(bid, centroid(pts(ch)))
        for bid, (poly, a) in bolls.items():
            if bid not in tips or bid not in bases or bid not in gt:
                continue
            t, b = tips[bid], bases[bid]
            rows.append(dict(stem=stem, boll_id=bid, W=W, H=H, poly=poly, tip=t, base=b,
                             tb_px=math.hypot(t[0] - b[0], t[1] - b[1]),
                             tip_vis=a.get("tip_visibility"), base_vis=a.get("base_visibility"),
                             occluded_by=a.get("occluded_by"), annot_type=a.get("annotation_type"),
                             gt_height_mm=gt[bid]))
    return rows


def sample_depth(rows, depth_dir, crop_top_px=0, crop_bottom_px=0):
    """Median raw depth inside the eroded boll mask.

    crop_top_px shifts annotation coords into the depth map's frame when the
    maps were produced from top-cropped images (--crop_top_frac), which is how
    the greenhouse roof is kept out of DA3's view."""
    out = []
    for r in rows:
        p = os.path.join(depth_dir, r["stem"] + ".npy")
        if not os.path.exists(p):
            continue
        D = np.load(p).astype(np.float32)
        dh, dw = D.shape
        ch = r["H"] - crop_top_px - crop_bottom_px
        q = np.array(r["poly"], np.float32).copy()
        q[:, 0] *= dw / r["W"]; q[:, 1] = (q[:, 1] - crop_top_px) * (dh / ch)
        m = np.zeros((dh, dw), np.uint8)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
        me = cv2.erode(m, np.ones((3, 3), np.uint8), 1)
        use = me if me.sum() >= 6 else m
        if use.sum() < 4:
            continue
        r = dict(r); r.pop("poly")
        r["z_raw"] = float(np.median(D[use > 0]))
        out.append(r)
    return out


def per_boll(rows, min_inst=3):
    """Aggregate to one point per tagged boll -- instances of the same boll are
    not independent, so fitting on them would overstate the precision badly."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["boll_id"]].append(r)
    B = []
    for bid, rs in sorted(by.items()):
        if len(rs) < min_inst:
            continue
        B.append(dict(boll_id=bid, n=len(rs),
                      z=float(np.median([r["z_raw"] for r in rs])),
                      tb=float(np.median([r["tb_px"] for r in rs])),
                      S=rs[0]["gt_height_mm"]))
    return B


def fit(B):
    x = np.log([b["z"] for b in B])
    y = np.log([b["S"] for b in B]) - np.log([b["tb"] for b in B])
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, C), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, C])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return 1.0 / slope, C, r2


def loo(B, gamma):
    """Leave-one-boll-out: fit C on the rest, predict the held-out boll's size."""
    errs = []
    for i in range(len(B)):
        tr = [b for j, b in enumerate(B) if j != i]
        te = B[i]
        C = float(np.mean([np.log(b["S"]) - np.log(b["tb"]) - (1 / gamma) * np.log(b["z"])
                           for b in tr]))
        pred = te["tb"] * te["z"] ** (1 / gamma) * np.exp(C)
        errs.append(abs(pred - te["S"]) / te["S"])
    return 100 * float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="data/annotations/left_down/left_down.xml")
    ap.add_argument("--depth_dir", default="depth_out/left_down_annot/depth")
    ap.add_argument("--gt", default="findings/cotton_pending_calibration/cotton_ground_truth.csv")
    ap.add_argument("--out", default="findings/cotton_tag_calibration")
    ap.add_argument("--middle", type=int, default=0,
                    help="use only the middle N annotated frames. The first and last "
                         "frames of a pass look down the row into open background, "
                         "which shifts what DA3 anchors its depth range on.")
    ap.add_argument("--crop_top_px", type=int, default=0,
                    help="0 = auto-detect from crop_info.json in --depth_dir")
    ap.add_argument("--all_instances", action="store_true",
                    help="keep partly-occluded / estimated annotations too")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    rows = parse_xml(a.xml, gt)
    crop_top_px = a.crop_top_px
    crop_bottom_px = 0
    ci = os.path.join(a.depth_dir, "crop_info.json")
    if os.path.exists(ci):
        info = json.load(open(ci))
        if crop_top_px == 0:
            crop_top_px = int(info["crop_top_px"])
        crop_bottom_px = int(info.get("crop_bottom_px", 0))
        print(f"crop_info.json found: shifting annotations by {crop_top_px}px, "
              f"bottom crop {crop_bottom_px}px")
    rows = sample_depth(rows, a.depth_dir, crop_top_px, crop_bottom_px)
    # A boll below the bottom crop is not in the depth map at all.
    if crop_bottom_px:
        keep = [r for r in rows if r["tip"][1] < r["H"] - crop_bottom_px
                and r["base"][1] < r["H"] - crop_bottom_px]
        if len(keep) < len(rows):
            print(f"dropped {len(rows)-len(keep)} instances below the bottom crop")
        rows = keep
    if not a.all_instances:
        rows = [r for r in rows if r["tip_vis"] == "visible" and r["base_vis"] == "visible"
                and r["annot_type"] == "whole_visible" and r["occluded_by"] == "none"]
    if a.middle > 0:
        stems = sorted({r["stem"] for r in rows})
        lo = max(0, (len(stems) - a.middle) // 2)
        band = set(stems[lo:lo + a.middle])
        rows = [r for r in rows if r["stem"] in band]
        print(f"middle {a.middle} of {len(stems)} annotated frames: "
              f"{sorted(band)[0]} .. {sorted(band)[-1]}")

    B = per_boll(rows)
    g, C, r2 = fit(B)
    print(f"instances {len(rows)}   bolls {len(B)}   "
          f"depth spread {min(b['z'] for b in B):.3f}-{max(b['z'] for b in B):.3f}")
    print(f"free fit: gamma = {g:.3f}   C = {C:.3f}   r2 = {r2:.3f}")

    scan = np.arange(0.40, 3.001, 0.01)
    errs = [loo(B, gg) for gg in scan]
    gstar = round(float(scan[int(np.argmin(errs))]), 2)
    print(f"leave-one-boll-out optimum: gamma = {gstar:.2f}  (mean |err| {min(errs):.1f}%)")
    for gg in (0.663, 1.0):
        print(f"   for reference, gamma={gg}: LOO mean |err| {loo(B, gg):.1f}%")

    Cstar = float(np.mean([np.log(b["S"]) - np.log(b["tb"]) - (1 / gstar) * np.log(b["z"])
                           for b in B]))
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(gamma=gstar, C=Cstar, scale=float(np.exp(Cstar)),
                   n_bolls=len(B), n_instances=len(rows),
                   note="size_mm = tb_px * z_raw**(1/gamma) * exp(C); z_raw is raw DA3, no focal",
                   fitted_on=a.xml),
              open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
    print(f"\nsize_mm = tb_px * z_raw**(1/{gstar:.2f}) * {np.exp(Cstar):.6f}")
    print("wrote", os.path.join(a.out, "calibration.json"))


if __name__ == "__main__":
    main()
