#!/usr/bin/env python
"""Score the RGB-D and the monocular-RGB size pipelines against tag ground truth.

Both inputs are viewer boll-report exports with a hand-filled real_id column
(real_id == the `boll` column of cotton_ground_truth.csv). The two runs are
DIFFERENT recordings of different bolls, so this compares the METHODS, not the
same physical objects.

    python scripts/depth/compare_rgbd_vs_rgb.py
"""
import argparse, csv, math, collections
import numpy as np

FITTED_ON = {7, 11, 14, 15, 20, 21, 24}   # bolls the DA3 gamma/scale was fitted on


def load_gt(p):
    return {int(r["boll"]): float(r["height_tip_to_base_mm"]) for r in csv.DictReader(open(p))}


def load_report(p, gt):
    """One row per tagged boll; fragments sharing a real_id are pooled by frame count."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        rid = (r.get("real_id") or "").strip()
        if not rid:
            continue
        med = r.get("median_u") or r.get("median_mm")
        by[int(rid)].append(dict(
            ids=r["track_ids"], frames=int(r["frames"]), used=int(r["used_after_outlier_removal"]),
            qc=int(r["qc_pass"]), med=float(med), sd=float(r.get("sd_u") or r.get("sd_mm")),
            dist=float(r.get("camera_dist_u") or r.get("camera_dist_mm") or "nan")))
    out = []
    for rid, v in sorted(by.items()):
        if rid not in gt:
            continue
        w = np.array([x["frames"] for x in v], float)
        out.append(dict(real_id=rid, n_frag=len(v), ids="+".join(x["ids"] for x in v),
                        frames=int(w.sum()), used=sum(x["used"] for x in v),
                        qc=sum(x["qc"] for x in v),
                        pred=float(np.average([x["med"] for x in v], weights=w)),
                        sd=float(np.average([x["sd"] for x in v], weights=w)),
                        dist=float(np.average([x["dist"] for x in v], weights=w)),
                        gt=gt[rid]))
    for r in out:
        r["err"] = r["pred"] - r["gt"]
        r["err_pct"] = 100 * r["err"] / r["gt"]
    return out


def stats(rows):
    e = np.array([r["err_pct"] for r in rows])
    a = np.array([r["err"] for r in rows])
    p = np.array([r["pred"] for r in rows])
    g = np.array([r["gt"] for r in rows])
    s = dict(n=len(rows), bias=e.mean(), mae=np.abs(e).mean(), medae=np.median(np.abs(e)),
             rmse=math.sqrt((a ** 2).mean()), sd=e.std(ddof=1) if len(e) > 1 else float("nan"),
             worst=np.abs(e).max(), gt_lo=g.min(), gt_hi=g.max(),
             within10=100 * (np.abs(e) <= 10).mean(), within20=100 * (np.abs(e) <= 20).mean(),
             repeat=float(np.median([100 * r["sd"] / r["pred"] for r in rows])),
             dist=float(np.median([r["dist"] for r in rows])))
    s["r"] = float(np.corrcoef(g, p)[0, 1]) if len(rows) > 2 else float("nan")
    # does the method track real size at all, or just predict the mean?
    s["slope"] = float(np.polyfit(g, p, 1)[0]) if len(rows) > 2 else float("nan")
    return s


def table(name, rows, mark_fitted):
    print(f"\n=== {name} ===")
    print(f"{'tag':>4} {'tracks':>12} {'frames':>7} {'used':>5} {'QC':>4} "
          f"{'pred':>7} {'GT':>7} {'err':>7} {'err%':>7} {'sd%':>6} {'dist':>7}")
    for r in sorted(rows, key=lambda r: abs(r["err_pct"])):
        flag = " *" if (mark_fitted and r["real_id"] in FITTED_ON) else ""
        print(f"{r['real_id']:>4} {r['ids']:>12} {r['frames']:>7} {r['used']:>5} {r['qc']:>4} "
              f"{r['pred']:>7.1f} {r['gt']:>7.2f} {r['err']:>+7.1f} {r['err_pct']:>+6.1f}% "
              f"{100*r['sd']/r['pred']:>5.1f}% {r['dist']:>7.0f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgbd", default="inference_results/rdbd_depth_inference/rgbd_size.csv")
    ap.add_argument("--rgb", default="inference_results/LeftDown_v3_soft/left_down_soft.csv")
    ap.add_argument("--gt", default="findings/cotton_pending_calibration/cotton_ground_truth.csv")
    a = ap.parse_args()

    gt = load_gt(a.gt)
    D = load_report(a.rgbd, gt)
    G = load_report(a.rgb, gt)
    table("RGB-D  (RealSense metric depth, pinhole, nothing fitted)", D, False)
    table("RGB    (DA3 monocular, gamma=0.70 + scale fitted on this recording)", G, True)
    print("   * = boll the DA3 gamma/scale was fitted on (in-sample)")

    # Both pipelines share the same predicted tip/base. Splitting each method's
    # error into a common scale factor and the leftover per-boll scatter says how
    # much is one fixable constant and how much is genuinely unreliable.
    def debias(rows):
        k = float(np.mean([r["gt"] / r["pred"] for r in rows]))
        out = [dict(r, pred=r["pred"] * k) for r in rows]
        for r in out:
            r["err"] = r["pred"] - r["gt"]
            r["err_pct"] = 100 * r["err"] / r["gt"]
        return k, out

    sD, sG = stats(D), stats(G)
    print(f"\n{'':28s}{'RGB-D':>12s}{'RGB (DA3)':>12s}")
    print('-' * 52)
    for k, lab, f in [
        ("n", "bolls matched to GT", "{:.0f}"), ("gt_lo", "GT range low (mm)", "{:.1f}"),
        ("gt_hi", "GT range high (mm)", "{:.1f}"), ("dist", "median cam dist (mm)", "{:.0f}"),
        ("bias", "mean signed error", "{:+.1f}%"), ("mae", "mean abs error", "{:.1f}%"),
        ("medae", "median abs error", "{:.1f}%"), ("rmse", "RMSE (mm)", "{:.1f}"),
        ("sd", "sd of signed error", "{:.1f}%"), ("worst", "worst boll", "{:.1f}%"),
        ("within10", "bolls within 10%", "{:.0f}%"), ("within20", "bolls within 20%", "{:.0f}%"),
        ("r", "corr(pred, GT)", "{:+.2f}"), ("slope", "slope pred vs GT", "{:+.2f}"),
        ("repeat", "within-boll sd (repeatability)", "{:.1f}%")]:
        print(f"{lab:28s}{f.format(sD[k]):>12s}{f.format(sG[k]):>12s}")

    kD, dD = debias(D)
    kG, dG = debias(G)
    tD, tG = stats(dD), stats(dG)
    print(f"\nafter removing each method's own scale factor (the one number a "
          f"recalibration could fix):")
    print(f"{'scale needed':28s}{'x%.3f' % kD:>12s}{'x%.3f' % kG:>12s}")
    print(f"{'residual mean abs error':28s}{'%.1f%%' % tD['mae']:>12s}{'%.1f%%' % tG['mae']:>12s}")
    print(f"{'residual worst boll':28s}{'%.1f%%' % tD['worst']:>12s}{'%.1f%%' % tG['worst']:>12s}")

    for lo in (20, 40):
        fD = [r for r in D if r["frames"] >= lo]
        fG = [r for r in G if r["frames"] >= lo]
        if len(fD) > 2 and len(fG) > 2:
            a, b = stats(fD), stats(fG)
            print(f"\nbolls seen in >={lo} frames only        "
                  f"n={a['n']}          n={b['n']}")
            print(f"{'  mean abs error':28s}{'%.1f%%' % a['mae']:>12s}{'%.1f%%' % b['mae']:>12s}")
            print(f"{'  mean signed error':28s}{'%+.1f%%' % a['bias']:>12s}{'%+.1f%%' % b['bias']:>12s}")


if __name__ == "__main__":
    main()
