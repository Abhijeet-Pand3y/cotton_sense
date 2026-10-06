#!/usr/bin/env python
"""Score any number of viewer boll-report exports against tag ground truth.

Each input is a viewer CSV with a hand-filled real_id column (real_id == the
`boll` column of cotton_ground_truth.csv). Runs are different recordings, so the
per-run tables compare METHODS; the cross-run section at the end compares the
SAME physical boll where a tag appears in more than one run.

    python scripts/depth/compare_size_runs.py
    python scripts/depth/compare_size_runs.py --run "label=path.csv" --run ...
"""
import argparse, csv, math, collections
import numpy as np

FITTED_ON = {7, 11, 14, 15, 20, 21, 24}   # bolls the DA3 gamma/scale was fitted on

DEFAULT_RUNS = [
    ("RGB-D (RealSense)", "inference_results/rdbd_depth_inference/rgbd_size.csv"),
    ("RGB left_down (DA3)", "inference_results/LeftDown_v3_soft/left_down_soft.csv"),
    ("RGB right_down (DA3)", "inference_results/RightDown_v5/right_down_size_da3.csv"),
]


def load_gt(p):
    return {int(r["boll"]): float(r["height_tip_to_base_mm"]) for r in csv.DictReader(open(p))}


def col(r, *names):
    for n in names:
        if r.get(n) not in (None, ""):
            return r[n]
    return None


def load_report(p, gt):
    """One row per tagged boll; fragments sharing a real_id are pooled by frame count."""
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        rid = (r.get("real_id") or "").strip()
        if not rid:
            continue
        by[int(rid)].append(dict(
            ids=r["track_ids"], frames=int(r["frames"]),
            used=int(r["used_after_outlier_removal"]), qc=int(r["qc_pass"]),
            med=float(col(r, "median_mm", "median_u")),
            sd=float(col(r, "sd_mm", "sd_u")),
            dist=float(col(r, "camera_dist_mm", "camera_dist_u") or "nan")))
    out = []
    for rid, v in sorted(by.items()):
        if rid not in gt:
            continue
        w = np.array([x["frames"] for x in v], float)
        r = dict(real_id=rid, n_frag=len(v), ids="+".join(x["ids"] for x in v),
                 frames=int(w.sum()), used=sum(x["used"] for x in v),
                 qc=sum(x["qc"] for x in v),
                 pred=float(np.average([x["med"] for x in v], weights=w)),
                 sd=float(np.average([x["sd"] for x in v], weights=w)),
                 dist=float(np.average([x["dist"] for x in v], weights=w)),
                 gt=gt[rid])
        r["err"] = r["pred"] - r["gt"]
        r["err_pct"] = 100 * r["err"] / r["gt"]
        out.append(r)
    return out


def stats(rows):
    e = np.array([r["err_pct"] for r in rows]); a = np.array([r["err"] for r in rows])
    p = np.array([r["pred"] for r in rows]); g = np.array([r["gt"] for r in rows])
    s = dict(n=len(rows), bias=e.mean(), mae=np.abs(e).mean(), medae=np.median(np.abs(e)),
             rmse=math.sqrt((a ** 2).mean()), sd=e.std(ddof=1) if len(e) > 1 else float("nan"),
             worst=np.abs(e).max(), gt_lo=g.min(), gt_hi=g.max(),
             within10=100 * (np.abs(e) <= 10).mean(), within20=100 * (np.abs(e) <= 20).mean(),
             repeat=float(np.median([100 * r["sd"] / r["pred"] for r in rows])),
             dist=float(np.median([r["dist"] for r in rows])),
             scale=float(np.mean([r["gt"] / r["pred"] for r in rows])))
    s["r"] = float(np.corrcoef(g, p)[0, 1]) if len(rows) > 2 else float("nan")
    s["slope"] = float(np.polyfit(g, p, 1)[0]) if len(rows) > 2 else float("nan")
    k = s["scale"]
    res = np.array([100 * (r["pred"] * k - r["gt"]) / r["gt"] for r in rows])
    s["resid_mae"] = np.abs(res).mean()
    return s


def table(name, rows):
    print(f"\n=== {name} ===")
    print(f"{'tag':>4} {'tracks':>10} {'frames':>7} {'used':>5} {'QC':>4} "
          f"{'pred':>7} {'GT':>7} {'err':>7} {'err%':>7} {'sd%':>6} {'dist':>6}")
    for r in sorted(rows, key=lambda r: abs(r["err_pct"])):
        flag = " *" if r["real_id"] in FITTED_ON else ""
        print(f"{r['real_id']:>4} {r['ids']:>10} {r['frames']:>7} {r['used']:>5} {r['qc']:>4} "
              f"{r['pred']:>7.1f} {r['gt']:>7.2f} {r['err']:>+7.1f} {r['err_pct']:>+6.1f}% "
              f"{100*r['sd']/r['pred']:>5.1f}% {r['dist']:>6.0f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None, help='"label=path.csv", repeatable')
    ap.add_argument("--gt", default="findings/cotton_pending_calibration/cotton_ground_truth.csv")
    a = ap.parse_args()
    runs = [tuple(x.split("=", 1)) for x in a.run] if a.run else DEFAULT_RUNS

    gt = load_gt(a.gt)
    data = [(lab, load_report(p, gt)) for lab, p in runs]
    for lab, rows in data:
        table(lab, rows)
    print("   * = boll the DA3 gamma/scale was fitted on (in-sample for the DA3 runs)")

    S = [(lab, stats(rows)) for lab, rows in data]
    w = 22
    print("\n" + " " * 30 + "".join(f"{lab[:w]:>{w}}" for lab, _ in S))
    print("-" * (30 + w * len(S)))
    for k, lab, f in [
        ("n", "bolls matched to GT", "{:.0f}"), ("gt_lo", "GT low (mm)", "{:.1f}"),
        ("gt_hi", "GT high (mm)", "{:.1f}"), ("dist", "median cam dist (mm)", "{:.0f}"),
        ("bias", "mean signed error", "{:+.1f}%"), ("mae", "mean abs error", "{:.1f}%"),
        ("medae", "median abs error", "{:.1f}%"), ("rmse", "RMSE (mm)", "{:.1f}"),
        ("worst", "worst boll", "{:.1f}%"), ("within10", "within 10%", "{:.0f}%"),
        ("within20", "within 20%", "{:.0f}%"), ("r", "corr(pred, GT)", "{:+.2f}"),
        ("slope", "slope pred vs GT", "{:+.2f}"),
        ("repeat", "within-boll sd", "{:.1f}%"),
        ("scale", "scale factor needed", "x{:.3f}"),
        ("resid_mae", "MAE after that scale", "{:.1f}%")]:
        print(f"{lab:30s}" + "".join(f"{f.format(s[k]):>{w}}" for _, s in S))

    # ---- same physical boll, measured by more than one run -------------------
    idx = {lab: {r["real_id"]: r for r in rows} for lab, rows in data}
    shared = sorted(set.intersection(*[set(d) for d in idx.values()])) if len(idx) > 1 else []
    everywhere = shared
    seen = collections.Counter(rid for d in idx.values() for rid in d)
    multi = sorted(rid for rid, c in seen.items() if c > 1)
    if multi:
        print(f"\n=== same boll across runs ({len(multi)} tags in >1 run, "
              f"{len(everywhere)} in all {len(data)}) ===")
        print(f"{'tag':>4} {'GT':>7}" + "".join(f"{lab[:w]:>{w}}" for lab, _ in data))
        for rid in multi:
            line = f"{rid:>4} {gt[rid]:>7.2f}"
            for lab, _ in data:
                r = idx[lab].get(rid)
                line += f"{(f'{r[chr(112)+chr(114)+chr(101)+chr(100)]:.1f} ({r[chr(101)+chr(114)+chr(114)+chr(95)+chr(112)+chr(99)+chr(116)]:+.0f}%)' if r else '—'):>{w}}"
            print(line)
        for rid in multi:
            preds = [idx[lab][rid]["pred"] for lab, _ in data if rid in idx[lab]]
            if len(preds) > 1:
                sp = 100 * (max(preds) - min(preds)) / np.mean(preds)
                print(f"     tag {rid}: methods disagree by {sp:.0f}% of their mean")


if __name__ == "__main__":
    main()
