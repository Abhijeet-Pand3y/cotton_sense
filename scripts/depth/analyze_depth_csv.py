"""
analyze_depth_csv.py — summarise the CSV from sample_boll_depth.py.

Separated from extraction so the analysis can be re-run, and re-tuned, without
regenerating depth maps or re-sampling.

REPORTS
  1. GATE HEALTH      what fraction of bolls fell back to the boll depth because
                      nothing passed the tolerance gate, and the distribution of
                      pass rates. Tells you whether --tol_pct is doing real work.
  2. STABILITY        per-track spread of boll depth across frames. This is the
                      per-frame noise floor and it caps every downstream number.
  3. AVERAGING CHECK  whether averaging k frames actually shrinks that spread by
                      the ~sqrt(k) an independent-noise model predicts. If it
                      does not, the error is systematic — a bias tied to
                      viewpoint — and more frames will not help.
  4. dz               tip-minus-base depth. Reported against the per-track noise
                      floor, because dz is only meaningful if it exceeds it.

USAGE
  python analyze_depth_csv.py --csv depths_left.csv
  python analyze_depth_csv.py --csv depths_left.csv --min_frames 5 --by occ_class
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def fnum(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def fbool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def spread(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    med = float(np.median(a))
    return {"n": a.size, "med": med,
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "cv_pct": (100 * float(a.std(ddof=1)) / med) if (a.size > 1 and med) else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def pct(a, p):
    return float(np.percentile(np.asarray(a, dtype=np.float64), p))


def frame_num(stem):
    """Leading frame index out of a stem like frame_002303_t76p84s."""
    import re
    m = re.search(r"(\d+)", str(stem))
    return int(m.group(1)) if m else None


def adjacent_noise(by_track_seq, max_gap=1):
    """Noise from consecutive-frame differences.

    Per-track spread over a long track is NOT noise: the camera moves, so the
    boll really is at different distances. Between adjacent frames at 30 fps it
    moves ~1 mm, so a larger step is measurement error.

    For independent noise of std sigma, the difference of two frames has std
    sigma*sqrt(2), so sigma = std(diff)/sqrt(2). Reported in mm, and relative to
    the local depth.
    """
    diffs, rels = [], []
    for _, seq in by_track_seq.items():
        seq = sorted(seq, key=lambda t: t[0])
        for (f0, z0), (f1, z1) in zip(seq, seq[1:]):
            if f1 - f0 > max_gap:
                continue
            diffs.append(z1 - z0)
            if z0:
                rels.append((z1 - z0) / z0)
    if len(diffs) < 10:
        return None
    d = np.asarray(diffs)
    r = np.asarray(rels)
    return {"n": d.size,
            "sigma_mm": float(d.std(ddof=1)) / np.sqrt(2) * 1000,
            "sigma_pct": float(r.std(ddof=1)) / np.sqrt(2) * 100,
            "mad_mm": float(np.median(np.abs(d - np.median(d)))) * 1000,
            "med_step_mm": float(np.median(np.abs(d))) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min_frames", type=int, default=5,
                    help="tracks need at least this many frames to be assessed")
    ap.add_argument("--by", default=None,
                    help="also break gate health down by this column "
                         "(e.g. occ_class)")
    ap.add_argument("--show_tracks", type=int, default=15,
                    help="how many per-track rows to print")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty CSV")
    print(f"{len(rows)} boll measurements from {args.csv}")
    frames = {r["frame"] for r in rows}
    print(f"{len(frames)} frames, "
          f"{len({r['track'] for r in rows if r['track'] not in ('', 'None')})} tracks")

    # ---- 1. gate health ---------------------------------------------------
    tip_est = [fbool(r["tip_est"]) for r in rows]
    base_est = [fbool(r["base_est"]) for r in rows]
    tip_keep = [fnum(r["tip_kept_pct"]) for r in rows if not fbool(r["tip_est"])]
    tip_keep = [v for v in tip_keep if v is not None]

    print("\n" + "=" * 70)
    print("1. GATE HEALTH")
    print("=" * 70)
    print(f"tip  fell back to boll depth : {sum(tip_est)}/{len(rows)} "
          f"({100*sum(tip_est)/len(rows):.1f}%)")
    print(f"base fell back to boll depth : {sum(base_est)}/{len(rows)} "
          f"({100*sum(base_est)/len(rows):.1f}%)")
    if tip_keep:
        print(f"tip pass rate (where it worked): "
              f"p10={pct(tip_keep,10):.0f}%  med={pct(tip_keep,50):.0f}%  "
              f"p90={pct(tip_keep,90):.0f}%")
        m = pct(tip_keep, 50)
        if m < 10:
            print("  -> very low: tolerance too tight, or keypoints off the boll")
        elif m > 90:
            print("  -> very high: tolerance too loose to reject anything")
        else:
            print("  -> the gate is discriminating")

    if args.by and args.by in rows[0]:
        print(f"\nby {args.by}:")
        g = defaultdict(list)
        for r in rows:
            g[r[args.by]].append(fbool(r["tip_est"]))
        for k, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
            print(f"  {str(k)[:14]:>14}  n={len(v):>5}  "
                  f"fallback={100*sum(v)/len(v):>5.1f}%")

    # ---- 2. per-track stability ------------------------------------------
    by_track = defaultdict(list)
    by_track_seq = defaultdict(list)
    for r in rows:
        t = r["track"]
        z = fnum(r["z_boll"])
        if t in ("", "None") or z is None:
            continue
        by_track[t].append(z)
        fn = frame_num(r["frame"])
        if fn is not None:
            by_track_seq[t].append((fn, z))
    multi = {t: v for t, v in by_track.items() if len(v) >= args.min_frames}

    print("\n" + "=" * 70)
    print(f"2. TRACK SPREAD — boll depth per track ({args.min_frames}+ frames)")
    print("=" * 70)
    print("CAUTION: over a long track the camera moves, so most of this spread is")
    print("REAL depth change, not error. Use section 2b for the noise floor.")
    if not multi:
        print(f"no track reached {args.min_frames} frames")
        cvs = []
    else:
        stats = {t: spread(v) for t, v in multi.items()}
        cvs = [s["cv_pct"] for s in stats.values()]
        print(f"{'track':>7} {'n':>5} {'med (m)':>9} {'std (mm)':>9} {'CV %':>7} "
              f"{'range (mm)':>11}")
        shown = sorted(stats.items(), key=lambda kv: -kv[1]["n"])[:args.show_tracks]
        for t, s in shown:
            print(f"{t:>7} {s['n']:>5} {s['med']:>9.3f} {s['std']*1000:>9.1f} "
                  f"{s['cv_pct']:>7.2f} {(s['max']-s['min'])*1000:>11.1f}")
        if len(stats) > len(shown):
            print(f"  ... {len(stats)-len(shown)} more tracks")
        print("-" * 70)
        print(f"median CV across {len(stats)} tracks : "
              f"{float(np.median(cvs)):.2f}%  (includes real motion)")

    # ---- 2b. the actual noise floor --------------------------------------
    an = adjacent_noise(by_track_seq)
    print("\n" + "=" * 70)
    print("2b. NOISE FLOOR — consecutive-frame steps")
    print("=" * 70)
    if an is None:
        print("too few consecutive-frame pairs to estimate")
        noise_mm = None
    else:
        print(f"pairs of adjacent frames : {an['n']}")
        print(f"median |step|            : {an['med_step_mm']:.1f} mm")
        print(f"noise sigma              : {an['sigma_mm']:.1f} mm "
              f"({an['sigma_pct']:.2f}%)")
        print(f"robust (MAD) step        : {an['mad_mm']:.1f} mm")
        noise_mm = an["sigma_mm"]
        print("-" * 70)
        print("At 30 fps the camera moves ~1 mm between frames, so nearly all of")
        print("this is measurement error. THIS is the floor, not the track spread.")
        print("MAD well below sigma => a few large jumps, not uniform jitter;")
        print("median filtering along the track will help more than averaging.")

    # ---- 3. does averaging help? -----------------------------------------
    if by_track_seq:
        print("\n" + "=" * 70)
        print("3. AVERAGING CHECK (detrended)")
        print("=" * 70)
        print("The camera trend is removed first (rolling median), so this asks")
        print("only whether averaging shrinks the NOISE around that trend.")
        resid = defaultdict(list)
        for t, seq in by_track_seq.items():
            seq = sorted(seq, key=lambda x: x[0])
            z = np.asarray([v for _, v in seq], dtype=np.float64)
            if z.size < 11:
                continue
            w = 9
            trend = np.array([np.median(z[max(0, i - w // 2):i + w // 2 + 1])
                              for i in range(z.size)])
            resid[t] = (z - trend).tolist()
        if not resid:
            print("no track long enough to detrend")
        else:
            base = float(np.median([np.std(v, ddof=1) for v in resid.values()])) * 1000
            print(f"{'k frames':>9} {'observed sigma (mm)':>21} {'sqrt(k) model':>15}")
            for k in (1, 2, 4, 8):
                obs = []
                for v in resid.values():
                    a = np.asarray(v)
                    if a.size < k * 3:
                        continue
                    nb = a.size // k
                    means = a[:nb * k].reshape(nb, k).mean(axis=1)
                    if means.size > 1:
                        obs.append(means.std(ddof=1) * 1000)
                if obs:
                    print(f"{k:>9} {np.median(obs):>21.2f} {base/np.sqrt(k):>15.2f}")
            print("-" * 70)
            print("observed follows the model -> noise is independent; average away")
            print("observed flattens          -> correlated noise; averaging caps out")

    # ---- 4. dz ------------------------------------------------------------
    dz = [fnum(r["dz"]) for r in rows]
    dz = [v * 1000 for v in dz if v is not None]           # mm
    real = [v * 1000 for r, v in zip(rows, [fnum(x["dz"]) for x in rows])
            if v is not None and not fbool(r["tip_est"]) and not fbool(r["base_est"])]
    print("\n" + "=" * 70)
    print("4. dz — tip depth minus base depth (mm)")
    print("=" * 70)
    if not dz:
        print("no dz values")
    else:
        s = spread(dz)
        print(f"all           n={s['n']:>5}  med={s['med']:>7.1f}  "
              f"p10={pct(dz,10):>7.1f}  p90={pct(dz,90):>7.1f}")
        if real:
            s2 = spread(real)
            print(f"gate-measured n={s2['n']:>5}  med={s2['med']:>7.1f}  "
                  f"p10={pct(real,10):>7.1f}  p90={pct(real,90):>7.1f}")
            print(f"|dz| median   : {np.median(np.abs(real)):.1f} mm")
        if noise_mm is not None:
            mag = np.median(np.abs(real)) if real else np.median(np.abs(dz))
            print(f"noise floor from z_boll (2b) : {noise_mm:.1f} mm")

            # dz has its OWN, lower floor: common-mode drift cancels in the
            # difference, so measure dz's adjacent-frame steps directly.
            dz_seq = defaultdict(list)
            for r in rows:
                t = r["track"]
                v = fnum(r["dz"])
                fn = frame_num(r["frame"])
                if (t not in ("", "None") and v is not None and fn is not None
                        and not fbool(r["tip_est"]) and not fbool(r["base_est"])):
                    dz_seq[t].append((fn, v))
            dzn = adjacent_noise(dz_seq)
            print("-" * 70)
            if dzn is None:
                print("too few adjacent gate-measured pairs to get dz's own floor")
                floor = noise_mm
            else:
                floor = dzn["sigma_mm"]
                print(f"dz's OWN noise floor        : {floor:.1f} mm "
                      f"(from {dzn['n']} adjacent pairs)")
                print(f"  median |step| in dz       : {dzn['med_step_mm']:.1f} mm")
                if floor < noise_mm:
                    print(f"  -> lower than the z_boll floor, as expected: "
                          f"common-mode drift cancels")
                else:
                    print(f"  -> NOT lower than the z_boll floor; tip and base errors")
                    print(f"     are not cancelling, so dz gains nothing from the")
                    print(f"     difference")
            print("-" * 70)
            if mag > 3 * floor:
                print(f"|dz| ({mag:.1f} mm) is >3x its floor ({floor:.1f}) -> real signal")
            elif mag > floor:
                print(f"|dz| ({mag:.1f} mm) is ~{mag/floor:.1f}x its floor ({floor:.1f})")
                print("     -> weak; may become usable with median filtering")
            else:
                print(f"|dz| ({mag:.1f} mm) is at or below its floor ({floor:.1f})")
                print("     -> not measurable per frame")


    # ---- 5. size proxy: does depth correction help? -----------------------
    if "size_proxy_axis" in rows[0]:
        print("\n" + "=" * 70)
        print("5. SIZE — raw pixels vs depth-corrected proxy")
        print("=" * 70)
        print("A boll's true size does not change, but its PIXEL size shrinks as it")
        print("gets farther away. Multiplying by depth should cancel that. So for")
        print("each track, does px*Z vary LESS than px alone? If yes, the depth")
        print("correction is doing real work — no focal length or ground truth")
        print("needed to check it.")
        print()

        for label, px_col, pr_col in (("kpt distance", "px_kpt", "size_proxy_kpt"),
                                      ("mask axis", "px_axis", "size_proxy_axis")):
            per_track = defaultdict(lambda: {"px": [], "pr": [], "z": []})
            for r in rows:
                t = r["track"]
                a, b, z = fnum(r[px_col]), fnum(r[pr_col]), fnum(r["z_boll"])
                if t in ("", "None") or a is None or b is None or z is None:
                    continue
                per_track[t]["px"].append(a)
                per_track[t]["pr"].append(b)
                per_track[t]["z"].append(z)

            cv_px, cv_pr, zrng, wins = [], [], [], 0
            for t, v in per_track.items():
                if len(v["px"]) < args.min_frames:
                    continue
                a, b, z = (np.asarray(v[k], dtype=np.float64) for k in ("px", "pr", "z"))
                if a.mean() <= 0 or b.mean() <= 0:
                    continue
                c1 = 100 * a.std(ddof=1) / a.mean()
                c2 = 100 * b.std(ddof=1) / b.mean()
                cv_px.append(c1)
                cv_pr.append(c2)
                zrng.append(100 * (z.max() - z.min()) / z.mean())
                wins += (c2 < c1)

            if not cv_px:
                print(f"{label}: no track with {args.min_frames}+ measurements")
                continue
            mpx, mpr = float(np.median(cv_px)), float(np.median(cv_pr))
            print(f"{label}:")
            print(f"  tracks assessed            : {len(cv_px)}")
            print(f"  depth varied within track  : {float(np.median(zrng)):.1f}% "
                  f"(median across tracks)")
            print(f"  CV of raw pixels           : {mpx:.2f}%")
            print(f"  CV of px x Z               : {mpr:.2f}%")
            print(f"  proxy better on            : {wins}/{len(cv_px)} tracks")
            if mpr < mpx * 0.85:
                print(f"  -> depth correction HELPS ({mpx/max(mpr,1e-9):.2f}x tighter)")
            elif mpr < mpx * 1.15:
                print(f"  -> about even; depth is neither helping nor hurting here")
            else:
                print(f"  -> depth correction HURTS; its noise exceeds the "
                      f"perspective effect it removes")
            print()

        print("-" * 70)
        print("Caveats: this only tests distance-invariance, not correctness. A")
        print("boll also rotates as the camera moves, so some of the pixel")
        print("variation is real foreshortening that depth cannot fix. And if the")
        print("depth barely varied within a track, there was little perspective")
        print("effect to remove, so the comparison is uninformative.")
        print()
        print("Units: size_proxy = px x Z = true_length x f, i.e. proportional to")
        print("real size. Multiply by 1/f once the focal length is known, or fit")
        print("the constant against a few measured bolls.")


if __name__ == "__main__":
    main()
