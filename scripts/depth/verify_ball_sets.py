"""
verify_ball_sets.py — independent re-derivation of the ball-set depth results.

WHY A SECOND SCRIPT RATHER THAN A FLAG ON THE FIRST
  Re-running the same code proves only that it is deterministic. This re-derives
  the numbers by a deliberately different route and then compares:

  analyse_ball_sets.py                     this script
  ------------------------------------     ------------------------------------
  median of z over frames, per ball        every frame kept separate
  one gamma per SET, from ball medians     one gamma per FRAME, then a
                                             distribution over frames
  disc mask built over the whole map       crop to a bounding box first
  median only                              median / mean / trimmed, and four
                                             disc sizes
  no uncertainty on gamma                  spread across frames + bootstrap

  Agreement between the two is evidence; disagreement localises a bug. Neither
  is trusted on its own.

WHAT IT ALSO TESTS THAT THE FIRST PASS COULD NOT
  1. Is gamma stable frame to frame, or is one number hiding a wide spread?
  2. Does the moving object change gamma? (The earlier study said clutter is
     nearly harmless; here that can be checked against real distance.)
  3. Is the 25/27 in ordering inversion present in every frame, or an artefact
     of one bad frame surviving into the median?
  4. Do the conclusions survive changing the sampling disc and the statistic?

USAGE
  python verify_ball_sets.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MM_PER_IN = 25.4
BALL_MM = 40.0
RNG = np.random.default_rng(20260817)


# ---------------------------------------------------------------- sampling ---
def sample_box(depth, cx, cy, r, w, h, frac, stat):
    """Independent sampler: crop first, then mask inside the crop.

    Deliberately not the whole-image mgrid the first script uses. If both agree
    the geometry of the mapping is right; if they do not, one of them indexes
    the depth map wrongly.
    """
    sx, sy = depth.shape[1] / w, depth.shape[0] / h
    dcx, dcy = cx * sx, cy * sy
    dr = r * frac * 0.5 * (sx + sy)
    x0, x1 = int(np.floor(dcx - dr)), int(np.ceil(dcx + dr)) + 1
    y0, y1 = int(np.floor(dcy - dr)), int(np.ceil(dcy + dr)) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, depth.shape[1]), min(y1, depth.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None, 0
    sub = depth[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = ((xx - dcx) ** 2 + (yy - dcy) ** 2) <= dr * dr
    v = sub[m & np.isfinite(sub) & (sub > 0)]
    if v.size < 4:
        return None, 0
    if stat == "median":
        return float(np.median(v)), v.size
    if stat == "mean":
        return float(v.mean()), v.size
    lo, hi = np.percentile(v, [20, 80])
    t = v[(v >= lo) & (v <= hi)]
    return float(t.mean() if t.size else np.median(v)), v.size


def load(frac=0.6, stat="median"):
    import cv2
    out = []
    for sd in sorted(Path("frames").glob("GX0101*_test")):
        pf = sd / "locate_out" / "per_frame_balls.csv"
        dd = sd / "depth" / "depth"
        md = sd / "metadata.csv"
        if not (pf.exists() and dd.exists() and md.exists()):
            continue
        mrows = list(csv.DictReader(open(md)))
        cols = sorted([c for c in mrows[0] if c.startswith("ball")
                       and c.endswith("_distance_in")],
                      key=lambda c: int(c[4:-12]) if c[4:-12].isdigit() else 0)
        dist = {}
        for i, c in enumerate(cols, 1):
            v = {r[c] for r in mrows if r.get(c)}
            if len(v) == 1:
                dist[f"ball{i}"] = float(v.pop()) * MM_PER_IN
        wh = {}
        for r in csv.DictReader(open(pf)):
            if r["status"] != "ok" or r["cx"] == "":
                continue
            npy = dd / f"{r['frame']}.npy"
            if not npy.exists() or r["ball"] not in dist:
                continue
            if r["frame"] not in wh:
                p = sd / f"{r['frame']}.jpg"
                if not p.exists():
                    p = sd / "baseline" / f"{r['frame']}.jpg"
                im = cv2.imread(str(p))
                wh[r["frame"]] = (im.shape[1], im.shape[0])
            w, h = wh[r["frame"]]
            z, n = sample_box(np.load(npy).astype(np.float32), float(r["cx"]),
                              float(r["cy"]), float(r["r"]), w, h, frac, stat)
            if z is None:
                continue
            out.append({"set": sd.name, "frame": r["frame"],
                        "is_base": int(r["is_base"]), "ball": r["ball"],
                        "r_px": float(r["r"]), "z": z, "n": n,
                        "Z": dist[r["ball"]]})
    return out


def gfit(Z, z):
    Z, z = np.asarray(Z, float), np.asarray(z, float)
    if np.unique(np.round(Z, 1)).size < 2:
        return None
    g, la = np.polyfit(np.log(Z), np.log(z), 1)
    return float(g), float(np.exp(la))


def main():
    base = load()
    print(f"independent sampling: {len(base)} ball-frames\n")

    # ---- 1. cross-check against the first implementation -------------------
    ref = {(r["set"], r["frame"], r["ball"]): float(r["z"])
           for r in csv.DictReader(open("findings/ball_depth_samples.csv"))}
    d = [(r["z"] - ref[(r["set"], r["frame"], r["ball"])])
         for r in base if (r["set"], r["frame"], r["ball"]) in ref]
    d = np.array(d)
    print("=== 1. cross-check vs analyse_ball_sets.py ===")
    print(f"  matched {d.size} of {len(base)} rows; "
          f"max |difference| = {np.abs(d).max():.2e}, "
          f"mean |difference| = {np.abs(d).mean():.2e}")
    print("  (two different samplers over the same discs; a non-trivial "
          "difference here would mean one of them maps image pixels to depth "
          "pixels wrongly)\n")

    by_set = defaultdict(list)
    for r in base:
        by_set[r["set"]].append(r)

    # ---- 2. per-frame gamma, per set ---------------------------------------
    print("=== 2. gamma per FRAME (first pass fitted one gamma per set) ===")
    print(f"{'set':<34}{'frames':>7}{'gamma med':>11}{'sd':>8}"
          f"{'min':>8}{'max':>8}{'set-level':>11}")
    frame_g, set_g = {}, {}
    for s, rows in sorted(by_set.items()):
        per_frame = defaultdict(list)
        for r in rows:
            per_frame[r["frame"]].append(r)
        gs = []
        for f, rr in per_frame.items():
            if len({round(x["Z"], 1) for x in rr}) < 2:
                continue
            g = gfit([x["Z"] for x in rr], [x["z"] for x in rr])
            if g:
                gs.append(g[0])
        agg = defaultdict(list)
        for r in rows:
            agg[r["ball"]].append(r["z"])
        gset = gfit([next(x["Z"] for x in rows if x["ball"] == b) for b in agg],
                    [float(np.median(v)) for v in agg.values()])
        frame_g[s], set_g[s] = gs, (gset[0] if gset else None)
        if gs:
            print(f"{s:<34}{len(gs):>7}{np.median(gs):>11.3f}"
                  f"{np.std(gs, ddof=1) if len(gs) > 1 else 0:>8.3f}"
                  f"{min(gs):>8.3f}{max(gs):>8.3f}"
                  f"{(gset[0] if gset else float('nan')):>11.3f}")
        else:
            print(f"{s:<34}{0:>7}{'-':>11}{'-':>8}{'-':>8}{'-':>8}"
                  f"{'-':>11}   (single distance)")
    print()

    # ---- 3. does the moving object shift gamma? ----------------------------
    print("=== 3. baseline frames vs object frames ===")
    print(f"{'set':<34}{'gamma base':>12}{'gamma obj':>11}{'shift':>9}")
    for s, rows in sorted(by_set.items()):
        out = []
        for flag in (1, 0):
            sub = [r for r in rows if r["is_base"] == flag]
            agg = defaultdict(list)
            for r in sub:
                agg[r["ball"]].append(r["z"])
            if len({round(next(x["Z"] for x in sub if x["ball"] == b), 1)
                    for b in agg}) < 2:
                out.append(None); continue
            g = gfit([next(x["Z"] for x in sub if x["ball"] == b) for b in agg],
                     [float(np.median(v)) for v in agg.values()])
            out.append(g[0] if g else None)
        if out[0] is not None and out[1] is not None:
            print(f"{s:<34}{out[0]:>12.3f}{out[1]:>11.3f}"
                  f"{out[1]-out[0]:>+9.3f}")
    print("  a large shift would mean the distractor changes the compression "
          "itself, not just adds noise\n")

    # ---- 4. is the ordering inversion real, frame by frame? ----------------
    print("=== 4. ordering, checked in every frame ===")
    for s, rows in sorted(by_set.items()):
        per_frame = defaultdict(list)
        for r in rows:
            per_frame[r["frame"]].append(r)
        tot = bad = 0
        detail = defaultdict(int)
        for f, rr in per_frame.items():
            if len(rr) < 2:
                continue
            tot += 1
            o = sorted(rr, key=lambda x: x["Z"])
            inv = [(o[i]["Z"], o[j]["Z"]) for i in range(len(o))
                   for j in range(i + 1, len(o)) if o[j]["z"] <= o[i]["z"]]
            if inv:
                bad += 1
                for a, b in inv:
                    detail[f"{a/MM_PER_IN:.0f}<{b/MM_PER_IN:.0f}"] += 1
        if tot:
            print(f"  {s:<34}{bad}/{tot} frames with an inversion"
                  + (f"   {dict(detail)}" if detail else ""))
    print("  a pair inverted in every frame is a systematic model error; one "
          "inverted occasionally is noise\n")

    # ---- 5. sensitivity to the sampling choice -----------------------------
    print("=== 5. sensitivity: does gamma depend on how we sample? ===")
    print(f"{'disc':<8}{'stat':<10}{'mean gamma':>12}{'sd':>8}{'n sets':>8}")
    for frac in (0.4, 0.5, 0.6, 0.7):
        for stat in ("median", "trimmed"):
            rows = load(frac, stat)
            bs = defaultdict(list)
            for r in rows:
                bs[r["set"]].append(r)
            gg = []
            for s, rr in bs.items():
                agg = defaultdict(list)
                for r in rr:
                    agg[r["ball"]].append(r["z"])
                g = gfit([next(x["Z"] for x in rr if x["ball"] == b) for b in agg],
                         [float(np.median(v)) for v in agg.values()])
                if g:
                    gg.append(g[0])
            print(f"{frac:<8}{stat:<10}{np.mean(gg):>12.3f}"
                  f"{np.std(gg, ddof=1):>8.3f}{len(gg):>8}")
    print("  gamma moving with the disc size would mean it is an artefact of "
          "where we sample, not a property of the depth model\n")

    # ---- 6. combined ------------------------------------------------------
    print("=== 6. COMBINED ===")
    gsets = [set_g[s] for s in set_g if set_g[s] is not None]
    allf = [g for s in frame_g for g in frame_g[s]]
    print(f"  gamma from per-set fits : mean {np.mean(gsets):.3f}  "
          f"sd {np.std(gsets, ddof=1):.3f}  n={len(gsets)} sets")
    print(f"  gamma from per-frame fits: median {np.median(allf):.3f}  "
          f"sd {np.std(allf, ddof=1):.3f}  n={len(allf)} frames")

    bs = [np.mean(RNG.choice(gsets, len(gsets), replace=True))
          for _ in range(20000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  bootstrap 95% CI on the set-level mean: [{lo:.3f}, {hi:.3f}]")
    print(f"  prior study 0.670 is {'INSIDE' if lo <= 0.670 <= hi else 'OUTSIDE'} "
          f"that interval")
    print(f"  gamma = 1 (no compression) is "
          f"{'INSIDE' if lo <= 1.0 <= hi else 'OUTSIDE'} it -> compression is "
          f"{'not ' if lo <= 1.0 <= hi else ''}established\n")

    # pooled focal, per set and combined
    print("  focal length, solved per set (independent of DA3):")
    fs = []
    for s, rows in sorted(by_set.items()):
        agg = defaultdict(list)
        for r in rows:
            agg[r["ball"]].append(r["r_px"])
        Z = np.array([next(x["Z"] for x in rows if x["ball"] == b) for b in agg])
        rp = np.array([np.mean(v) for v in agg.values()])
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        sl, ic = np.polyfit(Z, 1 / rp, 1)
        f_px = 2 / (sl * BALL_MM)
        fs.append(f_px)
        print(f"    {s:<34}f = {f_px:8.0f} px   (offset {ic/sl:+7.1f} mm)")
    print(f"    {'combined':<34}f = {np.mean(fs):8.0f} px   "
          f"sd {np.std(fs, ddof=1):.0f} ({100*np.std(fs, ddof=1)/np.mean(fs):.1f}%)")
    print("  these are fitted from ball size and tape alone. Consistency across "
          "sets means the rig and the tape are trustworthy.")


if __name__ == "__main__":
    main()
