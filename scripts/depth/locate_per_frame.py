"""
locate_per_frame.py — re-find and re-fit every ball in EVERY frame, instead of
reusing one static circle taken from the baseline median.

WHY THIS EXISTS
  ball_video_test.py locates the balls once, from a median stack, and applies
  that circle to every frame. That assumes the balls never move by even a pixel.
  They do: in this data the balls sit tens of pixels away from the median
  circle in the later frames, so the depth disc was sampling the wrong pixels.

HOW A BALL IS FOUND IN A FRAME
  1. COARSE — the ball's appearance from the baseline median is matched against
     a search window around where it used to be (normalised cross-correlation).
     This is what lets a ball that has moved still be found, and the match score
     is a direct measure of whether the ball is even there.
  2. FINE — a sub-pixel radial edge fit from that coarse centre, with the radius
     window held near the baseline radius so the rays cannot run away onto the
     stand or cup underneath.

OCCLUSION
  When the object covers a ball there is nothing to match and no edge to fit, so
  occlusion shows up as a low match score, a failed fit, or a distorted one.
  Thresholds are calibrated from the clean baseline frames themselves — those
  frames define what "unoccluded" looks like for this set — with fixed floors so
  a set with only one baseline frame still behaves.

  Every raw number is written to the CSV, so the call can be re-judged later
  without re-running anything.

MOVED vs OCCLUDED
  These are different and are reported separately. A ball that moved is still
  measurable at its new position; a ball that is occluded is not measurable at
  all. With two or more balls the script also reports whether the shift is
  common to all of them, which distinguishes the camera moving from the balls
  being knocked.

USAGE
  python locate_per_frame.py --set frames/GX010117_ball9in_test
  python locate_per_frame.py --frames <dir> --baseline <dir> \
      --balls <balls.json> --out <dir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_video_test import radial_fit, IMG_EXTS      # noqa: E402


def coarse_match(gray, tmpl, cx, cy, r, search):
    """Find the ball's centre by matching its baseline appearance nearby.

    Returns (cx, cy, score). score is normalised cross-correlation at the best
    position: high when the ball is there and looks like itself, low when
    something else is in front of it.
    """
    import cv2
    h, w = gray.shape
    pad = int(r * search)
    th, tw = tmpl.shape
    x0 = int(max(0, cx - tw / 2 - pad)); x1 = int(min(w, cx + tw / 2 + pad))
    y0 = int(max(0, cy - th / 2 - pad)); y1 = int(min(h, cy + th / 2 + pad))
    win = gray[y0:y1, x0:x1]
    if win.shape[0] < th or win.shape[1] < tw:
        return cx, cy, float("nan")
    res = cv2.matchTemplate(win.astype(np.uint8), tmpl.astype(np.uint8),
                            cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2, float(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", help="set folder with baseline/ and locate_out/")
    ap.add_argument("--frames", help="folder of object frames")
    ap.add_argument("--baseline", help="folder of clean frames")
    ap.add_argument("--balls", help="balls.json from locate")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--min_grad", type=float, default=4.0)
    ap.add_argument("--search", type=float, default=1.2,
                    help="coarse search radius, in ball radii, around the "
                         "baseline position")
    ap.add_argument("--min_rays", type=int, default=25)
    ap.add_argument("--max_dr", type=float, default=0.12,
                    help="fractional radius change vs baseline that is too much")
    ap.add_argument("--score_floor", type=float, default=0.45,
                    help="match score below this = ball not visible")
    ap.add_argument("--rms_floor", type=float, default=8.0)
    ap.add_argument("--no_viz", action="store_true")
    args = ap.parse_args()

    if args.set:
        s = Path(args.set)
        args.frames = args.frames or str(s)
        args.baseline = args.baseline or str(s / "baseline")
        args.balls = args.balls or str(s / "locate_out" / "balls.json")
        args.out = args.out or str(s / "locate_out")
    for k in ("frames", "baseline", "balls", "out"):
        if not getattr(args, k):
            raise SystemExit(f"--{k} is required (or use --set)")

    import cv2
    balls = json.load(open(args.balls))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    med = cv2.imread(str(Path(args.balls).parent / "median.jpg"))
    if med is None:
        raise SystemExit("median.jpg not found next to balls.json (rerun locate)")
    med_gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)

    # Template is the ball only — a square inside the circle, so the stand and
    # background do not drive the match.
    tmpl = {}
    for b in balls:
        s_ = b["r"] * 0.72
        x0, x1 = int(b["cx"] - s_), int(b["cx"] + s_)
        y0, y1 = int(b["cy"] - s_), int(b["cy"] + s_)
        tmpl[b["id"]] = med_gray[max(0, y0):y1, max(0, x0):x1].copy()

    obj = sorted(p for p in Path(args.frames).iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS)
    base = sorted(p for p in Path(args.baseline).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)
    frames = [(p, 1) for p in base] + [(p, 0) for p in obj]
    print(f"{len(base)} baseline + {len(obj)} object frames, {len(balls)} ball(s)")

    rows, viz = [], []
    for fp, is_base in frames:
        im = cv2.imread(str(fp))
        if im is None:
            print(f"  [skip] unreadable {fp.name}")
            continue
        graw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(graw, (5, 5), 1.2)
        vis = im.copy()
        for b in balls:
            mcx, mcy, score = coarse_match(graw, tmpl[b["id"]], b["cx"], b["cy"],
                                           b["r"], args.search)
            fit = radial_fit(gray, mcx, mcy, b["r"] * 0.85, b["r"] * 1.15,
                             min_grad=args.min_grad)
            # Focus on the ball itself. A soft frame has no strong edge for the
            # radial fit to find, which looks like a failure but is not an
            # occlusion — this number tells the two apart.
            s_ = int(b["r"])
            px0, px1 = int(max(0, mcx - s_)), int(min(graw.shape[1], mcx + s_))
            py0, py1 = int(max(0, mcy - s_)), int(min(graw.shape[0], mcy + s_))
            reg = graw[py0:py1, px0:px1]
            focus = (float(cv2.Laplacian(reg, cv2.CV_64F).var())
                     if reg.size > 100 else float("nan"))
            row = {"frame": fp.stem, "is_base": is_base, "ball": b["id"],
                   "base_cx": b["cx"], "base_cy": b["cy"], "base_r": b["r"],
                   "match": round(score, 3) if np.isfinite(score) else "",
                   "focus": round(focus, 1) if np.isfinite(focus) else "",
                   "coarse_cx": round(mcx, 1), "coarse_cy": round(mcy, 1)}
            if fit is None:
                row.update(cx="", cy="", r="", n_rays=0, rms="",
                           d_cx="", d_cy="", d_center="", dr_pct="")
            else:
                cx, cy, r, n, rms = fit
                row.update(cx=round(cx, 2), cy=round(cy, 2), r=round(r, 2),
                           n_rays=n, rms=round(rms, 2),
                           d_cx=round(cx - b["cx"], 2),
                           d_cy=round(cy - b["cy"], 2),
                           d_center=round(float(np.hypot(cx - b["cx"],
                                                         cy - b["cy"])), 2),
                           dr_pct=round(100 * (r - b["r"]) / b["r"], 2))
            rows.append(row)

            th_ = max(2, im.shape[1] // 900)
            cv2.circle(vis, (int(b["cx"]), int(b["cy"])), int(b["r"]),
                       (255, 160, 0), th_)                      # blue = baseline
            if fit is not None:
                cx, cy, r, _, _ = fit
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), th_)
                cv2.circle(vis, (int(cx), int(cy)), int(r * 0.6), (0, 255, 0),
                           max(1, th_ - 1))
            row["_vis"] = (int(b["cx"] - b["r"]), int(b["cy"] - b["r"] - 14))
        if not args.no_viz:
            xs = [b["cx"] for b in balls]; ys = [b["cy"] for b in balls]
            rr = max(b["r"] for b in balls)
            x0 = int(max(0, min(xs) - rr * 1.9)); x1 = int(min(vis.shape[1], max(xs) + rr * 1.9))
            y0 = int(max(0, min(ys) - rr * 1.9)); y1 = int(min(vis.shape[0], max(ys) + rr * 1.9))
            t = vis[y0:y1, x0:x1]
            h = 340
            t = cv2.resize(t, (max(1, int(t.shape[1] * h / t.shape[0])), h))
            tag = ("BASE " if is_base else "") + fp.stem[:26]
            for c, w_ in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(t, tag, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, w_)
            viz.append(t)
    for r in rows:
        r.pop("_vis", None)

    # ---- calibrate thresholds on the baseline frames, then classify ----------
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    thr = {}
    for b in balls:
        br = [r for r in rows if r["ball"] == b["id"] and r["is_base"] == 1]
        ms = np.array([num(r["match"]) for r in br]); ms = ms[np.isfinite(ms)]
        rm = np.array([num(r["rms"]) for r in br]); rm = rm[np.isfinite(rm)]
        thr[b["id"]] = {
            "score": min(args.score_floor, ms.min() - 0.15) if ms.size else args.score_floor,
            "rms": max(args.rms_floor, 2.5 * rm.max()) if rm.size else args.rms_floor,
        }

    # THE FIT IS THE AUTHORITY, NOT THE MATCH SCORE.
    #
    # It is tempting to treat a high match score as proof the ball is visible and
    # so blame a failed fit on something else. That is wrong here, and the data
    # showed it: when the hand-held ball is brought up to the lens it fills the
    # frame with smooth white, and a white-ball template matches that surface at
    # 0.84-0.97 while the real ball behind it is completely hidden. A template
    # made of a featureless object matches any featureless region.
    #
    # The radial fit does not have that failure mode. It needs a circular edge at
    # roughly the right radius, which an occluder does not provide. So a failed
    # fit means NOT MEASURABLE, whatever the match score says, and that is the
    # decision the depth step actually needs. A low match score is still useful
    # on top: it confirms occlusion rather than merely implying it.
    for r in rows:
        t = thr[r["ball"]]
        sc, rm, dr = num(r["match"]), num(r["rms"]), abs(num(r["dr_pct"]))
        low_match = np.isfinite(sc) and sc < t["score"]
        if r["cx"] == "":
            r["status"] = "occluded" if low_match else "not_measurable"
        elif low_match:
            r["status"] = "occluded"
        elif (np.isfinite(rm) and rm > t["rms"]) or r["n_rays"] < args.min_rays \
                or (np.isfinite(dr) and dr > 100 * args.max_dr):
            r["status"] = "suspect_badfit"
        else:
            r["status"] = "ok"

    # Movement is measured against the mean of the per-frame BASELINE fits, not
    # against the static circle. The static circle comes from a median stack,
    # whose blurred edges bias its radius and centre; measuring against it turns
    # that constant bias into fake movement in every frame.
    ref = {}
    for b in balls:
        br = [r for r in rows if r["ball"] == b["id"] and r["is_base"] == 1
              and r["cx"] != ""]
        if br:
            ref[b["id"]] = (float(np.mean([r["cx"] for r in br])),
                            float(np.mean([r["cy"] for r in br])),
                            float(np.mean([r["r"] for r in br])))
    for r in rows:
        if r["ball"] in ref and r["cx"] != "":
            rx, ry, rr = ref[r["ball"]]
            r["move_px"] = round(float(np.hypot(r["cx"] - rx, r["cy"] - ry)), 2)
            r["move_dx"] = round(r["cx"] - rx, 2)
            r["move_dy"] = round(r["cy"] - ry, 2)
            r["dr_vs_base_pct"] = round(100 * (r["r"] - rr) / rr, 2)
        else:
            r["move_px"] = r["move_dx"] = r["move_dy"] = r["dr_vs_base_pct"] = ""

    with open(out / "per_frame_balls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out/'per_frame_balls.csv'}")

    if viz and not args.no_viz:
        wmin = min(t.shape[1] for t in viz)
        viz = [t[:, :wmin] for t in viz]
        cols = min(3, len(viz))
        grid = [np.hstack(viz[i:i + cols]) for i in range(0, len(viz), cols)]
        wmin2 = min(g.shape[1] for g in grid)
        cv2.imwrite(str(out / "per_frame_check.jpg"),
                    np.vstack([g[:, :wmin2] for g in grid]))
        print(f"wrote {out/'per_frame_check.jpg'}")

    # ---- compare -----------------------------------------------------------
    print("\n=== per ball: own-fit radius, and movement vs the BASELINE fits ===")
    for b in balls:
        br = [r for r in rows if r["ball"] == b["id"] and r["is_base"] == 1]
        orr = [r for r in rows if r["ball"] == b["id"] and r["is_base"] == 0]
        rb = np.array([num(r["r"]) for r in br]); rb = rb[np.isfinite(rb)]
        ro = np.array([num(r["r"]) for r in orr
                       if r["status"] == "ok"]); ro = ro[np.isfinite(ro)]
        mv = np.array([num(r["move_px"]) for r in orr]); mv = mv[np.isfinite(mv)]
        t = thr[b["id"]]
        print(f"\n{b['id']}  static r={b['r']:.2f}   "
              f"[thresholds: match<{t['score']:.2f}, rms>{t['rms']:.1f}]")
        if rb.size:
            print(f"  baseline  n={rb.size}  r={rb.mean():7.2f} "
                  f"+/-{(rb.std(ddof=1) if rb.size > 1 else 0):.2f}  "
                  f"({100*(rb.mean()-b['r'])/b['r']:+.1f}% vs static circle)")
        if ro.size:
            print(f"  object ok n={ro.size}  r={ro.mean():7.2f} "
                  f"+/-{(ro.std(ddof=1) if ro.size > 1 else 0):.2f}  "
                  f"({100*(ro.mean()-rb.mean())/rb.mean():+.2f}% vs baseline)")
        if mv.size:
            print(f"  movement vs baseline: median {np.median(mv):.1f}px  "
                  f"max {mv.max():.1f}px  ({100*np.median(mv)/rb.mean():.1f}% of r)")
        for r in orr + br:
            if r["status"] == "ok" and num(r["move_px"]) < 0.06 * b["r"]:
                continue
            print(f"    {'BASE ' if r['is_base'] else '     '}"
                  f"{r['frame']:<24} {r['status']:<15} "
                  f"move={r['move_px'] or '-':>7}px  r={r['r'] or 'fail':>7}  "
                  f"dr={r['dr_vs_base_pct'] or '-':>6}%  match={r['match']}  "
                  f"focus={r['focus']}  rms={r['rms'] or '-'}")

    if len(balls) > 1:
        print("\n=== is the shift common to all balls (camera) or not (balls "
              "knocked)? ===")
        for fp, _ in frames:
            st = fp.stem
            v = [(r["ball"], num(r["move_dx"]), num(r["move_dy"]))
                 for r in rows if r["frame"] == st and r["move_px"] not in ("", None)]
            if len(v) < 2:
                continue
            dx = np.array([x for _, x, _ in v]); dy = np.array([y for _, _, y in v])
            print(f"  {st:<26} " + "  ".join(f"{n}({x:+.0f},{y:+.0f})"
                                             for n, x, y in v)
                  + f"   spread {np.hypot(dx.std(), dy.std()):.1f}px")
        print("  similar vectors on every ball = the camera moved; one ball "
              "moving alone = that ball was knocked.")

    print("\nblue = static baseline circle, red = this frame's own fit, "
          "green = inner 60% depth disc.")


if __name__ == "__main__":
    main()
