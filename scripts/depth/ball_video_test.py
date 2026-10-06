"""
ball_video_test.py — measure how much a moving foreground/background object
changes DA3's depth on a FIXED reference ball.

THE DESIGN THIS ASSUMES
  Camera clamped. One to three ping-pong balls placed in the scene and never
  moved. During the clip an object is brought in and out of frame, in front of
  or behind the balls, in various positions and orientations.

  Because the balls never move, their true distance is constant and their pixel
  radius is constant. So the spread of DA3's reported depth ACROSS FRAMES is the
  context effect directly — no focal length, no scale invariant, no calibration.

  And because the object's position differs in every frame, frames are NOT
  interchangeable. Every frame is its own condition, so all of them are processed
  rather than a sample.

TWO MEASUREMENTS
  1. STABILITY   per ball, the spread of depth over the clip. This is how much
                 the moving object moved the answer.
  2. RATIO       with 2+ balls, whether the depth RATIO between them holds even
                 when the absolute values drift. Ratios are what a size proxy
                 actually needs, so this can pass while absolute scale fails.

BALL MEASUREMENT
  The balls are located once, from a pixel-wise median over evenly spaced frames.
  The moving object averages out and sensor noise with it, leaving a clean static
  scene to measure against. That single radius then applies to every frame.

OCCLUSION
  When the object passes in front of a ball those frames must be dropped, not
  averaged in. Each frame's ball patch is compared with the template from the
  median stack; a large difference means something else is there.

USAGE
  # step 1 - locate the balls (writes balls.json, and a check image to eyeball)
  python ball_video_test.py locate --frames frames/clip1 --out clip1/

  # step 2 - depth on every frame, sampled at those fixed circles
  python ball_video_test.py measure --frames frames/clip1 --depth depth/clip1 \
      --balls clip1/balls.json --out clip1/

  # step 3 - analysis
  python ball_video_test.py analyse --csv clip1/per_frame.csv
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------- locate ----
def median_stack(paths, n_stack=60):
    """Pixel-wise median over evenly spaced frames.

    The balls are static so they survive; the moving object appears in a minority
    of frames at any given pixel and is removed, as is sensor noise. The result is
    a clean image to measure the balls from.
    """
    import cv2
    idx = np.linspace(0, len(paths) - 1, min(n_stack, len(paths))).astype(int)
    stack = []
    for i in idx:
        im = cv2.imread(str(paths[i]))
        if im is not None:
            stack.append(im)
    if not stack:
        raise SystemExit("no readable frames")
    print(f"  median-stacked {len(stack)} frames")
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def radial_fit(gray, cx, cy, r_lo, r_hi, n_rays=180, min_grad=6.0):
    """Sub-pixel circle fit from radial intensity gradients.

    From an approximate centre, cast rays outward and take the strongest
    gradient along each as an edge point. Rays whose gradient is weak are
    DISCARDED — that is the point of this method: where the ball's bright side
    meets a bright wall there is no real edge, and a fit that ignores those rays
    beats one that invents them. Least-squares circle through the survivors.

    Returns (cx, cy, r, n_used, rms_residual_px) or None.
    """
    pts = []
    for a in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        rs = np.arange(r_lo * 0.6, r_hi * 1.4, 0.5)
        xs = cx + rs * np.cos(a)
        ys = cy + rs * np.sin(a)
        ok = ((xs >= 1) & (xs < gray.shape[1] - 1) &
              (ys >= 1) & (ys < gray.shape[0] - 1))
        if ok.sum() < 10:
            continue
        rs, xs, ys = rs[ok], xs[ok], ys[ok]
        vals = gray[ys.astype(int), xs.astype(int)].astype(float)
        d = np.abs(np.gradient(vals))
        j = int(np.argmax(d))
        if d[j] < min_grad or not (r_lo * 0.7 < rs[j] < r_hi * 1.3):
            continue
        pts.append((xs[j], ys[j]))
    if len(pts) < 20:
        return None
    P = np.asarray(pts)
    # algebraic circle fit: x^2+y^2 + Dx + Ey + F = 0
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    b = -(P[:, 0] ** 2 + P[:, 1] ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    ccx, ccy = -D / 2, -E / 2
    rr = np.sqrt(max(ccx ** 2 + ccy ** 2 - F, 1e-9))
    resid = np.hypot(P[:, 0] - ccx, P[:, 1] - ccy) - rr
    return float(ccx), float(ccy), float(rr), len(P), float(np.sqrt((resid ** 2).mean()))


def cmd_locate(args):
    import cv2
    paths = sorted(p for p in Path(args.frames).iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")
    print(f"{len(paths)} frames")
    med = median_stack(paths, args.n_stack)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "median.jpg"), med)

    if args.debug:
        # Sweep thresholds and report every roundish blob, so the right
        # v_min/s_max/r range can be read off instead of guessed at.
        hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        print(f"  median image {W}x{H}; sweeping thresholds")
        for vm in (120, 140, 160, 180, 200, 220):
            for sm in (40, 60, 90):
                mm = ((hsv[:, :, 2] > vm) & (hsv[:, :, 1] < sm)).astype(np.uint8)
                mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
                nn, ll, ss, cc = cv2.connectedComponentsWithStats(mm, 8)
                hits = []
                for i in range(1, nn):
                    a = ss[i, cv2.CC_STAT_AREA]
                    bw, bh = ss[i, cv2.CC_STAT_WIDTH], ss[i, cv2.CC_STAT_HEIGHT]
                    r0 = np.sqrt(a / np.pi)
                    if r0 < 10 or r0 > 400:
                        continue
                    if not (0.6 < bw / max(bh, 1) < 1.7):
                        continue
                    hits.append((r0, cc[i][0], cc[i][1], bw / max(bh, 1)))
                hits.sort(key=lambda t: -t[0])
                if hits:
                    print(f"    V>{vm:3d} S<{sm:2d}: " + "  ".join(
                        f"r={r:.0f}@({x:.0f},{y:.0f})ar{a:.2f}" for r, x, y, a in hits[:4]))
        print("  pick the V/S where the ball appears with a stable radius,")
        print("  then set --v_min/--s_max and an --r_min/--r_max around it.")
        return

    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 2] > args.v_min) & (hsv[:, :, 1] < args.s_max)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)

    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    balls, vis = [], med.copy()

    # Manual seeds: automatic blob-finding fails when the ball and background are
    # both white, but the radial fit only needs a rough starting point — it finds
    # the edge itself and ignores the directions where there is no edge to find.
    # A seed within ~20px is plenty.
    if args.seed:
        for k, spec in enumerate(args.seed):
            parts = [float(v) for v in spec.split(",")]
            if len(parts) not in (2, 3):
                raise SystemExit(f"--seed wants x,y or x,y,r  (got '{spec}')")
            sx, sy = parts[0], parts[1]
            r0 = parts[2] if len(parts) == 3 else args.seed_r
            fit = radial_fit(gray, sx, sy, r0 * 0.7, r0 * 1.4,
                             min_grad=args.min_grad)
            if fit is None:
                print(f"  seed ({sx:.0f},{sy:.0f}) r~{r0:.0f}: fit failed — try a "
                      f"closer seed, a truer --seed_r, or a lower --min_grad")
                continue
            cx, cy, r, nu, rms = fit
            balls.append({"cx": round(cx, 2), "cy": round(cy, 2), "r": round(r, 2),
                          "rays_used": nu, "rms_px": round(rms, 2)})
            print(f"  ball: r={r:.2f}px centre=({cx:.1f},{cy:.1f})  "
                  f"{nu} rays, rms {rms:.2f}px   (seed {sx:.0f},{sy:.0f})")
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 3)
            cv2.circle(vis, (int(cx), int(cy)), int(r * 0.6), (0, 255, 0), 2)
        if not balls:
            raise SystemExit("no seed produced a fit")
        balls.sort(key=lambda b: -b["r"])
        for k, b in enumerate(balls):
            b["id"] = f"ball{k+1}"
        json.dump(balls, open(out / "balls.json", "w"), indent=1)
        cv2.imwrite(str(out / "balls_check.jpg"), vis)
        print(f"\nwrote {out/'balls.json'} and balls_check.jpg")
        print("LOOK AT balls_check.jpg — a low rms only means the points fitted a")
        print("circle well, not that it is the right circle.")
        if len(balls) > 1:
            rs = [b["r"] for b in balls]
            print("radius ratios (nearest/others): "
                  + ", ".join(f"{rs[0]/x:.3f}" for x in rs[1:]))
        return

    cands = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        r0 = np.sqrt(a / np.pi)
        if not (args.r_min < r0 < args.r_max):
            continue
        if not (0.7 < bw / max(bh, 1) < 1.4):
            continue
        cands.append((a, cen[i][0], cen[i][1], r0))
    cands.sort(key=lambda t: -t[0])
    for a, cx0, cy0, r0 in cands[:args.n_balls]:
        fit = radial_fit(gray, cx0, cy0, r0 * 0.8, r0 * 1.3)
        if fit is None:
            print(f"  blob at ({cx0:.0f},{cy0:.0f}) r~{r0:.0f}: radial fit failed")
            continue
        cx, cy, r, nu, rms = fit
        balls.append({"cx": round(cx, 2), "cy": round(cy, 2), "r": round(r, 2),
                      "rays_used": nu, "rms_px": round(rms, 2)})
        print(f"  ball: r={r:.2f}px centre=({cx:.1f},{cy:.1f})  "
              f"{nu} rays, rms {rms:.2f}px")
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 0, 255), 3)
        cv2.circle(vis, (int(cx), int(cy)), int(r * 0.6), (0, 255, 0), 2)
    if not balls:
        raise SystemExit("no balls found; adjust --v_min/--s_max/--r_min/--r_max")
    balls.sort(key=lambda b: -b["r"])            # nearest first
    for k, b in enumerate(balls):
        b["id"] = f"ball{k+1}"
    json.dump(balls, open(out / "balls.json", "w"), indent=1)
    cv2.imwrite(str(out / "balls_check.jpg"), vis)
    print(f"\nwrote {out/'balls.json'} and balls_check.jpg")
    print("LOOK AT balls_check.jpg before going further — every number downstream")
    print("depends on those circles being on the balls.")
    if len(balls) > 1:
        rs = [b["r"] for b in balls]
        print(f"radius ratios (nearest/others): "
              + ", ".join(f"{rs[0]/x:.3f}" for x in rs[1:]))
        print("true distance ratios are the inverse of these.")


# --------------------------------------------------------------- measure ----
def disc(shape, cx, cy, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def cmd_measure(args):
    import cv2
    balls = json.load(open(args.balls))
    # is_file() so a baseline subfolder like frames/clipNN/default is not swept in
    frames = sorted(p for p in Path(args.frames).iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    dmap = {p.stem: p for p in Path(args.depth).glob("*.npy")}
    pairs = [(f, dmap[f.stem]) for f in frames if f.stem in dmap]
    print(f"{len(frames)} frames, {len(pairs)} with depth")
    if not pairs:
        raise SystemExit("no frame/depth stem matches")

    med = cv2.imread(str(Path(args.balls).parent / "median.jpg"))
    if med is None:
        raise SystemExit("median.jpg not found next to balls.json (rerun locate)")
    tmpl = {}
    for b in balls:
        x0, x1 = int(b["cx"] - b["r"]), int(b["cx"] + b["r"])
        y0, y1 = int(b["cy"] - b["r"]), int(b["cy"] + b["r"])
        tmpl[b["id"]] = (cv2.cvtColor(med[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                         .astype(float), (x0, x1, y0, y1))

    rows = []
    for k, (fp, dp) in enumerate(pairs):
        depth = np.load(dp).astype(np.float32)
        im = cv2.imread(str(fp))
        H, W = im.shape[:2]
        sx, sy = depth.shape[1] / W, depth.shape[0] / H
        row = {"frame": fp.stem, "is_base": 0}
        for b in balls:
            # occlusion check against the static template
            t, (x0, x1, y0, y1) = tmpl[b["id"]]
            patch = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
            occ = float(np.abs(patch - t).mean()) if patch.shape == t.shape else 999.0
            # depth over the inner disc, mapped into depth-map pixels
            m = disc(depth.shape, b["cx"] * sx, b["cy"] * sy,
                     b["r"] * 0.6 * (sx + sy) / 2)
            v = depth[m & np.isfinite(depth) & (depth > 0)]
            row[f'{b["id"]}_z'] = float(np.median(v)) if v.size >= 5 else None
            row[f'{b["id"]}_iqr'] = (float(np.percentile(v, 75) - np.percentile(v, 25))
                                     if v.size >= 5 else None)
            row[f'{b["id"]}_n'] = int(v.size)
            row[f'{b["id"]}_occ'] = round(occ, 2)
        fin = depth[np.isfinite(depth) & (depth > 0)]
        row["scene_min"] = float(np.percentile(fin, 1)) if fin.size else None
        row["scene_max"] = float(np.percentile(fin, 99)) if fin.size else None
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(pairs)}")

    # Baseline frames: the scene with nothing added. Measured identically, then
    # flagged, so the analysis can express every other frame as a deviation from
    # a CLEAN reference rather than from the clip's own contaminated median.
    n_base = 0
    if args.baseline:
        bframes = sorted(p for p in Path(args.baseline).iterdir()
                         if p.is_file() and p.suffix.lower() in IMG_EXTS)
        bpairs = [(f, dmap[f.stem]) for f in bframes if f.stem in dmap]
        missing = len(bframes) - len(bpairs)
        if missing:
            print(f"  [warn] {missing} baseline frame(s) have no depth map — "
                  f"run run_da3_depth.py on {args.baseline} too")
        for fp, dp in bpairs:
            depth = np.load(dp).astype(np.float32)
            im = cv2.imread(str(fp))
            H, W = im.shape[:2]
            sx, sy = depth.shape[1] / W, depth.shape[0] / H
            row = {"frame": fp.stem, "is_base": 1}
            for b in balls:
                m = disc(depth.shape, b["cx"] * sx, b["cy"] * sy,
                         b["r"] * 0.6 * (sx + sy) / 2)
                v = depth[m & np.isfinite(depth) & (depth > 0)]
                row[f'{b["id"]}_z'] = float(np.median(v)) if v.size >= 5 else None
                row[f'{b["id"]}_iqr'] = (float(np.percentile(v, 75) -
                                               np.percentile(v, 25))
                                         if v.size >= 5 else None)
                row[f'{b["id"]}_n'] = int(v.size)
                row[f'{b["id"]}_occ'] = 0.0        # nothing there by definition
            fin = depth[np.isfinite(depth) & (depth > 0)]
            row["scene_min"] = float(np.percentile(fin, 1)) if fin.size else None
            row["scene_max"] = float(np.percentile(fin, 99)) if fin.size else None
            rows.append(row); n_base += 1
        print(f"  + {n_base} baseline frame(s)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out/'per_frame.csv'}  ({len(rows)} rows, {n_base} baseline)")


# --------------------------------------------------------------- analyse ----
def cmd_analyse(args):
    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty csv")
    ids = sorted({k[:-2] for k in rows[0] if k.endswith("_z")})
    print(f"{len(rows)} frames, balls: {', '.join(ids)}")

    def col(name, cast=float):
        out = []
        for r in rows:
            v = r.get(name)
            out.append(cast(v) if v not in ("", "None", None) else np.nan)
        return np.array(out)

    base = (col("is_base") == 1) if "is_base" in rows[0] else np.zeros(len(rows), bool)
    occ_thr = args.occ_thresh
    keep = np.ones(len(rows), bool)
    for i in ids:
        keep &= col(f"{i}_occ") < occ_thr
    print(f"{keep.sum()}/{len(rows)} frames pass the occlusion check "
          f"(mean |diff| < {occ_thr})")

    if base.any():
        print(f"\n=== 0. BASELINE — {int(base.sum())} clean frame(s), nothing added ===")
        print("  The reference. Its own spread is the floor: noise with a static")
        print("  scene and nothing moving, so no effect smaller than this is real.")
        bref = {}
        for i in ids:
            z = col(f"{i}_z")[base]; z = z[np.isfinite(z)]
            if z.size == 0:
                continue
            bref[i] = float(np.median(z))
            sd = 100 * z.std(ddof=1) / bref[i] if z.size > 1 else 0.0
            print(f"  {i}: z_base={bref[i]:.4f}  n={z.size}  sd={sd:.2f}%")

        print("\n=== 0b. DEVIATION FROM BASELINE — what the object actually did ===")
        print("  Depth error becomes size error one-for-one (size = px * z), so")
        print("  these percentages are also how wrong a size measurement would be.")
        for i in ids:
            if i not in bref:
                continue
            z = col(f"{i}_z")
            ok = keep & ~base & np.isfinite(z)
            if ok.sum() < 5:
                continue
            dev = 100 * (z[ok] - bref[i]) / bref[i]
            worse = np.abs(dev)
            print(f"  {i}: n={ok.sum():4d}  median {np.median(dev):+6.2f}%  "
                  f"mean|dev| {worse.mean():5.2f}%  p95|dev| "
                  f"{np.percentile(worse,95):5.2f}%  worst {worse.max():5.2f}%")
        print("  median far from 0 = the object shifts depth in a consistent")
        print("  direction; mean|dev| = the typical cost of it being there.")

    print("\n=== 1. STABILITY — the ball never moved, so this spread is the "
          "context effect ===")
    for i in ids:
        z = col(f"{i}_z")[keep & ~base]
        z = z[np.isfinite(z)]
        if z.size < 5:
            print(f"  {i}: too few clean frames"); continue
        print(f"  {i}: n={z.size:4d}  median={np.median(z):.4f}  "
              f"sd={z.std(ddof=1):.4f} ({100*z.std(ddof=1)/np.median(z):.2f}%)  "
              f"p5-p95 {np.percentile(z,5):.4f}-{np.percentile(z,95):.4f}  "
              f"full range {100*(z.max()-z.min())/np.median(z):.1f}%")
    print("  sd is the typical effect; full range is the worst case.")

    if len(ids) > 1:
        print("\n=== 2. RATIO — does relative depth survive even if absolute "
              "does not? ===")
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                za, zb = col(f"{ids[a]}_z"), col(f"{ids[b]}_z")
                ok = keep & np.isfinite(za) & np.isfinite(zb)
                if ok.sum() < 5:
                    continue
                ratio = za[ok] / zb[ok]
                print(f"  {ids[a]}/{ids[b]}: median={np.median(ratio):.4f}  "
                      f"sd={ratio.std(ddof=1):.4f} "
                      f"({100*ratio.std(ddof=1)/np.median(ratio):.2f}%)  n={ok.sum()}")
        print("  If the ratio sd is much smaller than the individual sds, the")
        print("  absolute scale wanders but relative depth holds — which is all")
        print("  a size proxy needs, and it needs only one calibration constant.")

    # ---- size proxy -------------------------------------------------------
    if args.balls:
        bl = {b["id"]: b for b in json.load(open(args.balls))}
        print("\n=== 2b. SIZE — every ball is the same 40mm sphere, so the "
              "proxy r*z should be IDENTICAL for all of them ===")
        print("  This is the size error a real measurement would inherit: two")
        print("  objects of equal true size, read as different because they sit")
        print("  at different distances.")
        prox = {}
        for i in ids:
            if i not in bl:
                continue
            z = col(f"{i}_z")[keep]
            z = z[np.isfinite(z)]
            if z.size < 5:
                continue
            r = bl[i]["r"]
            p = r * z
            prox[i] = p
            print(f"  {i}: r={r:7.2f}px  z_med={np.median(z):.4f}  "
                  f"r*z = {np.median(p):7.2f}  (sd {100*p.std(ddof=1)/np.median(p):.2f}%)")
        if len(prox) > 1:
            meds = np.array([np.median(v) for v in prox.values()])
            names = list(prox)
            print(f"\n  spread across balls: "
                  f"{100*(meds.max()-meds.min())/meds.mean():.1f}%  "
                  f"(would be 0% if depth were exact)")
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    e = 100 * (meds[a] - meds[b]) / meds[b]
                    sep = 100 * abs(bl[names[a]]["r"] - bl[names[b]]["r"]) / max(
                        bl[names[a]]["r"], bl[names[b]]["r"])
                    print(f"    {names[a]} vs {names[b]}: size differs {e:+.1f}%  "
                          f"(they are {sep:.0f}% apart in apparent size)")
            print("  A near object reading LARGER than an identical far one means")
            print("  depth is compressed: the far one is placed too close, so its")
            print("  size is under-estimated.")

    print("\n=== 3. WORST FRAMES — where the object hurt most ===")
    for i in ids:
        z = col(f"{i}_z"); m = np.median(z[keep & np.isfinite(z)])
        dev = np.abs(z - m) / m
        idx = np.argsort(-np.where(np.isfinite(dev) & keep, dev, -1))[:args.top]
        print(f"  {i} (median {m:.4f}):")
        for j in idx:
            if not np.isfinite(dev[j]):
                continue
            print(f"    {rows[j]['frame']:<32} z={z[j]:.4f} "
                  f"({100*(z[j]-m)/m:+.1f}%)  occ={rows[j][f'{i}_occ']}")
        print("  open those frames — what the object was doing there is the answer")
        print("  to 'when does it matter'.")


def cmd_check(args):
    """Overlay the circles on a spread of REAL frames, not the median.

    The circles were placed on a median image. This confirms they still land on
    the balls in actual frames — which they will only if the camera truly held
    still for the whole clip. Any drift shows up here as circles creeping off the
    balls across the sequence, and that would invalidate the clip.
    """
    import cv2
    balls = json.load(open(args.balls))
    frames = sorted(p for p in Path(args.frames).iterdir()
                    if p.suffix.lower() in IMG_EXTS)
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")
    idx = np.linspace(0, len(frames) - 1, min(args.n, len(frames))).astype(int)
    print(f"{len(frames)} frames; checking {len(idx)} spread across the clip")

    med_p = Path(args.balls).parent / "median.jpg"
    med = cv2.imread(str(med_p)) if med_p.exists() else None

    tiles = []
    for k in idx:
        im = cv2.imread(str(frames[k]))
        if im is None:
            continue
        for b in balls:
            cx, cy, r = b["cx"], b["cy"], b["r"]
            th = max(2, im.shape[1] // 900)
            cv2.circle(im, (int(cx), int(cy)), int(r), (0, 0, 255), th)
            cv2.circle(im, (int(cx), int(cy)), int(r * 0.6), (0, 255, 0), max(1, th - 1))
            if med is not None:
                # how different is this frame's ball patch from the static template?
                x0, x1 = int(cx - r), int(cx + r)
                y0, y1 = int(cy - r), int(cy + r)
                a = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
                t = cv2.cvtColor(med[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float)
                d = float(np.abs(a - t).mean()) if a.shape == t.shape else -1
                fs = max(0.5, r / 90.0)
                cv2.putText(im, f"{b['id']} d={d:.0f}", (int(cx - r), int(cy - r - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), int(fs * 4) + 2)
                cv2.putText(im, f"{b['id']} d={d:.0f}", (int(cx - r), int(cy - r - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(fs)))
        # crop to the region containing all the balls, so the tile is legible
        xs = [b["cx"] for b in balls]; ys = [b["cy"] for b in balls]
        rr = max(b["r"] for b in balls)
        x0 = int(max(0, min(xs) - rr * 2.2)); x1 = int(min(im.shape[1], max(xs) + rr * 2.2))
        y0 = int(max(0, min(ys) - rr * 2.2)); y1 = int(min(im.shape[0], max(ys) + rr * 2.2))
        t = im[y0:y1, x0:x1]
        h = 320
        t = cv2.resize(t, (max(1, int(t.shape[1] * h / t.shape[0])), h))
        cv2.putText(t, frames[k].stem[:22], (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(t, frames[k].stem[:22], (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        tiles.append(t)

    w = min(t.shape[1] for t in tiles)
    tiles = [t[:, :w] for t in tiles]
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    w2 = min(r.shape[1] for r in rows)
    grid = np.vstack([r[:, :w2] for r in rows])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "frame_check.jpg"), grid)
    print(f"wrote {out/'frame_check.jpg'}")
    print("Every circle should sit on its ball in EVERY tile. If they drift across")
    print("the sequence the camera moved and the clip cannot be used as-is.")
    print("d = mean |frame - median| in the patch: small when the ball is clear,")
    print("large when the object is passing in front of it.")


def cmd_frames(args):
    """Rank every frame by how wrong the ball SIZE comes out versus baseline.

    Size is what matters, so everything is expressed as size error, not depth.
    Because the ball never moves its pixel radius is fixed, so size = r*z and a
    depth error is a size error one-for-one. The baseline frames give the value a
    perfect measurement would return.

    The output is a list of frame names to open and look at, sorted worst-first,
    so the question 'what makes it fail' is answered by inspection rather than
    inference.
    """
    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("empty csv")
    ids = sorted({k[:-2] for k in rows[0] if k.endswith("_z")})
    balls = {b["id"]: b for b in json.load(open(args.balls))}

    def num(r, k):
        v = r.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    base = [r for r in rows if r.get("is_base") == "1"]
    rest = [r for r in rows if r.get("is_base") != "1"]
    if not base:
        raise SystemExit("no baseline rows — rerun measure with --baseline")

    ref = {}
    for i in ids:
        v = np.array([num(r, f"{i}_z") for r in base])
        v = v[np.isfinite(v)]
        if v.size:
            ref[i] = balls[i]["r"] * float(np.median(v))
    print(f"baseline from {len(base)} frame(s); reference size per ball:")
    for i, s_ in ref.items():
        print(f"  {i}: r={balls[i]['r']:.2f}px  size={s_:.2f}")

    recs = []
    for r in rest:
        e, per = [], {}
        for i in ids:
            z = num(r, f"{i}_z")
            if not np.isfinite(z) or i not in ref:
                continue
            err = 100 * (balls[i]["r"] * z - ref[i]) / ref[i]
            per[i] = err
            e.append(abs(err))
        if not e:
            continue
        recs.append({"frame": r["frame"], "worst": max(e), "mean": float(np.mean(e)),
                     "per": per,
                     "occ": max(num(r, f"{i}_occ") for i in ids)})
    recs.sort(key=lambda t: -t["worst"])

    n = len(recs)
    w = np.array([x["worst"] for x in recs])
    print(f"\n{n} non-baseline frames")
    print(f"size error: median {np.median(w):.1f}%  p90 {np.percentile(w,90):.1f}%  "
          f"max {w.max():.1f}%")
    for thr in (5, 10, 15, 20, 30):
        k = int((w >= thr).sum())
        print(f"  >= {thr:2d}% error: {k:4d} frames ({100*k/n:.1f}%)")

    def show(title, sel):
        print(f"\n{title}")
        hdr = "  " + f"{'frame':<34}{'worst':>8}" + "".join(f"{i:>10}" for i in ids) + f"{'occ':>7}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for x in sel:
            line = f"  {x['frame']:<34}{x['worst']:>7.1f}%"
            for i in ids:
                line += f"{x['per'].get(i, float('nan')):>+10.1f}"
            line += f"{x['occ']:>7.1f}"
            print(line)

    show(f"WORST {args.top} — open these first", recs[:args.top])
    show(f"BEST {args.top} — these worked, useful as a contrast", recs[-args.top:][::-1])

    if args.out:
        with open(args.out, "w", newline="") as f:
            cols = ["frame", "worst_pct", "mean_pct", "max_occ"] + [f"{i}_pct" for i in ids]
            wtr = csv.writer(f); wtr.writerow(cols)
            for x in recs:
                wtr.writerow([x["frame"], round(x["worst"], 2), round(x["mean"], 2),
                              round(x["occ"], 2)] +
                             [round(x["per"].get(i, float("nan")), 2) for i in ids])
        print(f"\nwrote {args.out} — every frame, worst-first")
    if args.copy_worst:
        import shutil
        d = Path(args.copy_worst); d.mkdir(parents=True, exist_ok=True)
        src = Path(args.frames) if args.frames else None
        if src is None:
            print("--copy_worst needs --frames")
        else:
            k = 0
            for rank, x in enumerate(recs[:args.top]):
                for p in src.glob(x["frame"] + ".*"):
                    shutil.copy2(p, d / f"{rank:02d}_{x['worst']:.0f}pct_{p.name}")
                    k += 1
            print(f"copied {k} worst frame(s) -> {d}")


def cmd_distance(args):
    """True camera-to-ball distance from the known 40mm ball, and DA3's error.

    Pinhole:  Z_true = f_px * D_mm / (2 * r_px)
    with D = 40mm for a regulation ping-pong ball.

    The focal length can come from three places, in order of preference:
      --solve_f   Two or more balls at different depths over-determine the
                  system: their radii fix the RATIO of their true distances, and
                  DA3's depths must follow the same ratio if it were exact. The
                  f that best reconciles the two is fitted here, so no camera
                  spec is needed. Only usable when the balls really are at
                  different distances.
      --known     One ball at a distance you measured with a tape. Exact.
      --focal_px  A spec-sheet value. Fine, but it is an estimate.

    With true distances in hand, DA3's compression can be fitted directly as
        z = a * Z^gamma
    and gamma is the exponent that would invert it.
    """
    rows = list(csv.DictReader(open(args.csv)))
    balls = {b["id"]: b for b in json.load(open(args.balls))}
    ids = sorted(balls)
    base = [r for r in rows if r.get("is_base") == "1"]
    if not base:
        raise SystemExit("no baseline rows — rerun measure with --baseline")

    def num(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return float("nan")

    z = {}
    for i in ids:
        v = np.array([num(r, f"{i}_z") for r in base])
        v = v[np.isfinite(v)]
        if v.size:
            z[i] = float(np.median(v))
    ids = [i for i in ids if i in z]
    r = {i: balls[i]["r"] for i in ids}
    D = args.diameter_mm

    print(f"baseline: {len(base)} frame(s); ball diameter {D} mm")
    print(f"{'ball':<7}{'r px':>9}{'z_da3':>9}")
    for i in ids:
        print(f"{i:<7}{r[i]:>9.2f}{z[i]:>9.4f}")

    f_px, how = None, ""
    if args.known:
        bid, Zmm = args.known.split(",")
        Zmm = float(Zmm)
        if bid not in r:
            raise SystemExit(f"--known names {bid}, which is not in balls.json")
        f_px = 2 * r[bid] * Zmm / D
        how = f"from --known {bid} at {Zmm:.0f}mm"
    elif args.solve_f and len(ids) > 1:
        # True distances are Z_i = f*D/(2*r_i), so their RATIOS are fixed by the
        # radii alone and f only sets the overall scale. f therefore cannot be
        # recovered from radii alone — it needs one absolute anchor. What CAN be
        # done without one is check whether DA3's depths are consistent with the
        # radii, which is the gamma fit below.
        print("\n[note] --solve_f cannot work from radii alone: the radii fix the")
        print("       distance RATIOS but not the scale, so f is unconstrained.")
        print("       Measure one ball with a tape and pass --known, or accept an")
        print("       estimated --focal_px. The gamma fit below needs neither.")
    if f_px is None and args.focal_px:
        f_px = args.focal_px
        how = f"from --focal_px (estimate)"

    if f_px:
        print(f"\nfocal length {f_px:.1f} px  ({how})")
        print(f"{'ball':<7}{'r px':>9}{'Z_true mm':>12}{'z_da3':>9}{'z/Z':>9}")
        for i in ids:
            Z = f_px * D / (2 * r[i])
            print(f"{i:<7}{r[i]:>9.2f}{Z:>12.1f}{z[i]:>9.4f}{z[i]/(Z/1000):>9.4f}")
        print("  z/Z constant across balls would mean DA3 is right up to one scale")
        print("  factor; it drifting with distance is the compression.")
    else:
        print("\nno focal length given, so no absolute distances — the relative")
        print("analysis below needs none.")

    if len(ids) > 1:
        print("\n=== relative distances, from the radii (no focal length needed) ===")
        near = min(ids, key=lambda i: -r[i])
        print(f"{'ball':<7}{'Z/Z_near':>10}{'z/z_near':>10}{'error':>9}")
        for i in ids:
            tr = r[near] / r[i]
            dr = z[i] / z[near]
            print(f"{i:<7}{tr:>10.4f}{dr:>10.4f}{100*(dr-tr)/tr:>+8.1f}%")
        print("  DA3 under-reports how much farther the far balls are: that is")
        print("  the compression, and it is what breaks size at range.")

        # gamma: z = a * Z^gamma. Z is known up to a scale, which only shifts a,
        # so gamma is recoverable without any focal length at all.
        Zrel = np.array([1.0 / r[i] for i in ids])
        zz = np.array([z[i] for i in ids])
        g, la = np.polyfit(np.log(Zrel), np.log(zz), 1)
        print(f"\n=== compression fit  z = a * Z^gamma ===")
        print(f"  gamma = {g:.3f}   (1.0 would be no compression)")
        print(f"  a true 2x depth difference is reported as "
              f"{2**g:.2f}x")
        if g < 0.95:
            print(f"  to invert: Z_corrected proportional to z^(1/gamma) = z^{1/g:.2f}")
            print("  fit gamma on several clips before trusting it — it has varied.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("locate", help="find the balls from a median stack")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_balls", type=int, default=3)
    p.add_argument("--n_stack", type=int, default=60)
    p.add_argument("--v_min", type=float, default=170, help="brightness threshold")
    p.add_argument("--s_max", type=float, default=60, help="saturation ceiling")
    p.add_argument("--r_min", type=float, default=15)
    p.add_argument("--r_max", type=float, default=300)
    p.add_argument("--debug", action="store_true",
                   help="sweep thresholds and list candidate blobs, then stop")
    p.add_argument("--seed", action="append",
                   help="manual seed 'x,y' or 'x,y,r' on the MEDIAN image; "
                        "repeat per ball. Within ~20px is close enough — the "
                        "radial fit refines it. Use when the ball and background "
                        "are both white and blob detection cannot separate them.")
    p.add_argument("--seed_r", type=float, default=80,
                   help="approximate radius when a seed gives only x,y")
    p.add_argument("--min_grad", type=float, default=6.0,
                   help="minimum edge gradient for a ray to count; lower it for "
                        "low-contrast edges")
    p.set_defaults(func=cmd_locate)

    p = sub.add_parser("measure", help="sample depth at the fixed circles")
    p.add_argument("--frames", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--balls", required=True)
    p.add_argument("--baseline", help="folder of clean frames (nothing added), "
                                      "measured and flagged as the reference")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("check", help="overlay the circles on real frames")
    p.add_argument("--frames", required=True)
    p.add_argument("--balls", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10, help="frames to check")
    p.add_argument("--cols", type=int, default=5)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("frames", help="rank every frame by ball-size error")
    p.add_argument("--csv", required=True)
    p.add_argument("--balls", required=True)
    p.add_argument("--out", help="write the full ranked list to this CSV")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--frames", help="frame folder, for --copy_worst")
    p.add_argument("--copy_worst", help="copy the worst frames here for viewing")
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("distance", help="true distance from the 40mm ball")
    p.add_argument("--csv", required=True)
    p.add_argument("--balls", required=True)
    p.add_argument("--diameter_mm", type=float, default=40.0)
    p.add_argument("--focal_px", type=float, help="estimated focal length")
    p.add_argument("--known", help="'ballN,Zmm' — one ball at a tape-measured "
                                   "distance, which gives f exactly")
    p.add_argument("--solve_f", action="store_true")
    p.set_defaults(func=cmd_distance)

    p = sub.add_parser("analyse")
    p.add_argument("--csv", required=True)
    p.add_argument("--occ_thresh", type=float, default=25.0,
                   help="mean |frame - template| above this = ball occluded")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--balls", help="balls.json, to also report the size proxy r*z")
    p.set_defaults(func=cmd_analyse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
