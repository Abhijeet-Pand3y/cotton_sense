"""
build_combined_csv.py — one table holding everything measured about every ball.

TWO FILES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

  combined_ball_summary.csv   one row per (set, ball). Every quantity appears
                              twice, as base_* and obj_*, so the clean scene and
                              the scene with a distractor in it sit side by side
                              and can be differenced directly.

  combined_ball_frames.csv    one row per (set, ball, frame). The same
                              quantities before any averaging, plus which object
                              was in that frame and how far away it was. Use
                              this to break the summary down by object distance.

WHAT "DISTANCE" MEANS IN HERE - THREE OF THEM, KEPT SEPARATE
  tape_*      what was measured with a tape. The ground truth.
  geom_*      from the ball's pixel radius and the solved focal length,
              Z = f*D/(2r) - c. Uses no depth model at all, so it is the
              independent check on the tape.
  da3_*       from DA3 depth inverted through the power law,
              Z = (z/a)^(1/gamma), with a fitted per frame.

  Keeping them in separate columns is the point: da3 error is only meaningful
  against tape, and geom is what says whether tape itself can be trusted.

WHERE A CELL IS LEFT EMPTY
  da3_* needs a scale a, and a is fitted per frame from the balls in that frame.
  With only one ball in the frame there is nothing to fit it against, so the
  answer would be forced onto the truth. Those cells are left empty rather than
  filled with a zero error that means nothing. da3_determinable records why.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MM_PER_IN = 25.4
BALL_MM = 40.0
SRC = {"GX010117": "GX010117.MP4", "GX010118": "GX010118.MP4",
       "GX010119": "GX010119.MP4"}


def meta_rows(sd):
    p = sd / "metadata.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def ball_distances(mrows):
    if not mrows:
        return {}
    cols = [c for c in mrows[0] if c.startswith("ball") and c.endswith("_distance_in")]
    cols.sort(key=lambda c: int(c[4:-12]) if c[4:-12].isdigit() else 0)
    out = {}
    for i, c in enumerate(cols, 1):
        v = {r[c] for r in mrows if r.get(c)}
        if len(v) == 1:
            out[f"ball{i}"] = float(v.pop()) * MM_PER_IN
    return out


def main():
    frames_root = Path("frames")
    focal = json.loads(Path("findings/focal.json").read_text()) \
        if Path("findings/focal.json").exists() else None

    # ---------- gather every per-frame measurement ------------------------
    recs = []
    for sd in sorted(frames_root.glob("GX0101*_test")):
        pf = sd / "locate_out" / "per_frame_balls.csv"
        dd = sd / "depth" / "depth"
        if not pf.exists():
            continue
        mrows = meta_rows(sd)
        dist = ball_distances(mrows)
        objd = {r["filename"].rsplit(".", 1)[0]: r.get("object_distance_in", "")
                for r in mrows}
        hand = {r["filename"].rsplit(".", 1)[0]:
                r.get("handheld_ball_distance_in", "") for r in mrows}
        import cv2
        wh, dep = {}, {}
        for r in csv.DictReader(open(pf)):
            fr, b = r["frame"], r["ball"]
            npy = dd / f"{fr}.npy"
            z = ""
            if r["status"] == "ok" and r["cx"] != "" and npy.exists():
                if fr not in wh:
                    p = sd / f"{fr}.jpg"
                    if not p.exists():
                        p = sd / "baseline" / f"{fr}.jpg"
                    im = cv2.imread(str(p))
                    wh[fr] = (im.shape[1], im.shape[0])
                    dep[fr] = np.load(npy).astype(np.float32)
                d = dep[fr]
                w, h = wh[fr]
                sx, sy = d.shape[1] / w, d.shape[0] / h
                cx, cy, rr = float(r["cx"]), float(r["cy"]), float(r["r"])
                dr = rr * 0.6 * (sx + sy) / 2
                yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
                m = ((xx - cx * sx) ** 2 + (yy - cy * sy) ** 2) <= dr * dr
                v = d[m & np.isfinite(d) & (d > 0)]
                z = float(np.median(v)) if v.size >= 4 else ""
            recs.append({
                "set": sd.name, "video": SRC.get(sd.name.split("_")[0], ""),
                "frame": fr, "is_base": int(r["is_base"]), "ball": b,
                "status": r["status"],
                "tape_mm": dist.get(b), "r_px": r["r"] or "",
                "z": z, "match": r["match"], "move_px": r["move_px"],
                "object_distance_in": objd.get(fr, ""),
                "handheld_ball_distance_in": hand.get(fr, ""),
            })

    # ---------- focal length, solved from every ball ----------------------
    rp = np.array([float(x["r_px"]) for x in recs if x["r_px"] and x["tape_mm"]])
    Zt = np.array([x["tape_mm"] for x in recs if x["r_px"] and x["tape_mm"]])
    sl, ic = np.polyfit(Zt, 1 / rp, 1)
    f_px, c_mm = 2 / (sl * BALL_MM), ic / sl

    # ---------- gamma per frame, then per set -----------------------------
    byframe = defaultdict(list)
    for x in recs:
        if x["z"] != "" and x["tape_mm"]:
            byframe[(x["set"], x["frame"])].append(x)
    fg = {}
    for k, rr in byframe.items():
        Z = np.array([x["tape_mm"] for x in rr], float)
        z = np.array([x["z"] for x in rr], float)
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        g, la = np.polyfit(np.log(Z), np.log(z), 1)
        fg[k] = (float(g), float(np.exp(la)))
    setg = {}
    for s in {k[0] for k in fg}:
        setg[s] = float(np.median([fg[k][0] for k in fg if k[0] == s]))
    gmed = float(np.median(list(setg.values())))
    cross = {s: float(np.median([v for t, v in setg.items() if t != s]))
             for s in setg}

    # ---------- per-frame derived distances -------------------------------
    for x in recs:
        x["ball_diameter_mm"] = BALL_MM
        x["ball_diameter_px"] = round(2 * float(x["r_px"]), 2) if x["r_px"] else ""
        x["tape_in"] = round(x["tape_mm"] / MM_PER_IN, 1) if x["tape_mm"] else ""
        x["geom_mm"] = (round(f_px * BALL_MM / (2 * float(x["r_px"])) - c_mm, 1)
                        if x["r_px"] else "")
        x["geom_err_pct"] = (round(100 * (x["geom_mm"] - x["tape_mm"]) / x["tape_mm"], 2)
                             if x["geom_mm"] != "" and x["tape_mm"] else "")
        k = (x["set"], x["frame"])
        g_use = setg.get(x["set"])
        if x["z"] != "" and x["tape_mm"] and k in fg and g_use:
            rr = byframe[k]
            Z = np.array([y["tape_mm"] for y in rr], float)
            z = np.array([y["z"] for y in rr], float)
            a = float(np.exp(np.mean(np.log(z) - g_use * np.log(Z))))
            x["da3_mm"] = round((float(x["z"]) / a) ** (1 / g_use), 1)
            x["da3_err_pct"] = round(100 * (x["da3_mm"] - x["tape_mm"]) / x["tape_mm"], 2)
            ac = float(np.exp(np.mean(np.log(z) - cross[x["set"]] * np.log(Z))))
            x["da3_cross_mm"] = round((float(x["z"]) / ac) ** (1 / cross[x["set"]]), 1)
            x["da3_cross_err_pct"] = round(
                100 * (x["da3_cross_mm"] - x["tape_mm"]) / x["tape_mm"], 2)
            x["da3_determinable"] = "yes"
        else:
            x["da3_mm"] = x["da3_err_pct"] = ""
            x["da3_cross_mm"] = x["da3_cross_err_pct"] = ""
            x["da3_determinable"] = ("no: frame has one distance only"
                                     if x["z"] != "" else "no: not measurable")
        x["gamma_set"] = round(setg[x["set"]], 3) if x["set"] in setg else ""
        x["gamma_cross"] = round(cross[x["set"]], 3) if x["set"] in cross else ""

    Path("findings").mkdir(exist_ok=True)
    cols = ["set", "video", "frame", "is_base", "ball", "status",
            "ball_diameter_mm", "ball_diameter_px", "r_px",
            "tape_in", "tape_mm", "geom_mm", "geom_err_pct",
            "z", "da3_mm", "da3_err_pct", "da3_cross_mm", "da3_cross_err_pct",
            "da3_determinable", "gamma_set", "gamma_cross",
            "object_distance_in", "handheld_ball_distance_in",
            "match", "move_px"]
    with open("findings/combined_ball_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda x: (x["set"], x["ball"],
                                                -x["is_base"], x["frame"])))

    # ---------- summary: base beside object -------------------------------
    grp = defaultdict(list)
    for x in recs:
        grp[(x["set"], x["ball"])].append(x)

    def agg(rows, key):
        v = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return (round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)
                if len(v) > 1 else 0.0, len(v)) if v else ("", "", 0)

    out = []
    for (s, b), rows in sorted(grp.items()):
        base = [r for r in rows if r["is_base"] == 1 and r["status"] == "ok"]
        obj = [r for r in rows if r["is_base"] == 0 and r["status"] == "ok"]
        objall = [r for r in rows if r["is_base"] == 0]
        d = {"set": s, "video": rows[0]["video"], "ball": b,
             "ball_diameter_mm": BALL_MM,
             "tape_in": rows[0]["tape_in"], "tape_mm": rows[0]["tape_mm"],
             "object_distances_in": ";".join(sorted(
                 {r["object_distance_in"] for r in objall
                  if r["object_distance_in"]}, key=lambda v: float(v))),
             "n_base_frames": len(base), "n_obj_frames": len(obj),
             "n_obj_dropped": len(objall) - len(obj),
             "dropped_reasons": ";".join(sorted(
                 {r["status"] for r in objall if r["status"] != "ok"})),
             "gamma_set": rows[0]["gamma_set"],
             "gamma_cross": rows[0]["gamma_cross"],
             "focal_px": round(f_px, 1), "tape_offset_mm": round(c_mm, 1)}
        for tag, rr in (("base", base), ("obj", obj)):
            for key, name in (("ball_diameter_px", "ball_diameter_px"),
                              ("r_px", "radius_px"), ("z", "da3_z"),
                              ("geom_mm", "geom_mm"),
                              ("geom_err_pct", "geom_err_pct"),
                              ("da3_mm", "da3_mm"),
                              ("da3_err_pct", "da3_err_pct"),
                              ("da3_cross_mm", "da3_cross_mm"),
                              ("da3_cross_err_pct", "da3_cross_err_pct")):
                m, sdv, n = agg(rr, key)
                d[f"{tag}_{name}"] = m
                if name in ("radius_px", "da3_z"):
                    d[f"{tag}_{name}_sd"] = sdv
        for name in ("radius_px", "da3_z", "da3_err_pct"):
            a_, b_ = d.get(f"base_{name}", ""), d.get(f"obj_{name}", "")
            d[f"delta_{name}_pct"] = (round(100 * (b_ - a_) / a_, 2)
                                      if a_ not in ("", 0) and b_ != "" else "")
        out.append(d)

    with open("findings/combined_ball_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    print(f"wrote findings/combined_ball_summary.csv  ({len(out)} rows, "
          f"{len(out[0])} columns)")
    print(f"wrote findings/combined_ball_frames.csv   ({len(recs)} rows)")
    print(f"focal {f_px:.0f} px, tape offset {c_mm:+.1f} mm, "
          f"gamma median across sets {gmed:.3f}")


if __name__ == "__main__":
    main()
