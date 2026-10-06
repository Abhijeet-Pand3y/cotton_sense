"""plot_noref_set.py — figure for a no-tape set (GX010120), for findings/team_share/.

Three panels, in the order the argument runs:
  A  what the method returns for a ball that never changed
  B  how that compares with the established no-reference baseline
  C  where the variation comes from — pixels or depth
"""
import csv
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1", "#fcfcfb"
BALL = 40.0

SET = sys.argv[1] if len(sys.argv) > 1 else "frames/GX010120_test"
OUT = sys.argv[2] if len(sys.argv) > 2 else "findings/team_share/noref_new_data.png"

rows = list(csv.DictReader(open(f"{SET}/noref_analysis.csv")))
rows.sort(key=lambda r: (r['object_distance_in'] != '', float(r['object_distance_in'] or 0)))
D = np.array([float(r["noref_size_mm"]) for r in rows])
z = np.array([float(r["z"]) for r in rows])
rpx = np.array([float(r["r_px"]) for r in rows])
isbase = np.array([r["scene"] == "ball only" for r in rows])
lab = ["ball\nalone" if b else f'bottle\n{r["object_distance_in"]}"'
       for b, r in zip(isbase, rows)]

old = np.abs([float(r["setup1_size_error_pct"])
              for r in csv.DictReader(open("findings/team_share/ball_measurements.csv"))])
new = np.abs([float(r["noref_size_error_pct"]) for r in rows])

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(True, color=GRID, lw=.8, zorder=0)
    a.set_axisbelow(True)

# ---- A: measured size, frame by frame -------------------------------------
x = np.arange(len(D))
ax[0].bar(x[isbase], D[isbase], .62, color=MUTED, zorder=3, edgecolor=SURF,
          label="ball alone (base)")
ax[0].bar(x[~isbase], D[~isbase], .62, color=S[0], zorder=3, edgecolor=SURF,
          label="bottle in shot")
ax[0].axhline(BALL, color=INK, lw=2, ls="--", zorder=4)
ax[0].annotate("true size 40 mm", (-.45, BALL), textcoords="offset points",
               xytext=(0, 6), ha="left", fontsize=9, color=INK)
for i, v in enumerate(D):
    ax[0].annotate(f"{v:.1f}", (i, v), textcoords="offset points", xytext=(0, 4),
                   ha="center", fontsize=8.5, color=INK)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=8, color=INK2)
ax[0].set_ylim(0, 56)
ax[0].set_ylabel("measured size of the ball (mm)", color=INK2, fontsize=10)
ax[0].set_title("A · the ball never changed — the answer did\n"
                f"spread {D.min():.1f}–{D.max():.1f} mm on one 40 mm sphere",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

# ---- B: new data vs the established no-reference baseline ------------------
means = [old.mean(), new.mean()]
ses = [old.std(ddof=1) / np.sqrt(len(old)), new.std(ddof=1) / np.sqrt(len(new))]
names = [f"established baseline\n{len(old)} balls, 10 recordings",
         f"this new recording\n{len(new)} frames, 1 ball"]
for i, (m, se, cl) in enumerate(zip(means, ses, (MUTED, S[1]))):
    ax[1].bar(i, m, .55, color=cl, zorder=3, edgecolor=SURF)
    ax[1].errorbar(i, m, yerr=1.96 * se, color=INK, lw=1.6, capsize=6, zorder=5)
    ax[1].annotate(f"{m:.1f}%\n({BALL*m/100:.1f} mm)", (i, m + 1.96 * se),
                   textcoords="offset points", xytext=(0, 8), ha="center",
                   fontsize=10, color=INK)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(names, fontsize=8.5, color=INK2)
ax[1].set_ylim(0, 26)
ax[1].set_ylabel("size error, nothing of known size (%)", color=INK2, fontsize=10)
ax[1].set_title("B · new data is not worse, and not proven better\n"
                "bars are means, whiskers 95% CI — they overlap (p = 0.07)",
                color=INK, fontsize=11.5, loc="left", pad=12)

# ---- C: where the variation lives ------------------------------------------
pc = lambda v: 100 * (v - v.mean()) / v.mean()
ax[2].plot(x, pc(rpx), "-o", ms=6, lw=2, color=S[2], mec=SURF, mew=1.4, zorder=4,
           label=f"ball width in pixels  (±{pc(rpx).max():.1f}%)")
ax[2].plot(x, pc(z), "-o", ms=6, lw=2, color=S[1], mec=SURF, mew=1.4, zorder=3,
           label=f"depth model output  (±{pc(z).max():.1f}%)")
ax[2].axhline(0, color=INK, lw=1.2, ls="--", zorder=2)
ax[2].set_xticks(x); ax[2].set_xticklabels(lab, fontsize=8, color=INK2)
ax[2].set_ylabel("deviation from this set's mean (%)", color=INK2, fontsize=10)
ax[2].set_title("C · the pixels are steady; the depth is not\n"
                "all of the error in panel A enters through depth",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor=SURF)
print("wrote", OUT)
