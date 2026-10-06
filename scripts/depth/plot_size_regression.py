"""plot_size_regression.py — figure for findings/size_from_depth.md.

A  which mapping wins, under each calibration regime
B  measured ball size against true distance, versus the true 40 mm
C  the distribution of size estimates, global vs one reference object
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_regression import load, solve_pinhole, loso, MODELS   # noqa: E402

BALL_MM = 40.0
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"
SURFACE = "#fcfcfb"

rows = load()
f_px, c_mm = solve_pinhole(rows)
for x in rows:
    x["Zp"] = x["Ztape"] + c_mm

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=SURFACE)
for a in ax:
    a.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(True, color=GRID, lw=0.8, zorder=0)
    a.set_axisbelow(True)

# ---- A --------------------------------------------------------------------
# 'inverse' is left out: it reaches 814 % under one-reference scaling and would
# flatten every other bar. It is reported in the text instead.
shown = [m for m in MODELS if m != "inverse"]
modes = [("global", "no reference\n(scene unknown)"),
         ("ref1", "one known-size\nobject in frame"),
         ("lsq", "perfect scale\n(ceiling)")]
w = 0.2
for i, m in enumerate(shown):
    vals = [np.mean([abs(v) for _, v in loso(rows, m, md)]) for md, _ in modes]
    xs = np.arange(len(modes)) + (i - 1.5) * w
    ax[0].bar(xs, vals, w * 0.88, color=SERIES[i], zorder=3,
              label=m, edgecolor=SURFACE, linewidth=1.2)
    for x_, v in zip(xs, vals):
        ax[0].annotate(f"{v:.1f}", (x_, v), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=7.5, color=INK2)
ax[0].set_xticks(np.arange(len(modes)))
ax[0].set_xticklabels([t for _, t in modes], fontsize=9)
ax[0].set_ylabel("mean |distance error| (%)   lower is better",
                 color=INK2, fontsize=10)
ax[0].set_title("A · Which mapping, and how much a reference helps\n"
                "leave-one-set-out, never in-sample",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2)
ax[0].grid(axis="x", visible=False)

# ---- B --------------------------------------------------------------------
gb = min(shown, key=lambda m: np.mean([abs(v) for _, v in loso(rows, m, "global")]))
pb = min(shown, key=lambda m: np.mean([abs(v) for _, v in loso(rows, m, "ref1")]))
for j, (md, lab, col, mk) in enumerate(
        (("global", f"no reference ({gb})", SERIES[0], "o"),
         ("ref1", f"one reference object ({pb})", SERIES[1], "s"))):
    acc = defaultdict(list)
    for x, e in loso(rows, gb if md == "global" else pb, md):
        acc[round(x["Ztape"] / 25.4)].append(BALL_MM * (1 + e / 100))
    xs = sorted(acc)
    ax[1].plot(xs, [np.mean(acc[k]) for k in xs], mk, ms=9, color=col,
               mec=SURFACE, mew=1.5, label=lab, zorder=4)
    for k in xs:
        v = acc[k]
        ax[1].plot([k, k], [np.min(v), np.max(v)], color=col, lw=2, alpha=.28,
                   zorder=3, solid_capstyle="round")
ax[1].axhline(BALL_MM, color=INK, lw=1.6, zorder=2)
ax[1].annotate("true 40 mm", (29, BALL_MM), textcoords="offset points",
               xytext=(-4, 5), ha="right", fontsize=9, color=INK)
ax[1].set_xlabel("true distance (inches)", color=INK2, fontsize=10)
ax[1].set_ylabel("measured ball diameter (mm)", color=INK2, fontsize=10)
ax[1].set_title("B · Ball size through the pinhole\nbars span the individual "
                "measurements", color=INK, fontsize=11.5, loc="left", pad=12)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

# ---- C --------------------------------------------------------------------
for j, (md, lab, col) in enumerate(
        (("global", "no reference", SERIES[0]),
         ("ref1", "one reference object", SERIES[1]))):
    d = np.array([BALL_MM * (1 + e / 100)
                  for _, e in loso(rows, gb if md == "global" else pb, md)])
    ax[2].hist(d, bins=np.arange(24, 66, 2.5), color=col, alpha=.62, zorder=3,
               label=f"{lab}\nmean {d.mean():.1f} mm, sd {d.std(ddof=1):.1f}")
ax[2].axvline(BALL_MM, color=INK, lw=1.6, zorder=4)
ax[2].annotate("40 mm", (BALL_MM, ax[2].get_ylim()[1]),
               textcoords="offset points", xytext=(4, -10), fontsize=9, color=INK)
ax[2].set_xlabel("measured ball diameter (mm)", color=INK2, fontsize=10)
ax[2].set_ylabel("ball-frames", color=INK2, fontsize=10)
ax[2].set_title("C · Unbiased on average, wide per measurement\n"
                "the spread is what limits a single reading",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

fig.tight_layout()
fig.savefig("findings/size_from_depth.png", dpi=170, facecolor=SURFACE)
print("wrote findings/size_from_depth.png")
