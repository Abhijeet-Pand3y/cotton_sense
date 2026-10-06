"""plot_boll_length.py — figure for findings/boll_length/README.md."""
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_regression import load, solve_pinhole, fit_model

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1", "#fcfcfb"

ball = load(); f, c = solve_pinhole(ball)
for x in ball: x["Zp"] = x["Ztape"] + c
bz = np.array([x["z"] for x in ball]); bZ = np.array([x["Zp"] for x in ball])
maps = {m: fit_model(m, bz, bZ) for m in ("proportional", "linear", "power")}
r = list(csv.DictReader(open("findings/boll_length/boll_length_per_frame.csv")))
by = defaultdict(list)
for x in r: by[x["track_id"]].append(x)
tr = [(t, v) for t, v in by.items() if len(v) >= 20]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2, labelsize=9); a.grid(True, color=GRID, lw=.8, zorder=0)
    a.set_axisbelow(True)

# A — extrapolation
cz = np.array([float(x["z_boll"]) for x in r])
ax[0].hist(bz, bins=np.arange(.2, 1.32, .04), color=S[0], alpha=.75, zorder=3,
           label=f"ping-pong calibration\n(n={bz.size})")
ax[0].hist(cz, bins=np.arange(.2, 1.32, .04), color=S[1], alpha=.7, zorder=3,
           label=f"cotton bolls\n(n={cz.size})")
ax[0].axvline(bz.max(), color=INK, lw=1.6, ls="--", zorder=4)
ax[0].annotate(f"calibration ends\n{bz.max():.2f}", (bz.max(), ax[0].get_ylim()[1]*.72),
               textcoords="offset points", xytext=(7, 0), fontsize=9, color=INK)
ax[0].set_xlabel("DA3 raw depth z", color=INK2, fontsize=10)
ax[0].set_ylabel("measurements", color=INK2, fontsize=10)
ax[0].set_title("A · 86% of bolls sit beyond the calibration\nabsolute lengths are "
                "extrapolated", color=INK, fontsize=11.5, loc="left", pad=12)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

# B — precision (within a boll)
raw = np.array([100*np.std([float(x["px_tip_base"]) for x in v], ddof=1)
                / np.mean([float(x["px_tip_base"]) for x in v]) for _, v in tr])
series = [("raw pixels", raw, MUTED)]
for i, m in enumerate(("proportional", "power")):
    cv = np.array([100*np.std(np.array([float(x["px_tip_base"]) for x in v])
                              * maps[m](np.array([float(x["z_boll"]) for x in v]))/f, ddof=1)
                   / np.mean(np.array([float(x["px_tip_base"]) for x in v])
                             * maps[m](np.array([float(x["z_boll"]) for x in v]))/f)
                   for _, v in tr])
    series.append((f"depth-corrected\n({m})", cv, S[i]))
bp = ax[1].boxplot([s[1] for s in series], widths=.55, patch_artist=True,
                   medianprops=dict(color=INK, lw=2), showfliers=False, zorder=3)
for p, (_, _, col) in zip(bp["boxes"], series):
    p.set_facecolor(col); p.set_alpha(.65); p.set_edgecolor(SURF)
for i, (_, v, col) in enumerate(series, 1):
    ax[1].plot(np.random.default_rng(1).normal(i, .06, v.size), v, "o", ms=4,
               color=col, alpha=.6, zorder=4)
    ax[1].annotate(f"{np.median(v):.1f}%", (i, np.median(v)), textcoords="offset points",
                   xytext=(16, -3), fontsize=9, color=INK)
ax[1].set_xticklabels([s[0] for s in series], fontsize=8.5)
ax[1].set_ylabel("within-boll length spread, CV (%)", color=INK2, fontsize=10)
ax[1].set_title("B · PRECISION gets worse\ndepth noise is added, not removed",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[1].grid(axis="x", visible=False)

# C — accuracy (between bolls)
Zm = np.array([np.median([float(x["Z_mm"]) for x in v]) for _, v in tr])
pxm = np.array([np.median([float(x["px_tip_base"]) for x in v]) for _, v in tr])
for lab, val, col, mk in (("raw pixels (scaled)", pxm/np.mean(pxm), MUTED, "o"),
                          ("depth-corrected (power)",
                           np.array([np.median([float(x["px_tip_base"])*maps["power"](float(x["z_boll"]))/f
                                                for x in v]) for _, v in tr]), S[1], "s")):
    v = val/np.mean(val)
    cc = np.corrcoef(Zm, v)[0, 1]
    ax[2].plot(Zm, v, mk, ms=8, color=col, mec=SURF, mew=1.4, alpha=.85, zorder=3,
               label=f"{lab}   r = {cc:+.2f}")
    b1, b0 = np.polyfit(Zm, v, 1)
    xs = np.linspace(Zm.min(), Zm.max(), 20)
    ax[2].plot(xs, b0+b1*xs, color=col, lw=2, zorder=4)
ax[2].axhline(1, color=INK, lw=1.2, ls=":", zorder=2)
ax[2].set_xlabel("boll distance from camera (mm)", color=INK2, fontsize=10)
ax[2].set_ylabel("measured length ÷ mean", color=INK2, fontsize=10)
ax[2].set_title("C · ACCURACY gets better\nthe distance bias is what it removes",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

fig.tight_layout()
fig.savefig("findings/boll_length/boll_length.png", dpi=170, facecolor=SURF)
print("wrote findings/boll_length/boll_length.png")
