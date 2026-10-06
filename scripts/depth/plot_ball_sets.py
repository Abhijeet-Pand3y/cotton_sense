"""plot_ball_sets.py — figure for findings/ball_depth_analysis.md.

A  the power law itself, with each frame's scale drift divided out
B  the exponent per set, with the spread across frames shown honestly
C  what the correction is worth in millimetres, on the six-ball set

Panel A divides each frame's z by that frame's own fitted scale a. Without
that the sets sit on parallel offset lines and the shared slope - the thing
being claimed - is invisible. It is also exactly what the analysis does.
"""
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM_PER_IN = 25.4
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1"
SURFACE = "#fcfcfb"
SHORT = {"GX010117_ball9in_21in_test": "9+21", "GX010117_ball9in_29in_test": "9+29",
         "GX010117_ball11in_19in_27in_test": "11+19+27",
         "GX010117_ball19in_27in_test": "19+27",
         "GX010117_ball21in_25in_27in_test": "21+25+27",
         "GX010118_6ball_handheld_test": "6-ball (9-27)"}

rows = [r for r in csv.DictReader(open("findings/ball_depth_samples.csv"))
        if r["Z_true_mm"]]
byset = defaultdict(lambda: defaultdict(list))
for r in rows:
    byset[r["set"]][r["frame"]].append((float(r["Z_true_mm"]), float(r["z"])))

fits = {}
for s, fr in byset.items():
    per = []
    for f, pts in fr.items():
        Z = np.array([p[0] for p in pts]); z = np.array([p[1] for p in pts])
        if np.unique(np.round(Z, 1)).size < 2:
            continue
        g, la = np.polyfit(np.log(Z), np.log(z), 1)
        per.append((f, g, np.exp(la), Z, z))
    if per:
        fits[s] = per
sets = [s for s in SHORT if s in fits]

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

# ---- A -------------------------------------------------------------------
# Ratio form: inside each frame, divide both distance and depth by the NEAREST
# ball's values. The per-frame scale a cancels exactly, every frame starts at
# (1,1), and what is left is the only thing gamma claims - how depth grows with
# relative distance. A slope of 1 on these axes would be no compression.
for i, s in enumerate(sets):
    g_med = float(np.median([p[1] for p in fits[s]]))
    for _, g, a_, Z, z in fits[s]:
        k = int(np.argmin(Z))
        ax[0].plot(Z / Z[k], z / z[k], "o", ms=7, color=SERIES[i], mec=SURFACE,
                   mew=1.2, alpha=.8, zorder=3)
    xx = np.linspace(1.0, 3.2, 60)
    ax[0].plot(xx, xx ** g_med, color=SERIES[i], lw=2, zorder=4,
               label=f"{SHORT[s]}  γ={g_med:.2f}")
xx = np.linspace(1.0, 3.2, 60)
ax[0].plot(xx, xx, "--", color=MUTED, lw=1.6, zorder=2)
ax[0].annotate("γ = 1\nno compression", (3.15, 3.15), textcoords="offset points",
               xytext=(-6, -2), ha="right", va="top", fontsize=9, color=MUTED)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xticks([1, 1.5, 2, 2.5, 3]); ax[0].set_yticks([1, 1.5, 2, 2.5, 3])
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel("true distance ÷ nearest ball in the frame", color=INK2, fontsize=10)
ax[0].set_ylabel("DA3 depth ÷ nearest ball in the frame", color=INK2, fontsize=10)
ax[0].set_title("A · Depth grows too slowly with distance\nall points below the "
                "dashed line", color=INK, fontsize=11.5, loc="left", pad=12)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

# ---- B -------------------------------------------------------------------
med = np.array([np.median([p[1] for p in fits[s]]) for s in sets])
sd = np.array([np.std([p[1] for p in fits[s]], ddof=1)
               if len(fits[s]) > 1 else 0 for s in sets])
y = np.arange(len(sets))
ax[1].axvline(med.mean(), color=INK, lw=1.6, zorder=2)
ax[1].annotate(f"mean {med.mean():.3f}", (med.mean(), -0.5),
               textcoords="offset points", xytext=(-7, 0), fontsize=9,
               color=INK, ha="right")
ax[1].axvline(0.670, color=MUTED, lw=1.6, ls="--", zorder=2)
ax[1].annotate("0.670\nprior study", (0.670, len(sets) - 0.15),
               textcoords="offset points", xytext=(4, -2), fontsize=8.5,
               color=MUTED, va="top", ha="left")
for i, s in enumerate(sets):
    ax[1].plot([med[i] - sd[i], med[i] + sd[i]], [y[i], y[i]], color=SERIES[i],
               lw=2.5, alpha=.35, zorder=3, solid_capstyle="round")
    for _, g, *_ in fits[s]:
        ax[1].plot(g, y[i], "|", ms=9, color=SERIES[i], alpha=.55, zorder=4)
    ax[1].plot(med[i], y[i], "o", ms=11, color=SERIES[i], mec=SURFACE, mew=2,
               zorder=5)
    ax[1].annotate(f"{med[i]:.3f}", (med[i], y[i]), textcoords="offset points",
                   xytext=(0, -21), fontsize=9, color=INK2, ha="center")
ax[1].set_yticks(y); ax[1].set_yticklabels([SHORT[s] for s in sets], fontsize=9)
ax[1].set_xlim(0.28, 0.90); ax[1].set_ylim(-0.75, len(sets) - 0.1)
ax[1].set_xlabel("exponent γ  (dot = set median, ticks = single frames)",
                 color=INK2, fontsize=10)
ax[1].set_title("B · γ per set\nspread is frame-to-frame noise, not bias",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[1].grid(axis="y", visible=False)

# ---- C -------------------------------------------------------------------
s6 = "GX010118_6ball_handheld_test"
cross = float(np.median([np.median([p[1] for p in fits[s]])
                         for s in sets if s != s6]))
own = float(np.median([p[1] for p in fits[s6]]))
for g, lab, col, mk in ((own, f"γ={own:.2f} from this set", SERIES[0], "o"),
                        (cross, f"γ={cross:.2f} from other sets", SERIES[1], "s"),
                        (1.0, "γ=1, no correction", MUTED, "^")):
    acc = defaultdict(list)
    for _, _, _, Z, z in fits[s6]:
        a_ = float(np.exp(np.mean(np.log(z) - g * np.log(Z))))
        for zt, ze in zip(Z, (z / a_) ** (1 / g)):
            acc[zt].append(100 * (ze - zt) / zt)
    Zs = np.array(sorted(acc))
    ax[2].plot(Zs / MM_PER_IN, [np.mean(acc[k]) for k in Zs], mk + "-", ms=8,
               lw=2, color=col, mec=SURFACE, mew=1.5, label=lab, zorder=3)
ax[2].axhline(0, color=INK, lw=1.2, zorder=2)
ax[2].axvspan(24, 28, color="#f0efe9", zorder=1)
ax[2].annotate("25/27 in:\nDA3 ranks\nthese inverted", (26, ax[2].get_ylim()[1]),
               textcoords="offset points", xytext=(0, -8), ha="center",
               va="top", fontsize=8.5, color=MUTED)
ax[2].set_xlabel("true distance (inches)", color=INK2, fontsize=10)
ax[2].set_ylabel("mean distance error (%)", color=INK2, fontsize=10)
ax[2].set_title("C · What the correction is worth\n6-ball set, error per distance",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2,
             loc="upper center", bbox_to_anchor=(0.5, -0.155))

fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("findings/ball_depth_analysis.png", dpi=170, facecolor=SURFACE)
print("wrote findings/ball_depth_analysis.png")
