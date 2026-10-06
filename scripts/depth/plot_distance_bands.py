"""plot_distance_bands.py — does the ball error depend on how close the ball is?

Cotton bolls sit further from the camera than the nearest calibration balls, so
the headline numbers are only relevant if they are not being driven by the close
end of the range. This splits the same leave-one-set-out results by distance.

Writes findings/team_share/ball_results_by_distance.png and
       findings/team_share/ball_error_by_distance.csv
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8880", "#e6e5e1", "#fcfcfb"
SC = [("A_no_reference", "nothing of known size", MUTED, "^"),
      ("B_one_reference", "one object of known size", S[1], "s"),
      ("C_several_references", "several objects of known size", S[0], "o")]
BANDS = [(0.0, "all balls\n9–29 in (23–74 cm)"),
         (11.5, "beyond 11 in\n(> 28 cm)"),
         (21.0, "beyond 21 in\n(> 53 cm)")]

rows = list(csv.DictReader(open("findings/final_validation_per_row.csv")))
dist = {(r["set"], r["frame"], r["ball"]): float(r["ball_distance_in"]) for r in rows}
for r in rows:
    r["d"] = float(r["ball_distance_in"])
    r["se"] = abs(float(r["size_err_pct"]))
    r["rd"] = dist.get((r["set"], r["frame"], r["reference_ball"])) if r["reference_ball"] else None


def sel(sc, lo):
    """Rows of one scenario where target AND reference are at or beyond `lo`."""
    return [r for r in rows if r["scenario"] == sc and r["d"] >= lo
            and (r["rd"] is None or r["rd"] >= lo)]


fig, ax = plt.subplots(1, 2, figsize=(11.6, 5.0), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2, labelsize=9)
    a.grid(True, color=GRID, lw=.8, zorder=0)
    a.set_axisbelow(True)

# A — error against target distance
ds = sorted({r["d"] for r in rows})
ax[0].axvspan(8, 11.5, color=GRID, alpha=.85, zorder=1)
ax[0].annotate("closer than cotton\nis ever shot", (9.6, 25.5), ha="center",
               fontsize=8.5, color=MUTED, zorder=6)
for sc, lab, col, mk in SC:
    y = [np.mean([r["se"] for r in rows if r["scenario"] == sc and r["d"] == d]) for d in ds]
    ax[0].plot(ds, y, "-" + mk, ms=6, lw=1.8, color=col, mec=SURF, mew=1.2,
               zorder=3, label=lab)
ax[0].set_xlabel("distance to the ball being measured (inches)", color=INK2, fontsize=10)
ax[0].set_ylabel("mean size error (%)", color=INK2, fontsize=10)
ax[0].set_xlim(8, 30)
ax[0].set_title("A · Error does not fall away with distance\nthe near balls are not what is hurting",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

# B — the three distance bands
w = .26
out = []
for j, (lo, blab) in enumerate(BANDS):
    for k, (sc, lab, col, mk) in enumerate(SC):
        v = [r["se"] for r in sel(sc, lo)]
        m = float(np.mean(v))
        out.append({"band": blab.replace("\n", " "), "scenario": lab,
                    "n": len(v), "mean_size_error_pct": round(m, 2)})
        ax[1].bar(j + (k - 1) * w, m, w * .92, color=col, zorder=3, edgecolor=SURF,
                  label=lab if j == 0 else None)
        ax[1].annotate(f"{m:.1f}", (j + (k - 1) * w, m), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=8, color=INK2)
ax[1].set_xticks(range(len(BANDS)))
ax[1].set_xticklabels([b[1] for b in BANDS], fontsize=8.5)
ax[1].set_ylabel("mean size error (%)", color=INK2, fontsize=10)
ax[1].set_title("B · Restricting to cotton working distance\nchanges little, and the far band is worse",
                color=INK, fontsize=11.5, loc="left", pad=12)
ax[1].grid(axis="x", visible=False)
ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

fig.tight_layout()
fig.savefig("findings/team_share/ball_results_by_distance.png", dpi=170, facecolor=SURF)
print("wrote findings/team_share/ball_results_by_distance.png")

p = Path("findings/team_share/ball_error_by_distance.csv")
with open(p, "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=["band", "scenario", "n", "mean_size_error_pct"])
    w_.writeheader(); w_.writerows(out)
print(f"wrote {p}")
