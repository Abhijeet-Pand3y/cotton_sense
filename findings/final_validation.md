# Is DA3 usable for cotton? — final leak-free validation

![results](final_validation.png)

Run: `python scripts/depth/final_validation.py`

## Protocol — nothing about the scored ball touches its constants

Each ball-frame is scored with one **set** held out entirely:

| constant | fitted on |
|---|---|
| `f`, `c` | the other sets only |
| `γ` | the other sets only |
| `a` (scale) | other balls in the frame, **never the target** — or other sets entirely |

The target's radius and depth are inputs; its true distance is used only to
score. The earlier `D_perframe_mm` column let the target help set its own scale,
which flattered it by 1.8 points (see `leakage_audit.md`).

**The one thing that cannot be held out** is `D = 40 mm`, because `f` is derived
from it. Every *absolute* number is therefore conditional on the balls really
being 40 mm. No spread, ratio or comparison depends on it.

## Results

| scenario | n | distance error | size error | size mean | size sd |
|---|---|---|---|---|---|
| **A · no reference at all** | 116 | 16.50% | **16.99%** | 39.16 mm | 7.83 |
| **B · one reference object in frame** | 302 | 8.83% | **8.86%** | 40.10 mm | 4.48 |
| **C · several references in frame** | 95 | 6.33% | **6.41%** | 40.24 mm | 3.46 |
| *measurement floor (closure check)* | 116 | — | *1.72%* | 40.05 mm | 0.84 |

Distance and size errors are nearly identical, as they must be: `D = 2rZ/f`, so
a fractional distance error is the same fractional size error. They differ only
by the per-fold change in `f` (<0.8%). Reporting both makes that visible rather
than assumed.

**A known-size object in shot halves the error.** Everything above the 1.72% floor
is DA3's contribution.

> **Corrected 20 Aug 2026.** The floor was quoted as 1.50%, which is the closure
> error under the *pooled* `f` and `c`. Scored the way every other row here is —
> with the constants refitted per fold — it is **1.72%** (median 1.21%). Both are
> mean absolute errors; the comparison in this table now uses matching constants.

## Does the verdict depend on the constants?

| variation | B: distance | B: size | C: size |
|---|---|---|---|
| as fitted (γ = 0.730) | 8.83% | 8.86% | 6.41% |
| γ = 0.55 | 19.83% | 19.95% | 15.33% |
| γ = 0.75 | 8.24% | 8.18% | **6.01%** |
| **γ = 1.0 (no compression)** | 12.84% | 12.40% | **11.36%** |
| f −5% | 8.83% | 10.01% | 8.32% |
| f +5% | 8.83% | 9.53% | 7.46% |

Three things follow:

1. **The compression is real and the correction earns its place.** γ = 1 nearly
   doubles the error (6.4% → 11.4%). This is out-of-sample, so it is not a
   fitting artefact.
2. **The optimum is broad.** Anything in γ ≈ 0.70–0.80 gives 6.0–6.5%. Our fitted
   0.730 sits inside it, so the exact value is not critical — which is why a
   borrowed γ transfers acceptably between scenes.
3. **`f` moves size but not distance.** It cancels in the scale fit and re-enters
   only at the pinhole step. A 5% focal error costs about 2 points of size error
   — worth pinning with a proper calibration before trusting absolute mm.

## Is the error concentrated at close range?

Bolls sit further out than the nearest balls, so the headline is only relevant if
the close positions are not carrying the error. Re-scored with near balls dropped
as target *and* reference (`scripts/depth/plot_distance_bands.py`):

| balls included | A | B | C |
|---|---|---|---|
| all, 9–29 in | 16.99% (116) | 8.86% (302) | 6.41% (95) |
| ≥ 11.5 in | 16.07% (90) | 8.62% (198) | 5.75% (68) |
| ≥ 21 in | 14.69% (53) | **10.23% (64)** | 7.26% (37) |

Dropping the close balls is worth ~0.3 points. The far band is *worse* under B
and C — it is where the 25″/27″ inversion lives, and it is the band nearest boll
working distance. The headline is therefore mildly optimistic for cotton
distances rather than conservative.

## What it means for cotton

A regulation ping-pong ball is **40.0 mm**; the 38 measured bolls average
**40.46 mm** tip-to-base, over an overlapping distance range. So the error
measured here transfers directly in magnitude.

**With one reference object in frame: 8.9% = ±3.6 mm on a typical boll.**

**Population statistics — usable.** True boll-to-boll spread is 21.2%. Adding
8.9% of independent measurement noise gives an observed spread of
√(21.2² + 8.9²) = **23.0%** — the measurement inflates the population spread by
under 2 points. Distributions, means and treatment comparisons survive this.

**Independent confirmation.** With several references (6.4%) the predicted
observed spread is √(21.2² + 6.4²) = 22.1%. The trimmed cotton run actually
measured **21.6%** against a true 21.2%. The real data lands slightly *better*
than this ball experiment predicts — two independent routes agreeing.

**Single-boll measurement — marginal.** ±3.6 mm against a true range of
20.4–52.4 mm separates a big boll from a small one, but not two bolls a few mm
apart. Averaging ~4 independent frames would reach 5% *if* the errors were
independent — they are not fully, because the scale error is shared along a
track. Frame averaging removes the depth noise, not the calibration error.

## Verdict

**DA3 is usable for cotton boll sizing, with two conditions:**

1. **A known-size reference must be in frame.** Without one the error is 17% and
   the scale is unrecoverable — that is the difference between a usable
   measurement and a useless one.
2. **Keep distant background out of shot.** The greenhouse roof pushed DA3's
   scale nearly 10× off and made the calibration extrapolate; cropping the top
   15% fixed it (`findings/trimmed_boll/`).

Good for: population distributions, ranking large vs small bolls, tracking
growth over time where the same boll is compared with itself.

Not good for: per-boll absolute size to better than a few mm, or separating
bolls of similar size in one frame — DA3 also mis-orders objects a couple of
inches apart in depth, and no exponent repairs a wrong ordering.

## Remaining assumptions

- `D = 40 mm` for the balls, uncalibrated. A 1% error here shifts every absolute
  number by 1%.
- `f` transfers between sessions. Same camera and firmware, but the digital-lens
  mode was not confirmed from metadata.
- Scenario B's n = 302 counts every (target, reference) pair, so those samples
  are correlated; the mean is sound but the effective sample size is nearer 95.
- One camera, one indoor scene for the balls; the cotton check is a single clip.
