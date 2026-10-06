# Monocular depth for boll sizing — what we measured, and whether it works in the field

**Question.** Can a monocular depth model turn boll pixel size into real size, given
that a boll's apparent size depends on how far away it is?

**Short answer.** Yes, but only with a correction we can now quantify. Raw model
depth makes the measurement *worse*; corrected depth cuts the systematic error by
about three times. Scene clutter — the thing we were most worried about for field
capture — turns out not to matter.

---

## Why this needed testing

Physical size follows `size = pixel_length × distance / focal_length`. The pixel
length we measure well. The distance comes from a monocular depth network
(Depth Anything 3), and monocular depth is known to be scale-ambiguous: it infers
absolute distance from context in the image. So two questions had to be answered
before relying on it:

1. Does what else is in the frame change the depth reading? (Field images will
   have foliage, equipment and varying background.)
2. Is the depth *linear* in true distance? If not, two bolls at different
   distances get different size errors, and a single calibration cannot fix it.

## How we tested it

A GoPro was clamped so it could not move. One to three **ping-pong balls**
(40 mm, known and identical) were placed at different distances and never moved
for the duration of a clip. During each clip an object was moved in and out of
frame, in front of and behind the balls.

Because the balls never move, two things follow that make the measurement clean:

- their true distance is constant, so **any change in reported depth is error**;
- their pixel radius gives the true distance ratio between them, so
  **the model can be scored without a tape measure or camera calibration**.

Three setups, ~700 analysed frames, plus 16–35 "baseline" frames per setup with
nothing but the balls in view, as a clean reference.

---

## Finding 1 — scene clutter does not meaningfully affect depth

| condition | spread in reported depth |
|---|---|
| baseline (nothing moving) | **3.6 %** |
| object moving through the frame | **4.8 %** |

The object costs about 1.2 percentage points above the noise floor — only 1.3×
the floor. In three of nine ball-measurements the cluttered frames were actually
*quieter* than the baseline.

**For field capture this is the favourable result.** Foliage brushing the lens,
equipment in shot, or a varying background do not corrupt the depth reading. It
also averages down: this error is random, so over ~30 frames of a tracked boll it
falls below 1 %.

Worst-case frames did reach 13–18 %, so individual frames can be bad; the fix is
temporal averaging, which the tracking pipeline already does.

## Finding 2 — depth is systematically compressed

All the balls are the same 40 mm sphere, so `radius × depth` should give the same
size for every one of them. It does not:

| setup | ball sizes (should be identical) | spread |
|---|---|---|
| A | 60.0, 50.1 | **18.1 %** |
| B | 64.2, 58.5, 51.6 | **21.7 %** |
| C | 65.4, 64.5, 57.1 | **13.4 %** |

The near ball always reads largest, the far ball smallest. The error scales with
how far apart the objects are:

**0.43 % size error per 1 % of depth separation** (r = 0.88, 7 pairs, 3 setups).

The cause is that the model under-reports how much farther the far objects are.
A ball genuinely 1.75× farther away is reported as only 1.45× farther.

> **Corrected 20 Aug 2026.** This line previously read "1.79× ... 1.45×", which
> spliced two different pairs: the 8 same-photo pairs at a true ratio of 1.795
> (the 9″/17″ balls) report 1.54–1.70, mean 1.62. The 1.45 belongs to the 15″/27″
> pairs, whose true ratio is 1.747 once the +32 mm pupil offset is applied
> (1.80 on raw tape). As printed it implied γ = 0.64; correctly paired, that pair
> gives γ ≈ 0.82 and the median over all 151 same-photo pairs is 0.729.

## Finding 3 — the compression follows a power law, and is correctable

Fitting `reported_depth ∝ true_depth^γ`:

| setup | γ |
|---|---|
| A | 0.592 |
| B | 0.662 |
| C | 0.755 |

mean **0.670**, sd 0.082. Consistently well below 1.0 — meaning compression — and
consistent in direction across all three setups.

Applying a single mean γ, even though γ varies between setups:

| | systematic size error at 48 % separation |
|---|---|
| raw depth | 9.2 – 14.8 % |
| corrected with `depth^(1/0.67)` | **0.4 – 5.1 %** |

**About a threefold reduction**, using one constant.

![Results](depth_findings.png)

---

## What this explains

An earlier attempt to depth-correct boll size made the measurement worse:
mask-axis size variability rose from 5.8 % to 8.1 % compared with using raw
pixels. We now know why. Bolls in that data spanned 0.507–0.751 m — a 48 %
depth separation, which is exactly where the compression is largest. The
correction was adding roughly 20 % of systematic error while removing a smaller
perspective effect.

That was a real failure with an identified cause, not a dead end.

## Limits of what has been shown

Stated plainly, because these bound how far the result can be pushed:

- **γ rests on 2–3 balls per setup.** These are slopes through very few points. A
  two-ball fit cannot fail — it just draws a line — so the power-law *shape* is
  assumed, not yet demonstrated.
- **γ varies 24 % between setups** and we do not know why. It does not obviously
  track the depth range covered (r = 0.24, n = 3).
- **Not yet tested on cotton.** That needs metric depth, which needs one tape
  measurement to fix the camera's focal length.
- **Occluded frames were excluded**, so behaviour when an object covers the target
  is uncharacterised. In practice occluded bolls are rejected anyway by the
  occlusion head.
- Absolute distances quoted anywhere rest on an estimated focal length
  (≈ 2450 px from lens and sensor specs), not a measured one. The γ result does
  **not** depend on this.

---

## Next experiment

Designed to fix the specific weaknesses above.

**More balls per frame, not more clips.** Five or six spheres spanning a wide
depth range in a single frame gives a properly fitted γ *with a residual* — which
tests whether the power law holds at all, rather than assuming it. This matters
more than repeating the current setup.

**Cover the boll depth range.** Current clips ran roughly 0.35–0.64 m; bolls sit
at 0.51–0.75 m. If γ drifts with absolute distance, it must be fitted in the band
where it will be used.

**Vary one scene property at a time.** Backdrop distance first, since that is the
main difference between the greenhouse (black cloth, close) and the field.

**One tape measurement.** Measuring a single ball's distance pins the focal length
exactly, converts everything to true metric distance, and tests the 2450 px
estimate. Costs nothing.

**Validate out-of-sample.** Fit γ on one set of balls, then apply the correction
to a *different* set at known distances and check whether their sizes come out
equal. Fitting and testing on the same data will always look good.

## Path to real data

1. Apply `depth^(1/γ)` to the existing cotton measurements and re-run the size
   analysis. If the corrected size variability drops below the 5.8 % raw-pixel
   baseline, the correction works on real bolls. **This is a one-line change and
   can be done immediately** — it does not need metric depth, because the
   remaining scale factor is absorbed by calibration.
2. Calibrate the constant against physically measured bolls. Ground-truth
   measurements for 38 bolls already exist; the outstanding piece is matching
   tracked bolls to their tag numbers.
3. Only then is per-boll metric size meaningful.

Worth noting that the boll-size measurement already matches ground truth in
*distribution* — measured spread 19.9 % against a true 21.2 % — using pixels
alone. Depth correction is about removing the distance-dependent bias between
bolls, not about making the measurement work at all.
