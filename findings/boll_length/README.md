# Real boll length from depth — LeftDown_v5

Applying the ping-pong-ball regression to real cotton, to measure tip-to-base
length in millimetres.

![results](boll_length.png)

## What was run

- `inference_results/LeftDown_v5` — 1882 frames, 1506 with a detection
- DA3 depth on **753 frames** (every 2nd frame carrying a detection)
- **2545 boll-frames**, **92 tracks**, 39 tracks with ≥20 frames
- Depth sampled over the **eroded boll mask**, not at the keypoints
- Mapping and `f = 2905.6 px` carried over from the ball calibration

Scripts: `scripts/depth/boll_length.py`, `scripts/depth/plot_boll_length.py`.
Per-frame output: `boll_length_per_frame.csv`.

## Headline: the correction helps and hurts, in different ways

These are two different error types and they move in opposite directions. Both
are real; neither cancels the other.

| | raw pixels | depth-corrected (power) |
|---|---|---|
| **precision** — spread of one boll across frames | **6.3%** | 9.2% (worse) |
| **accuracy** — length still correlating with distance | r = **−0.72** | r = **−0.14** (much better) |
| spread across different bolls | 26.9% | 17.9% |

**Precision gets worse** because the correction multiplies each frame's pixel
length by that frame's depth, and that depth is noisy: within a track the depth
CV is 5.2% while the pixel CV is only 6.3%. Multiplying a fairly stable number
by a comparably noisy one can only add variance. The power law is the worst of
the three mappings (9.2%) for a specific reason: `Z ∝ z^(1/γ)` with `1/γ ≈ 1.5`
**amplifies** relative depth noise by about 1.5×.

**Accuracy gets much better**, and this is what the correction is actually for.
With raw pixels alone, a boll's measured length correlates with its distance at
r = −0.723 — distant bolls simply read smaller, which is pure perspective bias,
not biology. The power correction cuts that to −0.137 and pulls the spread
across bolls from 26.9% to 17.9%. Of the three mappings the power law removes
the most bias (−0.137 vs −0.254 proportional, −0.217 linear), exactly as the
compression model predicts.

## This reproduces the earlier failure, and explains it

`depth_findings.md` recorded that depth-correcting boll size **made it worse**:
variability rose from 5.8% to 8.1% against raw pixels. Independently, on
different footage and with a differently-fitted mapping, this run gives
**6.3% → 8.7%**. Almost the same numbers.

That document attributed the failure to uncorrected compression. This run shows
that is not the whole story: the compression correction is applied here, and
precision still degrades — *more* so, because the correction amplifies noise.
The earlier measurement was not broken by missing the correction. It was
measuring precision, where per-frame depth can only hurt.

## The fix

Average depth over the track **before** correcting, not after. Per-frame depth
CV is 5.2%; over ~51 frames of a tracked boll that falls to roughly 0.7% if the
noise is independent. Doing it that way keeps the accuracy gain (r = −0.113,
spread 17.9%) and discards the per-frame noise that caused the precision loss.

One number per boll, from its whole track — not one number per frame averaged
afterwards.

## Absolute lengths are NOT trustworthy here

Median measured length is **67.7 mm** (proportional) or **76.0 mm** (power).
That is larger than a green cotton boll should be, and the reason is visible in
panel A:

- the ball calibration covers DA3 depth `z` from **0.250 to 0.748**
- the cotton bolls read `z` from **0.620 to 1.252**
- **86.2% of boll measurements sit above the calibrated maximum**

So almost every absolute number is an extrapolation of a curve fitted somewhere
else. The predicted distances (median 908 mm) also sit beyond the 507–751 mm
`depth_findings.md` reported for bolls. On top of that:

- **no reference object** is in these frames, which is the 14%-error regime from
  `size_from_depth.md` before extrapolation is even considered;
- **`f` is assumed to transfer.** Same camera and firmware (HERO11, 5312×2988),
  but the digital-lens/FOV mode of the two sessions could not be confirmed from
  metadata, and a different mode changes `f` outright.

The *relative* results above survive all of this — a CV and a correlation are
unchanged by a wrong global scale. The millimetre figures do not.

## Why depth came from the mask, not the keypoints

DA3 outputs 504×322 for a 2988×5312 frame, so one depth pixel spans about nine
image pixels. The tip and base sit on the boll's silhouette, where a single
depth pixel straddles boll and background. The eroded mask interior is the only
place the depth is reliably the boll's own.

The 3D variant — adding the tip/base depth difference back to recover tilt — was
computed (median 75.3 mm) and is **not recommended**: its per-track CV is 34.1%,
far worse than the 2D projection. The tilt it tries to recover is a depth
difference of a few millimetres, well below DA3's noise at this resolution.

Note the 2D length is a **lower bound** in principle: a boll tilted toward the
camera has part of its length along the view axis, which projection cannot see.

## Limits

- No physical ground truth in the repo. `validation_review.xlsx` is a
  model-prediction review, not boll measurements; the 38 measured bolls
  `depth_findings.md` mentions are not present and are not matched to track IDs.
  Every claim here is internal consistency, not absolute accuracy.
- Panel C assumes bolls are not systematically larger at one distance. Over 39
  tracks spanning a 70% distance range that is reasonable, but unverified.
- Tracks with high CV (three exceed 40%) are likely tracking or pose failures
  rather than depth failures; they are kept so the medians are not flattered.
- Only every 2nd detected frame carries depth.

## What to do next

1. **Put a known-size reference in shot.** Without one the absolute scale is
   unrecoverable, and here it is extrapolated as well. The tags already in frame
   are a candidate (use max observed length across the track, since projection
   only shrinks); a stake diameter or a clipped-on ping-pong ball is better.
2. **Re-calibrate at boll working distance.** The current calibration stops at
   0.75 m in `z` and the bolls live past it. A ball set at 0.7–1.3 m would put
   the fit where it is actually used.
3. **Aggregate per track, then correct.** Cheap, and it recovers the precision.
4. **Match tracks to tag IDs** to unlock the 38 ground-truth bolls. That is the
   only route to a real accuracy number rather than a consistency one.
