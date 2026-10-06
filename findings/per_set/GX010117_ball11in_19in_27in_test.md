# Depth analysis - GX010117_ball11in_19in_27in_test

11 measurable ball-frames, 3 ball(s), 4 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 11 | 279.4 | 188.54 | 0.3921 | 0.0122 | 4 |
| ball2 | 19 | 482.6 | 112.30 | 0.5591 | 0.0244 | 4 |
| ball3 | 27 | 685.8 | 81.27 | 0.7065 | 0.0192 | 3 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 279.4 | 276.2 | -3.2 | -1.14% |
| ball2 | 482.6 | 485.5 | +2.9 | +0.59% |
| ball3 | 685.8 | 683.1 | -2.7 | -0.40% |

mean |deviation| = 0.71%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

All balls correctly ordered by DA3 depth in all 4 frames.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 4 frame(s):

**gamma = 0.640** (median), sd 0.020, range 0.605-0.650

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t265s | 3 | 0.640 | 0.010556 |
| BASE baseline_t266s | 3 | 0.650 | 0.010429 |
| obj27in_t296s | 3 | 0.641 | 0.01075 |
| obj7in_t283s | 2 | 0.605 | 0.012496 |

Held at the set's median gamma, the scale a spans 0.010126 to 0.011062 - a **9.2% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- 3 distances per frame, so each fit is over-determined and the power-law shape is genuinely tested.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.640 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 279.4 | 279.4 | -0.01% | +1.53% | 4 |
| ball2 | 482.6 | 483.6 | +0.20% | -1.51% | 4 |
| ball3 | 685.8 | 684.2 | -0.24% | -0.71% | 3 |

mean |error| = **0.70%** over 11 ball-frames; without any compression correction (gamma = 1) it is 11.34%.

-> the correction helps here (16.2x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.685 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 279.4 | 287.3 | +2.84% | +3.26% | 4 |
| ball2 | 482.6 | 479.8 | -0.57% | -3.16% | 4 |
| ball3 | 685.8 | 665.9 | -2.91% | -3.34% | 3 |

mean |error| = **2.27%** over 11 ball-frames; without any compression correction (gamma = 1) it is 11.34%.

-> the correction helps here (5.0x lower error).
