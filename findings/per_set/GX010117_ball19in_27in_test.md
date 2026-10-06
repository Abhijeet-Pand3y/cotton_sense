# Depth analysis - GX010117_ball19in_27in_test

10 measurable ball-frames, 2 ball(s), 5 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 19 | 482.6 | 113.28 | 0.5449 | 0.0289 | 5 |
| ball2 | 27 | 685.8 | 81.32 | 0.6565 | 0.0454 | 5 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 482.6 | 481.0 | -1.6 | -0.33% |
| ball2 | 685.8 | 682.6 | -3.2 | -0.47% |

mean |deviation| = 0.40%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

All balls correctly ordered by DA3 depth in all 5 frames.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 5 frame(s):

**gamma = 0.699** (median), sd 0.160, range 0.336-0.705

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t305s | 2 | 0.703 | 0.0066384 |
| BASE baseline_t306s | 2 | 0.705 | 0.0065909 |
| obj5in_t323s | 2 | 0.699 | 0.0072436 |
| obj9in_t309s | 2 | 0.661 | 0.0097546 |
| obj9in_t313s | 2 | 0.336 | 0.068614 |

Held at the set's median gamma, the scale a spans 0.0068008 to 0.0076746 - a **12.8% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- The spread here is wide (sd 0.160). The median is reported rather than the mean so one bad frame cannot set the answer.

- Only two distances per frame, so each frame's fit passes exactly through its two points: the power-law SHAPE is assumed here, not tested. The spread across frames is still informative about how repeatable the exponent is.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.699 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 482.6 | 492.5 | +2.06% | +9.57% | 5 |
| ball2 | 685.8 | 672.8 | -1.89% | -8.73% | 5 |

mean |error| = **2.07%** over 10 ball-frames; without any compression correction (gamma = 1) it is 6.67%.

-> the correction helps here (3.2x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.640 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 482.6 | 485.6 | +0.62% | +8.72% | 5 |
| ball2 | 685.8 | 682.6 | -0.46% | -8.02% | 5 |

mean |error| = **2.81%** over 10 ball-frames; without any compression correction (gamma = 1) it is 6.67%.

-> the correction helps here (2.4x lower error).
