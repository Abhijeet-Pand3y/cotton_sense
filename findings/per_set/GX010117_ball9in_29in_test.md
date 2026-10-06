# Depth analysis - GX010117_ball9in_29in_test

5 measurable ball-frames, 2 ball(s), 3 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 9 | 228.6 | 229.70 | 0.3187 | 0.0434 | 3 |
| ball2 | 29 | 736.6 | 78.40 | 0.7251 | 0.0323 | 2 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 228.6 | 221.0 | -7.6 | -3.33% |
| ball2 | 736.6 | 709.3 | -27.3 | -3.71% |

mean |deviation| = 3.52%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

All balls correctly ordered by DA3 depth in all 2 frames.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 2 frame(s):

**gamma = 0.613** (median), sd 0.088, range 0.551-0.675

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t212s | 2 | 0.675 | 0.0081329 |
| obj21in_t228s | 2 | 0.551 | 0.019675 |

Held at the set's median gamma, the scale a spans 0.01182 to 0.013537 - a **14.5% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- The spread here is wide (sd 0.088). The median is reported rather than the mean so one bad frame cannot set the answer.

- Only two distances per frame, so each frame's fit passes exactly through its two points: the power-law SHAPE is assumed here, not tested. The spread across frames is still informative about how repeatable the exponent is.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.613 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 228.9 | +0.12% | +6.11% | 3 |
| ball2 | 736.6 | 737.9 | +0.18% | +6.11% | 2 |

mean |error| = **4.75%** over 5 ball-frames; without any compression correction (gamma = 1) it is 18.27%.

-> the correction helps here (3.8x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.685 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 238.5 | +4.33% | +12.14% | 3 |
| ball2 | 736.6 | 693.6 | -5.83% | -10.83% | 2 |

mean |error| = **4.93%** over 5 ball-frames; without any compression correction (gamma = 1) it is 18.27%.

-> the correction helps here (3.7x lower error).
