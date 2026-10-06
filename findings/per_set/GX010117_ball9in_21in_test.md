# Depth analysis - GX010117_ball9in_21in_test

10 measurable ball-frames, 2 ball(s), 5 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 9 | 228.6 | 236.27 | 0.3294 | 0.0242 | 5 |
| ball2 | 21 | 533.4 | 105.20 | 0.5887 | 0.0254 | 5 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 228.6 | 213.9 | -14.7 | -6.41% |
| ball2 | 533.4 | 520.4 | -13.0 | -2.44% |

mean |deviation| = 4.42%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

All balls correctly ordered by DA3 depth in all 5 frames.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 5 frame(s):

**gamma = 0.685** (median), sd 0.053, range 0.666-0.797

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t178s | 2 | 0.688 | 0.00807 |
| BASE baseline_t180s | 2 | 0.685 | 0.0079653 |
| BASE baseline_t181s | 2 | 0.682 | 0.0084288 |
| obj5in_t198s | 2 | 0.797 | 0.0037108 |
| obj7in_t192s | 2 | 0.666 | 0.0085826 |

Held at the set's median gamma, the scale a spans 0.0071432 to 0.0082496 - a **15.5% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- Only two distances per frame, so each frame's fit passes exactly through its two points: the power-law SHAPE is assumed here, not tested. The spread across frames is still informative about how repeatable the exponent is.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.685 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 226.1 | -1.08% | -6.68% | 5 |
| ball2 | 533.4 | 539.7 | +1.18% | +7.16% | 5 |

mean |error| = **1.70%** over 10 ball-frames; without any compression correction (gamma = 1) it is 12.60%.

-> the correction helps here (7.4x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.640 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 219.4 | -4.05% | -9.85% | 5 |
| ball2 | 533.4 | 556.4 | +4.32% | +10.92% | 5 |

mean |error| = **4.18%** over 10 ball-frames; without any compression correction (gamma = 1) it is 12.60%.

-> the correction helps here (3.0x lower error).
