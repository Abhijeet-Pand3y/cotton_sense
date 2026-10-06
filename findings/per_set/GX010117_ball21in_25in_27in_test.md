# Depth analysis - GX010117_ball21in_25in_27in_test

9 measurable ball-frames, 3 ball(s), 3 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 21 | 533.4 | 102.86 | 0.5683 | 0.0156 | 3 |
| ball2 | 25 | 635.0 | 86.01 | 0.6582 | 0.0072 | 3 |
| ball3 | 27 | 685.8 | 81.06 | 0.6604 | 0.0054 | 3 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 533.4 | 533.0 | -0.4 | -0.08% |
| ball2 | 635.0 | 643.6 | +8.6 | +1.36% |
| ball3 | 685.8 | 684.9 | -0.9 | -0.14% |

mean |deviation| = 0.53%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

- **27 in reads nearer than 25 in** in 1 of 3 frames  (intermittent)

A power law is monotonic, so no exponent can repair a wrong order - it can only stretch it.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 3 frame(s):

**gamma = 0.626** (median), sd 0.134, range 0.435-0.693

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t340s | 3 | 0.693 | 0.0073148 |
| BASE baseline_t341s | 3 | 0.626 | 0.011189 |
| obj7in_t354s | 3 | 0.435 | 0.038997 |

Held at the set's median gamma, the scale a spans 0.011189 to 0.011395 - a **1.8% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- The spread here is wide (sd 0.134). The median is reported rather than the mean so one bad frame cannot set the answer.

- 3 distances per frame, so each fit is over-determined and the power-law shape is genuinely tested.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.626 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 533.4 | 532.2 | -0.22% | +2.99% | 3 |
| ball2 | 635.0 | 658.7 | +3.74% | +4.51% | 3 |
| ball3 | 685.8 | 662.9 | -3.34% | -6.31% | 3 |

mean |error| = **3.09%** over 9 ball-frames; without any compression correction (gamma = 1) it is 4.14%.

-> the correction helps here (1.3x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.685 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 533.4 | 538.8 | +1.02% | +3.98% | 3 |
| ball2 | 635.0 | 654.8 | +3.12% | +3.83% | 3 |
| ball3 | 685.8 | 658.6 | -3.96% | -6.66% | 3 |

mean |error| = **2.99%** over 9 ball-frames; without any compression correction (gamma = 1) it is 4.14%.

-> the correction helps here (1.4x lower error).
