# Depth analysis - GX010118_6ball_handheld_test

52 measurable ball-frames, 6 ball(s), 10 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 9 | 228.6 | 220.28 | 0.2659 | 0.0290 | 8 |
| ball2 | 15 | 381.0 | 139.18 | 0.3745 | 0.0563 | 10 |
| ball3 | 17 | 431.8 | 124.49 | 0.4245 | 0.0375 | 9 |
| ball4 | 21 | 533.4 | 103.58 | 0.5095 | 0.0500 | 7 |
| ball5 | 25 | 635.0 | 86.43 | 0.6146 | 0.0546 | 9 |
| ball6 | 27 | 685.8 | 81.49 | 0.5523 | 0.0525 | 9 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 228.6 | 231.8 | +3.2 | +1.40% |
| ball2 | 381.0 | 385.5 | +4.5 | +1.19% |
| ball3 | 431.8 | 434.8 | +3.0 | +0.70% |
| ball4 | 533.4 | 529.0 | -4.4 | -0.82% |
| ball5 | 635.0 | 640.3 | +5.3 | +0.84% |
| ball6 | 685.8 | 681.1 | -4.7 | -0.69% |

mean |deviation| = 0.94%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 1c. Ordering, counted in every frame

- **27 in reads nearer than 25 in** in 9 of 9 frames  <- in every frame: systematic, not noise

- **17 in reads nearer than 15 in** in 1 of 9 frames  (intermittent)

A power law is monotonic, so no exponent can repair a wrong order - it can only stretch it.

## 2. Power law, fitted inside each frame

`z = a * Z^gamma` fitted separately in each of 9 frame(s):

**gamma = 0.708** (median), sd 0.093, range 0.512-0.808

| frame | balls | gamma | scale a |
|---|---|---|---|
| BASE baseline_t129s | 6 | 0.745 | 0.0048682 |
| BASE baseline_t135s | 6 | 0.759 | 0.0045647 |
| handheld7in_t143s | 5 | 0.512 | 0.018101 |
| handheld9in_t086s | 6 | 0.605 | 0.013449 |
| handheldunknown_t148s | 5 | 0.708 | 0.0055052 |
| handheldunknown_t153s | 6 | 0.773 | 0.0039028 |
| handheldunknown_t159s | 6 | 0.652 | 0.0077028 |
| handheldunknown_t165s | 5 | 0.685 | 0.0064765 |
| obj5in_t028s | 6 | 0.808 | 0.003141 |

Held at the set's median gamma, the scale a spans 0.0053261 to 0.007185 - a **34.9% drift between frames**. That is why gamma is fitted within a frame: pairing balls across frames at different scales corrupts the ratio gamma depends on. (Each frame's own a in the table above is not comparable frame to frame, because a and gamma trade off.)

- The spread here is wide (sd 0.093). The median is reported rather than the mean so one bad frame cannot set the answer.

- 6 distances per frame, so each fit is over-determined and the power-law shape is genuinely tested.

## 3. Distance error, gamma fitted on this set (in-sample)

gamma = 0.708 (optimistic: the exponent has seen this data). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 223.5 | -2.21% | -10.44% | 8 |
| ball2 | 381.0 | 381.7 | +0.17% | +17.16% | 10 |
| ball3 | 431.8 | 442.1 | +2.38% | +5.68% | 9 |
| ball4 | 533.4 | 544.5 | +2.08% | +4.88% | 7 |
| ball5 | 635.0 | 692.5 | +9.06% | +19.61% | 9 |
| ball6 | 685.8 | 619.1 | -9.72% | -14.51% | 9 |

mean |error| = **5.64%** over 52 ball-frames; without any compression correction (gamma = 1) it is 8.73%.

-> the correction helps here (1.5x lower error).

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

gamma = 0.640 (the honest number: this exponent never saw this set). Scale a re-fitted per frame.

| ball | tape mm | mean est mm | mean err % | worst err % | frames |
|---|---|---|---|---|---|
| ball1 | 228.6 | 207.8 | -9.12% | -17.61% | 8 |
| ball2 | 381.0 | 375.2 | -1.51% | +16.96% | 10 |
| ball3 | 431.8 | 440.5 | +2.01% | +4.23% | 9 |
| ball4 | 533.4 | 553.9 | +3.84% | +7.21% | 7 |
| ball5 | 635.0 | 723.5 | +13.94% | +26.25% | 9 |
| ball6 | 685.8 | 639.2 | -6.80% | -12.19% | 9 |

mean |error| = **7.01%** over 52 ball-frames; without any compression correction (gamma = 1) it is 8.73%.

-> the correction helps here (1.2x lower error).
