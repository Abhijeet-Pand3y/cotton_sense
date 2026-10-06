# Depth analysis - GX010117_ball25in_test

4 measurable ball-frames, 1 ball(s), 4 frame(s)

## 1. Measured, per ball

| ball | tape (in) | tape (mm) | radius px | DA3 z (median) | z sd | frames |
|---|---|---|---|---|---|---|
| ball1 | 25 | 635.0 | 85.14 | 0.7004 | 0.0155 | 4 |

The z column is shown per ball for orientation only. Nothing below is fitted from it: because DA3's output scale drifts between frames, the fits pair balls **inside** a frame instead (see section 2).

## 1b. Geometry check - distance from ball size alone (no DA3)

| ball | tape mm | from radius mm | diff mm | diff % |
|---|---|---|---|---|
| ball1 | 635.0 | 650.5 | +15.5 | +2.45% |

mean |deviation| = 2.45%. Tape, fitted circles and one global focal length agree to this much; it bounds how good any depth result could look.

## 2. Power law, fitted inside each frame

**Not fittable from this set.** No frame here holds two different distances, so there is no ratio to take and no slope to fit. This set still checks the geometry above, and the cross-set gamma is applied to it below.

## 3. Distance error, gamma fitted on the OTHER sets (cross-set)

**No error can be measured here.** This set spans one distance, and the scale a is fitted on it, so the estimate is forced onto the truth and would read 0.00 % however wrong DA3 was. That is arithmetic, not accuracy, so it is not reported.
