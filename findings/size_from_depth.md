# Real size from relative depth

Mapping DA3's relative depth to real distance, then measuring the ball through the pinhole and scoring it against its true 40 mm.

![results](size_from_depth.png)

## The chain

```
r_px          measured in every frame
Z             from DA3 depth through a fitted mapping   <- what is tested
D = 2*r*Z/f   pinhole size, compared with 40 mm
```

**One identity to be clear about.** `D_est/D_true = Z_est/Z_true` exactly, because both share the same `r` and the same `f`. The size result is therefore not independent evidence about the depth mapping - it is the same error in millimetres. It is still the number the boll work needs, but agreement here must not be read as a second confirmation.

## Distance is measured from the entrance pupil

The tape was read from the camera body, so every reading is short by a constant `c`, fitted jointly with the focal length: **f = 2906 px, c = +32.0 mm**.

Ignoring `c` makes the ball measure **37.15 mm instead of 40** - a 7 % error that looks like a depth failure and is purely bookkeeping. With `Z_pupil = Z_tape + c` the closure check gives **40.05 mm, sd 0.84**. That only confirms the algebra, since `f` and `c` were solved from these same balls.

gamma is likewise fitted against `Z_pupil`, not the tape: `c` is 4.5-13.7 % of these distances and varies with distance, so leaving it in would bias the exponent.

## 1. Which mapping?

Leave-one-**set**-out throughout - a flexible model always wins in-sample, which is exactly how one gets fooled.

| model | no reference | one known-size object | perfect scale (ceiling) |
|---|---|---|---|
| proportional | 14.01% | 13.79% | 8.47% |
| linear | 16.35% | 13.41% | 6.98% |
| power | 15.59% | 8.73% | 5.93% |
| quadratic | 16.35% | 10.29% | 6.95% |
| inverse | 19.38% | 814.30% | 14.17% |

- **no reference** - one fixed mapping, nothing known about the scene
- **one known-size object** - shape is global, scale set by a single reference object in the same frame. The realistic deployment case.
- **perfect scale** - scale least-squares over every ball in the frame. Not deployable, but it shows what remains once frame-to-frame scale drift is removed entirely.

`inverse` is unusable (814% under one-reference scaling): `1/z` explodes wherever z is small.

### Which model actually wins depends on the regime

| regime | power − proportional | 95% CI | verdict |
|---|---|---|---|
| no reference | +1.58 pts | [+0.27, +2.93] | **proportional better** |
| one reference | -5.06 pts | [-7.06, -3.01] | **power better** |
| perfect scale | -2.54 pts | [-3.52, -1.62] | **power better** |

Paired bootstrap on identical held-out rows. The reversal is the interesting part: **with no scale information the simplest model wins**, because the error is dominated by scene-to-scene scale drift and extra flexibility only fits that noise. **Once the scale is pinned by a reference object, the power law's shape earns its keep** and beats the straight line by 5 points.

So: best without a reference is **proportional** (14.01%); best with one is **power** (8.73%); the ceiling is **power** (5.93%). A known-size object in shot cuts the error 1.6x.

## 2. Ball size against the true 40 mm

| calibration | mean | sd | mean abs error | range |
|---|---|---|---|---|
| no reference (proportional) | 40.78 mm | 7.23 mm | 14.01% | 26.4-62.1 mm |
| one reference object (power) | 39.23 mm | 4.34 mm | 8.73% | 28.0-50.5 mm |

**Nearly unbiased on average, but wide on any single measurement.** Both calibrations centre within about 2 % of 40 mm, so there is no systematic size offset left to remove. What limits a single reading is the spread: a 40 mm ball is read anywhere from 28 to 50 mm with a reference object, and 26 to 62 without one. Averaging over a tracked object is what makes this usable, exactly as the earlier study concluded for depth.

## 3. DA3's own metric mode does not work here

`focal x depth / 300`, DA3's documented metric scaling, returns 2.42-7.24 m where the truth is 0.26-0.77 m - **9.8x out**, and the factor is not constant. Metric mode cannot be trusted as-is at this range; the mapping has to be fitted, which is what section 1 does.

## Limits

- `f` and `c` are calibrated from these same balls, so the closure check is not independent evidence. They would transfer to the boll work only if the camera and lens setting are unchanged.
- Size error equals distance error identically; nothing here tests the size step separately.
- The one-reference case uses the nearest ball as the reference, so that ball is excluded from its own frame's scoring and frames with a single ball contribute nothing.
- Held-out sets differ from each other mainly in scale, so leave-one-set-out is a genuinely hard test - which is why the global numbers look poor next to the earlier per-frame-calibrated ones.
- Still one camera, one indoor scene, spheres rather than bolls.

## What this means for the boll pipeline

- **Put a known-size reference in shot.** It is worth 1.6x on absolute size, and without it a single reading carries roughly 14 % error.
- **Use the power law once a reference exists**, the straight line otherwise. That is the opposite of the usual instinct to reach for the more flexible model when data is scarce.
- **Average over frames.** The measurement is close to unbiased, so averaging a tracked boll converges on the right answer; a single frame does not.

