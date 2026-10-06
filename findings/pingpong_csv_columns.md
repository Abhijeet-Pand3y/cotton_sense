# Column reference — `pingpong_ball_size_per_frame_v2.csv`

> ## ⚠ Working file — do not share or quote from this one
>
> Three of its size columns were produced with information that should not have
> been available, so they read better than the method really performs:
>
> | column | problem |
> |---|---|
> | `D_from_tape_mm` | uses the tape distance, and the camera constants were themselves derived from these balls assuming 40 mm. It must land near 40 — it is an arithmetic check, not a measurement. |
> | `D_perframe_mm` | the ball being measured helped set the scale used to measure it. Understates the error by about a third (5.22% vs a true 7.06%). |
> | `D_globalscale_mm` | the depth curve and scale were fitted using every ball, including the one being scored. |
>
> `D_strict_mm` is the only sound size column here.
>
> **For sharing, use `findings/team_share/` instead** — that file contains only
> results produced by predicting balls the calculation had never seen, and is
> written in plain language.
>
> This file is kept because the comparison between its columns is what revealed
> the problem, and because the raw measured inputs are useful.


**One row = one ball, in one frame.** 116 rows. A frame with 6 balls produces 6
rows sharing the same `frame`.

Built by `scripts/depth/ball_size_per_frame.py`.

## Where the numbers come from

Three different origins, and it matters which is which:

- **measured by you** — tape readings (`*_distance_*`)
- **measured from the image** — `radius_px`, `da3_depth_raw`
- **derived** — everything else, computed from those two plus a calibration

---

## 1 · Identity (columns 1–4)

| # | column | meaning |
|---|---|---|
| 1 | `set` | Which capture/setup. Named for the ball distances in it, e.g. `GX010117_ball11in_19in_27in_test` = three balls at 11″, 19″, 27″. |
| 2 | `frame` | The frame. `baseline_*` = clean scene, `obj*_*` = distractor present. The `t###s` suffix is the timestamp in the source video. |
| 3 | `is_baseline` | `1` = nothing added to the scene (the clean reference), `0` = an object was moved in. |
| 4 | `ball` | Which fixed ball. **`ball1` is always the nearest** — balls are numbered by descending radius, which for identical spheres is guaranteed to be ascending distance. |

## 2 · Your tape measurements (columns 5–12)

| # | column | units | meaning |
|---|---|---|---|
| 5 | `ball_distance_in` | in | Camera → this ball, as you measured it. |
| 6 | `ball_distance_mm` | mm | Same, converted (× 25.4). |
| 7 | `ball_pupil_distance_mm` | mm | `ball_distance_mm + 32.0`. **The pinhole measures from the lens entrance pupil, not the camera body.** Omitting this offset makes the ball read 37 mm instead of 40. The 32 mm was fitted, not measured. |
| 8 | `object_distance_in` | in | Camera → the distractor object. **Blank on baseline frames** (79 rows) — nothing was there. |
| 9 | `object_distance_mm` | mm | Same, converted. |
| 10 | `handheld_ball_distance_in` | in | GX010118 only — the 7th, hand-held ball. Approximate by nature, and *not* one of the fixed measured balls. Reads `unknown` where the position wasn't recorded. |
| 11 | `handheld_ball_distance_mm` | mm | Same, converted. |
| 12 | `object_minus_ball_mm` | mm | **Signed gap: negative = the object was in FRONT of this ball, positive = behind it.** This is the quantity that decides whether the object could perturb this particular ball's depth — an object 400 mm behind is a different situation from one 100 mm in front. |

## 3 · Measured from the image (columns 13–15)

| # | column | units | meaning |
|---|---|---|---|
| 13 | `radius_px` | px | Ball radius from a sub-pixel radial edge fit, **re-fitted in this frame** rather than reused from a reference frame. |
| 14 | `diameter_px` | px | `2 × radius_px`. The apparent size. |
| 15 | `da3_depth_raw` | — | DA3's depth, median over the **inner 60 %** of the ball disc. Relative units with no absolute meaning; the rim is excluded because that is where depth bleeds into the background. |

## 4 · Truth and the closure check (columns 16–18)

| # | column | units | meaning |
|---|---|---|---|
| 16 | `true_diameter_mm` | mm | `40.0` — a regulation ping-pong ball. The reference everything is scored against. |
| 17 | `D_from_tape_mm` | mm | `2 × radius_px × ball_pupil_distance_mm / f`, with `f = 2905.6 px`. **A closure check, not a result** — `f` and the 32 mm offset were solved from these same balls *assuming* 40 mm, so this cannot come out far from 40. It confirms the arithmetic. |
| 18 | `err_from_tape_pct` | % | Its deviation from 40. **This is the noise floor** — circle-fit error plus tape error, mean 1.50 % under the pooled `f` and `c` (median 1.08 %). Judge the depth columns against this, not against zero — but note the leave-one-set-out figure, which is what `final_validation.md` compares against, is **1.72 %**. |

## 5 · Depth-derived, per-frame calibration (columns 19–23)

The realistic deployment case: a known-size object in the same frame sets the scale.

| # | column | units | meaning |
|---|---|---|---|
| 19 | `n_balls_in_frame` | count | How many balls share this frame. More balls = better constrained. |
| 20 | `gamma_used` | — | The compression exponent, **taken from the other sets** so it never saw the set it is scoring. Blank where no fit was possible. |
| 21 | `Z_perframe_mm` | mm | Distance from DA3: `Z = (z/a)^(1/γ)`, with only the scale `a` fitted inside this frame at that fixed γ. |
| 22 | `D_perframe_mm` | mm | Diameter from that distance. **The realistic number.** Mean 40.24, mean abs error 5.22 %. |
| 23 | `err_perframe_pct` | % | Its deviation from 40. |

**Why 21 rows are blank here:** a single-ball frame has one free parameter (the
scale) and one data point, so the fit is forced onto the truth and would report
0 % error whatever DA3 did. That is arithmetic, not accuracy, so it is left
empty.

> **Fixed in v2.** Earlier this column fitted *both* γ and the scale per frame —
> two parameters through a 2-ball frame's two points, so `D_perframe_mm` came
> back exactly equal to `D_from_tape_mm`. Those rows were forced, not measured,
> and the error was understated at 3.51 %. Holding γ fixed leaves a real
> residual; 0 of 26 two-ball rows now match the tape answer.

## 5b · Depth-derived, sound protocol (columns 24–26)

**These are the trustworthy size columns in this file.** The scale comes from the
*other* balls in the frame, never the target, and the camera constants are
refitted with the whole recording held out.

| # | column | units | meaning |
|---|---|---|---|
| 24 | `Z_strict_mm` | mm | Distance, with nothing from this ball used to derive it. |
| 25 | `D_strict_mm` | mm | Diameter from that distance. Mean 40.27, sd 3.84, **mean abs error 7.06%**. |
| 26 | `err_strict_pct` | % | Deviation from 40 mm. |

Blank wherever the frame holds only one ball, since there is then no other object
to set the scale from.

## 6 · Depth-derived, no calibration (columns 27–29)

| # | column | units | meaning |
|---|---|---|---|
| 27 | `Z_globalscale_mm` | mm | Distance using **one** scale constant for every set — nothing known about the scene. |
| 28 | `D_globalscale_mm` | mm | Diameter from that. **The worst case**, mean abs error 14.34 %. |
| 29 | `err_globalscale_pct` | % | Its deviation from 40. |

---

## The three diameter columns, side by side

| column | distance source | mean | sd | mean abs err | status |
|---|---|---|---|---|---|
| `D_from_tape_mm` | tape | 40.05 | 0.84 | 1.50 % (1.72 % held out) | arithmetic check only |
| `D_perframe_mm` | depth, target helped calibrate | 40.24 | 2.83 | 5.22 % | **optimistic, do not quote** |
| `D_strict_mm` | depth, target excluded | 40.27 | 3.84 | **7.06 %** | **sound** |
| `D_globalscale_mm` | depth, fitted on all balls | 40.60 | 6.64 | 14.34 % | optimistic |

All three centre near 40 — the measurement is close to **unbiased**. What
separates them is **spread**, and that is what limits any single reading.

Accuracy by frame content: 3 balls → 2.32 %, 2 balls → 4.20 %, 6 balls →
7.23 %. The 6-ball set is worst because it contains the 25″/27″ pair that DA3
ranks in the wrong order — an error no exponent can repair.
