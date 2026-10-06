# Column reference — `final_validation_per_row.csv` and `_summary.csv`

Both files come from `scripts/depth/final_validation.py`. Every estimate in them
was produced for a ball the calculation had never encountered: the camera
constants, the depth curve and the scale were all derived from *other*
recordings, then applied. Nothing about a ball contributed to the numbers used
to measure it.

For a shareable, plain-language version see `findings/team_share/`.

---

# `final_validation_per_row.csv` — 513 rows, 24 columns

**One row = one ball, in one photo, under one setup.** The same ball appears
several times: once with nothing of known size available, once for each other
object that could have set the scale, and once using all of them together.

## Which setup the row belongs to

| column | meaning |
|---|---|
| `scenario` | `A_no_reference` (nothing of known size in the picture, 116 rows), `B_one_reference` (one object of known size, 302 rows — one per possible choice), `C_several_references` (all available objects, 95 rows). |
| `held_out_set` | the recording that was excluded when the constants for this row were worked out. Always equals `set` — that is the audit trail showing the row was predicted, not fitted. |
| `reference_ball` | which ball supplied the scale. Blank for setup A; one ball for B; a `+`-joined list for C. |

## Identity

| column | meaning |
|---|---|
| `set` | recording session. |
| `frame` | photo, with its timestamp. |
| `ball` | which ball; `ball1` is nearest. |
| `is_baseline` | 1 = clean scene, 0 = an extra object was in shot. |
| `n_balls_in_frame` | how many balls share the photo. |

## Truth

| column | meaning |
|---|---|
| `ball_distance_in` / `_mm` | tape distance from camera to this ball. |
| `true_pupil_distance_mm` | that distance measured from the lens rather than the camera body — the tape offset `c` added. This is what the estimate is graded against. |
| `true_diameter_mm` | 40.0. |

## Inputs measured from the photo

| column | meaning |
|---|---|
| `radius_px` | ball radius in pixels, re-fitted in this photo. |
| `da3_depth_raw` | the depth model's reading over the middle of the ball; relative, no units. |

## The constants used for this row

These vary by row because each is refitted with the row's recording excluded.
They are included so the protocol can be checked rather than taken on trust.

| column | meaning |
|---|---|
| `f_used_px` | focal length in pixels. |
| `c_used_mm` | camera-body-to-lens offset added to every tape reading. |
| `gamma_used` | strength of the distance correction; below 1 means the model compresses distance. |
| `a_used` | the scale constant, converting the model's relative depth to millimetres. |

## Outputs

| column | meaning |
|---|---|
| `Z_pred_mm` | estimated camera-to-ball distance. |
| `dist_err_mm`, `dist_err_pct` | signed error against `true_pupil_distance_mm`. |
| `D_pred_mm` | estimated ball diameter. |
| `size_err_mm`, `size_err_pct` | signed error against 40 mm. |

Distance and size errors are nearly identical by construction — size is distance
multiplied by the ball's pixel radius over the focal length, so a 5% distance
error is a 5% size error. They differ only by the small change in focal length
between folds.

---

# `final_validation_summary.csv` — 21 rows

One row per (setup, variation). Aggregates of the file above.

| column | meaning |
|---|---|
| `table` | `scenario` = the three headline results; `sensitivity` = the same measurement repeated with a constant deliberately changed. |
| `variant` | which setup. |
| `constant` | `as fitted`, or the constant that was overridden (`gamma=0.55` … `gamma=1.00`, `f-5pct`, `f+5pct`). |
| `n` | rows behind the number. |
| `dist_err_mean_pct`, `dist_err_median_pct` | average and typical distance error. |
| `dist_bias_pct` | signed average — whether distances run systematically long or short. |
| `size_err_mean_pct`, `size_err_median_pct` | the same for size. |
| `size_bias_pct` | signed average size error. |
| `size_mean_mm`, `size_sd_mm` | measured diameter and its spread. |

## Reading the sensitivity rows

- **`gamma`** is the strength of the distance correction. `gamma=1.00` turns it
  off and costs nearly double the error (6.4% → 11.4%), which is the evidence
  that the correction is real rather than curve-fitting. Anything from 0.70 to
  0.80 performs the same, so the value need not be precise.
- **`f-5pct` / `f+5pct`** change the focal length. Distance error is unchanged at
  8.83% while size *bias* swings by about ±5% — the focal length cancels when the
  scale is fitted and re-enters only at the final step, so it shifts sizes
  systematically rather than adding scatter. Worth pinning with a proper camera
  calibration before quoting absolute millimetres.

## One caveat on `n`

Setup B's 302 rows count every (target, possible-guide) pairing, so those rows
are related to one another. The averages are sound, but the effective number of
independent measurements is closer to 95.
