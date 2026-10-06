# Combined ball data — column reference

Two files, same measurements at two levels of aggregation.

| file | grain | rows |
|---|---|---|
| `combined_ball_summary.csv` | one row per (set, ball) — **base and object side by side** | 22 |
| `combined_ball_frames.csv` | one row per (set, ball, frame), before averaging | 126 |

Built by `scripts/depth/build_combined_csv.py`.

## Three different distances, deliberately kept apart

| prefix | source | uses DA3? | what it is for |
|---|---|---|---|
| `tape_*` | measured with a tape | no | the ground truth |
| `geom_*` | ball pixel radius + solved focal length, `Z = f·D/(2r) − c` | no | independent check that the tape and the circles agree |
| `da3_*` | DA3 depth inverted, `Z = (z/a)^(1/γ)` | yes | the thing being evaluated |

`da3_*` uses γ fitted on that ball's own set; `da3_cross_*` uses γ from the
**other** sets only, so it never saw this data. The cross-set columns are the
honest accuracy figure.

## Ball size

`ball_diameter_mm` = 40.0 (regulation ping-pong ball; the known physical size
everything is anchored to). `radius_px` / `ball_diameter_px` are the measured
apparent size, refit in every frame.

## base_ vs obj_

Every measurement appears twice:

- `base_*` — frames with only the balls in shot (the clean reference)
- `obj_*` — frames with a distractor object present, at `object_distances_in`
- `delta_*_pct` — the object frames' change from baseline, in percent

`n_base_frames` / `n_obj_frames` count what went into each; `n_obj_dropped`
and `dropped_reasons` record object frames excluded because the ball was
occluded or not measurable.

## Empty cells are meaningful

`da3_*` is blank wherever `da3_determinable` is not `yes`. The scale `a` is
fitted per frame from the balls in that frame, so a frame holding a single
distance has nothing to fit it against and the estimate would be forced onto
the truth — reporting 0 % error there would be arithmetic, not accuracy. Four
sets (9in, 17in, 25in, GX0119 17in) are blank throughout for this reason.

## Other columns

| column | meaning |
|---|---|
| `status` | per-frame fit outcome: `ok`, `occluded`, `not_measurable`, `suspect_badfit`. Only `ok` rows carry depth. |
| `match` | template-match score for the ball in that frame |
| `move_px` | how far the ball sat from its baseline position |
| `gamma_set`, `gamma_cross` | exponents used for `da3_*` and `da3_cross_*` |
| `focal_px`, `tape_offset_mm` | globally solved: 2908 px, +31.3 mm |
| `handheld_ball_distance_in` | GX010118 only — the hand-held 7th ball, not tracked as a fixed ball |

## Sanity figures for the file as a whole

- geometry vs tape: mean 1.86 % over 119 ball-frames
- DA3, own-set γ: mean 4.11 % over 95 ball-frames
- DA3, cross-set γ: mean 5.38 %, max 26.25 %
