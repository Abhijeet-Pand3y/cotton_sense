# `ball_measurements.csv` — what each column means

Part of the package in this folder; read `README.md` first for the experiment
and the results. This file is the measurement data behind them.

**One row = one ball, in one photo.** 116 rows. A photo containing six balls
produces six rows sharing the same `photo` value.

## What the experiment was

A camera was clamped still. Ping-pong balls — every one exactly **40 mm** across
— were placed at measured distances and never moved. In some photos an extra
object was moved into the scene. Because the balls never move and their true
size is known, any variation in the measured size is error, and can be measured
exactly.

The question being answered: **can a depth-estimating AI model, from a single
photo, tell us how big an object really is?**

## The three setups

The model gives *relative* depth — it can tell near from far, but not in
centimetres. To turn that into real sizes something in the picture has to fix
the scale. The three setups differ only in what was available to do that:

| setup | what is in the picture | realistic use |
|---|---|---|
| **setup1** | nothing of known size | point the camera and measure, no preparation |
| **setup2** | one object of known size | place one pre-measured item in shot |
| **setup3** | several objects of known size | place several pre-measured items in shot |

In the field, "an object of known size" would be something like a ping-pong ball
clipped to a stake, or a measured tag, placed at roughly the distance of whatever
is being measured.

## Identifying columns

| column | meaning |
|---|---|
| `recording` | which recording session. The name lists the ball distances, e.g. `GX010117_ball11in_19in_27in_test` = balls at 11, 19 and 27 inches. |
| `photo` | which photo, and its timestamp in the recording. |
| `scene` | `balls only` = clean scene; `extra object present` = something was moved into the shot. |
| `ball` | which ball. **`ball1` is always the closest** — balls are numbered nearest first. |
| `balls_in_photo` | how many balls share this photo. |

## Truth — what we know independently

| column | meaning |
|---|---|
| `true_distance_in` | distance from camera to this ball, measured with a tape. |
| `true_distance_mm` | the same distance in millimetres, adjusted so it is measured from the lens rather than the camera body. |
| `true_size_mm` | `40.0` — the real diameter of the ball. This is what every estimate is graded against. |

## What was measured from the photo

| column | meaning |
|---|---|
| `measured_radius_px` | the ball's radius in pixels, found by fitting a circle to its edge. Measured fresh in every photo. |
| `depth_model_output` | the AI model's depth reading over the middle of the ball. A relative number with no units — turning it into millimetres is the whole problem. |

## Results — setup 1 (nothing of known size)

| column | meaning |
|---|---|
| `setup1_distance_mm` | estimated camera-to-ball distance. |
| `setup1_size_mm` | estimated ball diameter. Compare against 40. |
| `setup1_size_error_pct` | how far off, as a percentage. |

## Results — setup 2 (one object of known size)

| column | meaning |
|---|---|
| `setup2_distance_mm` | estimated distance. |
| `setup2_size_mm` | estimated diameter — the middle answer across every object that could have been the reference. |
| `setup2_size_error_pct` | how far off. See the headline table: this is the *middle-of-choices* figure (6.7%), not the single-arbitrary-reference figure (8.9%). |
| `setup2_best_case_size_mm` | the answer if the *most helpful* available object had been chosen. |
| `setup2_worst_case_size_mm` | the answer if the *least helpful* had been chosen. |

Best and worst case are given because the choice matters: an object at a similar
distance to the target works much better than one much closer or farther. The
gap between these two columns is the cost of choosing badly.

## Results — setup 3 (several objects of known size)

| column | meaning |
|---|---|
| `setup3_distance_mm` | estimated distance. |
| `setup3_size_mm` | estimated diameter. |
| `setup3_size_error_pct` | how far off. |

## Why some cells are empty

21 rows are blank in the setup 2 and 3 columns. Those photos contain only **one**
ball, so there was no *other* object available to fix the scale. Filling them in
would have meant using the ball to measure itself, which always returns exactly
the right answer and tells us nothing — so they are left empty.

## Headline numbers

| setup | rows | average measured size | spread | typical error |
|---|---|---|---|---|
| nothing of known size | 116 | 39.16 mm | ±7.83 | **17.0%** |
| one object of known size, **chosen without care** | 302 pairings | 40.10 mm | ±4.48 | **8.9%** |
| one object of known size, *middle of the available choices* | 95 | 40.33 mm | ±3.51 | 6.7% |
| several objects of known size | 95 | 40.24 mm | ±3.46 | **6.4%** |

The two middle rows are the same measurements aggregated differently, and the
distinction matters when quoting a number.

- **8.9%** scores every ball against *each* other ball in turn, so it averages
  over good and bad choices of reference alike. **This is the figure to quote**
  for "one reference object in shot", and it is what the README and the figure
  use.
- **6.7%** is the middle answer once all the available references are combined —
  which is really a mild form of the several-references case, and reads better
  than a single arbitrary reference deserves. It is the `setup2_size_error_pct`
  column in this file, kept because the per-row best/worst columns beside it are
  what show how much the choice of reference costs.

All of them average close to the true 40 mm. **What separates them is
consistency**, and that is what limits any single measurement.

## The distance-band file

`ball_error_by_distance.csv` is a nine-row summary of the same measurements split
by how far the ball was from the camera — the check on whether the close-up balls
are flattering the headline. One row per (distance band, setup):

| column | meaning |
|---|---|
| `band` | which balls were included: all of them, only those beyond 11 in (28 cm), or only those beyond 21 in (53 cm). A ball is excluded as a *reference* as well as a target, so a band is a scene with nothing nearer than that. |
| `scenario` | which of the three setups. |
| `n` | measurements behind the number. |
| `mean_size_error_pct` | mean size error for that combination. |

Built by `scripts/depth/plot_distance_bands.py`, which also draws
`ball_results_by_distance.png`.

## How these numbers were checked

Every estimate was produced for a ball the calculation had never encountered.
The camera settings, the depth curve and the scale were all worked out from
*other* recordings and then applied to this one. No ball contributed to the
numbers used to measure it.

This matters because it is easy to get a flattering result by accident: an
earlier version of this analysis let each ball help set its own scale, which made
the method look 35% better than it is. Columns that could not be checked this way
are not in this file.

One assumption cannot be checked this way: the balls are taken to be exactly
40 mm because that is the regulation size — nobody put callipers on them. If the
real batch were 39.5 mm, every size in this file would shift by about 1%. No
comparison, spread or percentage in the file depends on it.
