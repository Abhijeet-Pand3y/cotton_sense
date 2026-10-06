# Cotton files — what each column means

Three files. Two are our measurements from video; one is the hand-measured
truth they are compared against. Read `README.md` first — it explains the
experiment these come from.

Built by `scripts/depth/build_team_cotton_csv.py` from the depth run in
`scripts/depth/trimmed_boll_analysis.py`.

---

## `cotton_ground_truth.csv` — 38 rows

The bolls physically measured by the Plant & Soil team, greenhouse, 27 Mar 2026.
This is the truth everything is graded against.

| column | meaning |
|---|---|
| `pot` | pot number |
| `boll` | boll number within that pot |
| `height_tip_to_base_mm` | **the tip-to-base axis — the one we measure from video.** mean 40.46 mm, sd 8.59, spread 21.2%, range 20.4–52.4 |
| `length_mm` | the perpendicular axis. mean 26.89 mm |
| `width_mm` | the third axis. mean 26.73 mm |

Only `height_tip_to_base_mm` is comparable with the video measurements. The
model's tip and base keypoints define that same axis, which is how we know the
columns correspond: its spread of 21.2% is the figure the earlier depth study
independently derived as the true boll-to-boll variation.

**These bolls are not matched to the ones in the video.** The clip covers part
of the same plants, but track IDs have not been tied to tag numbers. So the
comparison is distribution against distribution, never boll against boll.

---

## `cotton_boll_measurements.csv` — 18 rows

**One row = one physical boll, followed across the video clip.** Every value is
the median over that boll's frames, which is what makes it a single measurement
rather than a noisy per-frame one.

| column | meaning |
|---|---|
| `boll_track` | the tracker's ID for this boll. Arbitrary; not a tag number. |
| `frames_measured` | how many frames this boll was measured in. **This is the main quality signal** — a boll seen twice is not measured, it is glimpsed. |
| `used_in_headline` | 1 if `frames_measured` ≥ 10. The 12 bolls flagged here are the ones behind the headline 21.6% spread. |
| `median_visible_fraction` | how much of the boll was unobstructed, 0–1. |
| `median_detection_confidence` | how sure the detector was it was looking at a boll, 0–1. |
| `occluded_frame_fraction` | share of frames where a leaf, stem or the frame edge cut across the boll. |
| `apparent_length_px` | tip-to-base distance **in pixels**. The raw image measurement, before any distance is involved. |
| `depth_model_output` | DA3's depth over the boll's interior. Relative, no units — turning this into millimetres is the whole problem. |
| `length_ball_calibration_mm` | length in mm using the ball calibration **untouched**. A genuine blind prediction: nothing here had seen a cotton measurement. Averages 55.9 mm against a true 40.5 — **38% high**, because scene scale does not transfer. |
| `length_scaled_to_ground_truth_mm` | the same numbers after one single scale factor (0.724) chosen so the 18-boll mean matches the hand-measured mean. **The mean therefore agrees by construction and proves nothing** — what this column tests is the spread and the shape, which one factor cannot fake. |

### How to read it

- **Do not** read `length_scaled_to_ground_truth_mm` for a boll as its true size.
  It carries roughly ±4 mm, and this clip had no reference object.
- **Do** compare the spread of the 12 rows with `used_in_headline = 1` against
  `height_tip_to_base_mm` in the ground-truth file. That comparison is the
  result: 21.6% against 21.2%.
- Rows with few frames are the unreliable ones by construction. Track 13 (2
  frames, 14 mm) and track 7 (1 frame) are pose or tracking failures, kept in the
  file so the numbers are not flattered by quietly deleting them.
- The headline 12-boll range quoted in the README (23.3–53.0 mm) comes from
  re-centring those 12 rows on the ground-truth mean. The scale in this column
  was set from all 18, so the raw column runs slightly higher.
- **The visibility-filtered rows cannot be reproduced from this file.** Those
  filters (`≥10 frames and visibility ≥0.8`, `≥20 and ≥0.9`) drop individual
  *frames* first and then require the surviving count, re-taking each boll's
  median from what is left — so they need `cotton_boll_frames.csv`. Filtering
  this per-boll table instead keeps whole bolls on their median visibility and
  gives 17.2% (n = 11) and 16.7% (n = 8) against the correct 17.6% (n = 10) and
  17.4% (n = 7). The two frame-count rows reproduce from either file.

---

## `cotton_boll_frames.csv` — 542 rows

**One row = one boll, in one frame.** The raw material behind the file above.
Useful for checking how stable a boll's measurement is frame to frame, which is
where DA3's noise shows up.

| column | meaning |
|---|---|
| `boll_track` | same ID as the per-boll file |
| `frame` | frame name, with its timestamp in the clip (`t30p36s` = 30.36 s) |
| `apparent_length_px` | tip-to-base in pixels, this frame |
| `depth_model_output` | DA3 depth over the boll interior, this frame |
| `detection_confidence` | detector confidence, this frame |
| `visible_fraction` | unobstructed share of the boll, this frame |
| `occluded_by` | `none`, `leaf`, `stem` or `frame_edge` |

Only frames passing quality gates are present: detection confidence ≥ 0.5, both
tip and base keypoints found with confidence ≥ 0.5, and enough boll interior
falling inside the depth map to sample. Depth was computed on every second
frame carrying a detection.

**Why depth is sampled over the boll's interior and not at the tip and base.**
DA3 returns a 504×322 depth picture for a 2988×5312 frame, so one depth pixel
covers about nine image pixels. The tip and base sit on the boll's silhouette,
where a single depth pixel straddles boll and background and the reading is
part background. The eroded interior is the only place the depth is reliably the
boll's own.

**A note on the length being a projection.** Tip-to-base measured in the image
is the boll's length *as seen*. A boll tilted toward the camera has part of its
length along the viewing axis, which the image cannot see, so the measurement is
a lower bound in principle. Recovering the tilt from the tip-and-base depth
difference was tried and is **not** recommended: it made the per-boll spread far
worse (34% against 21%), because the tilt is a depth difference of a few
millimetres, well below DA3's resolution at this scale.
