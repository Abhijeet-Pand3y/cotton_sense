# Real measurements in Inference_Viewer.html

## Units: the numbers were always mm, the label was wrong

If you saw `65u` and `z636`, those were already **65 mm** and **636 mm**. The
viewer computed metres and multiplied by 1000, then appended a hardcoded `u`
because it was written before any calibration existed:

```js
(ds.zMedM * 1000).toFixed(1) + " u"     // metres -> mm, then labelled "u"
```

The viewer now reads `units` / `calibration` from `metadata.json` and labels
**mm** when the maps are calibrated, keeping `u` for raw uncalibrated runs. The
disclaimer text switches too. Original saved as `Inference_Viewer.html.bak`.

Nothing about the arithmetic changed — only what it is called.

---

The viewer already does the geometry correctly — it deprojects tip and base
through a pinhole model and reports a 3-D length, with a mask-axis cross-check,
a rolling median over 5 frames, and an outlier-rejected **Boll report** per
track. It needs two things it cannot invent: **camera intrinsics** and **depth
in metres**.

`scripts/depth/make_viewer_depth.py` supplies both. No HTML is modified.

## Load order — use the TRIMMED run

`left_down_v5_trimmed` is the one to use. Its top 15 % is cropped, which keeps
the greenhouse roof out of DA3's view; with the roof in frame the depth scale
was ~9x off and 86 % of readings were extrapolated. See
`findings/trimmed_boll/`.

1. **Load frames** → `inference_results/left_down_v5_trimmed/frames`
2. **Load detections** → `tracked_json`
3. **Load depth** → `inference_results/left_down_v5_trimmed/depth_metric/depth`
   (`metadata.json` sits in that folder and is picked up automatically)
4. Tick **Boll height labels (3-D, tip↔base)**, then **Boll report** for a
   per-track table.

The status line should read `intrinsics ✓`. If it says `✗ — load metadata.json`,
the folder selection missed the sidecar.

## Why a conversion step exists

The viewer treats a float32 `.npy` as **already metric** and does not apply
`depth_scale`. Raw DA3 output is not metric — it is relative, with an arbitrary
per-scene scale. So the calibration is baked into the maps and written as
float32 metres.

Two pieces go into that conversion, and they have very different status:

| | transfers between scenes? | why |
|---|---|---|
| **intrinsics** `fx = fy = 2905.6 px` | **yes** | focal length is a property of the camera and lens, solved from the 40 mm balls (R² 0.997) |
| **exponent** γ = 0.663 | probably | it is the shape of DA3's compression, consistent across every ball set |
| **scale** `a` | **no** | DA3 infers scale from scene content |

`ppx`, `ppy` are the image centre, read from an actual frame.

## The scale must be calibrated on THIS scene

Measured full-frame depth, same model, same camera:

```
ball calibration scene    p99 ≈ 1.03
cotton greenhouse scene   p99 ≈ 9.07
```

Nearly ten times apart, because one is a cloth backdrop about a metre away and
the other is a greenhouse with a roof many metres off. Carrying the ball
constant across gives bolls of ~90 mm, which is far too large.

**`LeftDown_v5/depth_metric` uses the ball constant and is provisional** — its
millimetres read roughly 2x high. **`left_down_v5_trimmed/depth_metric` is
calibrated against the 38 physically measured bolls** and is the one to trust:
its per-boll spread reproduces the true 21.2 % and its range matches the true
20.4–52.4 mm. Absolute scale there rests on one constant fitted to the measured
mean, so treat individual millimetres as good to roughly ±10 %.

## Fixing the scale — one measurement

Measure **one** boll physically, note its track ID in the viewer, then:

```bash
python scripts/depth/make_viewer_depth.py \
  --run inference_results/LeftDown_v5 \
  --ref_track 77 --ref_length_mm <measured mm>
```

Reload the depth folder. Every length rescales together. A measured
camera-to-boll distance works equally well:

```bash
python scripts/depth/make_viewer_depth.py --run ... \
  --ref_track 77 --ref_distance_mm <measured mm>
```

The maths: `Z_needed = L_true · f / px`, then `a = z_raw / Z_needed^γ`.

Better still, put a known-size object in shot for future captures — a ping-pong
ball clipped to a stake calibrates every frame it appears in, with no
foreshortening. See `findings/size_from_depth.md`.

## What to trust, and what not to

**Trust the relative numbers.** Ranking bolls, comparing one against another,
and tracking change over time are all unaffected by the scale constant — it
multiplies every length equally.

**Do not trust absolute millimetres until the scale is calibrated on this
scene.** Two separate reasons, both in `findings/boll_length/`:

- the scale is from the wrong scene (above);
- **86 % of boll depths fall outside the calibrated range.** The balls covered
  raw `z` 0.250–0.748; these bolls read 0.620–1.252. Even a correctly-scaled
  mapping is extrapolating for most measurements. A ball set at 0.7–1.3 m would
  fix this properly.

**Prefer the per-track report over single frames.** Per-frame depth noise is
about 5.2 % and the correction amplifies it — measured per-track spread rises
from 6.3 % (raw pixels) to 9.2 % (depth-corrected). Averaging over a track
removes that; the viewer's Boll report already does outlier-rejected
aggregation, which is the number to use.

**The 3-D label is noisier than the 2-D projection.** Recovering tilt needs a
tip-to-base depth difference of a few millimetres, well under DA3's noise at
504×280. Per-track CV was 34 % for the 3-D variant against 9 % for the
projection. The viewer already guards this by falling back to the mask axis when
the keypoint length disagrees by more than 30 %.

## Regenerating for another run

```bash
python scripts/depth/run_da3_depth.py --images_dir <run>/frames --out <run>/depth --no_viz
python scripts/depth/make_viewer_depth.py --run <run> --ref_track N --ref_length_mm M
```
