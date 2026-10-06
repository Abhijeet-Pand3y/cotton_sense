# Boll length on the trimmed set, against 38 measured bolls

![results](trimmed_boll.png)

## Your diagnosis was right

Cropping the roof out fixes the scale problem. DA3 has no absolute scale — it
reads it off scene content — and the greenhouse roof was pulling the whole map
into a different regime:

| scene | DA3 depth, 1st–99th pct |
|---|---|
| ball calibration | 0.37 – 1.01 |
| cotton, **roof in frame** | 0.78 – **9.07** |
| cotton, **top 15% cropped** | 0.56 – 1.23 |

Boll readings inside the calibrated range went from **13.8% → 67.5%**. The
mapping is no longer extrapolating for most measurements, which is what made the
earlier absolute numbers meaningless.

Cropping 15% (797 px) costs almost nothing: 0.3% of keypoints, **zero** bolls
lost entirely. Depth is inferred on the crop and the strip is padded back before
the viewer sees it, so detection coordinates stay valid.

## Ground truth

38 bolls. **"Height" is the tip↔base axis** — its CV of 21.2% is exactly the
figure `depth_findings.md` quotes as the true spread, which confirms the column.

| | mean | sd | CV | range |
|---|---|---|---|---|
| Height (tip↔base) | **40.46 mm** | 8.59 | **21.2%** | 20.4–52.4 |
| Length | 26.89 mm | 5.19 | 19.3% | 17.6–35.5 |
| Width | 26.73 mm | 4.81 | 18.0% | 18.1–35.7 |

Track IDs are not matched to boll IDs, so this compares **distributions**, not
boll against boll.

## Result 1 — the blind prediction is closer, still 38% high

Applying the ball calibration untouched, with nothing here having seen the
ground truth:

- predicted mean **55.91 mm** against a true **40.46 mm** → **+38.2%**

Much better than the roof-in run (which gave ~90 mm), but not usable as an
absolute. The residual is the scale constant: even trimmed, this scene reads
0.53–0.93 where the balls read 0.25–0.75, so DA3 still places the greenhouse
slightly deeper. **Scale still does not transfer between scenes** — consistent
with everything measured earlier. One measured boll fixes it; the factor here is
0.724.

## Result 2 — the spread matches, and depth beats pixels

With the scale fitted to the GT mean, the mean agrees *by construction* and
proves nothing. The **spread** is the real test — one scale factor cannot fake
it.

| tracks kept | depth-corrected CV | pixels-only CV | range (mm) |
|---|---|---|---|
| all 18 | 33.0% | 39.2% | 10.1–61.4 |
| **≥10 frames (12)** | **21.6%** | 27.6% | 23.3–53.0 |
| ≥10 fr, visfrac ≥0.8 (10) | 17.6% | 27.7% | 29.1–52.0 |
| ≥20 fr, visfrac ≥0.9 (7) | 17.4% | 29.6% | 29.5–49.3 |
| **ground truth (38)** | **21.2%** | — | 20.4–52.4 |

Two things stand out:

**Depth correction beats raw pixels at every filter level**, by 6–12 points.
That is the first time on real cotton that depth has demonstrably helped —
`findings/boll_length/` found the opposite on the untrimmed run, and the trim is
the difference.

**At the least-selected filter (≥10 frames) the measured spread is 21.6% against
a true 21.2%**, and the range 23.3–53.0 mm sits almost exactly on the true
20.4–52.4 mm. Panel C shows the quantiles tracking the diagonal.

The tighter filters give *lower* CV than ground truth (17.6%, 17.4%). That is
not better measurement — it is selection. Dropping short and occluded tracks
also drops small and awkward bolls, narrowing the population being sampled. The
≥10-frame row is the fair comparison.

## In the viewer

Metric depth is generated and ready:

```
inference_results/left_down_v5_trimmed/depth_metric/depth   (117 maps + metadata.json)
```

Load frames → detections → that depth folder. Reported lengths have a median of
**46.0 mm**, IQR 39.6–53.0, against a true mean of 40.46 mm and range 20.4–52.4.
The padded strip reads 0, so any detection above the crop line is skipped rather
than silently given a wrong depth.

Regenerate after measuring a real boll:

```bash
python scripts/depth/make_viewer_depth.py --run inference_results/left_down_v5_trimmed \
  --ref_track N --ref_length_mm M --pad_crop
```

## Limits

- **18 tracks, 12 usable**, against 38 measured bolls. The clip shows only part
  of the plants, and the distribution comparison rests on 12 points.
- **One free parameter came from the ground truth** (the scale). The mean is
  therefore not evidence; only the spread and the shape are.
- **Tracking is a simple greedy IoU tracker** written for this analysis, because
  the per-frame JSONs carry no track IDs. A boll split across two tracks would
  appear twice.
- IDs are unmatched, so no per-boll error can be quoted — only distributions.
  Matching tracks to tag numbers is what would turn this into a real per-boll
  accuracy figure, and the tags are legible in these frames.
- Still one camera, one session, one lens setting assumed.

## What this changes

The earlier conclusion — that depth correction makes boll size *worse* — was an
artefact of the roof breaking DA3's scale, not a property of the method. On
clean frames the correction reduces spread by 6 points and reproduces the true
distribution. The remaining gap is absolute scale, and that needs one physical
measurement per scene, or a known-size object in shot.
