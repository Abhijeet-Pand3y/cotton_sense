# Ball-set depth analysis - summary

Per-set detail is in `findings/per_set/`. This page holds what only appears when the sets are compared.

> **Method note.** gamma is fitted *inside* each frame and then aggregated, never from per-ball medians pooled across frames. DA3's output scale drifts frame to frame, so pooling first can pair a near-ball reading from one frame with a far-ball reading from another and corrupt the ratio gamma depends on. That is not a theoretical worry: doing it the pooled way gave 0.530 on the 19+27 set, where four of its five frames each say about 0.70.

## Focal length, solved from the 40 mm balls

`1/r = 2(Z + c)/(f*D)` over 116 ball-frames: **f = 2906 px**, tape offset c = +32.0 mm, R2 = 0.9970

- The earlier study assumed ~2450 px from lens specs; this is measured, and sits +18.6% from it.
- c is where the tape's zero sat relative to the entrance pupil - a constant bias in every stated distance.

## gamma per set

| set | balls | distances (in) | frames fitted | gamma (median) | sd | range |
|---|---|---|---|---|---|---|
| GX010117_ball11in_19in_27in_test | 3 | 11, 19, 27 | 4 | **0.640** | 0.020 | 0.605-0.650 |
| GX010117_ball17in_test | 1 | 17 | 0 | - (single distance) | - | - |
| GX010117_ball19in_27in_test | 2 | 19, 27 | 5 | **0.699** | 0.160 | 0.336-0.705 |
| GX010117_ball21in_25in_27in_test | 3 | 21, 25, 27 | 3 | **0.626** | 0.134 | 0.435-0.693 |
| GX010117_ball25in_test | 1 | 25 | 0 | - (single distance) | - | - |
| GX010117_ball9in_21in_test | 2 | 9, 21 | 5 | **0.685** | 0.053 | 0.666-0.797 |
| GX010117_ball9in_29in_test | 2 | 9, 29 | 2 | **0.613** | 0.088 | 0.551-0.675 |
| GX010117_ball9in_test | 1 | 9 | 0 | - (single distance) | - | - |
| GX010118_6ball_handheld_test | 6 | 9, 15, 17, 21, 25, 27 | 9 | **0.708** | 0.093 | 0.512-0.808 |
| GX010119_ball17in_test | 1 | 17 | 0 | - (single distance) | - | - |

- across sets: mean **0.662**, sd 0.040, n = 6 sets
- across all 28 individually fitted frames: median **0.678**, sd 0.102
- bootstrap 95% CI on the set-level mean: **[0.633, 0.691]**

gamma = 1 (no compression) is **outside** that interval, so the compression is established. `depth_findings.md` reported 0.670 from radius ratios with no metric ground truth; that value is inside this interval, an independent corroboration by a method sharing none of the same assumptions.

## Does a gamma from one set work on another?

Each set scored with the median gamma of the OTHER sets, against doing nothing (gamma = 1). Scale a is re-fitted per frame in both.

| set | n dist | cross gamma | mean err | mean err, gamma=1 | helps? |
|---|---|---|---|---|---|
| GX010117_ball11in_19in_27in_test | 3 | 0.685 | 2.27% | 11.34% | yes |
| GX010117_ball19in_27in_test | 2 | 0.640 | 2.81% | 6.67% | yes |
| GX010117_ball21in_25in_27in_test | 3 | 0.685 | 2.99% | 4.14% | yes |
| GX010117_ball9in_21in_test | 2 | 0.640 | 4.18% | 12.60% | yes |
| GX010117_ball9in_29in_test | 2 | 0.685 | 4.93% | 18.27% | yes |
| GX010118_6ball_handheld_test | 6 | 0.640 | 7.01% | 8.73% | yes |

The correction beats doing nothing on **6 of 6** sets.


![results](ball_depth_analysis.png)

## What this establishes

**The rig is trustworthy before DA3 is involved.** One focal length fitted over every ball reproduces the tape distances to about 1 %, R2 = 0.997, and re-solving it separately per set gives 2877-3023 px, a 1.8 % spread. Ball size, tape and pinhole model agree, so what follows is DA3's error, not the rig's.

**Compression is established against real distance.** The 95 % interval on gamma excludes 1 by a wide margin. This is the first time the exponent has been fitted here against tape-measured distance instead of radius ratios, and the earlier study's 0.670 falls inside the interval - two methods sharing no assumptions agreeing.

**The power-law shape now has a real test.** `depth_findings.md` asked for one frame with many spheres, because a two-ball fit draws a line through two points and cannot fail. The six-ball set supplies it: six distances fitted simultaneously in each of nine frames.

**The correction transfers.** Scored with an exponent fitted only on other sets, it beats doing nothing on 6 of 6 sets, one of them going from 11.3 % mean error to 2.3 %. gamma need not be exact to help - most of the benefit comes from moving away from gamma = 1 at all.

**But DA3 mis-orders close pairs, and that is not correctable.** In the six-ball set the 27 in ball reads nearer than the 25 in one in **every one of nine frames**, and the same pair inverts in the 21+25+27 set too. A power law is monotonic: it stretches an ordering, it cannot repair one. For cotton this is the binding limit, because comparing bolls at similar depth is the actual task and a 2 in gap at 25 in sits inside DA3's error.

## Verification

`scripts/depth/verify_ball_sets.py` re-derives all of this by a different route - separate sampler, per-frame rather than per-set fits, four disc sizes, three statistics - and reports:

- the two samplers agree exactly, so the image-to-depth-map mapping is right;
- gamma is insensitive to how the depth is sampled: 0.658-0.661 across discs from 40 % to 70 % of the ball radius, median or trimmed mean, so it is a property of the model and not of our disc;
- the moving object shifts gamma by less than 0.05 in every set with more than one object frame, consistent with the earlier finding that clutter adds noise rather than bias;
- the 25/27 in inversion is present frame by frame, not an artefact of averaging.

That verification is also what caught the one real error in the first pass: gamma had been fitted from per-ball medians pooled across frames, which mixes frames at different scales. Corrected, the 19+27 set moves from 0.530 to 0.699 and the six-ball set stops being an apparent failure of the correction.

## Limits

- DA3 returns 504x322 for a 4780x2988 frame, so the farthest balls are sampled over a disc about five pixels in radius. The ordering error sits exactly in that noisiest regime.
- The tape offset c is fitted, not measured; it is constant so it cannot create the compression, but the stated inches are nominal.
- Four of ten sets hold a single distance and cannot constrain gamma.
- Frames with only two balls give an exact fit each, so they measure how repeatable gamma is, not whether the power-law shape is right.
- One camera, one indoor scene, one backdrop. Nothing here says how gamma behaves outdoors, where the bolls are.

## Next

- Treat gamma as a **per-scene calibration**, not a published constant: it transfers well enough to help, but a reference sphere in shot would pin it per capture.
- Characterise the ordering failure directly - place spheres 1, 2 and 4 in apart at 25 in and find the gap DA3 can actually resolve. That number, not gamma, decides whether per-boll depth ranking is usable.
- Apply `depth^(1/gamma)` to the existing boll measurements and check whether size variability falls below the 5.8 % raw-pixel baseline, as `depth_findings.md` proposed.

- [GX010117_ball11in_19in_27in_test](per_set/GX010117_ball11in_19in_27in_test.md)
- [GX010117_ball17in_test](per_set/GX010117_ball17in_test.md)
- [GX010117_ball19in_27in_test](per_set/GX010117_ball19in_27in_test.md)
- [GX010117_ball21in_25in_27in_test](per_set/GX010117_ball21in_25in_27in_test.md)
- [GX010117_ball25in_test](per_set/GX010117_ball25in_test.md)
- [GX010117_ball9in_21in_test](per_set/GX010117_ball9in_21in_test.md)
- [GX010117_ball9in_29in_test](per_set/GX010117_ball9in_29in_test.md)
- [GX010117_ball9in_test](per_set/GX010117_ball9in_test.md)
- [GX010118_6ball_handheld_test](per_set/GX010118_6ball_handheld_test.md)
- [GX010119_ball17in_test](per_set/GX010119_ball17in_test.md)
