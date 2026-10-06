# How well can an AI depth model measure the size of an object in a photo?

*Applied Vision Lab · Texas Tech · 20 Aug 2026*

**Why this was asked.** We want to measure cotton bolls from ordinary GoPro
video. A photo alone cannot give size — a small object close to the camera and a
large object far away produce the identical picture. Size only follows if the
distance to the object is known, and an AI depth model claims to estimate that
distance from a single ordinary photo.

**What this document covers.** The controlled experiment that measures how well
that works, using ping-pong balls of known size at tape-measured distances.
Nothing here is cotton: the numbers below are what the method achieves under
laboratory conditions, and they set the ceiling for anything we do in the field.

> **Cotton results are deliberately not in this package.** The scale constant
> does not transfer between scenes — the balls prove it, and a first pass on
> greenhouse footage confirmed it. Real boll measurements need calibrating
> against physically measured bolls first, and that work is not finished. What
> follows is the part that is solid.

**Answer, in one line.** With one pre-measured object in the shot the method
measures a 40 mm sphere to about **±3.6 mm**. With nothing of known size in the
shot the error more than doubles and the scale cannot be recovered afterwards.

| what is in the picture | error on a 40 mm object |
|---|---|
| nothing of known size | ± 6.9 mm  |
| **one pre-measured object** | **± 3.6 mm** |
| several pre-measured objects | ± 2.6 mm |

**Since first writing, one further recording has been measured** — made later,
with nothing of known size in it, and with none of the constants fitted on it. It
came out at ± 4.4 mm, no worse than the table above, but wrong in one direction
on every single frame. Section 6.

---

## 1 · How the measurement works

Three ingredients, and only the middle one is uncertain.

**Pixel size** — how many pixels across the object is. Measured directly and
well: fitting a circle to a ball's edge is accurate to about 1.7%.

**Distance** — from Depth Anything 3 (DA3), which reads a single frame and
returns a depth picture. Two properties of it drive everything below:

- it is **relative**, not absolute. It reliably says which object is nearer, but
  has no idea whether the scene is a tabletop or a football field. Something of
  known size in the picture has to fix the scale.
- it **compresses** distance. Far objects are reported as closer than they really
  are. Measured here: a ball genuinely **1.75× farther** away was reported as
  only **1.45× farther** (the 15″/27″ pairs, 9 photos, median reported ratio
  1.45). Uncorrected, this makes distant objects read small and near ones read
  large.

**Focal length** — a fixed property of the camera, measured here at **≈2900
pixels** for this GoPro. The per-measurement spread is ±80 px, so three
significant figures is as far as the data goes.

Put together: `size = pixel size × distance ÷ focal length`. The compression is
undone by raising the depth to a power, `depth^(1/γ)`, with **γ ≈ 0.7**. Below 1
means compression; γ = 1 means none. Whether that one number earns its place is
Section 3.

## 2 · The experiment

A GoPro was **clamped so it could not move**. Ping-pong balls — every one exactly
**40 mm** across — were placed at **tape-measured distances from 9 to 29 inches**
and left untouched. In some clips an extra object was moved in and out of the
shot, in front of and behind the balls, to imitate the clutter of a real scene.
**116 ball measurements across 10 recordings.**

Because the balls never move and their true size is known, every estimate can be
graded exactly: any variation in measured size is error.

A ping-pong ball is also a fair stand-in for what we eventually want to measure —
a cotton boll is roughly 40 mm tip-to-base, at comparable distances from the
camera — so the accuracy measured here should transfer in magnitude, even though
the scale constant does not.

![ball experiment results](ball_results.png)

## 3 · Results

### The scale has to come from somewhere

Three situations, differing only in what was available to fix the scale:

| what is in the picture | size error on a 40 mm ball | n |
|---|---|---|
| nothing of known size | **17.0%** | 116 |
| **one object of known size** | **8.9%** | 302 pairings (≈95 independent) |
| several objects of known size | **6.4%** | 95 |
| *tape distance instead of the AI model* | *1.7%* | *116* |

That last row is the **floor**: repeat the calculation using the tape reading
instead of DA3 and 1.7% remains, from finding a ball's edge in the image. No
depth method could beat it here. Everything above it is DA3's contribution.
(Every figure in this table is a mean absolute error, scored with the same
held-out constants.)

**Putting one pre-measured object in the shot halves the error.** Adding more
barely helps. This is the single most important practical finding — it is the
difference between a usable measurement and an unusable one.

> **A note on the 8.9%.** Every ball was scored once against *each* other ball
> that could have served as the reference, and 8.9% averages over all those
> choices — including bad ones. **The choice matters:** a reference at a similar
> distance to the target works much better than one much nearer or farther. If
> instead you combine all the available references and take the middle answer,
> the error falls to 6.7%, which is really a mild form of the several-references
> case. **8.9% is the honest figure for “one reference, chosen without care”**,
> and the CSV's best-case/worst-case columns let you probe how much the choice
> costs.

### Does the close range flatter these numbers?

Anything we measure in the field will sit further away than the nearest balls, so
it matters whether the error is concentrated at the close end. **It is not.**
Re-scoring with the near balls excluded — as target *and* as reference — barely
moves anything:

| balls included | nothing of known size | one reference | several references |
|---|---|---|---|
| all, 9–29 in (23–74 cm) | 17.0% | 8.9% | 6.4% |
| **beyond 11 in (> 28 cm)** | 16.1% | **8.6%** | 5.8% |
| beyond 21 in (> 53 cm) | 14.7% | **10.2%** | 7.3% |

![error by distance](ball_results_by_distance.png)

- **Dropping the close balls gains about 0.3 points.** The headline numbers are
  not being dragged down by unrealistically close measurements, so quote them as
  they stand.
- **The far band is *worse*, not better** — 8.6% → 10.2% with one reference. That
  band is dominated by the 25″/27″ pair DA3 ranks in the wrong order, and it is
  the band nearest real working distance. So if anything the headline is mildly
  *optimistic* for the distances we care about, not conservative.

The real caveat is coverage: the farthest ball sits at 74 cm, and the experiment
says nothing beyond that. Re-calibrating at true working distance is on the list
in Section 5.

### The compression correction is real

Turn the correction off (γ = 1) and the error nearly doubles: **6.4% →
11.4%**. This is measured on balls the calculation had never seen, so it is not
curve-fitting.

It does **not** need precise tuning. Anything from γ = 0.70 to 0.80 gives
6.0–6.5%, and a value worked out on one recording transfers usefully to another:
scored with a γ fitted only on *other* recordings, the correction beat doing
nothing on **6 of 6** recordings.

### Clutter in the scene costs about 2.5 points

Frames with a distractor object present average **9.6%** against **7.2%** for
clean frames (one reference; 7.5% against 4.6% with several). Where the object
sits is what matters:

| object position | error |
|---|---|
| 5 in from the camera, well in front of the ball | 11.0% |
| out at the ball's own depth (27 in) | 4.1% — essentially clean |

A distractor close to the lens and well in front of the target corrupts the
reading; one at the target's own depth barely registers. The single worst
measurements in the dataset are all of this kind — `obj5in_t198s` (22.5%),
`obj9in_t313s` (19.5%), `obj5in_t028s` (19.4%).

### What no correction can fix

DA3 reported a ball at **27 inches as nearer than one at 25 inches in 10 of the
12 photos** where both were in shot — all 9 photos of one recording, and 1 of 3
in another. A power law stretches an ordering; it cannot repair one. DA3 cannot
reliably rank two objects a couple of inches apart in depth, and no amount of
calibration changes that. In the six-ball recording, the 25 in and 27 in balls
are the two worst measured of the six for exactly this reason.

## 4 · What this means for capture

**Two rules, from here on**

1. **Put a pre-measured object in every shot** — a ping-pong ball or a measured
   tag clipped to a stake, at roughly the distance of whatever is being measured.
   Worth more than any other change on this list, and it costs nothing.
2. **Keep obvious distractors out of the near foreground.** Something close to
   the lens and in front of the subject is what corrupts the reading; clutter at
   the subject's own depth is harmless.

**What the method supports**

- Population statistics — distributions, means, group comparisons. Measurement
  noise of ~9% widens an apparent 20% population spread only to about 22%.
- Ranking obviously large objects against obviously small ones.
- Tracking one object's change over time, where it is compared against itself.

**What it does not support**

- Absolute size of an individual object to better than a few millimetres.
- Separating two objects of similar size at similar depth in one frame.
- Any measurement at all without a known-size object in the shot.

## 5 · Honest limits, and what comes next

- **Every number was produced by prediction, not fitting.** For each recording,
  the camera constants, the depth curve and the scale were derived from the
  *other* recordings and then applied. An earlier version let each ball help set
  its own scale, which made the method look about a third better than it was
  (5.2% where the truth was 7.1%). That is corrected throughout.
- **The 40 mm is assumed, not measured.** Regulation size; nobody put callipers
  on the batch. A 39.5 mm batch shifts every absolute number by ~1%. No
  percentage, spread or comparison depends on it.
- **The scale constant does not transfer between scenes.** DA3 reads scale off
  scene content, so a value fitted in one setting is wrong in another. This is
  why a reference object in shot is not optional, and it is the specific reason
  the cotton work is not in this package yet.
- **One camera, one lens setting, one indoor scene.** Changing camera or zoom
  means redoing the camera calibration. Nothing here says how the method behaves
  outdoors.
- **The focal length matters for absolute sizes only.** A 5% error costs about
  2 points of size error, and it cancels entirely when a reference object sets
  the scale. Worth pinning with a proper camera calibration before quoting
  absolute millimetres.

**Next, in priority order**

1. **Put a known-size reference in shot from the next capture onward.** It halves
   the error and is the difference between recoverable and unrecoverable scale.
2. **Re-calibrate at true working distance.** The balls stop at 74 cm; the far
   band is already the worst one, so the error does not conveniently shrink out
   there.
3. **Characterise the ordering failure directly.** Place spheres 1, 2 and 4
   inches apart at 25 inches and find the smallest depth gap DA3 can resolve.
   That number, not γ, decides whether depth ranking of nearby objects is usable.
4. **Calibrate against physically measured subjects**, which is what turns any of
   this into an absolute accuracy figure rather than a consistency one.
5. **Test outdoors**, one scene property at a time — backdrop distance first.

## 6 · Held out: a recording the constants had never seen

Everything above was cross-validated *within* one body of footage. The obvious
question is what happens on a recording made later, that no constant was fitted
on. `GX010120` is that recording: one 40 mm ball on a stand, a water bottle
brought to 5, 11, 15, 19, 25 and 29 inches, and two frames of the ball alone as a
clean base. The ball's own distance was never tape-measured, so this set can only
be read in the hardest of the three setups — **nothing of known size in the
picture**. `f`, `c`, γ and the scale were all taken from the 116 measurements
above and applied unchanged.

![new data](noref_new_data.png)

**The headline.** The method returned **44.4 ± 3.1 mm** for a 40 mm ball — a mean
error of **11.1% (4.4 mm)**, against **17.0% (6.8 mm)** for the same setup in the
study above. That looks like an improvement and should not be read as one:
Welch t = −2.03, **p = 0.07**, on eight correlated frames of a single ball at a
single distance. The fair summary is *the new recording is not worse*.

**The result that matters is the sign.** In the study above the errors are
roughly balanced — 54% positive, signed mean −2.1%. Here **all eight frames
over-estimate**, signed mean +11.1%. A one-directional error is what a scale
constant looks like when it does not transfer to a new scene, and it is the same
failure this document already reports for the greenhouse footage. Cross-validation
inside one body of footage flatters the scale; a genuinely new scene does not.

**Compared with base.** The two ball-alone frames and the six bottle frames are
the same ball, so any difference is the bottle's doing. There isn't one that can
be told from noise: bottle-present reads +2.1 mm against ball-alone, while the
two ball-alone frames — one second apart, nothing changed — already differ by
2.7 mm. The bottle's distance does not predict the error either (r² = 0.07,
n = 6).

**What does move is the depth.** Across eight frames of a ball that never moved,
the fitted pixel radius holds to **1.4%** while DA3's output swings **14.7%**,
and the measured size spans **40.1–49.1 mm — 22.6% of the true size**. The
picture measurement is not the bottleneck; the depth is. Panel C makes the point
directly.

**Distance is cross-checked, not scored.** With no tape on the ball there is no
ground truth for distance. Its pixel radius puts it at **497.8 ± 2.7 mm
(19.6 in)** by geometry alone; DA3 with no reference says 588.4 mm, **+18.2%**,
on every frame. Which is right cannot be settled here — and note that this is the
size error restated, not a second piece of evidence, since size is `2·r·Z/f`.

Full per-set write-up: `findings/per_set/GX010120_test.md`.

## 7 · The files here

| file | what it holds |
|---|---|
| `README.md` | this document |
| `ball_measurements.csv` | every ball measurement — 116 rows, one per ball per photo |
| `ball_measurements_columns.md` | what each column means |
| `ball_error_by_distance.csv` | the nine numbers behind the distance-band figure |
| `ball_results.png` · `ball_results_by_distance.png` | the two figures above |
| `noref_new_data.csv` | the held-out recording in section 6 — 8 frames, one ball, no tape |
| `noref_new_data.png` | the section 6 figure |

**The constants**, for anyone re-deriving the numbers:

| constant | value | what it is |
|---|---|---|
| `f` | 2905.6 px *(≈2900 ± 80)* | camera focal length, solved from ball size and tape. The four figures are what the fit returned; the per-measurement spread is ±80 px and the value minimising size error is 2895, so treat it as ~2900. |
| `c` | +32.0 mm | offset from camera body to lens, added to every tape reading |
| `γ` | 0.730 | compression exponent; the optimum is broad, anything in 0.70–0.80 performs the same |
| scale | scene-specific | must be re-fixed per scene — the one that does not transfer |

Regenerate everything from the repository root:

```bash
python scripts/depth/final_validation.py        # the numbers
python scripts/depth/plot_team_results.py       # ball_results.png
python scripts/depth/plot_distance_bands.py     # ball_results_by_distance.png
python scripts/depth/build_team_csv.py          # ball_measurements.csv
python scripts/depth/analyse_noref_set.py --set frames/GX010120_test   # section 6
python scripts/depth/plot_noref_set.py          # noref_new_data.png
```

Fuller technical write-ups, including the failed attempts and why they failed,
are in `findings/` — `final_validation.md`, `leakage_audit.md`,
`ball_depth_analysis.md` and `size_from_depth.md`.
