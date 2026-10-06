# Leakage audit — is the calibration cheating?

Short answer: **partly, yes.** Three of the four constants were fitted using the
very rows they were then used to score. One of those leaks matters a lot.

Run it yourself: `python scripts/depth/leakage_audit.py`

## The four constants

Every size estimate is `D = 2·r·(z/a)^(1/γ)/f`, with `Z_pupil = Z_tape + c`.

| constant | fitted on | leaks? | cost |
|---|---|---|---|
| `f`, `c` | all 116 ball-frames, **assuming D = 40 mm** | **yes** | small — ≤0.78% |
| `γ` | median of the **other** sets | no | — |
| `a` (global column) | all rows | **yes** | not separated |
| `a` (per-frame column) | **every ball in the frame, including the one being scored** | **yes** | **+1.8 points** |

## 1 · `f` and `c` — small leak

Refitting with each set held out moves `f` by at most **0.78%** (2882.9–2922.4 px
against 2905.6). Real, but not what is distorting the numbers.

## 2 · `a` per-frame — the leak that matters

The per-frame column was described as "a known-size object in the frame sets the
scale." But it fitted the scale using **every** ball in the frame — including the
ball being measured. Using a ball's own true distance to set the scale, then
measuring that ball, is circular.

The honest version uses a *different* object as the reference and predicts the
target:

| protocol | n | mean | sd | mean abs err |
|---|---|---|---|---|
| as shipped (`a` uses target too) | 95 | 40.24 mm | 2.83 | **5.22%** |
| **strict** (`a` from other balls, `f` leave-one-set-out) | 95 | 40.27 mm | 3.84 | **7.06%** |

**The leak was worth 1.8 points — the error was understated by about a third.**

It bites hardest where the target is a large share of the evidence:

| frame content | as shipped | strict |
|---|---|---|
| 2 balls | 4.20% | **7.38%** |
| 3 balls | 2.32% | 3.52% |
| 6 balls | 7.23% | 8.60% |

Two-ball frames nearly double. That is exactly as expected: with two balls the
target supplied half the information used to calibrate against it. Removing it
leaves one reference ball — which is the real deployment case.

## 3 · A mean of 40 mm is not evidence

`D/40 = Z_est/Z_true`, and **two independent fits both drive that ratio to 1**:

- `f` was chosen so `2·r·Z_tape/f` averages 40 on these balls;
- `a` is chosen so DA3's depths match the true distances in the frame.

So the mean landing on 40 is built in and should never be quoted as accuracy.
Only the **spread** carries information.

## 4 · Leaks that are not in the arithmetic

Worth naming, since they do not show up in any cross-validation:

- **Model choice on the same data.** The power law, the 0.6 inner disc, the
  median statistic and the QC thresholds were all picked while looking at these
  balls. The sensitivity check (γ = 0.658–0.661 across four disc sizes and two
  statistics) says the disc choice at least is not doing work, but the model
  family was still chosen with the answers visible.
- **Filter chosen after seeing results.** In the cotton analysis the "≥10 frames"
  cut was selected after the CVs were on screen. Flagged there, and it is why the
  least-selected row is the one quoted.
- **`D = 40 mm` is an assumption, not a measurement.** Nobody put callipers on
  these balls. Regulation says 40 mm; if the real batch is 39.5 mm, `f` absorbs
  it and every absolute number shifts by 1.25%.

## What to quote

**7.06%** is the leak-free per-frame figure. Against the 1.50% noise floor from
the closure check, DA3 is contributing roughly 6–7 points of error once the scale
is honestly calibrated.

The CSV now carries both: `D_perframe_mm` (leaky, kept for comparison) and
**`D_strict_mm`** (the one to use), with `Z_strict_mm` and `err_strict_pct`.
