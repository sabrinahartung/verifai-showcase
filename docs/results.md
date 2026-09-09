# Current results

!!! tip "Reading these numbers"
    Every metric here is defined in plain language in the [Glossary](glossary.md).

`skin_cancer_clean` — ResNet18 trained on a lesion-grouped split of HAM10000 and evaluated
on **1,493 images the model provably never saw**. First run in this project where every
verdict gate is active, so the first whose verdicts claim anything.

## Summary

| Pillar | Verdict | Result |
|---|---|---|
| Integrity | :material-check: **pass** | 0 shared lesions, 0 shared images across 1,493 test images |
| Performance | :material-check: **pass** | 79.6% top-1, 72.8% balanced |
| Privacy | :material-check: **pass** | membership-inference AUC 0.558 (0.5 = ideal) |
| Robustness | :material-alert: **warn** | 71.7% of predictions survive corruption |
| Fairness | :material-close: **fail** | 21.3-point accuracy gap across ITA skin-tone bins |

## What the headline number hides

```mermaid
%%{init: {'theme':'base'}}%%
xychart-beta
    title "Recall per class (n = 1,493)"
    x-axis ["nevi", "BCC", "vascular", "dermatofib.", "melanoma", "keratosis", "actinic ker."]
    y-axis "Recall" 0 --> 1
    bar [0.862, 0.855, 0.818, 0.769, 0.638, 0.586, 0.566]
```

| Class | Sensitivity | 95% CI | Specificity | PPV | Support |
|---|---:|:--:|---:|---:|---:|
| melanocytic_Nevi | 0.862 | [0.84, 0.88] | 0.909 | 0.952 | 1,009 |
| basal_cell_carcinoma | 0.855 | [0.76, 0.92] | 0.972 | 0.619 | 76 |
| vascular_lesions | 0.818 | [0.61, 0.93] | 0.997 | 0.818 | 22 |
| dermatofibroma | 0.769 | [0.50, 0.92] | 0.991 | 0.435 | 13 |
| **melanoma** | **0.638** | **[0.56, 0.71]** | 0.920 | **0.495** | 163 |
| benign_keratosis-like_lesions | 0.586 | [0.51, 0.66] | 0.958 | 0.622 | 157 |
| actinic_keratoses | 0.566 | [0.43, 0.69] | 0.972 | 0.422 | 53 |

Overall: top-1 **0.796** [0.775, 0.816], balanced **0.728**, and top-3 differential accuracy
**0.976** [0.967, 0.983].

!!! tip "Two things only visible once intervals and PPV are reported"
    **The differential is strong even where the top-1 call is not.** The correct diagnosis is
    in the model's top three for 97.6% of images. As a tool that proposes a ranked differential
    to a clinician — which is how dermatologists actually work — this is a far better system
    than 0.796 suggests. Reporting only top-1 understates it.

    **PPV is where melanoma really hurts.** When this model says *melanoma*, it is right about
    half the time (0.495). Sensitivity and PPV fail in opposite directions and a single accuracy
    figure hides both.

    And `dermatofibroma` at 0.769 has an interval of **[0.50, 0.92]** on 13 images — spanning
    half the possible range. Without the interval it reads as a mid-table result; with it, it
    reads as *unknown*.

!!! danger "80% accuracy, and it misses one melanoma in three"
    The clinically most important class is the second worst performer. The headline is
    dominated by the 1,009 nevi — 68% of the test set. This is the entire argument for
    per-class reporting, and for the balanced accuracy (72.8%) that sits seven points below
    the plain figure.

## Fairness

| ITA bin | Accuracy | Images |
|---|---:|---:|
| light (I–II) | 78.5% | 1,325 |
| medium (III–IV) | 96.3% | 108 |
| dark (V–VI) | 75.0% | 60 |

| ITA bin | Accuracy | 95% CI | Images |
|---|---:|:--:|---:|
| light (I–II) | 0.785 | [0.76, 0.81] | 1,325 |
| medium (III–IV) | 0.963 | [0.91, 0.99] | 108 |
| dark (V–VI) | 0.750 | [0.63, 0.84] | 60 |

The 21.3-point gap exceeds the 15-point `fail` threshold, and the intervals for the two extreme
groups do **not** overlap, so the difference is supported rather than noise.

!!! warning "But it is not the gap you would assume"
    The supported gap runs between **medium (0.963) and dark (0.750)** — driven by the small
    medium bin scoring unusually *high*, not by dark skin collapsing. The comparison most people
    expect, **light vs dark**, has overlapping intervals ([0.76, 0.81] against [0.63, 0.84]) and
    is therefore **not** a demonstrated difference on this data.

    89% of the test set falls in one bin, which is HAM10000's documented skew. ITA is also
    estimated from pixels rather than clinically assessed. Reporting the gap without its
    intervals would have supported a confident and probably wrong story.

## Robustness

| Corruption | Predictions unchanged | 95% CI |
|---|---:|:--:|
| brightness ×1.4 | 77.6% | [75.4, 79.7] |
| JPEG q=25 | 73.8% | [71.5, 76.0] |
| Gaussian blur r=2 | 71.7% | [69.4, 74.0] |
| noise σ=18 | 63.8% | [61.3, 66.2] |

Mean 71.7%, below the 85% `pass` threshold. Roughly one prediction in three flips under
noise that does not change the diagnosis.

## Privacy

Membership-inference AUC **0.5577** [0.529, 0.587] from 750 training and 750 held-out images.
The verdict is taken on the interval's upper bound, not the point estimate — an AUC of 0.59
whose interval reached 0.68 would not have been shown to be low risk. Mean
confidence in the true class was **0.873** on training images against **0.777** on unseen
ones — a real but small separation, landing in the "low risk" band.

## Explainability

Mean deletion faithfulness **0.45** over 7 Grad-CAM overlays: masking the highlighted region
costs the predicted class 45 percentage points of confidence on average, which puts it in
the "partly faithful" band. Reported as `info` — there is no defensible universal threshold
for "faithful enough", and 7 overlays is an illustration rather than a measurement.

## Experiment 1 — a cost-sensitive decision rule

**No retraining.** Identical weights; only the rule that reads the probabilities changed.
`argmax` maximises expected accuracy, which on imbalanced data means under-calling rare
classes. Scaling melanoma's probability by *w* lets it win against a nevus it would otherwise
lose to. The weight was chosen by sweeping on the **validation** manifest — tuning it on test
would be fitting the decision rule to the test set, a quieter form of leakage that the
integrity check would not catch.

| Run | Melanoma sensitivity | Melanoma PPV | Nevi sensitivity | Top-1 | Balanced | Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| baseline (argmax) | 0.638 [0.56, 0.71] | 0.495 | 0.862 | **0.796** | 0.728 | 0.976 |
| melanoma ×5 | 0.804 [0.74, 0.86] | 0.379 | 0.784 | 0.754 | 0.721 | 0.979 |
| melanoma ×50 | **0.945** [0.90, 0.97] | 0.264 | 0.640 | 0.656 | 0.673 | 0.974 |

**Melanoma sensitivity rose from 0.638 to 0.945 — from missing one melanoma in three to
missing one in eighteen — without touching the model.** The cost is explicit: PPV falls from
0.495 to 0.264, so three in four melanoma flags become false alarms, and nevi sensitivity
drops from 0.862 to 0.640.

!!! danger "The result that justifies this whole pillar redesign"
    Top-1 accuracy **falls** from 0.796 to 0.656, and the performance verdict goes from
    `pass` to `warn`. An evaluation that measured only accuracy would have **rejected** the
    change that made the model dramatically better at catching cancer.

    This is not a hypothetical. It is why the clinical metrics had to be built before the
    first experiment, and it is the concrete argument against a single headline score.

Note also that top-3 accuracy is nearly unmoved (0.976 → 0.974): reordering the top-1 barely
disturbs which three diagnoses are in contention. As a ranked differential, all three
configurations are equally good — they differ only in what they *commit to*.

**Which one is correct?** None of them, until an intended use is declared. For a rule-out tool
("this is not cancer, go home"), 0.638 sensitivity is indefensible and ×50 is arguably still
too low. For a tool that reorders a dermatologist's worklist, the baseline's precision may be
worth more than the recall. Same model, same test set, opposite conclusions — which is exactly
why this project refuses to emit one number.

!!! note "Honest caveat on transfer"
    The weight was tuned for 0.85 melanoma sensitivity on validation and delivered 0.945 on
    test — better than targeted, and the intervals do not overlap (val [0.79, 0.90] against
    test [0.90, 0.97]). The operating point transferred, but validation and test evidently
    differ somewhat in melanoma difficulty, so a tuned threshold should not be assumed to
    carry over exactly.

## Experiment 2 — focal loss and oversampling (a negative result)

Two standard remedies for class imbalance, each changing exactly one thing against the
baseline, both trained on the same manifests and evaluated on the same frozen test set.
Oversampling and class weighting are competing answers to the same problem, so the
oversampling variant turns class weighting **off** rather than stacking them.

| Configuration | Melanoma sensitivity | Melanoma PPV | Top-1 | Balanced | Top-3 |
|---|---:|---:|---:|---:|---:|
| baseline (CE + class weights) | 0.638 [0.56, 0.71] | 0.495 | 0.796 | 0.728 | 0.976 |
| focal loss γ=2 + weights | 0.497 [0.42, 0.57] | **0.540** | 0.784 | 0.698 | 0.976 |
| balanced oversampling, no weights | 0.656 [0.58, 0.72] | 0.448 | 0.785 | 0.685 | 0.976 |
| *baseline + melanoma ×5 (rule only)* | *0.804 [0.74, 0.86]* | *0.379* | *0.754* | *0.721* | *0.979* |
| *baseline + melanoma ×50 (rule only)* | ***0.945** [0.90, 0.97]* | *0.264* | *0.656* | *0.673* | *0.974* |

**Neither training intervention produced a demonstrated improvement.** Focal loss *lowered*
melanoma sensitivity by 14 points and oversampling raised it by 1.8 — and in both cases the
intervals overlap the baseline's, so neither difference is established. Validation balanced
accuracy was 0.724 / 0.724 / 0.717: from the aggregate alone, nothing happened at all.

!!! note "The comparison that makes this worth reporting"
    A **free change to the decision rule** — no retraining, no new data — moved melanoma
    sensitivity from 0.638 to 0.945. Two days of standard imbalance engineering moved it by an
    amount indistinguishable from noise.

    Focal loss did do something, just not the intended thing: it made the model *more*
    conservative, buying the best melanoma PPV of any configuration (0.540) at the cost of
    sensitivity. That is a legitimate trade, but it is the opposite of what it was reached for.

**Top-3 accuracy is 0.974–0.979 across all five configurations.** None of these interventions
changed what the model *knows* — the correct diagnosis sits in its top three just as often
either way. They only change where it draws the line for its single committed answer. That is
precisely why a decision rule outperformed retraining here, and it suggests the next real gain
has to come from more information (more minority images, external data), not from reshaping
the same loss surface.

**No configuration dominates any other** across the eight compared metrics — five of five
survive as genuine trade-offs, each best at something.

## Training run

12 epochs, 4.6 minutes on MPS, batch size 32, lr 1e-4, class-weighted cross-entropy.

| Epoch | Train loss | Val accuracy | Val balanced |
|---:|---:|---:|---:|
| 1 | 1.081 | 0.670 | 0.652 |
| 4 | 0.462 | 0.737 | 0.708 |
| 8 | 0.303 | 0.755 | 0.707 |
| 11 | 0.188 | **0.807** | 0.698 |
| 12 | 0.213 | 0.792 | **0.724** ← kept |

!!! note "Why epoch 12 and not epoch 11"
    Epoch 11 had the best plain accuracy. Selection runs on **balanced** accuracy, because on
    a set that is 67% nevi, plain accuracy rewards a model for ignoring the rare classes.

## Reproducing

```bash
python scripts/materialize_images.py --max-size 320
python scripts/train_model.py scenarios/skin_cancer_clean.yaml
python scripts/run_scenario.py scenarios/skin_cancer_clean.yaml
```

Seed 42 throughout; the split manifests are committed. Results are bit-identical per device
but not across devices — `report.json` records `meta.device` for that reason.
