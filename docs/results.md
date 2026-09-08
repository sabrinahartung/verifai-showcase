# Current results

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
