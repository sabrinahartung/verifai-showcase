# Glossary

Every term this project uses, in plain language, with a real example from the
[current results](results.md) — the 1,493-image held-out run.

---

## Getting the answer right

**The three questions people confuse.** All three are about melanoma, all three are different:

| Question | Metric | This model |
|---|---|---:|
| Of the real melanomas, how many did we **catch**? | sensitivity | 0.638 |
| Of everything that wasn't melanoma, how often did we correctly **not** say melanoma? | specificity | 0.920 |
| When the model **says** melanoma, how often is it right? | PPV | 0.495 |

You could score a perfect sensitivity of 1.0 by calling *every* image melanoma — you'd catch
all of them. But then PPV collapses and every patient gets an unnecessary biopsy. Sensitivity
counts **missed cancers**; PPV counts **false alarms**. They fail in opposite directions.

Accuracy (top-1)
:   How often the model's single best guess is correct. **0.796** here. Simple, and misleading
    on unbalanced data: 1,009 of the 1,493 test images are nevi, so this mostly measures how
    well the model recognises common moles.

Balanced accuracy
:   The same idea, but averaged over *classes* instead of images, so melanoma counts as much as
    nevi. **0.728** — seven points below the headline. That gap *is* the class imbalance,
    made visible.

Sensitivity (recall, true positive rate)
:   Of the cases that really are X, how many did the model find? **Melanoma: 0.638** — it finds
    64 of every 100 melanomas and misses 36.

Specificity (true negative rate)
:   Of the cases that are *not* X, how many did the model correctly not call X?
    **Melanoma: 0.920.**

PPV (positive predictive value, precision)
:   When the model says X, how often is it actually X? **Melanoma: 0.495** — right about half
    the time. Depends heavily on **prevalence** (below), so a PPV measured on one population
    does not transfer to another.

NPV (negative predictive value)
:   When the model says *not* X, how often is that right? The mirror of PPV.

Top-3 (differential) accuracy
:   Is the correct diagnosis among the model's three best guesses? **0.976.** This matches how a
    dermatologist actually uses a suggestion — as a ranked shortlist, not a verdict — and it is
    why a seven-class model is worth keeping. Top-1 alone *understates* this system.

Support
:   How many test images a class has. **melanoma 163, dermatofibroma 13.** Always read a score
    next to its support.

Confusion matrix
:   A grid: rows are the true diagnosis, columns are what the model said. A perfect model is a
    bright diagonal. The bright cells *off* the diagonal are the confusions that matter — read
    the melanoma row to see what melanomas get mistaken for.

Prevalence
:   How common a condition is in the population you are testing. A screening clinic and a
    referral centre differ enormously. **Why it matters:** at sensitivity 0.85 and specificity
    0.90, PPV is 0.49 when prevalence is 10% and only 0.08 when it is 1% — same model, same
    sensitivity, wildly different usefulness.

---

## How sure are we?

Confidence interval (CI)
:   The range the true value plausibly sits in. Flip a coin ten times, get seven heads — you
    don't conclude the coin is 70% heads. Ten flips isn't enough to know, and the interval says
    so.

    **The comparison that makes this concrete:**

    | Class | Score | 95% CI | Images |
    |---|---:|:--:|---:|
    | melanocytic_Nevi | 0.862 | [0.84, 0.88] | 1,009 |
    | dermatofibroma | 0.769 | [0.50, 0.92] | **13** |

    Both look respectable in a plain table. But dermatofibroma rests on 13 images, so the truth
    could be anywhere from a coin flip to excellent. **That number is not a result, it's a
    shrug.**

The overlap rule
:   **If two intervals overlap, you have not shown the two things differ.** This changed a
    conclusion here: everyone assumes the fairness story is "worse on dark skin", but light
    [0.76, 0.81] and dark [0.63, 0.84] **overlap** — so on this data that claim is *not
    demonstrated*. The gap that is real runs between medium and dark.

Wilson score interval
:   The specific method used for proportions. The textbook "normal approximation" misbehaves
    exactly where these metrics live — small samples, and scores near 0 or 1, where it can
    produce impossible bounds below 0 or above 1. Wilson doesn't. **n=7 with 7 correct** reads
    `1.000 [0.65–1.00]` — perfect on paper, deeply uncertain in fact.

AUC (area under the ROC curve)
:   How well a score separates two groups. **0.5 = no better than guessing; 1.0 = perfect
    separation.** Used here for membership inference, where 0.5 is the *good* outcome.

---

## The data

Data leakage
:   When information from the test set was also available during training, so the model is
    partly *recalling* rather than generalising. The score comes out high and nothing crashes.
    **In the dataset this project started from: 80% of test images also appeared in training.**

Lesion-level grouping
:   HAM10000 photographs the same lesion several times. Splitting on *images* still puts a
    second photo of a memorised lesion into the test set — which is not a fair question. So the
    split is made on `lesion_id`: every photo of one lesion lands on the same side.

Stratified split
:   Keeping each class in the same proportion across train/val/test, so rare classes don't
    vanish from the test set. Here: 11–16% of every class landed in test, including
    **163 melanomas**.

Manifest
:   A small CSV listing exactly which images belong to which split, with labels and metadata.
    Committed to git, so "what did the model see" is a file you can diff rather than a claim
    you have to trust.

Train / validation / test
:   **Train** = learned from. **Validation** = used to choose between checkpoints (so the model
    is tuned to it — it counts as *seen*). **Test** = touched once, at the end. Here:
    7,014 / 1,508 / 1,493 images.

Distribution shift
:   When deployment data differs from training data — different scanner, clinic, or population.
    The commonest reason a medical model that scored well in one hospital fails in another.

Class imbalance
:   When some classes are far commoner than others. 67% of this data is nevi. Untreated, a
    model learns to say "nevi" and scores well doing it — which is why training uses **class
    weights** and why *balanced* accuracy is reported.

---

## How the model decides

Decision rule
:   How probabilities become an answer. Not a law of nature — a choice.

argmax
:   The default rule: pick whichever class has the highest probability. It maximises expected
    accuracy, which is *not* the same as being clinically useful, and it systematically
    under-calls rare classes.

Operating point / threshold
:   Choosing a different bar — e.g. flag melanoma when its probability exceeds τ, even if
    another class scores higher. Trades sensitivity against PPV, and the right trade depends on
    intended use. This is the next planned experiment.

Cost-sensitive decision
:   Weighting classes by the cost of missing them: `argmax(p × w)`. Still returns one of seven
    classes; melanoma simply clears a lower bar.

Calibration
:   Whether stated confidence matches reality — of all the cases called "80% melanoma", are
    about 80% melanoma? A model can be accurate and badly calibrated, which is dangerous in
    deployment.

Class weights · oversampling · focal loss
:   Three ways to stop a model ignoring rare classes: penalise their mistakes more, show them
    more often, or focus the loss on hard examples. All reshape the loss; **none add
    information** — which is why more real minority images has a higher ceiling than any of
    them.

---

## The pillars

Grad-CAM
:   A heatmap of which image regions drove the decision. Warm colours = influential. You want
    the heat on the lesion, not on hair, rulers, or the image border.

Deletion faithfulness
:   Checks whether the heatmap is *honest*: mask the highlighted region and measure how far
    confidence falls. A big drop means the highlight really mattered. **0.45 here** — partly
    faithful. A pretty heatmap is not proof of correct reasoning.

ITA (Individual Typology Angle)
:   A label-free estimate of skin tone computed from the healthy skin *around* the lesion, since
    HAM10000 carries no skin-type labels. Binned into light / medium / dark. An estimate from
    pixels, not a clinical assessment.

Corruption stability
:   Whether the prediction survives distortions that shouldn't change a diagnosis — noise, blur,
    brightness, JPEG. **71.7% here.** Note that stability is not correctness: a model that is
    confidently wrong both before and after scores a perfect 1.0.

Membership inference
:   Could an attacker tell whether a specific patient's image was in the training set, from the
    model's confidence alone? Reported as an AUC where **0.5 is ideal**. **0.558 here** — low
    risk.

Split integrity
:   The precondition check: does the test set overlap what the model trained on? Compared by
    lesion as well as by image. If this fails, every other number in the report is void — which
    is why it is not really a "pillar" at all, but a gate in front of them.

---

## This project's own vocabulary

Pillar
:   One dimension of the evaluation: performance, fairness, robustness, explainability, privacy —
    plus integrity, which is a precondition rather than a property.

Finding
:   One metric's result: a value, a verdict, a one-sentence summary, explanatory text, and a
    chart specification. The single unit that flows from engine to dashboard.

Verdict
:   `pass` · `warn` · `fail` · **`info`**. `info` means **no verdict was claimed** — the sample
    was too small or required data was missing. It is *not* a weak pass, and the dashboard says
    "No verdict" so it can't be misread as "we checked and it was fine".

Verdict gate
:   The minimum evidence before a metric is allowed to judge. Accuracy stays `info` below n=30;
    robustness below n=20; a fairness gap needs two groups of ≥10 whose intervals separate.

Snapshot
:   One immutable record per run, in `history/`. Carries the evaluation manifest's **content
    hash** and the integrity verdict — the two things that decide whether a later comparison is
    legitimate.

Comparability
:   Two runs may only be compared if they scored the same rows (identical manifest hash) and
    both had a verified split. Otherwise the tool refuses and says why, rather than drawing a
    chart of a difference it cannot support.
