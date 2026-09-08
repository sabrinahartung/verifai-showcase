# The six pillars

!!! tip "Unfamiliar terms?"
    Sensitivity, PPV, confidence intervals, ITA and the rest are defined in plain
    language with worked examples in the [Glossary](glossary.md).

Each pillar answers one plain question. The dashboard shows them in this order because
**integrity comes first** — every other number is conditional on it.

```mermaid
flowchart TB
    I["🔒 Integrity<br/><i>Can these results be trusted at all?</i>"]
    I --> P["🎯 Performance<br/><i>Does it get the answer right?</i>"]
    I --> F["⚖️ Fairness<br/><i>Does it work equally well for everyone?</i>"]
    I --> R["🌧️ Robustness<br/><i>Does it stay reliable on imperfect input?</i>"]
    I --> E["🔍 Explainability<br/><i>Can we see why it decided?</i>"]
    I --> V["🔐 Privacy<br/><i>Could it leak its training data?</i>"]
    style I fill:#FFF6E0,stroke:#C77700,stroke-width:3px,color:#1a1a2e
```

| Pillar | Metric | Needs held-out data? |
|---|---|---|
| Integrity | `split_leakage` | — it *is* the check |
| Performance | `top1_accuracy` | **yes** — this is what leakage corrupts |
| Fairness | `skin_tone_ita` | yes for subgroup accuracy, no for coverage |
| Robustness | `corruption_stability` | no — measures whether predictions *flip* |
| Explainability | `gradcam_faithfulness` | no — illustrative either way |
| Privacy | `membership_inference_auc` | **yes**, and needs members too |

## Integrity — `split_leakage`

Compares the evaluation manifest against everything the model trained on, by `lesion_id` as
well as `image_id`. Covered in full in [Split integrity](integrity.md).

Reports `pass` when nothing is shared, `fail` above 1% contamination, `warn` below it, and
`info` when the manifests carry no identifier to compare on — which is **not** a clean bill
of health, merely an unanswered question.

## Performance — `top1_accuracy`

Share of images whose most confident class is correct. The representation switches with
sample size, because one bar per image is useless at scale:

```mermaid
flowchart LR
    N{"n ≤ 50?"}
    N -->|yes| B["bar per image<br/><i>green = correct, red = wrong</i><br/>tall red = confidently wrong"]
    N -->|no| H["confusion matrix<br/><i>row-normalised</i>"]
    N -->|no| R["+ recall per class<br/><i>with support</i>"]
    style H fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
```

`value` carries `accuracy`, `balanced_accuracy`, `per_class_recall` and `support`.

!!! danger "Why the per-class view is not optional"
    This model scores **79.6% accuracy** and **63.8% recall on melanoma**. The averaged
    number is dominated by the 1,009 nevi in the test set. A rare, dangerous class can fail
    badly without moving the headline at all.

Verdict gate: no pass/fail below **n = 30**.

## Fairness — `skin_tone_ita`

HAM10000 carries no skin-type labels, so skin tone is estimated from the image itself. The
Individual Typology Angle is computed from the healthy skin *around* the lesion — the outer
frame of a dermatoscopic crop — using the median to resist hair, rulers and vignetting.

```mermaid
flowchart LR
    IM["image"] --> BD["border region<br/><i>skin, not lesion</i>"]
    BD --> LAB["sRGB → CIE Lab"]
    LAB --> MED["median L*, b*"]
    MED --> ITA["ITA = atan2(L−50, b) in degrees"]
    ITA --> BIN{"bin"}
    BIN -->|"≥ 41°"| L1["light (I–II)"]
    BIN -->|"19–41°"| L2["medium (III–IV)"]
    BIN -->|"< 19°"| L3["dark (V–VI)"]
```

Two things are reported: **coverage** (who is in the sample) and, only when the evidence
supports it, **subgroup accuracy** and the gap between groups.

Verdict gate: a gap requires **at least two populated bins with ≥ 10 images each**.

!!! warning "A regression worth remembering"
    The gate originally accepted a single populated bin. One group yields a gap of 0.0, which
    scored a green `pass` — on a dataset whose defining fairness problem is that it barely
    contains dark skin. A test now asserts that a single bin never reads as a pass.

## Robustness — `corruption_stability`

Applies four distortions that should not change a diagnosis, and measures how often the top-1
class survives.

```mermaid
flowchart LR
    C["clean image"] --> P0["predict"]
    C --> N["noise σ=18"] --> P1["predict"]
    C --> B["Gaussian blur r=2"] --> P2["predict"]
    C --> BR["brightness ×1.4"] --> P3["predict"]
    C --> J["JPEG q=25"] --> P4["predict"]
    P0 --> CMP{"same top-1?"}
    P1 & P2 & P3 & P4 --> CMP
    CMP --> S["stability per corruption"]
```

Verdict gate: no pass/fail below **n = 20**.

!!! note "Stability is not correctness"
    A model that is confidently wrong both before and after a distortion scores a perfect
    1.0 here. Read it next to performance, never alone.

## Explainability — `gradcam_faithfulness`

Grad-CAM highlights the regions that drove the decision. Faithfulness then checks whether
those highlights are honest, by masking the most-attended region and measuring the drop in
the predicted class's probability.

```mermaid
sequenceDiagram
    participant M as model
    participant G as Grad-CAM
    participant D as deletion test
    M->>G: forward, then backward on the top class
    G->>G: gradient-weighted activations at cam_layer, ReLU
    G-->>M: heatmap
    M->>D: p₀ = confidence on the original
    D->>D: grey out the top 20% most-attended pixels
    D->>M: p₁ = confidence on the masked image
    D-->>M: faithfulness = p₀ − p₁
```

The target layer comes from `model.cam_layer` (`layer4[-1]` for ResNets), so this is not
tied to one architecture. Overlays are capped at `gradcam_max_images` (default 7) to keep
artifacts small.

Always `info` — there is no defensible universal threshold for "faithful enough".

## Privacy — `membership_inference_auc`

Asks whether an attacker could tell that a specific patient was in the training set, using
the model's confidence in the true class as the signal.

```mermaid
flowchart LR
    ME["members<br/><i>train manifest</i>"] --> CM["confidence in true class"]
    NM["non-members<br/><i>test manifest</i>"] --> CN["confidence in true class"]
    CM & CN --> AUC["rank-based AUC<br/><i>ties averaged</i>"]
    AUC --> SC["0.5 = indistinguishable (good)<br/>1.0 = fully identifiable (leak)"]
    style SC fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
```

Verdict: `pass` below 0.60, `warn` below 0.75, `fail` above. Refuses to report below 50
images per side, and refuses entirely when no members set is declared.

!!! note "One attack, not all attacks"
    This is the cheap confidence-only attack. Shadow models or per-class calibration can do
    better, so a low score is evidence of low risk rather than a guarantee.
