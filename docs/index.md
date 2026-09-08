# VERIFAI Showcase

**Systematic, reproducible Responsible-AI evaluation of ML models** across six pillars —
integrity, performance, fairness, robustness, explainability and privacy.

The evaluation runs *once*, offline, and produces static artifacts. A small Streamlit app
reads those files and renders them. No server, no database, no running costs.

```mermaid
flowchart LR
    subgraph engine["⚙️ ENGINE — runs offline, once"]
        direction TB
        SC["scenario.yaml"] --> RUN["run_scenario()"]
        RUN --> FIND["Findings"]
        FIND --> ART["report.json<br/>card.json<br/>plots/*.png"]
    end
    subgraph shop["🪟 SHOWCASE — free & always on"]
        direction TB
        APP["Streamlit app"] --> TILE["tile gallery"]
        TILE --> DASH["pillar dashboard"]
    end
    ART -->|"committed to git"| APP
    style engine fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
    style shop fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
```

## Why this exists

A single accuracy number is not an evaluation. This model reports **79.6% top-1 accuracy**
on held-out data — and misses **one melanoma in three**. Both facts come from the same run;
only one of them fits in a headline.

!!! danger "The failure that motivated the integrity pillar"
    The dataset this project started from shipped `train`/`validation`/`test` splits in which
    **80% of the test images also appeared in training**. Every accuracy computed on it was a
    memorisation check wearing a benchmark's clothes. Nothing crashed; the number just came
    out high. See [Split integrity](integrity.md).

## The loop it is built for

Auditing a model once is the small use. The point is the cycle: the framework finds the
weakness that the headline number hides, you act on it, and the next run is compared against
the last one — but only when the comparison is legitimate.

```mermaid
flowchart LR
    subgraph once["done once"]
        BS["build_splits.py<br/><i>group by lesion, stratify</i>"] --> F["frozen test manifest"]
    end

    W["find the weakness<br/><i>melanoma recall 0.638,<br/>not accuracy 0.796</i>"]
    H["change ONE thing<br/><i>loss · sampling · threshold ·<br/>backbone — or more training data</i>"]
    T["train a variant"]
    E["evaluate on the frozen test set"]
    G{"is train ∩ test<br/>still empty?"}
    S["snapshot — admissible<br/><i>keyed to the test set's<br/>content hash</i>"]
    C["compare against baseline<br/><i>including what got worse</i>"]
    X["✋ excluded from the<br/>comparison, with the reason"]

    W --> H --> T --> E --> G
    G -->|yes| S --> C --> W
    G -->|no| X
    F --> E

    style W fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
    style C fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
    style X fill:#F5D3CE,stroke:#C0392B,color:#1a1a2e
    style once fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
```

**Why the integrity check repeats, when the split was built once.** Contamination is not a
property of the test set — it is a property of `train ∩ test`. Freezing the test set does not
freeze that relation, because the *training* side moves: the loop explicitly allows adding
training data as a variant. Pull in ISIC 2019, which incorporates HAM10000, and a test set you
correctly froze months ago is quietly contaminated again. The check is cheap, and it is the
only thing standing between that and a published number.

**What may change inside the loop:** the model (loss, sampling, operating point, backbone) and
the training data. **What may not:** the test manifest. Change that and you have not iterated —
you have started a separate comparison group, which is exactly how the tool treats it.

Every run writes a snapshot to `history/`, so an experiment cannot silently overwrite the
evidence of the one before it. A green *+12 points* against a leaked baseline is exactly the
claim this project exists to catch.

## The design in three claims

=== "Results are files"

    An evaluation produces `report.json`, `card.json` and some PNGs in a folder. Adding a
    folder adds a tile to the showcase. There is no database and no backend, so the public
    demo costs nothing to keep online and every result is reviewable in a diff.

=== "Honesty is enforced, not intended"

    Metrics withhold a verdict until the sample justifies one (`n>=30` for accuracy, `n>=20`
    for robustness, two populated bins of `>=10` for a fairness gap). A metric that cannot be
    computed says so instead of returning a number. The runner refuses to evaluate a
    contaminated split at all.

=== "Adding a model changes no code"

    A model is a scenario YAML. Run it, get a folder, get a tile. A new metric supplies its
    own chart specification and its own explanatory text, so the app never learns about it.

## Where to start

| If you want to… | Read |
|---|---|
| Understand how the parts fit together | [Architecture overview](architecture.md) |
| Know what a `Finding` is | [Data model](data-model.md) |
| Run the whole thing yourself | [The pipeline](pipeline.md) |
| Know what is actually measured | [The six pillars](pillars.md) |
| Understand the leakage story | [Split integrity](integrity.md) |
| See the numbers | [Current results](results.md) |
| Add a model or a metric | [Extending](extending.md) |
