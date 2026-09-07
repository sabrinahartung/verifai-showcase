# Data model

One shape flows end to end: a metric returns a `Finding`, the runner collects them into a
`Report`, the exporter serialises it, and the app renders it. Nothing translates between
formats along the way.

```mermaid
classDiagram
    class Report {
        +str scenario
        +Domain domain
        +str model_id
        +str dataset_id
        +list~Finding~ findings
        +dict meta
        +str created_at
        +add(finding)
        +to_dict() dict
    }
    class Finding {
        +Pillar pillar
        +str metric
        +Domain domain
        +Any value
        +Verdict verdict
        +str summary
        +dict details
        +list~str~ plots
    }
    class Details {
        +dict explain
        +dict chart
        +dict chart2
        +Any ...metric specific
    }
    class Explain {
        +str what
        +str how
        +str limits
    }
    class ChartSpec {
        +str kind
        +str title
        +... kind specific
    }
    Report "1" *-- "many" Finding
    Finding "1" *-- "1" Details
    Details "1" o-- "0..1" Explain
    Details "1" o-- "0..2" ChartSpec
```

## Vocabularies

`Pillar`
: `integrity` · `performance` · `fairness` · `robustness` · `explainability` · `privacy`

`Verdict`
: `pass` · `warn` · `fail` · `info`

`Domain`
: `image` · `text` · `tabular` · `llm`

!!! tip "`info` is not a weak `pass`"
    `info` means **no verdict was claimed** — the sample was too small, or the required data
    was missing. The dashboard renders it as "No verdict" rather than a neutral tick, because
    a reader must not mistake "we did not check" for "we checked and it was fine".

`Report.meta` records `seed`, `sample_size` (what was actually evaluated, not what the YAML
declared) and `device` — the last because results are bit-identical *per device*, not across
devices.

## The `explain` contract

Every metric ships its own explanatory text inside `details["explain"]`. This lives in the
engine, not the app, so a new metric brings its own wording and the app needs no change.

| Key | Answers | Rendered |
|---|---|---|
| `what` | What is measured, and why it matters | inline, always visible |
| `how` | How to read this particular chart | in the "How to read this chart" expander |
| `limits` | What this number does **not** tell you | same expander, under "What it does *not* tell you" |

`limits` is the one that earns its place. It is where each metric names the trap it sets:

> **Robustness** — "Stability is not correctness: a model that is confidently wrong both
> before and after a distortion scores a perfect 1.0 here."

> **Fairness** — "Coverage is not performance — this chart shows who is in the sample, not
> how well the model serves them."

> **Explainability** — "A convincing heatmap is not proof of medically correct reasoning —
> it shows where the model looked, not whether it looked for the right reason."

## Chart specifications

A metric describes a chart as data; `showcase/app.py::render_chart` draws it. Unknown kinds
fall back to `st.json`, so a malformed spec degrades rather than crashes.

```mermaid
flowchart LR
    F["Finding.details"]
    F --> C1["chart"]
    F --> C2["chart2 (optional)"]
    C1 & C2 --> R{{"render_chart(spec)"}}
    R -->|bar| B["grouped / coloured bars"]
    R -->|line| L["line + markers"]
    R -->|heatmap| H["confusion matrix"]
    R -->|scale| S["value on labelled bands"]
    R -->|images| I["PNG grid with captions"]
    R -->|unknown| J["st.json fallback"]
    style R fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
```

| `kind` | Required keys | Used by |
|---|---|---|
| `bar` | `x`, `y` (+ `color`/`colors`, `hover`) | performance, robustness, fairness |
| `line` | `x`, `y` | — |
| `heatmap` | `z` (+ `x`, `y`, `text`, `zmin`, `zmax`) | performance at n > 50 |
| `scale` | `value`, `min`, `max`, `bands` | integrity, explainability, privacy |
| `images` | `paths` (+ `captions`) | explainability |

### Why `scale` replaced the dial gauge

A dial shows a number. A `scale` shows whether it is a *good* number, because the bands are
named and supplied by the metric — so each metric defines its own semantics. For deletion
faithfulness, higher is better; for membership-inference AUC, the green band sits on the left.

```mermaid
flowchart LR
    subgraph faith["explainability: higher is better"]
        direction LR
        f1["0 — decorative"] --> f2["0.2 — partly faithful"] --> f3["0.5 — faithful → 1.0"]
    end
    subgraph mia["privacy: lower is better"]
        direction LR
        m1["0.5 — low risk"] --> m2["0.6 — moderate"] --> m3["0.75 — high risk → 1.0"]
    end
    style f3 fill:#CDE8D5,color:#1a1a2e
    style f1 fill:#F5D3CE,color:#1a1a2e
    style m1 fill:#CDE8D5,color:#1a1a2e
    style m3 fill:#F5D3CE,color:#1a1a2e
```

`kind: "gauge"` is still accepted as an alias that renders as a scale, so artifacts written
before the change keep rendering.

## Artifact layout

```mermaid
flowchart TB
    A["showcase/artifacts/"]
    A --> B["_sample_skin_resnet/<br/><i>dev fixture, sample: true</i>"]
    A --> C["skin_cancer/<br/><i>7 examples, integrity unverified</i>"]
    A --> D["skin_cancer_clean/<br/><i>1,493 held-out images</i>"]
    D --> D1["card.json<br/><i>tile metadata</i>"]
    D --> D2["report.json<br/><i>the Findings</i>"]
    D --> D3["plots/<br/><i>Grad-CAM overlays</i>"]
    style D fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
```

The app lists any directory containing **both** `card.json` and `report.json`. `card.json`
carries the tile metadata (`name`, `emoji`, `domain`, `dataset`, `description`, `hf_url`,
`sample`) and comes straight from the scenario's `card:` block.
