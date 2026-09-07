# Architecture overview

Two halves that never run at the same time, joined by a directory of files.

```mermaid
flowchart TB
    subgraph offline["⚙️ ENGINE — heavy, offline, run once"]
        direction TB
        Y["scenarios/*.yaml<br/><i>one run = one file</i>"]
        subgraph core["verifai/core"]
            R["run.py<br/>run_scenario()"]
            I["integrity.py<br/>audit_split()"]
            F["findings.py<br/>Finding · Report"]
        end
        M["verifai/models<br/><i>domain adapter</i>"]
        D["verifai/datasets<br/><i>manifest loader</i>"]
        MET["verifai/metrics<br/><i>six pillars</i>"]
        E["verifai/export<br/>write_report()"]
        Y --> R
        R -->|"guard"| I
        R --> M
        R --> D
        R --> MET
        MET --> F
        F --> E
    end

    subgraph files["📁 ARTIFACTS — the interface"]
        A["showcase/artifacts/&lt;id&gt;/<br/>├── card.json<br/>├── report.json<br/>└── plots/*.png"]
    end

    subgraph online["🪟 SHOWCASE — light, always on"]
        APP["showcase/app.py"]
        G["tile gallery"]
        DB["pillar dashboard"]
        APP --> G --> DB
    end

    E --> A
    A -->|"read-only"| APP

    style offline fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
    style files fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
    style online fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
```

The split is deliberate and load-bearing. The engine needs torch, torchvision and a few
hundred MB of images; the showcase needs `streamlit`, `pillow` and `plotly`. Because the
showcase only ever *reads* precomputed files, it deploys to a free tier and stays online.

!!! warning "Keep the dependency files separate"
    `requirements-engine.txt` is heavy; `showcase/requirements.txt` is deliberately light.
    Streamlit Community Cloud resolves the dependency file in the **entrypoint's directory**
    before the repo root — which is the only thing stopping it from installing the torch
    dependencies declared in the root `pyproject.toml`.

## What runs when you evaluate

```mermaid
sequenceDiagram
    autonumber
    participant CLI as run_scenario.py
    participant Run as core.run
    participant Int as core.integrity
    participant Mod as models.image
    participant DS as datasets.loaders
    participant Met as metrics.*
    participant Exp as export.artifacts

    CLI->>Run: run_scenario(yaml dict)
    Run->>Run: seed random / numpy / torch
    Run->>Mod: _build_model(spec["model"])
    Mod-->>Run: ImageClassifier (on device)
    Run->>DS: _build_dataset(spec["dataset"])
    DS-->>Run: ImageDataset (from manifest)

    rect rgb(255, 235, 235)
        Run->>Int: audit_split(test, train manifests)
        Int-->>Run: {verifiable, clean, contamination}
        Note over Run,Int: raises SplitLeakageError<br/>BEFORE any metric runs
    end

    loop each metric id in scenario["metrics"]
        Run->>Met: run(model, dataset, ctx)
        Met-->>Run: Finding (or list)
    end
    Run-->>CLI: Report
    CLI->>Exp: write_report(report, card)
    Exp-->>CLI: showcase/artifacts/<id>/
```

The guard runs **before** the metrics, not after. A contaminated split costs nothing to
detect and would otherwise produce a full report of flattering numbers.

## Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `core/findings.py` | `Finding`, `Report`, the pillar and verdict vocabularies | know about any specific metric |
| `core/run.py` | `METRIC_REGISTRY`, seeding, the integrity guard | import a metric directly |
| `core/integrity.py` | `audit_split()` — the one leakage implementation | be duplicated anywhere |
| `models/image.py` | `ImageClassifier`, device resolution, Grad-CAM layer lookup | hardcode a class list |
| `datasets/loaders.py` | manifest → `ImageDataset`, extra columns → `ImageSample.meta` | import from `verifai.models` |
| `metrics/**` | one `run(model, dataset, ctx) -> Finding` each | know the app exists |
| `export/artifacts.py` | `report.json` + `card.json` + `plots/` | compute anything |
| `showcase/app.py` | discovery and rendering | know any metric's name |

!!! note "One dependency direction that is asserted in tests"
    `datasets/loaders.py` once did `from verifai.models.image import CLASSES` — a dataset
    deriving its label vocabulary from a model. It now derives classes from its own labels,
    and `tests/test_engine_contracts.py` fails if that import ever returns.

## The registry

Metrics are referenced by string, resolved by import at run time. Nothing imports a metric
module directly, so a scenario picks its own subset and a new metric is one dictionary entry.

```mermaid
flowchart LR
    Y["scenario.yaml<br/>metrics:<br/>- integrity.split_leakage<br/>- performance.classification"]
    REG{{"METRIC_REGISTRY<br/><i>id → 'module:function'</i>"}}
    L["_load()<br/><i>importlib</i>"]
    FN["run(model, dataset, ctx)"]
    Y --> REG --> L --> FN
    FN --> FIND["Finding"]
    style REG fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
```

| Metric id | Implementation |
|---|---|
| `integrity.split_leakage` | `verifai.metrics.integrity.split_leakage:run` |
| `performance.classification` | `verifai.metrics.performance.classification:run` |
| `explainability.gradcam` | `verifai.metrics.explainability.gradcam:run` |
| `robustness.corruption` | `verifai.metrics.robustness.corruption:run` |
| `fairness.skin_tone` | `verifai.metrics.fairness.skin_tone_ita:run` |
| `privacy.mia` | `verifai.metrics.privacy.mia:run` |
