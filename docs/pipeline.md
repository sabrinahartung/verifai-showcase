# The pipeline

Everything runs locally. The full sequence from raw dataset to a deployed dashboard is about
fifteen minutes on an Apple M1 Max, most of it waiting for images to download.

```mermaid
flowchart TB
    subgraph prep["1 · Prepare data"]
        BS["build_splits.py<br/><i>stratified split on lesion_id</i>"]
        MAN["data/manifests/ham10000_*.csv<br/><b>committed to git</b>"]
        MI["materialize_images.py<br/><i>--max-size 320</i>"]
        IMG["data/raw/ham10000/<br/><b>gitignored, 145 MB</b>"]
        BS --> MAN --> MI --> IMG
    end
    subgraph train["2 · Train"]
        TM["train_model.py"]
        CK["artifacts_training/*.pt<br/><b>gitignored</b>"]
        TJ["*_training.json<br/><i>provenance</i>"]
        TM --> CK
        TM --> TJ
    end
    subgraph eval["3 · Evaluate"]
        RS["run_scenario.py"]
        GUARD{"split<br/>clean?"}
        ART["showcase/artifacts/&lt;id&gt;/"]
        RS --> GUARD
        GUARD -->|no| STOP["✋ SplitLeakageError"]
        GUARD -->|yes| ART
    end
    subgraph ship["4 · Deploy"]
        GIT["git commit + push"]
        SC["Streamlit Cloud<br/><i>redeploys on push</i>"]
        GIT --> SC
    end
    MAN -->|"train + val"| TM
    IMG --> TM
    CK --> RS
    MAN -->|"test only"| RS
    ART --> GIT
    style prep fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
    style train fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
    style eval fill:#E8F0FE,stroke:#1a73e8,color:#1a1a2e
    style ship fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
    style STOP fill:#F5D3CE,stroke:#C0392B,color:#1a1a2e
```

```bash
python scripts/build_splits.py                                   # manifests (already committed)
python scripts/materialize_images.py --max-size 320              # 145 MB
python scripts/train_model.py scenarios/skin_cancer_clean.yaml   # ~4.6 min on MPS
python scripts/run_scenario.py scenarios/skin_cancer_clean.yaml  # ~1:49
git add showcase/artifacts && git commit && git push             # push == deploy
```

## The manifests are the contract

This is the idea the whole pipeline is built around. Training reads `_train.csv` and
`_val.csv`; evaluation reads `_test.csv`; both are versioned in git. "What did the model see"
stops being a matter of trust and becomes a file you can diff.

```mermaid
flowchart LR
    subgraph m["data/manifests/ (in git)"]
        TR["ham10000_train.csv<br/>7,014 images<br/>5,230 lesions"]
        VA["ham10000_val.csv<br/>1,508 images<br/>1,120 lesions"]
        TE["ham10000_test.csv<br/>1,493 images<br/>1,120 lesions"]
    end
    TR --> T["train_model.py"]
    VA --> T
    TE -.->|"never opened<br/>during training"| T
    TE --> E["run_scenario.py"]
    TR --> G["integrity guard<br/><i>+ privacy.members</i>"]
    VA --> G
    G --> E
    style TE fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
    style G fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
```

Each row carries `filename, image_id, lesion_id, label, dx_type, age, sex, localization`.
Columns beyond `filename` and `label` land on `ImageSample.meta`, which is what makes real
demographic subgroups possible rather than only the ITA pixel proxy.

## Step 1 — building the split

HAM10000 photographs the same lesion more than once, so splitting on rows leaks even when
every `image_id` is unique. Splitting on `lesion_id` is what makes the test set meaningful.

Every lesion carries exactly one diagnosis — verified against the data — so the split can be
**stratified as well as grouped**, which `GroupShuffleSplit` cannot do.

```mermaid
flowchart TB
    P["dataset parquet<br/><i>metadata columns only, a few MB</i>"]
    P --> DE["deduplicate by image_id<br/>13,354 rows → 10,015 images"]
    DE --> GR["group by lesion_id<br/>→ 7,470 lesions"]
    GR --> ST["for each diagnosis:<br/>shuffle its lesions (seeded)<br/>slice 70 / 15 / 15"]
    ST --> W["write three manifests"]
    W --> V{"any shared<br/>lesion or image?"}
    V -->|yes| X["exit non-zero"]
    V -->|no| OK["✓ leak-free"]
    style OK fill:#CDE8D5,stroke:#2E9E5B,color:#1a1a2e
    style X fill:#F5D3CE,stroke:#C0392B,color:#1a1a2e
```

!!! tip "Metadata-only reads"
    `build_splits.py` reads just the metadata columns over HTTP range requests — a few MB
    instead of the 3.6 GB the embedded images would cost. The image bytes are only fetched
    later, by `materialize_images.py`.

## Step 2 — materialising images

| Option | Disk for 10,015 images |
|---|---|
| originals (600×450) | 2.70 GB |
| `--max-size 320` | **0.14 GB** (19× smaller) |
| `--max-size 256` | 0.10 GB |

The model resizes to 224×224 as its first transform, so 320 loses nothing for training.

!!! warning "Keep one resolution across anything you compare"
    A blur radius or a JPEG quality means something different at a different resolution, so
    **corruption stability shifts** with `--max-size`. The choice is recorded in
    `_materialize.json` beside the images, and the script refuses to write into a directory
    that already holds a different resolution.

## Step 3 — training

`train_model.py` reads the scenario's `training:` block, trains on train, selects on
**balanced** accuracy against val, and never opens the test manifest.

```mermaid
flowchart LR
    S["training: block<br/><i>epochs, lr, bs, workers</i>"] --> L["DataLoader<br/>workers=8<br/>persistent_workers"]
    L --> A["augment<br/><i>flips + colour jitter</i>"]
    A --> N["resnet18<br/><i>ImageNet weights</i>"]
    N --> C["CrossEntropy<br/><i>class-weighted</i>"]
    C --> O["Adam"]
    O --> V{"val balanced<br/>improved?"}
    V -->|yes| SV["save checkpoint"]
    V -->|no| NX["next epoch"]
    style SV fill:#CDE8D5,stroke:#2E9E5B,color:#1a1a2e
```

Two choices worth knowing:

**Class weights.** 67% of this data is `melanocytic_Nevi`. Unweighted cross-entropy learns
to say "nevi" and scores well doing it.

**Selection on balanced accuracy**, not accuracy, for the same reason — the run's best plain
accuracy (0.807, epoch 11) was *not* the checkpoint kept.

## Step 4 — evaluation and deploy

`run_scenario.py` builds the model and dataset, runs the integrity guard, then each metric in
the scenario's `metrics:` list, and writes the artifact folder. Because the showcase reads
committed files, **pushing is deploying** — Streamlit Community Cloud redeploys on push.

## Measured performance

On an Apple M1 Max (10 cores, 64 GB):

| Stage | Throughput | Time |
|---|---|---|
| ResNet18 fwd+bwd, MPS, bs=32 | 348 img/s | ~20 s/epoch |
| Decode + augment, 8 workers, 320px | 1,259 img/s | ~6 s/epoch |
| Realistic epoch (7,014 images) | 286 img/s | 24.5 s |
| **Full 12-epoch training** | | **4.6 min** |
| **Evaluation of 1,493 images** | | **1:49** |

Data loading is ~3.5× faster than the GPU consumes, so training is correctly compute-bound.

!!! note "MPS pays for itself after ~30 forward passes"
    MPS costs ~206 ms of one-time setup, then runs at 2.8 ms/img against CPU's 9.8. A
    scenario reaches break-even at about five images, since each costs ~7 passes. That is why
    `skin_cancer.yaml` (n=7) is pinned to `cpu` while `skin_cancer_clean.yaml` (n=1,493) uses
    `auto`. No cloud GPU is needed anywhere in this pipeline.
