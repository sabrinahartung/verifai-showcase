# Roadmap — towards a trustworthy, fully evaluated model

Status of this document: written 2026-09-07, after the leakage audit below.
Tick the boxes as you go; each step leaves the repo in a working state.

---

## Why this exists

The showcase currently evaluates on **7 images**. The obvious fix — "use the whole
test set" — does not work, because the model has already seen almost all of it.

The `skin-lesion-resnet18` checkpoint was trained on the Hugging Face dataset
`marmal88/skin_cancer`. Querying that dataset's parquet metadata directly:

| Check | Result |
|---|---|
| Its `test` split | 1,285 images |
| Test images whose **image_id also appears in `train`** | **1,025 (80%)** |
| Test images whose **lesion** was seen in training | 1,132 (88%) |
| Test images left after excluding train lesions | 153 |
| …after excluding train **and** validation lesions | **28** |
| HAM10000 images covered by `train` + `validation` | **9,964 of 10,015 (99.5%)** |
| HAM10000 lesions covered by `train` + `validation` | **7,442 of 7,470 (99.6%)** |

So evaluating on all 1,285 images would produce a high accuracy that means nothing —
it is mostly a memorisation check. `n=7` labelled "not a benchmark" is *more* honest
than a leaked `n=1285` carrying a green `pass` badge.

**The images were never the problem. The split was.** HAM10000 is a standard
dermatoscopic benchmark; its real limitations are class imbalance (67% nevi) and a
skew towards light skin — both of which the fairness pillar already reports.

### The fix

Re-split HAM10000 **grouped by `lesion_id`**, so no lesion appears on both sides.
That yields roughly **1,100 held-out lesions / ~1,500 test images** the model
provably never saw, and activates every verdict gate in the engine
(`n>=30` for accuracy, `n>=20` for robustness, `>=10` per bin for fairness).

Not every pillar is blocked by leakage, which is worth knowing:

| Pillar | Needs unseen data? |
|---|---|
| Performance, subgroup accuracy | **Yes** — this is what leakage corrupts |
| Robustness (corruption stability) | No — it measures whether predictions *flip*, a property of the model |
| Fairness (ITA coverage) | No — it describes the dataset, not generalisation |
| Explainability (Grad-CAM) | No — illustrative either way |
| Privacy (membership inference) | **Yes**, and worst hit: with 99.5% of the data in training there is almost no non-member set |

---

## Step 1 — Make the engine model-agnostic ✅ done 2026-09-07

*Goal: a second model can be added without editing engine code.*

- [x] `models/image.py` — move `CLASSES` out of the module constant and into the
      scenario spec (`model.classes`), keeping today's list as the default
- [x] `models/image.py` — make the architecture configurable (`model.arch`,
      default `resnet18`) instead of hardcoding `resnet18()`
- [x] `models/image.py` — **fix device handling**: it loads with
      `map_location="cpu"` and never moves the model, so the "free GPU" notebook
      silently runs on CPU and the Mac's MPS is unused. Add `model.device`
      (`auto` → cuda → mps → cpu) and record the resolved device in `Report.meta`
- [x] `datasets/loaders.py` — **break the backwards dependency**: the dataset
      loader currently does `from verifai.models.image import CLASSES`. Dataset
      classes must come from the data (sorted unique manifest labels) or from the
      dataset spec, never from the model
- [x] `explainability/gradcam.py` — `layer4[-1]` is ResNet-specific. Let the
      model adapter expose its own `cam_layer`, configurable per scenario

**Acceptance: met.** Re-running `scenarios/skin_cancer.yaml` with no scenario edits
produced findings identical to the pre-refactor baseline (top-1 100%, faithfulness
0.669, stability 75%). 13 contract tests in `tests/` cover the seams; run them with
`.venv/bin/python -m pytest tests/ -q`.

Measured while doing this: MPS is **slower** than CPU here (5.5s vs 3.2s at n=7),
because everything still runs at batch size 1 and per-image transfer dominates.
The scenario is therefore pinned to `device: cpu`, and the device is now recorded
in `report.json` under `meta.device`. Batching (see gaps below) is what makes a
GPU pay off — the device fix is its prerequisite, not a speedup on its own.

## Step 2 — Training, with the split as a contract ✅ code done 2026-09-07

*Goal: training and evaluation can never disagree about what was held out.*

- [x] `scripts/build_splits.py` — stratified split **on `lesion_id`**. Better than
      `GroupShuffleSplit`: every lesion carries exactly one diagnosis (verified), so
      the split is stratified *and* grouped. Reads only metadata columns over HTTP
      range requests — a few MB, not the 3.6 GB the images would cost
- [x] Manifests written and committed: `data/manifests/ham10000_{train,val,test}.csv`,
      carrying `filename, image_id, lesion_id, label, dx_type, age, sex, localization`
- [x] `scripts/materialize_images.py` — fetches the JPEGs the manifests name into
      the gitignored `data/raw/ham10000/` (~2.9 GB, deduplicated by image_id)
- [x] `scripts/train_model.py` — reads the `training:` block, trains on train,
      selects on val by **balanced** accuracy, never opens the test manifest, and
      writes `<name>_training.json` recording exactly which manifests were used
- [x] `scenarios/skin_cancer_clean.yaml` — the retrained model's scenario
- [ ] Upload the checkpoint to the HF Hub (`.pt` is gitignored)

**The resulting split:**

| Split | Images | Lesions |
|---|---|---|
| train | 7,014 | 5,230 |
| val | 1,508 | 1,120 |
| **test** | **1,493** | **1,120** |

Zero shared lesions and zero shared images between any pair of splits, checked by
the script itself, which exits non-zero rather than emit a leaking split. Class
stratification holds at 11–16% per class in test, including **163 melanomas** —
against 1 in the best clean subset the old split could offer.

**Verified end to end** on a 42-image miniature of the real split (build → fetch →
train 2 epochs on MPS → evaluate → artifacts), so the pipeline is known to work
before committing to the full download and training run.

Compute: ~7k images, ResNet18 — roughly 1–2 h on the Mac's MPS, or ~20 min on a
free Colab T4. Evaluation of 1,493 images is ~2 min on CPU.

## Step 3 — Make leakage a first-class finding ✅ done 2026-09-07

*Goal: the framework catches the mistake that invalidated the last run.*

- [x] New pillar `integrity` with metric `split_leakage`, reporting shared lesion
      IDs, duplicate image IDs and the contamination percentage
- [x] `run_scenario.py` raises `SplitLeakageError` **before any metric runs**
      rather than quietly producing a flattering number. `integrity.enforce: false`
      downgrades it to a reported finding
- [x] `verifai/core/integrity.py` is the single implementation, used by both the
      guard and the metric, so they cannot drift apart
- [x] Renders as a normal tile finding with its own `explain` block; `integrity`
      sits first in the dashboard because every other pillar is conditional on it

Two honesty bugs were found and fixed while building this:

- A manifest without `image_id`/`lesion_id` columns compared as *zero* overlap and
  scored a green "clean split" — publishing an unverified split as a verified one.
  `audit_split` now separates `verifiable` from `clean`, and unverifiable reports
  `info` with an explicit "this is not a clean bill of health".
- `scenarios/skin_cancer.yaml` now runs the check and states plainly that its
  integrity is unverified — which is the truthful status for a checkpoint trained
  on ~99.5% of HAM10000.

This is the step that turns the original flaw into the product's strongest claim:
*this tool catches the error behind most published accuracy numbers.*

## Step 4 — Retrain, evaluate, deploy

- [ ] Retrain on the clean lesion-grouped split
- [ ] Full evaluation over the real holdout (~1,500 images) — all verdict gates active
- [ ] Membership inference finally computable: members = train manifest,
      non-members = held-out manifest
- [ ] Replace the per-example bar chart (unreadable past ~50 bars) with a
      confusion matrix — the `heatmap` chart kind already exists — plus per-class accuracy
- [ ] Commit artifacts and push. **Streamlit Community Cloud redeploys on push, so
      committing the artifacts *is* the deploy.**

## Step 5 — A larger image set

*Goal: more data, and a test set the model has no relationship to at all.*

HAM10000 is 10,015 images and the model will have seen 70% of them. The next
honest gain comes from data drawn from somewhere else entirely.

- [ ] Add ISIC 2019 (~25k images, 8 classes) as additional **training** data
- [ ] **Check the overlap first.** ISIC 2019 incorporates HAM10000, so the same
      leakage trap is waiting: dedupe by `image_id`/`lesion_id` against our
      manifests before mixing, and re-run `build_splits.py` over the union
- [ ] Map the class vocabularies (ISIC's `SCC` has no HAM10000 equivalent —
      decide explicitly whether to add an eighth class or drop those rows)
- [ ] Consider a genuinely **external** test set (PH2, Derm7pt, PAD-UFES-20) as a
      separate scenario. Different camera, different clinic, different population:
      the gap between the internal and external number *is* the generalisation
      result, and is worth publishing as its own tile

Note that a larger training set does not by itself justify a bigger *test* set —
what makes the test number trustworthy is that no lesion in it was ever trained on.

## Step 6 — Snapshots and a comparison view

*Goal: show improvement over time, without inviting a dishonest comparison.*

- [ ] Keep every evaluation instead of overwriting: `run_scenario.py` writes
      `showcase/artifacts/<id>/history/<created_at>.json` alongside the current
      `report.json` (which stays "latest" so the dashboard is unchanged)
- [ ] Give each card a `lineage:` field, so the app can group snapshots that
      belong to the same model family across retrains
- [ ] Record in each snapshot what makes it comparable: `dataset_id`, the test
      manifest path and its content hash, plus the split-integrity verdict
- [ ] New chart kind for multi-series data (`grouped_bar` or `series`); today's
      `bar` takes a single x/y pair
- [ ] Comparison tile: metric-by-metric across snapshots, plus a delta column

**The trap this must avoid.** Two numbers are only comparable if they come from
the same test manifest *and* both snapshots were clean. Comparing the original
checkpoint against the retrained one is not a fair fight in either direction:

- Evaluated on its own contaminated split, the old model scores inflated numbers.
- Evaluated on the new clean manifest, it *still* scores inflated numbers, because
  it trained on ~99.5% of HAM10000 — the new test set is unseen for the new model,
  not for the old one.

So the comparison view must refuse to plot snapshots with mismatched test
manifests, and must render the integrity verdict beside every bar. A green
"+12 points" against a leaked baseline would be exactly the kind of claim this
project exists to catch. Where no fair comparison is possible, say so instead of
drawing the chart.

---

## Known scaling gaps (bite at n>1000, not at n=7)

- Everything runs at **batch size 1** (`to_tensor` does `.unsqueeze(0)` per image)
- `fairness/skin_tone_ita.py` recomputes the clean prediction that
  `performance/classification.py` already made — 7 forward passes per image, 2 redundant
- `details["per_example"]` is written for every image, so `report.json` grows linearly
- `ImageSample` carries only `id/path/label`, so the `sex`, `age`, `localization` and
  `lesion_id` columns that HAM10000 provides have nowhere to live — real demographic
  subgroups would beat the ITA pixel proxy
