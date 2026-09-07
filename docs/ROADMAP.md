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

## Step 2 — Training, with the split as a contract

*Goal: training and evaluation can never disagree about what was held out.*

- [ ] `scripts/train_model.py <scenario.yaml>` — trains from a `training:` block
      in the scenario (epochs, lr, batch size, class weights)
- [ ] Split with `GroupShuffleSplit` on **`lesion_id`**, not on rows
- [ ] Write `data/manifests/<name>_{train,val,test}.csv`, each carrying
      `image_id, lesion_id, label` — these are versioned in git and become the
      contract between training and evaluation
- [ ] Push weights to the HF Hub (a ResNet18 `.pt` is ~45 MB — too big for git,
      and `.gitignore` already excludes `*.pt`)
- [ ] `run_scenario.py` evaluates **only** on `<name>_test.csv`

Compute: ~8k images, ResNet18 — roughly 1–2 h on the Mac's MPS once the device bug
is fixed, or ~20 min on a free Colab T4. Evaluation of ~1,500 images is ~2 min on CPU.

## Step 3 — Make leakage a first-class finding

*Goal: the framework catches the mistake that invalidated the last run.*

- [ ] New pillar/metric `integrity.split_leakage` reporting: lesion IDs shared
      between train and test, duplicate image IDs, contamination percentage
- [ ] `run_scenario.py` **refuses to run** when the test manifest shares a
      `lesion_id` with the train manifest, rather than quietly producing a
      flattering number
- [ ] The check renders as a normal tile finding with its own `explain` block

This is the step that turns yesterday's flaw into the product's strongest claim:
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

---

## Known scaling gaps (bite at n>1000, not at n=7)

- Everything runs at **batch size 1** (`to_tensor` does `.unsqueeze(0)` per image)
- `fairness/skin_tone_ita.py` recomputes the clean prediction that
  `performance/classification.py` already made — 7 forward passes per image, 2 redundant
- `details["per_example"]` is written for every image, so `report.json` grows linearly
- `ImageSample` carries only `id/path/label`, so the `sex`, `age`, `localization` and
  `lesion_id` columns that HAM10000 provides have nowhere to live — real demographic
  subgroups would beat the ITA pixel proxy
