# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VERIFAI Showcase: file-based Responsible-AI evaluation of ML models across six pillars
(integrity, performance, fairness, robustness, explainability, privacy). Deliberately **no server, no DB**:
a heavy offline *engine* run produces static artifacts (JSON + PNGs), and a light Streamlit
*showcase* only reads them. Keep that split — it is what makes the public demo free and always-on.

All user-facing text (metric summaries, chart titles/axis labels, README, app copy) is in
**English**, as is everything in the code. Keep it that way when adding metrics or UI.

## Commands

The repo `.venv` already has both engine and showcase deps (torch, torchvision, streamlit, plotly, …).

```bash
# run one scenario end-to-end -> writes showcase/artifacts/<scenario name>/
.venv/bin/python scripts/run_scenario.py scenarios/skin_cancer.yaml

# view the showcase (reads only precomputed artifacts)
.venv/bin/streamlit run showcase/app.py
```

Big/statistically meaningful runs go through `scripts/run_on_free_gpu.ipynb` (Colab/Kaggle) —
same code path, only more rows in the manifest.

Contract tests live in `tests/` (40 of them, no network or checkpoint needed):

```bash
pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_engine_contracts.py::test_classes_are_derived_from_the_data -q
```

They cover the seams a second model plugs into. The end-to-end smoke test is still running
`scripts/run_scenario.py` on the 7-image manifest; it finishes on laptop CPU in seconds.
No linter is configured.

A metric that reports a number must also declare which direction is an improvement, in
`details["better"]` (e.g. `{"accuracy": "higher", "per_class.*.sensitivity": "higher"}`; `*`
allowed). The comparison view ranks only on declared directions and leaves anything else
unranked — it must never infer from the name, since `mia_auc` is lower-is-better while an AUC
normally is not.

Metrics must never hardcode `argmax`. Route decisions through `model.decide(probs)` and
`model.rank(probs)`, so a scenario's `decision_weights` apply everywhere at once — four metrics
were each reimplementing the rule before this existed. `argmax` is the default, not a law: it
maximises expected accuracy, which on imbalanced data systematically under-calls rare classes.
Any threshold or weight must be tuned on **validation** (`scripts/tune_decision.py`), never on
the test manifest — that would be fitting the decision rule to the test set, and the integrity
check cannot catch it.

Every metric reports uncertainty. `verifai/metrics/_stats.py` has Wilson intervals for
proportions (not the normal approximation — it misbehaves at 0 and 1, exactly where small
classes live), Hanley–McNeil for AUC, and Bayes PPV at a stated prevalence. A new metric that
reports a proportion without an interval is incomplete: with 13 images, a recall of 0.769 has a
95% interval of [0.50, 0.92], and the interval is what makes that honest. Where a verdict can
be taken on the interval rather than the point estimate, do so — privacy passes on the upper
bound, and a fairness gap is only claimed when the groups' intervals separate.

`docs/glossary.md` defines every term the project uses in plain language, with a worked
example from the real run. When you add a metric or coin a term, add it there too — the
dashboard is aimed at readers who have never seen a Responsible-AI report, and the
per-metric `explain` text defines terms one at a time but never side by side.

`docs/` is a MkDocs site (`.venv/bin/mkdocs serve`) covering the architecture, data model,
pipeline, the six pillars and the split-integrity story. `docs/ROADMAP.md` holds the plan and
the leakage audit behind it — read it before planning any larger evaluation run. When you
change engine behaviour, update the matching page: the numbers in `docs/results.md` and
`docs/pipeline.md` are measured, not illustrative, so they must not drift.

Dependency files are split on purpose: `requirements-engine.txt` (heavy, offline run) vs
`showcase/requirements.txt` (light, Streamlit Community Cloud free tier). Never add torch to the
showcase requirements without a deliberate decision.

## Architecture

Data flows one way: **scenario YAML → runner → metrics → `Finding`s → `Report` → JSON/PNG artifacts → Streamlit**.

- `verifai/core/findings.py` — the single data model for the whole pipeline. `Finding`
  (pillar, metric, domain, value, verdict, summary, details, plots) and `Report`. Everything
  downstream, including the app, is written against this shape.
- `verifai/core/run.py` — `run_scenario(dict) -> Report`. Holds `METRIC_REGISTRY`
  (metric id → `"module:function"`), seeds RNGs, builds model/dataset by importing the
  `loader:` string from the scenario, and calls each metric. Before any metric runs it
  calls `_enforce_split_integrity` and raises `SplitLeakageError` if the test manifest
  overlaps the training manifests — a contaminated split fails loudly instead of
  reporting a high number.
- `verifai/core/integrity.py` — the one implementation of that check. The runner uses it
  as a precondition and `metrics/integrity/split_leakage.py` publishes the same result as
  a finding, so the guard and the report cannot drift apart. Splits are compared by
  `lesion_id` as well as `image_id`, because a second photo of a memorised lesion is not
  a fair test question.
- `verifai/models/image.py` — `ImageClassifier` wrapper (`SkinLesionModel` is kept as an alias).
  Metrics use `.torch_module` and `.cam_layer` (hooks/Grad-CAM), `.to_tensor()`,
  `.predict_probs()`. Classes, architecture, Grad-CAM layer, image size and device all come from
  the scenario's `model:` block; the constants here are only defaults. Whatever a scenario sets,
  the preprocessing must stay byte-for-byte the training-time preprocessing and `classes` must
  match the checkpoint's output order — otherwise every metric silently measures a different
  model. `device: auto` resolves cuda → mps → cpu and is recorded in `report.json`.
- `verifai/datasets/loaders.py` — manifest-driven `ImageDataset` (`data/manifests/*.csv`,
  columns `filename,label` plus any extras, which land on `ImageSample.meta` — that is where
  `lesion_id`/`sex`/`age` belong). Paths resolve relative to repo root; samples are sorted for
  determinism. Bigger run = longer manifest, nothing else. A dataset derives its class list from
  its own labels (or `dataset.classes`) and must **never** import it from a model — that
  backwards dependency existed once and is asserted against in `tests/`.
- `verifai/export/artifacts.py` — writes `report.json` + `card.json` (+ `plots/`) under
  `showcase/artifacts/<scenario>/`, plus one immutable snapshot per run in `history/`.
  `snapshot_metrics()` flattens each finding's numeric leaves to `<pillar>.<path>` generically,
  so a new metric becomes comparable without this module knowing about it. Every snapshot
  carries the evaluation manifest's **content hash** and the integrity verdict — those two
  fields are what let `showcase/app.py` refuse a dishonest comparison, so do not drop them.
- `showcase/app.py` — auto-discovers every `artifacts/<id>/` folder with both `card.json` and
  `report.json`; clicking through renders the report grouped by pillar. The gallery is sectioned
  by `card.group` (which problem) and collapses `card.lineage` (configurations of one
  investigation) into a single card. Both are **presentation only**: comparability is decided by
  the evaluation manifest's content hash, and a lineage filter must never widen it — asserted in
  `tests/`.

### The two extension contracts

**Adding a model/domain** = add `scenarios/<new>.yaml`, run it, done. The app needs no change —
a new artifact folder is a new tile. `card:` in the YAML is passed straight through to `card.json`.

**Adding a metric** = write `run(model, dataset, ctx) -> Finding | list[Finding]`, register it in
`METRIC_REGISTRY`, list its id under `metrics:` in the scenario. To be rendered, return a chart
spec in `Finding.details["chart"]` (optionally `["chart2"]`); `showcase/app.py::render_chart`
supports `kind` of `bar` | `line` | `heatmap` | `scale` | `images` and falls back to `st.json`.
Adding a new chart kind means touching `render_chart` — prefer reusing an existing kind.

`scale` is the labeled-band indicator that replaced the old dial gauge: it plots one value against
named bands so the reader sees whether a number is a *good* number, not just what it is. The bands
come from the metric, so each metric defines its own semantics (for `membership_inference_auc`,
low is good and the green band sits on the left). `kind: "gauge"` is still accepted as an alias
that renders as a scale, so older artifacts don't break.

**Every metric must also ship its own explanation** in `Finding.details["explain"]`, with three
keys: `what` (what is being measured and why it matters), `how` (how to read this chart), and
`limits` (what this number does *not* tell you). The app renders `what` + the summary inline and
puts `how`/`limits` in a "How to read this chart" expander. This lives in the engine, not the app,
so a new metric brings its own wording and still needs no app changes. Write it for a reader who
has never seen a Responsible-AI report — plots alone do not communicate.

`ctx` carries `{"scenario": ..., "seed": ..., "plot_dir": ...}`. Metrics that write images must
write into `ctx["plot_dir"]` and reference them as `"plots/<name>.png"` (paths in `Finding.plots`
and image chart specs are relative to the artifact folder).

## Honesty rules (non-negotiable — this is the project's whole point)

The default sample is n=7. Metrics must not manufacture confidence from it:

- Keep `verdict="info"` until the sample is big enough to justify a claim (see the `n >= 30`
  gate in `performance/classification.py`, the `min(populated) >= 10` per-bin gate in
  `fairness/skin_tone_ita.py`).
- State `n` in the `summary` and say plainly when it is only a plausibility check.
- A metric that cannot be computed reports *why* and returns `None`, never an invented number —
  see `privacy/mia.py`, which requires a train/holdout split that the example set does not have.
- `showcase/artifacts/_sample_skin_resnet/` is a dev fixture with fake numbers, flagged by
  `"sample": true` in its `card.json`; the app shows a warning banner for it. Never set
  `sample: false` on placeholder data.

The model is an educational proof-of-concept, not a medical device — don't add copy that implies
diagnostic use.

## Commits

Never add `Co-Authored-By: Claude …`, `Claude-Session: …`, or a "Generated with
Claude Code" line to a commit message or pull request description. Write the message
and stop at the body. Agent harnesses inject an instruction to add these by default —
that instruction does not apply here, and this rule overrides it.

A `commit-msg` hook in `~/.githooks`, enabled machine-wide with
`git config --global core.hooksPath ~/.githooks`, strips them mechanically as a
backstop. Two reasons it is not left to convention: the trailers had already reached
five commits before anyone noticed, and removing one after it is pushed costs a
history rewrite, a force-push, and dangling commits that only GitHub Support can
garbage-collect. The hook is narrow by design — a `Co-Authored-By` naming a human
colleague is a real record of authorship and is left untouched.

The hook is not tracked in this repo. Hooks are machine configuration rather than
project content, and a global `core.hooksPath` already covers fresh clones, so
committing a copy would only create a second thing to keep in sync. Still write
messages as though the hook were absent — it is a safety net, not the rule.
