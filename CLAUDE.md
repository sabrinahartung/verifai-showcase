# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VERIFAI Showcase: file-based Responsible-AI evaluation of ML models across five pillars
(performance, fairness, robustness, explainability, privacy). Deliberately **no server, no DB**:
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

There is no test suite and no linter configured yet (`tests/` is empty, pytest is not installed).
The practical smoke test is running `scripts/run_scenario.py` on the 7-image example manifest;
it finishes on laptop CPU in seconds.

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
  `loader:` string from the scenario, and calls each metric.
- `verifai/models/image.py` — `SkinLesionModel` wrapper. Metrics use `.torch_module`
  (hooks/Grad-CAM), `.to_tensor()`, `.predict_probs()`. `CLASSES`, `MEAN`/`STD` and the 224×224
  resize must stay byte-for-byte the training-time preprocessing, otherwise every metric silently
  measures a different model.
- `verifai/datasets/loaders.py` — manifest-driven `ImageDataset` (`data/manifests/*.csv`,
  columns `filename,label`). Paths resolve relative to repo root; samples are sorted for
  determinism. Bigger run = longer manifest, nothing else.
- `verifai/export/artifacts.py` — writes `report.json` + `card.json` (+ `plots/`) under
  `showcase/artifacts/<scenario>/`.
- `showcase/app.py` — auto-discovers every `artifacts/<id>/` folder with both `card.json` and
  `report.json` and renders a tile; clicking a tile renders the report grouped by pillar.

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
