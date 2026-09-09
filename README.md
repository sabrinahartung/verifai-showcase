# VERIFAI Showcase — Responsible-AI Evaluation

**Systematic, reproducible evaluation of ML models along the Responsible-AI pillars** —
performance, fairness, robustness, explainability, privacy — across several data domains
(image first; text/LLM to follow).

> This version turns the larger VERIFAI framework into a **file-based, reproducible, free-to-run**
> form: the (potentially heavy) evaluation runs *once* — locally on CPU for small samples, or on a
> free GPU for large ones — and produces static **artifacts** (JSON + plots). A small Streamlit app
> shows them interactively: **tiles → click → dashboard**. No server, no database, no running costs.

**Full documentation:** `mkdocs serve` (or `docs/`) — architecture, the data model, the
pipeline, all six pillars, and the split-integrity story, with diagrams.

---

## Quickstart

**Engine — produce a real run** (in an environment with `torch` — e.g. your
`ML_Training_Dojo/.venv`, where everything is already installed):

```bash
pip install -r requirements-engine.txt          # only if torch is missing
python scripts/run_scenario.py scenarios/skin_cancer.yaml
```

This uses the **7 real, labeled HAM10000 example images** in `data/examples/` and your
**real ResNet18 from Hugging Face**, and writes `showcase/artifacts/skin_cancer/` (report.json,
card.json, plots/ with Grad-CAM overlays). Runs on a laptop in seconds.

> Tip: point `weights_path:` in `scenarios/skin_cancer.yaml` at your local `.pt` and even the
> Hugging Face download goes away.

**Showcase — take a look:**

```bash
pip install -r showcase/requirements.txt
streamlit run showcase/app.py
```

**Big run without GPU worries:** `scripts/run_on_free_gpu.ipynb` (Colab/Kaggle) — exactly so your
Mac does **not** have to compute the full subset.

--- thank

## Architecture at a glance

```
verifai/            ← ENGINE (offline: local / Kaggle / Colab)
  core/             findings data model + runner + metric registry
  models/           domain adapters (image: ResNet18 from Hugging Face)
  datasets/         small, pinned subsets via manifests (reproducible)
  metrics/          the pillars: performance / fairness / robustness / explainability / privacy
  export/           findings -> static artifacts (JSON + plots)

data/
  examples/         7 real, labeled HAM10000 images (MVP sample)
  manifests/        which images + labels (CSV, versioned)

scenarios/          one run = one YAML (e.g. skin_cancer.yaml)
scripts/            run_scenario.py (CLI) + run_on_free_gpu.ipynb

showcase/           SHOP WINDOW (deploys for free on Streamlit Community Cloud)
  app.py            tile gallery -> click a model -> Plotly dashboard
  artifacts/<id>/   one folder = one tile (card.json + report.json + plots/)
  requirements.txt  deliberately LIGHT (free-tier friendly)
```

## What the `skin_cancer` run measures (all really computed)

| Pillar | Metric | What it does |
|---|---|---|
| Performance | `top1_accuracy` | Top-1 hits + confidence per example (green=correct / red=wrong) |
| Explainability | `gradcam_faithfulness` | Grad-CAM overlays (ported from your `streamlit_app.py`) + deletion faithfulness |
| Robustness | `corruption_stability` | Does the prediction stay stable under noise/blur/brightness/JPEG? |
| Fairness | `skin_tone_ita` | Skin-type coverage via ITA (label-free); subgroup gap on the big run |
| Privacy | `membership_inference_auc` | **honest:** needs a train/holdout split → in the GPU run, no invented number |

**Honesty about the sample:** n=7 is a *plausibility check*, not a benchmark. Every metric says so
in its text and only claims a hard verdict (pass/fail) once there are enough data points. For
solid numbers, run the larger subset via the GPU notebook — **same code path**, just more rows in
the manifest.

## Extensible: adding a new model / a new domain

1. Create a scenario (`scenarios/<new>.yaml`) with model + dataset + metrics.
2. `python scripts/run_scenario.py scenarios/<new>.yaml` → produces `showcase/artifacts/<new>/`.
3. Done — the next time you open the app, **a new tile** appears automatically. No app code changes.

A new metric? It returns a small chart specification in `Finding.details["chart"]` (optionally
`["chart2"]`) — `{"kind": "bar"|"line"|"heatmap"|"gauge"|"images", ...}` — and the app renders it
generically with Plotly. The metric signature is the same everywhere:
`run(model, dataset, ctx) -> Finding`.

**No MongoDB, no forced Docker, no backend server.** Results are files.

---

## Transparency (important)

The public Streamlit demo shows **precomputed** results so it stays free and always reachable.
That is a deliberate design decision, not obfuscation:

- **Full code:** this repo — including every metric.
- **Model:** public on Hugging Face ([`sabrinahartung1010/skin-lesion-resnet18`](https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18)).
- **Data:** the real example images are in the repo (`data/examples/`), labels in the manifest.
- **Reproduce it yourself:** `scenarios/*.yaml` + `run_scenario.py` — every result is recomputable
  (the GPU notebook is included).
- **Real run on video:** see the portfolio.

> `showcase/artifacts/_sample_skin_resnet/` is a **dev fixture with SAMPLE data** (clearly marked
> as such) so the UI can be viewed immediately, before the first real run exists. After
> `run_scenario.py`, the real tile `skin_cancer/` appears next to it.

---

## Status

- [x] Engine + findings data model + runner + registry
- [x] Image metrics implemented across **all pillars** (performance, fairness, robustness, explainability; privacy honestly marked as "needs the full run")
- [x] Streamlit showcase: tile gallery → Plotly dashboard, auto-extensible
- [x] Reproducible example sample (7 real HAM10000 images + manifest)
- [x] **First real run** executed (`run_scenario.py`) → replace the SAMPLE tile with the real one
- [ ] Larger subset on a free GPU (solid fairness/privacy numbers)
- [ ] add Text scenario
- [ ] add LLM scenario
- [ ] Deploy to Streamlit Community Cloud + short video

## Data / license

The example images come from **HAM10000** (Tschandl et al., 2018; CC BY-NC 4.0) and serve
demonstration purposes only. The model is an **educational proof of concept — not a medical
device, not for diagnostic use.**
