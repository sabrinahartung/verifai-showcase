# VERIFAI Showcase — Responsible-AI-Evaluation

**Systematische, reproduzierbare Bewertung von ML-Modellen entlang der Responsible-AI-Säulen** —
Performance, Fairness, Robustheit, Erklärbarkeit, Datenschutz — über mehrere Datendomänen
(Bild zuerst; Text/LLM folgen).

> Diese Version überführt das große VERIFAI-Framework in eine **dateibasierte, reproduzierbare,
> gratis lauffähige** Form: die (potenziell schwere) Auswertung läuft *einmal* — lokal auf CPU für
> kleine Stichproben oder auf einer Gratis-GPU für große — und erzeugt statische **Artefakte**
> (JSON + Plots). Eine kleine Streamlit-App zeigt sie interaktiv: **Kacheln → Klick → Dashboard**.
> Ohne Server, ohne Datenbank, ohne laufende Kosten.

---

## Schnellstart

**Engine — einen echten Lauf erzeugen** (in einer Umgebung mit `torch` — z. B. deiner
`ML_Training_Dojo/.venv`, dort ist alles schon installiert):

```bash
pip install -r requirements-engine.txt          # nur falls torch fehlt
python scripts/run_scenario.py scenarios/skin_cancer.yaml
```

Das nutzt die **7 echten, gelabelten HAM10000-Beispielbilder** in `data/examples/` und dein
**echtes ResNet18 von Hugging Face** und schreibt `showcase/artifacts/skin_cancer/` (report.json,
card.json, plots/ mit Grad-CAM-Overlays). Läuft auf einem Laptop in Sekunden.

> Tipp: In `scenarios/skin_cancer.yaml` `weights_path:` auf deine lokale `.pt` zeigen lassen,
> dann entfällt sogar der Hugging-Face-Download.

**Showcase — anschauen:**

```bash
pip install -r showcase/requirements.txt
streamlit run showcase/app.py
```

**Großer Lauf ohne GPU-Sorgen:** `scripts/run_on_free_gpu.ipynb` (Colab/Kaggle) — genau dafür,
dass dein Mac den vollen Subset **nicht** rechnen muss.

---

## Architektur auf einen Blick

```
verifai/            ← ENGINE (offline: lokal / Kaggle / Colab)
  core/             Findings-Datenmodell + Runner + Metrik-Registry
  models/           Domänen-Adapter (Bild: ResNet18 von Hugging Face)
  datasets/         kleine, gepinnte Subsets über Manifeste (reproduzierbar)
  metrics/          die Säulen: performance / fairness / robustness / explainability / privacy
  export/           Findings -> statische Artefakte (JSON + Plots)

data/
  examples/         7 echte, gelabelte HAM10000-Bilder (MVP-Stichprobe)
  manifests/        welche Bilder + Labels (CSV, versioniert)

scenarios/          ein Lauf = eine YAML (z. B. skin_cancer.yaml)
scripts/            run_scenario.py (CLI) + run_on_free_gpu.ipynb

showcase/           SCHAUFENSTER (deployt gratis auf Streamlit Community Cloud)
  app.py            Kachel-Galerie -> Klick aufs Modell -> Plotly-Dashboard
  artifacts/<id>/   ein Ordner = eine Kachel (card.json + report.json + plots/)
  requirements.txt  bewusst LEICHT (Free-Tier-tauglich)
```

## Was der `skin_cancer`-Lauf misst (alle echt gerechnet)

| Säule | Metrik | Was sie tut |
|---|---|---|
| Performance | `top1_accuracy` | Top-1-Treffer + Konfidenz je Beispiel (grün=richtig / rot=falsch) |
| Erklärbarkeit | `gradcam_faithfulness` | Grad-CAM-Overlays (aus deinem `streamlit_app.py` portiert) + Deletion-Faithfulness |
| Robustheit | `corruption_stability` | Bleibt die Vorhersage unter Rauschen/Blur/Helligkeit/JPEG stabil? |
| Fairness | `skin_tone_ita` | Hauttyp-Abdeckung via ITA (label-frei); Subgruppen-Gap beim großen Lauf |
| Datenschutz | `membership_inference_auc` | **ehrlich:** braucht Train/Holdout-Split → im GPU-Lauf, keine erfundene Zahl |

**Ehrlichkeit zur Stichprobe:** n=7 ist ein *Plausibilitätscheck*, kein Benchmark. Jede Metrik
sagt das im Text und beansprucht erst ab genügend Datenpunkten ein hartes Urteil (pass/fail).
Für belastbare Zahlen den größeren Subset über das GPU-Notebook fahren — **gleicher Code-Pfad**,
nur mehr Zeilen im Manifest.

## Erweiterbar: ein neues Modell / eine neue Domäne hinzufügen

1. Szenario anlegen (`scenarios/<neu>.yaml`) mit Modell + Datensatz + Metriken.
2. `python scripts/run_scenario.py scenarios/<neu>.yaml` → erzeugt `showcase/artifacts/<neu>/`.
3. Fertig — beim nächsten Öffnen erscheint automatisch **eine neue Kachel**. Kein App-Code ändern.

Neue Metrik? Sie gibt in `Finding.details["chart"]` (optional `["chart2"]`) eine kleine
Chart-Spezifikation zurück (`{"kind": "bar"|"line"|"heatmap"|"gauge"|"images", ...}`) — die App
rendert sie generisch mit Plotly. Metrik-Signatur überall gleich: `run(model, dataset, ctx) -> Finding`.

**Kein MongoDB, kein Docker-Zwang, kein Backend-Server.** Ergebnisse sind Dateien.

---

## Transparenz (wichtig)

Die öffentliche Streamlit-Demo zeigt **vorab berechnete** Ergebnisse, damit sie gratis und
jederzeit erreichbar ist. Das ist eine bewusste Design-Entscheidung, keine Verschleierung:

- **Voller Code:** dieses Repo — inkl. jeder Metrik.
- **Modell:** öffentlich auf Hugging Face ([`sabrinahartung1010/skin-lesion-resnet18`](https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18)).
- **Daten:** die echten Beispielbilder liegen im Repo (`data/examples/`), Labels im Manifest.
- **Selbst reproduzieren:** `scenarios/*.yaml` + `run_scenario.py` — jedes Ergebnis nachrechenbar
  (GPU-Notebook liegt bei).
- **Echter Lauf im Video:** siehe Portfolio.

> `showcase/artifacts/_sample_skin_resnet/` ist ein **Dev-Fixture mit SAMPLE-Daten** (klar als
> solches markiert), damit man die UI sofort anschauen kann, bevor der erste echte Lauf da ist.
> Nach `run_scenario.py` erscheint daneben die echte Kachel `skin_cancer/`.

---

## Status

- [x] Engine + Findings-Datenmodell + Runner + Registry
- [x] Bild-Metriken über **alle Säulen** implementiert (Performance, Fairness, Robustheit, Erklärbarkeit; Datenschutz ehrlich als „braucht vollen Lauf")
- [x] Streamlit-Showcase: Kachel-Galerie → Plotly-Dashboard, auto-erweiterbar
- [x] Reproduzierbare Beispiel-Stichprobe (7 echte HAM10000-Bilder + Manifest)
- [ ] **Erster echter Lauf** ausführen (`run_scenario.py`) → SAMPLE-Kachel durch echte ersetzen
- [ ] Größerer Subset auf Gratis-GPU (belastbare Fairness-/Privacy-Zahlen)
- [ ] Text/LLM-Szenario
- [ ] Deploy auf Streamlit Community Cloud + kurzes Video

## Data / Lizenz

Die Beispielbilder stammen aus **HAM10000** (Tschandl et al., 2018; CC BY-NC 4.0) und dienen hier
nur der Demonstration. Das Modell ist ein **Bildungs-Proof-of-Concept — kein Medizinprodukt,
nicht für diagnostische Zwecke.**
