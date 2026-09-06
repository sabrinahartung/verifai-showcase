"""VERIFAI Showcase — Streamlit.

UX: a gallery of model tiles -> click a model -> a clean Plotly dashboard of its
Responsible-AI analysis across the four pillars.

Extensible by design: every model is a folder under ./artifacts/<model_id>/ with
  card.json    -> tile metadata (name, domain, description, hf_url, sample?)
  report.json  -> the Findings produced by the engine (verifai.export.artifacts)
  plots/       -> optional images (e.g. Grad-CAM overlays)
Add a folder -> a new tile appears. No code change needed.

Results are PRECOMPUTED (run once by the engine) so this app stays free & always-on.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

ART = Path(__file__).parent / "artifacts"
PILLARS = ["performance", "fairness", "robustness", "explainability", "privacy"]
VERDICT_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}

st.set_page_config(page_title="VERIFAI Showcase — Responsible AI", layout="wide")


# ---------- catalog ----------
def load_catalog() -> list[dict]:
    cards = []
    if not ART.exists():
        return cards
    for d in sorted(ART.iterdir()):
        card_f, report_f = d / "card.json", d / "report.json"
        if card_f.exists() and report_f.exists():
            card = json.loads(card_f.read_text(encoding="utf-8"))
            card["_dir"] = d
            cards.append(card)
    return cards


# ---------- generic Plotly renderer (the extensibility trick) ----------
def render_chart(spec: dict, base: Path):
    kind = spec.get("kind")
    title = spec.get("title", "")
    if kind == "bar":
        marker_color = spec.get("colors", spec.get("color", "#5B3FD6"))  # list = per-bar
        bar = go.Bar(x=spec["x"], y=spec["y"], marker_color=marker_color)
        if spec.get("hover"):
            bar.text = spec["hover"]
            bar.hovertemplate = "%{text}<br>%{y}<extra></extra>"
        fig = go.Figure(bar)
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""), yaxis_title=spec.get("y_title", ""))
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "line":
        fig = go.Figure(go.Scatter(x=spec["x"], y=spec["y"], mode="lines+markers"))
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""), yaxis_title=spec.get("y_title", ""))
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "heatmap":
        fig = go.Figure(go.Heatmap(z=spec["z"], x=spec.get("x"), y=spec.get("y"), colorscale="Blues"))
        fig.update_layout(title=title)
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "gauge":
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=spec["value"],
            gauge={"axis": {"range": [spec.get("min", 0), spec.get("max", 1)]}},
            title={"text": title}))
        st.plotly_chart(fig, use_container_width=True)
    elif kind == "images":
        cols = st.columns(min(3, max(1, len(spec.get("paths", [])))))
        for i, rel in enumerate(spec.get("paths", [])):
            p = base / rel
            if p.exists():
                cols[i % len(cols)].image(str(p), caption=spec.get("captions", [None] * 99)[i])
    else:
        st.json(spec)


# ---------- views ----------
def gallery(cards: list[dict]):
    st.title("VERIFAI — Responsible-AI Showcase")
    st.caption("Wähle ein Modell — und sieh seine Analyse über die vier Säulen: "
               "Fairness, Robustheit, Erklärbarkeit, Datenschutz.")
    if not cards:
        st.info("Noch keine Modelle. Erzeuge eins mit `python scripts/run_scenario.py scenarios/skin_cancer.yaml`.")
        return
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {card.get('emoji','🧠')} {card['name']}")
                st.caption(f"Domäne: {card.get('domain','?')}  ·  {card.get('dataset','')}")
                st.write(card.get("description", ""))
                if card.get("sample"):
                    st.warning("SAMPLE-Daten (Platzhalter, bis der echte Lauf drin ist)")
                if st.button("Analyse ansehen →", key=f"btn_{card['id']}"):
                    st.session_state["selected"] = card["id"]
                    st.rerun()


def dashboard(card: dict):
    base = card["_dir"]
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))

    if st.button("← Zurück zur Übersicht"):
        st.session_state.pop("selected", None); st.rerun()

    st.title(f"{card.get('emoji','🧠')} {card['name']}")
    st.write(f"**Domäne:** {report['domain']}  ·  **Modell:** `{report['model_id']}`  ·  "
             f"**Datensatz:** `{report['dataset_id']}`")
    if card.get("hf_url"):
        st.markdown(f"[🤗 Modell auf Hugging Face]({card['hf_url']})")
    if card.get("sample"):
        st.warning("Diese Ansicht zeigt SAMPLE-Daten — Platzhalter, bis der echte Engine-Lauf die Artefakte erzeugt.")

    by_pillar: dict[str, list] = {p: [] for p in PILLARS}
    for f in report["findings"]:
        by_pillar.setdefault(f["pillar"], []).append(f)

    # pillar overview row
    cols = st.columns(len(PILLARS))
    for c, p in zip(cols, PILLARS):
        items = by_pillar.get(p, [])
        worst = "info"
        for f in items:
            order = {"pass": 0, "info": 1, "warn": 2, "fail": 3}
            if order.get(f["verdict"], 1) >= order.get(worst, 1):
                worst = f["verdict"]
        c.metric(p.capitalize(), VERDICT_ICON.get(worst, "—") if items else "–")

    st.divider()
    for p in PILLARS:
        items = by_pillar.get(p, [])
        if not items:
            continue
        st.header(p.capitalize())
        for f in items:
            st.markdown(f"**{VERDICT_ICON.get(f['verdict'],'•')} {f['metric']}** — {f.get('summary','')}")
            details = f.get("details") or {}
            chart = details.get("chart")
            if chart:
                render_chart(chart, base)
            elif f.get("plots"):
                render_chart({"kind": "images", "paths": f["plots"]}, base)
            # optional secondary chart (e.g. a faithfulness gauge or subgroup gap)
            if details.get("chart2"):
                render_chart(details["chart2"], base)


# ---------- main ----------
cards = load_catalog()
sel = st.session_state.get("selected")
if sel:
    card = next((c for c in cards if c["id"] == sel), None)
    dashboard(card) if card else gallery(cards)
else:
    gallery(cards)
