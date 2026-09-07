"""VERIFAI Showcase — Streamlit.

UX: a gallery of model tiles -> click a model -> a Responsible-AI dashboard that a
non-specialist can actually read: every metric states what it measures, what the
chart shows, and what it cannot tell you.

Extensible by design: every model is a folder under ./artifacts/<model_id>/ with
  card.json    -> tile metadata (name, domain, description, hf_url, sample?)
  report.json  -> the Findings produced by the engine (verifai.export.artifacts)
  plots/       -> optional images (e.g. Grad-CAM overlays)
Add a folder -> a new tile appears. No code change needed.

The explanatory text travels *with* the finding (details["explain"]), so a new
metric brings its own wording and still needs no change here.

Results are PRECOMPUTED (run once by the engine) so this app stays free & always-on.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

ART = Path(__file__).parent / "artifacts"
# integrity comes first on purpose: every other pillar is conditional on it.
PILLARS = ["integrity", "performance", "fairness", "robustness", "explainability", "privacy"]

# The plain-language question each pillar answers, for readers who have never
# seen a Responsible-AI report.
PILLAR_QUESTION = {
    "integrity":      "Can these results be trusted at all?",
    "performance":    "Does the model get the answer right?",
    "fairness":       "Does it work equally well for everyone?",
    "robustness":     "Does it stay reliable when the image is imperfect?",
    "explainability": "Can we see why it decided what it decided?",
    "privacy":        "Could the model leak the data it was trained on?",
}

# icon, short label, what the verdict actually means
VERDICT = {
    "pass": ("✅", "Pass",       "Meets the threshold set for this metric."),
    "warn": ("⚠️", "Warning",    "Below the comfortable range — worth a closer look."),
    "fail": ("❌", "Fail",       "Clearly below the threshold set for this metric."),
    "info": ("ℹ️", "No verdict", "Measured, but the sample is too small (or required data "
                                "is missing) to claim a pass or fail."),
}
VERDICT_ORDER = {"pass": 0, "info": 1, "warn": 2, "fail": 3}

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
def _scale(spec: dict):
    """A value placed on a labeled band scale.

    Replaces the old dial gauge: a dial shows a number, this shows whether the
    number is a *good* number. Bands come from the metric, so each metric defines
    its own good/bad semantics (e.g. for MIA-AUC, low is good).
    """
    lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
    val = spec.get("value")
    bands = spec.get("bands") or [{"to": hi, "label": "", "color": "#D9E2EC"}]

    fig = go.Figure()
    # invisible trace so the axes exist for the shapes/annotations below
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0.5, 0.5], mode="markers",
                             marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    start = lo
    for b in bands:
        end = float(b.get("to", hi))
        fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1,
                      fillcolor=b.get("color", "#D9E2EC"), line_width=0, layer="below")
        if b.get("label"):
            fig.add_annotation(x=(start + end) / 2, y=0.5, text=b["label"], showarrow=False,
                               font=dict(size=13, color="#243B53"))
        start = end

    if val is not None:
        val = float(val)
        fig.add_shape(type="line", x0=val, x1=val, y0=-0.08, y1=1.08,
                      line=dict(color="#102A43", width=4))
        # no explicit colour: this sits on the plot background, so the Streamlit
        # template must pick it or it goes invisible in dark mode. Same below.
        fig.add_annotation(x=val, y=1.42, text=f"<b>{val:g}</b>", showarrow=False,
                           font=dict(size=20))
    if spec.get("value_label"):
        fig.add_annotation(x=(lo + hi) / 2, y=-0.75, text=spec["value_label"], showarrow=False,
                           font=dict(size=12), xanchor="center", align="center")

    fig.update_xaxes(range=[lo, hi], showgrid=False, zeroline=False,
                     tickvals=spec.get("ticks", [lo, hi]),
                     ticktext=spec.get("tick_labels"))
    fig.update_yaxes(range=[-1.0, 1.8], visible=False)
    # transparent plot area so only the bands carry colour, in either theme.
    # The band labels keep an explicit dark colour because they always sit on a
    # light pastel band.
    fig.update_layout(title=spec.get("title", ""), height=210, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig, width="stretch")


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
        st.plotly_chart(fig, width="stretch")
    elif kind == "line":
        fig = go.Figure(go.Scatter(x=spec["x"], y=spec["y"], mode="lines+markers"))
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""), yaxis_title=spec.get("y_title", ""))
        st.plotly_chart(fig, width="stretch")
    elif kind == "heatmap":
        fig = go.Figure(go.Heatmap(z=spec["z"], x=spec.get("x"), y=spec.get("y"), colorscale="Blues"))
        fig.update_layout(title=title)
        st.plotly_chart(fig, width="stretch")
    elif kind in ("scale", "gauge"):   # "gauge" kept as an alias so older reports still render
        _scale(spec)
    elif kind == "images":
        paths = spec.get("paths", [])
        captions = spec.get("captions") or []
        cols = st.columns(min(3, max(1, len(paths))))
        for i, rel in enumerate(paths):
            p = base / rel
            if p.exists():
                cols[i % len(cols)].image(str(p), caption=captions[i] if i < len(captions) else None)
    else:
        st.json(spec)


# ---------- explanatory text that ships with the finding ----------
def render_explain(finding: dict):
    """Render details["explain"] = {what, how, limits} — all keys optional."""
    ex = (finding.get("details") or {}).get("explain") or {}
    if ex.get("what"):
        st.write(ex["what"])
    if finding.get("summary"):
        st.info(f"**Result:** {finding['summary']}")
    return ex


def render_caveats(ex: dict):
    if not (ex.get("how") or ex.get("limits")):
        return
    with st.expander("How to read this chart"):
        if ex.get("how"):
            st.markdown(f"**Reading the chart**  \n{ex['how']}")
        if ex.get("limits"):
            st.markdown(f"**What it does _not_ tell you**  \n{ex['limits']}")


# ---------- views ----------
def gallery(cards: list[dict]):
    st.title("VERIFAI — Responsible-AI Showcase")
    st.caption("Pick a model — and see its analysis across the five pillars: performance, "
               "fairness, robustness, explainability, privacy.")

    with st.expander("What am I looking at?"):
        st.markdown(
            "Every model here has been put through the same battery of checks. Instead of a "
            "single accuracy number, each run asks five separate questions:\n\n"
            + "\n".join(f"- **{p.capitalize()}** — {PILLAR_QUESTION[p]}" for p in PILLARS)
            + "\n\nThe results were computed once by the engine in this repo and stored as "
              "files, so this page is just a reader — nothing is recomputed when you click. "
              "Every metric states what it measures and, just as importantly, when the sample "
              "is too small to justify a verdict."
        )

    if not cards:
        st.info("No models yet. Create one with `python scripts/run_scenario.py scenarios/skin_cancer.yaml`.")
        return
    cols = st.columns(3)
    for i, card in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {card.get('emoji','🧠')} {card['name']}")
                st.caption(f"Domain: {card.get('domain','?')}  ·  {card.get('dataset','')}")
                st.write(card.get("description", ""))
                if card.get("sample"):
                    st.warning("SAMPLE data (placeholder until the real run lands)")
                if st.button("View analysis →", key=f"btn_{card['id']}"):
                    st.session_state["selected"] = card["id"]
                    st.rerun()


def dashboard(card: dict):
    base = card["_dir"]
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))

    if st.button("← Back to overview"):
        st.session_state.pop("selected", None); st.rerun()

    st.title(f"{card.get('emoji','🧠')} {card['name']}")
    st.write(f"**Domain:** {report['domain']}  ·  **Model:** `{report['model_id']}`  ·  "
             f"**Dataset:** `{report['dataset_id']}`")
    if card.get("hf_url"):
        st.markdown(f"[🤗 Model on Hugging Face]({card['hf_url']})")
    if card.get("sample"):
        st.warning("This view shows SAMPLE data — a placeholder until the real engine run produces the artifacts.")

    n = (report.get("meta") or {}).get("sample_size")
    if n:
        st.caption(f"Everything below was computed on {n} image(s). Small samples are marked as "
                   f"such and deliberately do not get a pass/fail verdict.")

    by_pillar: dict[str, list] = {p: [] for p in PILLARS}
    for f in report["findings"]:
        by_pillar.setdefault(f["pillar"], []).append(f)

    # ---- pillar overview row: icon + what the icon means ----
    st.subheader("At a glance")
    cols = st.columns(len(PILLARS))
    for c, p in zip(cols, PILLARS):
        items = by_pillar.get(p, [])
        if not items:
            c.metric(p.capitalize(), "–", help="Not evaluated in this run.")
            continue
        worst = max((f["verdict"] for f in items), key=lambda v: VERDICT_ORDER.get(v, 1))
        icon, label, meaning = VERDICT[worst]
        c.metric(p.capitalize(), icon, help=f"{PILLAR_QUESTION[p]}\n\n**{label}** — {meaning}")
        c.caption(label)

    with st.expander("What do the icons mean?"):
        for v in ("pass", "warn", "fail", "info"):
            icon, label, meaning = VERDICT[v]
            st.markdown(f"{icon} **{label}** — {meaning}")

    st.divider()
    for p in PILLARS:
        items = by_pillar.get(p, [])
        if not items:
            continue
        st.header(p.capitalize())
        st.caption(PILLAR_QUESTION[p])
        for f in items:
            icon, label, _ = VERDICT.get(f["verdict"], VERDICT["info"])
            st.markdown(f"##### {icon} `{f['metric']}` · {label}")
            ex = render_explain(f)
            details = f.get("details") or {}
            chart = details.get("chart")
            if chart:
                render_chart(chart, base)
            elif f.get("plots"):
                render_chart({"kind": "images", "paths": f["plots"]}, base)
            # optional secondary chart (e.g. a faithfulness scale or subgroup gap)
            if details.get("chart2"):
                render_chart(details["chart2"], base)
            render_caveats(ex)
            st.divider()


# ---------- main ----------
cards = load_catalog()
sel = st.session_state.get("selected")
if sel:
    card = next((c for c in cards if c["id"] == sel), None)
    dashboard(card) if card else gallery(cards)
else:
    gallery(cards)
