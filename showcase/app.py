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


# ---------- snapshots: comparing runs, and refusing to ----------
def load_snapshots() -> list[dict]:
    """Every recorded run across every artifact folder, newest last."""
    snaps = []
    if not ART.exists():
        return snaps
    for d in sorted(ART.iterdir()):
        for f in sorted((d / "history").glob("*.json")) if (d / "history").is_dir() else []:
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            s["_file"] = f.name
            snaps.append(s)
    return sorted(snaps, key=lambda s: s.get("created_at", ""))


def comparability_key(snap: dict) -> tuple:
    """Runs are comparable only when scored on exactly the same rows.

    The manifest's content hash, not its path — a manifest can be regenerated
    with a different seed and keep its name.
    """
    ev = snap.get("eval_set") or {}
    return (ev.get("sha256"), ev.get("manifest"))


def group_snapshots(snaps: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for s in snaps:
        groups.setdefault(comparability_key(s), []).append(s)
    return groups


def _blocked_reason(snap: dict) -> str | None:
    """Why this run must not be plotted alongside the others."""
    if (snap.get("eval_set") or {}).get("sha256") is None:
        return "no evaluation manifest recorded, so there is nothing to match against"
    if snap.get("integrity") == "fail":
        return "its split was contaminated — the numbers are inflated by an unknown amount"
    if snap.get("integrity") != "pass":
        return "split integrity was never verified, so the numbers rest on an unchecked assumption"
    return None


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
        hm = go.Heatmap(z=spec["z"], x=spec.get("x"), y=spec.get("y"), colorscale="Blues",
                        zmin=spec.get("zmin"), zmax=spec.get("zmax"))
        if spec.get("text"):
            hm.text = spec["text"]
            hm.hovertemplate = "%{text}<extra></extra>"
        fig = go.Figure(hm)
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""),
                          yaxis_title=spec.get("y_title", ""),
                          # true classes read top-to-bottom, like a printed matrix
                          yaxis=dict(autorange="reversed"))
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
    snaps = load_snapshots()
    if snaps:
        if st.button(f"Compare runs ({len(snaps)} recorded) →"):
            st.session_state["compare"] = True
            st.rerun()

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


def comparison(snaps: list[dict]):
    st.title("Comparing runs")
    st.caption("Every recorded evaluation, grouped by the exact set of images it was scored on.")

    if st.button("← Back to overview"):
        st.session_state.pop("compare", None); st.rerun()

    with st.expander("Why runs are grouped, and when a comparison is refused"):
        st.markdown(
            "A difference between two numbers only means something if everything else was "
            "held equal. Two rules decide that here:\n\n"
            "1. **Same images.** Runs are grouped by the *content hash* of the evaluation "
            "manifest, not its filename — a manifest can be regenerated with a different "
            "seed and keep its name. Runs in different groups are never plotted together.\n"
            "2. **A verified split.** A run whose split was contaminated, or never checked, "
            "is excluded from the chart and listed with the reason. Its accuracy is inflated "
            "by an unknown amount, so plotting it beside an honest run would manufacture a "
            "comparison rather than report one.\n\n"
            "This is deliberately stricter than most dashboards. A green *+12 points* against "
            "a leaked baseline is exactly the claim this project exists to catch."
        )

    groups = group_snapshots(snaps)
    if not groups:
        st.info("No runs recorded yet. Every `run_scenario.py` invocation writes one.")
        return

    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for gi, (key, runs) in enumerate(order):
        ev = runs[0].get("eval_set") or {}
        name = (ev.get("manifest") or "unknown evaluation set").split("/")[-1]
        st.subheader(f"{name}  ·  {ev.get('n') or '?'} images")
        st.caption(f"content hash `{ev.get('sha256') or 'none'}`  ·  {len(runs)} run(s)")

        usable = [r for r in runs if _blocked_reason(r) is None]
        blocked = [(r, _blocked_reason(r)) for r in runs if _blocked_reason(r) is not None]

        for r, why in blocked:
            st.warning(f"**{r.get('label', r.get('scenario'))}** is excluded — {why}.")

        if len(usable) < 2:
            st.info(
                "Nothing to compare yet in this group: "
                + ("no run here has a verified split." if not usable
                   else "only one comparable run so far. Train a variant and re-run it "
                        "against this same manifest.")
            )
            st.divider(); continue

        keys = sorted({k for r in usable for k in r["metrics"]})
        default = [k for k in ("performance.accuracy", "performance.balanced_accuracy",
                               "robustness.mean_stability", "fairness.accuracy_gap",
                               "privacy.mia_auc") if k in keys]
        chosen = st.multiselect("Metrics", keys, default=default or keys[:5],
                                key=f"ms_{gi}")

        base = usable[0]
        rows = []
        for k in chosen:
            row = {"metric": k}
            for r in usable:
                row[r.get("label") or r["scenario"]] = r["metrics"].get(k)
            b, last = base["metrics"].get(k), usable[-1]["metrics"].get(k)
            row["Δ vs first"] = (None if b is None or last is None else round(last - b, 4))
            rows.append(row)
        st.dataframe(rows, width="stretch")

        if chosen:
            metric = st.selectbox("Chart", chosen, key=f"sb_{gi}")
            labels = [r.get("label") or r["scenario"] for r in usable]
            vals = [r["metrics"].get(metric) for r in usable]
            fig = go.Figure(go.Bar(x=labels, y=vals, marker_color="#5B3FD6"))
            fig.update_layout(title=metric, xaxis_title="Run", yaxis_title=metric)
            st.plotly_chart(fig, width="stretch")
        st.divider()

    if len(order) > 1:
        st.error(
            f"**{len(order)} groups above were not compared with each other.** They were "
            "scored on different sets of images, so a difference between them would measure "
            "the datasets, not the models."
        )


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
if st.session_state.get("compare"):
    comparison(load_snapshots())
    st.stop()
sel = st.session_state.get("selected")
if sel:
    card = next((c for c in cards if c["id"] == sel), None)
    dashboard(card) if card else gallery(cards)
else:
    gallery(cards)
