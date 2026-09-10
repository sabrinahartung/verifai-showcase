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


def direction_for(key: str, snaps: list[dict]) -> str | None:
    """'higher' | 'lower' | None — as declared by the metric, never inferred.

    Patterns may contain `*` (e.g. `performance.per_class.*.sensitivity`). An
    undeclared metric stays unranked: showing a value is honest, calling it better
    is not.
    """
    import fnmatch
    for s in snaps:
        for pattern, d in (s.get("directions") or {}).items():
            if key == pattern or fnmatch.fnmatch(key, pattern):
                return d
    return None


def best_run(key: str, runs: list[dict], direction: str | None) -> str | None:
    """Label of the run leading on this metric, or None if the metric is unranked."""
    if not direction:
        return None
    vals = [(r, r["metrics"].get(key)) for r in runs]
    vals = [(r, v) for r, v in vals if v is not None]
    if not vals:
        return None
    pick = max(vals, key=lambda rv: rv[1]) if direction == "higher" else min(vals, key=lambda rv: rv[1])
    return pick[0].get("label") or pick[0]["scenario"]


def dominated_by(runs: list[dict], keys: list[str], dirs: dict[str, str | None]) -> dict[str, str]:
    """Which runs are beaten on *every* ranked metric by some other run.

    A dominated run can be dismissed on the evidence alone. Anything left over is
    a genuine trade-off, where choosing requires saying what the model is *for* —
    which no amount of charting can decide.
    """
    ranked = [k for k in keys if dirs.get(k)]
    if not ranked:
        return {}
    out: dict[str, str] = {}
    for a in runs:
        la = a.get("label") or a["scenario"]
        for b in runs:
            if a is b:
                continue
            lb = b.get("label") or b["scenario"]
            better_somewhere = False
            worse_somewhere = False
            for k in ranked:
                va, vb = a["metrics"].get(k), b["metrics"].get(k)
                if va is None or vb is None:
                    worse_somewhere = True      # cannot claim dominance on missing data
                    break
                if va == vb:
                    continue
                a_wins = (va > vb) if dirs[k] == "higher" else (va < vb)
                better_somewhere |= a_wins
                worse_somewhere |= not a_wins
            if not worse_somewhere and better_somewhere:
                out[lb] = la                     # b is dominated by a
    return out


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
        if spec.get("y_lo") and spec.get("y_hi"):
            # Wilson intervals are asymmetric, so plus/minus arms differ.
            bar.error_y = dict(
                type="data", symmetric=False,
                array=[hi - y for y, hi in zip(spec["y"], spec["y_hi"])],
                arrayminus=[y - lo for y, lo in zip(spec["y"], spec["y_lo"])],
                thickness=1.4, width=6, color="#455A64")
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
DEFAULT_GROUP = "Models"


def _tile(card: dict, key: str):
    """One evaluated configuration."""
    with st.container(border=True):
        st.markdown(f"### {card.get('emoji','🧠')} {card['name']}")
        st.caption(f"Domain: {card.get('domain','?')}  ·  {card.get('dataset','')}")
        st.write(card.get("description", ""))
        if card.get("sample"):
            st.warning("SAMPLE data (placeholder until the real run lands)")
        if st.button("View analysis →", key=key):
            st.session_state["selected"] = card["id"]
            st.rerun()


def _lineage_card(lineage: str, members: list[dict], key: str):
    """Several configurations of one investigation, collapsed into a single card.

    Five tiles for five decision rules on one checkpoint is a wall, not a gallery.
    The card leads with the comparison, because these exist to be read against each
    other — a single one of them in isolation is the least useful view of the set.
    """
    first = members[0]
    with st.container(border=True):
        st.markdown(f"### {first.get('emoji','🧠')} {lineage}")
        st.caption(f"Domain: {first.get('domain','?')}  ·  {first.get('dataset','')}  ·  "
                   f"**{len(members)} configurations**")
        st.write("  ·  ".join(m["name"] for m in members))
        if st.button(f"Compare {len(members)} configurations →", key=f"{key}_cmp"):
            st.session_state["compare"] = True
            st.session_state["compare_lineage"] = lineage
            st.rerun()
        pick = st.selectbox("or open one", [m["name"] for m in members],
                            key=f"{key}_sel", label_visibility="collapsed")
        if st.button("View analysis →", key=f"{key}_one"):
            st.session_state["selected"] = next(m["id"] for m in members if m["name"] == pick)
            st.rerun()


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

    snaps = load_snapshots()
    if snaps:
        if st.button(f"Compare all runs ({len(snaps)} recorded) →"):
            st.session_state["compare"] = True
            st.session_state.pop("compare_lineage", None)
            st.rerun()

    if not cards:
        st.info("No models yet. Create one with `python scripts/run_scenario.py scenarios/skin_cancer.yaml`.")
        return

    # sections in a stable order, with placeholder data pushed to the end
    groups: dict[str, list[dict]] = {}
    for c in cards:
        groups.setdefault(c.get("group") or DEFAULT_GROUP, []).append(c)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0].startswith("Demo"), kv[0]))

    for gi, (group, members) in enumerate(ordered):
        st.divider()
        st.subheader(group)

        # collapse configurations of one investigation into a single card
        lineages: dict[str, list[dict]] = {}
        for c in members:
            lineages.setdefault(c.get("lineage") or c["id"], []).append(c)

        cols = st.columns(3)
        for i, (lineage, ms) in enumerate(sorted(lineages.items())):
            with cols[i % 3]:
                if len(ms) == 1:
                    _tile(ms[0], key=f"btn_{gi}_{i}")
                else:
                    _lineage_card(lineage, sorted(ms, key=lambda m: m["name"]),
                                  key=f"lin_{gi}_{i}")


def comparison(snaps: list[dict], cards: list[dict] | None = None):
    lineage = st.session_state.get("compare_lineage")
    st.title("Comparing runs" + (f" — {lineage}" if lineage else ""))
    st.caption("Every recorded evaluation, grouped by the exact set of images it was scored on.")

    if st.button("← Back to overview"):
        st.session_state.pop("compare", None)
        st.session_state.pop("compare_lineage", None)
        st.rerun()

    # A lineage narrows *what is shown*; it never widens what may be compared.
    # Grouping stays keyed on the evaluation set, so two runs of one lineage scored
    # on different manifests still land in different groups.
    if lineage and cards:
        ids = {c["id"] for c in cards if (c.get("lineage") or c["id"]) == lineage}
        snaps = [s for s in snaps if s.get("scenario") in ids]
        st.caption(f"Filtered to the {len(ids)} configuration(s) in this lineage. "
                   f"Comparability is still decided by the evaluation set, not the lineage.")

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

    with st.expander("Why several runs, and how to read them"):
        st.markdown(
            "Each run is one configuration — a model, plus the decision rule that reads "
            "its probabilities. They are not competitors in a race with a winner; "
            "together they map a **trade-off**.\n\n"
            "Two things here *are* decidable from the data:\n\n"
            "- **Which run leads on a given metric** — shown in the `best` column.\n"
            "- **Whether a run is beaten on everything.** If another run is at least as "
            "good on every selected metric, the loser can be dropped with no judgement "
            "call at all.\n\n"
            "What is *not* decidable: which of the surviving runs is best overall. A "
            "configuration catching 94% of melanomas while misreading a third of moles "
            "is better or worse than the reverse **depending entirely on what the tool "
            "is for**. Collapsing that into one score would not resolve the question, "
            "it would only hide it."
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

        # Re-running the same configuration records another snapshot, which is the
        # point of a history — but three identical rows help nobody read a table.
        # Collapse to the newest per configuration, with the full record a click away.
        by_label: dict[str, list[dict]] = {}
        for r in sorted(usable, key=lambda r: r.get("created_at", "")):
            by_label.setdefault(r.get("label") or r["scenario"], []).append(r)
        repeats = sum(len(v) - 1 for v in by_label.values())
        if repeats and not st.checkbox(
                f"Show every recorded run ({repeats} repeat(s) of a configuration hidden)",
                key=f"all_{gi}"):
            usable = [v[-1] for v in by_label.values()]

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
        default = [k for k in ("performance.per_class.melanoma.sensitivity",
                               "performance.per_class.melanoma.ppv_test_prevalence",
                               "performance.accuracy", "performance.balanced_accuracy",
                               "performance.top3_accuracy", "robustness.mean_stability",
                               "fairness.accuracy_gap", "privacy.mia_auc") if k in keys]
        chosen = st.multiselect("Metrics to compare", keys, default=default or keys[:6],
                                key=f"ms_{gi}")
        if not chosen:
            st.divider(); continue

        dirs = {k: direction_for(k, usable) for k in chosen}
        labels = [r.get("label") or r["scenario"] for r in usable]

        # --- is there an outright winner, or is this a trade-off? ---
        dom = dominated_by(usable, chosen, dirs)
        survivors = [l for l in labels if l not in dom]
        if len(survivors) == 1 and len(labels) > 1:
            st.success(f"**{survivors[0]}** is at least as good as every other run on all "
                       f"selected metrics. On this evidence it is the one to keep.")
        else:
            st.info(
                f"**No single best run.** {len(survivors)} of {len(labels)} runs trade off "
                "against each other: each is better on some selected metric and worse on "
                "another. Which one is *right* depends on what the model is for — a triage "
                "tool and a rule-out tool want opposite ends of this table. That is a "
                "decision about intended use, not one the data can settle."
            )
        for loser, winner in dom.items():
            st.caption(f"↳ **{loser}** is beaten by *{winner}* on every selected metric, "
                       f"so it can be dismissed without a value judgement.")

        # --- table: value, direction, and who leads each metric ---
        rows = []
        for k in chosen:
            d = dirs[k]
            arrow = {"higher": "↑ better", "lower": "↓ better"}.get(d, "—")
            leader = best_run(k, usable, d)
            row = {"metric": k, "good": arrow}
            for r in usable:
                lab = r.get("label") or r["scenario"]
                v = r["metrics"].get(k)
                row[lab] = None if v is None else round(v, 4)
            row["best"] = leader or "—"
            rows.append(row)
        st.dataframe(rows, width="stretch")
        st.caption("**best** is only filled in where the metric declared which direction is "
                   "an improvement. An undeclared metric is shown but not ranked.")

        # --- the trade-off, seen directly ---
        ranked_keys = [k for k in chosen if dirs.get(k)]
        if len(ranked_keys) >= 2:
            c1, c2 = st.columns(2)
            xk = c1.selectbox("Trade-off: x", ranked_keys, index=0, key=f"x_{gi}")
            yk = c2.selectbox("Trade-off: y", ranked_keys,
                              index=min(1, len(ranked_keys) - 1), key=f"y_{gi}")
            fig = go.Figure()
            for r in usable:
                lab = r.get("label") or r["scenario"]
                fig.add_trace(go.Scatter(
                    x=[r["metrics"].get(xk)], y=[r["metrics"].get(yk)], mode="markers+text",
                    text=[lab], textposition="top center", name=lab, marker=dict(size=14)))
            fig.update_layout(
                title=f"{yk} against {xk} — each point is one run",
                xaxis_title=f"{xk} ({dirs.get(xk, '')} is better)",
                yaxis_title=f"{yk} ({dirs.get(yk, '')} is better)", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        metric = st.selectbox("Bar chart", chosen, key=f"sb_{gi}")
        leader = best_run(metric, usable, dirs.get(metric))
        fig = go.Figure(go.Bar(
            x=labels, y=[r["metrics"].get(metric) for r in usable],
            marker_color=["#2E9E5B" if l == leader else "#5B3FD6" for l in labels]))
        fig.update_layout(title=f"{metric}"
                                + (f"  ·  best: {leader}" if leader else "  ·  unranked"),
                          xaxis_title="Run", yaxis_title=metric)
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
    comparison(load_snapshots(), cards)
    st.stop()
sel = st.session_state.get("selected")
if sel:
    card = next((c for c in cards if c["id"] == sel), None)
    dashboard(card) if card else gallery(cards)
else:
    gallery(cards)
