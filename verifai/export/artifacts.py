"""Export a Report to static artifacts the Streamlit showcase reads.

Writes:
  <out_dir>/<scenario>/report.json          (the full Report — always "latest")
  <out_dir>/<scenario>/card.json            (tile metadata)
  <out_dir>/<scenario>/plots/*.png          (referenced by Finding.plots)
  <out_dir>/<scenario>/history/<ts>.json    (one snapshot per run, kept)

`report.json` is overwritten every run so the dashboard is unchanged. Snapshots
accumulate beside it so a model can be compared against its own earlier versions —
without which each training experiment would silently overwrite the evidence of
the last one.

No DB, no server — just files that get committed into the showcase.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from verifai.core.findings import Report


def write_report(report: Report, out_dir: str = "showcase/artifacts",
                 card: dict | None = None) -> Path:
    """Write report.json (+ card.json so the showcase auto-lists it as a tile).

    `card` overrides/extends the auto-generated tile metadata, e.g.
    {"name": "...", "emoji": "🔬", "description": "...", "hf_url": "...", "dataset": "..."}.
    Adding a new model = one more run() -> one more folder -> one more tile.
    """
    base = Path(out_dir) / report.scenario
    (base / "plots").mkdir(parents=True, exist_ok=True)

    with open(base / "report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    card_out = {
        "id": report.scenario,
        "name": report.model_id,
        "emoji": "🧠",
        "domain": report.domain,
        "dataset": report.dataset_id,
        "description": "",
        "sample": False,
        **(card or {}),
    }
    with open(base / "card.json", "w", encoding="utf-8") as f:
        json.dump(card_out, f, ensure_ascii=False, indent=2)

    write_snapshot(report, base)
    return base / "report.json"


# Deep enough to reach value["per_class"]["melanoma"]["sensitivity"] — the number
# a cost-sensitive experiment is actually about. At depth 2 it was invisible, so
# the comparison could show accuracy moving while the metric that motivated the
# change was missing from the table.
MAX_FLATTEN_DEPTH = 3


def _flatten(value: Any, prefix: str, out: dict[str, float], depth: int = 0) -> None:
    """Collect the numeric leaves of a finding's `value` as dotted keys.

    Generic on purpose: a new metric becomes comparable without this module
    learning anything about it.
    """
    if isinstance(value, bool):            # bool is an int; not a metric
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict) and depth < MAX_FLATTEN_DEPTH:
        for k, v in value.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)


def snapshot_metrics(report: Report) -> dict[str, float]:
    """Every numeric result in the report, keyed as `<pillar>.<path>`."""
    flat: dict[str, float] = {}
    for f in report.findings:
        _flatten(f.value, f.pillar, flat)
    return flat


def write_snapshot(report: Report, base: Path) -> Path:
    """One immutable record per run, carrying what makes it comparable (or not)."""
    hist = base / "history"
    hist.mkdir(parents=True, exist_ok=True)
    meta = report.meta or {}

    integrity = next((f.verdict for f in report.findings if f.pillar == "integrity"), None)
    snap = {
        "created_at": report.created_at,
        "scenario": report.scenario,
        "label": meta.get("label") or report.model_id,
        "model_id": report.model_id,
        "dataset_id": report.dataset_id,
        # the comparability key: same rows, and an evaluation worth believing
        "eval_set": meta.get("eval_set") or {},
        "integrity": integrity,
        "device": meta.get("device"),
        "seed": meta.get("seed"),
        "verdicts": {f.pillar: f.verdict for f in report.findings},
        "metrics": snapshot_metrics(report),
    }
    name = re.sub(r"[^0-9A-Za-z]", "-", report.created_at) + ".json"
    path = hist / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    return path
