"""Export a Report to static artifacts the Streamlit showcase reads.

Writes:
  <out_dir>/<scenario>/report.json     (the full Report)
  <out_dir>/<scenario>/plots/*.png     (referenced by Finding.plots)

No DB, no server — just files that get committed into the showcase.
"""
from __future__ import annotations

import json
from pathlib import Path

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

    return base / "report.json"
