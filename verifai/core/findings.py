"""Findings data model — the single result format for the whole pipeline.

Ported/simplified from verifai_2_0's findings layer. The engine produces
`Finding`s, the exporter serializes them to JSON, the Streamlit showcase renders
them. One data model from end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

Pillar = Literal["integrity", "fairness", "robustness", "explainability",
                 "privacy", "performance"]
Verdict = Literal["pass", "warn", "fail", "info"]
Domain = Literal["image", "text", "tabular", "llm"]


@dataclass
class Finding:
    pillar: Pillar
    metric: str                     # e.g. "subgroup_sensitivity_gap"
    domain: Domain
    value: Any                      # scalar or dict of numbers
    verdict: Verdict = "info"
    summary: str = ""               # one human-readable sentence
    details: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)  # relative paths under the report's plot dir


@dataclass
class Report:
    scenario: str
    domain: Domain
    model_id: str
    dataset_id: str
    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)   # sample_size, seed, versions, timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
