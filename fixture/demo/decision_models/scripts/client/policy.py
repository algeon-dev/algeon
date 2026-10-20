"""Action thresholds: accept, review or reject each decision on its action noul.

The action question is the `noul` head tied to what the recompute will do with
the entity (merge, exclude, escalate). A choice head can be required to agree;
when the heads disagree the entity goes to the review queue instead of being
accepted or rejected on one head alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from client.decision_client import Decision

BANDS = ("accept", "review", "reject")


class Policy:
    def __init__(self, spec: dict):
        self.action_question = spec["action_question"]
        self.accept_min = float(spec.get("accept_min", 0.70))
        self.review_min = float(spec.get("review_min", 0.30))
        self.require = {name: set(options) for name, options in (spec.get("require") or {}).items()}
        self.spec = spec
        if not 0.0 <= self.review_min <= self.accept_min <= 1.0:
            raise ValueError("policy needs 0 <= review_min <= accept_min <= 1")

    @classmethod
    def load(cls, path: Path) -> "Policy":
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))

    def band(self, decision: Decision) -> str:
        value = float(decision.value(self.action_question))
        if value < self.review_min:
            return "reject"
        agrees = all(decision.value(name) in options for name, options in self.require.items())
        if value >= self.accept_min and agrees:
            return "accept"
        return "review"

    def apply(self, decisions: list[Decision]) -> tuple[dict[str, str], dict[str, int]]:
        bands = {decision.entity_key: self.band(decision) for decision in decisions}
        counts = {band: 0 for band in BANDS}
        for band in bands.values():
            counts[band] += 1
        return bands, counts
