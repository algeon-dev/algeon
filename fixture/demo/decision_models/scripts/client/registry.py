"""Semantic registry: a Parquet table of decisions keyed by content and question-set hash.

One registry file holds the decisions of one question set from one backend
family. A lookup hit means the entity state and the questions are byte-identical
to an earlier decision, so no backend call is needed. The engine-facing table
carries only the typed answer columns next to the case's key columns.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from client.decision_client import Decision, validate_questions

BASE_FIELDS = (
    ("entity_key", pa.string()),
    ("content_hash", pa.string()),
    ("question_set_hash", pa.string()),
    ("model_id", pa.string()),
    ("decided_at", pa.timestamp("us")),
)
TAIL_FIELDS = (
    ("input_tokens", pa.int64()),
    ("latency_ms", pa.int64()),
)


def answer_fields(questions: dict) -> list[tuple[str, pa.DataType]]:
    fields = []
    for name, spec in questions.items():
        if spec["type"] == "choice":
            fields.append((name, pa.string()))
            fields.append((f"{name}_p", pa.float64()))
        else:
            fields.append((name, pa.float64()))
    return fields


def registry_schema(questions: dict) -> pa.Schema:
    return pa.schema([pa.field(name, kind) for name, kind in (*BASE_FIELDS, *answer_fields(questions), *TAIL_FIELDS)])


def answer_columns(decision: Decision, questions: dict) -> dict:
    row = {}
    for name, spec in questions.items():
        answer = decision.answers[name]
        if spec["type"] == "choice":
            row[name] = answer["choice"]
            row[f"{name}_p"] = float(answer["p"])
        else:
            row[name] = float(answer["value"])
    return row


def answers_from_row(row: dict, questions: dict) -> dict[str, dict]:
    answers = {}
    for name, spec in questions.items():
        if spec["type"] == "choice":
            answers[name] = {"type": "choice", "choice": row[name], "p": row[f"{name}_p"]}
        else:
            answers[name] = {"type": spec["type"], "value": row[name]}
    return answers


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


class Registry:
    def __init__(self, path: Path, questions: dict):
        validate_questions(questions)
        self.path = Path(path)
        self.questions = questions
        self.schema = registry_schema(questions)
        self._rows: dict[tuple[str, str], dict] = {}
        # Keys written by this process; a refresh run re-decides a state once and then reuses its own answer.
        self.fresh: set[tuple[str, str]] = set()
        self.loaded_rows = 0
        if self.path.exists():
            table = pq.read_table(self.path)
            missing = set(self.schema.names) - set(table.schema.names)
            if missing:
                raise ValueError(
                    f"registry {self.path} lacks columns {sorted(missing)}; it was written for another question set")
            for row in table.select(self.schema.names).to_pylist():
                self._rows[(row["content_hash"], row["question_set_hash"])] = row
            self.loaded_rows = len(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def lookup(self, content_hash: str, question_set_hash: str) -> Decision | None:
        row = self._rows.get((content_hash, question_set_hash))
        if row is None:
            return None
        decided_at = row["decided_at"]
        if not isinstance(decided_at, datetime):
            decided_at = datetime.fromisoformat(str(decided_at))
        return Decision(
            entity_key=row["entity_key"], content_hash=content_hash, question_set_hash=question_set_hash,
            model_id=row["model_id"], decided_at=decided_at, answers=answers_from_row(row, self.questions),
            input_tokens=row["input_tokens"], output_tokens=None, cost_usd=None,
            latency_ms=row["latency_ms"] or 0, source="registry")

    def upsert(self, decisions: list[Decision]) -> None:
        for decision in decisions:
            row = {
                "entity_key": decision.entity_key,
                "content_hash": decision.content_hash,
                "question_set_hash": decision.question_set_hash,
                "model_id": decision.model_id,
                "decided_at": decision.decided_at,
                **answer_columns(decision, self.questions),
                "input_tokens": decision.input_tokens,
                "latency_ms": decision.latency_ms,
            }
            self._rows[(decision.content_hash, decision.question_set_hash)] = row
            self.fresh.add((decision.content_hash, decision.question_set_hash))

    def table(self) -> pa.Table:
        rows = sorted(self._rows.values(), key=lambda row: (row["entity_key"], row["content_hash"]))
        return pa.Table.from_pylist(rows, schema=self.schema)

    def save(self) -> None:
        write_parquet_atomic(self.table(), self.path)

    def engine_table(self, decisions: list[Decision], keys: dict[str, pa.Array | list]) -> pa.Table:
        """Typed decision table for `CREATE EXTERNAL TABLE`: key columns, then answer columns.

        `keys` supplies one typed column per entry, each as long as `decisions`
        and in the same order, so the recompute SQL joins on native types and
        never parses `entity_key`.
        """
        columns: dict[str, pa.Array | list] = {"entity_key": [decision.entity_key for decision in decisions]}
        for name, values in keys.items():
            if len(values) != len(decisions):
                raise ValueError(f"key column {name!r} has {len(values)} values for {len(decisions)} decisions")
            columns[name] = values
        for name, kind in answer_fields(self.questions):
            columns[name] = pa.array([answer_columns(decision, self.questions)[name] for decision in decisions], kind)
        return pa.table(columns)
