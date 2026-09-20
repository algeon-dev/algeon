#!/usr/bin/env python3
"""Classify ClickBench cross-engine outputs into parity behavior categories.

Reads rows.tsv or rows.jsonl produced by clickbench_cross_engine_outputs.py
and emits one classification row per query covering:

  all_engines_same   ; all engines returned identical rows and values
  tie_divergence     ; engines returned same row count but different tied rows
                        (expected for waiver queries q18,q25,q26,q31,q32,q33,q39,q40,q41,q42)
  float_variance     ; engines returned same row count but differ in Float64 value
                        (expected for q04)
  engine_instability ; same engine changed output between iterations
  correctness_bug    ; reference engines agree, algeon_datafusion differs (expected for q23)
  insufficient_data  ; fewer than two engines with successful output
  unexpected_mismatch; mismatch not explained by known SQL nondeterminism or float variance
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


FLOAT_VARIANCE_QUERY_IDS = frozenset(["q04"])
TIE_DIVERGENCE_QUERY_IDS = frozenset(
    ["q18", "q25", "q26", "q31", "q32", "q33", "q39", "q40", "q41", "q42"]
)
REFERENCE_ENGINES = ("duckdb", "cpu_datafusion")
NATIVE_ENGINE = "algeon_datafusion"

OUTPUT_HEADER = [
    "query_id",
    "classification",
    "engine_instability",
    "row_count_agreement",
    "fingerprint_agreement",
    "unstable_engines",
    "mismatched_engines",
    "reference_fingerprint",
    "algeon_datafusion_fingerprint",
    "notes",
]


@dataclass
class EngineEvidence:
    engine: str
    stable: bool
    row_count: str | None
    fingerprint: str | None


@dataclass
class ClassificationRow:
    query_id: str
    classification: str
    engine_instability: str  # "true" | "false"
    row_count_agreement: str  # "true" | "false" | "n/a"
    fingerprint_agreement: str  # "true" | "false" | "n/a"
    unstable_engines: str  # comma-sep or "none"
    mismatched_engines: str  # comma-sep or "none"
    reference_fingerprint: str  # consensus fp, "disagreement", or "none"
    algeon_datafusion_fingerprint: str  # fp or "none"
    notes: str


def main() -> int:
    args = parse_args()
    rows = load_input(args)

    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    query_ids = sorted(by_query, key=lambda q: int(q[1:]))
    classifications = [classify_query(qid, by_query[qid]) for qid in query_ids]

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = args.tsv or out_dir / "classification.tsv"
    jsonl_path = args.jsonl or out_dir / "classification.jsonl"

    write_tsv(tsv_path, classifications)
    write_jsonl(jsonl_path, classifications)

    print_summary(classifications)
    print(f"clickbench_cross_engine_classification_tsv={tsv_path}")
    print(f"clickbench_cross_engine_classification_jsonl={jsonl_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="rows.tsv or rows.jsonl from clickbench_cross_engine_outputs.py",
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: /tmp/clickbench_cross_engine_classify_<ts>).")
    parser.add_argument("--tsv", type=Path, help="Override classification TSV path.")
    parser.add_argument("--jsonl", type=Path, help="Override classification JSONL path.")
    return parser.parse_args()


def _default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"clickbench_cross_engine_classify_{stamp}"


def load_input(args: argparse.Namespace) -> list[dict]:
    path = args.input
    if path is None:
        raise SystemExit(
            "input file required; pass rows.tsv or rows.jsonl from clickbench_cross_engine_outputs.py"
        )
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    return _load_tsv(path)


def _load_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_query(query_id: str, rows: list[dict]) -> ClassificationRow:
    by_engine: dict[str, list[dict]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(row)

    evidences = {engine: _engine_evidence(engine, er) for engine, er in by_engine.items()}

    unstable_engines = sorted(e for e, ev in evidences.items() if not ev.stable and ev.row_count is None and _has_any_success(by_engine[e]))
    has_instability = bool(unstable_engines)

    stable_fps: dict[str, str] = {e: ev.fingerprint for e, ev in evidences.items() if ev.fingerprint is not None}
    stable_rcs: dict[str, str] = {e: ev.row_count for e, ev in evidences.items() if ev.row_count is not None}

    ref_fps = {e: fp for e, fp in stable_fps.items() if e in REFERENCE_ENGINES}
    ref_fp_values = set(ref_fps.values())
    ref_consensus: str | None = next(iter(ref_fp_values)) if len(ref_fp_values) == 1 else None
    native_fp = stable_fps.get(NATIVE_ENGINE)

    all_fps = list(stable_fps.values())
    all_rcs = list(stable_rcs.values())

    if len(all_fps) <= 1:
        fingerprint_agreement = "n/a"
        row_count_agreement = "n/a"
        mismatched_engines: list[str] = []
    else:
        fingerprint_agreement = "true" if len(set(all_fps)) == 1 else "false"
        row_count_agreement = "true" if len(set(all_rcs)) == 1 else "false"
        if fingerprint_agreement == "false":
            consensus = ref_consensus or _majority(all_fps)
            mismatched_engines = sorted(e for e, fp in stable_fps.items() if fp != consensus)
        else:
            mismatched_engines = []

    classification, notes = _classify(
        query_id=query_id,
        has_instability=has_instability,
        fingerprint_agreement=fingerprint_agreement,
        row_count_agreement=row_count_agreement,
        ref_consensus=ref_consensus,
        ref_fps=ref_fps,
        native_fp=native_fp,
    )

    if len(ref_fp_values) > 1:
        ref_fp_field = "disagreement"
    elif ref_consensus is not None:
        ref_fp_field = ref_consensus
    else:
        ref_fp_field = "none"

    return ClassificationRow(
        query_id=query_id,
        classification=classification,
        engine_instability="true" if has_instability else "false",
        row_count_agreement=row_count_agreement,
        fingerprint_agreement=fingerprint_agreement,
        unstable_engines=",".join(unstable_engines) or "none",
        mismatched_engines=",".join(mismatched_engines) or "none",
        reference_fingerprint=ref_fp_field,
        algeon_datafusion_fingerprint=native_fp or "none",
        notes=notes,
    )


def _has_any_success(rows: list[dict]) -> bool:
    return any(r["status"] == "succeeded" for r in rows)


def _engine_evidence(engine: str, rows: list[dict]) -> EngineEvidence:
    succeeded = [r for r in rows if r["status"] == "succeeded"]
    if not succeeded:
        return EngineEvidence(engine=engine, stable=True, row_count=None, fingerprint=None)
    fps = list(dict.fromkeys(r["fingerprint"] for r in succeeded))
    rcs = list(dict.fromkeys(r["row_count"] for r in succeeded))
    stable = len(fps) == 1 and len(rcs) == 1
    return EngineEvidence(
        engine=engine,
        stable=stable,
        row_count=rcs[0] if stable else None,
        fingerprint=fps[0] if stable else None,
    )


def _classify(
    query_id: str,
    has_instability: bool,
    fingerprint_agreement: str,
    row_count_agreement: str,
    ref_consensus: str | None,
    ref_fps: dict[str, str],
    native_fp: str | None,
) -> tuple[str, str]:
    if has_instability:
        return (
            "engine_instability",
            "one or more engines returned different outputs across iterations",
        )

    if fingerprint_agreement == "n/a":
        return "insufficient_data", "fewer than two engines with successful output"

    if fingerprint_agreement == "true":
        return "all_engines_same", "all engines returned identical rows and values"

    # Fingerprints differ: classify by query type.
    if query_id in FLOAT_VARIANCE_QUERY_IDS:
        if row_count_agreement == "true":
            return (
                "float_variance",
                f"{query_id} has Float64 aggregation; row counts agree but values differ "
                "(likely float accumulation order variance across engines)",
            )
        return (
            "unexpected_mismatch",
            f"{query_id} expected float_variance but row counts also differ; inspect raw rows",
        )

    if query_id in TIE_DIVERGENCE_QUERY_IDS:
        if row_count_agreement == "true":
            return (
                "tie_divergence",
                f"{query_id} has a LIMIT boundary with tied rows; row counts agree but engines chose different tied rows",
            )
        return (
            "unexpected_mismatch",
            f"{query_id} expected tie_divergence but row counts differ; inspect raw rows",
        )

    if query_id == "q23":
        if ref_consensus is not None and native_fp is not None and native_fp != ref_consensus:
            return (
                "correctness_bug",
                "reference engines agree; algeon_datafusion returns different COUNT(DISTINCT UserID) values "
                ": over-count bug in native grouped distinct aggregate",
            )
        if len(ref_fps) > 1:
            return (
                "unexpected_mismatch",
                "q23: reference engines disagree among themselves; inspect raw rows before drawing conclusions",
            )
        if native_fp is None:
            return "unexpected_mismatch", "q23: algeon_datafusion produced no successful output"
        # native_fp == ref_consensus: fingerprint_agreement would be "true", caught above.
        return (
            "unexpected_mismatch",
            "q23: unexpected state; inspect raw rows",
        )

    return (
        "unexpected_mismatch",
        f"{query_id}: fingerprints differ with no matching classification rule; inspect raw rows",
    )


def _majority(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def print_summary(classifications: list[ClassificationRow]) -> None:
    by_class: dict[str, list[str]] = {}
    for row in classifications:
        by_class.setdefault(row.classification, []).append(row.query_id)

    print("=== Cross-Engine Classification Summary ===")
    for cls in sorted(by_class):
        queries = by_class[cls]
        print(f"  {cls}: {','.join(queries)} ({len(queries)} queries)")

    unstable = [r.query_id for r in classifications if r.engine_instability == "true"]
    if unstable:
        print(f"  [engine_instability flag also set on]: {','.join(unstable)}")
    print()


def write_tsv(path: Path, rows: list[ClassificationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(path: Path, rows: list[ClassificationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False))
            fh.write("\n")


if __name__ == "__main__":
    sys.exit(main())
