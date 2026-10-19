#!/usr/bin/env python3
"""Materialize the EMBER2024 challenge node tables that case F reads.

Writes three Parquet tables under `fixture/demo/decision_models/data/ember/`:

- `ember_f_samples.parquet`: every challenge sample with its metadata id
  (`id`, `sha256`, `file_type`, `family`, `labeled`). Ids come from
  `input_metadata_challenge.parquet`; this script assigns no sample ids.
- `ember_f_nodes.parquet`: one row per graph node seen in a Win32 sample.
  Import symbols get `node_id` from 100000000 and behaviors from 200000000,
  each in (kind, name) order. `doc_freq` counts distinct Win32 samples.
- `ember_f_sample_nodes.parquet`: one row per Win32 sample and node it has.

Behaviors come from the raw challenge JSONL: CAPA `caps[].Capability`, MITRE
`ttps[].Technique` and MBC `mbc[].Behavior`, deduplicated by sha256. The script
checks that imports with doc frequency 3..100 reproduce the published V0 graph
(4,988 symbols, 80,673 edges) and prints the counts either way. Rerunning it
rewrites the same files.

Usage, from fixture/demo/decision_models/scripts:

    python3 cases/f_import_gate/prepare_tables.py
"""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.algeon import DEMO_ROOT, ROOT  # noqa: E402
from client.registry import write_parquet_atomic  # noqa: E402

DATA_DIR = DEMO_ROOT / "decision_models/data/ember"
RESEARCH_DIR = ROOT / "target/ember2024-research"
RAW_GLOB = str(DEMO_ROOT / "ember2024/raw/*_challenge_malicious.jsonl")
IMPORT_BASE = 100_000_000
BEHAVIOR_BASE = 200_000_000
# JSONL list field, item key and node kind for each behavior source.
BEHAVIOR_SOURCES = (("caps", "Capability", "capa"), ("ttps", "Technique", "mitre"), ("mbc", "Behavior", "mbc"))
PUBLISHED_V0 = {"symbols": 4988, "edges": 80673}

SAMPLES_SCHEMA = pa.schema([
    ("id", pa.int64()), ("sha256", pa.string()), ("file_type", pa.string()), ("family", pa.string()),
    ("labeled", pa.bool_()),
])
NODES_SCHEMA = pa.schema([
    ("node_id", pa.int64()), ("kind", pa.string()), ("name", pa.string()), ("dll_lower", pa.string()),
    ("doc_freq", pa.int64()),
])
SAMPLE_NODES_SCHEMA = pa.schema([("id", pa.int64()), ("node_id", pa.int64())])


def load_samples() -> list[dict]:
    rows = pq.read_table(RESEARCH_DIR / "input_metadata_challenge.parquet",
                         columns=["id", "sha256", "file_type", "family"]).to_pylist()
    for row in rows:
        row["labeled"] = bool(row["family"])
    return sorted(rows, key=lambda row: row["id"])


def load_imports(win32: set[int]) -> dict[tuple[str, str], set[int]]:
    table = pq.read_table(RESEARCH_DIR / "input_imports_challenge.parquet", columns=["id", "symbol"])
    members: dict[tuple[str, str], set[int]] = defaultdict(set)
    for sample_id, symbol in zip(table.column("id").to_pylist(), table.column("symbol").to_pylist()):
        if sample_id in win32:
            members[("import", symbol)].add(sample_id)
    return members


def load_behaviors(id_of_sha: dict[str, int], win32: set[int]) -> tuple[dict[tuple[str, str], set[int]], dict]:
    members: dict[tuple[str, str], set[int]] = defaultdict(set)
    seen: set[str] = set()
    stats = {"files": 0, "lines": 0, "duplicate_sha256": 0, "unknown_sha256": 0, "win32_with_behaviors": 0}
    for path in sorted(glob.glob(RAW_GLOB)):
        stats["files"] += 1
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["lines"] += 1
                record = json.loads(line)
                sha = record["sha256"]
                if sha in seen:
                    stats["duplicate_sha256"] += 1
                    continue
                seen.add(sha)
                sample_id = id_of_sha.get(sha)
                if sample_id is None:
                    stats["unknown_sha256"] += 1
                    continue
                if sample_id not in win32:
                    continue
                found = False
                for field, key, kind in BEHAVIOR_SOURCES:
                    for item in record.get(field) or []:
                        name = item.get(key)
                        if name:
                            members[(kind, name)].add(sample_id)
                            found = True
                stats["win32_with_behaviors"] += found
    return members, stats


def main() -> None:
    samples = load_samples()
    id_of_sha = {row["sha256"]: row["id"] for row in samples}
    win32 = {row["id"] for row in samples if row["file_type"] == "Win32"}
    imports = load_imports(win32)
    behaviors, behavior_stats = load_behaviors(id_of_sha, win32)

    nodes, sample_nodes = [], []
    for base, members in ((IMPORT_BASE, imports), (BEHAVIOR_BASE, behaviors)):
        for offset, (kind, name) in enumerate(sorted(members)):
            node_id = base + offset
            nodes.append({"node_id": node_id, "kind": kind, "name": name,
                          "dll_lower": name.split("!", 1)[0].lower() if kind == "import" else None,
                          "doc_freq": len(members[(kind, name)])})
            sample_nodes.extend({"id": sample_id, "node_id": node_id} for sample_id in sorted(members[(kind, name)]))
    sample_nodes.sort(key=lambda row: (row["id"], row["node_id"]))

    write_parquet_atomic(pa.Table.from_pylist(samples, schema=SAMPLES_SCHEMA), DATA_DIR / "ember_f_samples.parquet")
    write_parquet_atomic(pa.Table.from_pylist(nodes, schema=NODES_SCHEMA), DATA_DIR / "ember_f_nodes.parquet")
    write_parquet_atomic(pa.Table.from_pylist(sample_nodes, schema=SAMPLE_NODES_SCHEMA),
                         DATA_DIR / "ember_f_sample_nodes.parquet")

    v0 = [node for node in nodes if node["kind"] == "import" and 3 <= node["doc_freq"] <= 100]
    v0_counts = {"symbols": len(v0), "edges": sum(node["doc_freq"] for node in v0)}
    kinds = defaultdict(int)
    for node in nodes:
        kinds[node["kind"]] += 1
    labeled = [row for row in samples if row["id"] in win32 and row["labeled"]]
    report = {
        "samples": len(samples), "win32": len(win32), "win32_labeled": len(labeled),
        "nodes": len(nodes), "nodes_by_kind": dict(sorted(kinds.items())), "sample_nodes": len(sample_nodes),
        "behaviors": behavior_stats, "v0": v0_counts, "published_v0": PUBLISHED_V0,
        "v0_matches_published": v0_counts == PUBLISHED_V0,
    }
    print(json.dumps(report, indent=2))
    if v0_counts != PUBLISHED_V0:
        raise SystemExit(f"V0 construction gives {v0_counts}; the published graph has {PUBLISHED_V0}")


if __name__ == "__main__":
    main()
