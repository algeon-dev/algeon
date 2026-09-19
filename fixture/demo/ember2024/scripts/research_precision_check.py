#!/usr/bin/env python3
"""Prepare distance probes and compare saved SQL output with direct differences."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORT = ROOT / "docs/research/ember2024/results"
IDS = (20000424, 20000494, 20003070)


def main():
    source = pq.read_table(DATA / "input_research_vectors_challenge.parquet")
    positions = {row_id: index for index, row_id in enumerate(source["id"].to_pylist())}
    vectors = np.asarray([source["histogram"][positions[row_id]].as_py() for row_id in IDS], dtype=np.float32)
    for bits, dtype in ((32, pa.float32()), (64, pa.float64())):
        vector_type = pa.list_(pa.field("item", dtype, nullable=False), 512)
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("vector", vector_type, nullable=False)])
        table = pa.Table.from_arrays([pa.array(IDS), pa.array(vectors.tolist(), type=vector_type)], schema=schema)
        pq.write_table(table, DATA / f"input_distance_probe{bits}.parquet")

    oracle = {
        (first, second): float(np.square(vectors[i].astype(np.float64) - vectors[j].astype(np.float64)).sum())
        for i, first in enumerate(IDS) for j, second in enumerate(IDS)
    }
    checks = {}
    for query in ("48_distance_precision", "49_distance_float64", "51_distance_precision_reversed"):
        path = DATA / f"{query}.parquet"
        if not path.exists():
            continue
        table = pq.read_table(path)
        rows = []
        for row in table.to_pylist():
            expected = oracle[row["x_id"], row["y_id"]]
            rows.append({**row, "cpu_float64_direct_squared_l2": expected, "absolute_error": abs(row["distance"] - expected)})
        checks[query] = {"distance_dtype": str(table.schema.field("distance").type), "rows": rows}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "distance_precision_check.json").write_text(json.dumps({
        "source": "Identical stored Float32 histogram values; the Float64 probe widens those values without recomputation.",
        "selection": "IDs selected from the largest errors in query47 for diagnosis; this selection overrepresents large errors.",
        "checks": checks,
    }, indent=2) + "\n")
    print(f"Prepared {len(IDS)} vectors at both widths; compared {len(checks)} saved queries.")


if __name__ == "__main__":
    main()
