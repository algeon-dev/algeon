#!/usr/bin/env python3
"""Run EMBER2024 research SQL over Flight SQL and retain results and observations."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
AMAZON_RESEARCH = ROOT / "fixture/demo/amazon_reviews_2023/scripts/amazon_reviews_research.py"


def load_research_client():
    spec = importlib.util.spec_from_file_location("amazon_reviews_research", AMAZON_RESEARCH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared research client from {AMAZON_RESEARCH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORT = ROOT / "docs/research/ember2024"
    module.DATA = ROOT / "target/ember2024-research"

    return module


def register_tables(client, data_dir):
    parquet_dir = ROOT / "fixture/demo/ember2024/parquet"
    for split in ("train", "test", "challenge"):
        path = parquet_dir / f"{split}.parquet"
        client.execute(
            f"CREATE EXTERNAL TABLE IF NOT EXISTS ember_{split} "
            f"STORED AS PARQUET LOCATION '{path}'"
        )

    for path in sorted(data_dir.glob("input_*.parquet")):
        table = "ember_" + path.stem.removeprefix("input_")
        client.execute(f"CREATE EXTERNAL TABLE IF NOT EXISTS {table} STORED AS PARQUET LOCATION '{path}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--uri", default="grpc://127.0.0.1:50062")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    module = load_research_client()
    client = module.ResearchClient(args.uri)
    if args.register:
        register_tables(client, module.DATA)

    failed = False
    for path in args.files:
        name = path.stem + args.suffix
        record = client.run(name, path.read_text())
        if record.get("status") == "ok":
            result_path = module.REPORT / "results" / f"{name}.csv.gz"
            try:
                table = pq.read_table(module.DATA / f"{name}.parquet")
                for index, field in enumerate(table.schema):
                    if pa.types.is_string_view(field.type):
                        string_field = pa.field(
                            field.name,
                            pa.string(),
                            nullable=field.nullable,
                            metadata=field.metadata,
                        )
                        table = table.set_column(index, string_field, table.column(index).cast(pa.string()))
                with gzip.open(result_path, "wb") as output:
                    if table.num_columns:
                        csv.write_csv(table, output)
                record["csv_export_status"] = "ok"
            except Exception as error:
                record.update(csv_export_status="error", csv_error=str(error))
        else:
            record["csv_export_status"] = "not_attempted"
        (module.REPORT / "results" / f"{name}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {
                    "name": name,
                    "status": record.get("status"),
                    "csv_export_status": record.get("csv_export_status"),
                    "error": record.get("error"),
                    "csv_error": record.get("csv_error"),
                }
            ),
            flush=True,
        )
        failed |= record.get("status") == "error" or record.get("csv_export_status") == "error"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
