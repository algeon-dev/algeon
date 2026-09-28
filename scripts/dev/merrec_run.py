#!/usr/bin/env python3
"""Run MerRec research SQL through the Flight SQL server."""

import argparse
import contextlib
import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CLIENT_PATH = ROOT / "fixture/demo/amazon_reviews_2023/scripts/amazon_reviews_research.py"
REPORT = ROOT / "docs/research/merrec"
DATA = ROOT / "target/merrec-exploration/results"
MAX_CSV_ROWS = 100_000


def load_research_client():
    spec = importlib.util.spec_from_file_location("amazon_reviews_research", RESEARCH_CLIENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load research client: {RESEARCH_CLIENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORT = REPORT
    module.DATA = DATA
    return module


def write_csv_export(table, path):
    def csv_value(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, default=str, separators=(",", ":"))
        return value

    with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(table.column_names)
        for batch in table.to_batches(max_chunksize=8192):
            for row in batch.to_pylist():
                writer.writerow(csv_value(row[column]) for column in table.column_names)


def save_record(record):
    output = REPORT / "results"
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / f"{record['name']}.json"
    record_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("sql", "schema", "sample")}), flush=True)


def execute_sql_file(client, path):
    for index, statement in enumerate(path.read_text().split(";"), start=1):
        statement = statement.strip()
        if not statement:
            continue
        try:
            client.execute(statement)
        except Exception as error:
            record = {
                "name": f"prepare_{path.stem}_statement_{index}",
                "sql": statement,
                "statement": index,
                "status": "error",
                "error": str(error),
            }
            if client.last_query_id:
                record["query_id"] = client.last_query_id
            save_record(record)
            return False
    return True


def run_and_export(client, name, sql):
    with contextlib.redirect_stdout(io.StringIO()):
        record = client.run(name, sql)

    if record["status"] == "ok":
        if record["rows"] <= MAX_CSV_ROWS:
            export_path = REPORT / "exports" / f"{name}.csv.gz"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import pyarrow.parquet as pq

                write_csv_export(pq.read_table(DATA / f"{name}.parquet"), export_path)
                record["csv_export"] = {"status": "written",
                                        "path": str(export_path.relative_to(ROOT))}
            except Exception as error:
                record["csv_export"] = {"status": "error", "error": str(error)}
        else:
            record["csv_export"] = {"status": "skipped", "reason": "row_limit",
                                    "rows": record["rows"], "max_rows": MAX_CSV_ROWS}
    else:
        record["csv_export"] = {"status": "skipped", "reason": "query_failed"}
    save_record(record)
    return record


def prepare_graph_and_tastes(client):
    sql_dir = REPORT / "sql"
    for filename in ("00_register.sql", "01_graph_views.sql"):
        if not execute_sql_file(client, sql_dir / filename):
            return False

    for filename in ("prepare_transitions.sql", "prepare_markets.sql"):
        path = sql_dir / filename
        if run_and_export(client, path.stem, path.read_text())["status"] != "ok":
            return False

    if not execute_sql_file(client, sql_dir / "02_graph_snapshots.sql"):
        return False

    for filename in (
        "prepare_nodes.sql",
        "prepare_edges.sql",
        "prepare_undirected.sql",
        "prepare_audience.sql",
    ):
        path = sql_dir / filename
        if run_and_export(client, path.stem, path.read_text())["status"] != "ok":
            return False

    if not execute_sql_file(client, sql_dir / "03_register_graph.sql"):
        return False

    path = sql_dir / "prepare_tastes.sql"
    if run_and_export(client, path.stem, path.read_text())["status"] != "ok":
        return False

    return execute_sql_file(client, sql_dir / "04_register_tastes.sql")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="SQL files to execute")
    parser.add_argument("--uri", default="grpc://127.0.0.1:50063")
    setup = parser.add_mutually_exclusive_group()
    setup.add_argument("--register", action="store_true",
                       help="Execute docs/research/merrec/sql/00_register.sql first")
    setup.add_argument("--prepare", action="store_true",
                       help="Build and register MerRec graph and taste snapshots before query files")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180, help="Flight call timeout in seconds")
    parser.add_argument("--suffix", default="", help="Append this to each result name")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")
    if not re.fullmatch(r"[A-Za-z0-9_.-]*", args.suffix):
        parser.error("--suffix may contain only letters, digits, periods, underscores, and hyphens")
    for path in args.files:
        if not path.is_file():
            parser.error(f"SQL file does not exist: {path}")

    register_path = REPORT / "sql/00_register.sql"
    if args.register and not register_path.is_file():
        parser.error(f"registration SQL file does not exist: {register_path}")

    research = load_research_client()
    client = research.ResearchClient(args.uri)
    client.options = research.flight.FlightCallOptions(timeout=args.timeout)

    if args.register:
        register_sql = register_path.read_text()
        for index, statement in enumerate(register_sql.split(";"), start=1):
            statement = statement.strip()
            if not statement:
                continue
            try:
                client.execute(statement)
            except Exception as error:
                print(json.dumps({"name": "00_register", "statement": index,
                                  "status": "error", "error": str(error)}), file=sys.stderr)
                return 1

    if args.prepare and not prepare_graph_and_tastes(client):
        return 1

    had_query_error = False
    DATA.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        sql = path.read_text()
        for repeat in range(1, args.repeats + 1):
            repeat_suffix = f"_r{repeat}" if args.repeats > 1 else ""
            name = f"{path.stem}{args.suffix}{repeat_suffix}"
            record = run_and_export(client, name, sql)
            if record["status"] != "ok":
                had_query_error = True

    return 1 if had_query_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
