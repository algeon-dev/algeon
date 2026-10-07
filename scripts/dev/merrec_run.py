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
import shutil
import sys
import time

from pyarrow import flight
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CLIENT_PATH = ROOT / "fixture/demo/amazon_reviews_2023/scripts/amazon_reviews_research.py"
REPORT = ROOT / "docs/research/merrec"
DATA = ROOT / "target/merrec-exploration/results"
SOURCE_DATA = ROOT / "fixture/demo/merrec/parquet"
SOURCE_REGISTRATION = ROOT / "fixture/demo/merrec/scripts/scale/00_register_sources.sql"
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
    sql = path.read_text()
    for original, replacement in (
        ("fixture/demo/merrec/parquet/", str(SOURCE_DATA) + "/"),
        ("target/merrec-exploration/results/", str(DATA) + "/"),
    ):
        sql = sql.replace(original, replacement.replace("'", "''"))
    for index, statement in enumerate(sql.split(";"), start=1):
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


def capture_runtime_artifact(client, name, query_id):
    path = REPORT / "results" / f"{name}.runtime.jsonl.gz"
    pending_control = None
    try:
        path.unlink(missing_ok=True)
        if not query_id:
            return {"status": "skipped", "reason": "query_id_missing"}
        timeout = client.options.timeout if client.options is not None else -1
        deadline = time.monotonic() + (min(timeout, 30) if timeout > 0 else 30)
        remaining = deadline - time.monotonic()
        while True:
            options = flight.FlightCallOptions(
                timeout=remaining,
                headers=client.options.headers if client.options is not None else None,
                read_options=client.options.read_options if client.options is not None else None,
                write_options=client.options.write_options if client.options is not None else None,
            )
            frames = iter(client.client.do_action(
                flight.Action("GetQueryRuntimeArtifact", query_id.encode()), options))
            control = json.loads(bytes(next(frames).body))
            if control["runtime_jsonl_follows"]:
                break
            if control.get("retryable") is not True:
                return {"status": "unavailable", "control": control}
            # Query failure can precede coordinator sealing; only delivery is retried.
            if pending_control is None:
                pending_control = control
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "unavailable", "control": pending_control}
            time.sleep(min(0.1, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "unavailable", "control": pending_control}
        runtime_jsonl = bytes(next(frames).body)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as output:
            output.write(runtime_jsonl)
        return {"status": "written", "control": control,
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
    except Exception as error:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        if pending_control is not None and time.monotonic() >= deadline:
            return {"status": "unavailable", "control": pending_control}
        return {"status": "error", "error": str(error)}


def run_and_export(client, name, sql):
    with contextlib.redirect_stdout(io.StringIO()):
        record = client.run(name, sql)
    record["runtime_artifact"] = capture_runtime_artifact(client, name, record.get("query_id"))

    if record["status"] == "ok":
        if record["rows"] <= MAX_CSV_ROWS:
            export_path = REPORT / "exports" / f"{name}.csv.gz"
            export_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                write_csv_export(pq.read_table(DATA / f"{name}.parquet"), export_path)
                record["csv_export"] = {"status": "written",
                                        "path": str(export_path.relative_to(ROOT))
                                        if export_path.is_relative_to(ROOT) else str(export_path)}
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
    sql_dir = ROOT / "docs/research/merrec/sql"
    if not register_sources(client):
        return False
    if not execute_sql_file(client, sql_dir / "01_graph_views.sql"):
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


def prepare_scale_features(client):
    sql_dir = ROOT / "fixture/demo/merrec/scripts/scale"
    if not register_sources(client):
        return False
    for filename in ("03_user_features.sql", "14_taste64.sql"):
        path = sql_dir / filename
        if run_and_export(client, path.stem, path.read_text())["status"] != "ok":
            return False
    return execute_sql_file(client, sql_dir / "00_register_features.sql")


def capture_input_manifest(client, registered_data_dir=None):
    source_paths = {}
    for name in ("items", "events"):
        table, _ = client.execute(f"SHOW CREATE TABLE merrec_{name}")
        definition = table.to_pylist()[0]["definition"]
        location = re.search(r"\bLOCATION\s+'((?:[^']|'')*)'", definition, re.IGNORECASE)
        if location is None:
            raise ValueError(f"merrec_{name} has no external source location")
        source = Path(location.group(1).replace("''", "'")).resolve()
        if registered_data_dir is not None and source != (registered_data_dir / f"{name}.parquet").resolve():
            raise ValueError(f"merrec_{name} already uses {source}; start a fresh server for the requested dataset")
        source_paths[name] = source

    directories = {path.parent for path in source_paths.values()}
    if len(directories) != 1:
        raise ValueError("MerRec source tables belong to different dataset directories")
    manifest_path = directories.pop().parent / "manifest.json"
    report_manifest = REPORT / "input_manifest.json"
    if not manifest_path.is_file():
        # A reused report directory must not attribute unmanifested inputs to an older dataset.
        report_manifest.unlink(missing_ok=True)
        return
    manifest = json.loads(manifest_path.read_text())
    for name, source in source_paths.items():
        if (manifest_path.parent / manifest["tables"][name]["path"]).resolve() != source:
            raise ValueError(f"manifest does not describe the registered merrec_{name} source")
    REPORT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, report_manifest)


def register_sources(client):
    if not execute_sql_file(client, SOURCE_REGISTRATION):
        return False
    capture_input_manifest(client, SOURCE_DATA)
    return True


def main(argv=None):
    global REPORT, DATA, SOURCE_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="SQL files to execute")
    parser.add_argument("--uri", default="grpc://127.0.0.1:50063")
    setup = parser.add_mutually_exclusive_group()
    setup.add_argument("--register", action="store_true",
                       help="Register the source events and items tables first")
    setup.add_argument("--prepare", action="store_true",
                       help="Build and register MerRec graph and taste snapshots before query files")
    setup.add_argument("--prepare-scale", action="store_true",
                       help="Build and register full-scope user features before scale queries")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180, help="Flight call timeout in seconds")
    parser.add_argument("--suffix", default="", help="Append this to each result name")
    parser.add_argument("--report-dir", type=Path, default=REPORT,
                       help="Directory receiving results and CSV exports")
    parser.add_argument("--results-dir", type=Path, default=DATA,
                       help="Directory receiving result Parquet")
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA,
                       help="Directory containing events.parquet and items.parquet")
    args = parser.parse_args(argv)

    REPORT = args.report_dir.resolve()
    DATA = args.results_dir.resolve()
    SOURCE_DATA = args.data_dir.resolve()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")
    if not re.fullmatch(r"[A-Za-z0-9_.-]*", args.suffix):
        parser.error("--suffix may contain only letters, digits, periods, underscores, and hyphens")
    for path in args.files:
        if not path.is_file():
            parser.error(f"SQL file does not exist: {path}")

    register_path = SOURCE_REGISTRATION
    if args.register and not register_path.is_file():
        parser.error(f"registration SQL file does not exist: {register_path}")

    research = load_research_client()
    client = research.ResearchClient(args.uri)
    client.options = research.flight.FlightCallOptions(timeout=args.timeout)

    try:
        if args.register and not register_sources(client):
            return 1
        if args.prepare and not prepare_graph_and_tastes(client):
            return 1
        if args.prepare_scale and not prepare_scale_features(client):
            return 1
        if not (args.register or args.prepare or args.prepare_scale):
            capture_input_manifest(client)
    except Exception as error:
        save_record({"name": "prepare", "status": "error", "error": str(error)})
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
