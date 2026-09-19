#!/usr/bin/env python3
"""Run the Amazon research SQL over Flight SQL and retain results and observations."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import time

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
REPORT = ROOT / "docs/research/amazon_reviews_2023"
DATA = ROOT / "target/amazon-reviews-research"


def write_compressed(path, value):
    with gzip.open(str(path) + ".gz", "wt") as output:
        output.write(value)


def varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def fields(message):
    # Flight SQL Any and statement tickets contain length-delimited fields.
    offset = 0
    result = {}

    def read_varint():
        nonlocal offset
        value = shift = 0
        while True:
            byte = message[offset]
            offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    while offset < len(message):
        tag = read_varint()
        assert tag & 7 == 2, tag
        size = read_varint()
        result[tag >> 3] = message[offset:offset + size]
        offset += size
    return result


class ResearchClient:
    def __init__(self, uri="grpc://127.0.0.1:50061"):
        self.client = flight.FlightClient(uri)
        self.options = flight.FlightCallOptions(timeout=180)
        self.last_query_id = None

    def execute(self, sql):
        self.last_query_id = None
        message = field(1, b"type.googleapis.com/arrow.flight.protocol.sql.CommandStatementQuery")
        message += field(2, field(1, sql.encode()))
        info = self.client.get_flight_info(flight.FlightDescriptor.for_command(message), self.options)
        query_id = None
        tables = []
        for endpoint in info.endpoints:
            query_id = fields(fields(endpoint.ticket.ticket)[2])[1].decode()
            self.last_query_id = query_id
            tables.append(self.client.do_get(endpoint.ticket, self.options).read_all())
        table = pa.concat_tables(tables) if tables else pa.table({})
        return table, query_id

    def stats(self, query_id):
        return b"".join(bytes(result.body) for result in self.client.do_action(
            flight.Action("GetQueryStats", query_id.encode()), self.options
        )).decode()

    def run(self, name, sql, explain=True):
        output = REPORT / "results"
        output.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        record = {"name": name, "sql": sql}
        try:
            table, query_id = self.execute(sql)
            record.update(status="ok", elapsed_s=time.perf_counter() - started,
                          rows=table.num_rows, schema=str(table.schema), query_id=query_id,
                          sample=table.slice(0, 20).to_pylist())
            pq.write_table(table, DATA / f"{name}.parquet")
            if query_id:
                write_compressed(output / f"{name}.stats.txt", self.stats(query_id))
            if explain and sql.lstrip().upper().startswith(("SELECT", "WITH")):
                plan, _ = self.execute("EXPLAIN GPU FORMAT JSON " + sql)
                write_compressed(output / f"{name}.coverage.json", json.dumps(plan.to_pylist(), indent=2, default=str))
        except Exception as error:
            record.update(status="error", elapsed_s=time.perf_counter() - started, error=str(error))
            if self.last_query_id:
                record["query_id"] = self.last_query_id
                write_compressed(output / f"{name}.stats.txt", self.stats(self.last_query_id))
        (output / f"{name}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ("sql", "schema", "sample")}), flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--uri", default="grpc://127.0.0.1:50061")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Execute and snapshot the three feature queries")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    client = ResearchClient(args.uri)
    if args.register or args.prepare:
        for name in ("products", "reviews"):
            path = ROOT / f"fixture/demo/amazon_reviews_2023/parquet/{name}.parquet"
            client.execute(f"CREATE EXTERNAL TABLE IF NOT EXISTS {name} STORED AS PARQUET LOCATION '{path}'")
        for path in sorted(DATA.glob("input_*.parquet")):
            client.execute(f"CREATE EXTERNAL TABLE IF NOT EXISTS {path.stem[6:]} STORED AS PARQUET LOCATION '{path}'")
    if args.prepare:
        for name, table in (("00_review_rollup", "rollup"), ("01_headphone_complaints", "complaints"),
                            ("03_temporal_cohort", "cohort")):
            record = client.run(name, (REPORT / "sql" / f"{name}.sql").read_text())
            if record["status"] != "ok":
                raise RuntimeError(f"Feature preparation failed: {name}")
            path = DATA / f"input_{table}.parquet"
            shutil.copyfile(DATA / f"{name}.parquet", path)
            client.execute(f"CREATE EXTERNAL TABLE IF NOT EXISTS {table} STORED AS PARQUET LOCATION '{path}'")
    for path in args.files:
        client.run(path.stem + args.suffix, path.read_text())


if __name__ == "__main__":
    main()
