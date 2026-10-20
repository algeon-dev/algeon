"""Flight SQL helper for the demo: timed queries, coverage checks, decision-table registration.

Reuses `ResearchClient` from the Amazon Reviews research script for the Flight
SQL protocol framing. Every executed stage records elapsed time, the query id,
the plan-time coverage summary from `EXPLAIN GPU` and the runtime counters from
the query stats, so a case can prove `host_datafusion_boundary_count = 0`.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import time

import pyarrow as pa

from client.registry import write_parquet_atomic

ROOT = Path(__file__).resolve().parents[5]
DEMO_ROOT = ROOT / "fixture/demo"
RESEARCH_CLIENT = DEMO_ROOT / "amazon_reviews_2023/scripts/amazon_reviews_research.py"
DEFAULT_URI = "grpc://127.0.0.1:50064"
EXECUTION_FIELDS = (
    "native_plan_fragment_count", "host_datafusion_boundary_count", "allocator_high_water_bytes",
    "admission_wait_ms", "query_grant_bytes", "output_rows", "retry_count", "post_native_execution_path",
)

MERREC_TABLES = {
    "merrec_items": DEMO_ROOT / "merrec/parquet/items.parquet",
    "merrec_events": DEMO_ROOT / "merrec/parquet/events.parquet",
}
AMAZON_TABLES = {
    "products": DEMO_ROOT / "amazon_reviews_2023/parquet/products.parquet",
    "reviews": DEMO_ROOT / "amazon_reviews_2023/parquet/reviews.parquet",
    "semantic_reviews_text": ROOT / "target/amazon-reviews-bge-m3/reviews_text.parquet",
    "semantic_reviews_vectors": ROOT / "target/amazon-reviews-bge-m3/reviews_vectors_compact.parquet",
}
EMBER_RESEARCH = ROOT / "target/ember2024-research"
EMBER_TABLES = {
    "ember_metadata_challenge": EMBER_RESEARCH / "input_metadata_challenge.parquet",
    "ember_48_challenge_import_edges": EMBER_RESEARCH / "input_48_challenge_import_edges.parquet",
    "ember_48_challenge_import_symbols": EMBER_RESEARCH / "input_48_challenge_import_symbols.parquet",
    "ember_48_challenge_import_sample_counts": EMBER_RESEARCH / "input_48_challenge_import_sample_counts.parquet",
    "ember_49_challenge_graph_seed": EMBER_RESEARCH / "input_49_challenge_graph_seed.parquet",
}


def _research_module():
    spec = importlib.util.spec_from_file_location("amazon_reviews_research", RESEARCH_CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Stage:
    name: str
    table: pa.Table
    elapsed_s: float
    query_id: str | None
    explain: dict | None
    execution: dict | None

    @property
    def rows(self) -> int:
        return self.table.num_rows

    def record(self) -> dict:
        return {
            "name": self.name,
            "rows": self.rows,
            "elapsed_s": self.elapsed_s,
            "query_id": self.query_id,
            "explain": self.explain,
            "execution": self.execution,
        }


def _parse_number(text: str):
    if text in ("", "none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def statement_body(sql: str) -> str:
    """The statement without leading `--` comment lines and blank lines."""
    lines = sql.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("--")):
        lines.pop(0)
    return "\n".join(lines).lstrip()


def parse_execution_stats(stats: str) -> dict | None:
    """Selected counters from the `[execution]` TSV section of GetQueryStats."""
    if "[execution]" not in stats:
        return None
    section = stats.split("[execution]", 1)[1].lstrip("\n")
    lines = [line for line in section.splitlines() if line and not line.startswith("[")]
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    values = lines[1].split("\t")
    record = dict(zip(header, values))
    return {name: _parse_number(record[name]) for name in EXECUTION_FIELDS if name in record}


class Algeon:
    def __init__(self, uri: str = DEFAULT_URI, timeout: float = 1800.0):
        module = _research_module()
        self._client = module.ResearchClient(uri)
        self._client.options = module.flight.FlightCallOptions(timeout=timeout)
        self.uri = uri

    def execute(self, sql: str) -> tuple[pa.Table, str | None]:
        return self._client.execute(sql)

    def stats(self, query_id: str) -> str:
        return self._client.stats(query_id)

    def explain_summary(self, sql: str) -> dict:
        plan, _ = self.execute("EXPLAIN GPU FORMAT JSON " + sql)
        for row in plan.to_pylist():
            if row["row_kind"] == "summary":
                return {key: row[key] for key in ("gpu_path", "gpu_fragments", "cpu_boundaries", "outcome")}
        raise RuntimeError("EXPLAIN GPU returned no summary row")

    def run(self, name: str, sql: str, *, explain: bool = True) -> Stage:
        """Execute one query statement and collect its timing and coverage evidence."""
        is_query = statement_body(sql).upper().startswith(("SELECT", "WITH"))
        summary = self.explain_summary(sql) if explain and is_query else None
        started = time.perf_counter()
        table, query_id = self.execute(sql)
        elapsed = time.perf_counter() - started
        execution = parse_execution_stats(self.stats(query_id)) if query_id else None
        return Stage(name, table, elapsed, query_id, summary, execution)

    def register_parquet(self, table_name: str, path: Path, arrow_table: pa.Table | None = None) -> None:
        """Write `arrow_table` (when given) and register `path` as an external Parquet table."""
        path = Path(path).resolve()
        if arrow_table is not None:
            write_parquet_atomic(arrow_table, path)
        self.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.execute(f"CREATE EXTERNAL TABLE {table_name} STORED AS PARQUET LOCATION '{path}'")

    def register_sources(self, tables: dict[str, Path]) -> None:
        for name, path in tables.items():
            resolved = Path(path).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"source table {name} is missing at {resolved}")
            self.execute(f"CREATE EXTERNAL TABLE IF NOT EXISTS {name} STORED AS PARQUET LOCATION '{resolved}'")

    def register_merrec(self) -> None:
        self.register_sources(MERREC_TABLES)

    def register_amazon(self) -> None:
        self.register_sources(AMAZON_TABLES)

    def register_ember(self) -> None:
        self.register_sources(EMBER_TABLES)
