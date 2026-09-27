#!/usr/bin/env python3
"""Compatibility tests for the public fixture entrypoints.

The tests use /bin/true and /bin/false as generator backends so they validate
argument routing, defaults, and verification behavior without generating large
benchmark datasets.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SH = REPO_ROOT / "fixture" / "fixture.sh"
GEN_TPCH = REPO_ROOT / "fixture" / "tpc-h" / "generate.py"
GEN_TPCDS = REPO_ROOT / "fixture" / "tpc-ds" / "generate.py"
SPARK_WRITE_ICEBERG = REPO_ROOT / "fixture" / "iceberg-local" / "spark_write_iceberg.py"
TPC_PARQUET_LAYOUT = REPO_ROOT / "fixture" / "tpc" / "parquet_layout.py"
CITATION_NETWORK_PREPARE = REPO_ROOT / "fixture" / "demo" / "citation_network" / "prepare.py"
AMAZON_REVIEWS_PREPARE = REPO_ROOT / "fixture" / "demo" / "amazon_reviews_2023" / "prepare.py"


def run_cmd(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fake_tpcgen_root(root: Path, backend: str) -> Path:
    tpcgen_root = root / f"fake_tpcgen_{Path(backend).name}"
    bin_dir = tpcgen_root / "target" / "release"
    bin_dir.mkdir(parents=True)
    (bin_dir / "tpcgen-cli").symlink_to(backend)
    return tpcgen_root


def amazon_review(parent_asin: str, user_id: str, timestamp: int, rating: float, images: int = 0) -> dict:
    return {
        "rating": rating,
        "title": "t",
        "text": "body",
        "images": [{"small_image_url": "u", "attachment_type": "IMAGE"}] * images,
        "asin": parent_asin,
        "parent_asin": parent_asin,
        "user_id": user_id,
        "timestamp": timestamp,
        "helpful_vote": 0,
        "verified_purchase": True,
    }


def write_amazon_reviews_source(root: Path) -> tuple[Path, Path]:
    meta = root / "meta.jsonl"
    products = [
        {"parent_asin": "P2", "title": "b", "price": "from 1.00", "categories": [], "details": {}},
        {"parent_asin": "P1", "title": "a", "price": 9.5, "categories": ["Electronics", "Audio"],
         "details": {"Brand": "X", "Date First Available": "August 2, 2014"}, "author": {"name": "n"}},
        {"parent_asin": "P3", "title": "c", "price": None, "categories": [], "details": {}},
    ]
    meta.write_text("".join(json.dumps(product) + "\n" for product in products), encoding="utf-8")
    reviews = root / "reviews.jsonl"
    rows = [
        amazon_review("P2", "U2", 1_600_000_000_000, 5.0, images=2),
        amazon_review("P2", "U1", 1_700_000_000_000, 1.0),
        amazon_review("P3", "U1", 1_500_000_000_000, 3.0),
    ]
    reviews.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return meta, reviews


def amazon_reviews_prepare_command(root: Path, meta: Path, reviews: Path) -> list[str]:
    return [
        sys.executable,
        str(AMAZON_REVIEWS_PREPARE),
        "--meta-jsonl",
        str(meta),
        "--reviews-jsonl",
        str(reviews),
        "--out-dir",
        str(root / "out"),
        "--review-row-group-rows",
        "2",
        "--product-row-group-rows",
        "2",
        "--skip-sha256",
    ]


class FakeRow:
    def __init__(self, **values: object) -> None:
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def __getitem__(self, index: int) -> object:
        return list(self._values.values())[index]

    def __repr__(self) -> str:
        return f"FakeRow({self._values!r})"


class FakeResult:
    def __init__(self, rows: list[FakeRow] | None = None) -> None:
        self._rows = rows or []

    def collect(self) -> list[FakeRow]:
        return self._rows


class FakeWriteTo:
    def __init__(self, spark: "FakeSpark", table: str) -> None:
        self.spark = spark
        self.table = table
        self.properties: dict[str, str] = {}

    def using(self, fmt: str) -> "FakeWriteTo":
        assert fmt == "iceberg"
        return self

    def tableProperty(self, key: str, value: str) -> "FakeWriteTo":
        self.properties[key] = value
        return self

    def create(self) -> None:
        assert self.properties.get("format-version") == "2", "fixture tables must be Iceberg v2 minimum"
        self.spark.created_tables.append(self.table)


class FakeDataFrame:
    def __init__(self, spark: "FakeSpark", source: str) -> None:
        self.spark = spark
        self.source = source

    def limit(self, n: int) -> "FakeDataFrame":
        assert n == 0
        return self

    def writeTo(self, table: str) -> FakeWriteTo:
        return FakeWriteTo(self.spark, table)


class FakeSparkReader:
    def __init__(self, spark: "FakeSpark") -> None:
        self.spark = spark

    def parquet(self, source: str) -> FakeDataFrame:
        self.spark.parquet_reads.append(source)
        return FakeDataFrame(self.spark, source)


class FakeSpark:
    def __init__(self, *, existing: bool = False, file_count: int = 0, fail_on_add_files: bool = False) -> None:
        self.existing = existing
        self.file_count = file_count
        self.fail_on_add_files = fail_on_add_files
        self.sql_calls: list[str] = []
        self.parquet_reads: list[str] = []
        self.created_tables: list[str] = []
        self.read = FakeSparkReader(self)

    def sql(self, query: str) -> FakeResult:
        self.sql_calls.append(query)
        if query.startswith("SHOW TABLES"):
            rows = [FakeRow(tableName="papers")] if self.existing else []
            return FakeResult(rows)
        if ".files" in query and "COUNT(*)" in query:
            return FakeResult([FakeRow(file_count=self.file_count)])
        if "CALL lake.system.add_files" in query:
            if self.fail_on_add_files:
                raise RuntimeError("add_files failed")
            return FakeResult([FakeRow(added_files_count=1)])
        if "COUNT(*) AS row_count" in query:
            return FakeResult([FakeRow(row_count=4894081)])
        return FakeResult()


class FixtureGeneratorCompatibilityTests(unittest.TestCase):
    def test_dblp_edge_footer_carries_simple_topology_and_only_full_fixture_carries_vertex_bound(self) -> None:
        dblp = load_script_module("citation_network_footer_contract_under_test", CITATION_NETWORK_PREPARE)

        with tempfile.TemporaryDirectory(prefix="algeon_dblp_footer_contract.") as tmp:
            root = Path(tmp)
            source = root / "dblp.json"
            source.write_text(
                "[\n"
                '{"id": 101, "references": [202]},\n'
                '{"id": 202, "references": [101]}\n'
                "]\n",
                encoding="utf-8",
            )

            def convert_args(max_papers: int) -> SimpleNamespace:
                return SimpleNamespace(
                    source_json=source,
                    out_root=root / "output",
                    row_group_size=2,
                    profile_row_group_size=2,
                    bucket_width=10,
                    max_papers=max_papers,
                    overwrite=False,
                    sort_buffer_rows=2,
                    merge_fan_in=2,
                    sort_temp_dir=None,
                    keep_work=False,
                    skip_sha256=True,
                    progress_interval=0.0,
                    quiet=True,
                )

            full_manifest = dblp.convert(
                dblp.DatasetSpec(expected_papers=2, expected_edges=2),
                convert_args(max_papers=0),
            )
            metadata_key = dblp.GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY
            footer_key = metadata_key.encode("ascii")
            topology_key = dblp.GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY
            topology_footer_key = topology_key.encode("ascii")
            expected_topology = '{"destination_column_index":2,"source_column_index":1}'
            for order in ("src", "dst"):
                parquet_path = Path(full_manifest["root"]) / "parquet" / f"edges_by_{order}.parquet"
                parquet_footer = dblp.pq.ParquetFile(parquet_path).metadata.metadata
                self.assertEqual(parquet_footer[footer_key], b"2")
                self.assertEqual(parquet_footer[topology_footer_key], expected_topology.encode("utf-8"))
                self.assertEqual(
                    full_manifest["edge_layouts"][order]["footer_metadata"],
                    {metadata_key: "2", topology_key: expected_topology},
                )

            sample_manifest = dblp.convert(
                dblp.DatasetSpec(expected_papers=2, expected_edges=2),
                convert_args(max_papers=1),
            )
            self.assertEqual(sample_manifest["counts"]["papers"], 1)
            for order in ("src", "dst"):
                parquet_path = Path(sample_manifest["root"]) / "parquet" / f"edges_by_{order}.parquet"
                parquet_footer = dblp.pq.ParquetFile(parquet_path).metadata.metadata
                self.assertNotIn(footer_key, parquet_footer)
                self.assertEqual(parquet_footer[topology_footer_key], expected_topology.encode("utf-8"))
                self.assertEqual(
                    sample_manifest["edge_layouts"][order]["footer_metadata"],
                    {topology_key: expected_topology},
                )

    def test_dblp_vertex_bound_requires_every_citation_target_in_source(self) -> None:
        dblp = load_script_module("citation_network_vertex_bound_under_test", CITATION_NETWORK_PREPARE)

        with tempfile.TemporaryDirectory(prefix="algeon_dblp_vertex_bound.") as tmp:
            root = Path(tmp)
            source = root / "dblp.json"
            # Paper 101 cites 303, which is not a paper in the source.
            source.write_text(
                "[\n"
                '{"id": 101, "references": [202, 303]},\n'
                '{"id": 202, "references": [101]}\n'
                "]\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                source_json=source,
                out_root=root / "output",
                row_group_size=2,
                profile_row_group_size=2,
                bucket_width=10,
                max_papers=0,
                overwrite=False,
                sort_buffer_rows=2,
                merge_fan_in=2,
                sort_temp_dir=None,
                keep_work=False,
                skip_sha256=True,
                progress_interval=0.0,
                quiet=True,
            )

            with self.assertRaisesRegex(SystemExit, "references paper 303"):
                dblp.convert(dblp.DatasetSpec(expected_papers=2, expected_edges=3), args)

            # A sample emits no vertex bound, so the dangling target is kept.
            args.max_papers = 1
            args.overwrite = True
            sample = dblp.convert(dblp.DatasetSpec(expected_papers=2, expected_edges=3), args)
            self.assertEqual(sample["counts"]["edges"], 2)

    def test_dblp_multi_graph_slices_are_disjoint_discoverable_and_rowset_distinct(self) -> None:
        dblp = load_script_module("citation_network_multigraph_contract_under_test", CITATION_NETWORK_PREPARE)

        with tempfile.TemporaryDirectory(prefix="algeon_dblp_multigraph_contract.") as tmp:
            root = Path(tmp)
            source = root / "dblp.json"
            papers = [
                {"id": paper_id, "references": [paper_id % 128 + 1], "title": f"paper-{paper_id}"}
                for paper_id in range(1, 129)
            ]
            source.write_text("[\n" + ",\n".join(json.dumps(paper) for paper in papers) + "\n]\n")

            args = SimpleNamespace(
                source_json=source,
                out_root=root / "output",
                row_group_size=16,
                profile_row_group_size=16,
                bucket_width=10,
                max_papers=0,
                overwrite=False,
                sort_buffer_rows=16,
                merge_fan_in=2,
                sort_temp_dir=None,
                keep_work=False,
                skip_sha256=True,
                edge_slice_count=4,
                edge_slice_seed=20_260_804,
                progress_interval=0.0,
                quiet=True,
            )
            manifest = dblp.convert(dblp.DatasetSpec(expected_papers=128, expected_edges=128), args)
            parent_root = Path(manifest["root"])
            slice_index = manifest["multi_graph_slices"]
            self.assertEqual(slice_index["partition"], dblp.MULTI_GRAPH_SLICE_PARTITION)
            self.assertEqual(slice_index["slice_count"], 4)
            self.assertEqual(len(slice_index["slices"]), 4)

            full_edge_ids = set(
                dblp.pq.read_table(parent_root / "parquet" / "edges_by_src.parquet").column("edge_id").to_pylist()
            )
            seen_edge_ids: set[int] = set()
            rowset_fingerprints: set[str] = set()
            footer_key = dblp.GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY.encode("ascii")
            for entry in slice_index["slices"]:
                index = entry["index"]
                slice_root = Path(entry["root"])
                self.assertEqual(slice_root, parent_root / "slices" / f"source-hash-{index:02d}")
                self.assertTrue((slice_root / "manifest.json").is_file())
                self.assertTrue((slice_root / "source.json").is_file())
                for table in ("edges_by_src", "edges_by_dst", "profiles_by_paper_id", "paper_authors", "paper_fos"):
                    self.assertTrue((slice_root / "parquet" / f"{table}.parquet").is_file(), table)

                slice_manifest = json.loads((slice_root / "manifest.json").read_text())
                slice_contract = slice_manifest["multi_graph_slice"]
                self.assertEqual(slice_contract["slice_index"], index)
                self.assertEqual(slice_contract["slice_count"], 4)
                self.assertEqual(slice_contract["edge_rowset_sha256"], entry["edge_rowset_sha256"])
                rowset_fingerprints.add(entry["edge_rowset_sha256"])
                source_metadata = json.loads((slice_root / "source.json").read_text())
                self.assertEqual(source_metadata["derived_fixture"], slice_contract)

                edge_table = dblp.pq.read_table(slice_root / "parquet" / "edges_by_src.parquet")
                edge_ids = set(edge_table.column("edge_id").to_pylist())
                self.assertTrue(edge_ids)
                self.assertFalse(seen_edge_ids & edge_ids)
                seen_edge_ids.update(edge_ids)
                for src in edge_table.column("src").to_pylist():
                    self.assertEqual(dblp.edge_slice_index(src, 4, 20_260_804), index)
                for order in ("src", "dst"):
                    parquet_footer = dblp.pq.ParquetFile(
                        slice_root / "parquet" / f"edges_by_{order}.parquet"
                    ).metadata.metadata
                    self.assertNotIn(footer_key, parquet_footer)

            self.assertEqual(seen_edge_ids, full_edge_ids)
            self.assertEqual(len(rowset_fingerprints), 4)

            args.overwrite = True
            repeated = dblp.convert(dblp.DatasetSpec(expected_papers=128, expected_edges=128), args)
            self.assertEqual(
                [entry["edge_rowset_sha256"] for entry in repeated["multi_graph_slices"]["slices"]],
                [entry["edge_rowset_sha256"] for entry in slice_index["slices"]],
            )

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_amazon_reviews_prepare_writes_gpu_scannable_keyed_tables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_amazon_reviews.") as tmp:
            root = Path(tmp)
            meta, reviews = write_amazon_reviews_source(root)
            result = run_cmd(amazon_reviews_prepare_command(root, meta, reviews))
            self.assertEqual(result.returncode, 0, result.stderr)

            products = pq.read_table(root / "out" / "parquet" / "products.parquet")
            self.assertEqual(products.column("parent_asin").to_pylist(), ["P1", "P2", "P3"])
            self.assertEqual(products.column("product_id").to_pylist(), [0, 1, 2])
            self.assertEqual(products.column("price").to_pylist(), [9.5, None, None])
            self.assertEqual(products.column("category_path").to_pylist()[0], "Electronics > Audio")

            review_file = pq.ParquetFile(root / "out" / "parquet" / "reviews.parquet")
            review_table = review_file.read()
            # review_id is the source line; rows are ordered by review_time.
            self.assertEqual(review_table.column("review_id").to_pylist(), [2, 0, 1])
            self.assertEqual(review_table.column("product_id").to_pylist(), [2, 1, 1])
            self.assertEqual(review_table.column("user_key").to_pylist(), [0, 1, 0])
            self.assertEqual(review_table.column("rating").to_pylist(), [3, 5, 1])
            self.assertEqual(review_table.column("image_count").to_pylist(), [0, 2, 0])

            # Timezone-aware timestamps and nested columns cannot be filtered,
            # grouped, or joined by the native GPU engine.
            for table in (products, review_table):
                for field in table.schema:
                    self.assertFalse(pa.types.is_nested(field.type), field)
                    self.assertFalse(pa.types.is_dictionary(field.type), field)
            self.assertEqual(review_table.schema.field("review_time").type, pa.timestamp("ms"))
            metadata = review_file.metadata
            for index in range(metadata.num_row_groups):
                for column in range(metadata.num_columns):
                    self.assertEqual(metadata.row_group(index).column(column).compression, "SNAPPY")
            self.assertEqual(metadata.row_group(0).sorting_columns, ())

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_amazon_reviews_prepare_rejects_reviews_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_amazon_reviews_orphan.") as tmp:
            root = Path(tmp)
            meta, reviews = write_amazon_reviews_source(root)
            with reviews.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(amazon_review("P9", "U1", 1_600_000_000_000, 4.0)) + "\n")
            result = run_cmd(amazon_reviews_prepare_command(root, meta, reviews))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing from metadata", result.stderr)
            self.assertFalse((root / "out" / "parquet" / "reviews.parquet").exists())

    def test_generators_default_to_repo_sf1_roots(self) -> None:
        layout = load_script_module("tpc_parquet_layout_under_test", TPC_PARQUET_LAYOUT)

        scales = layout.parse_scales(None)

        self.assertEqual(scales, [1.0])
        self.assertEqual(
            layout.resolve_output_dir(workload="tpch", scale=scales[0], out=None, out_root=None),
            REPO_ROOT / "fixture" / "tpc-h" / "data" / "sf1" / "parquet",
        )
        self.assertEqual(
            layout.resolve_output_dir(workload="tpcds", scale=scales[0], out=None, out_root=None),
            REPO_ROOT / "fixture" / "tpc-ds" / "data" / "sf1" / "parquet",
        )

    def test_fixture_router_calls_generate_raw_without_verification_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_router.") as tmp:
            root = Path(tmp)
            true_root = fake_tpcgen_root(root, "/bin/true")
            tpch = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpch",
                    "generate",
                    "--scale",
                    "sf1",
                    "--out",
                    str(root / "router_tpch"),
                    "--variant",
                    "raw",
                    "--tpchgen-root",
                    str(true_root),
                    "--no-verify",
                ]
            )
            self.assertEqual(tpch.returncode, 0, tpch.stderr)
            self.assertIn("tpch parquet", tpch.stderr)

            tpcds = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpcds",
                    "generate",
                    "--scale",
                    "sf1",
                    "--out",
                    str(root / "router_tpcds"),
                    "--variant",
                    "raw",
                    "--tpchgen-root",
                    str(true_root),
                    "--no-verify",
                ]
            )
            self.assertEqual(tpcds.returncode, 0, tpcds.stderr)
            self.assertIn("tpcds parquet", tpcds.stderr)

    def test_fixture_router_exposes_read_only_dblp_ingest_help(self) -> None:
        result = run_cmd([str(FIXTURE_SH), "graph", "dblp-ingest", "--help"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-json", result.stdout)
        self.assertIn("--out-root", result.stdout)

    def test_force_flag_preserves_overwrite_behavior(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_force.") as tmp:
            root = Path(tmp)
            true_root = fake_tpcgen_root(root, "/bin/true")
            out_dir = Path(tmp) / "tpch_force"
            out_dir.mkdir()
            stale = out_dir / "stale.parquet"
            stale.write_bytes(b"stale")

            result = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpch",
                    "generate",
                    "--out",
                    str(out_dir),
                    "--variant",
                    "raw",
                    "--tpchgen-root",
                    str(true_root),
                    "--no-verify",
                    "--force",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(stale.exists())

    def test_new_generator_entrypoints_remain_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_strict.") as tmp:
            root = Path(tmp)
            true_root = fake_tpcgen_root(root, "/bin/true")

            tpch = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCH),
                    "--scale",
                    "1",
                    "--variant",
                    "raw",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-root",
                    str(true_root),
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)

            tpcds = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCDS),
                    "--scale",
                    "1",
                    "--variant",
                    "raw",
                    "--out-root",
                    str(root / "tpcds"),
                    "--tpchgen-root",
                    str(true_root),
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)

    def test_new_optimized_variant_reaches_backend_with_out_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_optimized.") as tmp:
            root = Path(tmp)
            false_root = fake_tpcgen_root(root, "/bin/false")

            tpch = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCH),
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-root",
                    str(false_root),
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)
            self.assertIn("tpch parquet", tpch.stderr)

            tpcds = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCDS),
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpcds"),
                    "--tpchgen-root",
                    str(false_root),
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)
            self.assertIn("tpcds parquet", tpcds.stderr)

    def test_fixture_router_new_optimized_variant_reaches_backend_with_out_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_router_optimized.") as tmp:
            root = Path(tmp)
            false_root = fake_tpcgen_root(root, "/bin/false")

            tpch = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpch",
                    "generate",
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-root",
                    str(false_root),
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)
            self.assertIn("tpch parquet", tpch.stderr)

            tpcds = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpcds",
                    "generate",
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpcds"),
                    "--tpchgen-root",
                    str(false_root),
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)
            self.assertIn("tpcds parquet", tpcds.stderr)

    def test_iceberg_rest_loader_help_includes_add_files_mode(self) -> None:
        result = run_cmd([str(FIXTURE_SH), "iceberg", "rest", "load", "--help"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workload tpch|tpcds|citation_network|equality_delete", result.stdout)
        self.assertIn("--load-mode create|add-files|equality-delete", result.stdout)
        self.assertIn("--reload", result.stdout)

    def test_citation_network_loader_fails_fast_with_acquisition_instructions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_citation_missing.") as tmp:
            root = Path(tmp)
            result = run_cmd(
                [
                    str(FIXTURE_SH),
                    "iceberg",
                    "rest",
                    "load",
                    "--workload",
                    "citation_network",
                    "--load-mode",
                    "add-files",
                ],
                env={
                    "CITATION_NETWORK_DATA_DIR": str(root / "missing"),
                    "DATA_ROOT": str(root / "data"),
                    "IVY_CACHE": str(root / "ivy"),
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("missing citation network parquet", result.stderr)
        self.assertIn("dblp.v12.json", result.stderr)
        self.assertIn("citation_network/prepare.py", result.stderr)

    def test_citation_network_loader_rejects_partial_parquet_before_staging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_fixture_citation_partial.") as tmp:
            root = Path(tmp)
            parquet = root / "partial"
            parquet.mkdir()
            (parquet / "edges_by_src.parquet").write_bytes(b"not real parquet")
            result = run_cmd(
                [
                    str(FIXTURE_SH),
                    "iceberg",
                    "rest",
                    "load",
                    "--workload",
                    "citation_network",
                    "--load-mode",
                    "add-files",
                ],
                env={
                    "CITATION_NETWORK_DATA_DIR": str(parquet),
                    "DATA_ROOT": str(root / "data"),
                    "IVY_CACHE": str(root / "ivy"),
                    "DOCKER": "/bin/false",
                },
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("missing parquet inputs for citation_network", result.stderr)
        self.assertIn("papers: expected", result.stderr)
        self.assertIn("citation_network/prepare.py", result.stderr)

    def test_spark_iceberg_writer_parses_add_files_s3_sources_without_pyspark(self) -> None:
        writer = load_script_module("spark_write_iceberg_under_test", SPARK_WRITE_ICEBERG)

        args = writer.parse_args(
            [
                "--mode",
                "add-files",
                "--reload",
                "citation_network",
                "/data",
                "papers=s3://lakehouse/staging/citation_network/v1/papers.parquet",
            ]
        )
        self.assertEqual(args.mode, "add-files")
        self.assertTrue(args.reload)

        specs = writer.parse_table_specs(args.tables)
        self.assertEqual(specs, [("papers", "s3://lakehouse/staging/citation_network/v1/papers.parquet", None)])
        self.assertEqual(
            writer.resolve_source("/data", specs[0][1]),
            "s3://lakehouse/staging/citation_network/v1/papers.parquet",
        )
        self.assertEqual(
            writer.parquet_relation_for_add_files(specs[0][1]),
            "`parquet`.`s3://lakehouse/staging/citation_network/v1/papers.parquet`",
        )

    def test_spark_iceberg_writer_parses_three_part_add_files_specs(self) -> None:
        writer = load_script_module("spark_write_iceberg_three_part_under_test", SPARK_WRITE_ICEBERG)

        specs = writer.parse_table_specs(
            "papers=profiles_by_paper_id=s3://lakehouse/staging/citation_network/v1/papers.parquet"
        )
        self.assertEqual(
            specs,
            [(
                "papers",
                "profiles_by_paper_id",
                "s3://lakehouse/staging/citation_network/v1/papers.parquet",
            )],
        )

    def test_spark_iceberg_writer_add_files_mode_creates_and_registers_staged_parquet(self) -> None:
        writer = load_script_module("spark_write_iceberg_add_files_under_test", SPARK_WRITE_ICEBERG)
        spark = FakeSpark()

        writer.add_files_mode(
            spark,
            "citation_network",
            [(
                "papers",
                "s3://lakehouse/staging/citation_network/v1/papers.parquet",
                None,
            )],
            "/data",
            reload=False,
        )

        # Schema-only create goes through the DataFrame API, not a SQL direct
        # file query (rejected when spark.sql.defaultCatalog is the REST catalog).
        self.assertEqual(
            spark.parquet_reads,
            ["s3://lakehouse/staging/citation_network/v1/papers.parquet"],
        )
        self.assertEqual(spark.created_tables, ["`lake`.`citation_network`.`papers`"])
        self.assertTrue(
            any(
                "CALL lake.system.add_files(" in call
                and "table => '`citation_network`.`papers`'" in call
                and "source_table => '`parquet`.`s3://lakehouse/staging/citation_network/v1/papers.parquet`'" in call
                for call in spark.sql_calls
            )
        )
        rewrite_index = next(
            index
            for index, call in enumerate(spark.sql_calls)
            if "CALL lake.system.rewrite_manifests(" in call
            and "table => '`citation_network`.`papers`'" in call
        )
        self.assertIn(
            "SET TBLPROPERTIES ('commit.manifest.target-size-bytes'='1024')",
            spark.sql_calls[rewrite_index - 1],
        )
        self.assertIn(
            "UNSET TBLPROPERTIES ('commit.manifest.target-size-bytes')",
            spark.sql_calls[rewrite_index + 1],
        )
        self.assertFalse(any("schema.name-mapping.default" in call for call in spark.sql_calls))

    def test_spark_iceberg_writer_add_files_mode_reads_schema_locally_for_three_part_specs(self) -> None:
        writer = load_script_module("spark_write_iceberg_local_schema_under_test", SPARK_WRITE_ICEBERG)
        spark = FakeSpark()

        with tempfile.TemporaryDirectory(prefix="algeon_add_files_schema.") as tmp:
            (Path(tmp) / "profiles_by_paper_id.parquet").touch()
            writer.add_files_mode(
                spark,
                "citation_network",
                [(
                    "papers",
                    "profiles_by_paper_id",
                    "s3://lakehouse/staging/citation_network/v1/papers.parquet",
                )],
                tmp,
                reload=False,
            )

            self.assertEqual(spark.parquet_reads, [f"{tmp}/profiles_by_paper_id.parquet"])
        self.assertEqual(spark.created_tables, ["`lake`.`citation_network`.`papers`"])
        self.assertTrue(
            any(
                "source_table => '`parquet`.`s3://lakehouse/staging/citation_network/v1/papers.parquet`'" in call
                for call in spark.sql_calls
            )
        )

    def test_spark_iceberg_writer_quotes_procedure_table_identifier_parts(self) -> None:
        writer = load_script_module("spark_write_iceberg_procedure_identifier_under_test", SPARK_WRITE_ICEBERG)

        self.assertEqual(
            writer.procedure_table_identifier("tpch_sf0.01", "customer"),
            "`tpch_sf0.01`.`customer`",
        )

    def test_spark_iceberg_writer_add_files_mode_skips_existing_tables_with_data_files(self) -> None:
        writer = load_script_module("spark_write_iceberg_skip_existing_under_test", SPARK_WRITE_ICEBERG)
        spark = FakeSpark(existing=True, file_count=1)

        writer.add_files_mode(
            spark,
            "citation_network",
            [("papers", "s3://lakehouse/staging/citation_network/v1/papers.parquet", None)],
            "/data",
            reload=False,
        )

        self.assertEqual(spark.created_tables, [])
        self.assertFalse(any("CALL lake.system.add_files" in call for call in spark.sql_calls))

    def test_spark_iceberg_writer_add_files_mode_drops_created_table_after_failure(self) -> None:
        writer = load_script_module("spark_write_iceberg_failure_cleanup_under_test", SPARK_WRITE_ICEBERG)
        spark = FakeSpark(fail_on_add_files=True)

        with self.assertRaises(RuntimeError):
            writer.add_files_mode(
                spark,
                "citation_network",
                [("papers", "s3://lakehouse/staging/citation_network/v1/papers.parquet", None)],
                "/data",
                reload=False,
            )

        self.assertIn("DROP TABLE IF EXISTS `lake`.`citation_network`.`papers`", spark.sql_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
