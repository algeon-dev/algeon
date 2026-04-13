#!/usr/bin/env python3
"""Compatibility tests for the public fixture entrypoints.

The tests use /bin/true and /bin/false as generator backends so they validate
argument routing, defaults, and verification behavior without generating large
benchmark datasets.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SH = REPO_ROOT / "fixture" / "scripts" / "fixture.sh"
GEN_TPCH = REPO_ROOT / "fixture" / "scripts" / "gen_tpch.py"
GEN_TPCDS = REPO_ROOT / "fixture" / "scripts" / "gen_tpcds.py"
SPARK_WRITE_ICEBERG = REPO_ROOT / "fixture" / "iceberg-local" / "spark_write_iceberg.py"
TPC_PARQUET_LAYOUT = REPO_ROOT / "fixture" / "scripts" / "tpc_parquet_layout.py"


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
    def test_generators_default_to_repo_sf1_roots(self) -> None:
        layout = load_script_module("tpc_parquet_layout_under_test", TPC_PARQUET_LAYOUT)

        scales = layout.parse_scales(None)

        self.assertEqual(scales, [1.0])
        self.assertEqual(
            layout.resolve_output_dir(workload="tpch", scale=scales[0], out=None, out_root=None),
            REPO_ROOT / "fixture" / "data" / "tpch" / "sf1" / "parquet",
        )
        self.assertEqual(
            layout.resolve_output_dir(workload="tpcds", scale=scales[0], out=None, out_root=None),
            REPO_ROOT / "fixture" / "data" / "tpcds" / "sf1" / "parquet",
        )

    def test_fixture_router_calls_generate_raw_without_verification_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_router.") as tmp:
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

    def test_force_flag_preserves_overwrite_behavior(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_force.") as tmp:
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
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_strict.") as tmp:
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
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_optimized.") as tmp:
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
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_router_optimized.") as tmp:
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
        self.assertIn("--load-mode create|add-files", result.stdout)
        self.assertIn("--reload", result.stdout)

    def test_citation_network_loader_fails_fast_with_acquisition_instructions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_citation_missing.") as tmp:
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
        self.assertIn("dblp_ingest.py", result.stderr)

    def test_citation_network_loader_rejects_partial_parquet_before_staging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_citation_partial.") as tmp:
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
        self.assertIn("dblp_ingest.py", result.stderr)

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

        with tempfile.TemporaryDirectory(prefix="nexus_add_files_schema.") as tmp:
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
