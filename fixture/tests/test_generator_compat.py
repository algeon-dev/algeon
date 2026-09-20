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
TPC_PARQUET_LAYOUT = REPO_ROOT / "fixture" / "tpc" / "parquet_layout.py"
CITATION_NETWORK_PREPARE = REPO_ROOT / "fixture" / "demo" / "citation_network" / "scripts" / "prepare.py"
AMAZON_REVIEWS_PREPARE = REPO_ROOT / "fixture" / "demo" / "amazon_reviews_2023" / "scripts" / "prepare.py"
KUAIRAND_PREPARE = REPO_ROOT / "fixture" / "demo" / "kuairand" / "scripts" / "prepare.py"
MERREC_PREPARE = REPO_ROOT / "fixture" / "demo" / "merrec" / "scripts" / "prepare.py"


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


def write_kuairand_source(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    users_csv = root / "user_features_pure.csv"
    user_header = (
        "user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,"
        "follow_user_num,follow_user_num_range,fans_user_num,fans_user_num_range,"
        "friend_user_num,friend_user_num_range,register_days,register_days_range,"
        + ",".join(f"onehot_feat{i}" for i in range(18))
        + "\n"
    )
    user_rows = [
        '1,full_active,0,1,1,514,500+,150,"[100,1k)",34,"[30,60)",799,730+,' + ",".join(["1"] * 18) + "\n",
        '2,full_active,0,0,1,457,"(250,500]",20,"[10,100)",3,"[1,5)",1474,730+,' + ",".join(["0"] * 18) + "\n",
    ]
    users_csv.write_text(user_header + "".join(user_rows), encoding="utf-8")

    videos_csv = root / "video_features_basic_pure.csv"
    video_header = "video_id,author_id,video_type,upload_dt,upload_type,visible_status,video_duration,server_width,server_height,music_id,music_type,tag\n"
    video_rows = [
        "10,7349781,NORMAL,2022-04-10,LongImport,0.0,87433.0,720.0,1280.0,9155697141,9.0,39\n",
        "20,2103883,NORMAL,2022-04-11,Kmovie,0.0,218066.0,720.0,1280.0,6355810746,9.0,2\n",
    ]
    videos_csv.write_text(video_header + "".join(video_rows), encoding="utf-8")

    stat_cols = [
        "show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration",
        "complete_play_cnt", "complete_play_user_num", "valid_play_cnt", "valid_play_user_num",
        "long_time_play_cnt", "long_time_play_user_num", "short_time_play_cnt", "short_time_play_user_num",
        "play_progress", "comment_stay_duration", "like_cnt", "like_user_num",
        "click_like_cnt", "double_click_cnt", "cancel_like_cnt", "cancel_like_user_num",
        "comment_cnt", "comment_user_num", "direct_comment_cnt", "reply_comment_cnt",
        "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt", "comment_like_user_num",
        "follow_cnt", "follow_user_num", "cancel_follow_cnt", "cancel_follow_user_num",
        "share_cnt", "share_user_num", "download_cnt", "download_user_num",
        "report_cnt", "report_user_num", "reduce_similar_cnt", "reduce_similar_user_num",
        "collect_cnt", "collect_user_num", "cancel_collect_cnt", "cancel_collect_user_num",
        "direct_comment_user_num", "reply_comment_user_num", "share_all_cnt", "share_all_user_num",
        "outsite_share_all_cnt",
    ]
    video_stats_csv = root / "video_features_statistic_pure.csv"
    stat_header = "video_id,counts," + ",".join(stat_cols) + "\n"
    stat_rows = [
        "10,100," + ",".join(["1.0"] * len(stat_cols)) + "\n",
        "20,200," + ",".join(["2.0"] * len(stat_cols)) + "\n",
    ]
    video_stats_csv.write_text(stat_header + "".join(stat_rows), encoding="utf-8")

    log_header = "user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,is_forward,is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,comment_stay_time,is_profile_enter,is_rand,tab\n"
    rand_log = root / "log_random_4_22_to_5_08_pure.csv"
    rand_rows = [
        "1,10,20220422,1800,1650615878629,1,0,0,0,0,0,1,1000,87433,0,0,0,1,1\n",
        "2,20,20220423,1900,1650702278629,0,1,0,0,0,0,0,500,218066,0,0,0,1,1\n",
    ]
    rand_log.write_text(log_header + "".join(rand_rows), encoding="utf-8")

    std_log = root / "log_standard_4_08_to_4_21_pure.csv"
    std_rows = [
        "1,20,20220410,1200,1649577878629,0,0,0,0,0,0,0,300,218066,0,0,0,0,1\n",
    ]
    std_log.write_text(log_header + "".join(std_rows), encoding="utf-8")

    return users_csv, videos_csv, video_stats_csv, rand_log, std_log


def kuairand_prepare_command(
    root: Path, users: Path, videos: Path, stats: Path, rand_log: Path, std_log: Path
) -> list[str]:
    return [
        sys.executable,
        str(KUAIRAND_PREPARE),
        "--users-csv",
        str(users),
        "--videos-csv",
        str(videos),
        "--video-stats-csv",
        str(stats),
        "--log-random-csv",
        str(rand_log),
        "--log-standard-csv",
        str(std_log),
        "--out-dir",
        str(root / "out"),
        "--interaction-row-group-rows",
        "2",
        "--entity-row-group-rows",
        "2",
        "--skip-sha256",
    ]


def write_merrec_source(root: Path, *, invalid_price: bool = False) -> Path:
    source_file = root / "raw_events.parquet"
    rows = {
        "user_id": [101, 101, 102],
        "stime": [
            1_697_844_941_000_000,
            1_697_844_943_000_000,
            1_697_991_139_000_000,
        ],
        "session_id": ["s1", "s1", "s2"],
        "sequence_id": ["seq1", "seq1", "seq2"],
        "sequence_length": [2, 2, 1],
        "event_id": ["item_view", "item_like", "item_view"],
        "item_id": [201, 202, 201],
        "product_id": ["p1", "p2", "p1"],
        "name": ["Running Shoes", "Leather Jacket", "Running Shoes"],
        "price": [-55.0, 120.0, -55.0] if invalid_price else [55.0, 120.0, 55.0],
        "c0_name": ["Footwear", "Clothing", "Footwear"],
        "c0_id": [1, 2, 1],
        "c1_name": ["Athletic", "Outerwear", "Athletic"],
        "c1_id": [10, 20, 10],
        "c2_name": ["Sneakers", "Jackets", "Sneakers"],
        "c2_id": [100, 200, 100],
        "brand_name": ["Nike", "Zara", "Nike"],
        "brand_id": [50, 60, 50],
        "item_condition_id": [1, 2, 1],
        "item_condition_name": ["New", "Like New", "New"],
        "size_name": ["10", "L", "10"],
        "size_id": [5, 6, 5],
        "color": ["Black", "Brown", "Black"],
        "shipper_id": [1, 2, 1],
        "shipper_name": ["Buyer", "Seller", "Buyer"],
    }
    schema = pa.schema(
        [
            pa.field("user_id", pa.int64()),
            pa.field("stime", pa.timestamp("us")),
            pa.field("session_id", pa.string()),
            pa.field("sequence_id", pa.string()),
            pa.field("sequence_length", pa.int64()),
            pa.field("event_id", pa.string()),
            pa.field("item_id", pa.int64()),
            pa.field("product_id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("c0_name", pa.string()),
            pa.field("c0_id", pa.int64()),
            pa.field("c1_name", pa.string()),
            pa.field("c1_id", pa.int64()),
            pa.field("c2_name", pa.string()),
            pa.field("c2_id", pa.int64()),
            pa.field("brand_name", pa.string()),
            pa.field("brand_id", pa.int64()),
            pa.field("item_condition_id", pa.int64()),
            pa.field("item_condition_name", pa.string()),
            pa.field("size_name", pa.string()),
            pa.field("size_id", pa.int64()),
            pa.field("color", pa.string()),
            pa.field("shipper_id", pa.int64()),
            pa.field("shipper_name", pa.string()),
        ]
    )
    table = pa.Table.from_pydict(rows, schema=schema)
    pq.write_table(table, source_file)
    return source_file


def merrec_prepare_command(root: Path, source_parquet: Path) -> list[str]:
    return [
        sys.executable,
        str(MERREC_PREPARE),
        "--input-parquet",
        str(source_parquet),
        "--out-dir",
        str(root / "out"),
        "--event-row-group-rows",
        "2",
        "--item-row-group-rows",
        "2",
        "--skip-sha256",
    ]


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

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_kuairand_prepare_writes_gpu_scannable_keyed_tables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_kuairand.") as tmp:
            root = Path(tmp)
            users, videos, stats, rand_log, std_log = write_kuairand_source(root)
            result = run_cmd(kuairand_prepare_command(root, users, videos, stats, rand_log, std_log))
            self.assertEqual(result.returncode, 0, result.stderr)

            out_dir = root / "out"
            parquet_dir = out_dir / "parquet"

            user_table = pq.read_table(parquet_dir / "users.parquet")
            self.assertEqual(user_table.column("user_id").to_pylist(), [1, 2])
            self.assertEqual(user_table.column("user_active_degree").to_pylist(), ["full_active", "full_active"])

            video_table = pq.read_table(parquet_dir / "videos.parquet")
            self.assertEqual(video_table.column("video_id").to_pylist(), [10, 20])
            self.assertEqual(video_table.column("video_type").to_pylist(), ["NORMAL", "NORMAL"])

            stat_table = pq.read_table(parquet_dir / "video_stats.parquet")
            self.assertEqual(stat_table.column("video_id").to_pylist(), [10, 20])
            self.assertEqual(stat_table.column("counts").to_pylist(), [100, 200])

            interaction_file = pq.ParquetFile(parquet_dir / "interactions.parquet")
            interaction_table = interaction_file.read()
            self.assertEqual(interaction_table.num_rows, 3)
            self.assertEqual(interaction_table.column("video_id").to_pylist(), [20, 10, 20])
            self.assertEqual(interaction_table.column("user_id").to_pylist(), [1, 1, 2])
            self.assertEqual(interaction_table.column("is_rand").to_pylist(), [0, 1, 1])
            self.assertEqual(interaction_table.column("is_click").to_pylist(), [0, 1, 0])
            self.assertEqual(interaction_table.column("is_like").to_pylist(), [0, 0, 1])

            for table in (user_table, video_table, stat_table, interaction_table):
                for field in table.schema:
                    self.assertFalse(pa.types.is_nested(field.type), field)
                    self.assertFalse(pa.types.is_dictionary(field.type), field)
            self.assertEqual(interaction_table.schema.field("interaction_time").type, pa.timestamp("ms"))

            metadata = interaction_file.metadata
            for index in range(metadata.num_row_groups):
                for column in range(metadata.num_columns):
                    self.assertEqual(metadata.row_group(index).column(column).compression, "SNAPPY")
            self.assertEqual(metadata.row_group(0).sorting_columns, ())

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_kuairand_prepare_rejects_orphan_foreign_keys(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_kuairand_orphan.") as tmp:
            root = Path(tmp)
            users, videos, stats, rand_log, std_log = write_kuairand_source(root)
            with rand_log.open("a", encoding="utf-8") as handle:
                handle.write("999,10,20220422,1800,1650615878629,1,0,0,0,0,0,1,1000,87433,0,0,0,1,1\n")
            result = run_cmd(kuairand_prepare_command(root, users, videos, stats, rand_log, std_log))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing from users", result.stderr)
            self.assertFalse((root / "out" / "parquet" / "interactions.parquet").exists())

    def test_fixture_router_exposes_kuairand_prepare_help(self) -> None:
        result = run_cmd([str(FIXTURE_SH), "kuairand", "prepare", "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--variant", result.stdout)
        self.assertIn("--download", result.stdout)

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_merrec_prepare_writes_gpu_scannable_tables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_merrec.") as tmp:
            root = Path(tmp)
            source_parquet = write_merrec_source(root)
            result = run_cmd(merrec_prepare_command(root, source_parquet))
            self.assertEqual(result.returncode, 0, result.stderr)

            out_dir = root / "out"
            parquet_dir = out_dir / "parquet"

            item_file = pq.ParquetFile(parquet_dir / "items.parquet")
            item_table = item_file.read()
            self.assertEqual(item_table.column("item_id").to_pylist(), [201, 202])
            self.assertEqual(item_table.column("name").to_pylist(), ["Running Shoes", "Leather Jacket"])
            self.assertEqual(item_table.column("price").to_pylist(), [55.0, 120.0])
            self.assertEqual(
                item_table.column("category_path").to_pylist(),
                ["Footwear > Athletic > Sneakers", "Clothing > Outerwear > Jackets"],
            )
            self.assertEqual(item_table.column("brand_name").to_pylist(), ["Nike", "Zara"])

            event_file = pq.ParquetFile(parquet_dir / "events.parquet")
            event_table = event_file.read()
            self.assertEqual(event_table.num_rows, 3)
            self.assertEqual(event_table.column("event_id").to_pylist(), [0, 1, 2])
            self.assertEqual(event_table.column("user_id").to_pylist(), [101, 101, 102])
            self.assertEqual(event_table.column("item_id").to_pylist(), [201, 202, 201])
            self.assertEqual(event_table.column("event_type").to_pylist(), ["item_view", "item_like", "item_view"])

            for table in (item_table, event_table):
                for field in table.schema:
                    self.assertFalse(pa.types.is_nested(field.type), field)
                    self.assertFalse(pa.types.is_dictionary(field.type), field)
            self.assertEqual(event_table.schema.field("event_time").type, pa.timestamp("ms"))

            for f in (item_file, event_file):
                metadata = f.metadata
                for index in range(metadata.num_row_groups):
                    for column in range(metadata.num_columns):
                        self.assertEqual(metadata.row_group(index).column(column).compression, "SNAPPY")
                self.assertEqual(metadata.row_group(0).sorting_columns, ())

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["events"], 3)
            self.assertEqual(manifest["counts"]["items"], 2)
            self.assertEqual(manifest["counts"]["users"], 2)

    @unittest.skipUnless(shutil.which("duckdb"), "duckdb CLI is required")
    def test_merrec_prepare_rejects_negative_price(self) -> None:
        with tempfile.TemporaryDirectory(prefix="algeon_merrec_invalid.") as tmp:
            root = Path(tmp)
            source_parquet = write_merrec_source(root, invalid_price=True)
            result = run_cmd(merrec_prepare_command(root, source_parquet))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("negative price", result.stderr)
            self.assertFalse((root / "out" / "parquet" / "events.parquet").exists())

    def test_fixture_router_exposes_merrec_prepare_help(self) -> None:
        result = run_cmd([str(FIXTURE_SH), "merrec", "prepare", "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--partition", result.stdout)
        self.assertIn("--input-parquet", result.stdout)
        self.assertIn("--download-sample", result.stdout)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
