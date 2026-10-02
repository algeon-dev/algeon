#!/usr/bin/env python3
"""Convert KuaiRand sequential recommendation dataset from CSV into GPU-scan-friendly Parquet.

Inputs are raw KuaiRand files (Pure, 1K, or 27K variant):
- ``user_features_<variant>.csv``
- ``video_features_basic_<variant>.csv``
- ``video_features_statistic_<variant>*.csv``
- ``log_random_*.csv``
- ``log_standard_*.csv``

Outputs under ``--out-dir``:
- ``parquet/users.parquet``         one row per user, sorted by ``user_id``
- ``parquet/videos.parquet``        one row per video, sorted by ``video_id``
- ``parquet/video_stats.parquet``   one row per video, sorted by ``video_id``
- ``parquet/interactions.parquet``  one row per interaction, sorted by ``(interaction_time, interaction_id)``
- ``manifest.json``                 source identity, counts, and per-table Parquet layout

Optional outputs when supplementary files are supplied:
- ``parquet/video_categories.parquet``
- ``parquet/video_captions.parquet``
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_DIR = Path(__file__).resolve().parents[1]
GENERATOR = "fixture/demo/kuairand/scripts/prepare.py"
SOURCE = "KuaiRand (https://kuairand.com/ / https://zenodo.org/records/10439422)"
DEFAULT_VARIANT = "pure"
DEFAULT_INTERACTION_ROW_GROUP_ROWS = 1_048_576
DEFAULT_ENTITY_ROW_GROUP_ROWS = 131_072
BLOOM_FPP = 0.01

ZENODO_ARCHIVES = {
    "pure": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
    "1k": "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
    "27k": "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
}

USER_SCHEMA = pa.schema(
    [
        pa.field("user_id", pa.int32(), nullable=False),
        pa.field("user_active_degree", pa.string()),
        pa.field("is_lowactive_period", pa.int8()),
        pa.field("is_live_streamer", pa.int8()),
        pa.field("is_video_author", pa.int8()),
        pa.field("follow_user_num", pa.int32()),
        pa.field("follow_user_num_range", pa.string()),
        pa.field("fans_user_num", pa.int32()),
        pa.field("fans_user_num_range", pa.string()),
        pa.field("friend_user_num", pa.int32()),
        pa.field("friend_user_num_range", pa.string()),
        pa.field("register_days", pa.int32()),
        pa.field("register_days_range", pa.string()),
        pa.field("onehot_feat0", pa.int16()),
        pa.field("onehot_feat1", pa.int16()),
        pa.field("onehot_feat2", pa.int16()),
        pa.field("onehot_feat3", pa.int16()),
        pa.field("onehot_feat4", pa.int16()),
        pa.field("onehot_feat5", pa.int16()),
        pa.field("onehot_feat6", pa.int16()),
        pa.field("onehot_feat7", pa.int16()),
        pa.field("onehot_feat8", pa.int16()),
        pa.field("onehot_feat9", pa.int16()),
        pa.field("onehot_feat10", pa.int16()),
        pa.field("onehot_feat11", pa.int16()),
        pa.field("onehot_feat12", pa.int16()),
        pa.field("onehot_feat13", pa.int16()),
        pa.field("onehot_feat14", pa.int16()),
        pa.field("onehot_feat15", pa.int16()),
        pa.field("onehot_feat16", pa.int16()),
        pa.field("onehot_feat17", pa.int16()),
    ]
)

VIDEO_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.int32(), nullable=False),
        pa.field("author_id", pa.int64(), nullable=False),
        pa.field("video_type", pa.string()),
        pa.field("upload_date", pa.date32()),
        pa.field("upload_type", pa.string()),
        pa.field("visible_status", pa.int16()),
        pa.field("video_duration", pa.int32()),
        pa.field("server_width", pa.int16()),
        pa.field("server_height", pa.int16()),
        pa.field("music_id", pa.int64()),
        pa.field("music_type", pa.int16()),
        pa.field("tag", pa.string()),
    ]
)

STATISTIC_FLOAT_COLUMNS = (
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
)

VIDEO_STATS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.int32(), nullable=False),
        pa.field("counts", pa.int64(), nullable=False),
        *[pa.field(col, pa.float64()) for col in STATISTIC_FLOAT_COLUMNS],
    ]
)

INTERACTION_SCHEMA = pa.schema(
    [
        pa.field("interaction_id", pa.int64(), nullable=False),
        pa.field("interaction_time", pa.timestamp("ms"), nullable=False),
        pa.field("user_id", pa.int32(), nullable=False),
        pa.field("video_id", pa.int32(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("hourmin", pa.int16(), nullable=False),
        pa.field("time_ms", pa.int64(), nullable=False),
        pa.field("is_click", pa.int8(), nullable=False),
        pa.field("is_like", pa.int8(), nullable=False),
        pa.field("is_follow", pa.int8(), nullable=False),
        pa.field("is_comment", pa.int8(), nullable=False),
        pa.field("is_forward", pa.int8(), nullable=False),
        pa.field("is_hate", pa.int8(), nullable=False),
        pa.field("long_view", pa.int8(), nullable=False),
        pa.field("play_time_ms", pa.int64(), nullable=False),
        pa.field("duration_ms", pa.int64(), nullable=False),
        pa.field("profile_stay_time", pa.int64(), nullable=False),
        pa.field("comment_stay_time", pa.int64(), nullable=False),
        pa.field("is_profile_enter", pa.int8(), nullable=False),
        pa.field("is_rand", pa.int8(), nullable=False),
        pa.field("tab", pa.int8(), nullable=False),
    ]
)

VIDEO_CATEGORIES_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.int32(), nullable=False),
        pa.field("first_level_category_id", pa.int32()),
        pa.field("first_level_category_name", pa.string()),
        pa.field("first_level_category_prob", pa.float64()),
        pa.field("second_level_category_id", pa.int32()),
        pa.field("second_level_category_name", pa.string()),
        pa.field("second_level_category_prob", pa.float64()),
        pa.field("third_level_category_id", pa.int32()),
        pa.field("third_level_category_name", pa.string()),
        pa.field("third_level_category_prob", pa.float64()),
        pa.field("fourth_level_category_id", pa.int32()),
        pa.field("fourth_level_category_name", pa.string()),
        pa.field("fourth_level_category_prob", pa.float64()),
    ]
)

VIDEO_CAPTIONS_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.int32(), nullable=False),
        pa.field("caption", pa.string()),
        pa.field("show_cover_text", pa.string()),
        pa.field("duration", pa.int32()),
    ]
)


@dataclass(frozen=True)
class TableLayout:
    name: str
    schema: pa.Schema
    sort_keys: tuple[str, ...]
    dictionary_columns: tuple[str, ...]
    bloom_columns: tuple[str, ...]
    free_text_columns: tuple[str, ...]

    def statistics_columns(self) -> list[str]:
        return [name for name in self.schema.names if name not in self.free_text_columns]


USERS = TableLayout(
    name="users",
    schema=USER_SCHEMA,
    sort_keys=("user_id",),
    dictionary_columns=(
        "user_active_degree",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
    ),
    bloom_columns=("user_id",),
    free_text_columns=tuple(f"onehot_feat{i}" for i in range(18)),
)

VIDEOS = TableLayout(
    name="videos",
    schema=VIDEO_SCHEMA,
    sort_keys=("video_id",),
    dictionary_columns=("video_type", "upload_type", "visible_status", "music_type"),
    bloom_columns=("video_id", "author_id", "music_id"),
    free_text_columns=("tag",),
)

VIDEO_STATS = TableLayout(
    name="video_stats",
    schema=VIDEO_STATS_SCHEMA,
    sort_keys=("video_id",),
    dictionary_columns=(),
    bloom_columns=("video_id",),
    free_text_columns=STATISTIC_FLOAT_COLUMNS,
)

INTERACTIONS = TableLayout(
    name="interactions",
    schema=INTERACTION_SCHEMA,
    sort_keys=("interaction_time", "interaction_id"),
    dictionary_columns=(
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
        "tab",
    ),
    bloom_columns=("user_id", "video_id"),
    free_text_columns=(),
)

VIDEO_CATEGORIES = TableLayout(
    name="video_categories",
    schema=VIDEO_CATEGORIES_SCHEMA,
    sort_keys=("video_id",),
    dictionary_columns=("first_level_category_name", "second_level_category_name"),
    bloom_columns=("video_id",),
    free_text_columns=(),
)

VIDEO_CAPTIONS = TableLayout(
    name="video_captions",
    schema=VIDEO_CAPTIONS_SCHEMA,
    sort_keys=("video_id",),
    dictionary_columns=(),
    bloom_columns=("video_id",),
    free_text_columns=("caption", "show_cover_text"),
)


def log(message: str) -> None:
    print(f"[kuairand] {message}", file=sys.stderr, flush=True)


def sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def download_variant(variant: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_url = ZENODO_ARCHIVES.get(variant.lower())
    if not archive_url:
        raise SystemExit(f"unknown variant '{variant}'; choices: pure, 1k, 27k")
    archive_name = archive_url.split("/")[-1]
    archive_path = raw_dir / archive_name
    log(f"downloading {archive_url} -> {archive_path}")
    urllib.request.urlretrieve(archive_url, archive_path)
    log(f"extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=raw_dir)
    log(f"download complete in {raw_dir}")


def resolve_file(explicit: Path | None, raw_dir: Path, patterns: list[str]) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"specified file not found: {explicit}")
        return explicit
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if not matches:
            matches = sorted(raw_dir.glob(f"*/{pattern}"))
        if not matches:
            matches = sorted(raw_dir.glob(f"*/*/{pattern}"))
        if matches:
            return matches[0]
    return None


def resolve_file_list(explicit: list[Path] | None, raw_dir: Path, patterns: list[str]) -> list[Path]:
    if explicit:
        for p in explicit:
            if not p.is_file():
                raise SystemExit(f"specified file not found: {p}")
        return explicit
    collected: list[Path] = []
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if not matches:
            matches = sorted(raw_dir.glob(f"*/{pattern}"))
        if not matches:
            matches = sorted(raw_dir.glob(f"*/*/{pattern}"))
        collected.extend(matches)
    # Deduplicate preserving order
    seen: set[Path] = set()
    result: list[Path] = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=["pure", "1k", "27k"], help=f"Dataset variant (default: {DEFAULT_VARIANT})")
    parser.add_argument("--raw-dir", type=Path, default=DATASET_DIR / "raw", help="Directory containing raw CSVs or archives")
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR, help="Directory receiving parquet/ and manifest.json")
    parser.add_argument("--work-dir", type=Path, help="DuckDB intermediate and spill directory (default: <out-dir>/work)")
    parser.add_argument("--users-csv", type=Path, help="User features CSV path")
    parser.add_argument("--videos-csv", type=Path, help="Video basic features CSV path")
    parser.add_argument("--video-stats-csv", type=Path, nargs="+", help="Video statistic features CSV path(s)")
    parser.add_argument("--logs-csv", type=Path, nargs="+", help="All interaction log CSV path(s)")
    parser.add_argument("--log-random-csv", type=Path, nargs="+", help="Random interaction log CSV path(s)")
    parser.add_argument("--log-standard-csv", type=Path, nargs="+", help="Standard recommendation log CSV path(s)")
    parser.add_argument("--categories-csv", type=Path, help="Optional kuairand_video_categories.csv")
    parser.add_argument("--captions-csv", type=Path, help="Optional kuairand_video_captions.csv")
    parser.add_argument("--interaction-row-group-rows", type=int, default=DEFAULT_INTERACTION_ROW_GROUP_ROWS)
    parser.add_argument("--entity-row-group-rows", type=int, default=DEFAULT_ENTITY_ROW_GROUP_ROWS)
    parser.add_argument("--duckdb-memory-limit", help="DuckDB memory limit (e.g. 64GB)")
    parser.add_argument("--duckdb-threads", type=int, help="DuckDB thread count")
    parser.add_argument("--download", action="store_true", help="Download raw archive from Zenodo if missing")
    parser.add_argument("--skip-sha256", action="store_true", help="Do not hash source CSV files")
    parser.add_argument("--keep-work", action="store_true", help="Keep per-run intermediate directory")
    parser.add_argument("--force", action="store_true", help="Replace existing output Parquet and manifest")
    args = parser.parse_args()

    for name in ("interaction_row_group_rows", "entity_row_group_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.duckdb_threads is not None and args.duckdb_threads < 1:
        parser.error("--duckdb-threads must be >= 1")

    args.work_dir = args.work_dir or args.out_dir / "work"
    return args


def duckdb_script(
    args: argparse.Namespace,
    users_csv: Path,
    videos_csv: Path,
    video_stats_csvs: list[Path],
    logs_csvs: list[Path],
    categories_csv: Path | None,
    captions_csv: Path | None,
    users_tmp: Path,
    videos_tmp: Path,
    video_stats_tmp: Path,
    interactions_tmp: Path,
    categories_tmp: Path | None,
    captions_tmp: Path | None,
    stats_json_path: Path,
) -> str:
    settings = [".mode trash", f"SET temp_directory = {sql_string(args.work_dir / 'duckdb-tmp')};"]
    settings.append("SET preserve_insertion_order = true;")
    if args.duckdb_memory_limit:
        settings.append(f"SET memory_limit = {sql_string(args.duckdb_memory_limit)};")
    if args.duckdb_threads:
        settings.append(f"SET threads = {args.duckdb_threads};")

    stats_csv_sql = ", ".join(sql_string(p) for p in video_stats_csvs)
    logs_csv_sql = ", ".join(sql_string(p) for p in logs_csvs)

    script_lines = settings + [
        ".print load and validate users",
        f"""
CREATE TABLE users_src AS
SELECT * FROM read_csv({sql_string(users_csv)}, header = true);

SELECT CASE WHEN count(*) FILTER (user_id IS NULL) > 0
            THEN error('users have rows without user_id')
            WHEN count(*) <> count(DISTINCT user_id)
            THEN error('users user_id is not unique') END
FROM users_src;

CREATE TABLE users AS
SELECT
    user_id::INTEGER AS user_id,
    nullif(trim(user_active_degree::VARCHAR), '') AS user_active_degree,
    try_cast(is_lowactive_period AS DOUBLE)::TINYINT AS is_lowactive_period,
    try_cast(is_live_streamer AS DOUBLE)::TINYINT AS is_live_streamer,
    try_cast(is_video_author AS DOUBLE)::TINYINT AS is_video_author,
    try_cast(follow_user_num AS DOUBLE)::INTEGER AS follow_user_num,
    nullif(trim(follow_user_num_range::VARCHAR), '') AS follow_user_num_range,
    try_cast(fans_user_num AS DOUBLE)::INTEGER AS fans_user_num,
    nullif(trim(fans_user_num_range::VARCHAR), '') AS fans_user_num_range,
    try_cast(friend_user_num AS DOUBLE)::INTEGER AS friend_user_num,
    nullif(trim(friend_user_num_range::VARCHAR), '') AS friend_user_num_range,
    try_cast(register_days AS DOUBLE)::INTEGER AS register_days,
    nullif(trim(register_days_range::VARCHAR), '') AS register_days_range,
    {" , ".join(f"try_cast(onehot_feat{i} AS DOUBLE)::SMALLINT AS onehot_feat{i}" for i in range(18))}
FROM users_src;
DROP TABLE users_src;
""",
        ".print load and validate videos",
        f"""
CREATE TABLE videos_src AS
SELECT * FROM read_csv({sql_string(videos_csv)}, header = true);

SELECT CASE WHEN count(*) FILTER (video_id IS NULL) > 0
            THEN error('videos have rows without video_id')
            WHEN count(*) <> count(DISTINCT video_id)
            THEN error('videos video_id is not unique') END
FROM videos_src;

CREATE TABLE videos AS
SELECT
    video_id::INTEGER AS video_id,
    try_cast(author_id AS DOUBLE)::BIGINT AS author_id,
    nullif(trim(video_type::VARCHAR), '') AS video_type,
    try_strptime(upload_dt::VARCHAR, '%Y-%m-%d')::DATE AS upload_date,
    nullif(trim(upload_type::VARCHAR), '') AS upload_type,
    try_cast(visible_status AS DOUBLE)::SMALLINT AS visible_status,
    try_cast(video_duration AS DOUBLE)::INTEGER AS video_duration,
    try_cast(server_width AS DOUBLE)::SMALLINT AS server_width,
    try_cast(server_height AS DOUBLE)::SMALLINT AS server_height,
    try_cast(music_id AS DOUBLE)::BIGINT AS music_id,
    try_cast(music_type AS DOUBLE)::SMALLINT AS music_type,
    nullif(trim(tag::VARCHAR), '') AS tag
FROM videos_src;
DROP TABLE videos_src;
""",
        ".print load and validate video stats",
        f"""
CREATE TABLE video_stats_src AS
SELECT * FROM read_csv([{stats_csv_sql}], header = true);

SELECT CASE WHEN count(*) FILTER (video_id IS NULL) > 0
            THEN error('video_stats have rows without video_id')
            WHEN count(*) <> count(DISTINCT video_id)
            THEN error('video_stats video_id is not unique') END
FROM video_stats_src;

SELECT CASE WHEN count(*) > 0 THEN error('video_stats references video_id missing from videos') END
FROM video_stats_src s ANTI JOIN videos v ON s.video_id = v.video_id;

CREATE TABLE video_stats AS
SELECT
    video_id::INTEGER AS video_id,
    counts::BIGINT AS counts,
    {" , ".join(f"try_cast({c} AS DOUBLE) AS {c}" for c in STATISTIC_FLOAT_COLUMNS)}
FROM video_stats_src;
DROP TABLE video_stats_src;
""",
        ".print load and validate interaction logs",
        f"""
CREATE TABLE logs_src AS
SELECT * FROM read_csv([{logs_csv_sql}], header = true);

SELECT CASE WHEN count(*) FILTER (user_id IS NULL OR video_id IS NULL OR time_ms IS NULL) > 0
            THEN error('logs have rows without user_id, video_id, or time_ms')
            WHEN count(*) FILTER (is_rand NOT IN (0, 1)) > 0
            THEN error('logs have rows with is_rand not in (0, 1)')
            WHEN count(*) FILTER (is_click NOT IN (0, 1) OR is_like NOT IN (0, 1)) > 0
            THEN error('logs have feedback flags outside 0..1') END
FROM logs_src;

SELECT CASE WHEN count(*) > 0 THEN error('logs reference user_id missing from users') END
FROM logs_src l ANTI JOIN users u ON l.user_id = u.user_id;

SELECT CASE WHEN count(*) > 0 THEN error('logs reference video_id missing from videos') END
FROM logs_src l ANTI JOIN videos v ON l.video_id = v.video_id;

CREATE TABLE interactions AS
SELECT
    (row_number() OVER (ORDER BY time_ms, user_id, video_id, play_time_ms, tab) - 1)::BIGINT AS interaction_id,
    epoch_ms(time_ms)::TIMESTAMP_MS AS interaction_time,
    user_id::INTEGER AS user_id,
    video_id::INTEGER AS video_id,
    try_strptime(date::VARCHAR, '%Y%m%d')::DATE AS date,
    try_cast(hourmin AS DOUBLE)::SMALLINT AS hourmin,
    time_ms::BIGINT AS time_ms,
    try_cast(is_click AS DOUBLE)::TINYINT AS is_click,
    try_cast(is_like AS DOUBLE)::TINYINT AS is_like,
    try_cast(is_follow AS DOUBLE)::TINYINT AS is_follow,
    try_cast(is_comment AS DOUBLE)::TINYINT AS is_comment,
    try_cast(is_forward AS DOUBLE)::TINYINT AS is_forward,
    try_cast(is_hate AS DOUBLE)::TINYINT AS is_hate,
    try_cast(long_view AS DOUBLE)::TINYINT AS long_view,
    try_cast(play_time_ms AS DOUBLE)::BIGINT AS play_time_ms,
    try_cast(duration_ms AS DOUBLE)::BIGINT AS duration_ms,
    try_cast(profile_stay_time AS DOUBLE)::BIGINT AS profile_stay_time,
    try_cast(comment_stay_time AS DOUBLE)::BIGINT AS comment_stay_time,
    try_cast(is_profile_enter AS DOUBLE)::TINYINT AS is_profile_enter,
    try_cast(is_rand AS DOUBLE)::TINYINT AS is_rand,
    try_cast(tab AS DOUBLE)::TINYINT AS tab
FROM logs_src;
DROP TABLE logs_src;
""",
        ".print write sorted tables",
        f"COPY (SELECT * FROM users ORDER BY user_id) TO {sql_string(users_tmp)} (FORMAT parquet, COMPRESSION snappy);",
        f"COPY (SELECT * FROM videos ORDER BY video_id) TO {sql_string(videos_tmp)} (FORMAT parquet, COMPRESSION snappy);",
        f"COPY (SELECT * FROM video_stats ORDER BY video_id) TO {sql_string(video_stats_tmp)} (FORMAT parquet, COMPRESSION snappy);",
        f"COPY (SELECT * FROM interactions ORDER BY interaction_time, interaction_id) TO {sql_string(interactions_tmp)} (FORMAT parquet, COMPRESSION snappy);",
    ]

    if categories_csv is not None and categories_tmp is not None:
        script_lines.extend(
            [
                ".print process video categories",
                f"""
CREATE TABLE categories_src AS
SELECT * FROM read_csv({sql_string(categories_csv)}, header = true);

CREATE TABLE video_categories AS
SELECT
    final_video_id::INTEGER AS video_id,
    try_cast(first_level_category_id AS DOUBLE)::INTEGER AS first_level_category_id,
    nullif(trim(first_level_category_name::VARCHAR), '') AS first_level_category_name,
    try_cast(first_level_category_prob AS DOUBLE) AS first_level_category_prob,
    try_cast(second_level_category_id AS DOUBLE)::INTEGER AS second_level_category_id,
    nullif(trim(second_level_category_name::VARCHAR), '') AS second_level_category_name,
    try_cast(second_level_category_prob AS DOUBLE) AS second_level_category_prob,
    try_cast(third_level_category_id AS DOUBLE)::INTEGER AS third_level_category_id,
    nullif(trim(third_level_category_name::VARCHAR), '') AS third_level_category_name,
    try_cast(third_level_category_prob AS DOUBLE) AS third_level_category_prob,
    try_cast(fourth_level_category_id AS DOUBLE)::INTEGER AS fourth_level_category_id,
    nullif(trim(fourth_level_category_name::VARCHAR), '') AS fourth_level_category_name,
    try_cast(fourth_level_category_prob AS DOUBLE) AS fourth_level_category_prob
FROM categories_src;
DROP TABLE categories_src;
COPY (SELECT * FROM video_categories ORDER BY video_id) TO {sql_string(categories_tmp)} (FORMAT parquet, COMPRESSION snappy);
""",
            ]
        )

    if captions_csv is not None and captions_tmp is not None:
        script_lines.extend(
            [
                ".print process video captions",
                f"""
CREATE TABLE captions_src AS
SELECT * FROM read_csv({sql_string(captions_csv)}, header = true);

CREATE TABLE video_captions AS
SELECT
    final_video_id::INTEGER AS video_id,
    nullif(trim(caption::VARCHAR), '') AS caption,
    nullif(trim(show_cover_text::VARCHAR), '') AS show_cover_text,
    try_cast(duration AS DOUBLE)::INTEGER AS duration
FROM captions_src;
DROP TABLE captions_src;
COPY (SELECT * FROM video_captions ORDER BY video_id) TO {sql_string(captions_tmp)} (FORMAT parquet, COMPRESSION snappy);
""",
            ]
        )

    script_lines.extend(
        [
            ".print collect summary statistics",
            f"""
COPY (
    SELECT
        (SELECT count(*) FROM users) AS users,
        (SELECT count(*) FROM videos) AS videos,
        (SELECT count(*) FROM video_stats) AS video_stats,
        (SELECT count(*) FROM interactions) AS interactions,
        (SELECT count(*) FROM interactions WHERE is_rand = 1) AS random_interactions,
        (SELECT count(*) FROM interactions WHERE is_rand = 0) AS standard_interactions,
        (SELECT min(time_ms) FROM interactions) AS min_timestamp_ms,
        (SELECT max(time_ms) FROM interactions) AS max_timestamp_ms
) TO {sql_string(stats_json_path)} (FORMAT json);
""",
        ]
    )

    return "\n".join(script_lines)


def run_duckdb(script: str) -> None:
    duckdb = shutil.which("duckdb")
    if duckdb is None:
        raise SystemExit("duckdb executable not found; install the DuckDB CLI")
    result = subprocess.run([duckdb, "-bail", ":memory:"], input=script, text=True)
    if result.returncode != 0:
        raise SystemExit(f"duckdb exited with status {result.returncode}; see the error above")


def bloom_options(layout: TableLayout, row_group_rows: int, table_rows: int) -> dict[str, dict[str, Any]]:
    ndv = max(1024, min(row_group_rows, table_rows))
    return {column: {"ndv": ndv, "fpp": BLOOM_FPP} for column in layout.bloom_columns}


def rewrite(layout: TableLayout, source: Path, destination: Path, row_group_rows: int) -> None:
    reader = pq.ParquetFile(source)
    if reader.schema_arrow.names != layout.schema.names:
        raise SystemExit(f"{source}: columns {reader.schema_arrow.names} do not match {layout.schema.names}")
    tmp_path = destination.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        tmp_path,
        layout.schema,
        compression="snappy",
        version="2.6",
        write_statistics=layout.statistics_columns(),
        write_page_index=True,
        use_dictionary=list(layout.dictionary_columns),
        bloom_filter_options=bloom_options(layout, row_group_rows, reader.metadata.num_rows),
    )
    written = 0
    try:
        for batch in reader.iter_batches(batch_size=row_group_rows):
            table = pa.Table.from_batches([batch]).cast(layout.schema)
            writer.write_table(table, row_group_size=row_group_rows)
            written += table.num_rows
            log(f"{layout.name}: wrote {written:,}/{reader.metadata.num_rows:,} rows")
    except BaseException:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    writer.close()
    tmp_path.replace(destination)


def verify(layout: TableLayout, path: Path, expected_rows: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if parquet_file.schema_arrow != layout.schema:
        raise SystemExit(f"{path}: schema {parquet_file.schema_arrow} does not match {layout.schema}")
    if metadata.num_rows != expected_rows:
        raise SystemExit(f"{path}: {metadata.num_rows} rows, expected {expected_rows}")
    key_index = layout.schema.get_field_index(layout.sort_keys[0])
    previous_max = None
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        for column in range(row_group.num_columns):
            compression = row_group.column(column).compression
            if compression != "SNAPPY":
                raise SystemExit(f"{path}: row group {index} column {column} uses {compression}, expected SNAPPY")
        stats = row_group.column(key_index).statistics
        if stats is None or not stats.has_min_max:
            raise SystemExit(f"{path}: row group {index} lacks min/max statistics for {layout.sort_keys[0]}")
        if previous_max is not None and stats.min < previous_max:
            raise SystemExit(f"{path}: row group {index} is not sorted by {layout.sort_keys[0]}")
        previous_max = stats.max
    return {
        "path": f"parquet/{path.name}",
        "bytes": path.stat().st_size,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "schema": layout.schema.names,
        "sort_keys": list(layout.sort_keys),
        "dictionary_columns": list(layout.dictionary_columns),
        "bloom_filter_columns": list(layout.bloom_columns),
        "free_text_columns_without_statistics": list(layout.free_text_columns),
    }


def source_entry(path: Path, skip_sha256: bool) -> dict[str, Any]:
    digest = None
    if not skip_sha256:
        log(f"hash {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    # Discover input CSV files
    var = args.variant.lower()
    users_csv = resolve_file(args.users_csv, args.raw_dir, [f"user_features_{var}.csv", "user_features_*.csv", "user_features.csv"])
    videos_csv = resolve_file(args.videos_csv, args.raw_dir, [f"video_features_basic_{var}.csv", "video_features_basic_*.csv", "video_features_basic.csv"])
    video_stats_csvs = resolve_file_list(
        args.video_stats_csv,
        args.raw_dir,
        [f"video_features_statistic_{var}*.csv", "video_features_statistic_*.csv", "video_features_statistic.csv"],
    )

    if args.logs_csv:
        logs_csvs = resolve_file_list(args.logs_csv, args.raw_dir, [])
    else:
        rand_csvs = resolve_file_list(args.log_random_csv, args.raw_dir, [f"log_random_*{var}*.csv", "log_random_*.csv"])
        std_csvs = resolve_file_list(args.log_standard_csv, args.raw_dir, [f"log_standard_*{var}*.csv", "log_standard_*.csv"])
        logs_csvs = rand_csvs + std_csvs

    missing = []
    if users_csv is None:
        missing.append("users CSV (user_features_*.csv)")
    if videos_csv is None:
        missing.append("videos CSV (video_features_basic_*.csv)")
    if not video_stats_csvs:
        missing.append("video stats CSV (video_features_statistic_*.csv)")
    if not logs_csvs:
        missing.append("interaction log CSVs (log_random_*.csv, log_standard_*.csv)")

    if missing:
        if args.download:
            download_variant(args.variant, args.raw_dir)
            return main()
        raise SystemExit(
            f"source CSV files missing from {args.raw_dir}: {', '.join(missing)}; "
            f"pass --download to fetch {args.variant} from Zenodo or see fixture/demo/kuairand/README.md"
        )

    categories_csv = resolve_file(args.categories_csv, args.raw_dir, ["kuairand_video_categories.csv", "*categories*.csv"])
    captions_csv = resolve_file(args.captions_csv, args.raw_dir, ["kuairand_video_captions.csv", "*captions*.csv"])

    parquet_dir = args.out_dir / "parquet"
    manifest_path = args.out_dir / "manifest.json"
    outputs = [
        parquet_dir / "users.parquet",
        parquet_dir / "videos.parquet",
        parquet_dir / "video_stats.parquet",
        parquet_dir / "interactions.parquet",
        manifest_path,
    ]
    if categories_csv:
        outputs.append(parquet_dir / "video_categories.parquet")
    if captions_csv:
        outputs.append(parquet_dir / "video_captions.parquet")

    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"output exists: {', '.join(map(str, existing))}; pass --force to replace it")
    for path in existing:
        if not path.is_symlink():
            path.unlink()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = Path(tempfile.mkdtemp(prefix="kuairand-", dir=args.work_dir))
    parquet_dir.mkdir(parents=True, exist_ok=True)

    users_tmp = args.work_dir / "users.sorted.parquet"
    videos_tmp = args.work_dir / "videos.sorted.parquet"
    video_stats_tmp = args.work_dir / "video_stats.sorted.parquet"
    interactions_tmp = args.work_dir / "interactions.sorted.parquet"
    categories_tmp = args.work_dir / "video_categories.sorted.parquet" if categories_csv else None
    captions_tmp = args.work_dir / "video_captions.sorted.parquet" if captions_csv else None
    stats_json_path = args.work_dir / "stats.json"

    try:
        log(f"DuckDB transforming KuaiRand ({args.variant})")
        script = duckdb_script(
            args,
            users_csv,
            videos_csv,
            video_stats_csvs,
            logs_csvs,
            categories_csv,
            captions_csv,
            users_tmp,
            videos_tmp,
            video_stats_tmp,
            interactions_tmp,
            categories_tmp,
            captions_tmp,
            stats_json_path,
        )
        run_duckdb(script)
        counts = json.loads(stats_json_path.read_text(encoding="utf-8").splitlines()[0])
        tables = {}

        tasks = [
            (USERS, users_tmp, args.entity_row_group_rows, counts["users"]),
            (VIDEOS, videos_tmp, args.entity_row_group_rows, counts["videos"]),
            (VIDEO_STATS, video_stats_tmp, args.entity_row_group_rows, counts["video_stats"]),
            (INTERACTIONS, interactions_tmp, args.interaction_row_group_rows, counts["interactions"]),
        ]
        if categories_tmp and categories_tmp.exists():
            tasks.append((VIDEO_CATEGORIES, categories_tmp, args.entity_row_group_rows, pq.ParquetFile(categories_tmp).metadata.num_rows))
        if captions_tmp and captions_tmp.exists():
            tasks.append((VIDEO_CAPTIONS, captions_tmp, args.entity_row_group_rows, pq.ParquetFile(captions_tmp).metadata.num_rows))

        for layout, source, row_group_rows, expected_rows in tasks:
            dest = parquet_dir / f"{layout.name}.parquet"
            rewrite(layout, source, dest, row_group_rows)
            tables[layout.name] = verify(layout, dest, expected_rows)
            tables[layout.name]["row_group_rows"] = row_group_rows
    finally:
        if not args.keep_work:
            shutil.rmtree(args.work_dir, ignore_errors=True)

    sources_map: dict[str, Any] = {
        "users": source_entry(users_csv, args.skip_sha256),
        "videos": source_entry(videos_csv, args.skip_sha256),
        "video_stats": [source_entry(p, args.skip_sha256) for p in video_stats_csvs],
        "logs": [source_entry(p, args.skip_sha256) for p in logs_csvs],
    }
    if categories_csv:
        sources_map["categories"] = source_entry(categories_csv, args.skip_sha256)
    if captions_csv:
        sources_map["captions"] = source_entry(captions_csv, args.skip_sha256)

    manifest = {
        "generator": GENERATOR,
        "generated_at_unix": int(time.time()),
        "dataset": {"source": SOURCE, "variant": args.variant},
        "sources": sources_map,
        "counts": counts,
        "layout": {"compression": "SNAPPY", "format_version": "2.6", "sorting_columns_metadata": False},
        "tables": tables,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"complete: {counts['interactions']:,} interactions, {counts['users']:,} users, {counts['videos']:,} videos -> {parquet_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
