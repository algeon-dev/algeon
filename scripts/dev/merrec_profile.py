#!/usr/bin/env python3
"""Profile the checked-in MerRec demo parquet with CPU DuckDB."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "fixture/demo/merrec"
DEFAULT_OUTPUT = ROOT / "docs/research/merrec/profile.json"

EVENT_COLUMNS = (
    "event_id",
    "event_time",
    "user_id",
    "item_id",
    "session_id",
    "sequence_id",
    "sequence_length",
    "event_type",
)
ITEM_COLUMNS = (
    "item_id",
    "product_id",
    "name",
    "price",
    "c0_id",
    "c0_name",
    "c1_id",
    "c1_name",
    "c2_id",
    "c2_name",
    "category_path",
    "brand_id",
    "brand_name",
    "item_condition_id",
    "item_condition_name",
    "size_id",
    "size_name",
    "color",
    "shipper_id",
    "shipper_name",
)


SQL: dict[str, str] = {
    "event_table_summary": """
        SELECT count(*) AS rows,
               count(DISTINCT event_id) AS distinct_event_ids,
               count(DISTINCT user_id) AS distinct_users,
               count(DISTINCT item_id) AS distinct_items,
               count(DISTINCT session_id) AS distinct_session_ids,
               count(DISTINCT sequence_id) AS distinct_sequence_ids,
               min(event_time) AS min_event_time,
               max(event_time) AS max_event_time
        FROM events
    """,
    "item_table_summary": """
        SELECT count(*) AS rows,
               count(DISTINCT item_id) AS distinct_item_ids,
               count(DISTINCT product_id) AS distinct_product_ids
        FROM items
    """,
    "item_duplicates": """
        SELECT count(*) AS duplicate_item_id_groups
        FROM (
            SELECT item_id FROM items GROUP BY item_id HAVING count(*) > 1
        ) duplicate_keys
    """,
    "sequence_scope": """
        WITH id_users AS (
            SELECT DISTINCT sequence_id, user_id FROM events
        ), id_summary AS (
            SELECT sequence_id, count(*) AS users_per_id
            FROM id_users GROUP BY sequence_id
        )
        SELECT (SELECT count(DISTINCT sequence_id) FROM events) AS sequence_ids,
               (SELECT count(*) FROM id_users) AS user_sequence_pairs,
               count(*) FILTER (WHERE users_per_id > 1) AS ids_used_by_multiple_users,
               coalesce(max(users_per_id), 0) AS max_users_for_one_id
        FROM id_summary
    """,
    "session_scope": """
        WITH id_users AS (
            SELECT DISTINCT session_id, user_id FROM events
        ), id_summary AS (
            SELECT session_id, count(*) AS users_per_id
            FROM id_users GROUP BY session_id
        )
        SELECT (SELECT count(DISTINCT session_id) FROM events) AS session_ids,
               (SELECT count(*) FROM id_users) AS user_session_pairs,
               count(*) FILTER (WHERE users_per_id > 1) AS ids_used_by_multiple_users,
               coalesce(max(users_per_id), 0) AS max_users_for_one_id
        FROM id_summary
    """,
    "event_types": """
        SELECT event_type, count(*) AS events,
               count(DISTINCT user_id) AS users,
               count(DISTINCT (user_id, session_id)) AS user_sessions
        FROM events GROUP BY event_type ORDER BY events DESC, event_type
    """,
    "timestamp_ties": """
        WITH timestamp_counts AS (
            SELECT event_time, count(*) AS rows_at_time
            FROM events GROUP BY event_time
        ), session_timestamp_counts AS (
            SELECT user_id, session_id, event_time, count(*) AS rows_at_time
            FROM events GROUP BY user_id, session_id, event_time
        )
        SELECT (SELECT count(*) FROM timestamp_counts WHERE rows_at_time > 1)
                    AS tied_timestamps,
               (SELECT coalesce(sum(rows_at_time - 1), 0) FROM timestamp_counts
                    WHERE rows_at_time > 1) AS rows_beyond_first_at_tied_timestamps,
               (SELECT coalesce(max(rows_at_time), 1) FROM timestamp_counts)
                    AS max_rows_at_one_timestamp,
               (SELECT count(*) FROM session_timestamp_counts WHERE rows_at_time > 1)
                    AS tied_user_session_timestamps,
               (SELECT coalesce(sum(rows_at_time - 1), 0)
                FROM session_timestamp_counts WHERE rows_at_time > 1)
                    AS user_session_rows_beyond_first_at_tied_timestamps,
               (SELECT coalesce(max(rows_at_time), 1) FROM session_timestamp_counts)
                    AS max_rows_at_one_user_session_timestamp
    """,
    "session_event_lengths": """
        WITH session_lengths AS (
            SELECT user_id, session_id, count(*) AS event_count,
                   count(*) FILTER (WHERE event_type = 'item_view') AS item_view_events,
                   count(DISTINCT item_id) FILTER (WHERE event_type = 'item_view')
                       AS distinct_viewed_items
            FROM events GROUP BY user_id, session_id
        )
        SELECT count(*) AS user_sessions,
               min(event_count) AS min_events,
               quantile_cont(event_count, 0.5) AS p50_events,
               quantile_cont(event_count, 0.9) AS p90_events,
               quantile_cont(event_count, 0.95) AS p95_events,
               quantile_cont(event_count, 0.99) AS p99_events,
               max(event_count) AS max_events,
               quantile_cont(item_view_events, 0.5) AS p50_item_view_events,
               quantile_cont(item_view_events, 0.9) AS p90_item_view_events,
               max(item_view_events) AS max_item_view_events,
               quantile_cont(distinct_viewed_items, 0.5) AS p50_distinct_viewed_items,
               quantile_cont(distinct_viewed_items, 0.9) AS p90_distinct_viewed_items,
               max(distinct_viewed_items) AS max_distinct_viewed_items,
               count(*) FILTER (WHERE distinct_viewed_items <= 50)
                   AS sessions_at_most_50_distinct_viewed_items,
               count(*) FILTER (WHERE item_view_events <= 50)
                   AS sessions_at_most_50_item_view_events
        FROM session_lengths
    """,
    "session_length_histogram": """
        WITH session_lengths AS (
            SELECT user_id, session_id, count(*) AS event_count
            FROM events GROUP BY user_id, session_id
        )
        SELECT CASE
                   WHEN event_count = 1 THEN '1'
                   WHEN event_count <= 5 THEN '2-5'
                   WHEN event_count <= 10 THEN '6-10'
                   WHEN event_count <= 20 THEN '11-20'
                   WHEN event_count <= 50 THEN '21-50'
                   WHEN event_count <= 100 THEN '51-100'
                   ELSE '101+'
               END AS event_count_band,
               count(*) AS sessions
        FROM session_lengths
        GROUP BY event_count_band
        ORDER BY min(event_count)
    """,
    "item_view_length_histogram": """
        WITH session_lengths AS (
            SELECT user_id, session_id,
                   count(DISTINCT item_id) AS distinct_viewed_items
            FROM events WHERE event_type = 'item_view'
            GROUP BY user_id, session_id
        )
        SELECT CASE
                   WHEN distinct_viewed_items = 1 THEN '1'
                   WHEN distinct_viewed_items <= 5 THEN '2-5'
                   WHEN distinct_viewed_items <= 10 THEN '6-10'
                   WHEN distinct_viewed_items <= 20 THEN '11-20'
                   WHEN distinct_viewed_items <= 50 THEN '21-50'
                   ELSE '51+'
               END AS distinct_item_band,
               count(*) AS sessions
        FROM session_lengths
        GROUP BY distinct_item_band
        ORDER BY min(distinct_viewed_items)
    """,
    "category_distribution": """
        SELECT coalesce(i.c0_name, '(missing)') AS category,
               count(DISTINCT i.item_id) AS catalog_items,
               count(e.event_id) AS events,
               count(*) FILTER (WHERE e.event_type = 'item_view') AS item_views
        FROM items i LEFT JOIN events e ON e.item_id = i.item_id
        GROUP BY category ORDER BY events DESC, category LIMIT 25
    """,
    "brand_distribution": """
        SELECT coalesce(i.brand_name, '(missing)') AS brand,
               count(DISTINCT i.item_id) AS catalog_items,
               count(e.event_id) AS events,
               count(*) FILTER (WHERE e.event_type = 'item_view') AS item_views
        FROM items i LEFT JOIN events e ON e.item_id = i.item_id
        GROUP BY brand ORDER BY events DESC, brand LIMIT 25
    """,
    "item_distribution": """
        SELECT i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
               i.brand_name, i.price,
               count(e.event_id) AS events,
               count(*) FILTER (WHERE e.event_type = 'item_view') AS item_views,
               count(*) FILTER (WHERE e.event_type IN ('item_like', 'buy_comp'))
                   AS likes_and_completed_buys
        FROM items i JOIN events e ON e.item_id = i.item_id
        GROUP BY i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
                 i.brand_name, i.price
        ORDER BY item_views DESC, events DESC, i.item_id
        LIMIT 20
    """,
    "electronics_products": """
        SELECT i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
               i.brand_name, i.price,
               count(e.event_id) AS events,
               count(*) FILTER (WHERE e.event_type = 'item_view') AS item_views,
               count(*) FILTER (WHERE e.event_type = 'item_like') AS likes,
               count(*) FILTER (WHERE e.event_type = 'buy_comp') AS completed_buys
        FROM items i JOIN events e ON e.item_id = i.item_id
        WHERE lower(i.c0_name) = 'electronics'
        GROUP BY i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
                 i.brand_name, i.price
        ORDER BY item_views DESC, events DESC, i.item_id LIMIT 20
    """,
    "video_games_products": """
        SELECT i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
               i.brand_name, i.price,
               count(e.event_id) AS events,
               count(*) FILTER (WHERE e.event_type = 'item_view') AS item_views,
               count(*) FILTER (WHERE e.event_type = 'item_like') AS likes,
               count(*) FILTER (WHERE e.event_type = 'buy_comp') AS completed_buys
        FROM items i JOIN events e ON e.item_id = i.item_id
        WHERE lower(i.c1_name) LIKE '%video games%'
           OR lower(i.c1_name) LIKE '%video game%'
        GROUP BY i.item_id, i.product_id, i.name, i.c0_name, i.c1_name,
                 i.brand_name, i.price
        ORDER BY item_views DESC, events DESC, i.item_id LIMIT 20
    """,
    "category_names": """
        SELECT c0_name, c1_name, count(*) AS catalog_items
        FROM items GROUP BY c0_name, c1_name
        ORDER BY catalog_items DESC, c0_name, c1_name LIMIT 100
    """,
    "consecutive_view_transition_summary": """
        WITH views AS (
            SELECT user_id, session_id, item_id, event_time, event_id,
                   lag(item_id) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_item_id,
                   lag(event_time) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_event_time
            FROM events WHERE event_type = 'item_view'
        ), transitions AS (
            SELECT user_id, session_id, previous_item_id AS from_item_id,
                   item_id AS to_item_id, previous_event_time, event_time
            FROM views
            WHERE previous_item_id IS NOT NULL
              AND item_id <> previous_item_id
              AND event_time - previous_event_time <= INTERVAL '30 minutes'
        )
        SELECT count(*) AS transition_steps,
               count(DISTINCT (user_id, session_id)) AS contributing_sessions,
               count(DISTINCT from_item_id) AS source_items,
               count(DISTINCT to_item_id) AS target_items
        FROM transitions
    """,
    "consecutive_view_item_edges": """
        WITH views AS (
            SELECT user_id, session_id, item_id, event_time, event_id,
                   lag(item_id) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_item_id,
                   lag(event_time) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_event_time
            FROM events WHERE event_type = 'item_view'
        ), transitions AS (
            SELECT user_id, session_id, previous_item_id AS from_item_id,
                   item_id AS to_item_id
            FROM views
            WHERE previous_item_id IS NOT NULL
              AND item_id <> previous_item_id
              AND event_time - previous_event_time <= INTERVAL '30 minutes'
        ), edges AS (
            SELECT from_item_id, to_item_id, count(*) AS transition_count,
                   count(DISTINCT (user_id, session_id)) AS session_support
            FROM transitions GROUP BY from_item_id, to_item_id
        )
        SELECT count(*) AS directed_edges,
               count(*) FILTER (WHERE session_support >= 2) AS edges_support_2,
               count(*) FILTER (WHERE session_support >= 3) AS edges_support_3,
               count(DISTINCT from_item_id) FILTER (WHERE session_support >= 1)
                   AS source_vertices_support_1,
               count(DISTINCT to_item_id) FILTER (WHERE session_support >= 1)
                   AS target_vertices_support_1,
               sum(transition_count) AS transition_steps,
               max(session_support) AS max_session_support
        FROM edges
    """,
    "top_consecutive_item_edges": """
        WITH views AS (
            SELECT user_id, session_id, item_id, event_time, event_id,
                   lag(item_id) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_item_id,
                   lag(event_time) OVER (
                       PARTITION BY user_id, session_id
                       ORDER BY event_time, event_id
                   ) AS previous_event_time
            FROM events WHERE event_type = 'item_view'
        ), transitions AS (
            SELECT user_id, session_id, previous_item_id AS from_item_id,
                   item_id AS to_item_id
            FROM views
            WHERE previous_item_id IS NOT NULL
              AND item_id <> previous_item_id
              AND event_time - previous_event_time <= INTERVAL '30 minutes'
        ), edges AS (
            SELECT from_item_id, to_item_id, count(*) AS transition_count,
                   count(DISTINCT (user_id, session_id)) AS session_support
            FROM transitions GROUP BY from_item_id, to_item_id
        )
        SELECT e.from_item_id, e.to_item_id, e.transition_count, e.session_support,
               f.name AS from_name, f.c0_name AS from_category,
               t.name AS to_name, t.c0_name AS to_category
        FROM edges e
        JOIN items f ON f.item_id = e.from_item_id
        JOIN items t ON t.item_id = e.to_item_id
        ORDER BY e.session_support DESC, e.transition_count DESC,
                 e.from_item_id, e.to_item_id LIMIT 20
    """,
    "category_transition_edges": """
        WITH views AS (
            SELECT e.user_id, e.session_id, e.item_id, e.event_time, e.event_id,
                   lag(e.item_id) OVER (
                       PARTITION BY e.user_id, e.session_id
                       ORDER BY e.event_time, e.event_id
                   ) AS previous_item_id,
                   lag(e.event_time) OVER (
                       PARTITION BY e.user_id, e.session_id
                       ORDER BY e.event_time, e.event_id
                   ) AS previous_event_time
            FROM events e WHERE e.event_type = 'item_view'
        ), transitions AS (
            SELECT v.user_id, v.session_id, f.c0_name AS from_category,
                   t.c0_name AS to_category
            FROM views v
            JOIN items f ON f.item_id = v.previous_item_id
            JOIN items t ON t.item_id = v.item_id
            WHERE v.previous_item_id IS NOT NULL
              AND v.item_id <> v.previous_item_id
              AND v.event_time - v.previous_event_time <= INTERVAL '30 minutes'
        )
        SELECT coalesce(from_category, '(missing)') AS from_category,
               coalesce(to_category, '(missing)') AS to_category,
               count(*) AS transition_count,
               count(DISTINCT (user_id, session_id)) AS session_support
        FROM transitions
        GROUP BY from_category, to_category
        ORDER BY session_support DESC, transition_count DESC,
                 from_category, to_category
    """,
    "product_transition_edges": """
        WITH views AS (
            SELECT e.user_id, e.session_id, e.item_id, e.event_time, e.event_id,
                   lag(e.item_id) OVER (
                       PARTITION BY e.user_id, e.session_id
                       ORDER BY e.event_time, e.event_id
                   ) AS previous_item_id,
                   lag(e.event_time) OVER (
                       PARTITION BY e.user_id, e.session_id
                       ORDER BY e.event_time, e.event_id
                   ) AS previous_event_time
            FROM events e WHERE e.event_type = 'item_view'
        ), transitions AS (
            SELECT v.user_id, v.session_id, f.product_id AS from_product_id,
                   t.product_id AS to_product_id
            FROM views v
            JOIN items f ON f.item_id = v.previous_item_id
            JOIN items t ON t.item_id = v.item_id
            WHERE v.previous_item_id IS NOT NULL
              AND v.item_id <> v.previous_item_id
              AND v.event_time - v.previous_event_time <= INTERVAL '30 minutes'
        )
        SELECT coalesce(from_product_id, '(missing)') AS from_product_id,
               coalesce(to_product_id, '(missing)') AS to_product_id,
               count(*) AS transition_count,
               count(DISTINCT (user_id, session_id)) AS session_support
        FROM transitions
        GROUP BY from_product_id, to_product_id
        ORDER BY session_support DESC, transition_count DESC,
                 from_product_id, to_product_id LIMIT 30
    """,
    "co_view_summary": """
        WITH session_items AS (
            SELECT user_id, session_id, item_id
            FROM events WHERE event_type = 'item_view'
            GROUP BY user_id, session_id, item_id
        ), session_lengths AS (
            SELECT user_id, session_id, count(*) AS distinct_items
            FROM session_items GROUP BY user_id, session_id
        )
        SELECT count(*) AS sessions_with_item_views,
               count(*) FILTER (WHERE distinct_items >= 2) AS sessions_with_pairs,
               count(*) FILTER (WHERE distinct_items <= 50) AS sessions_with_at_most_50,
               count(*) FILTER (WHERE distinct_items BETWEEN 2 AND 50)
                   AS eligible_sessions,
               max(distinct_items) AS max_distinct_items_before_bound,
               max(distinct_items) FILTER (WHERE distinct_items <= 50)
                   AS max_distinct_items_after_bound,
               sum(distinct_items) FILTER (WHERE distinct_items BETWEEN 2 AND 50)
                   AS eligible_session_item_occurrences
        FROM session_lengths
    """,
    "co_view_edges": """
        WITH session_items AS (
            SELECT user_id, session_id, item_id
            FROM events WHERE event_type = 'item_view'
            GROUP BY user_id, session_id, item_id
        ), session_lengths AS (
            SELECT user_id, session_id, count(*) AS distinct_items
            FROM session_items GROUP BY user_id, session_id
        ), bounded_items AS (
            SELECT si.user_id, si.session_id, si.item_id
            FROM session_items si
            JOIN session_lengths sl USING (user_id, session_id)
            WHERE sl.distinct_items BETWEEN 2 AND 50
        ), session_pairs AS (
            SELECT a.user_id, a.session_id, a.item_id AS item_a, b.item_id AS item_b
            FROM bounded_items a JOIN bounded_items b
              ON a.user_id = b.user_id AND a.session_id = b.session_id
             AND a.item_id < b.item_id
        )
        SELECT item_a, item_b,
               count(DISTINCT (user_id, session_id)) AS session_support
        FROM session_pairs GROUP BY item_a, item_b
    """,
    "co_view_graph_counts": """
        WITH edges AS ({co_view_edges}), tiers AS (
            SELECT 1 AS min_support UNION ALL SELECT 2 UNION ALL SELECT 3
        ), eligible_edges AS (
            SELECT tiers.min_support, edges.item_a, edges.item_b
            FROM tiers JOIN edges ON edges.session_support >= tiers.min_support
        ), graph_counts AS (
            SELECT min_support, count(*) AS edges
            FROM eligible_edges GROUP BY min_support
        ), vertices AS (
            SELECT min_support, item_a AS item_id FROM eligible_edges
            UNION SELECT min_support, item_b AS item_id FROM eligible_edges
        ), vertex_counts AS (
            SELECT min_support, count(*) AS vertices FROM vertices GROUP BY min_support
        )
        SELECT graph_counts.min_support, graph_counts.edges,
               vertex_counts.vertices,
               graph_counts.edges::DOUBLE /
                   nullif(vertex_counts.vertices * (vertex_counts.vertices - 1) / 2, 0)
                   AS undirected_density
        FROM graph_counts JOIN vertex_counts USING (min_support)
        ORDER BY min_support
    """,
    "top_co_view_edges": """
        WITH edges AS ({co_view_edges})
        SELECT e.item_a, e.item_b, e.session_support,
               a.name AS item_a_name, a.c0_name AS item_a_category,
               b.name AS item_b_name, b.c0_name AS item_b_category
        FROM edges e JOIN items a ON a.item_id = e.item_a
                     JOIN items b ON b.item_id = e.item_b
        ORDER BY e.session_support DESC, e.item_a, e.item_b LIMIT 20
    """,
}


def records(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    cursor = connection.execute(query)
    columns = [column[0] for column in cursor.description]
    result: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        result.append({key: json_value(value) for key, value in zip(columns, row)})
    return result


def one(connection: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    rows = records(connection, query)
    return rows[0] if rows else {}


def json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def null_counts_query(columns: tuple[str, ...]) -> str:
    expressions = ",\n".join(
        f"count(*) FILTER (WHERE {column} IS NULL) AS {column}"
        for column in columns
    )
    return f"SELECT {expressions} FROM {{table}}"


def table_null_counts(
    connection: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]
) -> dict[str, int]:
    query = null_counts_query(columns).format(table=table)
    return one(connection, query)


def table_schema(connection: duckdb.DuckDBPyConnection, table: str) -> list[dict[str, Any]]:
    return records(connection, f"DESCRIBE SELECT * FROM {table}")


def table_key_check(
    connection: duckdb.DuckDBPyConnection, table: str, columns: tuple[str, ...]
) -> dict[str, Any]:
    key = ", ".join(columns)
    sql = f"""
        SELECT count(*) AS rows,
               count(DISTINCT ({key})) AS distinct_keys,
               (SELECT count(*) FROM (
                   SELECT {key} FROM {table}
                   GROUP BY {key} HAVING count(*) > 1
               ) duplicate_keys) AS duplicate_key_groups
        FROM {table}
    """
    result = one(connection, sql)
    result["key"] = list(columns)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    parquet_dir = data_dir / "parquet"
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads = {max(1, args.threads)}")
    connection.execute("SET preserve_insertion_order = false")
    connection.read_parquet(str(parquet_dir / "events.parquet")).create_view("events")
    connection.read_parquet(str(parquet_dir / "items.parquet")).create_view("items")

    event_summary = one(connection, SQL["event_table_summary"])
    item_summary = one(connection, SQL["item_table_summary"])
    event_nulls = table_null_counts(connection, "events", EVENT_COLUMNS)
    item_nulls = table_null_counts(connection, "items", ITEM_COLUMNS)
    event_summary["null_counts"] = event_nulls
    item_summary["null_counts"] = item_nulls
    event_summary["schema"] = table_schema(connection, "events")
    item_summary["schema"] = table_schema(connection, "items")
    event_summary["key_checks"] = [
        table_key_check(connection, "events", ("event_id",)),
        table_key_check(connection, "events", ("user_id", "session_id", "event_id")),
    ]
    item_summary["key_checks"] = [
        table_key_check(connection, "items", ("item_id",)),
    ]
    sequence_scope = one(connection, SQL["sequence_scope"])
    session_scope = one(connection, SQL["session_scope"])
    timestamp_ties = one(connection, SQL["timestamp_ties"])
    session_lengths = one(connection, SQL["session_event_lengths"])
    co_view_summary = one(connection, SQL["co_view_summary"])

    category_transitions = records(connection, SQL["category_transition_edges"])
    product_transitions = records(connection, SQL["product_transition_edges"])
    co_view_edges_sql = SQL["co_view_edges"]
    co_view_graph_query = SQL["co_view_graph_counts"].format(
        co_view_edges=co_view_edges_sql
    )
    top_co_view_query = SQL["top_co_view_edges"].format(co_view_edges=co_view_edges_sql)

    profile = {
        "dataset": {
            "partition": manifest.get("dataset", {}).get("partition"),
            "source": manifest.get("dataset", {}).get("source"),
            "manifest_counts": manifest.get("counts", {}),
            "source_files": manifest.get("sources", []),
            "parquet_files": {
                table: {
                    "path": str(parquet_dir / f"{table}.parquet"),
                    "bytes": (parquet_dir / f"{table}.parquet").stat().st_size,
                }
                for table in ("events", "items")
            },
        },
        "tables": {"events": event_summary, "items": item_summary},
        "identity": {
            "sequence_id_scope": sequence_scope,
            "session_id_scope": session_scope,
        },
        "events": {
            "event_type_distribution": records(connection, SQL["event_types"]),
            "timestamp_ties": timestamp_ties,
        },
        "sessions": {
            "identity": "(user_id, session_id)",
            "length_summary": session_lengths,
            "event_count_histogram": records(
                connection, SQL["session_length_histogram"]
            ),
            "distinct_item_view_histogram": records(
                connection, SQL["item_view_length_histogram"]
            ),
        },
        "catalog_and_popularity": {
            "root_categories_by_joined_event_count": records(
                connection, SQL["category_distribution"]
            ),
            "brands_by_joined_event_count": records(
                connection, SQL["brand_distribution"]
            ),
            "top_items_by_views": records(connection, SQL["item_distribution"]),
            "top_electronics_products_by_views": records(
                connection, SQL["electronics_products"]
            ),
            "top_video_games_products_by_views": records(
                connection, SQL["video_games_products"]
            ),
            "root_and_second_level_category_names": records(
                connection, SQL["category_names"]
            ),
        },
        "graphs": {
            "consecutive_distinct_item_views": {
                "semantics": (
                    "Within (user_id, session_id), order item_view rows by "
                    "(event_time, event_id), compare each row with the immediately "
                    "previous item_view row, discard repeated adjacent item IDs, "
                    "and keep transitions whose adjacent event-time gap is at most "
                    "30 minutes. Edge support counts distinct user-session pairs."
                ),
                "summary": one(
                    connection, SQL["consecutive_view_transition_summary"]
                ),
                "edge_summary": one(
                    connection, SQL["consecutive_view_item_edges"]
                ),
                "top_directed_item_edges": records(
                    connection, SQL["top_consecutive_item_edges"]
                ),
                "directed_root_category_edges": category_transitions,
                "directed_product_id_edges": product_transitions,
            },
            "bounded_session_co_view": {
                "semantics": (
                    "For each (user_id, session_id), form unordered pairs of "
                    "distinct item_view item IDs, including sessions with 2 to 50 "
                    "distinct viewed items. Edge support is the number of unique "
                    "user-session pairs containing both items. Vertices are item IDs "
                    "incident to an edge meeting the support threshold; density is "
                    "edges divided by choose(vertices, 2)."
                ),
                "summary": co_view_summary,
                "graph_sizes_by_min_unique_session_support": records(
                    connection, co_view_graph_query
                ),
                "top_undirected_pairs": records(connection, top_co_view_query),
            },
        },
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "event_rows": event_summary["rows"],
                      "item_rows": item_summary["rows"],
                      "max_distinct_viewed_items_per_session": session_lengths[
                          "max_distinct_viewed_items"
                      ],
                      "co_view_graph_tiers": profile["graphs"]["bounded_session_co_view"][
                          "graph_sizes_by_min_unique_session_support"
                      ]}, indent=2, ensure_ascii=False))
    connection.close()


if __name__ == "__main__":
    main()
