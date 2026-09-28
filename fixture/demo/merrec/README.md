# MerRec (Mercari C2C Recommendation)

A relational e-commerce and interaction graph demo dataset derived from the
Mercari C2C marketplace dataset. It exercises temporal filtering, high-cardinality
group-bys, hash joins, and bipartite graph queries on the native GPU engine.

## Source

[mercari-us/merrec](https://huggingface.co/datasets/mercari-us/merrec),
described in the KDD 2025 paper *MerRec: A Large-scale Multipurpose Mercari Dataset
for Consumer-to-Consumer Recommendation Systems* (Lichi Li, Zainul Abi Din,
Zhen Tan, Sam London, Tianlong Chen, Ajay Daptardar).

Upstream organizes files across six monthly partitions in 2023:
`20230501`, `20230601`, `20230701`, `20230801`, `20230901`, and `20231001`.
Each partition holds approximately 300 to 400 Parquet shards (~65 MB each).

## Acquisition

Download shards using the Hugging Face CLI:

```bash
# Download a single shard for evaluation (~65 MB, ~494K events)
hf download mercari-us/merrec 20231001/000000000000.parquet \
  --repo-type dataset --local-dir fixture/demo/merrec/raw

# Download a complete monthly partition (e.g. 20231001)
hf download mercari-us/merrec --include "20231001/*.parquet" \
  --repo-type dataset --local-dir fixture/demo/merrec/raw
```

Alternatively, invoke `fixture/demo/merrec/scripts/prepare.py --download-sample` to retrieve the first shard
of partition `20231001` directly into `raw/sample/`.

## Conversion

```bash
fixture/fixture.sh merrec prepare
# same as: python3 fixture/demo/merrec/scripts/prepare.py
```

The converter requires the `duckdb` CLI and `pyarrow`. DuckDB parses source
Parquet shards, enforces key constraints, deduplicates item attributes to their
latest timestamp, and sorts intermediates using external-memory spilling. PyArrow
streams the sorted tables into Snappy-compressed format 2.6 Parquet with GPU scan
properties.

Options:

- `--partition NAME`: Monthly partition folder under `raw/` (default: `20231001`).
- `--input-parquet PATH`: Explicit Parquet file, directory, or glob pattern.
- `--out-dir PATH`: Directory receiving `parquet/` and `manifest.json`.
- `--work-dir PATH`: Directory for DuckDB scratch files and intermediate tables.
- `--event-row-group-rows N`: Row group size for `events` (default: 1,048,576).
- `--item-row-group-rows N`: Row group size for `items` (default: 131,072).
- `--limit N`: Limit total input rows read from source files.
- `--download-sample`: Fetch the first shard from Hugging Face if raw inputs are absent.
- `--force`: Overwrite existing Parquet tables and manifest.

Output layout:

```
manifest.json       source file metadata, row counts, and table layouts
parquet/
  items.parquet     one row per unique item_id, sorted by item_id
  events.parquet    one row per interaction event, sorted by (event_time, event_id)
```

The conversion verifies these invariants before writing final tables:

- Every event row contains non-null `item_id`, `user_id`, `stime`, `session_id`, `sequence_id`, and `event_id`.
- Every event references an `item_id` present in `items`.
- `items.item_id` is unique.
- Non-null prices are non-negative.

## Tables

`items`, sorted by `item_id`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `item_id` | Int64, not null | Source key, primary key |
| `product_id` | Utf8 | Source product taxonomy cluster id |
| `name` | Utf8 | Source item title |
| `price` | Float64 | Numeric price in USD |
| `c0_id` | Int32 | Root category identifier |
| `c0_name` | Utf8 | Root category name (e.g. Women, Men, Electronics) |
| `c1_id` | Int32 | Second-level category identifier |
| `c1_name` | Utf8 | Second-level category name |
| `c2_id` | Int32 | Leaf category identifier |
| `c2_name` | Utf8 | Leaf category name |
| `category_path` | Utf8 | `c0_name > c1_name > c2_name` |
| `brand_id` | Int32 | Brand identifier |
| `brand_name` | Utf8 | Brand name |
| `item_condition_id` | Int16 | Item condition rating (1 to 5) |
| `item_condition_name` | Utf8 | Condition label (New, Like New, Good, Fair, Poor) |
| `size_id` | Int32 | Item size identifier |
| `size_name` | Utf8 | Size label |
| `color` | Utf8 | Item color |
| `shipper_id` | Int16 | Shipping option identifier (1 or 2) |
| `shipper_name` | Utf8 | Shipping payer (Buyer or Seller) |

`events`, sorted by `(event_time, event_id)`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `event_id` | Int64, not null | 0-based dense event ordinal |
| `event_time` | Timestamp(ms), not null | Source `stime` as UTC wall time without timezone |
| `user_id` | Int64, not null | User identifier |
| `item_id` | Int64, not null | Joins `items.item_id` |
| `session_id` | Utf8, not null | Browsing session identifier |
| `sequence_id` | Utf8, not null | User session sequence identifier |
| `sequence_length` | Int32, not null | Total actions recorded in the sequence |
| `event_type` | Utf8, not null | Action type (item_view, item_like, item_add_to_cart_tap, offer_make, buy_start, buy_comp) |

## Parquet layout for the GPU engine

- **Types.** All columns use data types supported by the native GPU scan.
  Timestamps are timezone-naive milliseconds. Strings are dictionary-encoded on
  low-cardinality attributes (`event_type`, category names, condition, shipper).
- **Integer keys.** `item_id` and `user_id` are 64-bit integers. Joins,
  aggregations, and cuGraph graph construction operate directly on integer keys
  without string hashing.
- **Pruning.** `events` is physically ordered by `(event_time, event_id)`,
  allowing row-group min/max statistics to prune date and time ranges. `items` is
  ordered by `item_id` for key-range pruning on joins.
- **Bloom filters.** Configured on `user_id`, `item_id`, `session_id`, and
  `sequence_id` for `events`, and on `item_id`, `brand_name`, and `c0_name` for
  `items`.
- **Row groups.** 1,048,576 rows per row group on `events`, and 131,072 rows per
  row group on `items`.

## Native query examples

```sql
SELECT i.c0_name, e.event_type, count(*) AS event_count, avg(i.price) AS avg_item_price
FROM events e JOIN items i ON e.item_id = i.item_id
WHERE e.event_time >= TIMESTAMP '2023-10-01 00:00:00'
GROUP BY i.c0_name, e.event_type
ORDER BY event_count DESC;

SELECT session_id, count(*) AS session_events, min(event_time) AS start_time, max(event_time) AS end_time
FROM events
GROUP BY session_id
ORDER BY session_events DESC
LIMIT 10;

SELECT user_id, item_id, count(*) AS interaction_weight
FROM events
WHERE event_type IN ('item_like', 'buy_comp')
GROUP BY user_id, item_id;
```
