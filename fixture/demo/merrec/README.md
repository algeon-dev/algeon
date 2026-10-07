# MerRec (Mercari C2C Recommendation)

A relational e-commerce and interaction graph demo dataset derived from the
Mercari C2C marketplace dataset. It exercises temporal filtering, high-cardinality
group-bys, hash joins, and bipartite graph queries on the native GPU engine.

## Source

[mercari-us/merrec](https://huggingface.co/datasets/mercari-us/merrec),
described in the KDD 2025 paper *MerRec: A Large-scale Multipurpose Mercari Dataset
for Consumer-to-Consumer Recommendation Systems* (Lichi Li, Zainul Abi Din,
Zhen Tan, Sam London, Tianlong Chen, Ajay Daptardar).

The pinned source revision is `dd2b3c6f6f19d4814bb839ee6e8ab41926a14343`.
[`scripts/source_files.json`](scripts/source_files.json) records the byte size and
SHA-256 of all 2,170 shards across six monthly partitions. Their total compressed
size is 166,075,974,430 bytes. October has 409 shards and 247,882,433 events;
the six-month dataset has 1,268,092,738 events. Full conversion retains those
events and produces 82,614,392 unique items and 5,566,055 users. Item attributes
are stored once per item; normalized file sizes can be smaller than the raw
event shards without removing events.

## Acquisition and conversion

The default scope is the complete October partition. Missing shards cause an
error. The converter never substitutes `raw/sample/` for a monthly partition.
`--download` retrieves missing or wrong-size shards at the pinned revision,
checks downloaded bytes against the source inventory, and reuses complete files.

On the development host, put downloads, final tables, and DuckDB scratch on NVMe:

```bash
MERREC_ROOT=/mnt/algeon-nvme/algeon/fixture/demo/merrec

# Validate the conversion with 19,022,480 events from 32 complete shards.
fixture/fixture.sh merrec prepare --download --shards 32 \
  --raw-dir "$MERREC_ROOT/raw" --out-dir "$MERREC_ROOT/validation-32" \
  --duckdb-memory-limit 32GB --duckdb-threads 8

# Convert all 247,882,433 October events.
fixture/fixture.sh merrec prepare --download \
  --out-dir "$MERREC_ROOT" --duckdb-memory-limit 32GB --duckdb-threads 8 --force

# The formal demo dataset contains all six months, without a row limit.
fixture/fixture.sh merrec prepare --partition all --download --download-workers 32 \
  --out-dir "$MERREC_ROOT" --duckdb-memory-limit 32GB --duckdb-threads 8 --force
```

The 32GB DuckDB setting limits its managed host memory on this development host;
smaller hosts should set their own limit. DuckDB spills sorts and item
partitioning into `work/`. Item attributes are partitioned by a hash of
`item_id` into 64 groups before latest-row aggregation. Equal IDs from different
months share a group. Conversion preserves all selected events and uses NVMe
for partition files and external sorts; process RSS can exceed DuckDB's managed
memory limit. Events are sorted once; the streaming Parquet writer assigns
dense event IDs across row groups after that sort. The GPU query grant is configured separately by the
server. Arrow reads one source row group at a time, then assembles the configured
output row groups; decoded dictionary buffers do not accumulate across the file.
Generated paths in the repository point to the NVMe dataset through
`parquet/` and `manifest.json` symlinks. With those links, default download and
scratch directories follow the physical output location as well.

A sample requires an explicit command and a separate destination:

```bash
fixture/fixture.sh merrec prepare --download-sample \
  --raw-dir "$MERREC_ROOT/raw" --out-dir "$MERREC_ROOT/sample"
```

`--input-parquet` accepts a file, directory, or glob for caller-selected sources.
That scope is recorded as `explicit`; it carries no claim of upstream partition
completeness. `--shards N` selects the first N complete shards of one month.
`--limit N` is available only for an explicitly selected subset. Full monthly
and six-month conversions retain every source event.

The converter requires the `duckdb` CLI and `pyarrow`. DuckDB deduplicates item
attributes to the latest timestamp across the selected sources and sorts the
intermediates using external-memory spilling. PyArrow streams the results into
Snappy-compressed format 2.6 Parquet. Existing tables remain available until both
replacement tables pass validation; stop serving queries before installing a
new dataset because the table files are replaced individually.

Options:

- `--partition NAME|all`: Complete monthly partition, or all six months (default: `20231001`).
- `--download`: Acquire missing shards for the selected scope.
- `--download-workers N`: Concurrent downloads (default: 8).
- `--download-sample`: Select and acquire the first shard of one month.
- `--shards N`: Select the first N complete shards of one month.
- `--raw-dir PATH`: Source directory (default: beside the physical `parquet/` directory).
- `--input-parquet PATH`: Explicit Parquet file, directory, or glob pattern.
- `--out-dir PATH`: Directory receiving `parquet/` and `manifest.json`.
- `--work-dir PATH`: DuckDB scratch parent (default: beside the physical `parquet/` directory).
- `--duckdb-memory-limit SIZE`, `--duckdb-threads N`: Host conversion resources.
- `--event-row-group-rows N`: Event row group size (default: 1,048,576).
- `--item-row-group-rows N`: Item row group size (default: 131,072).
- `--limit N`: Input row limit for explicit subsets.
- `--skip-sha256`: Omit source hashing on conversion; downloads still verify hashes.
- `--keep-work`: Retain conversion intermediates.
- `--force`: Replace existing tables and manifest after validation.

Output layout:

```
manifest.json       source revision, selection, input rows, counts, and layouts
parquet/
  items.parquet     one row per unique item_id, sorted by item_id
  events.parquet    one row per interaction event, sorted by (event_time, event_id)
```

The conversion verifies non-null event keys and action types, non-negative
prices, and event references to the item table. Output event count must equal
the source footer row-count sum, or its explicitly requested row limit. The
manifest records the selected scope and each source file's SHA-256.

The item table represents the latest item state within the selected scope.
Temporal model training must use attributes available at the feature timestamp;
the existing supervised experiment reads raw events for that purpose.

The [scale demo workflow](scripts/scale/README.md) records server configuration,
CPU comparisons, batched vector inputs, and the observed graph staging limit.

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

`session_id` is not unique per user: in the six-month dataset, 15,138,756 of
the 25,864,474 distinct values appear for more than one user, and one value
appears for 54 users. Key sessions by `(user_id, session_id)`.

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

SELECT user_id, session_id, count(*) AS session_events, min(event_time) AS start_time, max(event_time) AS end_time
FROM events
GROUP BY user_id, session_id
ORDER BY session_events DESC
LIMIT 10;

SELECT user_id, item_id, count(*) AS interaction_weight
FROM events
WHERE event_type IN ('item_like', 'buy_comp')
GROUP BY user_id, item_id;
```
