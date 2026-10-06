# Amazon Reviews scale demo

The Amazon Reviews 2023 Electronics dataset contains 43,886,944 reviews and 1,610,012 products.
The product catalog has 1,609,918 entries with non-empty titles (94 null or empty).
Review records link to catalog items via `product_id`.

## Data flow, cohorts, and query scales

The evaluation comprises five distinct analytical queries that combine relational operations
with GPU algorithm UDTFs (cuVS KNN, cuVS KMeans, cuVS Top-K, and cuML UMAP).
The table below specifies the exact row count at each pipeline stage:

| Query | Source Tables | Initial Filter / Aggregation | Prepared Embedding Filter | UDTF Input Size | Result Rows | Execution Path | Host Boundaries |
|---|---|---|---|---|---:|---|---:|
| `01_catalog_batch.sql` | `products` | 1,010,000 products embedded | `sample_rank < 1000000` (corpus), `[1M, 1.01M)` (queries) | 1,000,000 corpus, 10,000 queries | 100,000 | native | 0 |
| `02_travel_shortlist.sql` | `reviews`, `products` | 2022 verified reviews (>=100 rev, <=20% low rating), $20-$150, Headphones: 179 products | `sample_rank < 1000000`: 117 products | 117 products against 1 search intent vector | 10 | native | 0 |
| `03_headphone_emerging.sql` | `reviews`, `products` | 2021-2022 rating <=2, text >=40 chars: 1,844,729 total (183,242 Headphones & Earbuds) | 1M review sample: 99,539 headphone reviews | 99,539 vectors into 24 clusters; 155 products qualify | 10 | native | 0 |
| `04_inline_kmeans_topk.sql` | `reviews`, `products` | 2022 rating <=2, Headphones: 139,606 reviews -> grouped by product (>=10 rev): 2,384 products | None (reads raw Parquet text directly) | 2,384 products into 6 clusters; Top-3 features per cluster | 18 | native | 0 |
| `05_headphone_umap.sql` | `reviews`, `products` | 99,539 headphone reviews | `sample_rank < 100000` (10% slice): 9,967 reviews | 9,967 vectors into 2D UMAP | 9,967 | native | 0 |

### Cohort distinction: business logic vs artificial sampling

1. **`02_travel_shortlist.sql` candidate pool**:
   The business filter across the full 43.9M reviews and 1.6M products yields 179 products meeting the review
   history thresholds (at least 100 verified reviews in 2022 with at most 20% negative ratings) and catalog
   constraints (category `Headphones & Earbuds`, price between $20 and $150).
   The prepared sample `sample_rank < 1000000` contains 117 of these 179 products because 62 eligible products
   fell outside the 1.01M product embedding sample.

2. **`03_headphone_emerging.sql` defect topic shift**:
   The total 2021-2022 corpus contains 1,844,729 low-rating reviews (rating <= 2, text length >= 40 chars)
   across all categories, of which 183,242 are in `Headphones & Earbuds`.
   The prepared 1.0M low-rating embedding sample captures 99,539 of these headphone reviews.
   cuVS KMeans groups these 99,539 vectors into 24 topic clusters.
   Subsequent relational aggregation and window functions calculate annual totals per product; 155 products
   meet the qualification thresholds (at least 30 low-rating reviews in both 2021 and 2022), from which the
   top 10 products with the largest increase in complaint cluster share are returned.

3. **`04_inline_kmeans_topk.sql` keyword complaint mining**:
   This query does not depend on precomputed embeddings. It directly scans the base `reviews` and `products`
   tables, evaluates regular expressions and substring mentions across 139,606 negative reviews in 2022,
   groups them into 2,384 products with at least 10 negative reviews, fits 6 KMeans clusters over 8 complaint
   dimensions, and extracts the top 3 complaint categories per cluster using cuVS Top-K.
   The 2,384 count is an exact business aggregation over the full catalog, not an artificial preparation slice.

## Native execution path: elimination of host boundaries

Under earlier server configurations, `03_headphone_emerging.sql` incurred 20 DataFusion host boundaries
(`mixed` execution) because DataFusion inserted a multi-partition `SortExec { preserve_partitioning: true }`
beneath `WindowAggExec`, and default SQL whole-partition window frames used `WindowFrameUnits::Range`.
Both patterns were unsupported by native lowering, triggering `NativePartitionPreservingSortUnsupported`.

Following the normalization and lowering extensions in `crates/algeon-datafusion`:
- Input normalization under `WindowAggExec` and `BoundedWindowAggExec` permits partition-preserving sorts.
- Whole-partition window frame lowering recognizes both `Rows` and `Range` unbounded boundary specifications.

As a result, all five demo queries lower into pure native GPU fragments and execute with **zero host boundaries**
under `execution_mode = "native_required"`.

## Server configuration

The evaluation runs on an NVIDIA RTX PRO 6000 Blackwell (96 GB device budget, 90 GiB query grant,
1 GiB backend reserve):

```toml
[[admission.device_profiles]]
device_ordinal = 0
memory_limit_bytes = 100931731456
backend_reserve_bytes = 1073741824
cache_cap_bytes = 0
default_query_limit_bytes = 96636764160
max_active_attempts = 1
source_work_per_attempt = 2
stream_pool_size = 8
worker_count = 4
lane_count = 4

[native]
execution_mode = "native_required"
```

## Running the demo

1. Build the release server binary:

```bash
cargo build --workspace --all-features --release --bin algeon_server
```

2. Start the server with `native_required`:

```bash
ALGEON_SERVER_CONFIG_FILE="$PWD/fixture/demo/amazon_reviews_2023/scripts/scale/server.toml" \
ALGEON_SERVER_EXECUTION_MODE=native_required \
ALGEON_SERVER_BIND=127.0.0.1:50062 \
flock -x /tmp/cudf-gpu.lock target/release/algeon_server
```

3. Register sources and execute queries:

```bash
python3 fixture/demo/amazon_reviews_2023/scripts/amazon_reviews_semantic.py --profile bge-m3 run \
  --register \
  --names catalog_batch travel_shortlist headphone_emerging inline_kmeans_topk headphone_umap2 \
  --repeats 3
```
