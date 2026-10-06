# MerRec scale demo

The formal dataset uses all six source months: 1,268,092,738 events across
2,170 shards, totaling 166,075,974,430 compressed bytes. The October validation
dataset has 247,882,433 events. Acquisition and conversion commands live in the
[fixture README](../../README.md).

The saved browsing visualizations describe the original 494,357-event source shard.
Their graph has 1,711 market vertices; their feature table has 1,919 users.
Scale runs write separate observations and leave those captured results intact.
Stop the server before replacing a dataset and start a fresh server for each
dataset scope, so registered schemas and Parquet metadata match its files.

## Server and measurements

[`server.toml`](server.toml) configures a 94 GiB device ledger and
a 90 GiB query grant on the development host's 96 GB GPU, with one active
attempt and no resident cache. Smaller devices need smaller service limits.
Source batches use the query grant and decoded working-set observations by
default. An explicit `native.max_source_chunk_bytes` still limits batching.
The allocator enforces the grant; a larger limit permits larger working sets
without requiring every query to allocate the entire limit. Compressed file
size does not determine allocation demand.

Build a release server for performance measurements:

```bash
cargo build --workspace --all-features --release --bin algeon_server

ALGEON_SERVER_CONFIG_FILE="$PWD/fixture/demo/merrec/scripts/scale/server.toml" \
ALGEON_SERVER_EXECUTION_MODE=native_required \
ALGEON_SERVER_BIND=127.0.0.1:50064 \
flock -x /tmp/cudf-gpu.lock target/release/algeon_server
```

The relational performance lane requires complete native execution. Run the
category join and user activity queries against the current dataset:

```bash
MERREC_ROOT=/mnt/algeon-nvme/algeon/fixture/demo/merrec
MERREC_REPORT="$PWD/target/merrec-scale/full/gpu"
MERREC_RESULTS="$MERREC_ROOT/scale-results/gpu"

python3 scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 --register \
  --data-dir "$MERREC_ROOT/parquet" \
  --report-dir "$MERREC_REPORT" --results-dir "$MERREC_RESULTS" \
  --timeout 1800 --repeats 5 \
  fixture/demo/merrec/scripts/scale/01_category_engagement.sql \
  fixture/demo/merrec/scripts/scale/02_user_activity.sql
```

For the CPU SQL baseline, stop that server and restart it with
`ALGEON_SERVER_EXECUTION_MODE=functions_only`. Run the identical SQL and input
files with separate `full/cpu` report and result directories. This mode keeps
ordinary relational SQL in DataFusion. Explicit cuGraph and cuVS functions
remain GPU functions, and are excluded from this CPU SQL comparison.

Every run retains Flight elapsed time, query statistics, an `EXPLAIN GPU`
coverage report, and complete results. The input manifest is copied from the
registered source directory after checking the table locations. Compare the
exact integer result rows between the two modes.
Report the first run separately from the median of subsequent runs, including
the CPU worker configuration and release binary checksum. These timings include
planning and complete result receipt; conversion and CSV export are excluded.
The first run may already read files cached by conversion. A filesystem cold
read requires a separately controlled measurement.

The validated six-month run used a release binary and cached source files.
The CPU baseline used 32 DataFusion target partitions. Native execution used
four source lanes and two source work items in flight. Both queries returned
identical CPU and GPU rows in all five runs. Medians below exclude the first run:

| Query | CPU | Native GPU |
|---|---:|---:|
| Category engagement join | 19.27 s | 5.16 s |
| User activity aggregation | 4.46 s | 2.15 s |

The former 64 MiB source target took 17.14 s for user activity. Grant-derived
batching and incremental integer-key aggregation reduced partial aggregations
from 11,072 to 256, and cumulative partial rows from 813,804,386 to 20,520,946.
Both measured queries used zero host boundaries and zero retries. Query
allocator peaks were 48.40 GiB for category engagement and 24.94 GiB for user
activity. Raw reports, binary checksums, and result comparisons for this run
are under `target/merrec-implementation/`.

## Item-user PageRank

[`04_item_pagerank.sql`](04_item_pagerank.sql) uses the existing item-user
experiment without changing its SQL. It filters `item_view` events, aggregates
repeated `(item_id, user_id)` interactions into edge weights, and maps users to
`-(user_id + 1)` to separate them from non-negative item IDs. The undirected
graph feeds PageRank; a result filter and Top-K return 20 item vertices.

```bash
python3 scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 \
  --report-dir "$MERREC_REPORT" --results-dir "$MERREC_RESULTS" \
  --timeout 1800 --repeats 5 \
  fixture/demo/merrec/scripts/scale/04_item_pagerank.sql
```

Native negation support keeps the edge projection in the same GPU execution
as its filter and aggregation. `native_required` checks the complete statement,
including relational operations around cuGraph. The query computes its graph
input from the registered events and needs no saved graph snapshot.

Report source events, normalized graph edges, and graph vertices separately.
Flight latency includes relational preparation, graph construction, PageRank,
and result receipt. The graph runtime row measures graph construction and
algorithm execution. It excludes the preceding event aggregation and edge
normalization, and does not measure the complete exploration latency.

The complete six-month input produces 1,682,503,472 normalized directed edges
for the undirected graph and 87,055,073 vertices. All five runs of the unchanged
statement completed within the 90 GiB grant. The first run took 19.24 s;
the median of the remaining four was 19.31 s. Each run had a query allocator
peak of 89.07 GiB, two native fragments, zero host boundaries, and zero retries.
PageRank converged after 75 iterations. No additional time or cohort filter
was needed.

The earlier allocation failures included redundant sort-key and edge-table
copies, an unused full-size native property buffer, and oversized weighted
sort workspace. Prepared-order deduplication now gathers the retained payload
once. Input owners retire after their last required read, and native weighted
construction uses the existing four-part sorting approach. PageRank also skips
the vertex-count probe used for betweenness seed policies; concatenating both
endpoint columns for that probe exceeded cuDF's row-count limit at this scale.

The runner saves the existing runtime artifact as `*.runtime.jsonl.gz`.
The footer's `data.attempt_report.final_domain_usage` records the query
allocator peak. GPU process samples from `nvidia-smi` include other allocation
overheads and are recorded separately when used. The two cuGraph host staging
limits remain finite; GPU-resident graph boundaries avoid host staging of the
complete edge and vertex tables.

## User vectors and batched search

`03_user_features.sql` joins all selected events to items and retains users with
at least 20 view events. Its 17 category-share features use the complete selected
time range. The original research's October 21 cutoff remains in
`prepare_tastes.sql` for the captured temporal experiment.

```bash
python3 scripts/dev/merrec_run.py --uri grpc://127.0.0.1:50064 --prepare-scale \
  --data-dir "$MERREC_ROOT/parquet" \
  --report-dir "$MERREC_REPORT" --results-dir "$MERREC_RESULTS" \
  --timeout 1800 --repeats 5 \
  fixture/demo/merrec/scripts/scale/06_batch_exact.sql \
  fixture/demo/merrec/scripts/scale/05_batch_cagra.sql
```

The CAGRA query builds an index over the complete feature table and searches the
first 10,000 users by ID in one batch, returning 10 neighbors each. Queries are
also members of the indexed dataset; self matches are allowed. Each SQL
invocation builds a fresh query-local index. Its elapsed time includes index
construction and search, and must be labeled accordingly.
`06_batch_exact.sql` searches the identical dataset and query batch with exact
brute-force kNN with no index-build phase. Compare its latency with CAGRA's
build-and-search latency before choosing an ANN method for these 17 features.
ANN comparisons need measured recall before publishing a
speedup. Category-share vectors can have equal distances, including identical
profiles; report ID overlap together with a check against each query's exact
kth-neighbor distance to distinguish ties from farther approximate neighbors.

The earlier vector measurement used a full feature table of 2,789,275 users.
The 10,000-query batch took
0.29 s for exact search and 14.23 s for CAGRA build and search (medians of the
last two of three runs). Use exact search as the vector demonstration for this
dataset. CAGRA's neighbor ID overlap with exact search was 94.165%; 98.603% of
its neighbors were within the exact kth-neighbor distance plus a squared-L2
tolerance of `1e-6`. Equal-distance profiles account for some ID differences.
