# Sort-lane sweep (plan F)

Entry: `crates/bench/src/bin/sort_lane_sweep.rs` (`bench::sort_lane_sweep`).
Method fixes the workload and the comparison contract; the numbers below are
one machine's observation, not a pooled speedup claim.

## Identity

- Algeon source: `f7fff63c29d0b6b694ff3b9928377595a17629ee` plus the F working
  tree (uncommitted at measurement time; tables below come from the final
  code state after review fixes).
- Native submodules: cudf `00e395cf1a`, cugraph `40119e4cb`, cuvs `25b1be43a`.
- GPU: NVIDIA GeForce RTX 5070 Ti, 16303 MiB. Driver 595.91.07.
  CUDA toolkit `/usr/local/cuda` (13.2). Build profile: debug (`dev`).
- Nsight Systems 2025.6.3 for the overlap trace.

## Workload

Two generated Parquet branches (`a`, `b`), fixed LCG seeds
`0x9E3779B97F4A7C15` / `0xBF58476D1CE4E5B9`, Int64 key with sparse nulls
(every 97th row) and a key-derived payload. Heavy query:

```sql
SELECT k, v FROM a UNION ALL SELECT k, v FROM b ORDER BY k DESC LIMIT 100
```

Light query for the mixed section:

```sql
SELECT COUNT(*) AS n, SUM(v) AS s FROM a WHERE k > 0
```

Widths run as separate backends with `source_work_per_attempt` 1, 2, and 4
(`max_active_attempts` 1). The profile width also sizes source concurrency,
but union reads finish before the sort merge begins, which keeps the width
comparison on the sort phase. Every cell asserts `Native` disposition and an
order-sensitive output fingerprint; any drift fails the run. Query cap 2 GB,
device capacity 8 GB, reps 10 after 1 warmup.

## Single-query results

Rows per branch 2M (66 MB total), wall ms:

| width | median | p95 | stddev | failures | fingerprint |
| --- | --- | --- | --- | --- | --- |
| 1 | 29.1 | 32.3 | 1.2 | 0 | `a92cf69c06743199:100` |
| 2 | 31.3 | 32.5 | 1.1 | 0 | match |
| 4 | 30.7 | 34.5 | 1.3 | 0 | match |

Rows per branch 8M (264 MB total), wall ms:

| width | median | p95 | stddev | failures | fingerprint |
| --- | --- | --- | --- | --- | --- |
| 1 | 90.5 | 98.7 | 2.5 | 0 | `a92cf69c06743199:100` |
| 2 | 98.9 | 100.3 | 3.9 | 0 | match |
| 4 | 91.4 | 93.4 | 1.1 | 0 | match |

Run-to-run variance on this shared machine reaches ~10% (standalone width-2
reruns measured 91.5 and 103.5 at 8M). Widths inside one invocation agree
within that envelope; no width shows a systematic advantage at these
shapes.

Raw per-rep rows live in the run TSV (`section width rep wall_ms gpu_exec_ms
rows fingerprint disposition outcome`). The heavy top-100 is null-led on both
datasets (null keys sort first under the sweep order), while each candidate
still sorts its full branch before truncation.

## Overlap evidence

Nsight trace of the 2M width-2 run, merge-sort kernels by stream:

- Streams 17/18 (one query's two candidates): busy 10.12 / 9.04 ms,
  overlapped 5.64 ms.
- Streams 42/43 (a later query's two candidates): busy 8.15 / 7.52 ms,
  overlapped 2.66 ms.
- No overlap across queries, as expected for sequential attempts.

The two candidates of one top-k run on disjoint fork subsets at the same
time. The deterministic GPU rendezvous test
(`lane_dispatch_sort_topk_candidates_overlap_lanes_and_match_serial`) proves
the same property without a profiler: both candidates in execution
simultaneously on distinct lanes and forks, one commit each.

## Mixed-load results

Mixed data (generated, fixed seeds): 8192-vertex ring-plus-shortcut edge
list (16384 edges) with float weights; 8192 two-dimensional grid vectors
plus 256 near-grid query vectors. Backend `max_active_attempts` 4, profile
width 2. Each of 5 rounds launches
the heavy sort, the light aggregate, a cuGraph PageRank aggregate, and a
cuVS brute-force KNN aggregate concurrently; every attempt must succeed with
a stable fingerprint or the run fails. Wall median / p95 ms with queue-wait
median / p95 ms:

2M mix:

| item | wall median | wall p95 | queue median | queue p95 | disposition |
| --- | --- | --- | --- | --- | --- |
| heavy sort | 32.0 | 32.7 | 0.2 | 0.2 | Native |
| light aggregate | 17.3 | 18.1 | 0.1 | 0.1 | Native |
| vector KNN | 17.4 | 18.5 | 0.1 | 0.3 | PartialNative |
| graph PageRank | 21.0 | 21.6 | 0.2 | 0.2 | Native |

8M mix:

| item | wall median | wall p95 | queue median | queue p95 | disposition |
| --- | --- | --- | --- | --- | --- |
| heavy sort | 106.7 | 108.0 | 0.2 | 0.2 | Native |
| light aggregate | 33.7 | 36.5 | 0.1 | 0.2 | Native |
| vector KNN | 20.0 | 21.4 | 0.2 | 0.3 | PartialNative |
| graph PageRank | 23.5 | 24.6 | 0.2 | 0.2 | Native |

Fingerprints hold across all 5 rounds per item (heavy `a92cf69c06743199:100`,
graph `360438c4017dd3aa:1`, vector `3888472016677431:1`). The vector item
reports `PartialNative` (cuVS island composes with CPU operators); the rest
are `Native`. Every attempt seals `Succeeded`, cleanup `Completed`, and
`final_domain_usage` `current_bytes` 0: no starvation, no residual attempt.
Domain peaks: 2M heavy 146000718 B; 8M heavy 584000334 B.

Cancel probe: a heavy attempt cancelled 10 ms after launch reaches
`Cancelled` with cleanup `Completed` and zero residual charge. Measured
cancel latency (cancel call to terminal report): 1.1 ms at 2M, 3.6 ms at
8M.

## Reading the numbers

Overlap without end-to-end gain. Widths agree within run-to-run variance;
the 2M width-4 median sits slightly above width 1 (four-lane dispatch for
two candidates: overhead without benefit). Parquet decode dominates kernel time
(decode kernels ~25% of the kernel summary against ~50% merge-sort), so
halving candidate sort time moves a fraction of the query. The API trace of
the same run shows 1489 `cudaStreamSynchronize` calls (~70 ms total): host
synchronization around source reads and D2H leaves little room for sort
overlap to change the wall time. Int64 single-key sorts at these sizes are
memory-bandwidth bound, so two concurrent sorts share the bus while the
makespan does not shrink.

The mechanism stage is complete; the performance stage is not claimed. The
revisit precondition is a profile where un-limited or sort-dominated work
holds lanes idle: final-sort time above roughly 20% of attempt wall time with
concurrent lanes free, on wider rows or multi-key string sorts where overlap
buys makespan.

## Sirius gap table

Same-workload comparison against Sirius on identical Parquet bytes with disk
spill off, per the baseline comparison rules.

| query | Algeon width 1 | Algeon width 2/4 | Sirius | root cause or status |
| --- | --- | --- | --- | --- |
| union sort/top-k 2M | 29.1 ms median | no gain (see above) | unavailable | no Sirius binary in this environment; listed, not estimated |
| union sort/top-k 8M | 90.5 ms median | no gain (see above) | unavailable | same |
| full ORDER BY without limit | serial by design | serial by design | unavailable | range-partitioned parallel full sort deferred; precondition above |

No per-query conclusion about Sirius follows from this table. A single
geomean across it would hide the per-shape result and is not reported.

## Known non-F issues met during this work

- `bounded_native_query_uses_hierarchical_topk_without_full_sort_first`
  fails on the clean HEAD tree (`f7fff63c`) as well as with this change: an
  aggregate-fed sort arrives as a frame stream while the test asserts
  frame-slice budget boundaries. Pre-existing; fixing aggregate output
  forms is outside F scope.
- Direct test-binary runs (outside nextest) loaded a stale prebuilt
  `libcudf_rust.so` whose fork-route ABI predated the working tree, which
  surfaced as duplicate fork claims. Cargo-driven flows rebuild the shim;
  non-test GPU commands must run from a fresh build.
