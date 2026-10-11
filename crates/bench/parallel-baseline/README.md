# GPU query execution baseline (A)

This record fixes the workload and measurement contract for phase A of the
[parallel execution plan](../../../docs/plan/2026-09-19-gpu-query-engine-parallel-execution-plan.md).
The baseline uses the current serial operator execution. Join plans use the
compiled pipeline; scan/aggregate plans can use the bounded aggregate path
without a compiled pipeline artifact. Operator width is **one and is not
configurable**. `ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT`
changes source work, not operator width. The runtime profile's stream pool also
includes root and source streams; its size is not a count of parallel operator
tasks.

The current capability-first compiled admission is entered for join/vector
plans. Q01/Q06 and the pure independent-union control can instead run through
the existing aggregate/generic path. B/C must bring these supported paths into
the same compiled execution model before scheduler parallelism can improve
them; their current native success is not compiled-scheduler coverage.

The TPCH entry wires `ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT`
into the source scheduler and the backend's source-work/derived stream-pool
bounds. Its native profile controls maximum source chunk bytes, row groups per
chunk, retry bounds, and source-chunk planning mode. This run resolved the
conservative profile to 512 MiB chunks, one row group per chunk, and at most
two source reads in flight. `query_service_goodput` separately controls active
attempts and service lanes. Neither entry exposes parallel operator-task
width; the trace's extra source/fork streams therefore do not prove operator
parallelism.

## Workload matrix

Use the same Parquet bytes and GPU for every comparable run. Keep unsupported,
fallback, error, and timeout rows in the results. Query labels are benchmark
identifiers and have no role in native planning.

| Case | Purpose | Existing entry |
| --- | --- | --- |
| TPC-H Q01, Q06, Q15 | scan/filter/partial and final aggregate; Q15 also reuses a CTE | `native_tpch_benchmark` |
| TPC-H Q17, Q18 | join with aggregate and correlated/shared input | `native_tpch_benchmark` |
| TPC-H Q03 | join and ordered top-k | `native_tpch_benchmark` |
| TPC-DS Q39, Q78 | broader aggregate/join and sort/shape coverage | `native_tpcds_parity_report` |
| Independent branches followed by union | branch scheduling control | `query_service_goodput` `adhoc.independent_union` |
| Repeated CTE | reuse/fanout control | TPC-H Q15 plan plus `query_service_goodput` `adhoc.repeated_cte` |
| Variable-width/high-cardinality group-by | memory/retained-output stress | `query_service_goodput` `adhoc.variable_width_group` |
| PageRank | graph smoke and mixed service load | `query_service_goodput` mix |
| cuVS brute-force KNN | vector smoke | existing cuVS GPU contract test; benchmark exposure pending |

Q01/Q06 and Q17 are the representative scan+aggregate and join profiler cases.
If planning rejects a case, retain its structured reason and report no native
latency. A successful CPU run alone is not native coverage.

## Comparison rules

Phase A records this source revision's serial baseline and fixes the comparison
method. After B/C makes operator width executable, record the modified executor
at width 1, 2, and 4; those later cells cannot be manufactured by raising
source-read concurrency.
Use the same dataset fingerprint and cap, and alternate baseline/candidate
ordering across runs. A width-1 regression is a regression even if width 4
improves. Compare Sirius on the identical Parquet bytes with disk spill off.
If its host cache or spill behavior cannot match, label a separate mode and
do not calculate a pooled speedup. Record every Sirius CPU fallback and
unsupported query.

For each cell, retain at least one warmup followed by ten timing samples,
median, p95, variance, failure count, parity, native coverage, and allocator
peak. Track cold-source and warm-source separately, and keep profiler runs
separate from unprofiled timing. Set the improvement and control cohorts before
looking at width results: Q01/Q06/Q15 and Q17/Q18 are benefit cases; Q03,
TPC-DS Q39/Q78, graph/vector smoke and service goodput are controls. Use the
baseline's measured variation to set the noise margin; the plan's greater
than 5% unexplained control regression threshold remains in force.

## Timing semantics

`native_tpch_benchmark` reports native execution wall time,
capability/compiled-graph coverage, source-read and operator elapsed, D2H
materialization count/bytes, and output fingerprints. Its
`allocator_high_water_bytes` may be absent; `max_task_observed_peak_bytes` is
a per-task observation and cannot substitute for the attempt domain peak.
The benchmark replans before each measured execution, so its stable
`native_wall_ms` excludes physical planning. Set
`ALGEON_TPCH_BENCH_ARTIFACT_DIR` to retain the first measured execution's
physical plan and sealed runtime observation, plus ten per-iteration planning, execution,
planning-plus-execution, and full benchmark-iteration wall samples. The
planning-plus-execution field ends before output fingerprint work; the
iteration-wall field ends after fingerprinting. Neither includes an external
client/network boundary. All six retained planning rows report zero host
DataFusion boundaries, so host child staging is absent on these native paths.
Operator/source duration metrics are host-side elapsed observations and may
overlap; they are not GPU kernel duration and must not be added to derive
query wall time. The stable TSV's `materialization_time_ms` is a native host
sink timer, not a GPU D2H copy interval. The timing boundaries are:

| Boundary | Evidence here | Limit |
| --- | --- | --- |
| Physical planning | Per-iteration `native_planning_ms` sidecar | Excluded from `native_wall_ms` |
| Native execution wall | TPCH `native_execution_ms` sidecar and stable TSV | Includes host orchestration and output materialization; not GPU busy time |
| GPU kernel/copy activity | Q01/Q17 Nsight kernel and direction-tagged copy intervals | Profiled runs only; interval unions are not query wall time |
| Host child staging | Six retained plans have zero host DataFusion boundaries | No nonzero staging sample in this workload |
| D2H | Stable TSV counts/bytes, host sink timer, and profiled CUPTI D2H intervals | No isolated elapsed host D2H stage for all ten samples |
| Full benchmark iteration | `native_iteration_wall_ms` sidecar | Ends after fingerprinting; excludes client/network |
| Service end-to-end | `goodput-ten` attempt rows | Scheduled request to terminal outcome, not isolated TPCH timing |

The observation JSONL is produced by the executed path. It contains actual
compiled task/edge/barrier/sink topology when compiled admission ran, or the
aggregate/generic path's topology when it did not. The files are written after
the measured timing intervals, once the backend seals the observation; no
diagnostic recompilation is substituted for the executed schedule.

An Nsight Systems trace supplies GPU kernel/copy intervals and host launch
gaps. Do not infer GPU utilization from a single kernel's occupancy.

The run manifest must retain the exact safe command/config, root and native
submodule identities and diffs, binary/build profile, GPU UUID, driver, CUDA
toolkit, query cap and cache share, source layout and row groups, CPU thread
settings, cache state, dataset fingerprint, per-query TSV, and trace paths.
Missing facts remain `unavailable`; they must not be substituted with an
estimate.

## Recorded serial baseline, 2026-09-19

The local artifacts are under `target/bench/parallel-a-2026-09-19/`. This was
a dev-profile build on one RTX 5070 Ti (GPU UUID
`GPU-18cc6584-d0bb-bf0c-56c4-50531dd340bb`, driver 595.91.07). The
source was root `550cc68d68e89a110422ff4ac5c99ce8ba3c6ca6` with a dirty
worktree; native submodules were cuDF `00e395cf1aa09f68a97eb2a7547edfc1cc231c16`,
cuGraph `40119e4cbb08da92bb3211635ac3d2819fdf8ecd`, and cuVS
`25b1be43a8c127e5ab6d2f29f20c62dbfd3351ab`. The manifest records diff
and binary hashes, toolkit, and CPU details. The SF1 TPC-H dataset had 8
Parquet files, 365,974,909 bytes and content fingerprint
`content-fnv1a64:6fc833c53ebc95d3`; its per-table row-group counts are in
the TSV metadata. The query cap/device ledger capacity was 16,086,990,848
bytes, device cache share zero, source reads in flight two, and operator width
one. The one warmup and preceding CPU runs make this a warmed-process
measurement; OS page-cache residency was not measured or forced. Tokio core
workers and Rayon threads were configured to 32, matching the CPU profile's
32 DataFusion target partitions; on-demand blocking/native helper thread
counts were not pinned.

The command used `flock /tmp/cudf-gpu.lock env` with
`TOKIO_WORKER_THREADS=32`, `RAYON_NUM_THREADS=32`,
`ALGEON_TPCH_BENCH_QUERY_IDS=q01,q06,q15,q17,q18,q03`,
`ALGEON_TPCH_BENCH_DATASET_MODE=sf1_external`,
`ALGEON_TPCH_BENCH_WARMUPS=1`, `ALGEON_TPCH_BENCH_ITERATIONS=10`,
`ALGEON_TPCH_BENCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT=2`, and
`ALGEON_TPCH_BENCH_ARTIFACT_DIR=target/bench/parallel-a-2026-09-19/artifacts-v3`
before `target/debug/native_tpch_benchmark`. Raw output is
`tpch-sf1-v3.tsv` and `tpch-sf1-v3.log`; each query has a `.plan.txt`,
`.observation.jsonl`, and `.samples.tsv` sidecar in `artifacts-v3/`. The TSV preserves every native
coverage field, outcome, fingerprint, and memory/copy counter.
`artifacts-v3/loaded-native-libraries.json` was written by that process after
execution. It hashes the mapped executable and 46 executable shared libraries,
including the loaded `cudf-install/lib/libcudf_rust.so` and
`libcudf.so`; the manifest verifies those recorded paths and hashes.
The exact benchmarked binaries are retained under `binaries-v3/`, since subsequent
Cargo checks may replace the files in `target/debug`; the manifests hash the
retained copies.
`baseline-manifest-v3.json` was generated with
`env TOKIO_WORKER_THREADS=32 RAYON_NUM_THREADS=32 python3 scripts/dev/capture_gpu_query_engine_baseline.py --output target/bench/parallel-a-2026-09-19/baseline-manifest-v3.json --run-config target/bench/parallel-a-2026-09-19/tpch-run-config-v3.json --run-log target/bench/parallel-a-2026-09-19/tpch-sf1-v3.log --result-tsv target/bench/parallel-a-2026-09-19/tpch-sf1-v3.tsv`.
It checks the executable hash recorded by the benchmark process against its
retained copy, rehashes all 46 mapped libraries, and records native source
SHAs, the Rust diff, toolkit, and CPU details. It also hashes 61 evidence
artifacts: measured TPCH plan/observation/sample sidecars, identity-bound
goodput attempts/summaries, TPC-DS parity, separate TPCH service calibration,
Sirius results, earlier diagnostic plan captures, trace exports/summaries,
and the GPU-counter support probe. The evidence list in
`tpch-run-config-v3.json` pins the exact files; the earlier calibration and
plan-capture artifacts have their own run identities and are not presented as
this TPCH run's executed plans.

All six queries selected native execution, completed, and matched the CPU
oracle. Times below are milliseconds from ten unprofiled iterations. P95 uses
linear interpolation at rank `1 + (n-1)*0.95`; `task peak` is the largest
per-task observed peak, **not** the unavailable attempt allocator high-water.

| Query | CPU mean | Native mean | Native median | Native p95 | Native SD | Planning mean | Iteration wall median | Task peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q01 | 316.561 | 561.117 | 558.079 | 577.150 | 10.746 | 59.780 | 617.758 | 2,490,882,232 |
| Q06 | 97.191 | 85.933 | 85.897 | 86.888 | 0.627 | 52.897 | 138.636 | 59,564,375 |
| Q15 | 210.548 | 90.191 | 90.117 | 91.373 | 0.720 | 93.242 | 183.250 | 5,543,776 |
| Q17 | 403.719 | 1,145.060 | 1,145.370 | 1,150.363 | 3.784 | 67.261 | 1,211.505 | 413,529,371 |
| Q18 | 453.675 | 2,095.108 | 2,094.854 | 2,100.558 | 3.906 | 73.056 | 2,168.302 | 381,059,025 |
| Q03 | 165.429 | 683.376 | 672.365 | 733.604 | 33.231 | 72.897 | 744.916 | 202,981,630 |

A separate `query_service_goodput` calibration of the same six SQL queries
used the same SF1 files and 16,086,990,848-byte cap, one warmup and three
measured single-flight attempts per item, a hot pool, and 32 configured
Tokio/Rayon threads. Its frozen `tpch_slo.tsv` records the service path's
end-to-end calibration latency and sealed attempt-aligned GPU peak. The
goodput session and the standalone TPCH benchmark have different physical
planning settings, so neither latency nor memory peak is a substitute for
the other surface. The service path fell back to CPU for Q15; its retained
attempt-bound physical plan has a `ScalarSubqueryExec` and no native wrapper.
The standalone TPCH report config disables that physical scalar-subquery
form and retains a whole-native plan. Calibration does not emit a structured
reason for a successful fallback, and no GPU attempt peak exists for Q15.
Keep that coverage difference visible.
From the same native build environment as the TPCH run, reproduce this
calibration with:

```bash
flock /tmp/cudf-gpu.lock env TOKIO_WORKER_THREADS=32 RAYON_NUM_THREADS=32 \
  ALGEON_GOODPUT_MODE=calibrate \
  ALGEON_GOODPUT_MIX=tpch.q01,tpch.q06,tpch.q15,tpch.q17,tpch.q18,tpch.q03 \
  ALGEON_GOODPUT_TPCH_DIR=fixture/tpc-h/data/sf1/parquet \
  ALGEON_GOODPUT_TPCH_SCALE=sf1 \
  ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES=16086990848 \
  ALGEON_GOODPUT_QUERY_CAP_BYTES=16086990848 \
  ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS=1 ALGEON_GOODPUT_POOL_STATE=hot \
  ALGEON_GOODPUT_CALIBRATION_WARMUP_REPS=1 ALGEON_GOODPUT_CALIBRATION_REPS=3 \
  ALGEON_GOODPUT_SLO_PATH=target/bench/parallel-a-2026-09-19/tpch_slo.tsv \
  ALGEON_GOODPUT_SOURCE_IDENTITY=550cc68d68e8-dirty \
  target/debug/query_service_goodput
```

| Item | Service disposition | E2E p50 ms | E2E p95 ms | Attempt peak max bytes |
| --- | --- | ---: | ---: | ---: |
| `tpch.q01` | native | 82.084 | 93.111 | 978,547,954 |
| `tpch.q06` | native | 55.229 | 56.615 | 59,565,029 |
| `tpch.q15` | CPU fallback | 301.964 | 302.034 | unavailable |
| `tpch.q17` | native | 129.629 | 130.092 | 370,159,863 |
| `tpch.q18` | native | 124.507 | 125.530 | 368,389,008 |
| `tpch.q03` | native | 102.686 | 102.781 | 304,212,345 |

Q01 and Q06 used the bounded aggregate/generic path: their measured
observations contain no compiled task definitions. Q15, Q17, Q18, and Q03
executed compiled graphs. Q17's measured observation has 17 task definitions,
17 compiled edges, and 10 barriers. Q39 and Q78
in `tpcds-sf1-v3.tsv` also selected native execution and matched the CPU oracle
under the same 16,086,990,848-byte native budget;
that parity entry has no latency or memory columns. A separate eight-second
goodput window gave each TPC-DS query two native, fingerprint-matched attempts:
Q39 end-to-end 159.303/145.482 ms with 31,849,794-byte sealed attempt peaks,
and Q78 313.978/273.109 ms with 165,083,380-byte peaks. These are service
attempt samples under load, not isolated ten-sample query timings. The goodput calibration
and ten-second hot-pool window used `tpch.q06`, `adhoc.wide_distinct`, and
`adhoc.global_sort` at 2 offered QPS with one active attempt and the same cap.
The window offered 17 requests: 17 succeeded, 17 met the frozen SLO, zero
output mismatches or memory failures, 1.7 goodput QPS; its maximum sealed
attempt peak was 411,668,403 bytes. This is a short smoke/control window, not
a capacity curve. Separate six-second hot-pool windows for
`adhoc.independent_union`/`adhoc.repeated_cte` and
`adhoc.variable_width_group` each offered six requests; all were native,
succeeded, and matched frozen fingerprints. Q15's native plan shows aggregate
node `n4` feeding two consumers (`n5` and `n9`), while the repeated CTE
workload establishes a reproducible SQL control without assuming its
optimizer retained sharing. Another six-second mixed window for PageRank on
the 156-edge karate fixture and TPC-H Q06 offered six requests; all six were
native, succeeded, and matched fingerprints. Every short window had one
active attempt, so none measures concurrent-query capacity.

The service windows retained one row per attempt in `goodput/<run>/attempts.tsv`.
The table gives per-item window results; every listed attempt had a `native`
disposition, succeeded, and matched its calibrated fingerprint. Median
end-to-end and execution times are milliseconds. The peak is the maximum
sealed attempt-aligned allocation, distinct from the TPCH task peaks above.

| Run | Item | Attempts | E2E median | Execution median | Attempt peak bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| `measure` | `tpch.q06` | 6 | 71.558 | 16.798 | 59,565,029 |
| `measure` | `adhoc.wide_distinct` | 4 | 54.968 | 23.627 | 352,155,315 |
| `measure` | `adhoc.global_sort` | 7 | 68.949 | 43.399 | 411,668,403 |
| `branch_measure` | `adhoc.independent_union` | 2 | 77.197 | 22.113 | 57,523,343 |
| `branch_measure` | `adhoc.repeated_cte` | 4 | 132.717 | 32.420 | 119,609,790 |
| `variable_width_measure` | `adhoc.variable_width_group` | 6 | 120.243 | 59.307 | 660,824,434 |
| `mixed_measure` | `graph.pagerank` | 2 | 24.382 | 11.537 | 18,832 |
| `mixed_measure` | `tpch.q06` | 4 | 79.171 | 19.194 | 59,565,029 |
| `tpcds_measure` | `tpcds.q39` | 2 | 152.392 | 63.553 | 31,849,794 |
| `tpcds_measure` | `tpcds.q78` | 2 | 293.543 | 121.714 | 165,083,380 |

The repeatable `scripts/dev/run_gpu_query_engine_controls.py` drives seven
isolated service controls from the frozen calibrations, with 2 s warmup,
12 s arrival windows at one offered QPS, one active attempt, the same device
cap, and 32 configured Tokio/Rayon threads. It takes the GPU lock for each
run and validates the run identity, summary, and every attempt row. The run
identity includes the executable hash, dataset file hashes, calibration hash,
settings, and runner hash; only an exact complete match is reused. Each new
identity writes to `goodput-ten/<item>-<identity>/`, leaving earlier evidence
intact. New goodput runs also save `loaded-native-libraries.json` from their
own `/proc/self/maps` after the measured window. Reuse requires this snapshot
and rechecks the executable and every mapped library's path, device, inode,
size, and SHA-256 against the current files; changed native code at the same
path therefore cannot reuse an earlier result. The seven runs recorded here
predate that additional reuse check and remain historical evidence. In these
seven identity-specific runs, all
12 window attempts for every item selected native execution, succeeded,
matched the calibrated fingerprint, and met its frozen SLO. The table reports
milliseconds over those 12 window attempts; SD is sample standard deviation
and p95 uses linear interpolation. Peaks are maximum sealed attempt-aligned
allocation, not TPCH task peaks. These controls measure serial service
behavior under scheduled arrivals, not concurrent-query capacity.

| Item | E2E median | E2E p95 | E2E SD | Execution median | Attempt peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adhoc.independent_union` | 77.957 | 79.543 | 0.769 | 22.427 | 57,523,343 |
| `adhoc.repeated_cte` | 133.887 | 134.393 | 5.819 | 32.068 | 119,609,790 |
| `adhoc.variable_width_group` | 116.778 | 121.628 | 2.731 | 59.023 | 660,824,434 |
| `adhoc.global_sort` | 70.902 | 77.304 | 3.887 | 44.327 | 411,668,403 |
| `graph.pagerank` | 24.672 | 25.267 | 0.343 | 11.412 | 18,832 |
| `tpcds.q39` | 156.502 | 159.767 | 5.181 | 66.519 | 31,849,794 |
| `tpcds.q78` | 271.032 | 273.295 | 7.371 | 111.544 | 165,083,380 |

Opt-in `ALGEON_GOODPUT_PLAN_ARTIFACT_DIR` captures the actual attempt-bound
physical plan on calibration warmup rep 0, then executes that same plan; the
measured service path has no plan-file I/O. `goodput-plans/` retains plans for
the independent union, repeated CTE, variable-width group, global sort,
PageRank, TPC-DS Q39/Q78, and service Q15. The separate
`plan-capture-manifest.json` pins their hashes, the rebuilt goodput binary,
the calibration, the Rust source diff, and the exact safe environment keys;
rerun those keys from the repository root with `flock /tmp/cudf-gpu.lock`
around `target/debug/query_service_goodput`. Each file labels its bounded
pipeline as a **recompiled diagnostic**, not the runtime admission schedule.
The union plan is native but reports `no_compiled_pipeline`, whereas the
repeated CTE has a compiled task graph. Plan-file I/O occurs only in the
calibration warmup, separately from the ten-sample TPCH and service windows.

Nsight Systems 2025.6.3 captured one Q01 and one Q17 run with
`--trace=cuda-sw,nvtx`; `q01-cuda-sw-v3` and `q17-cuda-sw-v3`
`.nsys-rep` and exported `.sqlite` files are retained.
The software CUDA trace was required to obtain kernel events on this GPU.
For each query, set `ALGEON_TPCH_BENCH_QUERY_IDS` to its ID, warmups to `0`,
iterations to `1`, and invoke
`flock /tmp/cudf-gpu.lock nsys profile --trace=cuda-sw,nvtx --output=<trace-prefix> env TOKIO_WORKER_THREADS=32 RAYON_NUM_THREADS=32 target/debug/native_tpch_benchmark`;
export with `nsys export --type=sqlite --output=<trace.sqlite> <trace.nsys-rep>`.
The two profiled runs had zero warmups and one iteration, so their durations
are not compared with the table above. In the interval from first to last
recorded kernel/copy event, Q01 had 683 kernels, 37.013 ms of kernel-busy
union within a 172.916 ms activity span, and 132.497 ms with neither kernel
nor copy active; Q17 had 12,502 kernels, 341.532 ms of kernel-busy union
within a 1,352.932 ms span, and 1,001.030 ms with neither active. Kernel-kernel
and kernel-copy overlap were zero in both traces. CUDA stream synchronization
API calls occupied 36.686 ms (Q01) and 352.273 ms (Q17) in aggregate; those
summed call durations are not whole-query wait time. The reproducible read-only
`scripts/dev/summarize_gpu_query_trace.py` writes `q01-trace-v3-summary.json` and
`q17-trace-v3-summary.json`. It joins CUDA launch APIs to kernels by correlation
ID: all 683 Q01 and 12,502 Q17 kernels matched a launch API. Total launch API
time was 18.528 ms and 54.033 ms respectively. The per-kernel
kernel-start-minus-launch-return deltas cannot be added into a query wait time
because they overlap. The largest observed GPU activity gaps were 12.649 ms
and 15.603 ms; both ended in a `cudaMemcpyAsync_v3020` operation whose host
API began 12.596 ms (Q01) and 15.593 ms (Q17) after the previous GPU event.
This establishes late host submission for those next events, but does not
assign each entire gap to the host. CUPTI recorded Q01 H2D/D2H/D2D copies of
79,245,663/2,858,105/82,048 bytes with respective copy-busy unions of
3.162/0.227/0.017 ms; Q17 recorded 149,429,638/2,276,343/152,393,872 bytes
and 7.154/1.746/1.469 ms. These are device copy intervals, not elapsed host
staging or result-materialization timings.
SM activity and memory-throughput counters were not qualified by these traces.
`nsys profile --gpu-metrics-devices=help` reported the GB203 device as
unsupported due to insufficient GPU-counter privilege; its output is saved as
`gpu-metrics-support.txt`.
The activity spans and overlap figures are unions/intersections of `start`/`end`
intervals from the exported `CUPTI_ACTIVITY_KIND_KERNEL` and
`CUPTI_ACTIVITY_KIND_MEMCPY` tables, not a GPU utilization percentage for the
whole query.

Existing GPU contracts passed for union over resident inputs, grouped
high-cardinality AVG, incremental high-cardinality merge, one graph CTE
lineage feeding two cuVS consumers, graph algorithm parity, and cuVS Parquet
execution.
These are correctness smoke results, not timed entries in the table.

The service controls have at least ten attempt samples per item, but no
isolated query timing series. A width comparison must run the same controls
and distinguish service end-to-end time from isolated TPCH execution time.

## Same-data Sirius reference

Sirius source `72aa972eb3a5be4fd5e612224e24d86bbf3ad91d` had a dirty
worktree. Its Pixi DuckDB 1.5.5 loaded the built extension despite a separate
DuckDB shell link failure. `scripts/dev/compare_sirius_tpch_baseline.py` reads
the **same SQL** directly from Algeon's `crates/workloads/src/tpch/catalog.rs`
and registers views over the **same eight SF1 Parquet files**. Sirius's bundled
TPC-H SQL uses different dates, predicates, and constants, so its stock query
files are not comparable. The script sets `gpu_execution=true` and
`enable_duckdb_fallback=false`; all six queries ran and matched the DuckDB CPU
oracle. Q01 differs only in floating AVG's final digits and passed the script's
explicit `1e-12` relative/absolute float tolerance; every other output matched
exactly. No query fell back, was unsupported, or was unrunnable in this mode.

The matched-cap mode uses this directory's `sirius-matched-cap.yaml`, copied from Sirius's
`test/cpp/integration/integration.yaml` with `usage_limit_fraction: 0.5`
replaced by `usage_limit_bytes: 16086990848`. It configures no disk tier, so
disk spill is off. It does not call `pin_table`, and its GPU-to-host downgrade
path remains enabled; no host-downgrade event count was collected. Sirius and
Algeon each used one warmup and ten unprofiled samples in a warm process.
The Sirius runner keeps a separate connection per query and retains an
unmeasured OS page-cache state, as did the Algeon baseline. The prior
`sirius-sf1-ten.json` run used Sirius's 50%-capacity integration config and
is explicitly a different-cap control; do not mix its values into this table.

From the Sirius root, reproduce the matched-cap run with
`flock /tmp/cudf-gpu.lock pixi run python scripts/dev/compare_sirius_tpch_baseline.py --sirius-root "$SIRIUS_ROOT" --config crates/bench/parallel-baseline/sirius-matched-cap.yaml --dataset fixture/tpc-h/data/sf1/parquet --output target/bench/parallel-a-2026-09-19/sirius-sf1-matched-cap.json --warmups 1 --iterations 10`.
The JSON retains all ten samples, per-file Parquet hashes,
SQL catalog/config/extension hashes, source
revision, tracked diff hash and untracked-path list, per-query parity and
explicit error/status fields. Sirius latency
includes DuckDB planning, execution, and fetched-row transfer; Algeon's native
latency column above is execution-only, so these are side-by-side reference
values rather than speedup ratios. Sirius's host tier and Algeon's
device-cache/attempt accounting are also different. Sirius allocator peaks
were unavailable; the Algeon task peaks above are not Sirius peak substitutes.

| Query | Sirius coverage | Parity | Sirius median ms | Sirius p95 ms | Sirius SD ms | Algeon native median ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Q01 | GPU, fallback off | float tolerance | 29.607 | 30.715 | 0.739 | 534.551 |
| Q06 | GPU, fallback off | exact | 23.787 | 25.390 | 0.988 | 82.143 |
| Q15 | GPU, fallback off | exact | 32.497 | 33.387 | 0.836 | 89.483 |
| Q17 | GPU, fallback off | exact | 55.216 | 56.293 | 1.071 | 1,120.612 |
| Q18 | GPU, fallback off | exact | 68.234 | 69.023 | 0.651 | 2,055.694 |
| Q03 | GPU, fallback off | exact | 42.421 | 43.396 | 0.591 | 656.450 |

The `sirius-q06-cuda-sw.nsys-rep` trace and SQLite/JSON exports provide an
independent GPU execution check: 112 kernels were recorded, with 4.239 ms of
kernel-busy union and 2.818 ms of kernel-kernel overlap in that profiled run.
The profiler result is not mixed with unprofiled latency samples.

The executor has no width 2/4 setting yet. Algeon modified width 1/2/4
candidate rows belong after B/C implements that control, using this manifest
and the same workload. Host staging on plans with host boundaries, D2H elapsed,
client end-to-end for TPCH,
attempt allocator high-water for TPCH, Sirius allocator peaks, SM activity,
and memory throughput are currently missing measurements.
