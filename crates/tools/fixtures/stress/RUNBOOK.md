# Service stress runbook

These profiles check shared-backend service behavior through public attempts.
Performance, SLO and output-correctness comparisons use the existing
[`query_service_goodput`](../../../bench/src/query_service_goodput/mod.rs)
harness. The stress runner records lifecycle, admission, cache and transport
evidence; it does not compare query result rows.

All shipped profiles configure GPU ordinal `0`, its CUDA-visible capacity of
`101975851008` bytes, zero backend reserve and an 8 GiB attempt cap. Cache also
enables a 6 GiB resident cap. The capacity comes from the current 96 GB GPU's
CUDA total; NVML reports a different total. On another Linux
CUDA/NVML host, set an explicit device limit within its CUDA-visible total.
Device limits and resident charges follow [memory and query lifetime](../../../../website/docs/design/admission.mdx).

| Profile | Service check | Submission window |
| --- | --- | --- |
| `mixed.json` | SF10 q01/q06, shared sessions, four concurrent attempts | 5 s warmup + 60 s measurement |
| `recovery.json` | SF10 q06, queue pressure followed by recovery | 5 s warmup + three 20 s steps |
| `lifecycle.json` | SF10 q01/q06, eight submitters against four active slots | 60 s, bounded by 2048 rounds |
| `cache.json` | SF10 q06 and full DBLP BFS, paired cache accounting and same-key probes | 5 s + 60 s for each half, plus dedicated cohorts |
| `flight-baseline.json` | SF10 q06 baseline for both Flight profiles | 5 s warmup + 30 s measurement |
| `flight-{pooled,churn}.flight-profile` | SF10 q06 over Flight with four clients | 5 s warmup + 30 s measurement |

Measured backend profiles allow 30 s after the submission window for existing
attempts to seal reports. The allowance adds no measurement time or new work.
Cache's dedicated source and BFS cohorts each have a separate 60 s bound.
Flight clients await already submitted attempts after their submission window.

## Preparation

Run commands from the repository root. Use existing fixtures at
`fixture/tpc-h/data/sf10/parquet` and
`fixture/demo/citation_network/v1/parquet`. Cache requires the complete DBLP
V12 fixture and its manifest. Fixture creation, downloads and builds belong
outside the measured window. When preparation is needed, use the recorded
commands below; graph ingestion requires the source JSON to be staged first.

The native libraries must already be built using the
[build guide](../../../../website/docs/build/from-source.md). Set their selected
prefixes for provenance before running the tools.

```bash
export CUDF_INSTALL_DIR="$PWD/target/native/cudf-install"
export CUVS_INSTALL_DIR="$PWD/target/native/cuvs-install"
export CUML_INSTALL_DIR="$PWD/target/native/cuml-install"
export LD_LIBRARY_PATH="$CUDF_INSTALL_DIR/lib:$CUVS_INSTALL_DIR/lib:$PWD/target/native/cugraph-install/lib:$CUML_INSTALL_DIR/lib:$PWD/target/native/cugraph-build"

fixture/fixture.sh tpch generate --scale 10 --out-root fixture/tpc-h/data
fixture/fixture.sh graph dblp-ingest \
  --source-json fixture/demo/citation_network/raw/dblp.v12.json \
  --out-root fixture/demo

cargo build --release -p tools --features single-node-multi-gpu-validate-run-cugraph
cargo build --release -p algeon-server --bin algeon_server
cargo build --release -p bench --all-features --bin query_service_goodput
mkdir -p target/stress
```

`--dry-run` accepts an empty `workloads` list and records configuration; it
cannot validate these workload profiles. `--fixture-preflight` retains its
exact contract of one TPC-H fixture plus one canonical DBLP V12 fixture.
`cache.json` satisfies that selection; the relational profiles do not. Each
`--measured-run` validates its complete selected fixture set before backend
construction or output creation. Preflight reads fixtures without preparing
them.

```bash
target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/cache.json \
  --out-dir target/stress/cache-preflight --fixture-preflight
```

Use a fresh output directory for every invocation. Backend and Flight runners
reject existing `--out-dir` paths; create their parent directories only. The
examples below assume their output paths do not exist. For a repeat, choose
new names and retain the previous artifacts. A repeated Flight baseline also
requires a copied Flight profile with `matched_in_process.run_artifact` updated
to the new artifact. Flight paths resolve relative to the profile file.

## Mixed load and recovery

Build before acquiring `/tmp/cudf-gpu.lock`. Run one GPU workload at a time.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/mixed.json \
  --out-dir target/stress/mixed --measured-run

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/recovery.json \
  --out-dir target/stress/recovery --measured-run
```

Recovery offers `1`, `256`, then `1` request/s against one active slot, four
queue slots and a 500 ms queue wait. Qualification requires a baseline queue
high-water mark below the queue cap, an intervening higher-rate step that
reaches the cap, GPU progress in each step, and recovery p95 total latency and
queue wait within three times the baseline. These rates are service-test
inputs; they are not measured sustainable QPS.

Each backend invocation writes `<out-dir>/run-artifact.json`. Require status
`measured_run`, `execution.measurement.qualifying = true`, and an empty
`execution.qualification_failures`. Recovery additionally requires
`execution.measurement.qps_envelope.status = "evidenced"`. The artifact retains
attempt rows, aggregate equality, time series and the quiescent close outcome.
Declared `not_evidenced` clauses remain evidence gaps even in a qualifying run.
Dirty source records `dirty_source` and makes the top-level artifact
non-qualifying even when the workload and lifecycle checks pass. Mixed load also requires
at least one successful GPU-reaching measurement attempt for each selected
query; aggregate progress alone cannot establish q01 and q06 coverage.

```bash
jq -e '.status == "measured_run" and
  .execution.measurement.qualifying == true and
  .execution.qualification_failures == [] and
  ([.execution.measurement.aggregate.attempts[] |
    select(.phase == "measurement" and .outcome == "succeeded" and
      (.disposition == "Native" or .disposition == "PartialNative")) | .query_label] | unique == ["q01", "q06"])' \
  target/stress/mixed/run-artifact.json

jq -e '.status == "measured_run" and
  .execution.measurement.qps_envelope.status == "evidenced"' \
  target/stress/recovery/run-artifact.json
```

## Lifecycle

Cancellation, deadline, close and quarantine controls are mutually exclusive.
Run each in a separate process and directory.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/lifecycle.json \
  --out-dir target/stress/cancel --measured-run --cancel-first-after-ms 0

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/lifecycle.json \
  --out-dir target/stress/deadline --measured-run --deadline-first-after-ms 100

flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/lifecycle.json \
  --out-dir target/stress/close --measured-run --close-under-load
```

In addition to `measured_run` status, inspect the selected barrier under
`execution.measurement`. Cancellation requires `cancellation_requested` and
nonzero `cancelled_reports`. Deadline requires an admitted witness before the
100 ms deadline, one typed deadline cancellation, capacity release and healthy
reuse. Cancellation is cooperative at native boundaries; a deadline cannot
interrupt a CUDA call already executing. Retain a failed witness as
non-qualifying evidence. Close requires live admitted and queued work at the
midpoint, sealed `backend_closing` reports for queued work, exact report
accounting and a quiescent backend close.

Quarantine is unvalidated on the one-GPU host. An operator must first copy
`lifecycle.json` to `target/stress/lifecycle-multi-gpu.json` and configure at
least two available devices, including their memory limits and installation
bounds. Then run the control separately.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config target/stress/lifecycle-multi-gpu.json \
  --out-dir target/stress/quarantine --measured-run --quarantine-device 0
```

Require `measured_run` and
`execution.measurement.aggregate.quarantine_fallback.status = "evidenced"`.
The proof includes live target work before acknowledgement, a closed target
ledger, zero target placements after acknowledgement, release of existing
target work and GPU completions on healthy devices. Merely adding a second
profile entry does not establish hardware or fallback evidence.

## Cache

The runner derives the cache-disabled companion from `cache.json`. Run the
pair together and retain their shared artifact.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/cache.json \
  --out-dir target/stress/cache --measured-run --cache-calibration

jq -e '.status == "cache_calibration" and
  .execution.qualification_failures == [] and
  .execution.cache_calibration.cache_disabled.qualifying == true and
  .execution.cache_calibration.cache_opt_in.qualifying == true and
  (.execution.cache_calibration.clauses | all(.status == "evidenced"))' \
  target/stress/cache/run-artifact.json
```

Qualification requires cache-domain ownership to match ledger charges, the
6 GiB cap to hold, source promotion coalescing and BFS followers with fewer
physical traversals than eligible requests. Additional cache-on admission
failures must have recorded explanations. The two-wide same-key cohorts run
separately from the measured window. This profile establishes cache behavior;
output fingerprints and performance comparisons belong to goodput calibration.

## Flight

Run the matched baseline before starting the server. `server.toml` matches
its admission and native settings. Clear inherited `ALGEON_SERVER_*` overrides
before starting the server so the TOML determines those settings. Each Flight
profile binds the baseline at
`target/stress/flight-baseline/run-artifact.json` and the companion server
configuration by hash. The Flight reader requires baseline status
`measured_run`; a development artifact with dirty-source qualification
failures cannot start the comparison. Use clean provenance for qualification.

```bash
flock /tmp/cudf-gpu.lock target/release/algeon_tools single-node-multi-gpu-validate \
  --config crates/tools/fixtures/stress/flight-baseline.json \
  --out-dir target/stress/flight-baseline --measured-run

flock /tmp/cudf-gpu.lock target/release/algeon_server \
  --config crates/tools/fixtures/stress/server.toml
```

After the baseline qualifies and the server accepts connections, use another
terminal to register the prepared table and run both clients. The server owns
the GPU lock throughout; the clients share that server process.

```bash
target/release/algeon_tools flight-sql-query http://127.0.0.1:50051 \
  "CREATE EXTERNAL TABLE lineitem STORED AS PARQUET LOCATION 'fixture/tpc-h/data/sf10/parquet/lineitem.parquet'"

target/release/algeon_tools flight-sql-stress \
  --config crates/tools/fixtures/stress/flight-pooled.flight-profile \
  --out-dir target/stress/flight-pooled

target/release/algeon_tools flight-sql-stress \
  --config crates/tools/fixtures/stress/flight-churn.flight-profile \
  --out-dir target/stress/flight-churn
```

Stop the server after the clients finish to release the lock. Flight writes
`attempts.tsv` and `summary.json`. Require nonzero measured `attempts`,
`succeeded = attempts`, `failed = 0` and `cancelled = 0` in each summary.
`matched_in_process` records the accepted baseline selection and hashes;
`delta_from_in_process` records server-minus-baseline application QPS and
latency. Server GPU-reaching QPS remains `not_observable_over_flight`.
Transport deltas do not establish an SLO or output-correctness comparison.

## Goodput, SLO and correctness comparisons

Use the existing [goodput harness](../../../bench/src/query_service_goodput/mod.rs)
and [calibration contract](../../../bench/calibration/README.md). Required
inputs are `ALGEON_GOODPUT_OUT_DIR` and `ALGEON_GOODPUT_SLO_PATH` with fresh
paths, `ALGEON_GOODPUT_TPCH_DIR=fixture/tpc-h/data/sf10/parquet`,
`ALGEON_GOODPUT_TPCH_SCALE=sf10`, `ALGEON_GOODPUT_MIX=tpch.q01@1,tpch.q06@1`,
`ALGEON_GOODPUT_DEVICE_ORDINALS=0`,
`ALGEON_GOODPUT_DEVICE_CAPACITY_BYTES=101975851008` and
`ALGEON_GOODPUT_QUERY_CAP_BYTES=8589934592`. Pin source chunk size, seed and
observation window. Clear optional TPC-DS/graph paths and request-cap manifests
for this relational comparison.
Record the binary hash and source state beside the results. Stop the Flight
server before invoking the GPU harness.

With those `ALGEON_GOODPUT_*` inputs set, calibrate separately, then compare one
and four active attempts while retaining the same cap and frozen SLO file.

```bash
ALGEON_GOODPUT_MODE=calibrate ALGEON_GOODPUT_RUN_ID=calibration \
ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS=1 \
flock /tmp/cudf-gpu.lock target/release/query_service_goodput

ALGEON_GOODPUT_MODE=measure ALGEON_GOODPUT_RUN_ID=active1 \
ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS=1 \
flock /tmp/cudf-gpu.lock target/release/query_service_goodput

ALGEON_GOODPUT_MODE=measure ALGEON_GOODPUT_RUN_ID=active4 \
ALGEON_GOODPUT_MAX_ACTIVE_ATTEMPTS=4 \
flock /tmp/cudf-gpu.lock target/release/query_service_goodput
```

Retain the SLO store and each run's `summary.tsv`, `attempts.tsv`,
`request_manifest.tsv`, `device_samples.tsv` and `slo_sensitivity.tsv`.
Calibration fixes output fingerprints and SLO deadlines before measurement;
the harness rejects a changed GPU model or dataset identity. Require
`output_mismatch = 0` and `reports_unpublished = 0`. Compare goodput, SLO counts
and latency alongside all outcome classes and offered counts. The
[existing matrix driver](../../../bench/scripts/run_goodput_matrix.sh) owns
broader comparisons and requires its dataset paths to be supplied explicitly.

## Retired experiments

The following sustained experiments are no longer shipped in this suite.
Owner tests remain bounded contract checks; they do not replace sustained
load experiments or supply measurements for the retired cases.

| Removed experiment | Retained contract owner |
| --- | --- |
| Feature-off cuGraph traffic beside relational load | [Feature-off SQL tests](../../../algeon-datafusion/tests/contract_tests_cpu/cugraph_sql_feature_off_tests.rs) |
| Live OTLP collector/export runs | [Operability evidence tests](../../src/single_node_multi_gpu_validate/operability.rs) |
| Nsight memory and CUDA overlap calibration | [Trace/artifact contracts](../../src/single_node_multi_gpu_validate/artifact.rs) and [sampling tests](../../src/single_node_multi_gpu_validate/timeseries.rs) |
| Telemetry on/off comparison | [Runtime metrics tests](../../../algeon-query-runtime/src/observability/metrics_summary/tests.rs) |
| Distinct graph population and churn | [Resident graph cache tests](../../../algeon-query-runtime/src/runtime/cache/resident_graph/tests.rs) |
| Cancellation/deadline storms and slow completion sinks | [Runner probe contracts](../../src/single_node_multi_gpu_validate/tests.rs) and [release tests](../../../algeon-query-runtime/src/runtime/query_service/tests/release_commit.rs) |

Retain non-qualifying artifacts and their process exit status. Change load or
hardware settings in a copied profile and report the new profile hash with its
results. A profile's presence in this directory does not establish a measured
pass on the current host.
