# Embedding Examples

Algeon is a GPU query engine for
[Apache DataFusion](https://datafusion.apache.org/) that runs inside the
embedding process. The service keeps its HTTP framework, session lifecycle,
and catalogs; Algeon evaluates supported physical plans on CUDA with cuDF and
exposes cuGraph/cuVS algorithms as SQL UDTFs.

This crate holds the runnable embedding examples. Each binary is a complete
program that builds and runs on its own, and each later example adds one layer
to the previous one in the order of the table below.

## The embedding model

Every example is assembled from the same three pieces:

1. **One bounded backend per process.** `AlgeonGpuBackend` owns the CUDA
   devices, the admission queue, and the per-device memory-grant ledger. All
   bounds (device memory limits, maximum active attempts) are fixed on the
   builder and shared by everything installed from it.
2. **Sessions installed from that backend.** `backend.install_on(...)` wires
   the native optimizer into a DataFusion `SessionStateBuilder`. Many sessions
  ; and therefore many frontends; share one backend and one admission
   ledger.
3. **An attempt per query.** Every query execution is an explicit attempt:
   begin it before planning, consume the result stream, then await the
   completion report. The `QueryAttemptReport` is the machine-readable record
   of what happened: disposition (native cuDF execution or DataFusion CPU
   fallback), admission grant, terminal outcome, and timing.

Planning decisions are structured values. Assert `PlanningReport`s and
attempt reports; printed physical plans are not a stable interface.
`algeon_explain_coverage('<sql>')` in SQL and `EXPLAIN GPU <sql>` over Flight
SQL return the same coverage evidence before a query runs.

## Choosing an example

| Example | Layer it teaches | Query entry point | Needs | Runs in |
| --- | --- | --- | --- | --- |
| [`attempt_lifecycle`](#attempt_lifecycle-the-per-query-contract) | The per-query contract every embedding uses | Explicit `backend.begin_attempt()` | One CUDA device | Seconds |
| [`embedded_backend`](#embedded_backend-one-process-two-frontends) | A production-shaped service topology | `SessionContext::sql` (REST) and Flight SQL tickets | CUDA device, demo dataset | Minutes |
| [`otlp_metrics_exporter`](#otlp_metrics_exporter-backend-metrics-over-otlp) | The metrics ownership split | `BackendMetricsOtlpExporter` in your process | Nothing; an OTLP collector is optional | Seconds |

`embedded_backend` does not call `begin_attempt`. Its admission is plan-scoped
(REST path) or ticket-scoped (Flight path), driven by the installed optimizer
rule and the Flight server. `attempt_lifecycle` drives the attempt API directly
and shows the contract those frameworks manage.

## `attempt_lifecycle`: the per-query contract

The minimal supported embedding path. One bounded backend is installed into
two sessions, and every invocation goes through the explicit execution-attempt
adapter:

```rust
let attempt = backend.begin_attempt(AttemptOptions::default())?; // before planning
let stream = attempt.sql(&ctx, sql).await?;                      // plan + execute
// ... consume the stream ...
let report = attempt.completion().await;                         // terminal report
```

The binary walks the three dispositions an attempt can reach:

- **CPU-only fallback.** `approx_distinct` is outside the native capability
  catalogue, so the plan keeps its executable DataFusion CPU baseline. The
  report records the fallback disposition.
- **Native execution.** Two attempts run `SUM ... WHERE value > 0`
  concurrently on the same explicitly owned device 0, proving that concurrent
  attempts keep distinct identities and receive independent admission grants
  from one ledger.
- **Stream drop.** Completion is independent of result consumption: dropping
  the stream without reading it still produces a bounded terminal report, so
  observability survives clients that disconnect early.

```bash
cargo run -p examples --all-features --bin attempt_lifecycle
```

Requires one CUDA device; exits with a clear message when none is present. Set
`ALGEON_ATTEMPT_LIFECYCLE_PROFILING_ROWS` (default 1,000,000) for an
Nsight-sized input; the binary doubles as a public-API profiling probe for
the native concurrent path.

The executable contract behind this example lives in
[`tests/attempt_lifecycle_tests.rs`](../../tests/attempt_lifecycle_tests.rs)
of the root `algeon-datafusion` package.

## `embedded_backend`: one process, two frontends

A realistic service shape built on the same primitives: one shared
`AlgeonGpuBackend` serving two frontends in one process.

- REST endpoints for predefined, parameter-validated GPU pipelines.
- Arrow Flight SQL for ad-hoc SQL, metadata, prepared statements, and
  cancellation.

REST requests use plan-scoped admission from the shared session. Flight SQL
issues statement tickets first and acquires GPU admission only when a native
plan starts executing. Native work from both paths draws from one
`QueryService` ledger, visible at `GET /api/gpu`.

```text
                                   one backend process
                        +----------------------------------------+
A: HTTP client ---------->   example HTTP API                    |
                        |    /api/pipelines, /api/gpu            |
                        |                  |                     |
                        |                  |                     |
B: Flight SQL client ---->   Flight SQL frontend                 |
                        |    SQL, metadata, cancel               |
                        |                  |                     |
                        |                  v                     |
                        |   shared DataFusion SessionContext     |
                        |                  |                     |
                        |                  v                     |
                        |   AlgeonGpuBackend                      |
                        |   QueryService admission ledger        |
                        |                  |                     |
                        |                  v                     |
                        |   cuDF / cuGraph on CUDA               |
                        +------------------|---------------------+
                                           |
                                           v
                         Citation Network Parquet files
                         registered as external tables
```

This topology fits an application with a small set of product-owned workflows
that also exposes SQL to operators, notebooks, or BI tools. Both frontends share
the process's backend, catalog view, and admission ledger.

### Data

The example runs against the DBLP / AMiner Citation Network V12 demo dataset:
4,894,081 papers and 45,564,149 citation edges. The REST pipelines query
these unqualified table names:

- `papers`
- `paper_fos`
- `citation_edges`
- `citation_edges_by_dst`

The source `dblp.v12.json` is about 12.5 GB. The local developer fixture path
is `fixture/demo/citation_network/raw/dblp.v12.json`; pass `--source-json` if
you keep the dataset outside the repo. Keep
`fixture/demo/citation_network/v1/parquet` on fast local storage, for example
with a symlink to NVMe, and allow roughly 30 GB of transient space for ingest
plus Parquet output. The ingest script requires Python with `pyarrow`.

```bash
python3 fixture/demo/citation_network/scripts/prepare.py --overwrite
```

Once the example is running, register the generated Parquet files as external
tables through Flight SQL, as shown in the
[citation network guide](https://algeon.dev/guides/datasets/citation-network).
Registrations last until the process stops.

### Run

#### GPU execution modes

Embedded and server sessions use the same three modes. `native_preferred` is
the default and keeps ordinary SQL eligible for native rewrite.
`functions_only` leaves scans, filters, joins, aggregates, sorts, windows, and
limits in DataFusion while explicit cuGraph and cuVS UDTFs cross the GPU
boundary. `native_required` rejects a completed plan that still contains
DataFusion CPU execution.

The `functions_only_modes` binary runs one PageRank query with CPU filter and
sort operators around the GPU UDTF. Select a mode without changing the
SQL:

```bash
ALGEON_EXAMPLE_EXECUTION_MODE=functions_only \
cargo run -p examples --all-features --bin functions_only_modes
```

Server deployments use `native.execution_mode` in TOML or
`ALGEON_SERVER_NATIVE_EXECUTION_MODE`. GPU inputs and outputs use bounded Arrow staging
with DataFusion memory accounting; staging does not spill. Unsupported
capability, OOM, cancellation, and runtime failures remain structured errors,
and a GPU UDTF is not replayed as a CPU algorithm after GPU execution
starts.

The example consumes the normal Flight SQL server environment plus
`EXAMPLE_REST_BIND`. `EXAMPLE_REST_BIND` is the bind address for this example's
own axum HTTP API (`/healthz`, `/api/pipelines`, `/api/gpu`).

The server has one bounded native admission contract; the example uses that
shared query-service ledger.

```bash
EXAMPLE_REST_BIND=127.0.0.1:8080 \
DATAFUSION_CATALOG_DEFAULT_CATALOG=datafusion \
DATAFUSION_CATALOG_DEFAULT_SCHEMA=public \
cargo run -p examples --all-features --bin embedded_backend
```

`embedded_backend` is the crate's `default-run` binary, so a bare
`cargo run -p examples --all-features` with the same environment works
too.

> **cuVS note.** Building with `--features cuvs` registers and executes the
> `cuvs_*` UDTFs inside the backend-managed query memory domain.

### Pipelines

The example exposes these predefined, parameter-validated GPU pipelines:

| Pipeline | Capability |
| --- | --- |
| `sql_aggregate_topk` | Filter, sort, and limit over `papers`. |
| `sql_join_facets` | Join `paper_fos` to `papers`, then group and aggregate. |
| `graph_pagerank_topk` | Run `cugraph_pagerank`, join vertices back to papers, and return top scores. |
| `graph_bfs_hops` | Run `cugraph_bfs`, join vertices back to papers, and aggregate by hop distance. |

`GET /api/pipelines/{name}/plan` returns the structured `PlanningReport` JSON
for that pipeline's default parameters. Use a `Selected` outcome as native
execution evidence; `NotSupported` and `NotSelectedByCost` retain the reason
for a non-native choice. Do not depend on printed physical plans.

### REST examples

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/api/pipelines

curl -X POST http://127.0.0.1:8080/api/pipelines/sql_aggregate_topk \
  -H 'content-type: application/json' \
  -d '{"year_from":2010,"year_to":2020,"limit":10}'

curl -X POST http://127.0.0.1:8080/api/pipelines/sql_join_facets \
  -H 'content-type: application/json' \
  -d '{"fos_name":"Artificial intelligence"}'

curl -X POST http://127.0.0.1:8080/api/pipelines/graph_pagerank_topk \
  -H 'content-type: application/json' \
  -d '{"limit":10}'

curl -X POST http://127.0.0.1:8080/api/pipelines/graph_bfs_hops \
  -H 'content-type: application/json' \
  -d '{"seed_paper_id":2963403868,"max_depth":2}'

curl http://127.0.0.1:8080/api/pipelines/sql_aggregate_topk/plan
curl http://127.0.0.1:8080/api/gpu
```

### Flight SQL

Flight SQL clients connect to `grpc://127.0.0.1:50051` and see the same
workspace. For example:

```bash
printf '%s\n' \
  "SELECT count(*) FROM papers" \
  "SELECT vertex, value FROM cugraph_pagerank(edges => (SELECT src, dst, weight FROM citation_edges)) ORDER BY value DESC LIMIT 5" |
  arrow_cli --host 127.0.0.1 -P 50051 --timeout 120 --output table
```

## `otlp_metrics_exporter`: backend metrics over OTLP

A reference embedding shape for backend-owned metrics, kept deliberately
opt-in via the `otlp-metrics` feature because it adds a tonic transport stack
that ordinary embeddings do not need.

The ownership split is the point of the example:

- **The engine owns measurement**: labels, units, bucket conversion, and
  aggregation semantics; and exposes one coherent `BackendMetricsSnapshot`
  per backend generation.
- **Your application owns the OpenTelemetry SDK**: the meter provider, export
  schedule, resource attributes, and endpoint policy.

Construct the SDK meter provider, pass the meter and a backend-snapshot
closure to `BackendMetricsOtlpExporter::new`, and call `export_histograms` on
the application's export schedule with its OTLP `Resource`, the current backend
generation's cumulative start time, and the collection timestamp. The
embeddable code is [`src/otlp_metrics.rs`](src/otlp_metrics.rs); its protobuf
contracts are pinned by
[`tests/otlp_metrics_exporter.rs`](tests/otlp_metrics_exporter.rs):

```bash
cargo test -p examples --all-features
```

No CUDA device is required for this one.

## Where to go next

- [Integrating Algeon](https://algeon.dev/integration): choosing between
  a DataFusion session, this embedded example, and the Flight SQL server.
- [DataFusion Session](https://algeon.dev/integration/datafusion-session);
  backend builder, admission configuration, and final-plan requirements.
- [Flight SQL Server](https://algeon.dev/integration/flight-sql-server);
  the server environment contract `embedded_backend` consumes.
- [Configuration](https://algeon.dev/integration/configuration): the
  full environment-variable reference.
- [GPU Coverage Validation](https://algeon.dev/integration/gpu-coverage-validation);
  `algeon_explain_coverage` / `EXPLAIN GPU` before shipping a query.
- [Handling Errors](https://algeon.dev/integration/handling-errors);
  typed errors at your application boundary.
