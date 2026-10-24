# Algeon tools

`algeon_tools --help` lists operator commands and explicit acceptance runners.
The small application examples live in [examples](../examples/README.md).

## REST/Flight acceptance fixture

[`embedded_backend`](src/bin/embedded_backend.rs) supplies the server exercised
by `algeon_tools embedded-backend-e2e`. Its HTTP routes, Flight session and
citation-network pipelines are in [`src/embedded_fixture`](src/embedded_fixture).
The runner builds the fixture binary automatically with workspace features.

```bash
ALGEON_EMBEDDED_BACKEND_E2E=1 flock /tmp/cudf-gpu.lock \
  cargo run -p tools --all-features -- embedded-backend-e2e
```

The fixture needs the prepared [citation-network dataset](../../website/docs/guides/datasets/citation-network.mdx).
For manual use, launch it with `cargo run -p tools --features embedded-backend-fixture --bin embedded_backend`
and register the dataset tables through Flight SQL. `EXAMPLE_REST_BIND` defaults
to `127.0.0.1:8080`; the Flight frontend consumes the normal `ALGEON_SERVER_*`
configuration. The HTTP routes remain `/healthz`, `/api/pipelines`,
`/api/pipelines/{name}`, `/api/pipelines/{name}/plan` and `/api/gpu`.

## S3 and Iceberg acceptance runs

[`s3-parquet-e2e`](src/e2e/s3_parquet_e2e.rs) scans generated Parquet objects
through the `[s3]` storage against real S3 or the RustFS service in
[`fixtures/s3-local`](fixtures/s3-local/compose.yaml).
[`iceberg-local-e2e`](src/e2e/iceberg_local_e2e.rs) writes Iceberg tables
through a REST and a Glue catalog and queries them over an in-process Flight
SQL server, against the services in
[`fixtures/iceberg-local`](fixtures/iceberg-local/compose.yaml).

```bash
docker compose -f crates/tools/fixtures/iceberg-local/compose.yaml up -d --wait
AWS_ACCESS_KEY_ID=algeon-e2e AWS_SECRET_ACCESS_KEY=algeon-e2e-secret \
  flock /tmp/cudf-gpu.lock cargo run -p tools --features iceberg-local-e2e -- iceberg-local-e2e
docker compose -f crates/tools/fixtures/iceberg-local/compose.yaml down -v
```

`algeon_tools <command> --help` prints each run's environment, and
[Tests](../../website/docs/build/tests.md#acceptance-runs-outside-the-gate)
lists what each run checks.

## Attempt profiling probe

[`attempt_lifecycle`](src/bin/attempt_lifecycle.rs) exercises CPU aggregation over
a native scan, concurrent native attempts and stream drop. It retains
`ALGEON_ATTEMPT_LIFECYCLE_PROFILING_ROWS` (default 1,000,000) for profiling input
size and uses CUDA device 0.

```bash
flock /tmp/cudf-gpu.lock cargo run -p tools --features attempt-lifecycle-probe --bin attempt_lifecycle
```

## OTLP metrics adapter

[`otlp_metrics`](src/otlp_metrics.rs) is used by the
`single-node-multi-gpu-validate-run-otlp` feature. It exports the backend's
metric descriptors and cumulative histogram projections without changing their
names, units or buckets. The application owns its SDK provider, resource
attributes, export schedule and endpoint. The adapter is also available under
the `otlp-metrics` feature; its existing protobuf contract tests live in
[`tests/otlp_metrics_exporter_cpu.rs`](tests/otlp_metrics_exporter_cpu.rs).

```bash
cargo nt -E 'binary_id(tools::otlp_metrics_exporter_cpu)'
```
