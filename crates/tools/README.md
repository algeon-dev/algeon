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
