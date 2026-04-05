---
sidebar_position: 3
---

# Build & Test

```bash
cargo build
cargo test
cargo test -p datafusion-nexus
cargo test -p nexus-query-engine
```

`cargo test --all-features` is the deterministic unit/integration surface. For
local turnaround, prefer the split `cargo nextest` lanes:

```bash
cargo nextest run --all-features -E 'not group(gpu)' --no-fail-fast
flock /tmp/cudf-gpu.lock cargo nextest run --all-features -E 'group(gpu)' --no-fail-fast
```

## GPU is a single external resource

Anything touching CUDA (GPU `cargo test`, the `gpu` nextest group, native
benchmarks, execution reports) must be serialized against other GPU users —
wrap it with `flock /tmp/cudf-gpu.lock <cmd>` when running concurrently.

## Feature flags

| Feature | Enables |
| --- | --- |
| `iceberg` | Iceberg / Glue / S3 support |
| `server` | Flight SQL server |
| `cugraph` | cuGraph graph algorithms |

## Formatting & lints

```bash
cargo +nightly fmt          # rustfmt.toml uses unstable options
cargo clippy --all-targets
```
