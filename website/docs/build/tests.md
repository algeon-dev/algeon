---
sidebar_position: 5
sidebar_label: Tests
description: The verified test surface — nextest lanes, GPU serialization, check_all.sh, feature flags, and formatting/lints.
---

# Tests

## The canonical check

One flocked nextest invocation runs everything — CPU test binaries at full
parallelism, GPU test binaries throttled by the `gpu` test group:

```bash
cd datafusion-nexus
flock /tmp/cudf-gpu.lock cargo nextest run --features cugraph,server --no-fail-fast
```

:::warning[Do not run only the GPU group]
`-E 'group(gpu)'` silently skips every non-GPU test binary. Run the full
suite; the group configuration already throttles the GPU lane internally.
:::

For the full local surface — nextest **plus** doctests, rustdoc lints, and
the GPU-lane classification tripwire:

```bash
bash scripts/check_all.sh
GPU_PARALLEL=1 GPU_TEST_THREADS=8 bash scripts/check_all.sh   # opt-in higher GPU fan-out
```

## GPU serialization

The GPU is a single external resource. The `gpu` test group in
`.config/nextest.toml` bounds concurrency *within* one invocation
(`max-threads=4`); the outer `flock /tmp/cudf-gpu.lock` serializes the
invocation against *other* GPU consumers — benchmarks, a running Flight SQL
server, E2E scripts, another checkout's test run. Both are required; neither
replaces the other. The same lock convention applies to `cudf-rs` and
`cugraph-rs` test runs.

GPU-lane membership is **binary-level and explicit**: a test binary belongs
to the `gpu` group iff any of its tests can reach the GPU at runtime.
`scripts/check_gpu_test_lane.sh` (run by `check_all.sh`) fails until every
test binary is classified.

## Per-crate suites

```bash
cargo test -p datafusion-nexus
cargo test -p nexus-query-engine
cargo test --test native_lowering_tests <substring>
```

`cargo nextest run` is the deterministic unit/integration surface. It
intentionally contains **no external-service E2E** — nothing in the Cargo test
targets needs a real AWS account or a running catalog. External E2E lives in
explicit tool invocations instead.

## Feature flags

| Feature | Enables |
| --- | --- |
| `cugraph` | cuGraph graph algorithms callable from SQL |
| `server` | Arrow Flight SQL server |

```bash
cargo run --features server --bin datafusion_nexus_server
```

## Formatting and lints

```bash
cargo +nightly fmt            # rustfmt.toml uses unstable options
cargo clippy --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

## Performance evidence

For any performance claim, use `--release` builds, compare rows produced by
the same harness, and label warmup count, iteration count, source cache
policy, and source chunk profile. See the
[benchmark reference](../reference/benchmarks/).
