---
sidebar_position: 5
sidebar_label: Tests
description: The single full gate, what it runs and prints, the partial edit loops, feature flags, and the maintainer diagnostics behind the harness.
---

# Tests

Before you start:

- The [build shell](./from-source.md#2-configure-the-build-shell) is
  configured and all three native layers plus the Rust workspace are built.
- `cargo-nextest` 0.9.140 and `cargo-hack` 0.6.45 are installed
  ([Prerequisites](./prerequisites.md#rust-toolchain-and-test-runner)). The
  gate checks both versions and exits with status 2 before any work if they
  differ.
- Nothing else is using the GPU. The gate serializes its own GPU phases with
  `flock /tmp/cudf-gpu.lock`; a benchmark, a running Flight SQL server, or
  another checkout's gate holding that lock blocks it until they finish.

## The full gate

Run it bare from the checkout root. Do not wrap it in `flock`: it takes
`/tmp/cudf-gpu.lock` itself for every GPU phase, so an outer hold deadlocks it.

```bash
bash scripts/check_all.sh
```

With no arguments it runs, in order: isolated policy self-tests, native-source
and checked-bindings verification, extracted component package checks, a
docs.rs-native-free workspace compile check, `msrv-check`, `feature-isolation-check`, nightly
`rustfmt --check`, lane-catalog structure checks, live nextest inventory, GPU
scheduling preflight, the CPU nextest lane, the shared-GPU and exclusive-GPU
nextest lanes, doctests, strict rustdoc, public API surface snapshots, and
clippy with `-D warnings`. Every Cargo phase uses `--all-features`; the one
exception is `feature-isolation-check`, described under
[Harness internals](#harness-internals).

### What success looks like

Every run prints a `CHECK CONFIG` line, then raw phase output, then a compact
phase table with one row per phase (effective command, wall time, status,
fail-closed test counts where applicable, GPU cap and lock, and the path of
that phase's combined-output log). A passing full run ends with:

```text
OK: canonical full pre-merge surface passed.
```

Partial modes end with `OK: ... PARTIAL: full pre-merge validation was not
run.` and never satisfy the pre-merge requirement.

### On failure

The run continues past a failed phase, then prints a `FAILURES` section that
repeats each failed phase's command, log path, and, for test lanes, the failed
test names. Logs are kept in a unique `target/check-all/run.*` directory; set
`CHECK_ALL_PHASE_LOG_DIR` to a dedicated directory when automation needs a
predictable path. The script exits non-zero after:

```text
FAIL: canonical full pre-merge surface reported problems.
```

- A native-library or CUDA-architecture failure in the GPU lanes is usually a
  stale or mismatched prefix: see [Troubleshooting](./troubleshooting.md).
- A single failing harness is fastest to iterate on with the
  [targeted edit loops](#targeted-edit-loops) below, then rerun the full gate.
- An exit with status 2 before any phase is an argument, environment, or
  tool-version error; the message names the exact fix.

## Partial edit loops

`--fast` runs the policy self-tests, native-source check, nightly `rustfmt
--check`, the lane-structure checks, and the CPU nextest lane. It skips
`msrv-check`, `feature-isolation-check`, live inventory, every GPU lane,
doctests, rustdoc, public API snapshots, and clippy:

```bash
bash scripts/check_all.sh --fast
```

`--lane` runs one test lane (or all three) with only its required preflight;
it never runs the compile-only checks, doctests, rustdoc, or clippy. GPU
selections keep the global lock. `--lane policy` runs the policy, format, and
structure checks without any test lane.

```bash
bash scripts/check_all.sh --lane cpu
bash scripts/check_all.sh --lane gpu            # shared then exclusive
bash scripts/check_all.sh --lane gpu-shared
bash scripts/check_all.sh --lane gpu-exclusive
bash scripts/check_all.sh --lane policy
bash scripts/check_all.sh --lane all
```

`--fast` and `--lane` conflict; any other option, a duplicate option, or an
unknown lane exits with status 2. Feature selection is not an option: the
gate always uses `--all-features`.

### Targeted edit loops

Select one package and one harness while keeping the workspace's all-feature
artifact graph:

```bash
# One CPU adapter planning harness.
cargo nextest run -p datafusion-nexus --all-features \
  --test native_planning_tests

# CPU adapter contracts (errors, policy reports, public facade, table-format).
cargo nextest run -p datafusion-nexus --all-features \
  --test cpu_contract_tests

# Pure native-engine contract harness.
cargo nextest run -p nexus-query-engine --all-features \
  --test native_contract_tests
```

A bare `cargo nextest run` does not take the cross-consumer GPU lock. Run GPU
harnesses through `bash scripts/check_all.sh --lane gpu-shared` or
`--lane gpu-exclusive`, or wrap the direct command in
`flock /tmp/cudf-gpu.lock`.

Never use plain `cargo test` for the workspace's non-doc tests: libtest runs a
binary's cases as threads, and serial threads cannot restore the engine's
process-global allocator or backend policy fixtures. The admitted test surface
requires nextest's per-test-process isolation.

Formatting, lints, and docs match the gate's phases:

```bash
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --keep-going
```

## Feature flags

| Feature | Enables |
| --- | --- |
| `cugraph` | cuGraph graph algorithms callable from SQL |
| `cuvs` | cuVS SQL functions (exact kNN, KMeans, PCA) and their GPU execution |
| `iceberg` | Iceberg catalogs and native scan integration |
| `nvml` | Optional NVML device diagnostics |

The Arrow Flight SQL server is the separate `nexus-server` package; its
`cugraph`, `cuvs`, `iceberg`, and `nvml` features forward the corresponding
adapter and engine capabilities.

```bash
cargo run -p nexus-server --all-features --bin datafusion_nexus_server
```

## Acceptance runs outside the gate

Nothing in the Cargo test targets needs a real dataset, cloud account,
running catalog, or Python cuGraph. Those are explicit acceptance runs, and
each takes the GPU lock itself.

Scale-factor workloads validate their dataset before starting and fail when it
is incomplete; the TPC-DS report emits one row per attempted query:

```bash
NEXUS_PARQUET_DIR=/path/to/complete/tpcds-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-bench --all-features --bin native_tpcds_parity_report

NEXUS_PARQUET_DIR=/path/to/complete/tpch-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-bench --all-features --bin native_tpch_execution_report
```

Sustained-load measurement uses the `query_service_goodput` binary through
its driver script, which pins every factor except the one under comparison
and records source, GPU, and native-library identity next to the results.
`calibrate` freezes each item's single-flight median latency and output
fingerprint; the measurement groups then refuse to run if the live GPU or
dataset differs from that calibration:

```bash
crates/nexus-bench/scripts/run_goodput_matrix.sh calibrate g1
```

Python-cuGraph comparisons run only on a host with Python cuGraph installed;
an unavailable oracle fails the run rather than skipping it:

```bash
flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-tools --all-features -- \
  cugraph-fixture-e2e bfs-include-edges
```

Replace `bfs-include-edges` with `pagerank`, `bfs-utf8-include-edges`,
`bfs-utf8-path`, or `bfs-utf8-predicates` for the other provisioned cases.

For any performance claim, use `--release` builds, compare rows produced by
the same harness, and label warmup count, iteration count, source cache
policy, and source chunk profile. See the
[benchmark reference](../reference/benchmarks/).

## Maintainer reference

Nothing below is needed to build, validate, or ship a change. It documents
how the harness reaches its verdicts, for people changing the harness, the
lane catalog, or a build script.

### Harness internals

`scripts/check_all.sh` is the only top-level script entry point. Supporting
files are grouped by responsibility, and the directory map is maintained in
[`scripts/README.md`](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/scripts/README.md):
`scripts/build/` (build support and packaging policy), `scripts/ci/`
(validation policies and parsers, with their fault-injection suites under
`scripts/ci/tests/`), and `scripts/dev/` (manually invoked helpers). Policy
data lives in `.config/policies/`; tool configuration such as
`.config/nextest.toml` stays at its tool-defined path.

The three nextest lanes are equivalent to:

```bash
AVAILABLE_CPU_CORES=$(nproc)
cargo nextest run --workspace --all-features \
  -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast
flock /tmp/cudf-gpu.lock \
  cargo nextest run -P default --workspace --all-features \
  -E 'group(gpu)' --no-fail-fast --test-threads "${AVAILABLE_CPU_CORES}"
flock /tmp/cudf-gpu.lock \
  cargo nextest run -P default --workspace --all-features \
  -E 'group(gpu-exclusive)' --no-fail-fast --test-threads 1
```

The shared `gpu` test group in `.config/nextest.toml` uses
`max-threads = "num-cpus"`; the harness validates the positive integer from
`nproc`, confirms with `nextest show-config` that the dynamic cap is still in
force, and passes the value explicitly. The `gpu-exclusive` group keeps
`max-threads = 1` for binaries that mutate process-global allocator state or
need whole-device isolation. The outer `flock` serializes each GPU lane against
other GPU consumers; the inner cap and the two separate invocations solve a
different problem, and the same lock convention applies to the GPU test
binaries under `components/*`.

Lane membership is binary-level and explicit.
`.config/policies/test-lanes.toml` is the single admission source: every live
pre-merge-admitted nextest binary has one CPU or GPU entry, every GPU entry
is `shared` or `exclusive`, and every entry records a reason.
`scripts/ci/check_gpu_test_lane.sh` (the `live-inventory` phase) fails closed
on unclassified, conflicting, stale, or malformed entries. Keep the catalog
complete when adding test binaries; without a component-local runtime
backstop, a misclassified CPU test is not rejected at CUDA-runtime creation.
Validate the catalog alone with:

```bash
python3 scripts/ci/test_lane_policy.py check-catalog
```

The two compile-only phases do not run tests, contact external services, or
take the GPU lock. `msrv-check` uses the workspace's declared Rust version:

```bash
cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo
```

`feature-isolation-check` is the one repository-owned exception to the
all-features rule; it proves each package's no-feature and non-default feature
configurations compile and proves nothing about tests:

```bash
cargo hack check --workspace --each-feature --exclude-features default --locked --keep-going
```

Doctests are GPU-capable until a mechanical classifier proves otherwise, so
they run as a third locked command:

```bash
flock /tmp/cudf-gpu.lock \
  cargo test --workspace --all-features --doc --no-fail-fast
```

The harness parses each `Doc-tests` result and reports exact per-harness and
aggregate counts; `executed` is `passed + failed + measured`, an actual zero is
reported as `executed=0`, and missing, duplicate, malformed, or inconsistent
harness output fails closed. The same fail-closed counting applies to the
nextest lanes' phase-table counts.

Rustdoc is always strict: the harness appends `-D warnings` after any
caller-provided `RUSTDOCFLAGS`, so an earlier `-A warnings` cannot weaken it.

`scripts/build/rapids.rs` is the std-only build-script support included by the
root adapter, `nexus-query-engine`, server, benchmark, tools, and examples
packages. Its isolated policy suite runs as one of the policy self-tests and
can be run alone; it neither loads nor modifies the real RAPIDS libraries:

```bash
bash scripts/build/check_build_support.sh
```

### Compile delay versus test delay

If a command sits at `Finished \`test\` profile ... target(s) in …` before any
`PASS`/`FAIL`, the cost is Cargo compile or link, not nextest execution.
Workspace `[profile.test]` is intentionally absent: tests inherit
`[profile.dev]` with line-table debug information and unpacked split
debuginfo. Expose the first dirty unit with:

```bash
CARGO_LOG=cargo::core::compiler::fingerprint=info \
  cargo nextest run --workspace --all-features --no-run 2>&1 \
  | rg 'fingerprint dirty|dirty:|Compiling'
```

Fingerprint logs can contain full local environment values such as `PATH`;
redact them before attaching to an issue. The live fingerprint matrix for
`cudf-nexus-sys` is
[`scripts/build/check_fingerprint_matrix.sh`](https://github.com/DataFusion-Nexus/datafusion-nexus/blob/main/scripts/build/check_fingerprint_matrix.sh)
(requires `CUDF_INSTALL_DIR`, `CC`, and `CXX`), and its static contract is:

```bash
cargo test -p cudf-nexus-sys --all-features --test fingerprint_matrix_contract
```

### Target artifact maintenance

Stale feature/hash variants and interrupted mold links can inflate `target/`
without reflecting the active surface. Cleanup is never a side effect of a
test command; report and clean explicitly:

```bash
bash scripts/dev/clean_target_artifacts.sh            # sizes + mold temps
bash scripts/dev/clean_target_artifacts.sh --mold-only --yes
bash scripts/dev/clean_target_artifacts.sh --clean-cargo-target --yes
```

After cleanup the next build is cold. Take any advisory size budget from a
clean rebuild of the active all-feature surface, not from a multi-day dirty
`target/` tree.

### Continuous enforcement

The only committed GitHub Actions workflow builds and deploys the website. No
workflow turns the full gate into a required status check, so
`bash scripts/check_all.sh` on the developer host is the pre-merge validation.
