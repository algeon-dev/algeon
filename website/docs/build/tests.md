---
sidebar_position: 5
sidebar_label: Tests
description: The verified test surface — nextest lanes, GPU serialization, check_all.sh, feature flags, and formatting/lints.
---

# Tests

Run these commands from the recursive checkout after configuring the build
shell in [Building from Source](./from-source.md#2-configure-the-build-shell).

## Tooling layout

`scripts/check_all.sh` is deliberately the only top-level script entry point.
Supporting files are grouped by responsibility:

- `scripts/build/` contains shared Cargo build support and packaging policy.
- `scripts/ci/` contains validation policies and parsers; `scripts/ci/tests/`
  contains their isolated fault-injection suites.
- `scripts/dev/` contains manually invoked developer and workload helpers.

Repository-owned policy data is under `.config/policies/`; standard tool
configuration such as `.config/nextest.toml` remains at its tool-defined path.
The concise directory map is maintained in `scripts/README.md`.

## The canonical full check

With no arguments, the harness runs the required full pre-merge surface:

```bash
cd datafusion-nexus
bash scripts/check_all.sh
```

The no-argument full gate adds `msrv-check` and `feature-isolation-check` after
the native/component structural checks and before nextest inventory and test
lanes. Both are compile-only phases: they do not run tests, contact external
services, or acquire the GPU lock. The phase table names these rows exactly.

`msrv-check` uses the workspace's declared Rust version (currently Rust 1.94),
not a second version constant maintained in this page:

```bash
cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo
```

`feature-isolation-check` is the repository-owned exception to the usual
all-features Cargo rule. Its exact command checks the live workspace's
no-feature and non-default feature configurations and proves compilation only;
it does not prove tests:

```bash
cargo hack check --workspace --each-feature --exclude-features default --locked --keep-going
```

The existing policy, build, test, doctest, rustdoc, public API, and Clippy
phases use one workspace-wide feature selection, so tests from bench, tools,
and examples packages remain part of the same surface. Operator and
external-service E2E tools are still explicit commands and are not run by this
gate.

The installation, diagnostic, and edit-loop examples for cargo-hack use the
repository's `cargo hack` command directly; they do not invoke a sibling
checkout or a local absolute path. Direct GPU tests and the full gate retain
the existing `flock /tmp/cudf-gpu.lock` contract.

Every invocation ends with a compact phase table. Each row records the
effective command, wall time, pass/fail/skipped state, fail-closed test counts
where applicable, GPU cap and lock, selected feature mode, and its persistent
combined-output log. By default logs are kept in a unique
`target/check-all/run.*` directory; set `CHECK_ALL_PHASE_LOG_DIR` to a valid
dedicated directory when automation needs a predictable artifact path. Raw
phase output remains visible while the log is written.

The canonical runner is exactly cargo-nextest 0.9.140. A missing or different
version exits 2 before policy, build, inventory, or test work. Its three nextest
commands are equivalent to:

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

The harness checks the live binary inventory before running lanes and verifies
the configured cap with `nextest show-config` before GPU work.

## Modes and options

`--fast` is a partial edit loop:

```bash
bash scripts/check_all.sh --fast
bash scripts/check_all.sh --fast --no-default-features
bash scripts/check_all.sh --fast --features iceberg
```

It runs isolated policy self-tests, nightly rustfmt, build support, the
catalog/generated lane-structure checks, and the CPU nextest lane. It does not
run `msrv-check` or `feature-isolation-check`, live nextest inventory, GPU tests,
doctests, rustdoc, public API snapshots, or clippy. It still checks the
committed lane catalog and generated scheduling overrides before the test
command runs. It remains partial validation.

By default `--fast` still uses `--all-features` for the packages it builds. That
narrows *which tests execute*, not necessarily *which unrelated test binaries
Cargo links*. For a true single-harness edit loop, use Cargo target selectors
(below) rather than assuming nextest filtering eliminated link work.

Use `--lane` for an intentionally partial lane-only run:

```bash
bash scripts/check_all.sh --lane cpu
bash scripts/check_all.sh --lane gpu
bash scripts/check_all.sh --lane gpu-shared
bash scripts/check_all.sh --lane gpu-exclusive
bash scripts/check_all.sh --lane policy
bash scripts/check_all.sh --lane all
```

Even `--lane all` runs only the three test lanes and their required preflight;
it does not run `msrv-check` or `feature-isolation-check` and does not claim
full validation. GPU selections preserve the global lock.
`--lane policy` runs policy/format/structure checks without test lanes.

Feature selection follows Cargo semantics:

```bash
bash scripts/check_all.sh --features cugraph,iceberg
bash scripts/check_all.sh --no-default-features --features iceberg
bash scripts/check_all.sh --all-features
```

`--all-features` conflicts with the other feature selectors.
`--no-default-features` may accompany `--features`. Any CLI feature selector
replaces, rather than merges with, the compatibility `FEATURES` value for the
existing workspace phases. In full mode, `feature-isolation-check` still runs
its complete package-local matrix regardless of a custom feature selector; the
default selection for the existing phases is `--all-features`.

Shared-GPU tests automatically use every CPU core available to the current
environment. `check_all.sh` validates the positive integer returned by `nproc`
and passes it explicitly to nextest; exclusive tests always use one.
CI helpers under `scripts/ci/` are internal callers. Developers use the root
`scripts/check_all.sh` entry point.

Cargo is never a test-runner fallback. The admitted test surface requires
`cargo-nextest` per-test-process isolation: allocator lifecycle cases mutate a
process-global `DeviceMemoryManager`, and native execution cases intentionally
retain one backend policy per process. Serial libtest threads cannot reset that
state. `--runner cargo-diagnostic`, `TEST_RUNNER=cargo`, and
`TEST_RUNNER=cargo-diagnostic` therefore fail closed before any test work.

The compatibility variables `FEATURES`, `TEST_RUNNER`, `GPU_LOCK`, and `STRICT`
remain recognized. `TEST_RUNNER=nextest` is its only accepted runner value;
`--runner nextest` can explicitly replace a retired runner environment value.
Every present value is validated before expensive work. Unknown options,
duplicates, conflicts, invalid values, missing option arguments, and an invalid
`nproc` result exit 2 before expensive work.
`--strict` and `STRICT=1` are compatibility spellings because rustdoc is always
strict when that phase runs.

`CHECK_ALL_PHASE_LOG_DIR` controls output location rather than execution
policy. It must be a non-empty single-line path without `|`, which delimits the
final phase table.

The binary-lane catalog is the admission control for CPU and GPU tests. Keep it
complete and run the live inventory check when changing test binaries: without
a component-local runtime-policy backstop, an incorrectly classified CPU test
is not rejected at CUDA-runtime creation.

## Doctests and rustdoc

Doctests are GPU-capable until a mechanical classifier proves otherwise. The
canonical harness therefore runs them as a separate, third flocked command:

```bash
flock /tmp/cudf-gpu.lock \
  cargo test --workspace --all-features --doc --no-fail-fast
```

`check_all.sh` preserves the command's combined output and exit status, parses
each `Doc-tests` harness result, and reports exact per-harness and aggregate
counts. `executed` is `passed + failed + measured`; an actual zero is reported
as `executed=0`, not treated as an assumed or skipped result. Missing result
lines, no complete harness, duplicate harnesses, malformed output, and
inconsistent outcomes fail closed.

Rustdoc is independently strict. The harness appends `-D warnings` after any
caller-provided `RUSTDOCFLAGS`, so an earlier `-A warnings` cannot weaken the
canonical check. `STRICT=1` is accepted for compatibility but changes no
behavior; leave it unset for normal use. Other `STRICT` values are rejected
before test work starts.

## GPU serialization

The GPU is a single external resource. The shared `gpu` test group in
`.config/nextest.toml` uses `max-threads = "num-cpus"`. The harness resolves the
environment's available CPU count with `nproc`, verifies that
`nextest show-config` still reports the dynamic `num-cpus` cap, and passes the
detected numeric value through `--test-threads`. The `gpu-exclusive` group retains
`max-threads=1` for binaries requiring process-global or whole-device isolation.
The harness runs the shared group first and the exclusive group second as
distinct locked invocations.

The outer `flock /tmp/cudf-gpu.lock` serializes each GPU lane against *other*
GPU consumers — benchmarks, a running Flight SQL server, E2E scripts, or
another checkout's test run. The inner cap, separate resource-class runs, and
outer lock solve different problems. The same lock convention applies to the
GPU test binaries in the `components/*` crates.

GPU-lane membership is **binary-level and explicit**. `.config/policies/test-lanes.toml`
is the single admission source: every live pre-merge-admitted nextest binary
in the selected workspace surface has one CPU or GPU entry, every GPU entry is
`shared` or `exclusive`, and every entry records a reason.
`scripts/ci/check_gpu_test_lane.sh` (run by `check_all.sh`) fails closed on
unclassified, conflicting, stale, or malformed entries and invalid nextest scheduling.
Validate the policy catalog with:

```bash
python3 scripts/ci/test_lane_policy.py check-catalog
```

## Build-support policy

`scripts/build/rapids.rs` is included by six build scripts: the root adapter,
`nexus-query-engine`, server, benchmark, tools, and examples packages. Its
environment capture is separate from pure direct/aggregate metadata merging,
platform path-list parsing, candidate ordering, and library-identity policy;
Cargo directive emission is a final stage. This keeps the native engine's
build path std-only and prevents DataFusion-specific behavior from crossing
that boundary.

```bash
bash scripts/build/check_build_support.sh
```

The isolated suite reports explicit counts for pure policy tests, five
inclusion compile smokes, and fake-library build-script scenarios. Fixtures
cover canonical paths, same inodes, byte-identical and divergent files,
required RMM failure, optional `rapids_logger` linkage, direct/aggregate
metadata precedence, and deterministic multi-candidate handling. It neither
loads nor modifies the real RAPIDS libraries.

## Per-crate suites and targeted edit loops

```bash
bash scripts/check_all.sh --fast
bash scripts/check_all.sh --lane gpu-shared
```

### One package, one harness

Prefer Cargo package and harness selectors for a focused edit loop while keeping
the workspace's all-feature artifact graph:

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

# Repository policy/fmt/CPU partial loop.
bash scripts/check_all.sh --fast
```

Targeted GPU work must still take `/tmp/cudf-gpu.lock`, preferably through
`bash scripts/check_all.sh --lane gpu-shared` or `--lane gpu-exclusive`.

Workspace `[profile.test]` is intentionally absent. Tests inherit
`[profile.dev]` with line-table debug information and unpacked split debuginfo,
which preserves file:line diagnostics without full debug records.

### Compile delay versus test delay

If a command sits at

```text
Finished `test` profile [unoptimized + debuginfo] target(s) in …
```

before any `PASS`/`FAIL`, the cost is Cargo compile/link, not nextest execution.
Expose the first dirty unit with:

```bash
CARGO_LOG=cargo::core::compiler::fingerprint=info \
  cargo nextest run --workspace --all-features --no-run 2>&1 \
  | rg 'fingerprint dirty|dirty:|Compiling'
```

Fingerprint logs can contain full local environment values such as `PATH`.
Redact them before attaching to an issue. The root component-support helper
documents the live matrix as `bash scripts/build/check_fingerprint_matrix.sh`
and the static contract as
`cargo test -p cudf-nexus-sys --all-features --test fingerprint_matrix_contract`.

### Target artifact maintenance

Stale feature/hash variants and interrupted mold links can inflate `target/`
without reflecting the active surface. Report and clean **explicitly**:

```bash
bash scripts/dev/clean_target_artifacts.sh            # sizes + mold temps
bash scripts/dev/clean_target_artifacts.sh --mold-only --yes
bash scripts/dev/clean_target_artifacts.sh --clean-cargo-target --yes
```

Never treat cleanup as a side effect of a test command. After cleanup the next
build is cold. An advisory size budget should be taken from a clean rebuild of
the active all-feature surface, not from a multi-day dirty `target/` tree.
`sccache` (or equivalent) can help clean CI and branch switches; it does not
remove the cost of linking many large test binaries.

Do not use broad `cargo test --all-features` for `datafusion-nexus`: libtest runs cases in a
binary as threads, and serial threads still cannot restore the engine's
process-global allocator or policy fixtures. The retired Cargo diagnostic
runner fails closed for that reason. `cargo nextest run --all-features` is the deterministic
unit/integration surface, but a bare invocation does **not** acquire the
cross-consumer GPU lock; use `check_all.sh` for repository validation. It
intentionally contains **no external-service E2E** — nothing in the Cargo test
targets needs a real AWS account or a running catalog. External E2E lives in
explicit tool invocations instead.

## Provisioned workload acceptance

Scale-factor datasets are not Cargo tests. They are explicit acceptance runs:

```bash
NEXUS_PARQUET_DIR=/path/to/complete/tpcds-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-bench --all-features --bin native_tpcds_parity_report

NEXUS_PARQUET_DIR=/path/to/complete/tpch-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-bench --all-features --bin native_tpch_execution_report
```

Both commands validate their dataset before the workload begins and fail when
it is incomplete. The TPC-DS report emits one row per attempted query, so its
TSV distinguishes attempted, matched, and failed query IDs rather than
converting an unavailable fixture into a passed Rust test.

Sustained-load measurement is a second explicit acceptance run. The
`query_service_goodput` binary replays a seeded open-loop arrival trace against
one backend and reports goodput, the outcome taxonomy, latency percentiles, and
per-attempt allocator peaks. It runs in two modes: `calibrate` freezes each
item's single-flight median latency and output fingerprint into a calibration
file, and `measure` compares a configuration against that frozen file and
refuses to run if the live GPU or dataset differs from the calibrated one.

```bash
crates/nexus-bench/scripts/run_goodput_matrix.sh calibrate g1
```

The driver script pins every factor that is not the one being compared, records
the source, GPU, and native-library identity next to the results, and writes one
summary row per comparison group.

Python-cuGraph comparisons are similarly provisioned acceptance coverage. They
are explicit tools commands, not Cargo test targets, and must run only on a
host with Python cuGraph installed:

```bash
flock /tmp/cudf-gpu.lock \
  cargo run -p nexus-tools --all-features -- \
  cugraph-fixture-e2e bfs-include-edges
```

An unavailable oracle is a failure for that selected acceptance run, never a
successful skipped assertion. Replace `bfs-include-edges` with `pagerank`,
`bfs-utf8-include-edges`, `bfs-utf8-path`, or `bfs-utf8-predicates` for the
other provisioned cases.

## Feature flags

| Feature | Enables |
| --- | --- |
| `cugraph` | cuGraph graph algorithms callable from SQL |
| `cuvs` | cuVS bindings and vector-function planning |
| `iceberg` | Iceberg catalogs and native scan integration |
| `nvml` | Optional NVML device diagnostics |

The Arrow Flight SQL server is the separate `nexus-server` package; its
`cugraph`, `cuvs`, `iceberg`, and `nvml` features forward the corresponding
adapter and engine capabilities.

```bash
cargo run -p nexus-server --all-features --bin datafusion_nexus_server
```

## Formatting and lints

```bash
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --keep-going
```

The MSRV is Rust 1.94. Formatting requires nightly.

## Continuous enforcement

The only repository-visible GitHub Actions workflow currently builds and
deploys the website on matching pushes to `main` or manual dispatch. It does
not run the Rust test suite. No committed workflow currently turns the
canonical pre-merge surface into required GitHub status checks, so use plain
`bash scripts/check_all.sh` for that validation. External-service acceptance
remains an explicit `nexus-tools` invocation, never a Cargo test
target.

## Performance evidence

For any performance claim, use `--release` builds, compare rows produced by
the same harness, and label warmup count, iteration count, source cache
policy, and source chunk profile. See the
[benchmark reference](../reference/benchmarks/).
