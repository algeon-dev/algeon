---
sidebar_position: 6
sidebar_label: Tests
description: The verified test surface — nextest lanes, GPU serialization, check_all.sh, feature flags, and formatting/lints.
---

# Tests

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

This is more than the three test lanes. It runs all isolated policy self-tests,
the dated-nightly rustfmt check, build-support checks, the live canonical
test-binary inventory, a default-member compile, CPU/shared-GPU/
exclusive-GPU tests, locked doctests with exact accounting, strict rustdoc,
clippy, and compile-only checks for the explicit tools and examples packages.
Those last checks compile code but never run tools or contact external services.

Every invocation ends with a compact phase table. Each row records the
effective command, wall time, pass/fail/skipped state, fail-closed test counts
where applicable, GPU cap and lock, selected feature mode, and its persistent
combined-output log. By default logs are kept in a unique
`target/check-all/run.*` directory; set `CHECK_ALL_PHASE_LOG_DIR` to a valid
dedicated directory when automation needs a predictable artifact path. Raw
phase output remains visible while the log is written.

The canonical runner is exactly cargo-nextest 0.9.140. A missing or different
version exits 2 before policy, build, inventory, or test work. Its three test
commands are equivalent to:

```bash
AVAILABLE_CPU_CORES=$(nproc)
CUDF_RUNTIME_POLICY=forbid cargo nextest run --all-features \
  -E 'not group(gpu) & not group(gpu-exclusive)' --no-fail-fast
CUDF_RUNTIME_POLICY=allow flock /tmp/cudf-gpu.lock \
  cargo nextest run -P default --all-features \
  -E 'group(gpu)' --no-fail-fast --test-threads "${AVAILABLE_CPU_CORES}"
CUDF_RUNTIME_POLICY=allow flock /tmp/cudf-gpu.lock \
  cargo nextest run -P default --all-features \
  -E 'group(gpu-exclusive)' --no-fail-fast --test-threads 1
```

Always run all three lanes for full coverage. The harness first runs the
binary-lane classification tripwire, verifies the configured cap with
`nextest show-config`, and applies explicit runtime policies.

## Modes and options

`--fast` is a partial edit loop:

```bash
bash scripts/check_all.sh --fast
```

It runs isolated policy self-tests, pinned rustfmt, build support, the
catalog/generated lane-structure checks, and the CPU nextest lane. It does not
run live nextest inventory, GPU tests, doctests, rustdoc, clippy, or the explicit
tools/examples checks. The CPU lane uses
`CUDF_RUNTIME_POLICY=forbid`, so an accidentally unclassified GPU test fails at
runtime creation before any CUDA query.

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
it does not claim full validation. GPU selections preserve the global lock and
runtime policy. `--lane policy` runs policy/format/structure checks without test
lanes.

Feature selection follows Cargo semantics:

```bash
bash scripts/check_all.sh --features cugraph,server
bash scripts/check_all.sh --no-default-features --features server
bash scripts/check_all.sh --all-features
```

`--all-features` conflicts with the other feature selectors.
`--no-default-features` may accompany `--features`. Any CLI feature selector
replaces, rather than merges with, the compatibility `FEATURES` value.

Shared-GPU tests automatically use every CPU core available to the current
environment. `check_all.sh` validates the positive integer returned by `nproc`
and passes it explicitly to nextest; exclusive tests always use one.
`bash scripts/ci/test_all.sh` is a pure alias that preserves every argument
passed to `check_all.sh`.

Cargo is never a silent fallback. It is available only as an explicit,
noncanonical diagnostic surface:

```bash
bash scripts/check_all.sh --runner cargo-diagnostic
```

This runs the whole non-doctest Cargo suite serially under the GPU lock and
prints `NONCANONICAL`. It cannot select an individual test lane and does not
satisfy full pre-merge validation. `TEST_RUNNER=cargo` remains a compatibility
mapping to this explicit mode.

The compatibility variables `FEATURES`, `TEST_RUNNER`, `GPU_LOCK`, and `STRICT`
remain supported. Every present value is validated before CLI precedence is
applied; a CLI value cannot hide an invalid environment value. Valid CLI values
win. Unknown options, duplicates, conflicts, invalid values, missing option
arguments, and an invalid `nproc` result exit 2 before expensive work.
`--strict` and `STRICT=1` are compatibility spellings because rustdoc is always
strict when that phase runs.

`CHECK_ALL_PHASE_LOG_DIR` controls output location rather than execution
policy. It must be a non-empty single-line path without `|`, which delimits the
final phase table.

The harness also provides a runtime backstop for catalog mistakes. It launches
the CPU nextest command with `CUDF_RUNTIME_POLICY=forbid`; cudf-rs checks that
policy in `RuntimeBuilder::build` before device resolution or CUDA FFI and
returns typed `RuntimeCreationForbidden` / `ExecutionContext` / `Permanent`
metadata. Shared GPU, exclusive GPU, doctest, and cargo-diagnostic commands each
set `CUDF_RUNTIME_POLICY=allow` explicitly, so a caller's ambient value cannot
silently change lane behavior. Invalid or non-Unicode policy values fail closed
as cudf-rs `InvalidRuntimePolicy` errors.

## Doctests and rustdoc

Doctests are GPU-capable until a mechanical classifier proves otherwise. The
canonical harness therefore runs them as a separate, third flocked command:

```bash
CUDF_RUNTIME_POLICY=allow flock /tmp/cudf-gpu.lock \
  cargo test --all-features --doc --no-fail-fast
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
outer lock solve different problems. The same lock convention applies to
`cudf-rs` and `cugraph-rs` test runs.

GPU-lane membership is **binary-level and explicit**. `.config/policies/test-lanes.toml`
is the single admission source: every live nextest binary has one CPU or GPU
entry, every GPU entry is `shared` or `exclusive`, and every entry records a
reason. `scripts/ci/check_gpu_test_lane.sh` (run by `check_all.sh`) fails closed on
missing, duplicate, stale all-feature, or malformed entries and on generated
nextest drift. After editing the catalog, regenerate the scheduling overrides:

```bash
python3 scripts/ci/test_lane_policy.py generate-nextest --write
```

Do not edit the generated override section in `.config/nextest.toml` directly.

## Build-support policy

`scripts/build/rapids.rs` is included by four build scripts: the root adapter,
`nexus-query-engine`, the benchmark package, and the examples package. Its
environment capture is separate from pure direct/aggregate metadata merging,
platform path-list parsing, candidate ordering, and library-identity policy;
Cargo directive emission is a final stage. This keeps the native engine's
build path std-only and prevents DataFusion-specific behavior from crossing
that boundary.

```bash
bash scripts/build/check_build_support.sh
```

The isolated suite reports explicit counts for pure policy tests, all four
inclusion compile smokes, and fake-library build-script scenarios. Fixtures
cover canonical paths, same inodes, byte-identical and divergent files,
required RMM failure, optional `rapids_logger` linkage, direct/aggregate
metadata precedence, and deterministic multi-candidate handling. It neither
loads nor modifies the real RAPIDS libraries.

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
cargo +nightly-2026-07-13 fmt --all --check  # canonical pinned check
cargo +nightly fmt --all                    # compatible developer command
cargo clippy --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --keep-going
```

Stable is pinned in `rust-toolchain.toml`; the canonical rustfmt nightly is
pinned separately in `rustfmt-toolchain.toml`.

## Continuous enforcement

The repository-visible `Validation` GitHub Actions workflow runs on every pull
request, merge-group candidate, and push to `main`, plus a nightly schedule.
Its stable status names and owners are:

| Required status | Runner | Contract |
| --- | --- | --- |
| `Validation / Policy and format` | GitHub-hosted Ubuntu | isolated policy suites, pinned rustfmt, and lane structure |
| `Validation / Website docs` | GitHub-hosted Ubuntu | clean Docusaurus install and build with `CI=true` |
| `Validation / CPU, lints, and rustdoc` | self-hosted `datafusion-nexus-cpu` | CPU lane, all-feature compile, strict rustdoc, clippy, tools, and examples |
| `Validation / Canonical GPU full` | self-hosted `datafusion-nexus-gpu` | plain `bash scripts/check_all.sh`, including the locked GPU lanes and doctests |

The native runners are provisioned with the sibling `cudf-rs`, `cugraph-rs`,
and `iceberg-rust` checkouts, matching native libraries, pinned Rust
toolchains, Python 3.12+, and cargo-nextest 0.9.140. The GPU runner must join
the same `/tmp/cudf-gpu.lock` convention as every other GPU consumer on that
host. The workflow verifies the expected checkout/artifact layout before it
runs the harness.

Repository administrators must require all four status names above on `main`.
The workflow is the versioned enforcement definition; branch-protection and
self-hosted-runner registration remain GitHub repository settings and cannot
be inferred from a green local run. External-service acceptance remains a
separately provisioned explicit `datafusion-nexus-tools` workflow, never a
Cargo test target.

## Performance evidence

For any performance claim, use `--release` builds, compare rows produced by
the same harness, and label warmup count, iteration count, source cache
policy, and source chunk profile. See the
[benchmark reference](../reference/benchmarks/).
