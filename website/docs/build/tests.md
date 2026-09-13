---
sidebar_position: 5
sidebar_label: Tests
description: The single full gate, what it runs and prints, the partial edit loop, feature flags, and the maintainer diagnostics behind the harness.
---

# Tests

Before you start:

- The [build shell](./from-source.md#2-configure-the-build-shell) is
  configured and all three native layers plus the Rust workspace are built.
- `cargo-nextest` 0.9.144 and `cargo-hack` 0.6.45 are installed
  ([Prerequisites](./prerequisites.md#rust-toolchain-and-test-runner)). The
  full gate checks both versions and exits with status 2 before any work if
  they differ; `--fast` checks only `cargo-nextest`.
- Nothing else is using the GPU. GPU tests take `/tmp/cudf-gpu.lock` through
  nextest's run-wrapper (shared for the `gpu` test group, exclusive for
  `gpu-exclusive`); a benchmark, a running Flight SQL server, or another
  checkout's gate holding that lock blocks a GPU test until it is free.

## The full gate

Run it bare from the checkout root. Do not wrap it in `flock`: its test phase
takes `/tmp/cudf-gpu.lock` itself through the same nextest run-wrapper, so an
outer hold deadlocks it.

```bash
bash scripts/check_all.sh
```

With no arguments it starts every phase at once, in four groups:

- **policy**: rustfmt, native-source verification, checked bindings, and the
  policy self-test scripts, run independently in parallel.
- **tests**: `cargo nt -P gate` runs the full workspace under the `gate`
  nextest profile, then doctests run through the GPU lock wrapper.
- **static**: clippy with `-D warnings`, then strict rustdoc, then the public
  API surface snapshot, in sequence.
- **hygiene**: the six feature-isolation shards plus their union check,
  `msrv-check`, a docs.rs native-free compile check, and the component-package
  check, each in its own target directory.

Hygiene never blocks tests. Every Cargo phase uses `--workspace
--all-features`; the one exception is feature-isolation, described under
[Harness internals](#harness-internals).

### What success looks like

Each phase logs to `target/check-all/run.*/<phase>.log`; `target/check-all/latest`
symlinks the newest run. That directory's `summary.tsv` (`phase`, `status`,
`started`, `duration`, `log`, with `gate` itself as the first row) is
rewritten every time a phase starts or finishes; poll it to watch a gate
started under `nohup`. A passing full run ends with:

```text
OK: full pre-merge gate passed.
```

`--fast` ends with `OK: fast checks passed. PARTIAL: this is not full
pre-merge validation.` and never satisfies the pre-merge requirement.

### On failure

The run continues past a failed phase, prints each phase's status and
duration, then lists the failed phases with their log paths. For a failed
`tests` or `cpu-tests` phase it also prints that phase's nextest summary and
the exact rerun command, `cargo nt -P gate -R latest` for `tests` or
`cargo nt -E 'not group(gpu) & not group(gpu-exclusive)' -R latest` for
`cpu-tests`. The script exits non-zero after:

```text
FAIL: full pre-merge gate reported problems.
```

- A native-library or CUDA-architecture failure in the `tests` phase is
  usually a stale or mismatched prefix: see [Troubleshooting](./troubleshooting.md).
- A single failing phase is fastest to iterate on with the
  [targeted edit loops](#targeted-edit-loops) below, then rerun the full gate.
- An exit with status 2 before any phase is an argument, environment, or
  tool-version error; the message names the exact fix.
- `cargo nextest replay` replays a stored run's per-test results without
  rerunning anything, for a closer look at what a run produced.

## Partial edit loop

`--fast` runs rustfmt, the native-source check, the policy self-test scripts,
and the CPU nextest tests (`cargo nt -E 'not group(gpu) & not
group(gpu-exclusive)'`). It skips `msrv-check`, feature-isolation, every GPU
test, doctests, rustdoc, the public API snapshot, and clippy:

```bash
bash scripts/check_all.sh --fast
```

Any option other than `--fast`, or a duplicate option, exits with status 2.
Feature selection is not an option: the gate always uses `--all-features`.

### Targeted edit loops

`.config/nextest.toml` is the only test-lane source, and every command below
selects with a `cargo nt` filterset (never `-p`, which resolves features
differently and rebuilds):

```bash
cargo nt -E 'package(<package>)'                              # what you changed
cargo nt -E 'rdeps(<package>)'                                # everything downstream
cargo nt -E 'binary_id(<package>::<target>) & test(<name>)'   # one test
cargo nt -E 'not group(gpu) & not group(gpu-exclusive)'       # CPU only
cargo nt -E '<same filterset>' -R latest                      # rerun only failures
```

A rerun must repeat the exact filterset used originally: with a different (or
missing) `-E`, nextest treats binaries the earlier filter skipped as new and
reruns the whole suite. After a gate failure, `cargo nt -P gate -R latest`
reruns only what that run left failed or unfinished.

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
| `nvml` | Optional NVML device diagnostics |

The Arrow Flight SQL server is launched by the separate `algeon-server` package; its
`cugraph`, `cuvs`, and `nvml` features forward the corresponding
adapter and engine capabilities.

Iceberg remains part of the supported source design. Its Cargo feature and
release acceptance resume after the upstream iceberg-rust 0.12 release is
available and validated.

```bash
cargo run -p algeon-server --all-features --bin algeon_server
```

## Acceptance runs outside the gate

Nothing in the Cargo test targets needs a real dataset, cloud account,
running catalog, or Python cuGraph. Those are explicit acceptance runs, and
each takes the GPU lock itself.

Scale-factor workloads validate their dataset before starting and fail when it
is incomplete; the TPC-DS report emits one row per attempted query:

```bash
ALGEON_PARQUET_DIR=/path/to/complete/tpcds-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p bench --all-features --bin native_tpcds_parity_report

ALGEON_PARQUET_DIR=/path/to/complete/tpch-sf1 \
  flock /tmp/cudf-gpu.lock \
  cargo run -p bench --all-features --bin native_tpch_execution_report
```

Sustained-load measurement uses the `query_service_goodput` binary through
its driver script, which pins every factor except the one under comparison
and records source, GPU, and native-library identity next to the results.
`calibrate` freezes each item's single-flight median latency and output
fingerprint; the measurement groups then refuse to run if the live GPU or
dataset differs from that calibration:

```bash
crates/bench/scripts/run_goodput_matrix.sh calibrate g1
```

Python-cuGraph comparisons run only on a host with Python cuGraph installed;
an unavailable oracle fails the run rather than skipping it:

```bash
flock /tmp/cudf-gpu.lock \
  cargo run -p tools --all-features -- \
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
nextest configuration, or a build script.

### Harness internals

`scripts/check_all.sh` is the only top-level script entry point. Supporting
files are grouped by responsibility, and the directory map is maintained in
[`scripts/README.md`](https://github.com/algeon-dev/algeon/blob/main/scripts/README.md):
`scripts/build/` (build support and packaging policy), `scripts/ci/`
(validation policies and parsers, with their fault-injection suites under
`scripts/ci/tests/`), and `scripts/dev/` (manually invoked helpers). Policy
data lives in `.config/policies/`; tool configuration such as
`.config/nextest.toml` stays at its tool-defined path.

Test scheduling lives entirely in `.config/nextest.toml`:

```toml
[test-groups]
gpu = { max-threads = 8 }
gpu-exclusive = { max-threads = 1 }
```

The NVIDIA driver serializes CUDA context creation, so on one GPU test
throughput peaks around 4-8 concurrent processes; `gpu` caps at 8.
`gpu-exclusive` tests run alone and last (`priority = -100`), reserved for
binaries that observe or occupy the whole device or need multiple GPUs;
nextest's one-process-per-test isolation already covers ordinary
process-global state, so exclusivity is for tests that need the *device* to
themselves, not just the process.

Because every test process pays for its own CUDA context, the cudf
contract binaries `semantic_contracts_gpu`, `execution_contracts_gpu`, and
`interop_contracts_gpu` group small tests into one table per file: the test
functions are plain `fn`s, and a single `#[test] fn <file_stem>_cases` passes
them to `common::run_named_cases`, which runs each case on the thread's shared
test runtime and fails once, naming every failing case. Add a new case to its
file's table. Keep a test as its own `#[test]` only when it arms one-shot
native fault injection, installs process-global or device-safety state, spawns
threads, is `#[should_panic]`, or uses a device other than 0. A table is run
like any test, for example
`cargo nt -E 'binary_id(cudf::semantic_contracts_gpu) & test(expr_tests_cases)'`.

Every test binary lands in exactly one of:

- **CPU, opt-in**: a `_cpu` target-name suffix, or a `binary_id` listed in
  the CPU override in `.config/nextest.toml`. To move a binary onto CPU, add
  its `binary_id` (from `cargo nextest show-config test-groups`, or a
  `binary_id(...)` filterset) to that override list; an unlisted binary
  defaults into `gpu`, so a misclassification is only slower, never a
  scheduling hazard.
- **`gpu-exclusive`**: any target whose name ends `_gpu_exclusive`.
- **`gpu`**: every remaining test, matched last.

`scripts/ci/nextest_gpu_lock.sh` is the run-wrapper nextest invokes for every
test (`[scripts.wrapper.gpu-lock]`, applied through
`[[profile.default.scripts]]`). It reads `NEXTEST_TEST_GROUP` and takes
`/tmp/cudf-gpu.lock` shared for `gpu`, exclusive for `gpu-exclusive`, and
nothing for an ungrouped CPU test; if a file descriptor under `/proc/$$/fd`
already points at the lock (an ancestor `flock`), it execs straight through
instead of re-locking. It prints `nextest_gpu_lock: waiting for <mode>
/tmp/cudf-gpu.lock` when a test blocks on another holder, so a stalled run
reads as a lock wait rather than a hang. Its self-test is:

```bash
bash scripts/ci/tests/check_nextest_gpu_lock_tests.sh
```

Inspect a binary's group, or a test's resolved filterset, with:

```bash
cargo nextest show-config test-groups
```

The `default` profile is quiet (only `FAIL`/`RETRY`/`SLOW` lines, failure
output held to the end) and stops after 20 failures; GPU tests time out after
5 minutes including lock waits, CPU tests after 2. The `gate` profile
disables fail-fast, writes JUnit to `target/nextest/gate/junit.xml`, raises
the GPU timeout to 20 minutes, and caps the whole run at a 90-minute global
timeout, since a gate's exclusive tests may wait for another run's shared GPU
tests to drain.

`-R latest` reruns, `cargo nextest replay`, and any other `nextest store`
command need recording enabled once per machine in nextest's own user config
(not this repository's); see
[Prerequisites](./prerequisites.md#rust-toolchain-and-test-runner).

The two compile-only hygiene phases do not run tests, contact external
services, or take the GPU lock. `msrv-check` uses the workspace's declared
Rust version:

```bash
cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo
```

Feature-isolation is the one repository-owned exception to the all-features
rule. It compiles every target in each package's no-feature and non-default
feature configurations. It does not execute tests:

```bash
cargo hack check --workspace --each-feature --exclude-features default --all-targets --locked --keep-going
```

`scripts/check_all.sh` runs that exact matrix as six package shards
(`FEATURE_ISOLATION_SHARDS`), each in its own target directory, then fails
the phase closed unless the union of commands the shards actually ran matches
the canonical command's own `--print-command-list` output.

Doctests run as a separate command through the same GPU lock wrapper, since
rustdoc does not build through nextest:

```bash
NEXTEST_TEST_GROUP=gpu scripts/ci/nextest_gpu_lock.sh /tmp/cudf-gpu.lock \
  env LD_LIBRARY_PATH="target/native/cudf-install/lib:target/native/cuvs-install/lib:target/native/cugraph-install/lib:target/native/cugraph-build" \
  cargo test --workspace --all-features --doc --no-fail-fast
```

rustdoc does not pass the build scripts' rpath link arguments to doctest
binaries, and runs them from temporary directories, so the loader needs
absolute native library directories.

Rustdoc is always strict: the harness appends `-D warnings` after any
caller-provided `RUSTDOCFLAGS`, so an earlier `-A warnings` cannot weaken it.

`scripts/build/rapids.rs` is the std-only build-script support included by the
DataFusion adapter, `query-engine`, server, benchmark, tools, and examples
packages. Its isolated policy suite runs as part of the policy phase's
self-test scripts and can be run alone; it neither loads nor modifies the
real RAPIDS libraries:

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
`cudf-sys` is
[`scripts/build/check_fingerprint_matrix.sh`](https://github.com/algeon-dev/algeon/blob/main/scripts/build/check_fingerprint_matrix.sh)
(requires `CUDF_INSTALL_DIR`, `CC`, and `CXX`), and its static contract is:

```bash
cargo test -p cudf-sys --all-features --test fingerprint_matrix_contract
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
