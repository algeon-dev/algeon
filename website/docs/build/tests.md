---
sidebar_position: 5
sidebar_label: Tests
description: The full pre-merge gate, its output and reruns, the partial edit loops, feature flags, and the acceptance runs kept outside Cargo test targets.
---

# Tests

This page is for anyone validating a change before merge. It covers the
commands, what they print, and how to rerun a failure. Lane rules, test
placement, and the harness scripts are contracts on the
[Test harness](./test-harness.md) page.

Before you start:

- The [build shell](./from-source.md#2-configure-the-build-shell) is
  configured and all four native layers plus the Rust workspace are built.
- `cargo-nextest` 0.9.144 and `cargo-hack` 0.6.45 are installed
  ([Prerequisites](./prerequisites.md#rust-toolchain-and-test-runner)); the
  full gate checks both versions before any work, and `--fast` checks only
  `cargo-nextest`.
- No benchmark, running Flight SQL server, or other checkout's gate holds
  `/tmp/cudf-gpu.lock`, since GPU tests wait on that lock until it is free.

## The full gate

Run it from the checkout root. Do not wrap nextest or the full gate in `flock`;
the run-wrapper owns test locking. Non-test GPU commands, including benchmarks,
servers and tools, take the lock separately.

```bash
bash scripts/check_all.sh
```

For a run that outlives a terminal or tool call, launch it with `nohup` and
poll `target/check-all/latest/summary.tsv`. With no arguments the script
starts all four phase groups at once:

| Group | What runs | Order |
| --- | --- | --- |
| policy | rustfmt, native-source verification, checked bindings, and the policy self-test scripts | Independent, in parallel |
| tests | `cargo nt -P gate` over the full workspace, then doctests through the GPU lock wrapper | Doctests start after nextest finishes |
| static | clippy with `-D warnings`, current-run orphan-source verification, strict rustdoc, then the public API surface snapshot | In sequence |
| hygiene | seven feature-isolation shards and their union check, the shipped-server clippy lane, `msrv-check`, a docs.rs native-free compile check, the hidden-API consumer probe, and the component-package check | Each in its own target directory; never blocks tests |

- A failed phase in any group fails the gate.
- Every other Cargo phase uses `--workspace --all-features`. Feature-isolation
  and the shipped-server clippy lane are the exceptions
  ([Feature coverage](./test-harness.md#feature-coverage)); the feature-off
  clippy supplement is described under
  [Test layout](./test-harness.md#test-layout).

### Passing run

Each phase logs to `target/check-all/run.*/<phase>.log`, and
`target/check-all/latest` symlinks the newest run. Its `summary.tsv` has the
columns `phase`, `status`, `started`, `duration` and `log`, with `gate` as the
first row, and the script rewrites it every time a phase starts or finishes.
A passing full run ends with:

```text
OK: full pre-merge gate passed.
```

Full validation requires one complete gate run on the final source state.
Focused checks, failure-only reruns, and `--fast` (which ends with
`OK: fast checks passed. PARTIAL: this is not full pre-merge validation.`)
are partial validation.

### On failure

The run continues past a failed phase, prints each phase's status and
duration, then lists the failed phases with their log paths. A failed `tests`
or `cpu-tests` phase also prints its nextest summary and the rerun command:
`cargo nt -P gate -R latest` for `tests`, or
`cargo nt -E 'not group(gpu) & not group(gpu-exclusive)' -R latest` for
`cpu-tests`. Either reruns only what that run left failed or unfinished. The
script exits non-zero after:

```text
FAIL: full pre-merge gate reported problems.
```

- A native-library or CUDA-architecture failure in the `tests` phase usually
  means a stale or mismatched prefix.
- A single failing phase is fastest to iterate on with the
  [targeted edit loops](#targeted-edit-loops); rerun the full gate afterwards.
- Exit status 2 before any phase means an argument, environment, or
  tool-version error, and the message names the fix.
- `cargo nextest replay` replays a stored run's results for each test
  without rerunning anything.

## Partial edit loop

`--fast` runs rustfmt, the native-source check, the policy self-test scripts,
and the CPU nextest tests (`cargo nt -E 'not group(gpu) & not
group(gpu-exclusive)'`). It skips `msrv-check`, feature-isolation, the
shipped-server clippy lane, all GPU tests, doctests, rustdoc, the public API
snapshot, the hidden-API consumer probe, and clippy.

```bash
bash scripts/check_all.sh --fast
```

Any other option, or a duplicate option, exits with status 2. Feature
selection is not a command-line option; the harness defines its workspace
checks and [coverage exceptions](./test-harness.md#feature-coverage).

### Targeted edit loops

Ordinary Cargo build, check, lint and test commands run from the repository
root with `--workspace --all-features`. `cargo nt` is the alias for
`cargo nextest run --workspace --all-features` defined in
[`.cargo/config.toml`](https://github.com/algeon-dev/algeon/blob/main/.cargo/config.toml).
Select tests with filtersets, never `-p`, to reuse that build configuration:

```bash
cargo nt -E 'package(<package>)'                              # what you changed
cargo nt -E 'rdeps(<package>)'                                # everything downstream
cargo nt -E 'binary_id(<package>::<target>) & test(<name>)'   # one test
cargo nt -E 'not group(gpu) & not group(gpu-exclusive)'       # CPU only
cargo nt -E 'package(bench)'                                  # the non-default bench package
cargo nt -E '<same filterset>' -R latest                      # rerun only failures
```

A rerun must repeat the original filterset, because nextest treats binaries
that a different or missing `-E` skipped earlier as new and reruns the whole
suite. Plain `cargo test` is not used for the workspace's non-doc tests,
because libtest runs a binary's cases as threads that cannot restore the
engine's process-global allocator or backend policy fixtures, and the
admitted test surface requires nextest's per-test-process isolation. The
[test lanes](./test-harness.md#harness-internals) live in
`.config/nextest.toml`.

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
| `cuvs` | cuVS UDTFs (exact kNN, KMeans, PCA) and their GPU execution |
| `cuml` | cuML UDTFs (truncated SVD, HDBSCAN, UMAP, logistic regression, random forests) and their GPU execution; requires the native cuML install from `scripts/build/build_libcuml.sh` |
| `nvml` | Optional NVML device diagnostics |

The Arrow Flight SQL server is launched by the separate `algeon-server`
package; its `cugraph`, `cuvs`, `cuml`, and `nvml` features forward the
corresponding adapter and engine capabilities.

```bash
cargo run -p algeon-server --all-features --bin algeon_server
```

## Acceptance runs outside the gate

Runs that need a real dataset, cloud account, running catalog, or Python
cuGraph stay outside Cargo test targets, and each one takes the GPU lock
itself.

| Purpose | Command | Requires |
| --- | --- | --- |
| TPC-DS parity report | `ALGEON_PARQUET_DIR=/path/to/complete/tpcds-sf1 flock /tmp/cudf-gpu.lock cargo run -p bench --all-features --bin native_tpcds_parity_report` | A complete TPC-DS dataset |
| TPC-H execution report | `ALGEON_PARQUET_DIR=/path/to/complete/tpch-sf1 flock /tmp/cudf-gpu.lock cargo run -p bench --all-features --bin native_tpch_execution_report` | A complete TPC-H dataset |
| Sustained-load goodput (`query_service_goodput`) | `crates/bench/scripts/run_goodput_matrix.sh calibrate g1` | The GPU and dataset recorded by `calibrate` |
| Python-cuGraph comparison | `flock /tmp/cudf-gpu.lock cargo run -p tools --all-features -- cugraph-fixture-e2e bfs-include-edges` | A host with Python cuGraph installed |

- Scale-factor workloads validate their dataset before starting and fail when
  it is incomplete. The TPC-DS report emits one row per attempted query.
- The goodput driver script pins every factor except the one under comparison
  and records source, GPU, and native-library identity next to the results.
  `calibrate` freezes each item's single-flight median latency and output
  fingerprint; the measurement groups refuse to run if the live GPU or
  dataset differs from that calibration.
- An unavailable Python cuGraph oracle fails the run; the run does not skip
  it. Replace `bfs-include-edges` with `pagerank`, `bfs-utf8-include-edges`,
  `bfs-utf8-path`, or `bfs-utf8-predicates` for the other provisioned cases.
- Tools owns the BFS predicate fixtures under `fixture/data/cugraph/`. The
  PageRank runner and required-oracle failure policy are shared through
  `algeon_datafusion::cugraph_sql::bench_support`. Upstream parity datasets
  prepared by `fixture/graph/prepare_cugraph.py` remain in
  `fixture/graph/cugraph/`.
- External-service E2E tests are invoked explicitly through
  `cargo run -p tools --all-features`.

For any performance claim, use `--release` builds, compare rows produced by
the same harness, and label warmup count, iteration count, source cache
policy, and source chunk profile. See the
[benchmark reference](../reference/benchmarks/).
