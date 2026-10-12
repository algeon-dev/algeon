---
sidebar_position: 5
sidebar_label: Tests
description: Focused development checks, the local pre-merge gate, failure reruns, and acceptance runs.
---

# Tests

Use the [targeted edit loops](#targeted-edit-loops) during development. Run
checks for the affected behavior and finish once they pass. Broaden validation
when failures or changes to shared contracts warrant it. Documentation-only
changes need no Cargo or GPU checks.

The [full gate](#the-full-gate) is for pre-merge validation or an explicit
request. Algeon currently runs these checks locally, without GitHub CI/CD.
Test placement is defined in
[Test layout](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#test-layout);
script maintainers use the
[harness reference](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md).

Before running Cargo checks:

- The [build shell](./from-source.md#2-configure-the-build-shell) is
  configured and all four native layers plus the Rust workspace are built.
- The latest stable `cargo-nextest` and `cargo-hack` 0.6.45 are installed
  ([Prerequisites](./prerequisites.md#rust-toolchain-and-test-runner)). The
  gate checks that nextest runs and prints its version; nextest is not pinned.
  The full gate also checks the exact `cargo-hack` version.
- No benchmark, running Flight SQL server, or other checkout's gate holds
  `/tmp/cudf-gpu.lock`, since GPU tests wait on that lock until it is free.

Set the native loader directories once before the Cargo commands below. The
gate exports this same value for every phase, replacing any caller value:

```bash
export LD_LIBRARY_PATH="$PWD/target/native/cudf-install/lib:$PWD/target/native/cuvs-install/lib:$PWD/target/native/cugraph-install/lib:$PWD/target/native/cuml-install/lib:$PWD/target/native/cugraph-build"
```

Repository executables receive native rpath arguments from their own package
build scripts. Installed cuGraph and cuML libraries also name their cuVS and
RMM dependency prefixes in RUNPATH. The gate exports the loader directories for
all phases, including doctests whose binaries carry no native rpath.
A downstream application does not inherit dependency build-script rpath
arguments. Its build and direct execution are covered separately by the
[Rust dependency quickstart](./rust-dependency.md#4-build-and-run-the-application)
and the [registry consumer check](https://github.com/algeon-dev/algeon/blob/main/scripts/ci/check_registry_consumer.sh).

## Development checks

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
[test lanes](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#harness-internals) live in
`.config/nextest.toml`.

Formatting, lints, and docs match the gate's phases:

```bash
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --keep-going
bash scripts/ci/check_public_api_surface.sh check --all-features
```

The public API check enforces the
[application documentation boundary](../design/index.mdx#api-documentation)
alongside the signature snapshots. Updating a snapshot does not waive missing
documentation on a reachable owner type. The isolated documentation regression
tests run as part of the snapshot policy self-test:

```bash
bash scripts/ci/tests/check_public_api_surface_tests.sh
```

### Broader CPU checks

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
checks and [coverage exceptions](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#feature-coverage).

## The full gate

Run it from the checkout root. Do not wrap nextest or the full gate in `flock`;
the run-wrapper owns test locking. Non-test GPU commands, including benchmarks,
servers and tools, take the lock separately.

```bash
bash scripts/check_all.sh
```

For a run that outlives a terminal or tool call, launch it with `nohup` and
poll `target/check-all/latest/summary.tsv`. With no arguments the script
starts policy checks and the `test-build` phase, which captures Cargo metadata
and builds the test binaries once. The other groups start after that phase
completes:

| Group | What runs | Order |
| --- | --- | --- |
| policy | rustfmt, native-source verification, checked bindings, and the policy self-test scripts | Independent, in parallel |
| tests | `cargo nextest run -P gate` reusing that run's Cargo and binaries metadata, then doctests through the GPU lock wrapper | Doctests start after nextest finishes; a failed test build skips nextest execution and still runs doctests |
| static | clippy with `-D warnings`, current-run orphan-source verification, strict rustdoc, then the public API snapshots and application documentation check | In sequence; Cargo and API checks use `nice -n 10` |
| hygiene | seven feature-isolation shards and their union check, shipped-server clippy, `msrv`, the `DOCS_RS=1` workspace compile check, `docsrs-native-free` rustdoc over the packaged publication closure, the API consumer probe and publication policy | Independent after the test build; compilation uses `nice -n 10` and separate target directories |

- A failed phase in any group fails the gate.
- Nextest reuses the test build's Cargo and binaries metadata, which fix the
  same full-workspace, all-features graph as `cargo nt`.
- Workspace Cargo phases normally use `--workspace --all-features`.
  Feature-isolation and shipped-server clippy use their defined feature
  selections ([Feature coverage](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#feature-coverage)).
  `docsrs-native-free` extracts the release archives and documents each crate
  separately with its own docs.rs metadata features, an isolated native-free
  environment and strict warnings. The
  [packaged release checks](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#doctests-and-build-support)
  define that archive lane; the feature-off clippy supplement is described under
  [Layout checks](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#layout-checks).

The full pre-merge gate includes packaged native-free rustdoc. The native
archive consumer gate runs separately:

```bash
bash scripts/ci/check_registry_consumer.sh
```

It requires the production native installs and a GPU. Its complete matrix
builds external applications against extracted release archives, checks loader
resolution and directly executes result assertions from another working
directory. See the
[packaged release checks](https://github.com/algeon-dev/algeon/blob/main/scripts/test-harness.md#doctests-and-build-support)
for native/toolchain selectors, evidence output and the fresh-resolution and
source-lockfile lanes. A focused `--features` or `--lane` rerun covers that
selection; the default complete matrix establishes the archive consumer
contract. Registry downloads without patches and hosted docs.rs results remain
post-publication checks.

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
  [targeted edit loops](#targeted-edit-loops). Rerun the full gate when
  completing pre-merge validation.
- Exit status 2 before any phase means an argument, environment, or
  tool-version error, and the message names the fix.
- `cargo nextest replay` replays a stored run's results for each test
  without rerunning anything.

## Doctests

Doctests run as a separate command through the GPU lock wrapper, since
rustdoc does not build through nextest. rustdoc does not pass the build
scripts' rpath link arguments to doctest binaries, and runs them from
temporary directories. The loader therefore needs absolute native library
directories.

<details>
<summary>Doctest command</summary>

```bash
NEXTEST_TEST_GROUP=gpu scripts/ci/nextest_gpu_lock.sh /tmp/cudf-gpu.lock \
  cargo test --workspace --all-features --doc --no-fail-fast
```

</details>

## Multi-GPU tests

Multi-GPU cases are ignored by default and require at least two visible
devices. On a host with that inventory, select them explicitly:

```bash
cargo nt --run-ignored only -E '
binary(/multi_gpu|multi_device/)
  | (binary_id(algeon-cudf::interop_contracts_gpu) & test(/::(device_event_export_rejects_readiness_from_another_device|helper_built_exports_reject_cross_device_inputs)$/))
  | (binary_id(algeon-cudf) & test(cross_device_secondary_is_rejected_with_public_classification))
  | (binary_id(algeon-query-runtime) & test(/^runtime::cache::local_parquet::tests::source_copy_rejects_a_cache_domain_on_another_device$/))
  | (binary_id(algeon-query-runtime) & test(/(multi_device_registry::|device_local_cleanup_fault_seals_retry_on_a_frozen_eligible_alternate|queued_placement_folds_a_native_fault_into_the_ledger_before_retargeting|query_admission_waiter_timeout_reports_the_selected_device_requirement)/))
'
```

These cases assert the device precondition when explicitly run. A single-GPU
invocation fails that precondition instead of reporting a pass.

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
