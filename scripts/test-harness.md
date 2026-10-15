# Test harness maintenance

Use [Test layout](#test-layout) when adding or moving tests. Read the other
sections when changing validation scripts or diagnosing the harness. Routine
commands and validation scope are in [Tests](../website/docs/build/tests.md).
Commands below run from the repository root.

## Test layout

Test placement follows the access the cases require. Unit tests that need
crate-private items or `cfg(test)` seams stay under `src/`. Tests that use
only public APIs, including hidden APIs enabled by `test-hooks`, belong under
`tests/`. A move must preserve item visibility. Any deliberate public API
expansion needs its own rationale and the corresponding
[public API ledger](https://github.com/algeon-dev/algeon/tree/main/.config/policies/public-api)
update.

The first level of `tests/` contains target roots only. A target with several
source files uses `tests/<target>/main.rs` and ordinary `mod` declarations.
Each test source belongs to one target; helpers shared across targets live in
`tests/support/`. cuDF contract targets group their files by binary and then
source area, for example
`tests/semantic_contracts_gpu/{core,frame,ops,internal}/`.

Unit tests may use `#[path]` to load a shared file from `tests/support/`.
Such a file supplies its own imports, accesses the crate through its public
library name, and has no `use super::*` dependency on its includer. This shared
support is the permitted reason for `#[cfg(test)] extern crate self as <lib>`
in `lib.rs`. Other references between `src/` and `tests/` are prohibited.
Cross-crate Rust sharing uses dependencies and the existing `bench-support`
or `test-hooks` feature surfaces. A build-script helper shared by `build.rs`
and tests is exempt from the source/test restriction.

Handwritten Rust uses `mod`; `include!` is reserved for generated sources,
including checked-in bindings and `OUT_DIR` output. `#[path]` is limited to
the shared-support and build-script exceptions above, and aliases selected
by `cfg` for alternative implementations. Declare these aliases with
separate `#[cfg]` and `#[path]` attributes so the checker visits each branch;
conditional `cfg_attr(..., path = ...)` attributes fail closed. A unit-test
module beside its owner uses `foo/tests.rs` with `#[cfg(test)] mod tests;` in
`foo.rs`.

Integration targets end in `_cpu`, `_gpu` or `_gpu_exclusive`. Their suffix selects the default nextest lane. CPU overrides also list lib
and bin unit-test targets, audited CPU modules, and exact CPU case names in
mixed targets. All tests selected by a CPU override must run without a
device context or GPU operation. New unmatched cases in a mixed target stay
in the GPU lane. Prefer a `_cpu` target for separable CPU modules. A target
that requires a feature declares Cargo `required-features`; it must not
replace that requirement with a crate-level `#![cfg(feature = ...)]` that
produces an empty test binary when the feature is disabled.

GPU tests in separate processes repeat CUDA context initialization. Group at least five related
cases taking less than three seconds each into a table, and split tables by
topic to keep each near ten seconds. The case functions are plain `fn`s;
one `#[test]` passes them to `cudf::testing::run_named_cases` through the
`test-hooks` feature. Table runners import the helper from that owner module.
Cases execute on the same thread in one process, with runtimes supplied by
their fixtures. The helper reports all failing case names. A lazily installed default device memory manager is permitted as
shared table state. Thread-local, self-clearing one-shot native faults do
not by themselves require a separate process; faults that can remain armed
after a panic need an unwind guard or an independent test. Keep separate
tests for process-global or device-safety installation, spawned threads,
`#[should_panic]`, and devices other than 0. A table is run like any test:
`cargo nt -E 'binary_id(algeon-cudf::semantic_contracts_gpu) & test(filter_misc_sorting_cases)'`.

`ALGEON_TEST_CASES` selects case names containing any comma-separated
substring, for example `ALGEON_TEST_CASES=left_build,null_keys cargo nt -E
'test(join_cases)'`. An unset or empty value runs the complete table. A
non-empty selector matching no cases triggers a failure; panicking cases still allow the
remaining selected cases to run. Each case must restore state during
unwinding when later cases depend on it. The sys crate keeps an equivalent
private helper because depending on cuDF would create a dependency cycle.

The `tools` package owns the BFS predicate fixtures under
`fixture/data/cugraph/`. The PageRank runner and required-oracle failure
policy are shared through `algeon_datafusion::cugraph_sql::bench_support`, and
upstream parity datasets prepared by `fixture/graph/prepare_cugraph.py` remain
in `fixture/graph/cugraph/`.

## Layout checks

`scripts/ci/check_test_layout.py` enforces source ownership, first-level
roots, allowed source boundaries, generated-only `include!`, and `#[path]`
exceptions in the policy phase. Its fault-injection suite uses temporary
repositories and real rustc dep-info, without native libraries:

```bash
python3 scripts/ci/tests/check_test_layout_tests.py
python3 scripts/ci/check_test_layout.py
```

The full gate's `orphan-sources` phase compares tracked workspace Rust files
present in the working tree with dep-info for the compiler artifacts reported
by that run's successful clippy commands. Pending deletions are excluded from
source coverage. It follows no old dep-info through shared `target` caches.
The main clippy command selects `--workspace --all-targets --all-features`;
a second command selects `algeon-query-engine` and `algeon-datafusion` with
`--all-targets --no-default-features` to compile feature-off implementations
and frontend tests that `--all-features` excludes. Both JSON artifact streams
are saved in the run directory. Missing dep-info, a failed clippy stream, or
an uncovered tracked source fails the check. Vendored submodules outside
workspace members are excluded; build helpers and checked-in bindings must
appear in dep-info. The phase is skipped when clippy fails, and the gate
still fails. `--fast` checks layout without source-coverage validation.

## Harness internals

`scripts/check_all.sh` is the only top-level script entry point. Supporting
files are grouped by responsibility, and the directory map is maintained in
[`scripts/README.md`](https://github.com/algeon-dev/algeon/blob/main/scripts/README.md):
`scripts/build/` (build support and packaging policy), `scripts/ci/`
(validation policies and parsers, with their fault-injection suites under
`scripts/ci/tests/`), and `scripts/dev/` (manually invoked helpers). Policy
data lives in `.config/policies/`; tool configuration such as
`.config/nextest.toml` stays at its tool-defined path.
Component-local umbrella commands must not introduce another full gate or
duplicate the root source, publication or test-lane registries.

Test scheduling lives entirely in `.config/nextest.toml`:

```toml
[test-groups]
gpu = { max-threads = 8 }
gpu-exclusive = { max-threads = 1 }
```

Every test lands in one lane:

| Lane | How a binary lands there | GPU lock | Max concurrent | Timeout (`default` / `gate`) |
| --- | --- | --- | --- | --- |
| CPU (opt-in) | A `_cpu` integration-target suffix, a lib/bin unit-test `binary_id`, or an audited CPU module or exact case filter within a mixed binary in `.config/nextest.toml`. | None | The run's global thread limit | 2 min / 2 min |
| `gpu-exclusive` | A `_gpu_exclusive` target, a test under a `gpu_exclusive` module, or an audited hardware case filter. | Exclusive | 1, alone and last (`priority = -100`) | 5 min / 20 min |
| `gpu` | Every remaining test, matched last. | Shared | 8 | 5 min / 20 min |

- An unlisted binary or case defaults into `gpu`, so an omitted CPU override
  stays safe to schedule.
- The NVIDIA driver serializes CUDA context creation. On one GPU, test
  throughput peaks around 4-8 concurrent processes, and `gpu` caps at 8.
- `gpu-exclusive` is reserved for binaries that observe or occupy an entire
  GPU or need several GPUs. nextest's one-process-per-test isolation already
  covers ordinary process-global state; exclusivity is for tests that need
  the *device* to themselves.
- GPU timeouts include lock waits.

The gate exports one canonical native `LD_LIBRARY_PATH` before all phases;
doctests inherit it without prepending directories again. It prints Cargo's
effective toolchain values from the merged configuration with shell and
`force` precedence applied. During `test-build`, the gate captures fresh
Cargo metadata and builds the workspace with all features through a
binaries-only nextest list. Both metadata files belong to that gate run.
The tests phase reuses them without another Cargo build or metadata query,
so static compilation cannot delay test execution through Cargo's build lock.
Policy checks run during `test-build`; static
and hygiene compilation starts after that phase completes with niceness
increased by 10, up to 19. A compile failure skips nextest execution while
the remaining checks continue. The isolated orchestration suite covers these conditions without
building Rust or loading native libraries:

```bash
python3 scripts/ci/tests/check_check_all_tests.py
```

`--native-free` is the subset GitHub Actions runs in
`.github/workflows/ci.yml`. It exports `DOCS_RS=1` and
`CARGO_TARGET_DIR=target/native-free` for every phase, and its compile phases
run one at a time, which fits a hosted runner's four cores and 16 GB.
Clippy over all targets replaces the full gate's separate `DOCS_RS=1`
compile check. `scripts/ci/native_free_packages.py` reads that run's Cargo
metadata and selects the test packages. A workspace member with a `links` key
owns a native library. A package qualifies when its own normal, build and dev
dependencies, and the normal and build dependencies below them, reach no such
member. The orchestration suite covers that selection with a mock dependency
graph.

`scripts/ci/nextest_gpu_lock.sh` is the run-wrapper nextest invokes for every
test (`[scripts.wrapper.gpu-lock]`, applied through
`[[profile.default.scripts]]`); it reads `NEXTEST_TEST_GROUP` and takes
`/tmp/cudf-gpu.lock` in the lane's mode from the table above. When a file
descriptor under `/proc/$$/fd` already points at the lock (an ancestor
`flock`), the wrapper execs straight through and does not lock again. A test
that blocks on another holder prints `nextest_gpu_lock: waiting for <mode>
/tmp/cudf-gpu.lock`, and the log then reads as a lock wait and not as a hang.

```bash
bash scripts/ci/tests/check_nextest_gpu_lock_tests.sh   # wrapper self-test
cargo nextest show-config test-groups                   # a binary's group, or a test's resolved filterset
```

The two nextest profiles differ as follows; the lane table lists their
timeouts.

| Setting | `default` | `gate` |
| --- | --- | --- |
| Failures | Stops after 20 failures | Fail-fast disabled |
| Output | Only `FAIL`/`RETRY`/`SLOW` lines, with failure output held to the end | Same as `default` |
| JUnit | None | `target/nextest/gate/junit.xml` |
| Global timeout | None | 90 minutes |

- The `gate` profile raises the GPU timeout and adds the global timeout
  because a gate's exclusive tests may wait until another run's shared GPU
  tests finish.
- `-R latest` reruns, `cargo nextest replay`, and any other `nextest store`
  command need recording enabled once per machine in nextest's own user
  config (outside this repository); see
  [Prerequisites](../website/docs/build/prerequisites.md#rust-toolchain-and-test-runner).

## Feature coverage

The MSRV and feature-isolation phases compile without running tests, contacting
external services or taking the GPU lock. `msrv-check` uses the workspace's
declared Rust version:

```bash
cargo hack check --workspace --all-features --locked --rust-version --workspace-behavior=cargo
```

Feature-isolation is the first repository-owned exception to the
all-features rule. It compiles every target in each package's no-feature and
non-default feature configurations. It does not execute tests:

```bash
cargo hack check --workspace --each-feature --exclude-features default --all-targets --locked --keep-going
```

`scripts/check_all.sh` runs that matrix as seven package shards
(`FEATURE_ISOLATION_SHARDS`), each in its own target directory. The phase
fails closed unless the union of commands the shards ran matches the
canonical command's own `--print-command-list` output.

The same phase runs `scripts/ci/check_nvml_opt_in.py`. For each workspace
member's no-feature and single-feature selection, it resolves the normal and
build graph with `cargo tree` and fails when `nvml-wrapper` appears outside an
`nvml` selection. It also requires the adapter, server, bench and tools NVML
selections to resolve the dependency. The facade consumer fixture enables
every capability feature and is exempt. Its fault-injection suite is
`scripts/ci/tests/check_nvml_opt_in_tests.py`.

The shipped-server clippy lane is the second exception. `--all-features`
unifies dev-dependency features such as `test-hooks` into every build, and
for that reason it does not lint the server as packaged. The lane selects only the
`algeon_server` binary through its manifest, with `--no-default-features` and
the packaging feature set that `scripts/build/cpu_packaging_policy.py
shipped-features` reads from `server.features` in
`.config/policies/cpu-packaging.toml`. Packaging uses that source too. The
lane selects no tests, examples or all-targets build, and keeps its own
persistent target directory:

```bash
cargo clippy --manifest-path crates/algeon-server/Cargo.toml --bin algeon_server --no-default-features \
  --features "$(python3 scripts/build/cpu_packaging_policy.py shipped-features)" --locked \
  --target-dir target/check-all/shipped-clippy -- -D warnings
```

## Doctests and build support

The full gate's `docsrs-native-free` phase checks the packaged publication
closure separately from workspace rustdoc and the `DOCS_RS=1` workspace
check. `scripts/ci/release_archives.py` reads the existing publication
allowlist, packages it with explicit adapter features
`cugraph,cuvs,cuml,server,nvml`, and checks normalized manifests, exact
first-party dependency versions, docs.rs metadata and required native build
inputs. `test-hooks` and `qualification` are excluded from release and
docs.rs feature selections.

Archives, extracted sources, `release.json`, `[patch.crates-io]` declarations,
the source `Cargo.lock` and preparation logs stay outside the checkout.
`scripts/ci/check_docsrs_native_free.sh` documents each extracted crate with
its own docs.rs metadata features. It uses an isolated cwd and Cargo home,
shares only the registry/Git download caches and the toolchain `bin`
directory, and replaces native/CUDA selectors with nonexistent paths. It
fetches every dependency before rustdoc, then runs with `DOCS_RS=1`,
`--locked --offline`, four build jobs and strict warnings. Cargo JSON
messages identify the sys-crate build-script outputs
used by this invocation, including cached executions. Every sys crate must
report empty native link libraries, search paths and linker directives;
`logs/docsrs/native-free-evidence.json` retains the selectors and output paths.
The existing `DOCS_RS` branches return before native discovery or builds.
This local check restricts Cargo dependency access with offline mode; it
does not hide host filesystem paths or enforce a network sandbox for build
scripts. The hosted docs.rs sandbox remains a separate publication check.

```bash
python3 scripts/ci/release_archives.py --output /tmp/algeon-release
bash scripts/ci/check_docsrs_native_free.sh --archives /tmp/algeon-release
python3 scripts/ci/tests/check_release_archives_tests.py
```

Use `--require-clean` on archive preparation for a frozen release source.
The default permits dirty development sources. An external `--target-dir`
on the docs.rs script reuses dependency compilation across preparations.

The separate release gate is `scripts/ci/check_registry_consumer.sh`, which
invokes `scripts/ci/registry_consumer.py`. It copies the documented application
fixture outside the checkout and patches only the extracted archives. Its
default matrix covers the adapter's default, each algorithm family, the
combined algorithm stack, `server`, `nvml` and all release features in both
fresh-resolution and source-lockfile lanes. Each row's resolved graph contains
`nvml-wrapper` only when the row selects `nvml`, and the application checks
that `nvml_status` is `FeatureDisabled` without it. Builds run offline after fetch;
runtime checks use the executable's ELF interpreter to verify selected
library identities and execute queries with
result assertions from another cwd. Native discovery cases use temporary
prefixes without rebuilding installed libraries. This gate requires the
existing production native installs and a GPU; it is invoked separately
from the full pre-merge gate.

```bash
bash scripts/ci/check_registry_consumer.sh --archives /tmp/algeon-release \
  --native-root "$PWD/target/native" --cuda-home /usr/local/cuda \
  --cc /usr/bin/gcc-14 --cxx /usr/bin/g++-14 \
  --output /tmp/algeon-registry-validation
python3 scripts/ci/tests/check_registry_consumer_tests.py
```

The registry command retains toolchain, feature, resolution, build, loader
and execution evidence under its output directory. An external `--target-dir`
reuses compilation across preparations. `--features` and `--lane`
select focused reruns; only the default complete matrix establishes the
full archive consumer contract. Successful local patches establish archive
behavior before publication. A registry download without patches and the
hosted docs.rs result remain post-publication checks.

Doctest commands and loader requirements are in [Tests](../website/docs/build/tests.md#doctests).

Rustdoc is always strict. The harness appends `-D warnings` after any
caller-provided `RUSTDOCFLAGS`, and an earlier `-A warnings` cannot weaken it.

The `public-api-surface` phase also checks the
[API documentation boundary](#api-documentation-boundary). After comparing ordinary API snapshots, it extracts complete workspace
rustdoc JSON once with private and hidden items included. This provides
method signatures for types exposed through another type's fields or methods.
Each invocation copies that successful build's JSON into its own temporary
directory; the checker does not inspect older target artifacts.

The extraction command disables only `rustdoc::broken_intra_doc_links`,
`rustdoc::private_intra_doc_links` and `rustdoc::redundant_explicit_links`.
Private link rendering is outside the documentation-presence check, and the
preceding public rustdoc phase retains its strict flags. Feature and target
arguments match the snapshot invocation. The snapshot policy self-test runs
the real-rustdoc regression suite, including a dependency documentation
removal that fails while the signatures still match:

```bash
bash scripts/ci/tests/check_public_api_surface_tests.sh
```

The [shared build-support crate](#workspace-and-publication-contracts)
has an isolated policy suite in the full gate's self-test phase. The suite can
also run alone, without loading or modifying RAPIDS libraries:

```bash
bash scripts/build/check_build_support.sh
```

## Workspace and publication contracts

Published packages use the `algeon-` prefix, and crate directory names match
package names. `[workspace.dependencies]` retains bare dependency aliases
through `package = "algeon-…"`. Rust imports and `dep:` features keep those
identities. Refer to layers by their bare alias. Every renamed crate pins its
`[lib] name` explicitly to preserve imports and public API snapshot names.

Each crate declares `publish` directly: `publish = ["crates-io"]` for the
release closure and `publish = false` otherwise. Do not set
`[workspace.package].publish`. The allowlist and its validation belong to
[`check_workspace_publication.py`](ci/check_workspace_publication.py), which
rejects undeclared or mismatched publication settings.

The release closure includes the normal and build dependencies needed to
install `algeon-datafusion` from crates.io. Published implementation packages
keep their owner APIs for workspace consumers. The
[API documentation boundary](#api-documentation-boundary) determines which
interfaces require application-facing rustdoc.

[`algeon-build-support`](../components/algeon-build-support/Cargo.toml)
(library `rapids_build_support`) is the shared build-dependency for packages
whose build scripts link RAPIDS libraries. It remains std-only and cannot
depend on crates built after it. Its policy suite is described under
[Doctests and build support](#doctests-and-build-support).

## Third-party data

The repository license is Apache-2.0. Content under other terms is listed in
[NOTICE](../NOTICE). The saved MerRec query results are derived from a CC
BY-NC 4.0 dataset and live only under `website/static/data/merrec-*.json`,
listed in [LICENSE-MERREC.md](../website/static/data/LICENSE-MERREC.md). Each
capture names its terms in a top-level `license` field, which the capture
writers under `scripts/dev` emit from `merrec_verify_common.MERREC_LICENSE`.
No published crate, Docker image or release archive may include those files:
the Docker scripts stage only the server binary and native libraries, and
`release_archives.py` packages crate directories.
[`check_third_party_data.py`](ci/check_third_party_data.py) enforces the
location, the field, the file list and the packaging boundary in the policy
phase. A new derived capture goes in that directory with the field and a line
in `LICENSE-MERREC.md`; a derived file of another dataset needs its own entry
in NOTICE and an extension of the check.

## API documentation boundary

`algeon-datafusion` is the application-facing Rust crate. The query and GPU
component crates, interop and build support are implementation dependencies
published with it. Workspace code imports each contract from its owner;
applications use the adapter and its re-exports. Query crates do not re-export
another crate's modules as a facade. Public adapter signatures may use
owner-crate types, with an adapter re-export for each exposed type; the
[owner-type policy](../.config/policies/public-api/algeon-datafusion-owner-types.tsv)
records that disposition.

Visible adapter APIs require rustdoc, including public fields, enum variants
and callable methods. The requirement also covers workspace-owned types
re-exported by the adapter or reachable through its public signatures. Their
documentation belongs at the defining item, so the adapter and internal
consumers share one description. Document units, valid inputs, `None` meanings,
errors and lifetime obligations where they affect callers.

Internal cross-crate interfaces retain documentation for ownership, safety and
other non-obvious invariants. They do not require application tutorials or
exhaustive `missing_docs` enforcement merely because their package is published.
Generated FFI declarations and existing hidden test interfaces stay outside
the application documentation surface. Handwritten safety contracts remain
with their owners. Hiding an item or changing its visibility requires the
existing API review and cannot substitute for documenting an application API.
Reports, metrics and native plan wrappers retain their existing contracts.

The `public-api-surface` phase enforces this scope alongside the API snapshots
([Doctests and build support](#doctests-and-build-support)). A passing
completeness check establishes that documentation exists; review must still
verify its claims against the implementation.

## Compile delay versus test delay

A command that sits at `` Finished `test` profile ... target(s) in … `` before
any `PASS`/`FAIL` is spending its time in Cargo compile or link, before
nextest executes anything. Workspace `[profile.test]` is intentionally
absent: tests inherit `[profile.dev]` with line-table debug information and
unpacked split debuginfo. Expose the first dirty unit with:

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
cargo nt -E 'binary_id(algeon-cudf-sys::fingerprint_matrix_contract_cpu)'
```

## Target artifact maintenance

Stale feature/hash variants and interrupted mold links can inflate `target/`
without reflecting the active surface. Cleanup is never a side effect of a
test command; report and clean explicitly:

```bash
bash scripts/dev/clean_target_artifacts.sh            # sizes + mold temps
bash scripts/dev/clean_target_artifacts.sh --mold-only --yes
bash scripts/dev/clean_target_artifacts.sh --clean-cargo-target --yes
```

After cleanup the next build is cold. Take any advisory size budget from a
clean rebuild of the active all-feature surface. A multi-day dirty `target/`
tree does not give a usable budget.
