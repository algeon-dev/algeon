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
their fixtures. The helper reports all failing case names. A lazily installed default device memory manager is allowed
shared table state. Thread-local, self-clearing one-shot native faults do
not by themselves require a separate process; faults that can remain armed
after a panic need an unwind guard or an independent test. Keep separate
tests for process-global or device-safety installation, spawned threads,
`#[should_panic]`, and devices other than 0. A table is run like any test:
`cargo nt -E 'binary_id(algeon-cudf::semantic_contracts_gpu) & test(filter_misc_sorting_cases)'`.

`ALGEON_TEST_CASES` selects case names containing any comma-separated
substring, for example `ALGEON_TEST_CASES=left_build,null_keys cargo nt -E
'test(join_cases)'`. An unset or empty value runs the complete table. A
nonempty selector matching no case fails; panicking cases still allow the
remaining selected cases to run. Each case must restore state during
unwinding when later cases depend on it. The sys crate keeps an equivalent
private helper because depending on cuDF would create a dependency cycle.

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

Doctest commands and loader requirements are in [Tests](../website/docs/build/tests.md#doctests).

Rustdoc is always strict. The harness appends `-D warnings` after any
caller-provided `RUSTDOCFLAGS`, and an earlier `-A warnings` cannot weaken it.

The [shared build-support crate](../website/docs/build/overview.md#workspace-and-publication-contracts)
has an isolated policy suite in the full gate's self-test phase. The suite can
also run alone, without loading or modifying RAPIDS libraries:

```bash
bash scripts/build/check_build_support.sh
```

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
