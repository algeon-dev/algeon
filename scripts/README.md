# Repository scripts

`scripts/check_all.sh` is the single canonical pre-merge entry point. The
remaining tooling is grouped by responsibility so developers do not need to
search an undifferentiated script list.

Read [AGENTS.md](AGENTS.md) before maintaining build, CI, or packaging tooling;
it links to [harness maintenance](test-harness.md) and the build contracts.

| Directory | Responsibility |
| --- | --- |
| `build/` | Shared Cargo build-script support, build-policy checks, and CPU packaging policy |
| `ci/` | Validation policies, parsers, and the compatibility `test_all.sh` alias |
| `ci/tests/` | Isolated fault-injection and orchestration tests for CI tooling |
| `dev/` | Explicit, manually invoked developer and workload helpers |

Developer maintenance helpers under `dev/` include
`bindings.sh` (the single checked-in cuDF/cuGraph/cuVS bindings entry) and
`clean_target_artifacts.sh` (report / mold-temp / `cargo clean`). Cleanup is
never an implicit side effect of a test command. Local policy checks keep native
source registry and workspace publication-policy failures separate in the full
gate.

`ci/nextest_gpu_lock.sh` is the nextest run-wrapper that every test runs
through (configured in `.config/nextest.toml`): it takes `/tmp/cudf-gpu.lock`
shared for the `gpu` test group and exclusive for `gpu-exclusive`, and takes
nothing for a CPU test. Its self-test is
`ci/tests/check_nextest_gpu_lock_tests.sh`.

`ci/check_test_layout.py` checks the test layout contract in the policy phase.
The full gate repeats it in `orphan-sources` after clippy, using only the
compiler artifacts named by that run's all-feature and feature-off JSON
streams. It reports uncovered tracked workspace Rust files and missing
current dep-info. Its fault-injection suite is
`python3 scripts/ci/tests/check_test_layout_tests.py`; the contract lives in
[Test layout](test-harness.md#test-layout).

Repository-owned policy data lives in `.config/policies/`; standard tool
configuration such as `.config/nextest.toml` (the only test-lane source)
remains at the location expected by that tool. When moving or adding a
script, update this map and all callers in the same change.

`check_all.sh` takes no arguments for the full pre-merge gate, or `--fast` for
a partial edit loop (policy checks, rustfmt, and CPU tests).
`ci/cargo_toolchain_environment.py` prints the effective Cargo toolchain
environment at gate startup. `ci/tests/check_check_all_tests.py` exercises
gate ordering, loader propagation and failure continuation with mocked Cargo.
`ci/tests/check_single_brand_tests.py` checks indexed blobs, dirty contents,
symlink targets and pending deletions against `ci/check_single_brand.sh`.
Pending deletions have no working-tree content; their indexed paths and blobs
remain subject to the branding check.

Native build helpers also run from this root and default to the repo-local
prefixes `target/native/cudf-install`, `target/native/cuvs-install`, and
`target/native/cugraph-build`. Native source is resolved only through the root
submodules under `components/*`; component-local wrapper scripts are not a
second entry point.
