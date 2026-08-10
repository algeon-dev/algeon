# Repository scripts

`scripts/check_all.sh` is the single canonical pre-merge entry point. The
remaining tooling is grouped by responsibility so developers do not need to
search an undifferentiated script list.

| Directory | Responsibility |
| --- | --- |
| `build/` | Shared Cargo build-script support, build-policy checks, and CPU packaging policy |
| `ci/` | Validation policies, parsers, and the compatibility `test_all.sh` alias |
| `ci/tests/` | Isolated fault-injection and orchestration tests for CI tooling |
| `dev/` | Explicit, manually invoked developer and workload helpers |

Developer maintenance helpers under `dev/` include
`bindings.sh` (the single checked-in cuDF/cuVS bindings entry) and
`clean_target_artifacts.sh` (report / mold-temp / `cargo clean`). Cleanup is
never an implicit side effect of a test command. Root CI checks keep native
source registry, component package archives, and test-lane policy failures
separate in the full gate.

Repository-owned policy data lives in `.config/policies/`; standard tool
configuration such as `.config/nextest.toml` remains at the location expected
by that tool. When moving or adding a script, update this map and all callers in
the same change.

Native build helpers also run from this root and default to the repo-local
prefixes `target/native/cudf-install`, `target/native/cuvs-install`, and
`target/native/cugraph-build`. Native source is resolved only through the root
submodules under `components/*`; component-local wrapper scripts are not a
second entry point.
