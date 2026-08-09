# Contributing

Contributions are very welcome. This component is still early, so the most
valuable contributions are changes that make the service-facing ownership,
execution, memory, and interop contracts clearer and more reliable.

Work from the monorepo root. The root workspace owns this component, its cuDF
and cuVS integrations, the single `Cargo.lock`, and the canonical validation
gate.

## Start With an Issue

Every pull request must start with an issue. Before opening a PR, open an issue
that explains why the change is needed, what problem it solves, and what
behavior or API contract it affects.

This keeps design context visible and reduces review churn, especially for
changes that touch GPU execution, memory movement, FFI behavior, or
DataFusion-Nexus integration paths.

## Pull Requests

In the PR description, include:

- The linked issue.
- A short summary of the change.
- The behavior, API, or contract that changed.
- The test environment you used.
- The exact tests, examples, or manual checks you ran.

## Testing

The root repository gate owns CI and GPU-lane admission for this component.
Contributors are still responsible for testing native changes in an
appropriate local or external GPU environment before requesting review.

Please make the test environment explicit in every PR. Include the relevant
CUDA version, GPU model when applicable, libcudf/libcugraph version or commit,
Rust version, OS, and any environment variables that affect the build or runtime.

If a change cannot be fully tested, state that clearly in the PR and explain
what was tested, what was not tested, and why.

Native source and build defaults are the root paths:

```shell
git submodule update --init --recursive components/cugraph/cugraph
CUDF_INSTALL_DIR="$PWD/target/native/cudf-install" \
CUVS_INSTALL_DIR="$PWD/target/native/cuvs-install" \
CUGRAPH_CMAKE_BUILD_DIR="$PWD/target/native/cugraph-build" \
  bash scripts/build/build_libcugraph.sh
```

Run focused component checks from the same root, then use the bare
`bash scripts/check_all.sh` for full validation. Do not wrap that gate in the
GPU lock; it acquires `/tmp/cudf-gpu.lock` internally.
