# Contributing

Contributions are very welcome. This component is still early, so the most
valuable contributions are changes that make the service-facing ownership,
execution, memory, and interop contracts clearer and more reliable.

Work from the monorepo root. The root workspace owns this component, the
shared cuDF/RMM identity, the single `Cargo.lock`, and the canonical validation
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
appropriate local or external GPU environment.

Please make the test environment explicit in every PR. Include the relevant
CUDA version, GPU model when applicable, library version or commit, Rust
version, OS, and any environment variables that affect the build or runtime.

If a change cannot be fully tested, state that clearly in the PR and explain
what was tested, what was not tested, and why.

Initialize the pinned official cuVS source and use the root native helpers:

```shell
git submodule update --init --recursive components/cuvs/cuvs
bash scripts/build/bootstrap_cuvs.sh
bash scripts/build/build_libcuvs.sh
```

The default install is `target/native/cuvs-install`; set `CUVS_INSTALL_DIR`
explicitly for a compatible external prefix. Run focused checks from the root,
then use the bare `bash scripts/check_all.sh` gate for full validation. Do not
wrap that gate in `/tmp/cudf-gpu.lock`.
