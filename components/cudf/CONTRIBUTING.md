# Contributing

Contributions are very welcome. This component is still early, so the most
valuable contributions are changes that make the service-facing ownership,
execution, memory, and interop contracts clearer and more reliable.

This guide is intentionally focused on the Rust crates and their native cuDF
boundary. The monorepo is Linux-only, and `components/cudf/cudf/` is the pinned
fork used for native builds.

## Start With an Issue

Every pull request must start with an issue. Before opening a PR, open an issue
that explains why the change is needed, what problem it solves, and what
behavior or API contract it affects.

This keeps design context visible and reduces review churn, especially for
changes that touch GPU execution, memory movement, FFI behavior, or
DataFusion-Nexus integration paths.

For a change whose design or scope is unclear, discuss the proposed solution in
the issue before writing code. A good contribution solves a demonstrated
problem or a concrete use case; unverified fixes, speculative abstractions, and
mechanical refactors without an agreed technical goal are not useful starting
points.

For a bug report, include the affected crate or native component, version or
commit, sanitized configuration, minimal reproduction steps, actual result,
and expected result. Source inspection alone is not confirmation of a bug; if
the behavior is not reproduced, describe it as a hypothesis and state what
remains unknown. When practical, verify the failure before fixing it and add a
regression test that fails when the fix is reverted.

## AI-Assisted Contributions

AI tools may help with investigation, implementation, and review, but a human
contributor must guide the work and remain accountable for the result.

- Participate meaningfully in every issue and pull request conversation. Do
  not delegate project communication entirely to an AI tool.
- Review, edit, and verify generated analysis or code before submitting it. You
  should understand every claim and change well enough to explain it during
  review.
- Disclose material AI assistance and any assumptions or unknowns that could
  affect review.
- Treat an AI-generated source hypothesis as a lead, not as evidence. For a
  bug report, reproduce the behavior against the affected Rust/native path in a
  relevant environment before calling it a confirmed bug.
- Do not include credentials, private configuration, or unredacted logs in
  prompts, issues, or pull requests.

The contributor, not the AI tool, is responsible for the correctness,
relevance, and maintainability of the submission.

## Workflow

### Branches

Use a descriptive topic branch for a pull request. Keep unrelated changes,
generated output, caches, and native build trees out of the pull request.
Extend the existing owner of a behavior when possible, and keep a change small
enough that its ownership and validation are easy to review.

### Pull Requests

The pull request title should be concise and make the affected layer and intent
clear. Conventional-commit-style titles are preferred, for example:

```text
fix(cudf-nexus-sys): preserve native ABI metadata
test: cover explicit execution provenance
docs: clarify GPU validation requirements
```

In the PR description, include:

- The linked issue.
- A short summary of the change.
- The user-facing, correctness, or maintenance value of the change.
- The behavior, API, or contract that changed.
- The layer that owns the change, especially when it crosses safe Rust, FFI,
  the C++ shim, native cuDF, runtime, or interop code.
- The test environment you used, including the CUDA version, GPU model when
  applicable, libcudf/cuDF version or commit, Rust version, OS, and relevant
  environment variables.
- The exact tests, examples, or manual checks you ran.
- Any known gaps, failures, or unverified paths.

For native or GPU changes, also identify the selected installed prefix or
preconfigured build tree and report the exact native checks and their results.
Do not describe a compile-only check as native or GPU qualification.

Keep pull requests focused and independently reviewable. If a change is large,
split it into coherent pieces when possible or explain in the issue why the
larger boundary is necessary. Changes that cross ownership boundaries should
receive review from the relevant maintainers when available.

## Local Setup

From the monorepo root, initialize the pinned cuDF source before building:

```shell
git submodule update --init --recursive components/cudf/cudf
```

The workspace MSRV is Rust 1.94. Formatting requires nightly Rust:

```shell
cargo +nightly fmt --all
```

Set at least one native discovery path before running checks. Use an installed
libcudf prefix:

```shell
export CUDF_INSTALL_DIR="$PWD/target/native/cudf-install"
```

or a preconfigured cuDF build tree:

```shell
export CUDF_CPP_BUILD_DIR="$PWD/components/cudf/cudf/cpp/build"
```

When both are set, `CUDF_INSTALL_DIR` takes precedence. The selected prefix,
CUDA loader environment, and root submodule pin must match the checkout being
tested.

## Testing

The root repository gate owns CI and GPU-lane admission for this component.
Contributors are still responsible for testing native changes in an
appropriate local or external GPU environment before requesting review.

Record the environment alongside every test result; the PR requirements above
specify the details to include.

Start with the narrowest relevant checks, then broaden validation according to
the change's blast radius. The standard workspace checks are:

```shell
cargo check --workspace --all-features --all-targets
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo +nightly fmt --all

cargo test -p rapids-interop --all-features
flock /tmp/cudf-gpu.lock cargo test -p cudf-nexus --all-features --test semantic_contracts
flock /tmp/cudf-gpu.lock cargo test --workspace --all-features --no-fail-fast
```

Serialize every GPU test with `flock /tmp/cudf-gpu.lock`. For changes to the
native shim or cuDF, run the relevant C++/CTest and Rust ABI or FFI contract
tests as well. Preserve the supplied runtime device, stream, and memory
resource in tests that exercise those contracts.

If a change cannot be fully tested, state that clearly in the PR and explain
what was tested, what was not tested, and why.
