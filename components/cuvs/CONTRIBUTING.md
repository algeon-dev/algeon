# Contributing

Contributions are very welcome. This repository is still early, so the most
valuable contributions are changes that make the service-facing ownership,
execution, memory, and interop contracts clearer and more reliable.

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

This repository does not currently have GitHub Actions runners with GPU
coverage. Contributors are responsible for testing their changes in an
appropriate local or external GPU environment.

Please make the test environment explicit in every PR. Include the relevant
CUDA version, GPU model when applicable, library version or commit, Rust
version, OS, and any environment variables that affect the build or runtime.

If a change cannot be fully tested, state that clearly in the PR and explain
what was tested, what was not tested, and why.
