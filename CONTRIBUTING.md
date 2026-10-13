# Contributing

Algeon builds and runs on Linux x86-64 hosts with an NVIDIA GPU;
[Prerequisites](website/docs/build/prerequisites.md) lists the exact
requirements. Questions and bug reports go to
[GitHub issues](https://github.com/algeon-dev/algeon/issues). Contributions
are licensed under [Apache-2.0](LICENSE), like the rest of the project.

## Getting started

- [Building from Source](website/docs/build/from-source.md) covers the
  recursive checkout, the four native builds and the Rust workspace.
- The [module ownership map](website/docs/design/index.mdx#crates-and-ownership)
  names the crate responsible for each area. Fix a gap in that crate instead
  of working around it downstream.
- [AGENTS.md](AGENTS.md) holds the coding conventions, the table of design
  documents that own each contract, and the validation rules. Component rules
  live in `components/{cudf,cugraph,cuvs,cuml}/AGENTS.md`. These files address
  coding agents; the same rules apply to people.

## Proposing a change

Open an issue before changing behavior or an API, and ask there when the
scope or design is unclear. A typo, a broken link or a small fix can go
straight to a pull request.

A bug report names the affected crate, the query or plan, reproduction steps,
sanitized configuration, and the expected and observed results. Note explicitly
if a finding has not yet been reproduced.

Work on a topic branch and keep each change to one topic.

## Lint exceptions

Workspace lint levels come from `[workspace.lints]` in the root `Cargo.toml`.
Fix the owning abstraction before suppressing a lint. When an exception is
necessary, use a per-item `#[expect(clippy::..., reason = "...")]` with a
concrete reason. Do not add crate-, module- or workspace-wide `#[allow]`
attributes, or use `#[allow]` as a migration baseline.

## Terminology

The [glossary](website/docs/reference/glossary.md) defines the project's
vocabulary. In code and documentation, `cugraph_*`, `cuvs_*` and `cuml_*` are
GPU UDTFs, and `gpu_*` and `algeon_explain_coverage` are metadata UDTFs. The
DataFusion API is `TableFunction`; UDF refers only to scalar, aggregate and
window functions. Preserve SQL names, Rust identifiers and catalog URLs.

## Validation

[Tests](website/docs/build/tests.md) has the commands. Run the formatting,
lint and behavior checks for the contracts you touched; ordinary work is done
once they pass. Run the full gate once the change is ready to merge.
Documentation-only changes need no Cargo or GPU checks.
[Acceptance runs](website/docs/build/tests.md#acceptance-runs-outside-the-gate)
against real datasets or external services stay outside the gate.

## Pull requests

There is no CI yet, so the pull request description carries the validation
evidence. Give it a conventional-commit title that names the area, such as
`fix(query-engine): release memory grant on cancelled pipeline tasks`, and
include:

- The linked issue, if there is one, and the changed behavior or contract.
- The exact commands you ran and their outcomes. Label focused checks,
  `--fast` and failure-only reruns as partial.
- Anything you could not verify.

For native or GPU checks, add the host environment and say whether the native
libraries came from this checkout's `target/native/` prefixes:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | grep release
rustc -V
. /etc/os-release && echo "$PRETTY_NAME"
```

A compile-only check says nothing about native or GPU correctness.

## AI-assisted contributions

You remain responsible for the change and for the conversation around it.
Review generated code and claims before submitting them, and reproduce a
suspected bug before reporting it. Say where assistance was material and which
assumptions or unknowns affect review. A human contributor must remain actively
engaged throughout the issue and pull request discussion.

Keep credentials, private data, internal endpoints and unredacted logs out of
prompts, issues and pull requests.
