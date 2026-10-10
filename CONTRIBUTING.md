# Contributing

Algeon development targets Linux. Keep this guide under 160 lines; detailed
contracts and commands belong in the linked owner documents.

## Getting started

- Follow [Prerequisites](website/docs/build/prerequisites.md) and
  [Building from Source](website/docs/build/from-source.md) for the recursive
  checkout, native libraries and workspace build.
- Use the [module ownership map](website/docs/design/index.mdx#crates-and-ownership)
  to find the crate responsible for a change. Fix gaps in that crate.
- Follow [AGENTS.md](AGENTS.md) and applicable component guidance for coding
  conventions, contract sources and repository maintenance rules.

## Proposing a change

Every pull request starts with an issue describing the problem and affected
behavior or API. Discuss unclear design or scope before implementing it.
For bug reports, include the affected crate, query or plan, reproduction
steps, sanitized configuration, and expected and observed results. Label
unreproduced findings as hypotheses.

Use a descriptive topic branch and keep changes scoped. Preserve unrelated
edits, generated output, caches and native build trees. For non-trivial work,
identify existing code to extend or replace before adding another path.
Avoid speculative abstractions and configuration.

## Code and documentation

Follow the owner documents for [native lowering](website/docs/design/native-lowering.mdx),
[memory and query lifetime](website/docs/design/admission.mdx), and
[error handling](website/docs/design/errors.mdx). Update the owning contract,
implementation and affected tests together when behavior changes.

Prefer extending existing abstractions. Add comments for rationale or safety
constraints that the code cannot express. Production `drop()` calls need a
one-line explanation of why the value cannot wait until scope exit.

### SQL function terminology

Call `cugraph_*`, `cuvs_*`, and `cuml_*` SQL calls GPU UDTFs: they return rows
in `FROM`. Call `gpu_list_functions`, `gpu_describe_function`,
`gpu_validate_call`, and `algeon_explain_coverage` metadata UDTFs. Use
DataFusion's `TableFunction` API name; reserve UDF for scalar, aggregate and
window APIs. Preserve SQL names, Rust identifiers and catalog URLs.

### Lint exceptions

Workspace lint levels come from `[workspace.lints]` in the root `Cargo.toml`.
Fix the owning abstraction before suppressing a lint. When an exception is
necessary, prefer per-item `#[expect(clippy::..., reason = "...")]` with a
concrete justification. Do not add crate-, module-, or workspace-wide
`#[allow]` attributes, or use `#[allow]` as a migration baseline.

## Testing and validation

Use [Tests](website/docs/build/tests.md) for commands. Run the relevant
formatting, lint and behavior checks, then finish once they pass. Broaden
coverage when failures or affected downstream contracts warrant it.
Documentation-only changes need no Cargo or GPU checks.

Reserve the full gate for pre-merge validation or an explicit request.
Focused checks, `--fast`, and failure-only reruns are partial validation.
Algeon currently runs checks locally, without GitHub CI/CD.

- Extend existing behavior coverage when possible. A bug fix needs a
  regression test that fails when the fix is reverted.
- Follow [Test layout](scripts/test-harness.md#test-layout) when adding or
  moving tests. Assert structured planning reports and diagnostics.
- Use nextest filtersets with the shared workspace configuration described in
  the test guide. Do not wrap nextest or the full gate in an outer GPU lock.
- Keep external-service and dataset acceptance runs outside default Cargo
  test targets; use the [acceptance commands](website/docs/build/tests.md#acceptance-runs-outside-the-gate).

Harness maintainers use [scripts/test-harness.md](scripts/test-harness.md).
Agents selecting checks or diagnosing failures use
[algeon-validation](.agents/skills/algeon-validation/SKILL.md).

## Pull requests

Use a concise conventional-commit title naming the affected area, such as
`fix(query-engine): release memory grant on cancelled pipeline tasks`.
Include the linked issue, changed behavior or contract, owning layer, exact
validation commands and outcomes, and any unverified paths.

For native checks, report the selected native prefixes, loader paths, GPU,
CUDA and Rust toolchains, and host OS. A compile-only check does not establish
native or GPU correctness.

## AI-assisted contributions

A human contributor remains responsible for the change and project
communication. Review generated code and claims before submitting them;
verify behavior before reporting an AI-generated hypothesis as a bug.
Disclose material AI assistance and assumptions or unknowns that affect review.
Do not delegate issue or PR conversations entirely to an AI tool.

Keep credentials, private data, sanitized cloud endpoints and unredacted
logs out of prompts, issues and pull requests.
