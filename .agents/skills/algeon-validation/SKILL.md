---
name: algeon-validation
description: Select focused checks for Algeon changes and diagnose validation failures. Use during development or when asked to run pre-merge validation.
---

# Algeon validation

Use the [test guide](../../../website/docs/build/tests.md) for commands and
[AGENTS.md](../../../AGENTS.md#validation) for when validation finishes.

## Select checks

- Inspect the diff and select existing tests at the affected behavior boundary.
  Use nextest filtersets from the test guide; keep the shared workspace build
  configuration to avoid unnecessary recompilation.
- For Rust changes, run the relevant behavior tests and formatting/lint checks.
  For script changes, use the affected script's existing self-test. Prose-only
  edits need a content/link review.
- Expand coverage only for a concrete failure or an affected downstream
  contract. `--fast` is not a routine follow-up to a passing focused check.
- For a full-gate run, follow the test guide's launch and monitoring
  instructions; inspect the run's summary and failing phase logs before
  choosing a rerun.

## Diagnose and report

- Repeat the original nextest filterset when rerunning failures. Do not wrap
  nextest or the full gate in an outer GPU lock.
- For lock waits or compile delays, consult the relevant section of
  [harness maintenance](../../../scripts/test-harness.md). Check the selected
  native prefixes and loader paths before considering a native rebuild.
- Report the commands run, their outcomes, and any unverified behavior.
  Label focused, `--fast`, and failure-only runs as partial validation.
  For native checks, include the selected prefixes, loader paths and GPU
  environment. Report an unavailable dependency as a validation gap.

When adding or moving tests, follow
[Test layout](../../../scripts/test-harness.md#test-layout). Load the harness's lane
and feature-coverage details only when changing those mechanisms.
