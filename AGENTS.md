# AGENTS.md

Before editing, read the applicable `AGENTS.md` files along each target path,
including nested guidance not loaded at startup. Component contracts live in
`components/{cudf,cugraph,cuvs,cuml}/AGENTS.md`. For build, CI, or packaging
maintenance (including configuration outside `scripts/`), also read
[scripts/AGENTS.md](scripts/AGENTS.md).

## Working principles

- Fix gaps in the owning abstraction or contract, including component runtime,
  interop, FFI, and native discovery; avoid adapter or algorithm workarounds.
- Preserve unrelated edits, submodule state, generated output, caches, native
  build trees, public behavior, and contract tests unless the task changes them.
  Avoid opportunistic cleanup and native rebuilds outside the task.
- Start with the smallest working end-to-end version; add capabilities while
  keeping the product working.
- Prefer the smallest change that fully solves the requirement. Check existing
  capabilities and types before adding dependencies or parallel paths. Reuse
  or consolidate within scope, with one truth source; ask only for an
  unrequested public-contract change or expansion of scope.
- Add abstractions, configuration, or defensive branches only for supported
  inputs, failure modes, or invariants. Remove redundant paths only after
  confirming no supported caller or contract requires them.
- Before adding an operator, identify its allocation, facts, lifetime,
  diagnostics and test owners in existing code. Ask for clarification only
  when unresolved ownership affects the contract.

## Comments

- Omit comments that only narrate code. Comment non-obvious rationale,
  invariants, safety constraints, external quirks, or business rules the code
  cannot express.
- Public API documentation describes observable contracts, not incidental
  implementation details.
- An explicit `drop()` in production code needs a one-line comment on why the
  value cannot wait until end of scope (lock window, re-entrancy, `RefCell`
  re-borrow); otherwise delete the `drop()`. Test code is exempt.

## Technical writing and anti-slop

- Name specific conditions, hardware constraints, or user inputs; never
  attribute human actions, consciousness, or feelings to algorithms, data, or
  compilers.
- Do not use em-dashes in comments or technical documentation, generic
  adjective triplets, theatrical binary contrasts, or colon reveals.
- Vary sentence length and shape.
- Every paragraph must advance a concrete invariant, benchmark number,
  architectural boundary, or API contract. Delete any sentence that functions
  purely as ungrounded promotional uplift.
- Omit conversational filler, flattery, and apologies. State changed files,
  line ranges, non-obvious rationale, verification commands and outcomes, and
  open technical decisions directly.
- Before writing about the GPU query engine, physical optimization, or CUDA
  memory allocation, read and follow
  [`.config/policies/algeon_domain_anti_slop.md`](.config/policies/algeon_domain_anti_slop.md).

## Contract sources

Read the linked sections before changing the corresponding area. These local
website sources own the design and behavior contracts; update the owner
document, implementation and affected tests together when a contract changes.
Keep other guidance as links to those definitions, without duplicate contracts.

| Task | Required source |
| --- | --- |
| Query crate or GPU component ownership and dependencies | [Module boundaries](website/docs/design/index.mdx#crates-and-ownership) |
| Native operators, lowering, planning reports or schedules | [Native lowering](website/docs/design/native-lowering.mdx) and [GPU execution](website/docs/design/gpu-execution.mdx) |
| Admission, allocation, attempt lifecycle, resident facts or metrics | [Memory and query lifetime](website/docs/design/admission.mdx) |
| Error types, mappings or DataFusion error boundaries | [Error ownership](website/docs/design/errors.mdx) and [Handling errors](website/docs/integration/handling-errors.mdx) |
| Workspace manifests, publication or native builds | [Build overview](website/docs/build/overview.md) and [Prerequisites](website/docs/build/prerequisites.md) |
| Running checks, adding tests or maintaining CI | [Tests](website/docs/build/tests.md) and the [test harness](website/docs/build/test-harness.md) contracts |

Machine-readable configuration and owner-crate constants/rustdoc define exact
values and API signatures. Reconcile discrepancies with the linked contracts
before changing behavior; drafts, blog posts and examples are not normative.

## Imports

- Import items in the file's top `use` block, or in `mod tests` when only tests
  use them. Reach variants and associated functions through the imported type.
- Use `super::` only for the direct parent; use `crate::` above it.
- On name collisions, the local item keeps the bare name and foreign items use
  an origin prefix, such as `CudfDataType`, `ArrowDataType`, or
  `DataFusionResult`. Prefix both sides when both are foreign. Keep the local
  `Result` bare; use `std::result::Result` inline only when its arity differs.
- Inline paths are otherwise limited to visibility qualifiers, attributes,
  doc links, calls through imported modules, sibling private modules, and enum
  translation matches. Gate imports with the same `cfg` as their uses.

## Validation policy

Select commands from the [test guide](website/docs/build/tests.md) for the
contracts affected. Broaden focused checks only when failures or scope warrant
it. Ordinary implementation work finishes after the relevant focused lint and
behavior tests pass, with no task-related failures. Read-only reviews and
prose-only edits need no Cargo or GPU checks unless requested.

Run the full gate only for pre-merge validation or an explicit request; follow
the guide's launch and monitoring instructions. Report focused, `--fast` and
failure-only reruns as partial validation. For native work, report commands,
selected native prefixes, loader paths and GPU environment.

Use `bash scripts/dev/clean_target_artifacts.sh` only to report removable build
artifacts; cleanup is never implicit.

## Writing tests

Follow the [test layout contract](website/docs/build/test-harness.md#test-layout).

- Prefer extending existing coverage at the behavior boundary before creating
  new tests.
- Add tests for realistic observable regressions, non-trivial invariants or
  boundaries, and concrete bugs. Code changing or coverage increasing is not
  sufficient justification by itself.
- Do not test what the type system guarantees or mirror literals, mappings,
  obvious control flow, implementation details, or removed features unless
  absence is itself a contract.
- For concurrency, prefer deterministic coordination or controlled scheduling
  over sleeps when practical.
- A new test must fail when the code under test is reverted; if it does not,
  delete it.

## Public contracts

- Reports, TSV families, metrics, and native execution-plan wrappers are
  contracts. Update each format atomically in place with its workspace
  consumers, goldens and fixtures; retain no legacy shapes, version-dependent
  readers or invented values for removed behavior. Format identifiers, fields
  and headers carry no version numbers; fingerprint domain tags are hash inputs
  and exempt.
- Preserve workload environment variables and item shapes.
