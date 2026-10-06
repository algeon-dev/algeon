---
sidebar_position: 6
description: One-line definitions of Algeon-specific terms, each linking to the page that owns the full contract.
---

# Glossary

Algeon-specific vocabulary, one or two lines per term. Each entry links to the
page that owns the full contract; that page stays authoritative when the two
disagree. Standard DataFusion, Arrow, and SQL terms are intentionally not
defined here.

## Planning and lowering

- `UDTF` (user-defined table function): a SQL function used in `FROM` that
  returns rows. Algeon's `cugraph_*`, `cuvs_*`, and `cuml_*` calls are GPU
  execution UDTFs. The `gpu_*` catalog calls and `algeon_explain_coverage`
  are metadata UDTFs; they do not run a GPU algorithm. Algeon's relation
  planners bind execution UDTFs, while DataFusion registers the catalog UDTFs
  as `TableFunction`s. The catalog API names each UDTF a function, as in
  `gpu_list_functions` and its `function_name` column. See
  [GPU UDTF Catalog API](./gpu-udtf-catalog.mdx).
- `Candidate`: a complete DataFusion physical plan (or subtree) that
  normalization recognizes as potentially executable natively. Every candidate
  receives a structured outcome in the
  [`PlanningReport`](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Whole relational candidate`: relational selection first tries to replace
  the complete query. When it does not lower, supported subtrees below the
  operators that stay on DataFusion can run as
  [native fragments](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Lowering`: translating a recognized candidate into the engine's
  `QueryPlan` IR after DataFusion physical planning has already fixed operator
  decisions. See [Walk through a grouped aggregate](../design/native-lowering.mdx#translate-the-work).
- `IR` (`QueryPlan`): planning's closed, DataFusion-free intermediate
  representation. Unlike DataFusion's open `ExecutionPlan` trait, it is an
  owned type the native stack fully controls, and the adapter's public API
  does not expose it.
  See [Module boundaries](../design/index.mdx#crates-and-ownership).
- `Scan facts`: the format-neutral description of a source scan (files,
  schema, pruning facts) that lowering hands to the engine in place of
  DataFusion source nodes. See
  [Projection and scan pruning](../design/data-path.mdx#projection-and-scan-pruning).
- `GPU island`: an explicit GPU execution fragment entered through a
  specialized UDTF call (`cugraph_*`, `cuvs_*`, `cuml_*`), as opposed to GPU work selected
  from a relational candidate. Counted together with native fragments in
  [coverage results](../integration/gpu-coverage-validation.mdx#final-disposition-and-gpu-path).
- `Native fragment`: a supported subtree that runs natively under a
  DataFusion operator the whole query could not lower. It replaces its
  DataFusion child only when the parent keeps its input schema, boundedness,
  ordering, distribution, and co-partitioning, and it shares the attempt's one
  grant with every other fragment. See
  [Partitions and result semantics](../design/native-lowering.mdx#partitions-and-result-semantics).
- `Final-plan requirement` / `disposition`: a requirement (such as
  `NoDataFusionCpu`) is checked against the exact completed plan; the
  disposition (`DataFusion`, `Native`, or `Mixed`) is derived by walking that
  plan. A planning rejection produces no executable plan. See
  [Require GPU execution](../design/native-lowering.mdx#baseline-preservation-and-final-plan-requirements)
  and [Final disposition and GPU path](../integration/gpu-coverage-validation.mdx#final-disposition-and-gpu-path).
- `Execution domain`: which runtime executes each part of the completed
  plan: GPU-native fragments and islands versus DataFusion CPU operators. This
  is what
  [`algeon_explain_coverage`](../integration/gpu-coverage-validation.mdx#final-disposition-and-gpu-path)
  reports before execution.

## Admission and runtime

- `Admission`: the runtime gate that turns a native plan into one bounded
  GPU execution: capability-first, rejected locally with structured reasons,
  never entered by `EXPLAIN`. See
  [Admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Attempt`: one bounded native execution of one admitted query on one
  selected device. Its evidence is sealed into a `QueryAttemptReport`. See
  [Admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Attempt capability evidence` (`AttemptCapabilityEvidence`):
  planning-owned evidence for the device capabilities and exact plan scopes
  admitted for one attempt. It carries no memory estimate. See
  [Admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Admission ticket`: an attempt's one lazy admission request. It claims a
  device and grant when the attempt's first native stream is polled, and every
  later native fragment of the attempt reuses that grant. See
  [Admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Grant` (`MemoryGrant`): the immutable device-byte budget an attempt
  receives at commit. It is the ceiling of the attempt's allocation domain and
  never grows, shrinks, borrows, or moves to another device. See
  [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants).
- `Ledger` (`DeviceLedger`): the per-device integer accounting authority
  for grants, resident cache charges, and attempt slots. It is capacity
  arithmetic, not a residency oracle. See
  [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants).
- `Envelope` (`AdmissionEnvelope`): the physical resources built around a
  committed grant: attempt lane, root stream, allocation domain, and worker
  scope. Execution sees it only complete. See
  [Admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Allocation domain`: the device-memory accounting scope an attempt
  allocates within, capped by its grant; each device's resident cache has its
  own shared domain outside any grant. See
  [The allocation path](../design/admission.mdx#allocation-enforcement).
- `Bounded execution`: a query runs inside the GPU memory cap it was
  granted; an allocation beyond that cap returns a typed allocation error
  instead of spilling or replaying on CPU. Planning checks kernels, types,
  semantics, and plan scope, not a predicted byte requirement. See
  [The allocation path](../design/admission.mdx#allocation-enforcement).

## Data and caching

- `Workspace overlay`: a mutable DataFusion catalog layered over a
  read-only source catalog, giving interactive DDL and short names without
  teaching the source catalog to accept views. See
  [Source and workspace](../design/data-path.mdx#lakehouse-and-workspace-boundary).
- `Residency`: the lifecycle of GPU-cached values:
  `AttemptOwned` → `Evictable` ↔ `Pinned` → `Destroyed`, with every resident
  byte charged to the device ledger. See
  [Completion and resident data](../design/admission.mdx#resident-handoff-and-release).
- `Provenance`: the recorded origin of a GPU-resident value: the device,
  stream, and producing source or graph contract. Interop hands values forward
  only within a matching provenance domain instead of round-tripping through
  host memory. See [Resident results and the query memory cap](../design/gpu-table-functions.mdx#resident-results-and-the-query-memory-cap).

## Diagnostics and errors

- `PlanningReport`: the public, structured planning evidence: one outcome
  per candidate (`Selected`, `NotSupported`, `NotSelectedByCost`). Assert it
  instead of printed plan text. See
  [Select an execution path](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Retry advice`: the sealed recovery signal in `QueryAttemptReport`,
  derived only after terminal outcome and GPU cleanup. The engine never
  retries from it; callers satisfy its prerequisite and submit a new attempt.
  See [Retry and recovery](../integration/handling-errors.mdx#retry-and-recovery-are-evidence-based).
- `Code / kind / status`: the three projections of one error identity:
  `code` is the most specific stable id, `kind` a coarse grouping, `status`
  the source condition (`Permanent`, `Temporary`, `Persistent`). See
  [One identity, several projections](../integration/handling-errors.mdx#one-identity-several-projections).
