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

- `Candidate`: a complete DataFusion physical plan (or subtree) that
  normalization recognizes as potentially executable natively. Every candidate
  receives a structured outcome in the
  [`PlanningReport`](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Whole relational candidate`: relational selection replaces a complete
  candidate or none of it; Algeon does not extract arbitrary fragments. The one
  narrow exception is the
  [certified mixed boundary](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Lowering`: translating a recognized candidate into the engine's
  `QueryPlan` IR after DataFusion physical planning has already fixed operator
  decisions. See [Native Lowering & Final Plans](../design/native-lowering.mdx).
- `IR (`QueryPlan`)`: planning's closed, DataFusion-free intermediate
  representation. Unlike DataFusion's open `ExecutionPlan` trait, it is an
  owned type the native stack fully controls and the engine facade re-exports.
  See [Crates and ownership](../design/index.mdx#crates-and-ownership).
- `Scan facts`: the format-neutral description of a source scan (files,
  schema, pruning facts) that lowering hands to the engine in place of
  DataFusion source nodes. See
  [How a query runs](../design/index.mdx#how-a-query-runs).
- `GPU island`: an explicit GPU execution fragment entered through a
  specialized SQL call (`cugraph_*`, `cuvs_*`), as opposed to GPU work selected
  from a relational candidate. Counted together with native fragments in
  [coverage results](../integration/gpu-coverage-validation.mdx#final-disposition-and-gpu-path).
- `Certified mixed boundary`: the single supported exception to
  whole-candidate replacement: a final CPU `ILIKE` filter over one
  local-Parquet source, certified by an exact shape-and-ownership proof. See
  [Selection, fallback, and mixed execution](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Final-plan requirement / disposition`: a requirement (such as
  `NoDataFusionCpu`) is checked against the exact completed plan; the
  disposition (`DataFusion`, `Native`, `Mixed`, or terminal `Rejected`) is
  derived by walking that plan. See
  [Baseline preservation](../design/native-lowering.mdx#baseline-preservation-and-final-plan-requirements).
- `Execution domain`: which runtime executes each part of the completed
  plan: GPU-native fragments and islands versus DataFusion CPU operators. This
  is what
  [`algeon_explain_coverage`](../integration/gpu-coverage-validation.mdx)
  reports before execution.

## Admission and runtime

- `Admission`: the runtime gate that turns a native plan into one bounded
  GPU execution: capability-first, rejected locally with structured reasons,
  never entered by `EXPLAIN`. See
  [Admission & Memory Governance](../design/admission.mdx).
- `Attempt`: one bounded native execution of one admitted query on one
  selected device. Its evidence is sealed into a `QueryAttemptReport`. See
  [Admission & Memory Governance](../design/admission.mdx).
- `Attempt capability evidence (`AttemptCapabilityEvidence`)`:
  planning-owned evidence for the device capabilities and exact plan scopes
  admitted for one attempt. It carries no memory estimate. See
  [From capability validation to admission](../design/admission.mdx#from-capability-validation-to-admission).
- `Admission ticket`: a planner-created, lazy admission request; device and
  grant acquisition wait until the first native stream is polled. See
  [Admission & Memory Governance](../design/admission.mdx).
- `Grant (`MemoryGrant`)`: the immutable device-byte budget an attempt
  receives at commit. It is the ceiling of the attempt's allocation domain and
  never grows, shrinks, borrows, or moves to another device. See
  [Device profiles and grants](../design/admission.mdx#device-profiles-and-grants).
- `Ledger (`DeviceLedger`)`: the per-device integer accounting authority
  for grants, resident cache charges, and attempt slots. It is capacity
  arithmetic, not a residency oracle. See
  [Device profiles and grants](../design/admission.mdx#device-profiles-and-grants).
- `Envelope (`AdmissionEnvelope`)`: the physical resources built around a
  committed grant: attempt lane, root stream, allocation domain, and worker
  scope. Execution sees it only complete. See
  [Resident handoff and release](../design/admission.mdx#resident-handoff-and-release).
- `Allocation domain`: the device-memory accounting scope an attempt
  allocates within, capped by its grant; each device's resident cache has its
  own shared domain outside any grant. See
  [Allocation enforcement](../design/admission.mdx#allocation-enforcement).
- `Bounded execution`: a query runs inside the GPU memory cap it was
  granted; an allocation beyond that cap returns a typed allocation error
  instead of spilling or replaying on CPU. Planning checks kernels, types,
  semantics, and plan scope, not a predicted byte requirement. See
  [Allocation enforcement](../design/admission.mdx#allocation-enforcement).

## Data and caching

- `Workspace overlay`: a mutable DataFusion catalog layered over a
  read-only source catalog, giving interactive DDL and short names without
  teaching the source catalog to accept views. See
  [Lakehouse and workspace boundary](../design/index.mdx#lakehouse-and-workspace-boundary).
- `Residency`: the lifecycle of GPU-cached values:
  `AttemptOwned -> Evictable <-> Pinned -> Destroyed`, with every resident
  byte charged to the device ledger.
- `Provenance`: the recorded origin of a GPU-resident value: the device,
  stream, and producing source or graph contract. Interop hands values forward
  only within a matching provenance domain instead of round-tripping through
  host memory. See [cuGraph SQL path](../design/index.mdx#cugraph-sql-path).

## Diagnostics and errors

- ``PlanningReport``: the public, structured planning evidence: one outcome
  per candidate (`Selected`, `NotSupported`, `NotSelectedByCost`). Assert it
  instead of printed plan text. See
  [Selection, fallback, and mixed execution](../design/native-lowering.mdx#selection-fallback-and-mixed-execution).
- `Retry advice`: the sealed recovery signal in `QueryAttemptReport`,
  derived only after terminal outcome and GPU cleanup. The engine never
  retries from it; callers satisfy its prerequisite and submit a new attempt.
  See [Retry and recovery](../integration/handling-errors.mdx#retry-and-recovery-are-evidence-based).
- `Code / kind / status`: the three projections of one error identity:
  `code` is the most specific stable id, `kind` a coarse grouping, `status`
  the source condition (`Permanent`, `Temporary`, `Persistent`). See
  [One identity, several projections](../integration/handling-errors.mdx#one-identity-several-projections).
