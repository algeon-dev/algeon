---
sidebar_position: 8
description: One-line definitions of Algeon-specific terms, each linking to the page that owns the full contract.
---

# Glossary

Algeon-specific vocabulary. Each term links to the page that owns its
contract, and that page is authoritative when the two disagree. Standard
DataFusion, Arrow and SQL terms are not defined here.

## Planning and lowering

| Term | Meaning | Defined in |
| --- | --- | --- |
| UDTF | A SQL function used in `FROM` that returns rows. `cugraph_*`, `cuvs_*` and `cuml_*` run a GPU algorithm; `gpu_*` and `algeon_explain_coverage` are metadata UDTFs that run none. | [GPU UDTF Catalog API](./gpu-udtf-catalog.mdx) |
| Candidate | A complete DataFusion physical plan, or a subtree, that normalization recognizes as possibly executable natively. Each one receives an outcome in the planning report. | [Selection, fallback and mixed execution](../design/native-lowering.mdx#selection-fallback-and-mixed-execution) |
| Whole relational candidate | The first thing selection tries: replacing the complete query. When it does not lower, supported subtrees can still run as native fragments. | [Selection, fallback and mixed execution](../design/native-lowering.mdx#selection-fallback-and-mixed-execution) |
| Native fragment | A supported subtree that runs natively under a DataFusion operator the whole query could not lower, sharing the attempt's one grant. | [Partitions and result semantics](../design/native-lowering.mdx#partitions-and-result-semantics) |
| GPU island | A GPU execution fragment entered through a UDTF call, as opposed to GPU work selected from a relational candidate. Coverage counts both. | [Final disposition and GPU path](./gpu-coverage-report.mdx#final-disposition-and-gpu-path) |
| Execution domain | Which runtime executes each part of the completed plan: GPU-native fragments and islands, or DataFusion CPU operators. | [Final disposition and GPU path](./gpu-coverage-report.mdx#final-disposition-and-gpu-path) |
| Final-plan requirement, disposition | A requirement such as `NoDataFusionCpu` is checked against the completed plan; the disposition (`DataFusion`, `Native` or `Mixed`) is read from that plan. | [Require GPU execution](../design/native-lowering.mdx#baseline-preservation-and-final-plan-requirements) |
| Lowering | Translating a recognized candidate into the engine's `QueryPlan` IR after DataFusion physical planning has fixed the operators. | [Walk through a grouped aggregate](../design/native-lowering.mdx#translate-the-work) |
| IR `QueryPlan` | Planning's closed, DataFusion-free intermediate representation, owned by the native stack and not exposed by the adapter's public API. | [Module boundaries](../design/index.mdx#crates-and-ownership) |
| Scan facts | The format-neutral description of a source scan (files, schema, pruning facts) that lowering hands to the engine in place of DataFusion source nodes. | [Projection and scan pruning](../design/data-path.mdx#projection-and-scan-pruning) |

## Admission and runtime

| Term | Meaning | Defined in |
| --- | --- | --- |
| Admission | The runtime gate that turns a native plan into one bounded GPU execution. `EXPLAIN` never enters it. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Grant `MemoryGrant` | The fixed device-byte budget an attempt receives at admission. | [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants) |
| Bounded execution | A query runs inside its grant; an allocation beyond it returns a typed allocation error instead of spilling or replaying on CPU. | [The allocation path](../design/admission.mdx#allocation-enforcement) |
| Attempt | One bounded native execution of one admitted query on one device. Its evidence is sealed into a `QueryAttemptReport`. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Admission ticket | An attempt's one lazy admission request, claimed when its first native stream is polled and reused by every later fragment. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Attempt capability evidence `AttemptCapabilityEvidence` | Planning-owned evidence of the device capabilities and plan scopes admitted for one attempt. It carries no memory estimate. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Envelope `AdmissionEnvelope` | The physical resources built around a committed grant: attempt lane, root stream, allocation domain and worker scope. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Allocation domain | The device-memory accounting scope an attempt allocates within, capped by its grant. Each device's resident cache has its own domain outside any grant. | [The allocation path](../design/admission.mdx#allocation-enforcement) |
| Ledger `DeviceLedger` | The per-device accounting authority for grants, resident cache charges and attempt slots. | [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants) |

## Data and caching

| Term | Meaning | Defined in |
| --- | --- | --- |
| Workspace overlay | A mutable DataFusion catalog layered over a read-only source catalog, for interactive DDL and short names. | [Source and workspace](../design/data-path.mdx#lakehouse-and-workspace-boundary) |
| Residency | The lifecycle of a GPU-cached value, from attempt-owned through evictable or pinned to destroyed, with every resident byte charged to the ledger. | [Completion and resident data](../design/admission.mdx#resident-handoff-and-release) |
| Provenance | The recorded origin of a GPU-resident value: device, stream and producing source or graph contract. Values move forward only within a matching provenance domain. | [Resident results and the query memory cap](../design/gpu-table-functions.mdx#resident-results-and-the-query-memory-cap) |

## Diagnostics and errors

| Term | Meaning | Defined in |
| --- | --- | --- |
| Planning report `PlanningReport` | The structured planning evidence: one outcome per candidate (`Selected`, `NotSupported`, `NotSelectedByCost`). | [Select an execution path](../design/native-lowering.mdx#selection-fallback-and-mixed-execution) |
| Retry advice | The recovery signal sealed into a `QueryAttemptReport` after the terminal outcome. The engine never retries from it; the caller submits a new attempt. | [Retry evidence](../design/errors.mdx#retry-evidence) |
| Code, kind, status | Three projections of one error identity: the most specific stable id, a coarse grouping, and the source condition (`Permanent`, `Temporary`, `Persistent`). | [One identity, several projections](../integration/handling-errors.mdx#one-identity-several-projections) |
