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
| Whole relational candidate | The primary stage of candidate selection, which attempts to lower the entire physical plan before decomposing it into subtrees. When it does not lower, supported subtrees can still run as native fragments. | [Selection, fallback and mixed execution](../design/native-lowering.mdx#selection-fallback-and-mixed-execution) |
| Native fragment | A supported subtree that executes natively beneath a DataFusion operator when the full plan cannot be lowered as a single unit, sharing the attempt's memory grant. | [Partitions and result semantics](../design/native-lowering.mdx#partitions-and-result-semantics) |
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
| Bounded execution | A query runs inside its grant. An allocation beyond it returns a typed allocation error instead of spilling or replaying on CPU. | [The allocation path](../design/admission.mdx#allocation-enforcement) |
| Attempt | One bounded native execution of one admitted query on one device. Its evidence is sealed into a `QueryAttemptReport`. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Admission ticket | An attempt's one lazy admission request, claimed when its first native stream is polled and reused by every later fragment. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Attempt capability evidence `AttemptCapabilityEvidence` | Planning-owned evidence of the device capabilities and plan scopes admitted for one attempt. It carries no memory estimate. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Envelope `AdmissionEnvelope` | The physical resources built around a committed grant: attempt lane, root stream, allocation domain and worker scope. | [Admission](../design/admission.mdx#from-capability-validation-to-admission) |
| Allocation domain | The device-memory accounting scope an attempt allocates within, capped by its grant. Each device's resident cache has its own domain outside any grant. | [The allocation path](../design/admission.mdx#allocation-enforcement) |
| Ledger `DeviceLedger` | The per-device accounting authority for grants, resident cache charges and attempt slots. | [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants) |
| GPU execution pool `GpuExecutionPool` | The per-device Tokio multi-thread runtime that runs an attempt's synchronous GPU work as jobs, on `worker_count` threads named `algeon-gpu-{d}`. | [Runtimes and threads](../design/index.mdx#runtimes-and-threads) |
| Job class (lifecycle / execution) | The permit class of a pool job. Lifecycle jobs (`ContextSetup`, `QueryExecution`, `Cleanup`) draw from `max_active_attempts` permits per device, and execution jobs (`AttemptWork`) from `max_active_attempts * source_work_per_attempt`. | [Runtimes and threads](../design/index.mdx#runtimes-and-threads) |
| Blocking lane | The pool's `spawn_blocking` path for synchronous work that must not occupy a pool worker, such as chunk lane jobs and source reads. A submission takes its job-class permit without waiting and fails with `service_overloaded` when none is free. | [Runtimes and threads](../design/index.mdx#runtimes-and-threads) |
| Attempt lane `lane_count` | A per-device index slot that each admitted attempt holds for its whole lifetime as identity evidence. `lane_count` defaults to `max_active_attempts` and an explicit value must be at least that. | [Admission configuration](../integration/configuration.mdx#admission) |
| Execution lane | One unit of auxiliary attempt work: an auxiliary permit, a leased stream and the execution context derived from both. | [CUDA streams and dependencies](../design/gpu-execution.mdx#cuda-streams-and-dependencies) |
| Root stream | The CUDA stream of the `Root`-class stream lease that an attempt takes when its envelope is built and holds until release. | [CUDA streams and dependencies](../design/gpu-execution.mdx#cuda-streams-and-dependencies) |
| Stream lease `StreamLease` | A `cudf::Runtime` (one CUDA stream and its memory resource) borrowed from the device's stream pool as `Root` or `Auxiliary` class. It returns to the pool only after the cleanup thread synchronizes the stream. | [Capacity and allocation](../design/admission.mdx#device-profiles-and-grants) |
| Cooperative cancellation `ExecutionCancellation` | A cancellation token with an optional deadline that the engine checks at native-engine boundaries, such as the next source chunk, task, operator or materialization step. It does not interrupt a CUDA call already running. | [Completion and resident data](../design/admission.mdx#resident-handoff-and-release) |

## Data and caching

| Term | Meaning | Defined in |
| --- | --- | --- |
| Residency | The lifecycle of a GPU-cached value, from attempt-owned through evictable or pinned to destroyed, with every resident byte charged to the ledger. | [Completion and resident data](../design/admission.mdx#resident-handoff-and-release) |
| Provenance | The recorded origin of a GPU-resident value: device, stream and producing source or graph contract. Values move forward only within a matching provenance domain. | [Resident results and the query memory cap](../design/gpu-table-functions.mdx#resident-results-and-the-query-memory-cap) |

## Diagnostics and errors

| Term | Meaning | Defined in |
| --- | --- | --- |
| Planning report `PlanningReport` | The structured planning evidence: one outcome per candidate (`Selected`, `NotSupported`, `NotSelectedByCost`). | [Select an execution path](../design/native-lowering.mdx#selection-fallback-and-mixed-execution) |
| Retry advice | The recovery signal sealed into a `QueryAttemptReport` after the terminal outcome. The engine never retries from it; the caller submits a new attempt. | [Retry evidence](../design/errors.mdx#retry-evidence) |
| Code, kind, status | Three projections of one error identity: the most specific stable id, a coarse grouping, and the source condition (`Permanent`, `Temporary`, `Persistent`). | [One identity, several projections](../integration/handling-errors.mdx#one-identity-several-projections) |
