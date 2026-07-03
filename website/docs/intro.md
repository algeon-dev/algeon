---
sidebar_position: 1
sidebar_label: Overview
slug: /
description: DataFusion-native backend and Flight SQL service layer for cuGraph, Iceberg, and structured GPU execution diagnostics.
---

# DataFusion Nexus

**DataFusion Nexus** is a DataFusion-native backend and service layer for
RAPIDS-backed analytics. Use it as a Rust dependency inside a custom backend, or
run the same integration as a Flight SQL service.

Nexus extends DataFusion with SQL-visible `cugraph_*` table functions,
Iceberg/lakehouse source integration, structured planning reports, typed error
facts, and optional cuDF execution for supported relational fragments. The GPU
path is an implementation surface behind DataFusion APIs, not a new SQL engine
boundary.

## What Nexus is for

- **Graph analytics in SQL** — build edge relations with ordinary SQL, pass the
  view to `cugraph_bfs`, `cugraph_pagerank`, `cugraph_louvain`, and other GPU
  algorithms, then compose the result as rows.
- **Custom backend services** — install `with_cudf_native` and
  `with_cugraph_sql` on a DataFusion `SessionStateBuilder`, then wrap the
  resulting session in your own API, authorization, tenancy, and domain model.
- **Lakehouse-aware execution** — read Iceberg tables from local or remote
  catalogs while keeping interactive views and edge relations in a mutable
  DataFusion workspace.
- **Structured diagnostics** — use planning reports, `FallbackReason`s,
  `ErrorCode`s, error `Facts`, and report schemas to explain admission,
  fallback, source access, graph validation, and runtime behavior.

## How it runs

Nexus can run in two shapes:

- **Embedded library** — your Rust service owns the DataFusion session and
  chooses which Nexus features to install.
- **Flight SQL service** — the provided server exposes the same DataFusion,
  cuGraph, Iceberg, memory-policy, and diagnostic surfaces over Arrow Flight SQL.

Supported relational fragments may be lowered into the DataFusion-free
`nexus-query-engine` IR and executed through cuDF. Unsupported fragments keep
the ordinary DataFusion CPU path, with structured reports explaining why.

Nexus does not currently accept Substrait plans, and it does not implement
single-query cross-GPU communication.

## Workspace at a glance

The repository has four Cargo workspace members:

- `datafusion-nexus` — DataFusion-facing adapter, optimizer wiring, table
  functions, server integration, Iceberg integration, execution wrappers, and
  report surfaces.
- `nexus-query-engine` — DataFusion-free native IR, admission, runtime policy,
  metrics, and executor.
- `datafusion-nexus-bench` — benchmark, report, fixture, and stress binaries.
- `datafusion-nexus-tools` — explicit developer/operator tools, including local
  E2E runners.

## Where to go next

- [Design](./design.mdx) — the integration surfaces, execution boundary, and
  diagnostic contracts.
- [cuGraph SQL API](./cugraph/overview.mdx) — discover and call the
  `cugraph_*` table functions.
- [Build & Test](./build-and-test.md) — commands, toolchain, and GPU test lanes.
- [Guides](./guides/benchmarks.md) — benchmark recipes and local E2E flows.
