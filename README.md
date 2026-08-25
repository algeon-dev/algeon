# DataFusion Nexus

[Website](https://datafusion.nexus) |
[Intro](https://datafusion.nexus/blog/introducing-datafusion-nexus) |
[API Docs](https://docs.rs/datafusion-nexus)

<img src="website/static/img/nexus-logo-dark-theme.png" alt="DataFusion Nexus" width="420">

DataFusion Nexus is an experimental GPU query engine for [Apache DataFusion](https://datafusion.apache.org/).
It evaluates physical-plan candidates, selects those supported by [cuDF](https://docs.rapids.ai/api/cudf/stable/)
for GPU execution, and exposes [cuGraph](https://docs.rapids.ai/api/cugraph/stable/) and other algorithms as
SQL table functions. Data can be read directly from local Parquet files or remote Iceberg tables.
Physical-plan candidates that are not selected for GPU execution fall back to their native DataFusion implementations.

Nexus is designed to fit into your existing architecture. Planning decisions, function
capabilities, and errors are exposed as structured, machine-readable values, allowing
application code to reason about queries before and after execution.

## Features

- **DataFusion-native GPU execution.** Nexus lowers supported physical-plan
  candidates to a DataFusion-independent cuDF IR. Rewrites are whole-candidate
  by default; narrowly certified mixed plans can retain a DataFusion CPU
  boundary. Unsupported or cost-rejected candidates retain their executable
  DataFusion plans, and callers can require a final plan with no DataFusion
  CPU execution.
- **Graph algorithms and GPU function discovery.** `cugraph_*` table functions
  run PageRank, BFS, Louvain, ForceAtlas2, and other algorithms over registered
  edge relations and return ordinary rows. The shared
  [GPU function catalog](website/docs/guides/gpu-function-discovery.mdx) lets
  clients list and describe the cuGraph and cuVS contracts, validate available
  calls, and check whether each function is executable in the current session.
- **Parquet and Iceberg sources.** Native scans read local and S3-compatible
  Parquet, including Iceberg tables resolved through REST and Glue catalogs.
  Remote reads use cuDF's KvikIO-backed S3 data source; unsupported native
  Iceberg scans retain their executable CPU delegate. Remote catalogs are
  read-only, with an optional mutable local workspace for views and scratch
  tables.
- **Structured planning diagnostics.** Planning reports record candidate
  outcomes and the final plan disposition.
  [`nexus_explain_coverage`](website/docs/integration/gpu-coverage-validation.mdx)
  and Flight SQL `EXPLAIN GPU` expose the same coverage evidence and
  remediation guidance.
  Errors retain typed identities at the DataFusion boundary.
- **Per-device memory admission.** Every native attempt is admitted against an
  immutable GPU memory grant. Shapes without the required memory proof are
  rejected before GPU work begins; eligible source-chunk failures may retry at
  a smaller granularity, with structured facts reported when recovery is not
  possible.

> [!WARNING]
> DataFusion Nexus is still an experimental project. Its public APIs,
> query-planning behavior, and overall architecture are evolving rapidly
> and may change incompatibly between releases. Evaluate it carefully
> before using it in production, and use it at your own risk.
