<p align="center">
  <img src="website/static/img/nexus-logo-dark-theme.png" alt="DataFusion Nexus" width="420">
  <br>
  <a href="https://datafusion.nexus">Website</a> |
  <a href="https://datafusion.nexus/blog/introducing-datafusion-nexus">Intro</a> |
  <a href="https://docs.rs/datafusion-nexus">API Docs</a>
</p>

---

DataFusion Nexus is an experimental GPU query engine for [Apache DataFusion](https://datafusion.apache.org/).
It accelerates supported physical plans with [cuDF](https://docs.rapids.ai/api/cudf/stable/)
and exposes GPU algorithms through SQL table functions.
Plans not selected for GPU execution retain their DataFusion implementations.

## Features

- **DataFusion-native GPU execution.** Integrates with DataFusion's physical
  optimizer, with an option to require a final plan without DataFusion CPU execution.
- **Graph algorithms and function discovery.** Run PageRank, BFS, Louvain,
  ForceAtlas2, and other [cuGraph](https://docs.rapids.ai/api/cugraph/stable/)
  algorithms through `cugraph_*` table functions. The
  [GPU function catalog](website/docs/guides/gpu-function-discovery.mdx)
  describes cuGraph and cuVS functions and their availability in the current session.
- **Parquet and Iceberg sources.** Read local and S3-compatible Parquet, plus
  Iceberg tables through REST and Glue catalogs. Remote catalogs are read-only.
  Iceberg requires a source build: it is withheld from the published crates
  until iceberg-rust ships a DataFusion 55 compatible release.
- **Structured planning diagnostics.** Inspect GPU coverage and remediation through
  [`nexus_explain_coverage`](website/docs/integration/gpu-coverage-validation.mdx)
  or Flight SQL `EXPLAIN GPU`. Planning reports and errors expose structured,
  machine-readable values.
- **GPU memory limits.** Native execution is admitted through a per-device ledger
  and runs within a service-configured memory cap enforced by the query allocator.

> [!WARNING]
> DataFusion Nexus is experimental. APIs and query behavior may change
> incompatibly between releases.
