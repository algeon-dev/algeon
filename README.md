<p align="center">
  <img src="https://algeon.dev/img/algeon-logo.png" alt="Algeon" width="420">
  <br>
  <em>Specialized GPU algorithms, composed through SQL.</em><br><br>
  <a href="https://algeon.dev">Website</a> |
  <a href="https://algeon.dev/blog/introducing-algeon">Intro</a> |
  <a href="https://docs.rs/algeon-datafusion">API Docs</a>
</p>

---

Algeon brings specialized GPU algorithms into composable SQL workflows over
open data. [Apache DataFusion](https://datafusion.apache.org/) is the first
integration: the `algeon-datafusion` adapter lowers supported physical plans to
a native [cuDF](https://docs.rapids.ai/api/cudf/stable/) engine and exposes
cuGraph and cuVS algorithms as SQL table functions. Plans that are not selected
for native execution retain their DataFusion implementations.

## Features

- **GPU execution through DataFusion.** Integrates with DataFusion's physical
  optimizer, with an option to require a final plan without DataFusion CPU execution.
- **Graph algorithms and function discovery.** Run PageRank, BFS, Louvain,
  ForceAtlas2, and other [cuGraph](https://docs.rapids.ai/api/cugraph/stable/)
  algorithms through `cugraph_*` table functions. The
  [GPU function catalog](website/docs/guides/gpu-function-discovery.mdx)
  describes cuGraph and cuVS functions and their availability in the current session.
- **Parquet sources.** Read local and S3-compatible Parquet. Remote sources are
  read-only. Iceberg is **not supported in this release** and cannot be enabled
  from source: the DataFusion integration moved from apache/iceberg-rust to
  [apache/datafusion-iceberg](https://github.com/apache/datafusion-iceberg),
  which is not yet published to crates.io. Iceberg support returns once it
  ships a release.
- **Structured planning diagnostics.** Inspect GPU coverage and remediation through
  [`algeon_explain_coverage`](website/docs/integration/gpu-coverage-validation.mdx)
  or Flight SQL `EXPLAIN GPU`. Planning reports and errors expose structured,
  machine-readable values.
- **GPU memory limits.** Native execution is admitted through a per-device ledger
  and runs within a service-configured memory cap enforced by the query allocator.

> [!WARNING]
> Algeon is experimental. APIs and query behavior may change
> incompatibly between releases.
