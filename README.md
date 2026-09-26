<p align="center">
  <img src="https://algeon.dev/img/algeon-logo.png" alt="Algeon" width="420">
  <br>
  <em>Specialized GPU algorithms, composed through SQL.</em><br><br>
  <a href="https://algeon.dev">Website</a> |
  <a href="https://algeon.dev/blog/introducing-algeon">Intro</a> |
  <a href="https://docs.rs/algeon-datafusion">API Docs</a>
</p>

---

Algeon (`/ˈæl.dʒi.ən/`, pronounced "AL-jee-on") is a GPU query engine that lets you call specialized
GPU algorithms from SQL. It integrates with query engines like [Apache DataFusion](https://datafusion.apache.org/):
the Algeon adapter lowers supported physical plans to
a native [cuDF](https://docs.rapids.ai/api/cudf/stable/) engine and exposes
GPU algorithms from [cuGraph](https://docs.rapids.ai/api/cugraph/stable/),
[cuVS](https://docs.rapids.ai/api/cuvs/stable/), and
[cuML](https://docs.rapids.ai/api/cuml/stable/) as SQL UDTFs (user-defined
table functions).
Algeon aims to extend OLAP infrastructure with GPUs, enabling users to run graph, vector,
and machine-learning algorithms in SQL without writing custom glue code or ad-hoc pipelines.

## Features

- **GPU execution through DataFusion.** Integrates with DataFusion's physical
  optimizer, so users can run the whole query on the GPU or only parts of it.
- **Specialized algorithms and function discovery.** Run PageRank, BFS, Louvain,
  kNN, PCA, and other cuGraph, cuVS, and cuML algorithms through
  `cugraph_*`, `cuvs_*`, and `cuml_*` UDTFs. The
  [GPU UDTF catalog](https://algeon.dev/guides/gpu-function-discovery)
  describes each function's availability in the current session.
- **Parquet sources.** Read Parquet files from local storage or remote Iceberg tables.
- **Structured planning diagnostics.** Inspect GPU coverage and remediation through
  [`algeon_explain_coverage`](https://algeon.dev/docs/integration/gpu-coverage-validation)
  or Flight SQL `EXPLAIN GPU`. Planning reports and errors expose structured,
  machine-readable values.
- **GPU memory and resource management.** Native execution is admitted through a per-device ledger
  and runs within a service-configured memory cap enforced by the query allocator.

> [!WARNING]
> Algeon is experimental. APIs and query behavior may change
> incompatibly between releases.
