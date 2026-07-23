# DataFusion Nexus

[Website](https://datafusion.nexus) |
[Intro](https://datafusion.nexus/blog/introducing-datafusion-nexus) |
[API Docs](https://docs.rs/datafusion-nexus)

<img src="website/static/img/nexus-logo-dark-theme.png" alt="DataFusion Nexus" width="420">

DataFusion Nexus is a GPU query engine for [Apache DataFusion](https://datafusion.apache.org/).
It takes the physical plans DataFusion already produces and executes the parts
it supports on the GPU: SQL runs on [Nvidia cuDF](https://docs.rapids.ai/api/cudf/stable/),
graph algorithms from [cuGraph](https://docs.rapids.ai/api/cugraph/stable/) are
callable directly from SQL, and data is read from
[Apache Iceberg](https://iceberg.apache.org/) and Parquet. Operators it does not
support remain on the standard DataFusion CPU path.

Nexus is designed to sit inside your own architecture. Graph workloads are
exposed as ordinary SQL table functions over your existing data, and planning
decisions and errors are returned as structured, machine-readable values — so
both application code and agents can reason about a query before and after it
runs.

## Features

- **GPU SQL execution.** The parts of a query that Nexus supports execute on
  cuDF. Under the default `GpuPreferred` policy, the remainder runs on the
  standard DataFusion CPU path, so an unsupported operator is a fallback rather
  than a failure.
- **Graph algorithms in SQL.** The `cugraph_*` table functions (PageRank, BFS,
  Louvain, ForceAtlas2, and more) run cuGraph over an edge table or view and
  return ordinary rows, so a graph traversal, a join, and an aggregation can
  live in one query over the same data — no separate graph database and no copy
  to keep in sync.
- **Vector search and vector ML _(in progress)_.** The next accelerator on the
  same pattern is [cuVS](https://github.com/NVIDIA/cuvs): `cuvs_*` table
  functions for GPU nearest-neighbor search and vector workloads such as
  clustering and dimensionality reduction, callable from SQL over your existing
  tables.
- **Lakehouse data access.** Query Iceberg tables from REST catalogs, AWS Glue,
  or S3-compatible object stores, with KvikIO streaming Parquet from object
  storage directly into GPU memory. Remote catalogs are read-only; views and
  scratch tables can live in an opt-in mutable local workspace overlay.
- **Explainable planning.** Planning reports record which parts of a plan ran on
  the GPU, which fell back to CPU, and why. `nexus_explain_coverage('<sql>')`
  returns that verdict as machine-readable JSON before a query runs, and errors
  cross the DataFusion boundary as typed values with stable identity rather than
  log text — a surface application code and agents can consume directly.
- **Bounded GPU memory.** Bounded mode plans each query against a GPU memory
  budget: shapes without a provable memory contract are rejected at planning
  time, admitted queries receive their reservation before execution, and runtime
  pressure is absorbed by chunk-split retries. A query that exhausts its retries
  fails with structured facts rather than an opaque OOM.
