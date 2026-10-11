# Algeon examples

Three standalone programs demonstrate relational SQL, graph analysis and vector
search through `algeon-datafusion`. Each creates a small dataset locally, prints
its result and attempt status, then closes the GPU backend. No dataset downloads
or running services are needed.

| Example | Question | API to read |
| --- | --- | --- |
| [query_parquet](src/bin/query_parquet.rs) | What is the paid revenue in each region? | Register Parquet, execute native SQL, read an attempt report |
| [graph_pagerank](src/bin/graph_pagerank.rs) | Which linked pages have the highest PageRank? | Register cuGraph SQL, run PageRank, join scores to page titles |
| [vector_search](src/bin/vector_search.rs) | Which catalog vectors are nearest to a query vector? | Register cuVS SQL, run exact kNN, join neighbors to item names |

## Run

Use a checkout configured according to [Building from Source](../../website/docs/build/from-source.md).
All programs use CUDA device 0 and declare `WholeDeviceExclusive` GPU memory
ownership. The [prerequisites](../../website/docs/build/prerequisites.md) describe
the GPU, CPU and toolchain requirements. Build cuDF for `query_parquet`, add
cuVS for `vector_search`, and add cuGraph for `graph_pagerank`.

From the repository root:

```bash
cargo run -p examples --bin query_parquet
cargo run -p examples --features cugraph --bin graph_pagerank
cargo run -p examples --features cuvs --bin vector_search
```

A bare `cargo run -p examples` runs `query_parquet`. On a shared development
host, prefix each command with `flock /tmp/cudf-gpu.lock` to coordinate with the
repository's GPU tests. These tiny inputs illustrate the API; they do not
measure performance.

## 1. Query Parquet

`query_parquet` writes four orders to a temporary Parquet file. SQL filters out
the unpaid order, groups by region and sums the amounts. The directory is
removed when the program finishes.

```text
+--------+---------+
| region | revenue |
+--------+---------+
| North  | 150     |
| South  | 80      |
+--------+---------+
```

The backend uses `NativeRequired`, which rejects a completed plan containing
DataFusion CPU execution. A successful run prints `execution=Native`.
To query another file, replace the fixture-writing block and pass its path to
`register_parquet`.

## 2. Rank linked pages

`graph_pagerank` registers four pages and five directed links. The
`cugraph_pagerank` table function computes scores on the GPU; SQL joins each
vertex to its title and orders the result. The expected order is `API`, `Home`,
`Guide`, `FAQ`. Scores are floating-point values and sum to approximately 1.

The backend uses `FunctionsOnly`: relational SQL runs in DataFusion, and the
explicit cuGraph function runs on the GPU. The session opts into graph SQL with
`try_with_cugraph_sql`. A successful run prints `execution=PartialNative`.

## 3. Search vectors

`vector_search` searches three two-dimensional catalog vectors for the two
nearest to `(0, 0)`. The UDTF receives an `Int64` id followed by `Float32`
dimensions; names are joined back after the search. `l2_expanded` returns
squared Euclidean distance.

The result contains `Notebook` at distance `0` and `Pencil` at distance `1`, in
that order. The backend uses `FunctionsOnly` and the session registers vector
SQL with `try_with_cuvs_sql`. A successful run prints `execution=PartialNative`.

## Query lifetime

Each program keeps backend construction, session installation and attempt
execution in its own source file. An attempt begins before SQL planning. Its
completion receiver is retained before execution, and awaited even when
planning or result collection fails. The disposition and terminal outcome are printed before propagating
the query error. `close_and_wait_async` runs after the session and streams
leave scope, including on an error return.

For application integration, see [DataFusion sessions](../../website/docs/integration/datafusion-session.mdx)
and [handling errors](../../website/docs/integration/handling-errors.mdx).
For remote clients, see [Flight SQL](../../website/docs/integration/flight-sql-server.mdx).
The previous REST/Flight acceptance fixture, lifecycle profiling probe and OTLP
adapter now live in [tools](../tools/README.md).
