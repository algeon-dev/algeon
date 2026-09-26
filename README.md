<p align="center">
  <img src="https://algeon.dev/img/algeon-logo.png" alt="Algeon" width="420">
  <br>
  <em>GPU analytics on any SQL result.</em><br><br>
  <a href="https://algeon.dev">Website</a> |
  <a href="https://algeon.dev/blog/introducing-algeon">Intro</a> |
  <a href="https://docs.rs/algeon-datafusion">API Docs</a>
</p>

---

Algeon (`/ˈæl.dʒi.ən/`, pronounced "AL-jee-on") is an experimental GPU query
engine. Any SQL result can feed a graph, vector, or machine-learning
algorithm on the GPU, and the algorithm's output comes back as rows the same
query can join, filter, and aggregate. Algeon is built on
[Apache DataFusion](https://datafusion.apache.org/): supported relational
plans run on [cuDF](https://docs.rapids.ai/api/cudf/stable/), and algorithms
from [cuGraph](https://docs.rapids.ai/api/cugraph/stable/),
[cuVS](https://docs.rapids.ai/api/cuvs/stable/), and
[cuML](https://docs.rapids.ai/api/cuml/stable/) are SQL table functions
(UDTFs) whose relation arguments are ordinary subqueries.

A relational query already reduces a terabyte-scale lakehouse to a working
set of a few gigabytes, and that working set fits on one GPU. The questions
worth asking about it (shortest paths, communities, nearest neighbors, model
scores) are the ones SQL alone handles badly, and today each of them costs an
export pipeline and a notebook. Algeon keeps them inside the query, so a
person or an agent can ask them for the price of one statement.

```sql
-- Rank the densely connected core of a citation graph, in one plan.
SELECT p.year, p.title
FROM cugraph_pagerank(edges => (
  SELECT src, dst
  FROM cugraph_k_core(edges => (SELECT src, dst FROM citation_edges), k => 30)
)) r
JOIN papers p ON p.paper_id = r.vertex
ORDER BY r.value DESC
LIMIT 5;
```

## What it does

- Runs supported DataFusion physical plans on cuDF, whole or in part, and
  keeps the DataFusion CPU plan for everything else.
- Exposes 27 cuGraph, 8 cuVS, and 7 cuML algorithms as `cugraph_*`,
  `cuvs_*`, and `cuml_*` UDTFs. Composed GPU stages pass data to each other
  without a host round trip, and `cuml_forest_predict` scores rows with a
  pretrained XGBoost, LightGBM, or Treelite forest.
- Reads Parquet from local disk or S3-compatible object storage, read-only.
- Reports where a query will run before it runs, through
  [`algeon_explain_coverage`](https://algeon.dev/integration/gpu-coverage-validation)
  or `EXPLAIN GPU` over Flight SQL, and lets a person or an agent list,
  describe, and dry-run validate every installed UDTF through the
  [GPU UDTF catalog](https://algeon.dev/guides/gpu-function-discovery).
- Returns failures with stable machine-readable codes, kinds, statuses, and
  remedies.
- Admits each query against a per-device memory ledger and runs it inside an
  immutable cap enforced by the query's allocator.
- Embeds as a Rust crate in a DataFusion session or runs as an Arrow Flight
  SQL server.

> [!WARNING]
> Algeon is experimental. APIs and query behavior may change incompatibly
> between releases, one query runs on one GPU, and nothing spills to host
> memory or disk. The [documentation](https://algeon.dev) states each boundary.

## Where to start

- [Build & Test](https://algeon.dev/build) for a source build or the Docker
  image.
- [Integrating Algeon](https://algeon.dev/integration) to embed a session or
  start the server.
- The [cuGraph](https://algeon.dev/cugraph), [cuVS](https://algeon.dev/cuvs),
  and [cuML](https://algeon.dev/cuml) SQL APIs, with runnable examples over a
  public citation network.
- [Introducing Algeon](https://algeon.dev/blog/introducing-algeon) for the
  reasoning behind the design.
