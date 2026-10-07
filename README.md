<p align="center">
  <img src="https://algeon.dev/img/algeon-logo.png" alt="Algeon" width="420">
  <br>
  <em>One Query, Any Data, Any Algorithm.</em><br><br>
  <a href="https://algeon.dev">Website</a> |
  <a href="https://algeon.dev/blog/introducing-algeon">Intro</a> |
  <a href="https://docs.rs/algeon-datafusion">API Docs</a>
</p>

---

Algeon (`/ˈæl.dʒi.ən/`, pronounced "AL-jee-on") is an experimental GPU query
engine that runs graph, vector, and machine-learning algorithms over any
relation SQL can produce, and returns their output as rows the same query
can join, filter, aggregate, or pass to the next function. Whatever a schema
or a semantic layer already maps to tables, one statement can now traverse,
cluster, search, or score. Algeon is built on
[Apache DataFusion](https://datafusion.apache.org/): supported relational
plans run on [cuDF](https://docs.rapids.ai/api/cudf/stable/), and algorithms
from [cuGraph](https://docs.rapids.ai/api/cugraph/stable/),
[cuVS](https://docs.rapids.ai/api/cuvs/stable/), and
[cuML](https://docs.rapids.ai/api/cuml/stable/) are SQL table functions
(UDTFs) whose relation arguments are ordinary subqueries.

Algeon runs GPU-accelerated SQL on fine-grained data: the exact rows and
columns one question needs, which a relational query typically reduces from
a whole lake to a few gigabytes. The follow-up questions about that working
set (which
supplier sits upstream of two late product lines, which hosts a stolen
credential reached next, which papers connect two ideas, which rows a model
would flag) have mature GPU implementations. Each of them has usually cost an
export pipeline and a notebook. Algeon keeps them inside the query, and a
person or an agent can test the next hypothesis for the price of one more
statement, which makes many more questions worth asking. Your application
supplies the agent and the domain definitions; Algeon supplies query
execution and the diagnostics beneath it.

```sql
-- Find foundational papers BERT relies on without directly citing them.
CREATE OR REPLACE VIEW bert_seed AS
SELECT paper_id AS vertex, CAST(1.0 AS DOUBLE) AS value
FROM papers
WHERE title = 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
  AND year = 2018;

SELECT p.year, p.title, ROUND(r.value, 6) AS ppr
FROM cugraph_personalized_pagerank(
  edges => (SELECT src, dst FROM citation_edges),
  personalization => (SELECT vertex, value FROM bert_seed)) r
JOIN papers p ON p.paper_id = r.vertex
WHERE NOT EXISTS (SELECT 1 FROM bert_seed s WHERE s.vertex = r.vertex)
  AND NOT EXISTS (SELECT 1 FROM citation_edges e JOIN bert_seed s ON s.vertex = e.src
                  WHERE e.dst = r.vertex)
ORDER BY r.value DESC
LIMIT 5;
```

The `bert_seed` view isolates the starting paper, and `cugraph_personalized_pagerank`
runs a GPU random walk biased toward that seed across 45.6 million citation edges.
The `NOT EXISTS` clauses prune BERT itself and its direct bibliography, leaving
foundational predecessors across continuous speech recognition and statistical
machine translation alongside LSTM. The output is ordinary rows that a `JOIN`
attaches titles to. The
[citation demo dataset](https://algeon.dev/guides/datasets/citation-network) behind this
query has 4.9 million papers and 45.6 million citation edges.

## What it does

- Runs supported DataFusion physical plans on cuDF, whole or in part, and
  keeps the DataFusion CPU plan for the rest unless the execution mode
  requires an all-GPU plan.
- Exposes cuGraph, cuVS, and cuML algorithms as `cugraph_*`, `cuvs_*`, and
  `cuml_*` UDTFs, from graph traversal and community detection to
  nearest-neighbor search, clustering, and pretrained forest inference. The
  function pages list every installed algorithm. When one GPU function's
  output feeds another directly, the data stays on the device.
- Reads Parquet from local disk or S3-compatible object storage on the GPU,
  read-only; other source types stay on the DataFusion path.
- Reports where a query will run before it runs, through
  [`algeon_explain_coverage`](https://algeon.dev/integration/gpu-coverage-validation)
  or `EXPLAIN GPU` over Flight SQL, and lets a person or an agent list,
  describe, and dry-run validate every installed UDTF through the
  [GPU UDTF catalog](https://algeon.dev/guides/gpu-udtf-discovery).
- Returns its own failures with a stable machine-readable code, kind, and
  status, which lets a wrong first query inform the second attempt.
- Admits each query against a per-device memory ledger and runs it inside an
  immutable cap enforced by the query's allocator.
- Embeds as a Rust crate in a DataFusion session or runs as an Arrow Flight
  SQL server. Linux with an NVIDIA GPU and a CUDA toolkit is required.

> [!WARNING]
> Algeon is experimental. Stability and result verification still need
> substantial work, and APIs and query behavior may change incompatibly
> between releases. One query runs on one GPU inside a configured memory cap,
> and nothing spills to host memory or disk; a small SQL input alone does not
> guarantee that an algorithm fits. The [documentation](https://algeon.dev)
> states each boundary. Evaluate Algeon on known datasets with trusted
> reference results.

## Where to start

- The [MerRec demos](https://algeon.dev/guides/datasets/merrec): eight
  statements over 1.27 billion marketplace events, from the brands that
  moved toward Barbie's audience to a campaign list built from a graph
  algorithm, a vector search and a trained model in one statement.
- [Build & Test](https://algeon.dev/build) for a source build or the Docker
  image.
- [Integrating Algeon](https://algeon.dev/integration) to embed a session or
  start the server.
- The [cuGraph](https://algeon.dev/cugraph), [cuVS](https://algeon.dev/cuvs),
  and [cuML](https://algeon.dev/cuml) SQL APIs, each with captured examples
  over the demo datasets.
- [Introducing Algeon](https://algeon.dev/blog/introducing-algeon) for the
  vision, the reasoning behind the design, and how Algeon could fit alongside
  an existing data platform.
