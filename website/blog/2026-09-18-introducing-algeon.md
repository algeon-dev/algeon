---
title: "Introducing Algeon"
description: Algeon is an experimental GPU query engine in which any SQL result can feed a graph, vector, or ML algorithm, and the output comes back as rows. This post explains why we are building it, who it is for, and how far along it is.
slug: introducing-algeon
tags: [announcement]
authors: morris
---

import {StackDiagram} from '@site/src/components/design/Diagram';
import {GlueVsPlanFigure, WorkingSetFigure} from '@site/src/components/blog/IntroFigures';

Algeon is a GPU query engine built on
[Apache DataFusion](https://datafusion.apache.org/). Its organizing idea is
that any SQL result can be the input of a specialized GPU algorithm, and that
the algorithm's output should come back as rows the same query can keep
joining, filtering, and aggregating. Graph algorithms from
[cuGraph](https://docs.rapids.ai/api/cugraph/stable/), vector search from
[cuVS](https://docs.rapids.ai/api/cuvs/stable/), and model fitting and scoring
from [cuML](https://docs.rapids.ai/api/cuml/stable/) appear as table functions
inside ordinary SQL. Relational work runs on
[cuDF](https://docs.rapids.ai/api/cudf/stable/) when the plan qualifies and
stays on DataFusion's CPU operators when it does not.

Algeon is experimental, and this post is as much about what it cannot do yet
as about what it can.

{/* truncate */}

## The query is no longer the expensive part

A lakehouse holds terabytes, sometimes petabytes, and almost no single
question needs more than a sliver of it. A question is about one tenant, one
quarter, one product line, one cohort. The relational layer does the
narrowing: a filter on a date range, a join against a dimension table, an
aggregation over a segment. Terabytes go in and a few gigabytes come out.
Fleet studies put numbers on how small the survivor usually is. In the Amazon
Redshift fleet the median query uses 0.6 GB of memory and the 99th-percentile
query 21.4 GB; among paying BigQuery customers, 90% of queries process less
than 100 MB.[^1]

A working set of that size fits on one GPU with room to spare; a workstation
card now ships with 96 GB of memory. DuckDB and DataFusion grew on the same
observation: after relational reduction, most analytical working sets do not
need a distributed system.

The questions people want to ask of that reduced set, though, are often the
ones relational algebra handles badly. Which of these accounts form a tightly
knit community? What is the shortest chain from this supplier to that
sanctioned entity? Which items sit nearest to this one in embedding space?
Which of these rows would the model we trained last quarter flag? Each has a
well-understood algorithm and a mature GPU implementation, and none of them
is a join.

<WorkingSetFigure
  caption={<>The relational layer reduces the lakehouse to a working set that
  fits one GPU. Algeon places the specialized algorithm after that reduction,
  inside the same statement, and returns its output as rows that ordinary SQL
  can keep working on, or that the next algorithm can take as input.</>}
/>

The handoff worth optimizing, then, is the one after the query: from a working
set that already fits one device to the algorithm that answers the question.
Algeon makes that handoff part of the query.

## The handoff is glue code today

Anyone who has run a GPU graph algorithm over warehouse data knows the shape
of the work. Export the selected rows to Parquet. Load them in a notebook.
Clean the IDs, drop the nulls, dedupe the edges, cast the types. Call
cuGraph. Write the result somewhere. Import it into the system where the rest
of the analysis lives and join it back. Every step is imperative code owned by
the application, and every step is specific to one dataset. The next dataset,
with slightly different noise, needs a slightly different pipeline, and a
pipeline that differs slightly has slightly different business semantics.
Nobody notices until two reports disagree.

GPU analytics never spread across heterogeneous enterprise data for the same
reason. The algorithms were fine. The relational work before and after them
was hand-written, which meant it could neither be iterated quickly nor reused.

<GlueVsPlanFigure
  caption={<>Left: the detour most teams build today, where each step is
  application code and most steps copy the data. Right: the same work as one
  SQL statement, with the algorithm as a table function between the SQL that
  prepares its input and the SQL that consumes its output.</>}
/>

Algeon's answer is a declarative execution contract. Selection and cleaning
happen in SQL before the algorithm, enrichment happens in SQL after it, and
the algorithm sits between them as a table function with a documented input
relation and a documented output schema. The physical decisions (what to
scan, what to push down, whether a stage can stay on the device) belong to the
planner. Over the [citation network](/cugraph/demo-dataset) our documentation
uses, the shortest citation chain from BERT back to the 1997 LSTM paper is one
statement:

```sql
SELECT b.path_index, b.distance, p.year, p.title
FROM cugraph_bfs(
  edges => (SELECT src, dst FROM citation_edges),
  sources => (SELECT paper_id AS vertex FROM papers
              WHERE title = 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
                AND year = 2018),
  output_mode => 'path',
  target_vertices => (SELECT paper_id AS vertex FROM papers
                      WHERE title = 'Long short-term memory' AND year = 1997)) b
JOIN papers p ON p.paper_id = b.vertex
ORDER BY b.path_index;
```

Because the edge relation is a subquery, restricting the graph to one decade
or one field of study is a `WHERE` clause. Because the output is rows,
attaching titles is a `JOIN`. The 45 million edges are read into GPU memory
when the statement runs and released when it finishes, and nothing is
exported.

Stages chain, too. One algorithm's output is often the candidate set, the
structure, or the feature the next stage needs, and often something plain SQL
could not have produced at all. The query below keeps only the densely
connected core of the citation graph and ranks it in the same plan. The core's
edges pass from [k-core](/cugraph/functions/cugraph_k_core) to
[PageRank](/cugraph/functions/cugraph_pagerank) without leaving the device:

```sql
SELECT p.year, p.title
FROM cugraph_pagerank(edges => (
  SELECT src, dst
  FROM cugraph_k_core(
    edges => (SELECT src, dst FROM citation_edges), k => 30)
)) r
JOIN papers p ON p.paper_id = r.vertex
ORDER BY r.value DESC
LIMIT 5;
```

## The next big query author is an agent

For most of the history of data infrastructure, queries came from two places:
people, and application code written by people. A third author is arriving
fast. Agents now explore lakehouses to answer questions nobody scheduled, and
they behave unlike either predecessor. They issue far more queries. They are
frequently wrong on the first attempt. They work by hypothesis: run
something, look at the result, revise, run again.

An agent in that loop should not spend its context on which partitions to
read, which columns to project, whether a filter can move below a join, which
side of a hash join to build, whether to aggregate before or after the join,
which operators have a GPU kernel, or whether 500,000 rows justify building an
index. Each of those is a planner's job. An agent that also plays optimizer,
scheduler, storage engine, GPU runtime, and ML engineer has less attention
left for noticing what the data says, the one job only it can do. Its
hypotheses get fewer and worse, and its context fills with mechanics that have
nothing to do with the question.

We think the right division of labor is a short stack.

<StackDiagram
  ariaLabel="Agent, SQL, planner, and physical operators"
  nodes={[
    {
      kind: 'asks', name: 'Agent or analyst',
      note: 'states the question and the shape of the answer', tone: 'plain',
    },
    {
      kind: 'declarative contract', name: 'SQL',
      note: ['tables · views · subqueries', 'cugraph_* · cuvs_* · cuml_*'], tone: 'plan',
      boundary: 'application responsibility ends here',
    },
    {
      kind: 'query engine', name: 'Planner and optimizer',
      note: ['partitions · columns · pushdown · join order', 'GPU selection · admission · memory grant'],
      tone: 'engine',
    },
    {
      kind: 'stable physical operators', name: 'cuDF · cuGraph · cuVS · cuML',
      note: 'one device · one grant · typed errors on the way back up', tone: 'gpu',
    },
  ]}
  caption={<>The agent states what it wants. Everything below the dashed line
  is the engine's job, and the engine reports back in structured terms the
  agent can act on.</>}
/>

The layering decided what the engine has to say back to its caller, and it
shaped Algeon more than any performance goal did.

Error messages are part of the product. An agent that receives a stack trace
learns nothing; an agent that receives a stable code, the column that failed,
its Arrow type, and a remedy can fix the query on the next attempt. Every
Algeon failure carries a machine-readable code, a kind, and a status
(permanent, temporary, or persistent) beside its message, and over Flight SQL
those fields travel as gRPC metadata. A graph algorithm that fails to converge
reports the iteration count and the configured limits and is marked as a
parameter problem. The contract is written up in
[Handling Errors](/integration/handling-errors).

The engine also has to answer questions before it spends GPU time. Three
metadata functions describe every installed algorithm: `gpu_list_functions`
says which ones this build can run, `gpu_describe_function` returns a
function's signature, option schema, and output schema as JSON, and
`gpu_validate_call` checks a concrete call against the named tables without
reading a row or launching a kernel. A fourth, `algeon_explain_coverage`,
plans a whole query the way execution would and reports which parts will run
on the GPU and why the rest will not:

```sql
SELECT row_kind, gpu_path, reason_code, remedy_kind, remedy
FROM algeon_explain_coverage(
  'SELECT year, COUNT(*) FROM papers GROUP BY year');
```

With those, an agent's second attempt is informed by its first. The
[discovery guide](/guides/gpu-function-discovery) and
[coverage validation](/integration/gpu-coverage-validation) pages walk through
the loop.

The payoff is signal detection at a volume no team could staff. Security
teams have seen this already: the misconfigurations a red team needed months
to find, an agent squad now turns up in hours. Data has the same property.
Most of the interesting signals in an enterprise's data were never found
because looking was expensive, whether that meant a graph database
procurement, an ETL pipeline, or a quarter of glue code for one question. When
each look costs one SQL statement, agents can afford millions of them, and the
unknown unknowns become findable. We sometimes describe the goal as a crystal
ball for agents, something they can ask about any slice of the business and
get an answer in rows, cheaply enough to ask again.

## The current build

Algeon is a DataFusion physical optimizer rule plus a native execution
engine. DataFusion keeps SQL parsing, logical planning, catalogs, and CPU
execution. Once a physical plan is complete, Algeon evaluates whole relational
candidates (scans, filters, projections, joins, aggregations, sorts, and
window functions over the common Arrow types) and lowers a supported candidate
to its own cuDF plan. A candidate it does not select keeps the executable
DataFusion baseline, and the reason is recorded.

The algorithms are explicit GPU islands in that plan:

- 27 [`cugraph_*`](/cugraph) functions: traversal (BFS, SSSP), centrality
  (PageRank, personalized PageRank, betweenness, Katz, HITS, eigenvector),
  community detection (Louvain, Leiden, ECG, spectral), connected components,
  a minimum spanning tree, similarity coefficients, degree and core structure,
  and a ForceAtlas2 layout.
- 8 [`cuvs_*`](/cuvs) functions: exact kNN, three approximate indexes
  (IVF-Flat, IVF-PQ, CAGRA), pairwise distances, per-row top-k, KMeans, and
  PCA.
- 7 [`cuml_*`](/cuml) functions: six fit a query-local model (TSVD, HDBSCAN,
  UMAP, logistic regression, random forest classification and regression), and
  [`cuml_forest_predict`](/cuml/functions/cuml_forest_predict) scores rows with
  a pretrained XGBoost, LightGBM, or Treelite forest read from a
  server-configured model directory.

The last of those functions reflects a belief about where enterprise ML
lives. The model is usually not trained by the query engine; it was trained in
PyTorch or XGBoost weeks ago by someone else. What the engine has to do
cheaply is get the features produced by joins, filters, and aggregations into
inference without a detour through files. Query-local fitting is for
exploration. Scoring a model somebody else trained is the production case.

Three execution modes decide how much of the relational work joins the GPU:
`native_preferred` runs supported SQL on cuDF as well, `functions_only` keeps
ordinary SQL on the CPU and reserves the GPU for the algorithm calls, and
`native_required` rejects any plan that would still contain CPU execution.

GPU memory is governed in one place. Each query is placed on one device and
receives an immutable memory grant from that device's ledger; the query's
allocator enforces the cap, and an allocation beyond it fails with a typed
error at its source. No estimate can veto a query before it runs. Between
composed GPU stages, data stays resident on the device.

Algeon embeds as a Rust crate inside a service that already owns its
DataFusion session, or runs as an Arrow Flight SQL server for ADBC clients,
notebooks, and agents. Sources are Parquet files on local disk or on
S3-compatible object storage, read-only. Results return as Arrow batches. The
[integration guide](/integration) covers both paths.

## Known gaps

Algeon is an experimental query engine. Both its stability and its
verifiability have a long way to go, and we would rather list the gaps than
have you find them.

- Correctness evidence is partial. TPC-H and TPC-DS at scale factor 100 run to
  completion on one GPU, and the [benchmark page](/reference/benchmarks)
  publishes CPU and GPU timings from one harness; for TPC-DS it also reports
  how many queries' row counts matched the CPU run. Every cuGraph example in
  the documentation was executed against the 4.9-million-paper citation
  network on 2026-09-26, and a few similarity scores were checked against
  independent calculations. There is no end-to-end graph benchmark and no
  value-level verification of every algorithm's output. One known defect is
  inherited from the pinned cuGraph release:
  [`cugraph_cosine`](/cugraph/functions/cugraph_cosine) normalizes
  incorrectly, and its example page records the bug and withholds the ranking.
- Performance is uneven. Exact edge betweenness over a 164,000-edge subgraph
  takes over a minute in a release build; we have profiled it and not yet
  fixed it.
- The relational subset is a subset. Nested and binary types and most list
  shapes stay on the CPU path, and a query that needs them runs there.
- A query runs on one GPU and never spills. Independent queries can be placed
  on different devices, but a query never spans two, and a working set larger
  than its grant fails with an allocation error.
- Every algorithm call is query-local. Indexes, fitted models, and graphs are
  built inside one statement and released after it; a bounded server can
  retain constructed graphs for a handful of algorithms. Persistent artifacts
  are still a direction.
- Sources are Parquet only. Iceberg is held out of this release until the
  DataFusion integration ships a release we can pin. cuOpt is not exposed.
  There is no Substrait input and no write path.

## Algeon next to Spark RAPIDS

The obvious comparison is NVIDIA's own Spark work.
[cudf-spark](https://github.com/NVIDIA/cudf-spark) replaces Spark SQL
operators with cuDF-backed ones, and
[cuml-spark](https://github.com/NVIDIA/cuml-spark) offers PySpark
ML-compatible estimators backed by cuML. Both are mature, and both target a
different moment in the data's life.

Spark is a cluster framework for jobs. Its unit of work is a pipeline over
the whole lake, scheduled, with a driver and executors, and its acceleration
is measured in job hours saved. Algeon targets the moment after that pipeline
has run: one working set, one device, one process, many small questions, most
of them ad hoc and many of them wrong on the first try. Latency per attempt
matters more than throughput per job.

The unit of composition differs too. In Spark, an ML estimator is an object
in Python or Scala, and moving between SQL and cuML is application code
again, the glue we set out to remove. Graph algorithms are absent from Spark
SQL. In Algeon the algorithm is a relation in the physical plan, and an agent
never leaves SQL to use it.

The two fit together. Spark RAPIDS makes the pipelines that build and refresh
the lake faster; Algeon makes the questions asked of the lake cheaper. Both
read Parquet, and both stand on the same RAPIDS libraries.

## Next

The nearest work is on artifacts. Today every index, model, and graph is
ephemeral. A retained index, a fitted model bound to a SQL name, and a
GPU-resident result that the next statement can read without a second upload
are all on the list, and each has to fit the ledger and grant model everything
else uses. cuOpt is the next algorithm family. DuckDB is the adapter direction
we are considering after DataFusion, and Spark a longer-term one; neither is
supported today.

If you run Algeon over your own data, the experiment we would most like to
hear about is simple. Point an agent at a working set you already have, let it
wander, and tell us which question it answered that you had written off as
too expensive to ask. The [build guide](/build) is the place to start.

[^1]: Redshift figures are from van Renen et al.,
[*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB 17(11),
2024, Table 6. The BigQuery figure is from Jordan Tigani,
[*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
MotherDuck, 2023.
