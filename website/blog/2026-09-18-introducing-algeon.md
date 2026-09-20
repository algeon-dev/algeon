---
title: "Introducing Algeon: One Query, Any Data, Any Algorithm"
description: "One Query, Any Data, Any Algorithm. Algeon is an experimental GPU query engine in which any relation SQL can produce feeds a graph, vector, or ML algorithm, and the output comes back as rows. This post explains why we are building it, who it is for, and how far along it is."
slug: introducing-algeon
tags: [announcement]
authors: morris
---

import {FlowDiagram} from '@site/src/components/design/Diagram';
import {AgentLoopFigure, GlueVsPlanFigure} from '@site/src/components/blog/IntroFigures';

Two product lines keep arriving late. Their suppliers look unrelated. Could
both depend on the same upstream company?

The evidence might already be in purchasing records and supplier
relationships. Investigating it would take SQL to select the relevant orders,
a graph traversal to trace dependencies, and more SQL to connect those
dependencies to the affected products. A different time window or a different
definition of a relationship could change the answer. The question is
hypothetical; the data and the definitions behind it belong to the
application.

The same shape appears in a security investigation, where the question is
which hosts a stolen credential reached and what those hosts touched next,
and in research, where it is which papers connect two ideas. Algeon is built
for that shape, whichever domain the records come from.

**Algeon is an experimental GPU query engine for this kind of exploration.**
An analyst or an AI agent applies a graph, vector, or machine-learning
algorithm to a dataset defined by SQL, and the algorithm's output comes back
as rows the same query can keep joining, filtering, and aggregating. Our aim
is to make the follow-up questions cheap enough to pursue while the
investigation is still unfolding. Algeon is experimental, and this post is as
much about what it cannot do yet as about what it can.

{/* truncate */}

## A working set for each question

A lakehouse holds terabytes, and almost no single
question needs more than a sliver of it. A question is about one tenant, one
quarter, one product line, one cohort. SQL does the narrowing: a filter on a
date range, a join against a dimension table, an aggregation over a segment,
and the domain rules that determine which records belong in the analysis
at all. Fleet studies put numbers on how small the survivor usually is. In
the Amazon Redshift fleet the median query uses 0.6 GB of memory and the
99th-percentile query 21.4 GB; among paying BigQuery customers, 90% of
queries process less than 100 MB.[^1]

That fine-grained slice is what Algeon runs GPU-accelerated SQL on. A
working set of that size fits in the memory of one GPU, although graph
construction and algorithm scratch space add to it. The questions worth
asking about the working set, though, are often the ones relational algebra
handles badly. Which of these companies form a tightly knit cluster? What is
the shortest chain from this supplier to that sanctioned entity? Which
products sit nearest to this one in embedding space? Which of these rows
would the model we trained last quarter flag? Each has a well-understood
algorithm and a mature GPU implementation, and none of them is expressible as
a join.

<FlowDiagram
  minWidth={720}
  ariaLabel="A lakehouse of terabytes is reduced by SQL to a working set of gigabytes, a GPU algorithm runs over it inside the same statement, and the output returns as rows or feeds the next stage."
  nodes={[
    {id: 'lake', kind: 'terabytes', name: 'Lakehouse', note: 'Parquet · tables · views', tone: 'lake'},
    {
      id: 'working-set', kind: 'gigabytes', name: 'Working set', tone: 'engine',
      edgeLabel: ['filter · join', 'aggregate'], note: 'fine-grained · fits one GPU',
    },
    {
      id: 'algorithm', kind: 'gpu algorithm', name: 'Algorithm output', tone: 'gpu',
      edgeLabel: ['cugraph_* · cuvs_*', 'cuml_*'], note: ['paths · communities', 'neighbors · scores'],
    },
    {id: 'answer', kind: 'rows', name: 'Answer', note: 'back to the caller', tone: 'plain',
      edgeLabel: ['join · filter', 'aggregate']},
  ]}
  groups={[{label: 'one SQL statement · one physical plan', nodes: ['working-set', 'algorithm', 'answer']}]}
  returns={[{from: 'algorithm', to: 'working-set', label: 'next stage input'}]}
  caption={<>SQL reduces the lakehouse to a working set that fits one GPU.
  Algeon places the specialized algorithm after that reduction, inside the
  same statement, and returns its output as rows that ordinary SQL can keep
  working on, or that the next algorithm can take as input.</>}
/>

Algeon accepts SQL subqueries as the inputs of graph, vector, and
machine-learning UDTFs (user-defined table functions), and their outputs are
relations too.
Supported relational operations and GPU algorithms compose in one physical
plan, and when one GPU UDTF's output feeds another directly, the data
stays on the device. Whether an analysis needs distributed execution depends
on the reduced working set and the algorithm's state, and Algeon runs each
query on one device inside a configured memory cap.

## The handoff is glue code today

Anyone who has run a GPU graph algorithm over warehouse data knows the shape
of the work. Export the selected rows and columns to a bucket. Load them in a
notebook. Clean the IDs, drop the nulls, dedupe the edges, cast the types.
Call cuGraph. Write the result somewhere. Import it into the system where the
rest of the analysis lives and join it back. Each step is an ad-hoc ETL
script, written for the small fraction of the dataset that one question needs.
The next dataset, with slightly different noise, needs a slightly different
pipeline, and two pipelines that differ slightly can encode different
semantics without anyone comparing them.

<GlueVsPlanFigure
  caption={<>Left: the detour most teams build today, where each step is an
  ad-hoc ETL script and most steps copy the data. Right: the same work as one
  SQL statement, with the algorithm as a UDTF between the SQL that
  prepares its input and the SQL that consumes its output.</>}
/>

Algeon's answer is a declarative execution contract. Selection and cleaning
happen in SQL before the algorithm, enrichment happens in SQL after it, and
the algorithm sits between them as a UDTF with a documented input
relation and a documented output schema. The physical decisions (what to
scan, what to push down, whether a stage can stay on the device) belong to
the planner; the choice of algorithm and its options stays with the query
author. The SQL keeps the working set's definition available for inspection
and revision as the question changes.

Our documentation's runnable examples use a public
[citation network](/guides/datasets/citation-network) of 4.9 million papers and 45.6
million directed citation edges, registered from Parquet files. Ranking the
foundational reading list for BERT beyond its own bibliography takes a seed
view and one query:

```sql
CREATE OR REPLACE VIEW bert_seed AS
SELECT paper_id AS vertex, CAST(1.0 AS DOUBLE) AS value
FROM papers
WHERE title = 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
  AND year = 2018;

SELECT ROUND(r.value, 6) AS ppr, p.year, p.title
FROM cugraph_personalized_pagerank(
  edges => (SELECT src, dst FROM citation_edges),
  personalization => (SELECT vertex, value FROM bert_seed)) r
JOIN papers p ON p.paper_id = r.vertex
WHERE NOT EXISTS (SELECT 1 FROM bert_seed s WHERE s.vertex = r.vertex)
  AND NOT EXISTS (SELECT 1 FROM citation_edges e JOIN bert_seed s ON s.vertex = e.src
                  WHERE e.dst = r.vertex)
ORDER BY r.value DESC
LIMIT 8;
```

Because the edge and personalization relations are subqueries, seed selection
and traversal parameters stay within the SQL session. Because the output is
tabular rows, attaching titles and pruning direct citations use standard
`JOIN` and `NOT EXISTS` clauses. The 45 million edges are read into GPU memory
when the statement runs and released when it finishes, unless a bounded server
retains the constructed graph for reuse, and nothing is exported. The
[captured example](/cugraph/udtfs/cugraph_personalized_pagerank#examples)
surfaces foundational work across continuous speech recognition and statistical
machine translation; other valid rankings exist, and the corpus stops in early
2020.

## Follow the result somewhere new

Stages chain. One algorithm's output is often the candidate set, the
structure, or the feature the next stage needs, and often something plain
SQL could not have produced at all. The
[overview's composed query](/#from-a-question-to-a-working-set) keeps only
the densely connected core of the citation graph with
[k-core](/cugraph/udtfs/cugraph_k_core) and ranks it with
[PageRank](/cugraph/udtfs/cugraph_pagerank) in the same plan. The core's
symmetrized edges pass between the two UDTFs without leaving the device.

In the supplier investigation, the same shape could carry the analysis
somewhere the original shortlist did not point. A traversal might identify an
upstream company that nobody had listed. A join to orders could quantify the
products exposed to it. Vector search over existing product features could
produce candidate substitutes, with a SQL filter keeping the ones currently
available. Each stage changes what the next stage can examine. A graph
produces a set of companies, and those companies define a new query.
Clustering can produce groups absent from the existing categories. A model score can narrow the candidates for a more expensive
analysis. Similarity alone would not establish that a product is an
acceptable substitute; the application supplies those criteria and chooses
the next question.

Within a supported query, there is no application script to export the
intermediate rows, construct a separate algorithm job, and load its output
back for a join. Data preparation still matters, and the SQL that does it
stays visible.

## Give agents room to investigate

Queries have come from people and from application code written by people.
Agents are a newer kind of author, and they behave unlike either. They issue
far more queries. They are frequently wrong on the first attempt. They work
by hypothesis: run something, look at the result, revise, run again.

An agent in that loop should not spend its context on which partitions to
read, which columns to project, whether a filter can move below a join, which
side of a hash join to build, or which operators have a GPU kernel. Each of
those is the planner's job. An agent that also plays optimizer, scheduler,
storage engine, and GPU runtime has less attention left for noticing what
the data shows, the one job only it can do. When each variation of a
question requires a custom program for data movement and GPU execution, the
application also has to maintain that program.

The application supplies the domain definitions, the choice of
algorithms and their options, and the hypotheses to test, and it handles the
rows or failures that come back. In practice the agent speaks the domain's
vocabulary, and a semantic layer, the application's ontology, maps that
vocabulary to the tables, keys, and joins Algeon sees. DataFusion and Algeon
own physical planning and execution, including join ordering and GPU
allocation.

<AgentLoopFigure
  caption={<>The application owns the agent, the ontology that maps its
  vocabulary to tables and keys, and the service that submits SQL. Algeon
  plans, admits, and executes each statement on one GPU over the lakehouse,
  and returns rows plus structured diagnostics the agent can act on. Each
  pass around the loop costs one statement.</>}
/>

A first query can be wrong, and error messages are part of the product. An
agent that receives a stack trace learns nothing; an agent that receives a
stable code, the kind of failure, and the context that triggered it can fix
the query on the next attempt. Every failure Algeon owns carries a
machine-readable code, a kind, and a status (permanent, temporary, or
persistent) beside its message, and over Flight SQL those fields travel as
gRPC metadata; ordinary DataFusion and transport errors keep their own
contracts. A graph algorithm that fails to converge reports the iteration
count and the configured limits and is marked as a parameter problem. The
status describes the condition; whether and when to retry an attempted query
comes from its sealed attempt report. The contract is written up in
[Handling Errors](/integration/handling-errors).

Before a query spends GPU time, the UDTF catalog describes what a call
needs. `gpu_list_functions` says which UDTFs this build can run,
`gpu_describe_function` returns a UDTF's signature, option schema, and
output schema as JSON, and `gpu_validate_call` checks a concrete call against
registered tables or views without reading a row or launching a kernel. For a
whole statement, `algeon_explain_coverage` plans the query the way execution
would and reports which parts will run on the GPU and why the rest will not.
With those, an agent's second attempt is informed by its first. The
[discovery guide](/guides/gpu-udtf-discovery) and
[coverage validation](/integration/gpu-coverage-validation) pages walk
through the loop. These interfaces do not establish that a hypothesis is
sound or that a correlation has meaning in the domain; that judgment stays with
the application and the people behind it.

With less work per attempt, exploration can become more specific. A team
could investigate the supplier network for one product line this week, then
repeat the analysis with a different definition of exposure. Most of the
interesting signals in an organization's data were never found because
looking was expensive, whether that meant a graph database procurement, an
ETL pipeline, or a quarter of glue code for one question. When a look costs
one SQL statement, an agent can afford many of them. We want Algeon to make
it practical to follow an unexpected result beyond the questions already
encoded in a dashboard.

## The current build

Algeon is a DataFusion physical optimizer rule plus a native execution
engine. DataFusion keeps SQL parsing, logical planning, catalogs, and CPU
execution. Once a physical plan is complete, Algeon evaluates whole
relational candidates (scans, filters, projections, joins, aggregations,
sorts, and window functions over the common Arrow types) and lowers a
supported candidate to its own cuDF plan. A candidate it does not select
keeps the executable DataFusion baseline, and the reason is recorded.

The algorithms are explicit GPU islands in that plan. The
[`cugraph_*`](/cugraph) UDTFs cover traversal, centrality, community
detection, components, similarity, core structure, and layout; the
[`cuvs_*`](/cuvs) UDTFs cover exact and approximate nearest neighbors,
pairwise distances, KMeans, and PCA; and the [`cuml_*`](/cuml) UDTFs fit
query-local models (TSVD, HDBSCAN, UMAP, logistic regression, random forests)
while [`cuml_forest_predict`](/cuml/udtfs/cuml_forest_predict) scores
rows with a pretrained XGBoost, LightGBM, or Treelite forest read from a
server-configured model directory. Each family's UDTF index lists every
installed UDTF, and the families keep growing.

The last of those reflects a belief about where trained models usually
live. The
model is usually trained elsewhere, in PyTorch or XGBoost, weeks earlier and
by someone else. What the query engine has to do cheaply is carry the
features produced by joins, filters, and aggregations into inference without
a detour through files. Query-local fitting serves exploration; scoring a
model somebody else trained is the case we expect in production. General
inference for neural networks trained in frameworks such as PyTorch remains
future work.

Three execution modes, configured on the backend, set how much of the
relational work joins the GPU: `native_preferred`, the default, runs
supported SQL on cuDF as well; `functions_only` keeps ordinary SQL on the CPU
and reserves the GPU for the algorithm calls; and `native_required` rejects
any plan that would still contain DataFusion CPU operators.

GPU memory is governed in one place. Each query is placed on one device and
receives an immutable memory grant from that device's ledger; the query's
allocator enforces the cap, and an allocation beyond it fails with a typed
error at its source. No estimate can veto a query before it runs. Between
directly composed GPU stages, data stays resident on the device.

Algeon embeds as a Rust crate inside a service that already owns its
DataFusion session, or runs as an Arrow Flight SQL server for ADBC clients,
notebooks, and agents. GPU scans read Parquet files on local disk or on
S3-compatible object storage, read-only; other source types stay on the
DataFusion path, and Iceberg integration is currently unavailable. Results return as Arrow batches. The
[integration guide](/integration) covers both paths.

## Known gaps

Both Algeon's stability and its verifiability have a long way to go, and we
would rather list the gaps than have you find them.

- Correctness evidence is partial. TPC-H and TPC-DS at scale factor 100 run
  to completion on one GPU. In the TPC-DS run recorded on 2026-07-20, all 99
  queries completed on both paths with matching row counts, in 101.7 seconds
  on the GPU against 379.4 seconds on the CPU, with 13 queries slower on the
  GPU; the [benchmark page](/reference/benchmarks) holds the per-query rows.
  Row-count agreement does not verify output values. Every cuGraph example in
  the documentation was executed against the citation network on 2026-09-26,
  and a few similarity scores were checked against independent calculations.
  There is no end-to-end graph benchmark and no value-level verification of
  every algorithm's output. One known defect is inherited from the pinned
  cuGraph release: [`cugraph_cosine`](/cugraph/udtfs/cugraph_cosine)
  normalizes incorrectly, and its example page records the bug and withholds
  the ranking.
- Performance is uneven. Exact edge betweenness over a 164,000-edge subgraph
  takes over a minute in a release build; we have profiled it and not yet
  found the cause.
- GPU relational coverage is a subset of SQL and of Arrow types. Nested and
  binary types and most list shapes stay on the CPU path, and a query that
  needs them runs there unless the execution mode forbids CPU operators.
- A query runs on one GPU and never spills. Independent queries can be placed
  on different devices, but a query never spans two, and a working set larger
  than its grant fails with an allocation error. Graph construction and
  algorithm scratch space count toward that grant, and a small SQL input
  alone does not guarantee that an algorithm fits.
- Vector indexes and fitted models live inside one statement, and forest
  inference reads its model file for each statement. Graphs are also built
  inside a statement, except that a bounded server can retain constructed
  graphs for a handful of algorithms. Persistent artifacts are still a
  direction.
- GPU file scans read Parquet only. Workspace views and `CREATE TABLE AS`
  tables exist, but nothing writes back to a source catalog. Iceberg is held
  out of this release until the DataFusion integration ships a release we
  can pin. cuOpt is not exposed, and there is no Substrait input.

## Alongside the existing data stack

The obvious comparison is NVIDIA's own Spark work.
[cuDF for Spark](https://github.com/NVIDIA/cudf-spark) replaces Spark SQL
operators with cuDF-backed ones, and
[cuML for Spark](https://github.com/NVIDIA/cuml-spark) offers PySpark
ML-compatible estimators backed by cuML. Teams with Spark workloads can use
both within their existing environment, and both target a different moment
in the data's life.

Spark is a cluster framework for jobs. Its unit of work is a pipeline over
the whole lake, scheduled, with a driver and executors, and its acceleration
is measured in job hours saved. Algeon targets the moment after that pipeline
has run: one working set, one device, one process, many small questions,
most of them ad hoc and many of them wrong on the first try. Latency per
attempt matters more there than throughput per job.

The unit of composition differs too. In Spark, an ML estimator is an object
in Python or Scala, and moving between SQL and cuML is application code
again. Spark SQL has no graph algorithms of its own; GraphX and GraphFrames
are separate APIs. In Algeon the algorithm is a relation in the physical
plan, and an agent never leaves SQL to use it.

The two can fit together. Spark could prepare shared datasets and train
models, with Algeon serving the exploratory queries asked of the result.
Both read Parquet and both stand on the same RAPIDS libraries. Algeon has no
direct Spark adapter today, and the division of work above is a possibility
we have not built.

## Next

The nearest work is on artifacts. Today every index and fitted model is
ephemeral, and graphs outlive a statement only in the bounded-server cache.
A retained index, a fitted model bound to a SQL name, and a SQL name bound
to a GPU-resident result that the next statement can read without a second
upload are all on the list, and each has to fit the ledger and grant model
everything else uses. cuOpt is the next algorithm family. DuckDB is the
adapter direction we are considering after DataFusion, and Spark a
longer-term one; neither is supported today.

The next useful experiment is a complete investigation on a known dataset.
Define a working set, apply an algorithm, and use its output in the next
query. Compare the results with a trusted reference and measure the full
query cost, including data preparation. If you run that experiment over your
own data, tell us which question it answered that you had written off as too
expensive to ask. The [overview](/) and [build guide](/build) are the places
to start.

[^1]: Redshift figures are from van Renen et al.,
[*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB 17(11),
2024, Table 6. The BigQuery figure is from Jordan Tigani,
[*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
MotherDuck, 2023.
