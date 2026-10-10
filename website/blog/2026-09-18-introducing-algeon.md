---
title: "Introducing Algeon: GPU OLAP for the Agent's Loop"
description: "Algeon is a GPU query engine that runs graph, vector and machine-learning algorithms on SQL subqueries and returns their output as rows, inside one statement and on one GPU. This post explains why that matters to an agent that asks many questions, how the engine is built, and how far along it is."
slug: introducing-algeon
tags: [announcement]
authors: morris
---

import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Table from '@site/src/theme/MDXComponents/Table';
import {FlowDiagram} from '@site/src/components/design/Diagram';
import {AgentLoopFigure, GlueVsPlanFigure} from '@site/src/components/blog/IntroFigures';
import session from '@site/static/data/merrec-agent-session.json';
import barbie from '@site/static/data/merrec-scale-barbie-tie-ins.json';
import funnel from '@site/static/data/merrec-scale-audience-funnel.json';
import {tpcdsSf100Benchmark} from '@site/src/data/tpcdsSf100Benchmark';

export const stepOf = id => session.steps.find(item => item.step === id);
export const errorOf = id => stepOf(id).error.replace(/^.*?message: /s, '');
export const seconds = value => value.toFixed(2);
export const ms = value => (value / 1000).toFixed(1);
export const tpcds = tpcdsSf100Benchmark.summary;
export const byStage = Object.fromEntries(funnel.records.map(row => [row.stage, row]));
export const pct = (row, digits = 2) => `${(row.purchase_rate * 100).toFixed(digits)}%`;

An agent session over 1.27 billion marketplace events asked which brands
moved toward Barbie's audience after the film opened in July 2023. It issued
six statements through Arrow Flight SQL. The first listed the graph functions
this build can run, and the second fetched the contract of personalized
PageRank. The third and fourth guessed column names and failed at planning
time, before any row was read. The fifth asked `EXPLAIN GPU` whether the
whole statement would run on the GPU. The sixth ran four graph calls over
the events and returned {stepOf(6).rows.length} brands in
{seconds(stepOf(6).elapsed_s)} seconds.

{/* truncate */}

<Table dense>
<thead><tr><th>Step</th><th>Statement</th><th>Outcome</th><th>Elapsed</th></tr></thead>
<tbody>{session.steps.map(item => <tr key={item.step}><td>{item.step}</td><td>{item.intent}</td><td>{item.status === 'ok' ? 'rows' : 'planning error'}</td><td>{Math.round(item.elapsed_s * 1000).toLocaleString('en-US')} ms</td></tr>)}</tbody>
</Table>

The shape recurs elsewhere. A supply-chain analyst asks which upstream
company two late product lines share. A security team asks which hosts a
stolen credential reached. A researcher asks which papers connect two ideas.
The evidence is already in the records, the question needs a graph
traversal, a similarity search or a model score in the middle, and the first
version of the question is usually wrong.

**Algeon is a GPU query engine for this kind of exploration.**
An analyst or an AI agent applies a graph, vector or machine-learning
algorithm to a relation defined by SQL, and the output comes back as rows the
same statement can keep joining, filtering and aggregating. Each question
costs one statement and runs on one GPU. This post is as much about what
Algeon cannot do yet as about what it can.

## A question now costs one statement

Most of the interesting signals in an organization's data were never found
because looking was expensive, whether that meant a graph database
procurement, an ETL job or a quarter of glue code for one question. When a
look costs one SQL statement, an agent can afford many of them.

Queries have come from people and from application code written by people.
Agents are a newer kind of author, and they behave unlike either. They issue
far more queries. They are frequently wrong on the first attempt. They work
by hypothesis: run something, look at the result, revise, run again.

Writing the query has become cheap. A model that knows a domain's vocabulary
and sees the schema writes the SQL for the next question in seconds, and a
smaller model specialized to that domain can do it at a cost that no longer
limits how often it asks. Any question that needed a graph traversal, a
nearest-neighbor search or a model score stayed expensive. Those steps ran as
separate jobs, with an export before and an import after, and the answer
arrived after the agent's turn had ended.

Algeon moves those steps into the statement. Three statements, measured over
Flight SQL from submission to the last result batch with planning included:

<Table>
<thead><tr><th>Statement</th><th>Work</th><th>Elapsed</th></tr></thead>
<tbody>
<tr><td><Link to="/guides/datasets/merrec#barbie">Barbie's July</Link></td><td>Two like graphs from 1.27 billion events, four PageRank calls, brand joins</td><td>{seconds(barbie.median_elapsed_s_after_first)} s</td></tr>
<tr><td><Link to="/guides/datasets/merrec#audience-funnel">A campaign list</Link></td><td>Personalized PageRank, exact nearest neighbors among the result, a trained forest, evaluation against later purchases</td><td>{seconds(funnel.median_elapsed_s_after_first)} s</td></tr>
<tr><td><Link to="/reference/benchmarks">TPC-DS SF100</Link></td><td>All {tpcds.queryCount} queries on the GPU path; the CPU path took {ms(tpcds.cpuTotalMs)} s</td><td>{ms(tpcds.gpuSuccessTotalMs)} s</td></tr>
</tbody>
</Table>

The MerRec figures are medians of repeated runs after the first, and the
fine print on each page names the server configuration, the binary and the
query IDs behind them.

A few seconds per attempt changes what an agent can afford. Spark's unit of
work is a job over the whole lake, scheduled with a driver and executors, and
its GPU acceleration is measured in job hours saved. Algeon targets the
moment after that job has run: one working set, one device, one process,
many small questions, most of them ad hoc and many of them wrong on the first
try. Latency per attempt matters more there than throughput per job, and an
answer that arrives inside the agent's turn can be acted on in the same turn.

## A working set for each question

A lakehouse holds terabytes, and almost no single question needs more than a
sliver of it. A question is about one tenant, one quarter, one product line,
one cohort. SQL does the narrowing: a filter on a date range, a join against
a dimension table, an aggregation over a segment, and the domain rules that
determine which records belong in the analysis at all. Fleet studies put
numbers on how small the survivor usually is. In the Amazon Redshift fleet
the median query uses 0.6 GB of memory and the 99th-percentile query 21.4 GB;
among paying BigQuery customers, 90% of queries process less than 100 MB.[^1]

That slice is what Algeon runs GPU-accelerated SQL on. A working set of that
size fits in the memory of one GPU, although graph construction and algorithm
scratch space add to it. The questions worth asking about the working set,
though, are often the ones relational algebra handles badly. Which of these
companies form a tightly knit cluster? What is the shortest chain from this
supplier to that sanctioned entity? Which products sit nearest to this one in
embedding space? Which of these rows would the model we trained last quarter
flag? Each has a well-understood algorithm and a mature GPU implementation,
and none of them is expressible as a join.

<FlowDiagram
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

Running the algorithm after the reduction also changes which algorithm fits.
NVIDIA's cuVS guide states the condition for exact search:
["Use brute-force for heavily filtered queries. If a filter excludes most
vectors, brute-force can search only the remaining candidates while IVF and
graph indexes may skip useful vectors before the filter is
applied."](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/brute-force#when-to-use-brute-force)
In SQL that filter is the `WHERE` clause and the joins that define the
working set. The [travel-headphone shortlist](/guides/datasets/amazon-reviews#travel-headphones)
restricts a million products by price, review count and low-rating share
before an exact nearest-neighbor search over the survivors, and the campaign
statement above searches taste vectors only among eligible shoppers. The
same holds for graphs. The Barbie statement builds a separate like graph for
each of its two periods from the filtered events. A result produced this way
stays tied to the conditions that admitted its input, and those conditions
are visible in the statement.

## The handoff is glue code today

Anyone who has run a GPU graph algorithm over warehouse data knows the shape
of the work. Export the selected rows and columns to a bucket. Load them in a
notebook. Clean the IDs, drop the nulls, dedupe the edges, cast the types.
Call cuGraph. Write the result somewhere. Import it into the system where the
rest of the analysis lives and join it back. Each step is an ad-hoc ETL
script, written for the small fraction of the dataset that one question
needs, and two scripts that differ slightly can encode different semantics
without anyone comparing them.

<GlueVsPlanFigure
  caption={<>Left: the detour most teams build today, where each step is an
  ad-hoc ETL script and most steps copy the data. Right: the same work as one
  SQL statement, with the algorithm as a UDTF between the SQL that
  prepares its input and the SQL that consumes its output.</>}
/>

Algeon's answer is a declarative execution contract. Selection and cleaning
happen in SQL before the algorithm, enrichment happens in SQL after it, and
the algorithm sits between them as a UDTF (user-defined table function) with
a documented input relation and a documented output schema. The physical
decisions (what to scan, what to push down, whether a stage can stay on the
device) belong to the planner; the choice of algorithm and its options stays
with the query author.

The shortest of the documented examples runs on a public
[citation network](/guides/datasets/citation-network) of 4.9 million papers
and 45.6 million citation edges. Ranking the foundational reading list for
BERT beyond its own bibliography takes a seed view and one query:

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

Because the edge and personalization relations are subqueries, the seed and
the traversal parameters stay in the SQL session; because the output is rows,
attaching titles and pruning direct citations are a `JOIN` and two
`NOT EXISTS` clauses. The 45 million edges are read into GPU memory when the
statement runs and released when it finishes, and nothing is exported. The
[captured result](/cugraph/udtfs/cugraph_personalized_pagerank?dataset=citation-network#examples)
ranks work on continuous speech recognition, statistical machine translation
and the Penn Treebank above BERT's direct references; other valid rankings
exist, and the corpus stops in early 2020.

## Follow the result somewhere new

Stages chain. One algorithm's output is often the candidate set, the
structure or the feature the next stage needs, and often something plain
SQL could not have produced at all. The
[campaign statement](/guides/datasets/merrec#audience-funnel) measured above
does this three times. Personalized PageRank from the Pokemon brand vertex
scores every shopper on the like graph; an exact nearest-neighbor search
finds the taste lookalikes of earlier buyers among eligible shoppers; a
forest trained offline ranks the union of the two lists, and a `GROUP BY` at
the end counts how many shoppers in each list bought a Pokemon listing in the
following two months. {pct(byStage[8])} of the final 5,000 did, against
{pct(byStage[1], 3)} of all eligible shoppers. The intermediate relations
stay on the device, and the list size, `k` and the PageRank cutoff are
parameters for the next run.

Revision is the other half of the loop. The Barbie result named four brands
whose graph affinity rose after the film. The agent then
[checked listing names](/guides/datasets/merrec#barbie) for the word "Barbie"
in each period with a separate statement, and re-ran the ranking with Hot
Wheels and then Nike as the seed to separate a Barbie-specific movement from
a general summer rise. Each check was one more statement with one changed
condition. Whether a rise in affinity means a partnership remains the
application's judgment; the engine's job is to make the next check cheap
enough to run.

## Built for the agent's loop

An agent in that loop should not spend its context on which partitions to
read, which columns to project, whether a filter can move below a join, or
which operators have a GPU kernel. Each of those is the planner's job. An
agent that also plays optimizer, scheduler, storage engine and GPU runtime
has less attention left for noticing what the data shows, the one job only
it can do.

The application supplies the domain definitions, the choice of algorithms
and their options, and the hypotheses to test. In practice the agent speaks
the domain's vocabulary, and a semantic layer, the application's ontology,
maps that vocabulary to the tables, keys and joins Algeon sees. DataFusion
and Algeon own physical planning and execution, including join ordering and
GPU allocation. SQL is the one interface all three already share, and the
agent's context, the ontology and the schema meet in the statement.

<AgentLoopFigure
  caption={<>The application owns the agent, the ontology that maps its
  vocabulary to tables and keys, and the service that submits SQL. Algeon
  plans, admits, and executes each statement on one GPU over the lakehouse,
  and returns rows plus structured diagnostics the agent can act on. Each
  pass around the loop costs one statement.</>}
/>

A first query can be wrong, and error messages are part of the product. An
agent that receives a stack trace learns nothing; an agent that receives a
stable code, the kind of failure and the context that triggered it can fix
the query on the next attempt. The two failures in the opening session read,
in full:

<CodeBlock language="text">{[errorOf(3), errorOf(4)].join('\n')}</CodeBlock>

Both returned at planning time. The first names the canonical column the
personalization relation lacks, and the second lists the output columns that
exist. Every failure Algeon owns carries a machine-readable code, a kind and
a status (permanent, temporary or persistent) beside its message, and over
Flight SQL those fields travel as gRPC metadata. A graph algorithm that fails
to converge [reports the iteration count and the configured limits](/integration/handling-errors#algorithm-did-not-converge)
and is marked as a parameter problem.
[Handling Errors](/integration/handling-errors) has the contract.

Before a query spends GPU time, the UDTF catalog describes what a call
needs. `gpu_list_functions` lists what this build can run,
`gpu_describe_function` returns a UDTF's signature, option schema and output
schema as JSON, and `gpu_validate_call` checks a call against registered
tables without reading a row. `EXPLAIN GPU` plans a whole statement the way
execution would and reports which parts will run on the GPU and why the rest
will not. The [discovery guide](/guides/gpu-udtf-discovery) walks through
the sequence and replays the opening session. None of these interfaces
establishes that a hypothesis is sound; that judgment stays with the
application and the people behind it.

## The current build

Algeon is a DataFusion physical optimizer rule plus a native execution
engine. DataFusion keeps SQL parsing, logical planning, catalogs and CPU
execution. Once a physical plan is complete, Algeon evaluates whole
relational candidates (scans, filters, projections, joins, aggregations,
sorts and window functions over the common Arrow types) and lowers a
supported candidate to its own cuDF plan. A candidate it does not select
keeps the DataFusion baseline, and the reason is recorded.

The algorithms are explicit GPU islands in that plan. The
[`cugraph_*`](/cugraph) UDTFs cover traversal, centrality, community
detection, components, similarity, core structure and layout; the
[`cuvs_*`](/cuvs) UDTFs cover exact and approximate nearest neighbors,
pairwise distances, KMeans and PCA; and the [`cuml_*`](/cuml) UDTFs fit
query-local models (TSVD, HDBSCAN, UMAP, logistic regression, random forests)
while [`cuml_forest_predict`](/cuml/udtfs/cuml_forest_predict) scores
rows with a pretrained XGBoost, LightGBM or Treelite forest. The last
reflects a belief about where trained models live. The model is usually
trained elsewhere, weeks earlier and by someone else, and the engine's job is
to carry the features produced by joins and aggregations into inference
without a detour through files. Query-local fitting serves exploration;
scoring a model somebody else trained is the case we expect in production.

Three [execution modes](/integration/datafusion-session#choose-an-execution-mode)
set how much relational work joins the GPU, from `functions_only`, which
keeps ordinary SQL on the CPU around the algorithm calls, to
`native_required`, which rejects any plan that would still contain CPU
operators. The MerRec statements above ran in `native_required`.

GPU memory is governed in one place. Each query is placed on one device and
receives an immutable memory grant from that device's ledger; the query's
allocator enforces the cap, and an allocation beyond it fails with a typed
error at its source.

Algeon embeds as a Rust crate inside a service that already owns its
DataFusion session, or runs as an Arrow Flight SQL server for ADBC clients,
notebooks and agents. GPU scans read Parquet files on local disk or on
S3-compatible object storage, read-only, and results return as Arrow
batches. The [integration guide](/integration) covers both paths.

## Known gaps

Both Algeon's stability and its verifiability have a long way to go, and we
would rather list the gaps than have you find them.

- Correctness evidence is partial. In the TPC-DS SF100 run recorded on
  2026-07-20, all 99 queries completed on both paths with matching row
  counts, and row-count agreement does not verify output values; the
  [benchmark page](/reference/benchmarks) holds the per-query rows and the
  TPC-H run. Every example on this site was captured with its SQL, query IDs
  and the binary that produced it, and a few similarity scores were checked
  against independent calculations. That check found a normalization defect
  in the pinned cuGraph release's cosine similarity; our fork carries the
  fix, and upstream does not yet. There is no end-to-end graph benchmark and
  no value-level verification of every algorithm's output.
- Performance is uneven. {tpcds.gpuSlowerCount} of the {tpcds.queryCount}
  TPC-DS queries ran slower on the GPU than on the CPU, and the UDTFs have no
  benchmark of their own beyond the timings captured on the demo pages.
- GPU relational coverage is a subset of SQL and of Arrow types. Nested and
  binary types and most list shapes stay on the CPU path.
- A query runs on one GPU and does not spill. Independent queries can be
  placed on different devices, but a query never spans two, and a working set
  larger than its grant fails with an allocation error. Graph construction
  and algorithm scratch space count toward that grant.
- Vector indexes and fitted models live inside one statement, and forest
  inference reads its model file for each statement. Graphs are also built
  inside a statement, except that a bounded server can retain constructed
  graphs for a handful of algorithms. Persistent artifacts are still a
  direction.
- GPU file scans read Parquet only. Workspace views and `CREATE TABLE AS`
  tables exist, but nothing writes back to a source catalog. Iceberg waits
  for a DataFusion integration release we can pin. cuOpt is not exposed, and
  there is no Substrait input.

## Alongside Spark

The obvious comparison is NVIDIA's own Spark work.
[cuDF for Spark](https://github.com/NVIDIA/cudf-spark) replaces Spark SQL
operators with cuDF-backed ones, and
[cuML for Spark](https://github.com/NVIDIA/cuml-spark) offers PySpark
ML-compatible estimators backed by cuML. Both serve the job over the whole
lake described above, and the unit of composition differs too. In Spark, an
ML estimator is an object in Python or Scala, and moving between SQL and cuML
is application code again; Spark SQL has no graph algorithms of its own, and
GraphX and GraphFrames are separate APIs. In Algeon the algorithm is a
relation in the physical plan, and an agent stays in SQL to use it. The two
can fit together, with Spark preparing shared datasets and training models
and Algeon serving the exploratory queries asked of the result. Both read
Parquet and both stand on the same RAPIDS libraries.

## Next

The [roadmap](/reference/roadmap) keeps the current order of work, starting
with an admission model that sizes grants closer to what a plan needs and
packaged releases that deploy without a native build. Adapters for engines
other than DataFusion are not planned.

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
