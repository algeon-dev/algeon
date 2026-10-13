---
title: "Introducing Algeon: Interactive Graph, Vector, and ML Analytics for the Agent Era"
description: "Algeon runs graph, vector and machine-learning algorithms on SQL subqueries and returns their output as rows, inside one SQL query. This post explains why that fits an agent that asks many questions through a backend service, how the engine is built, and what it cannot do yet."
slug: introducing-algeon
tags: [announcement]
authors: morris
---

import CodeBlock from '@theme/CodeBlock';
import Table from '@site/src/theme/MDXComponents/Table';
import {FlowDiagram} from '@site/src/components/design/Diagram';
import {AgentLoopFigure, GlueVsPlanFigure} from '@site/src/components/blog/IntroFigures';
import session from '@site/static/data/merrec-agent-session.json';
import barbie from '@site/static/data/merrec-scale-barbie-tie-ins.json';
import listingCheck from '@site/static/data/merrec-barbie-listing-check.json';
import funnel from '@site/static/data/merrec-scale-audience-funnel.json';

export const stepOf = id => session.steps.find(item => item.step === id);
export const seconds = value => value.toFixed(2);
export const wholeMs = value => Math.round(value * 1000).toLocaleString('en-US');
export const gib = bytes => (bytes / 2 ** 30).toFixed(2);
export const peakOf = capture => gib(Math.max(...capture.runs.map(run => run.allocator_peak_bytes)));
export const byStage = Object.fromEntries(funnel.records.map(row => [row.stage, row]));
export const nyx = listingCheck.records.find(row => row.brand_name === 'NYX');

An agent session over 1.27 billion marketplace events compared brand affinity
to Barbie's audience between May-June and July-August 2023, around the film's release.
It issued five queries to Algeon through Arrow Flight SQL. The first two read the catalog: which
graph functions this build installs, and what personalized PageRank takes and
returns. The third looked up Barbie's brand id to seed from, and the fourth
used `EXPLAIN GPU` to check whether the whole query would stay on the GPU.
The fifth ran four graph calls over the events and returned
{stepOf(5).rows.length} brands in {seconds(stepOf(5).elapsed_s)} seconds.
Personalized PageRank highlighted brands such as NYX whose affinity to Barbie rose
between the two periods, and follow-up queries verified the inventory tie-in
against listing titles within seconds.

{/* truncate */}

<Table>
<thead><tr><th>Step</th><th>Purpose</th><th>Outcome</th><th style={{textAlign: 'right'}}>Elapsed</th></tr></thead>
<tbody>{session.steps.map(item => <tr key={item.step}><td>{item.step}</td><td>{item.intent}</td><td style={{whiteSpace: 'nowrap'}}>{item.status === 'ok' ? 'rows' : 'planning error'}</td><td style={{whiteSpace: 'nowrap', textAlign: 'right'}}>{wholeMs(item.elapsed_s)} ms</td></tr>)}</tbody>
</Table>

<div className="fineprint">

Release build on one NVIDIA RTX PRO 6000, a 90 GiB query grant, `native_required` mode and the source Parquet on local NVMe.

</div>

This execution pattern reflects a shift in how humans interact with data
through personal assistants. Users expect assistants that are responsive enough
for interactive dialogue, capable of connecting non-obvious relationships
across large datasets, and grounded in verifiable evidence rather than model
guesses. To meet those expectations, an agent cannot rely on static dashboards
or parametric recall. It operates as an iterative investigator, inspecting
catalog metadata, resolving entity keys, formulating hypotheses, and validating
evidence across multiple queries.

That investigative loop concentrates latency pressure on the underlying engine.
When a conversational turn has an interactive budget of five to ten seconds and
the agent requires several consecutive attempts to verify its reasoning, each
step must complete in seconds or milliseconds. If the middle step involves
graph traversals, vector similarity, or model scoring, traditional lakehouse
batch queues and multi-system host transfers exceed that conversational budget.
In Algeon, an analyst or an AI agent applies a graph, vector, or
machine-learning algorithm to a relation defined by SQL, and the output comes back
as rows the same query can keep joining, filtering, and aggregating. Each
attempt is one query and one physical plan, submitted by the application's
backend service and answered in the same call.

## A working set for each question

A lakehouse holds terabytes, and almost no single question needs more than a
sliver of it. A question is about one tenant, one quarter, one product line,
one cohort. SQL does the narrowing: a filter on a date range, a join against
a dimension table, an aggregation over a segment, and the domain rules that
determine which records belong in the analysis at all. What survives is
the question's working set. Fleet studies put numbers on how small it
usually is. In the Amazon Redshift fleet
the median query uses 0.6 GB of memory and the 99th-percentile query 21.4 GB;
among paying BigQuery customers, 90% of queries process less than 100 MB.[^1]

Algeon executes GPU-accelerated SQL across that working set slice. A working
set of that size fits in GPU memory. The two queries in this post run over
[MerRec](/guides/datasets/merrec), the public marketplace dataset behind the
opening session, and peaked at {peakOf(barbie)} GiB and {peakOf(funnel)} GiB
across its 1.27 billion events.

The questions worth asking about the working set, though, are often the ones
relational algebra handles badly. Which of these companies form a tightly
knit cluster? What is the shortest chain from this supplier to that
sanctioned entity? Which products sit nearest to this one in embedding space?
Which of these rows would the model we trained last quarter flag? Each has a
well-understood algorithm and a mature GPU implementation. Relational algebra
can spell some of them out, a shortest path as a recursive CTE or a
nearest-neighbor search as a cross join with a per-row top-k, but the planner
then faces an iteration or a cross product with no single operator to place,
and modularity-based community detection has no practical SQL spelling at
all.

<FlowDiagram
  ariaLabel="A lakehouse of terabytes is reduced by SQL to a working set of gigabytes, a GPU algorithm runs over it inside the same query, and the output returns as rows or feeds the next stage."
  nodes={[
    {id: 'lake', kind: 'terabytes', name: 'Lakehouse', note: 'Parquet · tables · views', tone: 'lake'},
    {
      id: 'working-set', kind: 'gigabytes', name: 'Working set', tone: 'engine',
      edgeLabel: ['filter · join', 'aggregate'], note: 'fine-grained · fits GPU memory',
    },
    {
      id: 'algorithm', kind: 'gpu algorithm', name: 'Algorithm output', tone: 'gpu',
      edgeLabel: ['cugraph_* · cuvs_*', 'cuml_*'], note: ['paths · communities', 'neighbors · scores'],
    },
    {id: 'answer', kind: 'rows', name: 'Answer', note: 'back to the caller', tone: 'plain',
      edgeLabel: ['join · filter', 'aggregate']},
  ]}
  groups={[{label: 'one SQL query · one physical plan', nodes: ['working-set', 'algorithm', 'answer']}]}
  returns={[{from: 'algorithm', to: 'working-set', label: 'next stage input'}]}
  caption={<>SQL reduces the lakehouse to a working set that fits in GPU memory.
  Algeon places the specialized algorithm after that reduction, inside the
  same query, and returns its output as rows that ordinary SQL can keep
  working on, or that the next algorithm can take as input.</>}
/>

Running the algorithm after the reduction also changes which algorithm fits.
NVIDIA's cuVS guide states the condition for exact search:
["Use brute-force for heavily filtered queries. If a filter excludes most
vectors, brute-force can search only the remaining candidates while IVF and
graph indexes may skip useful vectors before the filter is
applied."](https://docs.nvidia.com/cuvs/user-guide/api-guides/indexing-guide/brute-force#when-to-use-brute-force)
In SQL that filter is the `WHERE` clause and the joins that define the
working set. For example, the campaign query below restricts candidate
embeddings to eligible shoppers before running an exact nearest-neighbor
search over the survivors. The same holds for graphs. The Barbie query that
opened this post builds a separate like graph for each of its two periods from
the filtered events. A result produced this way stays tied to the conditions
that admitted its input, and those conditions are visible in the SQL.

## Replacing glue code with in-plan UDTFs

Outside the query engine, a GPU graph algorithm over warehouse data runs as a
detour around the warehouse. In a typical workflow outside the engine, selected
rows are exported to cloud storage, loaded and transformed within a Python
notebook, processed through cuGraph, and imported back into the warehouse for
subsequent joins. Each step is an ad-hoc
ETL script, written for the small fraction of the dataset that one question
needs, and two scripts that differ slightly can encode different semantics
without anyone comparing them.

<GlueVsPlanFigure
  caption={<>Left: the detour as glue code, where each step is an
  ad-hoc ETL script and most steps copy the data. Right: the same work as one
  SQL query, with the algorithm as a UDTF between the SQL that
  prepares its input and the SQL that consumes its output.</>}
/>

In Algeon, selection and cleaning happen in SQL before the algorithm,
enrichment happens in SQL after it, and the algorithm sits between them as a
UDTF (user-defined table function) with a documented input relation and a
documented output schema. The physical
decisions (what to scan, what to push down, whether a stage can stay on the
device) belong to the planner; the choice of algorithm and its options stays
with the query author.

The detour also fixed where the work could run. A GPU graph step lived in a
notebook or a scheduled job, and a service that needed the result waited for
a file to land. With the algorithm inside the query, the same service submits
the SQL, over Arrow Flight SQL or through the DataFusion session it embeds,
and receives Arrow batches in the same call. No notebook or ad-hoc script sits
between the question and the rows. Graph construction, the algorithm and the
joins around it run inside the request, under the memory grant that request
was admitted with.

The Barbie query from the opening session has that shape. A `WITH`
clause selects distinct shopper-brand likes from May through August and
flags the ones after July 1. Personalized PageRank from the Barbie vertex
measures each brand's connection to Barbie's audience in each period, and
dividing it by ordinary PageRank adjusts for the brand's overall graph score.
The final `SELECT` compares that ratio across the two periods, keeps brands
with at least 1,000 fans in both, and attaches brand names with a join. The
preview ends after the four graph calls.

<CodeBlock language="sql" previewBefore="), fans AS (">{barbie.sql}</CodeBlock>

Because the edge and personalization relations are
[subqueries in the plan](/design/gpu-table-functions#an-algorithm-inside-a-query),
the period filter and the seed stay in the SQL session, and any additional filtering
condition can be modified directly within the SQL statement. The like graphs
live in GPU memory while the query
runs, and nothing is exported. In the opening session this query was the
fifth Flight SQL call, and it returned its rows in
{seconds(stepOf(5).elapsed_s)} seconds with both graph builds inside that
time.

## Validating hypotheses on the same working set

An initial graph query identifies records worth investigating, but a graph
score does not establish why an affinity score moved. Exploring the finding
requires follow-up queries that test competing explanations against the same
working set.

The initial query ranked brands by affinity change, defined as the ratio
of personalized to ordinary PageRank, requiring at least 1,000 distinct fans in
both periods. The four largest increases were retail and entertainment brands;
Taylor Swift ranked fifth and provides a comparison record.

<Table>
<thead><tr><th>Brand</th><th>May-June affinity</th><th>July-August affinity</th></tr></thead>
<tbody>{barbie.records.filter((row, index) => index < 4 || row.brand_name === 'Taylor Swift').map(row => <tr key={row.brand_name}><td>{row.brand_name}</td><td>{row.lift_before.toFixed(2)}</td><td>{row.lift_after.toFixed(2)}</td></tr>)}</tbody>
</Table>

To test whether these affinity increases coincided with Barbie-themed merchandise,
a follow-up query counted likes on listings whose names contained "Barbie".
For NYX, the share of likes on matching listings rose from
{nyx.barbie_pct_may_june.toFixed(2)}% to {nyx.barbie_pct_july_august.toFixed(2)}%,
confirming inventory aligned with the film release. Taylor Swift also had a high
graph score, but the name check found few matching likes, and that movement remains
unexplained by item titles.

To test whether the rise reflected a general summer retail trend, follow-up
queries substituted other seed brands. Nike served as an unrelated baseline; none
of the four brands appeared among the 25 largest affinity changes from Nike.
Hot Wheels, a Mattel brand whose collector audience overlaps Barbie's, produced
smaller increases for NYX, ALDO, and Primark. Each check ran as a single SQL
query with one modified parameter, documented in the
[full investigation](/guides/datasets/merrec#barbie) in the MerRec guide.

The engine executes each follow-up check in seconds, allowing an agent
or analyst to evaluate competing hypotheses in the same interactive session.

## Chaining multiple algorithms in one query

One algorithm's output is often the candidate set, the structure or the
feature the next stage needs, and often something plain SQL could not have
produced at all. The
[campaign query](/guides/datasets/merrec#audience-funnel) from the same MerRec dataset
does this three times for a planner who can contact 5,000 shoppers about
Pokemon listings. Personalized PageRank from the Pokemon brand vertex scores
every shopper on the like graph. An exact nearest-neighbor search finds the
taste lookalikes of earlier buyers among eligible shoppers. A forest trained
offline ranks the union of the two lists, and a `GROUP BY` at the end counts
how many shoppers in each list bought a Pokemon listing in the following two
months. The intermediate relations stay on the device, and the list size,
`k` and the PageRank cutoff are parameters for the next run.

The same query scores two controls with the same model and the same
list size, which is the comparison the planner needs. After ranking, the
combined graph-or-taste pool held
{byStage[8].pokemon_buyers_sep_oct - byStage[7].pokemon_buyers_sep_oct} more
later buyers than the rule a marketer would try first, previous Pokemon
viewers.

<Table>
<thead><tr><th>Candidate pool</th><th>Shortlist size</th><th>Bought Pokemon in Sep-Oct</th></tr></thead>
<tbody>{[
  {stage: 6, label: 'All eligible shoppers'},
  {stage: 7, label: 'Previous Pokemon viewers'},
  {stage: 8, label: 'Graph or taste candidates'},
].map(({stage, label}) => <tr key={stage}><td>{label}</td><td>{byStage[stage].shoppers.toLocaleString('en-US')}</td><td>{byStage[stage].pokemon_buyers_sep_oct}</td></tr>)}</tbody>
</Table>

The query evaluates composition across cuGraph, cuVS, and cuML inside one
physical plan, retaining intermediate buffers in GPU memory and returning the
ranked shortlist in about 20 seconds. The 93 to 72 buyer difference reflects
observational correlation in historical records. An online experiment would be
required to evaluate incremental marketing lift.

## Built for the agent's loop

An agent exploring data should not spend its context on partition pruning,
projection pushdown, or GPU kernel selection. Its role is to formulate
hypotheses in the application's domain vocabulary and present explainable evidence.
A semantic ontology maps that vocabulary to SQL tables, keys, and joins, while
DataFusion and Algeon manage physical planning, join ordering, and GPU execution.

<AgentLoopFigure
  caption={<>The application owns the agent, the ontology that maps its
  vocabulary to tables and keys, and the service that submits SQL. Algeon
  plans, admits, and executes each query on the GPU over the lakehouse,
  and returns rows plus structured diagnostics the agent can act on.</>}
/>

To power this loop, Algeon pairs Apache DataFusion with NVIDIA's RAPIDS libraries.
DataFusion provides the relational spine: SQL parsing, logical optimization,
and transparent CPU fallback for unsupported operators. RAPIDS executes
specialized algorithms from [`cuGraph`](/cugraph), [`cuVS`](/cuvs), and
[`cuML`](/cuml) directly as table functions in the physical plan. When the
surrounding SQL lowers natively, intermediate relations remain on the GPU
across stages, enabling single-query pipelines that
combine graph traversals, vector lookups, and model scoring without host memory round-trips.

Interactive exploration also requires that failures serve as structured interfaces.
Queries return Arrow record batches in seconds, and invalid calls are rejected at
planning time with machine-readable diagnostics, identifying missing columns or
parameter boundaries before device work begins. For example, if a query references
an assumed output column `score` instead of `value` on `cugraph_personalized_pagerank`,
Flight returns a schema error in 3 milliseconds:

```text
Flight returned invalid argument error, with message: No field named score. Valid fields are vertex, value.
```

The calling service receives the expected schema contract immediately,
without allocating device memory or waiting for a kernel launch. Each query runs under an immutable
memory grant from the device ledger, ensuring bounded allocation for multi-query backends.
The [design guide](/design) and [integration guide](/integration) document the
underlying physical lowering rules and memory contracts.

## Serving the working sets that Spark prepares

NVIDIA accelerates batch processing over the lake through its own Spark
integrations. [cuDF for Spark](https://github.com/NVIDIA/cudf-spark) replaces
Spark SQL operators with cuDF-backed ones, and
[cuML for Spark](https://github.com/NVIDIA/cuml-spark) offers PySpark
ML-compatible estimators backed by cuML. Both accelerate a Spark job, which a
driver schedules over partitions of the lake. Algeon targets the moment after
such a job has run, when one working set receives many small ad hoc
questions from a service that waits for each answer. Latency per attempt
matters more there than throughput per job. On the MerRec dataset, with the
source files already in the page cache, the Barbie query completed over Flight
SQL in a median of {seconds(barbie.median_elapsed_s_after_first)} s after its
first run (an equivalent DuckDB and igraph pipeline took 23.3 s on the same host,
producing identical top-25 rankings); the
[MerRec guide](/guides/datasets/merrec) records the first run beside the
median.

The unit of composition differs too. In Spark, an ML estimator is an object in
Python or Scala, and GraphX and GraphFrames require separate APIs outside Spark
SQL. Combining relational filters, graph algorithms, and model inference places
coordination and intermediate buffers in client application code. In Algeon,
each algorithm executes as an in-plan relation, allowing an agent to express the
workflow within SQL.

The two systems form a complementary lifecycle. Spark handles lakehouse-scale
batch extraction, heavy data preparation, and cluster-wide model training;
Algeon serves low-latency exploratory queries over the resulting working sets.
When interactive exploration identifies a durable query pattern, such as the
combined graph and taste candidate funnel, that logic can graduate into a
scheduled Spark batch job or a dedicated service operating over the broader
lake. Both read Parquet files directly and build upon the shared RAPIDS library
ecosystem.

## Known gaps

- Cloud object storage across major providers does not support GPU-direct
  RDMA, and emerging specifications such as NVIDIA cuObject remain early in
  adoption. The captured runs in this post stage working sets on local NVMe
  drives attached to the GPU host.
  The [storage transport design](/design/data-path#the-transport-depends-on-the-source)
  details how file reads route through KvikIO in compatibility mode, using
  pinned host bounce buffers to transfer data from NVMe into GPU memory. Local
  NVMe staging achieves interactive query latencies, but maintaining local disk
  copies is an operational compromise for lakehouse infrastructure. GPU-direct
  reads from shared object storage depend on wider ecosystem adoption, with
  GDS and cuObject tracked on the
  [roadmap](/reference/roadmap#longer-term).
- Lakehouse table formats lack mature Rust integration. Upstream crates
  including `datafusion-iceberg` and `iceberg-rust` remain under active
  development and have not reached the stability required for engine
  integration. Algeon currently reads standalone Parquet files directly,
  without resolving Iceberg metadata, manifest lists, or snapshot histories.
  While Iceberg sources are currently [unsupported](/integration#iceberg-support),
  integrating managed lakehouse tables is planned as a
  [mid-term roadmap item](/reference/roadmap#mid-term) as the open-source
  DataFusion and Iceberg ecosystems mature.
- A query runs on one GPU and does not spill. Independent queries can be
  placed on different devices, but a query never spans two, and a working set
  larger than its grant fails with an allocation error. Graph construction
  and algorithm scratch space count toward that grant. Disk spilling through
  cuCascade and multi-GPU execution are tracked on the
  [roadmap](/reference/roadmap#longer-term).
- Vector indexes, graph topologies, and fitted models are built per execution.
  Forest inference reads its model file during planning for each statement.
  While a bounded server can retain constructed graphs across calls for a subset
  of algorithms, persistent cross-query artifacts remain a roadmap goal.
- GPU relational coverage is a subset of SQL and of Arrow types. Nested and
  binary types and most list shapes stay on the CPU path. The
  [native SQL reference](/reference/native-sql-support) lists supported operators
  and type mappings.

## Next

The [roadmap](/reference/roadmap) keeps the current order of work, starting
with an admission model that sizes grants closer to what a plan needs and
packaged releases that deploy without a native build. Adapters for engines
other than DataFusion are not planned.

Until a packaged release exists, running Algeon means Linux, an NVIDIA GPU
and a [source build](/build/from-source). The
[MerRec setup](https://github.com/algeon-dev/algeon/blob/main/fixture/demo/merrec/scripts/scale/README.md)
prepares the dataset behind this post. The next useful experiment is a
complete investigation on a known dataset. Choose a question whose answer can
be checked independently, define the working set, apply an algorithm and use
its output in the next query. Measure the full query cost, including data
preparation, and compare the result with a trusted reference. A
[GitHub issue](https://github.com/algeon-dev/algeon/issues) with the SQL,
the expected output and the timing tells us where an operator is missing, a
result is wrong or a step costs more than it should.

[^1]: Redshift figures are from van Renen et al.,
[*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB 17(11),
2024, Table 6. The BigQuery figure is from Jordan Tigani,
[*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
MotherDuck, 2023.
