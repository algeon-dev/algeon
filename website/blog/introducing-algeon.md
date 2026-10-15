---
title: "Introducing Algeon: Interactive Graph, Vector, and ML Analytics for the Agent Era"
description: "Algeon runs graph, vector and machine-learning algorithms on SQL subqueries and returns their output as rows, inside one SQL query. This post explains why that fits an agent that asks many questions through a backend service, where Algeon sits beside the query engines already on the lake, how the engine is built, and what it cannot do yet."
slug: introducing-algeon
date: 2026-10-14
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
import controls from '@site/static/data/merrec-barbie-control-seeds.json';

export const stepOf = id => session.steps.find(item => item.step === id);
export const seconds = value => value.toFixed(2);
export const wholeMs = value => Math.round(value * 1000).toLocaleString('en-US');
export const gib = bytes => (bytes / 2 ** 30).toFixed(2);
export const peakOf = capture => gib(Math.max(...capture.runs.map(run => run.allocator_peak_bytes)));
export const byStage = Object.fromEntries(funnel.records.map(row => [row.stage, row]));
export const nyx = listingCheck.records.find(row => row.brand_name === 'NYX');

An agent session over 1.27 billion marketplace events compared brand affinity
to Barbie's audience between May-June and July-August 2023, around the film's release.
It issued five queries to Algeon through Arrow Flight SQL. The first two read
the catalog to check which graph functions this build installs and what
personalized PageRank takes and returns. The third looked up Barbie's brand id
to seed from, and the fourth used `EXPLAIN GPU` to check whether the whole query
would stay on the GPU.
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

Pre-aggregated rollups and reporting tables summarize metrics along predefined
dimensions, such as regional revenue or endpoint latency. Investigating weak
signals, such as unindexed supplier relationships or emerging brand affinities,
instead requires ad hoc joins across unaggregated relations.

An investigative agent tests competing hypotheses through sequences of
dependent queries. If each query takes minutes, interactive exploration becomes
impractical. Within a five-to-ten-second interaction budget, multiple dependent
queries require individual executions to complete in well under a second. Graph
traversals, vector similarity search, and model scoring make this target harder
to meet when each operation introduces separate scheduling, data movement, or
execution overhead.

Algeon operates alongside lakehouse engines, which handle scheduled batch ETL
and produce base tables. Algeon executes ad hoc queries involving graph, vector,
and machine learning operations. Rather than invoking these algorithms as
separate processing stages, Algeon integrates them as relational operators
within a single physical execution plan. Intermediate results flow between
operators as Arrow record batches, allowing these operations to participate in
the same query execution pipeline.

## A working set for each question

A lakehouse holds terabytes, and almost no single question needs more than a
sliver of it. A question is about one tenant, one quarter, one product line,
one cohort. SQL narrows the scan through date filters, dimension joins, and
segment aggregations that isolate relevant records. The surviving rows form
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

Executing the algorithm after the reduction also alters algorithm selection.
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

An initial graph query identifies candidate records across relations that no
pre-existing dashboard tracks, but graph centrality alone does not explain
affinity shifts. Confirmation of an observed correlation requires testing alternative
hypotheses on the same working set.

The opening query ranked brands by affinity change, defined as the ratio of
personalized to ordinary PageRank with at least 1,000 distinct fans in both
periods. Retail and entertainment tie-ins were expected, but the ranking also
highlighted cosmetics brand NYX alongside AMC and Primark; Taylor Swift ranked
fifth and provided a comparison record.

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

Each follow-up check incurred overhead comparable to the primary query. The Hot
Wheels and Nike control reruns completed over Flight SQL in
{seconds(controls.hot_wheels.median_elapsed_s_after_first)} s and
{seconds(controls.nike.median_elapsed_s_after_first)} s respectively, while the
name check executed as a standard relational aggregation. Together, the four
queries isolated the underlying effect. NYX, ALDO, and Primark expanded Barbie
inventories and shifted toward the Hot Wheels audience, AMC shifted exclusively
with Barbie, and Taylor Swift remained uncorrelated with title inventory.
Because execution occurs in-process without manual data exports, verification
queries execute in the same interactive session.

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

The query also evaluates two baseline controls using identical model parameters
and candidate pool sizes. After ranking, the combined graph-and-embedding pool
identified
{byStage[8].pokemon_buyers_sep_oct - byStage[7].pokemon_buyers_sep_oct} more
subsequent buyers than the baseline heuristic of targeting previous Pokemon
viewers.

<Table>
<thead><tr><th>Candidate pool</th><th>Shortlist size</th><th>Bought Pokemon in Sep-Oct</th></tr></thead>
<tbody>{[
  {stage: 6, label: 'All eligible shoppers'},
  {stage: 7, label: 'Previous Pokemon viewers'},
  {stage: 8, label: 'Graph or taste candidates'},
].map(({stage, label}) => <tr key={stage}><td>{label}</td><td>{byStage[stage].shoppers.toLocaleString('en-US')}</td><td>{byStage[stage].pokemon_buyers_sep_oct}</td></tr>)}</tbody>
</Table>

Targeting previous viewers relies on direct historical purchase signals.
Queries combining graph traversal with embedding similarity identify lookalike shoppers
lacking explicit prior interactions. Inside a single physical plan across cuGraph,
cuVS, and cuML, intermediate buffers remain in GPU memory, producing the ranked
shortlist in approximately 20 seconds. The 93 to 72 buyer difference reflects
observational correlation in historical records; evaluating incremental lift
would require an online randomized trial.

## Built for the agent's loop

The client application manages domain vocabularies, query formulation, and
result evaluation. An ontology maps application terms to SQL tables, keys, and
predicates. DataFusion and Algeon handle physical planning, join ordering,
and GPU memory allocation, allowing the client service to focus on hypothesis
testing.

<AgentLoopFigure
  caption={<>The application owns the agent, the ontology that maps its
  vocabulary to tables and keys, and the service that submits SQL. Algeon
  plans, admits, and executes each query on the GPU over the lakehouse,
  and returns rows plus structured diagnostics the agent can act on.</>}
/>

Algeon pairs Apache DataFusion with NVIDIA's RAPIDS libraries. DataFusion parses
SQL, optimizes the logical plan, and executes on the CPU when an operator lacks a
GPU implementation. RAPIDS runs specialized algorithms from
[`cuGraph`](/cugraph), [`cuVS`](/cuvs), and [`cuML`](/cuml) directly as table
functions in the physical plan. When surrounding operators lower natively,
intermediate relations stay in GPU memory across stages, avoiding host memory
copies between graph traversals, vector lookups, and model scoring.

Planning errors return as structured diagnostics. Queries return Arrow record
batches in seconds, and invalid statements fail at planning time before device
allocation starts. For example, if a query references an assumed output column
`score` instead of `value` on `cugraph_personalized_pagerank`, Flight returns
the schema mismatch in 3 milliseconds:

```text
Flight returned invalid argument error, with message: No field named score. Valid fields are vertex, value.
```

The calling service receives the expected schema without launching a kernel or
allocating device buffers. Each query executes under an immutable memory grant
from the device ledger, bounding memory usage across concurrent clients.
The [design guide](/design) and [integration guide](/integration) document the
underlying physical lowering rules and memory contracts.

## Beside the engines already on the lake

Distributed query engines and batch schedulers manage large-scale lakehouse
workloads. NVIDIA accelerates distributed batch execution through GPU-accelerated
Spark packages: [cuDF for Spark](https://github.com/NVIDIA/cudf-spark) replaces
Spark SQL operators with cuDF implementations, and
[cuML for Spark](https://github.com/NVIDIA/cuml-spark) provides PySpark ML
estimators backed by cuML. A Spark driver coordinates task execution across
cluster partitions.

Algeon targets the phase following batch preparation, when a filtered working set
receives repeated ad hoc queries from an interactive service. Latency per attempt
takes precedence over aggregate batch throughput. On the MerRec dataset, with
files staged in the OS page cache, the Barbie query completed over Flight SQL in
a median of {seconds(barbie.median_elapsed_s_after_first)} s after its first run
(an equivalent DuckDB and igraph pipeline required 23.3 s on the same host for
identical top-25 rankings); the [MerRec guide](/guides/datasets/merrec) documents
initial and median timings.

The execution model also differs. In Spark, an ML estimator is an external object
in Python or Scala, and graph operations require external GraphX or GraphFrames
APIs. A pipeline combining relational filters, graph algorithms, and inference
buffers intermediate results in driver or application memory. In Algeon, each
algorithm is an in-plan table function within a single SQL statement.

Spark and Algeon address distinct stages. Spark manages cluster extraction,
large-scale transformation, and distributed model training. Algeon executes
interactive analytical queries against the resulting Parquet files. When an
exploratory pattern proves persistent, such as the candidate generation funnel,
that logic can be deployed as a scheduled Spark job over the wider lake. Both
read Parquet directly and build on RAPIDS libraries.

## Known gaps

- Cloud object storage across major providers lacks GPU-direct RDMA support.
  The runs in this post stage working sets on local NVMe drives attached to the
  GPU host. File reads route through KvikIO in compatibility mode, using pinned
  host bounce buffers to transfer data from NVMe into GPU memory. Local NVMe
  staging delivers interactive query latencies, but maintaining local disk
  copies remains an operational trade-off for lakehouse setups. Direct reads from
  shared object storage via GDS and cuObject are tracked on the
  [roadmap](/reference/roadmap#longer-term).
- Lakehouse table formats lack mature Rust integration. Upstream crates
  including `datafusion-iceberg` and `iceberg-rust` remain under active
  development. Algeon currently reads standalone Parquet files directly,
  without resolving Iceberg metadata, manifest lists, or snapshot histories.
  Managed Iceberg table support is planned as a
  [mid-term roadmap item](/reference/roadmap#mid-term).
- A query runs on one GPU and does not spill. Independent queries can run on
  different devices, but a single query cannot span multiple GPUs. A working set
  larger than its grant fails with an allocation error. Graph construction and
  algorithm scratch space count toward that grant. Disk spilling through
  cuCascade and multi-GPU execution are tracked on the
  [roadmap](/reference/roadmap#longer-term).
- Vector indexes, graph topologies, and fitted models are built per execution.
  Forest inference reads its model file during planning for each statement.
  While a bounded server can retain constructed graphs across calls for a subset
  of algorithms, persistent cross-query artifacts remain planned for future
  releases.
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
prepares the dataset behind this post.

The next useful experiment is a complete investigation on a known dataset.
Identify a query with a verifiable baseline, define the working set, and execute
the algorithm in SQL. Measure total elapsed time, including scan and data
preparation, and compare outputs against a trusted reference. A
[GitHub issue](https://github.com/algeon-dev/algeon/issues) documenting the SQL,
expected results, and execution timings highlights where an operator is missing
or performance regresses.

[^1]: Redshift figures are from van Renen et al.,
[*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB 17(11),
2024, Table 6. The BigQuery figure is from Jordan Tigani,
[*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
MotherDuck, 2023.
