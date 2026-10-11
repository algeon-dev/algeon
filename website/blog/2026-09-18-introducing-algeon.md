---
title: "Introducing Algeon: GPU OLAP for the Agent's Loop"
description: "Algeon runs graph, vector and machine-learning algorithms on SQL subqueries and returns their output as rows, inside one statement and on one GPU. This post explains why that fits an agent that asks many questions, how the engine is built, and what it cannot do yet."
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
import listingCheck from '@site/static/data/merrec-barbie-listing-check.json';
import funnel from '@site/static/data/merrec-scale-audience-funnel.json';
import {tpcdsSf100Benchmark} from '@site/src/data/tpcdsSf100Benchmark';

export const stepOf = id => session.steps.find(item => item.step === id);
export const errorOf = id => stepOf(id).error.replace(/^.*?message: /s, '');
export const seconds = value => value.toFixed(2);
export const wholeMs = value => Math.round(value * 1000).toLocaleString('en-US');
export const gib = bytes => (bytes / 2 ** 30).toFixed(2);
export const peakOf = capture => gib(Math.max(...capture.runs.map(run => run.allocator_peak_bytes)));
export const tpcds = tpcdsSf100Benchmark.summary;
export const tpcdsDate = tpcdsSf100Benchmark.environment.measuredAtUtcDate;
export const byStage = Object.fromEntries(funnel.records.map(row => [row.stage, row]));
export const nyx = listingCheck.records.find(row => row.brand_name === 'NYX');
export const measurements = [
  {name: "Barbie's July", to: '/guides/datasets/merrec#barbie', capture: barbie},
  {name: 'A campaign list', to: '/guides/datasets/merrec#audience-funnel', capture: funnel},
];

An agent session over 1.27 billion marketplace events asked which brands
moved toward Barbie's audience after the film opened in July 2023. It issued
six statements through Arrow Flight SQL: two read the catalog of installed
graph functions, two guessed column names and failed at planning time before
any row was read, and one asked `EXPLAIN GPU` whether the whole statement
would stay on the GPU. The sixth ran four graph calls over the events and
returned {stepOf(6).rows.length} brands in {seconds(stepOf(6).elapsed_s)} seconds.

{/* truncate */}

<Table>
<thead><tr><th>Step</th><th>Statement</th><th>Outcome</th><th>Elapsed</th></tr></thead>
<tbody>{session.steps.map(item => <tr key={item.step}><td>{item.step}</td><td>{item.intent}</td><td>{item.status === 'ok' ? 'rows' : 'planning error'}</td><td>{wholeMs(item.elapsed_s)} ms</td></tr>)}</tbody>
</Table>

<div className="fineprint">

The [session capture](/data/merrec-agent-session.json) records each statement, its Flight status and the returned rows, as of {session.recorded_at_utc}. Elapsed times cover one statement each; they exclude model generation and the pauses between calls.

</div>

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
limits how often it asks. The step that stayed expensive was any question
with a graph traversal, a nearest-neighbor search or a model score in it.
Those ran as separate jobs, with an export before and an import after, and
the answer arrived after the agent's turn had ended.

Algeon moves those steps into the statement. The Barbie statement builds two
like graphs from the 1.27 billion events, runs four PageRank calls and joins
brand names. A second statement from the same dataset, a campaign list, runs
personalized PageRank, an exact nearest-neighbor search among its result, a
trained forest and an evaluation against later purchases. Each was run three
times over Flight SQL, with planning, data reads, algorithm execution and
receipt of the complete result inside the timing:

<Table>
<thead><tr><th>Statement</th><th>First run</th><th>Median of runs 2 and 3</th><th>Peak GPU allocation</th></tr></thead>
<tbody>{measurements.map(({name, to, capture}) => <tr key={name}><td><Link to={to}>{name}</Link></td><td>{seconds(capture.runs[0].elapsed_s)} s</td><td>{seconds(capture.median_elapsed_s_after_first)} s</td><td>{peakOf(capture)} GiB</td></tr>)}</tbody>
</Table>

Both ran on one 96 GB GPU with a 90 GiB query limit and cached source files.
The first run is listed on its own because it can be slower, and these
captures do not isolate why. The
[capture conditions](https://github.com/algeon-dev/algeon/blob/main/fixture/demo/merrec/scripts/scale/README.md#server-and-measurements)
and the saved [Barbie](/data/merrec-scale-barbie-tie-ins.json) and
[campaign](/data/merrec-scale-audience-funnel.json) results name the server
configuration, the binary and the query IDs behind each number.

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
size fits in the memory of one GPU, with the caveat that decoded columns,
graph storage and temporary algorithm memory all count against it. The
compressed size of the source does not decide the question; the two MerRec
statements peaked at {peakOf(barbie)} GiB and {peakOf(funnel)} GiB inside
their 90 GiB grant.

The questions worth asking about the working set, though, are often the ones
relational algebra handles badly. Which of these companies form a tightly
knit cluster? What is the shortest chain from this supplier to that
sanctioned entity? Which products sit nearest to this one in embedding space?
Which of these rows would the model we trained last quarter flag? Each has a
well-understood algorithm and a mature GPU implementation, and none of them
is expressible as a join.

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

The Barbie statement from the opening session has that shape. A `WITH`
clause selects distinct shopper-brand likes from May through August and
flags the ones after July 1. Personalized PageRank from the Barbie vertex
measures each brand's connection to Barbie's audience in each period, and
dividing it by ordinary PageRank adjusts for the brand's overall graph score.
The final `SELECT` compares that ratio across the two periods, keeps brands
with at least 1,000 fans in both, and attaches brand names with a join. The
preview ends after the four graph calls; **Show more** reveals the rest.

<CodeBlock language="sql" previewBefore="), fans AS (">{barbie.sql}</CodeBlock>

Because the edge and personalization relations are subqueries, the period
filter and the seed stay in the SQL session, and every other condition is one
edit away. The like graphs live in GPU memory for the duration of the
statement, and nothing is exported. The shortest documented example, a
[reading list for BERT](/cugraph/udtfs/cugraph_personalized_pagerank?dataset=citation-network#examples)
over a public citation network, is the same pattern in a query short enough
to read on one screen.

## Revise the question

The result named four brands whose affinity rose most. Each had at least
1,000 distinct fans in both periods. The affinity column is the ratio of
personalized to ordinary PageRank; 1 means the two scores are equal.

<Table>
<thead><tr><th>Brand</th><th>May-June affinity</th><th>July-August affinity</th></tr></thead>
<tbody>{barbie.records.slice(0, 4).map(row => <tr key={row.brand_name}><td>{row.brand_name}</td><td>{row.lift_before.toFixed(2)}</td><td>{row.lift_after.toFixed(2)}</td></tr>)}</tbody>
</Table>

A graph score identifies records worth investigating without explaining the
movement, and the next statements tested explanations. A search for "Barbie"
in listing names found that the share of NYX likes on matching listings rose
from {nyx.barbie_pct_may_june.toFixed(2)}% to {nyx.barbie_pct_july_august.toFixed(2)}%.
Taylor Swift also ranked highly in the graph, but the name check found few
matching likes, and that movement stays unexplained.

A second check re-ran the ranking with Hot Wheels and then Nike as the seed,
to separate a Barbie-specific movement from a general summer rise. With Nike,
none of the four brands appeared among the 25 largest changes. With Hot
Wheels, a Mattel brand whose collector audience overlaps Barbie's, NYX, ALDO
and Primark appeared with smaller changes. Each check was one more statement
with one changed condition, and the
[full investigation](/guides/datasets/merrec#barbie) has the SQL and the
result of each.

These comparisons describe associations in the records. The calendar split
is July 1, twenty days before the film's release, and listing names are the
latest retained attributes. Whether a rise in affinity means a partnership
remains the application's judgment; the engine's job is to make the next
check cheap enough to run.

## Follow the result somewhere new

Stages chain. One algorithm's output is often the candidate set, the
structure or the feature the next stage needs, and often something plain
SQL could not have produced at all. The
[campaign statement](/guides/datasets/merrec#audience-funnel) measured above
does this three times for a planner who can contact 5,000 shoppers about
Pokemon listings. Personalized PageRank from the Pokemon brand vertex scores
every shopper on the like graph. An exact nearest-neighbor search finds the
taste lookalikes of earlier buyers among eligible shoppers. A forest trained
offline ranks the union of the two lists, and a `GROUP BY` at the end counts
how many shoppers in each list bought a Pokemon listing in the following two
months. The intermediate relations stay on the device, and the list size,
`k` and the PageRank cutoff are parameters for the next run.

The same statement scores two controls with the same model and the same
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

The counts are later purchases in historical records; they do not measure
what a campaign would add. The taste vectors are a prepared table and the
model was trained before the query, so their preparation is outside the
timing, and ties at the neighbor cutoff can move one or two shoppers between
runs.

## Built for the agent's loop

An agent in that loop should not spend its context on which partitions to
read, which columns to project, whether a filter can move below a join, or
which operators have a GPU kernel. Each of those is the planner's job. An
agent that also plays optimizer, scheduler, storage engine and GPU runtime
has less attention left for what the data shows.

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

A first query can be wrong, and the error message is part of the interface.
An agent that receives a stack trace learns nothing; an agent that receives
a stable code, the kind of failure and the context that triggered it can fix
the query on the next attempt. The two failures in the opening session read,
in full:

<CodeBlock language="text" className="blog-code-message">{[errorOf(3), errorOf(4)].join('\n')}</CodeBlock>

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
will not; in the opening session it took {wholeMs(stepOf(5).elapsed_s)} ms
and reported no host boundary. The
[discovery guide](/guides/gpu-udtf-discovery) walks through the sequence and
replays the session. None of these interfaces establishes that a hypothesis
is sound; that judgment stays with the application and the people behind it.

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
rows with a pretrained XGBoost, LightGBM or Treelite forest. The last one
reflects where trained models usually live. The model is trained elsewhere,
weeks earlier and by someone else, and the engine's job is to carry the
features produced by joins and aggregations into inference without a detour
through files. Query-local fitting serves exploration; scoring a model
somebody else trained is the case we expect in production.

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
  {tpcdsDate}, all {tpcds.queryCount} queries completed on both paths with
  matching row counts; three of them were also compared value by value, and
  row-count agreement does not verify the rest. The
  [benchmark page](/reference/benchmarks) holds the per-query rows and the
  TPC-H run. A few similarity scores were checked against independent
  calculations; that check found a normalization defect in the pinned cuGraph
  release's cosine similarity, and the fork Algeon builds against carries the
  fix. There is no end-to-end graph benchmark and no value-level verification
  of every algorithm's output, and application results still need comparison
  with a trusted reference.
- Performance is uneven. {tpcds.gpuSlowerCount} of the {tpcds.queryCount}
  TPC-DS queries ran slower on the GPU than on the tuned CPU path, and the
  UDTFs have no benchmark of their own beyond the timings captured on the
  demo pages, which include no CPU comparison for the algorithm calls.
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

Until a packaged release exists, running Algeon means Linux, an NVIDIA GPU
and a [source build](/build/from-source); the
[MerRec setup](https://github.com/algeon-dev/algeon/blob/main/fixture/demo/merrec/scripts/scale/README.md)
prepares the dataset behind this post. The next useful experiment is a
complete investigation on a known dataset. Choose a question whose answer can
be checked independently, define the working set, apply an algorithm and use
its output in the next query. Measure the full query cost, including data
preparation, and compare the result with a trusted reference. A
[GitHub issue](https://github.com/algeon-dev/algeon/issues) with the SQL,
the expected output and the timing tells us where an operator is missing, a
result is wrong or a step costs more than it should. And if the experiment
answers a question you had written off as too expensive to ask, tell us
which one.

[^1]: Redshift figures are from van Renen et al.,
[*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB 17(11),
2024, Table 6. The BigQuery figure is from Jordan Tigani,
[*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
MotherDuck, 2023.
