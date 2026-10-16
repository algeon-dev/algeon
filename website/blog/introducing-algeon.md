---
title: "Introducing Algeon: Interactive Graph, Vector, and ML Analytics for the Agent's Loop"
description: "Follow a marketplace investigation to see how Algeon combines SQL with GPU graph, vector, and ML algorithms, supports follow-up queries, and fits beside existing lakehouse engines."
slug: introducing-algeon
date: 2026-10-15
tags: [announcement]
authors: morris
---

import CodeBlock from '@theme/CodeBlock';
import Link from '@docusaurus/Link';
import Table from '@site/src/theme/MDXComponents/Table';
import {AgentLoopFigure, BarbieTitleShareFigure, GlueVsPlanFigure, QueryInterfaceFigure} from '@site/src/components/blog/IntroFigures';
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
export const swift = listingCheck.records.find(row => row.brand_name === 'Taylor Swift');
export const nyxListing = listingCheck.listing_records.find(row => row.brand_name === 'NYX');

Which brands moved toward Barbie's audience around the July 2023 film release?
An agent put that question to Algeon over Arrow Flight SQL, using
[MerRec](/guides/datasets/merrec), a dataset of 1.27 billion marketplace events.
The query compared brand affinity in May-June with July-August. Primark and
AMC appeared near the top, alongside the cosmetics brand NYX in first place
and Taylor Swift in fifth. What could explain those changes?
{/* truncate */}

Algeon runs graph, vector, and machine-learning algorithms as table functions
inside SQL. Each function takes rows from a subquery and returns rows for the
rest of the query to use. In this investigation, SQL selected the likes,
graph algorithms measured brand affinity, and a final join attached brand
names to the ranking.

The ranking also raised questions that needed different SQL. An endpoint
built to return brand rankings might accept a new date range or seed brand,
but checking the titles of liked products requires another query. With
Algeon, an agent or analyst can express that follow-up through the SQL
interface, using the tables and functions available to the session.

## Replacing glue code with in-plan UDTFs

A workflow built from separate SQL and graph tools often exports the selected
rows, constructs a graph in application code, runs an algorithm, and imports
the scores for another join. Algeon's user-defined table functions (UDTFs)
put these dependencies in the query plan. The query author supplies the
input relations and algorithm parameters; the engine executes the operators
and manages their memory.

<GlueVsPlanFigure
  caption={<>A table function accepts the rows selected by SQL and returns
  rows that downstream joins and aggregations can use. The application
  does not need to export and reimport each intermediate result.</>}
/>

For the Barbie query, each period has a graph connecting shoppers to the
brands they liked. Personalized PageRank starts from the Barbie vertex and
scores connections through that graph. Dividing each score by ordinary
PageRank adjusts for the brand's overall graph score. We call this ratio
its affinity to Barbie's audience.

The query compares affinity across the periods, keeps brands with at least
1,000 distinct fans in both, and ranks the relative change. The July-August
window starts on July 1, twenty days before the film opened, so the comparison
includes the weeks leading up to the release. The SQL preview ends after
the four graph calls.

<CodeBlock language="sql" previewBefore="), fans AS (">{barbie.sql}</CodeBlock>

The ranking put NYX first, followed by AMC, Primark, ALDO, and Taylor Swift.

<Table>
<thead><tr><th>Brand</th><th>May-June affinity</th><th>July-August affinity</th><th>Change</th></tr></thead>
<tbody>{barbie.records.filter((row, index) => index < 4 || row.brand_name === 'Taylor Swift').map(row => <tr key={row.brand_name}><td>{row.brand_name}</td><td>{row.lift_before.toFixed(2)}</td><td>{row.lift_after.toFixed(2)}</td><td>{row.lift_change.toFixed(2)}×</td></tr>)}</tbody>
</Table>

The graph inputs are [subqueries in the plan](/design/gpu-table-functions#an-algorithm-inside-a-query).
Changing a date filter or seed requires editing the SQL. The like graphs and
intermediate results remain in GPU memory during this query; graph
construction is included in the measured time.

The ranking returned {stepOf(5).rows.length} brands in
{seconds(stepOf(5).elapsed_s)} seconds in the session replay. In separate
repeated runs, its median was {seconds(barbie.median_elapsed_s_after_first)}
seconds after the first run. An equivalent CPU workflow using 32-thread
DuckDB and igraph took a median of 23.3 seconds on the same host and returned
identical top-25 rankings. The [case study](/guides/datasets/merrec#barbie)
includes the timing breakdown.

<div className="fineprint">

GPU timings in this post use an NVIDIA RTX PRO 6000 with a 90 GiB query
grant and `native_required` mode. Source Parquet files were staged on local
NVMe and held in the OS page cache for both the GPU and CPU comparisons.

</div>

## Testing follow-up hypotheses on the working set

A change in graph affinity leaves its cause open. Barbie merchandise could
account for some of the ranking. Broader summer trends or an overlapping
collector audience could also contribute. We followed the ranking with a
product-title check and two reruns using other seed brands. These were
separate queries after the recorded agent session.

The title check counts likes on listings whose names contain "Barbie" over
the two periods. For NYX, that share rose from
{nyx.barbie_pct_may_june.toFixed(2)}% to
{nyx.barbie_pct_july_august.toFixed(2)}%, with listings such as
"{nyxListing.name}". For Taylor Swift, only {swift.barbie_likes_july_august}
of {swift.total_likes_july_august.toLocaleString('en-US')} July-August likes
went to matching listings.

<BarbieTitleShareFigure
  records={listingCheck.records}
  caption={<>Share of likes on listings with "Barbie" in the title, before
  and during the release period. NYX, Primark, ALDO, and AMC show substantial
  increases; Taylor Swift's share remains below 0.1%.</>}
/>

These results support a merchandise explanation for NYX, Primark, ALDO, and
AMC. The name match is an imperfect measure: seller descriptions and color
names can also contain "Barbie". It does not establish what caused the graph
affinity to change.

Next, we reran the graph query with Nike and Hot Wheels as seeds, changing
the seed brand id and its exclusion from the results. None of those four
brands appeared among Nike's 25 largest affinity changes. With Hot Wheels,
NYX, ALDO, and Primark appeared with smaller changes; AMC was absent from
the top 25. The Hot Wheels result is consistent with an overlapping Mattel
collector audience. These comparisons help distinguish the rankings, though
absence from a top-25 list cannot rule out a broader seasonal change.

The Hot Wheels and Nike reruns took
{seconds(controls.hot_wheels.median_elapsed_s_after_first)} and
{seconds(controls.nike.median_elapsed_s_after_first)} seconds respectively,
using medians after the first run. The [full investigation](/guides/datasets/merrec#barbie)
contains the follow-up SQL and all result rows. The checks support some
explanations and leave Taylor Swift's rise unresolved. Investigating that
connection would require another question and another query.

## Chaining multiple algorithms in one query

The Barbie investigation changed the query as new questions arose. A single
question can also require several algorithms. To select 5,000 shoppers for
a Pokemon promotion, the [campaign query](/guides/datasets/merrec#audience-funnel)
combines graph affinity, vector similarity, and model scores.

Personalized PageRank from the Pokemon brand identifies graph candidates.
Exact nearest-neighbor search finds eligible shoppers with tastes similar
to historical buyers. An offline-trained XGBoost model ranks the union of
these candidates, and the query keeps the top 5,000. A final aggregation
counts how many bought Pokemon products in the following two months.

<QueryInterfaceFigure
  caption={<>Eligibility filters, graph candidates, vector neighbors, and
  model scores form one SQL query. Candidate limits and algorithm parameters
  can be changed in that query.</>}
/>

For comparison, the model also ranks all eligible shoppers and those who
previously viewed a Pokemon listing. Each pool contributes its top 5,000.

<Table>
<thead><tr><th>Candidate pool</th><th>Shortlist size</th><th>Bought Pokemon in Sep-Oct</th></tr></thead>
<tbody>{[
  {stage: 6, label: 'All eligible shoppers'},
  {stage: 7, label: 'Previous Pokemon viewers'},
  {stage: 8, label: 'Graph or taste candidates'},
].map(({stage, label}) => <tr key={stage}><td>{label}</td><td>{byStage[stage].shoppers.toLocaleString('en-US')}</td><td>{byStage[stage].pokemon_buyers_sep_oct}</td></tr>)}</tbody>
</Table>

The graph-or-taste shortlist contained
{byStage[8].pokemon_buyers_sep_oct - byStage[7].pokemon_buyers_sep_oct} more
later buyers than the previous-viewer shortlist. Its candidates need not
have viewed a Pokemon listing before September. This is a historical
comparison; an online randomized trial would be needed to measure whether
sending the promotion caused additional purchases.

The query ran in a median of {seconds(funnel.median_elapsed_s_after_first)}
seconds after the first run. Intermediate results stayed in GPU memory
across cuGraph, cuVS, and cuML, including the candidate relations used by
the model and the scores consumed by the final SQL.

## A working set for each question

Both examples select algorithm inputs from a larger dataset. The Barbie
query builds graphs from distinct shopper-brand likes in each period. The
campaign query restricts vector search to eligible shoppers. SQL expresses
those filters and joins before the algorithm runs, forming the working set
for that question.

Memory requirements include graph construction, algorithm scratch space,
and intermediate results as well as the input rows. Across the recorded
runs, peak query allocations were {peakOf(barbie)} GiB for Barbie and
{peakOf(funnel)} GiB for the campaign, within the configured 90 GiB grant.

Algeon currently runs each query on one GPU and does not spill to disk.
A query whose allocations exceed its grant fails. Filters and candidate
limits therefore affect both the analytical question and whether it fits
the available memory.

## Built for the agent's loop

The opening session shows how an agent can discover and use this SQL
interface. It made five Flight SQL calls: two catalog reads for graph
functions and their signatures, a brand-id lookup, an `EXPLAIN GPU` check
for native coverage, and the ranking query.

<Table>
<thead><tr><th>Step</th><th>Purpose</th><th style={{textAlign: 'right'}}>Elapsed</th></tr></thead>
<tbody>{session.steps.map(item => <tr key={item.step}><td>{item.step}</td><td>{item.intent}</td><td style={{whiteSpace: 'nowrap', textAlign: 'right'}}>{wholeMs(item.elapsed_s)} ms</td></tr>)}</tbody>
</Table>

The agent wrote the statements in order after reading each response on
October 12. The timings above come from replaying those statements on
October 15 against an updated server. The [session record](/data/merrec-agent-session.json)
documents the replay method and results.

Errors also provide information for revising a query. A separate probe asked
for a `score` column when `cugraph_personalized_pagerank` returns `vertex`
and `value`. Flight returned this error in
{Math.round(session.probes.find(probe => probe.error.includes('No field named score')).elapsed_s * 1000)} milliseconds:

<CodeBlock language="text" className="wrap-lines">{'Flight returned invalid argument error, with message: No field named score. Valid fields are vertex, value.'}</CodeBlock>

The schema check returned valid column names during planning, before GPU
execution or device buffer allocation. The calling application can use
those names to revise the query.

The application remains responsible for choosing questions, generating SQL,
and evaluating the results. Its domain mapping, or ontology, connects terms
such as "brand" and "shopper" to tables, keys, and predicates. When several
engines serve the lake, the application also tracks their available
functions and chooses where to submit each query.

Algeon combines Apache DataFusion with NVIDIA RAPIDS. DataFusion parses SQL
and optimizes the plan. Algeon executes supported relational operators and
[`cuGraph`](/cugraph), [`cuVS`](/cuvs), and [`cuML`](/cuml) table functions on
the GPU. Native stages can keep intermediate relations there until the
final rows return as Arrow record batches. Execution mode controls whether
unsupported operators stay on DataFusion's CPU path or cause planning to
fail; the examples here require native execution. Each GPU query attempt
receives a fixed [memory grant](/design/admission), enforced during execution.

## Beside the engines already on the lake

Algeon targets investigations over prepared data. In an existing deployment,
Spark jobs can continue preparing Parquet tables, while engines such as
Trino, ClickHouse, or StarRocks serve the application's standard analytical
queries. The application can submit graph, vector, and ML queries to Algeon
against those Parquet files.

<AgentLoopFigure
  olap
  caption={<>The application maps domain terms to tables and functions,
  selects an engine, and submits SQL through its backend service. Algeon
  executes the graph, vector, and ML queries; results return to the
  application for evaluation and follow-up questions.</>}
/>

This arrangement requires data preparation. The examples use local Parquet
copies on the GPU host. Algeon does not provide Trino or ClickHouse
connectors, and it does not resolve Iceberg snapshots from table metadata.
Adapters for query engines other than DataFusion are not planned. The
current storage and execution limits are listed below.

## Known gaps

<Table>
<thead><tr><th>Gap</th><th>Today</th><th>Planned</th></tr></thead>
<tbody>
<tr><td style={{whiteSpace: 'nowrap'}}>Object storage reads</td><td>No GPU-direct RDMA from the major cloud providers. The runs in this post stage working sets on local NVMe attached to the GPU host, and file reads route through KvikIO in compatibility mode with pinned host bounce buffers. Keeping local copies is an operational trade-off for a lakehouse setup.</td><td>GDS and cuObject reads from shared object storage, <Link to="/reference/roadmap#longer-term">longer term</Link></td></tr>
<tr><td style={{whiteSpace: 'nowrap'}}>Lakehouse table formats</td><td>Algeon reads standalone Parquet files and resolves no Iceberg metadata, manifest lists, or snapshot histories.</td><td>Managed Iceberg tables, <Link to="/reference/roadmap#mid-term">mid term</Link></td></tr>
<tr><td style={{whiteSpace: 'nowrap'}}>One GPU, no spill</td><td>A query runs on one GPU. Independent queries can run on different devices, but a working set larger than its grant fails with an allocation error, and graph construction and algorithm scratch space count toward that grant.</td><td>Disk spilling through cuCascade and multi-GPU execution, <Link to="/reference/roadmap#longer-term">longer term</Link></td></tr>
<tr><td style={{whiteSpace: 'nowrap'}}>Per-query artifacts</td><td>Vector indexes, graph topologies, and fitted models are built per execution, and forest inference reads its model file during planning for each statement. A bounded server can retain constructed graphs across calls for a subset of algorithms.</td><td>Persistent cross-query artifacts in a future release</td></tr>
<tr><td style={{whiteSpace: 'nowrap'}}>GPU relational coverage</td><td>A subset of SQL and of Arrow types. Nested and binary types and most list shapes stay on the CPU path.</td><td>The <Link to="/reference/native-sql-support">native SQL reference</Link> lists supported operators and type mappings</td></tr>
</tbody>
</Table>

## Next

The Barbie investigation illustrates the workflow Algeon supports today.
A graph ranking led to a product-title check and comparisons with other
seed brands, all expressed through SQL. Some results gained supporting
evidence; Taylor Swift remained a question for further investigation.
The campaign example extended that interface to graph, vector, and model
operations within one query.

Algeon is available to try when the required data can be staged as Parquet
and the query fits on one GPU. Running it currently requires Linux, an
NVIDIA GPU, and a [source build](/build/from-source). Start with the
[MerRec setup](https://github.com/algeon-dev/algeon/blob/main/fixture/demo/merrec/scripts/scale/README.md)
to reproduce these queries and measurements. The [development plan](/reference/roadmap)
includes closer sizing of memory grants and packaged releases that remove
the native build requirement. Issues and operator coverage requests are
tracked on [GitHub](https://github.com/algeon-dev/algeon/issues).
