---
title: "DataFusion Nexus: Specialized GPU Algorithms, Now in SQL"
description: Why we built a GPU query engine on top of Apache DataFusion — the problems it solves, the design principles behind it, and where it stands today.
slug: introducing-datafusion-nexus
tags: [announcement]
authors: morris
---

:::note[Current repository layout]

This article describes the project at publication time. The maintained Rust
integration layers now live under `components/cudf`, `components/cugraph`, and
`components/cuvs` in the monorepo; the current one-clone build and supported
surface are documented in [Build & Test](/build) and [the overview](/).
Historical references in this post are not current API or capability promises.

:::

DataFusion Nexus is a GPU query engine built on
[Apache DataFusion](https://datafusion.apache.org/). It evaluates whole
relational physical-plan candidates and selects supported candidates for GPU
execution with [NVIDIA cuDF](https://docs.rapids.ai/api/cudf/stable/). Graph
algorithms from [cuGraph](https://docs.rapids.ai/api/cugraph/stable/) are
explicit GPU islands callable directly from SQL, and data is read from
[Apache Iceberg](https://iceberg.apache.org/) tables and Parquet files — on
S3-compatible object storage or on local disk. A relational candidate that is
not selected keeps the executable DataFusion baseline. It embeds as a Rust
library inside your own service, or serves remote clients over Flight SQL.

Before walking through what Nexus does, this first post explains why it
exists: where today's GPU analytics stack falls short, and what we think 
a fix should look like.

{/* truncate */}

## The problem: GPUs and query engines live in different worlds

GPU libraries for data are not the missing piece — they already exist, and
they are excellent. RAPIDS ships cuDF for DataFrame operations on an
Arrow-like columnar layout, cuGraph for graph analytics, cuVS for vector
search, cuML for machine learning primitives, cuOpt for optimization. For the
workloads they target, the gap between CPU and GPU is not incremental: a heavy
join or aggregation over a large fact table, or a traversal over a graph with
tens of millions of edges, can run one to two orders of magnitude faster on a
modern GPU.

Yet almost none of that shows up inside the systems where enterprises actually
query their data. In practice, a team that wants GPU acceleration over its own
data ends up building a detour:

1. **An ETL pipeline out of the warehouse.** The data an analysis needs is
   spread across tables, databases, and domains — the familiar silo problem of
   any lakehouse or warehouse estate. Before a GPU can touch it, someone has
   to extract it, reshape it, and land it somewhere the GPU job can read.
2. **A notebook-shaped job at the end of the pipeline.** The GPU ecosystem
   grew up around data science, not backend services. cuDF and cuGraph have
   clean Python APIs, but they are libraries you *program against*, not
   engines you *query*. So the GPU work becomes a scheduled script: load these
   files, run these operations, write the result back.

The result inherits the worst properties of both halves. Latency is measured
in pipeline runs, not query response times. The set of questions you can ask
is frozen at whatever the script was written to compute — there is no ad-hoc
query. Each new question means new glue code, each handoff between processes
means another serialization of intermediate data, and an expensive GPU sits
idle between batch windows. The common workaround — a dedicated graph database
here, a OLAP database there, each with its own copy of the data — only
multiplies the silos and the synchronization pipelines.

Meanwhile, the modern OLAP engines that do fit inside backend services—DataFusion,
DuckDB, and their peers—earned that fit through years of careful engineering 
and iterative refinement. Full SQL support, projection pruning, predicate pushdown,
statistics-based planning, catalog integration, lakehouse table formats. None
of that work is GPU-specific, and none of it exists in the GPU library world.

So the situation, simplified, is this: **the query-engine world has the
infrastructure but not the silicon, and the GPU world has the silicon but not
the infrastructure.** Nexus is an attempt to close that gap without rebuilding
either side.

## Why now

Two developments make this the right moment to try.

**Single-GPU capacity crossed a threshold.** A modern workstation-class GPU
offers up to 96 GB of VRAM and nearly two terabytes per second of memory
bandwidth, at a fraction of the cost of a top-tier LLM inference system.
Working sets that used to demand a scale-out CPU cluster — or a dedicated
graph database — now fit on one device: the
[45-million-edge citation graph](/cugraph/demo-dataset) used throughout the
cuGraph documentation loads alongside its metadata tables with room to spare,
and our published TPC-H SF100 run peaks below 60 GB of device memory. And
that scale is where most enterprises actually live.[^1] They will never buy a
top-tier AI server to run database queries; what they need — a mature
developer ecosystem, and a GPU with enough VRAM to hold the working set at a
price they can justify — only recently came to exist at once. At that scale
the interesting question is no longer "how do we shard this across GPUs?"
but "why does the query engine still route everything through the CPU?"

**DataFusion made the engine extensible at the right layer.** DataFusion is
not just a query engine; it is a query engine *library*, with a public,
composable physical-plan representation. A GPU backend does not need to
reimplement SQL parsing, logical optimization, catalogs, or the DataFrame
API — it can accept DataFusion's physical plan and select a whole relational
candidate when its native contract is met.

This layer is also where the memory math above actually holds up. An
enterprise lakehouse can run to terabytes or beyond — past any single card's
VRAM — but an ad-hoc query rarely needs more than a few columns and the rows
that survive a selective filter: a date range, one tenant, one region. The
bytes that must reach GPU memory are usually a small fraction of the
table[^2] — *if* the engine can figure out which fraction before reading
anything. That is exactly what a mature query-engine library does:
projection pruning drops the unread columns, predicate pushdown and
Parquet/Iceberg statistics skip the row groups and files that cannot match,
and only the surviving bytes are handed to the GPU. Without
that planning, "load it onto the GPU" means loading a terabyte you mostly
throw away; with it, tens of gigabytes — the difference between fitting on
one card and not.

## What Nexus is

DataFusion Nexus is a physical optimizer rule plus a native execution engine —
not a replacement for DataFusion, and not a universal GPU port of it. When a
query is planned, Nexus evaluates the exact DataFusion physical candidate for
scans, filters, projections, joins, aggregations, sorts, and window functions
over the common relational types, then lowers selected candidates into its own
cuDF plan. Every decision is local and reported: an unsupported candidate, or
one not selected by an enabled cost rule, leaves the executable DataFusion
baseline intact. A caller that needs a final plan without DataFusion CPU
execution expresses that as a requirement on the completed physical plan, not as a different candidate
policy. Iceberg follows the same model: one unified scan owns both its native
facts and its executable CPU delegate.

Around that core, five design decisions define the project.

### A library that lives inside your backend

Nexus follows DataFusion's own philosophy: it is a library before it is a
server. Embedded in a Rust process, the entire engine — DataFusion planning,
the GPU executor, the Iceberg readers — runs inside your service as a single
process. There is a Flight SQL server for remote clients, but the embedded
path is the one we consider primary, because it removes the tax every
"database over there" architecture pays: query results do not cross a
network, do not pass through a driver, and are not serialized and
deserialized between your query engine and your application logic. A handler
that returns a large result set hands you Arrow batches in your own address
space; if you chase zero-copy all the way down, nothing in the design stops
you from serving memory-mapped data.

Being a Rust library matters beyond raw copies. High-performance backends are
increasingly written in Rust on Tokio, and Nexus drops into that ecosystem as
a crate: your axum or tonic service can own a session, register tables, and
`await` query streams like any other async source. (One honest caveat: the
GPU work itself is not event-driven — device execution occupies real
threads — so Nexus integrates with the async runtime rather than pretending
kernel launches are `async` all the way down.) And because it is your
process, deep customization needs no plugin API or fork: register your own
DataFusion table providers and UDFs next to the GPU path, intercept planning
reports, or wire admission decisions into your service's own backpressure.

### GPU capabilities compose in SQL, and data stays on the GPU

The lesson of the ETL detour is that fragmentation — of languages, processes,
and copies — is the real cost. So in Nexus, GPU capabilities beyond relational
SQL are exposed *inside* SQL, as table functions. A cuGraph traversal is a
relation you can join, filter, and aggregate like any other:

```sql
CREATE VIEW lstm_target AS
SELECT paper_id AS vertex FROM papers WHERE title = 'Long short-term memory';

SELECT b.path_index, b.distance, p.year, p.title
FROM cugraph_bfs('citation_edges', 2896457183, 'src', 'dst', NULL,
                 '{"output_mode":"path",
                   "target_vertices_table":"lstm_target",
                   "target_vertex_col":"vertex"}') b
JOIN papers p ON p.paper_id = b.vertex
ORDER BY b.path_index;
```

This query finds the shortest citation path from BERT (2018) back to the LSTM
paper (1997) over a 45-million-edge citation network. Both directions of the
composition matter. Upstream, SQL defines the graph the GPU sees: any view
over an edge-shaped relation can be passed to a `cugraph_*` function, so "run
PageRank over only the 2015–2020 subgraph" is a `CREATE VIEW` away, not a new
export pipeline. The view is nothing more than a named query — no data moves
when it is created; the rows behind it are read into GPU memory when a query
that uses it runs, and freed when that query completes. Downstream, the
algorithm's output is ordinary rows, enriched by joining back to the `papers`
table in the same statement — no second process, no intermediate files, and
intermediate data stays on the device between the relational and graph
stages.

Twenty-three cuGraph algorithms — BFS, SSSP, PageRank, Louvain, betweenness,
connected components, and more — are
[documented and callable this way today](/cugraph/algorithms), each with
runnable examples. The same pattern is how the surface will grow: `cuvs_*`
table functions for GPU vector search are in progress, on exactly the same
contract.

### Lakehouse-native, not sidecar

Enterprises are converging on open table formats precisely to stop copying
data between engines. A GPU engine that requires its own ingestion format
would recreate the problem it set out to solve. Nexus therefore reads Iceberg
tables from REST and AWS Glue catalogs directly: table metadata drives file
selection at plan time, predicates are pushed into the Iceberg scan, and
KvikIO reads the surviving Parquet bytes from S3-compatible object storage
into GPU memory. The table your other engines query is the same table the GPU
reads — no export, no second copy to keep in sync.

Being a good lakehouse citizen also means a strict contract about what Nexus
will *never* do to your data: it does not write it. There is no INSERT,
UPDATE, DELETE, or COPY; no table Nexus exposes can be written into; and
nothing ever commits back to Iceberg or touches the files in storage. DDL
exists, but it is session-scoped metadata, not data: `CREATE EXTERNAL TABLE`
registers a pointer to Parquet files that already exist, and `CREATE VIEW`
saves a query definition — both live in an in-memory workspace overlay that
keeps the source catalogs read-only and disappears with the process. The data
behind a view is read into GPU memory only when a query that references it
executes, and is freed when that query completes. The deliberate exceptions
are the obvious ones: results are returned to the caller as host Arrow
batches, the opt-in source cache keeps scan bytes on local disk to accelerate
repeat reads, and a `CREATE TABLE ... AS` snapshot in the local workspace is
held in host memory for the session. Your lakehouse remains the system of
record; Nexus is a lens over it, not a second writer to it.

The same native reader also serves plain Parquet on local disk, registered
with ordinary `CREATE EXTERNAL TABLE` statements. That path matters more than
it sounds: when latency is the constraint, local NVMe is the fastest way to
feed the GPU, and it needs no catalog at all. One honesty note on I/O either
way: on hosts without GPUDirect Storage — which includes the cloud instances
we develop and benchmark on — KvikIO runs in its compatibility mode, staging
reads through a host-memory bounce buffer rather than DMA-ing straight into
the device. The programming model is the same; the "storage to GPU" arrow in
the diagram hides a hop through host RAM.

### Explainable planning, structured errors

A hybrid CPU/GPU engine has a failure mode all its own: a query runs on a
different execution domain than its owner expected and nobody knows why. We
consider this a contract problem, not a logging problem. Nexus records both
candidate selection evidence and the disposition of the exact final plan. You
can ask the question before running anything:

```sql
SELECT row_kind, gpu_path, candidate_shape,
       reason_code, remedy_code, remedy, coverage_json
FROM nexus_explain_coverage('SELECT l_returnflag, sum(l_quantity)
                             FROM lineitem GROUP BY l_returnflag');
```

The summary and candidate rows come back as columns and JSON, not prose:
`gpu_path` is `native`, `partial_native`, `cpu`, or `rejected`, while each
non-selected candidate carries a stable reason and actionable remedy. The
summary JSON retains the complete evidence; Flight SQL users can request the
same projection with `EXPLAIN GPU <query>`. A capability miss is not itself a
planning error. Errors cross the DataFusion boundary the same way — as typed
values with stable identity rather than log text.

This matters for the engineers operating the system, and it matters just as
much for the agents increasingly sitting between users and data. An agent
working against a Nexus session does not have to guess at the engine's
behavior: it can call `gpu_describe_function(...)` to read an execution
function's descriptor, `gpu_validate_call(...)` with a versioned envelope to
statically check named relations and options before spending GPU time on it,
and `nexus_explain_coverage(...)` to learn whether a query will hit the fast
path. As agents become a primary consumer of data infrastructure — issuing
more queries, and more varied ones, than the dashboards before them —
machine-readable contracts like these are how an engine earns a place under
them.

### GPU memory as a planned resource

GPU memory is the scarcest resource in the system, and "try it and catch the
OOM" is not a serving strategy. Integrating at the physical-plan layer gives
Nexus a better option: because planning already knows which columns, files,
and Parquet row groups a scan will touch, the engine can estimate the size of
a query's reads *before issuing them*, and use that estimate for admission
and scheduling. In the opt-in bounded mode, each query is planned against an
explicit device-memory budget: shapes without a provable memory contract are
rejected at admission with a structured reason, admitted queries receive
their reservation before execution, and runtime pressure is absorbed by
bounded chunk-split retries. A query that exhausts its retries fails with
structured facts about what it needed — not an opaque CUDA error that takes
the process down.

## Where it stands today

Correctness first: Nexus plans and executes the full TPC-H and TPC-DS suites
at scale factor 100 on a single GPU, with results matching CPU DataFusion
query for query. The [benchmark pages](/reference/benchmarks) publish the
per-query timings and full harness configuration for anyone who wants them.
We deliberately do not lead this post with those speedup numbers, because we
do not think they are the reason to adopt Nexus: the suites run against local
Parquet on one card, and an organization whose relational workload is large
enough to hurt on a modern CPU engine usually has far more data than one
GPU's memory — a scale Nexus, with no cross-GPU execution, does not chase.

The workload where a 96 GB card changes daily practice is graph. Traditional
query engines have no native graph search — a multi-hop traversal becomes a
tower of self-joins or a recursive CTE that falls over at depth — so
organizations that need one set up a separate graph database and feed it
with, once again, an ETL pipeline and a second copy of the data. That is
exactly the detour this project exists to remove: with cuGraph behind SQL
table functions, BFS, shortest paths, PageRank, community detection, and the
rest run on the GPU over graphs with tens of millions of edges, on the same
tables the rest of the query touches. We have not yet published an
end-to-end graph benchmark — one that measures the full query wall time,
graph construction and SQL enrichment included, at enterprise graph scale
against both a CPU engine and a dedicated graph database. That suite is in
preparation and will get its own post.

{/*
  TODO(benchmark): publish the end-to-end graph benchmark and link it here.
  Requirements settled on for that post:
  - end-to-end wall time (SQL in → rows out), including GPU graph build and
    the enrichment join — not algorithm-kernel time alone;
  - graphs at representative scale (tens of millions of edges or more, e.g.
    the 45M-edge citation network or the BloodHound-style AD/Entra graphs);
  - baselines: a CPU SQL engine on the same queries, and a dedicated graph
    database for the traversal portion;
  - note why the official nx-cugraph benchmarks
    (https://docs.rapids.ai/api/cugraph/stable/nx_cugraph/benchmarks/) are
    not sufficient here: datasets too small, and no end-to-end timing.
*/}

In the spirit of setting expectations honestly, the current limitations are
just as concrete:

- **A subset of Arrow types.** Native execution covers the common relational
  types — booleans, integers, floats, `Decimal128`, strings, dates,
  timestamps, and durations. Nested and binary types and most list shapes
  stay on the CPU path; on Iceberg scans, an unsupported projected type fails
  planning instead.
- **Single-GPU queries.** Independent queries can be placed on different
  devices, but one query does not exchange data across GPUs.
- **No disk spill.** A query's working set must fit the device memory budget;
  Nexus does not spill intermediate state to host memory or disk (there is no
  integration with out-of-core executors such as
  [cuCascade](https://github.com/NVIDIA/cuCascade)). Bounded admission and
  chunk-split retries manage pressure within the budget; they do not lift it.
- **No GPUDirect Storage in our tested environments.** KvikIO currently runs
  in compatibility mode on the instances we validate on, so reads — local
  NVMe or S3 — stage through a host-memory bounce buffer on the way to the
  device.
- **Parquet only**, and Iceberg merge-on-read support is partial: position
  deletes are applied natively (not yet on a fast path), while equality
  deletes and delete vectors are unsupported. The unified Iceberg scan retains
  its executable DataFusion CPU delegate when native acceleration is not
  selected.
- **Read-only, by design.** There is no INSERT, UPDATE, DELETE, or COPY, and
  no path that writes to Iceberg or to storage. If your workflow needs to
  persist query output back to the lakehouse, that is another engine's job.
- **SQL in, Arrow out.** Queries enter as SQL text through an embedded
  DataFusion session or Flight SQL; there is no Substrait plan input.

The [documentation](/) states each boundary precisely, and
`nexus_explain_coverage` will tell you where any specific query lands.

## Getting started

Nexus embeds as a Rust library in a DataFusion session, or runs as a Flight
SQL server for remote clients. The [integration guide](/integration)
covers both paths, the [cuGraph SQL API](/cugraph) documents every graph
function with runnable examples over a public citation-network dataset, and
the [build guide](/build) covers source builds and the Docker image.

Once the server is up, here is the experiment we would most like you to try —
and the one we think the design rewards. Point an agent at your own lakehouse
and let it wander. The [twenty-three cuGraph algorithms](/cugraph/algorithms)
now callable from SQL unlock graph analyses that a CPU query engine could never 
perform directly. Examples include identifying who sits on the most shortest 
paths between two business units, finding which accounts form a tightly-knit 
community, and measuring how far a compromised identity can reach in six hops. On a CPU workload those are the
questions you *can't* ask directly — they become a graph database procurement, an ETL
pipeline, and a quarter of glue code, so they quietly fall off the roadmap.
Behind a `cugraph_*` table function, over the tables you already have, they
are a `SELECT`. The machine-readable contracts described above are what make
this safe to hand to an agent: it can read an execution function's descriptor with
`gpu_describe_function(...)`, statically check a versioned call envelope with
`gpu_validate_call(...)` before spending GPU time, and ask
`nexus_explain_coverage(...)` whether the query it just wrote lands on the
fast path — so the exploration converges on queries that actually run on the
GPU rather than guessing.

When that wandering surfaces a query worth keeping, promote it. Register the
tables it touches, pin the SQL as a named workload on the Flight SQL server,
and run it against the CPU baseline on the same data. This is the loop the
whole architecture is built for: the interesting query and the production
workload are the same SQL, over the same lakehouse table, with no export step
between them. And it is the answer we are most curious about — not whether the
TPC-H numbers hold, but whether, turned loose on *your* graph at *your* scale,
Nexus lets you ask a question you had written off as too expensive to ask. If
you run that experiment, tell us what you found: the shape of the query, the
wall time against your old path, and whether it changed what you were willing
to ask. That is the feedback we are building toward.

GPU-accelerated SQL is an active space, and deservedly so. What we have not
seen elsewhere, and what Nexus is really about, is putting the *whole* GPU
data stack — relational execution today, graph algorithms today, vector
search next — behind one SQL surface, over the lakehouse and Parquet data you
already have, embeddable in the service you already run. If this is the kind
of system you want, we would love for you to try Nexus, file issues, and tell
us what breaks. The
[GitHub repository](https://github.com/DataFusion-Nexus/datafusion-nexus) is
the front door.

[^1]: Two independent looks at real fleets agree on the order of magnitude.
    Analyzing BigQuery customers, Jordan Tigani reports that the vast
    majority store less than a terabyte in total
    ([*Big Data is Dead*](https://motherduck.com/blog/big-data-is-dead/),
    MotherDuck, 2023). In the Amazon Redshift fleet, most tables hold fewer
    than a million rows, and 98% hold fewer than a billion (van Renen et
    al., [*Why TPC Is Not Enough: An Analysis of the Amazon Redshift
    Fleet*](https://www.vldb.org/pvldb/vol17/p3694-saxena.pdf), PVLDB
    17(11), 2024, Table 8).

[^2]: The same two studies quantify this. Among paying BigQuery customers,
    90% of queries processed less than 100 MB of data (Tigani, 2023). In
    the Redshift fleet, the median query uses 0.6 GB of memory and the
    99th-percentile query 21.4 GB (van Renen et al., 2024, Table 6) —
    comfortably inside one card's VRAM.
