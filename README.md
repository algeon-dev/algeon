<p align="center">
  <img src="website/static/img/algeon-logo.png" alt="Algeon" width="420">
  <br>
  <em>Compose algorithms with SQL.</em><br><br>
  <a href="https://algeon.dev">Website</a> |
  <a href="https://algeon.dev/build">Build</a> |
  <a href="https://docs.rs/algeon-datafusion">API Docs</a>
</p>

---

Algeon (`/ˈæl.dʒi.ən/`, "AL-jee-on") is a GPU-accelerated query engine built on
[Apache DataFusion](https://datafusion.apache.org/). It brings NVIDIA's
[cuGraph](https://docs.rapids.ai/api/cugraph/stable/),
[cuVS](https://docs.rapids.ai/api/cuvs/stable/) and
[cuML](https://docs.rapids.ai/api/cuml/stable/) algorithms into SQL as table
functions, alongside [cuDF](https://docs.rapids.ai/api/cudf/stable/) relational
execution. Subqueries supply algorithm inputs; the results feed joins,
aggregations or another algorithm within the same SQL statement.

For background and a worked SQL example, see
[Introducing Algeon](https://algeon.dev/blog/introducing-algeon).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="website/static/img/intro-flow-dark.png">
    <img src="website/static/img/intro-flow-light.png" width="100%" alt="A SQL working set passes as a subquery to a GPU UDTF, which returns rows that the same statement joins, filters, aggregates or passes to the next UDTF.">
  </picture>
</p>

## Where Algeon fits

- **Ad-hoc graph, vector and ML pipelines on lakehouse data.** One SQL query
  selects the input, runs the algorithm and joins the result back. A new
  pipeline is a new query, with no export scripts.
  [Examples](https://algeon.dev/guides/datasets).
- **Analysts and AI agents testing hypotheses.** A question no dashboard
  tracks needs many follow-up checks with other seeds, filters or controls.
  The [MerRec case study](https://algeon.dev/guides/datasets/merrec#audience-funnel)
  compares graph affinity, vector similarity and model scores for selecting
  a campaign audience.
- **Backends for agents and data services.** The
  [Flight SQL server](https://algeon.dev/integration/flight-sql-server)
  queues concurrent queries, lists its functions in the catalog and returns
  structured errors a program can act on.
- **Join- and aggregation-heavy SQL over Parquet.** Supported plan fragments
  run on cuDF. See the [benchmarks](https://algeon.dev/reference/benchmarks).
- **Engines built on DataFusion.** Algeon adds GPU execution and the
  algorithm functions to an existing
  [DataFusion session](https://algeon.dev/integration/datafusion-session);
  operators it does not support stay on the CPU.

**Not a fit yet:** working sets larger than one GPU's memory, sources other
than Parquet, and long-lived vector indexes or models. See
[scope and non-goals](https://algeon.dev/#scope).
Inputs, intermediate results and algorithm scratch space count toward the
query's GPU memory limit.
Lakehouse catalog integrations (WIP).

## Getting started

Algeon requires Linux, a supported NVIDIA GPU, the CUDA toolkit and native
RAPIDS libraries. Check the [prerequisites](https://algeon.dev/build/prerequisites)
before building.

- **Run a SQL server:** follow the
  [Flight SQL server build steps](https://algeon.dev/build#run-as-a-flight-sql-server),
  then submit a query with the
  [client quickstart](https://algeon.dev/integration/flight-sql-server#quickstart).
- **Embed in Rust:** follow the
  [DataFusion session guide](https://algeon.dev/integration/datafusion-session)
  for dependencies, native library setup and a query example.

## An algorithm in SQL

Explore related research beyond BERT's direct references:

```sql
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

| year | title | ppr |
|---|---|---|
| 1983 | A Maximum Likelihood Approach to Continuous Speech Recognition | 0.003127 |
| 2003 | A neural probabilistic language model | 0.002984 |
| 1997 | Long short-term memory | 0.002392 |
| 2014 | Adam: A Method for Stochastic Optimization | 0.002358 |
| 1990 | A statistical approach to machine translation | 0.002344 |

Personalized PageRank runs over
[45.6 million citation edges](https://algeon.dev/guides/datasets/citation-network)
on the GPU.
SQL joins the scores to paper titles and excludes BERT and its direct references.
[Run this example](https://algeon.dev/integration/flight-sql-server#start-the-citation-demo-server).

Within supported GPU plan fragments, intermediate relations stay in device
memory between SQL and algorithm stages. An audience campaign analysis
over the [MerRec dataset](https://algeon.dev/guides/datasets/merrec#audience-funnel)
computes seed affinity with `cugraph_personalized_pagerank`, identifies
taste lookalikes with `cuvs_brute_force_knn`, scores candidate shoppers with a
pretrained forest model through `cuml_forest_predict`, and aggregates conversion
with standard SQL. [Explore the examples](https://algeon.dev/guides/datasets).

## Discover signals, ask new questions

Exploratory analysis often starts before the right question is known. Graph
relationships, vector similarity and model scores can expose signals that
suggest what to investigate next. An analyst or agent can follow those signals
by changing seeds, comparing cohorts and testing controls in SQL.

Algeon supports this search for "unknown unknowns" by composing GPU algorithms
with SQL that inspects and tests their results. In the
[Barbie investigation](https://algeon.dev/guides/datasets/merrec#barbie), changes
in brand affinity prompted checks of listing names and alternative seed brands.
Those checks supported some explanations and left other changes unexplained.

## In-process execution

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="website/static/img/intro-architecture-dark.png">
    <img src="website/static/img/intro-architecture-light.png" width="100%" alt="A query enters through SQL, DataFrame or Flight SQL. Algeon's optimizer rule picks whole relational candidates for GPU execution, unselected candidates stay on the DataFusion CPU, and results return as Arrow batches. GPU execution reads Parquet from local disk or S3-compatible object storage through KvikIO.">
  </picture>
</p>

Queries enter through SQL, DataFusion DataFrames, or Arrow Flight SQL.
Algeon's DataFusion physical optimizer rule replaces supported plan candidates
with in-process GPU execution, returning results as standard Arrow batches.
Unselected candidates execute on the DataFusion CPU path. The
[design guide](https://algeon.dev/design) follows a query from SQL to GPU
columns and back.

## Documentation

- [Demo datasets](https://algeon.dev/guides/datasets): every example runs
  over a public dataset and is published with its SQL, result rows and
  query IDs.
- [Graph](https://algeon.dev/cugraph), [vector](https://algeon.dev/cuvs) and
  [machine learning](https://algeon.dev/cuml): the installed UDTFs of each
  family, their contracts and examples.
- [Build & Test](https://algeon.dev/build): prerequisites, the native builds,
  the checks and a Docker image.
- [Integration](https://algeon.dev/integration): an embedded
  DataFusion session or the Arrow Flight SQL server, configuration and error
  handling.
- [Design](https://algeon.dev/design): module boundaries, native lowering,
  memory admission and error ownership.

## Status

> [!WARNING]
> Algeon is experimental. APIs and query behavior may change incompatibly.
> Stability and result verification remain incomplete. Compare outputs
> against trusted reference results. The
> [roadmap](https://algeon.dev/reference/roadmap) lists what changes next.

GitHub Actions runs only the checks that need no GPU. GPU validation runs
locally because hosted runners have no NVIDIA GPU, and
[hardware sponsorships or dedicated GPU runners](.github/workflows/README.md)
are welcome.

## License

Algeon is licensed under the [Apache License 2.0](LICENSE). Third-party
content with other terms is listed in [NOTICE](NOTICE); the saved MerRec
results under `website/static/data/merrec-*.json` carry the dataset's
[CC BY-NC 4.0](website/static/data/LICENSE-MERREC.md) terms.
[SECURITY.md](SECURITY.md) explains how to report a vulnerability.
