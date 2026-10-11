# Citation Network scale demo

The Citation Network dataset uses the complete DBLP / AMiner Citation Network V12 snapshot:
4,894,081 papers and 45,564,149 directed citation edges touching 4,146,772 distinct papers.
No dangling citations or self-loops exist in the V12 source. Acquisition and conversion
commands live in the [dataset README](../../README.md).

## Graph scopes and analytical queries

The demo queries evaluate different graph scales derived from the same base corpus.
Algorithm input sizes are distinct from total corpus size:

| Graph / Subgraph | Source SQL View / Selection | Directed Edges | Active Vertices | Analytical Purpose |
|---|---|---:|---:|---|
| Full citation graph (`citation_edges`) | Base table | 45,564,149 | 4,146,772 | Global influence ranking via PageRank across the entire corpus |
| Pre-2000 subgraph (`edges_pre2000`) | Both endpoint papers dated 1901–2000 | 3,431,911 | 518,777 | Historical influence re-ranking of early literature |
| AI subgraph (`ai_edges`) | Year >= 2010, 10 AI `primary_fos` categories | 164,368 | 38,054 | Louvain community detection correlated with catalog research fields |
| Attention ego graph (`attention_ego_edges`) | *Attention Is All You Need*, 22 references, 110 top citers | 399 | 133 | ForceAtlas2 2D projection and visual community structure |
| NLP bibliography graph (`bibliography_edges`) | Top 12 NLP papers (2015–2020) by citations | 452 | 371 | Pairwise Sørensen bibliography overlap |

## Server and execution modes

The earlier website configuration specified `execution_mode = "functions_only"`, routing
relational filters, joins, aggregates, and window calculations back to DataFusion on CPU
while running only cuGraph calls on GPU. That mode generated between 6 and 16 host
boundaries per query and failed to execute Sørensen similarity across host-staged side inputs.

Configuring `execution_mode = "native_required"` ensures that the entire statement,
including relational preparation before cuGraph and enrichment joins and window rankings
after cuGraph, executes in native GPU fragments. Ordering propagation across renaming
projections and stacked window tasks was extended in `algeon-query-planning` so that
analytical statements with post-graph window ranking (such as Louvain community ranking)
compile and execute entirely within native GPU pipelines.

Build the release server:

```bash
cargo build --workspace --all-features --release --bin algeon_server

ALGEON_SERVER_CONFIG_FILE="$PWD/fixture/demo/citation_network/scripts/scale/server.toml" \
ALGEON_SERVER_BIND=127.0.0.1:50065 \
ALGEON_SERVER_CUGRAPH_ENABLED=true \
flock -x /tmp/cudf-gpu.lock target/release/algeon_server
```

Run the scale suite using the Python client runner:

```bash
python3 scripts/dev/citation_run.py --uri grpc://127.0.0.1:50065 --register --repeats 5 \
  --report-dir target/citation-scale/report --results-dir target/citation-scale/results \
  fixture/demo/citation_network/scripts/scale/01_pagerank_top.sql \
  fixture/demo/citation_network/scripts/scale/02_pagerank_paper_rank.sql \
  fixture/demo/citation_network/scripts/scale/03_pagerank_pre2000.sql \
  fixture/demo/citation_network/scripts/scale/04_louvain_ai_fields.sql \
  fixture/demo/citation_network/scripts/scale/05_force_atlas2_transformer.sql \
  fixture/demo/citation_network/scripts/scale/06_sorensen_nlp.sql
```

## Performance measurements

Timings were collected on an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB device budget,
90 GiB query grant, 1 GiB backend reserve). All five runs reported no host boundaries, allocation
retries, or CPU fallbacks. Medians below exclude the first run:

| Query | Rows | 1st Run (s) | Median (s) | Peak Memory (GiB) | Native Fragments | Host Boundaries |
|---|---:|---:|---:|---:|---:|---:|
| `01_pagerank_top.sql` | 5 | 0.167 | 0.161 | 2.068 | 2 | 0 |
| `02_pagerank_paper_rank.sql` | 1 | 0.154 | 0.152 | 2.068 | 2 | 0 |
| `03_pagerank_pre2000.sql` | 6 | 0.129 | 0.127 | 1.406 | 2 | 0 |
| `04_louvain_ai_fields.sql` | 15 | 0.164 | 0.166 | 1.374 | 2 | 0 |
| `05_force_atlas2_transformer.sql` | 133 | 0.991 | 0.963 | 2.377 | 3 | 0 |
| `06_sorensen_nlp.sql` | 5 | 0.253 | 0.241 | 1.346 | 4 | 0 |

In comparison, the `functions_only` mode introduced 6 to 16 host boundaries per statement and
ran between 3x and 10x slower on relational pre/post-processing, while `06_sorensen_nlp.sql`
failed due to host-staged vertex pair conversion errors. Under `native_required`, every statement
completed with identical validated rows.
