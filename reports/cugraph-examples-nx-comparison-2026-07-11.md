# cuGraph SQL examples vs Pandas+NetworkX — timing comparison

- Date: 2026-07-11
- Host: single NVIDIA RTX PRO 6000 (96 GB), 124 GB RAM box
- Server: `datafusion_nexus_server` release build (`server,iceberg,cugraph`), citation demo
  recipe (`website/docs/integration/flight-sql-server.mdx`), `max_in_flight=1`,
  `native_memory_control_policy=unbounded`
- Python: venv with networkx 3.6.1, pandas 3.0.3, scipy 1.18.0, pyarrow (CPython 3.10)
- Script: `reports/cugraph-examples-nx-comparison-2026-07-11-bench.py` (one subprocess per task, 30-minute cap per task)

## Method

Each task mirrors one of the new website example workloads end-to-end over the SAME Parquet
files the server reads:

- **datafusion-nexus**: the example page's SQL executed via `arrow_cli` against the running
  Flight SQL server; two runs, the warm (second) `rows received` time is reported. Includes
  Iceberg/Parquet scan, GPU graph build, algorithm, joins back to metadata, and result transfer.
- **Pandas+NetworkX**: `pd.read_parquet` on the same files, the same subgraph filters/joins in
  pandas, `networkx` algorithm, same final top-N extraction. Wall clock and stage breakdown
  reported; peak RSS from `getrusage`.

## Results

| Workload (example page) | Graph | nexus (warm) | Pandas+NetworkX | Speedup |
|---|---|---|---|---|
| PageRank top-5 w/ titles (`cugraph_pagerank`) | full citation, 45.6M edges directed | **0.33 s** | 188.6 s (read 1.2, build 115.8, pagerank 69.5, join 1.9) | ~570× |
| Triangle count aggregate (`cugraph_triangle_count_all`) | full citation, undirected | **0.44 s** | 273.3 s (read 0.6, build 100.3, triangles 165.5, python sum/max aggregate ~6.9) | ~620× |
| Exact betweenness top-6 (`cugraph_betweenness_centrality`) | AI subgraph, 38,054 vtx / 164,368 edges | **0.38 s** | ~675 s algo, extrapolated (k=200 sample = 3.54 s; exact = 38,054 sources) + 2.4 s prep | ~1,800× (extrapolated) |
| Louvain communities + size aggregates (leiden/ecg example subgraph; the fos cross-tab join was not replicated on either side) | AI subgraph, undirected | **0.39 s** | 8.0 s (read 3.0, filter 1.1, build 0.2, louvain 3.7) | ~21× |
| MST era-backbone top-5 w/ titles (`cugraph_minimum_spanning_tree`) | AI literature all-years, 50,300 vtx / 204,472 year-gap-weighted edges | **0.47 s** | 7.3 s (read 0.9, filter 1.3, build 0.3, mst 0.6, join 4.1) | ~16× |

Peak Python RSS: 21.6 GB (pagerank), 15.7 GB (triangles), 2.7 GB (betweenness/louvain).

## Correctness cross-checks

- **Triangle count: exact match.** NetworkX returns vertices 4,146,772 / triangle-corner sum
  202,557,720 / max 143,168 — identical to the cuGraph SQL result in the example page.
- **MST: exact match.** 49,263 forest edges over 50,300 papers, identical maximum surviving
  year gap (47) and the same top bridge pair (1972 character-recognition ↔ 2019
  feature-extraction comparison).
- **Betweenness:** NetworkX k=200 sampled top-6 is the same paper set as the GPU exact top-6
  (ordering differs — sampling noise).
- **Louvain:** community COUNTS differ (NetworkX 849 vs cuGraph 1,960 on the same subgraph) —
  different implementations/refinement of the same objective; both find modular structure, and
  the example pages only rely on cuGraph's own deterministic output.
- **PageRank: top-5 sets differ.** NetworkX ranks {GA book, Shannon '49, SIFT, LIBSVM, Fuzzy
  sets} vs cuGraph {automata-theory + Shannon canon}; NX top score 4.9e-4 vs cuGraph 1.39e-3.
  Likely dangling-node (out-degree-0 sink) mass handling differences — this graph has millions
  of sinks (papers whose references are outside the corpus). Not investigated further here;
  worth a parity note if PageRank semantics on sink-heavy graphs ever matter for docs claims.

## Reading the numbers

- NetworkX cost is dominated by **Python graph construction** (100–116 s for 45.6M edges before
  the algorithm even starts) — the GPU path's entire end-to-end run finishes ~300× before
  NetworkX finishes building its dict-of-dicts.
- At small-to-medium scale the gap narrows (the 204k-edge MST row is ~16×, and a
  few-thousand-edge graph earlier in this investigation measured ~4×) — for small graphs the
  pandas route is perfectly serviceable.
- The two spectral examples (`cugraph_spectral_modularity_maximization`,
  `cugraph_balanced_cut_clustering`) and ECG/Leiden have no NetworkX equivalent
  (sklearn/python-louvain/leidenalg would be needed), so they are excluded rather than compared
  unfairly.
- 2026-07-11 update: the MST example moved from the temporary Amazon co-review dataset to the
  citation network (year-gap weights); the MST row above was re-measured on the new workload
  and the exact-match cross-check repeated.
- Caps and isolation: every Python task ran in its own subprocess with a 30-minute timeout;
  nothing hit the cap. GPU timings include Flight SQL round-trip and metadata joins.
- Post-run script cleanups (ordered top-N merge instead of unordered `isin`, explicit triangle
  aggregate stage) were applied after the timed runs; they are millisecond-scale relative to
  the reported numbers.
