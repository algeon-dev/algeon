# Citation Network (DBLP / AMiner V12)

The cuGraph examples and the embedded-backend example query this dataset.
`website/docs/guides/datasets/citation-network.mdx` documents the tables from the SQL
side; this file covers acquisition and conversion.

## Source

`dblp.v12.json` from the DBLP / AMiner Citation Network Dataset V12 (Kaggle
`mathurinache/citation-network-dataset`): 12,520,828,317 bytes, SHA-256
`70e8e02a95e8861022372e7e73dc41a602cf36967d5a7eb928f7782b7e832339`. The file
is one JSON array with a paper object per line. Place it at
`raw/dblp.v12.json`.

## Conversion

```bash
fixture/fixture.sh graph dblp-ingest --overwrite
# same as: python3 fixture/demo/citation_network/scripts/prepare.py --overwrite
```

The defaults read `raw/dblp.v12.json` and write `v1/`. The full conversion
streams the JSON once and takes 17 to 19 minutes and 7.8 GB of RAM on a
32-core host with the output on local NVMe. Only `pyarrow` is required.
`--max-papers N` writes a prefix sample under `v1_sample_papersN/`, and
`--edge-slice-count K` adds `K` source-hash graph slices under `v1/slices/`.

```
v1/
  manifest.json                 counts, per-table Parquet layout, SQL registration
  source.json                   source path, size, SHA-256
  parquet/
    edges_by_src.parquet        45,564,149 edges sorted by (src, dst, edge_id)
    edges_by_dst.parquet        the same edges sorted by (dst, src, edge_id)
    profiles_by_paper_id.parquet  4,894,081 papers sorted by paper_id
    paper_authors.parquet       14,934,850 rows sorted by (paper_id, author_ordinal)
    paper_fos.parquet           45,029,892 rows sorted by (paper_id, fos_ordinal)
```

`crates/workloads/src/graph.rs` validates this layout against `manifest.json`,
`source.json`, and the Parquet footers before any benchmark or validation run
uses it.

## Parquet layout

- Snappy, format 2.6, statistics and page indexes on every column. Edge row
  groups hold 1,048,576 rows; paper profiles use 131,072 because abstracts
  dominate their size.
- Vertex ids are the raw AMiner Int64 paper ids, so cuGraph reads `src` and
  `dst` directly and enrichment joins use the same key.
- The two edge copies make both `src` and `dst` range filters selective on
  row-group statistics. Bloom filters cover `src`/`dst` on edges,
  `paper_id`/`venue`/`primary_fos` on profiles, and the keys of the side
  tables.
- Files declare `sorting_columns`, so DataFusion infers the scan ordering.
- Edge footers carry two contracts:
  - `algeon.graph.simple_directed_edge_topology.v1` certifies non-null
    `src`/`dst` and no duplicate directed pair. The writer checks every
    adjacent pair in sorted order.
  - `algeon.graph.vertex_upper_bound.v1` is the paper count. It is emitted only
    for the complete V12 rowset, and the writer fails if any citation target is
    not a paper in the source.

  V12 has no dangling citations and no self-loops. Its 45.6M edges touch
  4,146,772 distinct papers.
