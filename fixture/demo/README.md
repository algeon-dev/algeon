# Demo datasets

Algeon's examples, documentation, and demo benchmarks query these
datasets. They are real public data, too large to version, and kept apart from
the unit-test corpora (`fixture/graph/cugraph/`, `fixture/tests/`), the
benchmark generators (`fixture/tpc-h/`, `fixture/tpc-ds/`, `fixture/clickbench/`,
`fixture/graph/rmat_generate.py`), and the E2E stacks (`fixture/iceberg-local/`).

| Dataset | Upstream | Parquet tables | Converter | Consumers |
| :--- | :--- | :--- | :--- | :--- |
| [`citation_network/`](citation_network/README.md) | DBLP / AMiner V12 JSON, 12 GB | 2 edge layouts (45.6M edges), papers, authors, fields of study; 4.6 GB | `scripts/prepare.py` | cuGraph SQL docs and examples, embedded-backend example, `crates/workloads` graph discovery |
| [`amazon_reviews_2023/`](amazon_reviews_2023/README.md) | Amazon Reviews 2023 Electronics JSONL, 27 GB | `reviews` (43.9M), `products` (1.6M); 11 GB | `scripts/prepare.py` | semantic catalog search, review clustering and UMAP; relational joins and aggregation |
| [`ember2024/`](ember2024/README.md) | EMBER2024 archives and weekly JSONL, 24 GB compressed | `test`, `challenge`, `train` feature splits; 5.2 GB | `scripts/prepare.py` via `prepare.sh` | `cuml_forest_predict` demos, `forest_predict` bench |
| [`kuairand/`](kuairand/README.md) | KuaiRand recommendation logs (Pure / 1K / 27K), 194 MB to 46 GB | `interactions` (2.6M to 323M), `users` (27K), `videos` (7.6K to 32M), `video_stats`; 81 MB to 40 GB | `scripts/prepare.py` | sequential recommendation, unbiased offline evaluation, multi-table join and group-by demos |
| [`merrec/`](merrec/README.md) | MerRec (Mercari C2C recommendation), 6 monthly partitions | `events`, `items`; 80M items, 1B events | `scripts/prepare.py` | sequential recommendation, C2C e-commerce interaction graph, relational join and aggregation demos |

## Layout

```
fixture/demo/<dataset>/
  README.md, *.sh         versioned
  scripts/                versioned: acquisition notes and the converter
  raw/                    upstream downloads and upstream tool output
  parquet/                converted Parquet read by Algeon
  manifest.json           converter provenance, counts, per-table layout
```

The Citation Network keeps its established `v1/` root (`v1/manifest.json`,
`v1/source.json`, `v1/parquet/`), which `crates/workloads/src/graph.rs`
validates. EMBER2024 has no manifest, and `models/` holds its pretrained
LightGBM models next to `parquet/`. The root `.gitignore` ignores everything
under a dataset directory except `README.md`, `scripts/`, and `*.sh`.

Put the generated output on local NVMe with a symlink when the repository
sits on network storage; the converters write through the link. On the
development host `raw/` stays on the EBS root volume and every generated path
points into the same layout under `/mnt/algeon-nvme/algeon/fixture/demo/`:

```
citation_network/v1               -> .../citation_network/v1
amazon_reviews_2023/parquet       -> .../amazon_reviews_2023/parquet
amazon_reviews_2023/manifest.json -> .../amazon_reviews_2023/manifest.json
ember2024/parquet                 -> .../ember2024/parquet
kuairand/parquet                  -> .../kuairand/parquet
kuairand/manifest.json            -> .../kuairand/manifest.json
merrec/parquet                    -> .../merrec/parquet
merrec/manifest.json              -> .../merrec/manifest.json
```

To regenerate there directly:

```bash
N=/mnt/algeon-nvme/algeon/fixture/demo
fixture/fixture.sh graph dblp-ingest --out-root "$N" --overwrite
fixture/fixture.sh amazon-reviews prepare --out-dir "$N/amazon_reviews_2023" --force
fixture/fixture.sh ember2024 prepare --split challenge --split test --force   # writes through ember2024/parquet
fixture/fixture.sh ember2024 prepare --split train --rows 1000000 --force
fixture/fixture.sh kuairand prepare --out-dir "$N/kuairand" --force
fixture/fixture.sh merrec prepare --out-dir "$N/merrec" --force
```

The instance NVMe does not survive a stop, so rerun these after one. The
`train` export stays at 1,000,000 rows: the `forest_predict` `ember_train_full`
case is sized to it, and the full 5,252,000-row split fails that case against
the current GPU grant and host staging limit.

## Shared Parquet conventions

Each converter writes Snappy-compressed, format 2.6 Parquet with explicit
Arrow schemas:

- **Column types.** Only types the native GPU engine can filter, group, and
  join on. No Struct, Map, or Dictionary Arrow types, and no timezone-aware
  timestamps.
- **Sort order.** Rows are physically sorted on the main filter key, so
  row-group min/max statistics prune: citation edges by `src` or `dst`,
  reviews by `review_time`, entity tables by their id.
- **Statistics.** Min/max statistics only where they can prune. EMBER feature
  columns and free-text columns go without, which keeps the footer that every
  scan parses small.
- **Bloom filters.** On equality-lookup keys whose values scatter across row
  groups. Libcudf consults them after statistics pruning.
- **Row groups.** 1,048,576 rows for narrow tables. Text-heavy tables (paper
  profiles, products) use 131,072, and EMBER uses 100,000 because a row has
  2,570 columns.
