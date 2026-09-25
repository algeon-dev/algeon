# Fixtures

Run fixture commands from the monorepo root. Native consumers resolve the
root-owned component sources and `target/native/*` prefixes; fixture data never
requires a separate Rust checkout.

Fixtures are organized by workload. Generated and host-provisioned data is
gitignored; the small cuGraph parity corpus in `graph/cugraph/` is versioned.

```
fixture/
  fixture.sh                         # public fixture command
  tpc-h/
    generate.py                      # TPC-H generator
    data/sf<N>/parquet/              # local, gitignored benchmark data
    models/                          # local, gitignored forest models
  tpc-ds/
    generate.py                      # TPC-DS generator
    data/sf<N>/parquet/              # local, gitignored benchmark data
  tpc/parquet_layout.py              # shared TPC layout/verification logic
  clickbench/data/                   # local, gitignored ClickBench layouts
  ember2024/
    prepare.py                       # models, download, vectorize, Parquet export
    prepare.sh                       # venv-resolving entrypoint
    parquet/<split>.parquet          # local, gitignored id + f0..f2567 + label export
  graph/
    cugraph/                         # cuGraph parity corpus and oracle JSON
    demo/                            # local Citation Network Parquet
    dblp.v12.json                    # local source JSON
    dblp_ingest.py                   # Citation Network generator
    prepare.sh                       # cuGraph corpus preparation entrypoint
    rmat_generate.py                 # deterministic synthetic R-MAT generator
    synthetic/                       # local R-MAT Parquet fixtures
  iceberg-local/                     # local REST, RustFS, and Moto stack
```

## TPC-H and TPC-DS

`tpc-h/generate.py` and `tpc-ds/generate.py` own generation, layout
optimization, and Parquet metadata verification. They use `~/code/tpchgen-rs`
by default, write Snappy Parquet, use the release `tpcgen-cli` build, and make
the optimized layout the default:

```bash
# Generate optimized SF1 and SF10 into the in-repo workload directories.
fixture/fixture.sh tpch generate --scale 1 --scale 10 --force
fixture/fixture.sh tpcds generate --scale 1 --scale 10 --force

# Generate raw and optimized layouts on any chosen fast local filesystem.
export ALGEON_FIXTURE_ROOT=/mnt/fast/algeon-fixtures
fixture/tpc-h/generate.py --scale 1 --scale 10 --variant both --out-root "$ALGEON_FIXTURE_ROOT/tpch" --force
fixture/tpc-ds/generate.py --scale 1 --scale 10 --variant both --out-root "$ALGEON_FIXTURE_ROOT/tpcds" --force
```

Both scripts drive `tpcgen-cli`, then rewrite the final files as one Parquet
file per table, sorted by workload-specific pruning keys with row-group
statistics and Snappy compression. DuckDB is the default external-memory
optimizer; pass `--optimize-engine pyarrow` for the older in-memory path.
Verification checks the expected tables, SF1/SF10/SF100 row counts, row groups,
compression, and monotonic row-group statistics. The files deliberately omit
Parquet footer sorting-column metadata: physical ordering is for pruning
statistics, not an output-ordering promise to DataFusion.

### Lineitem forest model

`forest_predict` and `scripts/dev/bench_forest_stage0.py` score
`tpc-h/models/lineitem_xgboost_500t_d8.json` (override with
`ALGEON_FOREST_LINEITEM_MODEL`). No versioned script trains it; the stage 0
model is an XGBoost regressor trained on all 59,986,052 rows of
`tpc-h/data/sf10/parquet/lineitem.parquet`, with features `l_orderkey`,
`l_partkey`, `l_suppkey`, `l_linenumber`, `l_quantity`, `l_discount`, and
`l_tax` (in that order, each cast to Float32) and target `l_extendedprice`:

```python
import numpy as np, pyarrow.parquet as pq, xgboost as xgb

features = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_discount", "l_tax"]
table = pq.read_table("fixture/tpc-h/data/sf10/parquet/lineitem.parquet", columns=features + ["l_extendedprice"])
x = np.column_stack([table.column(c).to_numpy().astype(np.float32) for c in features])
y = table.column("l_extendedprice").to_numpy().astype(np.float32)
model = xgb.XGBRegressor(n_estimators=500, max_depth=8, tree_method="hist", device="cuda")
model.fit(x, y)
model.save_model("fixture/tpc-h/models/lineitem_xgboost_500t_d8.json")
```

## ClickBench

Fetch the official layouts into its own fixture directory:

```bash
fixture/fixture.sh clickbench fetch --layout both --out ./fixture/clickbench/data
```

`--layout single` writes `hits.parquet`; `--layout partitioned` writes
`partitioned/hits_0.parquet` through `hits_99.parquet`.

## EMBER2024

`ember2024/prepare.py` downloads the 14 pretrained EMBER2024 LightGBM models
and the raw weekly JSONL archives via `thrember`, vectorizes them into
`X_*.dat` / `y_*.dat`, and exports a requested split as Parquet
(`id` + `f0..f2567` + `label`, all non-nullable) in bounded-RAM row-group
chunks. Each step is skipped if its output already exists (`--force`
overrides). See `fixture/ember2024/README.md` for the full flow, the venv
setup (`~/.venvs/forest`, no `uvx` dependency), and a documented row-count
anomaly in the upstream `train`/`test` archives.

```bash
fixture/fixture.sh ember2024 prepare --split test --split challenge
fixture/fixture.sh ember2024 prepare --split train --rows 1000000 --force
```

## Graph fixtures

`graph/dblp_ingest.py` converts the DBLP / AMiner Citation Network V12 JSON
file into the Parquet files used by the cuGraph SQL examples. The source JSON
is about 12.5 GB, so place `fixture/graph/demo` on NVMe when the repository is
on slower storage:

```bash
python3 fixture/graph/dblp_ingest.py \
  --source-json /path/to/dblp.v12.json \
  --out-root fixture/graph/demo \
  --overwrite
```

This writes `fixture/graph/demo/citation_network/v1/parquet/` with
`edges_by_src.parquet`, `edges_by_dst.parquet`,
`profiles_by_paper_id.parquet`, `paper_authors.parquet`, and
`paper_fos.parquet`. Register it with the local Iceberg stack without
rewriting it:

```bash
docker compose -f fixture/iceberg-local/docker-compose.yml up -d
fixture/fixture.sh iceberg rest load \
  --workload citation_network \
  --load-mode add-files
```

`add-files` stages the Parquet under `s3://lakehouse/staging/...`, then Iceberg
references those staged objects in place. Do not delete that staging prefix
while the fixture metadata is live.

To refresh the smaller cuGraph parity corpus and its optional Python oracle,
run `fixture/graph/prepare.sh --oracle`. It uses
`ALGEON_CUGRAPH_PYTHON` when set, otherwise the repository virtual environment
or `python3`.

`graph/rmat_generate.py` generates a deterministic synthetic R-MAT
(recursive matrix) edge-list fixture as Parquet, sized by `--scale` (vertex
count `2**scale`) and `--edge-factor` (edge count `edge_factor * 2**scale`):

```bash
fixture/fixture.sh graph rmat \
  --scale 22 --edge-factor 16 --skew skewed --seed 20260910 \
  --out-root fixture/graph/synthetic --force
```

`--skew skewed` uses the Graph500 R-MAT quadrant probabilities
(a=0.57, b=0.19, c=0.19, d=0.05); `--skew uniform` uses a flat 0.25/0.25/0.25/0.25
split (Erdos-Renyi-like, low degree skew) to isolate the effect of degree skew
in comparisons. Output goes to `<out-root>/rmat_s<scale>_e<edge_factor>_<skew>/`,
containing `edges.parquet` (`src` int64, `dst` int64, `weight` float64,
self-loops and duplicate edges retained) and a `_generation.json` manifest
recording the command line, sizes, a sha256 of the Parquet file, and degree
statistics. Generation is deterministic for identical arguments (including
`--chunk-edges`, which sets the RNG draw/row-group chunking): rerunning with
the same flags reproduces a byte-identical `edges.parquet`. `--force`
overwrites an existing output directory; without it, generation refuses to
replace one.

## Local data placement

Point each workload-local data directory at fast storage with a symlink, or
generate/download directly into it:

```bash
export ALGEON_FIXTURE_ROOT=/mnt/fast/algeon-fixtures
ln -s "$ALGEON_FIXTURE_ROOT/tpch"       fixture/tpc-h/data
ln -s "$ALGEON_FIXTURE_ROOT/tpcds"      fixture/tpc-ds/data
ln -s "$ALGEON_FIXTURE_ROOT/clickbench" fixture/clickbench/data
ln -s "$ALGEON_FIXTURE_ROOT/demo"       fixture/graph/demo
ln -s "$ALGEON_FIXTURE_ROOT/synthetic"  fixture/graph/synthetic
```

For an explicit external fixture, set `ALGEON_PARQUET_DIR` to its Parquet root.
It is intentionally workload-neutral because one process executes one
workload at a time; the workload validates the expected layout. The in-repo
directories above remain the implicit local defaults.
