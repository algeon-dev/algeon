# Fixtures

Run fixture commands from the monorepo root. Native consumers resolve the
root-owned component sources and `target/native/*` prefixes; fixture data never
requires a separate Rust checkout.

Fixtures are organized by purpose. `demo/` holds the datasets Algeon's examples
and documentation query; everything else serves unit tests (`graph/cugraph/`,
`tests/`), or benchmarks (`tpc-h/`, `tpc-ds/`, `clickbench/`,
`graph/rmat_generate.py`). Generated and host-provisioned data is gitignored; the
small cuGraph parity corpus in `graph/cugraph/` is versioned.

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
  demo/                              # demo datasets; see demo/README.md
    citation_network/                # DBLP / AMiner V12 -> cuGraph edge and paper tables
    amazon_reviews_2023/             # Amazon Reviews 2023 -> products and reviews
    ember2024/                       # EMBER2024 -> forest-inference feature splits
    kuairand/                        # KuaiRand recommendation -> interactions, users, videos, stats
    merrec/                          # MerRec C2C recommendation -> events and items
  graph/
    cugraph/                         # cuGraph parity corpus and oracle JSON
    prepare.sh                       # cuGraph corpus preparation entrypoint
    rmat_generate.py                 # deterministic synthetic R-MAT generator
    synthetic/                       # local R-MAT Parquet fixtures
  tests/                             # Python tests for the fixture scripts
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
statistics and Snappy compression. DuckDB sorts in external memory. TPC-H
files are then written by Spark (below); TPC-DS files are written by DuckDB,
or by pyarrow's older in-memory path with `--optimize-engine pyarrow`.
Each DuckDB statement runs with half of the memory available when the table's
sort starts, so a sort next to other jobs spills to temp space instead of
being stopped by the OOM killer. A table whose estimated sort working set
exceeds a quarter of the available memory is sorted in value ranges of its
first sort key, so temp space stays bounded (one global sort of SF750 lineitem
needed more than 388 GiB). The DuckDB writer concatenates the sorted ranges
into the same file as one global sort, except that each range boundary can
leave one short row group.
Verification checks the expected tables, row counts for the scales in each
generator's expected-row table, row groups, compression, and monotonic
row-group statistics, and rejects a file with more than 32,767 row groups
(the most the parquet 59 footer decoder in DataFusion 55 reads). Use DuckDB
1.5.x: 1.2.0 writes string statistics that pyarrow
cannot read, and 2.0.0 writes every column as NOT NULL where the other scales
are nullable. The files deliberately omit
Parquet footer sorting-column metadata: physical ordering is for pruning
statistics, not an output-ordering promise to DataFusion.

Each optimized directory holds a `_generation.json` manifest with `layout`,
`scale_factor` (a number), `writer` (`spark`, `duckdb` or `pyarrow`),
`compression`, and per table its `sort_keys` (empty when unsorted), field
renames and casts. The DuckDB and pyarrow writers add `row_group_bytes`; the
Spark writer adds `spark_version`, `parquet_java_version` (read from the
written files' `created_by`), `parquet_block_size`, `parquet_page_size` and
`parquet_page_row_count_limit`.

### TPC-H Spark writer

TPC-H files are written by Spark 4.0.1 (parquet-java 1.15.2) in local mode.
Every setting is the Spark or parquet-java default except the single output
file per table:

| Setting | Value |
| --- | --- |
| `parquet.block.size` (row-group target) | 128 MiB |
| `parquet.page.size` | 1 MiB |
| `parquet.page.row.count.limit` | 20,000 rows |
| Dictionary encoding | on; a dictionary past 1 MiB falls back to plain |
| Compression | Snappy |
| Data page format | v1 |
| Decimal physical type | INT64 (`spark.sql.parquet.writeLegacyFormat=false`) |
| Output files per table | 1 (`coalesce(1)`) |

The block size is parquet-java's estimate of its write buffer, so a written
row group is larger uncompressed: the first SF1 lineitem row group holds 4.0
million rows and 230 MiB.

DuckDB sorts each raw table into one or more value-range files. One Spark task
reads them in range order and writes the table, and the generator renames
`part-00000-*.parquet` to `<table>.parquet` and deletes the Spark output
directory (`_SUCCESS`, `.crc` files) and the sorted files. Spark orders file
splits by size, largest first, so a single read over several files could
reorder rows; `tpc/spark_write.py` reads each file as exactly one split and
unions those reads in file order. Spark writes integer columns as plain
INT32 and INT64 without the `Int(bitWidth, signed)` annotation DuckDB adds;
both read back as the same Arrow types. Verification of Spark output also
requires a parquet-mr `created_by` and at most 20,000 rows in each data page,
read from the offset index.

| Table | Sort keys |
| --- | --- |
| `customer` | `c_nationkey`, `c_custkey` |
| `lineitem` | `l_shipdate`, `l_receiptdate`, `l_commitdate`, `l_orderkey`, `l_linenumber` |
| `nation` | `n_regionkey`, `n_nationkey` |
| `orders` | `o_orderdate`, `o_orderkey`, `o_custkey` |
| `part` | `p_brand`, `p_type`, `p_size`, `p_partkey` |
| `partsupp` | `ps_partkey`, `ps_suppkey` |
| `region` | `r_regionkey` |
| `supplier` | `s_nationkey`, `s_suppkey` |

Install PySpark from PyPI into its own environment and point the generator at
it. Spark needs Java 17 or 21 on `PATH` or in `JAVA_HOME`:

```bash
python3 -m venv ~/.venvs/pyspark
~/.venvs/pyspark/bin/pip install pyspark==4.0.1
export ALGEON_SPARK_PYTHON=~/.venvs/pyspark/bin/python
```

`--spark-python` overrides `ALGEON_SPARK_PYTHON`; without either, the
generator's own interpreter must have PySpark. The generator checks the
PySpark version and Java before it runs `tpcgen-cli`.
`--optimize-engine duckdb` writes TPC-H with DuckDB instead.

`--unsorted` skips the DuckDB sort; Spark writes tpchgen's rows in generation
order with the same settings. The manifest records the layout
`single_file_per_table_unsorted` and empty `sort_keys`, and verification skips
the monotonic statistics check:

```bash
fixture/tpc-h/generate.py --scale 100 --unsorted --out "$ALGEON_FIXTURE_ROOT/tpch/sf100-unsorted/parquet" --force
```

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

## Demo datasets

`demo/README.md` lists the demo datasets, their converters, and the
Parquet conventions they share. Each converter reads upstream files from its
dataset's `raw/` directory and writes Snappy Parquet next to it:

```bash
fixture/fixture.sh graph dblp-ingest --overwrite              # demo/citation_network/v1/
fixture/fixture.sh amazon-reviews prepare --force             # demo/amazon_reviews_2023/parquet/
fixture/fixture.sh ember2024 prepare --split test --split challenge   # demo/ember2024/parquet/
fixture/fixture.sh kuairand prepare --force                   # demo/kuairand/parquet/
fixture/fixture.sh merrec prepare --force                     # demo/merrec/parquet/
```

For the Citation Network, follow the
[demo walkthrough](../website/docs/guides/datasets/citation-network.mdx) to start a local
server, register the generated Parquet files, and run the first graph query.

## Graph fixtures

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
ln -s "$ALGEON_FIXTURE_ROOT/tpch"             fixture/tpc-h/data
ln -s "$ALGEON_FIXTURE_ROOT/tpcds"            fixture/tpc-ds/data
ln -s "$ALGEON_FIXTURE_ROOT/clickbench"       fixture/clickbench/data
ln -s "$ALGEON_FIXTURE_ROOT/citation_network" fixture/demo/citation_network/v1/parquet
ln -s "$ALGEON_FIXTURE_ROOT/amazon_reviews"   fixture/demo/amazon_reviews_2023/parquet
ln -s "$ALGEON_FIXTURE_ROOT/kuairand"         fixture/demo/kuairand/parquet
ln -s "$ALGEON_FIXTURE_ROOT/merrec"           fixture/demo/merrec/parquet
ln -s "$ALGEON_FIXTURE_ROOT/synthetic"        fixture/graph/synthetic
```

For an explicit external fixture, set `ALGEON_PARQUET_DIR` to its Parquet root.
It is intentionally workload-neutral because one process executes one
workload at a time; the workload validates the expected layout. The in-repo
directories above remain the implicit local defaults.
