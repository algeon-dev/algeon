# Fixtures

Fixtures are organized by workload. Generated and host-provisioned data is
gitignored; the small cuGraph parity corpus in `graph/cugraph/` is versioned.

```
fixture/
  fixture.sh                         # public fixture command
  tpc-h/
    generate.py                      # TPC-H generator
    data/sf<N>/parquet/              # local, gitignored benchmark data
  tpc-ds/
    generate.py                      # TPC-DS generator
    data/sf<N>/parquet/              # local, gitignored benchmark data
  tpc/parquet_layout.py              # shared TPC layout/verification logic
  clickbench/data/                   # local, gitignored ClickBench layouts
  graph/
    cugraph/                         # cuGraph parity corpus and oracle JSON
    demo/                            # local Citation Network Parquet
    dblp.v12.json                    # local source JSON
    dblp_ingest.py                   # Citation Network generator
    prepare.sh                       # cuGraph corpus preparation entrypoint
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
export NEXUS_FIXTURE_ROOT=/mnt/fast/datafusion-nexus-fixtures
fixture/tpc-h/generate.py --scale 1 --scale 10 --variant both --out-root "$NEXUS_FIXTURE_ROOT/tpch" --force
fixture/tpc-ds/generate.py --scale 1 --scale 10 --variant both --out-root "$NEXUS_FIXTURE_ROOT/tpcds" --force
```

Both scripts drive `tpcgen-cli`, then rewrite the final files as one Parquet
file per table, sorted by workload-specific pruning keys with row-group
statistics and Snappy compression. DuckDB is the default external-memory
optimizer; pass `--optimize-engine pyarrow` for the older in-memory path.
Verification checks the expected tables, SF1/SF10/SF100 row counts, row groups,
compression, and monotonic row-group statistics. The files deliberately omit
Parquet footer sorting-column metadata: physical ordering is for pruning
statistics, not an output-ordering promise to DataFusion.

## ClickBench

Fetch the official layouts into its own fixture directory:

```bash
fixture/fixture.sh clickbench fetch --layout both --out ./fixture/clickbench/data
```

`--layout single` writes `hits.parquet`; `--layout partitioned` writes
`partitioned/hits_0.parquet` through `hits_99.parquet`.

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
`NEXUS_CUGRAPH_PYTHON` when set, otherwise the repository virtual environment
or `python3`.

## Local data placement

Point each workload-local data directory at fast storage with a symlink, or
generate/download directly into it:

```bash
export NEXUS_FIXTURE_ROOT=/mnt/fast/datafusion-nexus-fixtures
ln -s "$NEXUS_FIXTURE_ROOT/tpch"       fixture/tpc-h/data
ln -s "$NEXUS_FIXTURE_ROOT/tpcds"      fixture/tpc-ds/data
ln -s "$NEXUS_FIXTURE_ROOT/clickbench" fixture/clickbench/data
ln -s "$NEXUS_FIXTURE_ROOT/demo"       fixture/graph/demo
```

For an explicit external fixture, set `NEXUS_PARQUET_DIR` to its Parquet root.
It is intentionally workload-neutral because one process executes one
workload at a time; the workload validates the expected layout. The in-repo
directories above remain the implicit local defaults.
