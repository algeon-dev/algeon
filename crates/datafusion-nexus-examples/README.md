# Embedded Backend Example

This crate is a runnable embedding example for a real-world process that
serves two frontends over one GPU backend:

- REST endpoints for predefined GPU pipelines.
- Arrow Flight SQL for ad-hoc SQL, metadata, prepared statements, and cancellation.

Both frontends use the same `NexusGpuBackend` and the same base
`SessionContext`. REST requests use plan-scoped admission from the shared
session. Flight SQL issues statement tickets first and acquires GPU admission
only when a native plan starts executing. Native work from both paths draws
from one `QueryService` ledger, visible at `GET /api/gpu`; CPU fallback work is
bounded separately by the server fallback statement limit.

```text
                                   one backend process
                        +----------------------------------------+
A: HTTP client ---------->   example HTTP API                    |
                        |    /api/pipelines, /api/gpu            |
                        |                  |                     |
                        |                  |                     |
B: Flight SQL client ---->   Flight SQL frontend                 |
                        |    SQL, metadata, cancel               |
                        |                  |                     |
                        |                  v                     |
                        |   shared DataFusion SessionContext     |
                        |                  |                     |
                        |                  v                     |
                        |   NexusGpuBackend                      |
                        |   QueryService admission ledger        |
                        |                  |                     |
                        |                  v                     |
                        |   cuDF / cuGraph on CUDA               |
                        +------------------|---------------------+
                                           |
                                           v
                         Iceberg REST catalog
                         lake.citation_network tables
                                           |
                                           v
                         RustFS/S3 staged Citation Network Parquet
```

This topology is useful when an application has a small set of product-owned
workflows but still needs an expert SQL side door for operators, notebooks, or
BI tools. The important property is not the REST framework; it is that every
frontend in the process shares one backend and one catalog view. GPU-native
execution shares one admission ledger; CPU fallback statements use the separate
host-side fallback bound.

## Data

The example runs against the DBLP / AMiner Citation Network V12 demo dataset:
4,894,081 papers and 45,564,149 citation edges. The E2E runner expects the
local Iceberg REST catalog to contain `lake.citation_network.*` tables, and the
server workspace overlay to expose these unqualified names:

- `papers`
- `paper_fos`
- `citation_edges`
- `citation_edges_by_dst`

The source `dblp.v12.json` is about 12.5 GB. The local developer fixture path
is `fixture/dblp.v12.json`; substitute your own path if you keep the dataset
outside the repo. Keep `fixture/data/demo` on fast local storage, for example
with a symlink to NVMe, and allow roughly 30 GB of transient space for ingest
plus Parquet output. The ingest script requires Python with `pyarrow`.

```bash
python3 fixture/scripts/dblp_ingest.py \
  --source-json fixture/dblp.v12.json \
  --out-root fixture/data/demo \
  --overwrite
```

Start the local REST/RustFS stack and register the generated Parquet files with
Iceberg `add_files`. This mode stages the Parquet files into RustFS and Iceberg
references them in place, so do not delete the `staging/` prefix while the REST
catalog metadata is live. The REST fixture stores metadata in ephemeral SQLite;
rerun the loader after every fresh `docker compose up`.

```bash
docker compose -f fixture/iceberg-local/docker-compose.yml up -d
fixture/scripts/fixture.sh iceberg rest load \
  --workload citation_network \
  --load-mode add-files
```

If the Parquet files are missing, the loader and E2E suite fail fast with the
same acquisition instructions instead of downloading the dataset automatically.

## Run

The example consumes the normal Flight SQL server environment plus
`EXAMPLE_REST_BIND`. `EXAMPLE_REST_BIND` is the bind address for this example's
own axum HTTP API (`/healthz`, `/api/pipelines`, `/api/gpu`). It is not the
Iceberg REST catalog endpoint; the Iceberg REST catalog endpoint is
`NEXUS_ICEBERG_REST_URI`.

Use a bounded native memory policy; the default server profile is bounded, and
the example rejects an unbounded policy because the point of the demo is shared
query-service admission. The manual run command assumes the local Iceberg
REST/RustFS stack is already running and the `citation_network` tables have
been loaded. A quick preflight:

```bash
curl --fail http://localhost:8181/v1/config
```

```bash
EXAMPLE_REST_BIND=127.0.0.1:8080 \
NEXUS_SERVER_BIND=127.0.0.1:50051 \
NEXUS_SERVER_SESSION_PROFILE=native \
NEXUS_SERVER_MAX_CONCURRENT_QUERIES=1 \
NEXUS_SERVER_CUGRAPH_ENABLED=true \
DATAFUSION_CATALOG_DEFAULT_CATALOG=datafusion \
DATAFUSION_CATALOG_DEFAULT_SCHEMA=public \
NEXUS_ICEBERG_CATALOG_KIND=rest \
NEXUS_ICEBERG_CATALOG_NAME=lake \
NEXUS_ICEBERG_NAMESPACE=citation_network \
NEXUS_ICEBERG_WAREHOUSE=s3://lakehouse/warehouse \
NEXUS_ICEBERG_REST_URI=http://localhost:8181 \
NEXUS_ICEBERG_S3_ENDPOINT=http://localhost:9000 \
NEXUS_ICEBERG_S3_REGION=us-east-1 \
NEXUS_ICEBERG_S3_PATH_STYLE=true \
NEXUS_ICEBERG_S3_ACCESS_KEY_ID=nexusadmin \
NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY=nexusadmin \
NEXUS_SERVER_WORKSPACE_CATALOG=datafusion \
NEXUS_SERVER_WORKSPACE_SCHEMA=public \
NEXUS_SERVER_WORKSPACE_BACKING_CATALOG=lake \
NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA=citation_network \
NEXUS_SERVER_WORKSPACE_BACKING_ALIASES=papers,paper_fos,citation_edges,citation_edges_by_dst \
flock /tmp/cudf-gpu.lock \
cargo run -p datafusion-nexus-examples --bin embedded_backend
```

Use the exact Iceberg keys shown above. `NEXUS_ICEBERG_CATALOG=rest` is not a
server config key, and `NEXUS_ICEBERG_REST_WAREHOUSE` is not read by the REST
catalog path. If startup reports `AWS_REGION is required for Glue Iceberg`, the
server did not see `NEXUS_ICEBERG_CATALOG_KIND=rest`.

REST examples:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/api/pipelines

curl -X POST http://127.0.0.1:8080/api/pipelines/sql_aggregate_topk \
  -H 'content-type: application/json' \
  -d '{"year_from":2010,"year_to":2020,"limit":10}'

curl -X POST http://127.0.0.1:8080/api/pipelines/sql_join_facets \
  -H 'content-type: application/json' \
  -d '{"fos_name":"Artificial intelligence"}'

curl -X POST http://127.0.0.1:8080/api/pipelines/graph_pagerank_topk \
  -H 'content-type: application/json' \
  -d '{"limit":10}'

curl -X POST http://127.0.0.1:8080/api/pipelines/graph_bfs_hops \
  -H 'content-type: application/json' \
  -d '{"seed_paper_id":2963403868,"max_depth":2}'

curl http://127.0.0.1:8080/api/pipelines/sql_aggregate_topk/plan
curl http://127.0.0.1:8080/api/gpu
```

Flight SQL clients connect to `grpc://127.0.0.1:50051` and see the same
workspace. For example:

```bash
printf '%s\n' \
  "SELECT count(*) FROM papers" \
  "SELECT vertex, value FROM cugraph_pagerank('citation_edges', 'src', 'dst', 'weight') ORDER BY value DESC LIMIT 5" |
  arrow_cli --host 127.0.0.1 -P 50051 --timeout 120 --output table
```

## Pipelines

| Pipeline | Capability |
| --- | --- |
| `sql_aggregate_topk` | Filter, sort, and limit over `papers`. |
| `sql_join_facets` | Join `paper_fos` to `papers`, then group and aggregate. |
| `graph_pagerank_topk` | Run `cugraph_pagerank`, join vertices back to papers, and return top scores. |
| `graph_bfs_hops` | Run `cugraph_bfs`, join vertices back to papers, and aggregate by hop distance. |

`GET /api/pipelines/{name}/plan` returns the structured `PlanningReport` JSON
for that pipeline's default parameters. Use the report's `Rewritten` and
`Rejected` outcomes as execution evidence; do not depend on printed physical
plans.

## E2E

The black-box E2E suite starts the local stack, loads the citation network with
`--load-mode add-files`, starts this example, and checks REST, Flight SQL,
shared ledger backpressure, and graceful shutdown:

```bash
crates/datafusion-nexus-examples/scripts/e2e.sh
```

- `--skip-stack` when the REST catalog and RustFS are already running.
- `--reload` to drop and re-add the Iceberg tables before the run.
- `--gpu-parity` flag is reserved but not implemented yet.
