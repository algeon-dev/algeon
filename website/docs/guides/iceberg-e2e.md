---
sidebar_position: 2
---

# Local Iceberg E2E

The local stack is a Docker Compose deployment of:

- Iceberg REST catalog on `http://localhost:8181`
- RustFS S3-compatible object storage on `http://localhost:9000`
- Moto Glue-compatible endpoint on `http://localhost:5001`

Start it with:

```bash
docker compose -f fixture/iceberg-local/docker-compose.yml up -d
```

The canonical benchmark-style REST E2E entrypoint still loads TPC-H by default:

```bash
fixture/iceberg-local/e2e.sh iceberg-local --reuse-parquet
```

## Load the citation network for website examples

For the website cuGraph examples, load the DBLP / AMiner citation network into
the same REST catalog:

```bash
fixture/fixture.sh iceberg rest load \
  --workload citation_network \
  --load-mode add-files
```

That creates namespace `citation_network` with `citation_edges`,
`citation_edges_by_dst`, `papers`, `paper_authors`, and `paper_fos`. Run the
loader again after a fresh compose stack starts; `add-files` registers staged
Parquet files in place. Run the server with `ALGEON_ICEBERG_CATALOG_KIND=rest`,
`ALGEON_ICEBERG_CATALOG_NAME=lake`, and
`ALGEON_ICEBERG_NAMESPACE=citation_network`; the cuGraph examples use the workspace
overlay so those tables are available as unqualified names.

## Glue tests are local-only

Glue E2E is local Moto + RustFS only. Rust tests must not require a real AWS
Glue/S3 account. Real AWS Glue is useful for remote demos, but it is not the
website's default setup path because users need an AWS account, warehouse
bucket, region, and credential chain.
