#[cfg(all(feature = "cugraph-oracle", not(test)))]
pub(super) mod cugraph_bfs_predicate_fixture;
#[cfg(all(feature = "embedded-backend-e2e", not(test)))]
pub(super) mod embedded_backend_e2e;
#[cfg(not(test))]
pub(super) mod flight_sql_cugraph_smoke;
#[cfg(all(feature = "s3-parquet-e2e", not(test)))]
pub(super) mod s3_parquet_e2e;
#[cfg(feature = "server-runtime-observability-e2e")]
#[cfg_attr(test, allow(dead_code))]
pub(super) mod server_runtime_observability;
