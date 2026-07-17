//! Unified developer/operator tool entrypoint.
//!
//! E2E runners are dispatched as explicit subcommands instead of standalone
//! Cargo bin targets, so `cargo test --all-targets` cannot accidentally turn
//! them into test harnesses.

use std::{env, error::Error};

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[cfg(not(test))]
#[path = "../connection_retry.rs"]
mod connection_retry;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/cugraph_bfs_predicate_fixture.rs"]
mod cugraph_bfs_predicate_fixture;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/cugraph_graph500_rmat_fixture.rs"]
mod cugraph_graph500_rmat_fixture;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/optional_fixture.rs"]
mod cugraph_optional_fixture;
#[cfg(all(feature = "cugraph-oracle", not(test)))]
#[path = "../../../../tests/cugraph/cugraph_sql_tests.rs"]
#[allow(dead_code, unused_imports)]
mod cugraph_sql_tests;
#[cfg(all(feature = "embedded-backend-e2e", not(test)))]
#[path = "../e2e/embedded_backend_e2e.rs"]
mod embedded_backend_e2e;
#[cfg(not(test))]
#[path = "../e2e/flight_sql_cugraph_smoke.rs"]
mod flight_sql_cugraph_smoke;
#[cfg(not(test))]
#[path = "../flight_sql_query.rs"]
mod flight_sql_query;
#[cfg(all(feature = "iceberg-e2e", not(test)))]
#[path = "../e2e/iceberg_glue_moto_e2e.rs"]
mod iceberg_glue_moto_e2e;
#[cfg(all(feature = "iceberg-e2e", not(test)))]
#[path = "../e2e/iceberg_rest_local_e2e.rs"]
mod iceberg_rest_local_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_glue_moto_e2e.rs"]
mod server_iceberg_glue_moto_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_glue_remote_direct_e2e.rs"]
mod server_iceberg_glue_remote_direct_e2e;
#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
#[path = "../e2e/server_iceberg_rest_multigpu_e2e.rs"]
mod server_iceberg_rest_multigpu_e2e;
#[cfg(all(feature = "validate-query", not(test)))]
#[path = "../validate_query.rs"]
mod validate_query;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ToolResult<()> {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("flight-sql-cugraph-smoke") => run_flight_sql_cugraph_smoke(env::args().nth(2)).await,
        Some("flight-sql-query") => run_flight_sql_query(env::args().skip(2).collect()).await,
        Some("iceberg-rest-local-e2e") => run_iceberg_rest_local_e2e().await,
        Some("iceberg-glue-moto-e2e") => run_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-moto-e2e") => run_server_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-remote-direct-e2e") => run_server_iceberg_glue_remote_direct_e2e().await,
        Some("server-iceberg-rest-multigpu-e2e") => run_server_iceberg_rest_multigpu_e2e().await,
        Some("embedded-backend-e2e") => run_embedded_backend_e2e(env::args().skip(2).collect()).await,
        Some("cugraph-fixture-e2e") => run_cugraph_fixture_e2e(env::args().nth(2)).await,
        Some("validate-query") => run_validate_query(env::args().skip(2).collect()).await,
        Some("-h" | "--help") | None => {
            print_usage();
            if command.is_none() { Err("missing tool command".into()) } else { Ok(()) }
        },
        Some(other) => {
            print_usage();
            Err(format!("unknown datafusion-nexus-tools command: {other}").into())
        },
    }
}

fn print_usage() {
    eprintln!(
        "datafusion_nexus_tools\n\
         \n\
         USAGE:\n\
           cargo run -p datafusion-nexus-tools --features validate-query -- validate-query --sql <SQL> [--format json|human] [--parquet TABLE=PATH]...\n\
           cargo run -p datafusion-nexus-tools -- flight-sql-cugraph-smoke [URL]\n\
           cargo run -p datafusion-nexus-tools -- flight-sql-query [URL] [SQL]\n\
           cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e\n\
           cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-glue-moto-e2e\n\
           cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-moto-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-remote-direct-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- server-iceberg-rest-multigpu-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features embedded-backend-e2e -- embedded-backend-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features cugraph-oracle -- cugraph-fixture-e2e <pagerank|bfs-include-edges|bfs-utf8-include-edges|bfs-utf8-path|bfs-utf8-predicates|graph500-rmat>"
    );
}

#[cfg(not(test))]
async fn run_flight_sql_cugraph_smoke(url: Option<String>) -> ToolResult<()> {
    flight_sql_cugraph_smoke::run(url).await
}

#[cfg(test)]
async fn run_flight_sql_cugraph_smoke(_url: Option<String>) -> ToolResult<()> {
    Err("flight-sql-cugraph-smoke requires normal cargo run execution".into())
}

#[cfg(not(test))]
async fn run_flight_sql_query(args: Vec<String>) -> ToolResult<()> {
    flight_sql_query::run(args).await
}

#[cfg(test)]
async fn run_flight_sql_query(_args: Vec<String>) -> ToolResult<()> {
    Err("flight-sql-query requires normal cargo run execution".into())
}

#[cfg(all(feature = "validate-query", not(test)))]
async fn run_validate_query(args: Vec<String>) -> ToolResult<()> {
    validate_query::run(args).await
}

#[cfg(not(all(feature = "validate-query", not(test))))]
async fn run_validate_query(_args: Vec<String>) -> ToolResult<()> {
    Err("validate-query requires --features validate-query and normal cargo run execution".into())
}

#[cfg(all(feature = "iceberg-e2e", not(test)))]
async fn run_iceberg_rest_local_e2e() -> ToolResult<()> {
    iceberg_rest_local_e2e::run().await
}

#[cfg(not(all(feature = "iceberg-e2e", not(test))))]
async fn run_iceberg_rest_local_e2e() -> ToolResult<()> {
    Err("iceberg-rest-local-e2e requires --features iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "iceberg-e2e", not(test)))]
async fn run_iceberg_glue_moto_e2e() -> ToolResult<()> {
    iceberg_glue_moto_e2e::run().await
}

#[cfg(not(all(feature = "iceberg-e2e", not(test))))]
async fn run_iceberg_glue_moto_e2e() -> ToolResult<()> {
    Err("iceberg-glue-moto-e2e requires --features iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_glue_moto_e2e() -> ToolResult<()> {
    server_iceberg_glue_moto_e2e::run().await
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_glue_moto_e2e() -> ToolResult<()> {
    Err("server-iceberg-glue-moto-e2e requires --features server-iceberg-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_glue_remote_direct_e2e() -> ToolResult<()> {
    server_iceberg_glue_remote_direct_e2e::run().await
}

#[cfg(all(feature = "server-iceberg-e2e", not(test)))]
async fn run_server_iceberg_rest_multigpu_e2e() -> ToolResult<()> {
    server_iceberg_rest_multigpu_e2e::run().await
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_rest_multigpu_e2e() -> ToolResult<()> {
    Err("server-iceberg-rest-multigpu-e2e requires --features server-iceberg-e2e and normal cargo run execution".into())
}

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_glue_remote_direct_e2e() -> ToolResult<()> {
    Err("server-iceberg-glue-remote-direct-e2e requires --features server-iceberg-e2e and normal cargo run execution"
        .into())
}

#[cfg(all(feature = "embedded-backend-e2e", not(test)))]
async fn run_embedded_backend_e2e(args: Vec<String>) -> ToolResult<()> {
    embedded_backend_e2e::run(args).await
}

#[cfg(not(all(feature = "embedded-backend-e2e", not(test))))]
async fn run_embedded_backend_e2e(_args: Vec<String>) -> ToolResult<()> {
    Err("embedded-backend-e2e requires --features embedded-backend-e2e and normal cargo run execution".into())
}

#[cfg(all(feature = "cugraph-oracle", not(test)))]
async fn run_cugraph_fixture_e2e(case: Option<String>) -> ToolResult<()> {
    match case.as_deref() {
        Some("pagerank") => cugraph_sql_tests::gpu::run_pagerank_filter_sort_limit_acceptance().await,
        Some("bfs-include-edges") => cugraph_bfs_predicate_fixture::run_numeric_include_edges_acceptance().await,
        Some("bfs-utf8-include-edges") => cugraph_bfs_predicate_fixture::run_utf8_include_edges_acceptance().await,
        Some("bfs-utf8-path") => cugraph_bfs_predicate_fixture::run_utf8_path_acceptance().await,
        Some("bfs-utf8-predicates") => cugraph_bfs_predicate_fixture::run_utf8_vertex_predicates_acceptance().await,
        Some("graph500-rmat") => cugraph_graph500_rmat_fixture::run_graph500_rmat_acceptance().await,
        Some(other) => return Err(format!("unknown cugraph fixture acceptance case: {other}").into()),
        None => return Err("cugraph-fixture-e2e requires an acceptance case".into()),
    }
    Ok(())
}

#[cfg(not(all(feature = "cugraph-oracle", not(test))))]
async fn run_cugraph_fixture_e2e(_case: Option<String>) -> ToolResult<()> {
    Err("cugraph-fixture-e2e requires --features cugraph-oracle and normal cargo run execution".into())
}
