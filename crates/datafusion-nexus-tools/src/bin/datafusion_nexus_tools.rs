//! Unified developer/operator tool entrypoint.
//!
//! E2E runners are dispatched as explicit subcommands instead of standalone
//! Cargo bin targets, so `cargo test --all-targets` cannot accidentally turn
//! them into test harnesses.

use std::{env, error::Error};

type ToolResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[cfg(not(test))]
#[path = "../e2e/flight_sql_cugraph_smoke.rs"]
mod flight_sql_cugraph_smoke;
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

#[tokio::main(flavor = "current_thread")]
async fn main() -> ToolResult<()> {
    let command = env::args().nth(1);
    match command.as_deref() {
        Some("flight-sql-cugraph-smoke") => run_flight_sql_cugraph_smoke(env::args().nth(2)).await,
        Some("iceberg-rest-local-e2e") => run_iceberg_rest_local_e2e().await,
        Some("iceberg-glue-moto-e2e") => run_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-moto-e2e") => run_server_iceberg_glue_moto_e2e().await,
        Some("server-iceberg-glue-remote-direct-e2e") => run_server_iceberg_glue_remote_direct_e2e().await,
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
           cargo run -p datafusion-nexus-tools -- flight-sql-cugraph-smoke [URL]\n\
           cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e\n\
           cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-glue-moto-e2e\n\
           cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-moto-e2e\n\
           flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- server-iceberg-glue-remote-direct-e2e"
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

#[cfg(not(all(feature = "server-iceberg-e2e", not(test))))]
async fn run_server_iceberg_glue_remote_direct_e2e() -> ToolResult<()> {
    Err("server-iceberg-glue-remote-direct-e2e requires --features server-iceberg-e2e and normal cargo run execution"
        .into())
}
