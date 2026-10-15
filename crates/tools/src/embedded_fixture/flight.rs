//! Flight SQL frontend wiring for the embedded backend fixture.
//!
//! Builds the `algeon-datafusion` Flight SQL server around the shared GPU
//! backend used by the fixture HTTP API.

use algeon_datafusion::{
    backend::AlgeonGpuBackend,
    server::{FlightSqlServer, ServerConfig, ServerShutdown},
};
use anyhow::Context;
use tokio_util::sync::CancellationToken;

pub async fn build_flight_server(
    config: ServerConfig,
    backend: AlgeonGpuBackend,
    shutdown: CancellationToken,
) -> anyhow::Result<FlightSqlServer> {
    let bind_addr = config.bind_addr;
    let listener = std::net::TcpListener::bind(bind_addr)
        .with_context(|| format!("failed to bind Flight SQL frontend to {bind_addr}"))?;
    FlightSqlServer::builder()
        .config(config)
        .backend(backend)
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown))
        .build()
        .await
        .context("failed to initialize Flight SQL server")
}
