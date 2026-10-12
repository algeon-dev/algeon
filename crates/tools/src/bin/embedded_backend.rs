//! REST/Flight server fixture for `embedded-backend-e2e`.
//!
//! This process creates one shared `AlgeonGpuBackend`, serves the example HTTP
//! API on `EXAMPLE_REST_BIND`, and serves Flight SQL on `ALGEON_SERVER_BIND`.

use std::{env, io, net::SocketAddr, process::ExitCode, sync::Arc};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    server::{
        ALGEON_SERVER_LOG_ENV, FlightSqlServer, Result as ServerResult, ServerConfig, backend_from_server_config,
        init_cuda_device,
    },
};
use anyhow::Context;
use axum::Router;
use tokio::{net::TcpListener, signal, task::JoinError};
use tokio_util::sync::CancellationToken;
use tools::embedded_fixture::{AppState, BindAddrs, flight, rest};
use tracing_subscriber::EnvFilter;

const EXAMPLE_REST_BIND_ENV: &str = "EXAMPLE_REST_BIND";
const DEFAULT_REST_BIND: &str = "127.0.0.1:8080";

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("embedded backend failed: {error:#}");
            ExitCode::FAILURE
        },
    }
}

async fn run() -> anyhow::Result<()> {
    let config = ServerConfig::from_env().context("failed to load server config from environment")?;
    let ownership = config
        .gpu_devices()
        .into_iter()
        .map(|ordinal| (ordinal, GpuMemoryOwnership::WholeDeviceExclusive))
        .collect::<Vec<_>>();
    let config = config.with_gpu_memory_ownership(ownership);
    init_tracing(&env::var(ALGEON_SERVER_LOG_ENV).unwrap_or_else(|_| "info".to_string()));

    init_cuda_device(0).context("failed to initialize CUDA device 0")?;
    let backend =
        backend_from_server_config(&config).context("failed to build shared GPU backend from server config")?;

    let rest_addr = rest_bind_addr()?;
    let shutdown = CancellationToken::new();
    let flight_server = flight::build_flight_server(config, backend.clone(), shutdown.clone())
        .await
        .context("failed to build Flight SQL frontend")?;
    let flight_addr = flight_server.local_addr();
    let state = AppState::new(Arc::clone(flight_server.session_context()), backend);
    let rest_router = rest::router(state);

    tracing::info!(
        rest_addr = %rest_addr,
        flight_addr = flight_addr.map(|addr| addr.to_string()).unwrap_or_else(|| "configured".to_string()),
        "starting embedded algeon-datafusion backend"
    );

    serve_frontends(rest_router, rest_addr, flight_server, shutdown).await.context("embedded frontends failed")
}

fn init_tracing(filter: &str) {
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn rest_bind_addr() -> anyhow::Result<SocketAddr> {
    let raw = env::var(EXAMPLE_REST_BIND_ENV).unwrap_or_else(|_| DEFAULT_REST_BIND.to_string());
    raw.parse().with_context(|| format!("failed to parse {EXAMPLE_REST_BIND_ENV} as a socket address: {raw}"))
}

async fn serve_frontends(
    router: Router,
    rest_addr: SocketAddr,
    flight_server: FlightSqlServer,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let rest_listener = TcpListener::bind(rest_addr)
        .await
        .with_context(|| format!("failed to bind example HTTP API frontend to {rest_addr}"))?;
    let bound_rest_addr = rest_listener.local_addr().context("failed to read bound example HTTP API address")?;
    let bind_addrs = BindAddrs { rest: bound_rest_addr, flight: flight_server.local_addr() };
    tracing::info!(
        rest_addr = %bind_addrs.rest,
        flight_addr = bind_addrs.flight.map(|addr| addr.to_string()).unwrap_or_else(|| "unknown".to_string()),
        "embedded frontends bound"
    );

    let rest_shutdown = shutdown.clone();
    let mut rest_task = tokio::spawn(async move {
        axum::serve(rest_listener, router)
            .with_graceful_shutdown(async move {
                rest_shutdown.cancelled().await;
            })
            .await
    });
    let mut flight_task = tokio::spawn(async move { flight_server.serve().await });

    let signal_shutdown = shutdown.clone();
    let mut signal_task = tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        signal_shutdown.cancel();
    });

    tokio::select! {
        result = &mut rest_task => {
            shutdown.cancel();
            handle_rest_result(result)?;
            handle_flight_result(flight_task.await)?;
        }
        result = &mut flight_task => {
            shutdown.cancel();
            handle_flight_result(result)?;
            handle_rest_result(rest_task.await)?;
        }
        _ = shutdown.cancelled() => {
            handle_rest_result(rest_task.await)?;
            handle_flight_result(flight_task.await)?;
        }
    }

    signal_task.abort();
    let _ = (&mut signal_task).await;
    Ok(())
}

fn handle_rest_result(result: Result<Result<(), io::Error>, JoinError>) -> anyhow::Result<()> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error).context("example HTTP API frontend failed"),
        Err(error) => Err(error).context("example HTTP API frontend task failed"),
    }
}

fn handle_flight_result(result: Result<ServerResult<()>, JoinError>) -> anyhow::Result<()> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(error).context("Flight SQL frontend failed"),
        Err(error) => Err(error).context("Flight SQL frontend task failed"),
    }
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate()).expect("install SIGTERM handler");
        tokio::select! {
            _ = signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}
