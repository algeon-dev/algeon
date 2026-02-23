//! Opt-in Flight SQL server for `datafusion-cudf`.

use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use cudf_query_engine::MemoryControlPolicy;
use tonic::transport::Server;

mod admission;
mod cancel;
pub mod config;
pub mod error;
mod flight;
mod session;

pub use config::{
    DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV,
    DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV, DATAFUSION_CUDF_ICEBERG_TABLES_ENV, DATAFUSION_CUDF_SERVER_BIND_ENV,
    DATAFUSION_CUDF_SERVER_LOG_ENV, DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, ServerConfig,
};
pub use error::{Result, ServerError};

/// Start the Flight SQL server and serve until SIGINT/SIGTERM.
///
/// Graceful shutdown sequence on signal:
/// 1. Close the admission semaphore so pending and future statement admits
///    fail fast instead of acquiring a permit during drain.
/// 2. Cancel the server-wide shutdown [`tokio_util::sync::CancellationToken`].
///    Every per-statement token is a child of this token, so cancellation
///    cascades to statements already admitted as well as any that race in
///    between permit acquisition and registry insert.
/// 3. Let tonic stop accepting new connections and drain existing RPCs; the
///    `serve_with_shutdown` future returns once the last RPC finishes.
pub async fn serve(config: ServerConfig) -> Result<()> {
    let addr = config.bind_addr;
    log_server_config_snapshot(&config);
    let ctx = Arc::new(session::build_session_context(&config).await?);
    let service = flight::DatafusionCudfFlightSqlService::new(ctx, config.max_in_flight);
    let admission = Arc::clone(service.admission());
    let cancellations = Arc::clone(service.cancellations());
    let shutdown_token = service.shutdown_token().clone();

    tracing::info!(
        bind_addr = %addr,
        max_in_flight = config.max_in_flight,
        "starting datafusion-cudf Flight SQL server"
    );

    // Hand back the escalation-watchdog handle so we can abort it once the
    // server has drained. Without this, a SIGINT received after a clean
    // shutdown (e.g. when `serve` is called repeatedly in tests) would still
    // trigger the spawned `process::exit`.
    let (watchdog_tx, watchdog_rx) = tokio::sync::oneshot::channel();
    let shutdown = async move {
        wait_for_shutdown_signal().await;
        admission.close();
        let in_flight = cancellations.len();
        shutdown_token.cancel();
        tracing::info!(in_flight, "received shutdown signal; draining datafusion-cudf Flight SQL server");
        // Stay listening for further signals during drain so a hung drain can
        // be force-exited with a second Ctrl-C / SIGTERM. 130 is the
        // conventional exit code for SIGINT-terminated processes.
        let watchdog = tokio::spawn(async {
            wait_for_shutdown_signal().await;
            tracing::warn!("received second shutdown signal during drain; forcing immediate exit");
            std::process::exit(130);
        });
        let _ = watchdog_tx.send(watchdog);
    };

    let server_result =
        Server::builder().add_service(FlightServiceServer::new(service)).serve_with_shutdown(addr, shutdown).await;

    // Abort the watchdog if it was armed. If serve_with_shutdown returned
    // before the shutdown future ran (e.g. a bind/listen error), the oneshot
    // sender was dropped — `recv` returns Err and there is nothing to abort.
    if let Ok(watchdog) = watchdog_rx.await {
        watchdog.abort();
    }
    server_result.map_err(ServerError::transport)?;

    tracing::info!("datafusion-cudf Flight SQL server stopped");
    Ok(())
}

#[cfg(feature = "iceberg")]
fn log_server_config_snapshot(config: &ServerConfig) {
    let native_execution = config.native_optimizer_config.native_execution_config();
    let native_memory = NativeMemorySnapshot::from(native_execution.memory_control_policy());
    let source_scheduler = native_execution.source_scheduler_policy();
    match &config.iceberg {
        Some(iceberg) => tracing::info!(
            bind_addr = %config.bind_addr,
            max_in_flight = config.max_in_flight,
            log_filter = %config.log_filter,
            native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
            native_memory_control_policy = %native_execution.memory_control_policy(),
            native_device_budget_bytes = %native_memory.device_budget_bytes,
            native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
            native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
            native_max_retry_attempts = %native_memory.max_retry_attempts,
            native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
            native_boundary_policy = %native_execution.boundary_policy(),
            native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
            iceberg_enabled = true,
            iceberg_catalog_name = %iceberg.catalog_name,
            iceberg_glue_catalog_handle = %iceberg.glue_catalog_handle,
            iceberg_namespace = %iceberg.namespace,
            iceberg_warehouse = %iceberg.warehouse,
            iceberg_region = %iceberg.region,
            iceberg_glue_catalog_id = %option_display(iceberg.catalog_id.as_deref()),
            iceberg_aws_profile = %option_display(iceberg.profile.as_deref()),
            iceberg_static_table_count = iceberg.tables.len(),
            "server config snapshot"
        ),
        None => tracing::info!(
            bind_addr = %config.bind_addr,
            max_in_flight = config.max_in_flight,
            log_filter = %config.log_filter,
            native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
            native_memory_control_policy = %native_execution.memory_control_policy(),
            native_device_budget_bytes = %native_memory.device_budget_bytes,
            native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
            native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
            native_max_retry_attempts = %native_memory.max_retry_attempts,
            native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
            native_boundary_policy = %native_execution.boundary_policy(),
            native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
            iceberg_enabled = false,
            "server config snapshot"
        ),
    }
}

#[cfg(not(feature = "iceberg"))]
fn log_server_config_snapshot(config: &ServerConfig) {
    let native_execution = config.native_optimizer_config.native_execution_config();
    let native_memory = NativeMemorySnapshot::from(native_execution.memory_control_policy());
    let source_scheduler = native_execution.source_scheduler_policy();
    tracing::info!(
        bind_addr = %config.bind_addr,
        max_in_flight = config.max_in_flight,
        log_filter = %config.log_filter,
        native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
        native_memory_control_policy = %native_execution.memory_control_policy(),
        native_device_budget_bytes = %native_memory.device_budget_bytes,
        native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
        native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
        native_max_retry_attempts = %native_memory.max_retry_attempts,
        native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
        native_boundary_policy = %native_execution.boundary_policy(),
        native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
        "server config snapshot"
    );
}

struct NativeMemorySnapshot {
    device_budget_bytes: String,
    max_source_chunk_bytes: String,
    max_row_groups_per_chunk: String,
    max_retry_attempts: String,
    min_retry_chunk_row_groups: String,
}

impl From<MemoryControlPolicy> for NativeMemorySnapshot {
    fn from(policy: MemoryControlPolicy) -> Self {
        match policy {
            MemoryControlPolicy::Unbounded => Self {
                device_budget_bytes: "none".to_string(),
                max_source_chunk_bytes: "none".to_string(),
                max_row_groups_per_chunk: "none".to_string(),
                max_retry_attempts: "none".to_string(),
                min_retry_chunk_row_groups: "none".to_string(),
            },
            MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            } => Self {
                device_budget_bytes: device_budget_bytes.to_string(),
                max_source_chunk_bytes: max_source_chunk_bytes.to_string(),
                max_row_groups_per_chunk: max_row_groups_per_chunk.to_string(),
                max_retry_attempts: max_retry_attempts.to_string(),
                min_retry_chunk_row_groups: min_retry_chunk_row_groups.to_string(),
            },
        }
    }
}

#[cfg(feature = "iceberg")]
fn option_display(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

/// Resolve when the process receives SIGINT or SIGTERM. cudf and cugraph are
/// Linux-only, so the whole server is Linux-only and we do not gate on `unix`.
/// The future also resolves if the signal stream errors out so the server
/// does not hang holding a broken listener.
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(stream) => stream,
        Err(error) => {
            tracing::error!(%error, "failed to install SIGTERM handler; relying on SIGINT only");
            let _ = tokio::signal::ctrl_c().await;
            return;
        },
    };
    tokio::select! {
        result = tokio::signal::ctrl_c() => {
            if let Err(error) = result {
                tracing::error!(%error, "ctrl-c listener failed; shutting down");
            }
        }
        _ = sigterm.recv() => {}
    }
}
