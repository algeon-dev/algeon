//! Opt-in Flight SQL server for `datafusion-cudf`.

use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use cudf_query_engine::MemoryControlPolicy;
use tonic::transport::Server;

mod admission;
mod cancel;
pub mod config;
pub mod diagnostics;
pub mod error;
mod flight;
mod session;

#[cfg(feature = "cugraph")]
pub use config::CugraphServerConfig;
pub use config::{
    DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV,
    DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV, DATAFUSION_CUDF_ICEBERG_TABLES_ENV, DATAFUSION_CUDF_SERVER_BIND_ENV,
    DATAFUSION_CUDF_SERVER_CUGRAPH_ALGORITHMS_ENV, DATAFUSION_CUDF_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    DATAFUSION_CUDF_SERVER_CUGRAPH_DIRECTED_ENV, DATAFUSION_CUDF_SERVER_CUGRAPH_ENABLED_ENV,
    DATAFUSION_CUDF_SERVER_CUGRAPH_RENUMBER_ENV, DATAFUSION_CUDF_SERVER_LOG_ENV,
    DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, DATAFUSION_CUDF_SERVER_NATIVE_MEMORY_POLICY_ENV,
    DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV, DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, ServerConfig,
    ServerSessionProfile,
};
pub use diagnostics::{FlightSqlNativeRejectClass, log_leaf_guard_rejection, log_native_planning_report};
pub use error::{Result, ServerError};
pub use session::{apply_session_profile, build_session_config, build_session_context};

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
    warn_if_testing_profile_active(&config);
    let ctx = Arc::new(session::build_session_context(&config).await?);
    let service = flight::DatafusionCudfFlightSqlService::new(
        ctx,
        config.native_optimizer_config.clone(),
        config.session_profile,
        config.max_in_flight,
        config.statement_ticket_ttl,
    );
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
            statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
            log_filter = %config.log_filter,
            session_profile = %config.session_profile,
            native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
            native_memory_control_policy = %native_execution.memory_control_policy(),
            native_device_budget_bytes = %native_memory.device_budget_bytes,
            native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
            native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
            native_max_retry_attempts = %native_memory.max_retry_attempts,
            native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
            native_boundary_policy = %native_execution.boundary_policy(),
            native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
            cugraph_enabled = cugraph_enabled(config),
            cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
            cugraph_allowed_algorithm_count = cugraph_allowed_algorithm_count(config),
            cugraph_allowed_algorithms = %cugraph_allowed_algorithms(config),
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
            statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
            log_filter = %config.log_filter,
            session_profile = %config.session_profile,
            native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
            native_memory_control_policy = %native_execution.memory_control_policy(),
            native_device_budget_bytes = %native_memory.device_budget_bytes,
            native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
            native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
            native_max_retry_attempts = %native_memory.max_retry_attempts,
            native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
            native_boundary_policy = %native_execution.boundary_policy(),
            native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
            cugraph_enabled = cugraph_enabled(config),
            cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
            cugraph_allowed_algorithm_count = cugraph_allowed_algorithm_count(config),
            cugraph_allowed_algorithms = %cugraph_allowed_algorithms(config),
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
        statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
        log_filter = %config.log_filter,
        session_profile = %config.session_profile,
        native_pipeline_execution_mode = %native_execution.native_pipeline_execution_mode(),
        native_memory_control_policy = %native_execution.memory_control_policy(),
        native_device_budget_bytes = %native_memory.device_budget_bytes,
        native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
        native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
        native_max_retry_attempts = %native_memory.max_retry_attempts,
        native_min_retry_chunk_row_groups = %native_memory.min_retry_chunk_row_groups,
        native_boundary_policy = %native_execution.boundary_policy(),
        native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
        cugraph_enabled = cugraph_enabled(config),
        cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
        cugraph_allowed_algorithm_count = cugraph_allowed_algorithm_count(config),
        cugraph_allowed_algorithms = %cugraph_allowed_algorithms(config),
        "server config snapshot"
    );
}

#[cfg(feature = "cugraph")]
fn cugraph_enabled(config: &ServerConfig) -> bool {
    config.cugraph.is_some()
}

#[cfg(not(feature = "cugraph"))]
fn cugraph_enabled(_config: &ServerConfig) -> bool {
    false
}

#[cfg(feature = "cugraph")]
fn cugraph_registry_algorithm_count() -> usize {
    crate::CUGRAPH_OPERATION_REGISTRY.len()
}

#[cfg(not(feature = "cugraph"))]
fn cugraph_registry_algorithm_count() -> usize {
    0
}

#[cfg(feature = "cugraph")]
fn cugraph_allowed_algorithm_count(config: &ServerConfig) -> usize {
    match config.cugraph.as_ref().and_then(|cugraph| cugraph.allowed_algorithms.as_ref()) {
        Some(algorithms) => algorithms.len(),
        None if config.cugraph.is_some() => crate::CUGRAPH_OPERATION_REGISTRY.len(),
        None => 0,
    }
}

#[cfg(not(feature = "cugraph"))]
fn cugraph_allowed_algorithm_count(_config: &ServerConfig) -> usize {
    0
}

#[cfg(feature = "cugraph")]
fn cugraph_allowed_algorithms(config: &ServerConfig) -> String {
    match config.cugraph.as_ref().and_then(|cugraph| cugraph.allowed_algorithms.as_ref()) {
        Some(algorithms) => algorithms.iter().map(|algorithm| algorithm.function_name()).collect::<Vec<_>>().join(","),
        None if config.cugraph.is_some() => "all".to_string(),
        None => "none".to_string(),
    }
}

#[cfg(not(feature = "cugraph"))]
fn cugraph_allowed_algorithms(_config: &ServerConfig) -> String {
    "unavailable".to_string()
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

/// Emit a one-shot startup warning when the operator selected the
/// [`ServerSessionProfile::CpuFallback`] profile against an Iceberg
/// deployment.
///
/// The combination is testing-only: `CudfTableFormatScanExec` has no CPU
/// Arrow reader yet (tracked as Phase 6 of
/// `docs/plan/2026-05-22-flight-sql-iceberg-cpu-fallback-plan.md`), so plain
/// `ORDER BY` shapes will hit the Phase 1 leaf guard and fail in planning.
/// Production Iceberg deployments should use [`ServerSessionProfile::Native`]
/// — which is the default when an Iceberg catalog is configured.
fn warn_if_testing_profile_active(config: &ServerConfig) {
    if !is_testing_profile_active(config) {
        return;
    }
    tracing::warn!(
        session_profile = %config.session_profile,
        "datafusion-cudf Flight SQL server starting with cpu_fallback session \
         profile while Iceberg is configured. This profile is testing-only and \
         does NOT actually fall back to CPU for Iceberg tables (GPU-only \
         CudfTableFormatScanExec has no CPU Arrow reader yet — tracked as \
         Phase 6 of docs/plan/2026-05-22-flight-sql-iceberg-cpu-fallback-plan.md). \
         Plain ORDER BY queries over Iceberg will fail with a structured \
         planning error. Set DATAFUSION_CUDF_SERVER_SESSION_PROFILE=native \
         (the default for Iceberg deployments) for production use."
    );
}

#[cfg(feature = "iceberg")]
pub(crate) fn is_testing_profile_active(config: &ServerConfig) -> bool {
    matches!(config.session_profile, ServerSessionProfile::CpuFallback) && config.iceberg.is_some()
}

#[cfg(not(feature = "iceberg"))]
pub(crate) fn is_testing_profile_active(_config: &ServerConfig) -> bool {
    // Without the iceberg feature there is no GPU-only table-format leaf, so
    // cpu_fallback is the right choice for any CPU-readable source.
    false
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudfOptimizerConfig;

    fn base_config(profile: ServerSessionProfile) -> ServerConfig {
        ServerConfig {
            bind_addr: "127.0.0.1:0".parse().expect("test bind address"),
            log_filter: "info".to_string(),
            max_in_flight: 1,
            statement_ticket_ttl: std::time::Duration::from_secs(60),
            native_optimizer_config: CudfOptimizerConfig::default(),
            session_profile: profile,
            #[cfg(feature = "iceberg")]
            iceberg: None,
            #[cfg(feature = "cugraph")]
            cugraph: None,
        }
    }

    #[test]
    fn testing_profile_inactive_for_native_profile() {
        let config = base_config(ServerSessionProfile::Native);
        assert!(!is_testing_profile_active(&config));
    }

    #[test]
    fn testing_profile_inactive_for_cpu_fallback_without_iceberg() {
        // Without an Iceberg catalog the cpu_fallback profile is the correct
        // choice for any CPU-readable source — no warning should fire.
        let config = base_config(ServerSessionProfile::CpuFallback);
        assert!(!is_testing_profile_active(&config));
    }

    #[cfg(feature = "iceberg")]
    #[test]
    fn testing_profile_active_only_for_cpu_fallback_plus_iceberg() {
        let iceberg = config::IcebergServerConfig {
            catalog_name: "glue".to_string(),
            glue_catalog_handle: "glue".to_string(),
            namespace: "demo_db".to_string(),
            warehouse: "s3://test/".to_string(),
            catalog_id: None,
            region: "us-east-1".to_string(),
            profile: None,
            tables: Vec::new(),
        };

        let mut config = base_config(ServerSessionProfile::CpuFallback);
        config.iceberg = Some(iceberg.clone());
        assert!(is_testing_profile_active(&config), "cpu_fallback + iceberg must arm the WARN");

        let mut config = base_config(ServerSessionProfile::Native);
        config.iceberg = Some(iceberg);
        assert!(
            !is_testing_profile_active(&config),
            "native + iceberg must stay silent — that is the recommended production setup",
        );
    }
}
