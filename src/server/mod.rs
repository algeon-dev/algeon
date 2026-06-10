//! Opt-in Flight SQL server for `datafusion-nexus`.

#[cfg(feature = "iceberg")]
use std::borrow::Cow;
use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use nexus_query_engine::MemoryControlPolicy;
use tonic::transport::Server;

#[cfg(feature = "iceberg")]
use crate::table_format::scan::credential_free_endpoint_identity;

mod admission;
pub mod auth;
mod cancel;
pub mod config;
pub mod diagnostics;
pub mod error;
mod flight;
mod session;

pub use auth::{AuthConfig, AuthInterceptor, AuthMode, Principal, StaticUserConfig, TokenStore};
pub use config::*;
pub use diagnostics::{FlightSqlNativeRejectClass, log_leaf_guard_rejection, log_native_planning_report};
pub use error::{Result, ServerError};
pub use session::{apply_session_profile, build_session_config, build_session_context};

#[cfg(feature = "iceberg")]
pub use crate::{
    NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
    NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
};

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
    ensure_single_librmm_runtime_mapping()?;
    let addr = config.bind_addr;
    log_server_config_snapshot(&config);
    warn_if_testing_profile_active(&config);
    warn_if_cugraph_built_but_disabled(&config);
    let ctx = Arc::new(session::build_session_context(&config).await?);
    let auth_interceptor = auth::AuthInterceptor::new(config.auth.mode.clone(), auth::TokenStore::new());
    log_auth_mode_snapshot(&config.auth.mode);
    let service = flight::DatafusionCudfFlightSqlService::new(
        ctx,
        config.native_optimizer_config.clone(),
        config.session_profile,
        config.max_in_flight,
        config.statement_ticket_ttl,
        auth_interceptor,
    );
    let admission = Arc::clone(service.admission());
    let cancellations = Arc::clone(service.cancellations());
    let shutdown_token = service.shutdown_token().clone();

    tracing::info!(
        bind_addr = %addr,
        max_in_flight = config.max_in_flight,
        "starting datafusion-nexus Flight SQL server"
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
        tracing::info!(in_flight, "received shutdown signal; draining datafusion-nexus Flight SQL server");
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

    tracing::info!("datafusion-nexus Flight SQL server stopped");
    Ok(())
}

#[cfg(feature = "iceberg")]
fn log_server_config_snapshot(config: &ServerConfig) {
    let native_execution = config.native_optimizer_config.native_execution_config();
    let native_memory = NativeMemorySnapshot::from(native_execution.memory_control_policy());
    let source_scheduler = native_execution.source_scheduler_policy();
    match &config.iceberg {
        Some(iceberg) => {
            let snapshot = iceberg_backend_snapshot(&iceberg.backend);
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
                iceberg_enabled = true,
                iceberg_catalog_name = %iceberg.catalog_name,
                iceberg_catalog_type = %snapshot.catalog_type,
                iceberg_namespace = %iceberg.namespace,
                iceberg_warehouse = %snapshot.warehouse,
                iceberg_glue_catalog_handle = %snapshot.glue_catalog_handle,
                iceberg_glue_region = %snapshot.glue_region,
                iceberg_glue_catalog_id = %snapshot.glue_catalog_id,
                iceberg_aws_profile = %snapshot.aws_profile,
                iceberg_rest_uri = %snapshot.rest_uri,
                iceberg_rest_s3_endpoint = %snapshot.rest_s3_endpoint,
                iceberg_scan_planning_timeout_secs = iceberg.scan_planning_timeout.as_secs(),
                iceberg_static_table_count = iceberg.tables.len(),
                "server config snapshot"
            )
        },
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

/// Emit a one-shot startup WARN when this binary was built with the `cugraph`
/// feature but cuGraph SQL functions were not enabled at runtime
/// (`NEXUS_SERVER_CUGRAPH_ENABLED` is not `true`). Without it the
/// `cugraph_*` table functions are never registered and queries fail with a
/// not-found error, which is easy to misread as a build problem when the
/// feature is actually compiled in. The cfg-gated definitions keep this a
/// no-op on builds without the feature (where the rebuild hint applies instead).
#[cfg(feature = "cugraph")]
fn warn_if_cugraph_built_but_disabled(config: &ServerConfig) {
    if config.cugraph.is_some() {
        return;
    }
    tracing::warn!(
        cugraph_enabled = false,
        "datafusion-nexus Flight SQL server was built with the `cugraph` feature \
         but cuGraph SQL functions are disabled at runtime, so they will not be \
         registered. Set NEXUS_SERVER_CUGRAPH_ENABLED=true to register \
         the cugraph_* table functions."
    );
}

#[cfg(not(feature = "cugraph"))]
fn warn_if_cugraph_built_but_disabled(_config: &ServerConfig) {}

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

/// Borrowed, secret-redacted view of an Iceberg backend config for the startup
/// snapshot log. Backend-specific fields are `"none"` for the other backend.
/// Secret values (`token`, `s3.secret-access-key`) are never included.
#[cfg(feature = "iceberg")]
struct IcebergBackendSnapshot<'a> {
    catalog_type: &'a str,
    warehouse: &'a str,
    glue_catalog_handle: &'a str,
    glue_region: &'a str,
    glue_catalog_id: &'a str,
    aws_profile: &'a str,
    rest_uri: Cow<'a, str>,
    rest_s3_endpoint: Cow<'a, str>,
}

#[cfg(feature = "iceberg")]
fn iceberg_backend_snapshot(backend: &config::IcebergCatalogBackendConfig) -> IcebergBackendSnapshot<'_> {
    match backend {
        config::IcebergCatalogBackendConfig::Glue(glue) => IcebergBackendSnapshot {
            catalog_type: "glue",
            warehouse: &glue.warehouse,
            glue_catalog_handle: &glue.catalog_handle,
            glue_region: &glue.region,
            glue_catalog_id: option_display(glue.catalog_id.as_deref()),
            aws_profile: option_display(glue.profile.as_deref()),
            rest_uri: Cow::Borrowed("none"),
            rest_s3_endpoint: Cow::Borrowed("none"),
        },
        config::IcebergCatalogBackendConfig::Rest(rest) => IcebergBackendSnapshot {
            catalog_type: "rest",
            warehouse: &rest.warehouse,
            glue_catalog_handle: "none",
            glue_region: "none",
            glue_catalog_id: "none",
            aws_profile: "none",
            rest_uri: Cow::Owned(credential_free_endpoint_identity(&rest.uri)),
            rest_s3_endpoint: rest
                .s3_endpoint
                .as_deref()
                .map(credential_free_endpoint_identity)
                .map(Cow::Owned)
                .unwrap_or(Cow::Borrowed("none")),
        },
    }
}

/// Emit a one-shot startup line summarizing the configured auth mode,
/// and a WARN when [`auth::AuthMode::Disabled`] is selected so operators
/// see in the log that the server is wide open.
fn log_auth_mode_snapshot(mode: &auth::AuthMode) {
    match mode {
        auth::AuthMode::Disabled => {
            tracing::warn!(
                auth_mode = "disabled",
                "datafusion-nexus Flight SQL server starting with auth DISABLED. \
                 Every RPC will be accepted. This is unsafe to expose outside \
                 a trusted network. Configure [auth] in the server TOML config \
                 to enable Basic+Bearer authentication."
            );
        },
        auth::AuthMode::BasicBearer { users, token_ttl_secs } => {
            tracing::info!(
                auth_mode = "basic_bearer",
                user_count = users.len(),
                token_ttl_secs = *token_ttl_secs,
                "datafusion-nexus Flight SQL server starting with Basic+Bearer auth"
            );
        },
    }
}

/// Warn when Iceberg uses `CpuFallback` without the CPU Iceberg reader.
fn warn_if_testing_profile_active(config: &ServerConfig) {
    if !is_testing_profile_active(config) {
        return;
    }
    tracing::warn!(
        session_profile = %config.session_profile,
        "datafusion-nexus Flight SQL server starting with cpu_fallback session \
         profile while Iceberg is configured, but this binary was built \
         without the `iceberg-cpu` feature. Iceberg tables will be served by \
         the GPU-only CudfTableFormatScanExec leaf and plain ORDER BY queries \
         will fail with a structured planning error. Either rebuild with \
         --features iceberg-cpu to enable the CPU Iceberg reader, or switch \
         to NEXUS_SERVER_SESSION_PROFILE=native."
    );
}

/// Returns true when the active configuration produces a CPU profile + Iceberg
/// pairing that this binary cannot actually serve.
///
/// With the `iceberg-cpu` feature compiled in, `CpuFallback` + Iceberg routes
/// through `iceberg_datafusion::IcebergStaticTableProvider` and is fully
/// supported, so this returns false. Without that feature the leaf is still
/// GPU-only and the warning fires.
#[cfg(all(feature = "iceberg", not(feature = "iceberg-cpu")))]
pub(crate) fn is_testing_profile_active(config: &ServerConfig) -> bool {
    matches!(config.session_profile, ServerSessionProfile::CpuFallback) && config.iceberg.is_some()
}

#[cfg(all(feature = "iceberg", feature = "iceberg-cpu"))]
pub(crate) fn is_testing_profile_active(_config: &ServerConfig) -> bool {
    // CpuFallback + Iceberg + iceberg-cpu serves Iceberg via
    // iceberg_datafusion::IcebergStaticTableProvider (real CPU Arrow reader),
    // so the pairing is production-supported.
    false
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

/// Refuse to serve when more than one distinct `librmm.so` build is mapped
/// into this process. RMM holds process-global device-resource state, and the
/// dynamic loader binds every exported rmm symbol to a single copy for the
/// whole process, so a second mapping (e.g. a stale copy resolved through
/// `LD_LIBRARY_PATH` next to the one libcugraph links) executes some consumers
/// against the wrong object layout and globals. The known failure mode is a
/// SIGSEGV inside `rmm::device_buffer::allocate_async` during the first
/// `DoGet`.
/// The build-time guard in `build.rs` pins what the link graph resolves; this
/// backstop catches what the loader actually mapped.
fn ensure_single_librmm_runtime_mapping() -> Result<()> {
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => maps,
        Err(error) => {
            tracing::warn!(%error, "could not read /proc/self/maps; skipping the single-librmm runtime check");
            return Ok(());
        },
    };
    let mappings = distinct_librmm_mappings(&maps);
    if mappings.len() <= 1 {
        return Ok(());
    }
    Err(ServerError::startup(format!(
        "{} distinct librmm.so files are mapped into this process: {}. Every RAPIDS consumer in one process must \
         bind the same RMM build; a second mapping means some consumer executes against the wrong rmm object \
         layout and device-resource state. Rebuild the cugraph C++ tree against the cudf install prefix \
         (cmake -Drmm_ROOT=<cudf-install>) and remove stale librmm.so copies from every library search path \
         (rpaths, LD_LIBRARY_PATH).",
        mappings.len(),
        mappings.join(", ")
    )))
}

/// Distinct file paths in `/proc/self/maps` content whose file name is
/// `librmm.so` or a versioned `librmm.so.*`. Deleted mappings still count:
/// an unlinked stale copy keeps executing for the life of the process.
fn distinct_librmm_mappings(maps: &str) -> Vec<String> {
    let mut mappings = Vec::new();
    for line in maps.lines() {
        let Some(path_start) = line.find('/') else { continue };
        let path = line[path_start..].trim_end();
        let path = path.strip_suffix(" (deleted)").unwrap_or(path);
        let Some(file_name) = std::path::Path::new(path).file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name != "librmm.so" && !file_name.starts_with("librmm.so.") {
            continue;
        }
        if !mappings.iter().any(|existing| existing == path) {
            mappings.push(path.to_string());
        }
    }
    mappings
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
            auth: auth::AuthConfig::default(),
            #[cfg(feature = "iceberg")]
            iceberg: None,
            #[cfg(feature = "cugraph")]
            cugraph: None,
        }
    }

    #[test]
    fn distinct_librmm_mappings_dedupes_segments_and_matches_versioned_sonames() {
        let maps = "7f0a1c000000-7f0a1c2bc000 r-xp 00000000 103:07 4584938  /opt/rapids/lib/librmm.so\n\
                    7f0a1c2bc000-7f0a1c2c0000 rw-p 002bb000 103:07 4584938  /opt/rapids/lib/librmm.so\n\
                    7f0a1d000000-7f0a1d100000 r-xp 00000000 103:07 4584939  /opt/other/librmm.so.26.06\n\
                    7f0a1e000000-7f0a1e100000 r-xp 00000000 103:07 4584940  /usr/lib/libstdc++.so.6\n";

        let mappings = distinct_librmm_mappings(maps);

        assert_eq!(mappings, vec!["/opt/rapids/lib/librmm.so".to_string(), "/opt/other/librmm.so.26.06".to_string()]);
    }

    #[test]
    fn distinct_librmm_mappings_counts_deleted_and_ignores_lookalikes() {
        let maps = "7f0a1c000000-7f0a1c2bc000 r-xp 00000000 103:07 4584938  /opt/rapids/lib/librmm.so (deleted)\n\
                    7f0a1d000000-7f0a1d100000 r-xp 00000000 103:07 4584939  /opt/other/librmm_helpers.so\n\
                    7f0a1e000000-7f0a1e100000 rw-p 00000000 00:00 0\n";

        let mappings = distinct_librmm_mappings(maps);

        assert_eq!(mappings, vec!["/opt/rapids/lib/librmm.so".to_string()]);
    }

    #[test]
    fn single_librmm_runtime_mapping_check_passes_in_this_process() {
        ensure_single_librmm_runtime_mapping()
            .expect("this test process must not map two librmm.so builds; a failure here is a real regression");
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
            namespace: "tpch_sf1".to_string(),
            tables: Vec::new(),
            backend: config::IcebergCatalogBackendConfig::Glue(config::GlueServerBackendConfig {
                catalog_handle: "glue".to_string(),
                warehouse: "s3://test/".to_string(),
                region: "us-east-1".to_string(),
                catalog_id: None,
                profile: None,
                uri: None,
                s3_endpoint: None,
                s3_region: None,
                s3_path_style_access: None,
            }),
            parquet_read_strategy_config: crate::TableFormatParquetReadStrategyConfig::default(),
            scan_planning_timeout: std::time::Duration::from_secs(60),
        };

        let mut config = base_config(ServerSessionProfile::CpuFallback);
        config.iceberg = Some(iceberg.clone());
        // With `iceberg-cpu` compiled in the pairing is supported by
        // iceberg_datafusion::IcebergStaticTableProvider; without it the
        // GPU-only leaf still applies and the WARN must fire.
        #[cfg(feature = "iceberg-cpu")]
        assert!(
            !is_testing_profile_active(&config),
            "cpu_fallback + iceberg + iceberg-cpu is production-supported via iceberg-datafusion",
        );
        #[cfg(not(feature = "iceberg-cpu"))]
        assert!(is_testing_profile_active(&config), "cpu_fallback + iceberg without iceberg-cpu must arm the WARN",);

        let mut config = base_config(ServerSessionProfile::Native);
        config.iceberg = Some(iceberg);
        assert!(
            !is_testing_profile_active(&config),
            "native + iceberg must stay silent — that is the recommended production setup",
        );
    }

    #[cfg(feature = "iceberg")]
    #[test]
    fn iceberg_backend_snapshot_sanitizes_rest_urls_for_logs() {
        let backend = config::IcebergCatalogBackendConfig::Rest(
            crate::table_format::iceberg::catalog::RestIcebergCatalogConfig::new(
                "lake",
                "https://user:rest-secret@rest.local:8181/catalog?token=secret#frag",
                "s3://lakehouse/warehouse",
            )
            .with_s3_endpoint("http://user:s3-secret@rustfs.local:9000/path?sig=secret#frag"),
        );

        let snapshot = iceberg_backend_snapshot(&backend);

        assert_eq!(snapshot.rest_uri, "https://rest.local:8181/catalog");
        assert_eq!(snapshot.rest_s3_endpoint, "http://rustfs.local:9000/path");
        assert!(!snapshot.rest_uri.contains("secret"));
        assert!(!snapshot.rest_s3_endpoint.contains("secret"));
        assert!(!snapshot.rest_s3_endpoint.contains("sig="));
    }
}
