//! Opt-in Flight SQL server for `datafusion-nexus`.

#[cfg(feature = "iceberg")]
use std::borrow::Cow;
use std::{future::Future, net::SocketAddr, pin::Pin, sync::Arc, time::Instant};

use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionContext;
use nexus_query_engine::runtime::MemoryControlPolicy;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Server, server::TcpIncoming};

use crate::backend::NexusGpuBackend;
#[cfg(feature = "iceberg")]
use crate::table_format::scan::credential_free_endpoint_identity;

mod admission;
mod auth;
mod cancel;
mod config;
mod datafusion_error;
mod diagnostics;
mod error;
mod flight;
mod observability;
mod session;
mod table_format_operator_fast_fail;
mod workspace;

pub use auth::{
    AuthConfig, AuthInterceptor, AuthMode, PasswordHashError, Principal, StaticUserConfig, TokenStore,
    handle_handshake, hash_password, verify_password,
};
#[cfg(feature = "cugraph")]
pub use config::CugraphServerConfig;
#[cfg(feature = "iceberg")]
pub use config::{
    GlueServerBackendConfig, IcebergCatalogBackendConfig, IcebergServerConfig, IcebergTableRegistrationConfig,
};
pub use config::{
    NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
    NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_DATABASE_ENV, NEXUS_ICEBERG_GLUE_TABLES_ENV,
    NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
    NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV, NEXUS_ICEBERG_REST_PREFIX_ENV, NEXUS_ICEBERG_REST_TOKEN_ENV,
    NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
    NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    NEXUS_ICEBERG_TABLES_ENV, NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV,
    NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV,
    NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV,
    NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV, NEXUS_SERVER_BIND_ENV,
    NEXUS_SERVER_CONFIG_FILE_ENV, NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS_ENV,
    NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS_ENV, NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, NEXUS_SERVER_CUGRAPH_ENABLED_ENV, NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
    NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, NEXUS_SERVER_GPU_DEVICES_ENV, NEXUS_SERVER_LOG_ENV,
    NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, NEXUS_SERVER_MAX_IN_FLIGHT_ENV, NEXUS_SERVER_NATIVE_CONFIG_OPTIONS,
    NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV, NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV,
    NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
    NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
    NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
    NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
    NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, NEXUS_SERVER_SESSION_PROFILE_ENV,
    NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV,
    NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV, NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
    NEXUS_SERVER_WORKSPACE_CATALOG_ENV, NEXUS_SERVER_WORKSPACE_SCHEMA_ENV, NexusServerNativeConfigOption, ServerConfig,
    ServerNativeConfigScope, ServerSessionProfile, WorkspaceOverlayConfig,
};
pub use diagnostics::{
    FlightSqlNativeRejectClass, log_leaf_guard_rejection, log_native_planning_report,
    log_native_planning_report_for_tables,
};
pub use error::{
    ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY, Result,
    ServerError, UNSUPPORTED_COLUMN_METADATA_KEY, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
    UNSUPPORTED_COLUMN_TYPE_METADATA_KEY, datafusion_error_to_status, datafusion_error_to_status_for_operation,
};
pub use session::{
    apply_session_profile, build_session_config, build_session_context, build_session_context_with_backend,
};

/// Flight SQL server shutdown source.
#[derive(Debug, Clone, Default)]
pub enum ServerShutdown {
    /// Install SIGINT/SIGTERM handlers and preserve the standalone server escalation watchdog.
    #[default]
    OsSignals,
    /// Drain when the caller-controlled token is cancelled.
    Token(CancellationToken),
}

/// Builder for the embeddable Flight SQL server.
#[derive(Default)]
pub struct FlightSqlServerBuilder {
    config: Option<ServerConfig>,
    backend: Option<NexusGpuBackend>,
    session_context: Option<Arc<SessionContext>>,
    listener: Option<std::net::TcpListener>,
    shutdown: ServerShutdown,
}

/// Embeddable Flight SQL server assembled from a [`ServerConfig`].
pub struct FlightSqlServer {
    service: flight::DatafusionCudfFlightSqlService,
    listen: Listen,
    shutdown: ServerShutdown,
    ctx: Arc<SessionContext>,
    backend: Option<NexusGpuBackend>,
    local_addr: Option<SocketAddr>,
}

impl FlightSqlServer {
    pub fn builder() -> FlightSqlServerBuilder {
        FlightSqlServerBuilder::default()
    }

    /// Shared base session used by the server. Embedded frontends can clone it.
    pub fn session_context(&self) -> &Arc<SessionContext> {
        &self.ctx
    }

    /// Shared backend used by the server, when bounded query-service admission is active.
    pub fn backend(&self) -> Option<&NexusGpuBackend> {
        self.backend.as_ref()
    }

    /// Address of an adopted listener, when the caller pre-bound the socket.
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    pub async fn serve(self) -> Result<()> {
        serve_built_server(self).await
    }
}

impl FlightSqlServerBuilder {
    pub fn config(mut self, config: ServerConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn backend(mut self, backend: NexusGpuBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn session_context(mut self, ctx: Arc<SessionContext>) -> Self {
        self.session_context = Some(ctx);
        self
    }

    pub fn listener(mut self, listener: std::net::TcpListener) -> Self {
        self.listener = Some(listener);
        self
    }

    pub fn shutdown(mut self, shutdown: ServerShutdown) -> Self {
        self.shutdown = shutdown;
        self
    }

    pub async fn build(self) -> Result<FlightSqlServer> {
        let config = self.config.ok_or_else(|| {
            ServerError::builder_validation(
                "server_config_required",
                "server config is required to build Flight SQL server",
            )
        })?;
        ensure_single_librmm_runtime_mapping()?;
        let listen = match self.listener {
            Some(listener) => Listen::Prebound(listener),
            None => Listen::Addr(config.bind_addr),
        };
        let addr = listen.addr()?;
        log_server_config_snapshot(&config);
        warn_if_testing_profile_active(&config);
        warn_if_cugraph_built_but_disabled(&config);

        if self.session_context.is_some() && self.backend.is_none() {
            return Err(ServerError::builder_validation(
                "server_session_without_backend",
                "an injected server session_context requires an injected backend",
            ));
        }

        let backend = match self.backend {
            Some(backend) => {
                validate_injected_backend_matches_config(&backend, &config)?;
                Some(backend)
            },
            None => NexusGpuBackend::from_server_config(&config)?,
        };

        let admission_init_start = Instant::now();
        let admission = Arc::new(admission::QueryAdmission::from_backend(backend.as_ref(), &config)?);
        tracing::info!(
            elapsed_ms = u64::try_from(admission_init_start.elapsed().as_millis()).unwrap_or(u64::MAX),
            "query admission controller initialized"
        );
        admission.log_query_service_state(None, "server_startup");

        let mut native_optimizer_config = backend
            .as_ref()
            .map_or_else(|| config.native_optimizer_config.clone(), |backend| backend.optimizer_config().clone());
        if let Some(limiter) = admission.unbounded_native_execution_limiter() {
            native_optimizer_config = native_optimizer_config.with_unbounded_native_execution_limiter(limiter);
        }

        let ctx = match self.session_context {
            Some(ctx) => {
                let backend = backend.as_ref().expect("injected session_context requires injected backend");
                validate_injected_session_matches_backend(&ctx, backend)?;
                ctx
            },
            None => {
                let mut session_config = config.clone();
                session_config.native_optimizer_config = native_optimizer_config.clone();
                Arc::new(session::build_session_context_with_backend(&session_config, backend.as_ref()).await?)
            },
        };

        let auth_interceptor = auth::AuthInterceptor::new(config.auth.mode.clone(), auth::TokenStore::new());
        log_auth_mode_snapshot(&config.auth.mode);
        let service = flight::DatafusionCudfFlightSqlService::new(
            Arc::clone(&ctx),
            native_optimizer_config.clone(),
            config.session_profile,
            admission,
            config.statement_ticket_ttl,
            auth_interceptor,
            #[cfg(feature = "cugraph")]
            config.cugraph.as_ref().map(|cugraph| {
                session::cugraph_sql_config(cugraph, native_optimizer_config.native_execution_config().clone())
            }),
        )
        .with_server_config_diagnostics(&config);
        let local_addr = matches!(&listen, Listen::Prebound(_)).then_some(addr);
        Ok(FlightSqlServer { service, listen, shutdown: self.shutdown, ctx, backend, local_addr })
    }
}

/// Initialize the process-wide CUDA device context for `ordinal`.
///
/// The first CUDA call in a process implicitly runs `cuInit()`, which establishes
/// the driver context — a one-time cost of a couple of seconds. [`serve`] triggers
/// this lazily while constructing the query-service admission controller; calling
/// it up front pays the cost before the server accepts connections, so binding is
/// not delayed by the cold start. The device count is cached, so
/// repeated calls within a process are cheap.
pub fn init_cuda_device(ordinal: usize) -> Result<()> {
    cudf::DeviceId::new(ordinal)
        .map(|_| ())
        .map_err(|error| ServerError::session("failed to initialize CUDA device context", error))
}

/// Start the Flight SQL server and serve until SIGINT/SIGTERM.
///
/// Graceful shutdown sequence on signal:
/// 1. Close the admission gate so pending and future execution admits
///    fail fast instead of acquiring a permit during drain.
/// 2. Cancel the server-wide shutdown [`tokio_util::sync::CancellationToken`].
///    Every per-statement token is a child of this token, so cancellation
///    cascades to statements already executing as well as tickets still waiting
///    for `DoGet`.
/// 3. Let tonic stop accepting new connections and drain existing RPCs; the
///    `serve_with_shutdown` future returns once the last RPC finishes.
pub async fn serve(config: ServerConfig) -> Result<()> {
    FlightSqlServer::builder().config(config).build().await?.serve().await
}

/// Serve on an already-bound listener instead of binding `config.bind_addr`.
///
/// This lets a caller reserve the port up front and hand the live socket to the
/// server, eliminating the bind/serve gap. Unlike [`serve`], the TCP socket accepts
/// connections before CUDA/session/admission startup finishes, so an early client
/// succeeds at the TCP layer and its first RPC blocks until gRPC is ready rather
/// than being refused. `config.bind_addr` is ignored in favour of the listener's address.
pub async fn serve_with_listener(config: ServerConfig, listener: std::net::TcpListener) -> Result<()> {
    FlightSqlServer::builder().config(config).listener(listener).build().await?.serve().await
}

/// How [`serve_inner`] obtains its TCP socket.
enum Listen {
    /// Bind `addr` only once startup is complete (matches historical `serve`
    /// behaviour: full init, then bind, so the socket never exists half-ready).
    Addr(std::net::SocketAddr),
    /// Adopt a listener the caller already bound.
    Prebound(std::net::TcpListener),
}

impl Listen {
    fn addr(&self) -> Result<SocketAddr> {
        match self {
            Listen::Addr(addr) => Ok(*addr),
            Listen::Prebound(listener) => listener
                .local_addr()
                .map_err(|error| ServerError::session("failed to read Flight SQL listener address", error)),
        }
    }
}

async fn serve_built_server(server: FlightSqlServer) -> Result<()> {
    let FlightSqlServer { service, listen, shutdown, .. } = server;
    let addr = listen.addr()?;
    let admission = Arc::clone(service.admission());
    let cancellations = Arc::clone(service.cancellations());
    let shutdown_token = service.shutdown_token().clone();
    tracing::info!(
        bind_addr = %addr,
        max_in_flight = admission.max_in_flight(),
        "starting datafusion-nexus Flight SQL server"
    );

    let (shutdown, watchdog_rx) = shutdown_future(shutdown, admission, cancellations, shutdown_token);

    let builder = Server::builder().add_service(FlightServiceServer::new(service));
    let server_result = match listen {
        Listen::Addr(addr) => builder.serve_with_shutdown(addr, shutdown).await,
        Listen::Prebound(listener) => {
            listener
                .set_nonblocking(true)
                .map_err(|error| ServerError::session("failed to set Flight SQL listener non-blocking", error))?;
            let tokio_listener = tokio::net::TcpListener::from_std(listener)
                .map_err(|error| ServerError::session("failed to adopt Flight SQL listener", error))?;
            // `TcpIncoming::from` leaves TCP_NODELAY unset; tonic's own
            // address-bound path defaults it to enabled, so restore it to keep
            // gRPC latency behaviour identical across the two entry points.
            let incoming = TcpIncoming::from(tokio_listener).with_nodelay(Some(true));
            builder.serve_with_incoming_shutdown(incoming, shutdown).await
        },
    };

    // Abort the watchdog if it was armed. If the serve future returned before
    // the shutdown future ran (e.g. a bind/listen error), the oneshot sender
    // was dropped — `recv` returns Err and there is nothing to abort.
    if let Some(mut watchdog_rx) = watchdog_rx
        && let Ok(watchdog) = (&mut watchdog_rx).await
    {
        watchdog.abort();
    }
    server_result.map_err(|error| ServerError::transport_at_addr(addr, error))?;

    tracing::info!("datafusion-nexus Flight SQL server stopped");
    Ok(())
}

type ShutdownFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

fn shutdown_future(
    shutdown: ServerShutdown,
    admission: Arc<admission::QueryAdmission>,
    cancellations: Arc<cancel::QueryCancellation>,
    shutdown_token: CancellationToken,
) -> (ShutdownFuture, Option<tokio::sync::oneshot::Receiver<tokio::task::JoinHandle<()>>>) {
    match shutdown {
        ServerShutdown::OsSignals => {
            // Hand back the escalation-watchdog handle so we can abort it once
            // the server has drained. Without this, a SIGINT received after a
            // clean shutdown would still trigger the spawned `process::exit`.
            let (watchdog_tx, watchdog_rx) = tokio::sync::oneshot::channel();
            let shutdown = async move {
                wait_for_shutdown_signal().await;
                admission.close();
                let in_flight = cancellations.len();
                shutdown_token.cancel();
                tracing::info!(in_flight, "received shutdown signal; draining datafusion-nexus Flight SQL server");
                let watchdog = tokio::spawn(async {
                    wait_for_shutdown_signal().await;
                    tracing::warn!("received second shutdown signal during drain; forcing immediate exit");
                    std::process::exit(130);
                });
                let _ = watchdog_tx.send(watchdog);
            };
            (Box::pin(shutdown), Some(watchdog_rx))
        },
        ServerShutdown::Token(token) => {
            let shutdown = async move {
                token.cancelled().await;
                admission.close();
                let in_flight = cancellations.len();
                shutdown_token.cancel();
                tracing::info!(in_flight, "received shutdown token; draining datafusion-nexus Flight SQL server");
            };
            (Box::pin(shutdown), None)
        },
    }
}

fn validate_injected_backend_matches_config(backend: &NexusGpuBackend, config: &ServerConfig) -> Result<()> {
    let MemoryControlPolicy::Bounded { device_budget_bytes: config_device_budget_bytes, .. } =
        config.native_optimizer_config.native_execution_config().memory_control_policy()
    else {
        return Err(ServerError::builder_validation(
            "server_backend_unbounded_config",
            "an injected backend requires bounded native memory policy in the server config",
        ));
    };

    let service_config = backend.query_service().config();
    let normalized_admission_policy = admission::server_admission_policy(config);
    if config.max_in_flight != service_config.max_concurrent_queries {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "max_in_flight",
            config.max_in_flight,
            service_config.max_concurrent_queries,
        ));
    }
    if config.gpu_devices != backend.device_ordinals() {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "gpu_devices",
            format_usize_list(&config.gpu_devices),
            format_usize_list(backend.device_ordinals()),
        ));
    }
    if normalized_admission_policy.slot_guarantee() != service_config.slot_guarantee {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "slot_guarantee",
            slot_guarantee_value(normalized_admission_policy.slot_guarantee()),
            slot_guarantee_value(service_config.slot_guarantee),
        ));
    }
    if normalized_admission_policy.max_queries_per_device() != service_config.max_queries_per_device {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.max_queries_per_device",
            format_optional_usize(normalized_admission_policy.max_queries_per_device()),
            format_optional_usize(service_config.max_queries_per_device),
        ));
    }
    if normalized_admission_policy.placement_policy() != service_config.placement_policy {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.placement_policy",
            placement_policy_value(normalized_admission_policy.placement_policy()),
            placement_policy_value(service_config.placement_policy),
        ));
    }
    if normalized_admission_policy.max_queued() != service_config.max_admission_queue_len {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.max_queued",
            normalized_admission_policy.max_queued(),
            service_config.max_admission_queue_len,
        ));
    }
    if normalized_admission_policy.wait_timeout() != service_config.admission_wait_timeout {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.wait_timeout",
            normalized_admission_policy.wait_timeout().as_millis(),
            service_config.admission_wait_timeout.as_millis(),
        ));
    }
    let backend_capacity = service_config.device_capacity_bytes.ok_or_else(|| {
        ServerError::builder_validation(
            "server_backend_admission_conflict",
            "injected backend does not expose a bounded device capacity",
        )
    })?;
    if config_device_budget_bytes != backend_capacity {
        let field = if config.device_budget_fraction.is_some() {
            config::NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV
        } else {
            config::NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV
        };
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            field,
            config_device_budget_bytes,
            backend_capacity,
        ));
    }

    let config_grant = normalized_admission_policy
        .resolve_grant_band_bytes(
            backend_capacity,
            config.gpu_devices.len(),
            config.native_optimizer_config.native_execution_config().native_pipeline_execution_mode(),
        )
        .map_err(|error| ServerError::session("failed to resolve server admission grant band", error))?;
    let MemoryControlPolicy::Bounded { device_budget_bytes: backend_min, .. } =
        backend.optimizer_config().native_execution_config().memory_control_policy()
    else {
        return Err(ServerError::builder_validation(
            "server_backend_admission_conflict",
            "injected backend native optimizer config is not bounded",
        ));
    };
    let backend_want = backend
        .query_service()
        .tenant_budget(backend.tenant_id())
        .map_err(|error| ServerError::session("failed to inspect injected backend tenant budget", error))?;
    if config_grant.min_bytes != backend_min {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.grant_min",
            config_grant.min_bytes,
            backend_min,
        ));
    }
    if config_grant.want_bytes != backend_want {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.grant_want",
            config_grant.want_bytes,
            backend_want,
        ));
    }
    let config_fallback_limit = config.native_optimizer_config.cpu_fallback_max_concurrent_statements();
    let backend_fallback_limit = backend.optimizer_config().cpu_fallback_max_concurrent_statements();
    if config_fallback_limit != backend_fallback_limit {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "fallback.max_concurrent_statements",
            config_fallback_limit.map(|value| value.get()).unwrap_or(0),
            backend_fallback_limit.map(|value| value.get()).unwrap_or(0),
        ));
    }
    let config_fallback_timeout = config_fallback_limit.and(config.native_optimizer_config.cpu_fallback_wait_timeout());
    let backend_fallback_timeout = backend_fallback_limit.and(backend.optimizer_config().cpu_fallback_wait_timeout());
    if config_fallback_timeout != backend_fallback_timeout {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "fallback.wait_timeout_ms",
            config_fallback_timeout.map(|value| value.as_millis()).unwrap_or(0),
            backend_fallback_timeout.map(|value| value.as_millis()).unwrap_or(0),
        ));
    }
    Ok(())
}

fn validate_injected_session_matches_backend(ctx: &SessionContext, backend: &NexusGpuBackend) -> Result<()> {
    let state = ctx.state();
    let Some(optimizer_config) = crate::session::installed_cudf_native_optimizer_config(&state) else {
        return Err(ServerError::builder_validation(
            "server_session_backend_mismatch",
            "injected session_context does not have the datafusion-nexus native optimizer installed",
        ));
    };
    if !optimizer_config
        .native_execution_config()
        .query_admission_source_matches(backend.query_service(), backend.tenant_id())
    {
        return Err(ServerError::builder_validation(
            "server_session_backend_mismatch",
            "injected session_context is not installed on the injected backend query service",
        ));
    }
    Ok(())
}

fn format_usize_list(values: &[usize]) -> String {
    values.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn slot_guarantee_value(slot_guarantee: nexus_query_engine::runtime::SlotGuarantee) -> &'static str {
    match slot_guarantee {
        nexus_query_engine::runtime::SlotGuarantee::GuaranteedMinimum => "guaranteed_minimum",
        nexus_query_engine::runtime::SlotGuarantee::Optimistic => "optimistic",
    }
}

fn format_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn placement_policy_value(policy: nexus_query_engine::runtime::QueryPlacementPolicy) -> String {
    match policy {
        nexus_query_engine::runtime::QueryPlacementPolicy::LeastLoaded => "least_loaded".to_string(),
        nexus_query_engine::runtime::QueryPlacementPolicy::BestFit => "best_fit".to_string(),
        nexus_query_engine::runtime::QueryPlacementPolicy::Pinned { device_ordinal } => {
            format!("pinned:{device_ordinal}")
        },
        _ => "unknown".to_string(),
    }
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
                cpu_fallback_max_concurrent_statements = config
                    .cpu_fallback_max_concurrent_statements
                    .map(|value| value.get())
                    .unwrap_or(0),
                cpu_fallback_wait_timeout_ms = config
                    .cpu_fallback_wait_timeout
                    .map(|value| value.as_millis())
                    .unwrap_or(0),
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
                native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
                native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
                cugraph_enabled = cugraph_enabled(config),
                cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
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
            cpu_fallback_max_concurrent_statements = config
                .cpu_fallback_max_concurrent_statements
                .map(|value| value.get())
                .unwrap_or(0),
            cpu_fallback_wait_timeout_ms = config
                .cpu_fallback_wait_timeout
                .map(|value| value.as_millis())
                .unwrap_or(0),
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
            native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
            native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
            cugraph_enabled = cugraph_enabled(config),
            cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
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
        cpu_fallback_max_concurrent_statements = config
            .cpu_fallback_max_concurrent_statements
            .map(|value| value.get())
            .unwrap_or(0),
        cpu_fallback_wait_timeout_ms = config
            .cpu_fallback_wait_timeout
            .map(|value| value.as_millis())
            .unwrap_or(0),
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
        native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
        native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
        cugraph_enabled = cugraph_enabled(config),
        cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(),
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
    crate::graph_sql::CUGRAPH_OPERATION_REGISTRY.len()
}

#[cfg(not(feature = "cugraph"))]
fn cugraph_registry_algorithm_count() -> usize {
    0
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
/// and a WARN when [`AuthMode::Disabled`] is selected so operators
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
    use std::num::NonZeroUsize;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        common::config::ConfigOptions,
        logical_expr::Operator,
        physical_expr::expressions::{binary, col},
        physical_optimizer::PhysicalOptimizerRule,
        physical_plan::{
            ExecutionPlan,
            empty::EmptyExec,
            projection::{ProjectionExec, ProjectionExpr},
        },
    };
    use nexus_query_engine::runtime::{QueryPlacementPolicy, SlotGuarantee};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        backend::{GrantBand, MemoryShare, NexusAdmissionPolicy, NexusMemoryPolicy},
        native_exec::NativeCudfExecutionConfig,
        planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig},
        policy::FallbackReason,
        report::CandidateOutcome,
    };

    const TEST_DEVICE_BUDGET_BYTES: u64 = 4096;
    const TEST_MAX_SOURCE_CHUNK_BYTES: u64 = 1024;
    const TEST_MAX_ROW_GROUPS_PER_CHUNK: usize = 1;
    const TEST_MAX_RETRY_ATTEMPTS: usize = 3;
    const TEST_MIN_RETRY_CHUNK_ROW_GROUPS: usize = 1;

    fn base_config(profile: ServerSessionProfile) -> ServerConfig {
        ServerConfig {
            bind_addr: "127.0.0.1:0".parse().expect("test bind address"),
            log_filter: "info".to_string(),
            max_in_flight: 1,
            admission_policy: crate::backend::NexusAdmissionPolicy::default().with_max_concurrent_queries(1),
            cpu_fallback_max_concurrent_statements: None,
            cpu_fallback_wait_timeout: None,
            gpu_devices: vec![0],
            device_budget_fraction: None,
            statement_ticket_ttl: std::time::Duration::from_secs(60),
            native_optimizer_config: CudfOptimizerConfig::default(),
            session_profile: profile,
            auth: auth::AuthConfig::default(),
            workspace: None,
            #[cfg(feature = "iceberg")]
            iceberg: None,
            #[cfg(feature = "cugraph")]
            cugraph: None,
        }
    }

    fn bounded_config(max_concurrent_queries: usize) -> ServerConfig {
        let mut config = base_config(ServerSessionProfile::Native);
        config.max_in_flight = max_concurrent_queries;
        config.admission_policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries);
        config.native_optimizer_config =
            CudfOptimizerConfig::default().with_native_execution_config(NativeCudfExecutionConfig::bounded(
                TEST_DEVICE_BUDGET_BYTES,
                TEST_MAX_SOURCE_CHUNK_BYTES,
                TEST_MAX_ROW_GROUPS_PER_CHUNK,
                TEST_MAX_RETRY_ATTEMPTS,
                TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
            ));
        config
    }

    fn unbounded_config() -> ServerConfig {
        let mut config = base_config(ServerSessionProfile::Native);
        config.max_in_flight = 1;
        config.admission_policy = NexusAdmissionPolicy::default().with_max_concurrent_queries(1);
        config.native_optimizer_config =
            CudfOptimizerConfig::default().with_native_execution_config(NativeCudfExecutionConfig::unbounded());
        config
    }

    fn admission_only_backend(max_concurrent_queries: usize) -> NexusGpuBackend {
        NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: TEST_DEVICE_BUDGET_BYTES,
                max_source_chunk_bytes: TEST_MAX_SOURCE_CHUNK_BYTES,
                max_row_groups_per_chunk: TEST_MAX_ROW_GROUPS_PER_CHUNK,
                max_retry_attempts: TEST_MAX_RETRY_ATTEMPTS,
                min_retry_chunk_row_groups: TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
            })
            .admission(NexusAdmissionPolicy::default().with_max_concurrent_queries(max_concurrent_queries))
            .build_admission_only()
            .expect("admission-only backend should build")
    }

    fn empty_exec_plan() -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(Arc::new(Schema::empty())))
    }

    fn unsupported_projection_plan() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("m00", DataType::Int64, false),
            Field::new("m01", DataType::Int64, false),
        ]));
        let input = Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>;
        let remainder =
            binary(col("m00", &schema).unwrap(), Operator::Modulo, col("m01", &schema).unwrap(), &schema).unwrap();

        Arc::new(
            ProjectionExec::try_new([ProjectionExpr { expr: remainder, alias: "remainder".to_string() }], input)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn builder_rejects_session_context_without_injected_backend() {
        let config = bounded_config(1);
        let ctx = Arc::new(SessionContext::new());

        let error = builder_error(FlightSqlServer::builder().config(config).session_context(ctx)).await;

        assert_eq!(error.fact_value("reason"), Some("server_session_without_backend"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_with_unbounded_config() {
        let backend = admission_only_backend(1);

        let error = builder_error(FlightSqlServer::builder().config(unbounded_config()).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_unbounded_config"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_admission_conflict() {
        let backend = admission_only_backend(1);

        let error = builder_error(FlightSqlServer::builder().config(bounded_config(2)).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("max_in_flight"));
    }

    #[tokio::test]
    async fn builder_validation_uses_normalized_max_in_flight_for_default_queue_limit() {
        let mut config = bounded_config(1);
        config.max_in_flight = 2;
        let backend = NexusGpuBackend::from_optimizer_config_for_admission_only(
            config.native_optimizer_config.clone(),
            config.gpu_devices.clone(),
            admission::server_admission_policy(&config),
            nexus_query_engine::runtime::TenantId::new("default"),
        )
        .unwrap()
        .expect("bounded config should build backend");

        validate_injected_backend_matches_config(&backend, &config).unwrap();
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_grant_band_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config
            .admission_policy
            .with_grant_band(GrantBand { min: MemoryShare::Bytes(256), want: MemoryShare::Bytes(2048) });

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("admission.grant_min"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_slot_guarantee_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config.admission_policy.with_slot_guarantee(SlotGuarantee::Optimistic);

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("slot_guarantee"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_per_device_cap_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config.admission_policy.with_max_queries_per_device(Some(1));

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("admission.max_queries_per_device"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_placement_policy_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config.admission_policy.with_placement_policy(QueryPlacementPolicy::BestFit);

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("admission.placement_policy"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_queue_limit_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config.admission_policy.with_max_queued(3);

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("admission.max_queued"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_wait_timeout_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.admission_policy = config.admission_policy.with_wait_timeout(std::time::Duration::from_secs(5));

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("admission.wait_timeout"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_gpu_device_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.gpu_devices = vec![1];

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("gpu_devices"));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_device_budget_fraction_conflict() {
        let backend = admission_only_backend(1);
        let mut config = bounded_config(1);
        config.device_budget_fraction = Some(0.5);
        config.native_optimizer_config =
            CudfOptimizerConfig::default().with_native_execution_config(NativeCudfExecutionConfig::bounded(
                TEST_DEVICE_BUDGET_BYTES / 2,
                TEST_MAX_SOURCE_CHUNK_BYTES,
                TEST_MAX_ROW_GROUPS_PER_CHUNK,
                TEST_MAX_RETRY_ATTEMPTS,
                TEST_MIN_RETRY_CHUNK_ROW_GROUPS,
            ));

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some(config::NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV));
    }

    #[tokio::test]
    async fn builder_rejects_injected_backend_fallback_wait_timeout_conflict() {
        let mut backend_config = bounded_config(1);
        backend_config.native_optimizer_config = backend_config
            .native_optimizer_config
            .with_cpu_fallback_wait_timeout(std::time::Duration::from_millis(7000))
            .with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
        let backend = NexusGpuBackend::from_optimizer_config_for_admission_only(
            backend_config.native_optimizer_config.clone(),
            backend_config.gpu_devices.clone(),
            admission::server_admission_policy(&backend_config),
            nexus_query_engine::runtime::TenantId::new("default"),
        )
        .unwrap()
        .expect("bounded config should build backend");

        let mut config = bounded_config(1);
        config.native_optimizer_config = config
            .native_optimizer_config
            .with_cpu_fallback_wait_timeout(std::time::Duration::from_millis(5000))
            .with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());

        let error = builder_error(FlightSqlServer::builder().config(config).backend(backend)).await;

        assert_eq!(error.fact_value("reason"), Some("server_backend_admission_conflict"));
        assert_eq!(error.fact_value("config_field"), Some("fallback.wait_timeout_ms"));
    }

    #[tokio::test]
    async fn build_session_context_with_backend_installs_backend_admission_source() {
        let config = bounded_config(1);
        let backend = admission_only_backend(1);

        let ctx = build_session_context_with_backend(&config, Some(&backend)).await.unwrap();
        let state = ctx.state();
        let optimizer_config =
            crate::session::installed_cudf_native_optimizer_config(&state).expect("native optimizer installed");

        assert!(
            optimizer_config
                .native_execution_config()
                .query_admission_source_matches(backend.query_service(), backend.tenant_id())
        );
    }

    #[test]
    fn server_cpu_fallback_limit_reaches_optimizer_fallback_plan() {
        let mut config = bounded_config(1);
        config.native_optimizer_config =
            config.native_optimizer_config.with_cpu_fallback_max_concurrent_statements(NonZeroUsize::new(1).unwrap());
        let rule = CudfNativePhysicalOptimizerRule::new(config.native_optimizer_config.clone());
        let fallback_plan = unsupported_projection_plan();

        let optimized = rule.optimize(fallback_plan, &ConfigOptions::new()).unwrap();
        let fallback_report = rule.last_report().expect("optimizer should retain a fallback planning report");

        assert_eq!(fallback_report.rewritten_count(), 0, "outcomes={:?}", fallback_report.outcomes());
        assert!(matches!(
            fallback_report.outcomes(),
            [CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }]
        ));
        assert!(
            crate::native_exec::plan_is_cpu_fallback_limited(optimized.as_ref()),
            "configured server fallback limit must wrap non-rewritten CPU fallback plans"
        );

        let optimized_empty = rule.optimize(empty_exec_plan(), &ConfigOptions::new()).unwrap();
        let empty_report = rule.last_report().expect("optimizer should retain an empty-input planning report");

        assert_eq!(empty_report.rewritten_count(), 1, "outcomes={:?}", empty_report.outcomes());
        assert_eq!(empty_report.rejected_count(), 0, "outcomes={:?}", empty_report.outcomes());
        assert!(
            !crate::native_exec::plan_is_cpu_fallback_limited(optimized_empty.as_ref()),
            "rewritten EmptyExec plans must not take the CPU fallback limiter"
        );
    }

    #[tokio::test]
    async fn builder_installs_unbounded_native_execution_limit() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut config = unbounded_config();
        config.bind_addr = listener.local_addr().unwrap();
        let server = FlightSqlServer::builder()
            .config(config)
            .listener(listener)
            .shutdown(ServerShutdown::Token(CancellationToken::new()))
            .build()
            .await
            .unwrap();

        assert_eq!(
            server
                .service
                .native_optimizer_config()
                .native_execution_config()
                .unbounded_native_max_concurrent_queries()
                .map(|value| value.get()),
            Some(1)
        );
        let state = server.session_context().state();
        let optimizer_config =
            crate::session::installed_cudf_native_optimizer_config(&state).expect("native optimizer installed");
        assert_eq!(
            optimizer_config
                .native_execution_config()
                .unbounded_native_max_concurrent_queries()
                .map(|value| value.get()),
            Some(1)
        );
    }

    #[tokio::test]
    async fn builder_rejects_injected_session_context_from_different_backend() {
        let config = bounded_config(1);
        let session_backend = admission_only_backend(1);
        let server_backend = admission_only_backend(1);
        let ctx = Arc::new(build_session_context_with_backend(&config, Some(&session_backend)).await.unwrap());

        let error =
            builder_error(FlightSqlServer::builder().config(config).backend(server_backend).session_context(ctx)).await;

        assert_eq!(error.fact_value("reason"), Some("server_session_backend_mismatch"));
    }

    async fn builder_error(builder: FlightSqlServerBuilder) -> ServerError {
        match builder.build().await {
            Ok(_) => panic!("builder should reject invalid configuration"),
            Err(error) => error,
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
            parquet_read_strategy_config: crate::table_format::TableFormatParquetReadStrategyConfig::default(),
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
