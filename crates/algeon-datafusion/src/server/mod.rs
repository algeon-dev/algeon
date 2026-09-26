//! Opt-in Flight SQL server for `algeon-datafusion`.
//!
//! DataFusion applications can embed this module directly. The workspace's
//! `algeon-server` package is only a standalone process launcher over this
//! API; configuration, session construction, Flight SQL behavior, and error
//! projection are owned here.

#[cfg(feature = "iceberg")]
use std::borrow::Cow;
use std::{net::SocketAddr, sync::Arc, time::Instant};

use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::prelude::SessionContext;
use futures::future::BoxFuture;
use query_engine::runtime::{GpuBackendDeviceStartupRecord, MemoryControlPolicy};
use tokio::{net::TcpListener, signal::ctrl_c, sync::oneshot, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use tonic::transport::{Server, server::TcpIncoming};

#[cfg(feature = "iceberg")]
use crate::server::table_format::scan::credential_free_endpoint_identity;
use crate::server::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceInstallationBounds, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy,
        GpuMemoryOwnership,
    },
    cugraph_sql::CUGRAPH_OPERATION_REGISTRY,
    planner::DataFusionCpuExecutionPolicy,
};
#[cfg(feature = "iceberg")]
pub(crate) use crate::table_format;
pub(crate) use crate::{
    attempt, backend, capability, cugraph_sql, cuml_sql, cuvs_sql, error as adapter_error, gpu_coverage, native_exec,
    native_report, planner, report, session as adapter_session,
};

mod admission;
mod auth;
mod cancel;
mod config;
mod coordinator;
mod diagnostics;
mod error;
mod flight;
mod io_lifecycle;
mod observability;
mod session;
mod workspace;

pub use auth::{AuthConfig, AuthMode, PasswordHashError, StaticUserConfig, hash_password, verify_password};
pub use config::{
    ALGEON_ICEBERG_CATALOG_KIND_ENV, ALGEON_ICEBERG_CATALOG_NAME_ENV, ALGEON_ICEBERG_GLUE_CATALOG_ID_ENV,
    ALGEON_ICEBERG_GLUE_CATALOG_NAME_ENV, ALGEON_ICEBERG_GLUE_URI_ENV, ALGEON_ICEBERG_NAMESPACE_ENV,
    ALGEON_ICEBERG_REST_CATALOG_HANDLE_ENV, ALGEON_ICEBERG_REST_PREFIX_ENV, ALGEON_ICEBERG_REST_TOKEN_ENV,
    ALGEON_ICEBERG_REST_URI_ENV, ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, ALGEON_ICEBERG_S3_ENDPOINT_ENV,
    ALGEON_ICEBERG_S3_PATH_STYLE_ENV, ALGEON_ICEBERG_S3_REGION_ENV, ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    ALGEON_ICEBERG_TABLES_ENV, ALGEON_ICEBERG_WAREHOUSE_ENV, ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV,
    ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV, ALGEON_SERVER_BIND_ENV, ALGEON_SERVER_CONFIG_FILE_ENV,
    ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, ALGEON_SERVER_CUGRAPH_ENABLED_ENV, ALGEON_SERVER_EXECUTION_MODE_ENV,
    ALGEON_SERVER_FINAL_PLAN_REQUIREMENT_ENV, ALGEON_SERVER_GPU_DEVICES_ENV, ALGEON_SERVER_LOG_ENV,
    ALGEON_SERVER_NATIVE_CONFIG_OPTIONS, ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
    ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
    ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
    ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
    ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
    ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
    ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
    ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
    ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, ALGEON_SERVER_WORKSPACE_BACKING_ALIASES_ENV,
    ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV, ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
    ALGEON_SERVER_WORKSPACE_CATALOG_ENV, ALGEON_SERVER_WORKSPACE_SCHEMA_ENV, AlgeonServerNativeConfigOption,
    CugraphServerConfig, ServerConfig, ServerNativeConfigScope, ServerObservabilityConfig, WorkspaceOverlayConfig,
};
#[cfg(feature = "iceberg")]
pub use config::{
    GlueServerBackendConfig, IcebergCatalogBackendConfig, IcebergServerConfig, IcebergTableRegistrationConfig,
};
pub use error::{
    ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY, ERROR_OPERATION_METADATA_KEY, ERROR_STATUS_METADATA_KEY,
    NATIVE_MESSAGE_METADATA_KEY, PREDICATE_ROLE_METADATA_KEY, REASON_METADATA_KEY, REQUIRED_OPTION_METADATA_KEY,
    Result, ServerError, UNSUPPORTED_COLUMN_METADATA_KEY, UNSUPPORTED_COLUMN_ROLE_METADATA_KEY,
    UNSUPPORTED_COLUMN_TYPE_METADATA_KEY,
};
pub use session::build_session_context_with_backend;

/// Build the server's bounded query-service backend from a validated config.
pub fn backend_from_server_config(config: &ServerConfig) -> Result<AlgeonGpuBackend> {
    let devices = config.resolve_gpu_devices()?;
    AlgeonGpuBackend::builder()
        .with_optimizer_config(config.native_optimizer_config.clone())
        .devices(devices)
        .queue_policy(admission::server_queue_policy(config))
        .build()
        .map_err(|error| ServerError::session("failed to initialize query service admission", error))
}

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
    backend: Option<AlgeonGpuBackend>,
    session_context: Option<Arc<SessionContext>>,
    listener: Option<std::net::TcpListener>,
    shutdown: ServerShutdown,
}

/// Embeddable Flight SQL server assembled from a [`ServerConfig`].
pub struct FlightSqlServer {
    service: flight::DatafusionAlgeonFlightSqlService,
    listen: Listen,
    shutdown: ServerShutdown,
    ctx: Arc<SessionContext>,
    backend: Option<AlgeonGpuBackend>,
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
    pub fn backend(&self) -> Option<&AlgeonGpuBackend> {
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

    pub fn backend(mut self, backend: AlgeonGpuBackend) -> Self {
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
        // `ServerConfig::from_env*` validates this during TOML loading, but
        // callers can also construct the public config struct directly.
        config.auth.mode.validate()?;
        ensure_single_librmm_runtime_mapping()?;
        let listen = match self.listener {
            Some(listener) => Listen::Prebound(listener),
            None => Listen::Addr(config.bind_addr),
        };
        let addr = listen.addr()?;
        if self.session_context.is_some() && self.backend.is_none() {
            return Err(ServerError::builder_validation(
                "server_session_without_backend",
                "an injected server session_context requires an injected backend",
            ));
        }

        let backend = match self.backend {
            Some(backend) => {
                validate_injected_backend_matches_config(&backend, &config)?;
                backend
            },
            None => backend_from_server_config(&config)?,
        };
        let mut effective_config = config.clone();
        effective_config.native_optimizer_config = backend.optimizer_config().clone();
        log_server_config_snapshot(&effective_config);
        warn_if_cugraph_built_but_disabled(&effective_config);

        let admission_init_start = Instant::now();
        let admission = Arc::new(admission::QueryAdmission::from_backend(&backend));
        tracing::info!(
            elapsed_ms = u64::try_from(admission_init_start.elapsed().as_millis()).unwrap_or(u64::MAX),
            "query admission controller initialized"
        );
        admission.log_query_service_state(None, "server_startup");

        let mut native_optimizer_config = effective_config.native_optimizer_config.clone();
        let io_lifecycle = Arc::new(io_lifecycle::ServerIoLifecycle::new());
        let native_execution_config = native_optimizer_config
            .native_execution_config()
            .clone()
            .with_io_lifecycle(io_lifecycle.handle())
            .with_host_output_memory_pool_accounting(true);
        native_optimizer_config = native_optimizer_config.with_native_execution_config(native_execution_config);

        let ctx = match self.session_context {
            Some(ctx) => {
                validate_injected_session_matches_server_config(&ctx, &backend, &effective_config)?;
                ctx
            },
            None => Arc::new(session::build_session_context_with_backend(&effective_config, &backend).await?),
        };

        let auth_interceptor = auth::AuthInterceptor::new(effective_config.auth.mode.clone(), auth::TokenStore::new());
        log_auth_mode_snapshot(&effective_config.auth.mode);
        let service = flight::DatafusionAlgeonFlightSqlService::new_with_io_lifecycle(
            Arc::clone(&ctx),
            native_optimizer_config.clone(),
            admission,
            effective_config.statement_ticket_ttl,
            auth_interceptor,
            io_lifecycle,
        )
        .with_server_config_diagnostics(&effective_config);
        let local_addr = matches!(&listen, Listen::Prebound(_)).then_some(addr);
        Ok(FlightSqlServer { service, listen, shutdown: self.shutdown, ctx, backend: Some(backend), local_addr })
    }
}

/// Initialize the process-wide CUDA device context for `ordinal`.
///
/// The first CUDA call in a process implicitly runs `cuInit()`, which establishes
/// the driver context; a one-time cost of a couple of seconds. [`serve`] triggers
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

/// How [`FlightSqlServer`] obtains its TCP socket.
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
    let io_lifecycle = Arc::clone(service.io_lifecycle());
    let cancellations = Arc::clone(service.cancellations());
    let coordinator = Arc::clone(service.coordinator());
    let _admission_close_guard = AdmissionCloseGuard { admission: Arc::clone(&admission) };
    let shutdown_token = service.shutdown_token().clone();
    let maintenance = coordinator::QueryCoordinatorMaintenance::start(Arc::clone(&coordinator));
    let mut shutdown_supervisor = ShutdownSupervisor::start(
        Arc::clone(&admission),
        Arc::clone(&io_lifecycle),
        Arc::clone(&coordinator),
        maintenance,
    );
    tracing::info!(bind_addr = %addr, "starting Algeon Flight SQL server");

    let (shutdown, watchdog_rx) =
        shutdown_future(shutdown, shutdown_supervisor.trigger(), cancellations, shutdown_token);

    let builder = Server::builder().add_service(FlightServiceServer::new(service));
    let server_result = match listen {
        Listen::Addr(addr) => builder.serve_with_shutdown(addr, shutdown).await,
        Listen::Prebound(listener) => {
            listener
                .set_nonblocking(true)
                .map_err(|error| ServerError::session("failed to set Flight SQL listener non-blocking", error))?;
            let tokio_listener = TcpListener::from_std(listener)
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
    // was dropped; `recv` returns Err and there is nothing to abort.
    if let Some(mut watchdog_rx) = watchdog_rx
        && let Ok(watchdog) = (&mut watchdog_rx).await
    {
        watchdog.abort();
    }
    // The serve future can return before its shutdown future runs (for
    // example, on a transport error). Start the same supervised shutdown path
    // there too; begin is idempotent and preserves the first absolute deadline.
    shutdown_supervisor.begin_shutdown(Instant::now());
    let shutdown_result = shutdown_supervisor.finish().await;
    server_result.map_err(|error| ServerError::transport_at_addr(addr, error))?;
    shutdown_result?;

    tracing::info!("Algeon Flight SQL server stopped");
    Ok(())
}

/// Closes admission even when the caller cancels or aborts the server future.
///
/// The normal shutdown path closes admission before coordinator drain. This
/// guard extends the same ordering to future cancellation, where execution
/// never resumes after the tonic serve await.
struct AdmissionCloseGuard {
    admission: Arc<admission::QueryAdmission>,
}

#[derive(Clone)]
struct ShutdownSupervisorTrigger {
    admission: Arc<admission::QueryAdmission>,
    io_lifecycle: Arc<io_lifecycle::ServerIoLifecycle>,
    coordinator: Arc<coordinator::QueryCoordinator>,
    plan: Arc<std::sync::OnceLock<coordinator::QueryCoordinatorShutdownPlan>>,
    started: CancellationToken,
}

impl ShutdownSupervisorTrigger {
    fn begin_shutdown(&self, now: Instant) -> coordinator::QueryCoordinatorShutdownPlan {
        let plan = *self.plan.get_or_init(|| {
            self.io_lifecycle.close();
            after_closing_admission(&self.admission, || self.coordinator.begin_shutdown(now))
        });
        self.started.cancel();
        plan
    }

    #[cfg(test)]
    fn install_shutdown_plan(
        &self,
        plan: coordinator::QueryCoordinatorShutdownPlan,
    ) -> coordinator::QueryCoordinatorShutdownPlan {
        let plan = *self.plan.get_or_init(|| {
            self.io_lifecycle.close();
            self.admission.close();
            plan
        });
        self.started.cancel();
        plan
    }
}

/// Owns the complete server teardown under the coordinator's first absolute
/// shutdown deadline. Coordinator drain and native query-service cleanup run
/// concurrently, so neither component receives a second grace period.
struct ShutdownSupervisor {
    trigger: ShutdownSupervisorTrigger,
    task: Option<JoinHandle<Result<()>>>,
}

impl ShutdownSupervisor {
    fn start(
        admission: Arc<admission::QueryAdmission>,
        io_lifecycle: Arc<io_lifecycle::ServerIoLifecycle>,
        coordinator: Arc<coordinator::QueryCoordinator>,
        maintenance: coordinator::QueryCoordinatorMaintenance,
    ) -> Self {
        let trigger = ShutdownSupervisorTrigger {
            admission: Arc::clone(&admission),
            io_lifecycle: Arc::clone(&io_lifecycle),
            coordinator: Arc::clone(&coordinator),
            plan: Arc::new(std::sync::OnceLock::new()),
            started: CancellationToken::new(),
        };
        let worker_trigger = trigger.clone();
        let task = tokio::spawn(async move {
            worker_trigger.started.cancelled().await;
            let plan = *worker_trigger.plan.get().expect("shutdown plan is installed before its start signal");
            let deadline = plan.deadline();
            let (maintenance_result, admission_result, io_result) = tokio::join!(
                maintenance.finish(plan),
                admission.close_and_wait_for_quiescence_until(deadline),
                io_lifecycle.close_wait_and_drain_until(deadline),
            );
            let forced = maintenance_result.map_err(|error| {
                ServerError::shutdown_with_source(
                    "coordinator_maintenance_task_failed",
                    "query coordinator maintenance task failed during shutdown",
                    error,
                )
            })?;
            admission_result?;
            io_result?;
            if forced {
                return Err(ServerError::shutdown(
                    "coordinator_shutdown_timeout",
                    "query coordinator shutdown deadline expired; retained incomplete query diagnostics",
                ));
            }
            if Instant::now() >= deadline {
                return Err(ServerError::shutdown(
                    "server_shutdown_deadline_timeout",
                    "server shutdown completed after its absolute deadline",
                ));
            }
            Ok(())
        });
        Self { trigger, task: Some(task) }
    }

    fn trigger(&self) -> ShutdownSupervisorTrigger {
        self.trigger.clone()
    }

    #[cfg(test)]
    fn start_with_plan(
        admission: Arc<admission::QueryAdmission>,
        io_lifecycle: Arc<io_lifecycle::ServerIoLifecycle>,
        coordinator: Arc<coordinator::QueryCoordinator>,
        maintenance: coordinator::QueryCoordinatorMaintenance,
        plan: coordinator::QueryCoordinatorShutdownPlan,
    ) -> Self {
        let supervisor = Self::start(admission, io_lifecycle, coordinator, maintenance);
        supervisor.trigger.install_shutdown_plan(plan);
        supervisor
    }

    fn begin_shutdown(&self, now: Instant) -> coordinator::QueryCoordinatorShutdownPlan {
        self.trigger.begin_shutdown(now)
    }

    async fn finish(&mut self) -> Result<()> {
        self.begin_shutdown(Instant::now());
        let task = self.task.take().ok_or_else(|| {
            ServerError::shutdown("shutdown_supervisor_joined_twice", "shutdown supervisor was joined more than once")
        })?;
        task.await.map_err(|error| {
            ServerError::shutdown_with_source(
                "shutdown_supervisor_task_failed",
                "shutdown supervisor task failed",
                error,
            )
        })?
    }
}

impl Drop for ShutdownSupervisor {
    fn drop(&mut self) {
        // If the outer serve future is cancelled, detach the already-owned
        // cleanup task after triggering the same ordered teardown.
        self.begin_shutdown(Instant::now());
    }
}

impl Drop for AdmissionCloseGuard {
    fn drop(&mut self) {
        self.admission.close();
    }
}

type ShutdownFuture = BoxFuture<'static, ()>;

fn shutdown_future(
    shutdown: ServerShutdown,
    supervisor: ShutdownSupervisorTrigger,
    cancellations: Arc<cancel::QueryCancellation>,
    shutdown_token: CancellationToken,
) -> (ShutdownFuture, Option<oneshot::Receiver<JoinHandle<()>>>) {
    match shutdown {
        ServerShutdown::OsSignals => {
            // Hand back the escalation-watchdog handle so we can abort it once
            // the server has drained. Without this, a SIGINT received after a
            // clean shutdown would still trigger the spawned `process::exit`.
            let (watchdog_tx, watchdog_rx) = oneshot::channel();
            let shutdown = async move {
                wait_for_shutdown_signal().await;
                let shutdown_plan = supervisor.begin_shutdown(Instant::now());
                let in_flight = cancellations.len();
                shutdown_token.cancel();
                tracing::info!(
                    in_flight,
                    coordinator_live = shutdown_plan.live_count(),
                    "received shutdown signal; draining Algeon Flight SQL server"
                );
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
                let shutdown_plan = supervisor.begin_shutdown(Instant::now());
                let in_flight = cancellations.len();
                shutdown_token.cancel();
                tracing::info!(
                    in_flight,
                    coordinator_live = shutdown_plan.live_count(),
                    "received shutdown token; draining Algeon Flight SQL server"
                );
            };
            (Box::pin(shutdown), None)
        },
    }
}

fn after_closing_admission<T>(admission: &admission::QueryAdmission, action: impl FnOnce() -> T) -> T {
    admission.close();
    action()
}

fn validate_injected_backend_matches_config(backend: &AlgeonGpuBackend, config: &ServerConfig) -> Result<()> {
    validate_injected_queue_policy(backend.queue_policy(), config)?;
    validate_injected_startup_profiles(backend.device_startup_records(), config)
}

fn validate_injected_queue_policy(injected: &AlgeonGpuQueuePolicy, config: &ServerConfig) -> Result<()> {
    let queue_policy = admission::server_queue_policy(config);
    if queue_policy.max_queued_attempts() != injected.max_queued_attempts() {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.max_queued_attempts",
            queue_policy.max_queued_attempts(),
            injected.max_queued_attempts(),
        ));
    }
    if queue_policy.max_queue_wait() != injected.max_queue_wait() {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.max_queue_wait_ms",
            queue_policy.max_queue_wait().as_millis(),
            injected.max_queue_wait().as_millis(),
        ));
    }
    Ok(())
}

/// Validate injected backends against the records frozen by the engine after
/// discovery and allocator installation.  A planner budget is intentionally
/// absent from this comparison: it is an unadmitted planning template, while
/// capacity and query limits are ordinal-local controller facts.
fn validate_injected_startup_profiles(records: &[GpuBackendDeviceStartupRecord], config: &ServerConfig) -> Result<()> {
    let configured_profiles = configured_profile_map(config)?;
    let configured_ownership = configured_ownership_map(config)?;
    if records.len() != configured_profiles.len() {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.device_profiles",
            configured_profiles.len(),
            records.len(),
        ));
    }
    for record in records {
        let ordinal = record.device_ordinal;
        let Some(profile) = configured_profiles.get(&ordinal) else {
            return Err(ServerError::builder_conflict(
                "server_backend_admission_conflict",
                "admission.device_profiles",
                format_usize_list(&config.gpu_devices),
                ordinal,
            ));
        };
        let expected_ownership = configured_ownership.get(&ordinal).ok_or_else(|| {
            ServerError::builder_validation(
                "server_backend_admission_conflict",
                "server config has no deployment ownership declaration for an injected device profile",
            )
            .with_fact("device_ordinal", ordinal)
        })?;
        compare_profile_field(
            ordinal,
            "gpu_memory_ownership",
            format!("{expected_ownership:?}"),
            format!("{:?}", record.memory_ownership),
        )?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.memory_limit",
            format!("{:?}", profile.memory_limit()),
            format!("{:?}", record.memory_limit),
        )?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.backend_reserve_bytes",
            profile.backend_reserve_bytes(),
            record.backend_reserve_bytes,
        )?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.max_active_attempts",
            profile.max_active_attempts(),
            record.resource_profile.max_active_attempts,
        )?;
        if let Some(default_query_limit_bytes) = profile.default_query_limit_bytes() {
            compare_profile_field(
                ordinal,
                "admission.device_profiles.default_query_limit_bytes",
                default_query_limit_bytes,
                record.resource_profile.default_query_limit_bytes,
            )?;
        }
        let source_work = profile.source_work_per_attempt();
        compare_profile_field(
            ordinal,
            "admission.device_profiles.source_work_per_attempt",
            source_work,
            record.resource_profile.source_work_per_attempt,
        )?;
        let bounds = expected_profile_bounds(profile)?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.stream_pool_size",
            bounds.stream_pool_size,
            record.stream_pool_size,
        )?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.worker_count",
            bounds.worker_count,
            record.worker_count,
        )?;
        compare_profile_field(ordinal, "admission.device_profiles.lane_count", bounds.lane_count, record.lane_count)?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.capabilities",
            format!("{:?}", profile.device_capabilities()),
            format!("{:?}", record.resource_profile.device_capabilities),
        )?;
        compare_profile_field(
            ordinal,
            "admission.device_profiles.install_telemetry",
            profile.install_telemetry(),
            record.install_telemetry,
        )?;
    }
    Ok(())
}

fn configured_profile_map(config: &ServerConfig) -> Result<std::collections::BTreeMap<usize, &AlgeonGpuDeviceProfile>> {
    let selected = config.gpu_devices.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let mut profiles = std::collections::BTreeMap::new();
    for profile in &config.gpu_device_profiles {
        let ordinal = profile.device_ordinal();
        if profiles.insert(ordinal, profile).is_some() {
            return Err(ServerError::builder_validation(
                "server_backend_admission_conflict",
                "server config contains duplicate device profiles",
            )
            .with_fact("device_ordinal", ordinal));
        }
    }
    let configured = profiles.keys().copied().collect::<std::collections::BTreeSet<_>>();
    if configured != selected {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "admission.device_profiles",
            format_usize_list(&config.gpu_devices),
            format_usize_list(&profiles.keys().copied().collect::<Vec<_>>()),
        ));
    }
    Ok(profiles)
}

fn configured_ownership_map(config: &ServerConfig) -> Result<std::collections::BTreeMap<usize, GpuMemoryOwnership>> {
    let mut ownership = std::collections::BTreeMap::new();
    for (ordinal, value) in config.gpu_memory_ownership.iter().copied() {
        if ownership.insert(ordinal, value).is_some() {
            return Err(ServerError::builder_validation(
                "server_backend_admission_conflict",
                "server config contains duplicate GPU memory ownership declarations",
            )
            .with_fact("device_ordinal", ordinal));
        }
    }
    let selected = config.gpu_devices.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let declared = ownership.keys().copied().collect::<std::collections::BTreeSet<_>>();
    if declared != selected {
        return Err(ServerError::builder_conflict(
            "server_backend_admission_conflict",
            "gpu_memory_ownership",
            format_usize_list(&config.gpu_devices),
            format_usize_list(&ownership.keys().copied().collect::<Vec<_>>()),
        ));
    }
    Ok(ownership)
}

fn expected_profile_bounds(profile: &AlgeonGpuDeviceProfile) -> Result<AlgeonGpuDeviceInstallationBounds> {
    if let Some(bounds) = profile.installation_bounds() {
        return Ok(bounds);
    }
    let auxiliary = profile.max_active_attempts().checked_mul(profile.source_work_per_attempt()).ok_or_else(|| {
        ServerError::builder_validation(
            "server_backend_admission_conflict",
            "configured source-work capacity overflowed while validating an injected backend",
        )
        .with_fact("device_ordinal", profile.device_ordinal())
    })?;
    let stream_pool_size = profile.max_active_attempts().checked_add(auxiliary).ok_or_else(|| {
        ServerError::builder_validation(
            "server_backend_admission_conflict",
            "configured stream capacity overflowed while validating an injected backend",
        )
        .with_fact("device_ordinal", profile.device_ordinal())
    })?;
    Ok(AlgeonGpuDeviceInstallationBounds::new(
        stream_pool_size,
        profile.max_active_attempts(),
        profile.max_active_attempts(),
    ))
}

fn compare_profile_field(
    device_ordinal: usize,
    field: &'static str,
    configured: impl ToString,
    actual: impl ToString,
) -> Result<()> {
    let configured = configured.to_string();
    let actual = actual.to_string();
    if configured == actual {
        return Ok(());
    }
    Err(ServerError::builder_conflict("server_backend_admission_conflict", field, configured, actual)
        .with_fact("device_ordinal", device_ordinal))
}

fn validate_injected_session_matches_server_config(
    ctx: &SessionContext,
    backend: &AlgeonGpuBackend,
    config: &ServerConfig,
) -> Result<()> {
    validate_injected_session_backend_binding(backend.is_session_state_bound(&ctx.state()))?;
    let requires_single_partition =
        config.native_optimizer_config.datafusion_cpu_execution_policy() == DataFusionCpuExecutionPolicy::Reject;
    let target_partitions = ctx.state().config_options().execution.target_partitions;
    if requires_single_partition && target_partitions != 1 {
        return Err(ServerError::builder_validation(
            "server_session_optimizer_policy_mismatch",
            "injected session_context target partitions were not derived from the strict injected backend",
        )
        .with_fact("config_field", "datafusion.execution.target_partitions")
        .with_fact("expected_value", 1)
        .with_fact("actual_value", target_partitions));
    }
    Ok(())
}

fn validate_injected_session_backend_binding(source_matches: bool) -> Result<()> {
    if !source_matches {
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

#[cfg(feature = "iceberg")]
fn log_server_config_snapshot(config: &ServerConfig) {
    let native_execution = config.native_optimizer_config.native_execution_config();
    let native_memory = NativeMemorySnapshot::from(native_execution.memory_control_policy());
    let source_scheduler = native_execution.source_scheduler_policy();
    match &config.iceberg {
        Some(iceberg) => {
            let snapshot = iceberg_backend_snapshot(&iceberg.backend, iceberg.source_access_policy());
            tracing::info!(
                bind_addr = %config.bind_addr,
                statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
                observability_per_query_max_records = config.observability.per_query_max_records,
                observability_per_query_max_bytes = config.observability.per_query_max_bytes,
                observability_active_logical_capture_max_bytes = config.observability.active_logical_capture_max_bytes,
                observability_global_sealed_max_count = config.observability.global_sealed_max_count,
                observability_global_sealed_max_bytes = config.observability.global_sealed_max_bytes,
                observability_sealed_ttl_secs = config.observability.sealed_ttl.as_secs(),
                observability_shutdown_grace_secs = config.observability.shutdown_grace.as_secs(),
                native_memory_control_policy = %native_execution.memory_control_policy(),
                native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
                native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
                native_max_retry_attempts = %native_memory.max_retry_attempts,
                native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
                native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
                native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
                cugraph_enabled = cugraph_enabled(config),
                cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(config),
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
            statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
            observability_per_query_max_records = config.observability.per_query_max_records,
            observability_per_query_max_bytes = config.observability.per_query_max_bytes,
            observability_active_logical_capture_max_bytes = config.observability.active_logical_capture_max_bytes,
            observability_global_sealed_max_count = config.observability.global_sealed_max_count,
            observability_global_sealed_max_bytes = config.observability.global_sealed_max_bytes,
            observability_sealed_ttl_secs = config.observability.sealed_ttl.as_secs(),
            observability_shutdown_grace_secs = config.observability.shutdown_grace.as_secs(),
            native_memory_control_policy = %native_execution.memory_control_policy(),
            native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
            native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
            native_max_retry_attempts = %native_memory.max_retry_attempts,
            native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
            native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
            native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
            cugraph_enabled = cugraph_enabled(config),
            cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(config),
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
        statement_ticket_ttl_secs = config.statement_ticket_ttl.as_secs(),
        observability_per_query_max_records = config.observability.per_query_max_records,
        observability_per_query_max_bytes = config.observability.per_query_max_bytes,
        observability_active_logical_capture_max_bytes = config.observability.active_logical_capture_max_bytes,
        observability_global_sealed_max_count = config.observability.global_sealed_max_count,
        observability_global_sealed_max_bytes = config.observability.global_sealed_max_bytes,
        observability_sealed_ttl_secs = config.observability.sealed_ttl.as_secs(),
        observability_shutdown_grace_secs = config.observability.shutdown_grace.as_secs(),
        native_memory_control_policy = %native_execution.memory_control_policy(),
        native_max_source_chunk_bytes = %native_memory.max_source_chunk_bytes,
        native_max_row_groups_per_chunk = %native_memory.max_row_groups_per_chunk,
        native_max_retry_attempts = %native_memory.max_retry_attempts,
        native_max_source_reads_in_flight = source_scheduler.max_source_reads_in_flight(),
        native_source_chunk_planning_mode = %native_execution.source_chunk_planning_mode(),
        native_statistical_aggregate_execution_mode = %native_execution.statistical_aggregate_execution_mode(),
        cugraph_enabled = cugraph_enabled(config),
        cugraph_registry_algorithm_count = cugraph_registry_algorithm_count(config),
        "server config snapshot"
    );
}

fn cugraph_enabled(config: &ServerConfig) -> bool {
    config.cugraph.is_some()
}

/// Emit a one-shot startup WARN when this binary was built with the `cugraph`
/// feature but cuGraph UDTFs were disabled at runtime
/// (`ALGEON_SERVER_CUGRAPH_ENABLED=false`). Without it the
/// the `cugraph_*` relation planner is not installed and queries fail with a
/// not-found error, which is easy to misread as a build problem when the
/// feature is actually compiled in. The cfg-gated definitions keep this a
/// no-op on feature-off builds, where enabling the runtime setting instead
/// registers the public frontend and returns its capability error at execution.
#[cfg(feature = "cugraph")]
fn warn_if_cugraph_built_but_disabled(config: &ServerConfig) {
    if config.cugraph.is_some() {
        return;
    }
    tracing::warn!(
        cugraph_enabled = false,
        "Algeon Flight SQL server was built with the `cugraph` feature \
         but cuGraph UDTFs are disabled at runtime. Set \
         ALGEON_SERVER_CUGRAPH_ENABLED=true to enable the cugraph_* SQL planner."
    );
}

#[cfg(not(feature = "cugraph"))]
fn warn_if_cugraph_built_but_disabled(_config: &ServerConfig) {}

fn cugraph_registry_algorithm_count(config: &ServerConfig) -> usize {
    config.cugraph.as_ref().map_or(0, |_| CUGRAPH_OPERATION_REGISTRY.len())
}

/// Emit a one-shot startup line summarizing the configured application-layer
/// authentication mode. Authentication is intentionally owned by Flight RPCs,
/// never by GPU admission or controller code.
fn log_auth_mode_snapshot(mode: &AuthMode) {
    match mode {
        AuthMode::Disabled => tracing::warn!(
            auth_mode = "disabled",
            "Algeon Flight SQL server starting with auth DISABLED; every RPC will be accepted"
        ),
        AuthMode::BasicBearer { users, token_ttl_secs } => tracing::info!(
            auth_mode = "basic_bearer",
            user_count = users.len(),
            token_ttl_secs = *token_ttl_secs,
            "Algeon Flight SQL server starting with Basic+Bearer auth"
        ),
    }
}

struct NativeMemorySnapshot {
    max_source_chunk_bytes: String,
    max_row_groups_per_chunk: String,
    max_retry_attempts: String,
}

impl From<MemoryControlPolicy> for NativeMemorySnapshot {
    fn from(policy: MemoryControlPolicy) -> Self {
        match policy {
            MemoryControlPolicy::Unbounded => Self {
                max_source_chunk_bytes: "none".to_string(),
                max_row_groups_per_chunk: "none".to_string(),
                max_retry_attempts: "none".to_string(),
            },
            MemoryControlPolicy::Bounded {
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                ..
            } => Self {
                max_source_chunk_bytes: max_source_chunk_bytes.to_string(),
                max_row_groups_per_chunk: max_row_groups_per_chunk.to_string(),
                max_retry_attempts: max_retry_attempts.to_string(),
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
fn iceberg_backend_snapshot<'a>(
    backend: &'a IcebergCatalogBackendConfig,
    source_access_policy: &'a crate::server::table_format::iceberg::SourceAccessPolicy,
) -> IcebergBackendSnapshot<'a> {
    match backend {
        IcebergCatalogBackendConfig::Glue(glue) => IcebergBackendSnapshot {
            catalog_type: "glue",
            warehouse: &glue.warehouse,
            glue_catalog_handle: &glue.catalog_handle,
            glue_region: &glue.region,
            glue_catalog_id: option_display(glue.catalog_id.as_deref()),
            aws_profile: option_display(glue.profile.as_deref()),
            rest_uri: Cow::Borrowed("none"),
            rest_s3_endpoint: Cow::Borrowed("none"),
        },
        IcebergCatalogBackendConfig::Rest(rest) => IcebergBackendSnapshot {
            catalog_type: "rest",
            warehouse: &rest.warehouse,
            glue_catalog_handle: "none",
            glue_region: "none",
            glue_catalog_id: "none",
            aws_profile: "none",
            rest_uri: Cow::Owned(credential_free_endpoint_identity(rest.uri())),
            rest_s3_endpoint: source_access_policy
                .s3_access()
                .endpoint
                .map(credential_free_endpoint_identity)
                .map(Cow::Owned)
                .unwrap_or(Cow::Borrowed("none")),
        },
    }
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
            let _ = ctrl_c().await;
            return;
        },
    };
    tokio::select! {
        result = ctrl_c() => {
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
mod tests;
