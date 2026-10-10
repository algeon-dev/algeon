use std::{sync::Arc, time::Duration};

use arrow::record_batch::RecordBatch;
#[cfg(test)]
use datafusion::execution::SessionStateBuilder;

use super::{
    AuthInterceptor, CancellationToken, CommandGetTables, DataFusionError, DataFusionErrorExt, ErrorLogDiagnostics,
    FailedStatement, GetTablesBuilder, GetTablesPrefilters, Owner, PreparedStatementStore, QueryAdmission,
    QueryCancellation, QueryCoordinator, QueryCoordinatorConfig, QueryOwnership, QueryPlanningLease, Request,
    RuntimeObservationConfig, Schema, SessionContext, Status, adapter_error_status,
    arrow_flight_error_to_status_for_operation, datafusion_error_to_status_for_operation, table_type_to_flight_sql,
};
#[cfg(test)]
use crate::server::{
    auth::{AuthMode, TokenStore},
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
};
use crate::server::{
    config::{ServerConfig, ServerObservabilityConfig},
    io_lifecycle::ServerIoLifecycle,
};

pub struct DatafusionAlgeonFlightSqlService {
    pub(super) ctx: Arc<SessionContext>,
    pub(super) admission: Arc<QueryAdmission>,
    pub(super) io_lifecycle: Arc<ServerIoLifecycle>,
    pub(super) cancellations: Arc<QueryCancellation>,
    pub(super) coordinator: Arc<QueryCoordinator>,
    pub(super) runtime_observation_config: RuntimeObservationConfig,
    /// Live prepared statements, keyed by handle. Populated by
    /// `do_action_create_prepared_statement` and removed by
    /// `do_action_close_prepared_statement`. The handle is an opaque UUID and
    /// is distinct from the per-execution `query_id` issued at
    /// `get_flight_info_*` time; one prepared statement can be executed
    /// multiple times, each producing a fresh `query_id` / Flight ticket.
    pub(super) prepared_statements: Arc<PreparedStatementStore>,
    /// Principal that issued each live `query_id` and prepared-statement
    /// handle. Consulted before every lookup by id so an authenticated caller
    /// cannot redeem, cancel, or read another principal's query.
    pub(super) ownership: Arc<QueryOwnership>,
    pub(super) statement_ticket_ttl: Duration,
    /// Parent token that every per-statement token is derived from. Cancelled
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so statements already
    /// registered for execution and tickets still waiting for `DoGet` see
    /// cancellation at the first await.
    pub(super) shutdown_token: CancellationToken,
    /// Flight/application-layer bearer-token validator. The underlying token
    /// store is shared with the handshake path and never reaches GPU control.
    pub(super) auth: AuthInterceptor,
}

impl DatafusionAlgeonFlightSqlService {
    pub(super) fn register_new_query(
        &self,
        sql: String,
        correlation_id: Option<String>,
    ) -> Result<(String, QueryPlanningLease), (String, Status)> {
        self.coordinator.register_new(sql, correlation_id, self.runtime_observation_config).map_err(|error| {
            let query_id = error
                .query_id()
                .expect("coordinator-owned registration failures retain their allocated query id")
                .to_string();
            let status =
                adapter_error_status(error.adapter_error("algeon_datafusion.server.flight.query_coordinator.register"));
            (query_id, status)
        })
    }

    /// Register a Flight result-stream invocation whose execution attempt
    /// owns runtime observation. The coordinator remains responsible for
    /// tickets, cancellation, and retained projections, but it must not mint
    /// an independent observation owner for the same invocation.
    pub(super) fn register_new_attempt_query(
        &self,
        sql: String,
        correlation_id: Option<String>,
    ) -> Result<(String, QueryPlanningLease), (String, Status)> {
        let registration = if self.admission.has_attempt_backend() {
            self.coordinator.register_new_with_attempt_observation(sql, correlation_id, self.runtime_observation_config)
        } else {
            // CPU-only unit fixtures intentionally have no runtime backend to
            // begin an attempt against. Production server construction always
            // takes the attempt-owned branch above.
            self.coordinator.register_new(sql, correlation_id, self.runtime_observation_config)
        };
        registration.map_err(|error| {
            let query_id = error
                .query_id()
                .expect("coordinator-owned registration failures retain their allocated query id")
                .to_string();
            let status =
                adapter_error_status(error.adapter_error("algeon_datafusion.server.flight.query_coordinator.register"));
            (query_id, status)
        })
    }

    #[cfg(test)]
    pub(super) fn register_query(
        &self,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
    ) -> Result<QueryPlanningLease, Status> {
        let registration = if self.admission.has_attempt_backend() {
            self.coordinator.register_with_id_with_attempt_observation(
                query_id,
                sql,
                correlation_id,
                self.runtime_observation_config,
            )
        } else {
            self.coordinator.register_with_id(query_id, sql, correlation_id, self.runtime_observation_config)
        };
        registration.map_err(|error| {
            adapter_error_status(error.adapter_error("algeon_datafusion.server.flight.query_coordinator.register"))
        })
    }

    #[cfg(test)]
    pub fn new(ctx: Arc<SessionContext>, admission: Arc<QueryAdmission>, statement_ticket_ttl: Duration) -> Self {
        let io_lifecycle = Arc::new(ServerIoLifecycle::new());
        Self::new_with_io_lifecycle(
            ctx,
            admission,
            statement_ticket_ttl,
            AuthInterceptor::new(AuthMode::Disabled, TokenStore::new()),
            io_lifecycle,
        )
    }

    pub(crate) fn new_with_io_lifecycle(
        ctx: Arc<SessionContext>,
        admission: Arc<QueryAdmission>,
        statement_ticket_ttl: Duration,
        auth: AuthInterceptor,
        io_lifecycle: Arc<ServerIoLifecycle>,
    ) -> Self {
        let observability_config = ServerObservabilityConfig::default();
        Self {
            ctx,
            admission,
            io_lifecycle,
            cancellations: Arc::default(),
            coordinator: QueryCoordinator::new(QueryCoordinatorConfig::from_observability(
                &observability_config,
                ServerObservabilityConfig::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES,
            )),
            runtime_observation_config: RuntimeObservationConfig::native_execution(),
            prepared_statements: Arc::default(),
            ownership: Arc::default(),
            statement_ticket_ttl,
            shutdown_token: CancellationToken::new(),
            auth,
        }
    }

    pub(crate) fn with_server_config_diagnostics(
        mut self,
        config: &ServerConfig,
        active_logical_capture_budget_bytes: u64,
    ) -> Self {
        self.runtime_observation_config = config.observability.runtime_observation_config();
        self.coordinator = QueryCoordinator::new(QueryCoordinatorConfig::from_observability(
            &config.observability,
            active_logical_capture_budget_bytes,
        ));
        self
    }

    /// Validate the bearer token on a Flight request and resolve the caller to
    /// the [`Owner`] that ids are registered against. Disabled auth keeps the
    /// existing trusted-network behavior: every caller is [`Owner::Anonymous`].
    pub(super) fn check_auth<T>(&self, request: &Request<T>) -> Result<Owner, Status> {
        self.auth.check(request.metadata()).map(|principal| Owner::from_principal(principal.as_ref()))
    }

    /// Authorize `caller` to name `id`, then return the caller so the
    /// authorized id and its owner stay bound at the call site.
    pub(super) fn authorize_id<T>(&self, request: &Request<T>, id: &str) -> Result<Owner, Status> {
        let caller = self.check_auth(request)?;
        self.ownership.authorize(id, &caller)?;
        Ok(caller)
    }

    pub(super) async fn build_get_tables_response(
        &self,
        query: CommandGetTables,
    ) -> Result<(Arc<Schema>, RecordBatch), Status> {
        let include_schema = query.include_schema;
        let filters = GetTablesPrefilters::from(&query);
        let mut builder = GetTablesBuilder::from(query);
        let empty_schema = Schema::empty();
        for catalog_name in self.ctx.catalog_names() {
            if !filters.catalog_matches(&catalog_name) {
                continue;
            }

            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                if !filters.schema_matches(&schema_name) {
                    continue;
                }

                let Some(schema_provider) = catalog.schema(&schema_name) else {
                    continue;
                };
                for table_name in schema_provider.table_names() {
                    if !filters.table_name_matches(&table_name) {
                        continue;
                    }

                    let (table_schema, table_type_str) = if include_schema && !filters.has_table_type_filter() {
                        match schema_provider.table(&table_name).await {
                            Ok(Some(table)) => (Some(table.schema()), table_type_to_flight_sql(table.table_type())),
                            Ok(None) => continue,
                            Err(error) => {
                                Self::warn_get_tables_metadata_error(
                                    &catalog_name,
                                    &schema_name,
                                    &table_name,
                                    "load_table",
                                    &error,
                                );
                                continue;
                            },
                        }
                    } else {
                        let table_type = match schema_provider.table_type(&table_name).await {
                            Ok(Some(table_type)) => table_type,
                            Ok(None) => continue,
                            Err(error) if include_schema => {
                                Self::warn_get_tables_metadata_error(
                                    &catalog_name,
                                    &schema_name,
                                    &table_name,
                                    "table_type",
                                    &error,
                                );
                                continue;
                            },
                            Err(error) => {
                                return Err(datafusion_error_to_status_for_operation(
                                    error,
                                    "algeon_datafusion.server.flight.do_get_tables.table_type",
                                    FailedStatement::Other,
                                ));
                            },
                        };

                        let table_type_str = table_type_to_flight_sql(table_type);
                        if !filters.table_type_matches(table_type_str) {
                            continue;
                        }

                        let table_schema = if include_schema {
                            match schema_provider.table(&table_name).await {
                                Ok(Some(table)) => Some(table.schema()),
                                Ok(None) => None,
                                Err(error) => {
                                    Self::warn_get_tables_metadata_error(
                                        &catalog_name,
                                        &schema_name,
                                        &table_name,
                                        "load_table",
                                        &error,
                                    );
                                    None
                                },
                            }
                        } else {
                            None
                        };
                        if include_schema && table_schema.is_none() {
                            continue;
                        }
                        (table_schema, table_type_str)
                    };

                    let table_schema_ref: &Schema = table_schema.as_deref().unwrap_or(&empty_schema);
                    builder
                        .append(&catalog_name, &schema_name, &table_name, table_type_str, table_schema_ref)
                        .map_err(|error| {
                            arrow_flight_error_to_status_for_operation(
                                error,
                                "algeon_datafusion.server.flight.do_get_tables.append",
                                "failed to append Flight SQL GetTables metadata row",
                            )
                        })?;
                }
            }
        }
        let schema = builder.schema();
        let batch = builder.build().map_err(|error| {
            arrow_flight_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.do_get_tables.build",
                "failed to build Flight SQL GetTables metadata batch",
            )
        })?;
        Ok((schema, batch))
    }

    pub(super) fn warn_get_tables_metadata_error(
        catalog_name: &str,
        schema_name: &str,
        table_name: &str,
        failure: &'static str,
        error: &DataFusionError,
    ) {
        if let Some(adapter_error) = error.find_algeon_error() {
            let diagnostics = ErrorLogDiagnostics::from_error(adapter_error);
            tracing::warn!(
                catalog = %catalog_name,
                schema = %schema_name,
                table = %table_name,
                failure,
                error_code = adapter_error.code().as_str(),
                error_kind = adapter_error.kind().as_str(),
                error_status = adapter_error.status().as_str(),
                error_operation = adapter_error.operation(),
                error_message = adapter_error.message(),
                error_cause = diagnostics.cause().filter(|cause| *cause != adapter_error.message()),
                error_source_path = diagnostics.source_path(),
                error_diagnostics_truncated = diagnostics.truncated().then_some(true),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        } else {
            let diagnostics = ErrorLogDiagnostics::from_error(error);
            tracing::warn!(
                catalog = %catalog_name,
                schema = %schema_name,
                table = %table_name,
                failure,
                error_cause = diagnostics.cause(),
                error_source_path = diagnostics.source_path(),
                error_diagnostics_truncated = diagnostics.truncated().then_some(true),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        }
    }

    pub(crate) fn admission(&self) -> &Arc<QueryAdmission> {
        &self.admission
    }

    pub(crate) fn io_lifecycle(&self) -> &Arc<ServerIoLifecycle> {
        &self.io_lifecycle
    }

    pub(crate) fn cancellations(&self) -> &Arc<QueryCancellation> {
        &self.cancellations
    }

    pub(crate) fn coordinator(&self) -> &Arc<QueryCoordinator> {
        &self.coordinator
    }

    pub(crate) fn shutdown_token(&self) -> &CancellationToken {
        &self.shutdown_token
    }
}

#[cfg(test)]
pub(super) struct BoundedTestFlightSqlService {
    service: DatafusionAlgeonFlightSqlService,
    _backend: AlgeonGpuBackend,
}

#[cfg(test)]
pub(super) fn default_test_backend() -> AlgeonGpuBackend {
    AlgeonGpuBackend::builder()
        .with_native_execution_config(AlgeonNativeExecutionConfig::new(
            SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
        ))
        .devices([AlgeonGpuDeviceConfig::new(
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )])
        .queue_policy(AlgeonGpuQueuePolicy::default())
        .build()
        .expect("flight test backend should build")
}

#[cfg(test)]
impl std::ops::Deref for BoundedTestFlightSqlService {
    type Target = DatafusionAlgeonFlightSqlService;

    fn deref(&self) -> &Self::Target {
        &self.service
    }
}

#[cfg(test)]
pub(super) fn bounded_test_service(statement_ticket_ttl: Duration) -> BoundedTestFlightSqlService {
    bounded_test_service_with_capture_budget(
        statement_ticket_ttl,
        ServerObservabilityConfig::DEFAULT_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES,
    )
}

#[cfg(test)]
pub(super) fn bounded_test_service_with_capture_budget(
    statement_ticket_ttl: Duration,
    max_active_logical_capture_bytes: u64,
) -> BoundedTestFlightSqlService {
    let backend = default_test_backend();
    let state = backend
        .install_on(SessionStateBuilder::new_with_default_features())
        .expect("bounded Flight SQL test backend should install")
        .build();
    let ctx = Arc::new(SessionContext::new_with_state(state));
    let admission = Arc::new(QueryAdmission::from_backend(&backend));
    let mut service = DatafusionAlgeonFlightSqlService::new(ctx, admission, statement_ticket_ttl);
    service.coordinator = QueryCoordinator::new(QueryCoordinatorConfig::from_observability(
        &ServerObservabilityConfig::default(),
        max_active_logical_capture_bytes,
    ));
    BoundedTestFlightSqlService { service, _backend: backend }
}
