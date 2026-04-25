use super::*;

pub struct DatafusionCudfFlightSqlService {
    pub(super) ctx: Arc<SessionContext>,
    pub(super) native_optimizer_config: CudfOptimizerConfig,
    /// Static session-profile selection from [`crate::server::ServerConfig`];
    /// per-statement diagnostics use it to add the "switch to native" hint
    /// when the testing-only `cpu_fallback` profile is active.
    pub(super) session_profile: ServerSessionProfile,
    pub(super) admission: Arc<QueryAdmission>,
    pub(super) cancellations: Arc<QueryCancellation>,
    pub(super) observability: Arc<QueryObservability>,
    pub(super) tickets: Arc<DashMap<String, StatementTicket>>,
    /// Map of prepared-statement handle → [`PreparedStatementEntry`].
    /// Populated by `do_action_create_prepared_statement` and removed by
    /// `do_action_close_prepared_statement` (or replaced when the same
    /// handle is reused). The handle is an opaque UUID and is distinct
    /// from the per-execution `query_id` issued at `get_flight_info_*`
    /// time; one prepared statement can be executed multiple times,
    /// each producing a fresh `query_id` / Flight ticket.
    pub(super) prepared_statements: Arc<DashMap<String, PreparedStatementEntry>>,
    pub(super) statement_ticket_ttl: Duration,
    /// Parent token that every per-statement token is derived from. Cancelled
    /// by [`crate::server::serve`] on SIGINT/SIGTERM so statements already
    /// registered for execution and tickets still waiting for `DoGet` see
    /// cancellation at the first await.
    pub(super) shutdown_token: CancellationToken,
    /// Bearer-token validator + handshake credential store. Cloned into
    /// the service from [`crate::server::serve`] so the same
    /// [`TokenStore`] is shared between the handshake mint path and the
    /// per-RPC validate path.
    pub(super) auth: AuthInterceptor,
    pub(super) catalog_diagnostics: ServerCatalogDiagnostics,
    #[cfg(feature = "cugraph")]
    pub(super) cugraph_sql_config: Option<CugraphSqlConfig>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ServerCatalogDiagnostics {
    pub(super) iceberg_enabled: bool,
    pub(super) workspace_catalog: Option<String>,
    pub(super) workspace_schema: Option<String>,
}

impl ServerCatalogDiagnostics {
    pub(super) fn from_server_config(config: &crate::server::config::ServerConfig) -> Self {
        Self {
            iceberg_enabled: server_config_iceberg_enabled(config),
            workspace_catalog: config.workspace.as_ref().map(|workspace| workspace.catalog.clone()),
            workspace_schema: config.workspace.as_ref().map(|workspace| workspace.schema.clone()),
        }
    }
}

#[cfg(feature = "iceberg")]
fn server_config_iceberg_enabled(config: &crate::server::config::ServerConfig) -> bool {
    config.iceberg.is_some()
}

#[cfg(not(feature = "iceberg"))]
fn server_config_iceberg_enabled(_config: &crate::server::config::ServerConfig) -> bool {
    false
}

/// Per-statement session context paired with the retained native rule that
/// will plan it. The native rule reference is kept alive for the lifetime of
/// planning so [`CudfNativePhysicalOptimizerRule::last_report`] can be read
impl DatafusionCudfFlightSqlService {
    pub fn new(
        ctx: Arc<SessionContext>,
        native_optimizer_config: CudfOptimizerConfig,
        session_profile: ServerSessionProfile,
        admission: Arc<QueryAdmission>,
        statement_ticket_ttl: Duration,
        auth: AuthInterceptor,
        #[cfg(feature = "cugraph")] cugraph_sql_config: Option<CugraphSqlConfig>,
    ) -> Self {
        Self {
            ctx,
            native_optimizer_config,
            session_profile,
            admission,
            cancellations: Arc::default(),
            observability: Arc::default(),
            tickets: Arc::default(),
            prepared_statements: Arc::default(),
            statement_ticket_ttl,
            shutdown_token: CancellationToken::new(),
            auth,
            catalog_diagnostics: ServerCatalogDiagnostics::default(),
            #[cfg(feature = "cugraph")]
            cugraph_sql_config,
        }
    }

    pub(in crate::server) fn with_server_config_diagnostics(
        mut self,
        config: &crate::server::config::ServerConfig,
    ) -> Self {
        self.catalog_diagnostics = ServerCatalogDiagnostics::from_server_config(config);
        self
    }

    #[cfg(all(test, feature = "iceberg"))]
    pub(super) fn with_catalog_diagnostics(mut self, diagnostics: ServerCatalogDiagnostics) -> Self {
        self.catalog_diagnostics = diagnostics;
        self
    }

    /// Validate the bearer token on a tonic request. Returns the
    /// authenticated [`crate::server::Principal`] when auth is enabled,
    /// or `None` when the server is running in
    /// [`AuthMode::Disabled`]. Errors as `Status::unauthenticated`.
    pub(super) fn check_auth<T>(&self, request: &Request<T>) -> Result<Option<crate::server::auth::Principal>, Status> {
        self.auth.check(request.metadata())
    }

    pub(super) async fn build_get_tables_response(
        &self,
        query: CommandGetTables,
    ) -> Result<(Arc<Schema>, arrow::record_batch::RecordBatch), Status> {
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
                                    "datafusion_nexus.server.flight.do_get_tables.table_type",
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
                                "datafusion_nexus.server.flight.do_get_tables.append",
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
                "datafusion_nexus.server.flight.do_get_tables.build",
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
        let diagnostics = ErrorSourceDiagnostics::from_error(error);
        if let Some(adapter_error) = error.find_nexus_error() {
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
                error_source_chain = diagnostics.chain(),
                error_root_cause = diagnostics.root_cause_when_distinct(),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        } else {
            tracing::warn!(
                catalog = %catalog_name,
                schema = %schema_name,
                table = %table_name,
                failure,
                error_source_chain = diagnostics.chain(),
                error_root_cause = diagnostics.root_cause_when_distinct(),
                "failed to inspect table metadata for Flight SQL GetTables"
            );
        }
    }

    pub(crate) fn admission(&self) -> &Arc<QueryAdmission> {
        &self.admission
    }

    pub(crate) fn cancellations(&self) -> &Arc<QueryCancellation> {
        &self.cancellations
    }

    pub(crate) fn shutdown_token(&self) -> &CancellationToken {
        &self.shutdown_token
    }

    #[cfg(test)]
    pub(crate) fn native_optimizer_config(&self) -> &CudfOptimizerConfig {
        &self.native_optimizer_config
    }
}
