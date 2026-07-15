use super::*;

/// after physical optimization runs (during `DataFrame::execute_stream`).
pub(super) struct QueryPlanning {
    pub(super) ctx: SessionContext,
    pub(super) native_rule: Arc<CudfNativePhysicalOptimizerRule>,
}
impl DatafusionCudfFlightSqlService {
    pub(super) fn query_planning(
        &self,
        query_id: &str,
        correlation_id: Option<&str>,
        cancellation: ExecutionCancellation,
    ) -> QueryPlanning {
        let observability: Arc<dyn NativeMetricsCaptureSink> = self.observability.clone();
        let metrics_capture = NativeMetricsCapture::new(query_id.to_string(), observability);
        let mut execution_config = self
            .native_optimizer_config
            .native_execution_config()
            .clone()
            .with_cancellation(cancellation)
            .with_metrics_capture(metrics_capture);
        if let Some(correlation_id) = correlation_id.map(str::trim).filter(|value| !value.is_empty()) {
            execution_config = execution_config.with_trace_id(correlation_id.to_string());
        }
        let optimizer_config =
            self.native_optimizer_config.clone().with_native_execution_config(execution_config.clone());
        // Retain the native rule so we can read its `last_report` for the
        // Milestone 4 planning-report log after physical optimization runs.
        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
        let mut session_extension = DataFusionNexusSessionConfig::from_optimizer_config(&optimizer_config);
        if optimizer_config.native_execution_config().has_query_admission_authority() {
            session_extension = session_extension.frozen_for_backend_admission();
        }
        let builder = install_cudf_native_rule_with_session_config(
            SessionStateBuilder::new_from_existing(self.ctx.state()),
            Arc::clone(&native_rule),
            session_extension,
        )
        .expect("server query planning should install a session config derived from the same optimizer config");
        #[cfg(feature = "cugraph")]
        let builder = if let Some(cugraph_sql_config) = &self.cugraph_sql_config {
            builder.with_cugraph_sql(cugraph_sql_config.clone().with_native_execution_config(execution_config))
        } else {
            builder
        };
        let state = builder.build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule }
    }

    pub(super) async fn statement_flight_info(
        &self,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        descriptor: FlightDescriptor,
        started_at: Instant,
        ticket_kind: StatementTicketKind,
        prepared_handle: Option<String>,
    ) -> Result<FlightInfo, Status> {
        // Before ticket issuance there is no per-query token, so planning
        // observes the server shutdown token directly. The native flag is still
        // shared with the optimizer rule, but `do_action_cancel_query` cannot
        // reach it until the ticket is registered.
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(&query_id, correlation_id.as_deref(), native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(None, &query_id, correlation_id.as_deref(), "planning")) => {
                result.map_err(|error| {
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, ticket_kind.create_logical_plan_operation()),
                    );
                    datafusion_error_to_status_for_operation(error, ticket_kind.create_logical_plan_operation())
                })?
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "planning",
                    server_shutdown_error(),
                );
                return Err(status);
            }
        };
        if let Err(error) = verify_flight_info_plan(&logical_plan, ticket_kind) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        if let Err(error) =
            reject_drop_object_type_mismatch(query_ctx, &logical_plan, ticket_kind.create_logical_plan_operation())
                .await
        {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        if let Some(error) = fast_fail_unsupported_table_format_operator_plan(&logical_plan) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            return Err(status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            ));
        }
        let schema = flight_info_result_schema(&logical_plan);

        // Both ticket variants carry the per-execution `query_id` bytes. For
        // prepared statements this reuses `prepared_statement_handle`; the
        // persistent handle remains in `self.prepared_statements`.
        let ticket_payload = match ticket_kind {
            StatementTicketKind::Statement => {
                let ticket = TicketStatementQuery { statement_handle: query_id.as_bytes().to_vec().into() };
                ticket.as_any().encode_to_vec().into()
            },
            StatementTicketKind::PreparedStatement => {
                let ticket =
                    CommandPreparedStatementQuery { prepared_statement_handle: query_id.as_bytes().to_vec().into() };
                ticket.as_any().encode_to_vec().into()
            },
        };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket { ticket: ticket_payload });
        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|error| {
                arrow_error_to_status_for_operation(
                    error,
                    "datafusion_nexus.server.flight.get_flight_info_query.schema",
                    "failed to build Flight SQL query FlightInfo schema",
                )
            })?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint)
            .with_ordered(false);
        self.tickets.insert(
            query_id.clone(),
            StatementTicket {
                query_id,
                sql,
                correlation_id,
                ticket_kind,
                prepared_handle,
                native_cancellation,
                started_at,
                issued_at: Instant::now(),
            },
        );
        Ok(info)
    }
}
