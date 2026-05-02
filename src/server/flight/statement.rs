use super::*;

/// after physical optimization runs (during `DataFrame::execute_stream`).
pub(super) struct QueryPlanning {
    pub(super) ctx: SessionContext,
    pub(super) native_rule: Arc<NexusNativePhysicalOptimizerRule>,
}
impl DatafusionNexusFlightSqlService {
    pub(super) fn coordinated_query_planning(
        &self,
        query_id: &str,
        correlation_id: Option<&str>,
        cancellation: ExecutionCancellation,
        observation: RuntimeObservationHandle,
        activity_factory: NativeObservationActivityFactory,
    ) -> QueryPlanning {
        self.query_planning_with_capture(
            correlation_id,
            cancellation,
            NativeMetricsCapture::coordinated(query_id.to_string(), observation)
                .with_observation_activity_factory(activity_factory),
        )
    }

    fn query_planning_with_capture(
        &self,
        correlation_id: Option<&str>,
        cancellation: ExecutionCancellation,
        metrics_capture: NativeMetricsCapture,
    ) -> QueryPlanning {
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
        let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
        let mut session_extension = DataFusionNexusSessionConfig::from_optimizer_config(&optimizer_config);
        if optimizer_config.native_execution_config().has_query_admission_authority() {
            session_extension = session_extension.frozen_for_backend_admission();
        }
        let builder = install_nexus_native_rule_with_session_config(
            SessionStateBuilder::new_from_existing(self.ctx.state()),
            Arc::clone(&native_rule),
            session_extension,
        )
        .expect("server query planning should install a session config derived from the same optimizer config");
        let builder = if let Some(cugraph_sql_config) = &self.cugraph_sql_config {
            builder.with_cugraph_sql(cugraph_sql_config.clone().with_native_execution_config(execution_config))
        } else {
            builder
        };
        let state = builder.build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule }
    }

    /// Retain the immutable report at the attempt-owned seal point.
    ///
    /// This is intentionally a synchronous seal observer rather than a
    /// detached receiver task: a Flight terminal lease must never become
    /// releasable before the coordinator has the identical sealed report.
    pub(super) fn register_attempt_report_handoff(
        &self,
        attempt: &crate::attempt::NexusAttempt,
        terminal: CoordinatorReportHandoffTerminal,
    ) {
        let key = terminal.key().clone();
        self.register_attempt_report_handoff_for_key(key, attempt, terminal);
    }

    #[cfg(test)]
    pub(super) fn register_attempt_report_handoff_for_test(
        &self,
        key: crate::server::coordinator::QueryKey,
        attempt: &crate::attempt::NexusAttempt,
        terminal: CoordinatorReportHandoffTerminal,
    ) {
        self.register_attempt_report_handoff_for_key(key, attempt, terminal);
    }

    fn register_attempt_report_handoff_for_key(
        &self,
        key: crate::server::coordinator::QueryKey,
        attempt: &crate::attempt::NexusAttempt,
        terminal: CoordinatorReportHandoffTerminal,
    ) {
        let coordinator = Arc::clone(&self.coordinator);
        let query_id = key.query_id().to_string();
        attempt.register_sealed_report_observer(move |report, snapshot| {
            let Some(report) = report else {
                Self::finish_missing_attempt_report_handoff(&terminal, &query_id);
                return;
            };
            let attempt_outcome = report.outcome.clone();
            if let Err(error) = coordinator.attach_attempt_report(&key, report, Some(snapshot)) {
                tracing::warn!(query_id, %error, "failed to retain sealed Flight attempt report");
                terminal.finish_attempt_report_delivery_failed(error, Some(attempt_outcome));
                return;
            }
            // Retain the terminal through asynchronous GPU cleanup. A weak
            // observer would let the final stream wrapper drop its pending
            // lease before this immutable handoff can release it.
            terminal.mark_attempt_report_attached();
        });
    }

    /// Fails a Flight delivery whose sealed attempt violated the engine
    /// invariant that a report is published before its snapshot observer.
    ///
    /// The coordinator must release its report gate even on this impossible
    /// production path; retaining a pending terminal would otherwise turn an
    /// observation invariant failure into an indefinitely live query.
    pub(super) fn finish_missing_attempt_report_handoff(terminal: &CoordinatorReportHandoffTerminal, query_id: &str) {
        let error = QueryCoordinatorError::new(
            QueryCoordinatorErrorKind::InvalidTransition,
            "attempt observation sealed before its immutable report was published",
        );
        tracing::warn!(query_id, %error, "sealed Flight attempt omitted its immutable report");
        terminal.finish_attempt_report_delivery_failed(error, None);
    }

    pub(super) async fn statement_flight_info(
        &self,
        mut planning_lease: QueryPlanningLease,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        descriptor: FlightDescriptor,
        started_at: Instant,
        ticket_kind: StatementTicketKind,
        prepared_handle: Option<String>,
    ) -> Result<FlightInfo, Status> {
        self.expire_stale_tickets(Instant::now());
        // Before ticket issuance there is no per-query token, so planning
        // observes the server shutdown token directly. The native flag is still
        // shared with the optimizer rule, but `do_action_cancel_query` cannot
        // reach it until the ticket is registered.
        let native_cancellation = ExecutionCancellation::new();
        let mut attempt = match self.admission.begin_attempt(crate::attempt::AttemptOptions {
            query_id: Some(query_id.clone()),
            trace_context: correlation_id.clone(),
            deadline: None,
            observation: self.runtime_observation_config,
        }) {
            Ok(attempt) => attempt,
            Err(error) => {
                self.coordinator.allow_missing_attempt_report(&query_id);
                let (status, captured) = adapter_error_to_status_and_capture(error);
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(attempt, CoordinatorReportHandoffTerminal::planning(&planning_lease));
        }
        let observation = match attempt.as_ref() {
            Some(attempt) => attempt.observation_handle(),
            None => match planning_lease.observation_handle() {
                Ok(observation) => observation,
                Err(error) => {
                    let (status, captured) = adapter_error_to_status_and_capture(
                        error.adapter_error("datafusion_nexus.server.flight.statement.observation_handle"),
                    );
                    planning_lease.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            },
        };
        let activity_factory = planning_lease.activity_factory();
        let planning = self.coordinated_query_planning(
            &query_id,
            correlation_id.as_deref(),
            native_cancellation.clone(),
            observation,
            activity_factory,
        );
        let query_ctx = match attempt.as_ref() {
            Some(attempt) => match attempt.planning_context(&planning.ctx) {
                Ok(context) => context,
                Err(error) => {
                    let (status, captured) =
                        datafusion_error_to_status_and_capture(error, ticket_kind.create_logical_plan_operation());
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    planning_lease.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            },
            None => planning.ctx.clone(),
        };
        let query_state = query_ctx.state();
        let logical_plan = tokio::select! {
            result = query_state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(None, &query_id, correlation_id.as_deref(), "planning")) => {
                match result {
                    Ok(plan) => plan,
                    Err(error) => {
                        if let Some(attempt) = attempt.as_ref() {
                            attempt.planning_failed(&error);
                        }
                        let (status, captured) = datafusion_error_to_status_and_capture(
                            error,
                            ticket_kind.create_logical_plan_operation(),
                        );
                        planning_lease.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                }
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                if let Some(attempt) = attempt.as_ref() {
                    attempt.cancel_before_stream();
                }
                let (status, captured) = adapter_error_to_status_and_capture(server_shutdown_error());
                let status = status_with_query_id_and_correlation_id(
                    status,
                    &query_id,
                    correlation_id.as_deref(),
                );
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            }
        };
        if let Err(error) = verify_flight_info_plan(&logical_plan, ticket_kind) {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed_adapter(&error);
            }
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        if let Err(error) =
            reject_drop_object_type_mismatch(&query_ctx, &logical_plan, ticket_kind.create_logical_plan_operation())
                .await
        {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed_adapter(&error);
            }
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
        let schema = flight_info_result_schema(&logical_plan);

        // Both ticket variants carry the per-execution `query_id` bytes. For
        // prepared statements this reuses `prepared_statement_handle`; the
        // persistent handle remains in `self.prepared_statements`.
        let ticket_payload = match ticket_kind {
            StatementTicketKind::Statement
            | StatementTicketKind::ExplainGpuRows
            | StatementTicketKind::ExplainGpuJson => {
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
        let info = match FlightInfo::new().try_with_schema(&schema) {
            Ok(info) => info,
            Err(error) => {
                if let Some(attempt) = attempt.as_ref() {
                    attempt.planning_failed_adapter(&Error::new(
                        ErrorCode::Internal,
                        "datafusion_nexus.server.flight.get_flight_info_query.schema",
                        "failed to build Flight SQL query FlightInfo schema",
                    ));
                }
                let (status, captured) = arrow_error_to_status_and_capture(
                    error,
                    "datafusion_nexus.server.flight.get_flight_info_query.schema",
                    "failed to build Flight SQL query FlightInfo schema",
                );
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        }
        .with_descriptor(descriptor)
        .with_endpoint(endpoint)
        .with_ordered(false);
        let issued_at = Instant::now();
        let deadline = issued_at.checked_add(self.statement_ticket_ttl).unwrap_or(issued_at);
        planning_lease
            .issue_ticket(CoordinatorTicketPayload {
                sql,
                correlation_id: correlation_id.clone(),
                prepared_handle,
                kind: ticket_kind.code(),
                native_cancellation,
                attempt: attempt.take(),
                started_at,
                deadline,
            })
            .map_err(|error| {
                status_with_query_id_and_correlation_id(
                    adapter_error_status(error.adapter_error("datafusion_nexus.server.query_coordinator.issue_ticket")),
                    &query_id,
                    correlation_id.as_deref(),
                )
            })?;
        Ok(info)
    }

    /// Issue an `EXPLAIN GPU` ticket without asking DataFusion to parse the
    /// extension as its ordinary `EXPLAIN` statement. The inner SQL is planned
    /// exactly once at DoGet by `validate_query`.
    pub(super) fn explain_gpu_flight_info(
        &self,
        mut planning_lease: QueryPlanningLease,
        query_id: String,
        inner_sql: String,
        output_mode: ExplainGpuOutputMode,
        correlation_id: Option<String>,
        descriptor: FlightDescriptor,
        started_at: Instant,
    ) -> Result<FlightInfo, Status> {
        self.expire_stale_tickets(Instant::now());
        let mut attempt = match self.admission.begin_attempt(crate::attempt::AttemptOptions {
            query_id: Some(query_id.clone()),
            trace_context: correlation_id.clone(),
            deadline: None,
            observation: self.runtime_observation_config,
        }) {
            Ok(attempt) => attempt,
            Err(error) => {
                self.coordinator.allow_missing_attempt_report(&query_id);
                let (status, captured) = adapter_error_to_status_and_capture(error);
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(attempt, CoordinatorReportHandoffTerminal::planning(&planning_lease));
        }
        let ticket_kind = match output_mode {
            ExplainGpuOutputMode::Rows => StatementTicketKind::ExplainGpuRows,
            ExplainGpuOutputMode::Json => StatementTicketKind::ExplainGpuJson,
        };
        let ticket = TicketStatementQuery { statement_handle: query_id.as_bytes().to_vec().into() };
        let endpoint = FlightEndpoint::new().with_ticket(Ticket { ticket: ticket.as_any().encode_to_vec().into() });
        let info = match FlightInfo::new().try_with_schema(coverage_schema().as_ref()) {
            Ok(info) => info,
            Err(error) => {
                if let Some(attempt) = attempt.as_ref() {
                    attempt.planning_failed_adapter(&Error::new(
                        ErrorCode::Internal,
                        "datafusion_nexus.server.flight.get_flight_info_statement.explain_gpu.schema",
                        "failed to build EXPLAIN GPU FlightInfo schema",
                    ));
                }
                let (status, captured) = arrow_error_to_status_and_capture(
                    error,
                    "datafusion_nexus.server.flight.get_flight_info_statement.explain_gpu.schema",
                    "failed to build EXPLAIN GPU FlightInfo schema",
                );
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        }
        .with_descriptor(descriptor)
        .with_endpoint(endpoint)
        .with_ordered(false);
        let issued_at = Instant::now();
        let deadline = issued_at.checked_add(self.statement_ticket_ttl).unwrap_or(issued_at);
        planning_lease
            .issue_ticket(CoordinatorTicketPayload {
                sql: inner_sql,
                correlation_id: correlation_id.clone(),
                prepared_handle: None,
                kind: ticket_kind.code(),
                native_cancellation: ExecutionCancellation::new(),
                attempt: attempt.take(),
                started_at,
                deadline,
            })
            .map_err(|error| {
                status_with_query_id_and_correlation_id(
                    adapter_error_status(error.adapter_error("datafusion_nexus.server.query_coordinator.issue_ticket")),
                    &query_id,
                    correlation_id.as_deref(),
                )
            })?;
        Ok(info)
    }
}
