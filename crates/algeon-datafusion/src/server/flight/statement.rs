use std::{ops::ControlFlow, sync::Arc, time::Instant};

use datafusion::{
    common::{
        TableReference,
        tree_node::{TreeNode, TreeNodeRecursion},
    },
    sql::{
        parser::Statement as DataFusionSqlStatement,
        sqlparser::ast::{
            DescribeAlias, ObjectName, ObjectNamePart, Query as SqlQuery, ShowStatementOptions,
            Statement as SqlStatement, TableFactor, Visit, Visitor,
        },
    },
};

use super::*;
use crate::{
    error::Result as AdapterResult,
    gpu_functions::{
        GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION,
        is_builtin_gpu_metadata_table_source,
    },
    server::{
        attempt::{AlgeonAttempt, AttemptOptions},
        coordinator::QueryKey,
    },
};

/// after physical optimization runs (during `DataFrame::execute_stream`).
pub(super) struct QueryPlanning {
    pub(super) ctx: SessionContext,
    pub(super) native_rule: Option<Arc<AlgeonNativePhysicalOptimizerRule>>,
}

pub(super) type ParsedFlightStatement = Result<DataFusionSqlStatement, DataFusionError>;

impl DatafusionAlgeonFlightSqlService {
    pub(super) fn parse_flight_statement(&self, sql: &str) -> ParsedFlightStatement {
        let state = self.ctx.state();
        let dialect = state.config_options().sql_parser.dialect;
        state.sql_to_statement(sql, &dialect)
    }

    pub(super) fn classify_statement_candidate(
        &self,
        datafusion_statement: &DataFusionSqlStatement,
    ) -> StatementExecutionDomain {
        let DataFusionSqlStatement::Statement(statement) = datafusion_statement else {
            return StatementExecutionDomain::Data;
        };
        match statement.as_ref() {
            SqlStatement::ShowTables {
                terse: false,
                history: false,
                extended: false,
                full: false,
                external: false,
                show_options,
            } if show_options_are_empty(show_options) => StatementExecutionDomain::Metadata,
            SqlStatement::ShowColumns { show_options, .. } if show_columns_options_are_supported(show_options) => {
                StatementExecutionDomain::Metadata
            },
            SqlStatement::ExplainTable { describe_alias: DescribeAlias::Describe | DescribeAlias::Desc, .. }
            | SqlStatement::Explain { describe_alias: DescribeAlias::Describe | DescribeAlias::Desc, .. } => {
                StatementExecutionDomain::Metadata
            },
            SqlStatement::Query(query) if query_sources_are_metadata(query) => StatementExecutionDomain::Metadata,
            _ => StatementExecutionDomain::Data,
        }
    }

    pub(super) fn verify_metadata_logical_plan(
        &self,
        plan: &LogicalPlan,
        operation: &'static str,
    ) -> AdapterResult<()> {
        if matches!(plan, LogicalPlan::DescribeTable(_)) {
            return Ok(());
        }

        let mut saw_metadata_scan = false;
        let mut rejection = None;
        plan.apply(|node| {
            match node {
                LogicalPlan::TableScan(scan)
                    if table_reference_is_information_schema(&scan.table_name)
                        || is_builtin_gpu_metadata_table_source(&scan.source) =>
                {
                    saw_metadata_scan = true;
                },
                LogicalPlan::TableScan(_) => rejection = Some("non_metadata_scan"),
                LogicalPlan::Dml(_) => rejection = Some("dml"),
                LogicalPlan::Copy(_) => rejection = Some("copy"),
                LogicalPlan::Ddl(_) => rejection = Some("ddl"),
                LogicalPlan::Extension(_) => rejection = Some("extension"),
                _ => {},
            }
            Ok(if rejection.is_some() { TreeNodeRecursion::Stop } else { TreeNodeRecursion::Continue })
        })
        .expect("metadata logical-plan inspection is infallible");

        let rejection = rejection.or((!saw_metadata_scan).then_some("missing_metadata_scan"));
        if let Some(reason) = rejection {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                operation,
                "statement no longer satisfies the metadata execution-domain proof",
            )
            .with_log_safe_fact("metadata_proof_rejection", reason));
        }
        Ok(())
    }

    pub(super) fn metadata_query_planning(&self) -> QueryPlanning {
        let rules = self
            .ctx
            .state()
            .physical_optimizers()
            .iter()
            .filter(|rule| {
                !matches!(
                    rule.name(),
                    ALGEON_NATIVE_COUNT_ONLY_PHYSICAL_OPTIMIZER_RULE_NAME
                        | ALGEON_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME
                        | ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME
                )
            })
            .cloned()
            .collect();
        let state =
            SessionStateBuilder::new_from_existing(self.ctx.state()).with_physical_optimizer_rules(rules).build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule: None }
    }

    pub(super) fn attempt_query_planning(
        &self,
        correlation_id: Option<&str>,
        cancellation: ExecutionCancellation,
    ) -> QueryPlanning {
        self.query_planning_with_capture(correlation_id, cancellation, NativeMetricsCapture::standalone())
    }

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
        let optimizer_config = self.native_optimizer_config.clone().with_native_execution_config(execution_config);
        // Retain the native rule so we can read its `last_report` for the
        // planning-report log after physical optimization runs.
        let native_rule = Arc::new(AlgeonNativePhysicalOptimizerRule::new(optimizer_config));
        let builder =
            SessionStateBuilder::new_from_existing(self.ctx.state()).with_algeon_native_rule(Arc::clone(&native_rule));
        let state = builder.build();
        QueryPlanning { ctx: SessionContext::new_with_state(state), native_rule: Some(native_rule) }
    }

    /// Retain the immutable report at the attempt-owned seal point.
    ///
    /// This is intentionally a synchronous seal observer rather than a
    /// detached receiver task: a Flight terminal lease must never become
    /// releasable before the coordinator has the identical sealed report.
    pub(super) fn register_attempt_report_handoff(
        &self,
        attempt: &AlgeonAttempt,
        terminal: CoordinatorReportHandoffTerminal,
    ) {
        let key = terminal.key().clone();
        self.register_attempt_report_handoff_for_key(key, attempt, terminal);
    }

    #[cfg(test)]
    pub(super) fn register_attempt_report_handoff_for_test(
        &self,
        key: QueryKey,
        attempt: &AlgeonAttempt,
        terminal: CoordinatorReportHandoffTerminal,
    ) {
        self.register_attempt_report_handoff_for_key(key, attempt, terminal);
    }

    fn register_attempt_report_handoff_for_key(
        &self,
        key: QueryKey,
        attempt: &AlgeonAttempt,
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

    #[expect(clippy::too_many_arguments, reason = "service wiring parameters are distinct ownership values")]
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
        parsed_statement: ParsedFlightStatement,
        execution_domain: StatementExecutionDomain,
    ) -> Result<FlightInfo, Status> {
        self.expire_stale_tickets(Instant::now());
        // Before ticket issuance there is no per-query token, so planning
        // observes the server shutdown token directly. The native flag is still
        // shared with the optimizer rule, but `do_action_cancel_query` cannot
        // reach it until the ticket is registered.
        let native_cancellation = ExecutionCancellation::new();
        let mut attempt = match execution_domain {
            StatementExecutionDomain::Metadata => None,
            StatementExecutionDomain::Data => {
                match self.admission.begin_attempt(AttemptOptions {
                    query_id: Some(query_id.clone()),
                    trace_context: correlation_id.clone(),
                    deadline: None,
                    observation: self.runtime_observation_config,
                }) {
                    Ok(attempt) => attempt,
                    Err(error) => {
                        self.coordinator.allow_missing_attempt_report(&query_id);
                        let (status, captured) = adapter_error_to_status_and_capture(error);
                        let status =
                            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                        planning_lease.finish_captured("planning", &status, captured);
                        return Err(status);
                    },
                }
            },
        };
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(attempt, CoordinatorReportHandoffTerminal::planning(&planning_lease));
        }
        let planning = match execution_domain {
            StatementExecutionDomain::Metadata => self.metadata_query_planning(),
            StatementExecutionDomain::Data => {
                self.attempt_query_planning(correlation_id.as_deref(), native_cancellation.clone())
            },
        };
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
            result = async {
                let statement = parsed_statement?;
                query_state.statement_to_plan(statement).await
            }
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
        if execution_domain == StatementExecutionDomain::Metadata
            && let Err(error) =
                self.verify_metadata_logical_plan(&logical_plan, ticket_kind.metadata_logical_proof_operation())
        {
            let (status, captured) = adapter_error_to_status_and_capture(error);
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            planning_lease.finish_captured("planning", &status, captured);
            return Err(status);
        }
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
                        "algeon_datafusion.server.flight.get_flight_info_query.schema",
                        "failed to build Flight SQL query FlightInfo schema",
                    ));
                }
                let (status, captured) = arrow_error_to_status_and_capture(
                    error,
                    "algeon_datafusion.server.flight.get_flight_info_query.schema",
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
                execution_domain,
                native_cancellation,
                attempt: attempt.take(),
                started_at,
                deadline,
            })
            .map_err(|error| {
                status_with_query_id_and_correlation_id(
                    adapter_error_status(
                        error.adapter_error("algeon_datafusion.server.query_coordinator.issue_ticket"),
                    ),
                    &query_id,
                    correlation_id.as_deref(),
                )
            })?;
        Ok(info)
    }

    /// Issue an `EXPLAIN GPU` ticket without asking DataFusion to parse the
    /// extension as its ordinary `EXPLAIN` statement. The inner SQL is planned
    /// exactly once at DoGet by `validate_query`.
    #[expect(clippy::too_many_arguments, reason = "service wiring parameters are distinct ownership values")]
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
        let mut attempt = match self.admission.begin_attempt(AttemptOptions {
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
                        "algeon_datafusion.server.flight.get_flight_info_statement.explain_gpu.schema",
                        "failed to build EXPLAIN GPU FlightInfo schema",
                    ));
                }
                let (status, captured) = arrow_error_to_status_and_capture(
                    error,
                    "algeon_datafusion.server.flight.get_flight_info_statement.explain_gpu.schema",
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
                execution_domain: StatementExecutionDomain::Data,
                native_cancellation: ExecutionCancellation::new(),
                attempt: attempt.take(),
                started_at,
                deadline,
            })
            .map_err(|error| {
                status_with_query_id_and_correlation_id(
                    adapter_error_status(
                        error.adapter_error("algeon_datafusion.server.query_coordinator.issue_ticket"),
                    ),
                    &query_id,
                    correlation_id.as_deref(),
                )
            })?;
        Ok(info)
    }
}

fn show_options_are_empty(options: &ShowStatementOptions) -> bool {
    options.show_in.is_none()
        && options.starts_with.is_none()
        && options.limit.is_none()
        && options.limit_from.is_none()
        && options.filter_position.is_none()
}

fn query_sources_are_metadata(query: &SqlQuery) -> bool {
    let mut proof = MetadataSourceCandidateProof::default();
    let _ = query.visit(&mut proof);
    proof.saw_source && !proof.rejected
}

#[derive(Default)]
struct MetadataSourceCandidateProof {
    saw_source: bool,
    rejected: bool,
}

impl Visitor for MetadataSourceCandidateProof {
    type Break = ();

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<Self::Break> {
        let source_is_metadata = match factor {
            TableFactor::Table { name, args: None, .. } => object_name_is_information_schema(name),
            TableFactor::Table { name, args: Some(_), .. } | TableFactor::Function { name, .. } => {
                object_name_is_builtin_gpu_metadata_function(name)
            },
            TableFactor::Derived { .. }
            | TableFactor::NestedJoin { .. }
            | TableFactor::Pivot { .. }
            | TableFactor::Unpivot { .. } => return ControlFlow::Continue(()),
            _ => false,
        };
        self.saw_source = true;
        if source_is_metadata {
            ControlFlow::Continue(())
        } else {
            self.rejected = true;
            ControlFlow::Break(())
        }
    }
}

fn object_name_is_information_schema(name: &ObjectName) -> bool {
    name.0
        .iter()
        .rev()
        .nth(1)
        .and_then(ObjectNamePart::as_ident)
        .is_some_and(|ident| ident.value.eq_ignore_ascii_case("information_schema"))
}

fn object_name_is_builtin_gpu_metadata_function(name: &ObjectName) -> bool {
    name.0.last().and_then(ObjectNamePart::as_ident).is_some_and(|ident| {
        [GPU_LIST_FUNCTIONS_FUNCTION, GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_VALIDATE_CALL_FUNCTION]
            .iter()
            .any(|function| ident.value.eq_ignore_ascii_case(function))
    })
}

fn show_columns_options_are_supported(options: &ShowStatementOptions) -> bool {
    options.starts_with.is_none()
        && options.limit.is_none()
        && options.limit_from.is_none()
        && options.filter_position.is_none()
        && options
            .show_in
            .as_ref()
            .is_some_and(|show_in| show_in.parent_type.is_none() && show_in.parent_name.is_some())
}

fn table_reference_is_information_schema(reference: &TableReference) -> bool {
    match reference {
        TableReference::Partial { schema, .. } | TableReference::Full { schema, .. } => {
            schema.as_ref() == "information_schema"
        },
        TableReference::Bare { .. } => false,
    }
}
