use super::*;

impl DatafusionNexusFlightSqlService {
    pub(super) async fn execute_update_sql(
        &self,
        mut planning_lease: QueryPlanningLease,
        sql: String,
        correlation_id: Option<String>,
        statement_kind: UpdateStatementKind,
    ) -> Result<i64, Status> {
        let query_id = planning_lease.key().query_id().to_string();
        let sql: Arc<str> = sql.into();
        let attempt = match self.admission.begin_attempt(crate::attempt::AttemptOptions {
            query_id: Some(query_id.clone()),
            trace_context: correlation_id.clone(),
            deadline: None,
            observation: self.runtime_observation_config,
        }) {
            Ok(attempt) => attempt,
            Err(error) => {
                // A production DoPut record is attempt-owned from registration.
                // If admission cannot mint that owner, explicitly let the
                // coordinator finalize the failed planning lease without a
                // report rather than retaining a permanently pending record.
                self.coordinator.allow_missing_attempt_report(&query_id);
                let (status, captured) = adapter_error_to_status_and_capture(error);
                let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        let coordinator_capture = if attempt.is_none() {
            let observation = match planning_lease.observation_handle() {
                Ok(observation) => observation,
                Err(error) => {
                    let (status, captured) = adapter_error_to_status_and_capture(
                        error.adapter_error("datafusion_nexus.server.flight.update.observation_handle"),
                    );
                    planning_lease.finish_captured("planning", &status, captured);
                    return Err(status);
                },
            };
            Some((observation, planning_lease.activity_factory()))
        } else {
            None
        };
        // Keep this exact-generation callback before `into_execution` moves
        // the planning lease. Its error arm may seal the attempt only after
        // that move has already terminalized the coordinator record.
        let planning_report_handoff = CoordinatorReportHandoffTerminal::planning(&planning_lease);
        let execution_lease = match planning_lease.into_execution() {
            Ok(execution_lease) => execution_lease,
            Err(error) => {
                let error = error.adapter_error(statement_kind.into_execution_operation());
                if let Some(attempt) = attempt.as_ref() {
                    self.register_attempt_report_handoff(attempt, planning_report_handoff);
                    attempt.planning_failed_adapter(&error);
                }
                let status = adapter_error_status(error);
                return Err(status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref()));
            },
        };
        let coordinator_terminal = CoordinatorExecutionTerminal::new_with_attempt_report(
            execution_lease,
            query_id.clone(),
            Arc::clone(&sql),
            Instant::now(),
            attempt.is_some(),
        );
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(
                attempt,
                CoordinatorReportHandoffTerminal::execution(Arc::clone(&coordinator_terminal)),
            );
        }
        let query_span = Self::query_span(&query_id, correlation_id.as_deref(), "update_statement");
        let _query_enter = query_span.enter();
        tracing::info!(
            query_id,
            correlation_id = tracing_correlation_id(correlation_id.as_deref()),
            sql = sql.as_ref(),
            statement_kind = ?statement_kind,
            "executing Flight SQL update statement"
        );

        drop(_query_enter);

        let native_cancellation = ExecutionCancellation::new();
        let planning = match coordinator_capture {
            Some((observation, activity_factory)) => self.coordinated_query_planning(
                &query_id,
                correlation_id.as_deref(),
                native_cancellation.clone(),
                observation,
                activity_factory,
            ),
            None => self.attempt_query_planning(correlation_id.as_deref(), native_cancellation.clone()),
        };
        let query_ctx = match attempt.as_ref() {
            Some(attempt) => match attempt.planning_context(&planning.ctx) {
                Ok(context) => context,
                Err(error) => {
                    let (status, captured) =
                        datafusion_error_to_status_and_capture(error, statement_kind.sql_operation());
                    let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    coordinator_terminal.wait_finished().await;
                    return Err(status);
                },
            },
            None => planning.ctx,
        };
        let state = query_ctx.state();
        let plan = tokio::select! {
            result = state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    let captured = datafusion_error_to_captured_error(&error, statement_kind.sql_operation());
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    coordinator_terminal.wait_finished().await;
                    return Err(status);
                },
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
                coordinator_terminal.finish_captured("planning", &status, captured);
                coordinator_terminal.wait_finished().await;
                return Err(status);
            },
        };
        if let Err(error) = reject_parameterized_plan(&plan, statement_kind.sql_operation()) {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed(&error);
            }
            let captured = datafusion_error_to_captured_error(&error, statement_kind.sql_operation());
            let status = status_with_query_id(
                datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                &query_id,
            );
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            coordinator_terminal.finish_captured("planning", &status, captured);
            coordinator_terminal.wait_finished().await;
            return Err(status);
        }
        if let Err(error) = verify_update_plan(&plan) {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed_adapter(&error);
            }
            let captured = CapturedError::from_adapter_error(&error);
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            coordinator_terminal.finish_captured("planning", &status, captured);
            coordinator_terminal.wait_finished().await;
            return Err(status);
        }
        if let Err(error) = reject_drop_object_type_mismatch(&query_ctx, &plan, statement_kind.sql_operation()).await {
            if let Some(attempt) = attempt.as_ref() {
                attempt.planning_failed_adapter(&error);
            }
            let captured = CapturedError::from_adapter_error(&error);
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            coordinator_terminal.finish_captured("planning", &status, captured);
            coordinator_terminal.wait_finished().await;
            return Err(status);
        }
        let _result = tokio::select! {
            result = query_ctx.execute_logical_plan(plan)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(df) => df,
                Err(error) => {
                    if let Some(attempt) = attempt.as_ref() {
                        attempt.planning_failed(&error);
                    }
                    let captured = datafusion_error_to_captured_error(&error, statement_kind.sql_operation());
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    let status = status_with_query_id_and_correlation_id(
                        status,
                        &query_id,
                        correlation_id.as_deref(),
                    );
                    coordinator_terminal.finish_captured("planning", &status, captured);
                    coordinator_terminal.wait_finished().await;
                    return Err(status);
                },
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
                coordinator_terminal.finish_captured("planning", &status, captured);
                coordinator_terminal.wait_finished().await;
                return Err(status);
            },
        };

        // `execute_logical_plan` has already completed every supported update
        // action. In particular, DataFusion materializes CTAS input there and
        // returns an empty DataFrame. Re-executing that placeholder under the
        // same attempt would create a second native admission after the CTAS
        // handle consumed the attempt's one post-cleanup finalizer.
        if let Some(attempt) = attempt.as_ref() {
            attempt.complete_without_result_stream();
        }

        coordinator_terminal.finish_success("collect");
        coordinator_terminal.wait_finished().await;
        // Flight SQL update success responses carry only a record count, so
        // there is no portable client-visible query_id. Failure captures are
        // still retained because their Status metadata exposes the query_id.
        Ok(NO_UPDATED_RECORDS_COUNT)
    }

    /// Finalize a prepared-update duplicate without replanning or executing
    /// the original DDL. Even this protocol-level no-op is an invocation, so
    /// it owns and publishes one sealed attempt report before the coordinator
    /// releases the completed query.
    pub(super) async fn complete_deduplicated_update(
        &self,
        planning_lease: QueryPlanningLease,
        sql: String,
        correlation_id: Option<String>,
    ) -> Result<i64, Status> {
        let query_id = planning_lease.key().query_id().to_string();
        let attempt = match self.admission.begin_attempt(crate::attempt::AttemptOptions {
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
                let mut planning_lease = planning_lease;
                planning_lease.finish_captured("planning", &status, captured);
                return Err(status);
            },
        };
        let planning_report_handoff = CoordinatorReportHandoffTerminal::planning(&planning_lease);
        let execution_lease = match planning_lease.into_execution() {
            Ok(execution_lease) => execution_lease,
            Err(error) => {
                let error = error
                    .adapter_error("datafusion_nexus.server.flight.do_put_prepared_statement_update.into_execution");
                if let Some(attempt) = attempt.as_ref() {
                    self.register_attempt_report_handoff(attempt, planning_report_handoff);
                    attempt.planning_failed_adapter(&error);
                }
                let status = adapter_error_status(error);
                return Err(status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref()));
            },
        };
        let coordinator_terminal = CoordinatorExecutionTerminal::new_with_attempt_report(
            execution_lease,
            query_id.clone(),
            sql.into(),
            Instant::now(),
            attempt.is_some(),
        );
        if let Some(attempt) = attempt.as_ref() {
            self.register_attempt_report_handoff(
                attempt,
                CoordinatorReportHandoffTerminal::execution(Arc::clone(&coordinator_terminal)),
            );
            attempt.complete_without_result_stream();
        }
        coordinator_terminal.finish_success("deduplicated_update");
        coordinator_terminal.wait_finished().await;
        Ok(NO_UPDATED_RECORDS_COUNT)
    }
}

pub(super) fn referenced_table_names(plan: &LogicalPlan) -> String {
    let mut tables = BTreeSet::new();
    collect_referenced_table_names(plan, &mut tables);
    tables.into_iter().collect::<Vec<_>>().join(",")
}

pub(super) fn collect_referenced_table_names(plan: &LogicalPlan, tables: &mut BTreeSet<String>) {
    if let LogicalPlan::TableScan(scan) = plan {
        tables.insert(scan.table_name.to_string());
    }
    for input in plan.inputs() {
        collect_referenced_table_names(input, tables);
    }
}

pub(super) async fn reject_drop_object_type_mismatch(
    ctx: &SessionContext,
    plan: &LogicalPlan,
    operation: &'static str,
) -> crate::adapter_error::Result<()> {
    let (object_name, expected_table_type) = match plan {
        LogicalPlan::Ddl(DdlStatement::DropTable(drop_table)) => (&drop_table.name, TableType::Base),
        LogicalPlan::Ddl(DdlStatement::DropView(drop_view)) => (&drop_view.name, TableType::View),
        _ => return Ok(()),
    };
    reject_drop_table_type_mismatch(ctx, object_name, expected_table_type, operation).await
}

async fn reject_drop_table_type_mismatch(
    ctx: &SessionContext,
    object_name: &TableReference,
    expected_table_type: TableType,
    operation: &'static str,
) -> crate::adapter_error::Result<()> {
    let table_name = object_name.table().to_string();
    let Ok(schema) = ctx.state().schema_for_ref(object_name.clone()) else {
        return Ok(());
    };
    let Ok(Some(actual_table_type)) = schema.table_type(&table_name).await else {
        return Ok(());
    };
    if actual_table_type == expected_table_type {
        return Ok(());
    };

    let object_name = object_name.to_string();
    let actual_label = table_type_client_label(actual_table_type);
    let expected_label = table_type_client_label(expected_table_type);
    let suggested_drop = drop_statement_for_table_type(actual_table_type);
    Err(Error::new(
        ErrorCode::InvalidRequest,
        operation,
        format!(
            "Object '{object_name}' is a {actual_label}, not a {expected_label}; use {suggested_drop} to remove it"
        ),
    )
    .with_fact("object_name", object_name)
    .with_log_safe_fact("requested_table_type", table_type_to_flight_sql(expected_table_type))
    .with_log_safe_fact("actual_table_type", table_type_to_flight_sql(actual_table_type)))
}

pub(super) fn table_type_client_label(table_type: TableType) -> &'static str {
    match table_type {
        TableType::Base => "base table",
        TableType::View => "view",
        TableType::Temporary => "temporary table",
    }
}

pub(super) fn drop_statement_for_table_type(table_type: TableType) -> &'static str {
    match table_type {
        TableType::Base | TableType::Temporary => "DROP TABLE",
        TableType::View => "DROP VIEW",
    }
}
