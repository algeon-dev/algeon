use super::*;

impl DatafusionCudfFlightSqlService {
    pub(super) async fn execute_update_sql(
        &self,
        query_id: String,
        sql: String,
        correlation_id: Option<String>,
        statement_kind: UpdateStatementKind,
    ) -> Result<i64, Status> {
        let started_at = Instant::now();
        let query_span = Self::query_span(&query_id, correlation_id.as_deref(), "update_statement");
        let _query_enter = query_span.enter();
        tracing::info!(
            query_id,
            correlation_id = correlation_id.as_deref().unwrap_or(""),
            sql,
            statement_kind = ?statement_kind,
            "executing Flight SQL update statement"
        );

        drop(_query_enter);

        let lifecycle = QueryLifecycle::new(query_id.clone(), started_at);
        let native_cancellation = ExecutionCancellation::new();
        let planning = self.query_planning(&query_id, correlation_id.as_deref(), native_cancellation.clone());
        let query_ctx = &planning.ctx;
        let state = query_ctx.state();
        let plan = tokio::select! {
            result = state.create_logical_plan(&sql)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(plan) => plan,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "planning",
                        datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
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
                lifecycle.finish_status("planning", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };
        if let Err(error) = reject_parameterized_plan(&plan, statement_kind.sql_operation()) {
            self.observability.finish_failed(
                &query_id,
                "planning",
                datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
            );
            let status = status_with_query_id(
                datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                &query_id,
            );
            let status = status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        if let Err(error) = verify_update_plan(&plan) {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        if let Err(error) = reject_drop_object_type_mismatch(query_ctx, &plan, statement_kind.sql_operation()).await {
            self.observability.finish_failed(&query_id, "planning", CapturedError::from_adapter_error(&error));
            let status = status_with_query_id_and_correlation_id(
                adapter_error_status(error),
                &query_id,
                correlation_id.as_deref(),
            );
            lifecycle.finish_status("planning", &status);
            self.observability.seal(&query_id);
            return Err(status);
        }
        let plan_for_error = plan.clone();
        let df = tokio::select! {
            result = query_ctx.execute_logical_plan(plan)
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "planning")) => match result {
                Ok(df) => df,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    let status = match self.ddl_read_only_registration_error(
                        query_ctx,
                        &plan_for_error,
                        error,
                        statement_kind.sql_operation(),
                    ) {
                        Ok(error) => finish_adapter_error_status_with_correlation_id(
                            self.observability.as_ref(),
                            &query_id,
                            correlation_id.as_deref(),
                            "planning",
                            error,
                        ),
                        Err(error) => {
                            self.observability.finish_failed(
                                &query_id,
                                "planning",
                                datafusion_error_to_captured_error(&error, statement_kind.sql_operation()),
                            );
                            let status = status_with_query_id(
                                datafusion_error_to_status_for_operation(error, statement_kind.sql_operation()),
                                &query_id,
                            );
                            status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref())
                        },
                    };
                    lifecycle.finish_status("planning", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
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
                lifecycle.finish_status("planning", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };

        let _batches = tokio::select! {
            result = df.collect()
                .instrument(Self::query_stage_span(Some(&query_span), &query_id, correlation_id.as_deref(), "native_execution")) => match result {
                Ok(batches) => batches,
                Err(error) => {
                    log_leaf_guard_rejection(&query_id, &error, self.session_profile);
                    self.observability.finish_failed(
                        &query_id,
                        "collect",
                        datafusion_error_to_captured_error(&error, statement_kind.collect_operation()),
                    );
                    let status = status_with_query_id(
                        datafusion_error_to_status_for_operation(error, statement_kind.collect_operation()),
                        &query_id,
                    );
                    let status =
                        status_with_query_id_and_correlation_id(status, &query_id, correlation_id.as_deref());
                    lifecycle.finish_status("collect", &status);
                    self.observability.seal(&query_id);
                    return Err(status);
                },
            },
            _ = self.shutdown_token.cancelled() => {
                native_cancellation.cancel();
                let status = finish_adapter_error_status_with_correlation_id(
                    self.observability.as_ref(),
                    &query_id,
                    correlation_id.as_deref(),
                    "collect",
                    server_shutdown_error(),
                );
                lifecycle.finish_status("collect", &status);
                self.observability.seal(&query_id);
                return Err(status);
            },
        };

        lifecycle.finish_success("collect");
        self.observability.finish_success(&query_id);
        self.observability.seal(&query_id);
        // Flight SQL update success responses carry only a record count, so
        // there is no portable client-visible query_id. Failure captures are
        // still retained because their Status metadata exposes the query_id.
        self.observability.delete(&query_id);
        Ok(NO_UPDATED_RECORDS_COUNT)
    }

    pub(super) fn ddl_read_only_registration_error(
        &self,
        ctx: &SessionContext,
        plan: &LogicalPlan,
        source: DataFusionError,
        operation: &'static str,
    ) -> Result<Error, DataFusionError> {
        if !datafusion_error_contains_message(&source, READ_ONLY_SCHEMA_PROVIDER_REGISTER_TABLE_MESSAGE) {
            return Err(source);
        }
        let Some((target, statement_kind)) = ddl_register_table_target(plan) else {
            return Err(source);
        };

        let options = ctx.state();
        let catalog_options = &options.config().options().catalog;
        let resolved = target.clone().resolve(&catalog_options.default_catalog, &catalog_options.default_schema);
        let message = format!(
            "schema provider does not support registering tables for DDL target {}.{}",
            resolved.catalog, resolved.schema
        );
        let mut error = Error::new(ErrorCode::DataFusionPlanning, operation, message)
            .with_fact("target_catalog", resolved.catalog.as_ref())
            .with_fact("target_schema", resolved.schema.as_ref())
            .with_fact("target_table", resolved.table.as_ref())
            .with_fact("statement_kind", statement_kind)
            .with_fact("iceberg_enabled", self.catalog_diagnostics.iceberg_enabled);
        if let Some(workspace_catalog) = &self.catalog_diagnostics.workspace_catalog {
            error = error.with_fact("workspace_catalog", workspace_catalog);
        }
        if let Some(workspace_schema) = &self.catalog_diagnostics.workspace_schema {
            error = error.with_fact("workspace_schema", workspace_schema);
        }
        Ok(error.with_source(source))
    }
}

pub(super) fn ddl_register_table_target(plan: &LogicalPlan) -> Option<(&TableReference, &'static str)> {
    let LogicalPlan::Ddl(ddl) = plan else {
        return None;
    };
    match ddl {
        DdlStatement::CreateExternalTable(statement) => Some((&statement.name, "CreateExternalTable")),
        DdlStatement::CreateMemoryTable(statement) => Some((&statement.name, "CreateMemoryTable")),
        DdlStatement::CreateView(statement) => Some((&statement.name, "CreateView")),
        _ => None,
    }
}

pub(super) fn datafusion_error_contains_message(error: &DataFusionError, needle: &str) -> bool {
    if error.to_string().contains(needle) {
        return true;
    }
    match error {
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            datafusion_error_contains_message(source, needle)
        },
        DataFusionError::Shared(source) => datafusion_error_contains_message(source, needle),
        DataFusionError::Collection(errors) => {
            errors.iter().any(|error| datafusion_error_contains_message(error, needle))
        },
        _ => false,
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
) -> crate::error::Result<()> {
    let (object_name, expected_table_type) = match plan {
        LogicalPlan::Ddl(DdlStatement::DropTable(drop_table)) => (&drop_table.name, TableType::Base),
        LogicalPlan::Ddl(DdlStatement::DropView(drop_view)) => (&drop_view.name, TableType::View),
        _ => return Ok(()),
    };
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
    .with_fact("requested_table_type", table_type_to_flight_sql(expected_table_type))
    .with_fact("actual_table_type", table_type_to_flight_sql(actual_table_type)))
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
