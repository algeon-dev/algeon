use super::*;

pub(in crate::exec) fn execute_groupby_aggregate(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct(overrides, frame, aggregate, operation);
    }

    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .with_execution(overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| QueryError::from_cudf(operation, "failed to execute nexus native aggregate", error))?;
    rename_groupby_key_columns(&mut output, aggregate, "nexus_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

pub(in crate::exec) fn execute_groupby_aggregate_with_budget(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    execute_groupby_aggregate_with_budget_observed(
        None, overrides, frame, aggregate, token, metrics, operation, message,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(in crate::exec) fn execute_groupby_aggregate_with_budget_observed(
    context: Option<&ExecutionContext>,
    _overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct_with_budget(_overrides, frame, aggregate, token, operation, message);
    }

    let input_rows = frame.num_rows() as u64;
    let stage_guard = context.map(|context| {
        let (observation, plan_execution_id, parent_span_id) =
            crate::observability::stage_context_from_execution(context);
        crate::observability::RuntimeOperatorStageGuard::start(
            observation,
            plan_execution_id,
            parent_span_id,
            crate::observability::RuntimeOperatorStageKind::LibcudfGroupby,
            crate::observability::RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
        )
    });
    let result = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .agg_with_budget(&aggregate.aggregations, token)
        .map_err(|error| QueryError::from_cudf_with_budget_details(operation, message, error));
    let mut output = match result {
        Ok(outcome) => {
            let output = budgeted_value(outcome, metrics, operation);
            if let Some(guard) = stage_guard {
                guard.finish(Some(output.num_rows() as u64), None);
            }
            output
        },
        Err(error) => {
            if let Some(guard) = stage_guard {
                guard.fail(&error);
            }
            return Err(error);
        },
    };
    rename_groupby_key_columns(&mut output, aggregate, "nexus_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

pub(super) fn rename_groupby_key_columns(
    output: &mut DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<()> {
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        *output = output
            .rename_columns(key_renames)
            .map_err(|error| QueryError::from_cudf(operation, "failed to rename nexus native aggregate keys", error))?;
    }
    Ok(())
}

fn execute_groupby_distinct(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let output = frame
        .with_execution(overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to execute nexus native group-only aggregate", error)
        })?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
}

fn execute_groupby_distinct_with_budget(
    _overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let budget_overrides = token.execution_overrides();
    let output = frame
        .with_execution(&budget_overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| attach_budget_token_failure_facts(QueryError::from_cudf(operation, message, error), token))?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
}

fn select_and_rename_groupby_distinct_keys(output: DataFrame, aggregate: &LoweredAggregate) -> Result<DataFrame> {
    let mut output =
        output.select(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.select_group_only_keys",
                "failed to select native group-only aggregate keys",
                error,
            )
        })?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    Ok(output)
}
