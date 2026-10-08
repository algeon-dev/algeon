use query_runtime::observability::{
    RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
};

use super::{
    DataFrame, DropDuplicatesOptions, DuplicateKeepOption, Exec, ExecutionBudgetToken, ExecutionContext,
    ExecutionMetrics, GroupByOptions, LoweredAggregate, Result, attach_budget_token_failure_facts, budgeted_value,
    cudf_query_error,
};

/// Runs `aggregate` as a libcudf groupby on `overrides`.
///
/// Rows with a NULL group key form groups of their own (`include_null_keys`), because libcudf
/// drops them by default and SQL GROUP BY keeps them. Group order is unspecified. An aggregate
/// with no aggregations runs as a distinct over the key columns. Key columns are renamed to
/// `output_key_names`.
pub(in crate::exec) fn execute_groupby_aggregate(
    overrides: Exec<'_>,
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
        .on(overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| cudf_query_error(operation, "failed to execute algeon native aggregate", error))?;
    rename_groupby_key_columns(&mut output, aggregate, "query_engine.aggregate.rename_keys")?;
    Ok(output)
}

/// `execute_groupby_aggregate` under `token`, without an operator-stage record.
pub(in crate::exec) fn execute_groupby_aggregate_with_budget(
    overrides: Exec<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    execute_groupby_aggregate_with_budget_observed(
        None, overrides, frame, aggregate, token, metrics, operation, message,
    )
}

/// `execute_groupby_aggregate` under `token`; the token selects the stream and memory resource,
/// so `_overrides` is unused.
///
/// With a `context`, the groupby is bracketed by a `LibcudfGroupby` operator stage that records
/// the input rows and then the output rows or the failure. The group-only form records no
/// stage. The budget report of the groupby is folded into `metrics`.
#[expect(
    clippy::too_many_arguments,
    reason = "aggregate occurrence inputs (spec, schemas, token, evidence) are distinct domain values"
)]
pub(in crate::exec) fn execute_groupby_aggregate_with_budget_observed(
    context: Option<&ExecutionContext>,
    _overrides: Exec<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct_with_budget(_overrides, frame, aggregate, token, operation, message);
    }

    let input_rows = frame.num_rows() as u64;
    let stage_guard = context.map(|context| {
        let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
        RuntimeOperatorStageGuard::start(
            observation,
            plan_execution_id,
            parent_span_id,
            RuntimeOperatorStageKind::LibcudfGroupby,
            RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
        )
    });
    let result = token
        .run("groupby::agg", |exec| {
            frame
                .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
                .options(GroupByOptions::builder().include_null_keys(true).build())
                .on(exec)
                .agg(&aggregate.aggregations)
        })
        .map_err(|error| cudf_query_error(operation, message, error));
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
    rename_groupby_key_columns(&mut output, aggregate, "query_engine.aggregate.rename_keys")?;
    Ok(output)
}

/// Renames the key columns of a groupby output from their input names to the aggregate's
/// `output_key_names`.
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
            .map_err(|error| cudf_query_error(operation, "failed to rename algeon native aggregate keys", error))?;
    }
    Ok(())
}

/// Group-only aggregate: one row per distinct key combination, taken with `drop_duplicates`.
/// Only key columns are kept, so which duplicate survives (`KeepAny`) does not matter.
fn execute_groupby_distinct(
    overrides: Exec<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let output = frame
        .on(overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| cudf_query_error(operation, "failed to execute algeon native group-only aggregate", error))?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
}

fn execute_groupby_distinct_with_budget(
    _overrides: Exec<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let budget_overrides = token.exec();
    let output = frame
        .on(budget_overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| attach_budget_token_failure_facts(cudf_query_error(operation, message, error), token))?;
    select_and_rename_groupby_distinct_keys(output, aggregate)
}

fn select_and_rename_groupby_distinct_keys(output: DataFrame, aggregate: &LoweredAggregate) -> Result<DataFrame> {
    let mut output =
        output.select(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            cudf_query_error(
                "query_engine.aggregate.select_group_only_keys",
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
            cudf_query_error(
                "query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    Ok(output)
}
