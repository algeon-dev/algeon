use std::time::Instant;

use cudf::interop::arrow_host::arrow_type_from_cudf_dtype;

use super::{
    ArrowDataType, DataFrame, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, Expr, GpuDataFrame,
    ProjectionNode, QueryError, QueryErrorCode, Result, attach_budget_token_failure_facts, budgeted_value,
    cudf_dtype_from_arrow_type, cudf_query_error, cudf_query_error_with_budget_details, datafusion_order_columns,
    is_supported_arrow_implicit_cast, lower_projection_output_expr, projection_schema_mismatch_error,
};
use crate::{
    expr::{alias_wrapped_column, representation_cast_column},
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    plan::ProjectionOutput,
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

pub(in crate::exec) fn execute_projection_with_budget(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    execute_projection_with_budget_and_prepared(context, node, input, token, None, metrics)
}

pub(in crate::exec) fn execute_projection_with_budget_and_prepared(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    prepared: Option<PreparedProjectionRegexpReplace>,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let frame = input.frame().clone();
    let timer = Instant::now();
    if let Some(output) = try_execute_column_only_projection(node, frame.clone(), Some(token), metrics)? {
        metrics.add_operator_time(timer.elapsed());
        return GpuDataFrame::try_new_with_source_residency(output, &input);
    }

    let frame = frame;
    let prepared = match prepared {
        Some(prepared) => Some(prepared),
        None if node.outputs.iter().any(|output| output.expr.contains_regexp_replace()) => {
            // Reuse prepared state when the caller already built it; direct
            // execution prepares under this operation's token.
            Some(prepare_projection_regexp_replace(context, node, &input, metrics)?)
        },
        None => None,
    };

    if let Some(prepared) = prepared {
        let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
        let input_rows = frame.num_rows() as u64;
        let stage_guard = RuntimeOperatorStageGuard::start(
            observation,
            plan_execution_id,
            parent_span_id,
            RuntimeOperatorStageKind::RegexReplaceExecute,
            RuntimeOperatorStageIdentity::new()
                .with_input_rows(input_rows)
                .with_reserved_bytes(prepared.required_budget_bytes),
        );
        match materialize_prepared_regexp_replace_projection(node, frame, prepared, token, metrics) {
            Ok(output_frame) => {
                let output = coerce_projection_output_schema(output_frame, node, Some(token), metrics)?;
                stage_guard.finish(Some(output.num_rows() as u64), None);
                metrics.add_operator_time(timer.elapsed());
                return GpuDataFrame::try_new_with_source_residency(output, &input);
            },
            Err(error) => {
                stage_guard.fail(&error);
                return Err(error);
            },
        }
    }

    let PreparedProjectionColumns { exprs, output_columns } = prepare_projection_columns(node, &frame)?;
    let output_columns = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output = budgeted_value(
        frame.with_columns_with_budget(exprs, token).map_err(|error| {
            cudf_query_error_with_budget_details(
                "query_engine.projection.execute",
                "failed to execute bounded algeon native projection",
                error,
            )
        })?,
        metrics,
        "query_engine.projection.execute",
    )
    .select(output_columns)
    .map_err(|error| {
        cudf_query_error(
            "query_engine.projection.execute",
            "failed to select bounded algeon native projection columns",
            error,
        )
    })?;
    output = coerce_projection_output_schema(output, node, Some(token), metrics)?;
    metrics.add_operator_time(timer.elapsed());
    GpuDataFrame::try_new_with_source_residency(output, &input)
}

/// Prepared projection components for column-input `REGEXP_REPLACE` outputs.
///
/// Independent handles are retained per regexp output. Nested non-column inputs
/// remain capability-rejected.
pub(in crate::exec) struct PreparedProjectionRegexpReplace {
    components: Vec<PreparedRegexpReplaceComponent>,
    required_budget_bytes: u64,
    retained_state_bytes: u64,
}

struct PreparedRegexpReplaceComponent {
    output_field: String,
    prepared: cudf::SeriesPreparedReplaceWithBackrefs,
}

impl PreparedProjectionRegexpReplace {
    pub(in crate::exec) fn retained_state_bytes(&self) -> u64 {
        self.retained_state_bytes
    }
}

/// Prepare reusable REGEXP_REPLACE state for a projection and return both
/// prepared handles and the hard budget required for execute.
///
/// Returns `None` when the projection has no regexp_replace. Nested non-column
/// regexp inputs are rejected fail-closed because preparation requires a
/// concrete strings column.
pub(in crate::exec) fn projection_regexp_replace_prepare(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: &GpuDataFrame,
    metrics: &ExecutionMetrics,
) -> Result<Option<PreparedProjectionRegexpReplace>> {
    if !node.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return Ok(None);
    }
    Ok(Some(prepare_projection_regexp_replace(context, node, input, metrics)?))
}

fn prepare_projection_regexp_replace(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: &GpuDataFrame,
    metrics: &ExecutionMetrics,
) -> Result<PreparedProjectionRegexpReplace> {
    let frame = input.frame();
    let input_rows = frame.num_rows() as u64;
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let stage_guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::RegexReplacePreflight,
        RuntimeOperatorStageIdentity::new().with_input_rows(input_rows),
    );
    let result = (|| {
        let mut components = Vec::new();
        let mut required_bytes = 0u64;
        let mut retained_state_bytes = 0u64;
        let mut saw_supported_regexp = false;
        for output in &node.outputs {
            match classify_regexp_replace_preflight_expr(&output.expr) {
                RegexpReplacePreflightClass::Absent => {},
                RegexpReplacePreflightClass::Unsupported => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "query_engine.projection.regexp_replace_preflight",
                        "REGEXP_REPLACE projection prepare currently requires a direct column input expression",
                    )
                    .with_fact("output_field", output.field.name().as_str())
                    .with_fact("required_contract", cudf::REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT));
                },
                RegexpReplacePreflightClass::DirectColumn { column_name, pattern, replacement } => {
                    saw_supported_regexp = true;
                    let series = frame.series(column_name).map_err(|error| {
                        cudf_query_error(
                            "query_engine.projection.regexp_replace_preflight",
                            "failed to access input column for REGEXP_REPLACE prepare",
                            error,
                        )
                        .with_fact("column", column_name)
                        .with_fact("output_field", output.field.name().as_str())
                    })?;
                    // Prepare without a stream-bound budget token so retained prepared
                    // state keeps the input series domain. A single execute token is
                    // attached later under the execute token; preparing under a
                    // separate token rebinds provenance and prevents reinsertion
                    // into the input frame.
                    let timer = Instant::now();
                    let prepared =
                        series.str().prepare_replace_with_backrefs(pattern, replacement).map_err(|error| {
                            cudf_query_error(
                                "query_engine.projection.regexp_replace_preflight",
                                "failed to prepare REGEXP_REPLACE reusable size state",
                                error,
                            )
                            .with_fact("column", column_name)
                            .with_fact("output_field", output.field.name().as_str())
                            .with_fact("required_contract", cudf::REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT)
                        })?;
                    metrics.add_operator_time(timer.elapsed());
                    let facts = prepared.output_size_facts();
                    let component_bytes = u64::try_from(facts.required_budget_bytes()).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            "query_engine.projection.regexp_replace_preflight",
                            "REGEXP_REPLACE prepare budget does not fit u64",
                        )
                        .with_fact("column", column_name)
                        .with_fact("required_budget_bytes", facts.required_budget_bytes())
                    })?;
                    let component_retained = u64::try_from(facts.retained_state_bytes()).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            "query_engine.projection.regexp_replace_preflight",
                            "REGEXP_REPLACE retained state bytes do not fit u64",
                        )
                        .with_fact("column", column_name)
                    })?;
                    required_bytes = required_bytes.checked_add(component_bytes).ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            "query_engine.projection.regexp_replace_preflight",
                            "REGEXP_REPLACE prepare budget overflowed u64",
                        )
                        .with_fact("column", column_name)
                    })?;
                    retained_state_bytes = retained_state_bytes.checked_add(component_retained).ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            "query_engine.projection.regexp_replace_preflight",
                            "REGEXP_REPLACE retained state budget overflowed u64",
                        )
                        .with_fact("column", column_name)
                    })?;
                    components
                        .push(PreparedRegexpReplaceComponent { output_field: output.field.name().clone(), prepared });
                },
            }
        }

        if !saw_supported_regexp {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "query_engine.projection.regexp_replace_preflight",
                "REGEXP_REPLACE projection prepare currently requires a direct column input expression",
            )
            .with_fact("required_contract", cudf::REPLACE_WITH_BACKREFS_OUTPUT_SIZE_PREFLIGHT_CONTRACT));
        }

        Ok(PreparedProjectionRegexpReplace { components, required_budget_bytes: required_bytes, retained_state_bytes })
    })();
    match result {
        Ok(value) => {
            stage_guard.finish(Some(input_rows), Some(value.retained_state_bytes()));
            Ok(value)
        },
        Err(error) => {
            stage_guard.fail(&error);
            Err(error)
        },
    }
}

fn materialize_prepared_regexp_replace_projection(
    node: &ProjectionNode,
    mut frame: DataFrame,
    prepared: PreparedProjectionRegexpReplace,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let prepared_by_field: std::collections::HashMap<String, cudf::SeriesPreparedReplaceWithBackrefs> =
        prepared.components.into_iter().map(|component| (component.output_field, component.prepared)).collect();

    // Materialize each prepared regexp component first (consumes its handle),
    // then evaluate remaining projection exprs against the augmented frame.
    let mut remaining = prepared_by_field;
    for output in &node.outputs {
        if let Some(handle) = remaining.remove(output.field.name().as_str()) {
            let series = budgeted_value(
                handle.execute_with_budget(token).map_err(|error| {
                    cudf_query_error_with_budget_details(
                        "query_engine.projection.execute",
                        "failed to execute prepared REGEXP_REPLACE projection component",
                        error,
                    )
                    .with_fact("output_field", output.field.name().as_str())
                })?,
                metrics,
                "query_engine.projection.execute",
            );
            frame = budgeted_value(
                frame.with_column_with_budget(output.field.name().as_str(), series, token).map_err(|error| {
                    cudf_query_error_with_budget_details(
                        "query_engine.projection.execute",
                        "failed to attach prepared REGEXP_REPLACE output column",
                        error,
                    )
                    .with_fact("output_field", output.field.name().as_str())
                })?,
                metrics,
                "query_engine.projection.execute",
            );
        }
    }

    let lower_exprs = node
        .outputs
        .iter()
        .filter(|output| {
            !matches!(
                classify_regexp_replace_preflight_expr(&output.expr),
                RegexpReplacePreflightClass::DirectColumn { .. }
            )
        })
        .map(|output| lower_projection_output_expr(output).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;

    if !lower_exprs.is_empty() {
        frame = budgeted_value(
            frame.with_columns_with_budget(lower_exprs, token).map_err(|error| {
                cudf_query_error_with_budget_details(
                    "query_engine.projection.execute",
                    "failed to execute bounded algeon native projection non-regexp expressions",
                    error,
                )
            })?,
            metrics,
            "query_engine.projection.execute",
        );
    }

    let mut output_columns = node.output_fields().map(|field| field.name().clone()).collect::<Vec<_>>();
    output_columns.extend(datafusion_order_columns(&frame));
    let output_column_refs = output_columns.iter().map(String::as_str).collect::<Vec<_>>();
    frame.select(output_column_refs).map_err(|error| {
        cudf_query_error(
            "query_engine.projection.execute",
            "failed to select bounded algeon native projection columns",
            error,
        )
    })
}

enum RegexpReplacePreflightClass<'a> {
    Absent,
    DirectColumn { column_name: &'a str, pattern: &'a str, replacement: &'a str },
    Unsupported,
}

fn classify_regexp_replace_preflight_expr(expr: &Expr) -> RegexpReplacePreflightClass<'_> {
    match expr {
        Expr::Alias { expr, .. } => classify_regexp_replace_preflight_expr(expr),
        Expr::RegexpReplace { expr, pattern, replacement, .. } => match strip_expr_aliases(expr.as_ref()) {
            Expr::Column(column) => RegexpReplacePreflightClass::DirectColumn {
                column_name: column.name(),
                pattern: pattern.as_str(),
                replacement: replacement.as_str(),
            },
            _ => RegexpReplacePreflightClass::Unsupported,
        },
        other if other.contains_regexp_replace() => RegexpReplacePreflightClass::Unsupported,
        _ => RegexpReplacePreflightClass::Absent,
    }
}

fn strip_expr_aliases(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias { expr, .. } => strip_expr_aliases(expr),
        other => other,
    }
}

struct PreparedProjectionColumns<'a> {
    exprs: Vec<(&'a str, cudf::Expr)>,
    output_columns: Vec<String>,
}

fn prepare_projection_columns<'a>(
    node: &'a ProjectionNode,
    frame: &DataFrame,
) -> Result<PreparedProjectionColumns<'a>> {
    let exprs = node
        .outputs
        .iter()
        .map(|output| lower_projection_output_expr(output).map(|expr| (output.field.name().as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let mut output_columns = node.output_fields().map(|field| field.name().clone()).collect::<Vec<_>>();
    output_columns.extend(datafusion_order_columns(frame));
    Ok(PreparedProjectionColumns { exprs, output_columns })
}

fn try_execute_column_only_projection(
    node: &ProjectionNode,
    frame: DataFrame,
    token: Option<&ExecutionBudgetToken>,
    metrics: &ExecutionMetrics,
) -> Result<Option<DataFrame>> {
    let mut mappings = Vec::with_capacity(node.outputs.len());
    for output in &node.outputs {
        let column = if let Some(column) = alias_wrapped_column(&output.expr) {
            column
        } else if let Some((column, _)) = representation_cast_column(&output.expr)
            && frame_column_cudf_data_type(&frame, column.name(), "query_engine.projection.select")
                .is_ok_and(|source| representation_cast_is_native_noop(output, source))
        {
            column
        } else {
            return Ok(None);
        };
        let source_name = column.name().to_string();
        let output_name = output.field.name().to_string();
        mappings.push((source_name, output_name));
    }
    for order_column in datafusion_order_columns(&frame) {
        let order_column = order_column.to_string();
        if mappings.iter().any(|(_, output)| output == &order_column) {
            return Ok(None);
        }
        mappings.push((order_column.clone(), order_column));
    }
    let mapping_refs = mappings.iter().map(|(source, output)| (source.as_str(), output.as_str())).collect::<Vec<_>>();
    let output = frame.select_as(mapping_refs).map_err(|error| {
        cudf_query_error("query_engine.projection.select", "failed to select native column-only projection", error)
    })?;
    coerce_projection_output_schema(output, node, token, metrics).map(Some)
}

fn representation_cast_is_native_noop(output: &ProjectionOutput, source_type: cudf::DataType) -> bool {
    representation_cast_column(&output.expr).is_some_and(|(_, target_type)| {
        target_type == output.field.data_type()
            && cudf_dtype_from_arrow_type(target_type).is_ok_and(|target| source_type == target)
    })
}

fn coerce_projection_output_schema(
    mut frame: DataFrame,
    node: &ProjectionNode,
    token: Option<&ExecutionBudgetToken>,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    for (field_index, field) in node.output_fields().enumerate() {
        let field_name = field.name();
        let actual_dtype = frame_column_cudf_data_type(&frame, field_name, "query_engine.projection.schema_coerce")?;
        let expected_type = field.data_type();
        let target_type = cudf_dtype_from_arrow_type(expected_type).map_err(|error| {
            projection_schema_mismatch_error_with_budget(
                token,
                field_index,
                field_name,
                expected_type,
                expected_type,
                "expected_type_not_supported",
            )
            .with_source(error)
        })?;
        if actual_dtype == target_type {
            continue;
        }
        let actual_type = arrow_type_from_cudf_dtype(actual_dtype).map_err(|error| {
            cudf_query_error(
                "query_engine.projection.schema_coerce",
                "failed to map native projection column type to Arrow",
                error,
            )
            .with_fact("field_name", field_name)
            .with_fact("field_index", field_index)
        })?;
        if !is_supported_arrow_implicit_cast(
            &actual_type,
            expected_type,
            rapids_analysis_context(PlanningPolicy::default()),
        ) {
            return Err(projection_schema_mismatch_error_with_budget(
                token,
                field_index,
                field_name,
                &actual_type,
                expected_type,
                "implicit_cast_not_supported",
            ));
        }
        frame = match token {
            Some(token) => budgeted_value(
                frame.cast_column_with_budget(field_name, target_type, token).map_err(|error| {
                    attach_budget_token_failure_facts(
                        cudf_query_error_with_budget_details(
                            "query_engine.projection.schema_coerce",
                            "failed to cast bounded native projection output column to declared schema type",
                            error,
                        ),
                        token,
                    )
                    .with_fact_if_absent("reason", "projection_output_schema_mismatch")
                    .with_fact("field_name", field_name)
                    .with_fact("field_index", field_index)
                    .with_fact("actual_type", format!("{actual_type:?}"))
                    .with_fact("expected_type", format!("{expected_type:?}"))
                })?,
                metrics,
                "query_engine.projection.schema_coerce",
            ),
            None => frame.cast_column(field_name, target_type).map_err(|error| {
                cudf_query_error(
                    "query_engine.projection.schema_coerce",
                    "failed to cast native projection output column to declared schema type",
                    error,
                )
                .with_fact("reason", "projection_output_schema_mismatch")
                .with_fact("field_name", field_name)
                .with_fact("field_index", field_index)
                .with_fact("actual_type", format!("{actual_type:?}"))
                .with_fact("expected_type", format!("{expected_type:?}"))
            })?,
        };
    }
    Ok(frame)
}

fn frame_column_cudf_data_type(frame: &DataFrame, column: &str, operation: &'static str) -> Result<cudf::DataType> {
    let series = frame.series(column).map_err(|error| {
        cudf_query_error(operation, "failed to access native column while inspecting its type", error)
            .with_fact("column", column)
    })?;
    series.dtype().map_err(|error| {
        cudf_query_error(operation, "failed to inspect native column type", error).with_fact("column", column)
    })
}

fn projection_schema_mismatch_error_with_budget(
    token: Option<&ExecutionBudgetToken>,
    field_index: usize,
    field_name: &str,
    actual_type: &ArrowDataType,
    expected_type: &ArrowDataType,
    reason_detail: &'static str,
) -> QueryError {
    let error = projection_schema_mismatch_error(field_index, field_name, actual_type, expected_type)
        .with_fact("reason_detail", reason_detail);
    match token {
        Some(token) => attach_budget_token_failure_facts(error, token),
        None => error,
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field};

    use super::*;

    #[test]
    fn string_representation_cast_is_native_noop() {
        let output = ProjectionOutput::new(
            Expr::cast(Expr::column("value"), DataType::Utf8View),
            Field::new("value", DataType::Utf8View, false),
        );

        assert!(representation_cast_is_native_noop(&output, cudf::DataType::new(cudf::TypeId::String)));
        assert!(!representation_cast_is_native_noop(&output, cudf::DataType::new(cudf::TypeId::Int64)));
    }
}
