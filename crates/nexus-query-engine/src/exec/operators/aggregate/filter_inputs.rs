use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::*;

pub(in crate::exec) fn prepare_filtered_aggregate_frame(
    context: &ExecutionContext,
    frame: DataFrame,
    input_schema: &ArrowSchemaRef,
    spec: &AggregateSpec,
    token: Option<&ExecutionBudgetToken>,
    metrics: &mut QueryMetrics,
) -> Result<(DataFrame, AggregateSpec, ArrowSchemaRef)> {
    if !aggregate_spec_has_filters(spec) {
        return Ok((frame, spec.clone(), Arc::clone(input_schema)));
    }

    let mut expressions = Vec::<(String, cudf::Expr)>::new();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut prepared_aggr_exprs = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, aggregate_expr) in spec.aggr_exprs.iter().enumerate() {
        let Some(filter) = &aggregate_expr.filter else {
            prepared_aggr_exprs.push(aggregate_expr.clone());
            continue;
        };
        let input_type = filtered_aggregate_input_type(aggregate_expr, input_schema)?;
        let value_expr = filtered_aggregate_value_expr(aggregate_expr);
        let alias = unique_filtered_aggregate_alias(index, &fields);
        let masked = Expr::case(
            vec![CaseWhen::new(filter.clone(), value_expr)],
            Expr::literal(LiteralValue::Null(input_type.clone())),
        );
        expressions.push((alias.clone(), lower_expr(&masked)?));
        fields.push(Field::new(alias.as_str(), input_type, true));
        prepared_aggr_exprs.push(
            AggregateExpr::new(aggregate_expr.function, Expr::column(alias), aggregate_expr.alias.clone())
                .with_distinct(aggregate_expr.distinct),
        );
    }

    let timer = Instant::now();
    let frame = match token {
        Some(token) => frame
            .with_columns_with_budget(expressions, token)
            .map_err(|error| {
                QueryError::from_cudf_with_budget_details(
                    "nexus_query_engine.aggregate.filter_inputs",
                    "failed to materialize bounded native aggregate filter inputs",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "nexus_query_engine.aggregate.filter_inputs"))?,
        None => {
            let overrides_lease = execution_overrides(context)?;
            let overrides = overrides_lease.overrides();
            frame.with_execution(&overrides).with_columns(expressions).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.aggregate.filter_inputs",
                    "failed to materialize native aggregate filter inputs",
                    error,
                )
            })?
        },
    };
    metrics.operator_time += timer.elapsed();
    let mut prepared_spec = AggregateSpec::new(spec.group_exprs.clone(), prepared_aggr_exprs);
    prepared_spec.grouping_sets = spec.grouping_sets.clone();
    Ok((frame, prepared_spec, Arc::new(Schema::new(fields))))
}

pub(super) fn filtered_aggregate_value_expr(aggregate_expr: &AggregateExpr) -> Expr {
    if aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && global_count_star_expr(&aggregate_expr.expr)
    {
        Expr::literal(LiteralValue::Int64(1))
    } else {
        aggregate_expr.expr.clone()
    }
}

pub(super) fn filtered_aggregate_input_type(
    aggregate_expr: &AggregateExpr,
    input_schema: &ArrowSchemaRef,
) -> Result<ArrowDataType> {
    if aggregate_expr.function == AggregateFunction::Count
        && !aggregate_expr.distinct
        && global_count_star_expr(&aggregate_expr.expr)
    {
        return Ok(ArrowDataType::Int64);
    }
    aggregate_expr_input_type(&aggregate_expr.expr, input_schema, "nexus_query_engine.aggregate.filter_input_type")
}

pub(super) fn aggregate_expr_input_type(
    expr: &Expr,
    input_schema: &ArrowSchemaRef,
    operation: &'static str,
) -> Result<ArrowDataType> {
    match expr {
        Expr::Column(column) => {
            input_schema.field_with_name(column.name()).map(|field| field.data_type().clone()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    operation,
                    aggregate_expr_input_type_missing_column_message(operation),
                )
                .with_fact("column", column.name())
                .with_fact("arrow_error", error.to_string())
            })
        },
        Expr::Alias { expr, .. } => aggregate_expr_input_type(expr, input_schema, operation),
        Expr::Literal(value) => Ok(value.data_type()),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            aggregate_expr_input_type_unsupported_message(operation),
        )
        .with_fact("expression", expr.to_string())),
    }
}

fn aggregate_expr_input_type_missing_column_message(operation: &'static str) -> &'static str {
    match operation {
        "nexus_query_engine.filtered_global_aggregate.state_schema" => {
            "aggregate state references a column missing from its input schema"
        },
        _ => "aggregate filter input references a missing column",
    }
}

fn aggregate_expr_input_type_unsupported_message(operation: &'static str) -> &'static str {
    match operation {
        "nexus_query_engine.filtered_global_aggregate.state_schema" => {
            "filtered global aggregate state input must be a prepared column or literal"
        },
        _ => "filtered aggregate input must be a prepared column or literal",
    }
}

fn unique_filtered_aggregate_alias(index: usize, fields: &[Field]) -> String {
    let base = format!("__nexus_native_filtered_aggregate_arg_{index}");
    crate::naming::unique_alias_with_suffix(base.as_str(), |alias| fields.iter().any(|field| field.name() == alias))
}
