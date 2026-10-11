use std::{sync::Arc, time::Instant};

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use query_planning::{expr::alias_wrapped_column, naming::unique_alias_with_suffix, prepared::avg_argument_cast_type};

use super::{
    AggregateExpr, AggregateFunction, AggregateSpec, ArrowDataType, CaseWhen, DataFrame, ExecutionBudgetToken,
    ExecutionContext, ExecutionMetrics, Expr, Field, LiteralValue, QueryError, QueryErrorCode, Result, Schema,
    budgeted_value, cudf_query_error, global_count_star_expr, lower_expr,
};

/// Materializes the aggregate arguments cuDF cannot read in place, and returns
/// the frame, spec and schema that reference them.
///
/// A `FILTER` clause masks its argument to null where the predicate is not
/// true. A grouped aggregate also casts an AVG argument that its sum state does
/// not accumulate as it is, such as an integer, to the state's type. A global
/// aggregate casts it inside the reduction.
pub(in crate::exec) fn prepare_aggregate_input_frame(
    context: &ExecutionContext,
    frame: DataFrame,
    input_schema: &ArrowSchemaRef,
    spec: &AggregateSpec,
    token: Option<&ExecutionBudgetToken>,
    metrics: &ExecutionMetrics,
) -> Result<(DataFrame, AggregateSpec, ArrowSchemaRef)> {
    let grouped = !spec.group_exprs.is_empty();
    let mut expressions = Vec::<(String, cudf::Expr)>::new();
    let mut fields = input_schema.fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    let mut prepared_aggr_exprs = Vec::with_capacity(spec.aggr_exprs.len());
    for (index, aggregate_expr) in spec.aggr_exprs.iter().enumerate() {
        let Some(argument) =
            materialized_aggregate_argument(aggregate_expr, aggregate_expr.filter.clone(), grouped, input_schema)?
        else {
            prepared_aggr_exprs.push(aggregate_expr.clone());
            continue;
        };
        let alias = unique_aggregate_input_alias(index, &fields);
        expressions.push((alias.clone(), lower_expr(&argument.value)?));
        fields.push(Field::new(alias.as_str(), argument.data_type, true));
        prepared_aggr_exprs.push(
            AggregateExpr::new(aggregate_expr.function, Expr::column(alias), aggregate_expr.alias.clone())
                .with_distinct(aggregate_expr.distinct),
        );
    }
    if expressions.is_empty() {
        return Ok((frame, spec.clone(), Arc::clone(input_schema)));
    }

    let timer = Instant::now();
    let frame = match token {
        Some(token) => token
            .run("dataframe::with_columns", |exec| frame.on(exec).with_columns(expressions))
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.prepare_inputs",
                    "failed to materialize bounded native aggregate inputs",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "query_engine.aggregate.prepare_inputs"))?,
        None => {
            let exec_runtime = context.runtime()?;
            let overrides = exec_runtime.exec();
            frame.on(overrides).with_columns(expressions).map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.prepare_inputs",
                    "failed to materialize native aggregate inputs",
                    error,
                )
            })?
        },
    };
    metrics.add_operator_time(timer.elapsed());
    let mut prepared_spec = AggregateSpec::new(spec.group_exprs.clone(), prepared_aggr_exprs);
    prepared_spec.grouping_sets = spec.grouping_sets.clone();
    Ok((frame, prepared_spec, Arc::new(Schema::new(fields))))
}

/// An aggregate argument computed into a column of its own.
pub(super) struct MaterializedAggregateArgument {
    pub(super) value: Expr,
    pub(super) data_type: ArrowDataType,
}

/// Returns how `aggregate_expr`'s argument is materialized, or `None` when
/// cuDF reads it in place. `predicate` is the aggregate's `FILTER` clause or the
/// column that holds its value. `cast_avg` casts an AVG argument to the type its
/// sum state accumulates in.
pub(super) fn materialized_aggregate_argument(
    aggregate_expr: &AggregateExpr,
    predicate: Option<Expr>,
    cast_avg: bool,
    input_schema: &ArrowSchemaRef,
) -> Result<Option<MaterializedAggregateArgument>> {
    let cast_type = if cast_avg { avg_input_cast_type(aggregate_expr, input_schema)? } else { None };
    if predicate.is_none() && cast_type.is_none() {
        return Ok(None);
    }
    let mut data_type = filtered_aggregate_input_type(aggregate_expr, input_schema)?;
    let mut value = filtered_aggregate_value_expr(aggregate_expr);
    if let Some(cast_type) = cast_type {
        value = Expr::cast(value, cast_type.clone());
        data_type = cast_type;
    }
    if let Some(predicate) = predicate {
        value = Expr::case(vec![CaseWhen::new(predicate, value)], Expr::literal(LiteralValue::Null(data_type.clone())));
    }
    Ok(Some(MaterializedAggregateArgument { value, data_type }))
}

/// Returns the type of the column that holds a grouped aggregate's argument
/// once it is materialized.
pub(super) fn grouped_aggregate_input_type(
    aggregate_expr: &AggregateExpr,
    input_schema: &ArrowSchemaRef,
) -> Result<ArrowDataType> {
    match avg_input_cast_type(aggregate_expr, input_schema)? {
        Some(cast_type) => Ok(cast_type),
        None => filtered_aggregate_input_type(aggregate_expr, input_schema),
    }
}

/// Returns the type the grouped sum state of an AVG accumulates its argument in,
/// when that is not the argument's own type.
fn avg_input_cast_type(aggregate_expr: &AggregateExpr, input_schema: &ArrowSchemaRef) -> Result<Option<ArrowDataType>> {
    if aggregate_expr.function != AggregateFunction::Avg || alias_wrapped_column(&aggregate_expr.expr).is_none() {
        return Ok(None);
    }
    let input_type =
        aggregate_expr_input_type(&aggregate_expr.expr, input_schema, "query_engine.aggregate.avg_input_type")?;
    Ok(avg_argument_cast_type(&input_type))
}

/// The value a filtered aggregate masks. COUNT(*) without DISTINCT has no argument, so it masks
/// the literal 1 to leave something that is NULL where the filter excludes a row; every other
/// aggregate masks its own argument.
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
    aggregate_expr_input_type(&aggregate_expr.expr, input_schema, "query_engine.aggregate.filter_input_type")
}

/// Arrow type of a prepared aggregate argument, which must be a column, an alias of one, or a
/// literal because arguments are materialized into columns before aggregation. Any other
/// expression fails with `UnsupportedExpression`.
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
        "query_engine.filtered_global_aggregate.state_schema" => {
            "aggregate state references a column missing from its input schema"
        },
        _ => "aggregate filter input references a missing column",
    }
}

fn aggregate_expr_input_type_unsupported_message(operation: &'static str) -> &'static str {
    match operation {
        "query_engine.filtered_global_aggregate.state_schema" => {
            "filtered global aggregate state input must be a prepared column or literal"
        },
        _ => "filtered aggregate input must be a prepared column or literal",
    }
}

fn unique_aggregate_input_alias(index: usize, fields: &[Field]) -> String {
    let base = format!("__algeon_native_aggregate_input_{index}");
    unique_alias_with_suffix(base.as_str(), |alias| fields.iter().any(|field| field.name() == alias))
}
