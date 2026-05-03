use super::*;

pub(super) fn lower_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Column(name) => Ok(col(name)),
        Expr::Literal(value) => lower_literal(value),
        Expr::Not(expr) => Ok(lower_expr(expr)?.not()),
        Expr::Cast { expr, target_type } => {
            let target_type = arrow_numeric_type_to_cudf_type_id(target_type).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.expr.lower_cast",
                    "native execution supports casts only to primitive numeric types",
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })?;
            Ok(lower_expr(expr)?.cast(target_type))
        },
        Expr::Binary { left, op, right } => lower_binary(lower_expr(left)?, *op, lower_expr(right)?),
        Expr::Alias { expr, alias } => Ok(lower_expr(expr)?.alias(alias)),
    }
}

pub(super) fn lower_literal(value: &LiteralValue) -> Result<cudf::Expr> {
    match value {
        LiteralValue::Boolean(value) => Ok(lit(*value)),
        LiteralValue::Int8(value) => Ok(lit(*value)),
        LiteralValue::Int16(value) => Ok(lit(*value)),
        LiteralValue::Int32(value) => Ok(lit(*value)),
        LiteralValue::Int64(value) => Ok(lit(*value)),
        LiteralValue::UInt8(value) => Ok(lit(*value)),
        LiteralValue::UInt16(value) => Ok(lit(*value)),
        LiteralValue::UInt32(value) => Ok(lit(*value)),
        LiteralValue::UInt64(value) => Ok(lit(*value)),
        LiteralValue::Float32(value) => Ok(lit(*value)),
        LiteralValue::Float64(value) => Ok(lit(*value)),
        LiteralValue::String(_) | LiteralValue::Null => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_literal",
            "native execution supports only non-null numeric and boolean literals",
        )
        .with_fact("literal", value)),
    }
}

pub(super) fn lower_binary(left: cudf::Expr, op: BinaryOp, right: cudf::Expr) -> Result<cudf::Expr> {
    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Subtract => Ok(left - right),
        BinaryOp::Multiply => Ok(left * right),
        BinaryOp::Divide => Ok(left / right),
        BinaryOp::Eq => Ok(left.eq(right)),
        BinaryOp::NotEq => Ok(left.ne(right)),
        BinaryOp::Lt => Ok(left.lt(right)),
        BinaryOp::LtEq => Ok(left.le(right)),
        BinaryOp::Gt => Ok(left.gt(right)),
        BinaryOp::GtEq => Ok(left.ge(right)),
        BinaryOp::And => Ok(left & right),
        BinaryOp::Or => Ok(left | right),
    }
}

pub(super) struct LoweredAggregate {
    pub(super) key_columns: Vec<String>,
    pub(super) output_key_names: Vec<String>,
    pub(super) aggregations: Vec<CudfAggSpec>,
    pub(super) output_names: Vec<String>,
}

pub(super) fn lower_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_expr")?;
            let function = lower_aggregate_function(expr.function);
            Ok(CudfAggSpec::new(input_column, function).with_alias(expr.alias.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;

    aggregate.aggregations = aggregations;
    aggregate.output_names =
        aggregate.output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect();
    Ok(aggregate)
}

pub(super) fn lower_partial_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let mut aggregate = lower_aggregate_spec_keys(spec)?;
    let mut aggregations = Vec::with_capacity(spec.aggr_exprs.len());
    let mut output_names = aggregate.output_key_names.clone();

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_partial_expr")?;
        match expr.function {
            AggregateFunction::Avg => {
                let sum_alias = avg_sum_state_alias(index);
                let count_alias = avg_count_state_alias(index);
                aggregations
                    .push(CudfAggSpec::new(input_column.as_str(), GroupByAgg::Sum).with_alias(sum_alias.as_str()));
                aggregations.push(
                    CudfAggSpec::new(input_column.as_str(), GroupByAgg::Count { include_nulls: false })
                        .with_alias(count_alias.as_str()),
                );
                output_names.push(sum_alias);
                output_names.push(count_alias);
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(input_column, lower_aggregate_function(expr.function))
                        .with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    aggregate.aggregations = aggregations;
    aggregate.output_names = output_names;
    Ok(aggregate)
}

pub(super) fn lower_final_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let partial = lower_partial_aggregate_spec(spec)?;
    let mut aggregations = Vec::with_capacity(partial.aggregations.len());
    let mut output_names = partial.output_key_names.clone();
    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let sum_alias = avg_sum_state_alias(index);
                let count_alias = avg_count_state_alias(index);
                aggregations.push(CudfAggSpec::new(sum_alias.as_str(), GroupByAgg::Sum).with_alias(sum_alias.as_str()));
                aggregations
                    .push(CudfAggSpec::new(count_alias.as_str(), GroupByAgg::Sum).with_alias(count_alias.as_str()));
                output_names.push(sum_alias);
                output_names.push(count_alias);
            },
            _ => {
                aggregations.push(
                    CudfAggSpec::new(expr.alias.as_str(), lower_final_aggregate_function(expr.function))
                        .with_alias(expr.alias.as_str()),
                );
                output_names.push(expr.alias.clone());
            },
        }
    }

    Ok(LoweredAggregate {
        key_columns: partial.output_key_names.clone(),
        output_key_names: partial.output_key_names,
        aggregations,
        output_names,
    })
}

pub(super) fn lower_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Count { include_nulls: false },
        AggregateFunction::Avg => GroupByAgg::Mean,
    }
}

pub(super) fn lower_final_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Sum,
        AggregateFunction::Avg => unreachable!("avg final aggregate is lowered as explicit sum/count state"),
    }
}

pub(super) fn lower_aggregate_spec_keys(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    if spec.group_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution currently requires at least one group key",
        ));
    }
    if spec.aggr_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution requires at least one aggregate expression",
        ));
    }

    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut output_key_names = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let (input_column, output_name) = lower_named_column_expr(expr, "cudf_query_engine.aggregate.lower_key")?;
        key_columns.push(input_column);
        output_key_names.push(output_name);
    }

    Ok(LoweredAggregate { key_columns, output_key_names, aggregations: Vec::new(), output_names: Vec::new() })
}

pub(super) fn aggregate_spec_has_avg(spec: &AggregateSpec) -> bool {
    spec.aggr_exprs.iter().any(|expr| expr.function == AggregateFunction::Avg)
}

pub(super) fn avg_sum_state_alias(index: usize) -> String {
    format!("__cudf_avg_{index}_sum")
}

pub(super) fn avg_count_state_alias(index: usize) -> String {
    format!("__cudf_avg_{index}_count")
}

pub(super) struct LoweredSort {
    pub(super) columns: Vec<String>,
    pub(super) descending: Vec<bool>,
    pub(super) nulls_first: Vec<bool>,
}

pub(super) fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "cudf_query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

pub(super) fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
    match expr {
        Expr::Column(name) => Ok((name.clone(), name.clone())),
        Expr::Alias { expr, alias } => Ok((lower_column_expr(expr, operation)?, alias.clone())),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

pub(super) fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
    match expr {
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias { expr, .. } => lower_column_expr(expr, operation),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}
