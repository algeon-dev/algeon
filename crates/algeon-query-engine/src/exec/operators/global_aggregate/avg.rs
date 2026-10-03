use arrow::datatypes::{DECIMAL128_MAX_PRECISION, SchemaRef as ArrowSchemaRef};

use super::{
    AggregateFunction, AggregateSpec, ArrowDataType, AvgStateAliases, BinaryOp, DataFrame, ExecutionBudgetToken,
    ExecutionContext, ExecutionMetrics, Expr, LiteralValue, QueryError, QueryErrorCode, Result, aggregate_spec_has_avg,
    budgeted_value, cudf_query_error, frame_column_arrow_data_type, lower_expr, validate_avg_output_fields,
};
use crate::expr::CaseWhen;

pub(in crate::exec) fn finalize_avg_output_columns(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, None)
}

pub(in crate::exec) fn finalize_avg_output_columns_with_budget(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    finalize_avg_output_columns_impl(context, frame, spec, output_schema, Some((token, metrics)))
}

fn finalize_avg_output_columns_impl(
    context: &ExecutionContext,
    frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &ArrowSchemaRef,
    token: Option<(&ExecutionBudgetToken, &ExecutionMetrics)>,
) -> Result<DataFrame> {
    if !aggregate_spec_has_avg(spec) {
        return Ok(frame);
    }

    validate_avg_output_fields(spec, output_schema)?;
    let avg_exprs = spec
        .aggr_exprs
        .iter()
        .enumerate()
        .filter(|(_, expr)| expr.function == AggregateFunction::Avg)
        .map(|(index, expr)| {
            let state = AvgStateAliases::for_aggregate_index(index);
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.aggregate.finalize_avg",
                    "avg aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            let state_sum_type =
                frame_column_arrow_data_type(&frame, state.sum(), "query_engine.aggregate.finalize_avg")?;
            let division_type = avg_division_data_type(&state_sum_type, field.data_type())?;
            let sum = Expr::cast(state.sum_expr(), division_type.clone());
            let count = Expr::cast(state.count_expr(), avg_count_output_data_type(&division_type));
            let alias = expr.alias.clone();
            let count_is_zero = Expr::binary(state.count_expr(), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(0)));
            let avg_divide = Expr::binary_with_output_type(sum, BinaryOp::Divide, count, division_type);
            let avg = Expr::cast(avg_divide, field.data_type().clone());
            let nullable_value = Expr::case(
                vec![CaseWhen::new(count_is_zero, Expr::literal(LiteralValue::Null(field.data_type().clone())))],
                avg,
            );
            lower_expr(&nullable_value).map(|lowered| (alias, lowered))
        })
        .collect::<Result<Vec<_>>>()?;
    match token {
        Some((token, metrics)) => token
            .run("dataframe::with_columns", |exec| frame.on(exec).with_columns(avg_exprs))
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.finalize_avg",
                    "failed to finalize bounded native avg aggregate output columns",
                    error,
                )
            })
            .map(|outcome| budgeted_value(outcome, metrics, "query_engine.aggregate.finalize_avg")),
        None => {
            let exec_runtime = context.runtime()?;
            let overrides = exec_runtime.exec();
            frame.on(overrides).with_columns(avg_exprs).map_err(|error| {
                cudf_query_error(
                    "query_engine.aggregate.finalize_avg",
                    "failed to finalize native avg aggregate output columns",
                    error,
                )
            })
        },
    }
}

pub(in crate::exec) fn avg_sum_state_data_type(
    input_type: &ArrowDataType,
    avg_output_type: &ArrowDataType,
) -> Result<ArrowDataType> {
    match avg_output_type {
        ArrowDataType::Float32 | ArrowDataType::Float64 => Ok(ArrowDataType::Float64),
        ArrowDataType::Decimal128(_, output_scale) => match input_type {
            ArrowDataType::Decimal128(_, input_scale) => {
                Ok(ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, *input_scale))
            },
            ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64 => Ok(ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.aggregate.avg_state_type",
                "avg sum state does not support this input type for Decimal128 output",
            )
            .with_fact("input_type", format!("{input_type:?}"))
            .with_fact("output_type", format!("{avg_output_type:?}"))),
        },
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.aggregate.avg_state_type",
            "avg sum state output schema must be Float32, Float64, or Decimal128",
        )
        .with_fact("input_type", format!("{input_type:?}"))
        .with_fact("output_type", format!("{avg_output_type:?}"))),
    }
}

fn avg_division_data_type(state_sum_type: &ArrowDataType, avg_output_type: &ArrowDataType) -> Result<ArrowDataType> {
    match avg_output_type {
        ArrowDataType::Float32 | ArrowDataType::Float64 => Ok(ArrowDataType::Float64),
        ArrowDataType::Decimal128(_, output_scale) => match state_sum_type {
            ArrowDataType::Decimal128(_, _)
            | ArrowDataType::Int8
            | ArrowDataType::Int16
            | ArrowDataType::Int32
            | ArrowDataType::Int64
            | ArrowDataType::UInt8
            | ArrowDataType::UInt16
            | ArrowDataType::UInt32
            | ArrowDataType::UInt64
            | ArrowDataType::Float32
            | ArrowDataType::Float64 => Ok(ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, *output_scale)),
            _ => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.aggregate.avg_division_type",
                "avg division does not support this sum state type for Decimal128 output",
            )
            .with_fact("state_sum_type", format!("{state_sum_type:?}"))
            .with_fact("output_type", format!("{avg_output_type:?}"))),
        },
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.aggregate.avg_division_type",
            "avg division output schema must be Float32, Float64, or Decimal128",
        )
        .with_fact("state_sum_type", format!("{state_sum_type:?}"))
        .with_fact("output_type", format!("{avg_output_type:?}"))),
    }
}

fn avg_count_output_data_type(avg_output_type: &ArrowDataType) -> ArrowDataType {
    match avg_output_type {
        ArrowDataType::Decimal128(_, _) => ArrowDataType::Decimal128(DECIMAL128_MAX_PRECISION, 0),
        _ => avg_output_type.clone(),
    }
}
