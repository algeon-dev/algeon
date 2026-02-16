use cudf::ops::ast::{AstExpression, AstOperator};

use super::*;
use crate::source::SourceColumnRef;

pub(super) fn lower_source_reader_predicate(
    context: &ExecutionContext,
    source: &LocalParquetSource,
) -> Result<Option<Arc<AstExpression>>> {
    let Some(predicate) = &source.reader_predicate else {
        return Ok(None);
    };
    if !source.reader_predicate_is_supported(predicate) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "local parquet reader predicate is not supported by the cudf reader contract",
        )
        .with_fact("predicate", predicate));
    }
    lower_source_reader_ast_expr(context, source, predicate).map(Arc::new).map(Some)
}

fn lower_source_reader_ast_expr(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    expr: &Expr,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let index = source_reader_column_index(source, column.name())?;
            AstExpression::left_column(index).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST source column reference",
                    error,
                )
                .with_fact("column", column.name())
                .with_fact("index", index)
            })
        },
        Expr::Literal(value) => lower_source_reader_ast_literal(context, value),
        Expr::Not(input) => {
            let input = lower_source_reader_ast_expr(context, source, input)?;
            AstExpression::unary(AstOperator::Not, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST NOT",
                    error,
                )
            })
        },
        Expr::IsNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, input)?;
            AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })
        },
        Expr::IsNotNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, input)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST IS NULL",
                    error,
                )
            })?;
            AstExpression::unary(AstOperator::Not, is_null).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST IS NOT NULL",
                    error,
                )
            })
        },
        Expr::Cast { expr, target_type } => {
            let input = lower_source_reader_ast_expr(context, source, expr)?;
            let expression = match target_type {
                ArrowDataType::Int64 => AstExpression::cast_to_int64(input),
                ArrowDataType::UInt64 => AstExpression::cast_to_uint64(input),
                ArrowDataType::Float64 => AstExpression::cast_to_float64(input),
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.local_parquet_source.reader_predicate",
                        "reader predicate cast is not supported by the cudf AST contract",
                    )
                    .with_fact("target_type", format!("{target_type:?}")));
                },
            };
            expression.map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST cast expression",
                    error,
                )
                .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        Expr::Binary { left, op, right, .. } => {
            let left = lower_source_reader_ast_expr(context, source, left)?;
            let right = lower_source_reader_ast_expr(context, source, right)?;
            let op = lower_source_reader_ast_operator(*op);
            AstExpression::binary(op, left, right).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST binary expression",
                    error,
                )
            })
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "reader predicate IN list must not be empty",
                ));
            }
            let input = lower_source_reader_ast_expr(context, source, expr)?;
            let values = list
                .iter()
                .map(|value| {
                    if !matches!(value, Expr::Literal(_)) {
                        return Err(QueryError::new(
                            QueryErrorCode::UnsupportedExpression,
                            "cudf_query_engine.local_parquet_source.reader_predicate",
                            "reader predicate IN list values must be literals",
                        )
                        .with_fact("value", value));
                    }
                    lower_source_reader_ast_expr(context, source, value)
                })
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.local_parquet_source.reader_predicate",
                    "failed to build cudf AST IN list expression",
                    error,
                )
                .with_fact("negated", negated.to_string())
                .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => lower_source_reader_ast_expr(context, source, expr),
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "reader predicate does not support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn source_reader_column_index(source: &LocalParquetSource, column: &str) -> Result<i32> {
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column) else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "reader predicate column must resolve to a physical read column",
        )
        .with_fact("column", column));
    };
    let Some(index) = source.read_column_file_indices.iter().position(|candidate| *candidate == file_index) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "reader predicate file column was not included in the read projection",
        )
        .with_fact("column", column)
        .with_fact("file_column_index", file_index));
    };
    i32::try_from(index).map_err(|_| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "reader predicate column index exceeds i32 range",
        )
        .with_fact("column", column)
        .with_fact("index", index)
    })
}

fn lower_source_reader_ast_operator(op: BinaryOp) -> AstOperator {
    match op {
        BinaryOp::Add => AstOperator::Add,
        BinaryOp::Subtract => AstOperator::Sub,
        BinaryOp::Multiply => AstOperator::Mul,
        BinaryOp::Divide => AstOperator::Div,
        BinaryOp::Modulo => AstOperator::Mod,
        BinaryOp::Eq => AstOperator::Equal,
        BinaryOp::NotEq => AstOperator::NotEqual,
        BinaryOp::Lt => AstOperator::Less,
        BinaryOp::LtEq => AstOperator::LessEqual,
        BinaryOp::Gt => AstOperator::Greater,
        BinaryOp::GtEq => AstOperator::GreaterEqual,
        BinaryOp::And => AstOperator::LogicalAnd,
        BinaryOp::Or => AstOperator::LogicalOr,
    }
}

fn lower_source_reader_ast_literal(context: &ExecutionContext, value: &LiteralValue) -> Result<AstExpression> {
    let runtime = context.runtime();
    let scalar = match value {
        LiteralValue::Boolean(value) => runtime.scalar(*value),
        LiteralValue::Int8(value) => runtime.scalar(*value),
        LiteralValue::Int16(value) => runtime.scalar(*value),
        LiteralValue::Int32(value) => runtime.scalar(*value),
        LiteralValue::Int64(value) => runtime.scalar(*value),
        LiteralValue::UInt8(value) => runtime.scalar(*value),
        LiteralValue::UInt16(value) => runtime.scalar(*value),
        LiteralValue::UInt32(value) => runtime.scalar(*value),
        LiteralValue::UInt64(value) => runtime.scalar(*value),
        LiteralValue::Float32(value) => runtime.scalar(*value),
        LiteralValue::Float64(value) => runtime.scalar(*value),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => runtime.string_scalar(value),
        LiteralValue::Decimal128 { raw, scale, .. } => {
            runtime.decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale))
        },
        LiteralValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), cudf::TimestampResolution::Days),
        LiteralValue::Timestamp { raw, unit } => {
            runtime.timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            runtime.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.local_parquet_source.reader_predicate",
                "reader predicate AST literals do not support interval values",
            )
            .with_fact("literal", value));
        },
        LiteralValue::Null(data_type) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                "cudf_query_engine.local_parquet_source.reader_predicate",
                "reader predicate AST literals do not yet support typed null values",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "failed to build cudf scalar for reader predicate AST literal",
            error,
        )
        .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.reader_predicate",
            "failed to build cudf AST literal",
            error,
        )
    })
}
