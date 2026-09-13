use std::sync::Arc;

use cudf::{
    DataType as CudfDataType, Error as CudfError, TimestampResolution, TypeId,
    core::Scalar as CudfScalar,
    io::source::{
        LocalParquetSource as CudfLocalParquetSource, SourceChunk as CudfSourceChunk,
        SourceReaderPredicate as CudfSourceReaderPredicate,
        SourceReaderPredicateColumn as CudfSourceReaderPredicateColumn,
        SourceReaderPredicateLiteral as CudfSourceReaderPredicateLiteral,
        SourceReaderPredicateOp as CudfSourceReaderPredicateOp, build_local_parquet_reader_batch_predicate,
        build_local_parquet_reader_predicate,
    },
    ops::ast::{AstExpression, AstOperator},
};

use super::{
    ArrowDataType, BinaryOp, ExecutionContext, Expr, LiteralValue, LocalParquetSource, QueryError, QueryErrorCode,
    Result, cudf_query_error, cudf_query_error_without_source, timestamp_unit_to_cudf_resolution,
};
use crate::{
    datatypes::{arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type},
    rapids_backend::rapids_analysis_context,
    source::{
        ReaderPredicateDecimalScaleError, SourceColumnRef, reader_predicate_expr_references_decimal_column,
        reader_predicate_integer_literal_as, reader_predicate_integer_literal_as_decimal_raw,
        reader_predicate_rescale_decimal_raw, reader_predicate_unsigned_literal_as,
    },
};

const OP_READER_PREDICATE: &str = "query_engine.local_parquet_source.reader_predicate";

pub(super) fn lower_source_reader_predicate(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    cudf_source: &CudfLocalParquetSource,
    cudf_chunk: &CudfSourceChunk,
) -> Result<Option<Arc<AstExpression>>> {
    let Some(predicate) = &source.reader_predicate else {
        return Ok(None);
    };
    if !source.reader_predicate_is_supported(predicate, rapids_analysis_context()) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "local parquet reader predicate is not supported by the cudf reader contract",
        )
        .with_fact("predicate", predicate));
    }
    if source_reader_predicate_references_decimal_column(source, predicate) {
        return lower_source_reader_predicate_with_cudf_contract(context, source, cudf_source, cudf_chunk, predicate)
            .map(Arc::new)
            .map(Some);
    }
    lower_source_reader_ast_expr(context, source, predicate, None).map(Arc::new).map(Some)
}

pub(super) fn lower_source_reader_batch_predicate(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    cudf_source: &CudfLocalParquetSource,
    cudf_chunks: &[CudfSourceChunk],
) -> Result<Option<Arc<AstExpression>>> {
    let Some(predicate) = &source.reader_predicate else {
        return Ok(None);
    };
    if !source.reader_predicate_is_supported(predicate, rapids_analysis_context()) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "local parquet reader predicate is not supported by the cudf reader contract",
        )
        .with_fact("predicate", predicate));
    }
    if source.reader_predicate_requires_source_contract(predicate) {
        let predicate = lower_cudf_source_reader_predicate(source, predicate)?;
        return build_local_parquet_reader_batch_predicate(&context.runtime()?, cudf_source, cudf_chunks, &predicate)
            .map_err(|error| {
                cudf_reader_predicate_error(
                    source,
                    OP_READER_PREDICATE,
                    "failed to build batch-stable source-aware local parquet reader predicate",
                    error,
                )
            })
            .map(Arc::new)
            .map(Some);
    }
    lower_source_reader_ast_expr(context, source, predicate, None).map(Arc::new).map(Some)
}

fn lower_source_reader_predicate_with_cudf_contract(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    cudf_source: &CudfLocalParquetSource,
    cudf_chunk: &CudfSourceChunk,
    predicate: &Expr,
) -> Result<AstExpression> {
    let predicate = lower_cudf_source_reader_predicate(source, predicate)?;
    build_local_parquet_reader_predicate(&context.runtime()?, cudf_source, cudf_chunk, &predicate).map_err(|error| {
        cudf_reader_predicate_error(
            source,
            OP_READER_PREDICATE,
            "failed to build cudf source-aware local parquet reader predicate",
            error,
        )
    })
}

fn cudf_reader_predicate_error(
    source: &LocalParquetSource,
    operation: &'static str,
    message: &'static str,
    error: CudfError,
) -> QueryError {
    if source.files.iter().any(|file| !file.location.is_local_path()) {
        cudf_query_error_without_source(operation, message, &error)
    } else {
        cudf_query_error(operation, message, error)
    }
}

fn lower_source_reader_ast_expr(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    expr: &Expr,
    literal_target_type: Option<CudfDataType>,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let index = source_reader_column_index(source, column.name())?;
            AstExpression::left_column(index).map_err(|error| {
                cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST source column reference", error)
                    .with_fact("column", column.name())
                    .with_fact("index", index)
            })
        },
        Expr::Literal(value) => lower_source_reader_ast_literal(context, value, literal_target_type),
        Expr::Not(input) => {
            let input = lower_source_reader_ast_expr(context, source, input, None)?;
            AstExpression::unary(AstOperator::Not, input)
                .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST NOT", error))
        },
        Expr::IsNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, input, None)?;
            AstExpression::unary(AstOperator::IsNull, input)
                .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST IS NULL", error))
        },
        Expr::IsNotNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, input, None)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input)
                .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST IS NULL", error))?;
            AstExpression::unary(AstOperator::Not, is_null)
                .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST IS NOT NULL", error))
        },
        Expr::Cast { expr, target_type } => {
            let input = lower_source_reader_ast_expr(context, source, expr, None)?;
            let expression = match target_type {
                ArrowDataType::Int64 => AstExpression::cast_to_int64(input),
                ArrowDataType::UInt64 => AstExpression::cast_to_uint64(input),
                ArrowDataType::Float64 => AstExpression::cast_to_float64(input),
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        OP_READER_PREDICATE,
                        "reader predicate cast is not supported by the cudf AST contract",
                    )
                    .with_fact("target_type", format!("{target_type:?}")));
                },
            };
            expression.map_err(|error| {
                cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST cast expression", error)
                    .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        // TRY_CAST is rejected from reader-predicate pushdown: the cudf AST
        // surface has no try-cast operator and null-on-failure changes the
        // filter semantics.
        Expr::TryCast { target_type, .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "TRY_CAST is not supported in reader predicate pushdown",
        )
        .with_fact("target_type", format!("{target_type:?}"))),
        Expr::Binary { left, op, right, .. } => {
            let (left_literal_target_type, right_literal_target_type) =
                source_reader_binary_literal_target_types(source, left, *op, right);
            let left = lower_source_reader_ast_expr(context, source, left, left_literal_target_type)?;
            let right = lower_source_reader_ast_expr(context, source, right, right_literal_target_type)?;
            let op = lower_source_reader_ast_operator(*op)?;
            AstExpression::binary(op, left, right).map_err(|error| {
                cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST binary expression", error)
            })
        },
        Expr::InList { expr, list, negated } => {
            if list.is_empty() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedExpression,
                    OP_READER_PREDICATE,
                    "reader predicate IN list must not be empty",
                ));
            }
            let list_literal_target_type = source_reader_column_dtype(source, expr);
            let input = lower_source_reader_ast_expr(context, source, expr, None)?;
            let values = list
                .iter()
                .map(|value| {
                    if !matches!(value, Expr::Literal(_)) {
                        return Err(QueryError::new(
                            QueryErrorCode::UnsupportedExpression,
                            OP_READER_PREDICATE,
                            "reader predicate IN list values must be literals",
                        )
                        .with_fact("value", value));
                    }
                    lower_source_reader_ast_expr(context, source, value, list_literal_target_type)
                })
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST IN list expression", error)
                    .with_fact("negated", negated.to_string())
                    .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => lower_source_reader_ast_expr(context, source, expr, literal_target_type),
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "reader predicate does not support this expression family",
        )
        .with_fact("expression", expr)),
    }
}

fn source_reader_column_index(source: &LocalParquetSource, column: &str) -> Result<i32> {
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column) else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.local_parquet_source.reader_predicate",
            "reader predicate column must resolve to a physical read column",
        )
        .with_fact("column", column));
    };
    let Some(index) = source.read_column_file_indices.iter().position(|candidate| *candidate == file_index) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.local_parquet_source.reader_predicate",
            "reader predicate file column was not included in the read projection",
        )
        .with_fact("column", column)
        .with_fact("file_column_index", file_index));
    };
    i32::try_from(index).map_err(|_| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.local_parquet_source.reader_predicate",
            "reader predicate column index exceeds i32 range",
        )
        .with_fact("column", column)
        .with_fact("index", index)
    })
}

fn source_reader_binary_literal_target_types(
    source: &LocalParquetSource,
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> (Option<CudfDataType>, Option<CudfDataType>) {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return (None, None);
    }
    (source_reader_column_dtype(source, right), source_reader_column_dtype(source, left))
}

fn source_reader_column_dtype(source: &LocalParquetSource, expr: &Expr) -> Option<CudfDataType> {
    let column = match expr {
        Expr::Column(column) => column,
        Expr::Alias { expr, .. } => return source_reader_column_dtype(source, expr),
        _ => return None,
    };
    source
        .schema
        .field_with_name(column.name())
        .ok()
        .and_then(|field| cudf_dtype_from_arrow_type(field.data_type()).ok())
}

fn lower_cudf_source_reader_predicate(source: &LocalParquetSource, expr: &Expr) -> Result<CudfSourceReaderPredicate> {
    match expr {
        Expr::Binary { left, op, right, .. } => match op {
            BinaryOp::And => Ok(CudfSourceReaderPredicate::and(
                lower_cudf_source_reader_predicate(source, left)?,
                lower_cudf_source_reader_predicate(source, right)?,
            )),
            BinaryOp::Or => Ok(CudfSourceReaderPredicate::or(
                lower_cudf_source_reader_predicate(source, left)?,
                lower_cudf_source_reader_predicate(source, right)?,
            )),
            _ => lower_cudf_source_reader_comparison(source, left, *op, right),
        },
        Expr::InList { expr, list, negated } => {
            let column = cudf_source_reader_predicate_column(source, expr)?;
            let literals = list.iter().map(cudf_source_reader_predicate_literal).collect::<Result<Vec<_>>>()?;
            CudfSourceReaderPredicate::in_list(column, literals, *negated).map_err(|error| {
                cudf_query_error(OP_READER_PREDICATE, "failed to build cudf source reader IN list predicate", error)
            })
        },
        Expr::Not(input) => {
            Ok(CudfSourceReaderPredicate::Not(Box::new(lower_cudf_source_reader_predicate(source, input)?)))
        },
        Expr::Alias { expr, .. } => lower_cudf_source_reader_predicate(source, expr),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "reader predicate cannot be expressed with the cudf source-aware predicate contract",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_cudf_source_reader_comparison(
    source: &LocalParquetSource,
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> Result<CudfSourceReaderPredicate> {
    if let Some((column, literal, op)) = cudf_source_reader_column_literal_comparison(source, left, op, right)? {
        return Ok(CudfSourceReaderPredicate::comparison(column, op, literal));
    }
    if let Some((column, literal, op)) =
        cudf_source_reader_column_literal_comparison(source, right, invert_binary_op(op)?, left)?
    {
        return Ok(CudfSourceReaderPredicate::comparison(column, op, literal));
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedExpression,
        OP_READER_PREDICATE,
        "source-aware reader predicates require column/literal comparisons",
    )
    .with_fact("left", left)
    .with_fact("operator", op.to_string())
    .with_fact("right", right))
}

fn cudf_source_reader_column_literal_comparison(
    source: &LocalParquetSource,
    column: &Expr,
    op: BinaryOp,
    literal: &Expr,
) -> Result<Option<(CudfSourceReaderPredicateColumn, CudfSourceReaderPredicateLiteral, CudfSourceReaderPredicateOp)>> {
    let Ok(column) = cudf_source_reader_predicate_column(source, column) else {
        return Ok(None);
    };
    let literal = cudf_source_reader_predicate_literal(literal)?;
    let op = cudf_source_reader_predicate_op(op)?;
    Ok(Some((column, literal, op)))
}

fn cudf_source_reader_predicate_column(
    source: &LocalParquetSource,
    expr: &Expr,
) -> Result<CudfSourceReaderPredicateColumn> {
    let column = match expr {
        Expr::Column(column) => column,
        Expr::Alias { expr, .. } => return cudf_source_reader_predicate_column(source, expr),
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "source-aware reader predicate input must be a source column",
            )
            .with_fact("expression", expr));
        },
    };
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicate column must resolve to a physical read column",
        )
        .with_fact("column", column.name()));
    };
    let Some(read_index) = source.read_column_file_indices.iter().position(|candidate| *candidate == file_index) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "source-aware reader predicate file column was not included in the read projection",
        )
        .with_fact("column", column.name())
        .with_fact("file_column_index", file_index));
    };
    CudfSourceReaderPredicateColumn::read_column(column.name(), read_index).map_err(|error| {
        cudf_query_error(OP_READER_PREDICATE, "failed to build cudf source reader predicate column reference", error)
            .with_fact("column", column.name())
            .with_fact("read_index", read_index)
    })
}

fn cudf_source_reader_predicate_literal(expr: &Expr) -> Result<CudfSourceReaderPredicateLiteral> {
    let Expr::Literal(value) = expr else {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicate value must be a literal",
        )
        .with_fact("expression", expr));
    };
    match value {
        LiteralValue::Boolean(value) => Ok(CudfSourceReaderPredicateLiteral::Boolean(*value)),
        LiteralValue::Int8(value) => Ok(CudfSourceReaderPredicateLiteral::Int64(i64::from(*value))),
        LiteralValue::Int16(value) => Ok(CudfSourceReaderPredicateLiteral::Int64(i64::from(*value))),
        LiteralValue::Int32(value) => Ok(CudfSourceReaderPredicateLiteral::Int64(i64::from(*value))),
        LiteralValue::Int64(value) => Ok(CudfSourceReaderPredicateLiteral::Int64(*value)),
        LiteralValue::UInt8(value) => Ok(CudfSourceReaderPredicateLiteral::UInt64(u64::from(*value))),
        LiteralValue::UInt16(value) => Ok(CudfSourceReaderPredicateLiteral::UInt64(u64::from(*value))),
        LiteralValue::UInt32(value) => Ok(CudfSourceReaderPredicateLiteral::UInt64(u64::from(*value))),
        LiteralValue::UInt64(value) => Ok(CudfSourceReaderPredicateLiteral::UInt64(*value)),
        LiteralValue::Float32(value) => Ok(CudfSourceReaderPredicateLiteral::Float64(f64::from(*value))),
        LiteralValue::Float64(value) => Ok(CudfSourceReaderPredicateLiteral::Float64(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => {
            Ok(CudfSourceReaderPredicateLiteral::String(value.clone()))
        },
        LiteralValue::Decimal128 { raw, scale, .. } => Ok(CudfSourceReaderPredicateLiteral::Decimal128 {
            raw: *raw,
            scale: arrow_decimal_scale_to_cudf_scale(*scale),
        }),
        LiteralValue::Date32(value) => Ok(CudfSourceReaderPredicateLiteral::Date32(*value)),
        LiteralValue::Date64(_)
        | LiteralValue::Time32 { .. }
        | LiteralValue::Time64 { .. }
        | LiteralValue::Timestamp { .. }
        | LiteralValue::Duration { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicates do not yet support temporal literals",
        )
        .with_fact("literal", value)),
        LiteralValue::Null(data_type) => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicates do not support typed null literals",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
        LiteralValue::IntervalMonthDayNano { .. } => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicates do not support interval literals",
        )
        .with_fact("literal", value)),
    }
}

fn cudf_source_reader_predicate_op(op: BinaryOp) -> Result<CudfSourceReaderPredicateOp> {
    match op {
        BinaryOp::Eq => Ok(CudfSourceReaderPredicateOp::Equal),
        BinaryOp::NotEq => Ok(CudfSourceReaderPredicateOp::NotEqual),
        BinaryOp::Lt => Ok(CudfSourceReaderPredicateOp::Less),
        BinaryOp::LtEq => Ok(CudfSourceReaderPredicateOp::LessEqual),
        BinaryOp::Gt => Ok(CudfSourceReaderPredicateOp::Greater),
        BinaryOp::GtEq => Ok(CudfSourceReaderPredicateOp::GreaterEqual),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicates only support comparison operators",
        )
        .with_fact("operator", op.to_string())),
    }
}

fn invert_binary_op(op: BinaryOp) -> Result<BinaryOp> {
    match op {
        BinaryOp::Eq => Ok(BinaryOp::Eq),
        BinaryOp::NotEq => Ok(BinaryOp::NotEq),
        BinaryOp::Lt => Ok(BinaryOp::Gt),
        BinaryOp::LtEq => Ok(BinaryOp::GtEq),
        BinaryOp::Gt => Ok(BinaryOp::Lt),
        BinaryOp::GtEq => Ok(BinaryOp::LtEq),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "source-aware reader predicates only support invertible comparison operators",
        )
        .with_fact("operator", op.to_string())),
    }
}

fn source_reader_predicate_references_decimal_column(source: &LocalParquetSource, expr: &Expr) -> bool {
    reader_predicate_expr_references_decimal_column(expr, |column_name| {
        source
            .schema
            .field_with_name(column_name)
            .ok()
            .is_some_and(|field| matches!(field.data_type(), ArrowDataType::Decimal128(_, _)))
    })
}

fn lower_source_reader_ast_operator(op: BinaryOp) -> Result<AstOperator> {
    match op {
        BinaryOp::Add => Ok(AstOperator::Add),
        BinaryOp::Subtract => Ok(AstOperator::Sub),
        BinaryOp::Multiply => Ok(AstOperator::Mul),
        BinaryOp::Divide => Ok(AstOperator::Div),
        BinaryOp::Modulo => Ok(AstOperator::Mod),
        BinaryOp::BitwiseAnd => Ok(AstOperator::BitwiseAnd),
        BinaryOp::BitwiseOr => Ok(AstOperator::BitwiseOr),
        BinaryOp::BitwiseXor => Ok(AstOperator::BitwiseXor),
        BinaryOp::Eq => Ok(AstOperator::Equal),
        BinaryOp::NotEq => Ok(AstOperator::NotEqual),
        BinaryOp::Lt => Ok(AstOperator::Less),
        BinaryOp::LtEq => Ok(AstOperator::LessEqual),
        BinaryOp::Gt => Ok(AstOperator::Greater),
        BinaryOp::GtEq => Ok(AstOperator::GreaterEqual),
        BinaryOp::And => Ok(AstOperator::NullLogicalAnd),
        BinaryOp::Or => Ok(AstOperator::NullLogicalOr),
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "query_engine.local_parquet_source.reader_predicate",
            "reader predicate does not support bit-shift operators",
        )
        .with_fact("operator", op.to_string())),
    }
}

fn lower_source_reader_ast_literal(
    context: &ExecutionContext,
    value: &LiteralValue,
    target_type: Option<CudfDataType>,
) -> Result<AstExpression> {
    let runtime = context.runtime()?;
    if let Some(target_type) = target_type
        && let Some(scalar) = lower_source_reader_ast_literal_with_target(context, value, target_type)?
    {
        return AstExpression::literal(scalar)
            .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST literal", error));
    }
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
        LiteralValue::Decimal128 { raw, scale, .. } => match target_type {
            Some(target_type) if target_type.id().is_decimal() => {
                let raw = rescale_decimal_raw_or_query_error(
                    *raw,
                    arrow_decimal_scale_to_cudf_scale(*scale),
                    target_type.scale(),
                )?;
                runtime.decimal_raw_scalar(raw, target_type)
            },
            _ => runtime.decimal128_raw_scalar(*raw, arrow_decimal_scale_to_cudf_scale(*scale)),
        },
        LiteralValue::Date32(value) => runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days),
        LiteralValue::Date64(value) => runtime.timestamp_raw_scalar(*value, TimestampResolution::Milliseconds),
        LiteralValue::Time32 { raw, unit } => {
            runtime.duration_raw_scalar(i64::from(*raw), timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Time64 { raw, unit } => {
            runtime.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Timestamp { raw, unit } => {
            runtime.timestamp_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::Duration { raw, unit } => {
            runtime.duration_raw_scalar(*raw, timestamp_unit_to_cudf_resolution(unit))
        },
        LiteralValue::IntervalMonthDayNano { .. } => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "reader predicate AST literals do not support interval values",
            )
            .with_fact("literal", value));
        },
        LiteralValue::Null(data_type) => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "reader predicate AST literals do not yet support typed null values",
            )
            .with_fact("data_type", format!("{data_type:?}")));
        },
    }
    .map_err(|error| {
        cudf_query_error(OP_READER_PREDICATE, "failed to build cudf scalar for reader predicate AST literal", error)
            .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar)
        .map_err(|error| cudf_query_error(OP_READER_PREDICATE, "failed to build cudf AST literal", error))
}

fn lower_source_reader_ast_literal_with_target(
    context: &ExecutionContext,
    value: &LiteralValue,
    target_type: CudfDataType,
) -> Result<Option<CudfScalar>> {
    let runtime = context.runtime()?;
    match target_type.id() {
        TypeId::Bool8 => match value {
            LiteralValue::Boolean(value) => runtime.scalar(*value).map(Some),
            _ => Ok(None),
        },
        TypeId::Int8 => integer_literal_as::<i8>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::Int16 => integer_literal_as::<i16>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::Int32 => integer_literal_as::<i32>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::Int64 => integer_literal_as::<i64>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::UInt8 => unsigned_literal_as::<u8>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::UInt16 => unsigned_literal_as::<u16>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::UInt32 => unsigned_literal_as::<u32>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::UInt64 => unsigned_literal_as::<u64>(value).map(|value| runtime.scalar(value)).transpose(),
        TypeId::Float32 => match value {
            LiteralValue::Float32(value) => runtime.scalar(*value).map(Some),
            LiteralValue::Int8(value) => runtime.scalar(f32::from(*value)).map(Some),
            LiteralValue::Int16(value) => runtime.scalar(f32::from(*value)).map(Some),
            LiteralValue::Int32(value) => runtime.scalar(*value as f32).map(Some),
            LiteralValue::UInt8(value) => runtime.scalar(f32::from(*value)).map(Some),
            LiteralValue::UInt16(value) => runtime.scalar(f32::from(*value)).map(Some),
            _ => Ok(None),
        },
        TypeId::Float64 => match value {
            LiteralValue::Float32(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::Float64(value) => runtime.scalar(*value).map(Some),
            LiteralValue::Int8(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::Int16(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::Int32(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::UInt8(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::UInt16(value) => runtime.scalar(f64::from(*value)).map(Some),
            LiteralValue::UInt32(value) => runtime.scalar(f64::from(*value)).map(Some),
            _ => Ok(None),
        },
        TypeId::String => match value {
            LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => runtime.string_scalar(value).map(Some),
            _ => Ok(None),
        },
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
            let raw = if let Some(raw) = integer_literal_as_decimal_raw_or_query_error(value, target_type.scale())? {
                raw
            } else if let LiteralValue::Decimal128 { raw, scale, .. } = value {
                rescale_decimal_raw_or_query_error(
                    *raw,
                    arrow_decimal_scale_to_cudf_scale(*scale),
                    target_type.scale(),
                )?
            } else {
                return Ok(None);
            };
            runtime.decimal_raw_scalar(raw, target_type).map(Some)
        },
        TypeId::TimestampDays => match value {
            LiteralValue::Date32(value) => {
                runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days).map(Some)
            },
            _ => Ok(None),
        },
        _ => Ok(None),
    }
    .map_err(|error| {
        cudf_query_error(
            OP_READER_PREDICATE,
            "failed to build cudf scalar for typed reader predicate AST literal",
            error,
        )
        .with_fact("literal", value)
        .with_fact("target_type_id", target_type.id())
        .with_fact("target_type_scale", target_type.scale())
    })
}

fn integer_literal_as<T>(value: &LiteralValue) -> Option<T>
where T: TryFrom<i128> {
    reader_predicate_integer_literal_as(value)
}

fn unsigned_literal_as<T>(value: &LiteralValue) -> Option<T>
where T: TryFrom<u128> {
    reader_predicate_unsigned_literal_as(value)
}

fn integer_literal_as_decimal_raw_or_query_error(value: &LiteralValue, target_scale: i32) -> Result<Option<i128>> {
    reader_predicate_integer_literal_as_decimal_raw(value, target_scale).map_err(decimal_scale_error_to_query_error)
}

fn rescale_decimal_raw_or_query_error(raw: i128, source_scale: i32, target_scale: i32) -> Result<i128> {
    reader_predicate_rescale_decimal_raw(raw, source_scale, target_scale).map_err(decimal_scale_error_to_query_error)
}

fn decimal_scale_error_to_query_error(error: ReaderPredicateDecimalScaleError) -> QueryError {
    match error {
        ReaderPredicateDecimalScaleError::ScaleFactorOverflow { exponent } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "decimal reader predicate scale factor overflows i128",
        )
        .with_fact("exponent", exponent),
        ReaderPredicateDecimalScaleError::IntegerRawOverflow { target_scale } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "integer reader predicate literal raw value overflows target decimal scale",
        )
        .with_fact("target_scale", target_scale),
        ReaderPredicateDecimalScaleError::IntegerInexact { target_scale } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "integer reader predicate literal cannot be represented exactly at target decimal scale",
        )
        .with_fact("target_scale", target_scale),
        ReaderPredicateDecimalScaleError::ScaleAlignmentOverflow { source_scale, target_scale } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "decimal reader predicate literal scale alignment overflowed",
        )
        .with_fact("source_scale", source_scale)
        .with_fact("target_scale", target_scale),
        ReaderPredicateDecimalScaleError::DecimalRawOverflow { source_scale, target_scale } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "decimal reader predicate literal raw value overflows target scale",
        )
        .with_fact("source_scale", source_scale)
        .with_fact("target_scale", target_scale),
        ReaderPredicateDecimalScaleError::DecimalInexact { source_scale, target_scale } => QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "decimal reader predicate literal cannot be represented exactly at target scale",
        )
        .with_fact("source_scale", source_scale)
        .with_fact("target_scale", target_scale),
    }
}
