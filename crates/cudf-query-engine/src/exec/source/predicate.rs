use std::path::Path;

use cudf::{
    DataType as CudfDataType, TypeId,
    ops::ast::{AstExpression, AstOperator},
};
use parquet::{
    basic::{ConvertedType, LogicalType, Type as ParquetPhysicalType},
    file::reader::{FileReader, SerializedFileReader},
    schema::types::ColumnDescriptor,
};

use super::*;
use crate::source::SourceColumnRef;

const OP_READER_PREDICATE: &str = "cudf_query_engine.local_parquet_source.reader_predicate";

pub(super) fn lower_source_reader_predicate(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    parquet_path: &Path,
) -> Result<Option<Arc<AstExpression>>> {
    let Some(predicate) = &source.reader_predicate else {
        return Ok(None);
    };
    if !source.reader_predicate_is_supported(predicate) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "local parquet reader predicate is not supported by the cudf reader contract",
        )
        .with_fact("predicate", predicate));
    }
    lower_source_reader_ast_expr(context, source, parquet_path, predicate, None).map(Arc::new).map(Some)
}

fn lower_source_reader_ast_expr(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    parquet_path: &Path,
    expr: &Expr,
    literal_target_type: Option<CudfDataType>,
) -> Result<AstExpression> {
    match expr {
        Expr::Column(column) => {
            let index = source_reader_column_index(source, column.name())?;
            AstExpression::left_column(index).map_err(|error| {
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST source column reference", error)
                    .with_fact("column", column.name())
                    .with_fact("index", index)
            })
        },
        Expr::Literal(value) => lower_source_reader_ast_literal(context, value, literal_target_type),
        Expr::Not(input) => {
            let input = lower_source_reader_ast_expr(context, source, parquet_path, input, None)?;
            AstExpression::unary(AstOperator::Not, input)
                .map_err(|error| QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST NOT", error))
        },
        Expr::IsNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, parquet_path, input, None)?;
            AstExpression::unary(AstOperator::IsNull, input)
                .map_err(|error| QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST IS NULL", error))
        },
        Expr::IsNotNull(input) => {
            let input = lower_source_reader_ast_expr(context, source, parquet_path, input, None)?;
            let is_null = AstExpression::unary(AstOperator::IsNull, input).map_err(|error| {
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST IS NULL", error)
            })?;
            AstExpression::unary(AstOperator::Not, is_null).map_err(|error| {
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST IS NOT NULL", error)
            })
        },
        Expr::Cast { expr, target_type } => {
            let input = lower_source_reader_ast_expr(context, source, parquet_path, expr, None)?;
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
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST cast expression", error)
                    .with_fact("target_type", format!("{target_type:?}"))
            })
        },
        Expr::Binary { left, op, right, .. } => {
            let (left_literal_target_type, right_literal_target_type) =
                source_reader_binary_literal_target_types(source, parquet_path, left, *op, right)?;
            let left = lower_source_reader_ast_expr(context, source, parquet_path, left, left_literal_target_type)?;
            let right = lower_source_reader_ast_expr(context, source, parquet_path, right, right_literal_target_type)?;
            let op = lower_source_reader_ast_operator(*op);
            AstExpression::binary(op, left, right).map_err(|error| {
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST binary expression", error)
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
            let list_literal_target_type = source_reader_decimal_column_dtype(source, parquet_path, expr)?;
            let input = lower_source_reader_ast_expr(context, source, parquet_path, expr, None)?;
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
                    lower_source_reader_ast_expr(context, source, parquet_path, value, list_literal_target_type)
                })
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST IN list expression", error)
                    .with_fact("negated", negated.to_string())
                    .with_fact("value_count", list.len())
            })
        },
        Expr::Alias { expr, .. } => {
            lower_source_reader_ast_expr(context, source, parquet_path, expr, literal_target_type)
        },
        Expr::Like { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Case { .. }
        | Expr::Extract { .. } => Err(QueryError::new(
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

fn source_reader_binary_literal_target_types(
    source: &LocalParquetSource,
    parquet_path: &Path,
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> Result<(Option<CudfDataType>, Option<CudfDataType>)> {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return Ok((None, None));
    }
    Ok((
        source_reader_decimal_column_dtype(source, parquet_path, right)?,
        source_reader_decimal_column_dtype(source, parquet_path, left)?,
    ))
}

fn source_reader_decimal_column_dtype(
    source: &LocalParquetSource,
    parquet_path: &Path,
    expr: &Expr,
) -> Result<Option<CudfDataType>> {
    let column = match expr {
        Expr::Column(column) => column,
        Expr::Alias { expr, .. } => return source_reader_decimal_column_dtype(source, parquet_path, expr),
        _ => return Ok(None),
    };
    let Some(field) = source.schema.field_with_name(column.name()).ok() else {
        return Ok(None);
    };
    if !matches!(field.data_type(), ArrowDataType::Decimal128(_, _)) {
        return Ok(None);
    }
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
        return Ok(None);
    };
    parquet_reader_decimal_dtype(parquet_path, file_index, column.name())
}

fn parquet_reader_decimal_dtype(
    parquet_path: &Path,
    file_column_index: usize,
    column_name: &str,
) -> Result<Option<CudfDataType>> {
    let reader = SerializedFileReader::try_from(parquet_path).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "failed to read Parquet footer for decimal reader predicate typing",
        )
        .with_fact("path", parquet_path.display())
        .with_fact("column", column_name)
        .with_fact("file_column_index", file_column_index)
        .with_fact("source_error", error.to_string())
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let descriptor = schema
        .columns()
        .get(file_column_index)
        .filter(|descriptor| parquet_descriptor_matches_column(descriptor, column_name))
        .or_else(|| {
            schema.columns().iter().find(|descriptor| parquet_descriptor_matches_column(descriptor, column_name))
        });
    let Some(descriptor) = descriptor else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OP_READER_PREDICATE,
            "reader predicate decimal column was not found in Parquet footer",
        )
        .with_fact("path", parquet_path.display())
        .with_fact("column", column_name)
        .with_fact("file_column_index", file_column_index));
    };

    let Some((precision, scale)) = parquet_decimal_precision_scale(descriptor) else {
        return Ok(None);
    };
    let type_id = parquet_reader_decimal_type_id(descriptor, precision)?;
    Ok(Some(CudfDataType::fixed_point(type_id, -scale)))
}

fn parquet_descriptor_matches_column(descriptor: &ColumnDescriptor, column_name: &str) -> bool {
    descriptor.name() == column_name || descriptor.path().parts().last().is_some_and(|part| part == column_name)
}

fn parquet_decimal_precision_scale(descriptor: &ColumnDescriptor) -> Option<(i32, i32)> {
    if let Some(LogicalType::Decimal { precision, scale }) = descriptor.logical_type_ref() {
        return Some((*precision, *scale));
    }
    (descriptor.converted_type() == ConvertedType::DECIMAL)
        .then(|| (descriptor.type_precision(), descriptor.type_scale()))
}

fn parquet_reader_decimal_type_id(descriptor: &ColumnDescriptor, precision: i32) -> Result<TypeId> {
    match descriptor.physical_type() {
        ParquetPhysicalType::INT32 => Ok(TypeId::Decimal32),
        ParquetPhysicalType::INT64 => Ok(TypeId::Decimal64),
        ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY => match descriptor.type_length() {
            0..=4 => Ok(TypeId::Decimal32),
            5..=8 => Ok(TypeId::Decimal64),
            9..=16 => Ok(TypeId::Decimal128),
            type_length => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                OP_READER_PREDICATE,
                "Parquet decimal fixed byte width exceeds libcudf Decimal128 storage",
            )
            .with_fact("type_length", type_length)),
        },
        ParquetPhysicalType::BYTE_ARRAY => match precision {
            1..=9 => Ok(TypeId::Decimal32),
            10..=18 => Ok(TypeId::Decimal64),
            19..=38 => Ok(TypeId::Decimal128),
            precision => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                OP_READER_PREDICATE,
                "Parquet decimal precision exceeds libcudf Decimal128 storage",
            )
            .with_fact("precision", precision)),
        },
        physical_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            OP_READER_PREDICATE,
            "Parquet decimal physical type is not supported by libcudf reader predicates",
        )
        .with_fact("physical_type", format!("{physical_type:?}"))),
    }
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

fn lower_source_reader_ast_literal(
    context: &ExecutionContext,
    value: &LiteralValue,
    target_type: Option<CudfDataType>,
) -> Result<AstExpression> {
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
        LiteralValue::Decimal128 { raw, scale, .. } => match target_type {
            Some(target_type) if target_type.id().is_decimal() => {
                let raw = rescale_decimal_raw(
                    *raw,
                    crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale),
                    target_type.scale(),
                )?;
                runtime.decimal_raw_scalar(raw, target_type)
            },
            _ => runtime.decimal128_raw_scalar(*raw, crate::datatypes::arrow_decimal_scale_to_cudf_scale(*scale)),
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
        QueryError::from_cudf(
            OP_READER_PREDICATE,
            "failed to build cudf scalar for reader predicate AST literal",
            error,
        )
        .with_fact("literal", value)
    })?;
    AstExpression::literal(scalar)
        .map_err(|error| QueryError::from_cudf(OP_READER_PREDICATE, "failed to build cudf AST literal", error))
}

fn rescale_decimal_raw(raw: i128, source_scale: i32, target_scale: i32) -> Result<i128> {
    let scale_delta = source_scale.checked_sub(target_scale).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            OP_READER_PREDICATE,
            "decimal reader predicate literal scale alignment overflowed",
        )
        .with_fact("source_scale", source_scale)
        .with_fact("target_scale", target_scale)
    })?;
    if scale_delta >= 0 {
        let multiplier = checked_pow10(scale_delta as u32)?;
        raw.checked_mul(multiplier).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "decimal reader predicate literal raw value overflows target scale",
            )
            .with_fact("source_scale", source_scale)
            .with_fact("target_scale", target_scale)
        })
    } else {
        let divisor = checked_pow10(scale_delta.unsigned_abs())?;
        if raw % divisor != 0 {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "decimal reader predicate literal cannot be represented exactly at target scale",
            )
            .with_fact("source_scale", source_scale)
            .with_fact("target_scale", target_scale));
        }
        Ok(raw / divisor)
    }
}

fn checked_pow10(exponent: u32) -> Result<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedExpression,
                OP_READER_PREDICATE,
                "decimal reader predicate scale factor overflows i128",
            )
            .with_fact("exponent", exponent)
        })?;
    }
    Ok(value)
}
