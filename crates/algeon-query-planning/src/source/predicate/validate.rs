//! Reader-predicate capability and source-contract validation.

use arrow::datatypes::DataType;

use crate::{
    backend::AnalysisContext,
    datatypes::{
        arrow_decimal_scale_to_cudf_scale, is_narrow_decimal_carry_type, is_supported_list_carry_type,
        is_timestamp_with_timezone,
    },
    expr::{BinaryOp, Expr, LiteralValue},
    source::{
        parquet::{LocalParquetSource, SourceColumnRef},
        reader_predicate_decimal_raw_for_integer_value, reader_predicate_expr_references_decimal_column,
        reader_predicate_integer_literal_raw, reader_predicate_rescale_decimal_raw,
        reader_predicate_unsigned_literal_raw,
    },
};

pub(in crate::source) fn reader_predicate_root_is_supported(expr: &Expr) -> bool {
    match expr {
        Expr::Alias { expr, .. } => reader_predicate_root_is_supported(expr),
        Expr::Column(_) | Expr::Literal(_) => false,
        Expr::Not(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::Binary { .. }
        | Expr::InList { .. }
        | Expr::Like { .. }
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
        | Expr::UnaryMath { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => true,
    }
}

pub(in crate::source) fn reader_predicate_expr_is_supported(
    source: &LocalParquetSource,
    expr: &Expr,
    ctx: AnalysisContext<'_>,
) -> bool {
    match expr {
        Expr::Column(column) => {
            let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
                return false;
            };
            source.read_column_file_indices.contains(&file_index)
                && source
                    .schema
                    .field_with_name(column.name())
                    .ok()
                    .is_some_and(|field| reader_predicate_column_type_is_supported(field.data_type(), ctx))
        },
        Expr::Literal(value) => reader_predicate_literal_is_supported(value),
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) | Expr::Alias { expr: inner, .. } => {
            reader_predicate_expr_is_supported(source, inner, ctx)
        },
        Expr::Cast { expr, target_type } => {
            matches!(target_type, DataType::Int64 | DataType::UInt64 | DataType::Float64)
                && reader_predicate_expr_is_supported(source, expr, ctx)
        },
        Expr::Binary { left, op, right, .. } => {
            reader_predicate_binary_operator_is_supported(*op)
                && reader_predicate_expr_is_supported(source, left, ctx)
                && reader_predicate_expr_is_supported(source, right, ctx)
                && reader_predicate_binary_operand_types_are_supported(source, left, *op, right, ctx)
        },
        Expr::InList { expr, list, .. } => {
            !list.is_empty()
                && reader_predicate_expr_is_supported(source, expr, ctx)
                && reader_predicate_in_list_literal_types_are_supported(source, expr, list, ctx)
                && list.iter().all(|value| {
                    matches!(value, Expr::Literal(_)) && reader_predicate_expr_is_supported(source, value, ctx)
                })
        },
        // TRY_CAST is rejected from reader-predicate pushdown: null-on-failure
        // semantics interact differently with filter logic than strict CAST.
        Expr::TryCast { .. }
        | Expr::Like { .. }
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
        | Expr::UnaryMath { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => false,
    }
}

fn reader_predicate_binary_operator_is_supported(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo
            | BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
    )
}

fn reader_predicate_operator_outputs_boolean(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
    )
}

fn reader_predicate_column_type_is_supported(data_type: &DataType, ctx: AnalysisContext<'_>) -> bool {
    ctx.backend().supports_type(data_type)
        && !is_timestamp_with_timezone(data_type)
        && !is_supported_list_carry_type(data_type)
        && !is_narrow_decimal_carry_type(data_type)
}

fn reader_predicate_literal_is_supported(value: &LiteralValue) -> bool {
    !matches!(value, LiteralValue::Null(_) | LiteralValue::IntervalMonthDayNano { .. })
}

fn reader_predicate_binary_operand_types_are_supported(
    source: &LocalParquetSource,
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
    ctx: AnalysisContext<'_>,
) -> bool {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return true;
    }
    reader_predicate_literal_matches_expr_type(source, left, right, ctx)
        && reader_predicate_literal_matches_expr_type(source, right, left, ctx)
}

fn reader_predicate_in_list_literal_types_are_supported(
    source: &LocalParquetSource,
    expr: &Expr,
    list: &[Expr],
    ctx: AnalysisContext<'_>,
) -> bool {
    list.iter().all(|value| reader_predicate_literal_matches_expr_type(source, value, expr, ctx))
}

fn reader_predicate_literal_matches_expr_type(
    source: &LocalParquetSource,
    literal: &Expr,
    typed_expr: &Expr,
    ctx: AnalysisContext<'_>,
) -> bool {
    let Expr::Literal(value) = literal else {
        return true;
    };
    let Some(data_type) = reader_predicate_expr_type(source, typed_expr) else {
        return true;
    };
    reader_predicate_literal_is_supported_for_type(value, &data_type, ctx)
}

fn reader_predicate_expr_type(source: &LocalParquetSource, expr: &Expr) -> Option<DataType> {
    match expr {
        Expr::Column(column) => {
            source.schema.field_with_name(column.name()).ok().map(|field| field.data_type().clone())
        },
        Expr::Literal(value) => Some(value.data_type()),
        Expr::Alias { expr, .. } => reader_predicate_expr_type(source, expr),
        Expr::Cast { target_type, .. }
        | Expr::TryCast { target_type, .. }
        | Expr::Round { output_type: target_type, .. } => Some(target_type.clone()),
        Expr::Not(_) | Expr::IsNull(_) | Expr::IsNotNull(_) | Expr::InList { .. } => Some(DataType::Boolean),
        Expr::StringContains { .. } | Expr::StartsWith { .. } | Expr::EndsWith { .. } => Some(DataType::Boolean),
        Expr::IsDistinctFrom { .. } | Expr::IsNotDistinctFrom { .. } => Some(DataType::Boolean),
        Expr::Binary { output_type: Some(output_type), .. } => Some(output_type.clone()),
        Expr::Binary { op, .. } if reader_predicate_operator_outputs_boolean(*op) => Some(DataType::Boolean),
        Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::Concat(_) => Some(DataType::Utf8),
        Expr::StringLength { output_type, .. } => Some(output_type.clone()),
        Expr::RegexpReplace { output_type, .. } => Some(output_type.clone()),
        Expr::DateTrunc { output_type, .. } => Some(output_type.clone()),
        Expr::Substring { .. }
        | Expr::Like { .. }
        | Expr::Abs(_)
        | Expr::UnaryMath { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::Binary { .. } => None,
    }
}

fn reader_predicate_literal_is_supported_for_type(
    value: &LiteralValue,
    data_type: &DataType,
    ctx: AnalysisContext<'_>,
) -> bool {
    if !reader_predicate_literal_is_supported(value) || !ctx.backend().supports_type(data_type) {
        return false;
    }
    match data_type {
        DataType::Boolean => matches!(value, LiteralValue::Boolean(_)),
        DataType::Int8 => signed_literal_fits(value, i128::from(i8::MIN), i128::from(i8::MAX)),
        DataType::Int16 => signed_literal_fits(value, i128::from(i16::MIN), i128::from(i16::MAX)),
        DataType::Int32 => signed_literal_fits(value, i128::from(i32::MIN), i128::from(i32::MAX)),
        DataType::Int64 => signed_literal_fits(value, i128::from(i64::MIN), i128::from(i64::MAX)),
        DataType::UInt8 => unsigned_literal_fits(value, u128::from(u8::MAX)),
        DataType::UInt16 => unsigned_literal_fits(value, u128::from(u16::MAX)),
        DataType::UInt32 => unsigned_literal_fits(value, u128::from(u32::MAX)),
        DataType::UInt64 => unsigned_literal_fits(value, u128::from(u64::MAX)),
        DataType::Float32 => matches!(
            value,
            LiteralValue::Float32(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
        ),
        DataType::Float64 => matches!(
            value,
            LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
        ),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            matches!(value, LiteralValue::Utf8(_) | LiteralValue::LargeUtf8(_))
        },
        DataType::Decimal128(_precision, scale) => decimal_literal_can_target_scale(value, *scale),
        DataType::Date32 => matches!(value, LiteralValue::Date32(_)),
        DataType::Date64 => matches!(value, LiteralValue::Date64(_)),
        DataType::Time32(unit) => {
            matches!(value, LiteralValue::Time32 { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Time64(unit) => {
            matches!(value, LiteralValue::Time64 { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Timestamp(unit, None) => {
            matches!(value, LiteralValue::Timestamp { unit: literal_unit, .. } if literal_unit == unit)
        },
        DataType::Duration(unit) => {
            matches!(value, LiteralValue::Duration { unit: literal_unit, .. } if literal_unit == unit)
        },
        _ => false,
    }
}

fn signed_literal_fits(value: &LiteralValue, min: i128, max: i128) -> bool {
    reader_predicate_integer_literal_raw(value).is_some_and(|raw| raw >= min && raw <= max)
}

fn unsigned_literal_fits(value: &LiteralValue, max: u128) -> bool {
    reader_predicate_unsigned_literal_raw(value).is_some_and(|raw| raw <= max)
}

fn decimal_literal_can_target_scale(value: &LiteralValue, target_arrow_scale: i8) -> bool {
    let target_cudf_scale = arrow_decimal_scale_to_cudf_scale(target_arrow_scale);
    if let Some(raw) = reader_predicate_integer_literal_raw(value) {
        return reader_predicate_decimal_raw_for_integer_value(raw, target_cudf_scale).is_ok();
    }
    if let LiteralValue::Decimal128 { raw, scale, .. } = value {
        return reader_predicate_rescale_decimal_raw(
            *raw,
            arrow_decimal_scale_to_cudf_scale(*scale),
            target_cudf_scale,
        )
        .is_ok();
    }
    false
}

pub(in crate::source) fn reader_predicate_references_decimal_column(source: &LocalParquetSource, expr: &Expr) -> bool {
    reader_predicate_expr_references_decimal_column(expr, |column_name| {
        source
            .schema
            .field_with_name(column_name)
            .ok()
            .is_some_and(|field| matches!(field.data_type(), DataType::Decimal128(_, _)))
    })
}

pub(in crate::source) fn reader_predicate_source_contract_shape_is_supported(
    source: &LocalParquetSource,
    expr: &Expr,
) -> bool {
    match expr {
        Expr::Binary { left, op: BinaryOp::And | BinaryOp::Or, right, .. } => {
            reader_predicate_source_contract_shape_is_supported(source, left)
                && reader_predicate_source_contract_shape_is_supported(source, right)
        },
        Expr::Binary { left, op, right, .. } => {
            reader_predicate_source_contract_comparison_is_supported(source, left, *op, right)
                || reader_predicate_source_contract_comparison_is_supported(source, right, invert_reader_op(*op), left)
        },
        Expr::InList { expr, list, .. } => {
            !list.is_empty()
                && reader_predicate_source_contract_column_type(source, expr).is_some()
                && list.iter().all(reader_predicate_source_contract_literal_is_supported)
        },
        Expr::Not(inner) | Expr::Alias { expr: inner, .. } => {
            reader_predicate_source_contract_shape_is_supported(source, inner)
        },
        Expr::Column(_)
        | Expr::Literal(_)
        | Expr::IsNull(_)
        | Expr::IsNotNull(_)
        | Expr::Cast { .. }
        | Expr::TryCast { .. }
        | Expr::Like { .. }
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
        | Expr::UnaryMath { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => false,
    }
}

fn reader_predicate_source_contract_comparison_is_supported(
    source: &LocalParquetSource,
    column: &Expr,
    op: BinaryOp,
    literal: &Expr,
) -> bool {
    matches!(op, BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq)
        && reader_predicate_source_contract_column_type(source, column).is_some()
        && reader_predicate_source_contract_literal_is_supported(literal)
}

fn reader_predicate_source_contract_column_type(source: &LocalParquetSource, expr: &Expr) -> Option<DataType> {
    let column = match expr {
        Expr::Column(column) => column,
        Expr::Alias { expr, .. } => return reader_predicate_source_contract_column_type(source, expr),
        _ => return None,
    };
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column.name()) else {
        return None;
    };
    if !source.read_column_file_indices.contains(&file_index) {
        return None;
    }
    let data_type = source.schema.field_with_name(column.name()).ok()?.data_type().clone();
    match data_type {
        DataType::Boolean
        | DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Decimal128(_, _)
        | DataType::Date32 => Some(data_type),
        _ => None,
    }
}

fn reader_predicate_source_contract_literal_is_supported(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal(
            LiteralValue::Boolean(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::Int64(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
                | LiteralValue::UInt64(_)
                | LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Utf8(_)
                | LiteralValue::LargeUtf8(_)
                | LiteralValue::Decimal128 { .. }
                | LiteralValue::Date32(_)
        )
    )
}

fn invert_reader_op(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        _ => op,
    }
}
