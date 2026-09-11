use std::sync::Arc;

use arrow::datatypes::{DataType, TimeUnit};
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CastExpr, Literal},
    },
};
use query_engine::datatypes::{
    is_narrow_decimal_carry_type, is_supported_arrow_cast, is_supported_arrow_type, is_supported_list_carry_type,
    is_timestamp_with_timezone,
};

use super::{ExpressionCapability, ExpressionRejection};
use crate::native::literal::parse_yyyy_mm_dd_date32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum SupportedBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    StringConcat,
    IsDistinctFrom,
    IsNotDistinctFrom,
}

impl SupportedBinaryOperator {
    pub(in crate::native) fn from_datafusion(op: &Operator) -> Result<Self, ExpressionRejection> {
        match op {
            Operator::Plus => Ok(Self::Add),
            Operator::Minus => Ok(Self::Subtract),
            Operator::Multiply => Ok(Self::Multiply),
            Operator::Divide => Ok(Self::Divide),
            Operator::Modulo => Ok(Self::Modulo),
            Operator::BitwiseAnd => Ok(Self::BitwiseAnd),
            Operator::BitwiseOr => Ok(Self::BitwiseOr),
            Operator::BitwiseXor => Ok(Self::BitwiseXor),
            Operator::BitwiseShiftLeft => Ok(Self::ShiftLeft),
            Operator::BitwiseShiftRight => Ok(Self::ShiftRight),
            Operator::Eq => Ok(Self::Eq),
            Operator::NotEq => Ok(Self::NotEq),
            Operator::Lt => Ok(Self::Lt),
            Operator::LtEq => Ok(Self::LtEq),
            Operator::Gt => Ok(Self::Gt),
            Operator::GtEq => Ok(Self::GtEq),
            Operator::And => Ok(Self::And),
            Operator::Or => Ok(Self::Or),
            Operator::StringConcat => Ok(Self::StringConcat),
            Operator::IsDistinctFrom => Ok(Self::IsDistinctFrom),
            Operator::IsNotDistinctFrom => Ok(Self::IsNotDistinctFrom),
            _ => Err(ExpressionRejection::unsupported_expression(ExpressionCapability::BinaryOperator)),
        }
    }

    pub(super) const fn rejects_timestamp_timezone(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Subtract
                | Self::Multiply
                | Self::Divide
                | Self::Modulo
                | Self::Eq
                | Self::NotEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::IsDistinctFrom
                | Self::IsNotDistinctFrom
        )
    }

    pub(super) const fn rejects_list_carry(self) -> bool {
        self.rejects_timestamp_timezone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum ScalarFunctionKind {
    Substring,
    Extract,
    DateTrunc,
    ToTimestampSeconds,
    Lower,
    Upper,
    Trim,
    LeftTrim,
    RightTrim,
    Contains,
    StartsWith,
    EndsWith,
    StringLength,
    RegexpReplace,
    Concat,
    Abs,
    Sqrt,
    Round,
    NullIf,
}

impl ScalarFunctionKind {
    pub(in crate::native) fn from_name(name: &str) -> Result<Self, ExpressionRejection> {
        match name.to_ascii_lowercase().as_str() {
            "substring" | "substr" => Ok(Self::Substring),
            "date_part" | "extract" => Ok(Self::Extract),
            "date_trunc" => Ok(Self::DateTrunc),
            "to_timestamp_seconds" => Ok(Self::ToTimestampSeconds),
            "lower" => Ok(Self::Lower),
            "upper" => Ok(Self::Upper),
            "trim" | "btrim" => Ok(Self::Trim),
            "ltrim" => Ok(Self::LeftTrim),
            "rtrim" => Ok(Self::RightTrim),
            "contains" => Ok(Self::Contains),
            "starts_with" => Ok(Self::StartsWith),
            "ends_with" => Ok(Self::EndsWith),
            "character_length" | "length" => Ok(Self::StringLength),
            "regexp_replace" => Ok(Self::RegexpReplace),
            "concat" => Ok(Self::Concat),
            "abs" => Ok(Self::Abs),
            "sqrt" => Ok(Self::Sqrt),
            "round" => Ok(Self::Round),
            "nullif" => Ok(Self::NullIf),
            _ => Err(ExpressionRejection::unsupported_expression(ExpressionCapability::ScalarFunction)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native) enum CastCapability {
    Native,
    BoundedNonnegativeIntegral,
    LiteralStringDate32,
}

pub(in crate::native) fn is_supported_native_type(data_type: &DataType) -> bool {
    is_supported_arrow_type(data_type)
}

pub(in crate::native) fn is_timestamp_with_timezone_type(data_type: &DataType) -> bool {
    is_timestamp_with_timezone(data_type)
}

pub(in crate::native) fn is_list_carry_type(data_type: &DataType) -> bool {
    is_supported_list_carry_type(data_type)
}

pub(in crate::native) fn is_narrow_decimal_carry(data_type: &DataType) -> bool {
    is_narrow_decimal_carry_type(data_type)
}

pub(in crate::native) fn is_supported_native_cast(source_type: &DataType, target_type: &DataType) -> bool {
    is_supported_arrow_cast(source_type, target_type, query_engine::rapids_backend::rapids_analysis_context())
}

pub(in crate::native) fn cast_capability(
    expr: &Arc<dyn PhysicalExpr>,
    source_type: &DataType,
    target_type: &DataType,
) -> Result<CastCapability, ExpressionRejection> {
    if is_supported_native_cast(source_type, target_type) {
        return Ok(CastCapability::Native);
    }
    if bounded_nonnegative_integral_cast_is_supported(expr, source_type, target_type) {
        return Ok(CastCapability::BoundedNonnegativeIntegral);
    }
    if literal_string_date32_cast_is_supported(expr, target_type) {
        return Ok(CastCapability::LiteralStringDate32);
    }
    Err(ExpressionRejection::unsupported_type(ExpressionCapability::CastPair))
}

pub(in crate::native) fn is_supported_try_cast_pair(source_type: &DataType, target_type: &DataType) -> bool {
    is_string_type(source_type) && is_integral_type(target_type)
}

pub(in crate::native) fn is_supported_to_timestamp_seconds_cast(
    source_type: &DataType,
    target_type: &DataType,
) -> bool {
    matches!(target_type, DataType::Timestamp(TimeUnit::Second, None))
        && matches!(
            source_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
        )
        && is_supported_arrow_cast(source_type, target_type, query_engine::rapids_backend::rapids_analysis_context())
}

pub(in crate::native) fn is_string_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

pub(in crate::native) fn is_integral_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

pub(in crate::native) fn is_numeric_or_decimal_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
    )
}

pub(in crate::native) fn compatible_types(left: &DataType, right: &DataType) -> bool {
    !is_timestamp_with_timezone_type(left)
        && !is_timestamp_with_timezone_type(right)
        && !is_list_carry_type(left)
        && !is_list_carry_type(right)
        && !is_narrow_decimal_carry(left)
        && !is_narrow_decimal_carry(right)
        && (left == right || is_supported_native_cast(left, right) || is_supported_native_cast(right, left))
}

fn literal_string_date32_cast_is_supported(expr: &Arc<dyn PhysicalExpr>, target_type: &DataType) -> bool {
    matches!(target_type, DataType::Date32)
        && expr
            .downcast_ref::<Literal>()
            .and_then(|literal| match literal.value() {
                ScalarValue::Utf8(Some(value))
                | ScalarValue::LargeUtf8(Some(value))
                | ScalarValue::Utf8View(Some(value)) => parse_yyyy_mm_dd_date32(value),
                _ => None,
            })
            .is_some()
}

pub(in crate::native) fn bounded_nonnegative_integral_cast_is_supported(
    expr: &Arc<dyn PhysicalExpr>,
    source_type: &DataType,
    target_type: &DataType,
) -> bool {
    if !is_integral_type(source_type) || !is_integral_type(target_type) {
        return false;
    }
    let Some(max_value) = nonnegative_integral_expr_max(expr) else {
        return false;
    };
    target_integral_max(target_type).is_some_and(|target_max| max_value <= target_max)
}

fn nonnegative_integral_expr_max(expr: &Arc<dyn PhysicalExpr>) -> Option<u128> {
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return nonnegative_integral_scalar_value(literal.value());
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return nonnegative_integral_expr_max(cast.expr());
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        return match binary.op() {
            Operator::BitwiseAnd => {
                nonnegative_integral_expr_max(binary.left()).or_else(|| nonnegative_integral_expr_max(binary.right()))
            },
            Operator::BitwiseOr | Operator::BitwiseXor => {
                Some(nonnegative_integral_expr_max(binary.left())? | nonnegative_integral_expr_max(binary.right())?)
            },
            Operator::BitwiseShiftRight => {
                let left_max = nonnegative_integral_expr_max(binary.left())?;
                let shift = nonnegative_integral_scalar_expr_value(binary.right())?;
                if shift >= 128 { Some(0) } else { Some(left_max >> shift) }
            },
            Operator::BitwiseShiftLeft => {
                let left_max = nonnegative_integral_expr_max(binary.left())?;
                let shift = u32::try_from(nonnegative_integral_scalar_expr_value(binary.right())?).ok()?;
                left_max.checked_shl(shift)
            },
            _ => None,
        };
    }
    None
}

fn nonnegative_integral_scalar_expr_value(expr: &Arc<dyn PhysicalExpr>) -> Option<u128> {
    expr.downcast_ref::<Literal>().and_then(|literal| nonnegative_integral_scalar_value(literal.value()))
}

fn nonnegative_integral_scalar_value(value: &ScalarValue) -> Option<u128> {
    match value {
        ScalarValue::Int8(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int16(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int32(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::Int64(Some(value)) => u128::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt32(Some(value)) => Some(u128::from(*value)),
        ScalarValue::UInt64(Some(value)) => Some(u128::from(*value)),
        _ => None,
    }
}

fn target_integral_max(data_type: &DataType) -> Option<u128> {
    match data_type {
        DataType::Int8 => Some(i8::MAX as u128),
        DataType::Int16 => Some(i16::MAX as u128),
        DataType::Int32 => Some(i32::MAX as u128),
        DataType::Int64 => Some(i64::MAX as u128),
        DataType::UInt8 => Some(u128::from(u8::MAX)),
        DataType::UInt16 => Some(u128::from(u16::MAX)),
        DataType::UInt32 => Some(u128::from(u32::MAX)),
        DataType::UInt64 => Some(u128::from(u64::MAX)),
        _ => None,
    }
}
