//! SQL scalar argument conversion shared by dense-relation function families.

use datafusion::{
    common::{DataFusionError, Result, ScalarValue},
    logical_expr::Expr,
};
use serde_json::{Number, Value};

pub(crate) fn scalar_literal(invalid: &impl Fn() -> DataFusionError, expression: Expr) -> Result<ScalarValue> {
    match expression {
        Expr::Literal(value, _) => Ok(value),
        Expr::Negative(inner) => negate_scalar(invalid, scalar_literal(invalid, *inner)?),
        Expr::Cast(cast) => scalar_literal(invalid, *cast.expr)?.cast_to(cast.field.data_type()).map_err(|_| invalid()),
        _ => Err(invalid()),
    }
}

fn negate_scalar(invalid: &impl Fn() -> DataFusionError, value: ScalarValue) -> Result<ScalarValue> {
    match value {
        ScalarValue::Int8(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int8(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int16(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int16(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int32(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int32(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int64(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int64(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Float32(Some(value)) => Ok(ScalarValue::Float32(Some(-value))),
        ScalarValue::Float64(Some(value)) => Ok(ScalarValue::Float64(Some(-value))),
        _ => Err(invalid()),
    }
}

pub(crate) fn scalar_json(invalid: &impl Fn() -> DataFusionError, value: ScalarValue) -> Result<Value> {
    let value = match value {
        ScalarValue::Null
        | ScalarValue::Boolean(None)
        | ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Float32(None)
        | ScalarValue::Float64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Utf8View(None) => Value::Null,
        ScalarValue::Boolean(Some(value)) => Value::Bool(value),
        ScalarValue::Int8(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int16(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int32(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int64(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt8(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt16(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt32(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt64(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Float32(Some(value)) => {
            Number::from_f64(f64::from(value)).map(Value::Number).ok_or_else(invalid)?
        },
        ScalarValue::Float64(Some(value)) => Number::from_f64(value).map(Value::Number).ok_or_else(invalid)?,
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Value::String(value)
        },
        _ => return Err(invalid()),
    };
    Ok(value)
}
