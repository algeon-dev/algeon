use std::fmt::Write;

use super::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DataType, Date32Array, Date64Array, Decimal128Array,
    Decimal256Array, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, ParityCompareOptions, ParityFloatTolerance, RecordBatch, Schema, StringArray, StringViewArray,
    TimeUnit, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array, i256,
};

pub(super) fn collect_canonical_rows(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<CanonicalRow>, String> {
    let mut rows = Vec::new();
    for batch in batches {
        if batch.schema().fields().len() != schema.fields().len() {
            return Err(format!(
                "batch has {} columns but schema has {}",
                batch.schema().fields().len(),
                schema.fields().len()
            ));
        }
        for row_index in 0..batch.num_rows() {
            let mut fields = Vec::with_capacity(schema.fields().len());
            for column_index in 0..schema.fields().len() {
                let array = batch.column(column_index);
                let data_type = schema.field(column_index).data_type();
                let field = canonicalize_value(array.as_ref(), data_type, row_index)
                    .map_err(|err| format!("column {column_index} row {row_index}: {err}"))?;
                fields.push(field);
            }
            rows.push(CanonicalRow { fields });
        }
    }
    Ok(rows)
}

/// Sortable row representation used for order-insensitive parity comparison.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CanonicalRow {
    pub(super) fields: Vec<CanonicalField>,
}

/// Value representation that preserves Arrow type distinctions in fingerprints.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CanonicalField {
    Null,
    Bool(bool),
    Int { width: u8, value: i128 },
    UInt { width: u8, value: u128 },
    Float { width: u8, bits: u64 },
    Decimal128 { precision: u8, scale: i8, value: i128 },
    Decimal256 { precision: u8, scale: i8, value: [u8; 32] },
    Date32(i32),
    Date64(i64),
    Timestamp { unit: TimeUnitTag, tz: Option<String>, value: i64 },
    Utf8(String),
    Binary(Vec<u8>),
    Unsupported(String),
}

/// Comparable timestamp unit tag, kept independent of Arrow's enum ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum TimeUnitTag {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl From<TimeUnit> for TimeUnitTag {
    fn from(value: TimeUnit) -> Self {
        match value {
            TimeUnit::Second => Self::Second,
            TimeUnit::Millisecond => Self::Millisecond,
            TimeUnit::Microsecond => Self::Microsecond,
            TimeUnit::Nanosecond => Self::Nanosecond,
        }
    }
}

pub(super) fn canonicalize_value(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
) -> Result<CanonicalField, String> {
    if array.is_null(row) {
        return Ok(CanonicalField::Null);
    }
    match data_type {
        DataType::Boolean => {
            let array = downcast::<BooleanArray>(array, "BooleanArray")?;
            Ok(CanonicalField::Bool(array.value(row)))
        },
        DataType::Int8 => {
            let array = downcast::<Int8Array>(array, "Int8Array")?;
            Ok(CanonicalField::Int { width: 8, value: array.value(row).into() })
        },
        DataType::Int16 => {
            let array = downcast::<Int16Array>(array, "Int16Array")?;
            Ok(CanonicalField::Int { width: 16, value: array.value(row).into() })
        },
        DataType::Int32 => {
            let array = downcast::<Int32Array>(array, "Int32Array")?;
            Ok(CanonicalField::Int { width: 32, value: array.value(row).into() })
        },
        DataType::Int64 => {
            let array = downcast::<Int64Array>(array, "Int64Array")?;
            Ok(CanonicalField::Int { width: 64, value: array.value(row).into() })
        },
        DataType::UInt8 => {
            let array = downcast::<UInt8Array>(array, "UInt8Array")?;
            Ok(CanonicalField::UInt { width: 8, value: array.value(row).into() })
        },
        DataType::UInt16 => {
            let array = downcast::<UInt16Array>(array, "UInt16Array")?;
            Ok(CanonicalField::UInt { width: 16, value: array.value(row).into() })
        },
        DataType::UInt32 => {
            let array = downcast::<UInt32Array>(array, "UInt32Array")?;
            Ok(CanonicalField::UInt { width: 32, value: array.value(row).into() })
        },
        DataType::UInt64 => {
            let array = downcast::<UInt64Array>(array, "UInt64Array")?;
            Ok(CanonicalField::UInt { width: 64, value: array.value(row).into() })
        },
        DataType::Float32 => {
            let array = downcast::<Float32Array>(array, "Float32Array")?;
            Ok(CanonicalField::Float { width: 32, bits: u64::from(array.value(row).to_bits()) })
        },
        DataType::Float64 => {
            let array = downcast::<Float64Array>(array, "Float64Array")?;
            Ok(CanonicalField::Float { width: 64, bits: array.value(row).to_bits() })
        },
        DataType::Decimal128(precision, scale) => {
            let array = downcast::<Decimal128Array>(array, "Decimal128Array")?;
            Ok(CanonicalField::Decimal128 { precision: *precision, scale: *scale, value: array.value(row) })
        },
        DataType::Decimal256(precision, scale) => {
            let array = downcast::<Decimal256Array>(array, "Decimal256Array")?;
            let value: i256 = array.value(row);
            Ok(CanonicalField::Decimal256 { precision: *precision, scale: *scale, value: value.to_le_bytes() })
        },
        DataType::Date32 => {
            let array = downcast::<Date32Array>(array, "Date32Array")?;
            Ok(CanonicalField::Date32(array.value(row)))
        },
        DataType::Date64 => {
            let array = downcast::<Date64Array>(array, "Date64Array")?;
            Ok(CanonicalField::Date64(array.value(row)))
        },
        DataType::Timestamp(unit, tz) => {
            let value = match unit {
                TimeUnit::Second => downcast::<TimestampSecondArray>(array, "TimestampSecondArray")?.value(row),
                TimeUnit::Millisecond => {
                    downcast::<TimestampMillisecondArray>(array, "TimestampMillisecondArray")?.value(row)
                },
                TimeUnit::Microsecond => {
                    downcast::<TimestampMicrosecondArray>(array, "TimestampMicrosecondArray")?.value(row)
                },
                TimeUnit::Nanosecond => {
                    downcast::<TimestampNanosecondArray>(array, "TimestampNanosecondArray")?.value(row)
                },
            };
            Ok(CanonicalField::Timestamp {
                unit: TimeUnitTag::from(*unit),
                tz: tz.as_ref().map(|s| s.to_string()),
                value,
            })
        },
        DataType::Utf8 => {
            let array = downcast::<StringArray>(array, "StringArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::LargeUtf8 => {
            let array = downcast::<LargeStringArray>(array, "LargeStringArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::Utf8View => {
            let array = downcast::<StringViewArray>(array, "StringViewArray")?;
            Ok(CanonicalField::Utf8(array.value(row).to_string()))
        },
        DataType::Binary => {
            let array = downcast::<BinaryArray>(array, "BinaryArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        DataType::LargeBinary => {
            let array = downcast::<LargeBinaryArray>(array, "LargeBinaryArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        DataType::BinaryView => {
            let array = downcast::<BinaryViewArray>(array, "BinaryViewArray")?;
            Ok(CanonicalField::Binary(array.value(row).to_vec()))
        },
        other => Ok(CanonicalField::Unsupported(format!("{other:?}"))),
    }
}

pub(super) fn downcast<'a, T: Array + 'static>(array: &'a dyn Array, name: &str) -> Result<&'a T, String> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| format!("expected {name}"))
}

pub(super) fn compare_canonical_rows(
    schema: &Schema,
    cpu: &CanonicalRow,
    native: &CanonicalRow,
    tolerance: ParityFloatTolerance,
    float_tolerant: bool,
    options: ParityCompareOptions,
) -> Option<String> {
    if cpu.fields.len() != native.fields.len() {
        return Some(format!("column count differs: cpu={} native={}", cpu.fields.len(), native.fields.len()));
    }
    for (index, (cpu_field, native_field)) in cpu.fields.iter().zip(native.fields.iter()).enumerate() {
        let name = schema.field(index).name();
        if !canonical_field_equal(cpu_field, native_field, tolerance, float_tolerant, options) {
            return Some(format!(
                "column {index} ({name}): cpu={} native={}",
                format_field_debug(cpu_field),
                format_field_debug(native_field),
            ));
        }
    }
    None
}

pub(super) fn canonical_field_equal(
    cpu: &CanonicalField,
    native: &CanonicalField,
    tolerance: ParityFloatTolerance,
    float_tolerant: bool,
    options: ParityCompareOptions,
) -> bool {
    if !float_tolerant {
        return canonical_field_exact_or_allowed(cpu, native, options);
    }
    match (cpu, native) {
        (CanonicalField::Float { width: w1, bits: b1 }, CanonicalField::Float { width: w2, bits: b2 }) => {
            if w1 != w2 {
                return false;
            }
            let cpu_value = float_bits_to_f64(*w1, *b1);
            let native_value = float_bits_to_f64(*w2, *b2);
            floats_close_enough(cpu_value, native_value, tolerance)
        },
        _ => canonical_field_exact_or_allowed(cpu, native, options),
    }
}

fn float_bits_to_f64(width: u8, bits: u64) -> f64 {
    match width {
        32 => f32::from_bits(bits as u32) as f64,
        _ => f64::from_bits(bits),
    }
}

pub(super) fn canonical_field_exact_or_allowed(
    cpu: &CanonicalField,
    native: &CanonicalField,
    options: ParityCompareOptions,
) -> bool {
    if cpu == native {
        return true;
    }
    if options.allow_decimal_precision_mismatch {
        match (cpu, native) {
            (
                CanonicalField::Decimal128 { scale: cpu_scale, value: cpu_value, .. },
                CanonicalField::Decimal128 { scale: native_scale, value: native_value, .. },
            ) => return cpu_scale == native_scale && cpu_value == native_value,
            (
                CanonicalField::Decimal256 { scale: cpu_scale, value: cpu_value, .. },
                CanonicalField::Decimal256 { scale: native_scale, value: native_value, .. },
            ) => return cpu_scale == native_scale && cpu_value == native_value,
            _ => {},
        }
    }
    false
}

pub(super) fn floats_close_enough(cpu: f64, native: f64, tolerance: ParityFloatTolerance) -> bool {
    if cpu.is_nan() && native.is_nan() {
        return true;
    }
    if cpu == native {
        return true;
    }
    let diff = (cpu - native).abs();
    if diff <= tolerance.absolute {
        return true;
    }
    if tolerance.relative > 0.0 {
        let scale = cpu.abs().max(native.abs());
        return diff <= tolerance.relative * scale;
    }
    false
}

pub(super) fn format_field_debug(field: &CanonicalField) -> String {
    match field {
        CanonicalField::Null => "null".to_string(),
        CanonicalField::Bool(value) => value.to_string(),
        CanonicalField::Int { value, .. } => value.to_string(),
        CanonicalField::UInt { value, .. } => value.to_string(),
        CanonicalField::Float { width, bits } => match width {
            32 => format!("{}", f32::from_bits(*bits as u32)),
            _ => format!("{}", f64::from_bits(*bits)),
        },
        CanonicalField::Decimal128 { value, scale, precision } => {
            format!("decimal128({precision},{scale}):{value}")
        },
        CanonicalField::Decimal256 { value, scale, precision } => {
            let mut hex = String::with_capacity(64);
            for byte in value.iter().rev() {
                let _ = write!(hex, "{byte:02x}");
            }
            format!("decimal256({precision},{scale}):0x{hex}")
        },
        CanonicalField::Date32(value) => format!("date32:{value}"),
        CanonicalField::Date64(value) => format!("date64:{value}"),
        CanonicalField::Timestamp { unit, tz, value } => format!("timestamp({unit:?},{tz:?}):{value}"),
        CanonicalField::Utf8(value) => format!("\"{}\"", escape_string(value)),
        CanonicalField::Binary(bytes) => format!("bytes(len={})", bytes.len()),
        CanonicalField::Unsupported(text) => format!("unsupported({text})"),
    }
}

pub(super) fn escape_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t").replace('\r', "\\r")
}
