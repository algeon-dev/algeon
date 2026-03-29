//! Dataset → DataFrame construction for the E2E conformance framework.
//!
//! Builds a `cudf::DataFrame` from a manifest [`Dataset`] definition,
//! translating JSON values and dtype strings into GPU-resident columns.

use cudf::{DataFrame, Series};

use super::manifest::Dataset;

/// Build a `cudf::DataFrame` from a manifest dataset definition.
///
/// Each column in the dataset is constructed as a nullable `Series`:
/// - Numeric types use `Series::new_nullable` with a validity mask derived
///   from JSON null positions.
/// - String types use `Series::from_strings_nullable`.
/// - Bool types convert JSON bools to Rust bools and use `Series::new_nullable`.
pub fn build_dataframe(dataset: &Dataset) -> cudf::Result<DataFrame> {
    let pairs: Vec<(String, Series)> = dataset
        .columns
        .iter()
        .map(|col| {
            let series = build_series(&col.name, &col.dtype, &col.values)?;
            Ok((col.name.clone(), series))
        })
        .collect::<cudf::Result<Vec<_>>>()?;

    DataFrame::from_series(pairs)
}

/// Build a single `Series` from a column name, dtype string, and JSON values.
fn build_series(name: &str, dtype: &str, values: &[serde_json::Value]) -> cudf::Result<Series> {
    match dtype {
        "int32" => build_numeric_series::<i32>(name, values, |v| v.as_i64().map(|n| n as i32)),
        "int64" => build_numeric_series::<i64>(name, values, |v| v.as_i64()),
        "uint32" => build_numeric_series::<u32>(name, values, |v| v.as_u64().map(|n| n as u32)),
        "uint64" => build_numeric_series::<u64>(name, values, |v| v.as_u64()),
        "float32" => build_numeric_series::<f32>(name, values, |v| v.as_f64().map(|n| n as f32)),
        "float64" => build_numeric_series::<f64>(name, values, |v| v.as_f64()),
        "bool8" => build_bool_series(name, values),
        "string" => build_string_series(name, values),
        other => {
            Err(cudf::Error::new(cudf::ErrorKind::InvalidOperation, format!("unsupported manifest dtype: {}", other))
                .with_operation("e2e::build_series")
                .with_context("dtype", other.to_string()))
        },
    }
}

/// Build a nullable numeric `Series` from JSON values.
///
/// `extract` maps a non-null `serde_json::Value` to the Rust type `T`.
/// JSON null values are marked invalid in the validity mask, and their
/// corresponding data slot is filled with `T`'s zero value.
fn build_numeric_series<T: cudf::CudfType + Default>(
    name: &str,
    values: &[serde_json::Value],
    extract: impl Fn(&serde_json::Value) -> Option<T>,
) -> cudf::Result<Series> {
    let mut data = Vec::with_capacity(values.len());
    let mut validity = Vec::with_capacity(values.len());

    for v in values {
        if v.is_null() {
            data.push(T::default());
            validity.push(false);
        } else {
            let val = extract(v).ok_or_else(|| {
                cudf::Error::new(cudf::ErrorKind::InvalidOperation, format!("cannot convert JSON value to {}", name))
                    .with_operation("e2e::build_numeric_series")
                    .with_context("value", v.to_string())
            })?;
            data.push(val);
            validity.push(true);
        }
    }

    let has_nulls = validity.iter().any(|&v| !v);
    if has_nulls { Series::new_nullable(name, &data, &validity) } else { Series::new(name, &data) }
}

/// Build a nullable bool `Series` from JSON boolean values.
fn build_bool_series(name: &str, values: &[serde_json::Value]) -> cudf::Result<Series> {
    build_numeric_series::<bool>(name, values, |v| v.as_bool())
}

/// Build a nullable string `Series` from JSON string values.
fn build_string_series(name: &str, values: &[serde_json::Value]) -> cudf::Result<Series> {
    let mut data = Vec::with_capacity(values.len());
    let mut validity = Vec::with_capacity(values.len());

    for v in values {
        if v.is_null() {
            data.push("");
            validity.push(false);
        } else {
            let s = v.as_str().ok_or_else(|| {
                cudf::Error::new(cudf::ErrorKind::InvalidOperation, "expected JSON string for string column")
                    .with_operation("e2e::build_string_series")
                    .with_context("value", v.to_string())
            })?;
            data.push(s);
            validity.push(true);
        }
    }

    let has_nulls = validity.iter().any(|&v| !v);
    if has_nulls { Series::from_strings_nullable(name, &data, &validity) } else { Series::from_strings(name, &data) }
}
