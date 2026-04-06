//! Dataset → DataFrame construction for the E2E conformance framework.
//!
//! Builds a `cudf::DataFrame` from a manifest [`Dataset`] definition,
//! translating JSON values and dtype strings into GPU-resident columns.

use std::{collections::HashSet, io::Write};

use cudf::{DataFrame, Series, TypeId, io::json::JsonReaderOptions, ops::dictionary};
use tempfile::NamedTempFile;

use super::manifest::Dataset;

/// Build a `cudf::DataFrame` from a manifest dataset definition.
///
/// Each column in the dataset is constructed as a nullable `Series`:
/// - Numeric types use `Series::new_nullable` with a validity mask derived
///   from JSON null positions.
/// - String types use `Series::from_strings_nullable`.
/// - Bool types convert JSON bools to Rust bools and use `Series::new_nullable`.
pub fn build_dataframe(dataset: &Dataset) -> cudf::Result<DataFrame> {
    let frame = if !dataset.columns.is_empty() {
        build_column_dataframe(dataset)?
    } else if !dataset.json_lines.is_empty() {
        build_json_lines_dataframe(dataset)?
    } else {
        return Err(cudf::Error::new(cudf::ErrorKind::InvalidOperation, "dataset must define columns or json_lines")
            .with_operation("e2e::build_dataframe")
            .with_context("dataset_id", dataset.id.clone()));
    };

    apply_dictionary_columns(frame, dataset)
}

fn build_column_dataframe(dataset: &Dataset) -> cudf::Result<DataFrame> {
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

fn build_json_lines_dataframe(dataset: &Dataset) -> cudf::Result<DataFrame> {
    let mut file = NamedTempFile::new().map_err(|error| {
        cudf::Error::new(cudf::ErrorKind::IoError, error.to_string())
            .with_operation("e2e::build_json_lines_dataframe")
            .with_context("dataset_id", dataset.id.clone())
    })?;

    for line in &dataset.json_lines {
        writeln!(file, "{line}").map_err(|error| {
            cudf::Error::new(cudf::ErrorKind::IoError, error.to_string())
                .with_operation("e2e::build_json_lines_dataframe")
                .with_context("dataset_id", dataset.id.clone())
        })?;
    }

    DataFrame::read_json_with_options(JsonReaderOptions::builder().filepath(file.path()).build())
}

fn apply_dictionary_columns(frame: DataFrame, dataset: &Dataset) -> cudf::Result<DataFrame> {
    if dataset.dictionary_columns.is_empty() {
        return Ok(frame);
    }

    let dictionary_columns: HashSet<&str> = dataset.dictionary_columns.iter().map(String::as_str).collect();
    let names = frame.column_names().to_vec();
    let pairs = names
        .iter()
        .map(|name| {
            let series = frame.series(name)?;
            if dictionary_columns.contains(*name) {
                let encoded = dictionary::encode(&series.as_column_view()?, TypeId::Int32)?;
                Ok(((*name).to_string(), Series::from_column(encoded)?.rename(*name)))
            } else {
                Ok(((*name).to_string(), series))
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn json_lines_dataset() -> Dataset {
        Dataset {
            id: "dictionary_nested".into(),
            columns: vec![],
            json_lines: vec![
                "{\"category\":\"beta\",\"nested\":[[1,2],[3]],\"point\":{\"coords\":[1,2],\"label\":\"one\"}}".into(),
                "{\"category\":\"alpha\",\"nested\":[[],[4,5]],\"point\":{\"coords\":[],\"label\":\"two\"}}".into(),
            ],
            dictionary_columns: vec!["category".into()],
        }
    }

    #[test]
    fn build_dataframe_supports_json_lines_with_dictionary_postprocessing() {
        let df = build_dataframe(&json_lines_dataset()).expect("build dataframe");

        assert_eq!(df.num_columns(), 3);
        assert_eq!(df.num_rows(), 2);
        assert_eq!(df.series("category").unwrap().dtype().unwrap().id(), TypeId::Dictionary32);
        assert_eq!(df.series("nested").unwrap().dtype().unwrap().id(), TypeId::List);
        assert_eq!(df.series("point").unwrap().dtype().unwrap().id(), TypeId::Struct);
    }

    #[test]
    fn build_dataframe_rejects_dataset_without_input_source() {
        let err = build_dataframe(&Dataset {
            id: "empty".into(),
            columns: vec![],
            json_lines: vec![],
            dictionary_columns: vec![],
        })
        .unwrap_err();

        assert_eq!(err.kind(), cudf::ErrorKind::InvalidOperation);
        assert_eq!(err.operation(), "e2e::build_dataframe");
    }
}
