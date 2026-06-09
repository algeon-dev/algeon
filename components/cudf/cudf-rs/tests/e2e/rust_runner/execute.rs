//! Operation dispatch and normalization for the E2E conformance framework.
//!
//! Executes manifest operations through `cudf-rs` and normalizes results
//! into the shared result schema for comparison with Python cuDF output.

use cudf::{DataFrame, DropNaOptions, InfoDetail, SortOptions, TypeId};

use super::{
    manifest::{Operation, canonical_dtype},
    normalize::{
        ColumnListEntry, DtypeCount, NormalizedColumnInfo, NormalizedColumnList, NormalizedColumnMeta,
        NormalizedDataFrame, NormalizedError, NormalizedInfoSnapshot, NormalizedResult,
    },
};

/// Error from E2E test execution.
#[derive(Debug)]
pub enum ExecutionError {
    /// cudf-rs returned an error (expected for negative test cases).
    CudfError(cudf::Error),
    /// Framework-level error (test infrastructure failure, not a test case result).
    Framework(String),
}

impl From<cudf::Error> for ExecutionError {
    fn from(e: cudf::Error) -> Self {
        ExecutionError::CudfError(e)
    }
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionError::CudfError(e) => write!(f, "cudf error: {}", e),
            ExecutionError::Framework(msg) => write!(f, "framework error: {}", msg),
        }
    }
}

/// Execute a manifest operation on a DataFrame and return a normalized result.
///
/// For operations that are expected to produce errors (negative test cases),
/// the cudf error is caught and normalized into `NormalizedResult::Error`.
pub fn execute_operation(df: &DataFrame, op: &Operation) -> Result<NormalizedResult, ExecutionError> {
    match op {
        Operation::Info { detail } => execute_info(df, detail),
        Operation::Schema {} => execute_schema(df),
        Operation::Dtypes {} => execute_dtypes(df),
        Operation::Select { columns } => execute_transform("select", || {
            let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            df.select(col_refs)
        }),
        Operation::Rename { mapping } => execute_transform("rename", || {
            let pairs: Vec<(&str, &str)> = mapping.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            df.rename_columns(pairs)
        }),
        Operation::SortValues { by, ascending, null_position } => execute_transform("sort_values", || {
            let descending: Vec<bool> = ascending.iter().map(|&a| !a).collect();
            let nulls_first: Vec<bool> = vec![null_position == "first"; by.len()];
            let options = SortOptions::new().with_descending(descending).with_nulls_first(nulls_first);
            let by_refs: Vec<&str> = by.iter().map(|s| s.as_str()).collect();
            df.sort_values(by_refs, options)
        }),
        Operation::FilterEq { column, value, value_dtype } => execute_filter_eq(df, column, value, value_dtype),
        Operation::DropNulls { subset } => execute_transform("drop_nulls", || {
            let opts = match subset {
                Some(names) => {
                    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                    DropNaOptions::builder().subset(&refs).build()
                },
                None => DropNaOptions::default(),
            };
            df.dropna(opts)
        }),
        Operation::Head { n } => execute_transform("head", || df.head(*n)),
        Operation::Tail { n } => execute_transform("tail", || df.tail(*n)),
    }
}

/// Execute an info operation and normalize the result.
fn execute_info(df: &DataFrame, detail_flags: &[String]) -> Result<NormalizedResult, ExecutionError> {
    let mut detail = InfoDetail::empty();
    for flag in detail_flags {
        match flag.as_str() {
            "columns" => detail |= InfoDetail::COLUMNS,
            "null_counts" => detail |= InfoDetail::NULL_COUNTS,
            "payload_bytes" => detail |= InfoDetail::PAYLOAD_BYTES,
            "device_allocated" => detail |= InfoDetail::DEVICE_ALLOCATED,
            other => {
                return Err(ExecutionError::Framework(format!("unknown info detail flag: {}", other)));
            },
        }
    }

    let info = df.info_with(detail)?;

    let dtype_counts: Vec<DtypeCount> = info
        .dtypes
        .iter()
        .map(|dtc| DtypeCount { dtype: type_id_to_canonical(dtc.data_type.id()), count: dtc.column_count })
        .collect();

    let columns: Vec<NormalizedColumnInfo> = info
        .columns
        .as_ref()
        .map(|cols| {
            cols.iter()
                .map(|c| NormalizedColumnInfo {
                    name: c.name.clone(),
                    dtype: type_id_to_canonical(c.data_type.id()),
                    nullable: c.nullable,
                    null_count: c.null_count,
                    non_null_count: c.non_null_count,
                    payload_bytes: c.memory_accounting.as_ref().and_then(|m| m.payload_bytes.map(|b| b as u64)),
                    device_allocated_bytes: c
                        .memory_accounting
                        .as_ref()
                        .and_then(|m| m.device_allocated_bytes.map(|b| b as u64)),
                })
                .collect()
        })
        .unwrap_or_default();

    let payload_bytes = info.memory_accounting.as_ref().and_then(|m| m.payload_bytes.map(|b| b as u64));
    let device_allocated_bytes =
        info.memory_accounting.as_ref().and_then(|m| m.device_allocated_bytes.map(|b| b as u64));

    Ok(NormalizedResult::InfoSnapshot(NormalizedInfoSnapshot {
        row_count: info.row_count,
        column_count: info.column_count,
        dtype_counts,
        columns,
        payload_bytes,
        device_allocated_bytes,
    }))
}

/// Execute schema and return a column list.
fn execute_schema(df: &DataFrame) -> Result<NormalizedResult, ExecutionError> {
    let entries = build_column_list_entries(df)?;
    Ok(NormalizedResult::ColumnList(NormalizedColumnList { entries }))
}

/// Execute dtypes and return a column list (same shape as schema).
fn execute_dtypes(df: &DataFrame) -> Result<NormalizedResult, ExecutionError> {
    let entries = build_column_list_entries(df)?;
    Ok(NormalizedResult::ColumnList(NormalizedColumnList { entries }))
}

/// Build column list entries from a DataFrame's schema.
fn build_column_list_entries(df: &DataFrame) -> Result<Vec<ColumnListEntry>, ExecutionError> {
    let names = df.column_names();
    let mut entries = Vec::with_capacity(names.len());
    for name in &names {
        let series = df.series(name)?;
        let dtype = series.dtype()?;
        entries.push(ColumnListEntry { name: name.to_string(), dtype: type_id_to_canonical(dtype.id()) });
    }
    Ok(entries)
}

/// Execute a transform operation, catching cudf errors for negative test cases.
fn execute_transform<F>(operation_kind: &'static str, op: F) -> Result<NormalizedResult, ExecutionError>
where F: FnOnce() -> cudf::Result<DataFrame> {
    match op() {
        Ok(result_df) => {
            let normalized = normalize_dataframe(&result_df)?;
            Ok(NormalizedResult::Dataframe(normalized))
        },
        Err(e) => Ok(NormalizedResult::Error(normalize_manifest_error(&e, operation_kind))),
    }
}

/// Execute filter_eq: compare a column against a scalar value.
fn execute_filter_eq(
    df: &DataFrame,
    column: &str,
    value: &serde_json::Value,
    value_dtype: &str,
) -> Result<NormalizedResult, ExecutionError> {
    let mask_result = build_filter_eq_mask(df, column, value, value_dtype);

    match mask_result {
        Ok(mask) => match df.filter(&mask) {
            Ok(result_df) => {
                let normalized = normalize_dataframe(&result_df)?;
                Ok(NormalizedResult::Dataframe(normalized))
            },
            Err(e) => Ok(NormalizedResult::Error(normalize_manifest_error(&e, "filter_eq"))),
        },
        Err(ExecutionError::CudfError(e)) => Ok(NormalizedResult::Error(normalize_manifest_error(&e, "filter_eq"))),
        Err(e) => Err(e),
    }
}

/// Build a boolean mask for filter_eq by dispatching on value_dtype.
fn build_filter_eq_mask(
    df: &DataFrame,
    column: &str,
    value: &serde_json::Value,
    value_dtype: &str,
) -> Result<cudf::Series, ExecutionError> {
    let series = df.series(column)?;
    let mask = match value_dtype {
        "int32" => {
            let v =
                value.as_i64().ok_or_else(|| ExecutionError::Framework("filter_eq: expected integer value".into()))?
                    as i32;
            series.eq_scalar(v)?
        },
        "int64" => {
            let v =
                value.as_i64().ok_or_else(|| ExecutionError::Framework("filter_eq: expected integer value".into()))?;
            series.eq_scalar(v)?
        },
        "uint32" => {
            let v = value
                .as_u64()
                .ok_or_else(|| ExecutionError::Framework("filter_eq: expected unsigned integer value".into()))?
                as u32;
            series.eq_scalar(v)?
        },
        "uint64" => {
            let v = value
                .as_u64()
                .ok_or_else(|| ExecutionError::Framework("filter_eq: expected unsigned integer value".into()))?;
            series.eq_scalar(v)?
        },
        "float32" => {
            let v = value.as_f64().ok_or_else(|| ExecutionError::Framework("filter_eq: expected float value".into()))?
                as f32;
            series.eq_scalar(v)?
        },
        "float64" => {
            let v =
                value.as_f64().ok_or_else(|| ExecutionError::Framework("filter_eq: expected float value".into()))?;
            series.eq_scalar(v)?
        },
        "bool8" => {
            let v =
                value.as_bool().ok_or_else(|| ExecutionError::Framework("filter_eq: expected bool value".into()))?;
            series.eq_scalar(v)?
        },
        "string" => {
            let v =
                value.as_str().ok_or_else(|| ExecutionError::Framework("filter_eq: expected string value".into()))?;
            series.eq_scalar_str(v)?
        },
        other => {
            return Err(ExecutionError::Framework(format!("filter_eq: unsupported value_dtype: {}", other)));
        },
    };
    Ok(mask)
}

/// Normalize a DataFrame into the shared result schema.
///
/// Materializes all column data to host memory as JSON-compatible values.
pub fn normalize_dataframe(df: &DataFrame) -> Result<NormalizedDataFrame, ExecutionError> {
    let names = df.column_names();
    let num_rows = df.num_rows();

    let mut columns_meta = Vec::with_capacity(names.len());
    let mut column_values: Vec<Vec<serde_json::Value>> = Vec::with_capacity(names.len());

    for name in &names {
        let series = df.series(name)?;
        let dtype = series.dtype()?;
        let canonical = type_id_to_canonical(dtype.id());
        columns_meta.push(NormalizedColumnMeta { name: name.to_string(), dtype: canonical.clone() });

        let values = materialize_column_values(&series, dtype.id())?;
        column_values.push(values);
    }

    // Transpose column-major to row-major
    let mut rows = Vec::with_capacity(num_rows);
    for row_idx in 0..num_rows {
        let row: Vec<serde_json::Value> = column_values.iter().map(|col| col[row_idx].clone()).collect();
        rows.push(row);
    }

    Ok(NormalizedDataFrame { columns: columns_meta, row_count: num_rows, rows })
}

/// Materialize a column's values to a Vec of JSON values, respecting nulls.
fn materialize_column_values(series: &cudf::Series, type_id: TypeId) -> Result<Vec<serde_json::Value>, ExecutionError> {
    let cv = series.as_column_view()?;
    let validity = cv.validity()?;

    match type_id {
        TypeId::Int32 => materialize_numeric::<i32>(&cv, &validity, serde_json::Value::from),
        TypeId::Int64 => materialize_numeric::<i64>(&cv, &validity, serde_json::Value::from),
        TypeId::UInt32 => materialize_numeric::<u32>(&cv, &validity, serde_json::Value::from),
        TypeId::UInt64 => materialize_numeric::<u64>(&cv, &validity, serde_json::Value::from),
        TypeId::Float32 => materialize_numeric::<f32>(&cv, &validity, |v| serde_json::Value::from(v as f64)),
        TypeId::Float64 => materialize_numeric::<f64>(&cv, &validity, serde_json::Value::from),
        TypeId::Bool8 => materialize_numeric::<bool>(&cv, &validity, serde_json::Value::from),
        TypeId::String => {
            let strings = cv.to_optional_strings()?;
            Ok(strings
                .into_iter()
                .map(|opt| match opt {
                    Some(s) => serde_json::Value::from(s),
                    None => serde_json::Value::Null,
                })
                .collect())
        },
        other => Err(ExecutionError::Framework(format!("unsupported TypeId for materialization: {:?}", other))),
    }
}

/// Materialize a fixed-width numeric column with null support.
fn materialize_numeric<T: cudf::CudfType>(
    cv: &cudf::core::ColumnView<'_>,
    validity: &Option<Vec<bool>>,
    to_json: impl Fn(T) -> serde_json::Value,
) -> Result<Vec<serde_json::Value>, ExecutionError> {
    let data = cv.to_vec::<T>()?;
    Ok(match validity {
        Some(valid) => data
            .into_iter()
            .zip(valid.iter())
            .map(|(val, &is_valid)| if is_valid { to_json(val) } else { serde_json::Value::Null })
            .collect(),
        None => data.into_iter().map(to_json).collect(),
    })
}

/// Convert a cudf `TypeId` to its manifest-canonical dtype string.
fn type_id_to_canonical(type_id: TypeId) -> String {
    let native = match type_id {
        TypeId::Int8 => "int8",
        TypeId::Int16 => "int16",
        TypeId::Int32 => "int32",
        TypeId::Int64 => "int64",
        TypeId::UInt8 => "uint8",
        TypeId::UInt16 => "uint16",
        TypeId::UInt32 => "uint32",
        TypeId::UInt64 => "uint64",
        TypeId::Float32 => "float32",
        TypeId::Float64 => "float64",
        TypeId::Bool8 => "bool8",
        TypeId::String => "string",
        TypeId::Dictionary32 => "dictionary32",
        TypeId::List => "list",
        TypeId::Struct => "struct",
        other => return format!("{:?}", other).to_lowercase(),
    };
    canonical_dtype(native).to_string()
}

/// Normalize a cudf::Error into a manifest-facing error taxonomy for comparison.
pub fn normalize_manifest_error(err: &cudf::Error, operation_kind: &'static str) -> NormalizedError {
    let kind = normalize_manifest_error_kind(err, operation_kind);
    let operation = operation_kind.to_string();
    let context: Vec<(String, String)> = err.context().iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    NormalizedError { kind, operation, context }
}

fn normalize_manifest_error_kind(err: &cudf::Error, _operation_kind: &'static str) -> String {
    let has_column_name = err.context().iter().any(|(k, _)| *k == "column_name");

    match err.kind() {
        cudf::ErrorKind::NotFound if has_column_name => "column_not_found".into(),
        cudf::ErrorKind::InvalidArgument
        | cudf::ErrorKind::NotSupported
        | cudf::ErrorKind::InvalidOperation
        | cudf::ErrorKind::TypeError
        | cudf::ErrorKind::NullError => "invalid_argument".into(),
        cudf::ErrorKind::IndexError => "index_out_of_bounds".into(),
        _ => "unknown".into(),
    }
}
