//! Arrow `RecordBatch` interop: GPU `DataFrame` ↔ Arrow `RecordBatch`
//! conversion.
//!
//! This module is specifically the Rust `arrow` crate / `RecordBatch` path.
//! It provides a **host-copy** conversion: data is transferred GPU → CPU →
//! Arrow (or Arrow → CPU → GPU).
//!
//! This is deliberately distinct from [`crate::interop::arrow_device`], which
//! implements Arrow C Device ABI handoff for existing device-backed buffers.
//! That device-interop path can preserve device pointers for many supported
//! layouts, but it is a different contract surface from `RecordBatch` and is
//! not the API implemented here.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DataFrame, Runtime};
//!
//! let rt = Runtime::new().unwrap();
//! let ids = rt.series(&[1i32, 2, 3]).unwrap();
//! let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
//! let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();
//!
//! // GPU → Arrow RecordBatch (host-copy)
//! let batch = df.to_record_batch().unwrap();
//! assert_eq!(batch.num_rows(), 3);
//!
//! // Arrow RecordBatch → GPU DataFrame (host-copy) inside the explicit runtime
//! let df2 = DataFrame::from_record_batch(&rt, &batch).unwrap();
//! assert_eq!(df2.num_columns(), 2);
//! ```

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use arrow::{
    array::*,
    datatypes::{DataType as ArrowDataType, Schema, TimeUnit},
    record_batch::RecordBatch,
};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_TYPE_ID, Result},
        types::{DataType, TypeId},
    },
    frame::dataframe::DataFrame,
    internal::{
        hash::fast_index_map,
        resolve::{missing_column_error, validate_no_duplicate_names},
    },
    runtime::{
        BoundaryKind, BoundaryObservation, QueryStageKind, Runtime, record_boundary_observation,
        record_boundary_observation_in_domain,
    },
};

pub(crate) mod testing_hooks;

static HOST_ARROW_EXPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_ARROW_IMPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_ARROW_LAST_BOUNDARY_OBSERVATION: Mutex<Option<BoundaryObservation>> = Mutex::new(None);
const DEFAULT_DECIMAL128_ARROW_PRECISION: u8 = 38;

/// Copy-free planner view of one Arrow host field as cudf-rs would import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowHostField {
    name: String,
    arrow_dtype: ArrowDataType,
    cudf_dtype: DataType,
    nullable: bool,
}

impl ArrowHostField {
    fn new(name: String, arrow_dtype: ArrowDataType, cudf_dtype: DataType, nullable: bool) -> Self {
        Self { name, arrow_dtype, cudf_dtype, nullable }
    }

    /// Field name in Arrow schema order.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Original Arrow logical type.
    pub fn arrow_data_type(&self) -> &ArrowDataType {
        &self.arrow_dtype
    }

    /// cudf-rs logical type that the host Arrow import path would create.
    pub fn cudf_data_type(&self) -> DataType {
        self.cudf_dtype
    }

    /// Arrow field nullability. cudf physical dtypes do not encode this bit.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// Copy-free planner view of an Arrow host schema as cudf-rs would import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowHostSchema {
    fields: Vec<ArrowHostField>,
    row_count: usize,
}

impl ArrowHostSchema {
    fn new(fields: Vec<ArrowHostField>, row_count: usize) -> Self {
        Self { fields, row_count }
    }

    /// Validated fields in schema order.
    pub fn fields(&self) -> &[ArrowHostField] {
        &self.fields
    }

    /// Return one validated field by ordinal.
    pub fn field(&self, index: usize) -> Option<&ArrowHostField> {
        self.fields.get(index)
    }

    /// Return one validated field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&ArrowHostField> {
        self.fields.iter().find(|field| field.name() == name)
    }

    /// Number of schema fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Logical row count supplied by the caller or observed from a RecordBatch.
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

/// Convert a supported Arrow host logical type into a cudf-rs logical type.
///
/// This is a copy-free planner/schema helper. It follows the same support
/// envelope as [`DataFrame::from_record_batch`]: numeric types, Boolean,
/// Decimal128, Date32, timezone-free timestamps, Utf8, and LargeUtf8.
pub fn cudf_dtype_from_arrow_type(dt: &ArrowDataType) -> Result<DataType> {
    cudf_dtype_from_arrow_type_in(dt, "arrow_host::cudf_dtype_from_arrow_type")
}

/// Convert a supported cudf-rs logical type into the Arrow host type used by
/// [`DataFrame::to_record_batch`].
///
/// Decimal128 export uses Arrow precision 38 because libcudf retains decimal
/// scale but not Arrow precision metadata.
pub fn arrow_type_from_cudf_dtype(dtype: DataType) -> Result<ArrowDataType> {
    arrow_type_from_cudf_dtype_in(dtype, "arrow_host::arrow_type_from_cudf_dtype")
}

/// Return whether this Arrow host type is supported by cudf-rs host import.
pub fn is_supported_arrow_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).is_ok()
}

/// Return whether this Arrow host type is a supported numeric cudf-rs type.
pub fn is_supported_arrow_numeric_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).map(|dtype| dtype.id().is_numeric()).unwrap_or(false)
}

/// Validate an Arrow schema without copying data and return cudf-rs planner metadata.
///
/// Nullability is retained as schema metadata rather than folded into the cudf
/// physical dtype.
pub fn validate_arrow_schema(schema: &Schema, row_count: usize) -> Result<ArrowHostSchema> {
    let operation = "arrow_host::validate_arrow_schema";
    let names = schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    validate_no_duplicate_names(&names, operation)?;

    let mut fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let cudf_dtype = cudf_dtype_from_arrow_type_in(field.data_type(), operation)
            .map_err(|error| error.with_detail("field_name", field.name().clone()))?;
        fields.push(ArrowHostField::new(
            field.name().clone(),
            field.data_type().clone(),
            cudf_dtype,
            field.is_nullable(),
        ));
    }

    Ok(ArrowHostSchema::new(fields, row_count))
}

/// Validate a RecordBatch schema without copying array data and return cudf-rs planner metadata.
pub fn validate_record_batch_schema(batch: &RecordBatch) -> Result<ArrowHostSchema> {
    validate_arrow_schema(batch.schema().as_ref(), batch.num_rows())
        .map_err(|error| error.reanchor("arrow_host::validate_record_batch_schema"))
}

fn cudf_dtype_from_arrow_type_in(dt: &ArrowDataType, operation: &'static str) -> Result<DataType> {
    let dtype = match dt {
        ArrowDataType::Int8 => DataType::new(TypeId::Int8),
        ArrowDataType::Int16 => DataType::new(TypeId::Int16),
        ArrowDataType::Int32 => DataType::new(TypeId::Int32),
        ArrowDataType::Int64 => DataType::new(TypeId::Int64),
        ArrowDataType::UInt8 => DataType::new(TypeId::UInt8),
        ArrowDataType::UInt16 => DataType::new(TypeId::UInt16),
        ArrowDataType::UInt32 => DataType::new(TypeId::UInt32),
        ArrowDataType::UInt64 => DataType::new(TypeId::UInt64),
        ArrowDataType::Float32 => DataType::new(TypeId::Float32),
        ArrowDataType::Float64 => DataType::new(TypeId::Float64),
        ArrowDataType::Boolean => DataType::new(TypeId::Bool8),
        ArrowDataType::Date32 => DataType::new(TypeId::TimestampDays),
        ArrowDataType::Timestamp(TimeUnit::Second, None) => DataType::new(TypeId::TimestampSeconds),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None) => DataType::new(TypeId::TimestampMilliseconds),
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None) => DataType::new(TypeId::TimestampMicroseconds),
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None) => DataType::new(TypeId::TimestampNanoseconds),
        ArrowDataType::Timestamp(_, Some(_)) => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                format!("unsupported Arrow timestamp with timezone for cudf conversion: {:?}", dt),
            )
            .with_detail("arrow_type", format!("{:?}", dt))
            .with_detail("reason", "timezone_not_supported"));
        },
        ArrowDataType::Decimal128(_, scale) => {
            DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale))
        },
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => DataType::new(TypeId::String),
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                format!("unsupported Arrow type for cudf conversion: {:?}", dt),
            )
            .with_detail("arrow_type", format!("{:?}", dt))
            .with_detail("reason", "unsupported_arrow_type"));
        },
    };

    Ok(dtype)
}

fn arrow_type_from_cudf_dtype_in(dtype: DataType, operation: &'static str) -> Result<ArrowDataType> {
    match dtype.id() {
        TypeId::Int8 => Ok(ArrowDataType::Int8),
        TypeId::Int16 => Ok(ArrowDataType::Int16),
        TypeId::Int32 => Ok(ArrowDataType::Int32),
        TypeId::Int64 => Ok(ArrowDataType::Int64),
        TypeId::UInt8 => Ok(ArrowDataType::UInt8),
        TypeId::UInt16 => Ok(ArrowDataType::UInt16),
        TypeId::UInt32 => Ok(ArrowDataType::UInt32),
        TypeId::UInt64 => Ok(ArrowDataType::UInt64),
        TypeId::Float32 => Ok(ArrowDataType::Float32),
        TypeId::Float64 => Ok(ArrowDataType::Float64),
        TypeId::Bool8 => Ok(ArrowDataType::Boolean),
        TypeId::TimestampDays => Ok(ArrowDataType::Date32),
        TypeId::TimestampSeconds => Ok(ArrowDataType::Timestamp(TimeUnit::Second, None)),
        TypeId::TimestampMilliseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Millisecond, None)),
        TypeId::TimestampMicroseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Microsecond, None)),
        TypeId::TimestampNanoseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)),
        TypeId::String => Ok(ArrowDataType::Utf8),
        TypeId::Decimal128 => {
            let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), operation)
                .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
            Ok(ArrowDataType::Decimal128(DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale))
        },
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            format!("unsupported cudf type for Arrow host conversion: {:?}", dtype.id()),
        )
        .with_fact(FACT_TYPE_ID, dtype.id())),
    }
}

impl DataFrame {
    /// Convert this GPU DataFrame to an Arrow [`RecordBatch`] via host copy.
    ///
    /// Each column is copied from GPU to host memory, then wrapped in the
    /// corresponding Arrow array type. The schema (column names and types)
    /// is preserved.
    ///
    /// If you need device-pointer handoff instead of host materialization, use
    /// [`crate::DataFrame::try_prepare_device_arrow_export`] from
    /// [`crate::interop::arrow_device`].
    ///
    /// # Supported types
    ///
    /// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
    ///   `UInt32`, `UInt64`, `Float32`, `Float64`
    /// - `Bool8`
    /// - `Decimal128` as Arrow `Decimal128(38, scale)`. cuDF stores decimal
    ///   scale but not precision, so this host path uses precision 38.
    /// - `TimestampDays` as Arrow `Date32`
    /// - `TimestampSeconds`, `TimestampMilliseconds`, `TimestampMicroseconds`,
    ///   and `TimestampNanoseconds` as Arrow timestamps without time zone.
    /// - `String` (converted to Arrow `Utf8`)
    ///
    /// # Errors
    ///
    /// Returns an error if any column has an unsupported type, or if the
    /// underlying GPU-to-host copy fails.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        HOST_ARROW_EXPORT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let names = self.column_names();
        let rows = self.num_rows();
        let column_count = self.num_columns();
        let placement = self.placement();

        let col_map = self.columns_map();
        let columns: Vec<(String, Arc<dyn Array>, bool)> = names
            .iter()
            .map(|&name| {
                let col = col_map.get(name).ok_or_else(|| missing_column_error("dataframe::to_record_batch", name))?;
                let cv = col.view()?;
                let dtype = cv.data_type()?;
                let nullable = cv.nullable()?;
                let arr = column_view_to_arrow(&cv, dtype)?;
                Ok((name.to_string(), arr, nullable))
            })
            .collect::<Result<_>>()?;

        let batch = RecordBatch::try_from_iter_with_nullable(columns).map_err(map_record_batch_error)?;
        let observation = self
            .boundary_observation_for_query_stage(
                BoundaryKind::DeviceToHost,
                QueryStageKind::FinalMaterialize,
                true,
                "dataframe::to_record_batch",
            )
            .map_err(|error| error.reanchor("dataframe::to_record_batch"))?;
        debug_assert_eq!(observation.placement, placement);
        debug_assert_eq!(observation.rows, rows);
        debug_assert_eq!(observation.columns, column_count);
        record_host_boundary_observation_in_domain(self.execution_domain(), observation);
        Ok(batch)
    }

    /// Create a GPU `DataFrame` from an Arrow [`RecordBatch`] via host copy
    /// inside the caller-provided runtime.
    ///
    /// This is the inverse of [`to_record_batch()`](Self::to_record_batch).
    /// Each Arrow array is copied from host memory to the GPU. Column names
    /// and supported logical values are preserved, and all imported columns
    /// are seeded with the caller runtime's retained provenance. Decimal128
    /// imports preserve values and scale, but cuDF does not retain Arrow
    /// precision metadata; re-export normalizes Decimal128 fields to
    /// `Decimal128(38, scale)`.
    ///
    /// This method imports from host-resident `RecordBatch` data. If you
    /// already have Arrow C Device ABI roots on the same CUDA device, prefer
    /// [`crate::DataFrame::try_import_from_device_arrow_table_into_runtime`]
    /// from [`crate::interop::arrow_device`].
    ///
    /// # Supported types
    ///
    /// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
    ///   `UInt32`, `UInt64`, `Float32`, `Float64`
    /// - `Boolean`
    /// - `Decimal128` (preserves values and scale; precision is re-exported
    ///   as 38)
    /// - `Date32`
    /// - `Timestamp(Second|Millisecond|Microsecond|Nanosecond, None)`
    /// - `Utf8` / `LargeUtf8` (converted to cudf `String`)
    ///
    /// # Errors
    ///
    /// Returns an error if any column has an unsupported Arrow type, or if
    /// the host-to-GPU copy fails.
    pub fn from_record_batch(runtime: &Runtime, batch: &RecordBatch) -> Result<Self> {
        runtime.dataframe_from_record_batch(batch)
    }
}

impl Runtime {
    /// Convert an Arrow [`RecordBatch`] to a GPU [`DataFrame`] via host copy
    /// inside this runtime.
    pub fn dataframe_from_record_batch(&self, batch: &RecordBatch) -> Result<DataFrame> {
        dataframe_from_record_batch(self, batch)
    }
}

/// Convert an Arrow [`RecordBatch`] to a GPU [`DataFrame`] via host copy
/// inside the caller-provided runtime.
///
/// Prefer [`Runtime::dataframe_from_record_batch()`] or
/// [`DataFrame::from_record_batch()`] for a more ergonomic API.
///
/// Each Arrow array is copied from host memory to the GPU. Column names and
/// supported logical values are preserved, and imported columns inherit the
/// supplied runtime's retained provenance. Decimal128 imports preserve values
/// and scale, but cuDF does not retain Arrow precision metadata; re-export
/// normalizes Decimal128 fields to `Decimal128(38, scale)`.
///
/// This is the host `RecordBatch` import path, not Arrow C Device ABI import.
/// For same-device device-root handoff, see
/// [`crate::DataFrame::try_import_from_device_arrow_table_into_runtime`].
///
/// # Supported types
///
/// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
///   `UInt32`, `UInt64`, `Float32`, `Float64`
/// - `Boolean`
/// - `Decimal128` (preserves values and scale; precision is re-exported as 38)
/// - `Date32`
/// - `Timestamp(Second|Millisecond|Microsecond|Nanosecond, None)`
/// - `Utf8` / `LargeUtf8` (converted to cudf `String`)
///
/// Unsupported Arrow types will return an error.
///
/// # Errors
///
/// Returns an error if any column has an unsupported Arrow type, or if
/// the host-to-GPU copy fails.
pub fn dataframe_from_record_batch(runtime: &Runtime, batch: &RecordBatch) -> Result<DataFrame> {
    HOST_ARROW_IMPORT_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    let schema = batch.schema();
    let num_cols = schema.fields().len();

    if num_cols == 0 {
        let frame =
            DataFrame::from_raw_parts(fast_index_map(), batch.num_rows(), Arc::clone(runtime.execution_domain()));
        let observation = frame.boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::InteropImport,
            false,
            "dataframe_from_record_batch",
        )?;
        record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
        return Ok(frame);
    }

    let mut names: Vec<String> = Vec::with_capacity(num_cols);
    let mut columns: Vec<Column> = Vec::with_capacity(num_cols);

    for (i, field) in schema.fields().iter().enumerate() {
        names.push(field.name().clone());
        let arr = batch.column(i);
        let col = arrow_array_to_column(runtime, arr.as_ref(), field.name())?;
        columns.push(col);
    }

    let frame = DataFrame::new(names, columns)?;
    let observation = frame.boundary_observation_for_query_stage(
        BoundaryKind::HostToDevice,
        QueryStageKind::InteropImport,
        false,
        "dataframe_from_record_batch",
    )?;
    record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
    Ok(frame)
}

// ---------------------------------------------------------------------------
// Trait-based cudf TypeId ↔ Arrow DataType mapping
// ---------------------------------------------------------------------------

impl TryFrom<TypeId> for ArrowDataType {
    type Error = Error;

    fn try_from(type_id: TypeId) -> Result<Self> {
        if type_id.is_decimal() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                "arrow::TryFrom<TypeId> for ArrowDataType",
                format!("decimal cudf type {:?} requires scale for Arrow host conversion", type_id),
            )
            .with_fact(FACT_TYPE_ID, type_id)
            .with_detail("reason", "decimal_scale_required")
            .with_detail("remedy", "use arrow_host::arrow_type_from_cudf_dtype with DataType::fixed_point"));
        }
        arrow_type_from_cudf_dtype_in(DataType::new(type_id), "arrow::TryFrom<TypeId> for ArrowDataType")
    }
}

impl TryFrom<&ArrowDataType> for TypeId {
    type Error = Error;

    fn try_from(dt: &ArrowDataType) -> Result<Self> {
        cudf_dtype_from_arrow_type_in(dt, "arrow::TryFrom<&ArrowDataType> for TypeId").map(|dtype| dtype.id())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: ColumnView → Arrow Array
// ---------------------------------------------------------------------------

/// Convert a single `ColumnView` to an Arrow `Array`.
fn column_view_to_arrow(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>> {
    match dtype.id() {
        TypeId::Int8 => numeric_to_arrow::<i8>(cv),
        TypeId::Int16 => numeric_to_arrow::<i16>(cv),
        TypeId::Int32 => numeric_to_arrow::<i32>(cv),
        TypeId::Int64 => numeric_to_arrow::<i64>(cv),
        TypeId::UInt8 => numeric_to_arrow::<u8>(cv),
        TypeId::UInt16 => numeric_to_arrow::<u16>(cv),
        TypeId::UInt32 => numeric_to_arrow::<u32>(cv),
        TypeId::UInt64 => numeric_to_arrow::<u64>(cv),
        TypeId::Float32 => numeric_to_arrow::<f32>(cv),
        TypeId::Float64 => numeric_to_arrow::<f64>(cv),
        TypeId::Bool8 => bool_to_arrow(cv),
        TypeId::String => string_to_arrow(cv),
        TypeId::Decimal128 => decimal128_to_arrow(cv, dtype),
        TypeId::TimestampDays => fixed_width_to_arrow::<i32, arrow::datatypes::Date32Type>(cv),
        TypeId::TimestampSeconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampSecondType>(cv),
        TypeId::TimestampMilliseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampMillisecondType>(cv),
        TypeId::TimestampMicroseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampMicrosecondType>(cv),
        TypeId::TimestampNanoseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampNanosecondType>(cv),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::column_view_to_arrow",
            format!("unsupported cudf type for Arrow conversion: {:?}", dtype.id()),
        )
        .with_fact(FACT_TYPE_ID, dtype.id())),
    }
}

pub(super) fn map_record_batch_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "dataframe::to_record_batch",
        error.to_string(),
        "record_batch",
        "invalid_arrow_record_batch",
        "ensure Arrow schemas and column arrays have matching row counts and compatible field layouts",
    )
    .with_source(error)
}

pub(super) fn map_numeric_to_arrow_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::numeric_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_primitive_array",
        "ensure Arrow primitive values and null buffers have matching logical lengths",
    )
    .with_source(error)
}

/// Convert a numeric column to the corresponding Arrow primitive array.
fn numeric_to_arrow<T>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where T: ArrowPrimitiveMapping {
    let data: Vec<T> = cv.to_vec()?;
    let validity = cv.to_host_validity()?;

    match validity {
        None => {
            let arr = PrimitiveArray::<T::ArrowPrimitive>::from_iter_values(data);
            Ok(Arc::new(arr))
        },
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let buffer = arrow::buffer::ScalarBuffer::from(data);
            let arr = PrimitiveArray::<T::ArrowPrimitive>::try_new(buffer, Some(null_buffer))
                .map_err(map_numeric_to_arrow_error)?;
            Ok(Arc::new(arr))
        },
    }
}

fn cudf_decimal_scale_to_arrow_scale(scale: i32, operation: &'static str) -> Result<i8> {
    let arrow_scale = scale.checked_neg().ok_or_else(|| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "decimal scale is outside Arrow Decimal128 range")
            .with_detail("cudf_scale", scale.to_string())
            .with_detail("reason", "decimal_scale_overflow")
    })?;
    if arrow_scale > i32::from(DEFAULT_DECIMAL128_ARROW_PRECISION) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "decimal scale exceeds cudf-rs Arrow Decimal128 precision",
        )
        .with_detail("cudf_scale", scale.to_string())
        .with_detail("arrow_scale", arrow_scale.to_string())
        .with_detail("arrow_precision", DEFAULT_DECIMAL128_ARROW_PRECISION.to_string())
        .with_detail("reason", "decimal_scale_exceeds_decimal128_precision"));
    }
    i8::try_from(arrow_scale).map_err(|_| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "decimal scale is outside Arrow Decimal128 range")
            .with_detail("cudf_scale", scale.to_string())
            .with_detail("arrow_scale", arrow_scale.to_string())
            .with_detail("reason", "decimal_scale_outside_arrow_i8_range")
    })
}

fn arrow_decimal_scale_to_cudf_scale(scale: i8) -> i32 {
    -i32::from(scale)
}

fn bytes_to_native_vec<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    assert_eq!(bytes.len() % elem_size, 0);
    let len = bytes.len() / elem_size;
    let mut out = vec![T::default(); len];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    out
}

fn native_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn fixed_width_host_values<T: Copy + Default>(cv: &ColumnView<'_>) -> Result<Vec<T>> {
    cv.to_fixed_width_host_bytes().map(|bytes| bytes_to_native_vec::<T>(&bytes))
}

fn fixed_width_to_arrow<T, A>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where
    T: Copy + arrow::datatypes::ArrowNativeType,
    A: arrow::datatypes::ArrowPrimitiveType<Native = T>,
{
    let data = fixed_width_host_values::<T>(cv)?;
    let validity = cv.to_host_validity()?;

    let arr = match validity {
        None => PrimitiveArray::<A>::from_iter_values(data),
        Some(mask) => PrimitiveArray::<A>::from_iter(
            data.into_iter().zip(mask).map(|(value, valid)| if valid { Some(value) } else { None }),
        ),
    };
    Ok(Arc::new(arr))
}

fn map_decimal_to_arrow_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::decimal128_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_decimal_array",
        "ensure cudf Decimal128 scale fits Arrow Decimal128 metadata",
    )
    .with_source(error)
}

fn decimal128_to_arrow(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>> {
    let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), "arrow::decimal128_to_arrow")
        .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
    let data = fixed_width_host_values::<i128>(cv)?;
    let validity = cv.to_host_validity()?;

    let arr = match validity {
        None => Decimal128Array::from(data),
        Some(mask) => Decimal128Array::from(
            data.into_iter().zip(mask).map(|(value, valid)| if valid { Some(value) } else { None }).collect::<Vec<_>>(),
        ),
    }
    .with_precision_and_scale(DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale)
    .map_err(map_decimal_to_arrow_error)?;

    Ok(Arc::new(arr))
}

pub(super) fn reset_host_interop_observation() {
    HOST_ARROW_EXPORT_CALL_COUNT.store(0, Ordering::Relaxed);
    HOST_ARROW_IMPORT_CALL_COUNT.store(0, Ordering::Relaxed);
    record_host_boundary_observation(None);
}

pub(super) fn host_export_call_count() -> usize {
    HOST_ARROW_EXPORT_CALL_COUNT.load(Ordering::Relaxed)
}

pub(super) fn host_import_call_count() -> usize {
    HOST_ARROW_IMPORT_CALL_COUNT.load(Ordering::Relaxed)
}

pub(super) fn last_host_boundary_observation() -> Option<BoundaryObservation> {
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.lock().ok().and_then(|last| *last)
}

fn record_host_boundary_observation(observation: Option<BoundaryObservation>) {
    if let Ok(mut last) = HOST_ARROW_LAST_BOUNDARY_OBSERVATION.lock() {
        *last = observation;
    }
    record_boundary_observation(observation);
}

fn record_host_boundary_observation_in_domain(
    domain: &Arc<crate::runtime::execution_domain::ExecutionDomain>,
    observation: BoundaryObservation,
) {
    if let Ok(mut last) = HOST_ARROW_LAST_BOUNDARY_OBSERVATION.lock() {
        *last = Some(observation);
    }
    record_boundary_observation_in_domain(domain, observation);
}

/// Convert a Bool8 column to an Arrow BooleanArray.
fn bool_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    // Avoid constructing Rust `bool` values directly from raw device bytes.
    let data: Vec<bool> = cv.to_raw_bytes()?.into_iter().map(|byte| byte != 0).collect();
    let validity = cv.to_host_validity()?;

    match validity {
        None => Ok(Arc::new(BooleanArray::from(data))),
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let bool_buffer = arrow::buffer::BooleanBuffer::from(data);
            let arr = BooleanArray::new(bool_buffer, Some(null_buffer));
            Ok(Arc::new(arr))
        },
    }
}

/// Convert a String column to an Arrow StringArray.
fn string_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    let data = cv.to_optional_strings()?;

    if data.iter().all(|value| value.is_some()) {
        let values = data.iter().map(|value| value.as_deref().unwrap_or("")).collect::<Vec<_>>();
        Ok(Arc::new(StringArray::from(values)))
    } else {
        let values = data.iter().map(|value| value.as_deref()).collect::<Vec<_>>();
        Ok(Arc::new(StringArray::from(values)))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: Arrow Array → Column
// ---------------------------------------------------------------------------

/// Convert a single Arrow array to an owned cudf `Column`.
fn arrow_array_to_column(runtime: &Runtime, arr: &dyn Array, field_name: &str) -> Result<Column> {
    let dt = arr.data_type();
    match dt {
        ArrowDataType::Int8 => primitive_to_column::<i8>(runtime, arr),
        ArrowDataType::Int16 => primitive_to_column::<i16>(runtime, arr),
        ArrowDataType::Int32 => primitive_to_column::<i32>(runtime, arr),
        ArrowDataType::Int64 => primitive_to_column::<i64>(runtime, arr),
        ArrowDataType::UInt8 => primitive_to_column::<u8>(runtime, arr),
        ArrowDataType::UInt16 => primitive_to_column::<u16>(runtime, arr),
        ArrowDataType::UInt32 => primitive_to_column::<u32>(runtime, arr),
        ArrowDataType::UInt64 => primitive_to_column::<u64>(runtime, arr),
        ArrowDataType::Float32 => primitive_to_column::<f32>(runtime, arr),
        ArrowDataType::Float64 => primitive_to_column::<f64>(runtime, arr),
        ArrowDataType::Boolean => bool_to_column(runtime, arr),
        ArrowDataType::Date32 => primitive_arrow_to_fixed_width_column::<arrow::datatypes::Date32Type>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampDays),
        ),
        ArrowDataType::Timestamp(TimeUnit::Second, None) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::TimestampSecondType,
        >(
            runtime, arr, DataType::new(TypeId::TimestampSeconds)
        ),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, None) => {
            primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampMillisecondType>(
                runtime,
                arr,
                DataType::new(TypeId::TimestampMilliseconds),
            )
        },
        ArrowDataType::Timestamp(TimeUnit::Microsecond, None) => {
            primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampMicrosecondType>(
                runtime,
                arr,
                DataType::new(TypeId::TimestampMicroseconds),
            )
        },
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, None) => {
            primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampNanosecondType>(
                runtime,
                arr,
                DataType::new(TypeId::TimestampNanoseconds),
            )
        },
        ArrowDataType::Timestamp(_, Some(_)) => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::arrow_array_to_column",
            format!("unsupported Arrow timestamp with timezone for column '{}': {:?}", field_name, dt),
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("arrow_type", format!("{:?}", dt))
        .with_detail("reason", "timezone_not_supported")),
        ArrowDataType::Decimal128(_, scale) => {
            let dtype = DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale));
            decimal128_array_to_column(runtime, arr, dtype)
        },
        ArrowDataType::Utf8 => string_array_to_column(runtime, arr),
        ArrowDataType::LargeUtf8 => large_string_array_to_column(runtime, arr),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::arrow_array_to_column",
            format!("unsupported Arrow type for column '{}': {:?}", field_name, dt),
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("arrow_type", format!("{:?}", dt))),
    }
}

/// Convert a primitive Arrow array to a cudf Column.
fn primitive_to_column<T>(runtime: &Runtime, arr: &dyn Array) -> Result<Column>
where T: ArrowPrimitiveMapping {
    let prim = arr.as_primitive::<T::ArrowPrimitive>();
    let values = prim.values().as_ref();

    match prim.nulls() {
        None => runtime.column_from_slice(values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_slice_nullable(values, &validity)
        },
    }
}

fn primitive_arrow_to_fixed_width_column<A>(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column>
where A: arrow::datatypes::ArrowPrimitiveType {
    let prim = arr.as_primitive::<A>();
    let values = prim.values().as_ref();
    let bytes = native_slice_as_bytes(values);
    let validity = prim.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

fn decimal128_array_to_column(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column> {
    let decimal = arr.as_primitive::<arrow::datatypes::Decimal128Type>();
    let values = decimal.values().as_ref();
    let bytes = native_slice_as_bytes(values);
    let validity = decimal.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

/// Convert a boolean Arrow array to a cudf Column (Bool8).
fn bool_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let bool_arr = arr.as_boolean();
    // Arrow booleans are bit-packed, while cudf Bool8 is byte-per-value.
    // This path still needs an explicit host-side unpacking step before the
    // required host-to-device copy.
    let values: Vec<bool> = bool_arr.values().iter().collect();

    match bool_arr.nulls() {
        None => runtime.column_from_slice(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_slice_nullable(&values, &validity)
        },
    }
}

/// Convert an Arrow StringArray (Utf8) to a cudf STRING Column.
fn string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let str_arr = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        error::foreign_invariant_violation(
            "arrow::string_array_to_column",
            "expected StringArray for Utf8 type",
            "arrow_array",
            "unexpected_arrow_array_layout",
            "ensure Utf8 fields are backed by Arrow StringArray values",
        )
        .with_detail("arrow_type", format!("{:?}", ArrowDataType::Utf8))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
    })?;

    let values: Vec<&str> = (0..str_arr.len()).map(|i| str_arr.value(i)).collect();

    match str_arr.nulls() {
        None => runtime.column_from_strs(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_strs_nullable(&values, &validity)
        },
    }
}

/// Convert an Arrow LargeStringArray (LargeUtf8) to a cudf STRING Column.
fn large_string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let str_arr = arr.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
        error::foreign_invariant_violation(
            "arrow::large_string_array_to_column",
            "expected LargeStringArray for LargeUtf8 type",
            "arrow_array",
            "unexpected_arrow_array_layout",
            "ensure LargeUtf8 fields are backed by Arrow LargeStringArray values",
        )
        .with_detail("arrow_type", format!("{:?}", ArrowDataType::LargeUtf8))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
    })?;

    let values: Vec<&str> = (0..str_arr.len()).map(|i| str_arr.value(i)).collect();

    match str_arr.nulls() {
        None => runtime.column_from_strs(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_strs_nullable(&values, &validity)
        },
    }
}

// ---------------------------------------------------------------------------
// Trait bounds helper: map Rust primitives to Arrow primitive types
// ---------------------------------------------------------------------------

/// Trait to associate a `CudfType` with its Arrow primitive type.
///
/// This avoids needing separate functions for each numeric type.
trait ArrowPrimitiveMapping: crate::core::types::CudfType + arrow::datatypes::ArrowNativeType {
    type ArrowPrimitive: arrow::datatypes::ArrowPrimitiveType<Native = Self>;
}

impl ArrowPrimitiveMapping for i8 {
    type ArrowPrimitive = arrow::datatypes::Int8Type;
}
impl ArrowPrimitiveMapping for i16 {
    type ArrowPrimitive = arrow::datatypes::Int16Type;
}
impl ArrowPrimitiveMapping for i32 {
    type ArrowPrimitive = arrow::datatypes::Int32Type;
}
impl ArrowPrimitiveMapping for i64 {
    type ArrowPrimitive = arrow::datatypes::Int64Type;
}
impl ArrowPrimitiveMapping for u8 {
    type ArrowPrimitive = arrow::datatypes::UInt8Type;
}
impl ArrowPrimitiveMapping for u16 {
    type ArrowPrimitive = arrow::datatypes::UInt16Type;
}
impl ArrowPrimitiveMapping for u32 {
    type ArrowPrimitive = arrow::datatypes::UInt32Type;
}
impl ArrowPrimitiveMapping for u64 {
    type ArrowPrimitive = arrow::datatypes::UInt64Type;
}
impl ArrowPrimitiveMapping for f32 {
    type ArrowPrimitive = arrow::datatypes::Float32Type;
}
impl ArrowPrimitiveMapping for f64 {
    type ArrowPrimitive = arrow::datatypes::Float64Type;
}
