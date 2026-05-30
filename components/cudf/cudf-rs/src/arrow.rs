//! Arrow interop: GPU `DataFrame` ↔ Arrow `RecordBatch` conversion.
//!
//! This module provides a **host-copy** path: data is transferred
//! GPU → CPU → Arrow (or Arrow → CPU → GPU). A zero-copy GPU Arrow
//! path would require future cudf-sys work.
//!
//! Feature-gated behind `"arrow"` in `Cargo.toml`.
//!
//! # Example
//!
//! ```no_run
//! use cudf::column::Column;
//! use cudf::dataframe::DataFrame;
//! use cudf::arrow::dataframe_from_record_batch;
//!
//! let df = DataFrame::new(
//!     vec!["id".into(), "score".into()],
//!     vec![
//!         Column::from_slice(&[1i32, 2, 3]).unwrap(),
//!         Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap(),
//!     ],
//! ).unwrap();
//!
//! // GPU → Arrow RecordBatch (host-copy)
//! let batch = df.to_record_batch().unwrap();
//! assert_eq!(batch.num_rows(), 3);
//!
//! // Arrow RecordBatch → GPU DataFrame (host-copy)
//! let df2 = dataframe_from_record_batch(&batch).unwrap();
//! assert_eq!(df2.num_columns(), 2);
//! ```

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::column::{Column, ColumnView};
use crate::dataframe::DataFrame;
use crate::error::{Error, ErrorKind, Result};
use crate::types::TypeId;

impl DataFrame {
    /// Convert this GPU DataFrame to an Arrow [`RecordBatch`] via host copy.
    ///
    /// Each column is copied from GPU to host memory, then wrapped in the
    /// corresponding Arrow array type. The schema (column names and types)
    /// is preserved.
    ///
    /// # Supported types
    ///
    /// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
    ///   `UInt32`, `UInt64`, `Float32`, `Float64`
    /// - `Bool8`
    ///
    /// Variable-width types (e.g., `String`, `List`) are not yet supported
    /// and will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if any column has an unsupported type, or if the
    /// underlying GPU-to-host copy fails.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let names = self.column_names();
        let num_cols = names.len();

        let mut fields: Vec<Field> = Vec::with_capacity(num_cols);
        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(num_cols);

        for &name in &names {
            let cv = self.column(name)?;
            let type_id = cv.data_type()?.id();
            let nullable = cv.nullable()?;

            let arrow_dt = cudf_type_to_arrow(type_id)?;
            fields.push(Field::new(name, arrow_dt, nullable));

            let arr = column_view_to_arrow(&cv, type_id)?;
            arrays.push(arr);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| {
            Error::new(ErrorKind::InvalidOperation, e.to_string())
                .with_operation("dataframe::to_record_batch")
        })
    }
}

/// Convert an Arrow [`RecordBatch`] to a GPU [`DataFrame`] via host copy.
///
/// Each Arrow array is copied from host memory to the GPU. The schema
/// (column names and types) is preserved.
///
/// # Supported types
///
/// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
///   `UInt32`, `UInt64`, `Float32`, `Float64`
/// - `Boolean`
///
/// Unsupported Arrow types will return an error.
///
/// # Errors
///
/// Returns an error if any column has an unsupported Arrow type, or if
/// the host-to-GPU copy fails.
pub fn dataframe_from_record_batch(batch: &RecordBatch) -> Result<DataFrame> {
    let schema = batch.schema();
    let num_cols = schema.fields().len();

    let mut names: Vec<String> = Vec::with_capacity(num_cols);
    let mut columns: Vec<Column> = Vec::with_capacity(num_cols);

    for (i, field) in schema.fields().iter().enumerate() {
        names.push(field.name().clone());
        let arr = batch.column(i);
        let col = arrow_array_to_column(arr.as_ref(), field.name())?;
        columns.push(col);
    }

    DataFrame::new(names, columns)
}

// ---------------------------------------------------------------------------
// Internal helpers: cudf TypeId ↔ Arrow DataType mapping
// ---------------------------------------------------------------------------

/// Map a cudf `TypeId` to the corresponding Arrow `DataType`.
fn cudf_type_to_arrow(type_id: TypeId) -> Result<ArrowDataType> {
    match type_id {
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
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("unsupported cudf type for Arrow conversion: {:?}", type_id),
        )
        .with_operation("arrow::cudf_type_to_arrow")
        .with_context("type_id", format!("{:?}", type_id).to_lowercase())),
    }
}

/// Map an Arrow `DataType` to the corresponding cudf `TypeId`.
#[allow(dead_code)]
fn arrow_type_to_cudf(dt: &ArrowDataType) -> Result<TypeId> {
    match dt {
        ArrowDataType::Int8 => Ok(TypeId::Int8),
        ArrowDataType::Int16 => Ok(TypeId::Int16),
        ArrowDataType::Int32 => Ok(TypeId::Int32),
        ArrowDataType::Int64 => Ok(TypeId::Int64),
        ArrowDataType::UInt8 => Ok(TypeId::UInt8),
        ArrowDataType::UInt16 => Ok(TypeId::UInt16),
        ArrowDataType::UInt32 => Ok(TypeId::UInt32),
        ArrowDataType::UInt64 => Ok(TypeId::UInt64),
        ArrowDataType::Float32 => Ok(TypeId::Float32),
        ArrowDataType::Float64 => Ok(TypeId::Float64),
        ArrowDataType::Boolean => Ok(TypeId::Bool8),
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("unsupported Arrow type for cudf conversion: {:?}", dt),
        )
        .with_operation("arrow::arrow_type_to_cudf")
        .with_context("arrow_type", format!("{:?}", dt))),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: ColumnView → Arrow Array
// ---------------------------------------------------------------------------

/// Convert a single `ColumnView` to an Arrow `Array`.
fn column_view_to_arrow(cv: &ColumnView<'_>, type_id: TypeId) -> Result<Arc<dyn Array>> {
    match type_id {
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
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("unsupported cudf type for Arrow conversion: {:?}", type_id),
        )
        .with_operation("arrow::column_view_to_arrow")
        .with_context("type_id", format!("{:?}", type_id).to_lowercase())),
    }
}

/// Convert a numeric column to the corresponding Arrow primitive array.
fn numeric_to_arrow<T>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where
    T: ArrowPrimitiveMapping,
{
    let data: Vec<T> = cv.to_vec()?;
    let validity = cv.validity()?;

    match validity {
        None => {
            let buffer = arrow::buffer::ScalarBuffer::from(data);
            let arr = PrimitiveArray::<T::ArrowPrimitive>::new(buffer, None);
            Ok(Arc::new(arr))
        }
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let buffer = arrow::buffer::ScalarBuffer::from(data);
            let arr = PrimitiveArray::<T::ArrowPrimitive>::try_new(buffer, Some(null_buffer))
                .map_err(|e: arrow::error::ArrowError| {
                    Error::new(ErrorKind::InvalidOperation, e.to_string())
                        .with_operation("arrow::numeric_to_arrow")
                })?;
            Ok(Arc::new(arr))
        }
    }
}

/// Convert a Bool8 column to an Arrow BooleanArray.
fn bool_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    let data: Vec<bool> = cv.to_vec()?;
    let validity = cv.validity()?;

    match validity {
        None => Ok(Arc::new(BooleanArray::from(data))),
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let bool_buffer = arrow::buffer::BooleanBuffer::from(data);
            let arr = BooleanArray::new(bool_buffer, Some(null_buffer));
            Ok(Arc::new(arr))
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: Arrow Array → Column
// ---------------------------------------------------------------------------

/// Convert a single Arrow array to an owned cudf `Column`.
fn arrow_array_to_column(arr: &dyn Array, field_name: &str) -> Result<Column> {
    let dt = arr.data_type();
    match dt {
        ArrowDataType::Int8 => primitive_to_column::<i8>(arr),
        ArrowDataType::Int16 => primitive_to_column::<i16>(arr),
        ArrowDataType::Int32 => primitive_to_column::<i32>(arr),
        ArrowDataType::Int64 => primitive_to_column::<i64>(arr),
        ArrowDataType::UInt8 => primitive_to_column::<u8>(arr),
        ArrowDataType::UInt16 => primitive_to_column::<u16>(arr),
        ArrowDataType::UInt32 => primitive_to_column::<u32>(arr),
        ArrowDataType::UInt64 => primitive_to_column::<u64>(arr),
        ArrowDataType::Float32 => primitive_to_column::<f32>(arr),
        ArrowDataType::Float64 => primitive_to_column::<f64>(arr),
        ArrowDataType::Boolean => bool_to_column(arr),
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            format!(
                "unsupported Arrow type for column '{}': {:?}",
                field_name, dt
            ),
        )
        .with_operation("arrow::arrow_array_to_column")
        .with_context("field_name", field_name.to_string())
        .with_context("arrow_type", format!("{:?}", dt))),
    }
}

/// Convert a primitive Arrow array to a cudf Column.
fn primitive_to_column<T>(arr: &dyn Array) -> Result<Column>
where
    T: ArrowPrimitiveMapping,
{
    let prim = arr
        .as_any()
        .downcast_ref::<PrimitiveArray<T::ArrowPrimitive>>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "failed to downcast Arrow array to expected primitive type",
            )
            .with_operation("arrow::primitive_to_column")
        })?;

    let values: Vec<T> = prim.values().iter().copied().collect();
    let nulls = prim.nulls();

    match nulls {
        None => Column::from_slice(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = (0..arr.len()).map(|i| null_buf.is_valid(i)).collect();
            Column::from_slice_nullable(&values, &validity)
        }
    }
}

/// Convert a boolean Arrow array to a cudf Column (Bool8).
fn bool_to_column(arr: &dyn Array) -> Result<Column> {
    let bool_arr = arr
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "failed to downcast Arrow array to BooleanArray",
            )
            .with_operation("arrow::bool_to_column")
        })?;

    let values: Vec<bool> = (0..bool_arr.len()).map(|i| bool_arr.value(i)).collect();
    let nulls = bool_arr.nulls();

    match nulls {
        None => Column::from_slice(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = (0..bool_arr.len()).map(|i| null_buf.is_valid(i)).collect();
            Column::from_slice_nullable(&values, &validity)
        }
    }
}

// ---------------------------------------------------------------------------
// Trait bounds helper: map Rust primitives to Arrow primitive types
// ---------------------------------------------------------------------------

/// Trait to associate a `CudfType` with its Arrow primitive type.
///
/// This avoids needing separate functions for each numeric type.
trait ArrowPrimitiveMapping: crate::types::CudfType + arrow::datatypes::ArrowNativeType {
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
