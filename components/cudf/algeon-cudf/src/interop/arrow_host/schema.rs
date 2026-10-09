//! Arrow host schema validation and cudf/Arrow type mapping.

use arrow::{
    datatypes::{
        DataType as ArrowDataType, Decimal32Type, Decimal64Type, Decimal128Type, DecimalType, FieldRef, Schema,
        TimeUnit, validate_decimal_precision_and_scale,
    },
    record_batch::RecordBatch,
};
use rapids_interop::{DataType as InteropDataType, TypeId as InteropTypeId};

use crate::{
    core::{
        error::{Error, ErrorCode, FACT_TYPE_ID, Result},
        types::{DataType, TypeId},
    },
    internal::resolve::validate_no_duplicate_names,
};

pub(super) const DEFAULT_DECIMAL128_ARROW_PRECISION: u8 = 38;

/// Copy-free planner view of one Arrow host field as cudf would import it.
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

    /// cudf logical type that the host Arrow import path would create.
    pub fn cudf_data_type(&self) -> DataType {
        self.cudf_dtype
    }

    /// Arrow field nullability. cudf physical dtypes do not encode this bit.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// Copy-free planner view of an Arrow host schema as cudf would import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowHostSchema {
    fields: Vec<ArrowHostField>,
    row_count: usize,
}

impl ArrowHostSchema {
    fn new(fields: Vec<ArrowHostField>, row_count: usize) -> Self {
        Self { fields, row_count }
    }

    pub(crate) fn with_row_count(&self, row_count: usize) -> Self {
        Self { fields: self.fields.clone(), row_count }
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

/// Convert a supported Arrow host logical type into a cudf logical type.
///
/// This is a copy-free planner/schema helper. It follows the same support
/// envelope as [`DataFrame::from_record_batch`](crate::DataFrame::from_record_batch): numeric types, Boolean,
/// Decimal128, Date32, timestamps, durations with Arrow time units, Utf8,
/// LargeUtf8, Utf8View, top-level `List` / `LargeList` with string-like or
/// numeric children, and positive-width `FixedSizeList` with numeric children.
/// Timestamp timezone metadata is accepted as an Arrow schema contract; cuDF
/// stores the physical epoch values and resolution.
pub fn cudf_dtype_from_arrow_type(dt: &ArrowDataType) -> Result<DataType> {
    cudf_dtype_from_arrow_type_in(dt, "arrow_host::cudf_dtype_from_arrow_type")
}

/// Convert a supported cudf logical type into the Arrow host type used by
/// [`DataFrame::to_record_batch`](crate::DataFrame::to_record_batch).
///
/// Decimal32, Decimal64, and Decimal128 export as Arrow Decimal128 with
/// precision 38 because this dtype-only path has no retained Arrow precision
/// metadata. Schema-aware exports preserve narrow logical decimal widths.
pub fn arrow_type_from_cudf_dtype(dtype: DataType) -> Result<ArrowDataType> {
    arrow_type_from_cudf_dtype_in(dtype, "arrow_host::arrow_type_from_cudf_dtype")
}

/// Return whether this Arrow host type is supported by cudf host import.
pub fn is_supported_arrow_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).is_ok()
}

/// Return whether this Arrow host type is a supported numeric cudf type.
pub fn is_supported_arrow_numeric_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).map(|dtype| dtype.id().is_numeric()).unwrap_or(false)
}

/// Return whether a cudf logical dtype can materialize into the requested
/// Arrow host type without changing the column's logical values.
///
/// This follows the export side of the host-copy interop contract, which differs
/// from Arrow import mapping. The predicate accepts exact Decimal32/64 mappings,
/// while schema-less export canonicalizes every physical decimal to Decimal128
/// because libcudf retains scale but not Arrow precision metadata.
///
/// Nested cudf dtypes do not retain their child schema in [`DataType`], so
/// this dtype-only predicate returns `false` for Arrow list types by design.
/// Owners with a concrete column view, such as `GpuRecordBatch`, must validate
/// the actual child dtype and any fixed-size shape contract.
pub fn cudf_dtype_can_export_to_arrow_type(cudf_dtype: DataType, arrow_type: &ArrowDataType) -> bool {
    match arrow_type {
        ArrowDataType::Decimal128(_, scale) if cudf_dtype.id().is_decimal() => {
            cudf_dtype.scale() == arrow_decimal_scale_to_cudf_scale(*scale)
        },
        ArrowDataType::List(_) | ArrowDataType::LargeList(_) | ArrowDataType::FixedSizeList(_, _)
            if cudf_dtype.id() == TypeId::List =>
        {
            false
        },
        _ => cudf_dtype_from_arrow_type(arrow_type)
            .map(|arrow_cudf_dtype| arrow_cudf_dtype == cudf_dtype)
            .unwrap_or(false),
    }
}

/// Validate an Arrow schema without copying data and return cudf planner metadata.
///
/// Nullability is kept as schema metadata, separate from the cudf physical
/// dtype.
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

/// Validate a RecordBatch schema without copying array data and return cudf planner metadata.
pub fn validate_record_batch_schema(batch: &RecordBatch) -> Result<ArrowHostSchema> {
    validate_arrow_schema(batch.schema().as_ref(), batch.num_rows())
        .map_err(|error| error.reanchor("arrow_host::validate_record_batch_schema"))
}

pub(super) fn cudf_dtype_from_arrow_type_in(dt: &ArrowDataType, operation: &'static str) -> Result<DataType> {
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
        ArrowDataType::Date64 => DataType::new(TypeId::TimestampMilliseconds),
        ArrowDataType::Time32(TimeUnit::Second) => DataType::new(TypeId::DurationSeconds),
        ArrowDataType::Time32(TimeUnit::Millisecond) => DataType::new(TypeId::DurationMilliseconds),
        ArrowDataType::Time64(TimeUnit::Microsecond) => DataType::new(TypeId::DurationMicroseconds),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => DataType::new(TypeId::DurationNanoseconds),
        ArrowDataType::Timestamp(unit, _) => cudf_timestamp_dtype(unit),
        ArrowDataType::Duration(TimeUnit::Second) => DataType::new(TypeId::DurationSeconds),
        ArrowDataType::Duration(TimeUnit::Millisecond) => DataType::new(TypeId::DurationMilliseconds),
        ArrowDataType::Duration(TimeUnit::Microsecond) => DataType::new(TypeId::DurationMicroseconds),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => DataType::new(TypeId::DurationNanoseconds),
        ArrowDataType::Decimal128(precision, scale) => {
            validate_arrow_decimal::<Decimal128Type>(*precision, *scale, operation)?;
            DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale))?
        },
        ArrowDataType::Decimal32(precision, scale) => {
            validate_arrow_decimal::<Decimal32Type>(*precision, *scale, operation)?;
            DataType::fixed_point(TypeId::Decimal32, arrow_decimal_scale_to_cudf_scale(*scale))?
        },
        ArrowDataType::Decimal64(precision, scale) => {
            validate_arrow_decimal::<Decimal64Type>(*precision, *scale, operation)?;
            DataType::fixed_point(TypeId::Decimal64, arrow_decimal_scale_to_cudf_scale(*scale))?
        },
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => DataType::new(TypeId::String),
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {
            DataType::new(TypeId::List)
        },
        ArrowDataType::FixedSizeList(field, dimension) => {
            if *dimension <= 0 {
                return Err(Error::new_with(
                    ErrorCode::ValueOutOfRange,
                    operation,
                    "Arrow FixedSizeList dimension must be positive for cudf conversion",
                )
                .with_detail("dimension", dimension.to_string())
                .with_detail("reason", "non_positive_fixed_size_list_dimension"));
            }
            if !list_numeric_child_is_supported(field) {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    format!("unsupported Arrow FixedSizeList child type for cudf conversion: {:?}", field.data_type()),
                )
                .with_detail("arrow_type", format!("{:?}", dt))
                .with_detail("reason", "unsupported_fixed_size_list_child_type"));
            }
            DataType::new(TypeId::List)
        },
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

pub(super) fn validate_arrow_decimal<T: DecimalType>(precision: u8, scale: i8, operation: &'static str) -> Result<()> {
    validate_decimal_precision_and_scale::<T>(precision, scale).map_err(|error| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "Arrow decimal precision and scale are invalid")
            .with_detail("precision", precision.to_string())
            .with_detail("scale", scale.to_string())
            .with_detail("reason", "invalid_arrow_decimal_precision_scale")
            .with_source(error)
    })
}

pub(super) fn list_string_child_is_supported(field: &FieldRef) -> bool {
    matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

pub(super) fn list_numeric_child_is_supported(field: &FieldRef) -> bool {
    numeric_interop_dtype(field.data_type()).is_some()
}

pub(super) fn list_fixed_width_child_is_supported(field: &FieldRef) -> bool {
    list_fixed_width_interop_dtype(field.data_type()).is_some()
}

pub(super) fn list_child_is_supported(field: &FieldRef) -> bool {
    list_string_child_is_supported(field) || list_fixed_width_child_is_supported(field)
}

pub(super) fn cudf_timestamp_dtype(unit: &TimeUnit) -> DataType {
    match unit {
        TimeUnit::Second => DataType::new(TypeId::TimestampSeconds),
        TimeUnit::Millisecond => DataType::new(TypeId::TimestampMilliseconds),
        TimeUnit::Microsecond => DataType::new(TypeId::TimestampMicroseconds),
        TimeUnit::Nanosecond => DataType::new(TypeId::TimestampNanoseconds),
    }
}

pub(super) fn arrow_type_from_cudf_dtype_in(dtype: DataType, operation: &'static str) -> Result<ArrowDataType> {
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
        TypeId::DurationSeconds => Ok(ArrowDataType::Duration(TimeUnit::Second)),
        TypeId::DurationMilliseconds => Ok(ArrowDataType::Duration(TimeUnit::Millisecond)),
        TypeId::DurationMicroseconds => Ok(ArrowDataType::Duration(TimeUnit::Microsecond)),
        TypeId::DurationNanoseconds => Ok(ArrowDataType::Duration(TimeUnit::Nanosecond)),
        TypeId::String => Ok(ArrowDataType::Utf8),
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
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

pub(super) fn cudf_decimal_scale_to_arrow_scale(scale: i32, operation: &'static str) -> Result<i8> {
    let arrow_scale = scale.checked_neg().ok_or_else(|| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "decimal scale is outside Arrow Decimal128 range")
            .with_detail("cudf_scale", scale.to_string())
            .with_detail("reason", "decimal_scale_overflow")
    })?;
    if arrow_scale > i32::from(DEFAULT_DECIMAL128_ARROW_PRECISION) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "decimal scale exceeds cudf Arrow Decimal128 precision",
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

pub(super) fn arrow_decimal_scale_to_cudf_scale(scale: i8) -> i32 {
    -i32::from(scale)
}

pub(super) fn numeric_interop_dtype(data_type: &ArrowDataType) -> Option<InteropDataType> {
    let type_id = match data_type {
        ArrowDataType::Int8 => InteropTypeId::Int8,
        ArrowDataType::Int16 => InteropTypeId::Int16,
        ArrowDataType::Int32 => InteropTypeId::Int32,
        ArrowDataType::Int64 => InteropTypeId::Int64,
        ArrowDataType::UInt8 => InteropTypeId::UInt8,
        ArrowDataType::UInt16 => InteropTypeId::UInt16,
        ArrowDataType::UInt32 => InteropTypeId::UInt32,
        ArrowDataType::UInt64 => InteropTypeId::UInt64,
        ArrowDataType::Float32 => InteropTypeId::Float32,
        ArrowDataType::Float64 => InteropTypeId::Float64,
        _ => return None,
    };
    Some(InteropDataType::new(type_id))
}
pub(super) fn list_fixed_width_interop_dtype(data_type: &ArrowDataType) -> Option<InteropDataType> {
    match data_type {
        ArrowDataType::Boolean => Some(InteropDataType::new(InteropTypeId::Bool8)),
        data_type => numeric_interop_dtype(data_type),
    }
}
