use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, PrimitiveArray},
    compute::cast,
    datatypes::{
        DataType as ArrowDataType, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, SchemaRef, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
        Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    },
    record_batch::{RecordBatch, RecordBatchOptions},
};

use super::DataFrame;
use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        schema,
    },
    interop::{
        arrow_device::DeviceArrowTableExport,
        arrow_host::{ArrowHostSchema, cudf_dtype_can_export_to_arrow_type, validate_arrow_schema},
    },
    runtime::ExecutionPlacement,
};

/// GPU-resident DataFrame plus an external Arrow schema contract.
///
/// This is a query-engine convenience wrapper. It keeps the cudf-native
/// [`DataFrame`] available for GPU operators while retaining the Arrow schema
/// expected by an external planner or result sink. Host materialization remains
/// explicit through [`to_host_record_batch`](Self::to_host_record_batch).
#[derive(Clone)]
pub struct GpuRecordBatch {
    frame: DataFrame,
    schema: SchemaRef,
    mapped_schema: ArrowHostSchema,
}

impl GpuRecordBatch {
    /// Validate and wrap a GPU DataFrame with an external Arrow schema.
    ///
    /// Validation is copy-free. Field count, field names in order, and
    /// Arrow-to-cudf logical dtypes must be materializable from the DataFrame
    /// schema under the host-copy export contract. Arrow nullability remains
    /// external schema metadata; it is not folded into the cudf physical dtype.
    pub fn new(frame: DataFrame, schema: SchemaRef) -> Result<Self> {
        let mapped_schema = validate_frame_schema(&frame, &schema)?;
        Ok(Self { frame, schema, mapped_schema })
    }

    /// Borrow the wrapped GPU DataFrame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Borrow the external Arrow schema contract.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Borrow the validated Arrow-to-cudf planner metadata.
    pub fn mapped_schema(&self) -> &ArrowHostSchema {
        &self.mapped_schema
    }

    /// Logical row count from the wrapped GPU DataFrame.
    pub fn num_rows(&self) -> usize {
        self.frame.num_rows()
    }

    /// Column count from the wrapped GPU DataFrame.
    pub fn num_columns(&self) -> usize {
        self.frame.num_columns()
    }

    /// Retained execution placement from the wrapped GPU DataFrame.
    pub fn placement(&self) -> ExecutionPlacement {
        self.frame.placement()
    }

    /// Materialize this GPU batch as a host Arrow RecordBatch.
    ///
    /// This delegates to [`DataFrame::to_record_batch`] for the device-to-host
    /// copy, then attaches this batch's stored Arrow schema contract. When
    /// cudf's host export type is a supported physical equivalent of the
    /// stored Arrow field type, the host array is cast before the final
    /// `RecordBatch` is built.
    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        let batch = self.frame.to_record_batch()?;
        record_batch_with_schema_contract(batch, &self.schema)
    }

    /// Prepare Arrow C Device ABI export for the wrapped GPU DataFrame.
    ///
    /// This is a device-backed interop boundary, not host materialization.
    pub fn try_prepare_device_arrow_export(&self) -> Result<DeviceArrowTableExport> {
        self.frame.try_prepare_device_arrow_export()
    }

    /// Consume the wrapper and return the GPU DataFrame.
    pub fn into_frame(self) -> DataFrame {
        self.frame
    }

    /// Consume the wrapper and return its parts, including validated planner metadata.
    pub fn into_parts(self) -> (DataFrame, SchemaRef, ArrowHostSchema) {
        (self.frame, self.schema, self.mapped_schema)
    }
}

fn validate_frame_schema(frame: &DataFrame, schema_ref: &SchemaRef) -> Result<ArrowHostSchema> {
    let operation = "dataframe::GpuRecordBatch::new";
    let schema = schema_ref.as_ref();
    if schema.fields().len() != frame.num_columns() {
        return Err(schema::column_name_count_mismatch_error(
            operation,
            "Arrow schema field count must match DataFrame column count",
            schema.fields().len(),
            frame.num_columns(),
        ));
    }

    let mapped =
        validate_arrow_schema(schema, frame.num_rows()).map_err(|error| error.reanchor_if_needed(operation))?;
    let frame_names = frame.column_names();
    debug_assert_eq!(mapped.len(), frame_names.len());

    for (index, (field, frame_name)) in mapped.fields().iter().zip(frame_names.iter()).enumerate() {
        if field.name() != *frame_name {
            return Err(crate::internal::resolve::missing_column_error(operation, field.name())
                .with_detail("field_name", field.name().to_string())
                .with_detail("expected_column_name", (*frame_name).to_string())
                .with_detail("field_index", index.to_string())
                .with_detail("reason", "schema_name_mismatch"));
        }

        let frame_dtype = frame.series(frame_name)?.dtype().map_err(|error| error.reanchor_if_needed(operation))?;
        let schema_dtype = field.cudf_data_type();
        if !cudf_dtype_can_export_to_arrow_type(frame_dtype, field.arrow_data_type()) {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                format!(
                    "Arrow schema field '{}' maps to {:?} scale {} but DataFrame column has {:?} scale {}",
                    field.name(),
                    schema_dtype.id(),
                    schema_dtype.scale(),
                    frame_dtype.id(),
                    frame_dtype.scale()
                ),
            )
            .with_fact(FACT_LEFT_TYPE_ID, frame_dtype.id())
            .with_fact(FACT_RIGHT_TYPE_ID, schema_dtype.id())
            .with_detail("field_name", field.name().to_string())
            .with_detail("field_index", index.to_string())
            .with_detail("frame_dtype_scale", frame_dtype.scale().to_string())
            .with_detail("schema_cudf_dtype_scale", schema_dtype.scale().to_string())
            .with_detail("reason", "schema_dtype_mismatch"));
        }
    }

    Ok(mapped)
}

fn record_batch_with_schema_contract(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    let operation = "dataframe::GpuRecordBatch::to_host_record_batch";
    if batch.num_columns() != schema.fields().len() {
        return Err(schema::column_name_count_mismatch_error(
            operation,
            "materialized Arrow column count must match stored schema field count",
            schema.fields().len(),
            batch.num_columns(),
        ));
    }

    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            let column = batch.column(index);
            if column.data_type() == field.data_type() {
                return Ok(column.clone());
            }

            if let Some(column) = timestamp_array_with_schema_timezone(column.as_ref(), field.data_type())? {
                return Ok(column);
            }

            if let Some(column) = time_array_with_schema_type(column.as_ref(), field.data_type())? {
                return Ok(column);
            }

            cast(column.as_ref(), field.data_type()).map_err(|error| {
                map_arrow_schema_contract_error(operation, error)
                    .with_detail("field_name", field.name().clone())
                    .with_detail("field_index", index.to_string())
                    .with_detail("actual_arrow_type", format!("{:?}", column.data_type()))
                    .with_detail("expected_arrow_type", format!("{:?}", field.data_type()))
                    .with_detail("schema_contract_failure", "schema_cast_failed")
            })
        })
        .collect::<Result<Vec<ArrayRef>>>()?;

    let options = RecordBatchOptions::new().with_row_count((columns.is_empty()).then_some(batch.num_rows()));
    RecordBatch::try_new_with_options(schema.clone(), columns, &options).map_err(|error| {
        map_arrow_schema_contract_error(operation, error)
            .with_detail("schema_contract_failure", "schema_record_batch_build_failed")
    })
}

fn timestamp_array_with_schema_timezone(column: &dyn Array, expected_type: &ArrowDataType) -> Result<Option<ArrayRef>> {
    let (actual_unit, expected_unit, expected_timezone) = match (column.data_type(), expected_type) {
        (
            ArrowDataType::Timestamp(actual_unit, _),
            ArrowDataType::Timestamp(expected_unit, Some(expected_timezone)),
        ) if actual_unit == expected_unit => (actual_unit, expected_unit, expected_timezone.clone()),
        _ => return Ok(None),
    };

    let timezone = Some(expected_timezone);
    let array: ArrayRef = match (actual_unit, expected_unit) {
        (TimeUnit::Second, TimeUnit::Second) => {
            Arc::new(timestamp_array_with_timezone::<TimestampSecondType>(column, timezone)?)
        },
        (TimeUnit::Millisecond, TimeUnit::Millisecond) => {
            Arc::new(timestamp_array_with_timezone::<TimestampMillisecondType>(column, timezone)?)
        },
        (TimeUnit::Microsecond, TimeUnit::Microsecond) => {
            Arc::new(timestamp_array_with_timezone::<TimestampMicrosecondType>(column, timezone)?)
        },
        (TimeUnit::Nanosecond, TimeUnit::Nanosecond) => {
            Arc::new(timestamp_array_with_timezone::<TimestampNanosecondType>(column, timezone)?)
        },
        _ => unreachable!("timestamp units checked above"),
    };
    Ok(Some(array))
}

fn time_array_with_schema_type(column: &dyn Array, expected_type: &ArrowDataType) -> Result<Option<ArrayRef>> {
    let column: ArrayRef = match (column.data_type(), expected_type) {
        (ArrowDataType::Duration(TimeUnit::Second), ArrowDataType::Time32(TimeUnit::Second)) => {
            Arc::new(PrimitiveArray::<Time32SecondType>::from(duration_to_time32_values::<DurationSecondType>(
                column,
                expected_type,
            )?))
        },
        (ArrowDataType::Duration(TimeUnit::Millisecond), ArrowDataType::Time32(TimeUnit::Millisecond)) => {
            Arc::new(PrimitiveArray::<Time32MillisecondType>::from(
                duration_to_time32_values::<DurationMillisecondType>(column, expected_type)?,
            ))
        },
        (ArrowDataType::Duration(TimeUnit::Microsecond), ArrowDataType::Time64(TimeUnit::Microsecond)) => {
            Arc::new(PrimitiveArray::<Time64MicrosecondType>::from(
                duration_to_time64_values::<DurationMicrosecondType>(column, expected_type)?,
            ))
        },
        (ArrowDataType::Duration(TimeUnit::Nanosecond), ArrowDataType::Time64(TimeUnit::Nanosecond)) => {
            Arc::new(PrimitiveArray::<Time64NanosecondType>::from(duration_to_time64_values::<DurationNanosecondType>(
                column,
                expected_type,
            )?))
        },
        _ => return Ok(None),
    };
    Ok(Some(column))
}

fn duration_to_time32_values<D>(column: &dyn Array, expected_type: &ArrowDataType) -> Result<Vec<Option<i32>>>
where D: arrow::datatypes::ArrowPrimitiveType<Native = i64> {
    let array = column.as_any().downcast_ref::<PrimitiveArray<D>>().ok_or_else(|| {
        duration_time_layout_error(
            "duration column has an unexpected Arrow primitive layout",
            column.data_type(),
            expected_type,
            "duration_time32_layout_mismatch",
        )
    })?;
    array
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .map(|value| {
                    i32::try_from(value).map_err(|_| {
                        Error::new_with(
                            ErrorCode::ValueOutOfRange,
                            "dataframe::GpuRecordBatch::to_host_record_batch",
                            "duration value does not fit Arrow Time32 schema contract",
                        )
                        .with_detail("value_index", index.to_string())
                        .with_detail("value", value.to_string())
                        .with_detail("expected_arrow_type", format!("{expected_type:?}"))
                        .with_detail("schema_contract_failure", "duration_time32_value_out_of_range")
                    })
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()
}

fn duration_to_time64_values<D>(column: &dyn Array, expected_type: &ArrowDataType) -> Result<Vec<Option<i64>>>
where D: arrow::datatypes::ArrowPrimitiveType<Native = i64> {
    let array = column.as_any().downcast_ref::<PrimitiveArray<D>>().ok_or_else(|| {
        duration_time_layout_error(
            "duration column has an unexpected Arrow primitive layout",
            column.data_type(),
            expected_type,
            "duration_time64_layout_mismatch",
        )
    })?;
    Ok(array.iter().collect::<Vec<_>>())
}

fn duration_time_layout_error(
    message: &'static str,
    actual_type: &ArrowDataType,
    expected_type: &ArrowDataType,
    schema_contract_failure: &'static str,
) -> Error {
    error::foreign_invariant_violation(
        "dataframe::GpuRecordBatch::to_host_record_batch",
        message,
        "record_batch",
        "invalid_arrow_schema_contract",
        "ensure materialized duration columns use the Arrow primitive array matching their unit",
    )
    .with_detail("actual_arrow_type", format!("{actual_type:?}"))
    .with_detail("expected_arrow_type", format!("{expected_type:?}"))
    .with_detail("schema_contract_failure", schema_contract_failure)
}

fn timestamp_array_with_timezone<T>(column: &dyn Array, timezone: Option<Arc<str>>) -> Result<PrimitiveArray<T>>
where T: arrow::datatypes::ArrowTimestampType {
    let array = column.as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        error::foreign_invariant_violation(
            "dataframe::GpuRecordBatch::to_host_record_batch",
            "timestamp column has an unexpected Arrow primitive layout",
            "record_batch",
            "invalid_arrow_schema_contract",
            "ensure materialized timestamp columns use the Arrow primitive array matching their unit",
        )
        .with_detail("actual_arrow_type", format!("{:?}", column.data_type()))
        .with_detail("expected_arrow_type", format!("{:?}", ArrowDataType::Timestamp(T::UNIT, timezone.clone())))
        .with_detail("schema_contract_failure", "timestamp_schema_timezone_layout_mismatch")
    })?;
    Ok(array.clone().with_timezone_opt(timezone))
}

fn map_arrow_schema_contract_error(operation: &'static str, error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        operation,
        error.to_string(),
        "record_batch",
        "invalid_arrow_schema_contract",
        "ensure materialized Arrow columns can be represented by the stored schema contract",
    )
    .with_source(error)
}

impl std::fmt::Debug for GpuRecordBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRecordBatch")
            .field("rows", &self.num_rows())
            .field("columns", &self.num_columns())
            .field("placement", &self.placement())
            .finish()
    }
}
