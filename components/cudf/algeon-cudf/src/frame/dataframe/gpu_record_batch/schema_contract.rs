//! Arrow schema contract validation and host record batch assembly.
use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, PrimitiveArray},
    compute::cast,
    datatypes::{
        ArrowPrimitiveType, ArrowTimestampType, DataType as ArrowDataType, DurationMicrosecondType,
        DurationMillisecondType, DurationNanosecondType, DurationSecondType, SchemaRef, Time32MillisecondType,
        Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
    },
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchOptions},
};

use super::{
    super::DataFrame,
    list_export::FixedSizeListStructuralProof,
    list_validation::{validate_fixed_size_list_offsets, validate_list_schema_contract},
};
use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        schema,
        types::TypeId,
    },
    internal::resolve::missing_column_error,
    interop::arrow_host::{ArrowHostSchema, cudf_dtype_can_export_to_arrow_type, validate_arrow_schema},
};

pub(super) struct ValidatedFrameSchema {
    pub(super) mapped_schema: ArrowHostSchema,
    pub(super) fixed_size_list_proofs: Arc<[FixedSizeListStructuralProof]>,
}

pub(super) fn validate_frame_schema(frame: &DataFrame, schema_ref: &SchemaRef) -> Result<ValidatedFrameSchema> {
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
    let mut fixed_size_list_proofs = Vec::new();

    for (index, (field, frame_name)) in mapped.fields().iter().zip(frame_names.iter()).enumerate() {
        if field.name() != *frame_name {
            return Err(missing_column_error(operation, field.name())
                .with_detail("field_name", field.name().to_string())
                .with_detail("expected_column_name", (*frame_name).to_string())
                .with_detail("field_index", index.to_string())
                .with_detail("reason", "schema_name_mismatch"));
        }

        let series = frame.series(frame_name)?;
        let frame_dtype = series.dtype().map_err(|error| error.reanchor_if_needed(operation))?;
        let schema_dtype = field.cudf_data_type();
        let can_export = match field.arrow_data_type() {
            ArrowDataType::List(child) | ArrowDataType::LargeList(child) if frame_dtype.id() == TypeId::List => {
                validate_list_schema_contract(
                    &series.as_column().view().map_err(|error| error.reanchor_if_needed(operation))?,
                    child,
                    field.name(),
                    index,
                    operation,
                )?;
                true
            },
            ArrowDataType::FixedSizeList(child, dimension) if frame_dtype.id() == TypeId::List => {
                let producer = series.arc();
                let view = producer.view().map_err(|error| error.reanchor_if_needed(operation))?;
                validate_list_schema_contract(&view, child, field.name(), index, operation)?;
                let proof =
                    validate_fixed_size_list_offsets(&view, &producer, *dimension, field.name(), index, operation)?;
                fixed_size_list_proofs.push(proof);
                true
            },
            _ => cudf_dtype_can_export_to_arrow_type(frame_dtype, field.arrow_data_type()),
        };
        if !can_export {
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

    Ok(ValidatedFrameSchema { mapped_schema: mapped, fixed_size_list_proofs: fixed_size_list_proofs.into() })
}

pub(super) fn record_batch_with_schema_contract(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
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
                return Ok(Arc::clone(column));
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
    RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options).map_err(|error| {
        map_arrow_schema_contract_error(operation, error)
            .with_detail("schema_contract_failure", "schema_record_batch_build_failed")
    })
}

fn timestamp_array_with_schema_timezone(column: &dyn Array, expected_type: &ArrowDataType) -> Result<Option<ArrayRef>> {
    let (actual_unit, expected_unit, expected_timezone) = match (column.data_type(), expected_type) {
        (
            ArrowDataType::Timestamp(actual_unit, _),
            ArrowDataType::Timestamp(expected_unit, Some(expected_timezone)),
        ) if actual_unit == expected_unit => (actual_unit, expected_unit, Arc::clone(expected_timezone)),
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
where D: ArrowPrimitiveType<Native = i64> {
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
where D: ArrowPrimitiveType<Native = i64> {
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
where T: ArrowTimestampType {
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

fn map_arrow_schema_contract_error(operation: &'static str, error: ArrowError) -> Error {
    error::foreign_invariant_violation(
        operation,
        error.to_string(),
        "record_batch",
        "invalid_arrow_schema_contract",
        "ensure materialized Arrow columns can be represented by the stored schema contract",
    )
    .with_source(error)
}
