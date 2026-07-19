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
        arrow_host::{
            ArrowHostSchema, cudf_dtype_can_export_to_arrow_type, estimate_arrow_host_record_batch_memory_bytes,
            validate_arrow_schema,
        },
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

/// Bounds one GPU-to-host materialization step by estimated final Arrow buffer
/// capacity and row count.
///
/// `target_materialization_bytes` is the soft chunking target. A row that
/// cannot be split may exceed that target, but it must not exceed
/// `max_materialization_bytes`, the hard pre-copy ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostBatchCursorConfig {
    pub target_materialization_bytes: usize,
    pub max_materialization_bytes: usize,
    pub max_rows: usize,
}

impl HostBatchCursorConfig {
    pub const DEFAULT_TARGET_MATERIALIZATION_BYTES: usize = 2 * 1024 * 1024;
    pub const DEFAULT_MAX_MATERIALIZATION_BYTES: usize = 4 * 1024 * 1024;
    pub const DEFAULT_MAX_ROWS: usize = 16 * 1024;
}

impl Default for HostBatchCursorConfig {
    fn default() -> Self {
        Self {
            target_materialization_bytes: Self::DEFAULT_TARGET_MATERIALIZATION_BYTES,
            max_materialization_bytes: Self::DEFAULT_MAX_MATERIALIZATION_BYTES,
            max_rows: Self::DEFAULT_MAX_ROWS,
        }
    }
}

/// A metadata-sized GPU slice ready for one bounded host copy.
///
/// Preparing a step performs no device-to-host value materialization. The
/// token keeps the exact sliced GPU candidate alive so a caller can reserve
/// host capacity before asking the originating cursor to copy it.
#[derive(Debug)]
pub struct PreparedHostBatch {
    batch: GpuRecordBatch,
    cursor_identity: Arc<()>,
    offset: usize,
    batch_index: usize,
    estimated_materialization_bytes: usize,
}

impl PreparedHostBatch {
    /// Conservative final Arrow buffer capacity that must be reserved before
    /// the host copy starts.
    pub fn estimated_materialization_bytes(&self) -> usize {
        self.estimated_materialization_bytes
    }

    /// Number of rows in this prepared chunk.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

/// Incremental GPU-to-host Arrow materializer.
///
/// The cursor retains the original GPU batch and at most one sliced GPU chunk.
/// An initially empty GPU batch still yields exactly one schema-preserving host
/// batch before reaching EOF, matching a full result materialization boundary.
/// It uses a schema-aware upper bound for final Arrow buffer capacities to
/// binary-reduce a candidate before copying it to host. A single row may exceed
/// the soft target, but an unsplittable row that exceeds the hard
/// materialization ceiling is rejected with a permanent structured resource
/// error before any host value copy.
pub struct HostBatchCursor {
    batch: GpuRecordBatch,
    config: HostBatchCursorConfig,
    offset: usize,
    emitted_batches: usize,
    identity: Arc<()>,
}

impl HostBatchCursor {
    pub fn new(batch: GpuRecordBatch, config: HostBatchCursorConfig) -> Result<Self> {
        if config.target_materialization_bytes == 0 || config.max_materialization_bytes == 0 || config.max_rows == 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::new",
                "host batch cursor bounds must be non-zero",
            )
            .with_detail("target_materialization_bytes", config.target_materialization_bytes.to_string())
            .with_detail("max_materialization_bytes", config.max_materialization_bytes.to_string())
            .with_detail("max_rows", config.max_rows.to_string()));
        }
        if config.target_materialization_bytes > config.max_materialization_bytes {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::new",
                "host batch cursor soft materialization target must not exceed its hard materialization ceiling",
            )
            .with_detail("target_materialization_bytes", config.target_materialization_bytes.to_string())
            .with_detail("max_materialization_bytes", config.max_materialization_bytes.to_string()));
        }
        Ok(Self { batch, config, offset: 0, emitted_batches: 0, identity: Arc::new(()) })
    }

    pub fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    pub fn remaining_rows(&self) -> usize {
        self.batch.num_rows().saturating_sub(self.offset)
    }

    /// Whether the cursor can still yield a host batch.
    ///
    /// This differs from `remaining_rows() > 0` for an initially empty result,
    /// which has no rows but still owns one typed materialization boundary.
    pub fn has_next_batch(&self) -> bool {
        self.remaining_rows() > 0 || (self.batch.num_rows() == 0 && self.emitted_batches == 0)
    }

    /// Select and size the next GPU chunk without copying values to host.
    pub fn prepare_next_batch(&self) -> Result<Option<PreparedHostBatch>> {
        let remaining = self.remaining_rows();
        let initial_empty_batch = self.batch.num_rows() == 0 && self.emitted_batches == 0;
        if !self.has_next_batch() {
            return Ok(None);
        }
        let candidate_rows = if initial_empty_batch { 0 } else { remaining.min(self.config.max_rows) };
        let mut candidate = self.candidate(candidate_rows)?;
        let mut estimated_bytes = candidate.estimated_materialization_bytes()?;
        if candidate_rows > 1 && estimated_bytes > self.config.target_materialization_bytes {
            drop(candidate);
            let mut low = 1usize;
            let mut high = candidate_rows - 1;
            let mut best_rows = 1usize;
            while low <= high {
                let middle = low + (high - low) / 2;
                let probe = self.candidate(middle)?;
                let estimated_bytes = probe.estimated_materialization_bytes()?;
                drop(probe);
                if estimated_bytes <= self.config.target_materialization_bytes {
                    best_rows = middle;
                    low = middle + 1;
                } else {
                    high = middle - 1;
                }
            }
            candidate = self.candidate(best_rows)?;
            estimated_bytes = candidate.estimated_materialization_bytes()?;
        }

        if estimated_bytes > self.config.max_materialization_bytes {
            return Err(Error::new_with(
                ErrorCode::HostMaterializationLimitExceeded,
                "dataframe::HostBatchCursor::prepare_next_batch",
                "unsplittable host output row exceeds the hard materialization ceiling",
            )
            .with_detail("reason", "host_materialization_hard_limit_exceeded")
            .with_detail("row_offset", self.offset.to_string())
            .with_detail("candidate_rows", candidate.num_rows().to_string())
            .with_detail("estimated_materialization_bytes", estimated_bytes.to_string())
            .with_detail("max_materialization_bytes", self.config.max_materialization_bytes.to_string()));
        }

        Ok(Some(PreparedHostBatch {
            batch: candidate,
            cursor_identity: Arc::clone(&self.identity),
            offset: self.offset,
            batch_index: self.emitted_batches,
            estimated_materialization_bytes: estimated_bytes,
        }))
    }

    /// Copy one previously prepared chunk to host and advance the cursor.
    pub fn materialize_prepared_batch(&mut self, prepared: PreparedHostBatch) -> Result<RecordBatch> {
        if !Arc::ptr_eq(&self.identity, &prepared.cursor_identity)
            || self.offset != prepared.offset
            || self.emitted_batches != prepared.batch_index
        {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "prepared host batch does not belong to the cursor's current position",
            )
            .with_detail("reason", "stale_or_foreign_prepared_host_batch")
            .with_detail("cursor_offset", self.offset.to_string())
            .with_detail("prepared_offset", prepared.offset.to_string())
            .with_detail("cursor_batch_index", self.emitted_batches.to_string())
            .with_detail("prepared_batch_index", prepared.batch_index.to_string()));
        }

        let rows = prepared.batch.num_rows();
        let next_offset = self.offset.checked_add(rows).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "host batch cursor row offset overflowed usize",
            )
            .with_detail("cursor_offset", self.offset.to_string())
            .with_detail("materialized_rows", rows.to_string())
        })?;
        let next_batch_index = self.emitted_batches.checked_add(1).ok_or_else(|| {
            Error::new_with(
                ErrorCode::SizeOverflow,
                "dataframe::HostBatchCursor::materialize_prepared_batch",
                "host batch cursor batch index overflowed usize",
            )
            .with_detail("batch_index", self.emitted_batches.to_string())
        })?;
        let batch = prepared.batch.to_host_record_batch()?;
        self.offset = next_offset;
        self.emitted_batches = next_batch_index;
        Ok(batch)
    }

    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(prepared) = self.prepare_next_batch()? else {
            return Ok(None);
        };
        self.materialize_prepared_batch(prepared).map(Some)
    }

    fn candidate(&self, rows: usize) -> Result<GpuRecordBatch> {
        if rows == 0 {
            debug_assert_eq!(self.batch.num_rows(), 0);
            return Ok(self.batch.clone());
        }
        let frame = self
            .batch
            .frame()
            .slice(self.offset, rows)
            .map_err(|error| error.reanchor("dataframe::HostBatchCursor::next_batch"))?;
        GpuRecordBatch::new(frame, Arc::clone(self.batch.schema()))
            .map_err(|error| error.reanchor("dataframe::HostBatchCursor::next_batch"))
    }
}

impl GpuRecordBatch {
    fn estimated_materialization_bytes(&self) -> Result<usize> {
        estimate_arrow_host_record_batch_memory_bytes(&self.frame, &self.mapped_schema)
            .map_err(|error| error.reanchor("dataframe::HostBatchCursor::prepare_next_batch"))
    }

    /// Consume this GPU batch into a bounded incremental host materializer.
    pub fn into_host_batch_cursor(self, config: HostBatchCursorConfig) -> Result<HostBatchCursor> {
        HostBatchCursor::new(self, config)
    }
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
