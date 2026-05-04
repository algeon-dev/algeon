use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use super::DataFrame;
use crate::{
    core::{
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        schema,
    },
    interop::{
        arrow_device::DeviceArrowTableExport,
        arrow_host::{ArrowHostSchema, validate_arrow_schema},
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
    /// Arrow-to-cudf logical dtypes must match the DataFrame schema. Arrow
    /// nullability remains external schema metadata; it is not folded into the
    /// cudf physical dtype.
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
    /// This delegates to [`DataFrame::to_record_batch`] and therefore copies
    /// device data to host memory and may synchronize before returning.
    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        self.frame.to_record_batch()
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
        if frame_dtype != schema_dtype {
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

impl std::fmt::Debug for GpuRecordBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRecordBatch")
            .field("rows", &self.num_rows())
            .field("columns", &self.num_columns())
            .field("placement", &self.placement())
            .finish()
    }
}
