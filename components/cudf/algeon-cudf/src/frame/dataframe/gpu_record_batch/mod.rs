//! GPU-resident record batch with an external Arrow schema contract.
use std::{fmt, sync::Arc};

use arrow::{
    datatypes::{DataType as ArrowDataType, SchemaRef},
    record_batch::RecordBatch,
};

use super::DataFrame;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, Result},
        types::TypeId,
    },
    interop::arrow_host::{ArrowHostSchema, estimate_arrow_host_record_batch_memory_bytes},
    ops::validation,
    runtime::{ExecutionPlacement, execution_domain::ExecutionDomain},
};

mod host_batch_cursor;
mod list_export;
mod list_validation;
mod schema_contract;

pub use host_batch_cursor::{HostBatchCursor, HostBatchCursorConfig, PreparedHostBatch};
use list_export::{FixedSizeListStructuralProof, ListExportContract, ListExportParts, UniformListProof};
pub use list_export::{ValidatedFixedSizeListExport, ValidatedUniformListExport};
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) use list_validation::{
    fixed_size_list_structural_proof_build_count_for_test, reset_fixed_size_list_structural_proof_build_count_for_test,
};
use list_validation::{rebind_fixed_size_list_structural_proof, validate_list_export, validate_uniform_list_offsets};
use schema_contract::{record_batch_with_schema_contract, validate_frame_schema};

fn validate_list_child_finite(
    child: &ColumnView<'_>,
    element: TypeId,
    operation: &'static str,
) -> Result<Option<validation::FiniteValidationProof>> {
    let actual = child.data_type().map_err(|error| error.reanchor(operation))?.id();
    if actual != element {
        return Err(Error::new_with(ErrorCode::TypeMismatch, operation, "list child differs from structural proof")
            .with_detail("actual_element", actual.name())
            .with_detail("proof_element", element.name()));
    }
    match element {
        TypeId::Float32 => validation::validate_finite_float32(child).map(Some),
        TypeId::Float64 => validation::validate_finite_float64(child).map(Some),
        TypeId::Int8 | TypeId::UInt8 => Ok(None),
        _ => Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "list child must be Float32, Float64, Int8, or UInt8",
        )
        .with_detail("supported_elements", "Float32, Float64, Int8, UInt8")),
    }
    .map_err(|error| error.reanchor(operation))
}

fn prepare_list_export<'a>(
    producer: &'a Arc<Column>,
    contract: ListExportContract,
    operation: &'static str,
) -> Result<(ListExportParts<'a>, Option<validation::FiniteValidationProof>)> {
    let export =
        producer.try_prepare_device_column_export().map_err(|error| error::from_interop_in(error, operation))?;
    // SAFETY: the column owns all projected buffers and its readiness stream.
    let export = unsafe { export.with_retained_owner(Arc::<Column>::clone(producer)) };
    let parts = validate_list_export(export, contract, operation)?;
    let child = producer
        .view()
        .map_err(|error| error.reanchor(operation))?
        .list_sliced_child()
        .map_err(|error| error.reanchor(operation))?;
    let finite_proof = validate_list_child_finite(&child, contract.element, operation)?;
    Ok((parts, finite_proof))
}

pub(super) struct RetainedProducerDomain(pub(super) Arc<ExecutionDomain>);

impl fmt::Debug for RetainedProducerDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedProducerDomain")
            .field("binding_id", &self.0.binding_id())
            .field("placement", &self.0.placement())
            .finish()
    }
}

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
    fixed_size_list_proofs: Arc<[FixedSizeListStructuralProof]>,
}

impl GpuRecordBatch {
    pub(super) fn estimated_materialization_bytes(&self) -> Result<usize> {
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
    /// Validation does not materialize column payload values. Field count,
    /// field names in order, and Arrow-to-cudf logical dtypes must be
    /// materializable from the DataFrame schema under the host-copy export
    /// contract. `FixedSizeList` additionally validates the structural list
    /// offsets so a variable-width cudf list cannot claim a fixed-width Arrow
    /// schema. Arrow nullability remains external schema metadata; it is not
    /// folded into the cudf physical dtype.
    pub fn new(frame: DataFrame, schema: SchemaRef) -> Result<Self> {
        let validated = validate_frame_schema(&frame, &schema)?;
        Ok(Self {
            frame,
            schema,
            mapped_schema: validated.mapped_schema,
            fixed_size_list_proofs: validated.fixed_size_list_proofs,
        })
    }

    pub(super) fn with_validated_slice(&self, frame: DataFrame) -> Result<Self> {
        debug_assert!(self.fixed_size_list_proofs.iter().all(|proof| proof.matches_schema(&self.schema)));
        let mapped_schema = self.mapped_schema.with_row_count(frame.num_rows());
        let fixed_size_list_proofs = self
            .fixed_size_list_proofs
            .iter()
            .map(|proof| rebind_fixed_size_list_structural_proof(*proof, &frame))
            .collect::<Result<Vec<_>>>()?
            .into();
        Ok(Self { frame, schema: Arc::clone(&self.schema), mapped_schema, fixed_size_list_proofs })
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

    /// Prepare a zero-copy device export for one `FixedSizeList<numeric>`
    /// field whose complete active range was structurally validated when this
    /// batch was created and finite-validated immediately before export.
    ///
    /// The returned witness is bound to the exact retained producer column,
    /// stream/memory-resource binding, semantic export graph, and active child
    /// range. Root or child nulls and non-finite child values are rejected
    /// because a dense vector consumer cannot represent them. Read access
    /// still requires activation through
    /// [`PreparedColumnExport::try_activate_read_lease`](rapids_interop::PreparedColumnExport::try_activate_read_lease).
    pub fn try_prepare_fixed_size_list_export(&self, field_index: usize) -> Result<ValidatedFixedSizeListExport<'_>> {
        let proof =
            self.fixed_size_list_proofs.iter().find(|proof| proof.field_index == field_index).copied().ok_or_else(
                || {
                    Error::new_with(
                        ErrorCode::UnsupportedDtype,
                        "dataframe::GpuRecordBatch::try_prepare_fixed_size_list_export",
                        "requested field is not a validated FixedSizeList",
                    )
                    .with_detail("field_index", field_index.to_string())
                    .with_detail("reason", "fixed_size_list_structural_proof_unavailable")
                },
            )?;
        self.try_prepare_fixed_size_list_export_with_proof(field_index, proof)
    }

    fn try_prepare_fixed_size_list_export_with_proof(
        &self,
        field_index: usize,
        proof: FixedSizeListStructuralProof,
    ) -> Result<ValidatedFixedSizeListExport<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_fixed_size_list_export";
        let (_, producer) = self.frame.columns_map().get_index(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "fixed-size-list field index is out of bounds")
                .with_detail("field_index", field_index.to_string())
                .with_detail("column_count", self.frame.num_columns().to_string())
        })?;
        if proof.field_index != field_index || !proof.matches_producer(producer) {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                OPERATION,
                "fixed-size-list structural proof does not belong to the requested producer column",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("proof_field_index", proof.field_index.to_string())
            .with_detail("proof_binding_id", proof.binding_id.to_string())
            .with_detail("producer_binding_id", producer.execution_domain().binding_id().to_string())
            .with_detail("reason", "fixed_size_list_proof_identity_mismatch"));
        }

        let (parts, finite_proof) = prepare_list_export(producer, proof.into(), OPERATION)?;
        if finite_proof.is_some_and(|finite_proof| {
            finite_proof.validated_values() != proof.child_len
                || finite_proof.binding_id() != proof.binding_id
                || finite_proof.placement() != proof.placement
        }) {
            return Err(error::invariant_violation(
                OPERATION,
                "FixedSizeList finite-value proof does not cover the structurally proven child range",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("structural_child_len", proof.child_len.to_string())
            .with_detail(
                "finite_validated_values",
                finite_proof.map(|proof| proof.validated_values()).unwrap_or(0).to_string(),
            )
            .with_detail("structural_binding_id", proof.binding_id.to_string())
            .with_detail(
                "finite_binding_id",
                finite_proof.map(|proof| proof.binding_id().to_string()).unwrap_or_default(),
            )
            .with_detail("reason", "fixed_size_list_finite_proof_mismatch"));
        }

        Ok(ValidatedFixedSizeListExport {
            export: parts.export,
            producer_domain: RetainedProducerDomain(Arc::clone(producer.execution_domain())),
            proof,
            root_column_id: parts.root_column_id,
            child_column_id: parts.child_column_id,
            child_values_buffer_id: parts.child_values_buffer_id,
        })
    }

    /// Prepare a zero-copy device export for a variable-width Arrow
    /// `List<numeric>` or `LargeList<numeric>` field whose active rows all
    /// have one uniform positive dimension.
    ///
    /// The dimension is derived from device-reduced active offset deltas; it
    /// is never supplied by the caller. Empty inputs are rejected because a
    /// variable-width schema cannot prove a dimension without any rows.
    /// Root/child nulls, non-monotonic or ragged offsets, zero-width rows, and
    /// non-finite child values are rejected before the proof-bearing witness
    /// is returned.
    pub fn try_prepare_uniform_list_export(&self, field_index: usize) -> Result<ValidatedUniformListExport<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_uniform_list_export";
        let field = self.schema.fields().get(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "list field index is out of bounds")
                .with_detail("field_index", field_index.to_string())
                .with_detail("field_count", self.schema.fields().len().to_string())
        })?;
        let supported_child = match field.data_type() {
            ArrowDataType::List(child) | ArrowDataType::LargeList(child) => {
                matches!(
                    child.data_type(),
                    ArrowDataType::Float32 | ArrowDataType::Float64 | ArrowDataType::Int8 | ArrowDataType::UInt8
                )
            },
            _ => false,
        };
        if !supported_child {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OPERATION,
                "uniform list export requires a variable-width list of Float32, Float64, Int8, or UInt8",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("arrow_type", format!("{:?}", field.data_type()))
            .with_detail("reason", "uniform_list_float32_schema_required"));
        }

        let (_, producer) = self.frame.columns_map().get_index(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "list producer index is out of bounds")
                .with_detail("field_index", field_index.to_string())
                .with_detail("column_count", self.frame.num_columns().to_string())
        })?;
        let view = producer.view().map_err(|error| error.reanchor(OPERATION))?;
        let proof = validate_uniform_list_offsets(&view, producer, field_index, OPERATION)?;
        self.try_prepare_uniform_list_export_with_proof(field_index, proof)
    }

    fn try_prepare_uniform_list_export_with_proof(
        &self,
        field_index: usize,
        proof: UniformListProof,
    ) -> Result<ValidatedUniformListExport<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_uniform_list_export";
        let (_, producer) = self.frame.columns_map().get_index(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "list producer index is out of bounds")
                .with_detail("field_index", field_index.to_string())
                .with_detail("column_count", self.frame.num_columns().to_string())
        })?;
        if proof.field_index != field_index || !proof.matches_producer(producer) {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                OPERATION,
                "uniform-list proof does not belong to the requested producer column",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("proof_field_index", proof.field_index.to_string())
            .with_detail("proof_binding_id", proof.binding_id.to_string())
            .with_detail("producer_binding_id", producer.execution_domain().binding_id().to_string())
            .with_detail("reason", "uniform_list_proof_identity_mismatch"));
        }

        let (parts, finite_proof) = prepare_list_export(producer, proof.into(), OPERATION)?;
        if finite_proof.is_some_and(|finite_proof| {
            finite_proof.validated_values() != proof.child_len
                || finite_proof.binding_id() != proof.binding_id
                || finite_proof.placement() != proof.placement
        }) {
            return Err(error::invariant_violation(
                OPERATION,
                "uniform-list finite proof does not cover the offset-proven active child range",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("offset_proof_child_len", proof.child_len.to_string())
            .with_detail(
                "finite_validated_values",
                finite_proof.map(|proof| proof.validated_values()).unwrap_or(0).to_string(),
            )
            .with_detail("offset_proof_binding_id", proof.binding_id.to_string())
            .with_detail(
                "finite_binding_id",
                finite_proof.map(|proof| proof.binding_id().to_string()).unwrap_or_default(),
            )
            .with_detail("reason", "uniform_list_finite_proof_mismatch"));
        }

        Ok(ValidatedUniformListExport {
            export: parts.export,
            producer_domain: RetainedProducerDomain(Arc::clone(producer.execution_domain())),
            proof,
            root_column_id: parts.root_column_id,
            offsets_buffer_id: parts.offsets_buffer_id,
            child_column_id: parts.child_column_id,
            child_values_buffer_id: parts.child_values_buffer_id,
        })
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn trigger_uniform_list_proof_identity_mismatch_for_test(
        &self,
        proof_owner: &Self,
        field_index: usize,
    ) -> Result<()> {
        const OPERATION: &str = "testing::trigger_uniform_list_proof_identity_mismatch";
        let (_, producer) = proof_owner.frame.columns_map().get_index(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "proof-owner list field index is out of bounds")
                .with_detail("field_index", field_index.to_string())
        })?;
        let view = producer.view().map_err(|error| error.reanchor(OPERATION))?;
        let proof = validate_uniform_list_offsets(&view, producer, field_index, OPERATION)?;
        self.try_prepare_uniform_list_export_with_proof(field_index, proof).map(drop)
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn trigger_fixed_size_list_proof_identity_mismatch_for_test(
        &self,
        proof_owner: &Self,
        field_index: usize,
    ) -> Result<()> {
        let proof = proof_owner
            .fixed_size_list_proofs
            .iter()
            .find(|proof| proof.field_index == field_index)
            .copied()
            .ok_or_else(|| {
            Error::new_with(
                ErrorCode::UnsupportedDtype,
                "testing::trigger_fixed_size_list_proof_identity_mismatch",
                "proof owner does not contain the requested FixedSizeList proof",
            )
        })?;
        self.try_prepare_fixed_size_list_export_with_proof(field_index, proof).map(drop)
    }

    /// Materialize this GPU batch as a host Arrow RecordBatch.
    ///
    /// This delegates to [`DataFrame::to_record_batch`] for the device-to-host
    /// copy, then attaches this batch's stored Arrow schema contract. When
    /// cudf's host export type is a supported physical equivalent of the
    /// stored Arrow field type, the host array is cast before the final
    /// `RecordBatch` is built.
    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        let batch = self.frame.to_record_batch_with_arrow_schema(&self.mapped_schema)?;
        record_batch_with_schema_contract(batch, &self.schema)
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

impl std::fmt::Debug for GpuRecordBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRecordBatch")
            .field("rows", &self.num_rows())
            .field("columns", &self.num_columns())
            .field("placement", &self.placement())
            .finish()
    }
}
