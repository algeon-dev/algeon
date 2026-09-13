//! List-of-Float32 export proofs, contracts, and child pack helpers.
use std::sync::Arc;

use arrow::datatypes::{DataType as ArrowDataType, SchemaRef};
use rapids_interop::{
    BufferBinding, BufferId, ColumnId, CompletionAuthority, PreparedColumnExport, ReadStartAuthority,
};

use super::RetainedProducerDomain;
use crate::{
    core::{
        Column,
        column::copy_view_to_owned,
        error::{self, Error, ErrorCode, Result},
        types::TypeId,
    },
    ops::validation::FiniteFloat32ValidationProof,
    runtime::{
        BoundaryObservation, DenseMatrixLayout, ExecutionBindingId, ExecutionPlacement, PackedFloat32Matrix, Runtime,
        execution_domain::ExecutionDomain,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FixedSizeListStructuralProof {
    pub(super) field_index: usize,
    pub(super) dimension: i32,
    pub(super) rows: usize,
    pub(super) child_offset: usize,
    pub(super) child_len: usize,
    pub(super) producer_column_identity: usize,
    pub(super) binding_id: ExecutionBindingId,
    pub(super) placement: ExecutionPlacement,
    pub(super) observation: Option<BoundaryObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct UniformListFloat32Proof {
    pub(super) field_index: usize,
    pub(super) dimension: usize,
    pub(super) rows: usize,
    pub(super) root_offset: usize,
    pub(super) child_offset: usize,
    pub(super) child_len: usize,
    pub(super) producer_column_identity: usize,
    pub(super) binding_id: ExecutionBindingId,
    pub(super) placement: ExecutionPlacement,
    pub(super) offsets_observation: BoundaryObservation,
}

impl UniformListFloat32Proof {
    pub(super) fn matches_producer(self, producer: &Arc<crate::core::Column>) -> bool {
        self.producer_column_identity == Arc::as_ptr(producer) as usize
            && self.binding_id == producer.execution_domain().binding_id()
            && self.placement == producer.execution_domain().placement()
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ListFloat32ExportContract {
    pub(super) field_index: usize,
    pub(super) dimension: usize,
    pub(super) rows: usize,
    pub(super) root_offset: Option<usize>,
    pub(super) child_offset: usize,
    pub(super) child_len: usize,
    pub(super) placement: ExecutionPlacement,
}

impl From<FixedSizeListStructuralProof> for ListFloat32ExportContract {
    fn from(proof: FixedSizeListStructuralProof) -> Self {
        Self {
            field_index: proof.field_index,
            dimension: proof.dimension as usize,
            rows: proof.rows,
            root_offset: None,
            child_offset: proof.child_offset,
            child_len: proof.child_len,
            placement: proof.placement,
        }
    }
}

impl From<UniformListFloat32Proof> for ListFloat32ExportContract {
    fn from(proof: UniformListFloat32Proof) -> Self {
        Self {
            field_index: proof.field_index,
            dimension: proof.dimension,
            rows: proof.rows,
            root_offset: Some(proof.root_offset),
            child_offset: proof.child_offset,
            child_len: proof.child_len,
            placement: proof.placement,
        }
    }
}

impl FixedSizeListStructuralProof {
    pub(super) fn matches_schema(self, schema: &SchemaRef) -> bool {
        matches!(
            schema.fields().get(self.field_index).map(|field| field.data_type()),
            Some(ArrowDataType::FixedSizeList(_, dimension)) if *dimension == self.dimension
        )
    }

    pub(super) fn matches_producer(self, producer: &Arc<crate::core::Column>) -> bool {
        self.producer_column_identity == Arc::as_ptr(producer) as usize
            && self.binding_id == producer.execution_domain().binding_id()
            && self.placement == producer.execution_domain().placement()
    }
}

/// A device prepared export whose active `LIST<Float32>` range is proven to
/// have one fixed-size child span per row and only finite child values.
///
/// The witness retains the ordinary [`PreparedColumnExport`] lease root while
/// binding the structural proof to the exact producer column, execution
/// binding, and active child range. Consumers must still activate and complete
/// the prepared export's read lease before using the child values buffer.
#[derive(Debug)]
pub struct ValidatedFixedSizeListFloat32Export<'a> {
    pub(super) export: PreparedColumnExport<'a>,
    pub(super) producer_domain: RetainedProducerDomain,
    pub(super) proof: FixedSizeListStructuralProof,
    pub(super) finite_proof: FiniteFloat32ValidationProof,
    pub(super) root_column_id: ColumnId,
    pub(super) child_column_id: ColumnId,
    pub(super) child_values_buffer_id: BufferId,
}

#[derive(Debug)]
pub(super) struct ListFloat32ExportParts<'a> {
    pub(super) export: PreparedColumnExport<'a>,
    pub(super) root_column_id: ColumnId,
    pub(super) offsets_buffer_id: BufferId,
    pub(super) child_column_id: ColumnId,
    pub(super) child_values_buffer_id: BufferId,
}

/// A device prepared export whose active variable-width `List<Float32>` range
/// was proven to have one uniform positive dimension and only finite values.
///
/// The proof is bound to the exact retained producer, root offset, semantic
/// offsets/child identities, active child range, execution binding, and
/// readiness stream. The ordinary prepared-export lease root remains
/// retained and must be activated before a consumer reads the child buffer.
#[derive(Debug)]
pub struct ValidatedUniformListFloat32Export<'a> {
    pub(super) export: PreparedColumnExport<'a>,
    pub(super) producer_domain: RetainedProducerDomain,
    pub(super) proof: UniformListFloat32Proof,
    pub(super) finite_proof: FiniteFloat32ValidationProof,
    pub(super) root_column_id: ColumnId,
    pub(super) offsets_buffer_id: BufferId,
    pub(super) child_column_id: ColumnId,
    pub(super) child_values_buffer_id: BufferId,
}

impl ValidatedUniformListFloat32Export<'_> {
    pub fn export(&self) -> &PreparedColumnExport<'_> {
        &self.export
    }

    pub fn field_index(&self) -> usize {
        self.proof.field_index
    }

    pub fn dimension(&self) -> usize {
        self.proof.dimension
    }

    pub fn rows(&self) -> usize {
        self.proof.rows
    }

    pub fn root_offset(&self) -> usize {
        self.proof.root_offset
    }

    pub fn child_offset(&self) -> usize {
        self.proof.child_offset
    }

    pub fn child_len(&self) -> usize {
        self.finite_proof.validated_values()
    }

    pub fn child_byte_offset(&self) -> usize {
        self.proof.child_offset * std::mem::size_of::<f32>()
    }

    pub fn binding_id(&self) -> ExecutionBindingId {
        self.finite_proof.binding_id()
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.finite_proof.placement()
    }

    pub fn offsets_observation(&self) -> BoundaryObservation {
        self.proof.offsets_observation
    }

    pub fn finite_observation(&self) -> BoundaryObservation {
        self.finite_proof.observation()
    }

    pub fn root_column_id(&self) -> ColumnId {
        self.root_column_id
    }

    pub fn offsets_buffer_id(&self) -> BufferId {
        self.offsets_buffer_id
    }

    pub fn child_column_id(&self) -> ColumnId {
        self.child_column_id
    }

    pub fn child_values_buffer_id(&self) -> BufferId {
        self.child_values_buffer_id
    }

    pub fn child_values_binding(&self) -> &BufferBinding<'_> {
        self.export
            .bindings()
            .iter()
            .find(|binding| binding.id() == self.child_values_buffer_id)
            .expect("validated uniform-list child binding must remain present")
    }

    /// Copy the exact proof-covered Float32 child range into an owned column
    /// on `runtime` without materializing vector payloads on the host.
    ///
    /// This is the owned-packing path for consumers such as mutable PCA input.
    /// Callers that only need a read-only tensor should instead activate the
    /// retained prepared export directly.
    pub fn copy_active_child_into_runtime(&self, runtime: &Runtime) -> Result<Column> {
        copy_validated_list_float32_child_into_runtime(
            self.export(),
            self.rows(),
            self.dimension(),
            self.child_len(),
            runtime,
            "dataframe::ValidatedUniformListFloat32Export::copy_active_child_into_runtime",
        )
    }

    /// Pack the exact proof-covered child range directly into one owned matrix
    /// allocation on `runtime`.
    ///
    /// Unlike composing [`copy_active_child_into_runtime`](Self::copy_active_child_into_runtime)
    /// with [`Runtime::pack_flat_float32_column`], this activates the retained
    /// prepared read once and performs only the final payload copy/reorder.
    pub fn pack_active_child_into_runtime(
        &self,
        runtime: &Runtime,
        layout: DenseMatrixLayout,
    ) -> Result<PackedFloat32Matrix> {
        pack_validated_list_float32_child_into_runtime(
            self.export(),
            &self.producer_domain.0,
            self.child_values_binding(),
            self.child_byte_offset(),
            self.rows(),
            self.dimension(),
            self.child_len(),
            runtime,
            layout,
            "dataframe::ValidatedUniformListFloat32Export::pack_active_child_into_runtime",
        )
    }
}

impl ValidatedFixedSizeListFloat32Export<'_> {
    /// The retained producer-rooted export used to activate read access.
    pub fn export(&self) -> &PreparedColumnExport<'_> {
        &self.export
    }

    /// Positional field index in the retained [`GpuRecordBatch`](super::GpuRecordBatch) schema.
    pub fn field_index(&self) -> usize {
        self.proof.field_index
    }

    /// Number of Float32 values in every logical list row.
    pub fn dimension(&self) -> usize {
        self.proof.dimension as usize
    }

    /// Number of active logical list rows.
    pub fn rows(&self) -> usize {
        self.proof.rows
    }

    /// Active child offset in Float32 elements from the values binding base.
    pub fn child_offset(&self) -> usize {
        self.proof.child_offset
    }

    /// Number of active Float32 child values.
    pub fn child_len(&self) -> usize {
        self.finite_proof.validated_values()
    }

    /// Active child byte offset from the values binding base.
    pub fn child_byte_offset(&self) -> usize {
        self.proof.child_offset * std::mem::size_of::<f32>()
    }

    /// Exact producer stream/memory-resource binding covered by the proof.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.finite_proof.binding_id()
    }

    /// Exact producer placement covered by the proof.
    pub fn placement(&self) -> ExecutionPlacement {
        self.finite_proof.placement()
    }

    /// Bounded control-scalar evidence used to build the structural proof.
    /// Empty inputs require no scalar read and therefore return `None`.
    pub fn structural_observation(&self) -> Option<BoundaryObservation> {
        self.proof.observation
    }

    /// Bounded control-scalar evidence that the exact active Float32 child
    /// range contains only finite values.
    pub fn finite_observation(&self) -> BoundaryObservation {
        self.finite_proof.observation()
    }

    /// Semantic root list-column identity retained by the prepared export.
    pub fn root_column_id(&self) -> ColumnId {
        self.root_column_id
    }

    /// Semantic Float32 child-column identity retained by the prepared export.
    pub fn child_column_id(&self) -> ColumnId {
        self.child_column_id
    }

    /// Symbolic binding identity for the child Float32 values allocation.
    pub fn child_values_buffer_id(&self) -> BufferId {
        self.child_values_buffer_id
    }

    /// Child values binding covered by this structural and finite witness.
    pub fn child_values_binding(&self) -> &BufferBinding<'_> {
        self.export
            .bindings()
            .iter()
            .find(|binding| binding.id() == self.child_values_buffer_id)
            .expect("validated fixed-size-list child binding must remain present")
    }

    /// Copy the exact proof-covered Float32 child range into an owned column
    /// on `runtime` without materializing vector payloads on the host.
    ///
    /// This is the owned-packing path for consumers such as mutable PCA input.
    /// Callers that only need a read-only tensor should instead activate the
    /// retained prepared export directly.
    pub fn copy_active_child_into_runtime(&self, runtime: &Runtime) -> Result<Column> {
        copy_validated_list_float32_child_into_runtime(
            self.export(),
            self.rows(),
            self.dimension(),
            self.child_len(),
            runtime,
            "dataframe::ValidatedFixedSizeListFloat32Export::copy_active_child_into_runtime",
        )
    }

    /// Pack the exact proof-covered child range directly into one owned matrix
    /// allocation on `runtime`.
    ///
    /// Unlike composing [`copy_active_child_into_runtime`](Self::copy_active_child_into_runtime)
    /// with [`Runtime::pack_flat_float32_column`], this activates the retained
    /// prepared read once and performs only the final payload copy/reorder.
    pub fn pack_active_child_into_runtime(
        &self,
        runtime: &Runtime,
        layout: DenseMatrixLayout,
    ) -> Result<PackedFloat32Matrix> {
        pack_validated_list_float32_child_into_runtime(
            self.export(),
            &self.producer_domain.0,
            self.child_values_binding(),
            self.child_byte_offset(),
            self.rows(),
            self.dimension(),
            self.child_len(),
            runtime,
            layout,
            "dataframe::ValidatedFixedSizeListFloat32Export::pack_active_child_into_runtime",
        )
    }
}

fn pack_validated_list_float32_child_into_runtime(
    export: &PreparedColumnExport<'_>,
    producer_domain: &Arc<ExecutionDomain>,
    child_values: &BufferBinding<'_>,
    child_byte_offset: usize,
    rows: usize,
    dimension: usize,
    child_len: usize,
    runtime: &Runtime,
    layout: DenseMatrixLayout,
    operation: &'static str,
) -> Result<PackedFloat32Matrix> {
    let expected_child_len = rows.checked_mul(dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "validated list matrix shape overflowed usize")
            .with_detail("rows", rows.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "validated_list_matrix_shape_overflow")
    })?;
    if child_len != expected_child_len {
        return Err(error::invariant_violation(
            operation,
            "validated list child length does not match its proof-covered matrix shape",
        )
        .with_detail("rows", rows.to_string())
        .with_detail("dimension", dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("validated_child_len", child_len.to_string())
        .with_detail("reason", "validated_list_matrix_shape_mismatch"));
    }

    let child_bytes = child_len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "validated list child byte length overflowed usize")
            .with_detail("validated_child_len", child_len.to_string())
            .with_detail("reason", "validated_list_child_byte_length_overflow")
    })?;
    let child_byte_end = child_byte_offset.checked_add(child_bytes).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "validated list child byte range overflowed usize")
            .with_detail("child_byte_offset", child_byte_offset.to_string())
            .with_detail("child_bytes", child_bytes.to_string())
            .with_detail("reason", "validated_list_child_byte_range_overflow")
    })?;
    if child_byte_end > child_values.bytes().byte_len() {
        return Err(error::invariant_violation(
            operation,
            "validated list child range exceeds its retained values binding",
        )
        .with_detail("child_byte_end", child_byte_end.to_string())
        .with_detail("binding_bytes", child_values.bytes().byte_len().to_string())
        .with_detail("reason", "validated_list_child_binding_too_small"));
    }

    let runtime_device = runtime.device().ordinal() as i32;
    if !matches!(
        child_values.placement(),
        rapids_interop::Placement::CudaDevice { device } if *device == runtime_device
    ) {
        return Err(Error::new_with(
            ErrorCode::RuntimeBindingMismatch,
            operation,
            "validated list input and destination runtime must use the same CUDA device",
        )
        .with_detail("runtime_device", runtime_device.to_string())
        .with_detail("reason", "validated_list_pack_device_mismatch"));
    }

    let child_address = (child_values.bytes().base_ptr() as usize).checked_add(child_byte_offset).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "validated list child device address overflowed usize")
            .with_detail("child_byte_offset", child_byte_offset.to_string())
            .with_detail("reason", "validated_list_child_address_overflow")
    })?;
    let stream = crate::interop::cuda_stream_handle(runtime.stream())
        .map_err(|error| error::from_interop_in(error, operation))?;
    let lease = export
        .try_activate_read_lease(
            ReadStartAuthority::ConsumerStream { stream },
            CompletionAuthority::SameDeviceStream { stream },
        )
        .map_err(|error| error::from_interop_in(error, operation))?;

    // SAFETY: the validated export binds this exact active child range to a
    // retained device allocation. The active lease orders producer writes
    // before `runtime` and retains the allocation through completion below.
    let packed = unsafe {
        runtime.pack_borrowed_flat_float32_device_values(
            child_address as *const std::ffi::c_void,
            producer_domain,
            rows,
            dimension,
            layout,
            operation,
        )
    }?;
    // The runtime's resolved execution domain has already installed the
    // reverse consumer-completion fence on the producer domain. Completing
    // the prepared lease records its stream witness, but no host wait is
    // needed before returning the independently owned packed allocation.
    let _completion = lease.complete().map_err(|error| error::from_interop_in(error, operation))?;
    Ok(packed)
}

fn copy_validated_list_float32_child_into_runtime(
    export: &PreparedColumnExport<'_>,
    rows: usize,
    dimension: usize,
    child_len: usize,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<Column> {
    let expected_child_len = rows.checked_mul(dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "validated list matrix shape overflowed usize")
            .with_detail("rows", rows.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "validated_list_matrix_shape_overflow")
    })?;
    if child_len != expected_child_len {
        return Err(error::invariant_violation(
            operation,
            "validated list child length does not match its proof-covered matrix shape",
        )
        .with_detail("rows", rows.to_string())
        .with_detail("dimension", dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("validated_child_len", child_len.to_string())
        .with_detail("reason", "validated_list_matrix_shape_mismatch"));
    }

    let root = Column::try_import_from_prepared_column_export_into_runtime(export, runtime)
        .map_err(|error| error::from_interop_in(error, operation))?;
    let child = root
        .view()
        .map_err(|error| error.reanchor(operation))?
        .list_sliced_child()
        .map_err(|error| error.reanchor(operation))?;
    let child_type = child.data_type().map_err(|error| error.reanchor(operation))?;
    if child_type.id() != TypeId::Float32 {
        return Err(error::invariant_violation(
            operation,
            "validated list import did not retain its Float32 child dtype",
        )
        .with_detail("actual_child_type", format!("{:?}", child_type.id()))
        .with_detail("reason", "validated_list_child_type_mismatch"));
    }
    let imported_child_len = child.len().map_err(|error| error.reanchor(operation))?;
    if imported_child_len != child_len {
        return Err(error::invariant_violation(
            operation,
            "validated list import did not retain the proof-covered child range",
        )
        .with_detail("expected_child_len", child_len.to_string())
        .with_detail("actual_child_len", imported_child_len.to_string())
        .with_detail("reason", "validated_list_child_range_mismatch"));
    }
    if child.null_count().map_err(|error| error.reanchor(operation))? != 0 {
        return Err(error::invariant_violation(
            operation,
            "validated list import introduced null Float32 child values",
        )
        .with_detail("reason", "validated_list_child_null_mismatch"));
    }
    copy_view_to_owned(&child).map_err(|error| error.reanchor(operation))
}
