use std::{cell::Cell, fmt, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, PrimitiveArray},
    compute::cast,
    datatypes::{
        DataType as ArrowDataType, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, FieldRef, SchemaRef, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
        Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    },
    record_batch::{RecordBatch, RecordBatchOptions},
};
use rapids_interop::{
    BufferBinding, BufferId, ColumnId, ColumnLayout, CompletionAuthority, PhysicalValidity, PreparedColumnExport,
    ProducerReadiness, ReadStartAuthority,
};

use super::DataFrame;
use crate::{
    core::{
        Column,
        column::{ColumnView, copy_view_to_owned},
        error::{
            self, Error, ErrorCode, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, FACT_VALIDATION_REASON,
            Result,
        },
        schema,
        table::TableView,
        types::{DataType, TypeId},
    },
    interop::{
        arrow_device::DeviceArrowTableExport,
        arrow_host::{
            ArrowHostSchema, cudf_dtype_can_export_to_arrow_type, cudf_dtype_from_arrow_type,
            estimate_arrow_host_record_batch_memory_bytes, validate_arrow_schema,
        },
    },
    ops::{
        binary::{self, BinaryOperator},
        filter,
        reduction::{self, ReduceAgg},
        unary,
        validation::{self, FiniteFloat32ValidationProof},
    },
    runtime::{
        BoundaryKind, BoundaryObservation, DenseMatrixLayout, ExecutionBindingId, ExecutionPlacement,
        PackedFloat32Matrix, QueryMemoryObservation, Runtime, execution_domain::ExecutionDomain,
        record_boundary_observation_in_domain,
    },
};

thread_local! {
    static FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT: Cell<usize> = const { Cell::new(0) };
}

struct RetainedProducerDomain(Arc<ExecutionDomain>);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedSizeListStructuralProof {
    field_index: usize,
    dimension: i32,
    rows: usize,
    child_offset: usize,
    child_len: usize,
    producer_column_identity: usize,
    binding_id: ExecutionBindingId,
    placement: ExecutionPlacement,
    observation: Option<BoundaryObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UniformListFloat32Proof {
    field_index: usize,
    dimension: usize,
    rows: usize,
    root_offset: usize,
    child_offset: usize,
    child_len: usize,
    producer_column_identity: usize,
    binding_id: ExecutionBindingId,
    placement: ExecutionPlacement,
    offsets_observation: BoundaryObservation,
}

impl UniformListFloat32Proof {
    fn matches_producer(self, producer: &Arc<crate::core::Column>) -> bool {
        self.producer_column_identity == Arc::as_ptr(producer) as usize
            && self.binding_id == producer.execution_domain().binding_id()
            && self.placement == producer.execution_domain().placement()
    }
}

#[derive(Debug, Clone, Copy)]
struct ListFloat32ExportContract {
    field_index: usize,
    dimension: usize,
    rows: usize,
    root_offset: Option<usize>,
    child_offset: usize,
    child_len: usize,
    placement: ExecutionPlacement,
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
    fn matches_schema(self, schema: &SchemaRef) -> bool {
        matches!(
            schema.fields().get(self.field_index).map(|field| field.data_type()),
            Some(ArrowDataType::FixedSizeList(_, dimension)) if *dimension == self.dimension
        )
    }

    fn matches_producer(self, producer: &Arc<crate::core::Column>) -> bool {
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
    export: PreparedColumnExport<'a>,
    producer_domain: RetainedProducerDomain,
    proof: FixedSizeListStructuralProof,
    finite_proof: FiniteFloat32ValidationProof,
    root_column_id: ColumnId,
    child_column_id: ColumnId,
    child_values_buffer_id: BufferId,
}

#[derive(Debug)]
struct ListFloat32ExportParts<'a> {
    export: PreparedColumnExport<'a>,
    root_column_id: ColumnId,
    offsets_buffer_id: BufferId,
    child_column_id: ColumnId,
    child_values_buffer_id: BufferId,
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
    export: PreparedColumnExport<'a>,
    producer_domain: RetainedProducerDomain,
    proof: UniformListFloat32Proof,
    finite_proof: FiniteFloat32ValidationProof,
    root_column_id: ColumnId,
    offsets_buffer_id: BufferId,
    child_column_id: ColumnId,
    child_values_buffer_id: BufferId,
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

    /// Positional field index in the retained [`GpuRecordBatch`] schema.
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

#[allow(clippy::too_many_arguments)]
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

struct ValidatedFrameSchema {
    mapped_schema: ArrowHostSchema,
    fixed_size_list_proofs: Arc<[FixedSizeListStructuralProof]>,
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
        self.batch.num_rows() - self.offset
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
            // Release the oversized GPU slice before allocating binary-search probes.
            drop(candidate);
            let mut low = 1usize;
            let mut high = candidate_rows - 1;
            let mut best_rows = 1usize;
            while low <= high {
                let middle = low + (high - low) / 2;
                let probe = self.candidate(middle)?;
                let estimated_bytes = probe.estimated_materialization_bytes()?;
                // Release each GPU slice before allocating the next probe.
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
        self.batch.with_validated_slice(frame)
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

    fn with_validated_slice(&self, frame: DataFrame) -> Result<Self> {
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

    /// Prepare a zero-copy device export for one `FixedSizeList<Float32>`
    /// field whose complete active range was structurally validated when this
    /// batch was created and finite-validated immediately before export.
    ///
    /// The returned witness is bound to the exact retained producer column,
    /// stream/memory-resource binding, semantic export graph, and active child
    /// range. Root or child nulls and non-finite child values are rejected
    /// because a dense vector consumer cannot represent them. Read access
    /// still requires activation through
    /// [`PreparedColumnExport::try_activate_read_lease`].
    pub fn try_prepare_fixed_size_list_float32_export(
        &self,
        field_index: usize,
    ) -> Result<ValidatedFixedSizeListFloat32Export<'_>> {
        let proof =
            self.fixed_size_list_proofs.iter().find(|proof| proof.field_index == field_index).copied().ok_or_else(
                || {
                    Error::new_with(
                        ErrorCode::UnsupportedDtype,
                        "dataframe::GpuRecordBatch::try_prepare_fixed_size_list_float32_export",
                        "requested field is not a validated FixedSizeList",
                    )
                    .with_detail("field_index", field_index.to_string())
                    .with_detail("reason", "fixed_size_list_structural_proof_unavailable")
                },
            )?;
        self.try_prepare_fixed_size_list_float32_export_with_proof(field_index, proof)
    }

    fn try_prepare_fixed_size_list_float32_export_with_proof(
        &self,
        field_index: usize,
        proof: FixedSizeListStructuralProof,
    ) -> Result<ValidatedFixedSizeListFloat32Export<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_fixed_size_list_float32_export";
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

        let export = producer
            .try_prepare_device_column_export()
            .map_err(|error| error::from_interop_in(error, OPERATION))?
            .with_retained_owner_arc(Arc::clone(producer));
        let parts = validate_list_float32_export(export, proof.into(), OPERATION)?;
        let child = producer
            .view()
            .map_err(|error| error.reanchor(OPERATION))?
            .list_sliced_child()
            .map_err(|error| error.reanchor(OPERATION))?;
        let finite_proof = validation::validate_finite_float32(&child).map_err(|error| error.reanchor(OPERATION))?;
        if finite_proof.validated_values() != proof.child_len
            || finite_proof.binding_id() != proof.binding_id
            || finite_proof.placement() != proof.placement
        {
            return Err(error::invariant_violation(
                OPERATION,
                "FixedSizeList finite-value proof does not cover the structurally proven child range",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("structural_child_len", proof.child_len.to_string())
            .with_detail("finite_validated_values", finite_proof.validated_values().to_string())
            .with_detail("structural_binding_id", proof.binding_id.to_string())
            .with_detail("finite_binding_id", finite_proof.binding_id().to_string())
            .with_detail("reason", "fixed_size_list_finite_proof_mismatch"));
        }

        Ok(ValidatedFixedSizeListFloat32Export {
            export: parts.export,
            producer_domain: RetainedProducerDomain(Arc::clone(producer.execution_domain())),
            proof,
            finite_proof,
            root_column_id: parts.root_column_id,
            child_column_id: parts.child_column_id,
            child_values_buffer_id: parts.child_values_buffer_id,
        })
    }

    /// Prepare a zero-copy device export for a variable-width Arrow
    /// `List<Float32>` or `LargeList<Float32>` field whose active rows all
    /// have one uniform positive dimension.
    ///
    /// The dimension is derived from device-reduced active offset deltas; it
    /// is never supplied by the caller. Empty inputs are rejected because a
    /// variable-width schema cannot prove a dimension without any rows.
    /// Root/child nulls, non-monotonic or ragged offsets, zero-width rows, and
    /// non-finite child values are rejected before the proof-bearing witness
    /// is returned.
    pub fn try_prepare_uniform_list_float32_export(
        &self,
        field_index: usize,
    ) -> Result<ValidatedUniformListFloat32Export<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_uniform_list_float32_export";
        let field = self.schema.fields().get(field_index).ok_or_else(|| {
            Error::new_with(ErrorCode::IndexOutOfBounds, OPERATION, "list field index is out of bounds")
                .with_detail("field_index", field_index.to_string())
                .with_detail("field_count", self.schema.fields().len().to_string())
        })?;
        let float32_child = match field.data_type() {
            ArrowDataType::List(child) | ArrowDataType::LargeList(child) => {
                matches!(child.data_type(), ArrowDataType::Float32)
            },
            _ => false,
        };
        if !float32_child {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OPERATION,
                "uniform list export requires a variable-width List<Float32> schema field",
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
        let proof = validate_uniform_list_float32_offsets(&view, producer, field_index, OPERATION)?;
        self.try_prepare_uniform_list_float32_export_with_proof(field_index, proof)
    }

    fn try_prepare_uniform_list_float32_export_with_proof(
        &self,
        field_index: usize,
        proof: UniformListFloat32Proof,
    ) -> Result<ValidatedUniformListFloat32Export<'_>> {
        const OPERATION: &str = "dataframe::GpuRecordBatch::try_prepare_uniform_list_float32_export";
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

        let export = producer
            .try_prepare_device_column_export()
            .map_err(|error| error::from_interop_in(error, OPERATION))?
            .with_retained_owner_arc(Arc::clone(producer));
        let parts = validate_list_float32_export(export, proof.into(), OPERATION)?;
        let child = producer
            .view()
            .map_err(|error| error.reanchor(OPERATION))?
            .list_sliced_child()
            .map_err(|error| error.reanchor(OPERATION))?;
        let finite_proof = validation::validate_finite_float32(&child).map_err(|error| error.reanchor(OPERATION))?;
        if finite_proof.validated_values() != proof.child_len
            || finite_proof.binding_id() != proof.binding_id
            || finite_proof.placement() != proof.placement
        {
            return Err(error::invariant_violation(
                OPERATION,
                "uniform-list finite proof does not cover the offset-proven active child range",
            )
            .with_detail("field_index", field_index.to_string())
            .with_detail("offset_proof_child_len", proof.child_len.to_string())
            .with_detail("finite_validated_values", finite_proof.validated_values().to_string())
            .with_detail("offset_proof_binding_id", proof.binding_id.to_string())
            .with_detail("finite_binding_id", finite_proof.binding_id().to_string())
            .with_detail("reason", "uniform_list_finite_proof_mismatch"));
        }

        Ok(ValidatedUniformListFloat32Export {
            export: parts.export,
            producer_domain: RetainedProducerDomain(Arc::clone(producer.execution_domain())),
            proof,
            finite_proof,
            root_column_id: parts.root_column_id,
            offsets_buffer_id: parts.offsets_buffer_id,
            child_column_id: parts.child_column_id,
            child_values_buffer_id: parts.child_values_buffer_id,
        })
    }

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
        let proof = validate_uniform_list_float32_offsets(&view, producer, field_index, OPERATION)?;
        self.try_prepare_uniform_list_float32_export_with_proof(field_index, proof).map(drop)
    }

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
        self.try_prepare_fixed_size_list_float32_export_with_proof(field_index, proof).map(drop)
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

fn validate_frame_schema(frame: &DataFrame, schema_ref: &SchemaRef) -> Result<ValidatedFrameSchema> {
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
            return Err(crate::internal::resolve::missing_column_error(operation, field.name())
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

fn validate_list_schema_contract(
    view: &ColumnView<'_>,
    child_field: &FieldRef,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<()> {
    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let actual_child_dtype = child_view.data_type().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_dtype =
        cudf_dtype_from_arrow_type(child_field.data_type()).map_err(|error| error.reanchor_if_needed(operation))?;
    if !cudf_dtype_can_export_to_arrow_type(actual_child_dtype, child_field.data_type()) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            format!(
                "Arrow schema field '{field_name}' requires list child {:?} but the DataFrame list child is {:?}",
                child_field.data_type(),
                actual_child_dtype.id()
            ),
        )
        .with_fact(FACT_LEFT_TYPE_ID, actual_child_dtype.id())
        .with_fact(FACT_RIGHT_TYPE_ID, expected_child_dtype.id())
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("actual_child_cudf_type", actual_child_dtype.id().name())
        .with_detail("expected_child_arrow_type", format!("{:?}", child_field.data_type()))
        .with_detail("reason", "schema_list_child_dtype_mismatch"));
    }
    let child_null_count = child_view.null_count().map_err(|error| error.reanchor_if_needed(operation))?;
    if !child_field.is_nullable() && child_null_count > 0 {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            format!(
                "Arrow schema field '{field_name}' declares a non-nullable list child but the DataFrame child contains nulls"
            ),
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_null_count", child_null_count.to_string())
        .with_detail("reason", "schema_list_child_nullability_mismatch"));
    }

    Ok(())
}

fn validate_fixed_size_list_offsets(
    view: &ColumnView<'_>,
    producer: &Arc<crate::core::Column>,
    dimension: i32,
    field_name: &str,
    field_index: usize,
    operation: &'static str,
) -> Result<FixedSizeListStructuralProof> {
    let expected_dimension = usize::try_from(dimension).map_err(|_| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow FixedSizeList dimension must be positive")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "non_positive_fixed_size_list_dimension")
    })?;
    if expected_dimension == 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "Arrow FixedSizeList dimension must be positive",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("dimension", "0")
        .with_detail("reason", "non_positive_fixed_size_list_dimension"));
    }

    let rows = view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset_raw = child_view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "FixedSizeList child offset must be non-negative",
            "list_child",
            "negative_fixed_size_list_child_offset",
            "ensure the cudf list column exposes a valid active child range",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_len = rows.checked_mul(expected_dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList active child length overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", expected_dimension.to_string())
            .with_detail("reason", "fixed_size_list_child_len_overflow")
    })?;
    if rows == 0 && child_len != expected_child_len {
        return Err(error::foreign_invariant_violation(
            operation,
            "empty FixedSizeList active child range must also be empty",
            "list_child",
            "fixed_size_list_active_child_len_mismatch",
            "ensure the cudf list active child range matches the fixed-size schema",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("row_count", rows.to_string())
        .with_detail("dimension", expected_dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }
    if rows == 0 {
        return Ok(record_fixed_size_list_structural_proof(
            field_index,
            dimension,
            rows,
            child_offset,
            child_len,
            producer,
            None,
        ));
    }
    let root_offset_raw = view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_offset = usize::try_from(root_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "cudf list root offset must be non-negative",
            "list_offsets",
            "negative_list_root_offset",
            "ensure the cudf list column exposes a valid active range",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("root_offset", root_offset_raw.to_string())
    })?;
    let root_end = root_offset.checked_add(rows).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList active offset range overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("reason", "fixed_size_list_offset_range_overflow")
    })?;
    let offsets_view = view.child(0).map_err(|error| error.reanchor_if_needed(operation))?;
    let offsets_len = offsets_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if offsets_len <= root_end {
        return Err(error::foreign_invariant_violation(
            operation,
            "cudf list offsets are too short for the FixedSizeList active range",
            "list_offsets",
            "fixed_size_list_offsets_too_short",
            "ensure the cudf list column exposes one more offset than active rows",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("required_offset_index", root_end.to_string())
        .with_detail("offset_count", offsets_len.to_string()));
    }

    let rows_i32 = i32::try_from(rows).map_err(|_| {
        error::range_limit_error(operation, "row_count", "i32::MAX", rows)
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
    })?;
    let end_offset = root_offset_raw.checked_add(1).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList end-offset slice start overflowed i32")
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset_raw.to_string())
            .with_detail("reason", "fixed_size_list_end_offset_start_overflow")
    })?;
    let starts = offsets_view
        .slice_with_operation(root_offset_raw, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let ends = offsets_view
        .slice_with_operation(end_offset, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let int32_type = DataType::new(TypeId::Int32);
    let widths = binary::binary_operation_vv(&ends, &starts, BinaryOperator::Sub, &int32_type)
        .map_err(|error| error.reanchor(operation))?;
    let widths_view = widths.view().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_null_count = view.null_count().map_err(|error| error.reanchor_if_needed(operation))?;
    let filtered_widths = if root_null_count == 0 || root_null_count == rows {
        None
    } else {
        let valid_rows = unary::is_valid(view).map_err(|error| error.reanchor(operation))?;
        let valid_rows_view = valid_rows.view().map_err(|error| error.reanchor_if_needed(operation))?;
        let widths_table =
            TableView::from_column_views(&[&widths_view]).map_err(|error| error.reanchor_if_needed(operation))?;
        let mut columns = filter::apply_boolean_mask(&widths_table, &valid_rows_view)
            .map_err(|error| error.reanchor(operation))?
            .into_columns()
            .map_err(|error| error.reanchor_if_needed(operation))?;
        Some(columns.pop().ok_or_else(|| {
            error::foreign_invariant_violation(
                operation,
                "filtering FixedSizeList widths returned no column",
                "list_offsets",
                "missing_filtered_fixed_size_list_widths",
                "ensure cuDF preserves the single input column when filtering valid list rows",
            )
            .with_detail("field_name", field_name.to_string())
            .with_detail("field_index", field_index.to_string())
        })?)
    };
    let (minimum, maximum, downloaded_scalars) = if root_null_count == rows {
        (dimension, dimension, 0usize)
    } else {
        let reduction_view = match filtered_widths.as_ref() {
            Some(filtered) => filtered.view().map_err(|error| error.reanchor_if_needed(operation))?,
            None => widths_view,
        };
        let minimum = reduction::reduce(&reduction_view, &ReduceAgg::Min, &int32_type)
            .map_err(|error| error.reanchor(operation))?;
        let maximum = reduction::reduce(&reduction_view, &ReduceAgg::Max, &int32_type)
            .map_err(|error| error.reanchor(operation))?;
        (
            minimum.value::<i32>().map_err(|error| error.reanchor(operation))?,
            maximum.value::<i32>().map_err(|error| error.reanchor(operation))?,
            2usize,
        )
    };
    let observation = BoundaryObservation::new(
        BoundaryKind::BoundedControlScalar,
        producer.execution_domain().placement(),
        rows,
        downloaded_scalars,
        true,
    )
    .with_memory(
        QueryMemoryObservation::empty().with_downloaded_bytes(downloaded_scalars * std::mem::size_of::<i32>()),
    );
    record_boundary_observation_in_domain(producer.execution_domain(), observation);
    if minimum != dimension || maximum != dimension {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "cudf list row widths do not match the Arrow FixedSizeList dimension",
        )
        .with_fact(FACT_EXPECTED_ROWS, expected_dimension)
        .with_detail("field_name", field_name.to_string())
        .with_detail("field_index", field_index.to_string())
        .with_detail("expected_dimension", expected_dimension.to_string())
        .with_detail("minimum_actual_dimension", minimum.to_string())
        .with_detail("maximum_actual_dimension", maximum.to_string())
        .with_detail("evidence", "device_reduced_list_offset_width_range")
        .with_detail("reason", "schema_fixed_size_list_dimension_mismatch"));
    }

    Ok(record_fixed_size_list_structural_proof(
        field_index,
        dimension,
        rows,
        child_offset,
        child_len,
        producer,
        Some(observation),
    ))
}

fn validate_uniform_list_float32_offsets(
    view: &ColumnView<'_>,
    producer: &Arc<crate::core::Column>,
    field_index: usize,
    operation: &'static str,
) -> Result<UniformListFloat32Proof> {
    let rows = view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if rows == 0 {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            operation,
            "variable-width List<Float32> dimension is unavailable for an empty active range",
        )
        .with_fact(FACT_VALIDATION_REASON, "uniform_list_dimension_unavailable")
        .with_detail("field_index", field_index.to_string())
        .with_detail("row_count", "0")
        .with_detail("remedy", "provide at least one vector row or use a trusted FixedSizeList schema dimension"));
    }

    let root_offset_raw = view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let root_offset = usize::try_from(root_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "cudf list root offset must be non-negative",
            "list_offsets",
            "negative_list_root_offset",
            "retain a valid cudf list active range",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("root_offset", root_offset_raw.to_string())
    })?;
    let root_end = root_offset.checked_add(rows).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform list active offset range overflowed usize")
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("reason", "uniform_list_offset_range_overflow")
    })?;
    let offsets_view = view.child(0).map_err(|error| error.reanchor_if_needed(operation))?;
    let offsets_len = offsets_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    if offsets_len <= root_end {
        return Err(error::foreign_invariant_violation(
            operation,
            "cudf list offsets are too short for the active uniform-list range",
            "list_offsets",
            "uniform_list_offsets_too_short",
            "retain one more list offset than active rows",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("required_offset_index", root_end.to_string())
        .with_detail("offset_count", offsets_len.to_string()));
    }

    let rows_i32 =
        i32::try_from(rows).map_err(|_| error::range_limit_error(operation, "row_count", "i32::MAX", rows))?;
    let end_offset = root_offset_raw.checked_add(1).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform-list end-offset start overflowed i32")
            .with_detail("field_index", field_index.to_string())
            .with_detail("root_offset", root_offset_raw.to_string())
            .with_detail("reason", "uniform_list_end_offset_start_overflow")
    })?;
    let starts = offsets_view
        .slice_with_operation(root_offset_raw, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let ends = offsets_view
        .slice_with_operation(end_offset, rows_i32, operation)
        .map_err(|error| error.reanchor_if_needed(operation))?;
    let int32_type = DataType::new(TypeId::Int32);
    let widths = binary::binary_operation_vv(&ends, &starts, BinaryOperator::Sub, &int32_type)
        .map_err(|error| error.reanchor(operation))?;
    let widths_view = widths.view().map_err(|error| error.reanchor_if_needed(operation))?;
    let minimum =
        reduction::reduce(&widths_view, &ReduceAgg::Min, &int32_type).map_err(|error| error.reanchor(operation))?;
    let maximum =
        reduction::reduce(&widths_view, &ReduceAgg::Max, &int32_type).map_err(|error| error.reanchor(operation))?;
    let minimum = minimum.value::<i32>().map_err(|error| error.reanchor(operation))?;
    let maximum = maximum.value::<i32>().map_err(|error| error.reanchor(operation))?;
    let offsets_observation = BoundaryObservation::new(
        BoundaryKind::BoundedControlScalar,
        producer.execution_domain().placement(),
        rows,
        2,
        true,
    )
    .with_memory(
        QueryMemoryObservation::empty().with_downloaded_bytes(2usize.saturating_mul(std::mem::size_of::<i32>())),
    );
    record_boundary_observation_in_domain(producer.execution_domain(), offsets_observation);

    if minimum < 0 {
        return Err(error::foreign_invariant_violation(
            operation,
            "List<Float32> offsets are not monotonic over the active range",
            "list_offsets",
            "list_offsets_not_monotonic",
            "repair the list offsets before requesting a dense-vector export",
        )
        .with_fact(FACT_VALIDATION_REASON, "list_offsets_not_monotonic")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    if minimum == 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "uniform List<Float32> dimension must be positive",
        )
        .with_fact(FACT_VALIDATION_REASON, "non_positive_vector_dimension")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    if minimum != maximum {
        return Err(Error::new_with(
            ErrorCode::ColumnLengthMismatch,
            operation,
            "List<Float32> rows do not have one uniform dimension",
        )
        .with_fact(FACT_VALIDATION_REASON, "non_uniform_vector_dimension")
        .with_detail("field_index", field_index.to_string())
        .with_detail("minimum_row_width", minimum.to_string())
        .with_detail("maximum_row_width", maximum.to_string())
        .with_detail("validation_evidence", "device_reduced_list_offset_width_range"));
    }
    let dimension = usize::try_from(minimum).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "positive uniform list dimension must fit usize",
            "list_offsets",
            "uniform_list_dimension_conversion_failed",
            "retain cudf-compatible list offsets",
        )
        .with_detail("dimension", minimum.to_string())
    })?;

    let child_view = view.list_sliced_child().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset_raw = child_view.offset().map_err(|error| error.reanchor_if_needed(operation))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            operation,
            "uniform List<Float32> child offset must be non-negative",
            "list_child",
            "negative_uniform_list_child_offset",
            "retain a valid cudf list active child range",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child_view.len().map_err(|error| error.reanchor_if_needed(operation))?;
    let expected_child_len = rows.checked_mul(dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "uniform List<Float32> child length overflowed usize")
            .with_detail("field_index", field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "uniform_list_child_len_overflow")
    })?;
    if child_len != expected_child_len {
        return Err(error::foreign_invariant_violation(
            operation,
            "offset-proven uniform list span differs from the active child range",
            "list_child",
            "uniform_list_active_child_len_mismatch",
            "retain a cudf list child range derived from the active offsets",
        )
        .with_detail("field_index", field_index.to_string())
        .with_detail("dimension", dimension.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }

    Ok(UniformListFloat32Proof {
        field_index,
        dimension,
        rows,
        root_offset,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        offsets_observation,
    })
}

fn record_fixed_size_list_structural_proof(
    field_index: usize,
    dimension: i32,
    rows: usize,
    child_offset: usize,
    child_len: usize,
    producer: &Arc<crate::core::Column>,
    observation: Option<BoundaryObservation>,
) -> FixedSizeListStructuralProof {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(|count| count.set(count.get() + 1));
    FixedSizeListStructuralProof {
        field_index,
        dimension,
        rows,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        observation,
    }
}

fn rebind_fixed_size_list_structural_proof(
    proof: FixedSizeListStructuralProof,
    frame: &DataFrame,
) -> Result<FixedSizeListStructuralProof> {
    const OPERATION: &str = "dataframe::GpuRecordBatch::with_validated_slice";
    let (_, producer) = frame.columns_map().get_index(proof.field_index).ok_or_else(|| {
        Error::new_with(
            ErrorCode::InvariantViolation,
            OPERATION,
            "validated FixedSizeList field disappeared while slicing a GPU record batch",
        )
        .with_detail("field_index", proof.field_index.to_string())
    })?;
    let view = producer.view().map_err(|error| error.reanchor(OPERATION))?;
    let rows = view.len().map_err(|error| error.reanchor(OPERATION))?;
    let child = view.list_sliced_child().map_err(|error| error.reanchor(OPERATION))?;
    let child_offset_raw = child.offset().map_err(|error| error.reanchor(OPERATION))?;
    let child_offset = usize::try_from(child_offset_raw).map_err(|_| {
        error::foreign_invariant_violation(
            OPERATION,
            "sliced FixedSizeList child offset must be non-negative",
            "list_child",
            "negative_fixed_size_list_child_offset",
            "retain a valid cudf list slice",
        )
        .with_detail("field_index", proof.field_index.to_string())
        .with_detail("child_offset", child_offset_raw.to_string())
    })?;
    let child_len = child.len().map_err(|error| error.reanchor(OPERATION))?;
    let expected_child_len = rows.checked_mul(proof.dimension as usize).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, OPERATION, "sliced FixedSizeList child length overflowed usize")
            .with_detail("field_index", proof.field_index.to_string())
            .with_detail("row_count", rows.to_string())
            .with_detail("dimension", proof.dimension.to_string())
    })?;
    let root_null_count = view.null_count().map_err(|error| error.reanchor(OPERATION))?;
    if root_null_count == 0 && child_len != expected_child_len {
        return Err(error::invariant_violation(
            OPERATION,
            "slicing a structurally validated FixedSizeList changed its fixed-width child span",
        )
        .with_detail("field_index", proof.field_index.to_string())
        .with_detail("expected_child_len", expected_child_len.to_string())
        .with_detail("actual_child_len", child_len.to_string()));
    }

    Ok(FixedSizeListStructuralProof {
        field_index: proof.field_index,
        dimension: proof.dimension,
        rows,
        child_offset,
        child_len,
        producer_column_identity: Arc::as_ptr(producer) as usize,
        binding_id: producer.execution_domain().binding_id(),
        placement: producer.execution_domain().placement(),
        observation: proof.observation,
    })
}

fn validate_list_float32_export<'a>(
    export: PreparedColumnExport<'a>,
    contract: ListFloat32ExportContract,
    operation: &'static str,
) -> Result<ListFloat32ExportParts<'a>> {
    let contract_error = |message: &'static str, reason: &'static str| {
        error::foreign_invariant_violation(
            operation,
            message,
            "fixed_size_list_export",
            reason,
            "rebuild the export from the validated GpuRecordBatch producer",
        )
        .with_detail("field_index", contract.field_index.to_string())
        .with_detail("dimension", contract.dimension.to_string())
    };

    let root = export.column();
    if root.data_type().id() != rapids_interop::TypeId::List || root.len() != contract.rows {
        return Err(contract_error(
            "prepared export root does not match the validated List<Float32> row contract",
            "list_float32_export_root_mismatch",
        )
        .with_detail("root_type", format!("{:?}", root.data_type().id()))
        .with_detail("root_rows", root.len().to_string())
        .with_detail("proof_rows", contract.rows.to_string()));
    }
    if let Some(root_offset) = contract.root_offset
        && root.offset() != root_offset
    {
        return Err(contract_error(
            "prepared list export root offset differs from the proven active range",
            "list_float32_export_root_offset_mismatch",
        )
        .with_detail("proof_root_offset", root_offset.to_string())
        .with_detail("export_root_offset", root.offset().to_string()));
    }
    if !validity_proves_no_nulls(root.validity()) {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            "validated List<Float32> vector rows must not contain nulls",
        )
        .with_detail("field_index", contract.field_index.to_string())
        .with_detail("reason", "null_vector_row"));
    }

    let (offsets_buffer_id, child_column_id) = match root.layout() {
        ColumnLayout::List { offsets, child } => (*offsets, *child),
        _ => {
            return Err(contract_error(
                "prepared export root does not retain a list layout",
                "fixed_size_list_export_layout_mismatch",
            ));
        },
    };
    if !export.bindings().iter().any(|binding| binding.id() == offsets_buffer_id) {
        return Err(contract_error(
            "prepared List<Float32> export lost its offsets binding",
            "list_float32_offsets_binding_missing",
        ));
    }
    let child = export.semantic_columns().iter().find(|column| column.id() == child_column_id).ok_or_else(|| {
        contract_error(
            "prepared List<Float32> export lost its semantic child column",
            "list_float32_child_column_missing",
        )
    })?;
    if child.data_type().id() != rapids_interop::TypeId::Float32 {
        return Err(contract_error("prepared list child is not Float32", "list_float32_child_type_mismatch")
            .with_detail("child_type", format!("{:?}", child.data_type().id())));
    }
    if child.len() != contract.child_len || child.offset() != contract.child_offset {
        return Err(contract_error(
            "prepared List<Float32> child active range differs from the structural proof",
            "list_float32_child_range_mismatch",
        )
        .with_detail("proof_child_offset", contract.child_offset.to_string())
        .with_detail("export_child_offset", child.offset().to_string())
        .with_detail("proof_child_len", contract.child_len.to_string())
        .with_detail("export_child_len", child.len().to_string()));
    }
    if !validity_proves_no_nulls(child.validity()) {
        return Err(Error::new_with(
            ErrorCode::NullUnexpected,
            operation,
            "validated List<Float32> child must not contain nulls",
        )
        .with_detail("field_index", contract.field_index.to_string())
        .with_detail("reason", "null_vector_value"));
    }
    let child_values_buffer_id = match child.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        _ => {
            return Err(contract_error(
                "prepared List<Float32> child is not fixed-width",
                "list_float32_child_layout_mismatch",
            ));
        },
    };
    let child_values =
        export.bindings().iter().find(|binding| binding.id() == child_values_buffer_id).ok_or_else(|| {
            contract_error(
                "prepared List<Float32> export lost its child values binding",
                "list_float32_child_values_binding_missing",
            )
        })?;
    let required_child_bytes = contract
        .child_offset
        .checked_add(contract.child_len)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            contract_error(
                "validated List<Float32> child byte range overflowed usize",
                "list_float32_child_byte_range_overflow",
            )
        })?;
    if child_values.bytes().byte_len() < required_child_bytes {
        return Err(contract_error(
            "prepared List<Float32> child values binding is smaller than the proven active range",
            "list_float32_child_values_binding_too_small",
        )
        .with_detail("required_child_bytes", required_child_bytes.to_string())
        .with_detail("binding_bytes", child_values.bytes().byte_len().to_string()));
    }

    let expected_device = contract.placement.device.ordinal() as i32;
    if !matches!(
        child_values.placement(),
        rapids_interop::Placement::CudaDevice { device } if *device == expected_device
    ) {
        return Err(contract_error(
            "prepared List<Float32> child values binding is not on the proven CUDA device",
            "list_float32_child_device_mismatch",
        ));
    }
    match export.readiness() {
        ProducerReadiness::DeviceStreamOrdered { stream }
            if stream.device() == expected_device && stream.as_raw() as usize == contract.placement.stream_debug => {},
        _ => {
            return Err(contract_error(
                "prepared List<Float32> export readiness differs from the proven execution stream",
                "list_float32_export_readiness_mismatch",
            ));
        },
    }

    Ok(ListFloat32ExportParts {
        root_column_id: root.id(),
        offsets_buffer_id,
        child_column_id,
        child_values_buffer_id,
        export,
    })
}

fn validity_proves_no_nulls(validity: &PhysicalValidity) -> bool {
    matches!(validity, PhysicalValidity::NoValidityBuffer | PhysicalValidity::ValidityMask { null_count: Some(0), .. })
}

pub(crate) fn reset_fixed_size_list_structural_proof_build_count_for_test() {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(|count| count.set(0));
}

pub(crate) fn fixed_size_list_structural_proof_build_count_for_test() -> usize {
    FIXED_SIZE_LIST_STRUCTURAL_PROOF_BUILD_COUNT.with(Cell::get)
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
