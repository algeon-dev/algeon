//! Frontend-neutral GPU-resident data, result, and reusable build handles.
//!
//! Handles in this module are runtime/device-affine. Cloning shares the
//! underlying native resources; host materialization is explicit.
//!
//! Besides the cuDF frame, a handle carries the lifetimes that make the frame
//! safe to keep on the device: a `ResidentOwnerLease` for the bytes charged to
//! the producing attempt's grant, transitive source owners for columns that alias
//! an input frame's memory, a source-cache pin that keeps a cache entry from
//! being evicted while the frame aliases it, and graph provenance. A consumer
//! that reads a handle on its own stream wraps these owners with `for_reader`, so
//! the owner's final release also waits for that stream's work; dropping a handle
//! therefore never frees buffers a stream may still read. The release rules are
//! owned by `website/docs/design/admission.mdx#resident-handoff-and-release`.

use std::{
    collections::{BTreeMap, HashSet},
    mem,
    sync::Arc,
};

use arrow::{
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use cudf::{
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, DeviceId, ExecutionPlacement,
    frame::dataframe::{
        ColumnMemoryIdentity, GpuRecordBatch, HostBatchCursor as CudfHostBatchCursor, HostBatchCursorConfig,
        InfoDetail, PreparedHostBatch as CudfPreparedHostBatch,
    },
    runtime::CudaStream,
};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    naming::is_datafusion_order_column_name,
    plan::{GraphResidencyDescriptor, JoinNullEquality, PlanFingerprint, QueryPlan},
};

use crate::{
    native_error::cudf_query_error,
    runtime::{
        ExecutionCancellation, ExecutionContext,
        cache::local_parquet::GpuLocalParquetSourceCachePin,
        gpu::ledger::{MemoryGrant, ResidentOwnerLease},
    },
};

/// Local lookup table for GPU-resident inputs consumed by a query.
///
/// Keys are plan-local input names, not global handles. All handles must be
/// reusable by the same [`ExecutionContext`] that executes the consuming plan.
pub type GpuResidentInputMap = BTreeMap<String, GpuResultHandle>;

/// Owned cuDF DataFrame plus the placement observed when it entered the engine.
///
/// The wrapper preserves the device placement contract while allowing callers
/// to move the underlying frame between native operators. Clone semantics are
/// those of `cudf::DataFrame`.
#[derive(Clone)]
pub struct GpuDataFrame {
    frame: DataFrame,
    placement: ExecutionPlacement,
    /// Grant retention for the bytes this frame published. `None` for frames
    /// whose memory is accounted only by the allocator of the attempt that
    /// holds them, including cache checkouts.
    resident_owner: Option<ResidentOwnerLease>,
    /// Owners of input frames whose column memory this frame still aliases.
    /// Narrowed on every projection so a derived frame retains only the
    /// owners of columns it carries.
    retained_source_owners: Vec<RetainedSourceOwner>,
    /// Owner of a graph normalization input kept alive until the graph
    /// consumer takes it; not propagated to derived frames.
    retained_normalization_source: Option<ResidentOwnerLease>,
    /// Keeps the local-Parquet cache entry this frame was copied from pinned,
    /// and therefore non-evictable, for the frame's lifetime.
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    graph_residency: Option<Arc<GraphResidencyDescriptor>>,
    producer_execution_scope: Option<PlanFingerprint>,
}

struct GpuDataFrameParts {
    frame: DataFrame,
    resident_owner: Option<ResidentOwnerLease>,
    retained_source_owners: Vec<RetainedSourceOwner>,
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    graph_residency: Option<Arc<GraphResidencyDescriptor>>,
    producer_execution_scope: Option<PlanFingerprint>,
}

// A transitive lease follows only aliases of the columns it protects. Keeping
// the identity set prevents both premature release and fresh-only over-retention.
#[derive(Clone)]
struct RetainedSourceOwner {
    owner: ResidentOwnerLease,
    column_identities: HashSet<ColumnMemoryIdentity>,
}

impl RetainedSourceOwner {
    fn for_reader(self, stream: &CudaStream) -> Self {
        Self { owner: self.owner.for_reader(stream.clone()), column_identities: self.column_identities }
    }

    fn retain_for(&self, output_identities: &HashSet<ColumnMemoryIdentity>) -> Option<Self> {
        let column_identities: HashSet<_> = self.column_identities.intersection(output_identities).copied().collect();
        (!column_identities.is_empty()).then(|| Self { owner: self.owner.clone(), column_identities })
    }
}

impl GpuDataFrame {
    /// Wrap a cuDF frame and record its current execution placement.
    pub fn new(frame: DataFrame) -> Self {
        Self::new_with_resident_owner(frame, None)
    }

    /// Wrap a derived cuDF frame with identity-aware live-byte accounting and
    /// the source's transitive resident and cache eviction-safety lifetimes.
    pub fn try_new_with_source_residency(frame: DataFrame, source: &Self) -> Result<Self> {
        if source.resident_owner.is_none()
            && source.retained_source_owners.is_empty()
            && source.source_cache_pin.is_none()
        {
            return Ok(Self::new(frame));
        }
        let source_identities = source_column_memory_identities(&source.frame);
        let output_identity = frame.memory_identity();
        let output_identities = output_identity.columns.iter().map(|column| column.identity).collect::<HashSet<_>>();
        let aliased_output_columns =
            output_identity.columns.iter().filter(|column| source_identities.contains(&column.identity)).count();
        let aliases_source = aliased_output_columns > 0;
        let all_output_columns_alias_source = aliases_source && aliased_output_columns == output_identity.columns.len();
        let mut retained_source_owners = source
            .retained_source_owners
            .iter()
            .filter_map(|owner| owner.retain_for(&output_identities))
            .collect::<Vec<_>>();
        let resident_owner = match &source.resident_owner {
            Some(source_owner) if all_output_columns_alias_source => {
                // Pure aliases, including column subsets, keep the source
                // owner alive instead of remeasuring a partial owner set.
                Some(source_owner.clone())
            },
            // A frozen grant accepts no further owner charge, so the derived
            // frame keeps only the alias retention. Its fresh columns are
            // accounted by the consuming attempt's own allocator.
            Some(source_owner) if source_owner.is_frozen_resident() => {
                if aliases_source {
                    retained_source_owners.push(RetainedSourceOwner {
                        owner: source_owner.clone(),
                        column_identities: output_identities.intersection(&source_identities).copied().collect(),
                    });
                }
                None
            },
            Some(source_owner) => {
                // Mixed alias/fresh outputs need one retention handle to cover the
                // whole escaping frame. This intentionally over-accounts aliased
                // columns while the input is alive, but avoids under-accounting
                // alias owners after the input wrapper drops.
                let observed_bytes = Self::new(frame.clone()).device_allocated_size_bytes()?;
                source_owner.retain_derived_owner(observed_bytes)?
            },
            None => None,
        };
        let mut output = Self::new_with_residency(frame, resident_owner, source.source_cache_pin.clone());
        output.retained_source_owners = retained_source_owners;
        Ok(output)
    }

    /// Marks every retained owner as read on `stream`, so each owner's final
    /// release waits for the work `stream` has enqueued when this frame and
    /// its derivatives drop.
    fn read_on_stream(&mut self, stream: &CudaStream) {
        self.resident_owner = self.resident_owner.take().map(|owner| owner.for_reader(stream.clone()));
        self.retained_source_owners = mem::take(&mut self.retained_source_owners)
            .into_iter()
            .map(|retained| retained.for_reader(stream))
            .collect();
    }

    fn new_with_resident_owner(frame: DataFrame, resident_owner: Option<ResidentOwnerLease>) -> Self {
        Self::new_with_residency(frame, resident_owner, None)
    }

    fn new_with_residency(
        frame: DataFrame,
        resident_owner: Option<ResidentOwnerLease>,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    ) -> Self {
        let placement = frame.placement();
        Self {
            frame,
            placement,
            resident_owner,
            retained_source_owners: Vec::new(),
            retained_normalization_source: None,
            source_cache_pin,
            graph_residency: None,
            producer_execution_scope: None,
        }
    }

    /// Borrow the underlying cuDF frame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Consume the wrapper and return the underlying cuDF frame.
    pub fn into_frame(self) -> DataFrame {
        let (frame, _) = self.into_frame_and_resident_owner();
        frame
    }

    fn into_frame_and_resident_owner(self) -> (DataFrame, Option<ResidentOwnerLease>) {
        let GpuDataFrameParts { frame, resident_owner, .. } = self.into_frame_and_residency();
        (frame, resident_owner)
    }

    fn into_frame_and_residency(self) -> GpuDataFrameParts {
        debug_assert_eq!(self.placement, self.frame.placement());
        GpuDataFrameParts {
            frame: self.frame,
            resident_owner: self.resident_owner,
            retained_source_owners: self.retained_source_owners,
            source_cache_pin: self.source_cache_pin,
            graph_residency: self.graph_residency,
            producer_execution_scope: self.producer_execution_scope,
        }
    }

    /// Attach graph source provenance to this GPU frame.
    pub fn with_graph_residency(mut self, graph_residency: GraphResidencyDescriptor) -> Self {
        self.graph_residency = Some(Arc::new(graph_residency));
        self
    }

    /// Graph source provenance carried by this frame, when known.
    pub fn graph_residency(&self) -> Option<&GraphResidencyDescriptor> {
        self.graph_residency.as_deref()
    }

    #[doc(hidden)]
    pub fn with_producer_execution_scope(mut self, scope: PlanFingerprint) -> Self {
        self.producer_execution_scope = Some(scope);
        self
    }

    /// Execution placement observed when the frame entered the engine.
    pub fn placement(&self) -> ExecutionPlacement {
        self.placement
    }

    /// The CUDA device of the frame.
    pub fn device(&self) -> DeviceId {
        self.placement.device
    }

    /// Current row count reported by the cuDF frame.
    pub fn num_rows(&self) -> usize {
        self.frame.num_rows()
    }

    /// Current column count reported by the cuDF frame.
    pub fn num_columns(&self) -> usize {
        self.frame.num_columns()
    }

    /// Observed device allocation for the frame in bytes.
    pub fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.frame.info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_dataframe.device_memory",
                "failed to observe native GPU dataframe memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.gpu_dataframe.device_memory",
                    "cudf dataframe memory accounting did not include device-allocated bytes",
                )
                .with_fact("rows", self.num_rows())
                .with_fact("columns", self.num_columns())
            })?;
        Ok(bytes as u64)
    }

    /// Bytes of this frame already charged to `grant`.
    ///
    /// This is intentionally distinct from the frame's physical allocation:
    /// callers use it only to avoid reserving an overlapping live input twice.
    #[cfg(any(feature = "cugraph", test))]
    #[cfg(test)]
    pub(crate) fn resident_owner_bytes_for(&self, grant: Option<&MemoryGrant>) -> u64 {
        match (self.resident_owner.as_ref(), grant) {
            (Some(owner), Some(grant)) if owner.belongs_to(grant) => owner.bytes(),
            _ => 0,
        }
    }

    #[doc(hidden)]
    pub fn take_resident_owner(&mut self) -> Option<ResidentOwnerLease> {
        self.resident_owner.take()
    }

    #[doc(hidden)]
    pub fn with_retained_normalization_source(mut self, owner: Option<ResidentOwnerLease>) -> Self {
        self.retained_normalization_source = owner;
        self
    }

    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub fn take_retained_normalization_source(&mut self) -> Option<ResidentOwnerLease> {
        self.retained_normalization_source.take()
    }
}

/// GPU-resident query result bound to an Arrow schema and runtime.
///
/// Cloning shares the underlying cuDF batch, runtime, and frozen resident
/// leases. Same-device consumers may borrow the result as resident input; host
/// materialization is explicit through [`to_host_record_batch`](Self::to_host_record_batch).
#[derive(Clone)]
pub struct GpuResultHandle {
    batch: GpuRecordBatch,
    runtime: Arc<cudf::Runtime>,
    resident_owner: Option<ResidentOwnerLease>,
    retained_source_owners: Vec<RetainedSourceOwner>,
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    graph_residency: Option<Arc<GraphResidencyDescriptor>>,
    producer_execution: Option<Arc<ProducerExecutionIdentity>>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct ProducerExecutionIdentity {
    scope: PlanFingerprint,
    plan: Option<Arc<QueryPlan>>,
}

/// Query-neutral GPU allocation retained by the local-Parquet source cache.
///
/// The allocation is owned by the cache allocation lease, never by the
/// attempt whose decoded frame was copied into the cache. The lease keeps the
/// device-scoped cache allocation domain alive until the concrete frame has
/// been destroyed and its residency charge can be released.
pub(crate) struct GpuLocalParquetCachedResult {
    batch: GpuRecordBatch,
    allocation_lease: Arc<cudf::CacheAllocationLease>,
}

/// Bounded native GPU-to-host cursor retaining every owner required by the
/// resident result until the final chunk is materialized or the cursor drops.
pub struct GpuHostBatchCursor {
    cursor: CudfHostBatchCursor,
    output_schema: SchemaRef,
    cancellation: ExecutionCancellation,
    /// The result's runtime, whose stream the cursor copies on. Held so the
    /// stream outlives the last copy and the owners below can fence on it.
    _runtime: Arc<cudf::Runtime>,
    _resident_owner: Option<ResidentOwnerLease>,
    _retained_source_owners: Vec<RetainedSourceOwner>,
    _source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    _graph_residency: Option<Arc<GraphResidencyDescriptor>>,
}

/// Metadata-sized GPU output chunk awaiting a reserved device-to-host copy.
#[derive(Debug)]
pub struct GpuPreparedHostBatch {
    prepared: CudfPreparedHostBatch,
}

impl GpuPreparedHostBatch {
    /// Conservative final Arrow buffer capacity that must be reserved before
    /// materializing this chunk.
    pub fn estimated_materialization_bytes(&self) -> usize {
        self.prepared.estimated_materialization_bytes()
    }

    pub fn num_rows(&self) -> usize {
        self.prepared.num_rows()
    }
}

fn host_batch_prepare_error(error: cudf::Error) -> QueryError {
    const HARD_LIMIT_REASON: &str = "host_materialization_hard_limit_exceeded";
    let hard_limit_details = (error.detail_value("reason") == Some(HARD_LIMIT_REASON)).then(|| {
        [
            ("reason", HARD_LIMIT_REASON.to_string()),
            ("row_offset", error.detail_value("row_offset").unwrap_or("unknown").to_string()),
            ("candidate_rows", error.detail_value("candidate_rows").unwrap_or("unknown").to_string()),
            (
                "estimated_materialization_bytes",
                error.detail_value("estimated_materialization_bytes").unwrap_or("unknown").to_string(),
            ),
            (
                "max_materialization_bytes",
                error.detail_value("max_materialization_bytes").unwrap_or("unknown").to_string(),
            ),
        ]
    });
    let mut error = cudf_query_error(
        "query_engine.host_batch_cursor.prepare",
        "failed to size bounded native output chunk before host materialization",
        error,
    );
    if let Some(details) = hard_limit_details {
        for (key, value) in details {
            error = error.with_fact(key, value);
        }
    }
    error
}

impl GpuHostBatchCursor {
    /// Size the next bounded host chunk without copying GPU values to host.
    pub fn prepare_next_batch(&self) -> Result<Option<GpuPreparedHostBatch>> {
        if !self.cursor.has_next_batch() {
            return Ok(None);
        }
        self.cancellation.check("query_engine.host_batch_cursor.prepare")?;
        self.cursor
            .prepare_next_batch()
            .map(|prepared| prepared.map(|prepared| GpuPreparedHostBatch { prepared }))
            .map_err(host_batch_prepare_error)
    }

    /// Materialize one prepared chunk. Cancellation is checked after the host
    /// reservation is acquired and immediately before the device-to-host copy.
    pub fn materialize_prepared_batch(&mut self, prepared: GpuPreparedHostBatch) -> Result<RecordBatch> {
        self.cancellation.check("query_engine.host_batch_cursor.materialize")?;
        let batch = self.cursor.materialize_prepared_batch(prepared.prepared).map_err(|error| {
            cudf_query_error(
                "query_engine.host_batch_cursor.materialize",
                "failed to materialize bounded native output chunk to host Arrow",
                error,
            )
        })?;
        bind_host_output_schema(batch, &self.output_schema)
    }

    /// Materialize the next bounded host chunk. Cancellation is checked before
    /// any new cuDF slice or device-to-host copy starts.
    #[cfg(test)]
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(prepared) = self.prepare_next_batch()? else {
            return Ok(None);
        };
        self.materialize_prepared_batch(prepared).map(Some)
    }

    #[cfg(test)]
    pub(crate) fn remaining_rows(&self) -> usize {
        self.cursor.remaining_rows()
    }
}

impl GpuResultHandle {
    /// Bind a GPU frame to a schema and the runtime from `context`.
    ///
    /// The frame must already reside on the context's device. Hidden
    /// DataFusion order columns may be stripped before schema binding.
    pub fn new(context: &ExecutionContext, frame: GpuDataFrame, schema: SchemaRef) -> Result<Self> {
        Self::new_with_resident_owner(context, frame, schema, None)
    }

    #[doc(hidden)]
    pub fn new_with_resident_owner(
        context: &ExecutionContext,
        frame: GpuDataFrame,
        schema: SchemaRef,
        resident_owner: Option<ResidentOwnerLease>,
    ) -> Result<Self> {
        if frame.device() != context.runtime()?.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu_result_handle",
                "native GPU result frame device does not match the execution context",
            )
            .with_fact("frame_device", frame.device().ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let frame = strip_hidden_datafusion_order_columns(frame, &schema)?;
        let GpuDataFrameParts {
            frame,
            resident_owner: frame_resident_owner,
            retained_source_owners,
            source_cache_pin,
            graph_residency,
            producer_execution_scope,
        } = frame.into_frame_and_residency();
        let resident_owner = resident_owner.or(frame_resident_owner);
        // Binding is where the producer's stream is known. Its fence makes
        // the owner's final release wait for the work that produced the
        // output, and the backing pointers let the report commit attest that
        // the frozen charge matches the registry's live allocations.
        if let Some(owner) = &resident_owner {
            owner.record_completion(context.runtime()?.stream())?;
            owner.set_backing_allocation_pointers(frame.backing_allocation_pointers().map_err(|error| {
                cudf_query_error(
                    "query_engine.gpu_result_handle",
                    "failed to identify the resident output's backing allocations",
                    error,
                )
            })?);
        }
        let batch = GpuRecordBatch::new(frame, schema).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_result_handle",
                "failed to bind native GPU result to its Arrow schema contract",
                error,
            )
        })?;
        Ok(Self {
            batch,
            runtime: context.shared_runtime()?,
            resident_owner,
            retained_source_owners,
            source_cache_pin,
            graph_residency,
            producer_execution: producer_execution_scope
                .map(|scope| Arc::new(ProducerExecutionIdentity { scope, plan: None })),
        })
    }

    pub(crate) fn retained_owner_identity_for(&self, grant: &MemoryGrant) -> Option<(usize, u64)> {
        self.resident_owner
            .as_ref()
            .filter(|owner| owner.belongs_to(grant))
            .map(|owner| (owner.identity_key(), owner.bytes()))
    }

    /// Borrow the underlying GPU cuDF frame.
    pub fn frame(&self) -> &DataFrame {
        self.batch.frame()
    }

    /// Borrows this result as a frame read on `context`'s stream. Once that
    /// frame and every frame derived from it have dropped, the resident
    /// owner's final release also waits for the work the stream had enqueued
    /// by then.
    pub fn read_on(&self, context: &ExecutionContext) -> Result<GpuDataFrame> {
        let stream = context.runtime()?.stream().clone();
        let mut frame = self.clone().into_gpu_frame();
        frame.read_on_stream(&stream);
        Ok(frame)
    }

    /// Consume the handle and return its GPU frame wrapper.
    pub fn into_gpu_frame(self) -> GpuDataFrame {
        let Self {
            batch,
            runtime: _,
            resident_owner,
            retained_source_owners,
            source_cache_pin,
            graph_residency,
            producer_execution,
        } = self;
        let mut frame = GpuDataFrame::new_with_residency(batch.into_frame(), resident_owner, source_cache_pin);
        frame.retained_source_owners = retained_source_owners;
        frame.graph_residency = graph_residency;
        frame.producer_execution_scope = producer_execution.map(|identity| identity.scope);
        frame
    }

    /// Arrow schema bound to this result.
    pub fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    /// Shared cuDF runtime that created the handle.
    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    /// Observed physical device bytes for this result handle.
    ///
    /// Source-cache promotion uses this as its pre-reservation upper bound
    /// before creating any cache-domain allocation.
    pub(crate) fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.batch.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_result_handle.device_memory",
                "failed to observe native GPU result device memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.gpu_result_handle.device_memory",
                    "GPU result memory accounting did not include device-allocated bytes",
                )
            })?;
        Ok(bytes as u64)
    }

    /// Graph source provenance carried by this handle, when known.
    pub fn graph_residency(&self) -> Option<&GraphResidencyDescriptor> {
        self.graph_residency.as_deref()
    }

    #[doc(hidden)]
    pub fn with_producer_execution_plan(mut self, plan: QueryPlan) -> Self {
        let scope = plan.execution_fingerprint();
        self.producer_execution = Some(Arc::new(ProducerExecutionIdentity { scope, plan: Some(Arc::new(plan)) }));
        self
    }

    /// Returns the opaque typed producer identity for an ownership transfer.
    #[doc(hidden)]
    pub fn producer_execution_identity(&self) -> Option<Arc<ProducerExecutionIdentity>> {
        self.producer_execution.clone()
    }

    /// Attaches an opaque typed producer identity retained before a source
    /// handle was consumed by a host transfer.
    #[doc(hidden)]
    pub fn with_producer_execution_identity(mut self, identity: Option<Arc<ProducerExecutionIdentity>>) -> Self {
        self.producer_execution = identity;
        self
    }

    pub(crate) fn producer_execution_scope(&self) -> Option<PlanFingerprint> {
        self.producer_execution.as_ref().map(|identity| identity.scope)
    }

    pub(crate) fn producer_execution_plan(&self) -> Option<&QueryPlan> {
        self.producer_execution.as_ref().and_then(|identity| identity.plan.as_deref())
    }

    /// Copies this attempt-owned source frame into the supplied cache-domain
    /// lease. The source handle is borrowed and remains owned by the attempt
    /// on every success, cancellation, or allocation-failure path.
    pub(crate) fn copy_for_local_parquet_cache(
        &self,
        allocation_lease: Arc<cudf::CacheAllocationLease>,
    ) -> Result<GpuLocalParquetCachedResult> {
        const OPERATION: &str = "query_engine.gpu_local_parquet_source_cache.copy_for_cache";
        if self.source_cache_pin.is_some() || self.graph_residency.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache input already carries non-attempt cache residency",
            )
            .with_fact("reason", "non_attempt_source_residency_in_cache_input"));
        }
        if self.runtime.device() != allocation_lease.domain().device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache allocation domain is on another CUDA device",
            )
            .with_fact("source_device", self.runtime.device().ordinal())
            .with_fact("cache_domain_device", allocation_lease.domain().device().ordinal()));
        }
        if self.runtime.binding_id() == allocation_lease.runtime().binding_id() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache allocation lease must use an independent execution binding",
            )
            .with_fact("reason", "local_parquet_cache_runtime_binding_not_independent"));
        }
        let frame = self.batch.frame().copy_into_runtime(allocation_lease.runtime()).map_err(|error| {
            cudf_query_error(OPERATION, "failed to copy attempt-owned local Parquet source into cache ownership", error)
        })?;
        let batch = GpuRecordBatch::new(frame, Arc::clone(self.schema())).map_err(|error| {
            cudf_query_error(OPERATION, "failed to bind query-neutral local Parquet cache value", error)
        })?;
        Ok(GpuLocalParquetCachedResult { batch, allocation_lease })
    }

    /// Current cuDF execution placement for the underlying batch.
    pub fn placement(&self) -> ExecutionPlacement {
        self.batch.placement()
    }

    /// Row count in the GPU result.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// Column count in the GPU result.
    pub fn num_columns(&self) -> usize {
        self.batch.num_columns()
    }

    /// Materialize this GPU result to host Arrow.
    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        self.batch.to_host_record_batch().map_err(|error| {
            cudf_query_error(
                "query_engine.host_arrow.materialize",
                "failed to materialize algeon native output to host Arrow",
                error,
            )
        })
    }

    /// Consume this result into a bounded incremental host materializer.
    pub fn into_host_batch_cursor(
        self,
        cancellation: ExecutionCancellation,
        config: HostBatchCursorConfig,
    ) -> Result<GpuHostBatchCursor> {
        let output_schema = Arc::clone(self.schema());
        self.into_host_batch_cursor_with_bound_schema(output_schema, cancellation, config)
    }

    /// Consume this result into a bounded host materializer using an external
    /// schema whose field names, nullability, and metadata are bound by ordinal.
    pub fn into_host_batch_cursor_with_host_schema(
        self,
        output_schema: SchemaRef,
        cancellation: ExecutionCancellation,
        config: HostBatchCursorConfig,
    ) -> Result<GpuHostBatchCursor> {
        validate_host_output_schema_binding(self.schema(), &output_schema)?;
        self.into_host_batch_cursor_with_bound_schema(output_schema, cancellation, config)
    }

    fn into_host_batch_cursor_with_bound_schema(
        self,
        output_schema: SchemaRef,
        cancellation: ExecutionCancellation,
        config: HostBatchCursorConfig,
    ) -> Result<GpuHostBatchCursor> {
        let Self {
            batch,
            runtime,
            resident_owner,
            retained_source_owners,
            source_cache_pin,
            graph_residency,
            producer_execution: _,
        } = self;
        // The cursor gathers its batches on the result's stream after the
        // owner bound it.
        let stream = runtime.stream().clone();
        let resident_owner = resident_owner.map(|owner| owner.for_reader(stream.clone()));
        let retained_source_owners =
            retained_source_owners.into_iter().map(|retained| retained.for_reader(&stream)).collect();
        let cursor = batch.into_host_batch_cursor(config).map_err(|error| {
            cudf_query_error(
                "query_engine.host_batch_cursor.create",
                "failed to create bounded native host-output cursor",
                error,
            )
        })?;
        Ok(GpuHostBatchCursor {
            cursor,
            output_schema,
            cancellation,
            _runtime: runtime,
            _resident_owner: resident_owner,
            _retained_source_owners: retained_source_owners,
            _source_cache_pin: source_cache_pin,
            _graph_residency: graph_residency,
        })
    }
}

fn validate_host_output_schema_binding(internal_schema: &SchemaRef, output_schema: &SchemaRef) -> Result<()> {
    const OPERATION: &str = "query_engine.host_output_schema_binding";
    if internal_schema.fields().len() != output_schema.fields().len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            OPERATION,
            "internal executable schema and external host schema have different field counts",
        )
        .with_fact("reason", "host_output_field_count_mismatch")
        .with_fact("internal_field_count", internal_schema.fields().len())
        .with_fact("external_field_count", output_schema.fields().len()));
    }

    for (ordinal, (internal, external)) in
        internal_schema.fields().iter().zip(output_schema.fields().iter()).enumerate()
    {
        if internal.data_type() != external.data_type() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "internal executable field and external host field have different Arrow types",
            )
            .with_fact("reason", "host_output_field_type_mismatch")
            .with_fact("ordinal", ordinal)
            .with_fact("internal_field", internal.name())
            .with_fact("external_field", external.name())
            .with_fact("internal_arrow_type", format!("{:?}", internal.data_type()))
            .with_fact("external_arrow_type", format!("{:?}", external.data_type())));
        }
    }
    Ok(())
}

fn bind_host_output_schema(batch: RecordBatch, output_schema: &SchemaRef) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(Arc::clone(output_schema), batch.columns().to_vec(), &options).map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.host_output_schema_binding",
            "failed to apply the validated external host schema to a materialized native output batch",
        )
        .with_fact("reason", "host_output_schema_projection_failed")
        .with_source(error)
    })
}

impl GpuLocalParquetCachedResult {
    pub(crate) fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    /// Completes work submitted on the cache lease before the cached value is
    /// destroyed. Cache code must call this before releasing its ledger charge.
    pub(crate) fn synchronize_for_destruction(&self) -> Result<()> {
        self.allocation_lease.runtime().synchronize().map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_local_parquet_source_cache.destroy",
                "failed to synchronize the cache allocation lease before source-cache destruction",
                error,
            )
        })
    }

    pub(crate) fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.batch.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_local_parquet_source_cache.device_memory",
                "failed to inspect cached local Parquet device memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.gpu_local_parquet_source_cache.device_memory",
                    "cached local Parquet memory accounting did not include device-allocated bytes",
                )
            })?;
        Ok(bytes as u64)
    }

    /// Copies the cached value into `context`'s runtime, so the attempt
    /// computes on memory its own allocation domain owns and the cache can
    /// account its bytes independently of any attempt. The returned handle
    /// holds `pin` for as long as the copied frame exists.
    pub(crate) fn checkout(
        &self,
        context: &ExecutionContext,
        pin: GpuLocalParquetSourceCachePin,
    ) -> Result<GpuResultHandle> {
        const OPERATION: &str = "query_engine.gpu_local_parquet_source_cache.checkout";
        let query_runtime = context.shared_runtime()?;
        if query_runtime.device() != self.allocation_lease.domain().device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache entry does not match the current query device placement",
            )
            .with_fact("cache_domain_device", self.allocation_lease.domain().device().ordinal())
            .with_fact("query_device", query_runtime.device().ordinal()));
        }
        let frame = self.batch.frame().copy_into_runtime(&query_runtime).map_err(|error| {
            cudf_query_error(OPERATION, "failed to copy cached local Parquet source into current attempt", error)
        })?;
        let batch = GpuRecordBatch::new(frame, Arc::clone(self.schema())).map_err(|error| {
            cudf_query_error(OPERATION, "failed to bind current-query local Parquet cache checkout", error)
        })?;
        Ok(GpuResultHandle {
            batch,
            runtime: query_runtime,
            resident_owner: None,
            retained_source_owners: Vec::new(),
            source_cache_pin: Some(pin),
            graph_residency: None,
            producer_execution: None,
        })
    }
}

/// Drops trailing DataFusion order columns the schema does not declare. The
/// resident owner is kept whole because the selection aliases the same
/// buffers; the transitive source owners are narrowed to the columns that
/// remain.
fn strip_hidden_datafusion_order_columns(frame: GpuDataFrame, schema: &SchemaRef) -> Result<GpuDataFrame> {
    if frame.num_columns() == schema.fields().len() {
        return Ok(frame);
    }

    let GpuDataFrameParts {
        frame: raw,
        resident_owner,
        retained_source_owners,
        source_cache_pin,
        graph_residency,
        producer_execution_scope,
    } = frame.into_frame_and_residency();
    let frame_names = raw.column_names();
    let expected_names = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    let visible_prefix_matches =
        frame_names.iter().take(expected_names.len()).copied().eq(expected_names.iter().copied());
    let trailing_are_order_columns =
        frame_names.iter().skip(expected_names.len()).all(|name| is_datafusion_order_column_name(name));
    if visible_prefix_matches && trailing_are_order_columns {
        let selected = raw.select(expected_names).map_err(|error| {
            cudf_query_error(
                "query_engine.gpu_result_handle",
                "failed to strip hidden native DataFusion order columns before binding result schema",
                error,
            )
        })?;
        let selected_identities = source_column_memory_identities(&selected);
        let retained_source_owners =
            retained_source_owners.iter().filter_map(|owner| owner.retain_for(&selected_identities)).collect();
        let mut frame = GpuDataFrame::new_with_residency(selected, resident_owner, source_cache_pin);
        frame.retained_source_owners = retained_source_owners;
        frame.graph_residency = graph_residency;
        frame.producer_execution_scope = producer_execution_scope;
        return Ok(frame);
    }

    let mut frame = GpuDataFrame::new_with_residency(raw, resident_owner, source_cache_pin);
    frame.retained_source_owners = retained_source_owners;
    frame.graph_residency = graph_residency;
    frame.producer_execution_scope = producer_execution_scope;
    Ok(frame)
}

/// Reusable cuDF hash-join build artifact.
///
/// The handle is runtime/device-affine and carries the retained hash-table byte
/// contract needed to reject unsafe reuse. Cloning shares the underlying cuDF
/// artifact.
#[derive(Clone)]
pub struct GpuHashJoinBuildHandle {
    hash_join: DataFrameHashJoin,
    runtime: Arc<cudf::Runtime>,
    device_allocated_size_bytes: u64,
}

/// Reusable cuDF filtered-join build artifact.
///
/// Like [`GpuHashJoinBuildHandle`], this is runtime/device-affine and reports
/// observed retained device bytes for cache accounting.
#[derive(Clone)]
pub struct GpuFilteredJoinBuildHandle {
    filtered_join: DataFrameFilteredJoin,
    runtime: Arc<cudf::Runtime>,
    device_allocated_size_bytes: u64,
}

impl GpuFilteredJoinBuildHandle {
    #[doc(hidden)]
    pub fn new(
        context: &ExecutionContext,
        filtered_join: DataFrameFilteredJoin,
        null_equality: JoinNullEquality,
    ) -> Result<Self> {
        if filtered_join.build().device() != context.runtime()?.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu_filtered_join_build_handle",
                "native GPU filtered-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", filtered_join.build().device().ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let device_allocated_size_bytes =
            filtered_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                cudf_query_error(
                    "query_engine.gpu_filtered_join_build_handle",
                    "failed to observe algeon native reusable filtered join memory",
                    error,
                )
                .with_fact("key_count", filtered_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        Ok(Self { filtered_join, runtime: context.shared_runtime()?, device_allocated_size_bytes })
    }

    #[doc(hidden)]
    pub fn filtered_join(&self) -> &DataFrameFilteredJoin {
        &self.filtered_join
    }

    /// Shared cuDF runtime that created this reusable build artifact.
    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    /// Current cuDF execution placement for the retained build side.
    pub fn placement(&self) -> ExecutionPlacement {
        self.filtered_join.build().placement()
    }

    /// Observed retained device bytes for this artifact.
    pub fn device_allocated_size_bytes(&self) -> u64 {
        self.device_allocated_size_bytes
    }
}

impl GpuHashJoinBuildHandle {
    #[doc(hidden)]
    pub fn new(
        context: &ExecutionContext,
        hash_join: DataFrameHashJoin,
        null_equality: JoinNullEquality,
    ) -> Result<Self> {
        if hash_join.build().device() != context.runtime()?.device() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu_hash_join_build_handle",
                "native GPU hash-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", hash_join.build().device().ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let device_allocated_size_bytes =
            hash_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                cudf_query_error(
                    "query_engine.gpu_hash_join_build_handle",
                    "failed to observe algeon native reusable hash join memory",
                    error,
                )
                .with_fact("key_count", hash_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        Ok(Self { hash_join, runtime: context.shared_runtime()?, device_allocated_size_bytes })
    }

    #[doc(hidden)]
    pub fn hash_join(&self) -> &DataFrameHashJoin {
        &self.hash_join
    }

    /// Shared cuDF runtime that created this reusable build artifact.
    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    /// Current cuDF execution placement for the retained build side.
    pub fn placement(&self) -> ExecutionPlacement {
        self.hash_join.build().placement()
    }

    /// Observed retained device bytes for this artifact.
    pub fn device_allocated_size_bytes(&self) -> u64 {
        self.device_allocated_size_bytes
    }
}

fn source_column_memory_identities(frame: &DataFrame) -> HashSet<ColumnMemoryIdentity> {
    frame.memory_identity().columns.into_iter().map(|column| column.identity).collect()
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow::{
        array::{Array, Int32Array, StringArray, UInt64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::{DataFrame, Exec, ops::expr::col};
    use query_planning::error::{QueryErrorCode, QueryErrorKind, QueryErrorStatus};

    use super::*;
    use crate::{
        observability::AdmissionRequestId,
        runtime::{
            StandaloneConfig,
            gpu::ledger::{DeviceLedger, MemoryGrant},
        },
    };

    fn single_i32_frame(runtime: &Arc<cudf::Runtime>, name: &str, values: Vec<i32>) -> DataFrame {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap();
        DataFrame::from_record_batch(runtime, &batch).unwrap()
    }

    fn frame_with_resident_owner(
        runtime: &Arc<cudf::Runtime>,
        grant: &MemoryGrant,
        name: &str,
        values: Vec<i32>,
    ) -> (GpuDataFrame, u64) {
        let frame = single_i32_frame(runtime, name, values);
        let observed_bytes = GpuDataFrame::new(frame.clone()).device_allocated_size_bytes().unwrap();
        let resident_owner = grant.retain_resident_owner(observed_bytes).expect("test frame owner is recorded");
        (GpuDataFrame::new_with_resident_owner(frame, resident_owner), observed_bytes)
    }

    #[test]
    fn gpu_frame_counts_resident_bytes_only_for_the_owning_grant() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 2048, 4);
        let owner = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(AdmissionRequestId::new(1), 1, 1024)
            .unwrap()
            .expect("owner grant must fit");
        let foreign = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(AdmissionRequestId::new(2), 1, 1024)
            .unwrap()
            .expect("foreign grant must fit");
        let (frame, bytes) = frame_with_resident_owner(&runtime, &owner, "id", vec![1, 2, 3]);

        assert_eq!(frame.resident_owner_bytes_for(Some(&owner)), bytes);
        assert_eq!(frame.resident_owner_bytes_for(Some(&foreign)), 0);
        assert_eq!(frame.resident_owner_bytes_for(None), 0);
    }

    #[test]
    fn retained_guard_keeps_gpu_result_handle_send_safe() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<GpuResultHandle>();
    }

    #[test]
    fn gpu_result_handle_retained_bytes_require_the_current_grant() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 2048, 4);
        let owner = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(AdmissionRequestId::new(11), 1, 1024)
            .unwrap()
            .expect("owner grant must fit");
        let foreign = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(AdmissionRequestId::new(12), 1, 1024)
            .unwrap()
            .expect("foreign grant must fit");
        let (frame, bytes) = frame_with_resident_owner(&runtime, &owner, "id", vec![1, 2, 3]);
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let handle = GpuResultHandle::new(&context, frame, Arc::clone(&schema)).unwrap();

        assert_eq!(handle.retained_owner_identity_for(&owner).map(|(_, value)| value), Some(bytes));
        assert_eq!(handle.retained_owner_identity_for(&foreign), None);

        let fresh =
            GpuResultHandle::new(&context, GpuDataFrame::new(single_i32_frame(&runtime, "id", vec![4, 5, 6])), schema)
                .unwrap();
        assert_eq!(fresh.retained_owner_identity_for(&owner), None);
        drop(fresh);
        assert_eq!(owner.resident_owner_bytes(), bytes);
        drop(handle);
        assert_eq!(owner.resident_owner_bytes(), 0);
    }

    #[test]
    fn host_batch_cursor_observes_cancellation_before_starting_the_next_device_to_host_copy() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            cancellation: cancellation.clone(),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let frame = GpuDataFrame::new(single_i32_frame(&runtime, "id", vec![1, 2, 3]));
        let handle = GpuResultHandle::new(&context, frame, schema).unwrap();
        let mut cursor = handle
            .into_host_batch_cursor(
                cancellation.clone(),
                HostBatchCursorConfig { target_materialization_bytes: 1024, max_rows: 1, ..Default::default() },
            )
            .unwrap();

        assert_eq!(cursor.next_batch().unwrap().unwrap().num_rows(), 1);
        cancellation.cancel();
        let error = cursor.next_batch().expect_err("cancellation must stop the next host chunk");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(cursor.remaining_rows(), 2);
    }

    #[test]
    fn host_output_schema_binding_preserves_duplicate_names_metadata_and_values() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let internal_schema = Arc::new(Schema::new(vec![
            Field::new("left_first", DataType::Utf8, true),
            Field::new("right_first", DataType::Utf8, true),
            Field::new("shared_targets_internal", DataType::UInt64, false),
        ]));
        let input = RecordBatch::try_new(
            Arc::clone(&internal_schema),
            vec![
                Arc::new(StringArray::from(vec![Some("a"), None])) as Arc<dyn Array>,
                Arc::new(StringArray::from(vec![Some("b"), Some("c")])) as Arc<dyn Array>,
                Arc::new(UInt64Array::from(vec![2, 1])) as Arc<dyn Array>,
            ],
        )
        .unwrap();
        let field_metadata = HashMap::from([("source".to_string(), "datafusion".to_string())]);
        let output_schema = Arc::new(
            Schema::new(vec![
                Field::new("first", DataType::Utf8, true).with_metadata(field_metadata),
                Field::new("first", DataType::Utf8, false),
                Field::new("shared_targets", DataType::UInt64, true),
            ])
            .with_metadata(HashMap::from([("query".to_string(), "self_join".to_string())])),
        );
        let frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &input).unwrap());
        let handle = GpuResultHandle::new(&context, frame, internal_schema).unwrap();
        let mut cursor = handle
            .into_host_batch_cursor_with_host_schema(
                Arc::clone(&output_schema),
                ExecutionCancellation::new(),
                HostBatchCursorConfig::default(),
            )
            .unwrap();

        let output = cursor.next_batch().unwrap().unwrap();
        assert_eq!(output.schema(), output_schema);
        assert_eq!(output.column(0).as_any().downcast_ref::<StringArray>().unwrap().value(0), "a");
        assert!(output.column(0).as_any().downcast_ref::<StringArray>().unwrap().is_null(1));
        assert_eq!(output.column(1).as_any().downcast_ref::<StringArray>().unwrap().value(1), "c");
        assert_eq!(output.column(2).as_any().downcast_ref::<UInt64Array>().unwrap().values(), &[2, 1]);
    }

    #[test]
    fn host_output_schema_binding_rejects_count_and_type_mismatches_with_ordinal_facts() {
        let internal = Arc::new(Schema::new(vec![
            Field::new("left_first", DataType::Utf8, true),
            Field::new("shared_targets_internal", DataType::UInt64, false),
        ]));
        let short = Arc::new(Schema::new(vec![Field::new("first", DataType::Utf8, true)]));
        let count_error = validate_host_output_schema_binding(&internal, &short).unwrap_err();
        assert_eq!(count_error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(count_error.fact_value("reason"), Some("host_output_field_count_mismatch"));
        assert_eq!(count_error.fact_value("internal_field_count"), Some("2"));
        assert_eq!(count_error.fact_value("external_field_count"), Some("1"));

        let reordered = Arc::new(Schema::new(vec![
            Field::new("shared_targets", DataType::UInt64, false),
            Field::new("first", DataType::Utf8, true),
        ]));
        let type_error = validate_host_output_schema_binding(&internal, &reordered).unwrap_err();
        assert_eq!(type_error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(type_error.fact_value("reason"), Some("host_output_field_type_mismatch"));
        assert_eq!(type_error.fact_value("ordinal"), Some("0"));
        assert_eq!(type_error.fact_value("internal_field"), Some("left_first"));
        assert_eq!(type_error.fact_value("external_field"), Some("shared_targets"));
        assert_eq!(type_error.fact_value("internal_arrow_type"), Some("Utf8"));
        assert_eq!(type_error.fact_value("external_arrow_type"), Some("UInt64"));
    }

    #[test]
    fn host_batch_cursor_preserves_structured_hard_limit_facts() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let value = "x".repeat(16 * 1024 * 1024);
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec![value.as_str()])) as Arc<dyn Array>],
        )
        .unwrap();
        let frame = GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap());
        let handle = GpuResultHandle::new(&context, frame, schema).unwrap();
        let cursor = handle.into_host_batch_cursor(cancellation, HostBatchCursorConfig::default()).unwrap();

        let error = cursor.prepare_next_batch().expect_err("oversized scalar must reject before D2H");
        assert_eq!(error.code(), QueryErrorCode::HostMaterializationLimitExceeded);
        assert_eq!(error.kind(), QueryErrorKind::ResourceExhausted);
        assert_eq!(error.status(), QueryErrorStatus::Permanent);
        assert_eq!(error.fact_value("reason"), Some("host_materialization_hard_limit_exceeded"));
        assert_eq!(error.fact_value("candidate_rows"), Some("1"));
        assert_eq!(
            error.fact_value("max_materialization_bytes"),
            Some(HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES.to_string().as_str())
        );
        assert!(
            error.fact_value("estimated_materialization_bytes").unwrap().parse::<usize>().unwrap()
                > HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES
        );
    }

    #[test]
    fn host_batch_cursor_reports_clean_eof_when_cancelled_after_final_chunk() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            cancellation: cancellation.clone(),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let frame = GpuDataFrame::new(single_i32_frame(&runtime, "id", vec![1]));
        let handle = GpuResultHandle::new(&context, frame, schema).unwrap();
        let mut cursor = handle
            .into_host_batch_cursor(
                cancellation.clone(),
                HostBatchCursorConfig { target_materialization_bytes: 1024, max_rows: 1, ..Default::default() },
            )
            .unwrap();

        assert_eq!(cursor.next_batch().unwrap().unwrap().num_rows(), 1);
        assert_eq!(cursor.remaining_rows(), 0);
        cancellation.cancel();
        assert!(cursor.next_batch().unwrap().is_none());
    }

    /// E3: the deferred host cursor prepares each bounded chunk inside the existing sink
    /// contract without declaring success early. Every prepared chunk materializes exactly
    /// once, the cursor reports more work until the final chunk drains, and dropping the
    /// cursor after a partial drain releases its retained owners instead of leaking a grant.
    #[test]
    fn host_batch_cursor_prepares_each_chunk_without_early_success() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            cancellation: cancellation.clone(),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let frame = GpuDataFrame::new(single_i32_frame(&runtime, "id", vec![1, 2, 3]));
        let handle = GpuResultHandle::new(&context, frame, schema).unwrap();
        let mut cursor = handle
            .into_host_batch_cursor(
                cancellation,
                HostBatchCursorConfig { target_materialization_bytes: 1024, max_rows: 1, ..Default::default() },
            )
            .unwrap();

        let mut rows = Vec::new();
        for expected in [1, 2, 3] {
            assert_eq!(cursor.remaining_rows(), 4 - expected, "unprepared chunks are not success");
            let prepared = cursor.prepare_next_batch().expect("chunk prepares").expect("chunk remains");
            assert_eq!(prepared.num_rows(), 1);
            let batch = cursor.materialize_prepared_batch(prepared).expect("prepared chunk materializes");
            assert_eq!(batch.num_rows(), 1);
            rows.push(batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap().value(0));
        }
        assert!(cursor.prepare_next_batch().expect("final prepare observes").is_none());
        assert_eq!(cursor.remaining_rows(), 0);
        rows.sort();
        assert_eq!(rows, vec![1, 2, 3], "every chunk materializes exactly once");
    }

    #[test]
    fn host_batch_cursor_prepare_reports_cancellation_before_any_copy() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::standalone(StandaloneConfig {
            runtime: Some(Arc::clone(&runtime)),
            cancellation: cancellation.clone(),
            ..StandaloneConfig::default()
        })
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let frame = GpuDataFrame::new(single_i32_frame(&runtime, "id", vec![1, 2, 3]));
        let handle = GpuResultHandle::new(&context, frame, schema).unwrap();
        let cursor = handle
            .into_host_batch_cursor(
                cancellation.clone(),
                HostBatchCursorConfig { target_materialization_bytes: 1024, max_rows: 1, ..Default::default() },
            )
            .unwrap();
        cancellation.cancel();
        let error = cursor.prepare_next_batch().expect_err("cancelled prepare must not size a chunk");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
    }

    #[test]
    fn source_residency_clones_owner_for_pure_alias_output() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 128 * 1024 * 1024, 4);
        let grant = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(
                AdmissionRequestId::new(1),
                64 * 1024 * 1024,
                64 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        let (source, source_bytes) = frame_with_resident_owner(&runtime, &grant, "id", vec![1, 2, 3]);
        assert!(source_bytes > 0);
        assert_eq!(grant.resident_owner_bytes(), source_bytes);

        let selected = source.frame().select(["id"]).unwrap();
        let derived = GpuDataFrame::try_new_with_source_residency(selected, &source).unwrap();
        assert_eq!(grant.resident_owner_bytes(), source_bytes);

        drop(source);
        assert_eq!(grant.resident_owner_bytes(), source_bytes);
        drop(derived);
        assert_eq!(grant.resident_owner_bytes(), 0);
    }

    #[test]
    fn source_residency_records_derived_owner_for_fresh_output() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 128 * 1024 * 1024, 4);
        let grant = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(
                AdmissionRequestId::new(2),
                64 * 1024 * 1024,
                64 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        let (source, source_bytes) = frame_with_resident_owner(&runtime, &grant, "id", vec![1, 2, 3]);
        let fresh = single_i32_frame(&runtime, "fresh", vec![10, 20, 30]);
        let fresh_bytes = GpuDataFrame::new(fresh.clone()).device_allocated_size_bytes().unwrap();
        assert!(fresh_bytes > 0);

        let derived = GpuDataFrame::try_new_with_source_residency(fresh, &source).unwrap();
        assert_eq!(grant.resident_owner_bytes(), source_bytes.saturating_add(fresh_bytes));

        drop(source);
        assert_eq!(grant.resident_owner_bytes(), fresh_bytes);
        drop(derived);
        assert_eq!(grant.resident_owner_bytes(), 0);
    }

    #[test]
    fn source_residency_retains_full_output_for_mixed_alias_output() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 128 * 1024 * 1024, 4);
        let grant = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(
                AdmissionRequestId::new(3),
                64 * 1024 * 1024,
                64 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        let (source, source_bytes) = frame_with_resident_owner(&runtime, &grant, "id", vec![1, 2, 3]);
        let mixed = source.frame().on(Exec::RETAINED).with_columns([("fresh", col("id") + 1i32)]).unwrap();
        let fresh_only = mixed.select(["fresh"]).unwrap();
        let fresh_bytes = GpuDataFrame::new(fresh_only).device_allocated_size_bytes().unwrap();
        let mixed_bytes = GpuDataFrame::new(mixed.clone()).device_allocated_size_bytes().unwrap();
        assert!(fresh_bytes > 0);
        assert!(
            mixed_bytes > fresh_bytes,
            "mixed output should include both the aliased source column and the fresh derived column"
        );

        let derived = GpuDataFrame::try_new_with_source_residency(mixed, &source).unwrap();
        assert_eq!(grant.resident_owner_bytes(), source_bytes.saturating_add(mixed_bytes));

        drop(source);
        assert_eq!(grant.resident_owner_bytes(), mixed_bytes);
        drop(derived);
        assert_eq!(grant.resident_owner_bytes(), 0);
    }

    #[test]
    fn source_residency_preserves_transitive_frozen_owner_across_alias_chain() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 128 * 1024 * 1024, 4);
        let grant = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(
                AdmissionRequestId::new(4),
                64 * 1024 * 1024,
                64 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        let (source, source_bytes) = frame_with_resident_owner(&runtime, &grant, "id", vec![1, 2, 3]);
        source.resident_owner.as_ref().unwrap().record_completion(runtime.stream()).unwrap();
        grant.freeze_resident(source_bytes).unwrap();

        let mixed = source.frame().on(Exec::RETAINED).with_columns([("fresh", col("id") + 1i32)]).unwrap();
        let mixed = GpuDataFrame::try_new_with_source_residency(mixed, &source).unwrap();
        let selected = mixed.frame().select(["id"]).unwrap();
        let selected = GpuDataFrame::try_new_with_source_residency(selected, &mixed).unwrap();

        drop(source);
        drop(mixed);
        assert_eq!(grant.resident_owner_bytes(), source_bytes);

        drop(selected);
        assert_eq!(grant.resident_owner_bytes(), 0);
    }

    #[test]
    fn source_residency_releases_transitive_owner_when_only_fresh_columns_survive() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = DeviceLedger::for_test(runtime.device().ordinal(), 128 * 1024 * 1024, 4);
        let grant = governor
            .admit_with_configured_floor_and_soft_working_set_hint_for_test(
                AdmissionRequestId::new(5),
                64 * 1024 * 1024,
                64 * 1024 * 1024,
            )
            .unwrap()
            .unwrap();
        let (source, source_bytes) = frame_with_resident_owner(&runtime, &grant, "id", vec![1, 2, 3]);
        source.resident_owner.as_ref().unwrap().record_completion(runtime.stream()).unwrap();
        grant.freeze_resident(source_bytes).unwrap();

        let mixed = source.frame().on(Exec::RETAINED).with_columns([("fresh", col("id") + 1i32)]).unwrap();
        let mixed = GpuDataFrame::try_new_with_source_residency(mixed, &source).unwrap();
        let selected = mixed.frame().select(["fresh"]).unwrap();
        let selected = GpuDataFrame::try_new_with_source_residency(selected, &mixed).unwrap();

        drop(source);
        drop(mixed);
        assert_eq!(grant.resident_owner_bytes(), 0);

        drop(selected);
    }
}
