//! Frontend-neutral GPU-resident data, result, and reusable build handles.
//!
//! Handles in this module are runtime/device-affine. Cloning shares the
//! underlying native resources; host materialization is explicit.

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, ExecutionPlacement,
    frame::dataframe::{
        ColumnMemoryIdentity, GpuRecordBatch, HostBatchCursor as CudfHostBatchCursor, HostBatchCursorConfig,
        InfoDetail, PreparedHostBatch as CudfPreparedHostBatch,
    },
};

use crate::{
    error::{QueryError, Result},
    naming::is_datafusion_order_column_name,
    plan::{GraphResidencyDescriptor, JoinNullEquality},
    runtime::{
        DevicePlacement, ExecutionCancellation, ExecutionContext,
        cache::local_parquet::GpuLocalParquetSourceCachePin,
        gpu::ledger::{MemoryGrant, MemoryGrantLiveReservation},
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
    live_reservation: Option<MemoryGrantLiveReservation>,
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    graph_residency: Option<Arc<GraphResidencyDescriptor>>,
}

impl GpuDataFrame {
    /// Wrap a cuDF frame and record its current execution placement.
    pub fn new(frame: DataFrame) -> Self {
        Self::new_with_live_reservation(frame, None)
    }

    /// Wrap a derived cuDF frame with identity-aware live-byte accounting and
    /// the source cache's transitive eviction-safety lifetime.
    pub fn try_new_with_source_residency(frame: DataFrame, source: &Self) -> Result<Self> {
        if source.live_reservation.is_none() && source.source_cache_pin.is_none() {
            return Ok(Self::new(frame));
        }
        let source_identities = source_column_memory_identities(&source.frame);
        let output_identity = frame.memory_identity();
        let aliased_output_columns =
            output_identity.columns.iter().filter(|column| source_identities.contains(&column.identity)).count();
        let aliases_source = aliased_output_columns > 0;
        let all_output_columns_alias_source = aliases_source && aliased_output_columns == output_identity.columns.len();
        let live_reservation = match &source.live_reservation {
            Some(source_live_reservation) if all_output_columns_alias_source => {
                // Pure aliases, including column subsets, keep the source
                // reservation alive instead of remeasuring a partial owner set.
                Some(source_live_reservation.clone())
            },
            Some(source_live_reservation) => {
                // Mixed alias/fresh outputs need one retention handle to cover the
                // whole escaping frame. This intentionally over-accounts aliased
                // columns while the input is alive, but avoids under-accounting
                // alias owners after the input wrapper drops.
                let observed_bytes = Self::new(frame.clone()).device_allocated_size_bytes()?;
                source_live_reservation.retain_derived_live_reservation(observed_bytes)?
            },
            None => None,
        };
        Ok(Self::new_with_residency(frame, live_reservation, source.source_cache_pin.clone()))
    }

    fn new_with_live_reservation(frame: DataFrame, live_reservation: Option<MemoryGrantLiveReservation>) -> Self {
        Self::new_with_residency(frame, live_reservation, None)
    }

    fn new_with_residency(
        frame: DataFrame,
        live_reservation: Option<MemoryGrantLiveReservation>,
        source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    ) -> Self {
        let placement = frame.placement();
        Self { frame, placement, live_reservation, source_cache_pin, graph_residency: None }
    }

    /// Borrow the underlying cuDF frame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    /// Consume the wrapper and return the underlying cuDF frame.
    pub fn into_frame(self) -> DataFrame {
        let (frame, _) = self.into_frame_and_live_reservation();
        frame
    }

    fn into_frame_and_live_reservation(self) -> (DataFrame, Option<MemoryGrantLiveReservation>) {
        let (frame, live_reservation, _source_cache_pin, _graph_residency) = self.into_frame_and_residency();
        (frame, live_reservation)
    }

    fn into_frame_and_residency(
        self,
    ) -> (
        DataFrame,
        Option<MemoryGrantLiveReservation>,
        Option<GpuLocalParquetSourceCachePin>,
        Option<Arc<GraphResidencyDescriptor>>,
    ) {
        debug_assert_eq!(self.placement, self.frame.placement());
        (self.frame, self.live_reservation, self.source_cache_pin, self.graph_residency)
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

    /// Execution placement observed when the frame entered the engine.
    pub fn placement(&self) -> ExecutionPlacement {
        self.placement
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
            QueryError::from_cudf(
                "nexus_query_engine.gpu_dataframe.device_memory",
                "failed to observe native GPU dataframe memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "nexus_query_engine.gpu_dataframe.device_memory",
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
    pub(crate) fn live_reservation_bytes_for(&self, grant: Option<&MemoryGrant>) -> u64 {
        match (self.live_reservation.as_ref(), grant) {
            (Some(reservation), Some(grant)) if reservation.belongs_to(grant) => reservation.bytes(),
            _ => 0,
        }
    }
}

/// GPU-resident query result bound to an Arrow schema and runtime.
///
/// Cloning shares the underlying cuDF batch/runtime handle. A handle is only
/// reusable with the same runtime and device placement that created it; host
/// materialization is explicit through [`to_host_record_batch`](Self::to_host_record_batch).
#[derive(Clone)]
pub struct GpuResultHandle {
    batch: GpuRecordBatch,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
    live_reservation: Option<MemoryGrantLiveReservation>,
    source_cache_pin: Option<GpuLocalParquetSourceCachePin>,
    graph_residency: Option<Arc<GraphResidencyDescriptor>>,
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
    device_placement: DevicePlacement,
}

/// Bounded native GPU-to-host cursor retaining every owner required by the
/// resident result until the final chunk is materialized or the cursor drops.
pub struct GpuHostBatchCursor {
    cursor: CudfHostBatchCursor,
    cancellation: ExecutionCancellation,
    _runtime: Arc<cudf::Runtime>,
    _device_placement: DevicePlacement,
    _live_reservation: Option<MemoryGrantLiveReservation>,
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
    let mut error = QueryError::from_cudf(
        "nexus_query_engine.host_batch_cursor.prepare",
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
        self.cancellation.check("nexus_query_engine.host_batch_cursor.prepare")?;
        self.cursor
            .prepare_next_batch()
            .map(|prepared| prepared.map(|prepared| GpuPreparedHostBatch { prepared }))
            .map_err(host_batch_prepare_error)
    }

    /// Materialize one prepared chunk. Cancellation is checked after the host
    /// reservation is acquired and immediately before the device-to-host copy.
    pub fn materialize_prepared_batch(&mut self, prepared: GpuPreparedHostBatch) -> Result<RecordBatch> {
        self.cancellation.check("nexus_query_engine.host_batch_cursor.materialize")?;
        self.cursor.materialize_prepared_batch(prepared.prepared).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.host_batch_cursor.materialize",
                "failed to materialize bounded native output chunk to host Arrow",
                error,
            )
        })
    }

    /// Materialize the next bounded host chunk. Cancellation is checked before
    /// any new cuDF slice or device-to-host copy starts.
    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(prepared) = self.prepare_next_batch()? else {
            return Ok(None);
        };
        self.materialize_prepared_batch(prepared).map(Some)
    }

    pub fn remaining_rows(&self) -> usize {
        self.cursor.remaining_rows()
    }
}

impl GpuResultHandle {
    /// Bind a GPU frame to a schema and the runtime from `context`.
    ///
    /// The frame must already reside on the context's device. Hidden
    /// DataFusion order columns may be stripped before schema binding.
    pub fn new(context: &ExecutionContext, frame: GpuDataFrame, schema: SchemaRef) -> Result<Self> {
        Self::new_with_live_reservation(context, frame, schema, None)
    }

    pub(crate) fn new_with_live_reservation(
        context: &ExecutionContext,
        frame: GpuDataFrame,
        schema: SchemaRef,
        live_reservation: Option<MemoryGrantLiveReservation>,
    ) -> Result<Self> {
        if frame.placement().device != context.runtime()?.device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_result_handle",
                "native GPU result frame device does not match the execution context",
            )
            .with_fact("frame_device", frame.placement().device.ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let frame = strip_hidden_datafusion_order_columns(frame, &schema)?;
        let (frame, frame_live_reservation, source_cache_pin, graph_residency) = frame.into_frame_and_residency();
        let live_reservation = live_reservation.or(frame_live_reservation);
        let batch = GpuRecordBatch::new(frame, schema).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_result_handle",
                "failed to bind native GPU result to its Arrow schema contract",
                error,
            )
        })?;
        Ok(Self {
            batch,
            runtime: context.shared_runtime()?,
            device_placement: context.device_placement(),
            live_reservation,
            source_cache_pin,
            graph_residency,
        })
    }

    /// Borrow the underlying GPU cuDF frame.
    pub fn frame(&self) -> &DataFrame {
        self.batch.frame()
    }

    /// Consume the handle and return its GPU frame wrapper.
    pub fn into_gpu_frame(self) -> GpuDataFrame {
        let Self { batch, runtime: _, device_placement: _, live_reservation, source_cache_pin, graph_residency } = self;
        let mut frame = GpuDataFrame::new_with_residency(batch.into_frame(), live_reservation, source_cache_pin);
        frame.graph_residency = graph_residency;
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

    /// Native-engine device placement recorded at handle creation.
    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    /// Observed physical device bytes for this result handle.
    ///
    /// Source-cache promotion uses this as its pre-reservation upper bound
    /// before creating any cache-domain allocation.
    pub(crate) fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.batch.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_result_handle.device_memory",
                "failed to observe native GPU result device memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "nexus_query_engine.gpu_result_handle.device_memory",
                    "GPU result memory accounting did not include device-allocated bytes",
                )
            })?;
        Ok(bytes as u64)
    }

    /// Graph source provenance carried by this handle, when known.
    pub fn graph_residency(&self) -> Option<&GraphResidencyDescriptor> {
        self.graph_residency.as_deref()
    }

    /// Copies this attempt-owned source frame into the supplied cache-domain
    /// lease. The source handle is borrowed and remains owned by the attempt
    /// on every success, cancellation, or allocation-failure path.
    pub(crate) fn copy_for_local_parquet_cache(
        &self,
        allocation_lease: Arc<cudf::CacheAllocationLease>,
    ) -> Result<GpuLocalParquetCachedResult> {
        const OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache.copy_for_cache";
        if self.source_cache_pin.is_some() || self.graph_residency.is_some() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache input already carries non-attempt cache residency",
            )
            .with_fact("reason", "non_attempt_source_residency_in_cache_input"));
        }
        if self.runtime.device() != allocation_lease.domain().device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache allocation domain is on another CUDA device",
            )
            .with_fact("source_device", self.runtime.device().ordinal())
            .with_fact("cache_domain_device", allocation_lease.domain().device().ordinal()));
        }
        if self.runtime.binding_id() == allocation_lease.runtime().binding_id() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache allocation lease must use an independent execution binding",
            )
            .with_fact("reason", "local_parquet_cache_runtime_binding_not_independent"));
        }
        let frame = self.batch.frame().copy_into_runtime(allocation_lease.runtime()).map_err(|error| {
            QueryError::from_cudf(
                OPERATION,
                "failed to copy attempt-owned local Parquet source into cache ownership",
                error,
            )
        })?;
        let batch = GpuRecordBatch::new(frame, Arc::clone(self.schema())).map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to bind query-neutral local Parquet cache value", error)
        })?;
        Ok(GpuLocalParquetCachedResult { batch, allocation_lease, device_placement: self.device_placement })
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
            QueryError::from_cudf(
                "nexus_query_engine.host_arrow.materialize",
                "failed to materialize nexus native output to host Arrow",
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
        let Self { batch, runtime, device_placement, live_reservation, source_cache_pin, graph_residency } = self;
        let cursor = batch.into_host_batch_cursor(config).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.host_batch_cursor.create",
                "failed to create bounded native host-output cursor",
                error,
            )
        })?;
        Ok(GpuHostBatchCursor {
            cursor,
            cancellation,
            _runtime: runtime,
            _device_placement: device_placement,
            _live_reservation: live_reservation,
            _source_cache_pin: source_cache_pin,
            _graph_residency: graph_residency,
        })
    }
}

impl GpuLocalParquetCachedResult {
    pub(crate) fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    /// Completes work submitted on the cache lease before the cached value is
    /// destroyed. Cache code must call this before releasing its ledger charge.
    pub(crate) fn synchronize_for_destruction(&self) -> Result<()> {
        self.allocation_lease.runtime().synchronize().map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_local_parquet_source_cache.destroy",
                "failed to synchronize the cache allocation lease before source-cache destruction",
                error,
            )
        })
    }

    pub(crate) fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.batch.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_local_parquet_source_cache.device_memory",
                "failed to inspect cached local Parquet device memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "nexus_query_engine.gpu_local_parquet_source_cache.device_memory",
                    "cached local Parquet memory accounting did not include device-allocated bytes",
                )
            })?;
        Ok(bytes as u64)
    }

    pub(crate) fn checkout(
        &self,
        context: &ExecutionContext,
        pin: GpuLocalParquetSourceCachePin,
    ) -> Result<GpuResultHandle> {
        const OPERATION: &str = "nexus_query_engine.gpu_local_parquet_source_cache.checkout";
        let query_runtime = context.shared_runtime()?;
        if query_runtime.device() != self.allocation_lease.domain().device()
            || self.device_placement != context.device_placement()
        {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                OPERATION,
                "local Parquet cache entry does not match the current query device placement",
            )
            .with_fact("cache_domain_device", self.allocation_lease.domain().device().ordinal())
            .with_fact("query_device", query_runtime.device().ordinal()));
        }
        let frame = self.batch.frame().copy_into_runtime(&query_runtime).map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to copy cached local Parquet source into current attempt", error)
        })?;
        let batch = GpuRecordBatch::new(frame, Arc::clone(self.schema())).map_err(|error| {
            QueryError::from_cudf(OPERATION, "failed to bind current-query local Parquet cache checkout", error)
        })?;
        Ok(GpuResultHandle {
            batch,
            runtime: query_runtime,
            device_placement: context.device_placement(),
            live_reservation: None,
            source_cache_pin: Some(pin),
            graph_residency: None,
        })
    }
}

fn strip_hidden_datafusion_order_columns(frame: GpuDataFrame, schema: &SchemaRef) -> Result<GpuDataFrame> {
    if frame.num_columns() == schema.fields().len() {
        return Ok(frame);
    }

    let (raw, live_reservation, source_cache_pin, graph_residency) = frame.into_frame_and_residency();
    let frame_names = raw.column_names();
    let expected_names = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    let visible_prefix_matches =
        frame_names.iter().take(expected_names.len()).copied().eq(expected_names.iter().copied());
    let trailing_are_order_columns =
        frame_names.iter().skip(expected_names.len()).all(|name| is_datafusion_order_column_name(name));
    if visible_prefix_matches && trailing_are_order_columns {
        let selected = raw.select(expected_names).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.gpu_result_handle",
                "failed to strip hidden native DataFusion order columns before binding result schema",
                error,
            )
        })?;
        let mut frame = GpuDataFrame::new_with_residency(selected, live_reservation, source_cache_pin);
        frame.graph_residency = graph_residency;
        return Ok(frame);
    }

    let mut frame = GpuDataFrame::new_with_residency(raw, live_reservation, source_cache_pin);
    frame.graph_residency = graph_residency;
    Ok(frame)
}

/// Reusable cuDF hash-join build artifact.
///
/// The handle is runtime/device-affine and carries the build key,
/// null-equality, and retained hash-table byte contract needed to reject unsafe
/// reuse. Cloning shares the underlying cuDF artifact.
#[derive(Clone)]
pub struct GpuHashJoinBuildHandle {
    hash_join: DataFrameHashJoin,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
    build_keys: Vec<String>,
    null_equality: JoinNullEquality,
    device_allocated_size_bytes: u64,
}

/// Reusable cuDF filtered-join build artifact.
///
/// Like [`GpuHashJoinBuildHandle`], this is runtime/device-affine and reports
/// observed retained device bytes for reservation/cache accounting.
#[derive(Clone)]
pub struct GpuFilteredJoinBuildHandle {
    filtered_join: DataFrameFilteredJoin,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
    build_keys: Vec<String>,
    null_equality: JoinNullEquality,
    device_allocated_size_bytes: u64,
}

impl GpuFilteredJoinBuildHandle {
    pub(crate) fn new(
        context: &ExecutionContext,
        filtered_join: DataFrameFilteredJoin,
        null_equality: JoinNullEquality,
    ) -> Result<Self> {
        if filtered_join.build().placement().device != context.runtime()?.device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_filtered_join_build_handle",
                "native GPU filtered-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", filtered_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let device_allocated_size_bytes =
            filtered_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu_filtered_join_build_handle",
                    "failed to observe nexus native reusable filtered join memory",
                    error,
                )
                .with_fact("key_count", filtered_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        Ok(Self {
            build_keys: filtered_join.build_keys().to_vec(),
            filtered_join,
            runtime: context.shared_runtime()?,
            device_placement: context.device_placement(),
            null_equality,
            device_allocated_size_bytes,
        })
    }

    pub(crate) fn filtered_join(&self) -> &DataFrameFilteredJoin {
        &self.filtered_join
    }

    /// Shared cuDF runtime that created this reusable build artifact.
    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    /// Native-engine device placement recorded at artifact creation.
    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    /// Current cuDF execution placement for the retained build side.
    pub fn placement(&self) -> ExecutionPlacement {
        self.filtered_join.build().placement()
    }

    /// Build-key column names used to construct the artifact.
    pub fn build_keys(&self) -> &[String] {
        &self.build_keys
    }

    /// Null-equality policy required for safe reuse.
    pub fn null_equality(&self) -> JoinNullEquality {
        self.null_equality
    }

    /// Observed retained device bytes for this artifact.
    pub fn device_allocated_size_bytes(&self) -> u64 {
        self.device_allocated_size_bytes
    }
}

impl GpuHashJoinBuildHandle {
    pub(crate) fn new(
        context: &ExecutionContext,
        hash_join: DataFrameHashJoin,
        null_equality: JoinNullEquality,
    ) -> Result<Self> {
        if hash_join.build().placement().device != context.runtime()?.device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle",
                "native GPU hash-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", hash_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime()?.device().ordinal()));
        }
        let device_allocated_size_bytes =
            hash_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu_hash_join_build_handle",
                    "failed to observe nexus native reusable hash join memory",
                    error,
                )
                .with_fact("key_count", hash_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        Ok(Self {
            build_keys: hash_join.build_keys().to_vec(),
            hash_join,
            runtime: context.shared_runtime()?,
            device_placement: context.device_placement(),
            null_equality,
            device_allocated_size_bytes,
        })
    }

    pub(crate) fn hash_join(&self) -> &DataFrameHashJoin {
        &self.hash_join
    }

    /// Shared cuDF runtime that created this reusable build artifact.
    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    /// Native-engine device placement recorded at artifact creation.
    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    /// Current cuDF execution placement for the retained build side.
    pub fn placement(&self) -> ExecutionPlacement {
        self.hash_join.build().placement()
    }

    /// Build-key column names used to construct the artifact.
    pub fn build_keys(&self) -> &[String] {
        &self.build_keys
    }

    /// Null-equality policy required for safe reuse.
    pub fn null_equality(&self) -> JoinNullEquality {
        self.null_equality
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
    use std::sync::Arc;

    use arrow::{
        array::{Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::{DataFrame, ops::expr::col};

    use super::*;
    use crate::{
        error::{QueryErrorCode, QueryErrorKind, QueryErrorStatus},
        observability::AdmissionRequestId,
        runtime::gpu::ledger::{DeviceLedger, MemoryGrant},
    };

    fn single_i32_frame(runtime: &Arc<cudf::Runtime>, name: &str, values: Vec<i32>) -> DataFrame {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap();
        DataFrame::from_record_batch(runtime, &batch).unwrap()
    }

    fn frame_with_live_reservation(
        runtime: &Arc<cudf::Runtime>,
        grant: &MemoryGrant,
        name: &str,
        values: Vec<i32>,
    ) -> (GpuDataFrame, u64) {
        let frame = single_i32_frame(runtime, name, values);
        let observed_bytes = GpuDataFrame::new(frame.clone()).device_allocated_size_bytes().unwrap();
        let live_reservation =
            grant.retain_live_reservation(observed_bytes).expect("test frame retention fits the immutable grant");
        (GpuDataFrame::new_with_live_reservation(frame, live_reservation), observed_bytes)
    }

    #[test]
    fn gpu_frame_counts_live_reservation_only_for_the_owning_grant() {
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
        let (frame, bytes) = frame_with_live_reservation(&runtime, &owner, "id", vec![1, 2, 3]);

        assert_eq!(frame.live_reservation_bytes_for(Some(&owner)), bytes);
        assert_eq!(frame.live_reservation_bytes_for(Some(&foreign)), 0);
        assert_eq!(frame.live_reservation_bytes_for(None), 0);
    }

    #[test]
    fn host_batch_cursor_observes_cancellation_before_starting_the_next_device_to_host_copy() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_cancellation(cancellation.clone())
            .build()
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
        assert_eq!(error.code(), crate::error::QueryErrorCode::Cancelled);
        assert_eq!(cursor.remaining_rows(), 2);
    }

    #[test]
    fn host_batch_cursor_preserves_structured_hard_limit_facts() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).build().unwrap();
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
        let context = ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_cancellation(cancellation.clone())
            .build()
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

    #[test]
    fn source_residency_clones_live_reservation_for_pure_alias_output() {
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
        let (source, source_bytes) = frame_with_live_reservation(&runtime, &grant, "id", vec![1, 2, 3]);
        assert!(source_bytes > 0);
        assert_eq!(grant.live_reserved_bytes(), source_bytes);

        let selected = source.frame().select(["id"]).unwrap();
        let derived = GpuDataFrame::try_new_with_source_residency(selected, &source).unwrap();
        assert_eq!(grant.live_reserved_bytes(), source_bytes);

        drop(source);
        assert_eq!(grant.live_reserved_bytes(), source_bytes);
        drop(derived);
        assert_eq!(grant.live_reserved_bytes(), 0);
    }

    #[test]
    fn source_residency_retains_derived_live_reservation_for_fresh_output() {
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
        let (source, source_bytes) = frame_with_live_reservation(&runtime, &grant, "id", vec![1, 2, 3]);
        let fresh = single_i32_frame(&runtime, "fresh", vec![10, 20, 30]);
        let fresh_bytes = GpuDataFrame::new(fresh.clone()).device_allocated_size_bytes().unwrap();
        assert!(fresh_bytes > 0);

        let derived = GpuDataFrame::try_new_with_source_residency(fresh, &source).unwrap();
        assert_eq!(grant.live_reserved_bytes(), source_bytes.saturating_add(fresh_bytes));

        drop(source);
        assert_eq!(grant.live_reserved_bytes(), fresh_bytes);
        drop(derived);
        assert_eq!(grant.live_reserved_bytes(), 0);
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
        let (source, source_bytes) = frame_with_live_reservation(&runtime, &grant, "id", vec![1, 2, 3]);
        let mixed = source.frame().with_columns([("fresh", col("id") + 1i32)]).unwrap();
        let fresh_only = mixed.select(["fresh"]).unwrap();
        let fresh_bytes = GpuDataFrame::new(fresh_only).device_allocated_size_bytes().unwrap();
        let mixed_bytes = GpuDataFrame::new(mixed.clone()).device_allocated_size_bytes().unwrap();
        assert!(fresh_bytes > 0);
        assert!(
            mixed_bytes > fresh_bytes,
            "mixed output should include both the aliased source column and the fresh derived column"
        );

        let derived = GpuDataFrame::try_new_with_source_residency(mixed, &source).unwrap();
        assert_eq!(grant.live_reserved_bytes(), source_bytes.saturating_add(mixed_bytes));

        drop(source);
        assert_eq!(grant.live_reserved_bytes(), mixed_bytes);
        drop(derived);
        assert_eq!(grant.live_reserved_bytes(), 0);
    }
}
