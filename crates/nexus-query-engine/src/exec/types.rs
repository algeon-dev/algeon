//! Public GPU-resident execution handles, outputs, and caches.
//!
//! Handles in this module are runtime/device-affine. Cloning shares the
//! underlying cuDF resources; host materialization is explicit. Cache byte
//! counters are observed retained device bytes, not reservation totals.

#[cfg(feature = "cugraph")]
use std::marker::PhantomData;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write,
    sync::{Arc, Mutex, Weak},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, DataType, ExecutionPlacement,
    frame::dataframe::{ColumnMemoryIdentity, GpuRecordBatch, InfoDetail},
};

#[cfg(feature = "cugraph")]
use crate::plan::ResidentGraphKey;
use crate::{
    chunk::SourceChunk,
    error::{QueryError, Result},
    exec::is_datafusion_order_column_name,
    metrics::QueryMetrics,
    plan::{GraphResidencyDescriptor, HashJoinBuildArtifactId, JoinNullEquality, ResultSink},
    runtime::{
        DevicePlacement, ExecutionContext,
        gpu::governor::{EvictableMemoryPressure, MemoryGovernor, MemoryGrant, MemoryGrantLiveReservation},
    },
    source::{LocalParquetSource, ParquetReaderDecimalType, ResolvedParquetSource, SourceColumnRef},
    util::escape_kv_delimited,
};

/// Local lookup table for GPU-resident inputs consumed by a query.
///
/// Keys are plan-local input names, not global handles. All handles must be
/// reusable by the same [`ExecutionContext`] that executes the consuming plan.
pub type GpuResidentInputMap = BTreeMap<String, GpuResultHandle>;

/// Output residency requested by a query sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryOutputMode {
    /// Keep the query output as a runtime/device-affine GPU handle.
    GpuResident,
    /// Materialize the query output to a host Arrow record batch.
    MaterializeToHost,
}

impl QueryOutputMode {
    /// Derive the output mode requested by a query plan sink.
    pub fn for_sink(sink: &ResultSink) -> Self {
        match sink {
            ResultSink::GpuResident { .. } => Self::GpuResident,
            ResultSink::HostArrow { .. } => Self::MaterializeToHost,
        }
    }

    /// Stable lowercase label used in metrics and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuResident => "gpu_resident",
            Self::MaterializeToHost => "materialize_to_host",
        }
    }
}

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

    /// Wrap a derived cuDF frame and retain source residency according to the
    /// observed column-owner relationship with `source`.
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
        let source_cache_pin = aliases_source.then(|| source.source_cache_pin.clone()).flatten();
        Ok(Self::new_with_residency(frame, live_reservation, source_cache_pin))
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

    pub(in crate::exec) fn source_cache_pin(&self) -> Option<GpuLocalParquetSourceCachePin> {
        self.source_cache_pin.clone()
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

impl GpuResultHandle {
    /// Bind a GPU frame to a schema and the runtime from `context`.
    ///
    /// The frame must already reside on the context's device. Hidden
    /// DataFusion order columns may be stripped before schema binding.
    pub fn new(context: &ExecutionContext, frame: GpuDataFrame, schema: SchemaRef) -> Result<Self> {
        Self::new_with_live_reservation(context, frame, schema, None)
    }

    pub(in crate::exec) fn new_with_live_reservation(
        context: &ExecutionContext,
        frame: GpuDataFrame,
        schema: SchemaRef,
        live_reservation: Option<MemoryGrantLiveReservation>,
    ) -> Result<Self> {
        if frame.placement().device != context.runtime().device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_result_handle",
                "native GPU result frame device does not match the execution context",
            )
            .with_fact("frame_device", frame.placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
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
            runtime: context.shared_runtime(),
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

    pub(crate) fn has_live_reservation(&self) -> bool {
        self.live_reservation.is_some()
    }

    pub(crate) fn has_source_cache_pin(&self) -> bool {
        self.source_cache_pin.is_some()
    }

    /// Graph source provenance carried by this handle, when known.
    pub fn graph_residency(&self) -> Option<&GraphResidencyDescriptor> {
        self.graph_residency.as_deref()
    }

    fn with_source_cache_pin(mut self, pin: GpuLocalParquetSourceCachePin) -> Self {
        debug_assert!(self.source_cache_pin.is_none(), "source-cache entry handles must not already carry pins");
        self.source_cache_pin = Some(pin);
        self
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
                "failed to materialize native cudf output to host Arrow",
                error,
            )
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
/// The handle is runtime/device-affine and carries the build key, build schema,
/// null-equality, and retained hash-table byte contract needed to reject unsafe
/// reuse. Cloning shares the underlying cuDF artifact.
#[derive(Clone)]
pub struct GpuHashJoinBuildHandle {
    hash_join: DataFrameHashJoin,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
    build_keys: Vec<String>,
    build_columns: Vec<GpuHashJoinBuildColumn>,
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
        if filtered_join.build().placement().device != context.runtime().device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_filtered_join_build_handle",
                "native GPU filtered-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", filtered_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        let device_allocated_size_bytes =
            filtered_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu_filtered_join_build_handle",
                    "failed to observe native cudf reusable filtered join memory",
                    error,
                )
                .with_fact("key_count", filtered_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        Ok(Self {
            build_keys: filtered_join.build_keys().to_vec(),
            filtered_join,
            runtime: context.shared_runtime(),
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
        if hash_join.build().placement().device != context.runtime().device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle",
                "native GPU hash-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", hash_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        let device_allocated_size_bytes =
            hash_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu_hash_join_build_handle",
                    "failed to observe native cudf reusable hash join memory",
                    error,
                )
                .with_fact("key_count", hash_join.build_keys().len())
                .with_fact("null_equality", null_equality)
            })?;
        let build_columns = gpu_hash_join_build_columns(hash_join.build())?;
        Ok(Self {
            build_keys: hash_join.build_keys().to_vec(),
            build_columns,
            hash_join,
            runtime: context.shared_runtime(),
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

    pub(crate) fn validate_for_reuse(
        &self,
        context: &ExecutionContext,
        build_keys: &[String],
        build_columns: &[GpuHashJoinBuildColumn],
        null_equality: JoinNullEquality,
    ) -> Result<()> {
        let context_runtime = context.shared_runtime();
        if !Arc::ptr_eq(&self.runtime, &context_runtime) {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact runtime does not match the execution context",
            )
            .with_fact("reason", "runtime_mismatch")
            .with_fact("artifact_device", self.runtime.device().ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        if self.device_placement != context.device_placement() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact device placement does not match the execution context",
            )
            .with_fact("reason", "device_placement_mismatch")
            .with_fact("artifact_device_placement", format!("{:?}", self.device_placement))
            .with_fact("runtime_device_placement", format!("{:?}", context.device_placement())));
        }
        if self.build_keys != build_keys {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact key contract does not match the requested join",
            )
            .with_fact("reason", "build_keys_mismatch")
            .with_fact("artifact_build_keys", self.build_keys.join(","))
            .with_fact("requested_build_keys", build_keys.join(",")));
        }
        if self.build_columns != build_columns {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact schema contract does not match the requested join",
            )
            .with_fact("reason", "build_schema_mismatch")
            .with_fact("artifact_build_schema", format_gpu_hash_join_build_columns(&self.build_columns))
            .with_fact("requested_build_schema", format_gpu_hash_join_build_columns(build_columns)));
        }
        if self.null_equality != null_equality {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact null-equality contract does not match the requested join",
            )
            .with_fact("reason", "null_equality_mismatch")
            .with_fact("artifact_null_equality", self.null_equality)
            .with_fact("requested_null_equality", null_equality));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuHashJoinBuildColumn {
    name: String,
    data_type: DataType,
    nullable: bool,
}

pub(crate) fn gpu_hash_join_build_columns(frame: &DataFrame) -> Result<Vec<GpuHashJoinBuildColumn>> {
    let info = frame.info_with(InfoDetail::COLUMNS).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.gpu_hash_join_build_contract",
            "failed to observe native cudf hash-join build schema contract",
            error,
        )
    })?;
    let columns = info.columns.ok_or_else(|| {
        QueryError::new(
            crate::error::QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_hash_join_build_contract",
            "native cudf hash-join build schema contract did not include column metadata",
        )
    })?;
    Ok(columns
        .into_iter()
        .map(|column| GpuHashJoinBuildColumn {
            name: column.name,
            data_type: column.data_type,
            nullable: column.nullable,
        })
        .collect())
}

fn format_gpu_hash_join_build_columns(columns: &[GpuHashJoinBuildColumn]) -> String {
    columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",")
}

/// Durable cache for reusable GPU hash-join build artifacts.
///
/// Entries are keyed by plan-local [`HashJoinBuildArtifactId`] values. Reuse is
/// still validated against runtime, device placement, build keys, build schema,
/// and null-equality before an artifact is consumed.
#[derive(Default)]
pub struct GpuHashJoinBuildCache {
    cache: Mutex<HashMap<HashJoinBuildArtifactId, GpuHashJoinBuildHandle>>,
}

impl std::fmt::Debug for GpuHashJoinBuildCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entry_count = self.cache.lock().ok().map(|cache| cache.len());
        f.debug_struct("GpuHashJoinBuildCache").field("entry_count", &entry_count).finish()
    }
}

impl GpuHashJoinBuildCache {
    /// Create an empty durable hash-join build cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of retained build artifacts.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.len())
    }

    /// Whether the cache currently holds no build artifacts.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Sum of observed retained hash-table device bytes.
    pub fn retained_hash_table_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.values().map(GpuHashJoinBuildHandle::device_allocated_size_bytes).sum())
    }

    pub(crate) fn get(&self, id: &HashJoinBuildArtifactId) -> Result<Option<GpuHashJoinBuildHandle>> {
        Ok(self.lock()?.get(id).cloned())
    }

    pub(crate) fn insert_if_absent(&self, id: HashJoinBuildArtifactId, handle: GpuHashJoinBuildHandle) -> Result<bool> {
        let mut cache = self.lock()?;
        if cache.contains_key(&id) {
            return Ok(false);
        }
        cache.insert(id, handle);
        Ok(true)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<HashJoinBuildArtifactId, GpuHashJoinBuildHandle>>> {
        self.cache.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_cache",
                "native GPU hash-join build artifact cache lock was poisoned",
            )
        })
    }
}

/// Stable key for a GPU-local Parquet source cache entry.
///
/// The key fingerprints the source scan contract: schema, projected/read
/// columns, predicates, file identities, row groups, scan limit, and chunk
/// boundaries. It is stable for equivalent local source scans, but is not a
/// cross-version or cross-process persistence format.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GpuLocalParquetSourceCacheKey {
    fingerprint: String,
    diagnostic_fingerprint: String,
}

impl std::fmt::Debug for GpuLocalParquetSourceCacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLocalParquetSourceCacheKey").field("fingerprint", &self.diagnostic_fingerprint).finish()
    }
}

impl GpuLocalParquetSourceCacheKey {
    /// Build a key for an unbounded local Parquet source scan.
    pub fn from_source(source: &LocalParquetSource) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "unbounded_local_parquet_source");
        push_source_scan_contract(&mut fingerprint, source);
        push_token(
            &mut fingerprint,
            "scan_limit",
            source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(&mut fingerprint, "file_count", source.files.len().to_string());
        for (index, file) in source.files.iter().enumerate() {
            push_token(&mut fingerprint, format!("file.{index}.location"), file.location.diagnostic_location());
            push_token(&mut fingerprint, format!("file.{index}.stable_id"), &file.stable_id);
            push_resolved_source_contract(&mut fingerprint, format!("file.{index}.resolved"), &file.resolved_source);
            push_token(
                &mut fingerprint,
                format!("file.{index}.size"),
                file.file_size_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
            );
            push_optional_usize_vec(&mut fingerprint, format!("file.{index}.row_groups"), file.row_groups.as_deref());
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.row_group_bytes"),
                file.row_group_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.projected_compressed"),
                file.projected_row_group_compressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.projected_uncompressed"),
                file.projected_row_group_uncompressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.pure_filter_compressed"),
                file.pure_filter_row_group_compressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.pure_filter_uncompressed"),
                file.pure_filter_row_group_uncompressed_bytes.as_deref(),
            );
            push_optional_u64_vec(
                &mut fingerprint,
                format!("file.{index}.row_group_rows"),
                file.row_group_rows.as_deref(),
            );
            push_reader_decimal_types(
                &mut fingerprint,
                format!("file.{index}.reader_decimal_types"),
                &file.reader_decimal_types,
            );
            push_position_delete_files(
                &mut fingerprint,
                format!("file.{index}.position_delete_files"),
                &file.position_delete_files,
            );
            push_equality_delete_files(
                &mut fingerprint,
                format!("file.{index}.equality_delete_files"),
                &file.equality_delete_files,
            );
            let partition_values = file.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
            push_string_vec(&mut fingerprint, format!("file.{index}.partition_values"), &partition_values);
        }
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_cache_fingerprint(&mut diagnostic_fingerprint, source);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Build a key for one bounded local Parquet source chunk.
    pub fn from_source_chunk(source: &LocalParquetSource, chunk: &SourceChunk) -> Self {
        let mut fingerprint = String::new();
        push_token(&mut fingerprint, "mode", "bounded_local_parquet_source_chunk");
        push_source_scan_contract(&mut fingerprint, source);
        push_token(
            &mut fingerprint,
            "source_scan_limit",
            source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(
            &mut fingerprint,
            "chunk_scan_limit",
            chunk.scan_limit_rows.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        );
        push_token(&mut fingerprint, "chunk.file_index", chunk.file_index.to_string());
        push_token(&mut fingerprint, "chunk.location", chunk.location.diagnostic_location());
        push_token(&mut fingerprint, "chunk.stable_id", &chunk.stable_id);
        push_resolved_source_contract(&mut fingerprint, "chunk.resolved", &chunk.resolved_source);
        push_token(&mut fingerprint, "chunk.reads_all_row_groups", chunk.reads_all_row_groups.to_string());
        push_usize_vec(&mut fingerprint, "chunk.row_groups", &chunk.row_group_indexes);
        push_optional_u64(&mut fingerprint, "chunk.known_file_bytes", chunk.known_file_bytes);
        push_optional_u64(&mut fingerprint, "chunk.known_row_group_bytes", chunk.known_row_group_bytes);
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_projected_compressed_bytes",
            chunk.known_projected_compressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_projected_uncompressed_bytes",
            chunk.known_projected_uncompressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_pure_filter_compressed_bytes",
            chunk.known_pure_filter_compressed_bytes,
        );
        push_optional_u64(
            &mut fingerprint,
            "chunk.known_pure_filter_uncompressed_bytes",
            chunk.known_pure_filter_uncompressed_bytes,
        );
        push_optional_usize_vec(
            &mut fingerprint,
            "chunk.source_file_row_groups",
            chunk.source_file_row_groups.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_row_group_bytes",
            chunk.source_file_row_group_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_projected_compressed_bytes",
            chunk.source_file_projected_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_projected_uncompressed_bytes",
            chunk.source_file_projected_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_pure_filter_compressed_bytes",
            chunk.source_file_pure_filter_compressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_pure_filter_uncompressed_bytes",
            chunk.source_file_pure_filter_uncompressed_bytes.as_deref(),
        );
        push_optional_u64_vec(
            &mut fingerprint,
            "chunk.source_file_row_group_rows",
            chunk.source_file_row_group_rows.as_deref(),
        );
        push_string_vec(&mut fingerprint, "chunk.read_columns", &chunk.read_columns);
        push_reader_decimal_types(&mut fingerprint, "chunk.reader_decimal_types", &chunk.reader_decimal_types);
        if let Some(file) = source.files.get(chunk.file_index) {
            push_position_delete_files(&mut fingerprint, "chunk.position_delete_files", &file.position_delete_files);
            push_equality_delete_files(&mut fingerprint, "chunk.equality_delete_files", &file.equality_delete_files);
        } else {
            push_token(&mut fingerprint, "chunk.position_delete_files.len", "missing");
            push_token(&mut fingerprint, "chunk.equality_delete_files.len", "missing");
        }
        push_token(
            &mut fingerprint,
            "chunk.reader_decimal_types_footer_verified",
            chunk.reader_decimal_types_footer_verified.to_string(),
        );
        let partition_values = chunk.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
        push_string_vec(&mut fingerprint, "chunk.partition_values", &partition_values);
        let mut diagnostic_fingerprint = fingerprint.clone();
        redact_source_chunk_cache_fingerprint(&mut diagnostic_fingerprint, source, chunk);
        Self { fingerprint, diagnostic_fingerprint }
    }

    /// Escaped textual fingerprint for diagnostics and tests.
    pub fn as_str(&self) -> &str {
        &self.diagnostic_fingerprint
    }
}

/// Default bound on same-graph replicas retained across service stream-pool
/// runtimes.
#[cfg(feature = "cugraph")]
pub const DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY: usize = 4;

/// Result of inserting a parked resident graph.
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuResidentGraphCacheInsert {
    /// The parked graph was retained by the cache.
    Inserted,
    /// An equivalent entry was already available for the same runtime.
    AlreadyPresent,
    /// The retained-byte cap could not admit this graph.
    CapacitySkipped,
    /// The per-resident-key replica cap would be exceeded.
    CopyLimitSkipped,
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) enum GpuResidentGraphEntry {
    Forward(cugraph::ParkedGraph<cugraph::Forward>),
    Transposed(cugraph::ParkedGraph<cugraph::Transposed>),
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) trait GpuResidentGraphTransposition:
    cugraph::Transposition + Send + Sized + 'static
{
    fn orientation() -> crate::plan::GraphResidencyOrientation;
    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry;
    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>>;
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphTransposition for cugraph::Forward {
    fn orientation() -> crate::plan::GraphResidencyOrientation {
        crate::plan::GraphResidencyOrientation::Forward
    }

    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Forward(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Forward(parked) => Some(parked),
            GpuResidentGraphEntry::Transposed(_) => None,
        }
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphTransposition for cugraph::Transposed {
    fn orientation() -> crate::plan::GraphResidencyOrientation {
        crate::plan::GraphResidencyOrientation::Transposed
    }

    fn into_entry(parked: cugraph::ParkedGraph<Self>) -> GpuResidentGraphEntry {
        GpuResidentGraphEntry::Transposed(parked)
    }

    fn take_entry(entry: GpuResidentGraphEntry) -> Option<cugraph::ParkedGraph<Self>> {
        match entry {
            GpuResidentGraphEntry::Transposed(parked) => Some(parked),
            GpuResidentGraphEntry::Forward(_) => None,
        }
    }
}

#[cfg(feature = "cugraph")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct GpuResidentGraphCacheEntryKey {
    resident_key: ResidentGraphKey,
    runtime_key: usize,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCacheEntryKey {
    fn for_context(resident_key: ResidentGraphKey, context: &ExecutionContext) -> Self {
        let runtime = context.shared_runtime();
        Self { resident_key, runtime_key: Arc::as_ptr(&runtime) as usize }
    }
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCacheEntry {
    graph: GpuResidentGraphEntry,
    retained_device_bytes: u64,
    last_access_epoch: u64,
    runtime_owner: Weak<cudf::Runtime>,
}

#[cfg(feature = "cugraph")]
#[derive(Default)]
struct GpuResidentGraphCacheEviction {
    entries: Vec<GpuResidentGraphCacheEntry>,
    device_bytes: u64,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCheckedOut {
    retained_device_bytes: u64,
    runtime_owner: Weak<cudf::Runtime>,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCompanionRuntime {
    pooled_runtime: Weak<cudf::Runtime>,
    companion: Arc<cudf::Runtime>,
}

#[cfg(feature = "cugraph")]
#[derive(Default)]
struct GpuResidentGraphCacheState {
    available: HashMap<GpuResidentGraphCacheEntryKey, GpuResidentGraphCacheEntry>,
    checked_out: HashMap<GpuResidentGraphCacheEntryKey, GpuResidentGraphCheckedOut>,
    preparation_pins: HashMap<GpuResidentGraphCacheEntryKey, u64>,
    companion_runtimes: HashMap<usize, GpuResidentGraphCompanionRuntime>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
    next_preparation_pin_id: u64,
}

#[cfg(feature = "cugraph")]
struct GpuResidentGraphCacheInner {
    max_retained_device_bytes: u64,
    max_copies_per_key: usize,
    governor: Option<MemoryGovernor>,
    state: Mutex<GpuResidentGraphCacheState>,
}

#[cfg(feature = "cugraph")]
impl Drop for GpuResidentGraphCacheInner {
    fn drop(&mut self) {
        let Some(governor) = &self.governor else {
            return;
        };
        let Ok(state) = self.state.get_mut() else {
            return;
        };
        let retained_device_bytes = state.retained_device_bytes;
        if retained_device_bytes > 0 {
            let _ = governor.release_evictable_bytes(retained_device_bytes);
            state.retained_device_bytes = 0;
        }
    }
}

/// Per-device cache for cuGraph graphs parked between query executions.
///
/// Entries are keyed by a device-agnostic [`ResidentGraphKey`] plus the
/// service stream-pool runtime identity that owns the parked graph's cuDF
/// runtime-bound handle leases. Checked-out graphs are exclusive: a hit moves
/// the parked graph out of the available map until the caller parks it again
/// or drops the checkout guard, at which point the governor evictable claim is
/// released.
#[cfg(feature = "cugraph")]
pub struct GpuResidentGraphCache {
    inner: Arc<GpuResidentGraphCacheInner>,
}

#[cfg(feature = "cugraph")]
impl std::fmt::Debug for GpuResidentGraphCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.state.lock().ok();
        f.debug_struct("GpuResidentGraphCache")
            .field("max_retained_device_bytes", &self.inner.max_retained_device_bytes)
            .field("max_copies_per_key", &self.inner.max_copies_per_key)
            .field("governor_backed", &self.is_governor_backed())
            .field("available_entry_count", &state.as_ref().map(|state| state.available.len()))
            .field("checked_out_entry_count", &state.as_ref().map(|state| state.checked_out.len()))
            .field("retained_device_bytes", &state.as_ref().map(|state| state.retained_device_bytes))
            .finish()
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCache {
    /// Create an empty resident graph cache with a retained-device-byte cap.
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self::new_with_governor(max_retained_device_bytes, DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY, None)
    }

    /// Create an empty resident graph cache charged to a device memory
    /// governor's evictable ledger.
    pub fn new_governor_backed(max_retained_device_bytes: u64, governor: MemoryGovernor) -> Self {
        Self::new_with_governor(
            max_retained_device_bytes,
            DEFAULT_RESIDENT_GRAPH_CACHE_MAX_COPIES_PER_KEY,
            Some(governor),
        )
    }

    /// Override the per-resident-key copy cap.
    pub fn with_max_copies_per_key(self, max_copies_per_key: usize) -> Self {
        Self::new_with_governor(self.inner.max_retained_device_bytes, max_copies_per_key, self.inner.governor.clone())
    }

    fn new_with_governor(
        max_retained_device_bytes: u64,
        max_copies_per_key: usize,
        governor: Option<MemoryGovernor>,
    ) -> Self {
        Self {
            inner: Arc::new(GpuResidentGraphCacheInner {
                max_retained_device_bytes,
                max_copies_per_key,
                governor,
                state: Mutex::new(GpuResidentGraphCacheState::default()),
            }),
        }
    }

    /// Maximum retained device bytes allowed by this cache.
    pub fn max_retained_device_bytes(&self) -> u64 {
        self.inner.max_retained_device_bytes
    }

    /// Maximum same-graph replicas retained across runtime identities.
    pub fn max_copies_per_key(&self) -> usize {
        self.inner.max_copies_per_key
    }

    /// Whether retained bytes are charged to a memory governor.
    pub fn is_governor_backed(&self) -> bool {
        self.inner.governor.is_some()
    }

    /// Register this cache as a synchronous evictable-memory observer.
    pub fn register_governor_pressure_observer(cache: &Arc<Self>) -> Result<()> {
        let Some(governor) = cache.inner.governor.as_ref().cloned() else {
            return Ok(());
        };
        let weak_cache: Weak<Self> = Arc::downgrade(cache);
        governor.register_evictable_memory_observer(Arc::new(move |pressure| {
            let Some(cache) = weak_cache.upgrade() else {
                return Ok(0);
            };
            cache.handle_memory_pressure(pressure)
        }))
    }

    /// Number of available parked graph entries.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.available.len())
    }

    /// Number of graphs currently checked out by running queries.
    pub fn checked_out_len(&self) -> Result<usize> {
        Ok(self.lock()?.checked_out.len())
    }

    /// Retained device bytes across available and checked-out parked graphs.
    pub fn retained_device_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.retained_device_bytes)
    }

    /// Whether this device cache currently has a live copy for `resident_key`.
    ///
    /// This is a host-side placement hint only: it must not perform CUDA work,
    /// call the memory governor, or acquire locks beyond the cache mutex because
    /// query-service admission may call it while holding the admission mutex.
    pub fn contains_resident_key(&self, resident_key: &ResidentGraphKey) -> Result<bool> {
        let state = self.lock()?;
        Ok(state
            .available
            .iter()
            .any(|(key, entry)| &key.resident_key == resident_key && entry.runtime_owner.upgrade().is_some())
            || state
                .checked_out
                .iter()
                .any(|(key, entry)| &key.resident_key == resident_key && entry.runtime_owner.upgrade().is_some()))
    }

    /// Pin an available graph for an imminent exact-runtime checkout.
    ///
    /// Unlike [`Self::contains_resident_key`], this is an exact-runtime lookup
    /// used by execution to decide whether source-side graph preparation can be
    /// skipped safely. Pressure eviction skips the pinned entry until checkout
    /// consumes the pin or the returned guard drops.
    pub(in crate::exec) fn try_pin_available_key_for_context(
        &self,
        resident_key: &ResidentGraphKey,
        context: &ExecutionContext,
    ) -> Result<Option<GpuResidentGraphPreparationPin>> {
        let entry_key = GpuResidentGraphCacheEntryKey::for_context(resident_key.clone(), context);
        let context_runtime = context.shared_runtime();
        let mut state = self.lock()?;
        let available = state
            .available
            .get(&entry_key)
            .is_some_and(|entry| runtime_owner_matches(&entry.runtime_owner, &context_runtime));
        if !available || state.preparation_pins.contains_key(&entry_key) {
            return Ok(None);
        }
        let pin_id = state.next_preparation_pin_id();
        state.preparation_pins.insert(entry_key.clone(), pin_id);
        Ok(Some(GpuResidentGraphPreparationPin { cache: Arc::clone(&self.inner), entry_key, pin_id }))
    }

    pub(in crate::exec) fn companion_runtime_for_context(
        &self,
        context: &ExecutionContext,
        operation: &'static str,
    ) -> Result<Arc<cudf::Runtime>> {
        let pooled_runtime = context.shared_runtime();
        let runtime_key = Arc::as_ptr(&pooled_runtime) as usize;
        if let Some(companion) = self.lock()?.companion_runtimes.get(&runtime_key)
            && runtime_owner_matches(&companion.pooled_runtime, &pooled_runtime)
        {
            return Ok(Arc::clone(&companion.companion));
        }
        let companion = Arc::new(
            cudf::Runtime::builder()
                .with_device(context.runtime().device())
                .with_stream(context.runtime().stream().clone())
                .with_device_memory_manager()
                .build()
                .map_err(|error| {
                    QueryError::from_cudf(
                        operation,
                        "failed to build stable companion runtime for resident graph cache",
                        error,
                    )
                    .with_fact("device_ordinal", context.runtime().device().ordinal())
                })?,
        );
        let mut state = self.lock()?;
        if let Some(existing) = state.companion_runtimes.get(&runtime_key)
            && runtime_owner_matches(&existing.pooled_runtime, &pooled_runtime)
        {
            return Ok(Arc::clone(&existing.companion));
        }
        state.companion_runtimes.insert(
            runtime_key,
            GpuResidentGraphCompanionRuntime {
                pooled_runtime: Arc::downgrade(&pooled_runtime),
                companion: Arc::clone(&companion),
            },
        );
        Ok(companion)
    }

    pub(in crate::exec) fn checkout<T>(
        &self,
        resident_key: &ResidentGraphKey,
        context: &ExecutionContext,
        preparation_pin_id: Option<u64>,
    ) -> Result<Option<GpuResidentGraphCheckout<T>>>
    where
        T: GpuResidentGraphTransposition,
    {
        let entry_key = GpuResidentGraphCacheEntryKey::for_context(resident_key.clone(), context);
        let mut state = self.lock()?;
        if !state.available.contains_key(&entry_key)
            || !state.consume_checkout_preparation_pin(&entry_key, preparation_pin_id)
        {
            return Ok(None);
        }
        let Some(entry) = state.available.remove(&entry_key) else {
            unreachable!("available resident graph entry checked immediately before removal")
        };
        let context_runtime = context.shared_runtime();
        let retained_device_bytes = entry.retained_device_bytes;
        if !runtime_owner_matches(&entry.runtime_owner, &context_runtime) {
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(retained_device_bytes);
            drop(state);
            let released = self.release_governor_evictable_bytes(retained_device_bytes);
            drop(entry);
            released?;
            return Ok(None);
        }
        let Some(parked) = T::take_entry(entry.graph) else {
            state.retained_device_bytes = state.retained_device_bytes.saturating_sub(retained_device_bytes);
            drop(state);
            self.release_governor_evictable_bytes(retained_device_bytes)?;
            return Ok(None);
        };
        state.checked_out.insert(
            entry_key.clone(),
            GpuResidentGraphCheckedOut { retained_device_bytes, runtime_owner: entry.runtime_owner },
        );
        Ok(Some(GpuResidentGraphCheckout {
            cache: Arc::clone(&self.inner),
            entry_key,
            retained_device_bytes,
            parked: Some(parked),
            completed: false,
            _transposition: PhantomData,
        }))
    }

    pub(in crate::exec) fn insert_if_absent<T>(
        &self,
        resident_key: ResidentGraphKey,
        context: &ExecutionContext,
        parked: cugraph::ParkedGraph<T>,
        retained_device_bytes: u64,
    ) -> Result<GpuResidentGraphCacheInsert>
    where
        T: GpuResidentGraphTransposition,
    {
        let context_runtime = context.shared_runtime();
        let entry_key = GpuResidentGraphCacheEntryKey::for_context(resident_key, context);
        if retained_device_bytes > self.inner.max_retained_device_bytes {
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let mut state = self.lock()?;
        if state.available.contains_key(&entry_key) || state.checked_out.contains_key(&entry_key) {
            return Ok(GpuResidentGraphCacheInsert::AlreadyPresent);
        }
        if state.copy_count_for_resident_key(&entry_key.resident_key) >= self.inner.max_copies_per_key {
            return Ok(GpuResidentGraphCacheInsert::CopyLimitSkipped);
        }
        let memory_grant = context.memory_grant();
        if !self.governor_can_admit_after_cache_eviction(&state, retained_device_bytes, memory_grant)? {
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let eviction = state.evict_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes);
        let evicted_device_bytes = eviction.device_bytes;
        // These governor ledger calls must not notify eviction observers while
        // holding the governor state lock; observers may call back into this
        // cache and take the cache mutex.
        if let Err(error) = self.release_governor_evictable_bytes(evicted_device_bytes) {
            drop(state);
            drop(eviction);
            return Err(error);
        }
        if state.retained_device_bytes.saturating_add(retained_device_bytes) > self.inner.max_retained_device_bytes {
            drop(state);
            drop(eviction);
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let reserved = match self.reserve_governor_evictable_bytes(retained_device_bytes, memory_grant) {
            Ok(reserved) => reserved,
            Err(error) => {
                drop(state);
                drop(eviction);
                return Err(error);
            },
        };
        if !reserved {
            drop(state);
            drop(eviction);
            return Ok(GpuResidentGraphCacheInsert::CapacitySkipped);
        }
        let last_access_epoch = state.next_epoch();
        state.available.insert(
            entry_key,
            GpuResidentGraphCacheEntry {
                graph: T::into_entry(parked),
                retained_device_bytes,
                last_access_epoch,
                runtime_owner: Arc::downgrade(&context_runtime),
            },
        );
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        drop(state);
        drop(eviction);
        Ok(GpuResidentGraphCacheInsert::Inserted)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GpuResidentGraphCacheState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache",
                "native GPU resident graph cache lock was poisoned",
            )
        })
    }

    fn handle_memory_pressure(&self, pressure: EvictableMemoryPressure) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(0);
        };
        if pressure.device_ordinal != governor.device_ordinal() {
            return Ok(0);
        }
        let mut state = self.lock()?;
        let eviction = state.evict_at_least(pressure.requested_evict_bytes);
        let evicted_device_bytes = eviction.device_bytes;
        drop(state);
        drop(eviction);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        Ok(evicted_device_bytes)
    }

    fn governor_can_admit_after_cache_eviction(
        &self,
        state: &GpuResidentGraphCacheState,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        let evictable_for_capacity =
            state.evictable_bytes_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes);
        let free_bytes = governor.free_bytes()?;
        if free_bytes.saturating_add(evictable_for_capacity) >= retained_device_bytes {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.grant_can_cover_evictable_bytes(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn reserve_governor_evictable_bytes(
        &self,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        if governor.try_reserve_evictable_bytes(retained_device_bytes)? {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.try_reserve_evictable_bytes_from_grant(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn release_governor_evictable_bytes(&self, released_device_bytes: u64) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(released_device_bytes);
        };
        governor.release_evictable_bytes(released_device_bytes)
    }
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }

    fn copy_count_for_resident_key(&self, resident_key: &ResidentGraphKey) -> usize {
        self.available.keys().filter(|key| &key.resident_key == resident_key).count()
            + self.checked_out.keys().filter(|key| &key.resident_key == resident_key).count()
    }

    fn evict_until_fits(
        &mut self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
    ) -> GpuResidentGraphCacheEviction {
        let mut eviction = GpuResidentGraphCacheEviction::default();
        while self.retained_device_bytes.saturating_add(incoming_retained_device_bytes) > max_retained_device_bytes {
            let Some(evict_key) = self.lru_available_key() else {
                break;
            };
            if let Some(evicted) = self.available.remove(&evict_key) {
                let retained_device_bytes = evicted.retained_device_bytes;
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(retained_device_bytes);
                eviction.device_bytes = eviction.device_bytes.saturating_add(retained_device_bytes);
                eviction.entries.push(evicted);
            }
        }
        eviction
    }

    fn evictable_bytes_until_fits(&self, max_retained_device_bytes: u64, incoming_retained_device_bytes: u64) -> u64 {
        let mut retained_device_bytes = self.retained_device_bytes;
        let mut evictable_device_bytes = 0u64;
        let mut candidates = self
            .available
            .iter()
            .filter(|(key, _entry)| !self.preparation_pins.contains_key(*key))
            .map(|(_key, entry)| (entry.last_access_epoch, entry.retained_device_bytes))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(last_access_epoch, _bytes)| *last_access_epoch);
        for (_last_access_epoch, candidate_bytes) in candidates {
            if retained_device_bytes.saturating_add(incoming_retained_device_bytes) <= max_retained_device_bytes {
                break;
            }
            retained_device_bytes = retained_device_bytes.saturating_sub(candidate_bytes);
            evictable_device_bytes = evictable_device_bytes.saturating_add(candidate_bytes);
        }
        evictable_device_bytes
    }

    fn evict_at_least(&mut self, target_device_bytes: u64) -> GpuResidentGraphCacheEviction {
        let mut eviction = GpuResidentGraphCacheEviction::default();
        while eviction.device_bytes < target_device_bytes {
            let Some(evict_key) = self.lru_available_key() else {
                break;
            };
            if let Some(evicted) = self.available.remove(&evict_key) {
                let retained_device_bytes = evicted.retained_device_bytes;
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(retained_device_bytes);
                eviction.device_bytes = eviction.device_bytes.saturating_add(retained_device_bytes);
                eviction.entries.push(evicted);
            }
        }
        eviction
    }

    fn lru_available_key(&self) -> Option<GpuResidentGraphCacheEntryKey> {
        self.available
            .iter()
            .filter(|(key, _entry)| !self.preparation_pins.contains_key(*key))
            .min_by_key(|(_key, entry)| entry.last_access_epoch)
            .map(|(key, _entry)| key.clone())
    }

    fn next_preparation_pin_id(&mut self) -> u64 {
        self.next_preparation_pin_id = self.next_preparation_pin_id.wrapping_add(1);
        if self.next_preparation_pin_id == 0 {
            self.next_preparation_pin_id = 1;
        }
        self.next_preparation_pin_id
    }

    fn consume_checkout_preparation_pin(
        &mut self,
        entry_key: &GpuResidentGraphCacheEntryKey,
        preparation_pin_id: Option<u64>,
    ) -> bool {
        match (self.preparation_pins.get(entry_key).copied(), preparation_pin_id) {
            (None, None) => true,
            (Some(expected), Some(actual)) if expected == actual => {
                self.preparation_pins.remove(entry_key);
                true
            },
            _ => false,
        }
    }
}

#[cfg(feature = "cugraph")]
fn runtime_owner_matches(owner: &Weak<cudf::Runtime>, runtime: &Arc<cudf::Runtime>) -> bool {
    owner.upgrade().is_some_and(|owner| Arc::ptr_eq(&owner, runtime))
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) struct GpuResidentGraphPreparationPin {
    cache: Arc<GpuResidentGraphCacheInner>,
    entry_key: GpuResidentGraphCacheEntryKey,
    pin_id: u64,
}

#[cfg(feature = "cugraph")]
impl GpuResidentGraphPreparationPin {
    pub(in crate::exec) fn pin_id(&self) -> u64 {
        self.pin_id
    }
}

#[cfg(feature = "cugraph")]
impl Drop for GpuResidentGraphPreparationPin {
    fn drop(&mut self) {
        if let Ok(mut state) = self.cache.state.lock()
            && state.preparation_pins.get(&self.entry_key) == Some(&self.pin_id)
        {
            state.preparation_pins.remove(&self.entry_key);
        }
    }
}

/// Exclusive checked-out parked graph.
#[cfg(feature = "cugraph")]
pub(in crate::exec) struct GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    cache: Arc<GpuResidentGraphCacheInner>,
    entry_key: GpuResidentGraphCacheEntryKey,
    retained_device_bytes: u64,
    parked: Option<cugraph::ParkedGraph<T>>,
    completed: bool,
    _transposition: PhantomData<T>,
}

#[cfg(feature = "cugraph")]
impl<T> GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    pub(in crate::exec) fn retained_device_bytes(&self) -> u64 {
        self.retained_device_bytes
    }

    pub(in crate::exec) fn take_parked(&mut self) -> cugraph::ParkedGraph<T> {
        self.parked.take().expect("resident graph checkout already consumed")
    }

    pub(in crate::exec) fn return_parked(mut self, parked: cugraph::ParkedGraph<T>) -> Result<()> {
        let mut state = self.cache.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache.return",
                "native GPU resident graph cache lock was poisoned",
            )
        })?;
        let Some(checked_out) = state.checked_out.remove(&self.entry_key) else {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_resident_graph_cache.return",
                "resident graph checkout was not registered",
            ));
        };
        let last_access_epoch = state.next_epoch();
        state.available.insert(
            self.entry_key.clone(),
            GpuResidentGraphCacheEntry {
                graph: T::into_entry(parked),
                retained_device_bytes: checked_out.retained_device_bytes,
                last_access_epoch,
                runtime_owner: checked_out.runtime_owner,
            },
        );
        self.completed = true;
        Ok(())
    }
}

#[cfg(feature = "cugraph")]
impl<T> Drop for GpuResidentGraphCheckout<T>
where T: GpuResidentGraphTransposition
{
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let Ok(mut state) = self.cache.state.lock() else {
            return;
        };
        let released =
            state.checked_out.remove(&self.entry_key).map_or(0, |checked_out| checked_out.retained_device_bytes);
        if released == 0 {
            return;
        }
        state.retained_device_bytes = state.retained_device_bytes.saturating_sub(released);
        let governor = self.cache.governor.clone();
        drop(state);
        if let Some(governor) = governor {
            let _ = governor.release_evictable_bytes(released);
        }
    }
}

/// Result of attempting to retain a GPU source chunk in the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLocalParquetSourceCacheInsert {
    /// Entry was inserted and may have evicted older entries.
    Inserted,
    /// Equivalent entry was already present; the existing handle is retained.
    AlreadyPresent,
    /// Entry was too large or protected entries prevented enough eviction.
    CapacitySkipped,
}

/// Accounting returned by a GPU-local Parquet source cache insert attempt.
///
/// `evicted_device_bytes` reports cache entries removed to make room, while
/// `retained_device_bytes` reports how many bytes the inserted or existing
/// entry contributes to cache residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuLocalParquetSourceCacheInsertResult {
    /// Whether the cache inserted the entry, found it already present, or
    /// skipped insertion because capacity could not be made available.
    pub(crate) outcome: GpuLocalParquetSourceCacheInsert,
    /// Device bytes released from evicted cache entries during this attempt.
    pub(crate) evicted_device_bytes: u64,
    /// Device bytes retained by the inserted or already-present entry.
    pub(crate) retained_device_bytes: u64,
}

#[derive(Clone)]
struct GpuLocalParquetSourceCacheEntry {
    handle: GpuResultHandle,
    retained_device_bytes: u64,
    last_access_epoch: u64,
    in_use_count: u64,
    generation: u64,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct GpuLocalParquetSourceCacheEntryKey {
    source_key: GpuLocalParquetSourceCacheKey,
    runtime_key: usize,
}

impl GpuLocalParquetSourceCacheEntryKey {
    fn new(source_key: GpuLocalParquetSourceCacheKey, runtime: &Arc<cudf::Runtime>) -> Self {
        Self { source_key, runtime_key: Arc::as_ptr(runtime) as usize }
    }

    fn for_context(source_key: &GpuLocalParquetSourceCacheKey, context: &ExecutionContext) -> Self {
        Self::new(source_key.clone(), &context.shared_runtime())
    }

    fn for_handle(source_key: GpuLocalParquetSourceCacheKey, handle: &GpuResultHandle) -> Self {
        Self::new(source_key, handle.runtime())
    }
}

#[derive(Default)]
struct GpuLocalParquetSourceCacheState {
    cache: HashMap<GpuLocalParquetSourceCacheEntryKey, GpuLocalParquetSourceCacheEntry>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
    next_entry_generation: u64,
}

struct GpuLocalParquetSourceCacheInner {
    max_retained_device_bytes: u64,
    governor: Option<MemoryGovernor>,
    state: Mutex<GpuLocalParquetSourceCacheState>,
}

impl Drop for GpuLocalParquetSourceCacheInner {
    fn drop(&mut self) {
        let Some(governor) = &self.governor else {
            return;
        };
        let Ok(state) = self.state.get_mut() else {
            return;
        };
        let retained_device_bytes = state.retained_device_bytes;
        if retained_device_bytes > 0 {
            let _ = governor.release_evictable_bytes(retained_device_bytes);
            state.retained_device_bytes = 0;
        }
    }
}

#[derive(Clone)]
pub(in crate::exec) struct GpuLocalParquetSourceCachePin {
    _inner: Arc<GpuLocalParquetSourceCachePinInner>,
}

struct GpuLocalParquetSourceCachePinInner {
    cache: Arc<GpuLocalParquetSourceCacheInner>,
    entry_key: GpuLocalParquetSourceCacheEntryKey,
    generation: u64,
}

impl Drop for GpuLocalParquetSourceCachePinInner {
    fn drop(&mut self) {
        if let Ok(mut state) = self.cache.state.lock()
            && let Some(entry) = state.cache.get_mut(&self.entry_key)
            && entry.generation == self.generation
        {
            entry.in_use_count = entry.in_use_count.saturating_sub(1);
        }
    }
}

/// LRU cache for GPU-resident local Parquet source results.
///
/// Capacity and retained sizes are device bytes. Cached handles remain
/// runtime/device-affine, and runtime identity is part of the internal cache
/// key. A lookup with a runtime that has not decoded the source yet misses
/// without evicting entries for other runtimes, while a schema mismatch is
/// treated as an invalid plan.
pub struct GpuLocalParquetSourceCache {
    inner: Arc<GpuLocalParquetSourceCacheInner>,
}

pub(in crate::exec) struct GpuLocalParquetSourceCachePinnedInsert {
    pub(in crate::exec) outcome: GpuLocalParquetSourceCacheInsert,
    pub(in crate::exec) handle: Option<GpuResultHandle>,
}

impl std::fmt::Debug for GpuLocalParquetSourceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.inner.state.lock().ok();
        f.debug_struct("GpuLocalParquetSourceCache")
            .field("max_retained_device_bytes", &self.inner.max_retained_device_bytes)
            .field("governor_backed", &self.is_governor_backed())
            .field("entry_count", &state.as_ref().map(|state| state.cache.len()))
            .field("retained_device_bytes", &state.as_ref().map(|state| state.retained_device_bytes))
            .finish()
    }
}

impl GpuLocalParquetSourceCache {
    /// Create an empty source cache with a retained-device-byte capacity.
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self::new_with_governor(max_retained_device_bytes, None)
    }

    /// Create an empty source cache whose retained bytes are charged to a
    /// device memory governor as evictable cache residency.
    pub fn new_governor_backed(max_retained_device_bytes: u64, governor: MemoryGovernor) -> Self {
        Self::new_with_governor(max_retained_device_bytes, Some(governor))
    }

    fn new_with_governor(max_retained_device_bytes: u64, governor: Option<MemoryGovernor>) -> Self {
        Self {
            inner: Arc::new(GpuLocalParquetSourceCacheInner {
                max_retained_device_bytes,
                governor,
                state: Mutex::new(GpuLocalParquetSourceCacheState::default()),
            }),
        }
    }

    /// Maximum retained device bytes allowed by this cache.
    pub fn max_retained_device_bytes(&self) -> u64 {
        self.inner.max_retained_device_bytes
    }

    /// Whether retained bytes are owned by the per-device memory governor's
    /// evictable ledger instead of by per-query reservation baselines.
    pub fn is_governor_backed(&self) -> bool {
        self.inner.governor.is_some()
    }

    /// Register this cache as a synchronous evictable-memory observer on its
    /// backing governor.
    pub fn register_governor_pressure_observer(cache: &Arc<Self>) -> Result<()> {
        let Some(governor) = cache.inner.governor.as_ref().cloned() else {
            return Ok(());
        };
        let weak_cache: Weak<Self> = Arc::downgrade(cache);
        governor.register_evictable_memory_observer(Arc::new(move |pressure| {
            let Some(cache) = weak_cache.upgrade() else {
                return Ok(0);
            };
            cache.handle_memory_pressure(pressure)
        }))
    }

    /// Number of retained source-result entries.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.cache.len())
    }

    /// Whether the cache currently holds no source-result entries.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Observed retained device bytes across all cache entries.
    pub fn retained_device_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.retained_device_bytes)
    }

    pub(crate) fn evict_at_least_protecting(
        &self,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<u64> {
        let mut state = self.lock()?;
        self.evict_at_least_locked(&mut state, target_device_bytes, protected_keys)
    }

    pub(crate) fn get(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<Option<(GpuResultHandle, u64)>> {
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_context(key, context);
        let mut state = self.lock()?;
        let access_epoch = state.next_epoch();
        let Some(entry) = state.cache.get_mut(&entry_key) else {
            return Ok(None);
        };
        if !entry.handle.can_reuse_for_source_cache(context, schema)? {
            let released = if entry.in_use_count == 0 { state.remove_entry(&entry_key) } else { 0 };
            drop(state);
            self.release_governor_evictable_bytes(released)?;
            return Ok(None);
        }
        entry.in_use_count = entry.in_use_count.saturating_add(1);
        entry.last_access_epoch = access_epoch;
        let pin = GpuLocalParquetSourceCachePin {
            _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                cache: Arc::clone(&self.inner),
                entry_key: entry_key.clone(),
                generation: entry.generation,
            }),
        };
        let handle = entry.handle.clone().with_source_cache_pin(pin);
        let retained_device_bytes = entry.retained_device_bytes;
        Ok(Some((handle, retained_device_bytes)))
    }

    pub(in crate::exec) fn insert_if_absent_from_grant_pinned(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<GpuLocalParquetSourceCachePinnedInsert> {
        let (insert, handle) = self.insert_if_absent_protecting_internal(
            key,
            handle,
            retained_device_bytes,
            &HashSet::new(),
            memory_grant,
            true,
        )?;
        Ok(GpuLocalParquetSourceCachePinnedInsert { outcome: insert.outcome, handle })
    }

    pub(crate) fn insert_if_absent_protecting(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<GpuLocalParquetSourceCacheInsertResult> {
        Ok(self
            .insert_if_absent_protecting_internal(
                key,
                handle,
                retained_device_bytes,
                protected_keys,
                memory_grant,
                false,
            )?
            .0)
    }

    fn insert_if_absent_protecting_internal(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        memory_grant: Option<&MemoryGrant>,
        pin_returned_handle: bool,
    ) -> Result<(GpuLocalParquetSourceCacheInsertResult, Option<GpuResultHandle>)> {
        let entry_key = GpuLocalParquetSourceCacheEntryKey::for_handle(key, &handle);
        let mut state = self.lock()?;
        let retained_before = state.retained_device_bytes;
        let access_epoch = state.next_epoch();
        if let Some(entry) = state.cache.get_mut(&entry_key) {
            let reusable = !pin_returned_handle || entry.handle.can_reuse_for_source_cache_handle(&handle)?;
            if !reusable {
                return Ok((
                    GpuLocalParquetSourceCacheInsertResult {
                        outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                        evicted_device_bytes: 0,
                        retained_device_bytes: retained_before,
                    },
                    None,
                ));
            }
            entry.last_access_epoch = access_epoch;
            let handle = if pin_returned_handle {
                entry.in_use_count = entry.in_use_count.saturating_add(1);
                Some(entry.handle.clone().with_source_cache_pin(GpuLocalParquetSourceCachePin {
                    _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                        cache: Arc::clone(&self.inner),
                        entry_key: entry_key.clone(),
                        generation: entry.generation,
                    }),
                }))
            } else {
                None
            };
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                handle,
            ));
        }
        if handle.has_live_reservation() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.insert",
                "cannot retain a source cache entry that carries a query live-reservation",
            )
            .with_fact("reason", "query_live_reservation_in_cache_handle"));
        }
        if handle.has_source_cache_pin() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.insert",
                "cannot retain a source cache entry that already carries a source-cache pin",
            )
            .with_fact("reason", "source_cache_pin_in_cache_handle"));
        }
        if retained_device_bytes > self.inner.max_retained_device_bytes {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes: 0,
                    retained_device_bytes: retained_before,
                },
                None,
            ));
        }
        if !self.governor_can_admit_after_cache_eviction(&state, retained_device_bytes, protected_keys, memory_grant)? {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes: 0,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        let evicted_device_bytes =
            state.evict_until_fits(self.inner.max_retained_device_bytes, retained_device_bytes, protected_keys);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        if state.retained_device_bytes.saturating_add(retained_device_bytes) > self.inner.max_retained_device_bytes {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        if !self.reserve_governor_evictable_bytes(retained_device_bytes, memory_grant)? {
            return Ok((
                GpuLocalParquetSourceCacheInsertResult {
                    outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                    evicted_device_bytes,
                    retained_device_bytes: state.retained_device_bytes,
                },
                None,
            ));
        }
        let generation = state.next_generation();
        let pin = pin_returned_handle.then(|| GpuLocalParquetSourceCachePin {
            _inner: Arc::new(GpuLocalParquetSourceCachePinInner {
                cache: Arc::clone(&self.inner),
                entry_key: entry_key.clone(),
                generation,
            }),
        });
        let returned_handle = pin.clone().map(|pin| handle.clone().with_source_cache_pin(pin));
        state.cache.insert(
            entry_key,
            GpuLocalParquetSourceCacheEntry {
                handle,
                retained_device_bytes,
                last_access_epoch: access_epoch,
                in_use_count: if pin.is_some() { 1 } else { 0 },
                generation,
            },
        );
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        Ok((
            GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::Inserted,
                evicted_device_bytes,
                retained_device_bytes: state.retained_device_bytes,
            },
            returned_handle,
        ))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GpuLocalParquetSourceCacheState>> {
        self.inner.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache",
                "native GPU local Parquet source cache lock was poisoned",
            )
        })
    }

    fn handle_memory_pressure(&self, pressure: EvictableMemoryPressure) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(0);
        };
        if pressure.device_ordinal != governor.device_ordinal() {
            return Ok(0);
        }
        let mut state = self.lock()?;
        self.evict_at_least_locked(&mut state, pressure.requested_evict_bytes, &HashSet::new())
    }

    fn evict_at_least_locked(
        &self,
        state: &mut GpuLocalParquetSourceCacheState,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<u64> {
        let evicted_device_bytes = state.evict_at_least(target_device_bytes, protected_keys);
        self.release_governor_evictable_bytes(evicted_device_bytes)?;
        Ok(evicted_device_bytes)
    }

    fn governor_can_admit_after_cache_eviction(
        &self,
        state: &GpuLocalParquetSourceCacheState,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        let evictable_for_capacity = state.evictable_bytes_until_fits(
            self.inner.max_retained_device_bytes,
            retained_device_bytes,
            protected_keys,
        );
        let free_bytes = governor.free_bytes()?;
        if free_bytes.saturating_add(evictable_for_capacity) >= retained_device_bytes {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.grant_can_cover_evictable_bytes(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn reserve_governor_evictable_bytes(
        &self,
        retained_device_bytes: u64,
        memory_grant: Option<&MemoryGrant>,
    ) -> Result<bool> {
        let Some(governor) = &self.inner.governor else {
            return Ok(true);
        };
        if governor.try_reserve_evictable_bytes(retained_device_bytes)? {
            return Ok(true);
        }
        match memory_grant {
            Some(memory_grant) => governor.try_reserve_evictable_bytes_from_grant(memory_grant, retained_device_bytes),
            None => Ok(false),
        }
    }

    fn release_governor_evictable_bytes(&self, released_device_bytes: u64) -> Result<u64> {
        let Some(governor) = &self.inner.governor else {
            return Ok(released_device_bytes);
        };
        governor.release_evictable_bytes(released_device_bytes)
    }
}

fn source_column_memory_identities(frame: &DataFrame) -> HashSet<ColumnMemoryIdentity> {
    frame.memory_identity().columns.into_iter().map(|column| column.identity).collect()
}

impl GpuLocalParquetSourceCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
    }

    fn next_generation(&mut self) -> u64 {
        let generation = self.next_entry_generation;
        self.next_entry_generation = self.next_entry_generation.saturating_add(1);
        generation
    }

    fn evict_until_fits(
        &mut self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut evicted_device_bytes = 0u64;
        while self.retained_device_bytes.saturating_add(incoming_retained_device_bytes) > max_retained_device_bytes {
            let Some(evict_key) = self.lru_unprotected_key(protected_keys) else {
                break;
            };
            if let Some(evicted) = self.cache.remove(&evict_key) {
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(evicted.retained_device_bytes);
                evicted_device_bytes = evicted_device_bytes.saturating_add(evicted.retained_device_bytes);
            }
        }
        evicted_device_bytes
    }

    fn evictable_bytes_until_fits(
        &self,
        max_retained_device_bytes: u64,
        incoming_retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut retained_device_bytes = self.retained_device_bytes;
        let mut evictable_device_bytes = 0u64;
        let mut candidates = self
            .cache
            .iter()
            .filter(|(key, entry)| entry.in_use_count == 0 && !protected_keys.contains(&key.source_key))
            .map(|(_key, entry)| (entry.last_access_epoch, entry.retained_device_bytes))
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|(last_access_epoch, _bytes)| *last_access_epoch);
        for (_last_access_epoch, candidate_bytes) in candidates {
            if retained_device_bytes.saturating_add(incoming_retained_device_bytes) <= max_retained_device_bytes {
                break;
            }
            retained_device_bytes = retained_device_bytes.saturating_sub(candidate_bytes);
            evictable_device_bytes = evictable_device_bytes.saturating_add(candidate_bytes);
        }
        evictable_device_bytes
    }

    fn evict_at_least(
        &mut self,
        target_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> u64 {
        let mut evicted_device_bytes = 0u64;
        while evicted_device_bytes < target_device_bytes {
            let Some(evict_key) = self.lru_unprotected_key(protected_keys) else {
                break;
            };
            if let Some(evicted) = self.cache.remove(&evict_key) {
                self.retained_device_bytes = self.retained_device_bytes.saturating_sub(evicted.retained_device_bytes);
                evicted_device_bytes = evicted_device_bytes.saturating_add(evicted.retained_device_bytes);
            }
        }
        evicted_device_bytes
    }

    fn remove_entry(&mut self, entry_key: &GpuLocalParquetSourceCacheEntryKey) -> u64 {
        let Some(removed) = self.cache.remove(entry_key) else {
            return 0;
        };
        self.retained_device_bytes = self.retained_device_bytes.saturating_sub(removed.retained_device_bytes);
        removed.retained_device_bytes
    }

    fn lru_unprotected_key(
        &self,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Option<GpuLocalParquetSourceCacheEntryKey> {
        self.cache
            .iter()
            .filter(|(key, entry)| entry.in_use_count == 0 && !protected_keys.contains(&key.source_key))
            .min_by_key(|(_key, entry)| entry.last_access_epoch)
            .map(|(key, _entry)| key.clone())
    }
}

impl GpuResultHandle {
    fn can_reuse_for_source_cache(&self, context: &ExecutionContext, schema: &SchemaRef) -> Result<bool> {
        if self.schema().as_ref() != schema.as_ref() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.reuse",
                "cached local Parquet source schema does not match the requested source schema",
            )
            .with_fact("reason", "schema_mismatch")
            .with_fact("cached_fields", self.schema().fields().len())
            .with_fact("requested_fields", schema.fields().len()));
        }
        let context_runtime = context.shared_runtime();
        if !Arc::ptr_eq(self.runtime(), &context_runtime) {
            return Ok(false);
        }
        if self.device_placement() != context.device_placement()
            || self.placement().device != context.runtime().device()
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn can_reuse_for_source_cache_handle(&self, requested: &Self) -> Result<bool> {
        if self.schema().as_ref() != requested.schema().as_ref() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache.reuse",
                "cached local Parquet source schema does not match the requested source schema",
            )
            .with_fact("reason", "schema_mismatch")
            .with_fact("cached_fields", self.schema().fields().len())
            .with_fact("requested_fields", requested.schema().fields().len()));
        }
        if !Arc::ptr_eq(self.runtime(), requested.runtime()) {
            return Ok(false);
        }
        if self.device_placement() != requested.device_placement()
            || self.placement().device != requested.placement().device
        {
            return Ok(false);
        }
        Ok(true)
    }
}

fn push_schema(output: &mut String, schema: &SchemaRef) {
    push_token(output, "schema_field_count", schema.fields().len().to_string());
    for (index, field) in schema.fields().iter().enumerate() {
        push_token(output, format!("schema.{index}.name"), field.name());
        push_token(output, format!("schema.{index}.data_type"), format!("{:?}", field.data_type()));
        push_token(output, format!("schema.{index}.nullable"), field.is_nullable().to_string());
        for (key, value) in field.metadata() {
            push_token(output, format!("schema.{index}.metadata.{}", escape_token(key)), value);
        }
    }
}

fn push_source_scan_contract(output: &mut String, source: &LocalParquetSource) {
    push_schema(output, &source.schema);
    push_token(output, "count_star_probe", source.is_count_star_probe_source().to_string());
    match &source.output_schema_override {
        Some(schema) => {
            push_token(output, "output_schema_override", "some");
            push_schema(output, schema);
        },
        None => push_token(output, "output_schema_override", "none"),
    }
    push_string_vec(output, "read_columns", &source.read_columns);
    push_usize_vec(output, "read_column_file_indices", &source.read_column_file_indices);
    push_string_vec(output, "projected_output_columns", &source.projected_output_columns);
    push_source_refs(output, "projected_output_column_source_refs", &source.projected_output_column_source_refs);
    push_string_vec(output, "pure_filter_columns", &source.pure_filter_columns);
    push_source_refs(output, "pure_filter_column_source_refs", &source.pure_filter_column_source_refs);
    push_token(
        output,
        "reader_predicate",
        source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string),
    );
}

fn push_resolved_source_contract(output: &mut String, label: impl AsRef<str>, resolved_source: &ResolvedParquetSource) {
    let label = label.as_ref();
    push_optional_string(
        output,
        format!("{label}.original_location"),
        Some(resolved_source.original_location.as_str()),
    );
    push_optional_string(output, format!("{label}.reader_location"), Some(resolved_source.reader_location.as_str()));
    push_optional_string(output, format!("{label}.storage_domain"), resolved_source.storage_domain.as_deref());
    push_optional_string(output, format!("{label}.content_identity"), resolved_source.content_identity.as_deref());
    push_optional_string(output, format!("{label}.storage_profile_id"), resolved_source.storage_profile_id.as_deref());
    push_token(output, format!("{label}.read_strategy"), resolved_source.read_strategy.as_str());
    push_token(output, format!("{label}.decode_input"), resolved_source.decode_input.as_str());
    push_token(output, format!("{label}.remote_read_mode"), resolved_source.remote_read_mode.as_str());
    push_token(output, format!("{label}.credential_mode"), resolved_source.credential_mode.as_str());
    push_token(output, format!("{label}.lakehouse_cache_status"), resolved_source.cache_status.as_str());
    push_token(
        output,
        format!("{label}.lakehouse_cache_read_mode"),
        resolved_source.lakehouse_cache_read_mode.as_str(),
    );
    push_optional_string(
        output,
        format!("{label}.lakehouse_cache_root"),
        resolved_source.lakehouse_cache_root.as_deref(),
    );
    push_optional_string(
        output,
        format!("{label}.lakehouse_cache_path"),
        resolved_source.lakehouse_cache_path.as_deref(),
    );
    push_optional_u64(output, format!("{label}.lakehouse_cache_max_bytes"), resolved_source.lakehouse_cache_max_bytes);
    push_token(
        output,
        format!("{label}.lakehouse_cache_fill_mode"),
        resolved_source.lakehouse_cache_fill_mode.as_str(),
    );
    push_token(
        output,
        format!("{label}.remote_host_buffer_ranged_enabled"),
        resolved_source.remote_host_buffer_ranged_enabled.to_string(),
    );
    push_optional_string(output, format!("{label}.fallback_reason"), resolved_source.fallback_reason.as_deref());
}

fn push_position_delete_files(
    output: &mut String,
    label: impl AsRef<str>,
    delete_files: &[crate::source::ParquetPositionDeleteFile],
) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), delete_files.len().to_string());
    for (index, delete_file) in delete_files.iter().enumerate() {
        let label = format!("{label}.{index}");
        push_token(output, format!("{label}.id"), delete_file.id.as_str());
        push_token(output, format!("{label}.location"), delete_file.location.reader_path());
        push_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        push_token(output, format!("{label}.stable_id"), delete_file.stable_id.as_str());
        push_optional_u64(output, format!("{label}.size"), delete_file.file_size_bytes);
        push_optional_u64(output, format!("{label}.rows"), delete_file.row_count);
        push_optional_string(
            output,
            format!("{label}.referenced_data_file"),
            delete_file.referenced_data_file.as_deref(),
        );
        push_optional_u64(output, format!("{label}.position_lower_bound"), delete_file.position_lower_bound);
        push_optional_u64(output, format!("{label}.position_upper_bound"), delete_file.position_upper_bound);
    }
}

fn push_equality_delete_files(
    output: &mut String,
    label: impl AsRef<str>,
    delete_files: &[crate::source::ParquetEqualityDeleteFile],
) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), delete_files.len().to_string());
    for (index, delete_file) in delete_files.iter().enumerate() {
        let label = format!("{label}.{index}");
        push_token(output, format!("{label}.id"), delete_file.id.as_str());
        push_token(output, format!("{label}.location"), delete_file.location.reader_path());
        push_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        push_token(output, format!("{label}.stable_id"), delete_file.stable_id.as_str());
        push_i32_vec(output, format!("{label}.equality_ids"), &delete_file.equality_ids);
        push_string_vec(output, format!("{label}.equality_columns"), &delete_file.equality_columns);
        push_optional_u64(output, format!("{label}.size"), delete_file.file_size_bytes);
        push_optional_u64(output, format!("{label}.rows"), delete_file.row_count);
    }
}

fn redact_source_cache_fingerprint(output: &mut String, source: &LocalParquetSource) {
    for (index, file) in source.files.iter().enumerate() {
        replace_token_value(
            output,
            format!("file.{index}.location"),
            file.location.reader_path().as_str(),
            file.location.diagnostic_location().as_str(),
        );
        replace_token_value(
            output,
            format!("file.{index}.stable_id"),
            file.stable_id.as_str(),
            file.location.diagnostic_stable_id(file.stable_id.as_str()).as_str(),
        );
        redact_resolved_source_contract(output, format!("file.{index}.resolved"), &file.resolved_source);
        for (delete_index, delete_file) in file.position_delete_files.iter().enumerate() {
            let label = format!("file.{index}.position_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            if let Some(referenced_data_file) = delete_file.referenced_data_file.as_deref() {
                replace_token_value(
                    output,
                    format!("{label}.referenced_data_file"),
                    referenced_data_file,
                    redacted_position_delete_referenced_data_file(referenced_data_file).as_str(),
                );
            }
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
        for (delete_index, delete_file) in file.equality_delete_files.iter().enumerate() {
            let label = format!("file.{index}.equality_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
    }
}

fn redact_source_chunk_cache_fingerprint(output: &mut String, source: &LocalParquetSource, chunk: &SourceChunk) {
    replace_token_value(
        output,
        "chunk.location",
        chunk.location.reader_path().as_str(),
        chunk.location.diagnostic_location().as_str(),
    );
    replace_token_value(
        output,
        "chunk.stable_id",
        chunk.stable_id.as_str(),
        chunk.location.diagnostic_stable_id(chunk.stable_id.as_str()).as_str(),
    );
    redact_resolved_source_contract(output, "chunk.resolved", &chunk.resolved_source);
    if let Some(file) = source.files.get(chunk.file_index) {
        for (delete_index, delete_file) in file.position_delete_files.iter().enumerate() {
            let label = format!("chunk.position_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            if let Some(referenced_data_file) = delete_file.referenced_data_file.as_deref() {
                replace_token_value(
                    output,
                    format!("{label}.referenced_data_file"),
                    referenced_data_file,
                    redacted_position_delete_referenced_data_file(referenced_data_file).as_str(),
                );
            }
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
        for (delete_index, delete_file) in file.equality_delete_files.iter().enumerate() {
            let label = format!("chunk.equality_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
    }
}

fn redacted_position_delete_referenced_data_file(path: &str) -> String {
    format!("stable_id_fnv64:{:016x}", stable_fnv1a64(path.as_bytes()))
}

fn redact_resolved_source_contract(
    output: &mut String,
    label: impl AsRef<str>,
    resolved_source: &ResolvedParquetSource,
) {
    let label = label.as_ref();
    replace_token_value(
        output,
        format!("{label}.original_location"),
        resolved_source.original_location.as_str(),
        stable_hashed_string(resolved_source.original_location.as_str()).as_str(),
    );
    replace_token_value(
        output,
        format!("{label}.reader_location"),
        resolved_source.reader_location.as_str(),
        stable_hashed_string(resolved_source.reader_location.as_str()).as_str(),
    );
    replace_optional_token_value(output, format!("{label}.storage_domain"), resolved_source.storage_domain.as_deref());
    replace_optional_token_value(
        output,
        format!("{label}.content_identity"),
        resolved_source.content_identity.as_deref(),
    );
    replace_optional_token_value(
        output,
        format!("{label}.lakehouse_cache_root"),
        resolved_source.lakehouse_cache_root.as_deref(),
    );
    replace_optional_token_value(
        output,
        format!("{label}.lakehouse_cache_path"),
        resolved_source.lakehouse_cache_path.as_deref(),
    );
}

fn replace_optional_token_value(output: &mut String, key: impl AsRef<str>, value: Option<&str>) {
    if let Some(value) = value {
        replace_token_value(output, key, value, stable_hashed_string(value).as_str());
    }
}

fn replace_token_value(output: &mut String, key: impl AsRef<str>, current_value: &str, new_value: &str) {
    let key = escape_token(key.as_ref());
    let current_value = escape_token(current_value);
    let new_value = escape_token(new_value);
    let needle = format!("{key}={current_value};");
    let replacement = format!("{key}={new_value};");
    *output = output.replace(&needle, replacement.as_str());
}

fn push_source_refs(output: &mut String, label: impl AsRef<str>, refs: &[SourceColumnRef]) {
    let values = refs.iter().map(format_source_ref).collect::<Vec<_>>();
    push_string_vec(output, label, &values);
}

fn format_source_ref(source_ref: &SourceColumnRef) -> String {
    match source_ref {
        SourceColumnRef::File(index) => format!("file:{index}"),
        SourceColumnRef::Partition(index) => format!("partition:{index}"),
        SourceColumnRef::MissingNull(index) => format!("missing_null:{index}"),
    }
}

fn push_string_vec(output: &mut String, label: impl AsRef<str>, values: &[String]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value);
    }
}

fn push_usize_vec(output: &mut String, label: impl AsRef<str>, values: &[usize]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
    }
}

fn push_i32_vec(output: &mut String, label: impl AsRef<str>, values: &[i32]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
    }
}

fn push_reader_decimal_types(output: &mut String, label: impl AsRef<str>, values: &[ParquetReaderDecimalType]) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{label}.{index}.column_name"), &value.column_name);
        push_token(output, format!("{label}.{index}.type_id"), value.type_id.as_str());
        push_token(output, format!("{label}.{index}.cudf_scale"), value.cudf_scale.to_string());
    }
}

fn push_optional_usize_vec(output: &mut String, label: impl AsRef<str>, values: Option<&[usize]>) {
    match values {
        Some(values) => push_usize_vec(output, label, values),
        None => push_token(output, label, "none"),
    }
}

fn push_optional_u64(output: &mut String, label: impl AsRef<str>, value: Option<u64>) {
    match value {
        Some(value) => push_token(output, label, value.to_string()),
        None => push_token(output, label, "none"),
    }
}

fn push_optional_u64_vec(output: &mut String, label: impl AsRef<str>, values: Option<&[u64]>) {
    match values {
        Some(values) => {
            push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
            for (index, value) in values.iter().enumerate() {
                push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
            }
        },
        None => push_token(output, label, "none"),
    }
}

fn push_optional_string(output: &mut String, label: impl AsRef<str>, value: Option<&str>) {
    match value {
        Some(value) => push_token(output, label, value),
        None => push_token(output, label, "none"),
    }
}

fn stable_hashed_string(value: &str) -> String {
    format!("fnv64:{:016x}:len:{}", stable_fnv1a64(value.as_bytes()), value.len())
}

fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn push_token(output: &mut String, key: impl AsRef<str>, value: impl ToString) {
    let _ = write!(output, "{}={};", escape_token(key.as_ref()), escape_token(&value.to_string()));
}

fn escape_token(value: &str) -> String {
    escape_kv_delimited(value)
}

/// Output produced by native query execution.
///
/// The variant is driven by the query plan's sink. Host Arrow output has
/// crossed the device-to-host boundary; GPU-resident output remains bound to
/// its originating runtime and must be materialized explicitly if a caller
/// needs Arrow.
pub enum QueryOutput {
    /// GPU-resident result bound to its originating runtime/device.
    GpuResident(GpuResultHandle),
    /// Host Arrow result after explicit device-to-host materialization.
    HostArrow(RecordBatch),
}

impl QueryOutput {
    /// Output residency mode represented by this value.
    pub fn mode(&self) -> QueryOutputMode {
        match self {
            Self::GpuResident(_) => QueryOutputMode::GpuResident,
            Self::HostArrow(_) => QueryOutputMode::MaterializeToHost,
        }
    }

    /// Borrow the host Arrow batch when the result was materialized to host.
    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    /// Borrow the GPU handle when the result stayed device-resident.
    pub fn as_gpu_resident(&self) -> Option<&GpuResultHandle> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

/// Native query output plus metrics collected during that execution.
///
/// Metrics ownership stays with the result so adapters can record them after
/// deciding whether to pass through a GPU handle or materialize host output.
pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

impl QueryExecutionResult {
    /// Create an execution result with its collected metrics.
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    /// Borrow the query output without consuming metrics.
    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    /// Borrow metrics collected during query execution.
    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    /// Consume the result into output and metrics.
    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::ops::expr::col;

    use super::*;
    use crate::runtime::gpu::governor::GrantRequest;
    #[cfg(feature = "cugraph")]
    use crate::{
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
            GraphConstructionPolicy, GraphEdgeColumnBindings, GraphResidencyOrientation, GraphResidencyPlanAnalysis,
            QueryPlanBuilder, graph_residency_descriptor_for_edge_plan,
        },
        source::{LocalParquetFile, LocalParquetSource, SourceNode},
    };

    #[cfg(feature = "cugraph")]
    fn test_resident_graph_entry_key() -> GpuResidentGraphCacheEntryKey {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int32, false),
            Field::new("dst", DataType::Int32, false),
        ]));
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/preparation-pin-edges.parquet")
                    .with_file_size_bytes(1)
                    .with_stable_id("local:/tmp/preparation-pin-edges.parquet:1:mtime=test"),
            ],
            schema,
            vec!["src".to_string(), "dst".to_string()],
        );
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source);
        let GraphResidencyPlanAnalysis::Keyable(descriptor) = graph_residency_descriptor_for_edge_plan(&plan, &spec)
        else {
            panic!("test edge source should produce a resident graph key")
        };
        GpuResidentGraphCacheEntryKey {
            resident_key: descriptor.resident_key(&spec, GraphResidencyOrientation::Forward),
            runtime_key: 1,
        }
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn resident_graph_preparation_pin_reserves_checkout_for_its_owner() {
        let entry_key = test_resident_graph_entry_key();
        let mut state = GpuResidentGraphCacheState::default();
        state.preparation_pins.insert(entry_key.clone(), 41);

        assert!(!state.consume_checkout_preparation_pin(&entry_key, None));
        assert_eq!(state.preparation_pins.get(&entry_key), Some(&41));
        assert!(!state.consume_checkout_preparation_pin(&entry_key, Some(42)));
        assert_eq!(state.preparation_pins.get(&entry_key), Some(&41));
        assert!(state.consume_checkout_preparation_pin(&entry_key, Some(41)));
        assert!(!state.preparation_pins.contains_key(&entry_key));
    }

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
        let live_reservation = grant.retain_live_reservation(observed_bytes);
        (GpuDataFrame::new_with_live_reservation(frame, live_reservation), observed_bytes)
    }

    #[test]
    fn source_residency_clones_live_reservation_for_pure_alias_output() {
        let runtime = Arc::new(cudf::Runtime::new().unwrap());
        let governor = MemoryGovernor::new(runtime.device().ordinal(), 128 * 1024 * 1024).unwrap();
        let grant = governor.try_request(GrantRequest::new(1, 64 * 1024 * 1024, 64 * 1024 * 1024)).unwrap().unwrap();
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
        let governor = MemoryGovernor::new(runtime.device().ordinal(), 128 * 1024 * 1024).unwrap();
        let grant = governor.try_request(GrantRequest::new(2, 64 * 1024 * 1024, 64 * 1024 * 1024)).unwrap().unwrap();
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
        let governor = MemoryGovernor::new(runtime.device().ordinal(), 128 * 1024 * 1024).unwrap();
        let grant = governor.try_request(GrantRequest::new(3, 64 * 1024 * 1024, 64 * 1024 * 1024)).unwrap().unwrap();
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
