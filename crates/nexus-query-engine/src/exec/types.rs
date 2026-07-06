//! Public GPU-resident execution handles, outputs, and caches.
//!
//! Handles in this module are runtime/device-affine. Cloning shares the
//! underlying cuDF resources; host materialization is explicit. Cache byte
//! counters are observed retained device bytes, not reservation totals.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write,
    sync::{Arc, Mutex},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, DataType, ExecutionPlacement,
    frame::dataframe::{GpuRecordBatch, InfoDetail},
};

use crate::{
    chunk::SourceChunk,
    error::{QueryError, Result},
    exec::is_datafusion_order_column_name,
    metrics::QueryMetrics,
    plan::{HashJoinBuildArtifactId, JoinNullEquality, ResultSink},
    runtime::{DevicePlacement, ExecutionContext, gpu::governor::MemoryGrantLiveReservation},
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
}

impl GpuDataFrame {
    /// Wrap a cuDF frame and record its current execution placement.
    pub fn new(frame: DataFrame) -> Self {
        Self::new_with_live_reservation(frame, None)
    }

    /// Wrap a derived cuDF frame while charging it to the same live-residency
    /// lineage as `source`.
    pub fn try_new_with_shared_residency(frame: DataFrame, source: &Self) -> Result<Self> {
        let live_reservation = if let Some(source_live_reservation) = &source.live_reservation {
            let observed_bytes = Self::new(frame.clone()).device_allocated_size_bytes()?;
            source_live_reservation.retain_sibling_live_reservation(observed_bytes)?
        } else {
            None
        };
        Ok(Self::new_with_live_reservation(frame, live_reservation))
    }

    fn new_with_live_reservation(frame: DataFrame, live_reservation: Option<MemoryGrantLiveReservation>) -> Self {
        let placement = frame.placement();
        Self { frame, placement, live_reservation }
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
        debug_assert_eq!(self.placement, self.frame.placement());
        (self.frame, self.live_reservation)
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
    _live_reservation: Option<MemoryGrantLiveReservation>,
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
        let (frame, frame_live_reservation) = frame.into_frame_and_live_reservation();
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
            _live_reservation: live_reservation,
        })
    }

    /// Borrow the underlying GPU cuDF frame.
    pub fn frame(&self) -> &DataFrame {
        self.batch.frame()
    }

    /// Consume the handle and return its GPU frame wrapper.
    pub fn into_gpu_frame(self) -> GpuDataFrame {
        let Self { batch, runtime: _, device_placement: _, _live_reservation } = self;
        GpuDataFrame::new_with_live_reservation(batch.into_frame(), _live_reservation)
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

    let (raw, live_reservation) = frame.into_frame_and_live_reservation();
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
        return Ok(GpuDataFrame::new_with_live_reservation(selected, live_reservation));
    }

    Ok(GpuDataFrame::new_with_live_reservation(raw, live_reservation))
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
        } else {
            push_token(&mut fingerprint, "chunk.position_delete_files.len", "missing");
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
}

#[derive(Default)]
struct GpuLocalParquetSourceCacheState {
    cache: HashMap<GpuLocalParquetSourceCacheKey, GpuLocalParquetSourceCacheEntry>,
    retained_device_bytes: u64,
    next_access_epoch: u64,
}

/// LRU cache for GPU-resident local Parquet source results.
///
/// Capacity and retained sizes are device bytes. Cached handles remain
/// runtime/device-affine; a lookup with a different runtime or device silently
/// misses, while a schema mismatch is treated as an invalid plan.
pub struct GpuLocalParquetSourceCache {
    max_retained_device_bytes: u64,
    state: Mutex<GpuLocalParquetSourceCacheState>,
}

impl std::fmt::Debug for GpuLocalParquetSourceCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().ok();
        f.debug_struct("GpuLocalParquetSourceCache")
            .field("max_retained_device_bytes", &self.max_retained_device_bytes)
            .field("entry_count", &state.as_ref().map(|state| state.cache.len()))
            .field("retained_device_bytes", &state.as_ref().map(|state| state.retained_device_bytes))
            .finish()
    }
}

impl GpuLocalParquetSourceCache {
    /// Create an empty source cache with a retained-device-byte capacity.
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self { max_retained_device_bytes, state: Mutex::new(GpuLocalParquetSourceCacheState::default()) }
    }

    /// Maximum retained device bytes allowed by this cache.
    pub fn max_retained_device_bytes(&self) -> u64 {
        self.max_retained_device_bytes
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
        Ok(state.evict_at_least(target_device_bytes, protected_keys))
    }

    pub(crate) fn get(
        &self,
        key: &GpuLocalParquetSourceCacheKey,
        context: &ExecutionContext,
        schema: &SchemaRef,
    ) -> Result<Option<(GpuResultHandle, u64)>> {
        let mut state = self.lock()?;
        let access_epoch = state.next_epoch();
        let Some(entry) = state.cache.get_mut(key) else {
            return Ok(None);
        };
        entry.last_access_epoch = access_epoch;
        let handle = entry.handle.clone();
        let retained_device_bytes = entry.retained_device_bytes;
        drop(state);
        if !handle.can_reuse_for_source_cache(context, schema)? {
            let mut state = self.lock()?;
            if let Some(removed) = state.cache.remove(key) {
                state.retained_device_bytes = state.retained_device_bytes.saturating_sub(removed.retained_device_bytes);
            }
            return Ok(None);
        }
        Ok(Some((handle, retained_device_bytes)))
    }

    pub(crate) fn insert_if_absent(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
    ) -> Result<GpuLocalParquetSourceCacheInsert> {
        Ok(self.insert_if_absent_protecting(key, handle, retained_device_bytes, &HashSet::new())?.outcome)
    }

    pub(crate) fn insert_if_absent_protecting(
        &self,
        key: GpuLocalParquetSourceCacheKey,
        handle: GpuResultHandle,
        retained_device_bytes: u64,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Result<GpuLocalParquetSourceCacheInsertResult> {
        let mut state = self.lock()?;
        let retained_before = state.retained_device_bytes;
        if state.cache.contains_key(&key) {
            return Ok(GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::AlreadyPresent,
                evicted_device_bytes: 0,
                retained_device_bytes: retained_before,
            });
        }
        if retained_device_bytes > self.max_retained_device_bytes {
            return Ok(GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                evicted_device_bytes: 0,
                retained_device_bytes: retained_before,
            });
        }
        let evicted_device_bytes =
            state.evict_until_fits(self.max_retained_device_bytes, retained_device_bytes, protected_keys);
        if state.retained_device_bytes.saturating_add(retained_device_bytes) > self.max_retained_device_bytes {
            return Ok(GpuLocalParquetSourceCacheInsertResult {
                outcome: GpuLocalParquetSourceCacheInsert::CapacitySkipped,
                evicted_device_bytes,
                retained_device_bytes: state.retained_device_bytes,
            });
        }
        let last_access_epoch = state.next_epoch();
        state.cache.insert(key, GpuLocalParquetSourceCacheEntry { handle, retained_device_bytes, last_access_epoch });
        state.retained_device_bytes = state.retained_device_bytes.saturating_add(retained_device_bytes);
        Ok(GpuLocalParquetSourceCacheInsertResult {
            outcome: GpuLocalParquetSourceCacheInsert::Inserted,
            evicted_device_bytes,
            retained_device_bytes: state.retained_device_bytes,
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GpuLocalParquetSourceCacheState>> {
        self.state.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_local_parquet_source_cache",
                "native GPU local Parquet source cache lock was poisoned",
            )
        })
    }
}

impl GpuLocalParquetSourceCacheState {
    fn next_epoch(&mut self) -> u64 {
        let epoch = self.next_access_epoch;
        self.next_access_epoch = self.next_access_epoch.saturating_add(1);
        epoch
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

    fn lru_unprotected_key(
        &self,
        protected_keys: &HashSet<GpuLocalParquetSourceCacheKey>,
    ) -> Option<GpuLocalParquetSourceCacheKey> {
        self.cache
            .iter()
            .filter(|(key, _entry)| !protected_keys.contains(*key))
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
