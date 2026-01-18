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
    metrics::QueryMetrics,
    plan::{HashJoinBuildArtifactId, JoinNullEquality, ResultSink},
    runtime::{DevicePlacement, ExecutionContext},
    source::{LocalParquetSource, SourceColumnRef},
};

pub type GpuResidentInputMap = BTreeMap<String, GpuResultHandle>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    GpuResident,
    MaterializeToHost,
}

impl ExecutionMode {
    pub fn for_sink(sink: &ResultSink) -> Self {
        match sink {
            ResultSink::GpuResident { .. } => Self::GpuResident,
            ResultSink::HostArrow { .. } => Self::MaterializeToHost,
        }
    }
}

#[derive(Clone)]
pub struct GpuDataFrame {
    frame: DataFrame,
    placement: ExecutionPlacement,
}

impl GpuDataFrame {
    pub fn new(frame: DataFrame) -> Self {
        let placement = frame.placement();
        Self { frame, placement }
    }

    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    pub fn into_frame(self) -> DataFrame {
        debug_assert_eq!(self.placement, self.frame.placement());
        self.frame
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.placement
    }

    pub fn num_rows(&self) -> usize {
        self.frame.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.frame.num_columns()
    }

    pub fn device_allocated_size_bytes(&self) -> Result<u64> {
        let info = self.frame.info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.gpu_dataframe.device_memory",
                "failed to observe native GPU dataframe memory",
                error,
            )
        })?;
        let bytes =
            info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
                QueryError::new(
                    crate::error::QueryErrorCode::Internal,
                    "cudf_query_engine.gpu_dataframe.device_memory",
                    "cudf dataframe memory accounting did not include device-allocated bytes",
                )
                .with_fact("rows", self.num_rows())
                .with_fact("columns", self.num_columns())
            })?;
        Ok(bytes as u64)
    }
}

#[derive(Clone)]
pub struct GpuResultHandle {
    batch: GpuRecordBatch,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
}

impl GpuResultHandle {
    pub fn new(context: &ExecutionContext, frame: GpuDataFrame, schema: SchemaRef) -> Result<Self> {
        if frame.placement().device != context.runtime().device() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.gpu_result_handle",
                "native GPU result frame device does not match the execution context",
            )
            .with_fact("frame_device", frame.placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        let batch = GpuRecordBatch::new(frame.into_frame(), schema).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.gpu_result_handle",
                "failed to bind native GPU result to its Arrow schema contract",
                error,
            )
        })?;
        Ok(Self { batch, runtime: context.shared_runtime(), device_placement: context.device_placement() })
    }

    pub fn frame(&self) -> &DataFrame {
        self.batch.frame()
    }

    pub fn into_gpu_frame(self) -> GpuDataFrame {
        GpuDataFrame::new(self.batch.into_frame())
    }

    pub fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.batch.placement()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.batch.num_columns()
    }

    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        self.batch.to_host_record_batch().map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.host_arrow.materialize",
                "failed to materialize native cudf output to host Arrow",
                error,
            )
        })
    }
}

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
                "cudf_query_engine.gpu_filtered_join_build_handle",
                "native GPU filtered-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", filtered_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        let device_allocated_size_bytes =
            filtered_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.gpu_filtered_join_build_handle",
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

    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.filtered_join.build().placement()
    }

    pub fn build_keys(&self) -> &[String] {
        &self.build_keys
    }

    pub fn null_equality(&self) -> JoinNullEquality {
        self.null_equality
    }

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
                "cudf_query_engine.gpu_hash_join_build_handle",
                "native GPU hash-join build artifact device does not match the execution context",
            )
            .with_fact("build_device", hash_join.build().placement().device.ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        let device_allocated_size_bytes =
            hash_join.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.gpu_hash_join_build_handle",
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

    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.hash_join.build().placement()
    }

    pub fn build_keys(&self) -> &[String] {
        &self.build_keys
    }

    pub fn null_equality(&self) -> JoinNullEquality {
        self.null_equality
    }

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
                "cudf_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact runtime does not match the execution context",
            )
            .with_fact("reason", "runtime_mismatch")
            .with_fact("artifact_device", self.runtime.device().ordinal())
            .with_fact("runtime_device", context.runtime().device().ordinal()));
        }
        if self.device_placement != context.device_placement() {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact device placement does not match the execution context",
            )
            .with_fact("reason", "device_placement_mismatch")
            .with_fact("artifact_device_placement", format!("{:?}", self.device_placement))
            .with_fact("runtime_device_placement", format!("{:?}", context.device_placement())));
        }
        if self.build_keys != build_keys {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact key contract does not match the requested join",
            )
            .with_fact("reason", "build_keys_mismatch")
            .with_fact("artifact_build_keys", self.build_keys.join(","))
            .with_fact("requested_build_keys", build_keys.join(",")));
        }
        if self.build_columns != build_columns {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.gpu_hash_join_build_handle.reuse",
                "native GPU hash-join build artifact schema contract does not match the requested join",
            )
            .with_fact("reason", "build_schema_mismatch")
            .with_fact("artifact_build_schema", format_gpu_hash_join_build_columns(&self.build_columns))
            .with_fact("requested_build_schema", format_gpu_hash_join_build_columns(build_columns)));
        }
        if self.null_equality != null_equality {
            return Err(QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "cudf_query_engine.gpu_hash_join_build_handle.reuse",
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
            "cudf_query_engine.gpu_hash_join_build_contract",
            "failed to observe native cudf hash-join build schema contract",
            error,
        )
    })?;
    let columns = info.columns.ok_or_else(|| {
        QueryError::new(
            crate::error::QueryErrorCode::InvalidPlan,
            "cudf_query_engine.gpu_hash_join_build_contract",
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

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
                "cudf_query_engine.gpu_hash_join_build_cache",
                "native GPU hash-join build artifact cache lock was poisoned",
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuLocalParquetSourceCacheKey {
    fingerprint: String,
}

impl GpuLocalParquetSourceCacheKey {
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
            push_token(&mut fingerprint, format!("file.{index}.path"), file.path.display().to_string());
            push_token(&mut fingerprint, format!("file.{index}.stable_id"), &file.stable_id);
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
            let partition_values = file.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
            push_string_vec(&mut fingerprint, format!("file.{index}.partition_values"), &partition_values);
        }
        Self { fingerprint }
    }

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
        push_token(&mut fingerprint, "chunk.path", chunk.path.display().to_string());
        push_token(&mut fingerprint, "chunk.stable_id", &chunk.stable_id);
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
        let partition_values = chunk.partition_values.iter().map(ToString::to_string).collect::<Vec<_>>();
        push_string_vec(&mut fingerprint, "chunk.partition_values", &partition_values);
        Self { fingerprint }
    }

    pub fn as_str(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLocalParquetSourceCacheInsert {
    Inserted,
    AlreadyPresent,
    CapacitySkipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuLocalParquetSourceCacheInsertResult {
    pub(crate) outcome: GpuLocalParquetSourceCacheInsert,
    pub(crate) evicted_device_bytes: u64,
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
    pub fn new(max_retained_device_bytes: u64) -> Self {
        Self { max_retained_device_bytes, state: Mutex::new(GpuLocalParquetSourceCacheState::default()) }
    }

    pub fn max_retained_device_bytes(&self) -> u64 {
        self.max_retained_device_bytes
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.cache.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

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
                "cudf_query_engine.gpu_local_parquet_source_cache",
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
                "cudf_query_engine.gpu_local_parquet_source_cache.reuse",
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
    push_string_vec(output, "read_columns", &source.read_columns);
    push_usize_vec(output, "read_column_file_indices", &source.read_column_file_indices);
    push_string_vec(output, "projected_output_columns", &source.projected_output_columns);
    push_source_refs(output, "projected_output_column_source_refs", &source.projected_output_column_source_refs);
    push_string_vec(output, "pure_filter_columns", &source.pure_filter_columns);
    push_source_refs(output, "pure_filter_column_source_refs", &source.pure_filter_column_source_refs);
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

fn push_token(output: &mut String, key: impl AsRef<str>, value: impl ToString) {
    let _ = write!(output, "{}={};", escape_token(key.as_ref()), escape_token(&value.to_string()));
}

fn escape_token(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '=' => escaped.push_str("\\="),
            ';' => escaped.push_str("\\;"),
            other => escaped.push(other),
        }
    }
    escaped
}

pub enum QueryOutput {
    GpuResident(GpuResultHandle),
    HostArrow(RecordBatch),
}

impl QueryOutput {
    pub fn mode(&self) -> ExecutionMode {
        match self {
            Self::GpuResident(_) => ExecutionMode::GpuResident,
            Self::HostArrow(_) => ExecutionMode::MaterializeToHost,
        }
    }

    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    pub fn as_gpu_resident(&self) -> Option<&GpuResultHandle> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

impl QueryExecutionResult {
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}
