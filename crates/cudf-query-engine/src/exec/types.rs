use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, DataType, ExecutionPlacement,
    frame::dataframe::{GpuRecordBatch, InfoDetail},
};

use crate::{
    error::{QueryError, Result},
    metrics::QueryMetrics,
    plan::{HashJoinBuildArtifactId, JoinNullEquality, ResultSink},
    runtime::{DevicePlacement, ExecutionContext},
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
