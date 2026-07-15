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
    DataFrame, DataFrameFilteredJoin, DataFrameHashJoin, DataType, ExecutionPlacement,
    frame::dataframe::{ColumnMemoryIdentity, GpuRecordBatch, InfoDetail},
};

use crate::{
    error::{QueryError, Result},
    naming::is_datafusion_order_column_name,
    plan::{GraphResidencyDescriptor, JoinNullEquality},
    runtime::{
        DevicePlacement, ExecutionContext, cache::local_parquet::GpuLocalParquetSourceCachePin,
        gpu::governor::MemoryGrantLiveReservation,
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

    pub(crate) fn source_cache_pin(&self) -> Option<GpuLocalParquetSourceCachePin> {
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

    pub(crate) fn new_with_live_reservation(
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

    pub(in crate::runtime) fn with_source_cache_pin(mut self, pin: GpuLocalParquetSourceCachePin) -> Self {
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

fn source_column_memory_identities(frame: &DataFrame) -> HashSet<ColumnMemoryIdentity> {
    frame.memory_identity().columns.into_iter().map(|column| column.identity).collect()
}

fn format_gpu_hash_join_build_columns(columns: &[GpuHashJoinBuildColumn]) -> String {
    columns
        .iter()
        .map(|column| format!("{}:{:?}:nullable={}", column.name, column.data_type, column.nullable))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use cudf::{DataFrame, ops::expr::col};

    use super::*;
    use crate::runtime::gpu::governor::{GrantRequest, MemoryGovernor, MemoryGrant};

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
