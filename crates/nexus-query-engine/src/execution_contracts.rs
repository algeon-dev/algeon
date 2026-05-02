//! Data-only execution policy and reservation contracts.
//!
//! This leaf is shared by error, metrics, runtime, and observability-facing
//! code.  It deliberately owns no runtime resources, which keeps metric
//! projection independent of the runtime implementation.

use crate::{
    correlation::{CompiledTaskId, PlanNodeId, QueryId},
    error::{QueryError, QueryErrorCode, Result},
    observability::{RuntimeAttemptId, RuntimeTaskId, TaskAttemptId},
    util::comma_separated_usize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DevicePlacement {
    SingleDevice { device_ordinal: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum MemorySpaceId {
    Device { device_ordinal: usize },
}

impl std::fmt::Display for MemorySpaceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Device { device_ordinal } => write!(f, "GPU:{device_ordinal}"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryResourcePolicy {
    #[default]
    DefaultCuda,
    Pool {
        initial_pool_size: usize,
        maximum_pool_size: Option<usize>,
    },
    ExternalRuntime,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryControlPolicy {
    #[default]
    Unbounded,
    Bounded {
        device_budget_bytes: u64,
        max_source_chunk_bytes: u64,
        max_row_groups_per_chunk: usize,
        max_retry_attempts: usize,
        min_retry_chunk_row_groups: usize,
    },
}

impl MemoryControlPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unbounded => "unbounded",
            Self::Bounded { .. } => "bounded",
        }
    }

    pub const fn is_bounded(self) -> bool {
        matches!(self, Self::Bounded { .. })
    }

    pub fn validate(self) -> Result<()> {
        let Self::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            min_retry_chunk_row_groups,
        } = self
        else {
            return Ok(());
        };
        validate_nonzero_policy_field("device_budget_bytes", device_budget_bytes == 0)?;
        validate_nonzero_policy_field("max_source_chunk_bytes", max_source_chunk_bytes == 0)?;
        validate_nonzero_policy_field("max_row_groups_per_chunk", max_row_groups_per_chunk == 0)?;
        validate_nonzero_policy_field("max_retry_attempts", max_retry_attempts == 0)?;
        validate_nonzero_policy_field("min_retry_chunk_row_groups", min_retry_chunk_row_groups == 0)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceChunkPlanningMode {
    #[default]
    Conservative,
    LocalParquetThroughput,
}

impl SourceChunkPlanningMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conservative => "conservative",
            Self::LocalParquetThroughput => "local_parquet_throughput",
        }
    }
}

impl std::fmt::Display for SourceChunkPlanningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Display for MemoryControlPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationKind {
    SourceRead,
    ChunkOperator,
    OperatorOutput,
    BuildSide,
    ProbeSide,
    HashTable,
    FilteredProbeState,
    MixedSizeData,
    GatherMap,
    JoinOutput,
    GraphAlgorithm,
    PartialState,
    FinalMerge,
    FinalSort,
    Materialization,
    CapabilityContract,
}

impl ReservationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SourceRead => "source_read",
            Self::ChunkOperator => "chunk_operator",
            Self::OperatorOutput => "operator_output",
            Self::BuildSide => "build_side",
            Self::ProbeSide => "probe_side",
            Self::HashTable => "hash_table",
            Self::FilteredProbeState => "filtered_probe_state",
            Self::MixedSizeData => "mixed_size_data",
            Self::GatherMap => "gather_map",
            Self::JoinOutput => "join_output",
            Self::GraphAlgorithm => "graph_algorithm",
            Self::PartialState => "partial_state",
            Self::FinalMerge => "final_merge",
            Self::FinalSort => "final_sort",
            Self::Materialization => "materialization",
            Self::CapabilityContract => "capability_contract",
        }
    }
}

impl std::fmt::Display for ReservationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionBoundaryPolicy {
    source_read: SourceReadBoundary,
    build_materialization: BuildMaterializationBoundary,
    probe_execution: ProbeExecutionBoundary,
    result_materialization: ResultMaterializationBoundary,
}

impl Default for ExecutionBoundaryPolicy {
    fn default() -> Self {
        Self {
            source_read: SourceReadBoundary::DirectDeviceLocalParquet,
            build_materialization: BuildMaterializationBoundary::QueryScopedGpuHashJoin,
            probe_execution: ProbeExecutionBoundary::ReusableGpuHashJoin,
            result_materialization: ResultMaterializationBoundary::ExplicitSink,
        }
    }
}

impl ExecutionBoundaryPolicy {
    pub const fn new(
        source_read: SourceReadBoundary,
        build_materialization: BuildMaterializationBoundary,
        probe_execution: ProbeExecutionBoundary,
        result_materialization: ResultMaterializationBoundary,
    ) -> Self {
        Self { source_read, build_materialization, probe_execution, result_materialization }
    }

    pub const fn source_read(self) -> SourceReadBoundary {
        self.source_read
    }

    pub const fn build_materialization(self) -> BuildMaterializationBoundary {
        self.build_materialization
    }

    pub const fn probe_execution(self) -> ProbeExecutionBoundary {
        self.probe_execution
    }

    pub const fn result_materialization(self) -> ResultMaterializationBoundary {
        self.result_materialization
    }

    pub const fn with_build_materialization(mut self, build_materialization: BuildMaterializationBoundary) -> Self {
        self.build_materialization = build_materialization;
        self
    }
}

impl std::fmt::Display for ExecutionBoundaryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "source_read={},build_materialization={},probe_execution={},result_materialization={}",
            self.source_read, self.build_materialization, self.probe_execution, self.result_materialization
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReadBoundary {
    DirectDeviceLocalParquet,
}

impl SourceReadBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectDeviceLocalParquet => "direct_device_local_parquet",
        }
    }
}

impl std::fmt::Display for SourceReadBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuildMaterializationBoundary {
    QueryScopedGpuHashJoin,
}

impl BuildMaterializationBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::QueryScopedGpuHashJoin => "query_scoped_gpu_hash_join",
        }
    }
}

impl std::fmt::Display for BuildMaterializationBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeExecutionBoundary {
    ReusableGpuHashJoin,
}

impl ProbeExecutionBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReusableGpuHashJoin => "reusable_gpu_hash_join",
        }
    }
}

impl std::fmt::Display for ProbeExecutionBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResultMaterializationBoundary {
    ExplicitSink,
}

impl ResultMaterializationBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitSink => "explicit_sink",
        }
    }
}

impl std::fmt::Display for ResultMaterializationBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reservation request submitted to bounded execution admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationRequest {
    pub query_id: Option<QueryId>,
    pub reservation_kind: ReservationKind,
    pub memory_space: MemorySpaceId,
    pub memory_boundary: String,
    pub requested_bytes: u64,
    pub budget_bytes: u64,
    pub node_id: Option<PlanNodeId>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub path: Option<String>,
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
    pub compiled_task_id: Option<CompiledTaskId>,
    pub task_attempt_id: Option<TaskAttemptId>,
    pub task_kind: Option<String>,
    pub resident_resource_id: Option<String>,
    pub source_read_contract_quality: Option<String>,
    pub source_read_planning_bytes: Option<u64>,
    pub source_read_estimated_bytes: Option<u64>,
}

impl MemoryReservationRequest {
    pub fn new(reservation_kind: ReservationKind, requested_bytes: u64, budget_bytes: u64) -> Self {
        Self {
            query_id: None,
            reservation_kind,
            memory_space: MemorySpaceId::Device { device_ordinal: 0 },
            memory_boundary: reservation_kind.as_str().to_string(),
            requested_bytes,
            budget_bytes,
            node_id: None,
            stage: None,
            chunk_index: None,
            file_index: None,
            path: None,
            row_groups: None,
            attempt: None,
            max_attempts: None,
            compiled_task_id: None,
            task_attempt_id: None,
            task_kind: None,
            resident_resource_id: None,
            source_read_contract_quality: None,
            source_read_planning_bytes: None,
            source_read_estimated_bytes: None,
        }
    }

    #[must_use]
    pub fn with_memory_space(mut self, memory_space: MemorySpaceId) -> Self {
        self.memory_space = memory_space;
        self
    }

    #[must_use]
    pub fn with_memory_boundary(mut self, memory_boundary: impl Into<String>) -> Self {
        self.memory_boundary = memory_boundary.into();
        self
    }

    #[must_use]
    pub fn with_requested_bytes(mut self, requested_bytes: u64) -> Self {
        self.requested_bytes = requested_bytes;
        self
    }

    #[must_use]
    pub fn with_node_id(mut self, node_id: impl ToString) -> Self {
        self.node_id = Some(PlanNodeId::new(node_id.to_string()));
        self
    }

    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.query_id = Some(query_id.into());
        self
    }

    #[must_use]
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    #[must_use]
    pub fn with_chunk_index(mut self, chunk_index: usize) -> Self {
        self.chunk_index = Some(chunk_index);
        self
    }

    #[must_use]
    pub fn with_file_index(mut self, file_index: usize) -> Self {
        self.file_index = Some(file_index);
        self
    }

    #[must_use]
    pub fn with_path(mut self, path: impl ToString) -> Self {
        self.path = Some(path.to_string());
        self
    }

    #[must_use]
    pub fn with_row_groups(mut self, row_groups: impl Into<Vec<usize>>) -> Self {
        self.row_groups = Some(row_groups.into());
        self
    }

    #[must_use]
    pub fn with_attempt(mut self, attempt: usize, max_attempts: usize) -> Self {
        self.attempt = Some(attempt);
        self.max_attempts = Some(max_attempts);
        self
    }

    #[must_use]
    pub fn with_compiled_task_id(mut self, compiled_task_id: impl ToString) -> Self {
        self.compiled_task_id = Some(CompiledTaskId::new(compiled_task_id));
        self
    }

    #[must_use]
    pub fn with_task_registration(
        mut self,
        task_id: impl ToString,
        attempt: u32,
        task_kind: impl Into<String>,
    ) -> Self {
        let task_id = task_id.to_string();
        self.compiled_task_id = Some(CompiledTaskId::new(&task_id));
        self.task_attempt_id = Some(TaskAttemptId::new(RuntimeTaskId::new(task_id), RuntimeAttemptId::new(attempt)));
        self.task_kind = Some(task_kind.into());
        self
    }

    #[must_use]
    pub fn with_resident_resource_id(mut self, resident_resource_id: impl ToString) -> Self {
        self.resident_resource_id = Some(resident_resource_id.to_string());
        self
    }

    #[must_use]
    pub fn with_source_read_contract(
        mut self,
        quality: impl Into<String>,
        planning_bytes: Option<u64>,
        estimated_bytes: Option<u64>,
    ) -> Self {
        self.source_read_contract_quality = Some(quality.into());
        self.source_read_planning_bytes = planning_bytes;
        self.source_read_estimated_bytes = estimated_bytes;
        self
    }

    pub fn denied_error(&self, granted_bytes: u64) -> QueryError {
        let mut error = QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.memory.reserve",
            "memory reservation failed before nexus native execution",
        )
        .with_fact("memory_boundary", &self.memory_boundary)
        .with_fact("reservation_kind", self.reservation_kind)
        .with_fact("requested_bytes", self.requested_bytes)
        .with_fact("granted_bytes", granted_bytes)
        .with_fact("budget_bytes", self.budget_bytes);
        if let Some(query_id) = &self.query_id {
            error = error.with_fact("query_id", query_id);
        }
        if let Some(node_id) = &self.node_id {
            error = error.with_fact("node_id", node_id);
        }
        if let Some(stage) = &self.stage {
            error = error.with_fact("stage", stage);
        }
        if let Some(chunk_index) = self.chunk_index {
            error = error.with_fact("chunk_index", chunk_index);
        }
        if let Some(file_index) = self.file_index {
            error = error.with_fact("file_index", file_index);
        }
        if let Some(path) = &self.path {
            error = error.with_fact("path", path);
        }
        if let Some(row_groups) = &self.row_groups {
            error = error
                .with_fact("row_group_indexes", comma_separated_usize(row_groups))
                .with_fact("row_group_count", row_groups.len());
        }
        if let Some(attempt) = self.attempt {
            error = error.with_fact("attempt", attempt);
        }
        if let Some(max_attempts) = self.max_attempts {
            error = error.with_fact("max_attempts", max_attempts);
        }
        if let Some(compiled_task_id) = &self.compiled_task_id {
            error = error.with_fact("compiled_task_id", compiled_task_id);
        }
        if let Some(resident_resource_id) = &self.resident_resource_id {
            error = error.with_fact("resident_resource_id", resident_resource_id);
        }
        if let Some(quality) = &self.source_read_contract_quality {
            error = error.with_fact("source_read_contract_quality", quality);
        }
        if let Some(planning_bytes) = self.source_read_planning_bytes {
            error = error.with_fact("source_read_planning_bytes", planning_bytes);
        }
        if let Some(estimated_bytes) = self.source_read_estimated_bytes {
            error = error.with_fact("source_read_estimated_bytes", estimated_bytes);
        }
        error
    }
}

fn validate_nonzero_policy_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.memory_policy.validate",
            "bounded memory-control policy fields must be greater than zero",
        )
        .with_fact("policy", "bounded")
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}
