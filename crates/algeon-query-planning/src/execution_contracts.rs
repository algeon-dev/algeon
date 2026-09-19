//! Data-only execution policy and reservation contracts.
//!
//! This leaf is shared by error, metrics, runtime, and observability-facing
//! code.  It deliberately owns no runtime resources, which keeps metric
//! projection independent of the runtime implementation.

use serde::{Deserialize, Serialize};

use crate::error::{QueryError, QueryErrorCode, Result};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeTaskId(String);

impl RuntimeTaskId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for RuntimeTaskId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for RuntimeTaskId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl std::fmt::Display for RuntimeTaskId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RuntimeAttemptId(u32);

impl RuntimeAttemptId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TaskAttemptId {
    pub task_id: RuntimeTaskId,
    pub attempt: RuntimeAttemptId,
}

impl TaskAttemptId {
    pub const fn new(task_id: RuntimeTaskId, attempt: RuntimeAttemptId) -> Self {
        Self { task_id, attempt }
    }
}

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

/// Native source scheduling controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSchedulerPolicy {
    max_source_reads_in_flight: usize,
}

impl SourceSchedulerPolicy {
    pub const fn serialized() -> Self {
        Self { max_source_reads_in_flight: 1 }
    }

    pub const fn new(max_source_reads_in_flight: usize) -> Self {
        Self { max_source_reads_in_flight }
    }

    pub const fn max_source_reads_in_flight(self) -> usize {
        self.max_source_reads_in_flight
    }

    pub const fn enables_prefetch(self) -> bool {
        self.max_source_reads_in_flight > 1
    }

    pub fn validate(self) -> Result<()> {
        if self.max_source_reads_in_flight == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.source_scheduler_policy.validate",
                "source scheduler max_source_reads_in_flight must be greater than zero",
            )
            .with_fact("field", "max_source_reads_in_flight")
            .with_fact("reason", "zero"));
        }
        Ok(())
    }
}

impl Default for SourceSchedulerPolicy {
    fn default() -> Self {
        Self { max_source_reads_in_flight: 2 }
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

fn validate_nonzero_policy_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.memory_policy.validate",
            "bounded memory-control policy fields must be greater than zero",
        )
        .with_fact("policy", "bounded")
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}
