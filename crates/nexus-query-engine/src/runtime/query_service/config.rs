use super::*;
use crate::graph_memory::GraphMemoryCalibrationOverride;

/// Bounded admission-queue policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueBounds {
    /// Maximum queued waiters. Must be nonzero.
    pub max_waiters: usize,
    /// Maximum time one waiter may remain queued. Must be nonzero.
    pub max_wait: Duration,
    /// Number of compatible-device bypasses before a waiter protects a slot.
    /// Must be nonzero to guarantee finite overtaking.
    pub max_overtakes_per_waiter: u32,
}

impl QueueBounds {
    /// Canonical bounded-admission queue defaults shared by every frontend.
    ///
    /// Frontends may project these values into their own configuration
    /// surfaces, but must not restate them. Keeping the construction here
    /// makes the runtime controller the sole owner of queue defaults.
    pub const fn canonical_default() -> Self {
        Self { max_waiters: 64, max_wait: Duration::from_secs(30), max_overtakes_per_waiter: 4 }
    }

    pub(super) fn validate(&self) -> Result<()> {
        validate_nonzero_config_field("queue_bounds.max_waiters", self.max_waiters == 0)?;
        validate_nonzero_config_field("queue_bounds.max_wait", self.max_wait.is_zero())?;
        validate_nonzero_config_field("queue_bounds.max_overtakes_per_waiter", self.max_overtakes_per_waiter == 0)
    }
}

impl Default for QueueBounds {
    fn default() -> Self {
        Self::canonical_default()
    }
}

/// Execution policy copied into admitted native query contexts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryExecutionConfig {
    /// Lakehouse cache-fill policy and shared resource limits.
    pub lakehouse_cache_fill: LakehouseCacheFillConfig,
    /// Source object-store profiles available to service-owned execution contexts.
    pub source_storage_profiles: Vec<SourceStorageProfile>,
    /// Source/build/probe/result boundary policy copied into admitted contexts.
    pub boundary_policy: ExecutionBoundaryPolicy,
    /// Bounded source scheduling policy copied into admitted contexts.
    pub source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode copied into admitted contexts.
    pub source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Floating statistical aggregate execution contract copied into admitted
    /// contexts. The field keeps its legacy STDDEV_SAMP name for source
    /// compatibility.
    pub stddev_samp_execution_mode: StddevSampExecutionMode,
    /// Dynamic-filter execution policy assigned to native plans using service handles.
    pub dynamic_filter_execution_options: DynamicFilterExecutionOptions,
    /// Keyed-hash join strategy policy assigned to native plans using service handles.
    pub keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions,
    /// Grouped aggregate strategy execution policy copied into admitted contexts.
    pub aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert local-throughput source chunk planning policy copied into admitted contexts.
    pub source_chunk_planning_options: SourceChunkPlanningOptions,
    /// Optional service-scoped graph calibration metadata copied into admitted contexts.
    pub graph_memory_calibration_override: GraphMemoryCalibrationOverride,
    /// Source chunk byte target copied into admitted bounded contexts.
    pub max_source_chunk_bytes: u64,
    /// Source row-group target copied into admitted bounded contexts.
    pub max_row_groups_per_chunk: usize,
    /// Bounded retry-attempt cap copied into admitted bounded contexts.
    pub max_retry_attempts: usize,
    /// Bounded retry row-group floor copied into admitted bounded contexts.
    pub min_retry_chunk_row_groups: usize,
}

impl QueryExecutionConfig {
    pub(super) fn validate(&self, deployment: &DeploymentIoCapability) -> Result<()> {
        let _ = deployment;
        self.lakehouse_cache_fill.validate()?;
        self.source_scheduler_policy.validate()?;
        self.dynamic_filter_execution_options.validate()?;
        self.keyed_hash_join_strategy_options.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
        validate_nonzero_config_field("max_source_chunk_bytes", self.max_source_chunk_bytes == 0)?;
        validate_nonzero_config_field("max_row_groups_per_chunk", self.max_row_groups_per_chunk == 0)?;
        validate_nonzero_config_field("max_retry_attempts", self.max_retry_attempts == 0)?;
        validate_nonzero_config_field("min_retry_chunk_row_groups", self.min_retry_chunk_row_groups == 0)?;
        Ok(())
    }

    pub(super) fn memory_policy_for_admitted_query(&self, device_budget_bytes: u64) -> MemoryControlPolicy {
        MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes: self.max_source_chunk_bytes,
            max_row_groups_per_chunk: self.max_row_groups_per_chunk,
            max_retry_attempts: self.max_retry_attempts,
            min_retry_chunk_row_groups: self.min_retry_chunk_row_groups,
        }
    }
}

impl Default for QueryExecutionConfig {
    fn default() -> Self {
        Self {
            lakehouse_cache_fill: LakehouseCacheFillConfig::default(),
            source_storage_profiles: Vec::new(),
            boundary_policy: ExecutionBoundaryPolicy::default(),
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            stddev_samp_execution_mode: StddevSampExecutionMode::default(),
            dynamic_filter_execution_options: DynamicFilterExecutionOptions::default(),
            keyed_hash_join_strategy_options: KeyedHashJoinStrategyOptions::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
            graph_memory_calibration_override: GraphMemoryCalibrationOverride::default(),
            max_source_chunk_bytes: 64 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    }
}

fn validate_nonzero_config_field(field: &'static str, is_zero: bool) -> Result<()> {
    if is_zero {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.config.validate",
            "query service config field must be greater than zero",
        )
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}

pub(super) fn minimum_stream_pool_size(
    max_active_attempts_per_device: usize,
    max_source_reads_in_flight: usize,
) -> usize {
    let streams_per_query = 1usize.saturating_add(max_source_reads_in_flight.max(1));
    max_active_attempts_per_device.saturating_mul(streams_per_query).max(max_active_attempts_per_device)
}

pub(super) fn default_stream_pool_size_for_ordinals(device_ordinals: &[usize]) -> Result<usize> {
    let ordinals = if device_ordinals.is_empty() {
        let device_count = super::NexusGpuInventory.device_count()?;
        (0..device_count).collect::<Vec<_>>()
    } else {
        device_ordinals.to_vec()
    };
    let mut default_size = 0;
    for ordinal in ordinals {
        let device = cudf::DeviceId::new(ordinal).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.query_service.device",
                "failed to resolve query service CUDA device",
                error,
            )
            .with_fact("device_ordinal", ordinal)
        })?;
        default_size = default_size.max(super::default_service_stream_pool_size_for_device(device));
    }
    Ok(default_size)
}
