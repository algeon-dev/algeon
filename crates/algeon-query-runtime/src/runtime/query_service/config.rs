use std::time::{Duration, Instant};

use query_planning::{
    capability::AggregateStrategyExecutionOptions,
    chunk::SourceChunkPlanningOptions,
    error::{QueryError, QueryErrorCode, Result},
    execution_contracts::{MemoryControlPolicy, SourceChunkPlanningMode, SourceSchedulerPolicy},
    mode::StatisticalAggregateExecutionMode,
    policy::PlanningPolicy,
};

use crate::runtime::{LakehouseCacheFillConfig, SourceStorageProfile};

/// Bounded admission-queue policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueBounds {
    /// Maximum queued waiters. Must be nonzero.
    pub max_waiters: usize,
    /// Maximum time one waiter may remain queued. Must be nonzero.
    pub max_wait: Duration,
}

impl QueueBounds {
    /// Canonical bounded-admission queue defaults shared by every frontend.
    ///
    /// Frontends may project these values into their own configuration
    /// surfaces, but must not restate them. Keeping the construction here
    /// makes the runtime controller the sole owner of queue defaults.
    pub(crate) const fn canonical_default() -> Self {
        Self { max_waiters: 64, max_wait: Duration::from_secs(30) }
    }

    /// The latest instant a wait that starts at `start` may last: `max_wait`
    /// after `start`, or the caller's `requested` deadline when that is
    /// earlier. A `max_wait` too long to add to `start` sets no limit of its
    /// own.
    pub(crate) fn wait_deadline(&self, start: Instant, requested: Option<Instant>) -> Option<Instant> {
        let configured = start.checked_add(self.max_wait);
        match (requested, configured) {
            (Some(requested), Some(configured)) => Some(requested.min(configured)),
            (requested, configured) => requested.or(configured),
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        validate_nonzero_config_field("queue_bounds.max_waiters", self.max_waiters == 0)?;
        validate_nonzero_config_field("queue_bounds.max_wait", self.max_wait.is_zero())
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
    /// Bounded source scheduling policy copied into admitted contexts.
    pub source_scheduler_policy: SourceSchedulerPolicy,
    /// Source chunk planning mode copied into admitted contexts.
    pub source_chunk_planning_mode: SourceChunkPlanningMode,
    /// Floating statistical aggregate execution contract copied into admitted
    /// contexts.
    pub statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode,
    /// Join and dynamic-filter thresholds copied into admitted contexts; a
    /// plan decided without an admission artifact is admitted under it.
    pub planning_policy: PlanningPolicy,
    /// Grouped aggregate strategy execution policy copied into admitted contexts.
    pub aggregate_strategy_execution_options: AggregateStrategyExecutionOptions,
    /// Expert local-throughput source chunk planning policy copied into admitted contexts.
    pub source_chunk_planning_options: SourceChunkPlanningOptions,
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
    pub(super) fn validate(&self) -> Result<()> {
        self.lakehouse_cache_fill.validate()?;
        self.source_scheduler_policy.validate()?;
        self.planning_policy.validate()?;
        self.aggregate_strategy_execution_options.validate()?;
        self.source_chunk_planning_options.validate()?;
        validate_nonzero_config_field("max_source_chunk_bytes", self.max_source_chunk_bytes == 0)?;
        validate_nonzero_config_field("max_row_groups_per_chunk", self.max_row_groups_per_chunk == 0)?;
        validate_nonzero_config_field("max_retry_attempts", self.max_retry_attempts == 0)?;
        validate_nonzero_config_field("min_retry_chunk_row_groups", self.min_retry_chunk_row_groups == 0)?;
        Ok(())
    }

    pub(in crate::runtime) fn memory_policy_for_admitted_query(&self, device_budget_bytes: u64) -> MemoryControlPolicy {
        MemoryControlPolicy {
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
            source_scheduler_policy: SourceSchedulerPolicy::default(),
            source_chunk_planning_mode: SourceChunkPlanningMode::default(),
            statistical_aggregate_execution_mode: StatisticalAggregateExecutionMode::default(),
            planning_policy: PlanningPolicy::default(),
            aggregate_strategy_execution_options: AggregateStrategyExecutionOptions::default(),
            source_chunk_planning_options: SourceChunkPlanningOptions::default(),
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
            "query_engine.query_service.config.validate",
            "query service config field must be greater than zero",
        )
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(())
}
