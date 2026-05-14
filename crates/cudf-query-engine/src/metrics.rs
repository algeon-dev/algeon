use std::{collections::BTreeMap, time::Duration};

use cudf::ExecutionBudgetReport;

use crate::runtime::{
    ExecutionBoundaryPolicy, MemoryControlPolicy, MemoryReservationRequest, MemorySpaceId, ReservationKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReservationEventOutcome {
    Granted,
    Denied,
}

impl MemoryReservationEventOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Granted => "granted",
            Self::Denied => "denied",
        }
    }
}

impl std::fmt::Display for MemoryReservationEventOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationEvent {
    pub reservation_kind: ReservationKind,
    pub memory_space: MemorySpaceId,
    pub memory_boundary: String,
    pub requested_bytes: u64,
    pub granted_bytes: u64,
    pub outcome: MemoryReservationEventOutcome,
    pub node_id: Option<String>,
    pub stage: Option<String>,
}

impl MemoryReservationEvent {
    pub fn granted(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::new(request, granted_bytes, MemoryReservationEventOutcome::Granted)
    }

    pub fn denied(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::new(request, granted_bytes, MemoryReservationEventOutcome::Denied)
    }

    fn new(request: &MemoryReservationRequest, granted_bytes: u64, outcome: MemoryReservationEventOutcome) -> Self {
        Self {
            reservation_kind: request.reservation_kind,
            memory_space: request.memory_space,
            memory_boundary: request.memory_boundary.clone(),
            requested_bytes: request.requested_bytes,
            granted_bytes,
            outcome,
            node_id: request.node_id.clone(),
            stage: request.stage.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    pub boundary: BoundaryMetrics,
    pub operators: Vec<OperatorMetrics>,
    pub join_strategy_decisions: Vec<JoinStrategyDecisionMetrics>,
    pub phase_c_memory_control_policy: MemoryControlPolicy,
    pub execution_boundary_policy: ExecutionBoundaryPolicy,
    pub input_batches: u64,
    pub input_rows: u64,
    /// Backward-compatible alias for `resident_dataframe_peak_bytes`.
    ///
    /// This is not an allocator high-water mark.
    pub peak_device_memory_bytes: Option<u64>,
    pub resident_dataframe_peak_bytes: Option<u64>,
    pub allocator_high_water_bytes: Option<u64>,
    pub reservation_requested_bytes: u64,
    pub reservation_granted_bytes: u64,
    pub reservation_denied_count: u64,
    pub reservation_events: Vec<MemoryReservationEvent>,
    /// Successful lower-level cuDF calls that returned an execution-budget report.
    ///
    /// Token-routed operations without a report-returning cuDF API may still attach budget failure
    /// facts on errors, but they do not synthesize success events here.
    pub budgeted_execution_events: Vec<BudgetedExecutionEvent>,
    pub source_chunks_planned: u64,
    pub source_chunks_attempted: u64,
    pub source_chunks_committed: u64,
    pub source_chunks_retried: u64,
    pub source_read_prefetches_started: u64,
    pub source_read_prefetches_completed: u64,
    pub source_read_prefetches_skipped: u64,
    pub source_read_prefetches_abandoned: u64,
    pub source_cache_hits: u64,
    pub source_cache_misses: u64,
    pub source_cache_inserts: u64,
    pub source_cache_insert_skipped: u64,
    pub source_cache_retained_bytes: u64,
    pub row_groups_planned: u64,
    pub row_groups_committed: u64,
    pub partial_aggregate_count: u64,
    pub partial_aggregate_rows: u64,
    pub final_merge_attempts: u64,
    pub final_merge_levels: u64,
    pub retry_count: u64,
    pub retry_resource_exhausted_count: u64,
    pub retry_reservation_failure_count: u64,
    /// Number of reusable cuDF hash-join artifacts built during this query.
    pub hash_join_build_count: u64,
    /// Number of reusable cuDF filtered-join artifacts built during this query.
    pub filtered_join_build_count: u64,
    /// Number of one-shot keyed equality joins executed during this query.
    pub one_shot_join_count: u64,
    /// Query-scoped reusable hash-join cache hits.
    pub hash_join_cache_hits: u64,
    /// Query-scoped reusable hash-join cache misses.
    pub hash_join_cache_misses: u64,
    /// Durable reusable hash-join build-artifact cache hits.
    pub hash_join_durable_cache_hits: u64,
    /// Durable reusable hash-join build-artifact cache misses.
    pub hash_join_durable_cache_misses: u64,
    /// Durable reusable hash-join build artifacts inserted by this query.
    pub hash_join_durable_cache_inserts: u64,
    /// Unique device bytes retained by reusable cuDF hash-join build artifacts.
    ///
    /// This is recorded when an artifact is constructed, not when it is reused.
    pub hash_join_hash_table_bytes: u64,
    /// Unique device bytes retained by reusable cuDF filtered-join build artifacts.
    ///
    /// This is recorded when an artifact is constructed, not when it is reused.
    pub filtered_join_hash_table_bytes: u64,
    /// Unique device bytes retained by filtered-join probe states.
    pub filtered_join_probe_state_bytes: u64,
    /// Unique device bytes retained by mixed inner/left join size-data states.
    pub mixed_join_size_data_bytes: u64,
    /// Device bytes retained by one-shot join index maps while materializing output.
    pub one_shot_join_index_bytes: u64,
    /// Reservation bytes requested for one-shot join output materialization.
    pub one_shot_join_materialization_bytes: u64,
    /// Observed device bytes of one-shot join output frames.
    pub one_shot_join_output_bytes: u64,
    pub parquet_files: u64,
    pub parquet_bytes_known: u64,
    pub row_groups_selected: u64,
    pub read_columns: u64,
    pub source_metadata_load_time: Duration,
    pub source_read_queue_wait_time: Duration,
    pub source_cache_lookup_time: Duration,
    pub parquet_read_time: Duration,
    pub source_read_time: Duration,
    pub operator_time: Duration,
    pub chunk_operator_time: Duration,
    pub final_merge_time: Duration,
    pub final_sort_time: Duration,
    pub materialization_time: Duration,
    /// Bounded queries whose capability-admitted graph was accepted by the
    /// compiled-pipeline executor before execution started.
    pub compiled_pipeline_attempts: u64,
    /// Compiled-pipeline tasks completed by the scheduler.
    pub compiled_pipeline_tasks_executed: u64,
    /// Host Arrow sink boundaries materialized by the compiled-pipeline executor.
    pub compiled_pipeline_host_arrow_sinks: u64,
    /// Adaptive executor retry actions taken by this query.
    pub adaptive_retry_count: u64,
    /// Adaptive executor spill actions taken by this query.
    pub adaptive_spill_count: u64,
    /// Adaptive executor downgrade actions taken by this query.
    pub adaptive_downgrade_count: u64,
    /// Adaptive memory-history lookups performed during admission/execution.
    pub adaptive_memory_history_lookup_count: u64,
    /// Adaptive memory-history lookups with a prior observation.
    pub adaptive_memory_history_hit_count: u64,
    /// Adaptive memory-history lookups without a prior observation.
    pub adaptive_memory_history_miss_count: u64,
}

impl QueryMetrics {
    pub fn record_operator_metrics(&mut self, metrics: OperatorMetrics) {
        self.operators.push(metrics);
    }

    pub fn record_join_strategy_decision(&mut self, node_id: impl ToString, facts: &BTreeMap<&'static str, String>) {
        self.join_strategy_decisions.push(JoinStrategyDecisionMetrics {
            node_id: node_id.to_string(),
            facts: facts.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect(),
        });
    }

    pub fn record_input_batch(&mut self, rows: u64) {
        self.input_batches += 1;
        self.input_rows += rows;
    }

    pub fn record_parquet_source(&mut self, known_bytes: Option<u64>, selected_row_groups: u64, read_columns: u64) {
        self.parquet_files += 1;
        self.parquet_bytes_known += known_bytes.unwrap_or(0);
        self.row_groups_selected += selected_row_groups;
        self.read_columns += read_columns;
    }

    pub fn record_device_memory_observation(&mut self, bytes: u64) {
        self.peak_device_memory_bytes = Some(self.peak_device_memory_bytes.map_or(bytes, |peak| peak.max(bytes)));
        self.resident_dataframe_peak_bytes =
            Some(self.resident_dataframe_peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }

    pub fn record_reservation_granted(&mut self, requested_bytes: u64, granted_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted_bytes);
    }

    pub fn record_reservation_denied(&mut self, requested_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(1);
    }

    pub fn record_reservation_granted_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_granted(request.requested_bytes, granted_bytes);
        self.reservation_events.push(MemoryReservationEvent::granted(request, granted_bytes));
    }

    pub fn record_reservation_denied_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_denied(request.requested_bytes);
        self.reservation_events.push(MemoryReservationEvent::denied(request, granted_bytes));
    }

    pub fn record_budgeted_execution(&mut self, operation: impl Into<String>, report: &ExecutionBudgetReport) {
        self.budgeted_execution_events.push(BudgetedExecutionEvent::new(operation, report));
    }

    pub fn record_source_chunk_planned(&mut self, row_groups: u64) {
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(1);
        self.row_groups_planned = self.row_groups_planned.saturating_add(row_groups);
    }

    pub fn record_source_chunk_attempt(&mut self) {
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(1);
    }

    pub fn record_source_chunk_commit(&mut self, row_groups: u64) {
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(1);
        self.row_groups_committed = self.row_groups_committed.saturating_add(row_groups);
    }

    pub fn record_source_chunk_retry(&mut self) {
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(1);
    }

    pub fn record_source_read_prefetch_started(&mut self) {
        self.source_read_prefetches_started = self.source_read_prefetches_started.saturating_add(1);
    }

    pub fn record_source_read_prefetch_completed(&mut self) {
        self.source_read_prefetches_completed = self.source_read_prefetches_completed.saturating_add(1);
    }

    pub fn record_source_read_prefetch_skipped(&mut self) {
        self.source_read_prefetches_skipped = self.source_read_prefetches_skipped.saturating_add(1);
    }

    pub fn record_source_read_prefetch_abandoned(&mut self) {
        self.source_read_prefetches_abandoned = self.source_read_prefetches_abandoned.saturating_add(1);
    }

    pub fn record_source_read_queue_wait(&mut self, elapsed: Duration) {
        self.source_read_queue_wait_time += elapsed;
    }

    pub fn record_source_cache_hit(&mut self, retained_bytes: u64) {
        self.source_cache_hits = self.source_cache_hits.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    pub fn record_source_cache_miss(&mut self, retained_bytes: u64) {
        self.source_cache_misses = self.source_cache_misses.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    pub fn record_source_cache_insert(&mut self, retained_bytes: u64) {
        self.source_cache_inserts = self.source_cache_inserts.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    pub fn record_source_cache_insert_skipped(&mut self, retained_bytes: u64) {
        self.source_cache_insert_skipped = self.source_cache_insert_skipped.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    pub fn record_source_cache_lookup_time(&mut self, elapsed: Duration) {
        self.source_cache_lookup_time += elapsed;
    }

    pub fn record_source_cache_retained_bytes(&mut self, retained_bytes: u64) {
        self.source_cache_retained_bytes = self.source_cache_retained_bytes.max(retained_bytes);
    }

    pub fn record_retry_resource_exhausted(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_resource_exhausted_count = self.retry_resource_exhausted_count.saturating_add(1);
    }

    pub fn record_retry_reservation_failure(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_reservation_failure_count = self.retry_reservation_failure_count.saturating_add(1);
    }

    pub fn record_hash_join_cache_hit(&mut self) {
        self.hash_join_cache_hits = self.hash_join_cache_hits.saturating_add(1);
    }

    pub fn record_hash_join_cache_miss(&mut self) {
        self.hash_join_cache_misses = self.hash_join_cache_misses.saturating_add(1);
    }

    pub fn record_hash_join_durable_cache_hit(&mut self) {
        self.hash_join_durable_cache_hits = self.hash_join_durable_cache_hits.saturating_add(1);
    }

    pub fn record_hash_join_durable_cache_miss(&mut self) {
        self.hash_join_durable_cache_misses = self.hash_join_durable_cache_misses.saturating_add(1);
    }

    pub fn record_hash_join_durable_cache_insert(&mut self) {
        self.hash_join_durable_cache_inserts = self.hash_join_durable_cache_inserts.saturating_add(1);
    }

    pub fn record_hash_join_build(&mut self) {
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(1);
    }

    pub fn record_filtered_join_build(&mut self) {
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(1);
    }

    pub fn record_one_shot_join(&mut self) {
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(1);
    }

    pub fn record_hash_join_hash_table_bytes(&mut self, bytes: u64) {
        self.hash_join_hash_table_bytes = self.hash_join_hash_table_bytes.saturating_add(bytes);
    }

    pub fn record_filtered_join_hash_table_bytes(&mut self, bytes: u64) {
        self.filtered_join_hash_table_bytes = self.filtered_join_hash_table_bytes.saturating_add(bytes);
    }

    pub fn record_filtered_join_probe_state_bytes(&mut self, bytes: u64) {
        self.filtered_join_probe_state_bytes = self.filtered_join_probe_state_bytes.saturating_add(bytes);
    }

    pub fn record_mixed_join_size_data_bytes(&mut self, bytes: u64) {
        self.mixed_join_size_data_bytes = self.mixed_join_size_data_bytes.saturating_add(bytes);
    }

    pub fn record_one_shot_join_index_bytes(&mut self, bytes: u64) {
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(bytes);
    }

    pub fn record_one_shot_join_materialization_bytes(&mut self, bytes: u64) {
        self.one_shot_join_materialization_bytes = self.one_shot_join_materialization_bytes.saturating_add(bytes);
    }

    pub fn record_one_shot_join_output_bytes(&mut self, bytes: u64) {
        self.one_shot_join_output_bytes = self.one_shot_join_output_bytes.saturating_add(bytes);
    }

    pub fn record_compiled_pipeline_attempt(&mut self) {
        self.compiled_pipeline_attempts = self.compiled_pipeline_attempts.saturating_add(1);
    }

    pub fn record_compiled_pipeline_tasks_executed(&mut self, tasks: u64) {
        self.compiled_pipeline_tasks_executed = self.compiled_pipeline_tasks_executed.saturating_add(tasks);
    }

    pub fn record_compiled_pipeline_host_arrow_sink(&mut self) {
        self.compiled_pipeline_host_arrow_sinks = self.compiled_pipeline_host_arrow_sinks.saturating_add(1);
    }

    pub fn record_adaptive_runtime_action_counts(&mut self, retry_count: u64, spill_count: u64, downgrade_count: u64) {
        self.adaptive_retry_count = self.adaptive_retry_count.saturating_add(retry_count);
        self.adaptive_spill_count = self.adaptive_spill_count.saturating_add(spill_count);
        self.adaptive_downgrade_count = self.adaptive_downgrade_count.saturating_add(downgrade_count);
    }

    pub fn record_adaptive_memory_history_lookup(&mut self, hit: bool) {
        self.adaptive_memory_history_lookup_count = self.adaptive_memory_history_lookup_count.saturating_add(1);
        if hit {
            self.adaptive_memory_history_hit_count = self.adaptive_memory_history_hit_count.saturating_add(1);
        } else {
            self.adaptive_memory_history_miss_count = self.adaptive_memory_history_miss_count.saturating_add(1);
        }
    }

    pub fn record_partial_aggregate(&mut self, rows: u64) {
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(1);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(rows);
    }

    pub fn record_final_merge_attempt(&mut self) {
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(1);
    }

    pub fn record_final_merge_level(&mut self) {
        self.final_merge_levels = self.final_merge_levels.saturating_add(1);
    }

    pub fn merge_from(&mut self, other: QueryMetrics) {
        self.boundary.merge_from(other.boundary);
        self.operators.extend(other.operators);
        self.join_strategy_decisions.extend(other.join_strategy_decisions);
        self.input_batches = self.input_batches.saturating_add(other.input_batches);
        self.input_rows = self.input_rows.saturating_add(other.input_rows);
        self.peak_device_memory_bytes = max_optional(self.peak_device_memory_bytes, other.peak_device_memory_bytes);
        self.resident_dataframe_peak_bytes =
            max_optional(self.resident_dataframe_peak_bytes, other.resident_dataframe_peak_bytes);
        self.allocator_high_water_bytes =
            max_optional(self.allocator_high_water_bytes, other.allocator_high_water_bytes);
        self.reservation_requested_bytes =
            self.reservation_requested_bytes.saturating_add(other.reservation_requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(other.reservation_granted_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(other.reservation_denied_count);
        self.reservation_events.extend(other.reservation_events);
        self.budgeted_execution_events.extend(other.budgeted_execution_events);
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(other.source_chunks_planned);
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(other.source_chunks_attempted);
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(other.source_chunks_committed);
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(other.source_chunks_retried);
        self.source_read_prefetches_started =
            self.source_read_prefetches_started.saturating_add(other.source_read_prefetches_started);
        self.source_read_prefetches_completed =
            self.source_read_prefetches_completed.saturating_add(other.source_read_prefetches_completed);
        self.source_read_prefetches_skipped =
            self.source_read_prefetches_skipped.saturating_add(other.source_read_prefetches_skipped);
        self.source_read_prefetches_abandoned =
            self.source_read_prefetches_abandoned.saturating_add(other.source_read_prefetches_abandoned);
        self.source_cache_hits = self.source_cache_hits.saturating_add(other.source_cache_hits);
        self.source_cache_misses = self.source_cache_misses.saturating_add(other.source_cache_misses);
        self.source_cache_inserts = self.source_cache_inserts.saturating_add(other.source_cache_inserts);
        self.source_cache_insert_skipped =
            self.source_cache_insert_skipped.saturating_add(other.source_cache_insert_skipped);
        self.source_cache_retained_bytes = self.source_cache_retained_bytes.max(other.source_cache_retained_bytes);
        self.row_groups_planned = self.row_groups_planned.saturating_add(other.row_groups_planned);
        self.row_groups_committed = self.row_groups_committed.saturating_add(other.row_groups_committed);
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(other.partial_aggregate_count);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(other.partial_aggregate_rows);
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(other.final_merge_attempts);
        self.final_merge_levels = self.final_merge_levels.saturating_add(other.final_merge_levels);
        self.retry_count = self.retry_count.saturating_add(other.retry_count);
        self.retry_resource_exhausted_count =
            self.retry_resource_exhausted_count.saturating_add(other.retry_resource_exhausted_count);
        self.retry_reservation_failure_count =
            self.retry_reservation_failure_count.saturating_add(other.retry_reservation_failure_count);
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(other.hash_join_build_count);
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(other.filtered_join_build_count);
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(other.one_shot_join_count);
        self.hash_join_cache_hits = self.hash_join_cache_hits.saturating_add(other.hash_join_cache_hits);
        self.hash_join_cache_misses = self.hash_join_cache_misses.saturating_add(other.hash_join_cache_misses);
        self.hash_join_durable_cache_hits =
            self.hash_join_durable_cache_hits.saturating_add(other.hash_join_durable_cache_hits);
        self.hash_join_durable_cache_misses =
            self.hash_join_durable_cache_misses.saturating_add(other.hash_join_durable_cache_misses);
        self.hash_join_durable_cache_inserts =
            self.hash_join_durable_cache_inserts.saturating_add(other.hash_join_durable_cache_inserts);
        self.hash_join_hash_table_bytes =
            self.hash_join_hash_table_bytes.saturating_add(other.hash_join_hash_table_bytes);
        self.filtered_join_hash_table_bytes =
            self.filtered_join_hash_table_bytes.saturating_add(other.filtered_join_hash_table_bytes);
        self.filtered_join_probe_state_bytes =
            self.filtered_join_probe_state_bytes.saturating_add(other.filtered_join_probe_state_bytes);
        self.mixed_join_size_data_bytes =
            self.mixed_join_size_data_bytes.saturating_add(other.mixed_join_size_data_bytes);
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(other.one_shot_join_index_bytes);
        self.one_shot_join_materialization_bytes =
            self.one_shot_join_materialization_bytes.saturating_add(other.one_shot_join_materialization_bytes);
        self.one_shot_join_output_bytes =
            self.one_shot_join_output_bytes.saturating_add(other.one_shot_join_output_bytes);
        self.parquet_files = self.parquet_files.saturating_add(other.parquet_files);
        self.parquet_bytes_known = self.parquet_bytes_known.saturating_add(other.parquet_bytes_known);
        self.row_groups_selected = self.row_groups_selected.saturating_add(other.row_groups_selected);
        self.read_columns = self.read_columns.saturating_add(other.read_columns);
        self.source_metadata_load_time += other.source_metadata_load_time;
        self.source_read_queue_wait_time += other.source_read_queue_wait_time;
        self.source_cache_lookup_time += other.source_cache_lookup_time;
        self.parquet_read_time += other.parquet_read_time;
        self.source_read_time += other.source_read_time;
        self.operator_time += other.operator_time;
        self.chunk_operator_time += other.chunk_operator_time;
        self.final_merge_time += other.final_merge_time;
        self.final_sort_time += other.final_sort_time;
        self.materialization_time += other.materialization_time;
        self.compiled_pipeline_attempts =
            self.compiled_pipeline_attempts.saturating_add(other.compiled_pipeline_attempts);
        self.compiled_pipeline_tasks_executed =
            self.compiled_pipeline_tasks_executed.saturating_add(other.compiled_pipeline_tasks_executed);
        self.compiled_pipeline_host_arrow_sinks =
            self.compiled_pipeline_host_arrow_sinks.saturating_add(other.compiled_pipeline_host_arrow_sinks);
        self.adaptive_retry_count = self.adaptive_retry_count.saturating_add(other.adaptive_retry_count);
        self.adaptive_spill_count = self.adaptive_spill_count.saturating_add(other.adaptive_spill_count);
        self.adaptive_downgrade_count = self.adaptive_downgrade_count.saturating_add(other.adaptive_downgrade_count);
        self.adaptive_memory_history_lookup_count =
            self.adaptive_memory_history_lookup_count.saturating_add(other.adaptive_memory_history_lookup_count);
        self.adaptive_memory_history_hit_count =
            self.adaptive_memory_history_hit_count.saturating_add(other.adaptive_memory_history_hit_count);
        self.adaptive_memory_history_miss_count =
            self.adaptive_memory_history_miss_count.saturating_add(other.adaptive_memory_history_miss_count);
    }
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinStrategyDecisionMetrics {
    pub node_id: String,
    pub facts: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedExecutionEvent {
    pub operation: String,
    pub boundary: String,
    pub budget_bytes: u64,
    pub enforcement: String,
    pub status: String,
    pub allocator_high_water_bytes: Option<u64>,
    pub retained_bytes: Option<u64>,
    pub output_bytes: Option<u64>,
    pub output_rows: Option<u64>,
    pub unresolved_contract: Option<String>,
    pub context_facts: BTreeMap<String, String>,
}

impl BudgetedExecutionEvent {
    fn new(operation: impl Into<String>, report: &ExecutionBudgetReport) -> Self {
        Self {
            operation: operation.into(),
            boundary: report.request().boundary().to_string(),
            budget_bytes: report.request().budget_bytes() as u64,
            enforcement: report.enforcement().as_str().to_string(),
            status: report.status().as_str().to_string(),
            allocator_high_water_bytes: report.allocator_high_water_bytes().map(|bytes| bytes as u64),
            retained_bytes: report.retained_bytes().map(|bytes| bytes as u64),
            output_bytes: report.output_bytes().map(|bytes| bytes as u64),
            output_rows: report.output_rows().map(|rows| rows as u64),
            unresolved_contract: report.unresolved_contract().map(str::to_string),
            context_facts: report.context_facts().clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
    pub node_id: String,
    pub operator_kind: String,
    pub input_rows: u64,
    pub input_bytes: u64,
    pub output_rows: u64,
    pub output_bytes: u64,
    pub join: Option<JoinOperatorMetrics>,
}

impl OperatorMetrics {
    pub fn join(node_id: impl ToString, join: JoinOperatorMetrics) -> Self {
        Self { node_id: node_id.to_string(), operator_kind: "join".to_string(), join: Some(join), ..Default::default() }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinOperatorMetrics {
    pub join_type: String,
    pub execution_strategy: String,
    pub key_count: usize,
    pub residual_predicate: bool,
    pub null_equality: String,
    /// Planner orientation used to execute or describe this join.
    ///
    /// For reusable hash joins this is the side that owns the hash table. For
    /// keyless conditional and cross joins no hash table is built, so this is an
    /// orientation label for comparable metrics rather than a physical build
    /// artifact.
    pub build_side: String,
    /// The side opposite `build_side` in the planner orientation.
    pub probe_side: String,
    pub build_rows: u64,
    pub build_bytes: u64,
    pub probe_rows: u64,
    pub probe_bytes: u64,
    /// Device bytes retained by the reusable cuDF join build artifact.
    ///
    /// This is a direct cudf-rs observation. For hash joins it is the retained
    /// hash table; for filtered joins it is the retained filtered-join build
    /// artifact. It remains `0` for join paths without a reusable build artifact.
    pub hash_table_bytes: u64,
    /// Device bytes retained by a filtered-join probe state.
    pub filtered_probe_state_bytes: u64,
    /// Device bytes retained by a mixed join retained size-data state.
    pub mixed_size_data_bytes: u64,
    /// Device bytes retained by join gather maps produced for this join.
    ///
    /// This is a direct cudf-rs observation over the returned device index
    /// columns, not a row-count estimate.
    pub gather_map_bytes: u64,
    pub output_reservation_bytes: u64,
    pub output_rows: u64,
    pub output_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundaryMetrics {
    pub gpu_resident_input_handoffs: u64,
    pub host_to_device_uploads: u64,
    pub host_to_device_uploaded_rows: u64,
    pub host_to_device_uploaded_bytes: u64,
    pub device_to_host_materializations: u64,
    pub device_to_host_materialized_rows: u64,
    pub device_to_host_materialized_bytes: u64,
    pub device_to_device_copies: u64,
    pub device_to_device_copied_bytes: u64,
}

impl BoundaryMetrics {
    pub fn record_gpu_resident_input_handoff(&mut self) {
        self.gpu_resident_input_handoffs += 1;
    }

    pub fn record_host_to_device_upload(&mut self, rows: u64, bytes: u64) {
        self.host_to_device_uploads += 1;
        self.host_to_device_uploaded_rows += rows;
        self.host_to_device_uploaded_bytes += bytes;
    }

    pub fn record_device_to_host_materialization(&mut self, rows: u64, bytes: u64) {
        self.device_to_host_materializations += 1;
        self.device_to_host_materialized_rows += rows;
        self.device_to_host_materialized_bytes += bytes;
    }

    pub fn record_device_to_device_copy(&mut self, bytes: u64) {
        self.device_to_device_copies += 1;
        self.device_to_device_copied_bytes += bytes;
    }

    fn merge_from(&mut self, other: BoundaryMetrics) {
        self.gpu_resident_input_handoffs =
            self.gpu_resident_input_handoffs.saturating_add(other.gpu_resident_input_handoffs);
        self.host_to_device_uploads = self.host_to_device_uploads.saturating_add(other.host_to_device_uploads);
        self.host_to_device_uploaded_rows =
            self.host_to_device_uploaded_rows.saturating_add(other.host_to_device_uploaded_rows);
        self.host_to_device_uploaded_bytes =
            self.host_to_device_uploaded_bytes.saturating_add(other.host_to_device_uploaded_bytes);
        self.device_to_host_materializations =
            self.device_to_host_materializations.saturating_add(other.device_to_host_materializations);
        self.device_to_host_materialized_rows =
            self.device_to_host_materialized_rows.saturating_add(other.device_to_host_materialized_rows);
        self.device_to_host_materialized_bytes =
            self.device_to_host_materialized_bytes.saturating_add(other.device_to_host_materialized_bytes);
        self.device_to_device_copies = self.device_to_device_copies.saturating_add(other.device_to_device_copies);
        self.device_to_device_copied_bytes =
            self.device_to_device_copied_bytes.saturating_add(other.device_to_device_copied_bytes);
    }
}
