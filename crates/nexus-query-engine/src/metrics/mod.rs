//! Query execution metrics and stable diagnostic event rows.
//!
//! Byte counters are device bytes unless the field name explicitly says host,
//! Arrow, or Parquet compressed bytes. Cumulative reservation totals can exceed
//! device memory; live-peak fields are instantaneous maxima.

use std::{collections::BTreeMap, time::Duration};

use crate::{
    graph_memory::GraphReservationReport,
    harness::max_option,
    runtime::{ExecutionBoundaryPolicy, MemoryControlPolicy},
};

mod adaptive;
mod aggregate;
mod boundary;
mod correlation;
mod descriptor;
mod dynamic_filter;
mod graph;
mod join;
mod json;
mod memory;
mod operator;
mod source_read;
mod task;

use correlation::inherited_query_id;
pub(crate) use memory::ReservationProjectionLedgerHandle;

pub use self::{
    aggregate::AggregateStrategyDecisionMetrics,
    boundary::BoundaryMetrics,
    correlation::{CompiledTaskId, PlanNodeId, PlanTaskMapping, QueryId, TraceId},
    descriptor::{MetricAggregation, MetricDescriptor, MetricDirection, MetricUnit, QUERY_METRIC_DESCRIPTORS},
    dynamic_filter::{DynamicFilterRuntimeMetrics, DynamicFilterSourceSnapshot},
    graph::GraphAlgorithmRuntimeMetrics,
    join::{JoinOperatorMetrics, JoinStrategyDecisionMetrics},
    memory::{MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome},
    operator::OperatorMetrics,
    source_read::{SourceReadParquetEvidenceMetrics, SourceReadRuntimeMetrics},
    task::{BudgetedExecutionEvent, TaskOperatorMetrics, TaskRuntimeMetrics},
};

/// Query-scoped metrics accumulated during native execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    pub(crate) reservation_projection_ledger: memory::ReservationProjectionLedger,
    /// Query-wide diagnostic correlation id.
    pub query_id: QueryId,
    /// Optional parent trace id.
    pub trace_id: Option<TraceId>,
    /// Data-boundary transfer counters.
    pub boundary: BoundaryMetrics,
    /// Operator-level metric rows in observation order.
    pub operators: Vec<OperatorMetrics>,
    /// Graph algorithm runtime rows in observation order.
    pub graph_algorithm_runtime_rows: Vec<GraphAlgorithmRuntimeMetrics>,
    /// Graph memory-tier reservation rows in observation order.
    pub graph_reservation_rows: Vec<GraphReservationReport>,
    /// Join-strategy diagnostics captured during planning/execution.
    pub join_strategy_decisions: Vec<JoinStrategyDecisionMetrics>,
    /// Aggregate-strategy diagnostics captured during planning/execution.
    pub aggregate_strategy_decisions: Vec<AggregateStrategyDecisionMetrics>,
    /// Dynamic-filter runtime event rows in observation order.
    pub dynamic_filter_runtime_rows: Vec<DynamicFilterRuntimeMetrics>,
    /// Source-read runtime event rows in observation order.
    pub source_read_runtime_rows: Vec<SourceReadRuntimeMetrics>,
    /// Memory policy used by the native bounded executor.
    pub memory_control_policy: MemoryControlPolicy,
    /// Boundary policy used for host/device materialization.
    pub execution_boundary_policy: ExecutionBoundaryPolicy,
    /// Number of input batches observed by native execution.
    pub input_batches: u64,
    /// Number of input rows observed by native execution.
    pub input_rows: u64,
    /// Backward-compatible alias for `resident_dataframe_peak_bytes`.
    ///
    /// This is not an allocator high-water mark.
    pub peak_device_memory_bytes: Option<u64>,
    /// Peak bytes retained by resident cudf DataFrames.
    pub resident_dataframe_peak_bytes: Option<u64>,
    /// cuDF allocator high-water bytes when reported by lower layers.
    pub allocator_high_water_bytes: Option<u64>,
    /// Cumulative reservation bytes requested over the query lifetime.
    pub reservation_requested_bytes: u64,
    /// Cumulative reservation bytes granted over the query lifetime.
    pub reservation_granted_bytes: u64,
    /// Peak live reserved bytes observed at any instant.
    pub reservation_live_peak_bytes: u64,
    /// Count of denied reservation requests.
    pub reservation_denied_count: u64,
    /// Largest compiled-pipeline task reservation request granted or applied during this query.
    pub max_task_reservation_bytes: u64,
    /// Largest lower-level budgeted cuDF peak reported by a task-like operation.
    pub max_task_observed_peak_bytes: Option<u64>,
    /// Largest retained byte count reported by a task-like lower-level cuDF operation.
    pub max_task_retained_bytes: Option<u64>,
    pub(crate) task_kind_by_compiled_task_id: BTreeMap<String, String>,
    /// Metrics aggregated by task operator kind.
    pub task_operator_metrics: BTreeMap<String, TaskOperatorMetrics>,
    /// Metrics keyed by compiled pipeline task id.
    pub task_runtime_metrics: BTreeMap<String, TaskRuntimeMetrics>,
    /// Query-local mapping rows between native plan nodes and compiled tasks.
    pub plan_task_mapping_rows: Vec<PlanTaskMapping>,
    /// Optional frontend physical-plan shape used to annotate plan/task mapping rows.
    pub plan_task_mapping_datafusion_shape: Option<String>,
    /// Exact memory reservation timeline events.
    ///
    /// These public rows are not bounded by runtime-observation retention
    /// limits; dropping them would make the public totals and task projections
    /// misleading when the separately retained trace is truncated.
    pub reservation_events: Vec<MemoryReservationEvent>,
    /// Successful lower-level cuDF calls that returned an execution-budget report.
    ///
    /// Token-routed operations without a report-returning cuDF API may still attach budget failure
    /// facts on errors, but they do not synthesize success events here.
    pub budgeted_execution_events: Vec<BudgetedExecutionEvent>,
    pub source_chunks_planned: u64,
    /// Includes both first attempts and retries.
    pub source_chunks_attempted: u64,
    pub source_chunks_committed: u64,
    pub source_chunks_retried: u64,
    pub source_read_prefetches_started: u64,
    pub source_read_prefetches_completed: u64,
    pub source_read_prefetches_skipped: u64,
    pub source_read_prefetches_abandoned: u64,
    /// Query-scoped source cache hits.
    pub source_cache_hits: u64,
    /// Query-scoped source cache misses.
    pub source_cache_misses: u64,
    /// Query-scoped source cache insertions.
    pub source_cache_inserts: u64,
    /// Query-scoped skipped source cache insertions.
    pub source_cache_insert_skipped: u64,
    /// Peak retained source-cache bytes.
    pub source_cache_retained_bytes: u64,
    /// Row groups planned across source chunks.
    pub row_groups_planned: u64,
    /// Row groups committed across source chunks.
    pub row_groups_committed: u64,
    /// Number of partial aggregate tasks.
    pub partial_aggregate_count: u64,
    /// Input rows consumed by partial aggregates.
    pub partial_aggregate_input_rows: u64,
    /// Rows produced by partial aggregates.
    pub partial_aggregate_rows: u64,
    /// Peak retained bytes from partial aggregate outputs.
    pub partial_aggregate_retained_bytes: u64,
    pub final_merge_attempts: u64,
    pub final_merge_levels: u64,
    /// Largest fan-in observed across final-merge levels.
    pub final_merge_max_fan_in: u64,
    pub retry_count: u64,
    /// Retries caused by lower-level resource exhaustion.
    pub retry_resource_exhausted_count: u64,
    /// Retries caused by reservation failure.
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
    /// Number of Parquet files observed by source reads.
    pub parquet_files: u64,
    /// Sum of known Parquet file bytes.
    pub parquet_bytes_known: u64,
    /// Row groups selected by source reads.
    pub row_groups_selected: u64,
    /// Physical read columns selected by source reads.
    pub read_columns: u64,
    /// Time spent loading source metadata.
    pub source_metadata_load_time: Duration,
    /// Time spent waiting in the source-read queue.
    pub source_read_queue_wait_time: Duration,
    /// Time spent looking up source cache entries.
    pub source_cache_lookup_time: Duration,
    /// Time spent inside Parquet read calls.
    pub parquet_read_time: Duration,
    /// Time spent in source-read orchestration.
    pub source_read_time: Duration,
    /// Time spent in non-source operators.
    pub operator_time: Duration,
    /// Time spent in per-chunk operators.
    pub chunk_operator_time: Duration,
    /// Time spent in final merge operators.
    pub final_merge_time: Duration,
    /// Time spent in final sort operators.
    pub final_sort_time: Duration,
    /// Time spent materializing final outputs.
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
    /// Dynamic-filter payloads successfully produced.
    pub dynamic_filter_produced_count: u64,
    /// Dynamic-filter routes whose producers completed without a usable payload.
    pub dynamic_filter_skipped_count: u64,
    /// Dynamic-filter payloads applied to a runtime consumer.
    pub dynamic_filter_applied_count: u64,
    /// Row groups removed from source reads by dynamic filters.
    pub dynamic_filter_row_groups_pruned: u64,
    /// Estimated source-read bytes removed by dynamic filters.
    pub dynamic_filter_source_bytes_pruned: u64,
    /// Source rows selected before dynamic reader predicates were applied.
    ///
    /// Counted once per source chunk, not once per dynamic-filter route, so
    /// multiple merged reader predicates do not duplicate query-level evidence.
    pub dynamic_filter_reader_predicate_rows_before: u64,
    /// Source rows materialized after dynamic reader predicates were applied.
    ///
    /// Counted once per source chunk, matching
    /// [`Self::dynamic_filter_reader_predicate_rows_before`].
    pub dynamic_filter_reader_predicate_rows_after: u64,
    /// Source rows removed by dynamic reader predicates.
    ///
    /// Counted once per source chunk and only when both before/after row counts
    /// are known.
    pub dynamic_filter_reader_predicate_rows_filtered: u64,
    /// Materialized rows before post-decode dynamic filters were applied.
    ///
    /// Counted once per post-decode filter application or stream chunk.
    pub dynamic_filter_post_decode_rows_before: u64,
    /// Materialized rows after post-decode dynamic filters were applied.
    ///
    /// Counted once per post-decode filter application or stream chunk.
    pub dynamic_filter_post_decode_rows_after: u64,
    /// Materialized rows removed by post-decode dynamic filters.
    ///
    /// Counted once per post-decode filter application or stream chunk and
    /// only when both before/after row counts are known.
    pub dynamic_filter_post_decode_rows_filtered: u64,
    /// Materialized rows removed by source-adjacent post-decode dynamic filters.
    pub dynamic_filter_source_post_decode_rows_filtered: u64,
    /// Materialized rows removed by filter-operator post-decode dynamic filters.
    pub dynamic_filter_filter_post_decode_rows_filtered: u64,
    /// Materialized rows removed by hash-join-probe post-decode dynamic filters.
    pub dynamic_filter_join_probe_post_decode_rows_filtered: u64,
}

impl QueryMetrics {
    /// Creates empty metrics for a query id minted or supplied by the caller.
    pub fn new(query_id: impl Into<QueryId>) -> Self {
        Self { query_id: query_id.into(), ..Default::default() }
    }

    /// Returns the query-wide diagnostic correlation id.
    pub fn query_id(&self) -> &QueryId {
        &self.query_id
    }

    /// Assigns a query id to this metrics value and any already-collected runtime rows.
    pub fn set_query_id(&mut self, query_id: impl Into<QueryId>) {
        self.query_id = query_id.into();
        self.stamp_existing_rows_with_query_id();
    }

    /// Returns a copy of this metrics value with a query id assigned.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        self.set_query_id(query_id);
        self
    }

    /// Returns a copy of this metrics value with a frontend shape attached to plan/task mappings.
    #[must_use]
    pub fn with_plan_task_mapping_datafusion_shape(mut self, shape: impl Into<String>) -> Self {
        let shape = shape.into();
        if !shape.is_empty() {
            self.plan_task_mapping_datafusion_shape = Some(shape.clone());
            for row in &mut self.plan_task_mapping_rows {
                if row.datafusion_shape.is_none() {
                    row.datafusion_shape = Some(shape.clone());
                }
            }
        }
        self
    }

    pub(super) fn inherited_query_id(&self) -> Option<QueryId> {
        inherited_query_id(&self.query_id)
    }

    fn stamp_existing_rows_with_query_id(&mut self) {
        let Some(query_id) = self.inherited_query_id() else {
            return;
        };
        for event in &mut self.reservation_events {
            event.query_id = Some(query_id.clone());
        }
        for row in &mut self.budgeted_execution_events {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.dynamic_filter_runtime_rows {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.source_read_runtime_rows {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.join_strategy_decisions {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.aggregate_strategy_decisions {
            row.query_id = Some(query_id.clone());
        }
        for row in self.task_operator_metrics.values_mut() {
            row.query_id = Some(query_id.clone());
        }
        for row in self.task_runtime_metrics.values_mut() {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.plan_task_mapping_rows {
            row.query_id = query_id.clone();
        }
        for row in &mut self.operators {
            row.query_id = Some(query_id.clone());
            if let Some(join) = &mut row.join {
                join.query_id = Some(query_id.clone());
            }
        }
        for row in &mut self.graph_algorithm_runtime_rows {
            row.query_id = Some(query_id.clone());
        }
        for row in &mut self.graph_reservation_rows {
            row.query_id = Some(query_id.clone());
        }
    }

    /// Merges another query metrics value into this one.
    pub fn merge_from(&mut self, mut other: QueryMetrics) {
        self.merge_reservation_projection_ledger(&other);
        if !self.query_id.is_assigned() && other.query_id.is_assigned() {
            self.query_id = other.query_id.clone();
            self.stamp_existing_rows_with_query_id();
        }
        if self.query_id.is_assigned() && !other.query_id.is_assigned() {
            other.set_query_id(self.query_id.clone());
        }
        if self.trace_id.is_none() {
            self.trace_id = other.trace_id.clone();
        }
        self.boundary.merge_from(other.boundary);
        self.operators.extend(other.operators);
        self.graph_algorithm_runtime_rows.extend(other.graph_algorithm_runtime_rows);
        self.graph_reservation_rows.extend(other.graph_reservation_rows);
        self.join_strategy_decisions.extend(other.join_strategy_decisions);
        self.aggregate_strategy_decisions.extend(other.aggregate_strategy_decisions);
        self.dynamic_filter_runtime_rows.extend(other.dynamic_filter_runtime_rows);
        self.source_read_runtime_rows.extend(other.source_read_runtime_rows);
        self.input_batches = self.input_batches.saturating_add(other.input_batches);
        self.input_rows = self.input_rows.saturating_add(other.input_rows);
        self.peak_device_memory_bytes = max_option(self.peak_device_memory_bytes, other.peak_device_memory_bytes);
        self.resident_dataframe_peak_bytes =
            max_option(self.resident_dataframe_peak_bytes, other.resident_dataframe_peak_bytes);
        self.allocator_high_water_bytes = max_option(self.allocator_high_water_bytes, other.allocator_high_water_bytes);
        self.reservation_requested_bytes =
            self.reservation_requested_bytes.saturating_add(other.reservation_requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(other.reservation_granted_bytes);
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(other.reservation_live_peak_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(other.reservation_denied_count);
        self.max_task_reservation_bytes = self.max_task_reservation_bytes.max(other.max_task_reservation_bytes);
        self.max_task_observed_peak_bytes =
            max_option(self.max_task_observed_peak_bytes, other.max_task_observed_peak_bytes);
        self.max_task_retained_bytes = max_option(self.max_task_retained_bytes, other.max_task_retained_bytes);
        self.task_kind_by_compiled_task_id.extend(other.task_kind_by_compiled_task_id);
        self.plan_task_mapping_rows.extend(other.plan_task_mapping_rows);
        if self.plan_task_mapping_datafusion_shape.is_none() {
            self.plan_task_mapping_datafusion_shape = other.plan_task_mapping_datafusion_shape;
        }
        for (operator_kind, metrics) in other.task_operator_metrics {
            self.task_operator_metrics
                .entry(operator_kind.clone())
                .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() })
                .merge_from(metrics);
        }
        for (compiled_task_id, metrics) in other.task_runtime_metrics {
            self.task_runtime_metrics
                .entry(compiled_task_id.clone())
                .or_insert_with(|| TaskRuntimeMetrics { compiled_task_id, ..Default::default() })
                .merge_from(metrics);
        }
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
        self.partial_aggregate_input_rows =
            self.partial_aggregate_input_rows.saturating_add(other.partial_aggregate_input_rows);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(other.partial_aggregate_rows);
        self.partial_aggregate_retained_bytes =
            self.partial_aggregate_retained_bytes.max(other.partial_aggregate_retained_bytes);
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(other.final_merge_attempts);
        self.final_merge_levels = self.final_merge_levels.saturating_add(other.final_merge_levels);
        self.final_merge_max_fan_in = self.final_merge_max_fan_in.max(other.final_merge_max_fan_in);
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
        self.dynamic_filter_produced_count =
            self.dynamic_filter_produced_count.saturating_add(other.dynamic_filter_produced_count);
        self.dynamic_filter_skipped_count =
            self.dynamic_filter_skipped_count.saturating_add(other.dynamic_filter_skipped_count);
        self.dynamic_filter_applied_count =
            self.dynamic_filter_applied_count.saturating_add(other.dynamic_filter_applied_count);
        self.dynamic_filter_row_groups_pruned =
            self.dynamic_filter_row_groups_pruned.saturating_add(other.dynamic_filter_row_groups_pruned);
        self.dynamic_filter_source_bytes_pruned =
            self.dynamic_filter_source_bytes_pruned.saturating_add(other.dynamic_filter_source_bytes_pruned);
        self.dynamic_filter_reader_predicate_rows_before = self
            .dynamic_filter_reader_predicate_rows_before
            .saturating_add(other.dynamic_filter_reader_predicate_rows_before);
        self.dynamic_filter_reader_predicate_rows_after = self
            .dynamic_filter_reader_predicate_rows_after
            .saturating_add(other.dynamic_filter_reader_predicate_rows_after);
        self.dynamic_filter_reader_predicate_rows_filtered = self
            .dynamic_filter_reader_predicate_rows_filtered
            .saturating_add(other.dynamic_filter_reader_predicate_rows_filtered);
        self.dynamic_filter_post_decode_rows_before =
            self.dynamic_filter_post_decode_rows_before.saturating_add(other.dynamic_filter_post_decode_rows_before);
        self.dynamic_filter_post_decode_rows_after =
            self.dynamic_filter_post_decode_rows_after.saturating_add(other.dynamic_filter_post_decode_rows_after);
        self.dynamic_filter_post_decode_rows_filtered = self
            .dynamic_filter_post_decode_rows_filtered
            .saturating_add(other.dynamic_filter_post_decode_rows_filtered);
        self.dynamic_filter_source_post_decode_rows_filtered = self
            .dynamic_filter_source_post_decode_rows_filtered
            .saturating_add(other.dynamic_filter_source_post_decode_rows_filtered);
        self.dynamic_filter_filter_post_decode_rows_filtered = self
            .dynamic_filter_filter_post_decode_rows_filtered
            .saturating_add(other.dynamic_filter_filter_post_decode_rows_filtered);
        self.dynamic_filter_join_probe_post_decode_rows_filtered = self
            .dynamic_filter_join_probe_post_decode_rows_filtered
            .saturating_add(other.dynamic_filter_join_probe_post_decode_rows_filtered);
        self.stamp_existing_rows_with_query_id();
    }
}
fn optional_str(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_row_groups(value: Option<&[usize]>) -> String {
    value
        .map(|row_groups| row_groups.iter().map(usize::to_string).collect::<Vec<_>>().join(","))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "none".to_string())
}
