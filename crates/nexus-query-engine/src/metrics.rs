//! Query execution metrics and stable diagnostic event rows.
//!
//! Byte counters are device bytes unless the field name explicitly says host,
//! Arrow, or Parquet compressed bytes. Cumulative reservation totals can exceed
//! device memory; live-peak fields are instantaneous maxima.

use std::{collections::BTreeMap, time::Duration};

use cudf::ExecutionBudgetReport;

use crate::{
    dynamic_filter::{DynamicFilter, DynamicFilterConsumerReductionStage, DynamicFilterRoute},
    runtime::{ExecutionBoundaryPolicy, MemoryControlPolicy, MemoryReservationRequest, MemorySpaceId, ReservationKind},
};

/// Outcome of one memory-reservation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReservationEventOutcome {
    /// Requested reservation was granted.
    Granted,
    /// Requested reservation was denied.
    Denied,
    /// Existing reservation state was changed without a new admission decision.
    Applied,
}

impl MemoryReservationEventOutcome {
    /// Stable snake-case label emitted in debug rows.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Granted => "granted",
            Self::Denied => "denied",
            Self::Applied => "applied",
        }
    }
}

impl std::fmt::Display for MemoryReservationEventOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reservation timeline action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReservationEventAction {
    /// Initial reservation request.
    Reserve,
    /// Reservation grew by a positive delta.
    Grow,
    /// Reservation shrank by a negative delta.
    Shrink,
    /// Reservation was fully released.
    Release,
    /// Reservation ownership moved across boundaries.
    Transfer,
    /// Alias acquired a shared reservation.
    AliasAcquire,
    /// Alias changed its report-facing label.
    AliasRelabel,
    /// Alias released its shared reservation.
    AliasRelease,
}

impl MemoryReservationEventAction {
    /// Stable snake-case label emitted in debug rows.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reserve => "reserve",
            Self::Grow => "grow",
            Self::Shrink => "shrink",
            Self::Release => "release",
            Self::Transfer => "transfer",
            Self::AliasAcquire => "alias_acquire",
            Self::AliasRelabel => "alias_relabel",
            Self::AliasRelease => "alias_release",
        }
    }
}

impl std::fmt::Display for MemoryReservationEventAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One memory-reservation timeline row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReservationEvent {
    /// Reservation class affected by this event.
    pub reservation_kind: ReservationKind,
    /// Memory space whose budget was charged.
    pub memory_space: MemorySpaceId,
    /// Stable boundary label from the reservation request.
    pub memory_boundary: String,
    /// Reservation action applied at this point in the timeline.
    pub action: MemoryReservationEventAction,
    /// Requested bytes for reserve/grow decisions.
    pub requested_bytes: u64,
    /// Bytes granted or otherwise associated with this event.
    pub granted_bytes: u64,
    /// Signed change to live charged bytes.
    pub charged_delta_bytes: i128,
    /// Live reserved bytes after the event.
    pub current_reserved_bytes: u64,
    /// Budget limit in bytes for the memory space.
    pub budget_bytes: u64,
    /// Admission or application outcome.
    pub outcome: MemoryReservationEventOutcome,
    pub node_id: Option<String>,
    pub stage: Option<String>,
    pub chunk_index: Option<usize>,
    pub file_index: Option<usize>,
    pub path: Option<String>,
    /// Present on source-read events; selects row groups within the file.
    pub row_groups: Option<Vec<usize>>,
    pub attempt: Option<usize>,
    pub max_attempts: Option<usize>,
    pub compiled_task_id: Option<String>,
    pub resident_resource_id: Option<String>,
    pub source_read_contract_quality: Option<String>,
    /// Bytes used for admission sizing; may differ from the requested reservation.
    pub source_read_planning_bytes: Option<u64>,
    /// Byte estimate carried for diagnostics only; not used for admission.
    pub source_read_estimated_bytes: Option<u64>,
}

impl MemoryReservationEvent {
    /// Builds a granted reserve event from a reservation request.
    pub fn granted(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::from_request(
            request,
            MemoryReservationEventOutcome::Granted,
            MemoryReservationEventAction::Reserve,
            granted_bytes,
            i128::from(granted_bytes),
            granted_bytes,
        )
    }

    /// Builds a denied reserve event from a reservation request.
    pub fn denied(request: &MemoryReservationRequest, granted_bytes: u64) -> Self {
        Self::from_request(
            request,
            MemoryReservationEventOutcome::Denied,
            MemoryReservationEventAction::Reserve,
            granted_bytes,
            0,
            request.budget_bytes.saturating_sub(granted_bytes),
        )
    }

    /// Builds an event row from a request and explicit timeline state.
    pub fn from_request(
        request: &MemoryReservationRequest,
        outcome: MemoryReservationEventOutcome,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) -> Self {
        Self {
            reservation_kind: request.reservation_kind,
            memory_space: request.memory_space,
            memory_boundary: request.memory_boundary.clone(),
            action,
            requested_bytes: request.requested_bytes,
            granted_bytes,
            charged_delta_bytes,
            current_reserved_bytes,
            budget_bytes: request.budget_bytes,
            outcome,
            node_id: request.node_id.clone(),
            stage: request.stage.clone(),
            chunk_index: request.chunk_index,
            file_index: request.file_index,
            path: request.path.clone(),
            row_groups: request.row_groups.clone(),
            attempt: request.attempt,
            max_attempts: request.max_attempts,
            compiled_task_id: request.compiled_task_id.clone(),
            resident_resource_id: request.resident_resource_id.clone(),
            source_read_contract_quality: request.source_read_contract_quality.clone(),
            source_read_planning_bytes: request.source_read_planning_bytes,
            source_read_estimated_bytes: request.source_read_estimated_bytes,
        }
    }

    /// Serializes the event as a stable comma-separated debug row.
    pub fn to_stable_debug_row(&self) -> String {
        format!(
            "reservation_kind={},memory_space={},memory_boundary={},stage={},action={},outcome={},requested_bytes={},granted_bytes={},remaining_bytes={},current_reserved_bytes={},budget_bytes={},node_id={},compiled_task_id={},resident_resource_id={},chunk_index={},file_index={},path={},row_groups={},attempt={},max_attempts={},source_read_contract_quality={},source_read_planning_bytes={},source_read_estimated_bytes={}",
            self.reservation_kind,
            self.memory_space,
            self.memory_boundary,
            optional_str(self.stage.as_deref()),
            self.action,
            self.outcome,
            self.requested_bytes,
            self.granted_bytes,
            self.budget_bytes.saturating_sub(self.current_reserved_bytes),
            self.current_reserved_bytes,
            self.budget_bytes,
            optional_str(self.node_id.as_deref()),
            optional_str(self.compiled_task_id.as_deref()),
            optional_str(self.resident_resource_id.as_deref()),
            optional_usize(self.chunk_index),
            optional_usize(self.file_index),
            optional_str(self.path.as_deref()),
            optional_row_groups(self.row_groups.as_deref()),
            optional_usize(self.attempt),
            optional_usize(self.max_attempts),
            optional_str(self.source_read_contract_quality.as_deref()),
            optional_u64(self.source_read_planning_bytes),
            optional_u64(self.source_read_estimated_bytes),
        )
    }
}

/// Query-scoped metrics accumulated during native execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryMetrics {
    /// Data-boundary transfer counters.
    pub boundary: BoundaryMetrics,
    /// Operator-level metric rows in observation order.
    pub operators: Vec<OperatorMetrics>,
    /// Graph algorithm runtime rows in observation order.
    pub graph_algorithm_runtime_rows: Vec<GraphAlgorithmRuntimeMetrics>,
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
    pub(crate) current_compiled_task_id: Option<String>,
    /// Memory reservation timeline events.
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
    /// Object-store bytes fetched into host buffers.
    pub source_fetch_bytes: u64,
    /// Count of object-store SourceFetch operations.
    pub source_fetch_count: u64,
    /// Time spent waiting for host-buffer byte budget.
    pub host_fetch_budget_wait_time: Duration,
    /// Time spent waiting for object-store SourceFetch concurrency permits.
    pub source_fetch_concurrency_wait_time: Duration,
    /// Time spent fetching object-store bytes into host buffers.
    pub source_fetch_time: Duration,
    /// Time spent decoding host-buffer Parquet bytes into device frames.
    pub source_decode_time: Duration,
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
    /// Appends one operator metric row.
    pub fn record_operator_metrics(&mut self, metrics: OperatorMetrics) {
        self.operators.push(metrics);
    }

    /// Appends one graph algorithm runtime row.
    pub fn record_graph_algorithm_runtime(&mut self, metrics: GraphAlgorithmRuntimeMetrics) {
        self.graph_algorithm_runtime_rows.push(metrics);
    }

    /// Records join-strategy diagnostics for a node.
    pub fn record_join_strategy_decision(&mut self, node_id: impl ToString, facts: &BTreeMap<&'static str, String>) {
        self.join_strategy_decisions.push(JoinStrategyDecisionMetrics {
            node_id: node_id.to_string(),
            facts: facts.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect(),
        });
    }

    /// Records aggregate-strategy diagnostics for a node or execution stage.
    pub fn record_aggregate_strategy_decision(
        &mut self,
        node_id: impl ToString,
        facts: &BTreeMap<&'static str, String>,
    ) {
        self.aggregate_strategy_decisions.push(AggregateStrategyDecisionMetrics {
            node_id: node_id.to_string(),
            facts: facts.iter().map(|(key, value)| ((*key).to_string(), value.clone())).collect(),
        });
    }

    /// Records a dynamic-filter payload produced at runtime.
    pub fn record_dynamic_filter_produced(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        producer_rows: u64,
    ) {
        self.dynamic_filter_produced_count = self.dynamic_filter_produced_count.saturating_add(1);
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::produced(route, filter, producer_rows));
    }

    /// Records a dynamic-filter producer that completed without a usable payload.
    pub fn record_dynamic_filter_skipped(&mut self, route: &DynamicFilterRoute, reason: impl Into<String>) {
        self.dynamic_filter_skipped_count = self.dynamic_filter_skipped_count.saturating_add(1);
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::skipped(route, reason));
    }

    /// Records a dynamic-filter payload applied to a local-Parquet source consumer.
    pub fn record_dynamic_filter_applied(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        before: DynamicFilterSourceSnapshot,
        after: DynamicFilterSourceSnapshot,
    ) {
        let row_groups_pruned = before.row_groups.saturating_sub(after.row_groups);
        let source_bytes_pruned = before.source_bytes.unwrap_or(0).saturating_sub(after.source_bytes.unwrap_or(0));
        self.dynamic_filter_applied_count = self.dynamic_filter_applied_count.saturating_add(1);
        self.dynamic_filter_row_groups_pruned = self.dynamic_filter_row_groups_pruned.saturating_add(row_groups_pruned);
        self.dynamic_filter_source_bytes_pruned =
            self.dynamic_filter_source_bytes_pruned.saturating_add(source_bytes_pruned);
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::applied(
            route,
            filter,
            before,
            after,
            row_groups_pruned,
            source_bytes_pruned,
        ));
    }

    /// Records a dynamic-filter payload applied to a non-source consumer.
    pub fn record_dynamic_filter_applied_post_decode(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.dynamic_filter_applied_count = self.dynamic_filter_applied_count.saturating_add(1);
        self.record_dynamic_filter_post_decode_reduction(route.consumer_reduction_stage(), rows_before, rows_after);
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::applied_post_decode(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records an additional post-decode source stage for a route that was
    /// already counted as applied to its source consumer.
    pub fn record_dynamic_filter_source_post_decode_stage(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_post_decode_reduction(
            DynamicFilterConsumerReductionStage::SourcePostDecode,
            rows_before,
            rows_after,
        );
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::source_post_decode_stage(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records post-decode row evidence for a route already counted as applied.
    ///
    /// Lazy stream consumers install a route once, but the row reduction is
    /// only known as chunks are consumed.
    pub fn record_dynamic_filter_post_decode_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.record_dynamic_filter_post_decode_reduction(route.consumer_reduction_stage(), rows_before, rows_after);
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::post_decode_rows(
            route,
            filter,
            rows_before,
            rows_after,
        ));
    }

    /// Records source-reader row evidence for a dynamic-filter reader predicate.
    pub fn record_dynamic_filter_source_reader_predicate_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        source_file_index: usize,
        source_chunk_index: usize,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::source_reader_predicate_rows(
            route,
            filter,
            Some(source_file_index),
            Some(source_chunk_index),
            rows_before,
            rows_after,
        ));
    }

    /// Records source-reader row evidence for a batched dynamic-filter reader predicate.
    pub fn record_dynamic_filter_source_reader_predicate_batch_rows(
        &mut self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.dynamic_filter_runtime_rows.push(DynamicFilterRuntimeMetrics::source_reader_predicate_rows(
            route,
            filter,
            None,
            None,
            rows_before,
            rows_after,
        ));
    }

    /// Records de-duplicated source-reader row reduction for a chunk.
    pub fn record_dynamic_filter_reader_predicate_source_reduction(
        &mut self,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        if let Some(rows_before) = rows_before {
            self.dynamic_filter_reader_predicate_rows_before =
                self.dynamic_filter_reader_predicate_rows_before.saturating_add(rows_before);
        }
        if let Some(rows_after) = rows_after {
            self.dynamic_filter_reader_predicate_rows_after =
                self.dynamic_filter_reader_predicate_rows_after.saturating_add(rows_after);
        }
        if let Some(rows_filtered) = rows_before.zip(rows_after).map(|(before, after)| before.saturating_sub(after)) {
            self.dynamic_filter_reader_predicate_rows_filtered =
                self.dynamic_filter_reader_predicate_rows_filtered.saturating_add(rows_filtered);
        }
    }

    /// Records one local-Parquet source read or source-batch read.
    pub fn record_source_read_runtime(&mut self, metrics: SourceReadRuntimeMetrics) {
        self.source_read_runtime_rows.push(metrics);
    }

    fn record_dynamic_filter_post_decode_reduction(
        &mut self,
        stage: DynamicFilterConsumerReductionStage,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        if let Some(rows_before) = rows_before {
            self.dynamic_filter_post_decode_rows_before =
                self.dynamic_filter_post_decode_rows_before.saturating_add(rows_before);
        }
        if let Some(rows_after) = rows_after {
            self.dynamic_filter_post_decode_rows_after =
                self.dynamic_filter_post_decode_rows_after.saturating_add(rows_after);
        }
        if let Some(rows_filtered) = rows_before.zip(rows_after).map(|(before, after)| before.saturating_sub(after)) {
            self.dynamic_filter_post_decode_rows_filtered =
                self.dynamic_filter_post_decode_rows_filtered.saturating_add(rows_filtered);
            match stage {
                DynamicFilterConsumerReductionStage::SourcePostDecode => {
                    self.dynamic_filter_source_post_decode_rows_filtered =
                        self.dynamic_filter_source_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::FilterPostDecode => {
                    self.dynamic_filter_filter_post_decode_rows_filtered =
                        self.dynamic_filter_filter_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::JoinProbePostDecode => {
                    self.dynamic_filter_join_probe_post_decode_rows_filtered =
                        self.dynamic_filter_join_probe_post_decode_rows_filtered.saturating_add(rows_filtered);
                },
                DynamicFilterConsumerReductionStage::SourceReader => {},
            }
        }
    }

    /// Records one input batch and its row count.
    pub fn record_input_batch(&mut self, rows: u64) {
        self.input_batches += 1;
        self.input_rows += rows;
    }

    /// Records one Parquet source file observation.
    pub fn record_parquet_source(&mut self, known_bytes: Option<u64>, selected_row_groups: u64, read_columns: u64) {
        self.parquet_files += 1;
        self.parquet_bytes_known += known_bytes.unwrap_or(0);
        self.row_groups_selected += selected_row_groups;
        self.read_columns += read_columns;
    }

    /// Records a resident DataFrame memory observation in bytes.
    pub fn record_device_memory_observation(&mut self, bytes: u64) {
        self.peak_device_memory_bytes = Some(self.peak_device_memory_bytes.map_or(bytes, |peak| peak.max(bytes)));
        self.resident_dataframe_peak_bytes =
            Some(self.resident_dataframe_peak_bytes.map_or(bytes, |peak| peak.max(bytes)));
    }

    /// Adds a granted reservation to cumulative counters.
    pub fn record_reservation_granted(&mut self, requested_bytes: u64, granted_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted_bytes);
    }

    /// Adds a denied reservation to cumulative counters.
    pub fn record_reservation_denied(&mut self, requested_bytes: u64) {
        self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(requested_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(1);
    }

    /// Records a granted reservation event and updates live/task peaks.
    pub fn record_reservation_granted_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_granted(request.requested_bytes, granted_bytes);
        let event = MemoryReservationEvent::granted(request, granted_bytes);
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a denied reservation event and updates live/task peaks.
    pub fn record_reservation_denied_event(&mut self, request: &MemoryReservationRequest, granted_bytes: u64) {
        self.record_reservation_denied(request.requested_bytes);
        let event = MemoryReservationEvent::denied(request, granted_bytes);
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a granted reservation timeline event with an explicit action.
    pub fn record_reservation_granted_timeline_event(
        &mut self,
        request: &MemoryReservationRequest,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        charged_delta_bytes: i128,
        current_reserved_bytes: u64,
    ) {
        self.record_reservation_granted(request.requested_bytes, granted_bytes);
        let event = MemoryReservationEvent::from_request(
            request,
            MemoryReservationEventOutcome::Granted,
            action,
            granted_bytes,
            charged_delta_bytes,
            current_reserved_bytes,
        );
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records a denied reservation timeline event with an explicit action.
    pub fn record_reservation_denied_timeline_event(
        &mut self,
        request: &MemoryReservationRequest,
        action: MemoryReservationEventAction,
        granted_bytes: u64,
        current_reserved_bytes: u64,
    ) {
        self.record_reservation_denied(request.requested_bytes);
        let event = MemoryReservationEvent::from_request(
            request,
            MemoryReservationEventOutcome::Denied,
            action,
            granted_bytes,
            0,
            current_reserved_bytes,
        );
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    /// Records an already-built reservation timeline event.
    pub fn record_reservation_timeline_event(&mut self, event: MemoryReservationEvent) {
        self.record_reservation_live_peak(event.current_reserved_bytes);
        self.record_task_reservation_event(&event);
        self.reservation_events.push(event);
    }

    fn record_reservation_live_peak(&mut self, current_reserved_bytes: u64) {
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(current_reserved_bytes);
    }

    /// Records a lower-level cuDF execution-budget report.
    pub fn record_budgeted_execution(&mut self, operation: impl Into<String>, report: &ExecutionBudgetReport) {
        let observed_peak_bytes = report.allocator_high_water_bytes().map(|bytes| bytes as u64);
        let retained_bytes = report.retained_bytes().map(|bytes| bytes as u64);
        self.max_task_observed_peak_bytes = max_optional(self.max_task_observed_peak_bytes, observed_peak_bytes);
        self.max_task_retained_bytes = max_optional(self.max_task_retained_bytes, retained_bytes);
        if let Some(compiled_task_id) = self.current_compiled_task_id.clone() {
            self.record_task_budget_observation(&compiled_task_id, observed_peak_bytes, retained_bytes);
        }
        self.budgeted_execution_events.push(BudgetedExecutionEvent::new(operation, report));
    }

    /// Records execution of a compiled pipeline task.
    pub fn record_compiled_pipeline_task_execution(
        &mut self,
        compiled_task_id: impl ToString,
        operator_kind: impl Into<String>,
    ) {
        self.record_task_execution(compiled_task_id, operator_kind);
    }

    /// Records execution of a task id and operator kind.
    pub fn record_task_execution(&mut self, task_id: impl ToString, operator_kind: impl Into<String>) {
        let compiled_task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        self.task_kind_by_compiled_task_id.insert(compiled_task_id.clone(), operator_kind.clone());
        self.task_runtime_metrics.entry(compiled_task_id.clone()).or_insert_with(|| TaskRuntimeMetrics {
            compiled_task_id,
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        let row = self
            .task_operator_metrics
            .entry(operator_kind.clone())
            .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() });
        row.task_count = row.task_count.saturating_add(1);
    }

    /// Adds elapsed time to a task and its operator-kind aggregate.
    pub fn record_task_elapsed(&mut self, task_id: impl ToString, operator_kind: impl Into<String>, elapsed: Duration) {
        let compiled_task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        self.task_kind_by_compiled_task_id.insert(compiled_task_id.clone(), operator_kind.clone());
        let row = self.task_runtime_metrics.entry(compiled_task_id.clone()).or_insert_with(|| TaskRuntimeMetrics {
            compiled_task_id,
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        row.operator_kind = operator_kind.clone();
        row.elapsed_time += elapsed;

        let operator = self
            .task_operator_metrics
            .entry(operator_kind.clone())
            .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() });
        operator.elapsed_time += elapsed;
    }

    /// Sets the compiled task id associated with subsequent budget reports.
    pub fn set_current_compiled_task(&mut self, compiled_task_id: impl ToString) {
        self.set_current_task(compiled_task_id);
    }

    /// Sets the task id associated with subsequent budget reports.
    pub fn set_current_task(&mut self, task_id: impl ToString) {
        self.current_compiled_task_id = Some(task_id.to_string());
    }

    /// Clears the current compiled task id.
    pub fn clear_current_compiled_task(&mut self) {
        self.clear_current_task();
    }

    /// Clears the current task id.
    pub fn clear_current_task(&mut self) {
        self.current_compiled_task_id = None;
    }

    fn record_task_reservation_event(&mut self, event: &MemoryReservationEvent) {
        let Some(compiled_task_id) = event.compiled_task_id.as_deref() else {
            return;
        };
        if !self.task_kind_by_compiled_task_id.contains_key(compiled_task_id)
            && let Some(operator_kind) = legacy_bounded_operator_kind_from_task_id(compiled_task_id)
        {
            self.task_kind_by_compiled_task_id.insert(compiled_task_id.to_string(), operator_kind.to_string());
        }
        self.record_task_runtime_reservation_event(compiled_task_id, event);
        if matches!(
            event.action,
            MemoryReservationEventAction::Reserve
                | MemoryReservationEventAction::Grow
                | MemoryReservationEventAction::Shrink
        ) && event.outcome != MemoryReservationEventOutcome::Denied
        {
            let event_reservation_bytes = if event.charged_delta_bytes > 0 {
                u64::try_from(event.charged_delta_bytes).unwrap_or(u64::MAX)
            } else {
                event.granted_bytes
            };
            self.max_task_reservation_bytes = self.max_task_reservation_bytes.max(event_reservation_bytes);
        }

        let Some(operator_kind) = self.task_kind_by_compiled_task_id.get(compiled_task_id).cloned() else {
            return;
        };
        let row = self
            .task_operator_metrics
            .entry(operator_kind.clone())
            .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() });
        if event.charged_delta_bytes >= 0 {
            row.live_reserved_bytes =
                row.live_reserved_bytes.saturating_add(u64::try_from(event.charged_delta_bytes).unwrap_or(u64::MAX));
        } else {
            row.live_reserved_bytes =
                row.live_reserved_bytes.saturating_sub(u64::try_from(-event.charged_delta_bytes).unwrap_or(u64::MAX));
        }
        row.reservation_live_peak_bytes = row.reservation_live_peak_bytes.max(row.live_reserved_bytes);
    }

    fn record_task_runtime_reservation_event(&mut self, compiled_task_id: &str, event: &MemoryReservationEvent) {
        let operator_kind =
            self.task_kind_by_compiled_task_id.get(compiled_task_id).cloned().unwrap_or_else(|| "unknown".to_string());
        let is_new_legacy_task =
            is_legacy_bounded_task_id(compiled_task_id) && !self.task_runtime_metrics.contains_key(compiled_task_id);
        let row = self.task_runtime_metrics.entry(compiled_task_id.to_string()).or_insert_with(|| TaskRuntimeMetrics {
            compiled_task_id: compiled_task_id.to_string(),
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        row.operator_kind = operator_kind.clone();
        row.record_reservation_event(event);

        let operator = self
            .task_operator_metrics
            .entry(operator_kind.clone())
            .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() });
        if is_new_legacy_task {
            operator.task_count = operator.task_count.saturating_add(1);
        }
        operator.record_reservation_event(event);
    }

    fn record_task_budget_observation(
        &mut self,
        compiled_task_id: &str,
        observed_peak_bytes: Option<u64>,
        retained_bytes: Option<u64>,
    ) {
        let operator_kind =
            self.task_kind_by_compiled_task_id.get(compiled_task_id).cloned().unwrap_or_else(|| "unknown".to_string());
        let row = self.task_runtime_metrics.entry(compiled_task_id.to_string()).or_insert_with(|| TaskRuntimeMetrics {
            compiled_task_id: compiled_task_id.to_string(),
            operator_kind: operator_kind.clone(),
            ..Default::default()
        });
        row.operator_kind = operator_kind.clone();
        row.max_observed_peak_bytes = max_optional(row.max_observed_peak_bytes, observed_peak_bytes);
        row.max_retained_bytes = max_optional(row.max_retained_bytes, retained_bytes);

        let operator = self
            .task_operator_metrics
            .entry(operator_kind.clone())
            .or_insert_with(|| TaskOperatorMetrics { operator_kind, ..Default::default() });
        operator.max_observed_peak_bytes = max_optional(operator.max_observed_peak_bytes, observed_peak_bytes);
        operator.max_retained_bytes = max_optional(operator.max_retained_bytes, retained_bytes);
    }

    /// Records that a source chunk was planned.
    pub fn record_source_chunk_planned(&mut self, row_groups: u64) {
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(1);
        self.row_groups_planned = self.row_groups_planned.saturating_add(row_groups);
    }

    /// Records one source chunk attempt.
    pub fn record_source_chunk_attempt(&mut self) {
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(1);
    }

    /// Records one committed source chunk and row-group count.
    pub fn record_source_chunk_commit(&mut self, row_groups: u64) {
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(1);
        self.row_groups_committed = self.row_groups_committed.saturating_add(row_groups);
    }

    /// Records one source chunk retry.
    pub fn record_source_chunk_retry(&mut self) {
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(1);
    }

    /// Records a started source-read prefetch.
    pub fn record_source_read_prefetch_started(&mut self) {
        self.source_read_prefetches_started = self.source_read_prefetches_started.saturating_add(1);
    }

    /// Records a completed source-read prefetch.
    pub fn record_source_read_prefetch_completed(&mut self) {
        self.source_read_prefetches_completed = self.source_read_prefetches_completed.saturating_add(1);
    }

    /// Records a skipped source-read prefetch.
    pub fn record_source_read_prefetch_skipped(&mut self) {
        self.source_read_prefetches_skipped = self.source_read_prefetches_skipped.saturating_add(1);
    }

    /// Records an abandoned source-read prefetch.
    pub fn record_source_read_prefetch_abandoned(&mut self) {
        self.source_read_prefetches_abandoned = self.source_read_prefetches_abandoned.saturating_add(1);
    }

    /// Adds source-read queue wait time.
    pub fn record_source_read_queue_wait(&mut self, elapsed: Duration) {
        self.source_read_queue_wait_time += elapsed;
    }

    /// Records a source cache hit and retained byte peak.
    pub fn record_source_cache_hit(&mut self, retained_bytes: u64) {
        self.source_cache_hits = self.source_cache_hits.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    /// Records a source cache miss and retained byte peak.
    pub fn record_source_cache_miss(&mut self, retained_bytes: u64) {
        self.source_cache_misses = self.source_cache_misses.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    /// Records a source cache insertion and retained byte peak.
    pub fn record_source_cache_insert(&mut self, retained_bytes: u64) {
        self.source_cache_inserts = self.source_cache_inserts.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    /// Records a skipped source cache insertion and retained byte peak.
    pub fn record_source_cache_insert_skipped(&mut self, retained_bytes: u64) {
        self.source_cache_insert_skipped = self.source_cache_insert_skipped.saturating_add(1);
        self.record_source_cache_retained_bytes(retained_bytes);
    }

    /// Adds source cache lookup time.
    pub fn record_source_cache_lookup_time(&mut self, elapsed: Duration) {
        self.source_cache_lookup_time += elapsed;
    }

    /// Records one object-store SourceFetch payload.
    pub fn record_source_fetch(&mut self, bytes: u64, elapsed: Duration) {
        self.source_fetch_count = self.source_fetch_count.saturating_add(1);
        self.source_fetch_bytes = self.source_fetch_bytes.saturating_add(bytes);
        self.source_fetch_time += elapsed;
    }

    /// Adds time spent waiting on host-buffer byte budget.
    pub fn record_host_fetch_budget_wait(&mut self, elapsed: Duration) {
        self.host_fetch_budget_wait_time += elapsed;
    }

    /// Adds time spent waiting on the SourceFetch concurrency gate.
    pub fn record_source_fetch_concurrency_wait(&mut self, elapsed: Duration) {
        self.source_fetch_concurrency_wait_time += elapsed;
    }

    /// Adds time spent decoding host-buffer Parquet bytes.
    pub fn record_source_decode_time(&mut self, elapsed: Duration) {
        self.source_decode_time += elapsed;
    }

    /// Updates the peak retained source-cache bytes.
    pub fn record_source_cache_retained_bytes(&mut self, retained_bytes: u64) {
        self.source_cache_retained_bytes = self.source_cache_retained_bytes.max(retained_bytes);
    }

    /// Records one retry caused by resource exhaustion.
    pub fn record_retry_resource_exhausted(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_resource_exhausted_count = self.retry_resource_exhausted_count.saturating_add(1);
    }

    /// Records one retry caused by reservation failure.
    pub fn record_retry_reservation_failure(&mut self) {
        self.retry_count = self.retry_count.saturating_add(1);
        self.retry_reservation_failure_count = self.retry_reservation_failure_count.saturating_add(1);
    }

    /// Records one query-scoped hash-join cache hit.
    pub fn record_hash_join_cache_hit(&mut self) {
        self.hash_join_cache_hits = self.hash_join_cache_hits.saturating_add(1);
    }

    /// Records one query-scoped hash-join cache miss.
    pub fn record_hash_join_cache_miss(&mut self) {
        self.hash_join_cache_misses = self.hash_join_cache_misses.saturating_add(1);
    }

    /// Records one durable hash-join cache hit.
    pub fn record_hash_join_durable_cache_hit(&mut self) {
        self.hash_join_durable_cache_hits = self.hash_join_durable_cache_hits.saturating_add(1);
    }

    /// Records one durable hash-join cache miss.
    pub fn record_hash_join_durable_cache_miss(&mut self) {
        self.hash_join_durable_cache_misses = self.hash_join_durable_cache_misses.saturating_add(1);
    }

    /// Records one durable hash-join cache insertion.
    pub fn record_hash_join_durable_cache_insert(&mut self) {
        self.hash_join_durable_cache_inserts = self.hash_join_durable_cache_inserts.saturating_add(1);
    }

    /// Records one reusable hash-join build.
    pub fn record_hash_join_build(&mut self) {
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(1);
    }

    /// Records one reusable filtered-join build.
    pub fn record_filtered_join_build(&mut self) {
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(1);
    }

    /// Records one one-shot join execution.
    pub fn record_one_shot_join(&mut self) {
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(1);
    }

    /// Adds retained hash-join hash table bytes.
    pub fn record_hash_join_hash_table_bytes(&mut self, bytes: u64) {
        self.hash_join_hash_table_bytes = self.hash_join_hash_table_bytes.saturating_add(bytes);
    }

    /// Adds retained filtered-join hash table bytes.
    pub fn record_filtered_join_hash_table_bytes(&mut self, bytes: u64) {
        self.filtered_join_hash_table_bytes = self.filtered_join_hash_table_bytes.saturating_add(bytes);
    }

    /// Adds retained filtered-join probe-state bytes.
    pub fn record_filtered_join_probe_state_bytes(&mut self, bytes: u64) {
        self.filtered_join_probe_state_bytes = self.filtered_join_probe_state_bytes.saturating_add(bytes);
    }

    /// Adds retained mixed-join size-data bytes.
    pub fn record_mixed_join_size_data_bytes(&mut self, bytes: u64) {
        self.mixed_join_size_data_bytes = self.mixed_join_size_data_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join index-map bytes.
    pub fn record_one_shot_join_index_bytes(&mut self, bytes: u64) {
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join output materialization reservation bytes.
    pub fn record_one_shot_join_materialization_bytes(&mut self, bytes: u64) {
        self.one_shot_join_materialization_bytes = self.one_shot_join_materialization_bytes.saturating_add(bytes);
    }

    /// Adds one-shot join output bytes.
    pub fn record_one_shot_join_output_bytes(&mut self, bytes: u64) {
        self.one_shot_join_output_bytes = self.one_shot_join_output_bytes.saturating_add(bytes);
    }

    /// Records one compiled-pipeline execution attempt.
    pub fn record_compiled_pipeline_attempt(&mut self) {
        self.compiled_pipeline_attempts = self.compiled_pipeline_attempts.saturating_add(1);
    }

    /// Adds completed compiled-pipeline task count.
    pub fn record_compiled_pipeline_tasks_executed(&mut self, tasks: u64) {
        self.compiled_pipeline_tasks_executed = self.compiled_pipeline_tasks_executed.saturating_add(tasks);
    }

    /// Records one compiled-pipeline Host Arrow sink.
    pub fn record_compiled_pipeline_host_arrow_sink(&mut self) {
        self.compiled_pipeline_host_arrow_sinks = self.compiled_pipeline_host_arrow_sinks.saturating_add(1);
    }

    /// Adds adaptive runtime action counters.
    pub fn record_adaptive_runtime_action_counts(&mut self, retry_count: u64, spill_count: u64, downgrade_count: u64) {
        self.adaptive_retry_count = self.adaptive_retry_count.saturating_add(retry_count);
        self.adaptive_spill_count = self.adaptive_spill_count.saturating_add(spill_count);
        self.adaptive_downgrade_count = self.adaptive_downgrade_count.saturating_add(downgrade_count);
    }

    /// Records one adaptive memory-history lookup.
    pub fn record_adaptive_memory_history_lookup(&mut self, hit: bool) {
        self.adaptive_memory_history_lookup_count = self.adaptive_memory_history_lookup_count.saturating_add(1);
        if hit {
            self.adaptive_memory_history_hit_count = self.adaptive_memory_history_hit_count.saturating_add(1);
        } else {
            self.adaptive_memory_history_miss_count = self.adaptive_memory_history_miss_count.saturating_add(1);
        }
    }

    /// Records one partial aggregate task summary.
    pub fn record_partial_aggregate(&mut self, input_rows: u64, output_rows: u64, retained_bytes: u64) {
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(1);
        self.partial_aggregate_input_rows = self.partial_aggregate_input_rows.saturating_add(input_rows);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(output_rows);
        self.partial_aggregate_retained_bytes = self.partial_aggregate_retained_bytes.max(retained_bytes);
    }

    /// Records one final merge attempt and fan-in.
    pub fn record_final_merge_attempt(&mut self, merge_fan_in: usize) {
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(1);
        self.final_merge_max_fan_in = self.final_merge_max_fan_in.max(merge_fan_in as u64);
    }

    /// Records one final merge level.
    pub fn record_final_merge_level(&mut self) {
        self.final_merge_levels = self.final_merge_levels.saturating_add(1);
    }

    /// Merges another query metrics value into this one.
    pub fn merge_from(&mut self, other: QueryMetrics) {
        self.boundary.merge_from(other.boundary);
        self.operators.extend(other.operators);
        self.graph_algorithm_runtime_rows.extend(other.graph_algorithm_runtime_rows);
        self.join_strategy_decisions.extend(other.join_strategy_decisions);
        self.aggregate_strategy_decisions.extend(other.aggregate_strategy_decisions);
        self.dynamic_filter_runtime_rows.extend(other.dynamic_filter_runtime_rows);
        self.source_read_runtime_rows.extend(other.source_read_runtime_rows);
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
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(other.reservation_live_peak_bytes);
        self.reservation_denied_count = self.reservation_denied_count.saturating_add(other.reservation_denied_count);
        self.max_task_reservation_bytes = self.max_task_reservation_bytes.max(other.max_task_reservation_bytes);
        self.max_task_observed_peak_bytes =
            max_optional(self.max_task_observed_peak_bytes, other.max_task_observed_peak_bytes);
        self.max_task_retained_bytes = max_optional(self.max_task_retained_bytes, other.max_task_retained_bytes);
        self.task_kind_by_compiled_task_id.extend(other.task_kind_by_compiled_task_id);
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
        self.source_fetch_bytes = self.source_fetch_bytes.saturating_add(other.source_fetch_bytes);
        self.source_fetch_count = self.source_fetch_count.saturating_add(other.source_fetch_count);
        self.host_fetch_budget_wait_time += other.host_fetch_budget_wait_time;
        self.source_fetch_concurrency_wait_time += other.source_fetch_concurrency_wait_time;
        self.source_fetch_time += other.source_fetch_time;
        self.source_decode_time += other.source_decode_time;
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
    }
}

/// Source-read footprint before or after applying a dynamic filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynamicFilterSourceSnapshot {
    /// Planned row groups visible to the source consumer.
    pub row_groups: u64,
    /// Estimated source-read bytes, when known.
    pub source_bytes: Option<u64>,
}

/// Runtime diagnostic row for one dynamic-filter route event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicFilterRuntimeMetrics {
    /// Plan-local dynamic-filter route id.
    pub dynamic_filter_id: String,
    /// Event kind: `produced`, `skipped`, `applied`, `reader_predicate_rows`,
    /// `applied_post_decode`, or `post_decode_rows`.
    pub event: String,
    /// Producer kind label.
    pub producer_kind: String,
    /// Producer plan node id.
    pub producer_node_id: String,
    /// Consumer kind label.
    pub consumer_kind: String,
    /// Consumer plan node id.
    pub consumer_node_id: String,
    /// Consumer lowering label selected during route discovery.
    pub consumer_lowering: String,
    /// Stage where the consumer can first reduce work.
    pub consumer_reduction_stage: String,
    /// Payload kind, or `none` for skipped events.
    pub filter_kind: String,
    /// Exact payload value count, when available.
    pub payload_value_count: Option<u64>,
    /// Inclusive integer-like minimum, when the payload carries one.
    pub payload_min_i64: Option<i64>,
    /// Inclusive integer-like maximum, when the payload carries one.
    pub payload_max_i64: Option<i64>,
    /// Producer output rows observed while creating the payload, when known.
    pub producer_rows: Option<u64>,
    /// Stable reason code for skipped events.
    pub skipped_reason: Option<String>,
    /// Source row groups before application.
    pub selected_row_groups_before: Option<u64>,
    /// Source row groups after application.
    pub selected_row_groups_after: Option<u64>,
    /// Source-read bytes before application.
    pub selected_source_bytes_before: Option<u64>,
    /// Source-read bytes after application.
    pub selected_source_bytes_after: Option<u64>,
    /// Source file index for source-reader row evidence.
    pub source_file_index: Option<u64>,
    /// Source chunk index for source-reader row evidence.
    pub source_chunk_index: Option<u64>,
    /// Selected source rows before the reader predicate was applied.
    pub reader_predicate_rows_before: Option<u64>,
    /// Materialized source rows after the reader predicate was applied.
    pub reader_predicate_rows_after: Option<u64>,
    /// Source rows removed by the reader predicate.
    pub reader_predicate_rows_filtered: u64,
    /// Materialized rows before a post-decode dynamic-filter stage.
    pub post_decode_rows_before: Option<u64>,
    /// Materialized rows after a post-decode dynamic-filter stage.
    pub post_decode_rows_after: Option<u64>,
    /// Materialized rows removed by a post-decode dynamic-filter stage.
    pub post_decode_rows_filtered: u64,
    /// Row groups removed by this application.
    pub row_groups_pruned: u64,
    /// Source-read bytes removed by this application.
    pub source_bytes_pruned: u64,
}

impl DynamicFilterRuntimeMetrics {
    fn produced(route: &DynamicFilterRoute, filter: &DynamicFilter, producer_rows: u64) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        Self {
            dynamic_filter_id: route.id.to_string(),
            event: "produced".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: Some(producer_rows),
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn skipped(route: &DynamicFilterRoute, reason: impl Into<String>) -> Self {
        Self {
            dynamic_filter_id: route.id.to_string(),
            event: "skipped".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: "none".to_string(),
            payload_value_count: None,
            payload_min_i64: None,
            payload_max_i64: None,
            producer_rows: None,
            skipped_reason: Some(reason.into()),
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn applied(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        before: DynamicFilterSourceSnapshot,
        after: DynamicFilterSourceSnapshot,
        row_groups_pruned: u64,
        source_bytes_pruned: u64,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        Self {
            dynamic_filter_id: route.id.to_string(),
            event: "applied".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: Some(before.row_groups),
            selected_row_groups_after: Some(after.row_groups),
            selected_source_bytes_before: before.source_bytes,
            selected_source_bytes_after: after.source_bytes,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned,
            source_bytes_pruned,
        }
    }

    fn source_reader_predicate_rows(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        source_file_index: Option<usize>,
        source_chunk_index: Option<usize>,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        let rows_filtered = rows_before.zip(rows_after).map_or(0, |(before, after)| before.saturating_sub(after));
        Self {
            dynamic_filter_id: route.id.to_string(),
            event: "reader_predicate_rows".to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.as_str().to_string(),
            consumer_reduction_stage: route.consumer_reduction_stage().to_string(),
            filter_kind: filter.kind().as_str().to_string(),
            payload_value_count: filter.value_count().map(|value| value as u64),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: source_file_index.map(|index| index as u64),
            source_chunk_index: source_chunk_index.map(|index| index as u64),
            reader_predicate_rows_before: rows_before,
            reader_predicate_rows_after: rows_after,
            reader_predicate_rows_filtered: rows_filtered,
            post_decode_rows_before: None,
            post_decode_rows_after: None,
            post_decode_rows_filtered: 0,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    fn applied_post_decode(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event(route, filter, "applied_post_decode", rows_before, rows_after)
    }

    fn source_post_decode_stage(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event_with_stage(
            route,
            filter,
            "applied_post_decode",
            DynamicFilterConsumerReductionStage::SourcePostDecode,
            rows_before,
            rows_after,
        )
    }

    fn post_decode_rows(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event(route, filter, "post_decode_rows", rows_before, rows_after)
    }

    fn post_decode_event(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        event: &'static str,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        Self::post_decode_event_with_stage(
            route,
            filter,
            event,
            route.consumer_reduction_stage(),
            rows_before,
            rows_after,
        )
    }

    fn post_decode_event_with_stage(
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        event: &'static str,
        consumer_reduction_stage: DynamicFilterConsumerReductionStage,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) -> Self {
        let (payload_min_i64, payload_max_i64) =
            filter.i64_range().map_or((None, None), |(min, max)| (Some(min), Some(max)));
        let rows_filtered = rows_before.zip(rows_after).map_or(0, |(before, after)| before.saturating_sub(after));
        Self {
            dynamic_filter_id: route.id.to_string(),
            event: event.to_string(),
            producer_kind: route.producer.kind().to_string(),
            producer_node_id: route.producer.node_id().to_string(),
            consumer_kind: route.consumer.kind().to_string(),
            consumer_node_id: route.consumer.node_id().to_string(),
            consumer_lowering: route.consumer_lowering.to_string(),
            consumer_reduction_stage: consumer_reduction_stage.to_string(),
            filter_kind: filter.kind().to_string(),
            payload_value_count: filter.value_count().and_then(|value| u64::try_from(value).ok()),
            payload_min_i64,
            payload_max_i64,
            producer_rows: None,
            skipped_reason: None,
            selected_row_groups_before: None,
            selected_row_groups_after: None,
            selected_source_bytes_before: None,
            selected_source_bytes_after: None,
            source_file_index: None,
            source_chunk_index: None,
            reader_predicate_rows_before: None,
            reader_predicate_rows_after: None,
            reader_predicate_rows_filtered: 0,
            post_decode_rows_before: rows_before,
            post_decode_rows_after: rows_after,
            post_decode_rows_filtered: rows_filtered,
            row_groups_pruned: 0,
            source_bytes_pruned: 0,
        }
    }

    /// Serializes this row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "dynamic_filter_id={},event={},producer_kind={},producer_node_id={},consumer_kind={},consumer_node_id={},consumer_lowering={},consumer_reduction_stage={},filter_kind={},payload_value_count={},payload_min_i64={},payload_max_i64={},producer_rows={},skipped_reason={},selected_row_groups_before={},selected_row_groups_after={},selected_source_bytes_before={},selected_source_bytes_after={},source_file_index={},source_chunk_index={},reader_predicate_rows_before={},reader_predicate_rows_after={},reader_predicate_rows_filtered={},post_decode_rows_before={},post_decode_rows_after={},post_decode_rows_filtered={},row_groups_pruned={},source_bytes_pruned={}",
            self.dynamic_filter_id,
            self.event,
            self.producer_kind,
            self.producer_node_id,
            self.consumer_kind,
            self.consumer_node_id,
            self.consumer_lowering,
            self.consumer_reduction_stage,
            self.filter_kind,
            optional_u64(self.payload_value_count),
            optional_i64(self.payload_min_i64),
            optional_i64(self.payload_max_i64),
            optional_u64(self.producer_rows),
            optional_str(self.skipped_reason.as_deref()),
            optional_u64(self.selected_row_groups_before),
            optional_u64(self.selected_row_groups_after),
            optional_u64(self.selected_source_bytes_before),
            optional_u64(self.selected_source_bytes_after),
            optional_u64(self.source_file_index),
            optional_u64(self.source_chunk_index),
            optional_u64(self.reader_predicate_rows_before),
            optional_u64(self.reader_predicate_rows_after),
            self.reader_predicate_rows_filtered,
            optional_u64(self.post_decode_rows_before),
            optional_u64(self.post_decode_rows_after),
            self.post_decode_rows_filtered,
            self.row_groups_pruned,
            self.source_bytes_pruned,
        )
    }
}

/// Runtime evidence for one local-Parquet source read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceReadRuntimeMetrics {
    /// Native source node id.
    pub source_node_id: String,
    /// Source read shape, for example `chunk` or `chunk_batch`.
    pub read_kind: String,
    /// Parquet read strategy used by the chunk or batch.
    pub read_strategy: String,
    /// Batch index within the source stream, or the chunk index for singleton reads.
    pub batch_index: u64,
    /// Number of source chunks represented by this read.
    pub chunk_count: u64,
    /// First source chunk index represented by this read.
    pub first_chunk_index: u64,
    /// Last source chunk index represented by this read.
    pub last_chunk_index: u64,
    /// First source file index represented by this read.
    pub first_file_index: u64,
    /// Last source file index represented by this read.
    pub last_file_index: u64,
    /// Number of distinct files represented by this read.
    pub file_count: u64,
    /// Selected row groups represented by this read.
    pub row_group_count: u64,
    /// Number of physical read columns per represented file.
    pub read_columns_per_file: u64,
    /// Sum of physical read-column selections across represented files.
    pub read_columns_total: u64,
    /// Sum of known file bytes for represented files, when available.
    pub known_file_bytes: u64,
    /// Best available source-read byte estimate for represented chunks.
    pub estimated_source_read_bytes: u64,
    /// Sum of known selected row-group bytes, when available.
    pub known_row_group_bytes: u64,
    /// Sum of known projected compressed bytes, when available.
    pub known_projected_compressed_bytes: u64,
    /// Sum of known projected uncompressed bytes, when available.
    pub known_projected_uncompressed_bytes: u64,
    /// Sum of known pure-filter compressed bytes, when available.
    pub known_pure_filter_compressed_bytes: u64,
    /// Sum of known pure-filter uncompressed bytes, when available.
    pub known_pure_filter_uncompressed_bytes: u64,
    /// Number of output rows materialized by the read after reader predicates.
    pub output_rows: u64,
    /// Whether this read carried a source reader predicate.
    pub reader_predicate: bool,
    /// Number of dynamic-filter reader predicate routes attached to this read.
    pub reader_dynamic_filter_count: u64,
    /// Stable diagnostic reason for the source chunk batching decision.
    pub source_batching_reason: String,
    /// Source chunk planning mode requested for this source read.
    pub source_chunk_planning_mode: String,
    /// Stable diagnostic reason for the source chunk row-group and byte target.
    pub source_chunk_planning_reason: String,
    /// Elapsed time spent in the source read.
    pub elapsed: Duration,
}

impl SourceReadRuntimeMetrics {
    /// Serializes this row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "source_node_id={},read_kind={},read_strategy={},batch_index={},chunk_count={},first_chunk_index={},last_chunk_index={},first_file_index={},last_file_index={},file_count={},row_group_count={},read_columns_per_file={},read_columns_total={},known_file_bytes={},estimated_source_read_bytes={},known_row_group_bytes={},known_projected_compressed_bytes={},known_projected_uncompressed_bytes={},known_pure_filter_compressed_bytes={},known_pure_filter_uncompressed_bytes={},output_rows={},reader_predicate={},reader_dynamic_filter_count={},source_batching_reason={},source_chunk_planning_mode={},source_chunk_planning_reason={},elapsed_ns={}",
            self.source_node_id,
            self.read_kind,
            self.read_strategy,
            self.batch_index,
            self.chunk_count,
            self.first_chunk_index,
            self.last_chunk_index,
            self.first_file_index,
            self.last_file_index,
            self.file_count,
            self.row_group_count,
            self.read_columns_per_file,
            self.read_columns_total,
            self.known_file_bytes,
            self.estimated_source_read_bytes,
            self.known_row_group_bytes,
            self.known_projected_compressed_bytes,
            self.known_projected_uncompressed_bytes,
            self.known_pure_filter_compressed_bytes,
            self.known_pure_filter_uncompressed_bytes,
            self.output_rows,
            self.reader_predicate,
            self.reader_dynamic_filter_count,
            self.source_batching_reason,
            self.source_chunk_planning_mode,
            self.source_chunk_planning_reason,
            self.elapsed.as_nanos().min(u128::from(u64::MAX)),
        )
    }
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
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

fn is_legacy_bounded_task_id(task_id: &str) -> bool {
    task_id.starts_with("legacy_bounded.")
}

fn legacy_bounded_operator_kind_from_task_id(task_id: &str) -> Option<&str> {
    task_id.strip_prefix("legacy_bounded.")?.split('.').next()
}

/// Join-strategy decision facts captured for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinStrategyDecisionMetrics {
    /// Native join node id.
    pub node_id: String,
    /// Stable decision fact keys and values.
    pub facts: BTreeMap<String, String>,
}

/// Aggregate-strategy decision facts captured for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AggregateStrategyDecisionMetrics {
    /// Native aggregate node id, or a stage-local fallback label when unavailable.
    pub node_id: String,
    /// Stable decision fact keys and values.
    pub facts: BTreeMap<String, String>,
}

/// Metrics aggregated across tasks of the same operator kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskOperatorMetrics {
    /// Operator-kind label shared by the aggregated tasks.
    pub operator_kind: String,
    /// Number of tasks observed for this operator kind.
    pub task_count: u64,
    /// Cumulative elapsed time across tasks.
    pub elapsed_time: Duration,
    /// Cumulative reservation bytes requested.
    pub reservation_requested_bytes: u64,
    /// Cumulative reservation bytes granted.
    pub reservation_granted_bytes: u64,
    /// Cumulative reservation bytes released.
    pub reservation_released_bytes: u64,
    /// Peak live reserved bytes for this operator kind.
    pub reservation_live_peak_bytes: u64,
    /// Largest lower-level observed allocator peak for this operator kind.
    pub max_observed_peak_bytes: Option<u64>,
    /// Largest lower-level retained byte count for this operator kind.
    pub max_retained_bytes: Option<u64>,
    pub(crate) live_reserved_bytes: u64,
}

impl TaskOperatorMetrics {
    fn record_reservation_event(&mut self, event: &MemoryReservationEvent) {
        if matches!(event.action, MemoryReservationEventAction::Reserve | MemoryReservationEventAction::Grow) {
            self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(event.requested_bytes);
        }
        if event.outcome != MemoryReservationEventOutcome::Denied && event.charged_delta_bytes > 0 {
            let granted = u64::try_from(event.charged_delta_bytes).unwrap_or(u64::MAX);
            self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted);
        }
        if event.charged_delta_bytes < 0 {
            let released = u64::try_from(-event.charged_delta_bytes).unwrap_or(u64::MAX);
            self.reservation_released_bytes = self.reservation_released_bytes.saturating_add(released);
        }
    }

    fn merge_from(&mut self, other: Self) {
        self.task_count = self.task_count.saturating_add(other.task_count);
        self.elapsed_time += other.elapsed_time;
        self.reservation_requested_bytes =
            self.reservation_requested_bytes.saturating_add(other.reservation_requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(other.reservation_granted_bytes);
        self.reservation_released_bytes =
            self.reservation_released_bytes.saturating_add(other.reservation_released_bytes);
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(other.reservation_live_peak_bytes);
        self.max_observed_peak_bytes = max_optional(self.max_observed_peak_bytes, other.max_observed_peak_bytes);
        self.max_retained_bytes = max_optional(self.max_retained_bytes, other.max_retained_bytes);
        self.live_reserved_bytes = self.live_reserved_bytes.saturating_add(other.live_reserved_bytes);
    }
}

/// Metrics for one compiled pipeline task.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskRuntimeMetrics {
    /// Stable compiled task id.
    pub compiled_task_id: String,
    /// Operator-kind label for the task.
    pub operator_kind: String,
    /// Cumulative elapsed task time.
    pub elapsed_time: Duration,
    /// Cumulative reservation bytes requested by the task.
    pub reservation_requested_bytes: u64,
    /// Cumulative reservation bytes granted to the task.
    pub reservation_granted_bytes: u64,
    /// Cumulative reservation bytes released by the task.
    pub reservation_released_bytes: u64,
    /// Peak live reserved bytes for the task.
    pub reservation_live_peak_bytes: u64,
    /// Largest lower-level observed allocator peak for the task.
    pub max_observed_peak_bytes: Option<u64>,
    /// Largest lower-level retained byte count for the task.
    pub max_retained_bytes: Option<u64>,
    pub(crate) live_reserved_bytes: u64,
}

impl TaskRuntimeMetrics {
    fn record_reservation_event(&mut self, event: &MemoryReservationEvent) {
        if matches!(event.action, MemoryReservationEventAction::Reserve | MemoryReservationEventAction::Grow) {
            self.reservation_requested_bytes = self.reservation_requested_bytes.saturating_add(event.requested_bytes);
        }
        if event.outcome != MemoryReservationEventOutcome::Denied && event.charged_delta_bytes > 0 {
            let granted = u64::try_from(event.charged_delta_bytes).unwrap_or(u64::MAX);
            self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(granted);
        }
        if event.charged_delta_bytes < 0 {
            let released = u64::try_from(-event.charged_delta_bytes).unwrap_or(u64::MAX);
            self.reservation_released_bytes = self.reservation_released_bytes.saturating_add(released);
        }
        if event.charged_delta_bytes >= 0 {
            self.live_reserved_bytes =
                self.live_reserved_bytes.saturating_add(u64::try_from(event.charged_delta_bytes).unwrap_or(u64::MAX));
        } else {
            self.live_reserved_bytes =
                self.live_reserved_bytes.saturating_sub(u64::try_from(-event.charged_delta_bytes).unwrap_or(u64::MAX));
        }
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(self.live_reserved_bytes);
    }

    fn merge_from(&mut self, other: Self) {
        if self.operator_kind.is_empty() || self.operator_kind == "unknown" {
            self.operator_kind = other.operator_kind;
        }
        self.elapsed_time += other.elapsed_time;
        self.reservation_requested_bytes =
            self.reservation_requested_bytes.saturating_add(other.reservation_requested_bytes);
        self.reservation_granted_bytes = self.reservation_granted_bytes.saturating_add(other.reservation_granted_bytes);
        self.reservation_released_bytes =
            self.reservation_released_bytes.saturating_add(other.reservation_released_bytes);
        self.reservation_live_peak_bytes = self.reservation_live_peak_bytes.max(other.reservation_live_peak_bytes);
        self.max_observed_peak_bytes = max_optional(self.max_observed_peak_bytes, other.max_observed_peak_bytes);
        self.max_retained_bytes = max_optional(self.max_retained_bytes, other.max_retained_bytes);
        self.live_reserved_bytes = self.live_reserved_bytes.saturating_add(other.live_reserved_bytes);
    }
}

/// Successful lower-level execution-budget report captured as a metrics row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetedExecutionEvent {
    /// Native operation label that produced the report.
    pub operation: String,
    /// Lower-level budget boundary label.
    pub boundary: String,
    /// Budget limit in bytes.
    pub budget_bytes: u64,
    /// Lower-level budget enforcement label.
    pub enforcement: String,
    /// Lower-level budget status label.
    pub status: String,
    /// Allocator high-water bytes reported by lower layers.
    pub allocator_high_water_bytes: Option<u64>,
    /// Retained bytes reported by lower layers.
    pub retained_bytes: Option<u64>,
    /// Output bytes reported by lower layers.
    pub output_bytes: Option<u64>,
    /// Output rows reported by lower layers.
    pub output_rows: Option<u64>,
    /// Unresolved contract label when the lower layer reports one.
    pub unresolved_contract: Option<String>,
    /// Additional lower-level context facts.
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

/// Generic operator metric row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperatorMetrics {
    /// Native node id.
    pub node_id: String,
    /// Operator-kind label.
    pub operator_kind: String,
    /// Input rows consumed by the operator.
    pub input_rows: u64,
    /// Input bytes consumed by the operator.
    pub input_bytes: u64,
    /// Output rows produced by the operator.
    pub output_rows: u64,
    /// Output bytes produced by the operator.
    pub output_bytes: u64,
    /// Join-specific metrics when this is a join row.
    pub join: Option<JoinOperatorMetrics>,
}

impl OperatorMetrics {
    /// Creates a join operator metric row.
    pub fn join(node_id: impl ToString, join: JoinOperatorMetrics) -> Self {
        Self { node_id: node_id.to_string(), operator_kind: "join".to_string(), join: Some(join), ..Default::default() }
    }
}

/// cuGraph algorithm runtime metric row.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphAlgorithmRuntimeMetrics {
    /// Native graph algorithm node id.
    pub node_id: String,
    /// Stable graph algorithm label.
    pub algorithm: String,
    /// Runtime outcome label.
    pub runtime_outcome: String,
    /// Input boundary label.
    pub input_boundary: String,
    /// Output boundary label.
    pub output_boundary: String,
    /// Graph construction policy label.
    pub construction_policy: String,
    /// Source vertex column name.
    pub source_column: String,
    /// Destination vertex column name.
    pub destination_column: String,
    /// Optional edge-weight column name.
    pub weight_column: Option<String>,
    /// Whether the graph was treated as directed.
    pub directed: bool,
    /// Whether cuGraph renumbering was enabled.
    pub renumber: bool,
    /// Stable serialized algorithm parameters.
    pub algorithm_parameters: String,
    /// Stable serialized algorithm metadata.
    pub algorithm_metadata: String,
    /// Input row count.
    pub input_rows: u64,
    /// Input column count.
    pub input_columns: u64,
    /// Output row count.
    pub output_rows: u64,
    /// Output column count.
    pub output_columns: u64,
    /// Output device bytes when known.
    pub output_device_bytes: Option<u64>,
    /// Runtime elapsed nanoseconds.
    pub elapsed_ns: u64,
    /// GPU-resident input handoffs performed for this algorithm.
    pub gpu_resident_input_handoffs: u64,
    /// Host-to-device uploads performed for this algorithm.
    pub host_to_device_uploads: u64,
    /// Device-to-host materializations performed for this algorithm.
    pub device_to_host_materializations: u64,
}

/// Join-specific operator metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinOperatorMetrics {
    /// Join type label.
    pub join_type: String,
    /// Execution strategy label.
    pub execution_strategy: String,
    /// Number of equality key pairs.
    pub key_count: usize,
    /// Whether a residual predicate was present.
    pub residual_predicate: bool,
    /// Null-equality policy label.
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
    /// Build-side input row count.
    pub build_rows: u64,
    /// Build-side input bytes.
    pub build_bytes: u64,
    /// Probe-side input row count.
    pub probe_rows: u64,
    /// Probe-side input bytes.
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
    /// Bytes reserved for join output materialization.
    pub output_reservation_bytes: u64,
    /// Join output row count.
    pub output_rows: u64,
    /// Join output bytes.
    pub output_bytes: u64,
}

/// Host/device boundary transfer metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundaryMetrics {
    /// Number of GPU-resident input handoffs.
    pub gpu_resident_input_handoffs: u64,
    /// Number of host-to-device uploads.
    pub host_to_device_uploads: u64,
    /// Rows uploaded from host to device.
    pub host_to_device_uploaded_rows: u64,
    /// Bytes uploaded from host to device.
    pub host_to_device_uploaded_bytes: u64,
    /// Number of device-to-host materializations.
    pub device_to_host_materializations: u64,
    /// Rows materialized from device to host.
    pub device_to_host_materialized_rows: u64,
    /// Bytes materialized from device to host.
    pub device_to_host_materialized_bytes: u64,
    /// Number of device-to-device copies.
    pub device_to_device_copies: u64,
    /// Bytes copied device-to-device.
    pub device_to_device_copied_bytes: u64,
}

impl BoundaryMetrics {
    /// Records one GPU-resident input handoff.
    pub fn record_gpu_resident_input_handoff(&mut self) {
        self.gpu_resident_input_handoffs += 1;
    }

    /// Records one host-to-device upload.
    pub fn record_host_to_device_upload(&mut self, rows: u64, bytes: u64) {
        self.host_to_device_uploads += 1;
        self.host_to_device_uploaded_rows += rows;
        self.host_to_device_uploaded_bytes += bytes;
    }

    /// Records one device-to-host materialization.
    pub fn record_device_to_host_materialization(&mut self, rows: u64, bytes: u64) {
        self.device_to_host_materializations += 1;
        self.device_to_host_materialized_rows += rows;
        self.device_to_host_materialized_bytes += bytes;
    }

    /// Records one device-to-device copy.
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
