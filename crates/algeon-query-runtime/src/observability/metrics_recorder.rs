//! The only way execution code reports query metrics.
//!
//! A `MetricsRecorder` holds no metrics. Each method is one typed fact sent to
//! the plan execution's observation owner, which is the single store; a
//! `QueryMetrics` value is only ever a projection of that store. Method names
//! match the `QueryMetrics` reductions they drive, so a call site reads the same
//! as before while no longer owning the value it reports into.

use std::{collections::BTreeMap, time::Duration};

use cudf::ExecutionBudgetReport;
use query_planning::{
    dynamic_filter::{DynamicFilter, DynamicFilterRoute},
    graph_observation::GraphMeasuredPeakProvenance,
    plan::{GraphAlgorithm, NodeId},
};

use super::{
    BudgetWindow, BudgetWindowPeaks, PlanExecutionId, QueryMetricsCheckpoint, RuntimeObservationConfig,
    RuntimeObservationHandle, RuntimeObservationSession, RuntimeQueryId, RuntimeTraceId, SourceReadFact,
};
use crate::metrics::{
    BoundaryMetrics, DynamicFilterSourceSnapshot, GraphAlgorithmRuntimeMetrics, OperatorMetrics, PlanNodeId,
    QueryMetrics, SourceReadRuntimeMetrics,
};

/// Records one plan execution's metrics into its observation owner.
#[derive(Debug, Clone)]
pub struct MetricsRecorder {
    observation: RuntimeObservationHandle,
    plan_execution_id: PlanExecutionId,
}

impl MetricsRecorder {
    /// Binds a recorder to one plan execution of a query-bound session.
    pub fn new(observation: RuntimeObservationHandle, plan_execution_id: PlanExecutionId) -> Self {
        Self { observation, plan_execution_id }
    }

    /// Binds a recorder to one plan execution, owning the metrics locally when
    /// the caller's observation has no owner.
    ///
    /// A context built without an observation session still produces metrics:
    /// its plan execution gets a private owner with the native retention
    /// profile. That owner records metrics only; the caller's trace stays off.
    pub fn bind(
        observation: &RuntimeObservationHandle,
        plan_execution_id: PlanExecutionId,
        query_id: &str,
        trace_id: Option<&str>,
    ) -> Self {
        let observation = if observation.has_owner() {
            observation.clone()
        } else {
            RuntimeObservationSession::new(
                RuntimeQueryId::new(query_id),
                trace_id.map(RuntimeTraceId::new),
                RuntimeObservationConfig::native_execution(),
            )
            .handle()
        };
        Self::new(observation, plan_execution_id)
    }

    /// The plan execution this recorder reports into.
    pub const fn plan_execution_id(&self) -> &PlanExecutionId {
        &self.plan_execution_id
    }

    /// One internally consistent view of this plan execution's metrics.
    pub fn checkpoint(&self) -> Option<QueryMetricsCheckpoint> {
        self.observation.metrics_checkpoint(&self.plan_execution_id)
    }

    /// Source chunk and partial aggregate counts so far, for diagnostic facts.
    /// Never an input to execution decisions.
    pub fn progress(&self) -> ExecutionProgress {
        self.observation.execution_progress(&self.plan_execution_id)
    }

    /// The data-boundary counters so far, for reporting one operation's share.
    /// Never an input to execution decisions.
    pub fn boundary(&self) -> BoundaryMetrics {
        self.observation.boundary_metrics(&self.plan_execution_id)
    }

    fn apply(&self, reduce: impl FnOnce(&mut QueryMetrics)) {
        self.observation.apply_metrics(&self.plan_execution_id, reduce);
    }

    fn source(&self, fact: SourceReadFact) {
        self.observation.record_source_read(&self.plan_execution_id, fact);
    }

    // ---- source reads --------------------------------------------------

    /// Records one source-read fact.
    pub fn record_source_read(&self, fact: SourceReadFact) {
        self.source(fact);
    }

    /// Offers one source-read detail row for bounded retention.
    pub fn record_source_read_runtime(&self, row: SourceReadRuntimeMetrics) {
        self.observation.record_source_read_row(&self.plan_execution_id, row);
    }

    /// Records one input batch and its row count.
    pub fn record_input_batch(&self, rows: u64) {
        self.source(SourceReadFact::InputBatch { rows });
    }

    /// Records one Parquet source file observation.
    pub fn record_parquet_source(&self, known_bytes: Option<u64>, selected_row_groups: u64, read_columns: u64) {
        self.source(SourceReadFact::ParquetSource {
            files: 1,
            known_bytes: known_bytes.unwrap_or(0),
            selected_row_groups,
            read_columns,
        });
    }

    /// Records that a source chunk was planned.
    pub fn record_source_chunk_planned(&self, row_groups: u64) {
        self.source(SourceReadFact::ChunkPlanned { row_groups });
    }

    /// Records one source chunk attempt.
    pub fn record_source_chunk_attempt(&self) {
        self.source(SourceReadFact::ChunkAttempted);
    }

    /// Records one committed source chunk and row-group count.
    pub fn record_source_chunk_commit(&self, row_groups: u64) {
        self.source(SourceReadFact::ChunkCommitted { chunk_count: 1, row_groups });
    }

    /// Records committed source chunks and row-group count.
    pub fn record_source_chunk_commit_count(&self, chunk_count: u64, row_groups: u64) {
        self.source(SourceReadFact::ChunkCommitted { chunk_count, row_groups });
    }

    /// Records one source chunk retry.
    pub fn record_source_chunk_retry(&self) {
        self.source(SourceReadFact::ChunkRetried);
    }

    /// Records a started source-read prefetch.
    pub fn record_source_read_prefetch_started(&self) {
        self.source(SourceReadFact::PrefetchStarted);
    }

    /// Records a completed source-read prefetch.
    pub fn record_source_read_prefetch_completed(&self) {
        self.source(SourceReadFact::PrefetchCompleted);
    }

    /// Records a skipped source-read prefetch.
    pub fn record_source_read_prefetch_skipped(&self) {
        self.source(SourceReadFact::PrefetchSkipped);
    }

    /// Records an abandoned source-read prefetch.
    pub fn record_source_read_prefetch_abandoned(&self) {
        self.source(SourceReadFact::PrefetchAbandoned);
    }

    /// Adds source-read queue wait time.
    pub fn record_source_read_queue_wait(&self, elapsed: Duration) {
        self.source(SourceReadFact::QueueWait(elapsed));
    }

    /// Adds source metadata load time.
    pub fn add_source_metadata_load_time(&self, elapsed: Duration) {
        self.source(SourceReadFact::MetadataLoadTime(elapsed));
    }

    /// Adds time spent inside the Parquet read call.
    pub fn add_parquet_read_time(&self, elapsed: Duration) {
        self.source(SourceReadFact::ParquetReadTime(elapsed));
    }

    /// Adds source-read orchestration time.
    pub fn add_source_read_time(&self, elapsed: Duration) {
        self.source(SourceReadFact::SourceReadTime(elapsed));
    }

    // ---- operator, merge, sort, and materialization time ---------------

    /// Adds native operator time.
    pub fn add_operator_time(&self, elapsed: Duration) {
        self.apply(|metrics| saturating_add_duration(&mut metrics.operator_time, elapsed));
    }

    /// Adds operator time spent inside source chunks.
    pub fn add_chunk_operator_time(&self, elapsed: Duration) {
        self.apply(|metrics| saturating_add_duration(&mut metrics.chunk_operator_time, elapsed));
    }

    /// Adds final-merge time.
    pub fn add_final_merge_time(&self, elapsed: Duration) {
        self.apply(|metrics| saturating_add_duration(&mut metrics.final_merge_time, elapsed));
    }

    /// Adds final-sort time.
    pub fn add_final_sort_time(&self, elapsed: Duration) {
        self.apply(|metrics| saturating_add_duration(&mut metrics.final_sort_time, elapsed));
    }

    /// Adds host materialization time.
    pub fn add_materialization_time(&self, elapsed: Duration) {
        self.apply(|metrics| saturating_add_duration(&mut metrics.materialization_time, elapsed));
    }

    // ---- data boundary -------------------------------------------------

    /// Records a GPU-resident input handoff.
    pub fn record_gpu_resident_input_handoff(&self) {
        self.apply(|metrics| metrics.boundary.record_gpu_resident_input_handoff());
    }

    /// Records a host-to-device upload.
    pub fn record_host_to_device_upload(&self, rows: u64, bytes: u64) {
        self.apply(|metrics| metrics.boundary.record_host_to_device_upload(rows, bytes));
    }

    /// Records a device-to-host materialization.
    pub fn record_device_to_host_materialization(&self, rows: u64, bytes: u64) {
        self.apply(|metrics| metrics.boundary.record_device_to_host_materialization(rows, bytes));
    }

    /// Records a device-to-device copy.
    pub fn record_device_to_device_copy(&self, bytes: u64) {
        self.apply(|metrics| metrics.boundary.record_device_to_device_copy(bytes));
    }

    // ---- memory and operators -----------------------------------------

    /// Records one device memory observation.
    pub fn record_device_memory_observation(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_device_memory_observation(bytes));
    }

    /// Records one operator occurrence.
    pub fn record_operator_metrics(&self, row: OperatorMetrics) {
        self.apply(|metrics| metrics.record_operator_metrics(row));
    }

    // ---- aggregates ------------------------------------------------------

    /// Records one aggregate strategy decision.
    pub fn record_aggregate_strategy_decision(&self, node_id: impl ToString, facts: &BTreeMap<&'static str, String>) {
        let node_id = node_id.to_string();
        let facts = facts.clone();
        self.apply(|metrics| metrics.record_aggregate_strategy_decision(node_id, &facts));
    }

    /// Records one partial aggregate.
    pub fn record_partial_aggregate(&self, input_rows: u64, output_rows: u64, retained_bytes: u64) {
        self.apply(|metrics| metrics.record_partial_aggregate(input_rows, output_rows, retained_bytes));
    }

    /// Records one final-merge attempt.
    pub fn record_final_merge_attempt(&self, merge_fan_in: usize) {
        self.apply(|metrics| metrics.record_final_merge_attempt(merge_fan_in));
    }

    /// Records one final-merge level.
    pub fn record_final_merge_level(&self) {
        self.apply(QueryMetrics::record_final_merge_level);
    }

    // ---- joins -----------------------------------------------------------

    /// Records one join strategy decision.
    pub fn record_join_strategy_decision(&self, node_id: impl ToString, facts: &BTreeMap<&'static str, String>) {
        let node_id = node_id.to_string();
        let facts = facts.clone();
        self.apply(|metrics| metrics.record_join_strategy_decision(node_id, &facts));
    }

    /// Records a hash-join cache hit.
    pub fn record_hash_join_cache_hit(&self) {
        self.apply(QueryMetrics::record_hash_join_cache_hit);
    }

    /// Records a hash-join cache miss.
    pub fn record_hash_join_cache_miss(&self) {
        self.apply(QueryMetrics::record_hash_join_cache_miss);
    }

    /// Records a hash-join build.
    pub fn record_hash_join_build(&self) {
        self.apply(QueryMetrics::record_hash_join_build);
    }

    /// Records a filtered-join build.
    pub fn record_filtered_join_build(&self) {
        self.apply(QueryMetrics::record_filtered_join_build);
    }

    /// Records a one-shot join.
    pub fn record_one_shot_join(&self) {
        self.apply(QueryMetrics::record_one_shot_join);
    }

    /// Records hash-join hash table bytes.
    pub fn record_hash_join_hash_table_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_hash_join_hash_table_bytes(bytes));
    }

    /// Records filtered-join hash table bytes.
    pub fn record_filtered_join_hash_table_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_filtered_join_hash_table_bytes(bytes));
    }

    /// Records filtered-join probe state bytes.
    pub fn record_filtered_join_probe_state_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_filtered_join_probe_state_bytes(bytes));
    }

    /// Records mixed-join size data bytes.
    pub fn record_mixed_join_size_data_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_mixed_join_size_data_bytes(bytes));
    }

    /// Records one-shot join index bytes.
    pub fn record_one_shot_join_index_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_one_shot_join_index_bytes(bytes));
    }

    /// Records one-shot join materialization bytes.
    pub fn record_one_shot_join_materialization_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_one_shot_join_materialization_bytes(bytes));
    }

    /// Records one-shot join output bytes.
    pub fn record_one_shot_join_output_bytes(&self, bytes: u64) {
        self.apply(|metrics| metrics.record_one_shot_join_output_bytes(bytes));
    }

    // ---- dynamic filters -------------------------------------------------

    /// Records a produced dynamic filter.
    pub fn record_dynamic_filter_produced(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        producer_rows: u64,
    ) {
        self.apply(|metrics| metrics.record_dynamic_filter_produced(route, filter, producer_rows));
    }

    /// Records a skipped dynamic filter.
    pub fn record_dynamic_filter_skipped(&self, route: &DynamicFilterRoute, reason: impl Into<String>) {
        let reason = reason.into();
        self.apply(|metrics| metrics.record_dynamic_filter_skipped(route, reason));
    }

    /// Records a dynamic filter applied to a source.
    pub fn record_dynamic_filter_applied(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        before: DynamicFilterSourceSnapshot,
        after: DynamicFilterSourceSnapshot,
    ) {
        self.apply(|metrics| metrics.record_dynamic_filter_applied(route, filter, before, after));
    }

    /// Records a dynamic filter applied after decode.
    pub fn record_dynamic_filter_applied_post_decode(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| metrics.record_dynamic_filter_applied_post_decode(route, filter, rows_before, rows_after));
    }

    /// Records one source post-decode dynamic-filter stage.
    pub fn record_dynamic_filter_source_post_decode_stage(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| {
            metrics.record_dynamic_filter_source_post_decode_stage(route, filter, rows_before, rows_after);
        });
    }

    /// Records post-decode dynamic-filter row counts.
    pub fn record_dynamic_filter_post_decode_rows(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| metrics.record_dynamic_filter_post_decode_rows(route, filter, rows_before, rows_after));
    }

    /// Records reader-predicate dynamic-filter row counts for one source chunk.
    pub fn record_dynamic_filter_source_reader_predicate_rows(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        source_file_index: usize,
        source_chunk_index: usize,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| {
            metrics.record_dynamic_filter_source_reader_predicate_rows(
                route,
                filter,
                source_file_index,
                source_chunk_index,
                rows_before,
                rows_after,
            );
        });
    }

    /// Records reader-predicate dynamic-filter row counts for one source batch.
    pub fn record_dynamic_filter_source_reader_predicate_batch_rows(
        &self,
        route: &DynamicFilterRoute,
        filter: &DynamicFilter,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| {
            metrics.record_dynamic_filter_source_reader_predicate_batch_rows(route, filter, rows_before, rows_after);
        });
    }

    /// Records a reader-predicate dynamic-filter source reduction.
    pub fn record_dynamic_filter_reader_predicate_source_reduction(
        &self,
        rows_before: Option<u64>,
        rows_after: Option<u64>,
    ) {
        self.apply(|metrics| metrics.record_dynamic_filter_reader_predicate_source_reduction(rows_before, rows_after));
    }

    // ---- graph -----------------------------------------------------------

    /// Records one graph algorithm runtime row.
    pub fn record_graph_algorithm_runtime(&self, row: GraphAlgorithmRuntimeMetrics) {
        self.apply(|metrics| metrics.record_graph_algorithm_runtime(row));
    }

    /// Opens a pending graph measured-peak row.
    pub fn record_graph_claim_calibration(&self, node_id: NodeId, algorithm: GraphAlgorithm) {
        self.apply(|metrics| metrics.record_graph_claim_calibration(node_id, algorithm));
    }

    /// Completes the most recent pending graph measured-peak row.
    pub fn record_graph_claim_calibration_measurement(
        &self,
        node_id: NodeId,
        algorithm: GraphAlgorithm,
        measured_peak_bytes: Option<u64>,
        measured_peak_provenance: GraphMeasuredPeakProvenance,
    ) {
        self.apply(|metrics| {
            metrics.record_graph_claim_calibration_measurement(
                node_id,
                algorithm,
                measured_peak_bytes,
                measured_peak_provenance,
            );
        });
    }

    /// Records the transition into the native graph algorithm executor.
    pub fn record_graph_claim_calibration_algorithm_started(&self, node_id: NodeId, algorithm: GraphAlgorithm) {
        self.apply(|metrics| metrics.record_graph_claim_calibration_algorithm_started(node_id, algorithm));
    }

    // ---- tasks and budgets -------------------------------------------------

    /// Records a lower-level cuDF execution-budget report.
    pub fn record_budgeted_execution(&self, operation: impl Into<String>, report: &ExecutionBudgetReport) {
        let operation = operation.into();
        self.apply(|metrics| metrics.record_budgeted_execution(operation, report));
    }

    /// Opens a window whose budget reports are later folded into one task.
    pub fn open_budget_window(&self) -> BudgetWindow {
        self.observation.open_budget_window(&self.plan_execution_id)
    }

    /// Closes a budget window and returns the peaks it saw, for an operation
    /// that reports them itself.
    pub fn take_budget_window(&self, window: BudgetWindow) -> BudgetWindowPeaks {
        self.observation.take_budget_window(&self.plan_execution_id, window)
    }

    /// Folds the peaks a budget window saw into one compiled task.
    pub fn close_budget_window(&self, compiled_task_id: impl ToString, window: BudgetWindow) {
        self.observation.close_budget_window(&self.plan_execution_id, window, &compiled_task_id.to_string());
    }

    /// Records execution of a compiled pipeline task.
    pub fn record_compiled_pipeline_task_execution(
        &self,
        compiled_task_id: impl ToString,
        operator_kind: impl Into<String>,
    ) {
        let compiled_task_id = compiled_task_id.to_string();
        let operator_kind = operator_kind.into();
        self.apply(|metrics| metrics.record_compiled_pipeline_task_execution(compiled_task_id, operator_kind));
    }

    /// Records execution of a task id and operator kind.
    pub fn record_task_execution(&self, task_id: impl ToString, operator_kind: impl Into<String>) {
        let task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        self.apply(|metrics| metrics.record_task_execution(task_id, operator_kind));
    }

    /// Registers one task definition.
    pub fn register_task_definition(&self, task_id: impl ToString, operator_kind: impl Into<String>) {
        let task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        self.apply(|metrics| metrics.register_task_definition(task_id, operator_kind));
    }

    /// Records one compiled task's plan mapping.
    pub fn record_compiled_pipeline_task_mapping(
        &self,
        compiled_task_id: impl ToString,
        operator_kind: impl Into<String>,
        plan_node_id: Option<impl Into<PlanNodeId>>,
    ) {
        let compiled_task_id = compiled_task_id.to_string();
        let operator_kind = operator_kind.into();
        let plan_node_id = plan_node_id.map(Into::into);
        self.apply(|metrics| {
            metrics.record_compiled_pipeline_task_mapping(compiled_task_id, operator_kind, plan_node_id);
        });
    }

    /// Records elapsed time for one task.
    pub fn record_task_elapsed(&self, task_id: impl ToString, operator_kind: impl Into<String>, elapsed: Duration) {
        let task_id = task_id.to_string();
        let operator_kind = operator_kind.into();
        self.apply(|metrics| metrics.record_task_elapsed(task_id, operator_kind, elapsed));
    }

    /// Records a resource-exhausted retry.
    pub fn record_retry_resource_exhausted(&self) {
        self.apply(QueryMetrics::record_retry_resource_exhausted);
    }

    /// Records one compiled pipeline attempt.
    pub fn record_compiled_pipeline_attempt(&self) {
        self.apply(QueryMetrics::record_compiled_pipeline_attempt);
    }

    /// Records executed compiled pipeline tasks.
    pub fn record_compiled_pipeline_tasks_executed(&self, tasks: u64) {
        self.apply(|metrics| metrics.record_compiled_pipeline_tasks_executed(tasks));
    }

    /// Records a compiled pipeline host Arrow sink.
    pub fn record_compiled_pipeline_host_arrow_sink(&self) {
        self.apply(QueryMetrics::record_compiled_pipeline_host_arrow_sink);
    }
}

/// How far a plan execution has progressed, as reported in diagnostic facts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecutionProgress {
    /// Source chunks planned.
    pub source_chunks_planned: u64,
    /// Source chunks committed.
    pub source_chunks_committed: u64,
    /// Partial aggregates recorded.
    pub partial_aggregate_count: u64,
}

fn saturating_add_duration(total: &mut Duration, value: Duration) {
    *total = total.checked_add(value).unwrap_or(Duration::MAX);
}
