//! Native execution metric rows projected from a DataFusion `MetricsSet`.
//!
//! See [`NativeDataFusionExecutionMetrics`] for the per-field unit and
//! cumulative-versus-instantaneous semantics.

use datafusion::physical_plan::metrics::MetricsSet;

use super::tsv::{metric_count, optional_observed_metric_count, optional_observed_metric_duration_nanos};
use crate::native_exec::SERVER_CAPTURE_LOSS_METRIC_NAME;

mod dynamic_filter;
mod graph;
mod labels;
mod lakehouse_cache;
mod plan_task;
mod source_read;
mod strategy;
mod task;

use dynamic_filter::dynamic_filter_runtime_rows;
pub use dynamic_filter::{
    NativeDynamicFilterMeasuredReductionStatus, NativeDynamicFilterPostDecodeReductionStage,
    NativeDynamicFilterRuntimeMetricRow, NativeDynamicFilterWorkAvoidanceStatus,
};
pub use graph::NativeGraphRuntimeMetricRow;
use graph::graph_runtime_rows;
pub use lakehouse_cache::NativeLakehouseCacheFillEventMetricRow;
use lakehouse_cache::lakehouse_cache_fill_event_snapshot_rows;
pub use plan_task::NativePlanTaskMappingMetricRow;
use plan_task::plan_task_mapping_rows;
pub use source_read::NativeSourceReadRuntimeMetricRow;
use source_read::source_read_runtime_rows;
pub use strategy::{NativeAggregateStrategyMetricRow, NativeJoinStrategyMetricRow};
use strategy::{aggregate_strategy_rows, join_strategy_rows};
pub use task::{NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow};
use task::{task_operator_rows, task_runtime_rows};

/// Native metrics projected from DataFusion `MetricsSet` into report rows.
///
/// Count fields are cumulative over one query attempt. Byte fields are bytes;
/// reservation requested/granted bytes are cumulative admission accounting,
/// while live peak, observed peak, retained, cache, and hash-table byte fields
/// describe resident device memory at an instant or retained by a cache.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeDataFusionExecutionMetrics {
    /// Count of GPU-resident input handoffs.
    pub gpu_resident_input_handoffs: usize,
    /// Count of host-to-device uploads.
    pub host_to_device_uploads: usize,
    /// Rows uploaded from host to device.
    pub host_to_device_uploaded_rows: usize,
    /// Bytes uploaded from host to device.
    pub host_to_device_uploaded_bytes: usize,
    /// Count of device-to-host materializations.
    pub device_to_host_materializations: usize,
    /// Rows materialized from device to host.
    pub device_to_host_materialized_rows: usize,
    /// Bytes materialized from device to host.
    pub device_to_host_materialized_bytes: usize,
    /// Count of device-to-device copies.
    pub device_to_device_copies: usize,
    /// Bytes copied device-to-device.
    pub device_to_device_copied_bytes: usize,
    /// Parquet files observed by native source reads.
    pub parquet_files: usize,
    /// Sum of known Parquet file bytes.
    pub parquet_bytes_known: usize,
    /// Row groups selected by native source reads.
    pub row_groups_selected: usize,
    /// Physical read columns selected by native source reads.
    pub read_columns: usize,
    /// Query-service admission sequence, when execution used QueryService admission.
    pub admission_sequence: Option<usize>,
    /// Query-service admission wait time in nanoseconds, when observed.
    pub admission_wait_time_ns: Option<usize>,
    /// CUDA device ordinal selected by QueryService admission, when observed.
    pub device_ordinal: Option<usize>,
    /// Query-service device-memory grant bytes, when observed.
    pub query_grant_bytes: Option<usize>,
    /// Peak bytes retained by resident cudf DataFrames, when observed.
    pub resident_dataframe_peak_bytes: Option<usize>,
    /// cuDF allocator high-water bytes, when reported by lower layers.
    pub allocator_high_water_bytes: Option<usize>,
    /// Largest lower-level task allocator peak, when reported.
    pub max_task_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count, when reported.
    pub max_task_retained_bytes: Option<usize>,
    /// Stable nested graph runtime metric rows.
    pub graph_runtime_rows: Vec<NativeGraphRuntimeMetricRow>,
    /// Stable nested task metrics aggregated by operator kind.
    pub task_operator_rows: Vec<NativeTaskOperatorMetricRow>,
    /// Stable nested task runtime metric rows keyed by compiled task id.
    pub task_runtime_rows: Vec<NativeTaskRuntimeMetricRow>,
    /// Stable nested mapping rows between native plan nodes and compiled tasks.
    pub plan_task_mapping_rows: Vec<NativePlanTaskMappingMetricRow>,
    /// Count of source chunks planned.
    pub source_chunks_planned: usize,
    /// Count of source chunk attempts, including retries.
    pub source_chunks_attempted: usize,
    /// Count of source chunks successfully committed.
    pub source_chunks_committed: usize,
    /// Count of source chunk retries.
    pub source_chunks_retried: usize,
    /// Count of source-read prefetches started.
    pub source_read_prefetches_started: usize,
    /// Count of source-read prefetches completed.
    pub source_read_prefetches_completed: usize,
    /// Count of source-read prefetches skipped.
    pub source_read_prefetches_skipped: usize,
    /// Count of source-read prefetches abandoned.
    pub source_read_prefetches_abandoned: usize,
    /// Row groups planned across source chunks.
    pub row_groups_planned: usize,
    /// Row groups committed across source chunks.
    pub row_groups_committed: usize,
    /// Count of partial aggregate tasks.
    pub partial_aggregate_count: usize,
    /// Input rows consumed by partial aggregates.
    pub partial_aggregate_input_rows: usize,
    /// Rows produced by partial aggregates.
    pub partial_aggregate_rows: usize,
    /// Peak retained bytes from partial aggregate outputs.
    pub partial_aggregate_retained_bytes: usize,
    /// Count of final merge attempts.
    pub final_merge_attempts: usize,
    /// Count of final merge levels.
    pub final_merge_levels: usize,
    /// Largest final-merge fan-in.
    pub final_merge_max_fan_in: usize,
    /// Total retry count.
    pub retry_count: usize,
    /// Retries caused by lower-level resource exhaustion.
    pub retry_resource_exhausted_count: usize,
    /// Dynamic-filter payloads successfully produced.
    pub dynamic_filter_produced_count: usize,
    /// Dynamic-filter routes whose producers completed without a usable payload.
    pub dynamic_filter_skipped_count: usize,
    /// Dynamic-filter payloads applied to a runtime consumer.
    pub dynamic_filter_applied_count: usize,
    /// Row groups removed from source reads by dynamic filters.
    pub dynamic_filter_row_groups_pruned: usize,
    /// Estimated source-read bytes removed by dynamic filters.
    pub dynamic_filter_source_bytes_pruned: usize,
    /// Source rows selected before dynamic reader predicates were applied.
    ///
    /// Counted once per source chunk. Dynamic-filter routes do not add to the count.
    pub dynamic_filter_reader_predicate_rows_before: usize,
    /// Source rows materialized after dynamic reader predicates were applied.
    ///
    /// Counted once per source chunk.
    pub dynamic_filter_reader_predicate_rows_after: usize,
    /// Source rows removed by dynamic reader predicates.
    ///
    /// Counted once per source chunk and only when both before/after row counts
    /// are known.
    pub dynamic_filter_reader_predicate_rows_filtered: usize,
    /// Materialized rows before post-decode dynamic filters were applied.
    ///
    /// Counted once per post-decode filter application or stream chunk.
    pub dynamic_filter_post_decode_rows_before: usize,
    /// Materialized rows after post-decode dynamic filters were applied.
    ///
    /// Counted once per post-decode filter application or stream chunk.
    pub dynamic_filter_post_decode_rows_after: usize,
    /// Materialized rows removed by post-decode dynamic filters.
    ///
    /// Counted once per post-decode filter application or stream chunk and
    /// only when both before/after row counts are known.
    pub dynamic_filter_post_decode_rows_filtered: usize,
    /// Materialized rows removed by source-adjacent post-decode dynamic filters.
    pub dynamic_filter_source_post_decode_rows_filtered: usize,
    /// Materialized rows removed by filter-operator post-decode dynamic filters.
    pub dynamic_filter_filter_post_decode_rows_filtered: usize,
    /// Materialized rows removed by hash-join-probe post-decode dynamic filters.
    pub dynamic_filter_join_probe_post_decode_rows_filtered: usize,
    /// Stable nested dynamic-filter runtime rows.
    pub dynamic_filter_runtime_rows: Vec<NativeDynamicFilterRuntimeMetricRow>,
    /// Stable nested source-read runtime rows.
    pub source_read_runtime_rows: Vec<NativeSourceReadRuntimeMetricRow>,
    /// Stable nested background lakehouse cache-fill events observed before metrics collection.
    pub lakehouse_cache_fill_event_snapshot_rows: Vec<NativeLakehouseCacheFillEventMetricRow>,
    /// Count of reusable cuDF hash-join artifacts built.
    pub hash_join_build_count: usize,
    /// Count of reusable cuDF filtered-join artifacts built.
    pub filtered_join_build_count: usize,
    /// Count of one-shot keyed equality joins executed.
    pub one_shot_join_count: usize,
    /// Unique device bytes retained by reusable hash-join build artifacts.
    pub hash_join_hash_table_bytes: usize,
    /// Unique device bytes retained by reusable filtered-join build artifacts.
    pub filtered_join_hash_table_bytes: usize,
    /// Unique device bytes retained by filtered-join probe states.
    pub filtered_join_probe_state_bytes: usize,
    /// Device bytes retained by mixed join size-data states.
    pub mixed_join_size_data_bytes: usize,
    /// Device bytes retained by one-shot join index maps.
    pub one_shot_join_index_bytes: usize,
    /// Reservation bytes requested for one-shot join output materialization.
    pub one_shot_join_materialization_bytes: usize,
    /// Observed device bytes of one-shot join output frames.
    pub one_shot_join_output_bytes: usize,
    /// Device bytes retained by join gather maps.
    pub join_gather_map_bytes: usize,
    /// Stable nested aggregate strategy rows.
    pub aggregate_strategy_rows: Vec<NativeAggregateStrategyMetricRow>,
    /// Stable nested join strategy rows.
    pub join_strategy_rows: Vec<NativeJoinStrategyMetricRow>,
    /// Whether detail rows were lost before these metrics were rendered. The
    /// source only flags a loss, so how many rows were lost is unknown.
    pub detail_capture_lost: bool,
}

impl NativeDataFusionExecutionMetrics {
    /// Extracts native execution metrics from a DataFusion metrics set.
    pub fn from_metrics(metrics: &MetricsSet) -> Self {
        Self {
            gpu_resident_input_handoffs: metric_count(metrics, "gpu_resident_input_handoffs"),
            host_to_device_uploads: metric_count(metrics, "host_to_device_uploads"),
            host_to_device_uploaded_rows: metric_count(metrics, "host_to_device_uploaded_rows"),
            host_to_device_uploaded_bytes: metric_count(metrics, "host_to_device_uploaded_bytes"),
            device_to_host_materializations: metric_count(metrics, "device_to_host_materializations"),
            device_to_host_materialized_rows: metric_count(metrics, "device_to_host_materialized_rows"),
            device_to_host_materialized_bytes: metric_count(metrics, "device_to_host_materialized_bytes"),
            device_to_device_copies: metric_count(metrics, "device_to_device_copies"),
            device_to_device_copied_bytes: metric_count(metrics, "device_to_device_copied_bytes"),
            parquet_files: metric_count(metrics, "parquet_files"),
            parquet_bytes_known: metric_count(metrics, "parquet_bytes_known"),
            row_groups_selected: metric_count(metrics, "row_groups_selected"),
            read_columns: metric_count(metrics, "read_columns"),
            admission_sequence: optional_observed_metric_count(metrics, "admission_sequence"),
            admission_wait_time_ns: optional_observed_metric_duration_nanos(metrics, "admission_wait_time"),
            device_ordinal: optional_observed_metric_count(metrics, "device_ordinal"),
            query_grant_bytes: optional_observed_metric_count(metrics, "query_grant_bytes"),
            resident_dataframe_peak_bytes: optional_observed_metric_count(metrics, "resident_dataframe_peak_bytes"),
            allocator_high_water_bytes: optional_observed_metric_count(metrics, "allocator_high_water_bytes"),
            max_task_observed_peak_bytes: optional_observed_metric_count(metrics, "max_task_observed_peak_bytes"),
            max_task_retained_bytes: optional_observed_metric_count(metrics, "max_task_retained_bytes"),
            graph_runtime_rows: graph_runtime_rows(metrics),
            task_operator_rows: task_operator_rows(metrics),
            task_runtime_rows: task_runtime_rows(metrics),
            plan_task_mapping_rows: plan_task_mapping_rows(metrics),
            source_chunks_planned: metric_count(metrics, "source_chunks_planned"),
            source_chunks_attempted: metric_count(metrics, "source_chunks_attempted"),
            source_chunks_committed: metric_count(metrics, "source_chunks_committed"),
            source_chunks_retried: metric_count(metrics, "source_chunks_retried"),
            source_read_prefetches_started: metric_count(metrics, "source_read_prefetches_started"),
            source_read_prefetches_completed: metric_count(metrics, "source_read_prefetches_completed"),
            source_read_prefetches_skipped: metric_count(metrics, "source_read_prefetches_skipped"),
            source_read_prefetches_abandoned: metric_count(metrics, "source_read_prefetches_abandoned"),
            row_groups_planned: metric_count(metrics, "row_groups_planned"),
            row_groups_committed: metric_count(metrics, "row_groups_committed"),
            partial_aggregate_count: metric_count(metrics, "partial_aggregate_count"),
            partial_aggregate_input_rows: metric_count(metrics, "partial_aggregate_input_rows"),
            partial_aggregate_rows: metric_count(metrics, "partial_aggregate_rows"),
            partial_aggregate_retained_bytes: metric_count(metrics, "partial_aggregate_retained_bytes"),
            final_merge_attempts: metric_count(metrics, "final_merge_attempts"),
            final_merge_levels: metric_count(metrics, "final_merge_levels"),
            final_merge_max_fan_in: metric_count(metrics, "final_merge_max_fan_in"),
            retry_count: metric_count(metrics, "retry_count"),
            retry_resource_exhausted_count: metric_count(metrics, "retry_resource_exhausted_count"),
            dynamic_filter_produced_count: metric_count(metrics, "dynamic_filter_produced_count"),
            dynamic_filter_skipped_count: metric_count(metrics, "dynamic_filter_skipped_count"),
            dynamic_filter_applied_count: metric_count(metrics, "dynamic_filter_applied_count"),
            dynamic_filter_row_groups_pruned: metric_count(metrics, "dynamic_filter_row_groups_pruned"),
            dynamic_filter_source_bytes_pruned: metric_count(metrics, "dynamic_filter_source_bytes_pruned"),
            dynamic_filter_reader_predicate_rows_before: metric_count(
                metrics,
                "dynamic_filter_reader_predicate_rows_before",
            ),
            dynamic_filter_reader_predicate_rows_after: metric_count(
                metrics,
                "dynamic_filter_reader_predicate_rows_after",
            ),
            dynamic_filter_reader_predicate_rows_filtered: metric_count(
                metrics,
                "dynamic_filter_reader_predicate_rows_filtered",
            ),
            dynamic_filter_post_decode_rows_before: metric_count(metrics, "dynamic_filter_post_decode_rows_before"),
            dynamic_filter_post_decode_rows_after: metric_count(metrics, "dynamic_filter_post_decode_rows_after"),
            dynamic_filter_post_decode_rows_filtered: metric_count(metrics, "dynamic_filter_post_decode_rows_filtered"),
            dynamic_filter_source_post_decode_rows_filtered: metric_count(
                metrics,
                "dynamic_filter_source_post_decode_rows_filtered",
            ),
            dynamic_filter_filter_post_decode_rows_filtered: metric_count(
                metrics,
                "dynamic_filter_filter_post_decode_rows_filtered",
            ),
            dynamic_filter_join_probe_post_decode_rows_filtered: metric_count(
                metrics,
                "dynamic_filter_join_probe_post_decode_rows_filtered",
            ),
            dynamic_filter_runtime_rows: dynamic_filter_runtime_rows(metrics),
            source_read_runtime_rows: source_read_runtime_rows(metrics),
            lakehouse_cache_fill_event_snapshot_rows: lakehouse_cache_fill_event_snapshot_rows(metrics),
            hash_join_build_count: metric_count(metrics, "hash_join_build_count"),
            filtered_join_build_count: metric_count(metrics, "filtered_join_build_count"),
            one_shot_join_count: metric_count(metrics, "one_shot_join_count"),
            hash_join_hash_table_bytes: metric_count(metrics, "hash_join_hash_table_bytes"),
            filtered_join_hash_table_bytes: metric_count(metrics, "filtered_join_hash_table_bytes"),
            filtered_join_probe_state_bytes: metric_count(metrics, "filtered_join_probe_state_bytes"),
            mixed_join_size_data_bytes: metric_count(metrics, "mixed_join_size_data_bytes"),
            one_shot_join_index_bytes: metric_count(metrics, "one_shot_join_index_bytes"),
            one_shot_join_materialization_bytes: metric_count(metrics, "one_shot_join_materialization_bytes"),
            one_shot_join_output_bytes: metric_count(metrics, "one_shot_join_output_bytes"),
            join_gather_map_bytes: metric_count(metrics, "join_gather_map_bytes"),
            aggregate_strategy_rows: aggregate_strategy_rows(metrics),
            join_strategy_rows: join_strategy_rows(metrics),
            detail_capture_lost: metric_count(metrics, SERVER_CAPTURE_LOSS_METRIC_NAME) > 0,
        }
    }

    /// Merges one query attempt's metrics into this cross-attempt projection.
    ///
    /// Cumulative fields saturate, peak fields retain the maximum, and
    /// per-attempt admission identities remain present only when both attempts
    /// reported the same value. Nested rows retain their existing merge and
    /// stable-order rules.
    pub fn merge_from(&mut self, other: Self) {
        let Self {
            gpu_resident_input_handoffs,
            host_to_device_uploads,
            host_to_device_uploaded_rows,
            host_to_device_uploaded_bytes,
            device_to_host_materializations,
            device_to_host_materialized_rows,
            device_to_host_materialized_bytes,
            device_to_device_copies,
            device_to_device_copied_bytes,
            parquet_files,
            parquet_bytes_known,
            row_groups_selected,
            read_columns,
            admission_sequence,
            admission_wait_time_ns,
            device_ordinal,
            query_grant_bytes,
            resident_dataframe_peak_bytes,
            allocator_high_water_bytes,
            max_task_observed_peak_bytes,
            max_task_retained_bytes,
            graph_runtime_rows,
            task_operator_rows,
            task_runtime_rows,
            plan_task_mapping_rows,
            source_chunks_planned,
            source_chunks_attempted,
            source_chunks_committed,
            source_chunks_retried,
            source_read_prefetches_started,
            source_read_prefetches_completed,
            source_read_prefetches_skipped,
            source_read_prefetches_abandoned,
            row_groups_planned,
            row_groups_committed,
            partial_aggregate_count,
            partial_aggregate_input_rows,
            partial_aggregate_rows,
            partial_aggregate_retained_bytes,
            final_merge_attempts,
            final_merge_levels,
            final_merge_max_fan_in,
            retry_count,
            retry_resource_exhausted_count,
            dynamic_filter_produced_count,
            dynamic_filter_skipped_count,
            dynamic_filter_applied_count,
            dynamic_filter_row_groups_pruned,
            dynamic_filter_source_bytes_pruned,
            dynamic_filter_reader_predicate_rows_before,
            dynamic_filter_reader_predicate_rows_after,
            dynamic_filter_reader_predicate_rows_filtered,
            dynamic_filter_post_decode_rows_before,
            dynamic_filter_post_decode_rows_after,
            dynamic_filter_post_decode_rows_filtered,
            dynamic_filter_source_post_decode_rows_filtered,
            dynamic_filter_filter_post_decode_rows_filtered,
            dynamic_filter_join_probe_post_decode_rows_filtered,
            dynamic_filter_runtime_rows,
            source_read_runtime_rows,
            lakehouse_cache_fill_event_snapshot_rows,
            hash_join_build_count,
            filtered_join_build_count,
            one_shot_join_count,
            hash_join_hash_table_bytes,
            filtered_join_hash_table_bytes,
            filtered_join_probe_state_bytes,
            mixed_join_size_data_bytes,
            one_shot_join_index_bytes,
            one_shot_join_materialization_bytes,
            one_shot_join_output_bytes,
            join_gather_map_bytes,
            aggregate_strategy_rows,
            join_strategy_rows,
            detail_capture_lost,
        } = other;

        self.gpu_resident_input_handoffs = self.gpu_resident_input_handoffs.saturating_add(gpu_resident_input_handoffs);
        self.host_to_device_uploads = self.host_to_device_uploads.saturating_add(host_to_device_uploads);
        self.host_to_device_uploaded_rows =
            self.host_to_device_uploaded_rows.saturating_add(host_to_device_uploaded_rows);
        self.host_to_device_uploaded_bytes =
            self.host_to_device_uploaded_bytes.saturating_add(host_to_device_uploaded_bytes);
        self.device_to_host_materializations =
            self.device_to_host_materializations.saturating_add(device_to_host_materializations);
        self.device_to_host_materialized_rows =
            self.device_to_host_materialized_rows.saturating_add(device_to_host_materialized_rows);
        self.device_to_host_materialized_bytes =
            self.device_to_host_materialized_bytes.saturating_add(device_to_host_materialized_bytes);
        self.device_to_device_copies = self.device_to_device_copies.saturating_add(device_to_device_copies);
        self.device_to_device_copied_bytes =
            self.device_to_device_copied_bytes.saturating_add(device_to_device_copied_bytes);
        self.parquet_files = self.parquet_files.saturating_add(parquet_files);
        self.parquet_bytes_known = self.parquet_bytes_known.saturating_add(parquet_bytes_known);
        self.row_groups_selected = self.row_groups_selected.saturating_add(row_groups_selected);
        self.read_columns = self.read_columns.saturating_add(read_columns);
        self.admission_sequence = match (self.admission_sequence, admission_sequence) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        };
        self.admission_wait_time_ns = self.admission_wait_time_ns.max(admission_wait_time_ns);
        self.device_ordinal = match (self.device_ordinal, device_ordinal) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        };
        self.query_grant_bytes = self.query_grant_bytes.max(query_grant_bytes);
        self.resident_dataframe_peak_bytes = self.resident_dataframe_peak_bytes.max(resident_dataframe_peak_bytes);
        self.allocator_high_water_bytes = self.allocator_high_water_bytes.max(allocator_high_water_bytes);
        self.max_task_observed_peak_bytes = self.max_task_observed_peak_bytes.max(max_task_observed_peak_bytes);
        self.max_task_retained_bytes = self.max_task_retained_bytes.max(max_task_retained_bytes);
        self.merge_graph_runtime_rows(graph_runtime_rows);
        self.merge_task_operator_rows(task_operator_rows);
        self.merge_task_runtime_rows(task_runtime_rows);
        self.plan_task_mapping_rows.extend(plan_task_mapping_rows);
        self.plan_task_mapping_rows.sort_by_key(|row| row.to_stable_kv_row());
        self.plan_task_mapping_rows.dedup();
        self.source_chunks_planned = self.source_chunks_planned.saturating_add(source_chunks_planned);
        self.source_chunks_attempted = self.source_chunks_attempted.saturating_add(source_chunks_attempted);
        self.source_chunks_committed = self.source_chunks_committed.saturating_add(source_chunks_committed);
        self.source_chunks_retried = self.source_chunks_retried.saturating_add(source_chunks_retried);
        self.source_read_prefetches_started =
            self.source_read_prefetches_started.saturating_add(source_read_prefetches_started);
        self.source_read_prefetches_completed =
            self.source_read_prefetches_completed.saturating_add(source_read_prefetches_completed);
        self.source_read_prefetches_skipped =
            self.source_read_prefetches_skipped.saturating_add(source_read_prefetches_skipped);
        self.source_read_prefetches_abandoned =
            self.source_read_prefetches_abandoned.saturating_add(source_read_prefetches_abandoned);
        self.row_groups_planned = self.row_groups_planned.saturating_add(row_groups_planned);
        self.row_groups_committed = self.row_groups_committed.saturating_add(row_groups_committed);
        self.partial_aggregate_count = self.partial_aggregate_count.saturating_add(partial_aggregate_count);
        self.partial_aggregate_input_rows =
            self.partial_aggregate_input_rows.saturating_add(partial_aggregate_input_rows);
        self.partial_aggregate_rows = self.partial_aggregate_rows.saturating_add(partial_aggregate_rows);
        self.partial_aggregate_retained_bytes =
            self.partial_aggregate_retained_bytes.max(partial_aggregate_retained_bytes);
        self.final_merge_attempts = self.final_merge_attempts.saturating_add(final_merge_attempts);
        self.final_merge_levels = self.final_merge_levels.saturating_add(final_merge_levels);
        self.final_merge_max_fan_in = self.final_merge_max_fan_in.max(final_merge_max_fan_in);
        self.retry_count = self.retry_count.saturating_add(retry_count);
        self.retry_resource_exhausted_count =
            self.retry_resource_exhausted_count.saturating_add(retry_resource_exhausted_count);
        self.dynamic_filter_produced_count =
            self.dynamic_filter_produced_count.saturating_add(dynamic_filter_produced_count);
        self.dynamic_filter_skipped_count =
            self.dynamic_filter_skipped_count.saturating_add(dynamic_filter_skipped_count);
        self.dynamic_filter_applied_count =
            self.dynamic_filter_applied_count.saturating_add(dynamic_filter_applied_count);
        self.dynamic_filter_row_groups_pruned =
            self.dynamic_filter_row_groups_pruned.saturating_add(dynamic_filter_row_groups_pruned);
        self.dynamic_filter_source_bytes_pruned =
            self.dynamic_filter_source_bytes_pruned.saturating_add(dynamic_filter_source_bytes_pruned);
        self.dynamic_filter_reader_predicate_rows_before = self
            .dynamic_filter_reader_predicate_rows_before
            .saturating_add(dynamic_filter_reader_predicate_rows_before);
        self.dynamic_filter_reader_predicate_rows_after =
            self.dynamic_filter_reader_predicate_rows_after.saturating_add(dynamic_filter_reader_predicate_rows_after);
        self.dynamic_filter_reader_predicate_rows_filtered = self
            .dynamic_filter_reader_predicate_rows_filtered
            .saturating_add(dynamic_filter_reader_predicate_rows_filtered);
        self.dynamic_filter_post_decode_rows_before =
            self.dynamic_filter_post_decode_rows_before.saturating_add(dynamic_filter_post_decode_rows_before);
        self.dynamic_filter_post_decode_rows_after =
            self.dynamic_filter_post_decode_rows_after.saturating_add(dynamic_filter_post_decode_rows_after);
        self.dynamic_filter_post_decode_rows_filtered =
            self.dynamic_filter_post_decode_rows_filtered.saturating_add(dynamic_filter_post_decode_rows_filtered);
        self.dynamic_filter_source_post_decode_rows_filtered = self
            .dynamic_filter_source_post_decode_rows_filtered
            .saturating_add(dynamic_filter_source_post_decode_rows_filtered);
        self.dynamic_filter_filter_post_decode_rows_filtered = self
            .dynamic_filter_filter_post_decode_rows_filtered
            .saturating_add(dynamic_filter_filter_post_decode_rows_filtered);
        self.dynamic_filter_join_probe_post_decode_rows_filtered = self
            .dynamic_filter_join_probe_post_decode_rows_filtered
            .saturating_add(dynamic_filter_join_probe_post_decode_rows_filtered);
        self.merge_dynamic_filter_runtime_rows(dynamic_filter_runtime_rows);
        self.source_read_runtime_rows.extend(source_read_runtime_rows);
        self.source_read_runtime_rows.sort_by_key(|row| row.to_stable_kv_row());
        self.lakehouse_cache_fill_event_snapshot_rows.extend(lakehouse_cache_fill_event_snapshot_rows);
        self.lakehouse_cache_fill_event_snapshot_rows.sort_by_key(|row| row.to_stable_kv_row());
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(hash_join_build_count);
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(filtered_join_build_count);
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(one_shot_join_count);
        self.hash_join_hash_table_bytes = self.hash_join_hash_table_bytes.saturating_add(hash_join_hash_table_bytes);
        self.filtered_join_hash_table_bytes =
            self.filtered_join_hash_table_bytes.saturating_add(filtered_join_hash_table_bytes);
        self.filtered_join_probe_state_bytes =
            self.filtered_join_probe_state_bytes.saturating_add(filtered_join_probe_state_bytes);
        self.mixed_join_size_data_bytes = self.mixed_join_size_data_bytes.saturating_add(mixed_join_size_data_bytes);
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(one_shot_join_index_bytes);
        self.one_shot_join_materialization_bytes =
            self.one_shot_join_materialization_bytes.saturating_add(one_shot_join_materialization_bytes);
        self.one_shot_join_output_bytes = self.one_shot_join_output_bytes.saturating_add(one_shot_join_output_bytes);
        self.join_gather_map_bytes = self.join_gather_map_bytes.saturating_add(join_gather_map_bytes);
        self.merge_aggregate_strategy_rows(aggregate_strategy_rows);
        self.merge_join_strategy_rows(join_strategy_rows);
        self.detail_capture_lost |= detail_capture_lost;
    }

    /// Classifies dynamic-filter runtime evidence by the strongest measured reduction.
    ///
    /// `dynamic_filter_applied_count` is intentionally not treated as a
    /// reduction signal. It only marks that a route reached a consumer.
    pub fn dynamic_filter_measured_reduction_status(&self) -> NativeDynamicFilterMeasuredReductionStatus {
        if self.dynamic_filter_source_bytes_pruned > 0 || self.dynamic_filter_row_groups_pruned > 0 {
            return NativeDynamicFilterMeasuredReductionStatus::MeasuredSourcePruned;
        }
        if self.dynamic_filter_reader_predicate_rows_filtered > 0 {
            return NativeDynamicFilterMeasuredReductionStatus::MeasuredReaderRowsReduced;
        }
        if self.dynamic_filter_post_decode_rows_filtered > 0 {
            return NativeDynamicFilterMeasuredReductionStatus::MeasuredPostDecodeReduced;
        }
        if self.dynamic_filter_produced_count > 0
            || self.dynamic_filter_skipped_count > 0
            || self.dynamic_filter_applied_count > 0
            || !self.dynamic_filter_runtime_rows.is_empty()
        {
            return NativeDynamicFilterMeasuredReductionStatus::MeasuredNoReduction;
        }
        NativeDynamicFilterMeasuredReductionStatus::Unknown
    }

    /// Classifies the strongest observed work avoided by dynamic filters.
    pub fn dynamic_filter_work_avoidance_status(&self) -> NativeDynamicFilterWorkAvoidanceStatus {
        if self.dynamic_filter_source_bytes_pruned > 0 || self.dynamic_filter_row_groups_pruned > 0 {
            return NativeDynamicFilterWorkAvoidanceStatus::SourceReadPruned;
        }
        if self.dynamic_filter_reader_predicate_rows_filtered > 0 {
            return NativeDynamicFilterWorkAvoidanceStatus::SourceReaderRowsReduced;
        }
        if self.dynamic_filter_post_decode_rows_filtered > 0 {
            return match self.dynamic_filter_post_decode_reduction_stage() {
                NativeDynamicFilterPostDecodeReductionStage::SourcePostDecode => {
                    NativeDynamicFilterWorkAvoidanceStatus::SourcePostDecodeRowsReduced
                },
                NativeDynamicFilterPostDecodeReductionStage::FilterPostDecode => {
                    NativeDynamicFilterWorkAvoidanceStatus::FilterPostDecodeRowsReduced
                },
                NativeDynamicFilterPostDecodeReductionStage::JoinProbePostDecode => {
                    NativeDynamicFilterWorkAvoidanceStatus::JoinProbePostDecodeRowsReduced
                },
                NativeDynamicFilterPostDecodeReductionStage::MixedPostDecode => {
                    NativeDynamicFilterWorkAvoidanceStatus::MixedPostDecodeRowsReduced
                },
                NativeDynamicFilterPostDecodeReductionStage::None
                | NativeDynamicFilterPostDecodeReductionStage::Unknown => {
                    NativeDynamicFilterWorkAvoidanceStatus::Unknown
                },
            };
        }
        if self.dynamic_filter_produced_count > 0
            || self.dynamic_filter_skipped_count > 0
            || self.dynamic_filter_applied_count > 0
            || !self.dynamic_filter_runtime_rows.is_empty()
        {
            return NativeDynamicFilterWorkAvoidanceStatus::MeasuredNoReduction;
        }
        NativeDynamicFilterWorkAvoidanceStatus::Unknown
    }

    /// Classifies the post-decode consumer stage that reduced rows.
    pub fn dynamic_filter_post_decode_reduction_stage(&self) -> NativeDynamicFilterPostDecodeReductionStage {
        if self.dynamic_filter_post_decode_rows_filtered == 0 {
            return NativeDynamicFilterPostDecodeReductionStage::None;
        }

        let mut classified_rows_filtered = 0_usize;
        let source_post_decode = self.dynamic_filter_source_post_decode_rows_filtered > 0;
        if source_post_decode {
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_source_post_decode_rows_filtered);
        }
        let filter_post_decode = self.dynamic_filter_filter_post_decode_rows_filtered > 0;
        if filter_post_decode {
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_filter_post_decode_rows_filtered);
        }
        let join_probe_post_decode = self.dynamic_filter_join_probe_post_decode_rows_filtered > 0;
        if join_probe_post_decode {
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_join_probe_post_decode_rows_filtered);
        }

        let stage_count = usize::from(source_post_decode)
            + usize::from(filter_post_decode)
            + usize::from(join_probe_post_decode)
            + usize::from(classified_rows_filtered < self.dynamic_filter_post_decode_rows_filtered);
        if stage_count > 1 {
            return NativeDynamicFilterPostDecodeReductionStage::MixedPostDecode;
        }
        if source_post_decode {
            return NativeDynamicFilterPostDecodeReductionStage::SourcePostDecode;
        }
        if filter_post_decode {
            return NativeDynamicFilterPostDecodeReductionStage::FilterPostDecode;
        }
        if join_probe_post_decode {
            return NativeDynamicFilterPostDecodeReductionStage::JoinProbePostDecode;
        }
        debug_assert!(classified_rows_filtered < self.dynamic_filter_post_decode_rows_filtered);
        NativeDynamicFilterPostDecodeReductionStage::Unknown
    }

    /// Counts partial aggregate strategy rows that used libcudf groupby only
    /// because no prior cardinality evidence was available.
    pub fn aggregate_libcudf_runtime_probe_count(&self) -> usize {
        self.aggregate_strategy_rows.iter().filter(|row| row.is_libcudf_runtime_probe()).count()
    }

    /// Total elapsed nanoseconds for libcudf runtime-probe aggregate rows.
    pub fn aggregate_libcudf_runtime_probe_elapsed_ns(&self) -> usize {
        self.aggregate_strategy_rows
            .iter()
            .filter(|row| row.is_libcudf_runtime_probe())
            .map(|row| row.strategy_operator_elapsed_ns)
            .sum()
    }

    /// Input rows consumed by libcudf runtime-probe aggregate rows.
    pub fn aggregate_libcudf_runtime_probe_input_rows(&self) -> usize {
        self.aggregate_strategy_rows.iter().filter(|row| row.is_libcudf_runtime_probe()).map(|row| row.input_rows).sum()
    }

    /// Counts partial aggregate strategy rows that measured exact single-key cardinality.
    pub fn aggregate_exact_cardinality_probe_count(&self) -> usize {
        self.aggregate_strategy_rows.iter().filter(|row| row.has_exact_cardinality_probe()).count()
    }

    /// Total elapsed nanoseconds for exact single-key cardinality probes.
    pub fn aggregate_exact_cardinality_probe_elapsed_ns(&self) -> usize {
        self.aggregate_strategy_rows
            .iter()
            .filter(|row| row.has_exact_cardinality_probe())
            .map(|row| row.exact_cardinality_probe_elapsed_ns)
            .sum()
    }

    /// Input rows consumed by exact single-key cardinality probes.
    pub fn aggregate_exact_cardinality_probe_input_rows(&self) -> usize {
        self.aggregate_strategy_rows
            .iter()
            .filter(|row| row.has_exact_cardinality_probe())
            .map(|row| row.exact_cardinality_probe_input_rows)
            .sum()
    }

    /// Total elapsed nanoseconds for prepared sort-groupby aggregate rows.
    pub fn aggregate_prepared_sort_groupby_elapsed_ns(&self) -> usize {
        self.aggregate_strategy_rows
            .iter()
            .filter(|row| row.selected_strategy == "prepared_sort_groupby")
            .map(|row| row.strategy_operator_elapsed_ns)
            .sum()
    }

    /// Total elapsed nanoseconds for encoded-key partial aggregate rows.
    pub fn aggregate_encoded_key_partial_elapsed_ns(&self) -> usize {
        self.aggregate_strategy_rows
            .iter()
            .filter(|row| row.selected_strategy == "encoded_key_partial")
            .map(|row| row.strategy_operator_elapsed_ns)
            .sum()
    }

    /// Merges aggregate strategy rows by aggregate site and keeps stable sort order.
    fn merge_aggregate_strategy_rows(&mut self, rows: Vec<NativeAggregateStrategyMetricRow>) {
        for row in rows {
            if let Some(existing) = self.aggregate_strategy_rows.iter_mut().find(|existing| existing.same_site(&row)) {
                existing.merge_runtime_from(row);
            } else {
                self.aggregate_strategy_rows.push(row);
            }
        }
        self.aggregate_strategy_rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    }

    /// Merges join strategy rows by join site and keeps stable sort order.
    fn merge_join_strategy_rows(&mut self, rows: Vec<NativeJoinStrategyMetricRow>) {
        for row in rows {
            if let Some(existing) = self.join_strategy_rows.iter_mut().find(|existing| existing.same_site(&row)) {
                existing.merge_runtime_from(row);
            } else {
                self.join_strategy_rows.push(row);
            }
        }
        self.join_strategy_rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    }

    /// Appends graph runtime rows and restores stable sort order.
    fn merge_graph_runtime_rows(&mut self, rows: Vec<NativeGraphRuntimeMetricRow>) {
        self.graph_runtime_rows.extend(rows);
        self.graph_runtime_rows.sort_by(|left, right| {
            left.node_id
                .cmp(&right.node_id)
                .then_with(|| left.algorithm.cmp(&right.algorithm))
                .then_with(|| left.source_column.cmp(&right.source_column))
                .then_with(|| left.destination_column.cmp(&right.destination_column))
                .then_with(|| left.to_stable_kv_row().cmp(&right.to_stable_kv_row()))
        });
    }

    /// Merges task operator rows by operator kind and keeps stable sort order.
    fn merge_task_operator_rows(&mut self, rows: Vec<NativeTaskOperatorMetricRow>) {
        for row in rows {
            if let Some(existing) =
                self.task_operator_rows.iter_mut().find(|existing| existing.operator_kind == row.operator_kind)
            {
                existing.task_count = existing.task_count.saturating_add(row.task_count);
                existing.elapsed_ns = existing.elapsed_ns.saturating_add(row.elapsed_ns);
                existing.max_observed_peak_bytes = existing.max_observed_peak_bytes.max(row.max_observed_peak_bytes);
                existing.max_retained_bytes = existing.max_retained_bytes.max(row.max_retained_bytes);
            } else {
                self.task_operator_rows.push(row);
            }
        }
        self.task_operator_rows.sort_by(|left, right| left.operator_kind.cmp(&right.operator_kind));
    }

    /// Merges task runtime rows by compiled task id and keeps stable sort order.
    fn merge_task_runtime_rows(&mut self, rows: Vec<NativeTaskRuntimeMetricRow>) {
        for row in rows {
            if let Some(existing) =
                self.task_runtime_rows.iter_mut().find(|existing| existing.compiled_task_id == row.compiled_task_id)
            {
                existing.operator_kind = row.operator_kind;
                existing.elapsed_ns = existing.elapsed_ns.saturating_add(row.elapsed_ns);
                existing.max_observed_peak_bytes = existing.max_observed_peak_bytes.max(row.max_observed_peak_bytes);
                existing.max_retained_bytes = existing.max_retained_bytes.max(row.max_retained_bytes);
            } else {
                self.task_runtime_rows.push(row);
            }
        }
        self.task_runtime_rows.sort_by(|left, right| left.compiled_task_id.cmp(&right.compiled_task_id));
    }

    /// Appends dynamic-filter runtime rows and restores stable sort order.
    fn merge_dynamic_filter_runtime_rows(&mut self, rows: Vec<NativeDynamicFilterRuntimeMetricRow>) {
        self.dynamic_filter_runtime_rows.extend(rows);
        self.dynamic_filter_runtime_rows.sort_by_key(|row| row.to_stable_kv_row());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_from_preserves_scalar_peak_optional_and_identity_semantics() {
        let mut metrics = NativeDataFusionExecutionMetrics {
            gpu_resident_input_handoffs: usize::MAX - 2,
            partial_aggregate_retained_bytes: 23,
            final_merge_levels: 29,
            final_merge_max_fan_in: 31,
            hash_join_hash_table_bytes: 37,
            admission_sequence: Some(41),
            admission_wait_time_ns: Some(43),
            device_ordinal: Some(47),
            query_grant_bytes: Some(53),
            resident_dataframe_peak_bytes: Some(59),
            allocator_high_water_bytes: Some(61),
            max_task_observed_peak_bytes: Some(67),
            max_task_retained_bytes: Some(71),
            ..NativeDataFusionExecutionMetrics::default()
        };

        metrics.merge_from(NativeDataFusionExecutionMetrics {
            gpu_resident_input_handoffs: 5,
            partial_aggregate_retained_bytes: 97,
            final_merge_levels: 101,
            final_merge_max_fan_in: 103,
            hash_join_hash_table_bytes: 107,
            admission_sequence: Some(41),
            admission_wait_time_ns: Some(109),
            device_ordinal: Some(47),
            query_grant_bytes: Some(113),
            resident_dataframe_peak_bytes: Some(127),
            allocator_high_water_bytes: Some(131),
            max_task_observed_peak_bytes: Some(137),
            max_task_retained_bytes: Some(139),
            ..NativeDataFusionExecutionMetrics::default()
        });

        assert_eq!(metrics.gpu_resident_input_handoffs, usize::MAX);
        assert_eq!(metrics.partial_aggregate_retained_bytes, 97);
        assert_eq!(metrics.final_merge_levels, 130);
        assert_eq!(metrics.final_merge_max_fan_in, 103);
        assert_eq!(metrics.hash_join_hash_table_bytes, 144);
        assert_eq!(metrics.admission_sequence, Some(41));
        assert_eq!(metrics.admission_wait_time_ns, Some(109));
        assert_eq!(metrics.device_ordinal, Some(47));
        assert_eq!(metrics.query_grant_bytes, Some(113));
        assert_eq!(metrics.resident_dataframe_peak_bytes, Some(127));
        assert_eq!(metrics.allocator_high_water_bytes, Some(131));
        assert_eq!(metrics.max_task_observed_peak_bytes, Some(137));
        assert_eq!(metrics.max_task_retained_bytes, Some(139));
        assert_eq!(
            metrics,
            NativeDataFusionExecutionMetrics {
                gpu_resident_input_handoffs: usize::MAX,
                partial_aggregate_retained_bytes: 97,
                final_merge_levels: 130,
                final_merge_max_fan_in: 103,
                hash_join_hash_table_bytes: 144,
                admission_sequence: Some(41),
                admission_wait_time_ns: Some(109),
                device_ordinal: Some(47),
                query_grant_bytes: Some(113),
                resident_dataframe_peak_bytes: Some(127),
                allocator_high_water_bytes: Some(131),
                max_task_observed_peak_bytes: Some(137),
                max_task_retained_bytes: Some(139),
                ..Default::default()
            }
        );

        metrics.merge_from(NativeDataFusionExecutionMetrics {
            admission_sequence: Some(149),
            device_ordinal: Some(151),
            ..NativeDataFusionExecutionMetrics::default()
        });
        assert_eq!(metrics.admission_sequence, None);
        assert_eq!(metrics.device_ordinal, None);
    }

    #[test]
    fn merge_from_keeps_a_single_attempt_identity_and_rejects_missing_identity() {
        let first = NativeDataFusionExecutionMetrics {
            admission_sequence: Some(41),
            device_ordinal: Some(7),
            ..NativeDataFusionExecutionMetrics::default()
        };
        let mut aggregate = first.clone();

        assert_eq!(aggregate, first);

        aggregate.merge_from(NativeDataFusionExecutionMetrics::default());
        assert_eq!(aggregate.admission_sequence, None);
        assert_eq!(aggregate.device_ordinal, None);
    }

    #[test]
    fn merge_from_keeps_every_nested_row_collection_ordered() {
        let mapping = |plan_node_id: &str| NativePlanTaskMappingMetricRow {
            query_id: "q1".to_string(),
            plan_node_id: Some(plan_node_id.to_string()),
            ..Default::default()
        };
        let mut metrics = NativeDataFusionExecutionMetrics {
            graph_runtime_rows: vec![NativeGraphRuntimeMetricRow {
                node_id: "z-runtime".to_string(),
                ..Default::default()
            }],
            task_operator_rows: vec![NativeTaskOperatorMetricRow {
                operator_kind: "source".to_string(),
                task_count: 2,
                elapsed_ns: 3,
                ..Default::default()
            }],
            task_runtime_rows: vec![NativeTaskRuntimeMetricRow {
                compiled_task_id: "task-z".to_string(),
                operator_kind: "source".to_string(),
                elapsed_ns: 5,
                ..Default::default()
            }],
            plan_task_mapping_rows: vec![mapping("z-plan")],
            dynamic_filter_runtime_rows: vec![NativeDynamicFilterRuntimeMetricRow {
                dynamic_filter_id: "z-filter".to_string(),
                ..Default::default()
            }],
            source_read_runtime_rows: vec![NativeSourceReadRuntimeMetricRow {
                source_node_id: "z-source".to_string(),
                ..Default::default()
            }],
            lakehouse_cache_fill_event_snapshot_rows: vec![NativeLakehouseCacheFillEventMetricRow {
                outcome: "z-cache".to_string(),
                ..Default::default()
            }],
            aggregate_strategy_rows: vec![NativeAggregateStrategyMetricRow {
                node_id: "z-aggregate".to_string(),
                input_rows: 7,
                ..Default::default()
            }],
            join_strategy_rows: vec![NativeJoinStrategyMetricRow {
                node_id: "z-join".to_string(),
                hash_join_build_count: 11,
                ..Default::default()
            }],
            ..NativeDataFusionExecutionMetrics::default()
        };

        metrics.merge_from(NativeDataFusionExecutionMetrics {
            graph_runtime_rows: vec![NativeGraphRuntimeMetricRow {
                node_id: "a-runtime".to_string(),
                ..Default::default()
            }],
            task_operator_rows: vec![
                NativeTaskOperatorMetricRow {
                    operator_kind: "aggregate".to_string(),
                    task_count: 13,
                    ..Default::default()
                },
                NativeTaskOperatorMetricRow {
                    operator_kind: "source".to_string(),
                    task_count: 17,
                    elapsed_ns: 19,
                    ..Default::default()
                },
            ],
            task_runtime_rows: vec![
                NativeTaskRuntimeMetricRow {
                    compiled_task_id: "task-a".to_string(),
                    operator_kind: "aggregate".to_string(),
                    ..Default::default()
                },
                NativeTaskRuntimeMetricRow {
                    compiled_task_id: "task-z".to_string(),
                    operator_kind: "source-updated".to_string(),
                    elapsed_ns: 23,
                    ..Default::default()
                },
            ],
            plan_task_mapping_rows: vec![mapping("a-plan"), mapping("z-plan")],
            dynamic_filter_runtime_rows: vec![NativeDynamicFilterRuntimeMetricRow {
                dynamic_filter_id: "a-filter".to_string(),
                ..Default::default()
            }],
            source_read_runtime_rows: vec![NativeSourceReadRuntimeMetricRow {
                source_node_id: "a-source".to_string(),
                ..Default::default()
            }],
            lakehouse_cache_fill_event_snapshot_rows: vec![NativeLakehouseCacheFillEventMetricRow {
                outcome: "a-cache".to_string(),
                ..Default::default()
            }],
            aggregate_strategy_rows: vec![
                NativeAggregateStrategyMetricRow { node_id: "a-aggregate".to_string(), ..Default::default() },
                NativeAggregateStrategyMetricRow {
                    node_id: "z-aggregate".to_string(),
                    input_rows: 29,
                    ..Default::default()
                },
            ],
            join_strategy_rows: vec![
                NativeJoinStrategyMetricRow { node_id: "a-join".to_string(), ..Default::default() },
                NativeJoinStrategyMetricRow {
                    node_id: "z-join".to_string(),
                    hash_join_build_count: 31,
                    ..Default::default()
                },
            ],
            ..NativeDataFusionExecutionMetrics::default()
        });

        assert_eq!(
            metrics.graph_runtime_rows.iter().map(|row| row.node_id.as_str()).collect::<Vec<_>>(),
            ["a-runtime", "z-runtime"]
        );
        assert_eq!(
            metrics.task_operator_rows.iter().map(|row| row.operator_kind.as_str()).collect::<Vec<_>>(),
            ["aggregate", "source"]
        );
        assert_eq!(metrics.task_operator_rows[1].task_count, 19);
        assert_eq!(metrics.task_operator_rows[1].elapsed_ns, 22);
        assert_eq!(
            metrics.task_runtime_rows.iter().map(|row| row.compiled_task_id.as_str()).collect::<Vec<_>>(),
            ["task-a", "task-z"]
        );
        assert_eq!(metrics.task_runtime_rows[1].operator_kind, "source-updated");
        assert_eq!(metrics.task_runtime_rows[1].elapsed_ns, 28);
        assert_eq!(
            metrics.plan_task_mapping_rows.iter().map(|row| row.plan_node_id.as_deref()).collect::<Vec<_>>(),
            [Some("a-plan"), Some("z-plan")]
        );
        assert_eq!(
            metrics.dynamic_filter_runtime_rows.iter().map(|row| row.dynamic_filter_id.as_str()).collect::<Vec<_>>(),
            ["a-filter", "z-filter"]
        );
        assert_eq!(
            metrics.source_read_runtime_rows.iter().map(|row| row.source_node_id.as_str()).collect::<Vec<_>>(),
            ["a-source", "z-source"]
        );
        assert_eq!(
            metrics.lakehouse_cache_fill_event_snapshot_rows.iter().map(|row| row.outcome.as_str()).collect::<Vec<_>>(),
            ["a-cache", "z-cache"]
        );
        assert_eq!(
            metrics.aggregate_strategy_rows.iter().map(|row| row.node_id.as_str()).collect::<Vec<_>>(),
            ["a-aggregate", "z-aggregate"]
        );
        assert_eq!(metrics.aggregate_strategy_rows[1].input_rows, 36);
        assert_eq!(
            metrics.join_strategy_rows.iter().map(|row| row.node_id.as_str()).collect::<Vec<_>>(),
            ["a-join", "z-join"]
        );
        assert_eq!(metrics.join_strategy_rows[1].hash_join_build_count, 42);
    }
}
