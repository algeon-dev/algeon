//! Native execution metric rows projected from a DataFusion `MetricsSet`.
//!
//! See [`NativeDataFusionExecutionMetrics`] for the per-field unit and
//! cumulative-versus-instantaneous semantics.

use std::collections::BTreeSet;

use datafusion::physical_plan::metrics::MetricsSet;

use super::tsv::{metric_count, optional_observed_metric_count, optional_observed_metric_duration_nanos};

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
pub use graph::{NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow};
use graph::{graph_reservation_rows, graph_runtime_rows};
use labels::max_optional_usize;
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
    /// Cumulative reservation bytes requested over the query lifetime.
    pub reservation_requested_bytes: usize,
    /// Cumulative reservation bytes granted over the query lifetime.
    pub reservation_granted_bytes: usize,
    /// Peak live reserved bytes observed at any instant.
    pub reservation_live_peak_bytes: usize,
    /// Count of denied reservation requests.
    pub reservation_denied_count: usize,
    /// Largest task reservation request granted or applied.
    pub max_task_reservation_bytes: usize,
    /// Largest lower-level task allocator peak, when reported.
    pub max_task_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count, when reported.
    pub max_task_retained_bytes: Option<usize>,
    /// Stable nested graph runtime metric rows.
    pub graph_reservation_rows: Vec<NativeGraphReservationMetricRow>,
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
    /// Query-scoped source cache hits.
    pub source_cache_hits: usize,
    /// Query-scoped source cache misses.
    pub source_cache_misses: usize,
    /// Query-scoped source cache insertions.
    pub source_cache_inserts: usize,
    /// Query-scoped skipped source cache insertions.
    pub source_cache_insert_skipped: usize,
    /// Peak retained source-cache bytes.
    pub source_cache_retained_bytes: usize,
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
    /// Retries caused by reservation failure.
    pub retry_reservation_failure_count: usize,
    /// Adaptive executor retry actions.
    pub adaptive_retry_count: usize,
    /// Adaptive executor spill actions.
    pub adaptive_spill_count: usize,
    /// Adaptive executor downgrade actions.
    pub adaptive_downgrade_count: usize,
    /// Adaptive memory-history lookups.
    pub adaptive_memory_history_lookup_count: usize,
    /// Adaptive memory-history lookup hits.
    pub adaptive_memory_history_hit_count: usize,
    /// Adaptive memory-history lookup misses.
    pub adaptive_memory_history_miss_count: usize,
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
    /// Counted once per source chunk, not once per dynamic-filter route.
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
    /// Query-scoped reusable hash-join cache hits.
    pub hash_join_cache_hits: usize,
    /// Query-scoped reusable hash-join cache misses.
    pub hash_join_cache_misses: usize,
    /// Durable reusable hash-join cache hits.
    pub hash_join_durable_cache_hits: usize,
    /// Durable reusable hash-join cache misses.
    pub hash_join_durable_cache_misses: usize,
    /// Durable reusable hash-join cache insertions.
    pub hash_join_durable_cache_inserts: usize,
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
            reservation_requested_bytes: metric_count(metrics, "reservation_requested_bytes"),
            reservation_granted_bytes: metric_count(metrics, "reservation_granted_bytes"),
            reservation_live_peak_bytes: metric_count(metrics, "reservation_live_peak_bytes"),
            reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
            max_task_reservation_bytes: metric_count(metrics, "max_task_reservation_bytes"),
            max_task_observed_peak_bytes: optional_observed_metric_count(metrics, "max_task_observed_peak_bytes"),
            max_task_retained_bytes: optional_observed_metric_count(metrics, "max_task_retained_bytes"),
            graph_reservation_rows: graph_reservation_rows(metrics),
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
            source_cache_hits: metric_count(metrics, "source_cache_hits"),
            source_cache_misses: metric_count(metrics, "source_cache_misses"),
            source_cache_inserts: metric_count(metrics, "source_cache_inserts"),
            source_cache_insert_skipped: metric_count(metrics, "source_cache_insert_skipped"),
            source_cache_retained_bytes: metric_count(metrics, "source_cache_retained_bytes"),
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
            retry_reservation_failure_count: metric_count(metrics, "retry_reservation_failure_count"),
            adaptive_retry_count: metric_count(metrics, "adaptive_retry_count"),
            adaptive_spill_count: metric_count(metrics, "adaptive_spill_count"),
            adaptive_downgrade_count: metric_count(metrics, "adaptive_downgrade_count"),
            adaptive_memory_history_lookup_count: metric_count(metrics, "adaptive_memory_history_lookup_count"),
            adaptive_memory_history_hit_count: metric_count(metrics, "adaptive_memory_history_hit_count"),
            adaptive_memory_history_miss_count: metric_count(metrics, "adaptive_memory_history_miss_count"),
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
            hash_join_cache_hits: metric_count(metrics, "hash_join_cache_hits"),
            hash_join_cache_misses: metric_count(metrics, "hash_join_cache_misses"),
            hash_join_durable_cache_hits: metric_count(metrics, "hash_join_durable_cache_hits"),
            hash_join_durable_cache_misses: metric_count(metrics, "hash_join_durable_cache_misses"),
            hash_join_durable_cache_inserts: metric_count(metrics, "hash_join_durable_cache_inserts"),
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
        }
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

    /// Classifies the post-decode consumer stage that actually reduced rows.
    pub fn dynamic_filter_post_decode_reduction_stage(&self) -> NativeDynamicFilterPostDecodeReductionStage {
        if self.dynamic_filter_post_decode_rows_filtered == 0 {
            return NativeDynamicFilterPostDecodeReductionStage::None;
        }

        let mut stages = BTreeSet::new();
        let mut classified_rows_filtered = 0_usize;
        if self.dynamic_filter_source_post_decode_rows_filtered > 0 {
            stages.insert("source_post_decode");
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_source_post_decode_rows_filtered);
        }
        if self.dynamic_filter_filter_post_decode_rows_filtered > 0 {
            stages.insert("filter_post_decode");
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_filter_post_decode_rows_filtered);
        }
        if self.dynamic_filter_join_probe_post_decode_rows_filtered > 0 {
            stages.insert("join_probe_post_decode");
            classified_rows_filtered =
                classified_rows_filtered.saturating_add(self.dynamic_filter_join_probe_post_decode_rows_filtered);
        }

        if classified_rows_filtered < self.dynamic_filter_post_decode_rows_filtered {
            stages.insert("unknown");
        }

        if stages.len() > 1 {
            return NativeDynamicFilterPostDecodeReductionStage::MixedPostDecode;
        }

        match stages.iter().next().copied() {
            Some("source_post_decode") => NativeDynamicFilterPostDecodeReductionStage::SourcePostDecode,
            Some("filter_post_decode") => NativeDynamicFilterPostDecodeReductionStage::FilterPostDecode,
            Some("join_probe_post_decode") => NativeDynamicFilterPostDecodeReductionStage::JoinProbePostDecode,
            Some(_) | None => NativeDynamicFilterPostDecodeReductionStage::Unknown,
        }
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
    pub fn merge_aggregate_strategy_rows(&mut self, rows: Vec<NativeAggregateStrategyMetricRow>) {
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
    pub fn merge_join_strategy_rows(&mut self, rows: Vec<NativeJoinStrategyMetricRow>) {
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
    pub fn merge_graph_runtime_rows(&mut self, rows: Vec<NativeGraphRuntimeMetricRow>) {
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

    /// Appends graph reservation rows and restores stable sort order.
    pub fn merge_graph_reservation_rows(&mut self, rows: Vec<NativeGraphReservationMetricRow>) {
        self.graph_reservation_rows.extend(rows);
        self.graph_reservation_rows.sort_by(|left, right| {
            left.node_id
                .cmp(&right.node_id)
                .then_with(|| left.algorithm.cmp(&right.algorithm))
                .then_with(|| left.to_stable_kv_row().cmp(&right.to_stable_kv_row()))
        });
    }

    /// Merges task operator rows by operator kind and keeps stable sort order.
    pub fn merge_task_operator_rows(&mut self, rows: Vec<NativeTaskOperatorMetricRow>) {
        for row in rows {
            if let Some(existing) =
                self.task_operator_rows.iter_mut().find(|existing| existing.operator_kind == row.operator_kind)
            {
                existing.task_count = existing.task_count.saturating_add(row.task_count);
                existing.elapsed_ns = existing.elapsed_ns.saturating_add(row.elapsed_ns);
                existing.reservation_requested_bytes =
                    existing.reservation_requested_bytes.saturating_add(row.reservation_requested_bytes);
                existing.reservation_granted_bytes =
                    existing.reservation_granted_bytes.saturating_add(row.reservation_granted_bytes);
                existing.reservation_released_bytes =
                    existing.reservation_released_bytes.saturating_add(row.reservation_released_bytes);
                existing.reservation_live_peak_bytes =
                    existing.reservation_live_peak_bytes.max(row.reservation_live_peak_bytes);
                existing.max_observed_peak_bytes =
                    max_optional_usize(existing.max_observed_peak_bytes, row.max_observed_peak_bytes);
                existing.max_retained_bytes = max_optional_usize(existing.max_retained_bytes, row.max_retained_bytes);
            } else {
                self.task_operator_rows.push(row);
            }
        }
        self.task_operator_rows.sort_by(|left, right| left.operator_kind.cmp(&right.operator_kind));
    }

    /// Merges task runtime rows by compiled task id and keeps stable sort order.
    pub fn merge_task_runtime_rows(&mut self, rows: Vec<NativeTaskRuntimeMetricRow>) {
        for row in rows {
            if let Some(existing) =
                self.task_runtime_rows.iter_mut().find(|existing| existing.compiled_task_id == row.compiled_task_id)
            {
                existing.operator_kind = row.operator_kind;
                existing.elapsed_ns = existing.elapsed_ns.saturating_add(row.elapsed_ns);
                existing.reservation_requested_bytes =
                    existing.reservation_requested_bytes.saturating_add(row.reservation_requested_bytes);
                existing.reservation_granted_bytes =
                    existing.reservation_granted_bytes.saturating_add(row.reservation_granted_bytes);
                existing.reservation_released_bytes =
                    existing.reservation_released_bytes.saturating_add(row.reservation_released_bytes);
                existing.reservation_live_peak_bytes =
                    existing.reservation_live_peak_bytes.max(row.reservation_live_peak_bytes);
                existing.max_observed_peak_bytes =
                    max_optional_usize(existing.max_observed_peak_bytes, row.max_observed_peak_bytes);
                existing.max_retained_bytes = max_optional_usize(existing.max_retained_bytes, row.max_retained_bytes);
            } else {
                self.task_runtime_rows.push(row);
            }
        }
        self.task_runtime_rows.sort_by(|left, right| left.compiled_task_id.cmp(&right.compiled_task_id));
    }

    /// Appends plan/task mapping rows and restores stable sort order.
    pub fn merge_plan_task_mapping_rows(&mut self, rows: Vec<NativePlanTaskMappingMetricRow>) {
        self.plan_task_mapping_rows.extend(rows);
        self.plan_task_mapping_rows.sort_by_key(|row| row.to_stable_kv_row());
        self.plan_task_mapping_rows.dedup();
    }

    /// Appends dynamic-filter runtime rows and restores stable sort order.
    pub fn merge_dynamic_filter_runtime_rows(&mut self, rows: Vec<NativeDynamicFilterRuntimeMetricRow>) {
        self.dynamic_filter_runtime_rows.extend(rows);
        self.dynamic_filter_runtime_rows.sort_by_key(|row| row.to_stable_kv_row());
    }

    /// Appends source-read runtime rows and restores stable sort order.
    pub fn merge_source_read_runtime_rows(&mut self, rows: Vec<NativeSourceReadRuntimeMetricRow>) {
        self.source_read_runtime_rows.extend(rows);
        self.source_read_runtime_rows.sort_by_key(|row| row.to_stable_kv_row());
    }

    /// Appends lakehouse cache-fill event snapshot rows and restores stable sort order.
    pub fn merge_lakehouse_cache_fill_event_snapshot_rows(
        &mut self,
        rows: Vec<NativeLakehouseCacheFillEventMetricRow>,
    ) {
        self.lakehouse_cache_fill_event_snapshot_rows.extend(rows);
        self.lakehouse_cache_fill_event_snapshot_rows.sort_by_key(|row| row.to_stable_kv_row());
    }
}
