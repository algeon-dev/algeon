//! Native execution metric rows projected from a DataFusion `MetricsSet`.
//!
//! See [`NativeDataFusionExecutionMetrics`] for the per-field unit and
//! cumulative-versus-instantaneous semantics.

use std::collections::{BTreeMap, BTreeSet};

use datafusion::physical_plan::metrics::MetricsSet;

use super::tsv::{metric_count, metric_duration_nanos, optional_observed_metric_count};

const NATIVE_AGGREGATE_STRATEGY_ROWS_METRIC_NAME: &str = "native_aggregate_strategy_rows";
const NATIVE_JOIN_STRATEGY_ROWS_METRIC_NAME: &str = "native_join_strategy_rows";
const NATIVE_GRAPH_RUNTIME_ROWS_METRIC_NAME: &str = "native_graph_runtime_rows";
const NATIVE_TASK_OPERATOR_ROWS_METRIC_NAME: &str = "native_task_operator_rows";
const NATIVE_TASK_RUNTIME_ROWS_METRIC_NAME: &str = "native_task_runtime_rows";
const NATIVE_DYNAMIC_FILTER_RUNTIME_ROWS_METRIC_NAME: &str = "native_dynamic_filter_runtime_rows";
const NATIVE_SOURCE_READ_RUNTIME_ROWS_METRIC_NAME: &str = "native_source_read_runtime_rows";

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
    pub graph_runtime_rows: Vec<NativeGraphRuntimeMetricRow>,
    /// Stable nested task metrics aggregated by operator kind.
    pub task_operator_rows: Vec<NativeTaskOperatorMetricRow>,
    /// Stable nested task runtime metric rows keyed by compiled task id.
    pub task_runtime_rows: Vec<NativeTaskRuntimeMetricRow>,
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
    /// Object-store bytes fetched into host buffers.
    pub source_fetch_bytes: usize,
    /// Count of object-store SourceFetch operations.
    pub source_fetch_count: usize,
    /// Time spent waiting for host-buffer byte budget, in nanoseconds.
    pub host_fetch_budget_wait_time_ns: Option<usize>,
    /// Time spent waiting for SourceFetch concurrency permits, in nanoseconds.
    pub source_fetch_concurrency_wait_time_ns: Option<usize>,
    /// Time spent fetching object-store bytes into host buffers, in nanoseconds.
    pub source_fetch_time_ns: Option<usize>,
    /// Time spent decoding host-buffer Parquet bytes, in nanoseconds.
    pub source_decode_time_ns: Option<usize>,
}

/// Stable query-level classification of measured dynamic-filter reduction.
///
/// The value describes observed runtime reduction, not whether a filter route
/// was produced or reached a consumer. The string values are public TSV
/// sentinels and should remain stable within a report schema version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterMeasuredReductionStatus {
    /// No dynamic-filter runtime evidence was observed.
    Unknown,
    /// Dynamic filters removed source row groups or source-read bytes.
    MeasuredSourcePruned,
    /// Dynamic reader predicates reduced materialized source rows.
    MeasuredReaderRowsReduced,
    /// Post-decode dynamic filters reduced rows after source decode.
    MeasuredPostDecodeReduced,
    /// Dynamic-filter routes ran or were considered but no measured reduction was observed.
    MeasuredNoReduction,
}

impl NativeDynamicFilterMeasuredReductionStatus {
    /// Stable TSV label for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::MeasuredSourcePruned => "measured_source_pruned",
            Self::MeasuredReaderRowsReduced => "measured_reader_rows_reduced",
            Self::MeasuredPostDecodeReduced => "measured_post_decode_reduced",
            Self::MeasuredNoReduction => "measured_no_reduction",
        }
    }
}

/// Stable query-level classification of which work dynamic filters avoided.
///
/// This is stricter than [`NativeDynamicFilterMeasuredReductionStatus`]:
/// it preserves where the strongest observed reduction occurred so admission
/// and gate code can avoid treating route lifecycle events as wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterWorkAvoidanceStatus {
    /// No dynamic-filter runtime evidence was observed.
    Unknown,
    /// Source row groups or source bytes were pruned before read/decode work.
    SourceReadPruned,
    /// Source-reader predicates reduced materialized source rows.
    SourceReaderRowsReduced,
    /// Source-adjacent post-decode filters reduced rows before downstream operators.
    SourcePostDecodeRowsReduced,
    /// Filter-operator post-decode filters reduced rows before downstream operators.
    FilterPostDecodeRowsReduced,
    /// Hash-join-probe post-decode filters reduced rows at the probe boundary.
    JoinProbePostDecodeRowsReduced,
    /// More than one post-decode consumer stage reduced rows.
    MixedPostDecodeRowsReduced,
    /// Dynamic-filter routes ran or were considered but no measured reduction was observed.
    MeasuredNoReduction,
}

impl NativeDynamicFilterWorkAvoidanceStatus {
    /// Stable TSV label for this status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::SourceReadPruned => "source_read_pruned",
            Self::SourceReaderRowsReduced => "source_reader_rows_reduced",
            Self::SourcePostDecodeRowsReduced => "source_post_decode_rows_reduced",
            Self::FilterPostDecodeRowsReduced => "filter_post_decode_rows_reduced",
            Self::JoinProbePostDecodeRowsReduced => "join_probe_post_decode_rows_reduced",
            Self::MixedPostDecodeRowsReduced => "mixed_post_decode_rows_reduced",
            Self::MeasuredNoReduction => "measured_no_reduction",
        }
    }
}

/// Stable query-level classification of where post-decode row reduction occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDynamicFilterPostDecodeReductionStage {
    /// No post-decode row reduction was measured.
    None,
    /// Post-decode reduction happened at a local source after decode.
    SourcePostDecode,
    /// Post-decode reduction happened at a filter operator before downstream work.
    FilterPostDecode,
    /// Post-decode reduction happened at a hash-join probe boundary.
    JoinProbePostDecode,
    /// More than one post-decode consumer stage reduced rows.
    MixedPostDecode,
    /// Top-level counters show reduction but nested rows lack usable stage evidence.
    Unknown,
}

impl NativeDynamicFilterPostDecodeReductionStage {
    /// Stable TSV label for this stage.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SourcePostDecode => "source_post_decode",
            Self::FilterPostDecode => "filter_post_decode",
            Self::JoinProbePostDecode => "join_probe_post_decode",
            Self::MixedPostDecode => "mixed_post_decode",
            Self::Unknown => "unknown",
        }
    }
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
            resident_dataframe_peak_bytes: optional_observed_metric_count(metrics, "resident_dataframe_peak_bytes"),
            allocator_high_water_bytes: optional_observed_metric_count(metrics, "allocator_high_water_bytes"),
            reservation_requested_bytes: metric_count(metrics, "reservation_requested_bytes"),
            reservation_granted_bytes: metric_count(metrics, "reservation_granted_bytes"),
            reservation_live_peak_bytes: metric_count(metrics, "reservation_live_peak_bytes"),
            reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
            max_task_reservation_bytes: metric_count(metrics, "max_task_reservation_bytes"),
            max_task_observed_peak_bytes: optional_observed_metric_count(metrics, "max_task_observed_peak_bytes"),
            max_task_retained_bytes: optional_observed_metric_count(metrics, "max_task_retained_bytes"),
            graph_runtime_rows: graph_runtime_rows(metrics),
            task_operator_rows: task_operator_rows(metrics),
            task_runtime_rows: task_runtime_rows(metrics),
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
            source_fetch_bytes: metric_count(metrics, "source_fetch_bytes"),
            source_fetch_count: metric_count(metrics, "source_fetch_count"),
            host_fetch_budget_wait_time_ns: metric_duration_nanos(metrics, "host_fetch_budget_wait_time"),
            source_fetch_concurrency_wait_time_ns: metric_duration_nanos(metrics, "source_fetch_concurrency_wait_time"),
            source_fetch_time_ns: metric_duration_nanos(metrics, "source_fetch_time"),
            source_decode_time_ns: metric_duration_nanos(metrics, "source_decode_time"),
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
}

/// Stable nested metric row for one graph algorithm execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeGraphRuntimeMetricRow {
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
    pub input_rows: usize,
    /// Input column count.
    pub input_columns: usize,
    /// Output row count.
    pub output_rows: usize,
    /// Output column count.
    pub output_columns: usize,
    /// Output device bytes, when known.
    pub output_device_bytes: Option<usize>,
    /// Runtime elapsed nanoseconds.
    pub elapsed_ns: usize,
    /// GPU-resident input handoffs for this algorithm.
    pub gpu_resident_input_handoffs: usize,
    /// Host-to-device uploads for this algorithm.
    pub host_to_device_uploads: usize,
    /// Device-to-host materializations for this algorithm.
    pub device_to_host_materializations: usize,
}

/// Stable nested metric row for one dynamic-filter runtime event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeDynamicFilterRuntimeMetricRow {
    /// Plan-local dynamic-filter route id.
    pub dynamic_filter_id: String,
    /// Event kind, such as `produced`, `skipped`, `applied`,
    /// `reader_predicate_rows`, or `applied_post_decode`.
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
    pub payload_value_count: Option<usize>,
    /// Inclusive integer-like minimum, when present.
    pub payload_min_i64: Option<i64>,
    /// Inclusive integer-like maximum, when present.
    pub payload_max_i64: Option<i64>,
    /// Producer rows observed while creating the payload, when known.
    pub producer_rows: Option<usize>,
    /// Stable reason for skipped events.
    pub skipped_reason: Option<String>,
    /// Source row groups before application.
    pub selected_row_groups_before: Option<usize>,
    /// Source row groups after application.
    pub selected_row_groups_after: Option<usize>,
    /// Source-read bytes before application.
    pub selected_source_bytes_before: Option<usize>,
    /// Source-read bytes after application.
    pub selected_source_bytes_after: Option<usize>,
    /// Source file index for source-reader row evidence.
    pub source_file_index: Option<usize>,
    /// Source chunk index for source-reader row evidence.
    pub source_chunk_index: Option<usize>,
    /// Selected source rows before the reader predicate was applied.
    pub reader_predicate_rows_before: Option<usize>,
    /// Materialized source rows after the reader predicate was applied.
    pub reader_predicate_rows_after: Option<usize>,
    /// Source rows removed by the reader predicate.
    pub reader_predicate_rows_filtered: usize,
    /// Materialized rows before a post-decode dynamic-filter stage.
    pub post_decode_rows_before: Option<usize>,
    /// Materialized rows after a post-decode dynamic-filter stage.
    pub post_decode_rows_after: Option<usize>,
    /// Materialized rows removed by a post-decode dynamic-filter stage.
    pub post_decode_rows_filtered: usize,
    /// Row groups removed by this application.
    pub row_groups_pruned: usize,
    /// Source-read bytes removed by this application.
    pub source_bytes_pruned: usize,
}

impl NativeDynamicFilterRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
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
            optional_usize(self.payload_value_count),
            optional_i64(self.payload_min_i64),
            optional_i64(self.payload_max_i64),
            optional_usize(self.producer_rows),
            optional_str(self.skipped_reason.as_deref()),
            optional_usize(self.selected_row_groups_before),
            optional_usize(self.selected_row_groups_after),
            optional_usize(self.selected_source_bytes_before),
            optional_usize(self.selected_source_bytes_after),
            optional_usize(self.source_file_index),
            optional_usize(self.source_chunk_index),
            optional_usize(self.reader_predicate_rows_before),
            optional_usize(self.reader_predicate_rows_after),
            self.reader_predicate_rows_filtered,
            optional_usize(self.post_decode_rows_before),
            optional_usize(self.post_decode_rows_after),
            self.post_decode_rows_filtered,
            self.row_groups_pruned,
            self.source_bytes_pruned,
        )
    }
}

/// Stable nested metric row for one local-Parquet source read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeSourceReadRuntimeMetricRow {
    /// Native source node id.
    pub source_node_id: String,
    /// Source read shape, for example `chunk` or `chunk_batch`.
    pub read_kind: String,
    /// Parquet read strategy used by the chunk or batch.
    pub read_strategy: String,
    /// Batch index within the source stream, or chunk index for singleton reads.
    pub batch_index: usize,
    /// Number of source chunks represented by this read.
    pub chunk_count: usize,
    /// First source chunk index represented by this read.
    pub first_chunk_index: usize,
    /// Last source chunk index represented by this read.
    pub last_chunk_index: usize,
    /// First source file index represented by this read.
    pub first_file_index: usize,
    /// Last source file index represented by this read.
    pub last_file_index: usize,
    /// Number of distinct files represented by this read.
    pub file_count: usize,
    /// Selected row groups represented by this read.
    pub row_group_count: usize,
    /// Number of physical read columns per represented file.
    pub read_columns_per_file: usize,
    /// Sum of physical read-column selections across represented files.
    pub read_columns_total: usize,
    /// Sum of known file bytes for represented files, when available.
    pub known_file_bytes: usize,
    /// Best available source-read byte estimate for represented chunks.
    pub estimated_source_read_bytes: usize,
    /// Sum of known selected row-group bytes, when available.
    pub known_row_group_bytes: usize,
    /// Sum of known projected compressed bytes, when available.
    pub known_projected_compressed_bytes: usize,
    /// Sum of known projected uncompressed bytes, when available.
    pub known_projected_uncompressed_bytes: usize,
    /// Sum of known pure-filter compressed bytes, when available.
    pub known_pure_filter_compressed_bytes: usize,
    /// Sum of known pure-filter uncompressed bytes, when available.
    pub known_pure_filter_uncompressed_bytes: usize,
    /// Number of output rows materialized by the read after reader predicates.
    pub output_rows: usize,
    /// Whether this read carried a source reader predicate.
    pub reader_predicate: bool,
    /// Number of dynamic-filter reader predicate routes attached to this read.
    pub reader_dynamic_filter_count: usize,
    /// Stable diagnostic reason for the source chunk batching decision.
    pub source_batching_reason: String,
    /// Elapsed time spent in the source read, in nanoseconds.
    pub elapsed_ns: usize,
}

impl NativeSourceReadRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "source_node_id={},read_kind={},read_strategy={},batch_index={},chunk_count={},first_chunk_index={},last_chunk_index={},first_file_index={},last_file_index={},file_count={},row_group_count={},read_columns_per_file={},read_columns_total={},known_file_bytes={},estimated_source_read_bytes={},known_row_group_bytes={},known_projected_compressed_bytes={},known_projected_uncompressed_bytes={},known_pure_filter_compressed_bytes={},known_pure_filter_uncompressed_bytes={},output_rows={},reader_predicate={},reader_dynamic_filter_count={},source_batching_reason={},elapsed_ns={}",
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
            self.elapsed_ns,
        )
    }
}

impl NativeGraphRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},algorithm={},runtime_outcome={},input_boundary={},output_boundary={},construction_policy={},source_column={},destination_column={},weight_column={},directed={},renumber={},algorithm_parameters={},algorithm_metadata={},input_rows={},input_columns={},output_rows={},output_columns={},output_device_bytes={},elapsed_ns={},gpu_resident_input_handoffs={},host_to_device_uploads={},device_to_host_materializations={}",
            self.node_id,
            self.algorithm,
            self.runtime_outcome,
            self.input_boundary,
            self.output_boundary,
            self.construction_policy,
            self.source_column,
            self.destination_column,
            optional_str(self.weight_column.as_deref()),
            self.directed,
            self.renumber,
            self.algorithm_parameters,
            self.algorithm_metadata,
            self.input_rows,
            self.input_columns,
            self.output_rows,
            self.output_columns,
            optional_usize(self.output_device_bytes),
            self.elapsed_ns,
            self.gpu_resident_input_handoffs,
            self.host_to_device_uploads,
            self.device_to_host_materializations,
        )
    }
}

/// Aggregated task metrics by native operator kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTaskOperatorMetricRow {
    /// Operator-kind label shared by aggregated tasks.
    pub operator_kind: String,
    /// Number of tasks observed for this operator kind.
    pub task_count: usize,
    /// Cumulative elapsed nanoseconds across tasks.
    pub elapsed_ns: usize,
    /// Cumulative reservation bytes requested.
    pub reservation_requested_bytes: usize,
    /// Cumulative reservation bytes granted.
    pub reservation_granted_bytes: usize,
    /// Cumulative reservation bytes released.
    pub reservation_released_bytes: usize,
    /// Peak live reserved bytes for this operator kind.
    pub reservation_live_peak_bytes: usize,
    /// Largest lower-level observed allocator peak for this operator kind.
    pub max_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count for this operator kind.
    pub max_retained_bytes: Option<usize>,
}

impl NativeTaskOperatorMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "operator_kind={},task_count={},elapsed_ns={},reservation_requested_bytes={},reservation_granted_bytes={},reservation_released_bytes={},reservation_live_peak_bytes={},max_observed_peak_bytes={},max_retained_bytes={}",
            self.operator_kind,
            self.task_count,
            self.elapsed_ns,
            self.reservation_requested_bytes,
            self.reservation_granted_bytes,
            self.reservation_released_bytes,
            self.reservation_live_peak_bytes,
            optional_usize(self.max_observed_peak_bytes),
            optional_usize(self.max_retained_bytes)
        )
    }
}

/// Runtime metric row for one compiled pipeline task.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeTaskRuntimeMetricRow {
    /// Stable compiled task id.
    pub compiled_task_id: String,
    /// Operator-kind label for the task.
    pub operator_kind: String,
    /// Cumulative elapsed task time in nanoseconds.
    pub elapsed_ns: usize,
    /// Cumulative reservation bytes requested by the task.
    pub reservation_requested_bytes: usize,
    /// Cumulative reservation bytes granted to the task.
    pub reservation_granted_bytes: usize,
    /// Cumulative reservation bytes released by the task.
    pub reservation_released_bytes: usize,
    /// Peak live reserved bytes for the task.
    pub reservation_live_peak_bytes: usize,
    /// Largest lower-level observed allocator peak for the task.
    pub max_observed_peak_bytes: Option<usize>,
    /// Largest lower-level retained byte count for the task.
    pub max_retained_bytes: Option<usize>,
}

impl NativeTaskRuntimeMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "compiled_task_id={},operator_kind={},elapsed_ns={},reservation_requested_bytes={},reservation_granted_bytes={},reservation_released_bytes={},reservation_live_peak_bytes={},max_observed_peak_bytes={},max_retained_bytes={}",
            self.compiled_task_id,
            self.operator_kind,
            self.elapsed_ns,
            self.reservation_requested_bytes,
            self.reservation_granted_bytes,
            self.reservation_released_bytes,
            self.reservation_live_peak_bytes,
            optional_usize(self.max_observed_peak_bytes),
            optional_usize(self.max_retained_bytes)
        )
    }
}

/// Aggregate strategy diagnostics with runtime counters merged by aggregate site.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeAggregateStrategyMetricRow {
    /// Native aggregate node id, or a stage-local fallback label.
    pub node_id: String,
    /// Strategy selected for this aggregate stage.
    pub selected_strategy: String,
    /// Aggregate execution stage label.
    pub execution_stage: String,
    /// Native memory mode label.
    pub memory_mode: String,
    /// Number of group-key expressions.
    pub group_key_count: usize,
    /// Stable group-key expression summary.
    pub group_key_exprs: String,
    /// Stable group-key dtype summary.
    pub group_key_dtypes: String,
    /// Stable aggregate function summary.
    pub aggregate_functions: String,
    /// Whether this aggregate row carries a floating statistical aggregate contract.
    pub floating_statistical_aggregate: String,
    /// Selected floating statistical aggregate execution mode.
    pub statistical_aggregate_execution_mode: String,
    /// Required tolerance contract for floating statistical parity.
    pub statistical_aggregate_tolerance_contract: String,
    /// Floating statistical aggregate functions covered by the contract.
    pub statistical_aggregate_functions: String,
    /// Degrees-of-freedom value used by statistical aggregate kernels.
    pub statistical_aggregate_ddof: String,
    /// Migration projection for logical `STDDEV_SAMP`.
    pub stddev_samp_execution_mode: String,
    /// Migration projection for logical `STDDEV_SAMP` ddof.
    pub stddev_samp_ddof: String,
    /// Count of aggregate filter predicates.
    pub aggregate_filter_count: usize,
    /// Count of distinct aggregate expressions.
    pub distinct_aggregate_count: usize,
    /// Input rows consumed by this aggregate stage.
    pub input_rows: usize,
    /// Output groups produced by this aggregate stage.
    pub output_group_count: usize,
    /// `output_group_count / input_rows` scaled by 1,000,000.
    pub reduction_ratio_ppm: usize,
    /// Chunk rows for partial stages, or zero for non-partial stages.
    pub chunk_row_count: usize,
    /// Source chunk count visible at this decision point.
    pub chunk_count: usize,
    /// Partial aggregate count visible at this decision point.
    pub partial_aggregate_count: usize,
    /// Whether this partial aggregate consumed source-adjacent stage output.
    pub source_adjacent_partial: String,
    /// Source row groups feeding this partial aggregate, or `unknown`.
    pub source_chunk_row_group_count: String,
    /// Whether the source chunk reads a whole file with unknown row groups.
    pub source_chunk_reads_all_row_groups: String,
    /// Known source row count for this chunk before source-adjacent stages.
    pub source_chunk_known_row_count: String,
    /// Best available source-read byte estimate for this chunk.
    pub source_chunk_estimated_source_read_bytes: String,
    /// Whether chunk sizing relied on file-size estimates.
    pub source_chunk_size_estimate_based: String,
    /// Source-adjacent partition strategy, or `unknown`/`none`.
    pub source_adjacent_partition_strategy: String,
    /// Planned source-adjacent partition count, or `unknown`.
    pub source_adjacent_partition_count: String,
    /// Non-empty source-adjacent partition count, or `unknown`.
    pub source_adjacent_non_empty_partition_count: String,
    /// Group-key column count used for source-adjacent partitioning, or `unknown`.
    pub source_adjacent_partition_key_count: String,
    /// Largest number of chunk fragments feeding one source-adjacent partition.
    pub source_adjacent_max_partition_fan_in: String,
    /// Planned partition count for partitioned final merge, or `unknown`/`none`.
    pub partitioned_merge_partition_count: String,
    /// Non-empty partition count for partitioned final merge, or `unknown`/`none`.
    pub partitioned_merge_non_empty_partition_count: String,
    /// Largest fan-in for one partitioned final-merge bucket, or `unknown`/`none`.
    pub partitioned_merge_max_partition_fan_in: String,
    /// Strategy-selection elapsed nanoseconds.
    pub selection_elapsed_ns: usize,
    /// Key-encoding elapsed nanoseconds.
    pub encode_elapsed_ns: usize,
    /// Groupby elapsed nanoseconds.
    pub groupby_elapsed_ns: usize,
    /// Key-decoding elapsed nanoseconds.
    pub decode_elapsed_ns: usize,
    /// Total operator elapsed nanoseconds charged to this strategy.
    pub strategy_operator_elapsed_ns: usize,
    /// Stable structural fingerprint used for cross-node aggregate evidence reuse.
    pub aggregate_structural_key: String,
    /// Evidence source used by partial aggregate cardinality cost models.
    pub prior_cardinality_evidence_scope: String,
    /// Runtime cost-model name used for encoded-key partial selection.
    pub encoded_key_partial_cost_model: String,
    /// Runtime cost-model status for encoded-key partial selection.
    pub encoded_key_partial_cost_model_status: String,
    /// Stable reason for the encoded-key partial runtime selection decision.
    pub encoded_key_partial_selection_reason: String,
    /// Encoded-key partial reduction-ratio threshold scaled by 1,000,000.
    pub encoded_key_partial_reduction_ratio_threshold_ppm: usize,
    /// Prior partial reduction ratio used by the encoded-key cost model.
    pub encoded_key_partial_prior_reduction_ratio_ppm: usize,
    /// Strategy used by the prior partial evidence row.
    pub encoded_key_partial_prior_strategy: String,
    /// Exact-cardinality probe status for a single-key partial aggregate.
    pub exact_cardinality_probe_status: String,
    /// Group-key column measured by the exact-cardinality probe.
    pub exact_cardinality_probe_key: String,
    /// Rows consumed by the exact-cardinality probe.
    pub exact_cardinality_probe_input_rows: usize,
    /// Non-null distinct values measured by the exact-cardinality probe.
    pub exact_cardinality_probe_non_null_distinct_count: String,
    /// Null rows observed by the exact-cardinality probe.
    pub exact_cardinality_probe_null_count: String,
    /// Group count measured by the exact-cardinality probe, including the null group.
    pub exact_cardinality_probe_num_groups: String,
    /// Exact-cardinality probe reduction ratio scaled by 1,000,000.
    pub exact_cardinality_probe_reduction_ratio_ppm: String,
    /// Exact-cardinality probe elapsed nanoseconds.
    pub exact_cardinality_probe_elapsed_ns: usize,
    /// Stable local rejection reason for prepared sort groupby.
    pub prepared_sort_groupby_rejected_reason: String,
    /// Prepared sort groupby contract status.
    pub prepared_sort_groupby_contract_status: String,
    /// Required contract label for prepared sort groupby.
    pub prepared_sort_groupby_required_contract: String,
    /// Runtime cost-model name used for prepared sort groupby selection.
    pub prepared_sort_groupby_cost_model: String,
    /// Runtime cost-model status for prepared sort groupby selection.
    pub prepared_sort_groupby_cost_model_status: String,
    /// Stable reason for the prepared sort groupby runtime selection decision.
    pub prepared_sort_groupby_selection_reason: String,
    /// Exact prepared group count, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_num_groups: String,
    /// Exact grouped-key output bytes, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_grouped_key_output_bytes: String,
    /// Exact maximum group size, or `unknown` when the prepare step did not run.
    pub prepared_sort_groupby_max_group_size: String,
    /// Stable candidate rejection summary.
    pub rejected_candidate_strategies: String,
}

impl NativeAggregateStrategyMetricRow {
    /// Returns true for the expensive first-observation path where libcudf
    /// groupby is selected to collect cardinality evidence for future chunks.
    pub fn is_libcudf_runtime_probe(&self) -> bool {
        self.execution_stage == "partial"
            && self.selected_strategy == "libcudf_groupby"
            && self.prepared_sort_groupby_cost_model_status == "selected_libcudf_runtime_probe"
            && self.prepared_sort_groupby_selection_reason == "runtime_cardinality_probe_required"
    }

    /// Returns true when a single-key exact-cardinality probe was executed.
    pub fn has_exact_cardinality_probe(&self) -> bool {
        self.execution_stage == "partial" && self.exact_cardinality_probe_status == "measured"
    }

    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},selected_strategy={},execution_stage={},memory_mode={},group_key_count={},group_key_exprs={},group_key_dtypes={},aggregate_functions={},floating_statistical_aggregate={},statistical_aggregate_execution_mode={},statistical_aggregate_tolerance_contract={},statistical_aggregate_functions={},statistical_aggregate_ddof={},stddev_samp_execution_mode={},stddev_samp_ddof={},aggregate_filter_count={},distinct_aggregate_count={},input_rows={},output_group_count={},reduction_ratio_ppm={},chunk_row_count={},chunk_count={},partial_aggregate_count={},source_adjacent_partial={},source_chunk_row_group_count={},source_chunk_reads_all_row_groups={},source_chunk_known_row_count={},source_chunk_estimated_source_read_bytes={},source_chunk_size_estimate_based={},source_adjacent_partition_strategy={},source_adjacent_partition_count={},source_adjacent_non_empty_partition_count={},source_adjacent_partition_key_count={},source_adjacent_max_partition_fan_in={},partitioned_merge_partition_count={},partitioned_merge_non_empty_partition_count={},partitioned_merge_max_partition_fan_in={},selection_elapsed_ns={},encode_elapsed_ns={},groupby_elapsed_ns={},decode_elapsed_ns={},strategy_operator_elapsed_ns={},aggregate_structural_key={},prior_cardinality_evidence_scope={},encoded_key_partial_cost_model={},encoded_key_partial_cost_model_status={},encoded_key_partial_selection_reason={},encoded_key_partial_reduction_ratio_threshold_ppm={},encoded_key_partial_prior_reduction_ratio_ppm={},encoded_key_partial_prior_strategy={},exact_cardinality_probe_status={},exact_cardinality_probe_key={},exact_cardinality_probe_input_rows={},exact_cardinality_probe_non_null_distinct_count={},exact_cardinality_probe_null_count={},exact_cardinality_probe_num_groups={},exact_cardinality_probe_reduction_ratio_ppm={},exact_cardinality_probe_elapsed_ns={},prepared_sort_groupby_rejected_reason={},prepared_sort_groupby_contract_status={},prepared_sort_groupby_required_contract={},prepared_sort_groupby_cost_model={},prepared_sort_groupby_cost_model_status={},prepared_sort_groupby_selection_reason={},prepared_sort_groupby_num_groups={},prepared_sort_groupby_grouped_key_output_bytes={},prepared_sort_groupby_max_group_size={},rejected_candidate_strategies={}",
            self.node_id,
            self.selected_strategy,
            self.execution_stage,
            self.memory_mode,
            self.group_key_count,
            self.group_key_exprs,
            self.group_key_dtypes,
            self.aggregate_functions,
            self.floating_statistical_aggregate,
            self.statistical_aggregate_execution_mode,
            self.statistical_aggregate_tolerance_contract,
            self.statistical_aggregate_functions,
            self.statistical_aggregate_ddof,
            self.stddev_samp_execution_mode,
            self.stddev_samp_ddof,
            self.aggregate_filter_count,
            self.distinct_aggregate_count,
            self.input_rows,
            self.output_group_count,
            self.reduction_ratio_ppm,
            self.chunk_row_count,
            self.chunk_count,
            self.partial_aggregate_count,
            self.source_adjacent_partial,
            self.source_chunk_row_group_count,
            self.source_chunk_reads_all_row_groups,
            self.source_chunk_known_row_count,
            self.source_chunk_estimated_source_read_bytes,
            self.source_chunk_size_estimate_based,
            self.source_adjacent_partition_strategy,
            self.source_adjacent_partition_count,
            self.source_adjacent_non_empty_partition_count,
            self.source_adjacent_partition_key_count,
            self.source_adjacent_max_partition_fan_in,
            self.partitioned_merge_partition_count,
            self.partitioned_merge_non_empty_partition_count,
            self.partitioned_merge_max_partition_fan_in,
            self.selection_elapsed_ns,
            self.encode_elapsed_ns,
            self.groupby_elapsed_ns,
            self.decode_elapsed_ns,
            self.strategy_operator_elapsed_ns,
            self.aggregate_structural_key,
            self.prior_cardinality_evidence_scope,
            self.encoded_key_partial_cost_model,
            self.encoded_key_partial_cost_model_status,
            self.encoded_key_partial_selection_reason,
            self.encoded_key_partial_reduction_ratio_threshold_ppm,
            self.encoded_key_partial_prior_reduction_ratio_ppm,
            self.encoded_key_partial_prior_strategy,
            self.exact_cardinality_probe_status,
            self.exact_cardinality_probe_key,
            self.exact_cardinality_probe_input_rows,
            self.exact_cardinality_probe_non_null_distinct_count,
            self.exact_cardinality_probe_null_count,
            self.exact_cardinality_probe_num_groups,
            self.exact_cardinality_probe_reduction_ratio_ppm,
            self.exact_cardinality_probe_elapsed_ns,
            self.prepared_sort_groupby_rejected_reason,
            self.prepared_sort_groupby_contract_status,
            self.prepared_sort_groupby_required_contract,
            self.prepared_sort_groupby_cost_model,
            self.prepared_sort_groupby_cost_model_status,
            self.prepared_sort_groupby_selection_reason,
            self.prepared_sort_groupby_num_groups,
            self.prepared_sort_groupby_grouped_key_output_bytes,
            self.prepared_sort_groupby_max_group_size,
            self.rejected_candidate_strategies,
        )
    }

    fn same_site(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.selected_strategy == other.selected_strategy
            && self.execution_stage == other.execution_stage
            && self.memory_mode == other.memory_mode
            && self.group_key_count == other.group_key_count
            && self.group_key_exprs == other.group_key_exprs
            && self.group_key_dtypes == other.group_key_dtypes
            && self.aggregate_functions == other.aggregate_functions
            && self.floating_statistical_aggregate == other.floating_statistical_aggregate
            && self.statistical_aggregate_execution_mode == other.statistical_aggregate_execution_mode
            && self.statistical_aggregate_tolerance_contract == other.statistical_aggregate_tolerance_contract
            && self.statistical_aggregate_functions == other.statistical_aggregate_functions
            && self.statistical_aggregate_ddof == other.statistical_aggregate_ddof
            && self.stddev_samp_execution_mode == other.stddev_samp_execution_mode
            && self.stddev_samp_ddof == other.stddev_samp_ddof
            && self.aggregate_filter_count == other.aggregate_filter_count
            && self.distinct_aggregate_count == other.distinct_aggregate_count
            && self.source_adjacent_partial == other.source_adjacent_partial
            && self.source_chunk_row_group_count == other.source_chunk_row_group_count
            && self.source_chunk_reads_all_row_groups == other.source_chunk_reads_all_row_groups
            && self.source_chunk_known_row_count == other.source_chunk_known_row_count
            && self.source_chunk_estimated_source_read_bytes == other.source_chunk_estimated_source_read_bytes
            && self.source_chunk_size_estimate_based == other.source_chunk_size_estimate_based
            && self.source_adjacent_partition_strategy == other.source_adjacent_partition_strategy
            && self.source_adjacent_partition_count == other.source_adjacent_partition_count
            && self.source_adjacent_non_empty_partition_count == other.source_adjacent_non_empty_partition_count
            && self.source_adjacent_partition_key_count == other.source_adjacent_partition_key_count
            && self.source_adjacent_max_partition_fan_in == other.source_adjacent_max_partition_fan_in
            && self.partitioned_merge_partition_count == other.partitioned_merge_partition_count
            && self.partitioned_merge_non_empty_partition_count == other.partitioned_merge_non_empty_partition_count
            && self.partitioned_merge_max_partition_fan_in == other.partitioned_merge_max_partition_fan_in
            && self.aggregate_structural_key == other.aggregate_structural_key
            && self.prior_cardinality_evidence_scope == other.prior_cardinality_evidence_scope
            && self.encoded_key_partial_cost_model == other.encoded_key_partial_cost_model
            && self.encoded_key_partial_cost_model_status == other.encoded_key_partial_cost_model_status
            && self.encoded_key_partial_selection_reason == other.encoded_key_partial_selection_reason
            && self.encoded_key_partial_reduction_ratio_threshold_ppm
                == other.encoded_key_partial_reduction_ratio_threshold_ppm
            && self.encoded_key_partial_prior_reduction_ratio_ppm == other.encoded_key_partial_prior_reduction_ratio_ppm
            && self.encoded_key_partial_prior_strategy == other.encoded_key_partial_prior_strategy
            && self.exact_cardinality_probe_status == other.exact_cardinality_probe_status
            && self.exact_cardinality_probe_key == other.exact_cardinality_probe_key
            && self.exact_cardinality_probe_non_null_distinct_count
                == other.exact_cardinality_probe_non_null_distinct_count
            && self.exact_cardinality_probe_null_count == other.exact_cardinality_probe_null_count
            && self.exact_cardinality_probe_num_groups == other.exact_cardinality_probe_num_groups
            && self.exact_cardinality_probe_reduction_ratio_ppm == other.exact_cardinality_probe_reduction_ratio_ppm
            && self.prepared_sort_groupby_rejected_reason == other.prepared_sort_groupby_rejected_reason
            && self.prepared_sort_groupby_contract_status == other.prepared_sort_groupby_contract_status
            && self.prepared_sort_groupby_required_contract == other.prepared_sort_groupby_required_contract
            && self.prepared_sort_groupby_cost_model == other.prepared_sort_groupby_cost_model
            && self.prepared_sort_groupby_cost_model_status == other.prepared_sort_groupby_cost_model_status
            && self.prepared_sort_groupby_selection_reason == other.prepared_sort_groupby_selection_reason
            && self.rejected_candidate_strategies == other.rejected_candidate_strategies
    }

    fn merge_runtime_from(&mut self, other: Self) {
        self.input_rows = self.input_rows.saturating_add(other.input_rows);
        self.output_group_count = self.output_group_count.saturating_add(other.output_group_count);
        self.chunk_row_count = self.chunk_row_count.saturating_add(other.chunk_row_count);
        self.chunk_count = self.chunk_count.max(other.chunk_count);
        self.partial_aggregate_count = self.partial_aggregate_count.max(other.partial_aggregate_count);
        self.selection_elapsed_ns = self.selection_elapsed_ns.saturating_add(other.selection_elapsed_ns);
        self.encode_elapsed_ns = self.encode_elapsed_ns.saturating_add(other.encode_elapsed_ns);
        self.groupby_elapsed_ns = self.groupby_elapsed_ns.saturating_add(other.groupby_elapsed_ns);
        self.decode_elapsed_ns = self.decode_elapsed_ns.saturating_add(other.decode_elapsed_ns);
        self.strategy_operator_elapsed_ns =
            self.strategy_operator_elapsed_ns.saturating_add(other.strategy_operator_elapsed_ns);
        self.exact_cardinality_probe_input_rows =
            self.exact_cardinality_probe_input_rows.saturating_add(other.exact_cardinality_probe_input_rows);
        self.exact_cardinality_probe_elapsed_ns =
            self.exact_cardinality_probe_elapsed_ns.saturating_add(other.exact_cardinality_probe_elapsed_ns);
        self.prepared_sort_groupby_num_groups = merge_numeric_string_fact_sum(
            &self.prepared_sort_groupby_num_groups,
            &other.prepared_sort_groupby_num_groups,
        );
        self.prepared_sort_groupby_grouped_key_output_bytes = merge_numeric_string_fact_sum(
            &self.prepared_sort_groupby_grouped_key_output_bytes,
            &other.prepared_sort_groupby_grouped_key_output_bytes,
        );
        self.prepared_sort_groupby_max_group_size = merge_numeric_string_fact_max(
            &self.prepared_sort_groupby_max_group_size,
            &other.prepared_sort_groupby_max_group_size,
        );
        self.reduction_ratio_ppm = if self.input_rows == 0 {
            0
        } else {
            self.output_group_count.saturating_mul(1_000_000).saturating_div(self.input_rows)
        };
    }

    fn sort_key(&self) -> (&str, &str, &str, &str, &str) {
        (
            self.node_id.as_str(),
            self.execution_stage.as_str(),
            self.selected_strategy.as_str(),
            self.group_key_dtypes.as_str(),
            self.aggregate_functions.as_str(),
        )
    }
}

fn merge_numeric_string_fact_sum(left: &str, right: &str) -> String {
    match (left.parse::<usize>(), right.parse::<usize>()) {
        (Ok(left), Ok(right)) => left.saturating_add(right).to_string(),
        _ => "unknown".to_string(),
    }
}

fn merge_numeric_string_fact_max(left: &str, right: &str) -> String {
    match (left.parse::<usize>(), right.parse::<usize>()) {
        (Ok(left), Ok(right)) => left.max(right).to_string(),
        _ => "unknown".to_string(),
    }
}

/// Join strategy diagnostics with runtime counters merged by join site.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeJoinStrategyMetricRow {
    /// Native join node id.
    pub node_id: String,
    /// Planner-selected join strategy label.
    pub join_strategy: String,
    /// Runtime execution strategy label.
    pub execution_strategy: String,
    /// Join type label.
    pub join_type: String,
    /// Build-side orientation label.
    pub build_side: String,
    /// Number of equality key pairs.
    pub key_count: usize,
    /// Estimated build-side rows, if known.
    pub estimated_build_rows: Option<usize>,
    /// Observed build-side rows, if known.
    pub observed_build_rows: Option<usize>,
    /// Build-side input bytes, if known.
    pub build_side_bytes: Option<usize>,
    /// Probe-side input bytes, if known.
    pub probe_side_bytes: Option<usize>,
    /// Probe reuse count, if known.
    pub probe_reuse_count: Option<usize>,
    /// Partition count for partitioned join strategies, if known.
    pub partition_count: Option<usize>,
    /// Partition target bytes for partitioned join strategies, if known.
    pub partition_target_bytes: Option<usize>,
    /// Build partition batch count, if known.
    pub build_partition_batch_count: Option<usize>,
    /// Probe partition batch count, if known.
    pub probe_partition_batch_count: Option<usize>,
    /// Build-side partitioned bytes, if known.
    pub build_partitioned_bytes: Option<usize>,
    /// Probe-side partitioned bytes, if known.
    pub probe_partitioned_bytes: Option<usize>,
    /// Rejection reason for reusable hash join, or `None` when not rejected/reported.
    pub reusable_hash_join_rejected_reason: Option<String>,
    /// Rejection reason for partitioned hash join, or `None` when not rejected/reported.
    pub partitioned_hash_join_rejected_reason: Option<String>,
    /// Rejection reason for one-shot equi join, or `None` when not rejected/reported.
    pub one_shot_equi_rejected_reason: Option<String>,
    /// Count of partitioned hash joins executed at this site.
    pub partitioned_hash_join_count: usize,
    /// Count of reusable hash-join builds executed at this site.
    pub hash_join_build_count: usize,
    /// Device bytes retained by hash-join build artifacts.
    pub hash_join_hash_table_bytes: usize,
    /// Count of filtered-join builds executed at this site.
    pub filtered_join_build_count: usize,
    /// Device bytes retained by filtered-join build artifacts.
    pub filtered_join_hash_table_bytes: usize,
    /// Device bytes retained by filtered-join probe states.
    pub filtered_join_probe_state_bytes: usize,
    /// Device bytes retained by mixed join size-data states.
    pub mixed_join_size_data_bytes: usize,
    /// Count of one-shot joins executed at this site.
    pub one_shot_join_count: usize,
    /// Device bytes retained by one-shot join index maps.
    pub one_shot_join_index_bytes: usize,
    /// Reservation bytes requested for one-shot output materialization.
    pub one_shot_join_materialization_bytes: usize,
    /// Observed device bytes of one-shot join output frames.
    pub one_shot_join_output_bytes: usize,
    /// Device bytes retained by join gather maps.
    pub join_gather_map_bytes: usize,
    /// Bytes reserved for join output materialization.
    pub join_output_reservation_bytes: usize,
    /// Join output row count.
    pub join_output_rows: usize,
    /// Join output bytes.
    pub join_output_bytes: usize,
}

impl NativeJoinStrategyMetricRow {
    /// Serializes this nested row as stable comma-separated key/value pairs.
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},join_strategy={},execution_strategy={},join_type={},build_side={},key_count={},estimated_build_rows={},observed_build_rows={},build_side_bytes={},probe_side_bytes={},probe_reuse_count={},partition_count={},partition_target_bytes={},build_partition_batch_count={},probe_partition_batch_count={},build_partitioned_bytes={},probe_partitioned_bytes={},reusable_hash_join_rejected_reason={},partitioned_hash_join_rejected_reason={},one_shot_equi_rejected_reason={},partitioned_hash_join_count={},hash_join_build_count={},hash_join_hash_table_bytes={},filtered_join_build_count={},filtered_join_hash_table_bytes={},filtered_join_probe_state_bytes={},mixed_join_size_data_bytes={},one_shot_join_count={},one_shot_join_index_bytes={},one_shot_join_materialization_bytes={},one_shot_join_output_bytes={},join_gather_map_bytes={},join_output_reservation_bytes={},join_output_rows={},join_output_bytes={}",
            self.node_id,
            self.join_strategy,
            self.execution_strategy,
            self.join_type,
            self.build_side,
            self.key_count,
            optional_usize(self.estimated_build_rows),
            optional_usize(self.observed_build_rows),
            optional_usize(self.build_side_bytes),
            optional_usize(self.probe_side_bytes),
            optional_usize(self.probe_reuse_count),
            optional_usize(self.partition_count),
            optional_usize(self.partition_target_bytes),
            optional_usize(self.build_partition_batch_count),
            optional_usize(self.probe_partition_batch_count),
            optional_usize(self.build_partitioned_bytes),
            optional_usize(self.probe_partitioned_bytes),
            optional_str(self.reusable_hash_join_rejected_reason.as_deref()),
            optional_str(self.partitioned_hash_join_rejected_reason.as_deref()),
            optional_str(self.one_shot_equi_rejected_reason.as_deref()),
            self.partitioned_hash_join_count,
            self.hash_join_build_count,
            self.hash_join_hash_table_bytes,
            self.filtered_join_build_count,
            self.filtered_join_hash_table_bytes,
            self.filtered_join_probe_state_bytes,
            self.mixed_join_size_data_bytes,
            self.one_shot_join_count,
            self.one_shot_join_index_bytes,
            self.one_shot_join_materialization_bytes,
            self.one_shot_join_output_bytes,
            self.join_gather_map_bytes,
            self.join_output_reservation_bytes,
            self.join_output_rows,
            self.join_output_bytes,
        )
    }

    fn same_site(&self, other: &Self) -> bool {
        self.node_id == other.node_id
            && self.join_strategy == other.join_strategy
            && self.execution_strategy == other.execution_strategy
            && self.join_type == other.join_type
            && self.build_side == other.build_side
            && self.key_count == other.key_count
            && self.estimated_build_rows == other.estimated_build_rows
            && self.probe_reuse_count == other.probe_reuse_count
            && self.partition_count == other.partition_count
            && self.partition_target_bytes == other.partition_target_bytes
            && self.reusable_hash_join_rejected_reason == other.reusable_hash_join_rejected_reason
            && self.partitioned_hash_join_rejected_reason == other.partitioned_hash_join_rejected_reason
            && self.one_shot_equi_rejected_reason == other.one_shot_equi_rejected_reason
    }

    fn merge_runtime_from(&mut self, other: Self) {
        self.observed_build_rows = max_optional_usize(self.observed_build_rows, other.observed_build_rows);
        self.build_side_bytes = max_optional_usize(self.build_side_bytes, other.build_side_bytes);
        self.probe_side_bytes = max_optional_usize(self.probe_side_bytes, other.probe_side_bytes);
        self.build_partition_batch_count =
            max_optional_usize(self.build_partition_batch_count, other.build_partition_batch_count);
        self.probe_partition_batch_count =
            max_optional_usize(self.probe_partition_batch_count, other.probe_partition_batch_count);
        self.build_partitioned_bytes = max_optional_usize(self.build_partitioned_bytes, other.build_partitioned_bytes);
        self.probe_partitioned_bytes = max_optional_usize(self.probe_partitioned_bytes, other.probe_partitioned_bytes);
        self.partitioned_hash_join_count =
            self.partitioned_hash_join_count.saturating_add(other.partitioned_hash_join_count);
        self.hash_join_build_count = self.hash_join_build_count.saturating_add(other.hash_join_build_count);
        self.hash_join_hash_table_bytes =
            self.hash_join_hash_table_bytes.saturating_add(other.hash_join_hash_table_bytes);
        self.filtered_join_build_count = self.filtered_join_build_count.saturating_add(other.filtered_join_build_count);
        self.filtered_join_hash_table_bytes =
            self.filtered_join_hash_table_bytes.saturating_add(other.filtered_join_hash_table_bytes);
        self.filtered_join_probe_state_bytes =
            self.filtered_join_probe_state_bytes.saturating_add(other.filtered_join_probe_state_bytes);
        self.mixed_join_size_data_bytes =
            self.mixed_join_size_data_bytes.saturating_add(other.mixed_join_size_data_bytes);
        self.one_shot_join_count = self.one_shot_join_count.saturating_add(other.one_shot_join_count);
        self.one_shot_join_index_bytes = self.one_shot_join_index_bytes.saturating_add(other.one_shot_join_index_bytes);
        self.one_shot_join_materialization_bytes =
            self.one_shot_join_materialization_bytes.saturating_add(other.one_shot_join_materialization_bytes);
        self.one_shot_join_output_bytes =
            self.one_shot_join_output_bytes.saturating_add(other.one_shot_join_output_bytes);
        self.join_gather_map_bytes = self.join_gather_map_bytes.saturating_add(other.join_gather_map_bytes);
        self.join_output_reservation_bytes =
            self.join_output_reservation_bytes.saturating_add(other.join_output_reservation_bytes);
        self.join_output_rows = self.join_output_rows.saturating_add(other.join_output_rows);
        self.join_output_bytes = self.join_output_bytes.saturating_add(other.join_output_bytes);
    }

    fn sort_key(&self) -> (&str, &str, &str, &str, &str, usize) {
        (
            self.node_id.as_str(),
            self.join_strategy.as_str(),
            self.execution_strategy.as_str(),
            self.join_type.as_str(),
            self.build_side.as_str(),
            self.key_count,
        )
    }
}

fn join_strategy_rows(metrics: &MetricsSet) -> Vec<NativeJoinStrategyMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_JOIN_STRATEGY_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "node_id"))
        .map(|metric| labels_to_join_strategy_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    rows
}

fn aggregate_strategy_rows(metrics: &MetricsSet) -> Vec<NativeAggregateStrategyMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_AGGREGATE_STRATEGY_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "node_id"))
        .map(|metric| labels_to_aggregate_strategy_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    rows
}

fn task_operator_rows(metrics: &MetricsSet) -> Vec<NativeTaskOperatorMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_TASK_OPERATOR_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "operator_kind"))
        .map(|metric| labels_to_task_operator_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.operator_kind.cmp(&right.operator_kind));
    rows
}

fn graph_runtime_rows(metrics: &MetricsSet) -> Vec<NativeGraphRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_GRAPH_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "algorithm"))
        .map(|metric| labels_to_graph_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.algorithm.cmp(&right.algorithm))
            .then_with(|| left.source_column.cmp(&right.source_column))
            .then_with(|| left.destination_column.cmp(&right.destination_column))
    });
    rows
}

fn task_runtime_rows(metrics: &MetricsSet) -> Vec<NativeTaskRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_TASK_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "compiled_task_id"))
        .map(|metric| labels_to_task_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| left.compiled_task_id.cmp(&right.compiled_task_id));
    rows
}

fn dynamic_filter_runtime_rows(metrics: &MetricsSet) -> Vec<NativeDynamicFilterRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_DYNAMIC_FILTER_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "dynamic_filter_id"))
        .map(|metric| labels_to_dynamic_filter_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows
}

fn source_read_runtime_rows(metrics: &MetricsSet) -> Vec<NativeSourceReadRuntimeMetricRow> {
    let mut rows = metrics
        .iter()
        .filter(|metric| metric.value().name() == NATIVE_SOURCE_READ_RUNTIME_ROWS_METRIC_NAME)
        .filter(|metric| metric.labels().iter().any(|label| label.name() == "source_node_id"))
        .map(|metric| labels_to_source_read_runtime_row(labels_by_name(metric.labels())))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows
}

fn labels_by_name(labels: &[datafusion::physical_plan::metrics::Label]) -> BTreeMap<&str, &str> {
    labels.iter().map(|label| (label.name(), label.value())).collect()
}

fn labels_to_graph_runtime_row(labels: BTreeMap<&str, &str>) -> NativeGraphRuntimeMetricRow {
    NativeGraphRuntimeMetricRow {
        node_id: string_label(&labels, "node_id"),
        algorithm: string_label(&labels, "algorithm"),
        runtime_outcome: string_label(&labels, "runtime_outcome"),
        input_boundary: string_label(&labels, "input_boundary"),
        output_boundary: string_label(&labels, "output_boundary"),
        construction_policy: string_label(&labels, "construction_policy"),
        source_column: string_label(&labels, "source_column"),
        destination_column: string_label(&labels, "destination_column"),
        weight_column: optional_string_label(&labels, "weight_column"),
        directed: bool_label(&labels, "directed"),
        renumber: bool_label(&labels, "renumber"),
        algorithm_parameters: string_label(&labels, "algorithm_parameters"),
        algorithm_metadata: string_label(&labels, "algorithm_metadata"),
        input_rows: usize_label(&labels, "input_rows"),
        input_columns: usize_label(&labels, "input_columns"),
        output_rows: usize_label(&labels, "output_rows"),
        output_columns: usize_label(&labels, "output_columns"),
        output_device_bytes: optional_usize_label(&labels, "output_device_bytes"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        gpu_resident_input_handoffs: usize_label(&labels, "gpu_resident_input_handoffs"),
        host_to_device_uploads: usize_label(&labels, "host_to_device_uploads"),
        device_to_host_materializations: usize_label(&labels, "device_to_host_materializations"),
    }
}

fn labels_to_join_strategy_row(labels: BTreeMap<&str, &str>) -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: string_label(&labels, "node_id"),
        join_strategy: string_label(&labels, "join_strategy"),
        execution_strategy: string_label(&labels, "execution_strategy"),
        join_type: string_label(&labels, "join_type"),
        build_side: string_label(&labels, "build_side"),
        key_count: usize_label(&labels, "key_count"),
        estimated_build_rows: optional_usize_label(&labels, "estimated_build_rows"),
        observed_build_rows: optional_usize_label(&labels, "observed_build_rows"),
        build_side_bytes: optional_usize_label(&labels, "build_side_bytes"),
        probe_side_bytes: optional_usize_label(&labels, "probe_side_bytes"),
        probe_reuse_count: optional_usize_label(&labels, "probe_reuse_count"),
        partition_count: optional_usize_label(&labels, "partition_count"),
        partition_target_bytes: optional_usize_label(&labels, "partition_target_bytes"),
        build_partition_batch_count: optional_usize_label(&labels, "build_partition_batch_count"),
        probe_partition_batch_count: optional_usize_label(&labels, "probe_partition_batch_count"),
        build_partitioned_bytes: optional_usize_label(&labels, "build_partitioned_bytes"),
        probe_partitioned_bytes: optional_usize_label(&labels, "probe_partitioned_bytes"),
        reusable_hash_join_rejected_reason: optional_string_label(&labels, "reusable_hash_join_rejected_reason"),
        partitioned_hash_join_rejected_reason: optional_string_label(&labels, "partitioned_hash_join_rejected_reason"),
        one_shot_equi_rejected_reason: optional_string_label(&labels, "one_shot_equi_rejected_reason"),
        partitioned_hash_join_count: usize_label(&labels, "partitioned_hash_join_count"),
        hash_join_build_count: usize_label(&labels, "hash_join_build_count"),
        hash_join_hash_table_bytes: usize_label(&labels, "hash_join_hash_table_bytes"),
        filtered_join_build_count: usize_label(&labels, "filtered_join_build_count"),
        filtered_join_hash_table_bytes: usize_label(&labels, "filtered_join_hash_table_bytes"),
        filtered_join_probe_state_bytes: usize_label(&labels, "filtered_join_probe_state_bytes"),
        mixed_join_size_data_bytes: usize_label(&labels, "mixed_join_size_data_bytes"),
        one_shot_join_count: usize_label(&labels, "one_shot_join_count"),
        one_shot_join_index_bytes: usize_label(&labels, "one_shot_join_index_bytes"),
        one_shot_join_materialization_bytes: usize_label(&labels, "one_shot_join_materialization_bytes"),
        one_shot_join_output_bytes: usize_label(&labels, "one_shot_join_output_bytes"),
        join_gather_map_bytes: usize_label(&labels, "join_gather_map_bytes"),
        join_output_reservation_bytes: usize_label(&labels, "join_output_reservation_bytes"),
        join_output_rows: usize_label(&labels, "join_output_rows"),
        join_output_bytes: usize_label(&labels, "join_output_bytes"),
    }
}

fn labels_to_aggregate_strategy_row(labels: BTreeMap<&str, &str>) -> NativeAggregateStrategyMetricRow {
    NativeAggregateStrategyMetricRow {
        node_id: string_label(&labels, "node_id"),
        selected_strategy: string_label(&labels, "selected_strategy"),
        execution_stage: string_label(&labels, "execution_stage"),
        memory_mode: string_label(&labels, "memory_mode"),
        group_key_count: usize_label(&labels, "group_key_count"),
        group_key_exprs: string_label(&labels, "group_key_exprs"),
        group_key_dtypes: string_label(&labels, "group_key_dtypes"),
        aggregate_functions: string_label(&labels, "aggregate_functions"),
        floating_statistical_aggregate: string_label(&labels, "floating_statistical_aggregate"),
        statistical_aggregate_execution_mode: string_label(&labels, "statistical_aggregate_execution_mode"),
        statistical_aggregate_tolerance_contract: string_label(&labels, "statistical_aggregate_tolerance_contract"),
        statistical_aggregate_functions: string_label(&labels, "statistical_aggregate_functions"),
        statistical_aggregate_ddof: string_label(&labels, "statistical_aggregate_ddof"),
        stddev_samp_execution_mode: string_label(&labels, "stddev_samp_execution_mode"),
        stddev_samp_ddof: string_label(&labels, "stddev_samp_ddof"),
        aggregate_filter_count: usize_label(&labels, "aggregate_filter_count"),
        distinct_aggregate_count: usize_label(&labels, "distinct_aggregate_count"),
        input_rows: usize_label(&labels, "input_rows"),
        output_group_count: usize_label(&labels, "output_group_count"),
        reduction_ratio_ppm: usize_label(&labels, "reduction_ratio_ppm"),
        chunk_row_count: usize_label(&labels, "chunk_row_count"),
        chunk_count: usize_label(&labels, "chunk_count"),
        partial_aggregate_count: usize_label(&labels, "partial_aggregate_count"),
        source_adjacent_partial: string_label(&labels, "source_adjacent_partial"),
        source_chunk_row_group_count: string_label(&labels, "source_chunk_row_group_count"),
        source_chunk_reads_all_row_groups: string_label(&labels, "source_chunk_reads_all_row_groups"),
        source_chunk_known_row_count: string_label(&labels, "source_chunk_known_row_count"),
        source_chunk_estimated_source_read_bytes: string_label(&labels, "source_chunk_estimated_source_read_bytes"),
        source_chunk_size_estimate_based: string_label(&labels, "source_chunk_size_estimate_based"),
        source_adjacent_partition_strategy: string_label(&labels, "source_adjacent_partition_strategy"),
        source_adjacent_partition_count: string_label(&labels, "source_adjacent_partition_count"),
        source_adjacent_non_empty_partition_count: string_label(&labels, "source_adjacent_non_empty_partition_count"),
        source_adjacent_partition_key_count: string_label(&labels, "source_adjacent_partition_key_count"),
        source_adjacent_max_partition_fan_in: string_label(&labels, "source_adjacent_max_partition_fan_in"),
        partitioned_merge_partition_count: string_label(&labels, "partitioned_merge_partition_count"),
        partitioned_merge_non_empty_partition_count: string_label(
            &labels,
            "partitioned_merge_non_empty_partition_count",
        ),
        partitioned_merge_max_partition_fan_in: string_label(&labels, "partitioned_merge_max_partition_fan_in"),
        selection_elapsed_ns: usize_label(&labels, "selection_elapsed_ns"),
        encode_elapsed_ns: usize_label(&labels, "encode_elapsed_ns"),
        groupby_elapsed_ns: usize_label(&labels, "groupby_elapsed_ns"),
        decode_elapsed_ns: usize_label(&labels, "decode_elapsed_ns"),
        strategy_operator_elapsed_ns: usize_label(&labels, "strategy_operator_elapsed_ns"),
        aggregate_structural_key: string_label(&labels, "aggregate_structural_key"),
        prior_cardinality_evidence_scope: string_label(&labels, "prior_cardinality_evidence_scope"),
        encoded_key_partial_cost_model: string_label(&labels, "encoded_key_partial_cost_model"),
        encoded_key_partial_cost_model_status: string_label(&labels, "encoded_key_partial_cost_model_status"),
        encoded_key_partial_selection_reason: string_label(&labels, "encoded_key_partial_selection_reason"),
        encoded_key_partial_reduction_ratio_threshold_ppm: usize_label(
            &labels,
            "encoded_key_partial_reduction_ratio_threshold_ppm",
        ),
        encoded_key_partial_prior_reduction_ratio_ppm: usize_label(
            &labels,
            "encoded_key_partial_prior_reduction_ratio_ppm",
        ),
        encoded_key_partial_prior_strategy: string_label(&labels, "encoded_key_partial_prior_strategy"),
        exact_cardinality_probe_status: string_label(&labels, "exact_cardinality_probe_status"),
        exact_cardinality_probe_key: string_label(&labels, "exact_cardinality_probe_key"),
        exact_cardinality_probe_input_rows: usize_label(&labels, "exact_cardinality_probe_input_rows"),
        exact_cardinality_probe_non_null_distinct_count: string_label(
            &labels,
            "exact_cardinality_probe_non_null_distinct_count",
        ),
        exact_cardinality_probe_null_count: string_label(&labels, "exact_cardinality_probe_null_count"),
        exact_cardinality_probe_num_groups: string_label(&labels, "exact_cardinality_probe_num_groups"),
        exact_cardinality_probe_reduction_ratio_ppm: string_label(
            &labels,
            "exact_cardinality_probe_reduction_ratio_ppm",
        ),
        exact_cardinality_probe_elapsed_ns: usize_label(&labels, "exact_cardinality_probe_elapsed_ns"),
        prepared_sort_groupby_rejected_reason: string_label(&labels, "prepared_sort_groupby_rejected_reason"),
        prepared_sort_groupby_contract_status: string_label(&labels, "prepared_sort_groupby_contract_status"),
        prepared_sort_groupby_required_contract: string_label(&labels, "prepared_sort_groupby_required_contract"),
        prepared_sort_groupby_cost_model: string_label(&labels, "prepared_sort_groupby_cost_model"),
        prepared_sort_groupby_cost_model_status: string_label(&labels, "prepared_sort_groupby_cost_model_status"),
        prepared_sort_groupby_selection_reason: string_label(&labels, "prepared_sort_groupby_selection_reason"),
        prepared_sort_groupby_num_groups: string_label(&labels, "prepared_sort_groupby_num_groups"),
        prepared_sort_groupby_grouped_key_output_bytes: string_label(
            &labels,
            "prepared_sort_groupby_grouped_key_output_bytes",
        ),
        prepared_sort_groupby_max_group_size: string_label(&labels, "prepared_sort_groupby_max_group_size"),
        rejected_candidate_strategies: string_label(&labels, "rejected_candidate_strategies"),
    }
}

fn labels_to_task_operator_row(labels: BTreeMap<&str, &str>) -> NativeTaskOperatorMetricRow {
    NativeTaskOperatorMetricRow {
        operator_kind: string_label(&labels, "operator_kind"),
        task_count: usize_label(&labels, "task_count"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        reservation_requested_bytes: usize_label(&labels, "reservation_requested_bytes"),
        reservation_granted_bytes: usize_label(&labels, "reservation_granted_bytes"),
        reservation_released_bytes: usize_label(&labels, "reservation_released_bytes"),
        reservation_live_peak_bytes: usize_label(&labels, "reservation_live_peak_bytes"),
        max_observed_peak_bytes: optional_usize_label(&labels, "max_observed_peak_bytes"),
        max_retained_bytes: optional_usize_label(&labels, "max_retained_bytes"),
    }
}

fn labels_to_task_runtime_row(labels: BTreeMap<&str, &str>) -> NativeTaskRuntimeMetricRow {
    NativeTaskRuntimeMetricRow {
        compiled_task_id: string_label(&labels, "compiled_task_id"),
        operator_kind: string_label(&labels, "operator_kind"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
        reservation_requested_bytes: usize_label(&labels, "reservation_requested_bytes"),
        reservation_granted_bytes: usize_label(&labels, "reservation_granted_bytes"),
        reservation_released_bytes: usize_label(&labels, "reservation_released_bytes"),
        reservation_live_peak_bytes: usize_label(&labels, "reservation_live_peak_bytes"),
        max_observed_peak_bytes: optional_usize_label(&labels, "max_observed_peak_bytes"),
        max_retained_bytes: optional_usize_label(&labels, "max_retained_bytes"),
    }
}

fn labels_to_dynamic_filter_runtime_row(labels: BTreeMap<&str, &str>) -> NativeDynamicFilterRuntimeMetricRow {
    NativeDynamicFilterRuntimeMetricRow {
        dynamic_filter_id: string_label(&labels, "dynamic_filter_id"),
        event: string_label(&labels, "event"),
        producer_kind: string_label(&labels, "producer_kind"),
        producer_node_id: string_label(&labels, "producer_node_id"),
        consumer_kind: string_label(&labels, "consumer_kind"),
        consumer_node_id: string_label(&labels, "consumer_node_id"),
        consumer_lowering: string_label(&labels, "consumer_lowering"),
        consumer_reduction_stage: string_label(&labels, "consumer_reduction_stage"),
        filter_kind: string_label(&labels, "filter_kind"),
        payload_value_count: optional_usize_label(&labels, "payload_value_count"),
        payload_min_i64: optional_i64_label(&labels, "payload_min_i64"),
        payload_max_i64: optional_i64_label(&labels, "payload_max_i64"),
        producer_rows: optional_usize_label(&labels, "producer_rows"),
        skipped_reason: optional_string_label(&labels, "skipped_reason"),
        selected_row_groups_before: optional_usize_label(&labels, "selected_row_groups_before"),
        selected_row_groups_after: optional_usize_label(&labels, "selected_row_groups_after"),
        selected_source_bytes_before: optional_usize_label(&labels, "selected_source_bytes_before"),
        selected_source_bytes_after: optional_usize_label(&labels, "selected_source_bytes_after"),
        source_file_index: optional_usize_label(&labels, "source_file_index"),
        source_chunk_index: optional_usize_label(&labels, "source_chunk_index"),
        reader_predicate_rows_before: optional_usize_label(&labels, "reader_predicate_rows_before"),
        reader_predicate_rows_after: optional_usize_label(&labels, "reader_predicate_rows_after"),
        reader_predicate_rows_filtered: usize_label(&labels, "reader_predicate_rows_filtered"),
        post_decode_rows_before: optional_usize_label(&labels, "post_decode_rows_before"),
        post_decode_rows_after: optional_usize_label(&labels, "post_decode_rows_after"),
        post_decode_rows_filtered: usize_label(&labels, "post_decode_rows_filtered"),
        row_groups_pruned: usize_label(&labels, "row_groups_pruned"),
        source_bytes_pruned: usize_label(&labels, "source_bytes_pruned"),
    }
}

fn labels_to_source_read_runtime_row(labels: BTreeMap<&str, &str>) -> NativeSourceReadRuntimeMetricRow {
    NativeSourceReadRuntimeMetricRow {
        source_node_id: string_label(&labels, "source_node_id"),
        read_kind: string_label(&labels, "read_kind"),
        read_strategy: string_label(&labels, "read_strategy"),
        batch_index: usize_label(&labels, "batch_index"),
        chunk_count: usize_label(&labels, "chunk_count"),
        first_chunk_index: usize_label(&labels, "first_chunk_index"),
        last_chunk_index: usize_label(&labels, "last_chunk_index"),
        first_file_index: usize_label(&labels, "first_file_index"),
        last_file_index: usize_label(&labels, "last_file_index"),
        file_count: usize_label(&labels, "file_count"),
        row_group_count: usize_label(&labels, "row_group_count"),
        read_columns_per_file: usize_label(&labels, "read_columns_per_file"),
        read_columns_total: usize_label(&labels, "read_columns_total"),
        known_file_bytes: usize_label(&labels, "known_file_bytes"),
        estimated_source_read_bytes: usize_label(&labels, "estimated_source_read_bytes"),
        known_row_group_bytes: usize_label(&labels, "known_row_group_bytes"),
        known_projected_compressed_bytes: usize_label(&labels, "known_projected_compressed_bytes"),
        known_projected_uncompressed_bytes: usize_label(&labels, "known_projected_uncompressed_bytes"),
        known_pure_filter_compressed_bytes: usize_label(&labels, "known_pure_filter_compressed_bytes"),
        known_pure_filter_uncompressed_bytes: usize_label(&labels, "known_pure_filter_uncompressed_bytes"),
        output_rows: usize_label(&labels, "output_rows"),
        reader_predicate: bool_label(&labels, "reader_predicate"),
        reader_dynamic_filter_count: usize_label(&labels, "reader_dynamic_filter_count"),
        source_batching_reason: string_label(&labels, "source_batching_reason"),
        elapsed_ns: usize_label(&labels, "elapsed_ns"),
    }
}

fn string_label(labels: &BTreeMap<&str, &str>, name: &str) -> String {
    labels.get(name).copied().unwrap_or("unknown").to_string()
}

fn optional_string_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<String> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.to_string())
}

fn usize_label(labels: &BTreeMap<&str, &str>, name: &str) -> usize {
    labels.get(name).and_then(|value| value.parse().ok()).unwrap_or(0)
}

fn bool_label(labels: &BTreeMap<&str, &str>, name: &str) -> bool {
    labels.get(name).and_then(|value| value.parse().ok()).unwrap_or(false)
}

fn optional_usize_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<usize> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.parse().ok()).flatten()
}

fn optional_i64_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<i64> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.parse().ok()).flatten()
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn optional_str(value: Option<&str>) -> String {
    value.unwrap_or("none").to_string()
}

fn max_optional_usize(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}
