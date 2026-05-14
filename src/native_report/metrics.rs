use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::MetricsSet;

use super::tsv::{metric_count, optional_observed_metric_count};

const NATIVE_JOIN_STRATEGY_ROWS_METRIC_NAME: &str = "native_join_strategy_rows";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeDataFusionExecutionMetrics {
    pub gpu_resident_input_handoffs: usize,
    pub host_to_device_uploads: usize,
    pub host_to_device_uploaded_rows: usize,
    pub host_to_device_uploaded_bytes: usize,
    pub device_to_host_materializations: usize,
    pub device_to_host_materialized_rows: usize,
    pub device_to_host_materialized_bytes: usize,
    pub device_to_device_copies: usize,
    pub device_to_device_copied_bytes: usize,
    pub parquet_files: usize,
    pub parquet_bytes_known: usize,
    pub row_groups_selected: usize,
    pub read_columns: usize,
    pub resident_dataframe_peak_bytes: Option<usize>,
    pub allocator_high_water_bytes: Option<usize>,
    pub reservation_requested_bytes: usize,
    pub reservation_granted_bytes: usize,
    pub reservation_denied_count: usize,
    pub source_chunks_planned: usize,
    pub source_chunks_attempted: usize,
    pub source_chunks_committed: usize,
    pub source_chunks_retried: usize,
    pub source_read_prefetches_started: usize,
    pub source_read_prefetches_completed: usize,
    pub source_read_prefetches_skipped: usize,
    pub source_read_prefetches_abandoned: usize,
    pub source_cache_hits: usize,
    pub source_cache_misses: usize,
    pub source_cache_inserts: usize,
    pub source_cache_insert_skipped: usize,
    pub source_cache_retained_bytes: usize,
    pub row_groups_planned: usize,
    pub row_groups_committed: usize,
    pub partial_aggregate_count: usize,
    pub partial_aggregate_rows: usize,
    pub final_merge_attempts: usize,
    pub final_merge_levels: usize,
    pub retry_count: usize,
    pub retry_resource_exhausted_count: usize,
    pub retry_reservation_failure_count: usize,
    pub adaptive_retry_count: usize,
    pub adaptive_spill_count: usize,
    pub adaptive_downgrade_count: usize,
    pub adaptive_memory_history_lookup_count: usize,
    pub adaptive_memory_history_hit_count: usize,
    pub adaptive_memory_history_miss_count: usize,
    pub hash_join_build_count: usize,
    pub filtered_join_build_count: usize,
    pub one_shot_join_count: usize,
    pub hash_join_cache_hits: usize,
    pub hash_join_cache_misses: usize,
    pub hash_join_durable_cache_hits: usize,
    pub hash_join_durable_cache_misses: usize,
    pub hash_join_durable_cache_inserts: usize,
    pub hash_join_hash_table_bytes: usize,
    pub filtered_join_hash_table_bytes: usize,
    pub filtered_join_probe_state_bytes: usize,
    pub mixed_join_size_data_bytes: usize,
    pub one_shot_join_index_bytes: usize,
    pub one_shot_join_materialization_bytes: usize,
    pub one_shot_join_output_bytes: usize,
    pub join_gather_map_bytes: usize,
    pub join_strategy_rows: Vec<NativeJoinStrategyMetricRow>,
}

impl NativeDataFusionExecutionMetrics {
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
            reservation_denied_count: metric_count(metrics, "reservation_denied_count"),
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
            partial_aggregate_rows: metric_count(metrics, "partial_aggregate_rows"),
            final_merge_attempts: metric_count(metrics, "final_merge_attempts"),
            final_merge_levels: metric_count(metrics, "final_merge_levels"),
            retry_count: metric_count(metrics, "retry_count"),
            retry_resource_exhausted_count: metric_count(metrics, "retry_resource_exhausted_count"),
            retry_reservation_failure_count: metric_count(metrics, "retry_reservation_failure_count"),
            adaptive_retry_count: metric_count(metrics, "adaptive_retry_count"),
            adaptive_spill_count: metric_count(metrics, "adaptive_spill_count"),
            adaptive_downgrade_count: metric_count(metrics, "adaptive_downgrade_count"),
            adaptive_memory_history_lookup_count: metric_count(metrics, "adaptive_memory_history_lookup_count"),
            adaptive_memory_history_hit_count: metric_count(metrics, "adaptive_memory_history_hit_count"),
            adaptive_memory_history_miss_count: metric_count(metrics, "adaptive_memory_history_miss_count"),
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
            join_strategy_rows: join_strategy_rows(metrics),
        }
    }

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
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeJoinStrategyMetricRow {
    pub node_id: String,
    pub join_strategy: String,
    pub execution_strategy: String,
    pub join_type: String,
    pub build_side: String,
    pub key_count: usize,
    pub estimated_build_rows: Option<usize>,
    pub observed_build_rows: Option<usize>,
    pub build_side_bytes: Option<usize>,
    pub probe_reuse_count: Option<usize>,
    pub reusable_hash_join_rejected_reason: Option<String>,
    pub one_shot_equi_rejected_reason: Option<String>,
    pub hash_join_build_count: usize,
    pub hash_join_hash_table_bytes: usize,
    pub filtered_join_build_count: usize,
    pub filtered_join_hash_table_bytes: usize,
    pub filtered_join_probe_state_bytes: usize,
    pub mixed_join_size_data_bytes: usize,
    pub one_shot_join_count: usize,
    pub one_shot_join_index_bytes: usize,
    pub one_shot_join_materialization_bytes: usize,
    pub one_shot_join_output_bytes: usize,
    pub join_gather_map_bytes: usize,
}

impl NativeJoinStrategyMetricRow {
    pub fn to_stable_kv_row(&self) -> String {
        format!(
            "node_id={},join_strategy={},execution_strategy={},join_type={},build_side={},key_count={},estimated_build_rows={},observed_build_rows={},build_side_bytes={},probe_reuse_count={},reusable_hash_join_rejected_reason={},one_shot_equi_rejected_reason={},hash_join_build_count={},hash_join_hash_table_bytes={},filtered_join_build_count={},filtered_join_hash_table_bytes={},filtered_join_probe_state_bytes={},mixed_join_size_data_bytes={},one_shot_join_count={},one_shot_join_index_bytes={},one_shot_join_materialization_bytes={},one_shot_join_output_bytes={},join_gather_map_bytes={}",
            self.node_id,
            self.join_strategy,
            self.execution_strategy,
            self.join_type,
            self.build_side,
            self.key_count,
            optional_usize(self.estimated_build_rows),
            optional_usize(self.observed_build_rows),
            optional_usize(self.build_side_bytes),
            optional_usize(self.probe_reuse_count),
            optional_str(self.reusable_hash_join_rejected_reason.as_deref()),
            optional_str(self.one_shot_equi_rejected_reason.as_deref()),
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
            && self.reusable_hash_join_rejected_reason == other.reusable_hash_join_rejected_reason
            && self.one_shot_equi_rejected_reason == other.one_shot_equi_rejected_reason
    }

    fn merge_runtime_from(&mut self, other: Self) {
        self.observed_build_rows = max_optional_usize(self.observed_build_rows, other.observed_build_rows);
        self.build_side_bytes = max_optional_usize(self.build_side_bytes, other.build_side_bytes);
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

fn labels_by_name(labels: &[datafusion::physical_plan::metrics::Label]) -> BTreeMap<&str, &str> {
    labels.iter().map(|label| (label.name(), label.value())).collect()
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
        probe_reuse_count: optional_usize_label(&labels, "probe_reuse_count"),
        reusable_hash_join_rejected_reason: optional_string_label(&labels, "reusable_hash_join_rejected_reason"),
        one_shot_equi_rejected_reason: optional_string_label(&labels, "one_shot_equi_rejected_reason"),
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

fn optional_usize_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<usize> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.parse().ok()).flatten()
}

fn optional_usize(value: Option<usize>) -> String {
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
