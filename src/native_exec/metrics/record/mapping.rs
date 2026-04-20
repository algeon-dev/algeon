//! Compile-time mapping from ordered metric descriptors to recording paths.

use nexus_query_engine::{metrics::QueryMetrics, runtime::LakehouseCacheFillEventSnapshot};

use super::{NativeCudfMetrics, add_optional_count, count, join_gather_map_bytes};
use crate::native_exec::{
    admission::QueryAdmissionFacts,
    metrics::contract::{for_each_native_cudf_metric, recording_marker},
};

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::native_exec::metrics) enum NativeMetricRecordingKind {
    Query,
    Admission,
    CacheEvents,
}

/// Compile-time recording contract for one ordered metric descriptor.
///
/// The ordered descriptor macro calls this trait for every marker. Adding a
/// descriptor without an implementation is therefore a compile error instead
/// of a silently registered metric that is never populated.
trait NativeMetricRecording<Marker> {
    #[cfg(test)]
    const KIND: NativeMetricRecordingKind;

    #[inline]
    fn record_query(&self, _metrics: &QueryMetrics) {}

    #[inline]
    fn record_admission(&self, _facts: QueryAdmissionFacts) {}

    #[inline]
    fn record_cache_events(&self, _events: &[LakehouseCacheFillEventSnapshot]) {}
}

macro_rules! define_descriptor_recorders {
    ($( $field:ident => $name:literal: $kind:ident, )*) => {
        pub(super) fn record_descriptor_query_metrics(&self, metrics: &QueryMetrics) {
            $(
                <Self as NativeMetricRecording<recording_marker::$field>>::record_query(self, metrics);
            )*
        }

        pub(super) fn record_descriptor_admission_facts(&self, facts: QueryAdmissionFacts) {
            $(
                <Self as NativeMetricRecording<recording_marker::$field>>::record_admission(self, facts);
            )*
        }

        pub(super) fn record_descriptor_cache_events(&self, events: &[LakehouseCacheFillEventSnapshot]) {
            $(
                <Self as NativeMetricRecording<recording_marker::$field>>::record_cache_events(self, events);
            )*
        }
    };
}

#[cfg(test)]
macro_rules! define_recording_descriptors {
    ($( $field:ident => $name:literal: $kind:ident, )*) => {
        pub(in crate::native_exec::metrics) const NATIVE_CUDF_RECORDING_DESCRIPTORS: &[(&str, NativeMetricRecordingKind)] = &[
            $(
                (
                    $name,
                    <NativeCudfMetrics as NativeMetricRecording<recording_marker::$field>>::KIND,
                ),
            )*
        ];
    };
}

#[cfg(test)]
for_each_native_cudf_metric!(define_recording_descriptors);

macro_rules! recording_kind {
    ($kind:ident) => {
        #[cfg(test)]
        const KIND: NativeMetricRecordingKind = NativeMetricRecordingKind::$kind;
    };
}

macro_rules! impl_query_count_recording {
    ($( $field:ident => $($source:ident).+ ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!(Query);

                fn record_query(&self, metrics: &QueryMetrics) {
                    self.$field.add(count(metrics$(.$source)+));
                }
            }
        )*
    };
}

macro_rules! impl_query_timer_recording {
    ($( $field:ident => $source:ident ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!(Query);

                fn record_query(&self, metrics: &QueryMetrics) {
                    self.$field.add_duration(metrics.$source);
                }
            }
        )*
    };
}

macro_rules! impl_query_optional_recording {
    ($( $field:ident => $source:ident, observed = $observed:ident ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!(Query);

                fn record_query(&self, metrics: &QueryMetrics) {
                    add_optional_count(&self.$field, Some(&self.$observed), metrics.$source);
                }
            }
        )*
    };
}

macro_rules! impl_query_optional_unobserved_recording {
    ($( $field:ident => $source:ident ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!(Query);

                fn record_query(&self, metrics: &QueryMetrics) {
                    add_optional_count(&self.$field, None, metrics.$source);
                }
            }
        )*
    };
}

macro_rules! impl_query_rows_recording {
    ($( $field:ident => $method:ident ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!(Query);

                fn record_query(&self, metrics: &QueryMetrics) {
                    self.$method(metrics);
                }
            }
        )*
    };
}

macro_rules! impl_noop_recording {
    ($kind:ident; $( $field:ident ),* $(,)?) => {
        $(
            impl NativeMetricRecording<recording_marker::$field> for NativeCudfMetrics {
                recording_kind!($kind);
            }
        )*
    };
}

impl_query_count_recording! {
    input_batches => input_batches,
    input_rows => input_rows,
    gpu_resident_input_handoffs => boundary.gpu_resident_input_handoffs,
    host_to_device_uploads => boundary.host_to_device_uploads,
    host_to_device_uploaded_rows => boundary.host_to_device_uploaded_rows,
    host_to_device_uploaded_bytes => boundary.host_to_device_uploaded_bytes,
    device_to_host_materializations => boundary.device_to_host_materializations,
    device_to_host_materialized_rows => boundary.device_to_host_materialized_rows,
    device_to_host_materialized_bytes => boundary.device_to_host_materialized_bytes,
    device_to_device_copies => boundary.device_to_device_copies,
    device_to_device_copied_bytes => boundary.device_to_device_copied_bytes,
    reservation_requested_bytes => reservation_requested_bytes,
    reservation_granted_bytes => reservation_granted_bytes,
    reservation_live_peak_bytes => reservation_live_peak_bytes,
    reservation_denied_count => reservation_denied_count,
    max_task_reservation_bytes => max_task_reservation_bytes,
    source_chunks_planned => source_chunks_planned,
    source_chunks_attempted => source_chunks_attempted,
    source_chunks_committed => source_chunks_committed,
    source_chunks_retried => source_chunks_retried,
    source_read_prefetches_started => source_read_prefetches_started,
    source_read_prefetches_completed => source_read_prefetches_completed,
    source_read_prefetches_skipped => source_read_prefetches_skipped,
    source_read_prefetches_abandoned => source_read_prefetches_abandoned,
    source_cache_hits => source_cache_hits,
    source_cache_misses => source_cache_misses,
    source_cache_inserts => source_cache_inserts,
    source_cache_insert_skipped => source_cache_insert_skipped,
    source_cache_retained_bytes => source_cache_retained_bytes,
    row_groups_planned => row_groups_planned,
    row_groups_committed => row_groups_committed,
    partial_aggregate_count => partial_aggregate_count,
    partial_aggregate_input_rows => partial_aggregate_input_rows,
    partial_aggregate_rows => partial_aggregate_rows,
    partial_aggregate_retained_bytes => partial_aggregate_retained_bytes,
    final_merge_attempts => final_merge_attempts,
    final_merge_levels => final_merge_levels,
    final_merge_max_fan_in => final_merge_max_fan_in,
    compiled_pipeline_attempts => compiled_pipeline_attempts,
    compiled_pipeline_tasks_executed => compiled_pipeline_tasks_executed,
    compiled_pipeline_host_arrow_sinks => compiled_pipeline_host_arrow_sinks,
    parquet_files => parquet_files,
    parquet_bytes_known => parquet_bytes_known,
    row_groups_selected => row_groups_selected,
    read_columns => read_columns,
    retry_count => retry_count,
    retry_resource_exhausted_count => retry_resource_exhausted_count,
    retry_reservation_failure_count => retry_reservation_failure_count,
    adaptive_retry_count => adaptive_retry_count,
    adaptive_spill_count => adaptive_spill_count,
    adaptive_downgrade_count => adaptive_downgrade_count,
    adaptive_memory_history_lookup_count => adaptive_memory_history_lookup_count,
    adaptive_memory_history_hit_count => adaptive_memory_history_hit_count,
    adaptive_memory_history_miss_count => adaptive_memory_history_miss_count,
    dynamic_filter_produced_count => dynamic_filter_produced_count,
    dynamic_filter_skipped_count => dynamic_filter_skipped_count,
    dynamic_filter_applied_count => dynamic_filter_applied_count,
    dynamic_filter_row_groups_pruned => dynamic_filter_row_groups_pruned,
    dynamic_filter_source_bytes_pruned => dynamic_filter_source_bytes_pruned,
    dynamic_filter_reader_predicate_rows_before => dynamic_filter_reader_predicate_rows_before,
    dynamic_filter_reader_predicate_rows_after => dynamic_filter_reader_predicate_rows_after,
    dynamic_filter_reader_predicate_rows_filtered => dynamic_filter_reader_predicate_rows_filtered,
    dynamic_filter_post_decode_rows_before => dynamic_filter_post_decode_rows_before,
    dynamic_filter_post_decode_rows_after => dynamic_filter_post_decode_rows_after,
    dynamic_filter_post_decode_rows_filtered => dynamic_filter_post_decode_rows_filtered,
    dynamic_filter_source_post_decode_rows_filtered => dynamic_filter_source_post_decode_rows_filtered,
    dynamic_filter_filter_post_decode_rows_filtered => dynamic_filter_filter_post_decode_rows_filtered,
    dynamic_filter_join_probe_post_decode_rows_filtered => dynamic_filter_join_probe_post_decode_rows_filtered,
    hash_join_build_count => hash_join_build_count,
    filtered_join_build_count => filtered_join_build_count,
    one_shot_join_count => one_shot_join_count,
    hash_join_cache_hits => hash_join_cache_hits,
    hash_join_cache_misses => hash_join_cache_misses,
    hash_join_durable_cache_hits => hash_join_durable_cache_hits,
    hash_join_durable_cache_misses => hash_join_durable_cache_misses,
    hash_join_durable_cache_inserts => hash_join_durable_cache_inserts,
    hash_join_hash_table_bytes => hash_join_hash_table_bytes,
    filtered_join_hash_table_bytes => filtered_join_hash_table_bytes,
    filtered_join_probe_state_bytes => filtered_join_probe_state_bytes,
    mixed_join_size_data_bytes => mixed_join_size_data_bytes,
    one_shot_join_index_bytes => one_shot_join_index_bytes,
    one_shot_join_materialization_bytes => one_shot_join_materialization_bytes,
    one_shot_join_output_bytes => one_shot_join_output_bytes,
    source_fetch_bytes => source_fetch_bytes,
    source_fetch_count => source_fetch_count,
}

impl_query_optional_recording! {
    max_task_observed_peak_bytes => max_task_observed_peak_bytes,
        observed = max_task_observed_peak_bytes_observed,
    max_task_retained_bytes => max_task_retained_bytes,
        observed = max_task_retained_bytes_observed,
    resident_dataframe_peak_bytes => resident_dataframe_peak_bytes,
        observed = resident_dataframe_peak_bytes_observed,
    allocator_high_water_bytes => allocator_high_water_bytes,
        observed = allocator_high_water_bytes_observed,
}

impl_query_optional_unobserved_recording! {
    peak_device_memory_bytes => peak_device_memory_bytes,
}

impl_query_timer_recording! {
    source_metadata_load_time => source_metadata_load_time,
    source_read_queue_wait_time => source_read_queue_wait_time,
    source_cache_lookup_time => source_cache_lookup_time,
    host_fetch_budget_wait_time => host_fetch_budget_wait_time,
    source_fetch_concurrency_wait_time => source_fetch_concurrency_wait_time,
    source_fetch_time => source_fetch_time,
    source_decode_time => source_decode_time,
    parquet_read_time => parquet_read_time,
    source_read_time => source_read_time,
    operator_time => operator_time,
    chunk_operator_time => chunk_operator_time,
    final_merge_time => final_merge_time,
    final_sort_time => final_sort_time,
    materialization_time => materialization_time,
}

impl_query_rows_recording! {
    native_graph_reservation_rows => record_graph_reservation_rows,
    native_graph_runtime_rows => record_graph_runtime_rows,
    native_task_operator_rows => record_task_operator_rows,
    native_task_runtime_rows => record_task_runtime_rows,
    native_dynamic_filter_runtime_rows => record_dynamic_filter_runtime_rows,
    native_source_read_runtime_rows => record_source_read_runtime_rows,
    native_aggregate_strategy_rows => record_aggregate_strategy_rows,
    native_join_strategy_rows => record_join_strategy_rows,
    native_plan_task_mapping_rows => record_plan_task_mapping_rows,
}

impl NativeMetricRecording<recording_marker::source_read_boundary_direct_device> for NativeCudfMetrics {
    recording_kind!(Query);

    fn record_query(&self, metrics: &QueryMetrics) {
        self.record_boundary_policy(metrics.execution_boundary_policy);
    }
}

impl_noop_recording! {
    Query;
    build_materialization_boundary_query_scoped,
    build_materialization_boundary_durable,
    probe_execution_boundary_reusable,
    result_materialization_boundary_explicit,
    max_task_observed_peak_bytes_observed,
    max_task_retained_bytes_observed,
    resident_dataframe_peak_bytes_observed,
    allocator_high_water_bytes_observed,
}

impl NativeMetricRecording<recording_marker::join_gather_map_bytes> for NativeCudfMetrics {
    recording_kind!(Query);

    fn record_query(&self, metrics: &QueryMetrics) {
        self.join_gather_map_bytes.add(count(join_gather_map_bytes(metrics)));
    }
}

impl NativeMetricRecording<recording_marker::admission_sequence> for NativeCudfMetrics {
    recording_kind!(Admission);

    fn record_admission(&self, facts: QueryAdmissionFacts) {
        add_optional_count(
            &self.admission_sequence,
            Some(&self.admission_sequence_observed),
            Some(facts.admission_sequence),
        );
    }
}

impl NativeMetricRecording<recording_marker::admission_wait_time> for NativeCudfMetrics {
    recording_kind!(Admission);

    fn record_admission(&self, facts: QueryAdmissionFacts) {
        self.admission_wait_time_observed.add(1);
        self.admission_wait_time.add_duration(facts.admission_wait);
    }
}

impl NativeMetricRecording<recording_marker::device_ordinal> for NativeCudfMetrics {
    recording_kind!(Admission);

    fn record_admission(&self, facts: QueryAdmissionFacts) {
        add_optional_count(
            &self.device_ordinal,
            Some(&self.device_ordinal_observed),
            Some(facts.device_ordinal as u64),
        );
    }
}

impl NativeMetricRecording<recording_marker::query_grant_bytes> for NativeCudfMetrics {
    recording_kind!(Admission);

    fn record_admission(&self, facts: QueryAdmissionFacts) {
        add_optional_count(
            &self.query_grant_bytes,
            Some(&self.query_grant_bytes_observed),
            Some(facts.query_grant_bytes),
        );
    }
}

impl_noop_recording! {
    Admission;
    admission_sequence_observed,
    admission_wait_time_observed,
    device_ordinal_observed,
    query_grant_bytes_observed,
}

impl NativeMetricRecording<recording_marker::native_lakehouse_cache_fill_event_snapshot_rows> for NativeCudfMetrics {
    recording_kind!(CacheEvents);

    fn record_cache_events(&self, events: &[LakehouseCacheFillEventSnapshot]) {
        self.record_lakehouse_cache_fill_event_rows(events);
    }
}

impl NativeCudfMetrics {
    for_each_native_cudf_metric!(define_descriptor_recorders);
}
