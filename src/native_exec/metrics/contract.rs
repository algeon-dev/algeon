//! Ordered native-to-DataFusion metric contract.

/// Storage and DataFusion category attached to a native metric descriptor.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeMetricKind {
    Rows,
    Bytes,
    Timer,
}

/// Registration metadata generated from the ordered public contract.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeMetricDescriptor {
    pub(super) name: &'static str,
    pub(super) kind: NativeMetricKind,
}

// This is the single ordered source for public names, recorder field identifiers,
// DataFusion registration type/category, and native-to-DataFusion recording ids.
// Keep entries in public contract order; consumers parse that order verbatim.
macro_rules! for_each_native_cudf_metric {
    ($macro:ident) => {
        $macro! {
            input_batches => "input_batches": rows,
            input_rows => "input_rows": rows,
            gpu_resident_input_handoffs => "gpu_resident_input_handoffs": rows,
            host_to_device_uploads => "host_to_device_uploads": rows,
            host_to_device_uploaded_rows => "host_to_device_uploaded_rows": rows,
            host_to_device_uploaded_bytes => "host_to_device_uploaded_bytes": bytes,
            device_to_host_materializations => "device_to_host_materializations": rows,
            device_to_host_materialized_rows => "device_to_host_materialized_rows": rows,
            device_to_host_materialized_bytes => "device_to_host_materialized_bytes": bytes,
            device_to_device_copies => "device_to_device_copies": rows,
            device_to_device_copied_bytes => "device_to_device_copied_bytes": bytes,
            source_read_boundary_direct_device => "source_read_boundary_direct_device": rows,
            build_materialization_boundary_query_scoped => "build_materialization_boundary_query_scoped": rows,
            build_materialization_boundary_durable => "build_materialization_boundary_durable": rows,
            probe_execution_boundary_reusable => "probe_execution_boundary_reusable": rows,
            result_materialization_boundary_explicit => "result_materialization_boundary_explicit": rows,
            admission_sequence => "admission_sequence": rows,
            admission_sequence_observed => "admission_sequence_observed": rows,
            admission_wait_time => "admission_wait_time": timer,
            admission_wait_time_observed => "admission_wait_time_observed": rows,
            device_ordinal => "device_ordinal": rows,
            device_ordinal_observed => "device_ordinal_observed": rows,
            query_grant_bytes => "query_grant_bytes": bytes,
            query_grant_bytes_observed => "query_grant_bytes_observed": rows,
            reservation_requested_bytes => "reservation_requested_bytes": bytes,
            reservation_granted_bytes => "reservation_granted_bytes": bytes,
            reservation_live_peak_bytes => "reservation_live_peak_bytes": bytes,
            reservation_denied_count => "reservation_denied_count": rows,
            max_task_reservation_bytes => "max_task_reservation_bytes": bytes,
            max_task_observed_peak_bytes => "max_task_observed_peak_bytes": bytes,
            max_task_observed_peak_bytes_observed => "max_task_observed_peak_bytes_observed": rows,
            max_task_retained_bytes => "max_task_retained_bytes": bytes,
            max_task_retained_bytes_observed => "max_task_retained_bytes_observed": rows,
            native_graph_reservation_rows => "native_graph_reservation_rows": rows,
            native_graph_runtime_rows => "native_graph_runtime_rows": rows,
            native_task_operator_rows => "native_task_operator_rows": rows,
            native_task_runtime_rows => "native_task_runtime_rows": rows,
            native_dynamic_filter_runtime_rows => "native_dynamic_filter_runtime_rows": rows,
            native_source_read_runtime_rows => "native_source_read_runtime_rows": rows,
            native_lakehouse_cache_fill_event_snapshot_rows => "native_lakehouse_cache_fill_event_snapshot_rows": rows,
            peak_device_memory_bytes => "peak_device_memory_bytes": bytes,
            resident_dataframe_peak_bytes => "resident_dataframe_peak_bytes": bytes,
            resident_dataframe_peak_bytes_observed => "resident_dataframe_peak_bytes_observed": rows,
            allocator_high_water_bytes => "allocator_high_water_bytes": bytes,
            allocator_high_water_bytes_observed => "allocator_high_water_bytes_observed": rows,
            source_chunks_planned => "source_chunks_planned": rows,
            source_chunks_attempted => "source_chunks_attempted": rows,
            source_chunks_committed => "source_chunks_committed": rows,
            source_chunks_retried => "source_chunks_retried": rows,
            source_read_prefetches_started => "source_read_prefetches_started": rows,
            source_read_prefetches_completed => "source_read_prefetches_completed": rows,
            source_read_prefetches_skipped => "source_read_prefetches_skipped": rows,
            source_read_prefetches_abandoned => "source_read_prefetches_abandoned": rows,
            source_cache_hits => "source_cache_hits": rows,
            source_cache_misses => "source_cache_misses": rows,
            source_cache_inserts => "source_cache_inserts": rows,
            source_cache_insert_skipped => "source_cache_insert_skipped": rows,
            source_cache_retained_bytes => "source_cache_retained_bytes": bytes,
            row_groups_planned => "row_groups_planned": rows,
            row_groups_committed => "row_groups_committed": rows,
            partial_aggregate_count => "partial_aggregate_count": rows,
            partial_aggregate_input_rows => "partial_aggregate_input_rows": rows,
            partial_aggregate_rows => "partial_aggregate_rows": rows,
            partial_aggregate_retained_bytes => "partial_aggregate_retained_bytes": bytes,
            final_merge_attempts => "final_merge_attempts": rows,
            final_merge_levels => "final_merge_levels": rows,
            final_merge_max_fan_in => "final_merge_max_fan_in": rows,
            compiled_pipeline_attempts => "compiled_pipeline_attempts": rows,
            compiled_pipeline_tasks_executed => "compiled_pipeline_tasks_executed": rows,
            compiled_pipeline_host_arrow_sinks => "compiled_pipeline_host_arrow_sinks": rows,
            parquet_files => "parquet_files": rows,
            parquet_bytes_known => "parquet_bytes_known": bytes,
            row_groups_selected => "row_groups_selected": rows,
            read_columns => "read_columns": rows,
            retry_count => "retry_count": rows,
            retry_resource_exhausted_count => "retry_resource_exhausted_count": rows,
            retry_reservation_failure_count => "retry_reservation_failure_count": rows,
            adaptive_retry_count => "adaptive_retry_count": rows,
            adaptive_spill_count => "adaptive_spill_count": rows,
            adaptive_downgrade_count => "adaptive_downgrade_count": rows,
            adaptive_memory_history_lookup_count => "adaptive_memory_history_lookup_count": rows,
            adaptive_memory_history_hit_count => "adaptive_memory_history_hit_count": rows,
            adaptive_memory_history_miss_count => "adaptive_memory_history_miss_count": rows,
            dynamic_filter_produced_count => "dynamic_filter_produced_count": rows,
            dynamic_filter_skipped_count => "dynamic_filter_skipped_count": rows,
            dynamic_filter_applied_count => "dynamic_filter_applied_count": rows,
            dynamic_filter_row_groups_pruned => "dynamic_filter_row_groups_pruned": rows,
            dynamic_filter_source_bytes_pruned => "dynamic_filter_source_bytes_pruned": bytes,
            dynamic_filter_reader_predicate_rows_before => "dynamic_filter_reader_predicate_rows_before": rows,
            dynamic_filter_reader_predicate_rows_after => "dynamic_filter_reader_predicate_rows_after": rows,
            dynamic_filter_reader_predicate_rows_filtered => "dynamic_filter_reader_predicate_rows_filtered": rows,
            dynamic_filter_post_decode_rows_before => "dynamic_filter_post_decode_rows_before": rows,
            dynamic_filter_post_decode_rows_after => "dynamic_filter_post_decode_rows_after": rows,
            dynamic_filter_post_decode_rows_filtered => "dynamic_filter_post_decode_rows_filtered": rows,
            dynamic_filter_source_post_decode_rows_filtered => "dynamic_filter_source_post_decode_rows_filtered": rows,
            dynamic_filter_filter_post_decode_rows_filtered => "dynamic_filter_filter_post_decode_rows_filtered": rows,
            dynamic_filter_join_probe_post_decode_rows_filtered => "dynamic_filter_join_probe_post_decode_rows_filtered": rows,
            hash_join_build_count => "hash_join_build_count": rows,
            filtered_join_build_count => "filtered_join_build_count": rows,
            one_shot_join_count => "one_shot_join_count": rows,
            hash_join_cache_hits => "hash_join_cache_hits": rows,
            hash_join_cache_misses => "hash_join_cache_misses": rows,
            hash_join_durable_cache_hits => "hash_join_durable_cache_hits": rows,
            hash_join_durable_cache_misses => "hash_join_durable_cache_misses": rows,
            hash_join_durable_cache_inserts => "hash_join_durable_cache_inserts": rows,
            hash_join_hash_table_bytes => "hash_join_hash_table_bytes": bytes,
            filtered_join_hash_table_bytes => "filtered_join_hash_table_bytes": bytes,
            filtered_join_probe_state_bytes => "filtered_join_probe_state_bytes": bytes,
            mixed_join_size_data_bytes => "mixed_join_size_data_bytes": bytes,
            one_shot_join_index_bytes => "one_shot_join_index_bytes": bytes,
            one_shot_join_materialization_bytes => "one_shot_join_materialization_bytes": bytes,
            one_shot_join_output_bytes => "one_shot_join_output_bytes": bytes,
            join_gather_map_bytes => "join_gather_map_bytes": bytes,
            native_aggregate_strategy_rows => "native_aggregate_strategy_rows": rows,
            native_join_strategy_rows => "native_join_strategy_rows": rows,
            source_metadata_load_time => "source_metadata_load_time": timer,
            source_read_queue_wait_time => "source_read_queue_wait_time": timer,
            source_cache_lookup_time => "source_cache_lookup_time": timer,
            parquet_read_time => "parquet_read_time": timer,
            source_read_time => "source_read_time": timer,
            operator_time => "operator_time": timer,
            chunk_operator_time => "chunk_operator_time": timer,
            final_merge_time => "final_merge_time": timer,
            final_sort_time => "final_sort_time": timer,
            materialization_time => "materialization_time": timer,
            native_plan_task_mapping_rows => "native_plan_task_mapping_rows": rows,
        }
    };
}

pub(super) use for_each_native_cudf_metric;

#[cfg(test)]
macro_rules! metric_kind {
    (rows) => {
        NativeMetricKind::Rows
    };
    (bytes) => {
        NativeMetricKind::Bytes
    };
    (timer) => {
        NativeMetricKind::Timer
    };
}

macro_rules! define_contract {
    ($( $field:ident => $name:literal: $kind:ident, )*) => {
        /// Stable public DataFusion metric-name contract exposed by native cuDF execution plans.
        ///
        /// Every name is registered as a DataFusion `MetricType::Summary` metric. Timer names
        /// use DataFusion `Time` storage in nanoseconds; all other entries use `Count` storage.
        pub const NATIVE_CUDF_DATAFUSION_METRIC_NAMES: &[&str] = &[$($name),*];

        #[cfg(test)]
        pub(super) const NATIVE_CUDF_METRIC_DESCRIPTORS: &[NativeMetricDescriptor] = &[
            $(NativeMetricDescriptor { name: $name, kind: metric_kind!($kind) }),*
        ];

        /// Generated recording identifiers for labelled summary rows.
        ///
        /// Recorders use these accessors instead of repeating public metric-name literals.
        /// The ordered macro emits one accessor per public metric; only the labelled-row
        /// recorders read them today.
        #[allow(dead_code)] // macro emits one accessor per contract entry; subset is referenced
        pub(super) mod metric_name {
            $(
                pub(in crate::native_exec::metrics) const fn $field() -> &'static str {
                    $name
                }
            )*
        }

        /// Type-level ids used to require one recording implementation for
        /// every ordered descriptor at compile time.
        #[allow(non_camel_case_types)]
        pub(super) mod recording_marker {
            $(
                pub(in crate::native_exec::metrics) enum $field {}
            )*
        }
    };
}

for_each_native_cudf_metric!(define_contract);
