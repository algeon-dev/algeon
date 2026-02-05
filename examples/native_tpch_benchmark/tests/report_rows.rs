use super::*;

#[test]
fn benchmark_report_accepted_rows_include_bounded_status_contract_rows_and_resource_metrics() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q01".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let row = NativeTpchBenchmarkRow {
        query_id: "q01".to_string(),
        planning_outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
        strict_bounded_contract_rows: Vec::new(),
        strict_bounded_compiled_graphs: Vec::new(),
        reason_code: None,
        unsupported_category: None,
        native_candidate_shape_family: Some("sort_local_parquet_aggregate".to_string()),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Native,
        rejected_before_execution: false,
        cpu_runtime_started: true,
        native_runtime_started: true,
        runtime_outcome: NativeDataFusionRuntimeOutcome::Succeeded,
        cpu_failure: None,
        cpu_failure_details: execution::FailureDetails::default(),
        runtime_failure: None,
        runtime_failure_details: execution::FailureDetails::default(),
        parity_status: validation::ParityStatus::Matched,
        cpu_wall_ms: Some(20.0),
        native_wall_ms: Some(5.0),
        speedup_vs_cpu: Some(4.0),
        cpu_output_batches: Some(1),
        native_output_batches: Some(1),
        cpu_output_rows: Some(4),
        native_output_rows: Some(4),
        cpu_output_bytes: Some(128),
        native_output_bytes: Some(128),
        cpu_output_fingerprint: Some("output-fnv1a64:1111111111111111".to_string()),
        native_output_fingerprint: Some("output-fnv1a64:1111111111111111".to_string()),
        output_fingerprint: Some("output-fnv1a64:1111111111111111".to_string()),
        metrics: NativeDataFusionExecutionMetrics {
            gpu_resident_input_handoffs: 1,
            host_to_device_uploads: 2,
            host_to_device_uploaded_rows: 3,
            host_to_device_uploaded_bytes: 4,
            device_to_host_materializations: 5,
            device_to_host_materialized_rows: 6,
            device_to_host_materialized_bytes: 7,
            device_to_device_copies: 8,
            device_to_device_copied_bytes: 9,
            parquet_files: 10,
            parquet_bytes_known: 11,
            row_groups_selected: 12,
            read_columns: 13,
            resident_dataframe_peak_bytes: Some(14),
            allocator_high_water_bytes: Some(15),
            reservation_requested_bytes: 16,
            reservation_granted_bytes: 17,
            reservation_denied_count: 18,
            source_chunks_planned: 19,
            source_chunks_attempted: 20,
            source_chunks_committed: 21,
            source_chunks_retried: 22,
            source_read_prefetches_started: 23,
            source_read_prefetches_completed: 24,
            source_read_prefetches_skipped: 25,
            source_read_prefetches_abandoned: 26,
            source_cache_hits: 27,
            source_cache_misses: 28,
            source_cache_inserts: 29,
            source_cache_insert_skipped: 30,
            source_cache_retained_bytes: 31,
            row_groups_planned: 32,
            row_groups_committed: 33,
            partial_aggregate_count: 34,
            partial_aggregate_rows: 35,
            final_merge_attempts: 36,
            final_merge_levels: 37,
            retry_count: 38,
            retry_resource_exhausted_count: 39,
            retry_reservation_failure_count: 40,
            hash_join_build_count: 41,
            filtered_join_build_count: 42,
            hash_join_cache_hits: 43,
            hash_join_cache_misses: 44,
            hash_join_durable_cache_hits: 45,
            hash_join_durable_cache_misses: 46,
            hash_join_durable_cache_inserts: 47,
            hash_join_hash_table_bytes: 48,
            filtered_join_hash_table_bytes: 49,
            filtered_join_probe_state_bytes: 50,
            mixed_join_size_data_bytes: 51,
            join_gather_map_bytes: 52,
        },
        timing_metrics: NativeTpchTimingMetrics {
            parquet_read_time_ms: Some(1.0),
            source_read_time_ms: Some(2.0),
            source_read_queue_wait_time_ms: Some(3.0),
            source_cache_lookup_time_ms: Some(4.0),
            operator_time_ms: Some(5.0),
            chunk_operator_time_ms: Some(6.0),
            final_merge_time_ms: Some(7.0),
            final_sort_time_ms: Some(8.0),
            materialization_time_ms: Some(9.0),
        },
    };
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    assert!(text.contains(
        "q01\taccepted_replaced\tbounded_accepted_parity_matched\tnone\tnone\tnone\tnone\tsort_local_parquet_aggregate\tnative\tfalse\ttrue\ttrue\tsucceeded"
    ));
    assert!(text.contains(
        "\t20.000\t5.000\t4.000\t1\t1\t4\t4\t128\t128\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\t"
    ));
    assert!(text.contains(
        "\t27\t28\t29\t30\t31\t32\t33\t34\t35\t36\t37\t38\t39\t40\t41\t42\t43\t44\t45\t46\t47\t48\t49\t50\t51\t"
    ));
    assert!(text.contains("\t1.000\t2.000\t3.000\t4.000\t5.000\t6.000\t7.000\t8.000\t9.000\n"));
}

#[test]
fn benchmark_report_accepted_join_rows_emit_strict_contract_rows() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q15".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q15");
    planning.native_candidate_shape_family = Some("sort_hash_join_input".to_string());
    planning.strict_bounded_contract_rows = accepted_strict_bounded_contract_rows();
    planning.strict_bounded_compiled_graphs = accepted_strict_bounded_compiled_graphs();
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config,
        rows: vec![NativeTpchBenchmarkRow::accepted(
            "q15",
            planning,
            Ok(timed_cpu_run("output-fnv1a64:0000000000000015")),
            Ok(timed_native_run("output-fnv1a64:0000000000000015")),
        )],
    };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q15 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let contract_rows_index = header_index(&header_fields, "strict_bounded_contract_rows");
    let graph_index = header_index(&header_fields, "strict_bounded_compiled_graphs");
    assert_eq!(fields[0], "q15");
    assert!(fields[contract_rows_index].contains("execution_mode=strict_bounded"));
    assert!(fields[contract_rows_index].contains("contract_status=modeled"));
    assert!(fields[contract_rows_index].contains("contract_quality=hard_budgeted_report_with_unresolved_preflight"));
    assert!(
        fields[contract_rows_index].contains("unresolved_preflight_gap=libcudf_final_sort_order_preflight_unavailable")
    );
    assert!(!fields[contract_rows_index].contains("bounded_join_execution_family"));
    assert!(fields[graph_index].contains("sink_kind=host_arrow_materialization"));
    assert!(text.contains("\tmatched\t"));
}
