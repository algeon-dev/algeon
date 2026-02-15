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
        adaptive_pipeline_rows: Vec::new(),
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
            reservation_live_peak_bytes: 18,
            reservation_denied_count: 19,
            max_task_reservation_bytes: 64,
            max_task_observed_peak_bytes: Some(65),
            max_task_retained_bytes: Some(66),
            task_operator_rows: vec![NativeTaskOperatorMetricRow {
                operator_kind: "source_read".to_string(),
                task_count: 3,
                reservation_live_peak_bytes: 64,
            }],
            source_chunks_planned: 20,
            source_chunks_attempted: 21,
            source_chunks_committed: 22,
            source_chunks_retried: 23,
            source_read_prefetches_started: 24,
            source_read_prefetches_completed: 25,
            source_read_prefetches_skipped: 26,
            source_read_prefetches_abandoned: 27,
            source_cache_hits: 28,
            source_cache_misses: 29,
            source_cache_inserts: 30,
            source_cache_insert_skipped: 31,
            source_cache_retained_bytes: 32,
            row_groups_planned: 33,
            row_groups_committed: 34,
            partial_aggregate_count: 35,
            partial_aggregate_input_rows: 36,
            partial_aggregate_rows: 37,
            partial_aggregate_retained_bytes: 38,
            final_merge_attempts: 39,
            final_merge_levels: 40,
            final_merge_max_fan_in: 41,
            retry_count: 42,
            retry_resource_exhausted_count: 43,
            retry_reservation_failure_count: 44,
            adaptive_retry_count: 45,
            adaptive_spill_count: 46,
            adaptive_downgrade_count: 47,
            adaptive_memory_history_lookup_count: 48,
            adaptive_memory_history_hit_count: 49,
            adaptive_memory_history_miss_count: 50,
            hash_join_build_count: 51,
            filtered_join_build_count: 52,
            one_shot_join_count: 53,
            hash_join_cache_hits: 54,
            hash_join_cache_misses: 55,
            hash_join_durable_cache_hits: 56,
            hash_join_durable_cache_misses: 57,
            hash_join_durable_cache_inserts: 58,
            hash_join_hash_table_bytes: 59,
            filtered_join_hash_table_bytes: 60,
            filtered_join_probe_state_bytes: 61,
            mixed_join_size_data_bytes: 62,
            one_shot_join_index_bytes: 63,
            one_shot_join_materialization_bytes: 64,
            one_shot_join_output_bytes: 65,
            join_gather_map_bytes: 66,
            join_strategy_rows: vec![NativeJoinStrategyMetricRow {
                node_id: "n7".to_string(),
                join_strategy: "one_shot_equi".to_string(),
                execution_strategy: "bounded_one_shot_equi_join".to_string(),
                join_type: "inner".to_string(),
                build_side: "right".to_string(),
                key_count: 1,
                estimated_build_rows: Some(100),
                observed_build_rows: Some(101),
                build_side_bytes: Some(1024),
                probe_side_bytes: Some(2048),
                probe_reuse_count: Some(1),
                partition_count: None,
                partition_target_bytes: None,
                build_partition_batch_count: None,
                probe_partition_batch_count: None,
                build_partitioned_bytes: None,
                probe_partitioned_bytes: None,
                reusable_hash_join_rejected_reason: Some("build_side_too_large".to_string()),
                partitioned_hash_join_rejected_reason: Some("partitioned_hash_equi_unsupported".to_string()),
                one_shot_equi_rejected_reason: None,
                partitioned_hash_join_count: 0,
                hash_join_build_count: 0,
                hash_join_hash_table_bytes: 0,
                filtered_join_build_count: 0,
                filtered_join_hash_table_bytes: 0,
                filtered_join_probe_state_bytes: 0,
                mixed_join_size_data_bytes: 0,
                one_shot_join_count: 1,
                one_shot_join_index_bytes: 2048,
                one_shot_join_materialization_bytes: 4096,
                one_shot_join_output_bytes: 8192,
                join_gather_map_bytes: 2048,
                join_output_reservation_bytes: 4096,
                join_output_rows: 4,
                join_output_bytes: 8192,
            }],
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
        "q01\taccepted_replaced\tbounded_accepted_parity_matched\tnone\tnone\tnone\tnone\tnone\tsort_local_parquet_aggregate\tnative\tfalse\ttrue\ttrue\tsucceeded"
    ));
    assert!(text.contains(
        "\t20.000\t5.000\t4.000\t1\t1\t4\t4\t128\t128\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\t"
    ));
    assert!(text.contains(
        "\t28\t29\t30\t31\t32\t33\t34\t35\t36\t37\t38\t39\t40\t41\t42\t43\t44\t45\t46\t47\t48\t49\t50\t51\t52\t53\t54\t55\t"
    ));
    assert!(text.contains("\t64\t65\t66\toperator_kind=source_read,task_count=3,reservation_live_peak_bytes=64\t"));
    assert!(text.contains("\tnode_id=n7,join_strategy=one_shot_equi,execution_strategy=bounded_one_shot_equi_join"));
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
    assert!(
        fields[contract_rows_index]
            .contains("unresolved_preflight_gap=libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")
    );
    assert!(!fields[contract_rows_index].contains("bounded_join_execution_family"));
    assert!(!fields[contract_rows_index].contains("retry"));
    assert!(!fields[contract_rows_index].contains("spill"));
    assert!(!fields[contract_rows_index].contains("downgrade"));
    assert!(!fields[contract_rows_index].contains("memory_history"));
    assert!(fields[graph_index].contains("sink_kind=host_arrow_materialization"));
    assert!(text.contains("\tmatched\t"));
}
