use super::*;

#[test]
fn benchmark_report_accepted_rows_include_bounded_status_contract_rows_and_resource_metrics() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string()],
        native_fallback_policy: GpuFallbackPolicy::GpuOnly,
        ..NativeTpchBenchmarkConfig::default()
    };
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
        source_predicate_pushdown_status: "reader_pushdown".to_string(),
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
            graph_runtime_rows: vec![NativeGraphRuntimeMetricRow {
                node_id: "g0".to_string(),
                algorithm: "pagerank".to_string(),
                runtime_outcome: "succeeded".to_string(),
                input_boundary: "device".to_string(),
                output_boundary: "device".to_string(),
                construction_policy: "python_cugraph".to_string(),
                source_column: "src".to_string(),
                destination_column: "dst".to_string(),
                weight_column: Some("weight".to_string()),
                directed: true,
                renumber: true,
                algorithm_parameters: "alpha=0.85,epsilon=0.000001,max_iterations=100".to_string(),
                algorithm_metadata: "converged=true,iterations=12".to_string(),
                input_rows: 100,
                input_columns: 3,
                output_rows: 4,
                output_columns: 2,
                output_device_bytes: Some(320),
                graph_peak_bytes: Some(2048),
                elapsed_ns: 123_000,
                gpu_resident_input_handoffs: 1,
                host_to_device_uploads: 0,
                device_to_host_materializations: 0,
            }],
            task_operator_rows: vec![NativeTaskOperatorMetricRow {
                operator_kind: "source_read".to_string(),
                task_count: 3,
                elapsed_ns: 7_000_000,
                reservation_requested_bytes: 0,
                reservation_granted_bytes: 0,
                reservation_released_bytes: 0,
                reservation_live_peak_bytes: 64,
                max_observed_peak_bytes: None,
                max_retained_bytes: None,
            }],
            task_runtime_rows: vec![NativeTaskRuntimeMetricRow {
                compiled_task_id: "t0".to_string(),
                operator_kind: "source_read".to_string(),
                elapsed_ns: 7_000_000,
                reservation_requested_bytes: 64,
                reservation_granted_bytes: 64,
                reservation_released_bytes: 64,
                reservation_live_peak_bytes: 64,
                max_observed_peak_bytes: Some(65),
                max_retained_bytes: Some(66),
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
            source_fetch_bytes: 33,
            source_fetch_count: 34,
            row_groups_planned: 35,
            row_groups_committed: 36,
            partial_aggregate_count: 37,
            partial_aggregate_input_rows: 38,
            partial_aggregate_rows: 39,
            partial_aggregate_retained_bytes: 40,
            final_merge_attempts: 41,
            final_merge_levels: 42,
            final_merge_max_fan_in: 43,
            retry_count: 44,
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
            aggregate_strategy_rows: vec![NativeAggregateStrategyMetricRow {
                node_id: "n3".to_string(),
                selected_strategy: "libcudf_groupby".to_string(),
                execution_stage: "partial".to_string(),
                memory_mode: "bounded".to_string(),
                group_key_count: 1,
                group_key_exprs: "l_returnflag".to_string(),
                group_key_dtypes: "Utf8".to_string(),
                aggregate_functions: "sum,count".to_string(),
                floating_statistical_aggregate: "".to_string(),
                statistical_aggregate_execution_mode: "".to_string(),
                statistical_aggregate_tolerance_contract: "".to_string(),
                statistical_aggregate_functions: "".to_string(),
                statistical_aggregate_ddof: "".to_string(),
                stddev_samp_execution_mode: "".to_string(),
                stddev_samp_ddof: "".to_string(),
                aggregate_filter_count: 0,
                filtered_input_materialization_strategy: "".to_string(),
                filtered_input_materialization_estimated_bytes: 0,
                filtered_input_materialization_budget_bytes: 0,
                filtered_input_materialization_threshold_bytes: 0,
                filtered_input_lower_level_unresolved_contracts: "".to_string(),
                filtered_global_rows_per_chunk: 0,
                filtered_global_inputs_per_segment: 0,
                filtered_global_segment_target_temporary_bytes: 0,
                filtered_global_actual_chunk_count: 0,
                filtered_global_actual_segment_count: 0,
                filtered_global_max_segment_rows: 0,
                filtered_global_observed_temporary_peak_bytes: 0,
                filtered_global_observed_retained_peak_bytes: 0,
                filtered_global_state_rows_merged: 0,
                filtered_grouped_rows_per_chunk: 0,
                filtered_grouped_inputs_per_segment: 0,
                filtered_grouped_segment_target_temporary_bytes: 0,
                filtered_grouped_actual_chunk_count: 0,
                filtered_grouped_actual_segment_count: 0,
                filtered_grouped_max_segment_rows: 0,
                filtered_grouped_observed_temporary_peak_bytes: 0,
                filtered_grouped_observed_retained_peak_bytes: 0,
                filtered_grouped_state_rows_merged: 0,
                filtered_grouped_high_cardinality_admission: "".to_string(),
                distinct_aggregate_count: 0,
                input_rows: 6001215,
                output_group_count: 4,
                reduction_ratio_ppm: 0,
                chunk_row_count: 6001215,
                chunk_count: 1,
                partial_aggregate_count: 0,
                source_adjacent_partial: "true".to_string(),
                source_chunk_row_group_count: "1".to_string(),
                source_chunk_reads_all_row_groups: "false".to_string(),
                source_chunk_known_row_count: "6001215".to_string(),
                source_chunk_estimated_source_read_bytes: "123456".to_string(),
                source_chunk_size_estimate_based: "false".to_string(),
                source_adjacent_partition_strategy: "row_balanced".to_string(),
                source_adjacent_partition_count: "32".to_string(),
                source_adjacent_non_empty_partition_count: "32".to_string(),
                source_adjacent_partition_key_count: "0".to_string(),
                source_adjacent_max_partition_fan_in: "17".to_string(),
                partitioned_merge_partition_count: "unknown".to_string(),
                partitioned_merge_non_empty_partition_count: "unknown".to_string(),
                partitioned_merge_max_partition_fan_in: "unknown".to_string(),
                selection_elapsed_ns: 0,
                encode_elapsed_ns: 0,
                groupby_elapsed_ns: 123456789,
                decode_elapsed_ns: 0,
                strategy_operator_elapsed_ns: 123456789,
                aggregate_structural_key: "agg-v1-fnv64-0123456789abcdef".to_string(),
                prior_cardinality_evidence_scope: "absent".to_string(),
                encoded_key_partial_cost_model: "prior_partial_reduction_ratio".to_string(),
                encoded_key_partial_cost_model_status: "rejected_by_cost_model".to_string(),
                encoded_key_partial_selection_reason: "no_prior_cardinality_evidence".to_string(),
                encoded_key_partial_reduction_ratio_threshold_ppm: 100000,
                encoded_key_partial_prior_reduction_ratio_ppm: 0,
                encoded_key_partial_prior_strategy: "".to_string(),
                exact_cardinality_probe_status: "not_attempted".to_string(),
                exact_cardinality_probe_key: "none".to_string(),
                exact_cardinality_probe_input_rows: 0,
                exact_cardinality_probe_non_null_distinct_count: "unknown".to_string(),
                exact_cardinality_probe_null_count: "unknown".to_string(),
                exact_cardinality_probe_num_groups: "unknown".to_string(),
                exact_cardinality_probe_reduction_ratio_ppm: "unknown".to_string(),
                exact_cardinality_probe_elapsed_ns: 0,
                prepared_sort_groupby_rejected_reason: "none".to_string(),
                prepared_sort_groupby_contract_status: "available".to_string(),
                prepared_sort_groupby_required_contract: "prepared_sort_groupby_execution_contract".to_string(),
                prepared_sort_groupby_cost_model: "exact_grouped_key_facts_required".to_string(),
                prepared_sort_groupby_cost_model_status: "selected_libcudf_runtime_probe".to_string(),
                prepared_sort_groupby_selection_reason: "runtime_cardinality_probe_required".to_string(),
                prepared_sort_groupby_num_groups: "unknown".to_string(),
                prepared_sort_groupby_grouped_key_output_bytes: "unknown".to_string(),
                prepared_sort_groupby_max_group_size: "unknown".to_string(),
                rejected_candidate_strategies:
                    "encoded_key_partial:encoded_key_partial_cost_model_no_prior_cardinality_evidence;partitioned_merge:partitioned_merge_requires_final_merge_stage"
                        .to_string(),
                ..NativeAggregateStrategyMetricRow::default()
            }],
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
                estimated_output_rows: Some(102),
                estimated_output_bytes: Some(4096),
                build_key_unique: Some("true".to_string()),
                build_key_uniqueness_probe_status: Some("measured".to_string()),
                build_key_uniqueness_probe_error_code: None,
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
            host_fetch_budget_wait_time_ns: Some(5_000_000),
            source_fetch_concurrency_wait_time_ns: Some(6_000_000),
            source_fetch_time_ns: Some(7_000_000),
            source_decode_time_ns: Some(8_000_000),
            ..Default::default()
        },
        timing_metrics: NativeTpchTimingMetrics {
            parquet_read_time_ms: Some(1.0),
            source_read_time_ms: Some(2.0),
            source_read_queue_wait_time_ms: Some(3.0),
            source_cache_lookup_time_ms: Some(4.0),
            host_fetch_budget_wait_time_ms: Some(5.0),
            source_fetch_concurrency_wait_time_ms: Some(6.0),
            source_fetch_time_ms: Some(7.0),
            source_decode_time_ms: Some(8.0),
            operator_time_ms: Some(9.0),
            chunk_operator_time_ms: Some(10.0),
            final_merge_time_ms: Some(11.0),
            final_sort_time_ms: Some(12.0),
            materialization_time_ms: Some(13.0),
        },
    };
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    assert!(text.contains("# tpch_benchmark_native_fallback_policy=gpu_only\n"));
    assert!(text.contains(
        "q01\taccepted_replaced\tbounded_accepted_parity_matched\tnone\tnone\tnone\tnone\tnone\tsort_local_parquet_aggregate\tnative\tfalse\ttrue\ttrue\tsucceeded"
    ));
    assert!(text.contains(
        "\t20.000\t5.000\t4.000\t1\t1\t4\t4\t128\t128\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\t"
    ));
    assert!(text.contains("\t64\t65\t66\tnone\tnode_id=g0,algorithm=pagerank,runtime_outcome=succeeded,input_boundary=device,output_boundary=device,construction_policy=python_cugraph,source_column=src,destination_column=dst,weight_column=weight,directed=true,renumber=true,algorithm_parameters=alpha=0.85,epsilon=0.000001,max_iterations=100,algorithm_metadata=converged=true,iterations=12,input_rows=100,input_columns=3,output_rows=4,output_columns=2,output_device_bytes=320,graph_peak_bytes=2048,elapsed_ns=123000,gpu_resident_input_handoffs=1,host_to_device_uploads=0,device_to_host_materializations=0\toperator_kind=source_read,task_count=3,elapsed_ns=7000000,reservation_requested_bytes=0,reservation_granted_bytes=0,reservation_released_bytes=0,reservation_live_peak_bytes=64,max_observed_peak_bytes=none,max_retained_bytes=none\tcompiled_task_id=t0,operator_kind=source_read,elapsed_ns=7000000,reservation_requested_bytes=64,reservation_granted_bytes=64,reservation_released_bytes=64,reservation_live_peak_bytes=64,max_observed_peak_bytes=65,max_retained_bytes=66\t"));
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let fields = lines.next().expect("benchmark report should contain a q01 row").split('\t').collect::<Vec<_>>();
    let header_fields = header.split('\t').collect::<Vec<_>>();
    assert_eq!(header, NATIVE_TPCH_BENCHMARK_TSV_HEADER);
    assert_eq!(fields.len(), header_fields.len());
    assert_eq!(fields[header_index(&header_fields, "peak_memory_operator_class")], "source");
    assert_eq!(fields[header_index(&header_fields, "peak_memory_operator_class_bytes")], "64");
    assert_eq!(fields[header_index(&header_fields, "task_reservation_operator_class")], "source");
    assert_eq!(fields[header_index(&header_fields, "task_reservation_operator_class_bytes")], "64");
    assert_eq!(fields[header_index(&header_fields, "runtime_operator_class")], "materialization");
    assert_eq!(fields[header_index(&header_fields, "runtime_operator_class_ms")], "13.000");
    assert_eq!(fields[header_index(&header_fields, "source_predicate_pushdown_status")], "reader_pushdown");
    assert_eq!(fields[header_index(&header_fields, "partitioned_join_count")], "0");
    assert_eq!(fields[header_index(&header_fields, "join_partitioning_status")], "not_partitioned");
    assert_eq!(fields[header_index(&header_fields, "aggregate_merge_status")], "thresholded");
    assert_eq!(fields[header_index(&header_fields, "source_fetch_bytes")], "33");
    assert_eq!(fields[header_index(&header_fields, "source_fetch_count")], "34");
    assert_eq!(fields[header_index(&header_fields, "row_groups_planned")], "35");
    assert_eq!(fields[header_index(&header_fields, "row_groups_committed")], "36");
    assert_eq!(fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_count")], "1");
    assert_eq!(fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_elapsed_ns")], "123456789");
    assert_eq!(fields[header_index(&header_fields, "aggregate_libcudf_runtime_probe_input_rows")], "6001215");
    assert_eq!(fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_count")], "0");
    assert_eq!(fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_elapsed_ns")], "0");
    assert_eq!(fields[header_index(&header_fields, "aggregate_exact_cardinality_probe_input_rows")], "0");
    assert_eq!(fields[header_index(&header_fields, "aggregate_prepared_sort_groupby_elapsed_ns")], "0");
    assert_eq!(fields[header_index(&header_fields, "aggregate_encoded_key_partial_elapsed_ns")], "0");
    assert!(
        fields[header_index(&header_fields, "task_high_water_rows")]
            .contains("operator_class=source,task_count=3,max_task_reservation_bytes=64")
    );
    assert!(
        fields[header_index(&header_fields, "task_high_water_rows")]
            .contains("operator_class=final_merge,task_count=41")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("node_id=n3,selected_strategy=libcudf_groupby,execution_stage=partial")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("source_adjacent_partition_strategy=row_balanced")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("partitioned_merge_partition_count=unknown")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")].contains("filtered_grouped_rows_per_chunk=0")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("filtered_input_lower_level_unresolved_contracts=")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("prepared_sort_groupby_cost_model=exact_grouped_key_facts_required")
    );
    assert!(
        fields[header_index(&header_fields, "aggregate_strategy_rows")]
            .contains("prepared_sort_groupby_num_groups=unknown")
    );
    assert!(text.contains("\tnode_id=n7,join_strategy=one_shot_equi,execution_strategy=bounded_one_shot_equi_join"));
    assert!(
        text.contains(
            "\t1.000\t2.000\t3.000\t4.000\t5.000\t6.000\t7.000\t8.000\t9.000\t10.000\t11.000\t12.000\t13.000\n"
        )
    );
}

#[test]
fn benchmark_report_partitioned_join_count_sums_runtime_executions_but_status_counts_sites() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q07".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut native = timed_native_run("output-fnv1a64:0000000000000007");
    native.metrics = NativeDataFusionExecutionMetrics {
        join_strategy_rows: vec![NativeJoinStrategyMetricRow {
            node_id: "n7".to_string(),
            join_strategy: "partitioned_hash_equi".to_string(),
            execution_strategy: "bounded_partitioned_hash_equi_join".to_string(),
            join_type: "inner".to_string(),
            build_side: "right".to_string(),
            key_count: 1,
            partitioned_hash_join_count: 2,
            ..Default::default()
        }],
        ..Default::default()
    };
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config,
        rows: vec![NativeTpchBenchmarkRow::accepted(
            "q07",
            accepted_planning_row("q07"),
            Ok(timed_cpu_run("output-fnv1a64:0000000000000007")),
            Ok(native),
        )],
    };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let fields = lines.next().expect("benchmark report should contain a q07 row").split('\t').collect::<Vec<_>>();
    let header_fields = header.split('\t').collect::<Vec<_>>();

    assert_eq!(fields[header_index(&header_fields, "partitioned_join_count")], "2");
    assert_eq!(fields[header_index(&header_fields, "join_partitioning_status")], "partitioned");
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
