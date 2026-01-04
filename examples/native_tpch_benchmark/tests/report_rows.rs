use super::*;

#[test]
fn benchmark_report_accepted_rows_include_bounded_status_and_resource_metrics() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q01".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let row = NativeTpchBenchmarkRow {
        query_id: "q01".to_string(),
        planning_outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
        bounded_join_reservation_facts: BTreeMap::new(),
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
            row_groups_planned: 27,
            row_groups_committed: 28,
            partial_aggregate_count: 29,
            partial_aggregate_rows: 30,
            final_merge_attempts: 31,
            final_merge_levels: 32,
            retry_count: 33,
            retry_resource_exhausted_count: 34,
            retry_reservation_failure_count: 35,
            hash_join_build_count: 36,
            filtered_join_build_count: 37,
            hash_join_cache_hits: 38,
            hash_join_cache_misses: 39,
            hash_join_durable_cache_hits: 40,
            hash_join_durable_cache_misses: 41,
            hash_join_durable_cache_inserts: 42,
            hash_join_hash_table_bytes: 43,
            filtered_join_hash_table_bytes: 44,
            filtered_join_probe_state_bytes: 45,
            mixed_join_size_data_bytes: 46,
            join_gather_map_bytes: 47,
        },
        timing_metrics: NativeTpchTimingMetrics {
            parquet_read_time_ms: Some(1.0),
            source_read_time_ms: Some(2.0),
            source_read_queue_wait_time_ms: Some(3.0),
            operator_time_ms: Some(4.0),
            chunk_operator_time_ms: Some(5.0),
            final_merge_time_ms: Some(6.0),
            final_sort_time_ms: Some(7.0),
            materialization_time_ms: Some(8.0),
        },
    };
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    assert!(text.contains(
        "q01\taccepted_replaced\tbounded_accepted_parity_matched\tnone\tnone\tnone\tsort_local_parquet_aggregate\tnative\tfalse\ttrue\ttrue\tsucceeded"
    ));
    assert!(text.contains(
        "\t20.000\t5.000\t4.000\t1\t1\t4\t4\t128\t128\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\toutput-fnv1a64:1111111111111111\t"
    ));
    assert!(text.contains("\t27\t28\t29\t30\t31\t32\t33\t34\t35\t36\t37\t38\t39\t40\t41\t42\t43\t44\t45\t46\t"));
    assert!(text.contains("\t1.000\t2.000\t3.000\t4.000\t5.000\t6.000\t7.000\t8.000\n"));
}

#[test]
fn benchmark_report_q07_accepted_join_rows_preserve_bounded_join_facts() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q07".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q07");
    planning.native_candidate_shape_family = Some("sort_hash_join_aggregate".to_string());
    planning.bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q07",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000007")),
        Ok(timed_native_run("output-fnv1a64:0000000000000007")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q07 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q07 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");

    assert_eq!(fields[0], "q07");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(fields[bounded_join_facts_index].contains("bounded_join_lower_level_blockers="));
    assert_eq!(fields[shape_family_index], "sort_hash_join_aggregate");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
}

#[test]
fn benchmark_report_q13_accepted_join_rows_preserve_bounded_facts_and_parity_fields() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q13".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q13");
    planning.native_candidate_shape_family = Some("sort_hash_join_aggregate".to_string());
    planning.bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q13",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000013")),
        Ok(timed_native_run("output-fnv1a64:0000000000000013")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q13 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q13 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let cpu_started_index = header_index(&header_fields, "cpu_runtime_started");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let parity_status_index = header_index(&header_fields, "parity_status");
    let cpu_fingerprint_index = header_index(&header_fields, "cpu_output_fingerprint");
    let native_fingerprint_index = header_index(&header_fields, "native_output_fingerprint");
    let output_fingerprint_index = header_index(&header_fields, "output_fingerprint");

    assert_eq!(fields[0], "q13");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(fields[bounded_join_facts_index].contains("bounded_join_lower_level_blockers="));
    assert_eq!(fields[shape_family_index], "sort_hash_join_aggregate");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[cpu_started_index], "true");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
    assert_eq!(fields[parity_status_index], "matched");
    assert_eq!(fields[cpu_fingerprint_index], "output-fnv1a64:0000000000000013");
    assert_eq!(fields[native_fingerprint_index], "output-fnv1a64:0000000000000013");
    assert_eq!(fields[output_fingerprint_index], "output-fnv1a64:0000000000000013");
}

#[test]
fn benchmark_report_q16_accepted_join_rows_preserve_bounded_facts_and_parity_fields() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q16".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q16");
    planning.native_candidate_shape_family = Some("sort_hash_join_aggregate".to_string());
    planning.bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q16",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000016")),
        Ok(timed_native_run("output-fnv1a64:0000000000000016")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q16 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q16 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let cpu_started_index = header_index(&header_fields, "cpu_runtime_started");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let parity_status_index = header_index(&header_fields, "parity_status");
    let cpu_fingerprint_index = header_index(&header_fields, "cpu_output_fingerprint");
    let native_fingerprint_index = header_index(&header_fields, "native_output_fingerprint");
    let output_fingerprint_index = header_index(&header_fields, "output_fingerprint");

    assert_eq!(fields[0], "q16");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(fields[bounded_join_facts_index].contains("bounded_join_lower_level_blockers="));
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_mixed_keyed_residual_multi_join_capability=multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline"
    ));
    assert_eq!(fields[shape_family_index], "sort_hash_join_aggregate");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[cpu_started_index], "true");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
    assert_eq!(fields[parity_status_index], "matched");
    assert_eq!(fields[cpu_fingerprint_index], "output-fnv1a64:0000000000000016");
    assert_eq!(fields[native_fingerprint_index], "output-fnv1a64:0000000000000016");
    assert_eq!(fields[output_fingerprint_index], "output-fnv1a64:0000000000000016");
}

#[test]
fn benchmark_report_q21_accepted_join_rows_preserve_expanded_mixed_residual_facts() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q21".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q21");
    planning.native_candidate_shape_family = Some("sort_fetch_hash_join_aggregate".to_string());
    planning.bounded_join_reservation_facts = expanded_mixed_residual_multi_join_facts();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q21",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000021")),
        Ok(timed_native_run("output-fnv1a64:0000000000000021")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q21 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q21 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");

    assert_eq!(fields[0], "q21");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_mixed_keyed_residual_multi_join_capability=multi_join_mixed_keyed_residual_inner_left_semi_left_anti_pipeline"
    ));
    assert!(fields[bounded_join_facts_index]
        .contains("bounded_join_mixed_keyed_residual_multi_join_capability_detail=mixed_keyed_residual_multi_join_types=inner,left_semi,left_anti"));
    assert_eq!(fields[shape_family_index], "sort_fetch_hash_join_aggregate");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
}

#[test]
fn benchmark_report_q11_accepted_rows_preserve_keyless_aggregate_over_join_subplan_inputs_facts() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q11".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q11");
    planning.native_candidate_shape_family = Some("sort_mixed_hash_nested_loop_join_input".to_string());
    planning.bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q11",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000011")),
        Ok(timed_native_run("output-fnv1a64:0000000000000011")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q11 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q11 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let parity_status_index = header_index(&header_fields, "parity_status");

    assert_eq!(fields[0], "q11");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(
        fields[bounded_join_facts_index]
            .contains("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
    );
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability=keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline"
    ));
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail=root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort"
    ));
    assert_eq!(fields[shape_family_index], "sort_mixed_hash_nested_loop_join_input");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
    assert_eq!(fields[parity_status_index], "matched");
}

#[test]
fn benchmark_report_q22_accepted_rows_preserve_keyless_input_subplan_facts() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q22".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let mut planning = accepted_planning_row("q22");
    planning.native_candidate_shape_family = Some("sort_mixed_hash_nested_loop_join_aggregate".to_string());
    planning.bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    let expected_bounded_join_facts = expected_key_value_facts(&planning.bounded_join_reservation_facts);
    let row = NativeTpchBenchmarkRow::accepted(
        "q22",
        planning,
        Ok(timed_cpu_run("output-fnv1a64:0000000000000022")),
        Ok(timed_native_run("output-fnv1a64:0000000000000022")),
    );
    let report = NativeTpchBenchmarkReport { metadata: benchmark_metadata(), config, rows: vec![row] };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q22 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q22 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let shape_family_index = header_index(&header_fields, "native_candidate_shape_family");
    let execution_path_index = header_index(&header_fields, "post_native_execution_path");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");

    assert_eq!(fields[0], "q22");
    assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(fields[bounded_join_facts_index]
        .contains("bounded_join_execution_families=root_reusable_hash_join,root_filtered_semi_anti_join,keyless_conditional_join,keyless_conditional_join_input_subplan_pipeline"));
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability=keyless_conditional_join_input_subplan_pipeline"
    ));
    assert!(fields[bounded_join_facts_index].contains(
        "bounded_join_keyless_conditional_join_input_subplan_pipeline_capability_detail=root_join=keyless_conditional_inner;input_subplans=source_filter_projection_or_projection_wrapped_left_anti_hash_join_or_scalar_global_aggregate"
    ));
    assert_eq!(fields[shape_family_index], "sort_mixed_hash_nested_loop_join_aggregate");
    assert_eq!(fields[execution_path_index], "native");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
}
