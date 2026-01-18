use super::*;

#[test]
fn benchmark_report_header_and_rejected_rows_are_stable() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q02".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = benchmark_metadata();
    let mut bounded_join_reservation_facts = BTreeMap::new();
    bounded_join_reservation_facts
        .insert("bounded_join_execution_status".to_string(), "unsupported_plan_shape".to_string());
    bounded_join_reservation_facts.insert(
        "bounded_join_unsupported_shape".to_string(),
        cudf_query_engine::BoundedJoinReservationContract.supported_shape_requirement(),
    );
    let planning = NativeDataFusionPlanningRow {
        query_id: "q02".to_string(),
        physical_root: "SortExec".to_string(),
        physical_shape: "SortExec(HashJoinExec)".to_string(),
        native_candidate_shape: Some("SortExec(HashJoinExec)".to_string()),
        native_candidate_shape_family: Some("sort_hash_join_aggregate".to_string()),
        outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some("native_bounded_join_unsupported".to_string()),
        unsupported_category: Some("operator_hashjoin".to_string()),
        bounded_join_reservation_facts,
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "SortExec".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        rejected_before_execution: true,
    };
    let report =
        NativeTpchBenchmarkReport { metadata, config, rows: vec![NativeTpchBenchmarkRow::rejected("q02", planning)] };

    let text = report.to_tsv();
    assert!(text.starts_with("# native_tpch_benchmark_tsv_schema_version=6\n"));
    assert!(text.contains(&format!("{NATIVE_TPCH_BENCHMARK_TSV_HEADER}\n")));
    assert!(text.contains("# tpch_dataset_scale_factor=sf1\n"));
    assert!(text.contains("# tpch_benchmark_query_ids=q02\n"));
    assert!(text.contains("# tpch_benchmark_wall_time_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_native_counter_fields=measured_iteration_totals\n"));
    assert!(text.contains("# tpch_benchmark_native_peak_memory_fields=max_across_measured_iterations\n"));
    assert!(text.contains("# tpch_benchmark_native_timing_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_dataset_mode=sf1_external\n"));
    assert!(text.contains("# tpch_benchmark_native_device_budget_bytes=15032385536\n"));
    assert!(text.contains("# tpch_benchmark_native_max_source_chunk_bytes=536870912\n"));
    assert!(text.contains("# tpch_benchmark_native_max_source_reads_in_flight=2\n"));
    assert!(text.contains("# tpch_benchmark_native_source_cache_policy=disabled\n"));
    assert!(text.contains("# tpch_benchmark_native_source_cache_max_retained_bytes=none\n"));
    assert!(text.contains(
        "q02\trejected_before_execution\tbounded_rejected\tnative_bounded_join_unsupported\toperator_hashjoin"
    ));
    assert!(text.contains(&format!(
        "bounded_join_execution_status=unsupported_plan_shape;bounded_join_unsupported_shape={}",
        cudf_query_engine::BoundedJoinReservationContract.supported_shape_requirement()
    )));
    assert!(text.contains("\ttrue\tfalse\tfalse\tnot_started\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnot_run\t"));
}

#[test]
fn benchmark_report_q15_accepted_rows_preserve_modeled_contract_facts() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q15".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = benchmark_metadata();
    let mut bounded_join_reservation_facts = cudf_query_engine::BoundedJoinReservationContract
        .facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect::<BTreeMap<_, _>>();
    bounded_join_reservation_facts
        .insert("bounded_join_execution_family".to_string(), "keyed_hash_join_resident_subplan_pipeline".to_string());
    bounded_join_reservation_facts.insert(
        "bounded_join_required_shape_contract".to_string(),
        "sort_root_keyed_inner_hash_join_with_optional_projection_wrapped_keyed_inner_join_and_scalar_revenue_aggregate"
            .to_string(),
    );
    bounded_join_reservation_facts.insert(
        "bounded_join_contract_modes".to_string(),
        "source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report".to_string(),
    );
    bounded_join_reservation_facts.insert(
        "bounded_join_unresolved_preflight_contracts".to_string(),
        "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable,final_sort_contract:libcudf_final_sort_order_preflight_unavailable".to_string(),
    );
    let expected_bounded_join_facts = expected_key_value_facts(&bounded_join_reservation_facts);
    let planning = NativeDataFusionPlanningRow {
        query_id: "q15".to_string(),
        physical_root: "SortExec".to_string(),
        physical_shape: "SortExec(HashJoinExec)".to_string(),
        native_candidate_shape: Some("SortExec(HashJoinExec)".to_string()),
        native_candidate_shape_family: Some("sort_hash_join_input".to_string()),
        outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
        reason_code: None,
        unsupported_category: None,
        bounded_join_reservation_facts,
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "NativeCudfExec".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Native,
        rejected_before_execution: false,
    };
    let report = NativeTpchBenchmarkReport {
        metadata,
        config,
        rows: vec![NativeTpchBenchmarkRow::accepted(
            "q15",
            planning,
            Ok(timed_cpu_run("output-fnv1a64:0000000000000015")),
            Ok(timed_native_run("output-fnv1a64:0000000000000015")),
        )],
    };

    let text = report.to_tsv();
    assert!(text.contains("q15\taccepted_replaced\tbounded_accepted_parity_matched\tnone\tnone"));
    assert!(text.contains("bounded_join_execution_status=enabled"));

    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("benchmark report should contain the q15 row");
    assert!(lines.next().is_none(), "benchmark report should contain only the q15 row");
    let fields = row.split('\t').collect::<Vec<_>>();
    assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");

    let bounded_join_facts_index = header_index(&header_fields, "bounded_join_reservation_facts");
    let cpu_started_index = header_index(&header_fields, "cpu_runtime_started");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let parity_status_index = header_index(&header_fields, "parity_status");
    let output_fingerprint_index = header_index(&header_fields, "output_fingerprint");

    assert_eq!(fields[0], "q15");
    assert_eq!(fields[bounded_join_facts_index], expected_bounded_join_facts);
    assert!(fields[bounded_join_facts_index].contains("bounded_join_execution_status=enabled"));
    assert!(
        fields[bounded_join_facts_index]
            .contains("bounded_join_execution_family=keyed_hash_join_resident_subplan_pipeline")
    );
    assert!(fields[bounded_join_facts_index].contains("blocked_reservations="));
    assert!(fields[bounded_join_facts_index].contains("missing_reservations="));
    assert!(fields[bounded_join_facts_index].contains("bounded_join_lower_level_blockers="));
    assert!(
        fields[bounded_join_facts_index]
            .contains("bounded_join_contract_modes=source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report,final_sort_contract:hard_budgeted_report")
    );
    assert!(
        fields[bounded_join_facts_index]
            .contains("bounded_join_unresolved_preflight_contracts=source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable,final_sort_contract:libcudf_final_sort_order_preflight_unavailable")
    );
    assert_eq!(fields[cpu_started_index], "true");
    assert_eq!(fields[native_started_index], "true");
    assert_eq!(fields[runtime_outcome_index], "succeeded");
    assert_eq!(fields[parity_status_index], "matched");
    assert_eq!(fields[output_fingerprint_index], "output-fnv1a64:0000000000000015");
}
