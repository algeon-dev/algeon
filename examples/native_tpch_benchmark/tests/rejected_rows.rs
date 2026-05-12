use super::*;

#[test]
fn benchmark_report_header_and_rejected_rows_are_stable() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q02".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = benchmark_metadata();
    let mut planning = rejected_planning_row("q02");
    planning.strict_bounded_contract_rows = vec![cudf_query_engine::StrictBoundedContractGateRow {
        execution_mode: cudf_query_engine::StrictBoundedExecutionMode::UnsupportedStrictBounded,
        contract_scope: cudf_query_engine::capability::ReservationRequirementScope::Node,
        gate_id: cudf_query_engine::StrictBoundedContractGateId::Compiled(cudf_query_engine::ReservationGateId::new(0)),
        owner: cudf_query_engine::StrictBoundedContractOwner::Compiled(cudf_query_engine::ReservationGateOwner::Task(
            cudf_query_engine::PipelineTaskId::new(0),
        )),
        reservation_kind: cudf_query_engine::ReservationKind::MixedSizeData,
        memory_boundary: "mixed_keyed_residual_or_conditional_full_retained_size_data",
        required_contract: "mixed_keyed_residual_or_conditional_full_retained_size_data",
        contract_status: cudf_query_engine::capability::ContractStatus::BlockedByLowerLevel,
        contract_quality: cudf_query_engine::capability::ContractQuality::ObservedRetainedState,
        unresolved_preflight_gap: None,
    }];
    let report =
        NativeTpchBenchmarkReport { metadata, config, rows: vec![NativeTpchBenchmarkRow::rejected("q02", planning)] };

    let text = report.to_tsv();
    assert!(text.starts_with("# native_tpch_benchmark_tsv_schema_version=8\n"));
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
    assert!(text.contains("execution_mode=unsupported_strict_bounded"));
    assert!(text.contains("contract_status=blocked_by_lower_level"));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(!text.contains("bounded_join_execution_family"));
    assert!(text.contains("\ttrue\tfalse\tfalse\tnot_started\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnot_run\t"));
}
