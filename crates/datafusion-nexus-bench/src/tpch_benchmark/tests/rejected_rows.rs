use super::*;

#[test]
fn benchmark_report_header_and_rejected_rows_are_stable() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q02".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = benchmark_metadata();
    let mut planning = rejected_planning_row("q02");
    planning.strict_bounded_contract_rows = vec![nexus_query_engine::contract_report::StrictBoundedContractGateRow {
        execution_mode: nexus_query_engine::mode::StrictBoundedExecutionMode::UnsupportedStrictBounded,
        contract_scope: nexus_query_engine::capability::ReservationRequirementScope::Node,
        gate_id: nexus_query_engine::contract_report::StrictBoundedContractGateId::Compiled(
            nexus_query_engine::pipeline::ReservationGateId::new(0),
        ),
        owner: nexus_query_engine::contract_report::StrictBoundedContractOwner::Compiled(
            nexus_query_engine::pipeline::ReservationGateOwner::Task(
                nexus_query_engine::pipeline::PipelineTaskId::new(0),
            ),
        ),
        reservation_kind: nexus_query_engine::runtime::ReservationKind::MixedSizeData,
        memory_boundary: "mixed_keyed_residual_or_conditional_full_retained_size_data",
        required_contract: "mixed_keyed_residual_or_conditional_full_retained_size_data",
        contract_status: nexus_query_engine::capability::ContractStatus::BlockedByLowerLevel,
        contract_quality: nexus_query_engine::capability::ContractQuality::ObservedRetainedState,
        unresolved_preflight_gap: None,
    }];
    let report =
        NativeTpchBenchmarkReport { metadata, config, rows: vec![NativeTpchBenchmarkRow::rejected("q02", planning)] };

    let text = report.to_tsv();
    assert!(text.starts_with("# tpch_dataset_root=/tmp/tpch\n"));
    assert!(text.contains(&format!("{NATIVE_TPCH_BENCHMARK_TSV_HEADER}\n")));
    assert!(text.contains("# tpch_dataset_scale_factor=sf1\n"));
    assert!(text.contains("# tpch_benchmark_query_ids=q02\n"));
    assert!(text.contains("# tpch_benchmark_wall_time_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_native_counter_fields=measured_iteration_totals\n"));
    assert!(text.contains("# tpch_benchmark_native_peak_memory_fields=max_across_measured_iterations\n"));
    assert!(text.contains("# tpch_benchmark_native_timing_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_dataset_mode=sf1_external\n"));
    assert!(text.contains("# tpch_benchmark_native_profile_name=conservative\n"));
    assert!(text.contains("# tpch_benchmark_native_profile_source=default\n"));
    assert!(text.contains("# tpch_benchmark_native_device_budget_bytes=15032385536\n"));
    assert!(text.contains("# tpch_benchmark_native_max_source_chunk_bytes=536870912\n"));
    assert!(text.contains("# tpch_benchmark_native_source_chunk_planning_mode=conservative\n"));
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

#[test]
fn benchmark_report_metadata_identifies_named_sf100_profile() {
    let profile =
        tpch_native_profile_settings_from([(CUDF_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]).unwrap();
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string()],
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        native_memory_policy: NativeTpchMemoryPolicy::bounded_from_profile_settings(profile),
        ..NativeTpchBenchmarkConfig::default()
    };
    let report = NativeTpchBenchmarkReport { metadata: benchmark_sf100_metadata(), config, rows: Vec::new() };

    let text = report.to_tsv();

    assert!(text.contains("# tpch_benchmark_native_profile_name=sf100_gpu_throughput\n"));
    assert!(text.contains("# tpch_benchmark_native_profile_source=env_profile\n"));
    assert!(text.contains("# tpch_benchmark_native_device_budget_bytes=102005473280\n"));
    assert!(text.contains("# tpch_benchmark_native_max_source_chunk_bytes=34359738368\n"));
    assert!(text.contains("# tpch_benchmark_native_max_row_groups_per_chunk=1024\n"));
    assert!(text.contains("# tpch_benchmark_native_source_chunk_planning_mode=local_parquet_throughput\n"));
}
