use super::*;

#[test]
fn benchmark_report_header_and_not_selected_rows_are_stable() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q02".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = benchmark_metadata();
    let mut planning = not_supported_planning_row("q02");
    planning.capability_decision_rows = vec![nexus_query_engine::contract_report::CapabilityDecisionRow {
        execution_mode: nexus_query_engine::mode::CapabilityEnvelopeStatus::Rejected,
        scope: nexus_query_engine::capability::CapabilityScope::Subgraph,
        decision_id: nexus_query_engine::contract_report::CapabilityDecisionId {
            scope: nexus_query_engine::capability::CapabilityScope::Subgraph,
            index: 0,
        },
        owner: nexus_query_engine::contract_report::CapabilityDecisionOwner::Admission("bounded_pipeline"),
        category: Some(
            nexus_query_engine::capability::CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
        ),
        boundary: "bounded_pipeline",
        reason: "unsupported_compiled_pipeline_envelope",
        status: nexus_query_engine::contract_report::CapabilityDecisionStatus::Rejected,
        facts: std::collections::BTreeMap::new(),
    }];
    let report = NativeTpchBenchmarkReport {
        metadata,
        config,
        rows: vec![NativeTpchBenchmarkRow::not_selected("q02", planning)],
    };

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
    assert!(!text.contains("tpch_benchmark_native_source_cache_"));
    assert!(text.contains("q02\tnot_supported\tbounded_rejected\tnative_bounded_join_unsupported\toperator_hashjoin"));
    assert!(text.contains("execution_mode=unsupported_bounded"));
    assert!(text.contains("status=rejected"));
    assert!(!text.contains("bounded_join_reservation_facts"));
    assert!(!text.contains("bounded_join_execution_family"));
    assert!(text.contains("\tfalse\tfalse\tfalse\tnot_started\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnone\tnot_run\t"));
}

#[test]
fn benchmark_report_metadata_identifies_named_sf100_profile() {
    let profile =
        tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]).unwrap();
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
