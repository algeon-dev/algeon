use arrow::{
    array::{Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
};
use datafusion_cudf::TpchTableMetadata;

use super::*;

#[test]
fn default_config_selects_all_tpch_queries() {
    let config = NativeTpchBenchmarkConfig::default();
    assert_eq!(config.query_ids.len(), TPCH_QUERIES.len());
    assert_eq!(config.selected_queries().len(), TPCH_QUERIES.len());
    assert_eq!(config.iterations, 1);
    assert_eq!(config.dataset_mode.as_str(), "sf1_external");
    assert_eq!(config.native_memory_policy.as_str(), "bounded");
    config.validate().unwrap();
}

#[test]
fn sf1_external_dataset_mode_rejects_tiny_or_unknown_metadata() {
    dataset::validate_sf1_external_metadata(&benchmark_metadata()).unwrap();

    let tiny = TpchDatasetMetadata {
        scale_factor: "tiny".to_string(),
        provenance: "generated_tiny".to_string(),
        ..benchmark_metadata()
    };
    let error =
        dataset::validate_sf1_external_metadata(&tiny).expect_err("SF1 benchmark mode must not accept tiny metadata");
    assert!(error.to_string().contains("requires CUDF_TPCH_SCALE_FACTOR=sf1"));

    let unknown = TpchDatasetMetadata {
        scale_factor: "unknown".to_string(),
        provenance: "external_parquet_root".to_string(),
        ..benchmark_metadata()
    };
    let error = dataset::validate_sf1_external_metadata(&unknown)
        .expect_err("SF1 benchmark mode must not accept an unknown scale factor");
    assert!(error.to_string().contains("observed unknown"));

    let without_content_fingerprint = TpchDatasetMetadata { content_fingerprint: None, ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&without_content_fingerprint)
        .expect_err("SF1 benchmark mode must preserve a content fingerprint");
    assert!(error.to_string().contains("requires a content fingerprint"));

    let unknown_generation =
        TpchDatasetMetadata { generation_command: "unknown_external".to_string(), ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&unknown_generation)
        .expect_err("SF1 benchmark mode must preserve reproducible generation metadata");
    assert!(error.to_string().contains("requires a reproducible CUDF_TPCH_GENERATION_COMMAND"));

    let mut mislabeled_tiny = benchmark_metadata();
    mislabeled_tiny.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_count = 3;
    let error = dataset::validate_sf1_external_metadata(&mislabeled_tiny)
        .expect_err("SF1 benchmark mode must reject mislabeled tiny data");
    assert!(error.to_string().contains("requires SF1 row count lineitem=6001215"));
}

#[test]
fn sf1_external_dataset_mode_rejects_inconsistent_row_group_metadata_lengths() {
    let mut rows_len_mismatch = benchmark_metadata();
    let table = rows_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000];
    table.row_group_bytes = vec![1, 1];
    let error = dataset::validate_sf1_external_metadata(&rows_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group row length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_rows length for table lineitem"));

    let mut bytes_len_mismatch = benchmark_metadata();
    let table = bytes_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000, 3_001_215];
    table.row_group_bytes = vec![1];
    let error = dataset::validate_sf1_external_metadata(&bytes_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group byte length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_bytes length for table lineitem"));

    let mut zero_row_group_bytes = benchmark_metadata();
    zero_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_group_bytes = vec![0];
    let error = dataset::validate_sf1_external_metadata(&zero_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject zero-byte row groups");
    assert!(error.to_string().contains("zero-byte row groups for table lineitem"));

    let mut oversized_row_group_bytes = benchmark_metadata();
    let table = oversized_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.byte_size = 1;
    table.row_group_bytes = vec![2];
    let error = dataset::validate_sf1_external_metadata(&oversized_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject row-group byte totals larger than table bytes");
    assert!(error.to_string().contains("row-group bytes greater than table bytes for table lineitem"));
}

#[test]
fn query_id_filter_preserves_configured_order_and_rejects_duplicates() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q03".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };

    let selected = config.selected_queries().into_iter().map(|query| query.id).collect::<Vec<_>>();

    assert_eq!(selected, vec!["q03", "q01"]);

    let duplicate = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };
    let error = duplicate.validate().expect_err("duplicate query ids must not silently collapse rows");
    assert!(error.to_string().contains("duplicate TPC-H query id q01"));
}

#[test]
fn direct_config_rejects_empty_query_list() {
    let empty = NativeTpchBenchmarkConfig { query_ids: Vec::new(), ..NativeTpchBenchmarkConfig::default() };
    let error = empty.validate().expect_err("empty direct query lists must not produce empty benchmark reports");
    assert!(error.to_string().contains("must name at least one query id"));
}

#[tokio::test]
async fn benchmark_runner_validates_direct_config_before_selecting_queries() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q99".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let dir = unique_temp_dir("datafusion-cudf-native-tpch-invalid-config");
    let error = run_native_tpch_benchmark(&config, &dir).await.expect_err("invalid direct configs must not panic");

    assert!(error.to_string().contains("unknown TPC-H query id q99"));
    fs::remove_dir_all(dir).unwrap();
}

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
    assert!(text.starts_with("# native_tpch_benchmark_tsv_schema_version=4\n"));
    assert!(text.contains(&format!("{NATIVE_TPCH_BENCHMARK_TSV_HEADER}\n")));
    assert!(text.contains("# tpch_dataset_scale_factor=sf1\n"));
    assert!(text.contains("# tpch_benchmark_query_ids=q02\n"));
    assert!(text.contains("# tpch_benchmark_wall_time_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_native_counter_fields=measured_iteration_totals\n"));
    assert!(text.contains("# tpch_benchmark_native_peak_memory_fields=max_across_measured_iterations\n"));
    assert!(text.contains("# tpch_benchmark_native_timing_fields=per_measured_iteration_mean\n"));
    assert!(text.contains("# tpch_benchmark_dataset_mode=sf1_external\n"));
    assert!(text.contains("# tpch_benchmark_native_device_budget_bytes=8589934592\n"));
    assert!(text.contains("# tpch_benchmark_native_max_source_chunk_bytes=536870912\n"));
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
            row_groups_planned: 23,
            row_groups_committed: 24,
            partial_aggregate_count: 25,
            partial_aggregate_rows: 26,
            final_merge_attempts: 27,
            final_merge_levels: 28,
            retry_count: 29,
            retry_resource_exhausted_count: 30,
            retry_reservation_failure_count: 31,
            hash_join_build_count: 32,
            filtered_join_build_count: 33,
            hash_join_cache_hits: 34,
            hash_join_cache_misses: 35,
            hash_join_durable_cache_hits: 36,
            hash_join_durable_cache_misses: 37,
            hash_join_durable_cache_inserts: 38,
            hash_join_hash_table_bytes: 39,
            filtered_join_hash_table_bytes: 40,
            filtered_join_probe_state_bytes: 41,
            mixed_join_size_data_bytes: 42,
            join_gather_map_bytes: 43,
        },
        timing_metrics: NativeTpchTimingMetrics {
            parquet_read_time_ms: Some(1.0),
            source_read_time_ms: Some(2.0),
            operator_time_ms: Some(3.0),
            chunk_operator_time_ms: Some(4.0),
            final_merge_time_ms: Some(5.0),
            final_sort_time_ms: Some(6.0),
            materialization_time_ms: Some(7.0),
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
    assert!(text.contains("\t23\t24\t25\t26\t27\t28\t29\t30\t31\t32\t33\t34\t35\t36\t37\t38\t39\t40\t41\t42\t"));
    assert!(text.contains("\t1.000\t2.000\t3.000\t4.000\t5.000\t6.000\t7.000\n"));
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
fn accepted_row_preserves_native_not_started_failure_status() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Err(execution::NativeRunFailure {
            runtime_started: false,
            message: "native planning failed\nwith details".to_string(),
            details: execution::FailureDetails::default(),
        }),
    );

    assert!(row.cpu_runtime_started);
    assert!(!row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Failed);
    assert_eq!(row.runtime_failure.as_deref(), Some("native planning failed with details"));
    assert_eq!(row.parity_status, validation::ParityStatus::NativeFailed);
    assert_eq!(row.cpu_output_rows, Some(4));
    assert_eq!(row.native_output_rows, None);
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(NativeTpchMemoryPolicy::default(), &row), "bounded_accepted_native_failed");
}

#[test]
fn accepted_row_distinguishes_cpu_planning_failure_from_started_runtime() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Err(execution::CpuRunFailure {
            runtime_started: false,
            message: "cpu physical planning failed\tbefore execution".to_string(),
            details: execution::FailureDetails::default(),
        }),
        Ok(timed_native_run("output-fnv1a64:0000000000000001")),
    );

    assert!(!row.cpu_runtime_started);
    assert!(row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.cpu_failure.as_deref(), Some("cpu physical planning failed before execution"));
    assert_eq!(row.runtime_failure, None);
    assert_eq!(row.parity_status, validation::ParityStatus::CpuFailed);
    assert_eq!(row.cpu_output_rows, None);
    assert_eq!(row.native_output_rows, Some(4));
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(NativeTpchMemoryPolicy::default(), &row), "bounded_accepted_cpu_failed");
}

#[test]
fn accepted_row_reports_fingerprint_mismatch_without_canonical_output_fingerprint() {
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Ok(timed_native_run("output-fnv1a64:0000000000000002")),
    );

    assert!(row.cpu_runtime_started);
    assert!(row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, validation::ParityStatus::Mismatched);
    assert_eq!(row.cpu_output_fingerprint.as_deref(), Some("output-fnv1a64:0000000000000001"));
    assert_eq!(row.native_output_fingerprint.as_deref(), Some("output-fnv1a64:0000000000000002"));
    assert_eq!(row.output_fingerprint, None);
    assert_eq!(bounded_status(NativeTpchMemoryPolicy::default(), &row), "bounded_accepted_parity_mismatched");
}

#[test]
fn native_runtime_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_cudf::ErrorCode::MemoryReservationFailed,
        "datafusion_cudf.native_exec.execute",
        "native cudf query execution failed",
    )
    .with_fact("native_code", "memory_reservation_failed")
    .with_fact("native_kind", "resource_exhausted")
    .with_fact("native_status", "temporary")
    .with_fact("native_operation", "cudf_query_engine.memory.reserve")
    .with_fact("memory_boundary", "final_sort")
    .with_fact("requested_bytes", 2048)
    .with_fact("granted_bytes", 1024);
    let datafusion_error = DataFusionError::from(adapter_error);
    let details = execution::FailureDetails::from_datafusion_error(&datafusion_error);

    assert_eq!(details.code.as_deref(), Some("memory_reservation_failed"));
    assert_eq!(details.kind.as_deref(), Some("resource_exhausted"));
    assert_eq!(details.status.as_deref(), Some("temporary"));
    assert_eq!(details.operation.as_deref(), Some("datafusion_cudf.native_exec.execute"));
    assert_eq!(details.facts.get("memory_boundary").map(String::as_str), Some("final_sort"));
    assert_eq!(details.facts.get("requested_bytes").map(String::as_str), Some("2048"));

    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
        Err(execution::NativeRunFailure { runtime_started: true, message: datafusion_error.to_string(), details }),
    );
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config: NativeTpchBenchmarkConfig {
            query_ids: vec!["q01".to_string()],
            ..NativeTpchBenchmarkConfig::default()
        },
        rows: vec![row],
    };
    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let row = lines.next().expect("benchmark report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row_fields = row.split('\t').collect::<Vec<_>>();
    let facts = row_fields[header_index(&header_fields, "runtime_failure_facts")];

    assert_eq!(row_fields[header_index(&header_fields, "runtime_outcome")], "failed");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_code")], "memory_reservation_failed");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_kind")], "resource_exhausted");
    assert_eq!(row_fields[header_index(&header_fields, "runtime_failure_status")], "temporary");
    assert_eq!(
        row_fields[header_index(&header_fields, "runtime_failure_operation")],
        "datafusion_cudf.native_exec.execute"
    );
    assert!(facts.contains("memory_boundary=final_sort"));
    assert!(facts.contains("native_operation=cudf_query_engine.memory.reserve"));
    assert!(facts.contains("requested_bytes=2048"));
}

#[test]
fn planning_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_cudf::ErrorCode::InvalidExecutionPlan,
        "datafusion_cudf.native_planner",
        "native physical planning failed",
    )
    .with_fact("query_id", "q01")
    .with_fact("planner_stage", "native_lowering");
    let row = NativeTpchBenchmarkRow::planning_failure("q01", DataFusionError::from(adapter_error));

    assert_eq!(row.reason_code.as_deref(), Some("physical_planning_failed"));
    assert_eq!(row.unsupported_category.as_deref(), Some("planning"));
    assert!(!row.cpu_runtime_started);
    assert!(!row.native_runtime_started);
    assert_eq!(row.runtime_outcome, NativeDataFusionRuntimeOutcome::NotStarted);
    assert_eq!(row.runtime_failure_details.code.as_deref(), Some("invalid_execution_plan"));
    assert_eq!(row.runtime_failure_details.kind.as_deref(), Some("planning"));
    assert_eq!(row.runtime_failure_details.status.as_deref(), Some("permanent"));
    assert_eq!(row.runtime_failure_details.operation.as_deref(), Some("datafusion_cudf.native_planner"));
    assert_eq!(row.runtime_failure_details.facts.get("planner_stage").map(String::as_str), Some("native_lowering"));
    assert_eq!(bounded_status(NativeTpchMemoryPolicy::default(), &row), "bounded_planning_failed");
}

#[test]
fn cpu_runtime_failure_preserves_adapter_error_facts() {
    let adapter_error = AdapterError::new(
        datafusion_cudf::ErrorCode::InvalidExecutionPlan,
        "datafusion_cudf.cpu_collect",
        "cpu collection failed",
    )
    .with_fact("query_id", "q01")
    .with_fact("phase", "collect");
    let datafusion_error = DataFusionError::from(adapter_error);
    let details = execution::FailureDetails::from_datafusion_error(&datafusion_error);
    let row = NativeTpchBenchmarkRow::accepted(
        "q01",
        accepted_planning_row("q01"),
        Err(execution::CpuRunFailure { runtime_started: true, message: datafusion_error.to_string(), details }),
        Ok(timed_native_run("output-fnv1a64:0000000000000001")),
    );
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config: NativeTpchBenchmarkConfig {
            query_ids: vec!["q01".to_string()],
            ..NativeTpchBenchmarkConfig::default()
        },
        rows: vec![row],
    };
    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let row = lines.next().expect("benchmark report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row_fields = row.split('\t').collect::<Vec<_>>();
    let facts = row_fields[header_index(&header_fields, "cpu_failure_facts")];

    assert_eq!(row_fields[header_index(&header_fields, "parity_status")], "cpu_failed");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_code")], "invalid_execution_plan");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_kind")], "planning");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_status")], "permanent");
    assert_eq!(row_fields[header_index(&header_fields, "cpu_failure_operation")], "datafusion_cudf.cpu_collect");
    assert!(facts.contains("phase=collect"));
    assert!(facts.contains("query_id=q01"));
}

#[test]
fn allocator_high_water_placeholder_is_none_without_runtime_contract() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q01".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config,
        rows: vec![NativeTpchBenchmarkRow::accepted(
            "q01",
            accepted_planning_row("q01"),
            Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
            Ok(timed_native_run("output-fnv1a64:0000000000000001")),
        )],
    };
    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let row = lines.next().expect("benchmark report should contain a data row");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    let row_fields = row.split('\t').collect::<Vec<_>>();

    assert_eq!(
        row_fields[header_index(&header_fields, "allocator_high_water_bytes")],
        "none",
        "allocator high-water must stay unset until the lower runtime exposes a real contract"
    );
}

#[test]
fn output_fingerprint_is_independent_of_record_batch_boundaries() {
    let single_batch = vec![int_batch(&[1, 2, 3, 4])];
    let split_batches = vec![int_batch(&[1, 2]), int_batch(&[3, 4])];
    let different_rows = vec![int_batch(&[1, 2, 4, 3])];
    let schema = single_batch[0].schema();

    let single_fingerprint = validation::fingerprint_batches(&schema, &single_batch).unwrap();
    let split_fingerprint = validation::fingerprint_batches(&schema, &split_batches).unwrap();
    let different_fingerprint = validation::fingerprint_batches(&schema, &different_rows).unwrap();

    assert_eq!(single_fingerprint, split_fingerprint);
    assert_ne!(single_fingerprint, different_fingerprint);
    assert_eq!(
        validation::parity_status(
            Some(&execution::QueryMeasurement {
                output_batches: single_batch.len(),
                output_rows: 4,
                output_bytes: 0,
                output_fingerprint: single_fingerprint,
            }),
            Some(&execution::QueryMeasurement {
                output_batches: split_batches.len(),
                output_rows: 4,
                output_bytes: 0,
                output_fingerprint: split_fingerprint,
            }),
        ),
        validation::ParityStatus::Matched
    );
}

#[test]
fn output_fingerprint_includes_schema_nullability_and_metadata() {
    let nullable = vec![int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[1, 2])];
    let non_nullable = vec![int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[1, 2])];
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("logical_source".to_string(), "cpu".to_string());
    let schema_metadata =
        vec![int_batch_with_field(Field::new("value", DataType::Int32, false), Some(metadata), &[1, 2])];

    let nullable_fingerprint = validation::fingerprint_batches(&nullable[0].schema(), &nullable).unwrap();
    let non_nullable_fingerprint = validation::fingerprint_batches(&non_nullable[0].schema(), &non_nullable).unwrap();
    let schema_metadata_fingerprint =
        validation::fingerprint_batches(&schema_metadata[0].schema(), &schema_metadata).unwrap();

    assert_ne!(nullable_fingerprint, non_nullable_fingerprint);
    assert_ne!(non_nullable_fingerprint, schema_metadata_fingerprint);
}

#[test]
fn output_fingerprint_includes_schema_for_empty_outputs() {
    let nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, true), None, &[]).schema();
    let non_nullable_schema = int_batch_with_field(Field::new("value", DataType::Int32, false), None, &[]).schema();

    assert_ne!(
        validation::fingerprint_batches(&nullable_schema, &[]).unwrap(),
        validation::fingerprint_batches(&non_nullable_schema, &[]).unwrap()
    );
}

#[test]
fn output_fingerprint_distinguishes_null_from_empty_string() {
    let null_string = vec![string_batch(&[None])];
    let empty_string = vec![string_batch(&[Some("")])];

    assert_ne!(
        validation::fingerprint_batches(&null_string[0].schema(), &null_string).unwrap(),
        validation::fingerprint_batches(&empty_string[0].schema(), &empty_string).unwrap()
    );
}

#[test]
fn benchmark_planning_row_selection_rejects_ambiguous_reports() {
    let selected = execution::select_benchmark_planning_row("q01", vec![accepted_planning_row("q01")]).unwrap();
    assert_eq!(selected.query_id, "q01");

    let error = execution::select_benchmark_planning_row(
        "q01",
        vec![accepted_planning_row("q01"), rejected_planning_row("q01")],
    )
    .expect_err("benchmark planning must not silently choose one row from an ambiguous report");
    assert!(error.to_string().contains("requires exactly one native planning row per query"));

    let error = execution::select_benchmark_planning_row("q01", Vec::new())
        .expect_err("benchmark planning must reject empty reports explicitly");
    assert!(error.to_string().contains("native planning report was empty for q01"));

    let error = execution::select_benchmark_planning_row("q01", vec![accepted_planning_row("q02")])
        .expect_err("benchmark planning must reject a single row for a different query");
    assert!(error.to_string().contains("planning row query_id mismatch"));
}

#[test]
fn accepted_planning_row_must_use_native_execution_path() {
    execution::validate_decision_grade_native_planning_row("q01", &accepted_planning_row("q01")).unwrap();

    let mut gpu_island = accepted_planning_row("q01");
    gpu_island.post_native_execution_path = NativeDataFusionPostNativeExecutionPath::GpuIsland;
    let error = execution::validate_decision_grade_native_planning_row("q01", &gpu_island)
        .expect_err("TPC-H native readiness benchmark must not accept GPU-island execution");
    assert!(error.to_string().contains("requires post_native_execution_path=native"));
}

#[test]
fn accepted_planning_row_must_not_carry_rejection_fields() {
    let mut rejected_flag = accepted_planning_row("q01");
    rejected_flag.rejected_before_execution = true;
    let error = execution::validate_decision_grade_native_planning_row("q01", &rejected_flag)
        .expect_err("accepted benchmark planning rows must not carry rejected_before_execution");
    assert!(error.to_string().contains("must not set rejected_before_execution=true"));

    let mut reason_code = accepted_planning_row("q01");
    reason_code.reason_code = Some("unsupported_operator".to_string());
    let error = execution::validate_decision_grade_native_planning_row("q01", &reason_code)
        .expect_err("accepted benchmark planning rows must not carry a rejection reason");
    assert!(error.to_string().contains("must not include a rejection reason_code"));

    let mut unsupported_category = accepted_planning_row("q01");
    unsupported_category.unsupported_category = Some("operator_hashjoin".to_string());
    let error = execution::validate_decision_grade_native_planning_row("q01", &unsupported_category)
        .expect_err("accepted benchmark planning rows must not carry an unsupported category");
    assert!(error.to_string().contains("must not include an unsupported_category"));
}

#[test]
fn rejected_planning_row_must_carry_stable_rejection_fields() {
    execution::validate_decision_grade_native_rejection_row("q02", &rejected_planning_row("q02")).unwrap();

    let mut missing_rejected_flag = rejected_planning_row("q02");
    missing_rejected_flag.rejected_before_execution = false;
    let error = execution::validate_decision_grade_native_rejection_row("q02", &missing_rejected_flag)
        .expect_err("benchmark rejected rows must identify pre-execution rejection");
    assert!(error.to_string().contains("must set rejected_before_execution=true"));

    let mut missing_reason = rejected_planning_row("q02");
    missing_reason.reason_code = None;
    let error = execution::validate_decision_grade_native_rejection_row("q02", &missing_reason)
        .expect_err("benchmark rejected rows must carry stable reason codes");
    assert!(error.to_string().contains("must include a stable reason_code"));

    let mut native_path = rejected_planning_row("q02");
    native_path.post_native_execution_path = NativeDataFusionPostNativeExecutionPath::Native;
    let error = execution::validate_decision_grade_native_rejection_row("q02", &native_path)
        .expect_err("benchmark rejected rows must not be classified as native execution");
    assert!(error.to_string().contains("must not report post_native_execution_path=native"));
}

#[test]
fn benchmark_report_schema_v4_rows_match_header_columns() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string(), "q02".to_string(), "q03".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config,
        rows: vec![
            NativeTpchBenchmarkRow::accepted(
                "q01",
                accepted_planning_row("q01"),
                Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
                Ok(timed_native_run("output-fnv1a64:0000000000000001")),
            ),
            NativeTpchBenchmarkRow::rejected("q02", rejected_planning_row("q02")),
            NativeTpchBenchmarkRow::planning_failure(
                "q03",
                DataFusionError::Plan("planning failed before execution".to_string()),
            ),
        ],
    };

    let text = report.to_tsv();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().expect("benchmark report should contain a TSV header");
    let header_fields = header.split('\t').collect::<Vec<_>>();
    assert_eq!(header, NATIVE_TPCH_BENCHMARK_TSV_HEADER);

    let bounded_status_index = header_index(&header_fields, "bounded_status");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let output_fingerprint_index = header_index(&header_fields, "output_fingerprint");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let rows = lines.collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let fields = row.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");
        match fields[0] {
            "q01" => {
                assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
                assert_eq!(fields[native_started_index], "true");
                assert_eq!(fields[runtime_outcome_index], "succeeded");
                assert_eq!(fields[output_fingerprint_index], "output-fnv1a64:0000000000000001");
            },
            "q02" => {
                assert_eq!(fields[bounded_status_index], "bounded_rejected");
                assert_eq!(fields[native_started_index], "false");
                assert_eq!(fields[runtime_outcome_index], "not_started");
                assert_eq!(fields[output_fingerprint_index], "not_run");
            },
            "q03" => {
                assert_eq!(fields[bounded_status_index], "bounded_planning_failed");
                assert_eq!(fields[native_started_index], "false");
                assert_eq!(fields[runtime_outcome_index], "not_started");
                assert_eq!(fields[output_fingerprint_index], "not_run");
            },
            query_id => panic!("unexpected benchmark row {query_id}"),
        }
    }
}

fn benchmark_metadata() -> TpchDatasetMetadata {
    TpchDatasetMetadata {
        root: PathBuf::from("/tmp/tpch"),
        scale_factor: "sf1".to_string(),
        provenance: "external_parquet_root".to_string(),
        generation_command: "dbgen".to_string(),
        table_count: 8,
        file_count: 8,
        total_rows: 8_661_245,
        total_bytes: 8,
        metadata_fingerprint: "metadata-fnv1a64:0000000000000001".to_string(),
        content_fingerprint: Some("content-fnv1a64:0000000000000001".to_string()),
        tables: benchmark_sf1_tables(),
    }
}

fn benchmark_sf1_tables() -> Vec<TpchTableMetadata> {
    [
        (
            "customer",
            150_000,
            &["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"][..],
        ),
        (
            "lineitem",
            6_001_215,
            &[
                "l_orderkey",
                "l_partkey",
                "l_suppkey",
                "l_linenumber",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_returnflag",
                "l_linestatus",
                "l_shipdate",
                "l_commitdate",
                "l_receiptdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ][..],
        ),
        ("nation", 25, &["n_nationkey", "n_name", "n_regionkey", "n_comment"][..]),
        (
            "orders",
            1_500_000,
            &[
                "o_orderkey",
                "o_custkey",
                "o_orderstatus",
                "o_totalprice",
                "o_orderdate",
                "o_orderpriority",
                "o_clerk",
                "o_shippriority",
                "o_comment",
            ][..],
        ),
        (
            "part",
            200_000,
            &[
                "p_partkey",
                "p_name",
                "p_mfgr",
                "p_brand",
                "p_type",
                "p_size",
                "p_container",
                "p_retailprice",
                "p_comment",
            ][..],
        ),
        ("partsupp", 800_000, &["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"][..]),
        ("region", 5, &["r_regionkey", "r_name", "r_comment"][..]),
        (
            "supplier",
            10_000,
            &["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"][..],
        ),
    ]
    .into_iter()
    .map(|(table, row_count, columns)| benchmark_table_metadata(table, row_count, columns))
    .collect()
}

fn benchmark_table_metadata(table: &str, row_count: u64, columns: &[&str]) -> TpchTableMetadata {
    TpchTableMetadata {
        table: table.to_string(),
        path: PathBuf::from(format!("/tmp/tpch/{table}.parquet")),
        schema: columns
            .iter()
            .map(|column| format!("{column}:Int64:nullable=false:metadata={{}}"))
            .collect::<Vec<_>>()
            .join("|"),
        files: Vec::new(),
        file_count: 1,
        row_count,
        byte_size: 1,
        row_group_count: 1,
        row_group_rows: vec![row_count],
        row_group_bytes: vec![1],
    }
}

fn accepted_planning_row(query_id: &str) -> NativeDataFusionPlanningRow {
    NativeDataFusionPlanningRow {
        query_id: query_id.to_string(),
        physical_root: "NativeCudfExec".to_string(),
        physical_shape: "NativeCudfExec".to_string(),
        native_candidate_shape: Some("NativeCudfExec".to_string()),
        native_candidate_shape_family: Some("sort_local_parquet_aggregate".to_string()),
        outcome: NativeDataFusionPlanningOutcome::AcceptedReplaced,
        reason_code: None,
        unsupported_category: None,
        bounded_join_reservation_facts: BTreeMap::new(),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "NativeCudfExec".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Native,
        rejected_before_execution: false,
    }
}

fn rejected_planning_row(query_id: &str) -> NativeDataFusionPlanningRow {
    NativeDataFusionPlanningRow {
        query_id: query_id.to_string(),
        physical_root: "SortExec".to_string(),
        physical_shape: "SortExec(HashJoinExec)".to_string(),
        native_candidate_shape: Some("SortExec(HashJoinExec)".to_string()),
        native_candidate_shape_family: Some("sort_hash_join_aggregate".to_string()),
        outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some("native_bounded_join_unsupported".to_string()),
        unsupported_category: Some("operator_hashjoin".to_string()),
        bounded_join_reservation_facts: BTreeMap::new(),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "SortExec".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        rejected_before_execution: true,
    }
}

fn timed_cpu_run(fingerprint: &str) -> execution::TimedQueryRun {
    execution::TimedQueryRun {
        samples: vec![Duration::from_millis(10)],
        final_measurement: query_measurement(fingerprint),
    }
}

fn timed_native_run(fingerprint: &str) -> execution::TimedNativeQueryRun {
    execution::TimedNativeQueryRun {
        samples: vec![Duration::from_millis(5)],
        final_measurement: query_measurement(fingerprint),
        metrics: NativeDataFusionExecutionMetrics::default(),
        timing_metrics: NativeTpchTimingMetrics::default(),
    }
}

fn query_measurement(fingerprint: &str) -> execution::QueryMeasurement {
    execution::QueryMeasurement {
        output_batches: 1,
        output_rows: 4,
        output_bytes: 128,
        output_fingerprint: fingerprint.to_string(),
    }
}

fn int_batch(values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

fn int_batch_with_field(
    field: Field,
    metadata: Option<std::collections::HashMap<String, String>>,
    values: &[i32],
) -> RecordBatch {
    let schema = match metadata {
        Some(metadata) => Schema::new_with_metadata(vec![field], metadata),
        None => Schema::new(vec![field]),
    };
    let schema = Arc::new(schema);
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
}

fn string_batch(values: &[Option<&str>]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, true)]));
    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(values.to_vec()))]).unwrap()
}

fn header_index(header: &[&str], name: &str) -> usize {
    header.iter().position(|field| *field == name).unwrap_or_else(|| panic!("missing TSV field {name}"))
}

fn expected_key_value_facts(facts: &BTreeMap<String, String>) -> String {
    if facts.is_empty() {
        return "none".to_string();
    }

    facts
        .iter()
        .map(|(key, value)| format!("{}={}", expected_stable_tsv_field(key), expected_stable_tsv_field(value)))
        .collect::<Vec<_>>()
        .join(";")
}

fn expected_stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
