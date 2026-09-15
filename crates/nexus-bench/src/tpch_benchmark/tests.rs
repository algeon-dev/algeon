use std::{path::PathBuf, time::Duration};

use nexus_workloads::tpch::TpchTableMetadata;

use super::*;

mod config_dataset;
mod output_fingerprint;
mod provenance;
mod rejected_rows;
mod report_rows;
mod runtime_failures;
#[path = "tests/strategy_gate.rs"]
mod strategy_gate_tests;

#[test]
fn benchmark_metric_aggregation_delegates_to_the_report_metric_owner() {
    let source = include_str!("execution.rs");
    let aggregate = source
        .split_once("fn aggregate_native_execution_metrics")
        .expect("TPC-H benchmark must aggregate native execution metrics")
        .1
        .split_once("fn native_started")
        .expect("metric aggregation must end before native failure conversion")
        .0;

    assert!(aggregate.contains(".reduce("));
    assert!(aggregate.contains("aggregate.merge_from(current);"));
    assert!(!aggregate.contains("saturating_add"));
    assert!(!aggregate.contains("max_optional_metric"));
}

#[test]
fn allocator_high_water_placeholder_is_none_without_runtime_contract() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q01".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let report = NativeTpchBenchmarkReport {
        metadata: benchmark_metadata(),
        config,
        devices: Vec::new(),
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
fn benchmark_planning_row_selection_handles_multi_candidate_reports() {
    let selected = execution::select_benchmark_planning_row("q01", vec![accepted_planning_row("q01")]).unwrap();
    assert_eq!(selected.query_id, "q01");

    let selected = execution::select_benchmark_planning_row(
        "q01",
        vec![accepted_planning_row("q01"), not_supported_planning_row("q01")],
    )
    .expect("benchmark planning should select a unique whole-native accepted row from retained candidate reports");
    assert_eq!(selected.query_id, "q01");

    let error = execution::select_benchmark_planning_row(
        "q01",
        vec![accepted_planning_row("q01"), accepted_planning_row("q01"), not_supported_planning_row("q01")],
    )
    .expect_err("benchmark planning must reject genuinely ambiguous accepted reports");
    assert!(error.to_string().contains("requires exactly one whole-native accepted planning row"));

    let error = execution::select_benchmark_planning_row("q01", Vec::new())
        .expect_err("benchmark planning must reject empty reports explicitly");
    assert!(error.to_string().contains("native planning report was empty for q01"));

    let error = execution::select_benchmark_planning_row("q01", vec![accepted_planning_row("q02")])
        .expect_err("benchmark planning must reject a single row for a different query");
    assert!(error.to_string().contains("planning row query_id mismatch"));
}

#[test]
fn benchmark_query_progress_line_identifies_query_and_profile() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q07".to_string()],
        native_memory_policy: NativeTpchMemoryPolicy::bounded_custom(
            NativeTpchDeviceBudget::Explicit { bytes: 94_489_280_512 },
            34_359_738_368,
            1024,
            DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
            nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput,
        ),
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        ..NativeTpchBenchmarkConfig::default()
    };
    let query = TPCH_QUERIES.iter().find(|query| query.id == "q07").unwrap();

    let line = benchmark_query_progress_line(&config, query, 2, 3);

    assert!(line.starts_with("[native_tpch_benchmark] stage=query_start query=q07 index=3/3"));
    assert!(line.contains("dataset_mode=sf100_external"));
    assert!(line.contains("execution_mode=bounded"));
    assert!(line.contains("relational_execution=native_no_datafusion_cpu"));
    assert!(line.contains("memory_policy=bounded"));
    assert!(line.contains("native_profile=custom"));
    assert!(line.contains("source_chunk_planning_mode=local_parquet_throughput"));
    assert!(line.contains("max_row_groups_per_chunk=1024"));
    assert!(line.contains("max_source_chunk_bytes=34359738368"));
    assert!(line.contains("skip_cpu="));
}

#[test]
fn benchmark_query_progress_line_identifies_named_sf100_profile() {
    let profile =
        tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]).unwrap();
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q07".to_string()],
        native_memory_policy: NativeTpchMemoryPolicy::bounded_from_profile_settings(profile),
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        ..NativeTpchBenchmarkConfig::default()
    };
    let query = TPCH_QUERIES.iter().find(|query| query.id == "q07").unwrap();

    let line = benchmark_query_progress_line(&config, query, 0, 1);

    assert!(line.contains("native_profile=sf100_gpu_throughput"));
    assert!(line.contains("source_chunk_planning_mode=local_parquet_throughput"));
    assert!(line.contains("max_row_groups_per_chunk=1024"));
    assert!(line.contains("max_source_chunk_bytes=34359738368"));
}

#[test]
fn accepted_planning_row_must_use_native_execution_path() {
    execution::validate_decision_grade_native_planning_row("q01", &accepted_planning_row("q01")).unwrap();

    let mut mixed_plan = accepted_planning_row("q01");
    mixed_plan.final_disposition = Some(FinalPlanDisposition::Mixed);
    let error = execution::validate_decision_grade_native_planning_row("q01", &mixed_plan)
        .expect_err("TPC-H native readiness benchmark must not accept mixed-plan execution");
    assert!(error.to_string().contains("requires final_disposition=native"));
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
fn not_selected_planning_row_preserves_the_executable_baseline() {
    execution::validate_decision_grade_native_not_selected_row("q02", &not_supported_planning_row("q02")).unwrap();

    let mut terminal_flag = not_supported_planning_row("q02");
    terminal_flag.rejected_before_execution = true;
    let error = execution::validate_decision_grade_native_not_selected_row("q02", &terminal_flag)
        .expect_err("not-selected benchmark rows must not identify a terminal rejection");
    assert!(error.to_string().contains("must not set rejected_before_execution=true"));

    let mut missing_reason = not_supported_planning_row("q02");
    missing_reason.reason_code = None;
    let error = execution::validate_decision_grade_native_not_selected_row("q02", &missing_reason)
        .expect_err("not-selected benchmark rows must carry stable reason codes");
    assert!(error.to_string().contains("must include a stable reason_code"));

    let mut native_path = not_supported_planning_row("q02");
    native_path.final_disposition = Some(FinalPlanDisposition::Native);
    let error = execution::validate_decision_grade_native_not_selected_row("q02", &native_path)
        .expect_err("not-selected benchmark rows must not be classified as native execution");
    assert!(error.to_string().contains("must preserve an executable DataFusion or mixed final disposition"));
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
        devices: Vec::new(),
        rows: vec![
            NativeTpchBenchmarkRow::accepted(
                "q01",
                accepted_planning_row("q01"),
                Ok(timed_cpu_run("output-fnv1a64:0000000000000001")),
                Ok(timed_native_run("output-fnv1a64:0000000000000001")),
            ),
            NativeTpchBenchmarkRow::not_selected("q02", not_supported_planning_row("q02")),
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
    assert_eq!(NATIVE_TPCH_BENCHMARK_TSV_COLUMNS.join("\t"), NATIVE_TPCH_BENCHMARK_TSV_HEADER);
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tgraph_runtime_rows\ttask_operator_rows\t"));
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains(
        "\tdynamic_filter_post_decode_rows_before\tdynamic_filter_post_decode_rows_after\tdynamic_filter_post_decode_rows_filtered\tdynamic_filter_source_post_decode_rows_filtered\tdynamic_filter_filter_post_decode_rows_filtered\tdynamic_filter_join_probe_post_decode_rows_filtered\tdynamic_filter_post_decode_reduction_stage\t"
    ));
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tdynamic_filter_post_decode_reduction_stage\t"));
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tdynamic_filter_measured_reduction_status\t"));
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tdynamic_filter_work_avoidance_status\t"));
    assert!(
        NATIVE_TPCH_BENCHMARK_TSV_HEADER
            .contains("\taggregate_libcudf_runtime_probe_count\taggregate_libcudf_runtime_probe_elapsed_ns\t")
    );
    assert!(
        NATIVE_TPCH_BENCHMARK_TSV_HEADER
            .contains("\taggregate_exact_cardinality_probe_count\taggregate_exact_cardinality_probe_elapsed_ns\t")
    );

    let planning_outcome_index = header_index(&header_fields, "planning_outcome");
    let bounded_status_index = header_index(&header_fields, "bounded_status");
    assert!(NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tpost_native_execution_path\t"));
    assert!(!NATIVE_TPCH_BENCHMARK_TSV_HEADER.contains("\tfinal_disposition\t"));
    for removed in [
        "source_cache_hits",
        "source_cache_misses",
        "source_cache_inserts",
        "source_cache_insert_skipped",
        "source_cache_retained_bytes",
        "source_cache_lookup_time_ms",
        "hash_join_durable_cache_hits",
        "hash_join_durable_cache_misses",
        "hash_join_durable_cache_inserts",
    ] {
        assert!(
            !NATIVE_TPCH_BENCHMARK_TSV_HEADER.split('\t').any(|column| column == removed),
            "report must not project removed field {removed}"
        );
    }
    // This frozen column name carries the current final-plan disposition value.
    let final_disposition_index = header_index(&header_fields, "post_native_execution_path");
    let rejected_before_execution_index = header_index(&header_fields, "rejected_before_execution");
    let native_started_index = header_index(&header_fields, "native_runtime_started");
    let output_fingerprint_index = header_index(&header_fields, "output_fingerprint");
    let runtime_outcome_index = header_index(&header_fields, "runtime_outcome");
    let rows = lines.collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let fields = row.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), NATIVE_TPCH_BENCHMARK_TSV_COLUMNS.len(), "row has wrong contract width: {row}");
        assert_eq!(fields.len(), header_fields.len(), "row has wrong schema width: {row}");
        match fields[0] {
            "q01" => {
                assert_eq!(fields[bounded_status_index], "bounded_accepted_parity_matched");
                assert_eq!(fields[final_disposition_index], "native");
                assert_eq!(fields[native_started_index], "true");
                assert_eq!(fields[runtime_outcome_index], "succeeded");
                assert_eq!(fields[output_fingerprint_index], "output-fnv1a64:0000000000000001");
            },
            "q02" => {
                assert_eq!(fields[planning_outcome_index], "not_supported");
                assert_eq!(fields[bounded_status_index], "bounded_rejected");
                assert_eq!(fields[final_disposition_index], "datafusion");
                assert_eq!(fields[rejected_before_execution_index], "false");
                assert_eq!(fields[native_started_index], "false");
                assert_eq!(fields[runtime_outcome_index], "not_started");
                assert_eq!(fields[output_fingerprint_index], "not_run");
            },
            "q03" => {
                assert_eq!(fields[planning_outcome_index], "none");
                assert_eq!(fields[bounded_status_index], "bounded_planning_failed");
                assert_eq!(fields[final_disposition_index], "none");
                assert_eq!(fields[rejected_before_execution_index], "true");
                assert_eq!(fields[native_started_index], "false");
                assert_eq!(fields[runtime_outcome_index], "not_started");
                assert_eq!(fields[output_fingerprint_index], "not_run");
            },
            query_id => panic!("unexpected benchmark row {query_id}"),
        }
    }
}

fn benchmark_metadata() -> TpchDatasetMetadata {
    benchmark_metadata_for_scale("sf1", 8_661_245, benchmark_sf1_tables())
}

fn benchmark_sf10_metadata() -> TpchDatasetMetadata {
    benchmark_metadata_for_scale("sf10", 86_586_082, benchmark_sf10_tables())
}

fn benchmark_sf100_metadata() -> TpchDatasetMetadata {
    benchmark_metadata_for_scale("sf100", 866_037_932, benchmark_sf100_tables())
}

fn benchmark_metadata_for_scale(
    scale_factor: impl Into<String>,
    total_rows: u64,
    tables: Vec<TpchTableMetadata>,
) -> TpchDatasetMetadata {
    TpchDatasetMetadata {
        root: PathBuf::from("/tmp/tpch"),
        scale_factor: scale_factor.into(),
        provenance: "external_parquet_root".to_string(),
        generation_command: "dbgen".to_string(),
        table_count: 8,
        file_count: 8,
        total_rows,
        total_bytes: 8,
        metadata_fingerprint: "metadata-fnv1a64:0000000000000001".to_string(),
        content_fingerprint: Some("content-fnv1a64:0000000000000001".to_string()),
        tables,
    }
}

fn benchmark_sf1_tables() -> Vec<TpchTableMetadata> {
    benchmark_tables(&[
        ("customer", 150_000),
        ("lineitem", 6_001_215),
        ("nation", 25),
        ("orders", 1_500_000),
        ("part", 200_000),
        ("partsupp", 800_000),
        ("region", 5),
        ("supplier", 10_000),
    ])
}

fn benchmark_sf10_tables() -> Vec<TpchTableMetadata> {
    benchmark_tables(&[
        ("customer", 1_500_000),
        ("lineitem", 59_986_052),
        ("nation", 25),
        ("orders", 15_000_000),
        ("part", 2_000_000),
        ("partsupp", 8_000_000),
        ("region", 5),
        ("supplier", 100_000),
    ])
}

fn benchmark_sf100_tables() -> Vec<TpchTableMetadata> {
    benchmark_tables(&[
        ("customer", 15_000_000),
        ("lineitem", 600_037_902),
        ("nation", 25),
        ("orders", 150_000_000),
        ("part", 20_000_000),
        ("partsupp", 80_000_000),
        ("region", 5),
        ("supplier", 1_000_000),
    ])
}

fn benchmark_tables(row_counts: &[(&str, u64)]) -> Vec<TpchTableMetadata> {
    row_counts
        .iter()
        .map(|(table, row_count)| benchmark_table_metadata(table, *row_count, benchmark_table_columns(table)))
        .collect()
}

fn benchmark_table_columns(table: &str) -> &'static [&'static str] {
    match table {
        "customer" => {
            &["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]
        },
        "lineitem" => &[
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
        ],
        "nation" => &["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        "orders" => &[
            "o_orderkey",
            "o_custkey",
            "o_orderstatus",
            "o_totalprice",
            "o_orderdate",
            "o_orderpriority",
            "o_clerk",
            "o_shippriority",
            "o_comment",
        ],
        "part" => &[
            "p_partkey",
            "p_name",
            "p_mfgr",
            "p_brand",
            "p_type",
            "p_size",
            "p_container",
            "p_retailprice",
            "p_comment",
        ],
        "partsupp" => &["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        "region" => &["r_regionkey", "r_name", "r_comment"],
        "supplier" => &["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"],
        other => panic!("unknown benchmark table {other}"),
    }
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
        physical_root: "NexusNativeExec".to_string(),
        physical_shape: "NexusNativeExec".to_string(),
        native_candidate_shape: Some("NexusNativeExec".to_string()),
        native_candidate_shape_family: Some("sort_local_parquet_aggregate".to_string()),
        outcome: Some(CandidateOutcomeKind::Selected),
        reason_code: None,
        unsupported_category: None,
        capability_decision_rows: Vec::new(),
        capability_compiled_graphs: Vec::new(),
        native_optimizer: Default::default(),
        edge_normalize_inserted: false,
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "NexusNativeExec".to_string(),
        final_disposition: Some(FinalPlanDisposition::Native),
        native_plan_fragment_count: Some(1),
        host_datafusion_boundary_count: Some(0),
        rejected_before_execution: false,
    }
}

fn not_supported_planning_row(query_id: &str) -> NativeDataFusionPlanningRow {
    NativeDataFusionPlanningRow {
        query_id: query_id.to_string(),
        physical_root: "SortExec".to_string(),
        physical_shape: "SortExec(HashJoinExec)".to_string(),
        native_candidate_shape: Some("SortExec(HashJoinExec)".to_string()),
        native_candidate_shape_family: Some("sort_hash_join_aggregate".to_string()),
        outcome: Some(CandidateOutcomeKind::NotSupported),
        reason_code: Some("native_bounded_join_unsupported".to_string()),
        unsupported_category: Some("operator_hashjoin".to_string()),
        capability_decision_rows: Vec::new(),
        capability_compiled_graphs: Vec::new(),
        native_optimizer: Default::default(),
        edge_normalize_inserted: false,
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "SortExec".to_string(),
        final_disposition: Some(FinalPlanDisposition::DataFusion),
        native_plan_fragment_count: Some(0),
        host_datafusion_boundary_count: Some(0),
        rejected_before_execution: false,
    }
}

fn accepted_capability_compiled_graphs() -> Vec<datafusion_nexus::planner::NativeCapabilityCompiledGraphDiagnostics> {
    vec![datafusion_nexus::planner::NativeCapabilityCompiledGraphDiagnostics {
        task_count: 3,
        edge_count: 2,
        barrier_count: 0,
        resource_count: 3,
        sink_kind: "host_arrow_materialization".to_string(),
    }]
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

fn header_index(header: &[&str], name: &str) -> usize {
    header.iter().position(|field| *field == name).unwrap_or_else(|| panic!("missing TSV field {name}"))
}

fn tpch_native_profile_settings_from<const N: usize>(
    entries: [(&'static str, &'static str); N],
) -> DataFusionResult<NativeTpchProfileSettings> {
    let env = entries.into_iter().collect::<std::collections::BTreeMap<_, _>>();
    native_tpch_profile_settings_with_env(|name| env.get(name).copied().map(ToString::to_string))
}
