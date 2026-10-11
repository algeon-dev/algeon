use std::{path::PathBuf, time::Duration};

use algeon_datafusion::planner::NativeCapabilityCompiledGraphDiagnostics;
use workloads::tpch::TpchTableMetadata;

use super::*;

mod config_dataset;
mod large_keyed_join_gate;
mod output_fingerprint;
mod provenance;
mod rejected_rows;
mod report_rows;
mod runtime_failures;

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
    let query_id_index = header_index(&header_fields, "query_id");
    let planning_outcome_index = header_index(&header_fields, "planning_outcome");
    let bounded_status_index = header_index(&header_fields, "bounded_status");
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
        match fields[query_id_index] {
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
        physical_root: "AlgeonNativeExec".to_string(),
        physical_shape: "AlgeonNativeExec".to_string(),
        native_candidate_shape: Some("AlgeonNativeExec".to_string()),
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
        post_native_plan_root: "AlgeonNativeExec".to_string(),
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

fn accepted_capability_compiled_graphs() -> Vec<NativeCapabilityCompiledGraphDiagnostics> {
    vec![NativeCapabilityCompiledGraphDiagnostics {
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
        planning_samples: vec![Duration::from_millis(2)],
        plan_plus_execution_samples: vec![Duration::from_millis(7)],
        iteration_wall_samples: vec![Duration::from_millis(8)],
        final_measurement: query_measurement(fingerprint),
        metrics: NativeDataFusionExecutionMetrics::default(),
        timing_metrics: NativeTpchTimingMetrics::default(),
    }
}

#[test]
fn benchmark_sidecar_keeps_each_planning_and_execution_sample() {
    let dir = unique_temp_dir("algeon-tpch-samples-test");
    let mut native = timed_native_run("same");
    native.samples.push(Duration::from_millis(11));
    native.planning_samples.push(Duration::from_millis(3));
    native.plan_plus_execution_samples.push(Duration::from_millis(14));
    native.iteration_wall_samples.push(Duration::from_millis(15));

    artifacts::write_samples(&dir, "q01", None, Some(&native)).expect("write samples");
    let text = std::fs::read_to_string(dir.join("q01.samples.tsv")).expect("read samples");
    let lines = text.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[1], "1\t\t2.000\t5.000\t7.000\t8.000");
    assert_eq!(lines[2], "2\t\t3.000\t11.000\t14.000\t15.000");
    std::fs::remove_dir_all(dir).expect("remove samples directory");
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
