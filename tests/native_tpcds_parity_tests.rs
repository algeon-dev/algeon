use std::sync::Arc;

use arrow::{
    array::{Decimal128Array, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion_cudf::{
    NATIVE_TPCDS_PARITY_TSV_HEADER, NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, NativeTpcdsParityCpuOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityReport, NativeTpcdsParityRow,
    NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityFloatTolerance, ParityStatus, compare_record_batches,
    sql_has_top_level_order_by,
};

#[test]
fn parity_tsv_header_pins_schema_version_and_columns() {
    assert_eq!(NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, 1);
    let columns: Vec<&str> = NATIVE_TPCDS_PARITY_TSV_HEADER.split('\t').collect();
    assert_eq!(
        columns,
        vec![
            "query_id",
            "cpu_planning_status",
            "cpu_runtime_status",
            "native_planning_outcome",
            "native_reason_code",
            "native_unsupported_category",
            "native_runtime_started",
            "native_runtime_outcome",
            "parity_status",
            "parity_detail",
            "cpu_output_rows",
            "native_output_rows",
            "cpu_output_batches",
            "native_output_batches",
            "cpu_output_fingerprint",
            "native_output_fingerprint",
            "comparison_mode",
            "native_reservation_live_peak_bytes",
            "native_reservation_denied_count",
            "native_retry_count",
            "native_runtime_failure",
        ]
    );
}

#[test]
fn parity_report_to_tsv_emits_schema_version_and_header() {
    let row = NativeTpcdsParityRow {
        query_id: "q01".to_string(),
        cpu_planning_status: datafusion_cudf::ParityStageStatus::Succeeded,
        cpu_runtime_status: datafusion_cudf::ParityStageStatus::Succeeded,
        native_planning_outcome: datafusion_cudf::NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution,
        native_reason_code: Some("unsupported_operator".to_string()),
        native_unsupported_category: Some("operator_join".to_string()),
        native_runtime_started: false,
        native_runtime_outcome: datafusion_cudf::NativeDataFusionRuntimeOutcome::NotStarted,
        parity_status: ParityStatus::NativeRejected,
        parity_detail: None,
        cpu_output_rows: Some(0),
        native_output_rows: None,
        cpu_output_batches: Some(0),
        native_output_batches: None,
        cpu_output_fingerprint: None,
        native_output_fingerprint: None,
        comparison_mode: ParityComparisonMode::NotApplicable,
        native_reservation_live_peak_bytes: None,
        native_reservation_denied_count: None,
        native_retry_count: None,
        native_runtime_failure: None,
    };
    let report = NativeTpcdsParityReport::from_rows(vec![row], ParityFloatTolerance::exact());
    let tsv = report.to_tsv();
    let first_line = tsv.lines().next().unwrap();
    assert_eq!(
        first_line,
        format!("# native_tpcds_parity_tsv_schema_version={NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION}")
    );
    assert!(tsv.contains(NATIVE_TPCDS_PARITY_TSV_HEADER));
    // 21 columns separated by tabs in the data row.
    let data_line = tsv.lines().find(|line| line.starts_with("q01\t")).expect("data row should exist");
    assert_eq!(data_line.split('\t').count(), 21);
}

#[test]
fn compare_record_batches_matched_ordered_exact_for_int_and_string() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), None])];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), None])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedExact);
    assert!(result.detail.is_none());
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_mismatched_ordered_exact_first_mismatch_row_and_column() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), Some("c")])];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("DIFF"), Some("c")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedExact);
    let detail = result.detail.expect("mismatch should have detail");
    assert!(detail.contains("row 1"), "detail must report first mismatch row: {detail}");
    assert!(detail.contains("label"), "detail must include column name: {detail}");
    assert_ne!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_unordered_match_across_batch_boundaries() {
    let schema = int_string_schema();
    let cpu = vec![
        int_string_batch(&schema, &[3, 1], &[Some("c"), Some("a")]),
        int_string_batch(&schema, &[2], &[Some("b")]),
    ];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), Some("c")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, false, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.mode, ParityComparisonMode::UnorderedExact);
}

#[test]
fn compare_record_batches_distinguishes_null_from_empty_string() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1], &[None])];
    let native = vec![int_string_batch(&schema, &[1], &[Some("")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("mismatch should have detail");
    assert!(detail.contains("label"), "detail must include column name: {detail}");
}

#[test]
fn compare_record_batches_decimal_scale_difference_is_mismatch() {
    let schema_a = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let schema_b = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 4), false)]));
    let batch_a = RecordBatch::try_new(
        Arc::clone(&schema_a),
        vec![Arc::new(Decimal128Array::from(vec![100_i128]).with_precision_and_scale(15, 2).unwrap())],
    )
    .unwrap();
    let batch_b = RecordBatch::try_new(
        Arc::clone(&schema_b),
        vec![Arc::new(Decimal128Array::from(vec![10000_i128]).with_precision_and_scale(15, 4).unwrap())],
    )
    .unwrap();

    let result =
        compare_record_batches(&schema_a, &[batch_a], &schema_b, &[batch_b], true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("schema mismatch must have detail");
    assert!(detail.contains("type differs"), "detail must explain schema mismatch: {detail}");
}

#[test]
fn compare_record_batches_float_tolerance_matches_close_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let cpu =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![0.30000000000000004_f64]))])
            .unwrap();
    let native = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![0.3_f64]))]).unwrap();

    let exact = compare_record_batches(
        &schema,
        &[cpu.clone()],
        &schema,
        &[native.clone()],
        true,
        ParityFloatTolerance::exact(),
    );
    assert_eq!(exact.status, ParityStatus::Mismatched);
    assert_eq!(exact.mode, ParityComparisonMode::OrderedExact);

    let tolerant =
        compare_record_batches(&schema, &[cpu], &schema, &[native], true, ParityFloatTolerance::new(1e-9, 0.0));
    assert_eq!(tolerant.status, ParityStatus::Matched);
    assert_eq!(tolerant.mode, ParityComparisonMode::OrderedFloatTolerant);
}

#[test]
fn compare_record_batches_float_tolerance_rejects_distant_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let cpu = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![1.0_f64]))]).unwrap();
    let native = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![2.0_f64]))]).unwrap();

    let result =
        compare_record_batches(&schema, &[cpu], &schema, &[native], true, ParityFloatTolerance::new(1e-3, 1e-3));
    assert_eq!(result.status, ParityStatus::Mismatched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedFloatTolerant);
}

#[test]
fn compare_record_batches_row_count_mismatch_is_explicit() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2], &[Some("a"), Some("b")])];
    let native = vec![int_string_batch(&schema, &[1], &[Some("a")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());
    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("row count mismatch must have detail");
    assert!(detail.contains("row count"), "detail must explain row count: {detail}");
}

#[test]
fn parity_status_for_cpu_failure_flags_cpu_failed_and_skips_comparison() {
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "q01".to_string(),
        sql: "SELECT 1 ORDER BY 1".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::PlanningFailed(datafusion_cudf::NativePlanningFailure {
            stage: "sql_to_logical_plan".to_string(),
            reason_code: "sql_planning_failed".to_string(),
            message: "boom".to_string(),
        }),
        native_planning: datafusion_cudf::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_cudf::NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: "boom".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
    };
    let report = NativeTpcdsParityReport::collect([item], ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.rows[0].parity_status, ParityStatus::CpuFailed);
    assert_eq!(report.rows[0].comparison_mode, ParityComparisonMode::NotApplicable);
}

#[test]
fn sql_has_top_level_order_by_detects_outer_order_by() {
    assert!(sql_has_top_level_order_by("SELECT * FROM t ORDER BY a"));
    assert!(sql_has_top_level_order_by("\nSELECT * FROM t\nORDER BY a\nLIMIT 100\n"));
}

#[test]
fn sql_has_top_level_order_by_ignores_order_by_inside_subquery() {
    let sql = "SELECT * FROM (SELECT * FROM t ORDER BY a) sub";
    assert!(!sql_has_top_level_order_by(sql));
}

#[test]
fn sql_has_top_level_order_by_ignores_order_by_inside_string_or_comment() {
    let sql = "SELECT 'order by foo' AS x FROM t -- order by trailing comment";
    assert!(!sql_has_top_level_order_by(sql));
    let block = "SELECT /* ORDER BY ignored */ 1 FROM t";
    assert!(!sql_has_top_level_order_by(block));
}

fn int_string_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("label", DataType::Utf8, true)]))
}

fn int_string_batch(schema: &Arc<Schema>, ids: &[i64], labels: &[Option<&str>]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![Arc::new(Int64Array::from(ids.to_vec())), Arc::new(StringArray::from(labels.to_vec()))],
    )
    .unwrap()
}
