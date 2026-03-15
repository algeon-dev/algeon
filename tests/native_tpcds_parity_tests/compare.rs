use std::sync::Arc;

use arrow::{
    array::{Decimal128Array, Float32Array, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::physical_plan::{ExecutionPlan, empty::EmptyExec, metrics::MetricsSet};
use datafusion_nexus::{
    GpuCandidate, NativeAdaptivePipelineDiagnostics, NativePlanningDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityNativePlanningOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityReport, NativeTpcdsParityWorkloadItem,
    ParityComparisonMode, ParityFloatTolerance, ParityStatus, PlanningReport, compare_record_batches,
    sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};

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
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_matched_uses_comparison_fingerprints_for_compatible_schema_names() {
    let cpu_schema = int_string_schema();
    let native_schema = Arc::new(Schema::new(vec![
        Field::new("native_id", DataType::Int64, false),
        Field::new("native_label", DataType::Utf8, true),
    ]));
    let cpu = vec![int_string_batch(&cpu_schema, &[1, 2], &[Some("a"), Some("b")])];
    let native = vec![int_string_batch(&native_schema, &[1, 2], &[Some("a"), Some("b")])];

    let result =
        compare_record_batches(&cpu_schema, &cpu, &native_schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
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
fn external_oracle_decimal_precision_widening_matches_cpu_value() {
    let cpu_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(17, 2), false)]));
    let duckdb_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(38, 2), false)]));
    let cpu_batch = RecordBatch::try_new(
        Arc::clone(&cpu_schema),
        vec![Arc::new(Decimal128Array::from(vec![12345_i128]).with_precision_and_scale(17, 2).unwrap())],
    )
    .unwrap();
    let duckdb_batch = RecordBatch::try_new(
        Arc::clone(&duckdb_schema),
        vec![Arc::new(Decimal128Array::from(vec![12345_i128]).with_precision_and_scale(38, 2).unwrap())],
    )
    .unwrap();
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "qxx".to_string(),
        sql: "SELECT amount FROM t ORDER BY amount".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded { schema: cpu_schema, batches: vec![cpu_batch] },
        native_planning: datafusion_nexus::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_nexus::NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: "not relevant".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        duckdb_output: NativeTpcdsParityExternalOutput::Succeeded {
            schema: duckdb_schema,
            batches: vec![duckdb_batch],
        },
    };

    let report = NativeTpcdsParityReport::collect([item], ParityFloatTolerance::exact());

    assert_eq!(report.rows[0].duckdb_output.cpu_parity_status, ParityStatus::Matched);
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
        std::slice::from_ref(&cpu),
        &schema,
        std::slice::from_ref(&native),
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
fn compare_record_batches_float32_tolerance_rejects_distant_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float32, false)]));
    let cpu = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float32Array::from(vec![1.0_f32]))]).unwrap();
    let native = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float32Array::from(vec![42.0_f32]))]).unwrap();

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
fn parity_report_retries_unordered_only_for_partial_order_by_ties() {
    let schema = int_string_schema();
    let cpu_batch = int_string_batch(&schema, &[1, 1], &[Some("a"), Some("b")]);
    let native_batch = int_string_batch(&schema, &[1, 1], &[Some("b"), Some("a")]);
    let partial_item = NativeTpcdsParityWorkloadItem {
        query_id: "partial".to_string(),
        sql: "SELECT id, label FROM t ORDER BY id".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![cpu_batch.clone()],
        },
        native_planning: accepted_native_planning(&schema),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![native_batch.clone()],
            metrics: MetricsSet::new(),
        },
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };
    let total_item = NativeTpcdsParityWorkloadItem {
        query_id: "total".to_string(),
        sql: "SELECT id, label FROM t ORDER BY id, label".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded { schema: Arc::clone(&schema), batches: vec![cpu_batch] },
        native_planning: accepted_native_planning(&schema),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![native_batch],
            metrics: MetricsSet::new(),
        },
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };

    let report = NativeTpcdsParityReport::collect([partial_item, total_item], ParityFloatTolerance::exact());
    let partial = report.rows.iter().find(|row| row.query_id == "partial").unwrap();
    assert_eq!(partial.parity_status, ParityStatus::Matched);
    assert_eq!(partial.comparison_mode, ParityComparisonMode::UnorderedExact);

    let total = report.rows.iter().find(|row| row.query_id == "total").unwrap();
    assert_eq!(total.parity_status, ParityStatus::Mismatched);
    assert_eq!(total.comparison_mode, ParityComparisonMode::OrderedExact);
}

#[test]
fn parity_status_for_cpu_failure_flags_cpu_failed_and_skips_comparison() {
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "q01".to_string(),
        sql: "SELECT 1 ORDER BY 1".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::PlanningFailed(datafusion_nexus::NativePlanningFailure {
            stage: "sql_to_logical_plan".to_string(),
            reason_code: "sql_planning_failed".to_string(),
            message: "boom".to_string(),
        }),
        native_planning: datafusion_nexus::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_nexus::NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: "boom".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
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

#[test]
fn sql_top_level_order_by_covers_output_columns_detects_total_order() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    assert!(sql_top_level_order_by_covers_output_columns("SELECT id, label FROM t ORDER BY id, label", &schema));
    assert!(sql_top_level_order_by_covers_output_columns(
        "SELECT id, label FROM t ORDER BY t.id DESC NULLS FIRST, label ASC LIMIT 10",
        &schema,
    ));
}

#[test]
fn sql_top_level_order_by_covers_output_columns_rejects_partial_order() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("brand_id", DataType::Int64, false),
        Field::new("brand", DataType::Utf8, true),
        Field::new("t_hour", DataType::Int64, false),
        Field::new("t_minute", DataType::Int64, false),
        Field::new("ext_price", DataType::Decimal128(15, 2), true),
    ]));
    let q71_like = "
SELECT i_brand_id brand_id, i_brand brand, t_hour, t_minute, sum(ext_price) ext_price
FROM t
GROUP BY i_brand, i_brand_id, t_hour, t_minute
ORDER BY ext_price DESC NULLS FIRST, i_brand_id NULLS FIRST, t_hour NULLS FIRST
";
    assert!(!sql_top_level_order_by_covers_output_columns(q71_like, &schema));
    assert!(!sql_top_level_order_by_covers_output_columns(
        "SELECT id, label, value FROM t ORDER BY id, label",
        &Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, true),
            Field::new("value", DataType::Int64, false),
        ]),
    ));
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

fn accepted_native_planning(schema: &Arc<Schema>) -> NativeTpcdsParityNativePlanningOutcome {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("test_native_plan", schema.fields().len()));
    NativeTpcdsParityNativePlanningOutcome::Planned {
        accepted: true,
        planned_plan: Arc::new(EmptyExec::new(Arc::clone(schema))) as Arc<dyn ExecutionPlan>,
        diagnostics: Box::new(NativePlanningDiagnostics {
            input_plan_root: "EmptyExec".to_string(),
            input_plan_shape: "EmptyExec".to_string(),
            input_join_diagnostics: Vec::new(),
            report,
            native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics::default(),
            source_diagnostics: Vec::new(),
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
        }),
    }
}
