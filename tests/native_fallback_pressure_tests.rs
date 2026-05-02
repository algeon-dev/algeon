use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt,
    GpuFallbackPolicy, NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeCudfExecutionConfig,
    NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath,
    NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureWorkloadItem,
};

#[tokio::test]
async fn native_fallback_pressure_report_summarizes_workload_outcomes() {
    let dir = unique_temp_dir("datafusion-cudf-native-fallback-pressure");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let workload = [
        ("sum_rewrite", SQL_SUM_QUERY),
        ("not_rewrite", SQL_NOT_QUERY),
        ("cast_rewrite", SQL_CAST_QUERY),
        ("avg_rewrite", SQL_AVG_QUERY),
        ("filtered_reject", SQL_FILTERED_QUERY),
        ("modulo_reject", SQL_MODULO_QUERY),
    ];
    let mut items = Vec::with_capacity(workload.len());
    for (query_id, sql) in workload {
        items.push(NativeFallbackPressureWorkloadItem::new(
            query_id,
            sql_plan(&path, direct_friendly_sql_session_config(), sql).await,
        ));
    }

    let report = NativeFallbackPressureReport::collect(items);

    assert_eq!(report.rows.len(), 6);
    assert_eq!(report.rewritten_count(), 4);
    assert_eq!(report.rejected_count(), 2);

    let rewritten = report_row(&report, "sum_rewrite");
    assert_eq!(rewritten.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(rewritten.reason_code, None);
    assert_eq!(rewritten.unsupported_category, None);
    assert!(!rewritten.rejected_before_execution);
    assert_eq!(rewritten.source_diagnostics.len(), 1);
    assert!(rewritten.source_diagnostics[0].direct_device_read);
    assert_eq!(rewritten.source_diagnostics[0].predicate_disposition, "pruning_only");

    let not = report_row(&report, "not_rewrite");
    assert_eq!(not.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(not.reason_code, None);
    assert_eq!(not.unsupported_category, None);
    assert!(!not.rejected_before_execution);

    let cast = report_row(&report, "cast_rewrite");
    assert_eq!(cast.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(cast.reason_code, None);
    assert_eq!(cast.unsupported_category, None);
    assert!(!cast.rejected_before_execution);

    let avg = report_row(&report, "avg_rewrite");
    assert_eq!(avg.outcome, NativeFallbackPressureOutcome::Rewritten);
    assert_eq!(avg.reason_code, None);
    assert_eq!(avg.unsupported_category, None);
    assert!(!avg.rejected_before_execution);

    let filtered = report_row(&report, "filtered_reject");
    assert_eq!(filtered.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(filtered.unsupported_category.as_deref(), Some("aggregate_filter"));
    assert!(filtered.rejected_before_execution);

    let modulo = report_row(&report, "modulo_reject");
    assert_eq!(modulo.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(modulo.unsupported_category.as_deref(), Some("expression_binary_modulo"));
    assert!(modulo.rejected_before_execution);

    let text = report.to_tsv();
    assert_eq!(NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, 1);
    assert_eq!(
        NATIVE_FALLBACK_PRESSURE_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tsource_diagnostics\trejected_before_execution"
    );
    assert!(text.starts_with("# native_fallback_pressure_tsv_schema_version=1\n"));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("sum_rewrite\tSortExec"));
    assert!(text.contains("not_rewrite\tSortExec"));
    assert!(text.contains("cast_rewrite\tSortExec"));
    assert!(text.contains("avg_rewrite\tSortExec"));
    assert!(text.contains("\trewritten\tnone\tnone\t"));
    assert!(text.contains("\tunsupported_expression\taggregate_filter\t"));
    assert!(text.contains("\tunsupported_expression\texpression_binary_modulo\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_planning_report_summarizes_session_optimizer_outcomes() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-planning-report");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    let accepted = native_datafusion_planning_report("sum_session", &path, SQL_SUM_QUERY).await;
    assert_eq!(accepted.rows.len(), 1);
    assert_eq!(accepted.accepted_count(), 1);
    assert_eq!(accepted.rejected_count(), 0);
    let accepted_row = &accepted.rows[0];
    assert_eq!(accepted_row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(accepted_row.reason_code, None);
    assert_eq!(accepted_row.unsupported_category, None);
    assert_eq!(accepted_row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(accepted_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(!accepted_row.rejected_before_execution);
    assert_eq!(accepted_row.source_diagnostics.len(), 1);
    assert!(accepted_row.source_diagnostics[0].direct_device_read);

    let rejected = native_datafusion_planning_report("filtered_session", &path, SQL_FILTERED_QUERY).await;
    assert_eq!(rejected.rows.len(), 1);
    assert_eq!(rejected.accepted_count(), 0);
    assert_eq!(rejected.rejected_count(), 1);
    let rejected_row = &rejected.rows[0];
    assert_eq!(rejected_row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
    assert_eq!(rejected_row.reason_code.as_deref(), Some("unsupported_expression"));
    assert_ne!(rejected_row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert!(rejected_row.rejected_before_execution);
    assert_eq!(rejected_row.source_diagnostics.len(), 1);
    assert!(rejected_row.source_diagnostics[0].direct_device_read);

    let text = accepted.to_tsv();
    assert_eq!(NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION, 1);
    assert_eq!(
        NATIVE_DATAFUSION_PLANNING_TSV_HEADER,
        "query_id\tphysical_root\tphysical_shape\tnative_candidate_shape\toutcome\treason_code\tunsupported_category\tsource_diagnostics\tpost_native_plan_root\tpost_native_execution_path\trejected_before_execution"
    );
    assert!(text.starts_with("# native_datafusion_planning_tsv_schema_version=1\n"));
    assert!(text.contains(&format!("{NATIVE_DATAFUSION_PLANNING_TSV_HEADER}\n")));
    assert!(text.contains("sum_session\t"));
    assert!(text.contains("\taccepted_replaced\tnone\tnone\t"));
    assert!(text.contains("\tNativeCudfExec\tnative\tfalse"));

    fs::remove_dir_all(dir).unwrap();
}

fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a datafusion_cudf::NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1));
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let gpu_island_rule = Arc::new(CudfPhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(direct_friendly_sql_session_config())
        .with_cudf_native_and_gpu_island_rules(Arc::clone(&native_rule), gpu_island_rule)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    NativeDataFusionPlanningReport::from_retained_rule(query_id, plan, native_rule.as_ref())
}

fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 15, 20, 25, 30])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

const SQL_SUM_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_NOT_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE NOT (payload < 10)
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_CAST_QUERY: &str = "
SELECT key, SUM(CAST(key AS BIGINT)) AS sum_cast
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_cast DESC, key ASC
LIMIT 2
";

const SQL_AVG_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_FILTERED_QUERY: &str = "
SELECT key, SUM(m00) FILTER (WHERE payload >= 10) AS sum_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_m00 DESC, key ASC
LIMIT 2
";

const SQL_MODULO_QUERY: &str = "
SELECT key, SUM(payload % 2) AS modulo_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY modulo_payload DESC, key ASC
LIMIT 2
";
