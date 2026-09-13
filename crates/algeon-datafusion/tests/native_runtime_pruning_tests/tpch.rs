use std::{fs, fs::File, sync::Arc};

use arrow::array::{Date32Array, Int32Array, Int64Array, StringArray};
use cudf::testing::arrow::{
    arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
    reset_arrow_host_interop_observation_for_test,
};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::{fixtures::date32, *};

#[tokio::test]
async fn native_runtime_tpch_q6_tiny_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q6-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    assert_tpch_q6_native_parity(&dir.join("lineitem.parquet")).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q1_tiny_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q1-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q1 = TPCH_QUERIES.iter().find(|query| query.id == "q01").expect("TPC-H Q1 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q1.sql).await;
    assert!(
        cpu_batches.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0,
        "tiny TPC-H Q1 should exercise non-empty output parity"
    );

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q1.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "TPC-H Q1 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q3_tiny_bounded_join_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q3-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q3 = TPCH_QUERIES.iter().find(|query| query.id == "q03").expect("TPC-H Q3 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q3.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q3.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "bounded TPC-H Q3 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);
    assert_batches_equal(&cpu_batches, &native_batches);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q7_tiny_bounded_duplicate_nation_identity_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q7-tiny-bounded");
    write_q7_nonempty_tpch_parquet_dataset(&dir);
    let q7 = TPCH_QUERIES.iter().find(|query| query.id == "q07").expect("TPC-H Q7 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q7.sql).await;
    assert_q7_nation_identity_rows(&cpu_batches);

    let physical_plan = tpch_sql_plan(&dir, tpch_report_session_config(), q7.sql).await;
    let optimization = optimize_with_pruning_template(physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "bounded TPC-H Q7 should be replaced by AlgeonNativeExec");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q7.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q7 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
    assert_q7_nation_identity_rows(&native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);

    let metrics = native_plan.metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize() >= 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q2_tiny_bounded_aggregate_over_join_subplan_topk_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q2-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q2 = TPCH_QUERIES.iter().find(|query| query.id == "q02").expect("TPC-H Q2 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q2.sql).await;

    let physical_plan = tpch_sql_plan(&dir, tpch_report_session_config(), q2.sql).await;
    let optimization = optimize_with_pruning_template(physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "bounded TPC-H Q2 should be replaced by AlgeonNativeExec");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q2.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q2 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q19_tiny_bounded_join_residual_in_list_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q19-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q19 = TPCH_QUERIES.iter().find(|query| query.id == "q19").expect("TPC-H Q19 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q19.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q19.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "bounded TPC-H Q19 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);
    assert_batches_equal(&cpu_batches, &native_batches);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q04_tiny_bounded_left_semi_full_sort_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q04-tiny-bounded");
    write_q04_nonempty_tpch_parquet_dataset(&dir);
    let q4 = TPCH_QUERIES.iter().find(|query| query.id == "q04").expect("TPC-H Q4 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q4.sql).await;
    assert_eq!(record_batch_row_count(&cpu_batches), 1, "tiny TPC-H Q4 CPU output should be non-empty");

    let physical_plan = tpch_sql_plan(&dir, tpch_report_session_config(), q4.sql).await;
    let optimization = optimize_with_pruning_template(physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "bounded TPC-H Q4 should be replaced by AlgeonNativeExec");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q4.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q4 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);

    let metrics = native_plan.metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize() >= 1);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_attempts").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_host_arrow_sinks").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q15_tiny_bounded_root_join_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q15-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q15 = TPCH_QUERIES.iter().find(|query| query.id == "q15").expect("TPC-H Q15 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q15.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q15.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "bounded TPC-H Q15 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q18_tiny_bounded_resident_aggregate_topk_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q18-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q18 = TPCH_QUERIES.iter().find(|query| query.id == "q18").expect("TPC-H Q18 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q18.sql).await;

    let physical_plan = tpch_sql_plan(&dir, tpch_report_session_config(), q18.sql).await;
    let optimization = optimize_with_pruning_template(physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "bounded TPC-H Q18 should be replaced by AlgeonNativeExec");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q18.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q18 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);

    let metrics = native_plan.metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize() >= 1);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_attempts").unwrap().as_usize(), 1);
    assert!(metrics.sum_by_name("compiled_pipeline_tasks_executed").unwrap().as_usize() > 0);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_host_arrow_sinks").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q20_tiny_bounded_resident_aggregate_subquery_filter_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q20-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q20 = TPCH_QUERIES.iter().find(|query| query.id == "q20").expect("TPC-H Q20 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q20.sql).await;

    let physical_plan = tpch_sql_plan(&dir, tpch_report_session_config(), q20.sql).await;
    let optimization = optimize_with_pruning_template(physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "bounded TPC-H Q20 should be replaced by AlgeonNativeExec");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q20.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q20 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);

    let metrics = native_plan.metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize() >= 1);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_attempts").unwrap().as_usize(), 1);
    assert!(metrics.sum_by_name("compiled_pipeline_tasks_executed").unwrap().as_usize() > 0);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_host_arrow_sinks").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q16_tiny_bounded_distinct_anti_join_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q16-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q16 = TPCH_QUERIES.iter().find(|query| query.id == "q16").expect("TPC-H Q16 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q16.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q16.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "bounded TPC-H Q16 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q11_tiny_bounded_aggregate_over_join_subplan_inputs_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q11-tiny-bounded");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q11.sql).await;
    assert_eq!(record_batch_row_count(&cpu_batches), 1, "tiny TPC-H Q11 CPU output should be non-empty");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q11.sql).await;
    assert_algeon_native_exec(&native_plan, "bounded TPC-H Q11 should be replaced by AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);

    let metrics = native_plan.metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize() >= 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q17_tiny_bounded_decimal_residual_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q17-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q17 = TPCH_QUERIES.iter().find(|query| query.id == "q17").expect("TPC-H Q17 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q17.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q17.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "TPC-H Q17 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert!(arrow_host_export_call_count_for_test() >= 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q22_tiny_bounded_scalar_subquery_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-tpch-q22-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q22 = TPCH_QUERIES.iter().find(|query| query.id == "q22").expect("TPC-H Q22 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q22.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q22.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "TPC-H Q22 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

fn record_batch_row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

async fn assert_tpch_q6_native_parity(lineitem_path: &std::path::Path) {
    assert!(lineitem_path.exists(), "missing lineitem parquet file: {}", lineitem_path.display());
    let q6 = TPCH_QUERIES.iter().find(|query| query.id == "q06").expect("TPC-H Q6 query must exist");
    let cpu_batches = cpu_lineitem_sql_output(lineitem_path, q6.sql).await;
    assert!(
        cpu_batches.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0,
        "tiny TPC-H Q6 should exercise non-empty output parity"
    );

    let native_plan = native_datafusion_lineitem_sql_plan(lineitem_path, q6.sql).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("AlgeonNativeExec"),
        "TPC-H Q6 should be replaced by AlgeonNativeExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);
}

fn write_q7_nonempty_tpch_parquet_dataset(root: &std::path::Path) {
    write_tiny_tpch_parquet_dataset(root).unwrap();
    write_tpch_test_table(
        &root.join("customer.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_nationkey", DataType::Int64, false),
        ])),
        vec![Arc::new(Int64Array::from(vec![10, 20])), Arc::new(Int64Array::from(vec![3, 4]))],
    );
    write_tpch_test_table(
        &root.join("supplier.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("s_suppkey", DataType::Int64, false),
            Field::new("s_nationkey", DataType::Int64, false),
        ])),
        vec![Arc::new(Int64Array::from(vec![100, 200])), Arc::new(Int64Array::from(vec![3, 4]))],
    );
    write_tpch_test_table(
        &root.join("orders.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_custkey", DataType::Int64, false),
        ])),
        vec![Arc::new(Int64Array::from(vec![1000, 2000, 3000])), Arc::new(Int64Array::from(vec![20, 10, 20]))],
    );
    write_tpch_test_table(
        &root.join("nation.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("n_nationkey", DataType::Int64, false),
            Field::new("n_name", DataType::Utf8, false),
        ])),
        vec![Arc::new(Int64Array::from(vec![3, 4])), Arc::new(StringArray::from(vec!["FRANCE", "GERMANY"]))],
    );
    write_tpch_test_table(
        &root.join("lineitem.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_suppkey", DataType::Int64, false),
            Field::new("l_shipdate", DataType::Date32, false),
            Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
            Field::new("l_discount", DataType::Decimal128(15, 2), false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1000, 2000, 3000])),
            Arc::new(Int64Array::from(vec![100, 200, 200])),
            Arc::new(Date32Array::from(vec![date32(1995, 6, 1), date32(1996, 7, 1), date32(1995, 8, 1)])),
            Arc::new(
                Decimal128Array::from_iter_values([10_000, 20_000, 30_000]).with_precision_and_scale(15, 2).unwrap(),
            ),
            Arc::new(Decimal128Array::from_iter_values([10, 5, 0]).with_precision_and_scale(15, 2).unwrap()),
        ],
    );
}

fn write_q04_nonempty_tpch_parquet_dataset(root: &std::path::Path) {
    write_tiny_tpch_parquet_dataset(root).unwrap();
    write_tpch_test_table(
        &root.join("lineitem.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_partkey", DataType::Int64, false),
            Field::new("l_suppkey", DataType::Int64, false),
            Field::new("l_linenumber", DataType::Int32, false),
            Field::new("l_quantity", DataType::Decimal128(15, 2), false),
            Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
            Field::new("l_discount", DataType::Decimal128(15, 2), false),
            Field::new("l_tax", DataType::Decimal128(15, 2), false),
            Field::new("l_returnflag", DataType::Utf8, false),
            Field::new("l_linestatus", DataType::Utf8, false),
            Field::new("l_shipdate", DataType::Date32, false),
            Field::new("l_commitdate", DataType::Date32, false),
            Field::new("l_receiptdate", DataType::Date32, false),
            Field::new("l_shipinstruct", DataType::Utf8, false),
            Field::new("l_shipmode", DataType::Utf8, false),
            Field::new("l_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![1, 2, 2, 3])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 1])),
            Arc::new(
                Decimal128Array::from_iter_values([1_700, 3_600, 3_700, 800]).with_precision_and_scale(15, 2).unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([211_685, 459_831, 459_832, 133_096])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(Decimal128Array::from_iter_values([6, 9, 4, 2]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Decimal128Array::from_iter_values([2, 6, 1, 1]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(StringArray::from(vec!["N", "R", "R", "A"])),
            Arc::new(StringArray::from(vec!["O", "F", "F", "F"])),
            Arc::new(Date32Array::from(vec![
                date32(1994, 3, 13),
                date32(1995, 9, 18),
                date32(1995, 9, 19),
                date32(1998, 8, 2),
            ])),
            Arc::new(Date32Array::from(vec![
                date32(1994, 2, 12),
                date32(1995, 9, 10),
                date32(1995, 9, 11),
                date32(1998, 7, 20),
            ])),
            Arc::new(Date32Array::from(vec![
                date32(1994, 3, 22),
                date32(1995, 10, 1),
                date32(1995, 10, 2),
                date32(1998, 8, 15),
            ])),
            Arc::new(StringArray::from(vec!["DELIVER IN PERSON", "TAKE BACK RETURN", "TAKE BACK RETURN", "NONE"])),
            Arc::new(StringArray::from(vec!["TRUCK", "MAIL", "SHIP", "AIR"])),
            Arc::new(StringArray::from(vec![
                "lineitem one",
                "lineitem two",
                "lineitem two duplicate",
                "lineitem three",
            ])),
        ],
    );
    write_tpch_test_table(
        &root.join("orders.parquet"),
        Arc::new(Schema::new(vec![
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_custkey", DataType::Int64, false),
            Field::new("o_orderstatus", DataType::Utf8, false),
            Field::new("o_totalprice", DataType::Decimal128(15, 2), false),
            Field::new("o_orderdate", DataType::Date32, false),
            Field::new("o_orderpriority", DataType::Utf8, false),
            Field::new("o_clerk", DataType::Utf8, false),
            Field::new("o_shippriority", DataType::Int32, false),
            Field::new("o_comment", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["O", "F", "O"])),
            Arc::new(
                Decimal128Array::from_iter_values([17_279_949, 3_842_632, 20_565_430])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(Date32Array::from(vec![date32(1995, 3, 15), date32(1993, 8, 14), date32(1994, 7, 30)])),
            Arc::new(StringArray::from(vec!["1-URGENT", "2-HIGH", "5-LOW"])),
            Arc::new(StringArray::from(vec!["Clerk#000000001", "Clerk#000000002", "Clerk#000000003"])),
            Arc::new(Int32Array::from(vec![0, 0, 0])),
            Arc::new(StringArray::from(vec!["order one", "special requests", "order three"])),
        ],
    );
}

fn write_tpch_test_table(path: &std::path::Path, schema: Arc<Schema>, columns: Vec<ArrayRef>) {
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn assert_q7_nation_identity_rows(batches: &[RecordBatch]) {
    assert!(!batches.is_empty());
    assert_eq!(batches[0].schema().field(0).name(), "supp_nation");
    assert_eq!(batches[0].schema().field(1).name(), "cust_nation");
    assert_eq!(batches[0].schema().field(2).name(), "l_year");
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push((
                array_value(batch.column(0), row),
                array_value(batch.column(1), row),
                array_value(batch.column(2), row),
            ));
        }
    }
    rows.sort_unstable();
    assert_eq!(
        rows,
        vec![
            ("FRANCE".to_string(), "GERMANY".to_string(), "1995".to_string()),
            ("GERMANY".to_string(), "FRANCE".to_string(), "1996".to_string()),
        ]
    );
}
