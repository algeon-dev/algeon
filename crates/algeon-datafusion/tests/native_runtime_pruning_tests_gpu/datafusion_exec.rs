use std::{fs, sync::Arc};

use cudf::testing::arrow::{
    arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
    reset_arrow_host_interop_observation_for_test,
};
use datafusion::physical_plan::metrics::MetricsSet;

use super::*;

fn assert_source_batch_metrics(metrics: &MetricsSet) {
    let planned = metric_usize(metrics, "source_chunks_planned");
    let committed = metric_usize(metrics, "source_chunks_committed");

    assert!(planned > 0);
    assert!(committed > 0);
    assert!(committed <= planned);
}

fn assert_row_group_coverage_metrics(metrics: &MetricsSet) {
    assert!(metric_usize(metrics, "row_groups_planned") > 0);
    assert_eq!(metric_usize(metrics, "row_groups_planned"), metric_usize(metrics, "row_groups_committed"));
}

#[tokio::test]
async fn native_datafusion_exec_sql_count_only_uses_zero_column_count_source() {
    let dir = unique_temp_dir("algeon-datafusion-native-datafusion-count-only");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_COUNT_ONLY_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_COUNT_ONLY_QUERY).await;

    assert_eq!(below_native_admission(&plan).name(), "AlgeonNativeExec");
    assert!(below_native_admission(&plan).downcast_ref::<AlgeonNativeExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    let metrics = below_native_admission(&plan).metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "read_columns"), 0);
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_default_sql_uses_bounded_native_engine() {
    let dir = unique_temp_dir("algeon-datafusion-native-datafusion-exec");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_SUM_EXPRESSION_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_SUM_EXPRESSION_QUERY).await;

    assert_eq!(below_native_admission(&plan).name(), "AlgeonNativeExec");
    assert!(below_native_admission(&plan).downcast_ref::<AlgeonNativeExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    let metrics = below_native_admission(&plan).metrics().expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(metric_usize(&metrics, "row_groups_selected"), 2);
    assert!(metric_usize(&metrics, "row_groups_selected") < total_row_groups);
    assert_source_batch_metrics(&metrics);
    assert_row_group_coverage_metrics(&metrics);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_unsupported_sql_retains_datafusion_baseline_before_native_execution() {
    let dir = unique_temp_dir("algeon-datafusion-native-datafusion-baseline");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let plan = native_datafusion_sql_plan(&path, SQL_UNSUPPORTED_APPROX_DISTINCT_QUERY).await;

    assert_ne!(below_native_admission(&plan).name(), "AlgeonNativeExec");
    assert!(below_native_admission(&plan).downcast_ref::<AlgeonNativeExec>().is_none());
    assert!(
        below_native_admission(&plan)
            .metrics()
            .is_none_or(|metrics| metrics.sum_by_name("host_to_device_uploads").is_none())
    );

    fs::remove_dir_all(dir).unwrap();
}
