use super::*;

#[tokio::test]
async fn native_datafusion_exec_q6_shaped_global_decimal_sum_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-q6-shaped");
    let path = dir.join("lineitem.parquet");
    write_q6_lineitem_parquet(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_Q6_SHAPED_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_Q6_SHAPED_QUERY, native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert!(metrics.sum_by_name("source_chunks_planned").unwrap().as_usize() > 0);
    assert_eq!(
        metrics.sum_by_name("source_chunks_planned").unwrap().as_usize(),
        metrics.sum_by_name("source_chunks_committed").unwrap().as_usize(),
    );
    assert!(metrics.sum_by_name("row_groups_planned").unwrap().as_usize() > 0);
    assert_eq!(
        metrics.sum_by_name("row_groups_planned").unwrap().as_usize(),
        metrics.sum_by_name("row_groups_committed").unwrap().as_usize(),
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_sql_count_only_uses_zero_column_count_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-count-only");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_COUNT_ONLY_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_COUNT_ONLY_QUERY, native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("read_columns").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_default_sql_uses_bounded_native_engine() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-exec");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_SUM_EXPRESSION_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_SUM_EXPRESSION_QUERY, native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("row_groups_selected").unwrap().as_usize(), 2);
    assert!(metrics.sum_by_name("row_groups_selected").unwrap().as_usize() < total_row_groups);
    assert_eq!(
        metrics.sum_by_name("source_chunks_planned").unwrap().as_usize(),
        metrics.sum_by_name("source_chunks_committed").unwrap().as_usize(),
    );
    assert_eq!(
        metrics.sum_by_name("row_groups_planned").unwrap().as_usize(),
        metrics.sum_by_name("row_groups_committed").unwrap().as_usize(),
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_default_sql_distributed_avg_uses_bounded_native_engine() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-avg");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_AVG_AGGREGATE_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_AVG_AGGREGATE_QUERY, native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("row_groups_selected").unwrap().as_usize(), 2);
    assert!(metrics.sum_by_name("row_groups_selected").unwrap().as_usize() < total_row_groups);
    assert_eq!(
        metrics.sum_by_name("source_chunks_planned").unwrap().as_usize(),
        metrics.sum_by_name("source_chunks_committed").unwrap().as_usize(),
    );
    assert_eq!(
        metrics.sum_by_name("row_groups_planned").unwrap().as_usize(),
        metrics.sum_by_name("row_groups_committed").unwrap().as_usize(),
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_nullable_avg_output_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-nullable-avg-sort");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);
    let cpu_batches = cpu_sql_output(&path, SQL_AVG_ORDER_BY_VALUE_QUERY).await;

    let plan = native_datafusion_sql_plan(&path, SQL_AVG_ORDER_BY_VALUE_QUERY, native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(
        metrics.sum_by_name("source_chunks_planned").unwrap().as_usize(),
        metrics.sum_by_name("source_chunks_committed").unwrap().as_usize(),
    );
    assert_eq!(
        metrics.sum_by_name("row_groups_planned").unwrap().as_usize(),
        metrics.sum_by_name("row_groups_committed").unwrap().as_usize(),
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_unsupported_sql_falls_back_before_native_execution() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-fallback");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let plan =
        native_datafusion_sql_plan(&path, SQL_UNSUPPORTED_APPROX_DISTINCT_QUERY, native_datafusion_config()).await;

    assert_ne!(plan.name(), "NativeCudfExec");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_none());
    assert!(plan.metrics().is_none_or(|metrics| metrics.sum_by_name("host_to_device_uploads").is_none()));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_datafusion_exec_resource_failure_preserves_adapter_facts() {
    let dir = unique_temp_dir("datafusion-cudf-native-datafusion-resource-failure");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let plan =
        native_datafusion_sql_plan(&path, SQL_SUM_EXPRESSION_QUERY, constrained_native_datafusion_config()).await;

    assert_eq!(plan.name(), "NativeCudfExec");
    let error = collect(plan, Arc::new(TaskContext::default())).await.unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.fact_value("native_code"), Some("memory_reservation_failed"));
    assert_eq!(adapter_error.fact_value("native_datafusion_phase"), Some("runtime_execution"));
    assert_eq!(adapter_error.fact_value("cpu_fallback_after_native_start"), Some("false"));
    assert!(adapter_error.fact_value("native_operation").is_some());
    assert!(adapter_error.fact_value("memory_boundary").is_some());
    assert!(adapter_error.fact_value("reservation_kind").is_some());

    fs::remove_dir_all(dir).unwrap();
}
