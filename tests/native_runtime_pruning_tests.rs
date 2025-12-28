use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Date32Array, Decimal128Array, Int32Array, Int64Array, StringArray, TimestampNanosecondArray},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use cudf_query_engine::{ExecutionContext, MemoryControlPolicy, QueryOutput, execute_query_plan};
use datafusion::{
    common::DataFusionError,
    execution::{SessionStateBuilder, context::TaskContext},
    functions_aggregate::{count::count_udaf, sum::sum_udaf},
    logical_expr::Operator,
    parquet::{
        arrow::ArrowWriter,
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    },
    physical_expr::{
        LexOrdering, PhysicalExpr, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{CastExpr, NotExpr, binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect, displayable,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode, GpuFallbackPolicy, NativeCudfExec,
    NativeCudfExecutionConfig, TPCH_QUERIES, TPCH_TABLES, lower_physical_plan_to_native, tpch_report_session_config,
    write_tiny_tpch_parquet_dataset,
};
use tokio::sync::Mutex;

static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::const_new(());

#[tokio::test]
async fn native_runtime_default_sql_sum_expression_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-sum");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_min_max_count_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-min-max-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_MIN_MAX_COUNT_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_count_star_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-count-star");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_COUNT_STAR_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-default-avg");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_AGGREGATE_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_nullable_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-nullable-avg");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_uneven_row_group_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-uneven-avg");
    let path = dir.join("input.parquet");
    write_uneven_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_sort_null_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-sort-nulls");
    let path = dir.join("input.parquet");
    write_nullable_sort_input_parquet(&path);

    for sql in [
        SQL_SORT_NULLS_ASC_FIRST_QUERY,
        SQL_SORT_NULLS_ASC_LAST_QUERY,
        SQL_SORT_NULLS_DESC_FIRST_QUERY,
        SQL_SORT_NULLS_DESC_LAST_QUERY,
    ] {
        assert_default_sql_native_parity(&path, sql, None).await;
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_bounded_topk_null_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-topk-nulls");
    let path = dir.join("input.parquet");
    write_nullable_sort_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_SORT_NULLS_TOPK_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_nullable_avg_output_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-nullable-avg-sort");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_ORDER_BY_VALUE_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_q6_shaped_global_decimal_sum_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-q6-shaped");
    let path = dir.join("lineitem.parquet");
    write_q6_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_Q6_SHAPED_QUERY).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 4);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));
    let revenue = decimal128_array(native_batch, 0);
    assert_eq!(revenue.value(0), 260_000);
    assert_eq!(revenue.scale(), 4);

    fs::remove_dir_all(dir).unwrap();
}

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
async fn native_runtime_global_decimal_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-global-decimal-avg");
    let path = dir.join("lineitem.parquet");
    write_q6_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let native_physical_plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q6_tiny_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q6-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    assert_tpch_q6_native_parity(&dir.join("lineitem.parquet")).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q1_tiny_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q1-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q1 = TPCH_QUERIES.iter().find(|query| query.id == "q01").expect("TPC-H Q1 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q1.sql).await;
    assert!(
        cpu_batches.iter().map(|batch| batch.num_rows()).sum::<usize>() > 0,
        "tiny TPC-H Q1 should exercise non-empty output parity"
    );

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q1.sql, native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q1 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q3_tiny_join_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q3-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q3 = TPCH_QUERIES.iter().find(|query| query.id == "q03").expect("TPC-H Q3 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q3.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q3.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q3 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q2_tiny_root_topk_join_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q2-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q2 = TPCH_QUERIES.iter().find(|query| query.id == "q02").expect("TPC-H Q2 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q2.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q2.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q2 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q19_tiny_join_residual_in_list_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q19-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q19 = TPCH_QUERIES.iter().find(|query| query.id == "q19").expect("TPC-H Q19 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q19.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q19.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q19 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q15_tiny_root_join_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q15-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q15 = TPCH_QUERIES.iter().find(|query| query.id == "q15").expect("TPC-H Q15 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q15.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q15.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q15 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q16_tiny_distinct_anti_join_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q16-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q16 = TPCH_QUERIES.iter().find(|query| query.id == "q16").expect("TPC-H Q16 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q16.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q16.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q16 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q11_tiny_nested_loop_having_subquery_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q11-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q11.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q11.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q11 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q11_tiny_bounded_aggregate_over_join_subplan_inputs_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q11-tiny-bounded");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q11.sql).await;
    assert_eq!(record_batch_row_count(&cpu_batches), 1, "tiny TPC-H Q11 CPU output should be non-empty");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q11.sql, native_datafusion_config()).await;
    assert_native_cudf_exec(&native_plan, "bounded TPC-H Q11 should be replaced by NativeCudfExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let metrics = native_plan.metrics().expect("NativeCudfExec should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("reservation_denied_count").unwrap().as_usize(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q11_tiny_bounded_budget_failure_preserves_reservation_facts() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q11-tiny-budget-failure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q11.sql, constrained_native_datafusion_config()).await;
    assert_native_cudf_exec(
        &native_plan,
        "bounded TPC-H Q11 should be admitted before the constrained runtime budget fails",
    );

    let error = collect(Arc::clone(&native_plan), Arc::new(TaskContext::default())).await.unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::MemoryReservationFailed);
    assert_eq!(adapter_error.fact_value("native_code"), Some("memory_reservation_failed"));
    assert_eq!(adapter_error.fact_value("native_datafusion_phase"), Some("runtime_execution"));
    assert_eq!(
        adapter_error.fact_value("bounded_join_execution_family"),
        Some("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
    );
    for fact in [
        "memory_boundary",
        "reservation_kind",
        "requested_bytes",
        "granted_bytes",
        "budget_bytes",
        "node_id",
        "stage",
        "attempt",
        "max_attempts",
    ] {
        assert!(adapter_error.fact_value(fact).is_some(), "missing Q11 reservation failure fact {fact}");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q17_tiny_decimal_residual_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q17-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q17 = TPCH_QUERIES.iter().find(|query| query.id == "q17").expect("TPC-H Q17 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q17.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q17.sql, unbounded_native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q17 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_tpch_q22_tiny_bounded_scalar_subquery_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-tpch-q22-tiny");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q22 = TPCH_QUERIES.iter().find(|query| query.id == "q22").expect("TPC-H Q22 query must exist");
    let cpu_batches = cpu_tpch_sql_output(&dir, q22.sql).await;

    let native_plan = native_datafusion_tpch_sql_plan(&dir, q22.sql, native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q22 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
#[ignore = "requires local TPC-H SF1 Parquet data"]
async fn native_runtime_tpch_q6_sf1_matches_cpu() {
    let root = std::env::var("CUDF_TPCH_PARQUET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("tests/data/tpch/sf1/parquet"));
    assert_tpch_q6_native_parity(&root.join("lineitem.parquet")).await;
}

#[tokio::test]
async fn native_runtime_zero_column_count_matches_cpu_without_parquet_data_columns() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-zero-column-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let native_physical_plan = count_only_sort_limit_plan(parquet_scan(&path).await);
    let cpu_batches =
        collect(Arc::clone(&native_physical_plan), Arc::new(datafusion::execution::context::TaskContext::default()))
            .await
            .unwrap();
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 0);
    assert_eq!(lowering.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.read_columns, 0);
    assert_eq!(metrics.input_rows, 8);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

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
async fn native_runtime_default_sql_not_filter_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-not-filter");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan(&path).await;
    let scan_schema = scan.schema();
    let payload_lt_10 = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::Lt,
        lit(datafusion::common::ScalarValue::Int32(Some(10))),
        &scan_schema,
    )
    .unwrap();
    let predicate = Arc::new(NotExpr::new(payload_lt_10));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);
    assert_native_physical_plan_parity(plan, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_widening_numeric_cast_matches_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-cast");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan(&path).await;
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(datafusion::common::ScalarValue::Int32(Some(10))),
        &scan_schema,
    )
    .unwrap();
    let cast_payload = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int64, None));
    let x00 = binary(cast_payload, Operator::Plus, col("m00", &scan_schema).unwrap(), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_projected_x00(scan, predicate, x00);
    assert_native_physical_plan_parity(plan, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_phase2_scalar_expressions_match_cpu_and_collect_required_columns() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-phase2-scalars");
    let path = dir.join("input.parquet");
    write_phase2_scalar_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].pure_filter_column_count, 0);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_phase2_nullable_scalar_expressions_match_cpu() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-phase2-nullable-scalars");
    let path = dir.join("input.parquet");
    write_nullable_phase2_scalar_input_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let native_physical_plan =
        sql_plan(&path, SessionConfig::new(), SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY).await;
    let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}\n{plan_display}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}\n{plan_display}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].projected_output_column_count, 3);
    assert_eq!(lowering.source_diagnostics[0].pure_filter_column_count, 0);

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_phase2_null_semantics_match_cpu_by_expression_family() {
    let cases = [
        ("like", SQL_PHASE2_NULL_SEMANTICS_LIKE_QUERY),
        ("substring", SQL_PHASE2_NULL_SEMANTICS_SUBSTRING_QUERY),
        ("in-list", SQL_PHASE2_NULL_SEMANTICS_IN_LIST_QUERY),
        ("case", SQL_PHASE2_NULL_SEMANTICS_CASE_QUERY),
        ("case-no-else", SQL_PHASE2_NULL_SEMANTICS_CASE_NO_ELSE_QUERY),
        ("extract", SQL_PHASE2_NULL_SEMANTICS_EXTRACT_QUERY),
        ("interval", SQL_PHASE2_NULL_SEMANTICS_INTERVAL_QUERY),
        ("modulo", SQL_PHASE2_NULL_SEMANTICS_MODULO_QUERY),
        ("is-null", SQL_PHASE2_NULL_SEMANTICS_IS_NULL_QUERY),
    ];

    for (name, sql) in cases {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-runtime-phase2-null-semantics-{name}").as_str());
        let path = dir.join("input.parquet");
        write_nullable_phase2_scalar_input_parquet(&path);

        let cpu_batches = cpu_sql_output(&path, sql).await;
        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let plan_display = displayable(native_physical_plan.as_ref()).indent(true).to_string();
        let lowering = lower_physical_plan_to_native(native_physical_plan);
        assert_eq!(lowering.rewritten_count(), 1, "{name} outcomes={:?}\n{plan_display}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 0, "{name} outcomes={:?}\n{plan_display}", lowering.report.outcomes());

        let context = bounded_execution_context();
        let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
        cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
        let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
        assert_eq!(result.metrics().boundary.host_to_device_uploads, 0, "{name}");
        assert_eq!(result.metrics().boundary.device_to_host_materializations, 1, "{name}");

        let QueryOutput::HostArrow(native_batch) = result.output() else {
            panic!("{name} native host sink should return a RecordBatch");
        };
        assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_invalid_phase2_scalar_expressions_reject_before_native_execution() {
    for (name, sql) in [
        ("substring-zero-start", SQL_INVALID_SUBSTRING_ZERO_START_QUERY),
        ("substring-dynamic-start", SQL_INVALID_SUBSTRING_DYNAMIC_START_QUERY),
        ("substring-dynamic-length", SQL_INVALID_SUBSTRING_DYNAMIC_LENGTH_QUERY),
        ("substring-negative-length", SQL_INVALID_SUBSTRING_NEGATIVE_LENGTH_QUERY),
        ("substring-stop-overflow", SQL_INVALID_SUBSTRING_STOP_OVERFLOW_QUERY),
    ] {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-runtime-invalid-{name}").as_str());
        let path = dir.join("input.parquet");
        write_phase2_scalar_input_parquet(&path);

        let native_physical_plan = sql_plan(&path, SessionConfig::new(), sql).await;
        let lowering = lower_physical_plan_to_native(native_physical_plan);

        assert_eq!(lowering.rewritten_count(), 0, "{name} should not rewrite: {:?}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 1, "{name} should reject: {:?}", lowering.report.outcomes());
        assert!(
            matches!(
                lowering.report.outcomes()[0],
                datafusion_cudf::CandidateOutcome::Rejected {
                    reason: datafusion_cudf::FallbackReason::UnsupportedExpression,
                    ..
                }
            ),
            "{name} outcomes={:?}",
            lowering.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test]
async fn native_runtime_substring_max_start_length_one_is_in_contract() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-substring-max-start-boundary");
    let path = dir.join("input.parquet");
    write_phase2_scalar_input_parquet(&path);

    let native_physical_plan = sql_plan(&path, SessionConfig::new(), SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);

    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate() {
    let dir = unique_temp_dir("datafusion-cudf-native-runtime-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, Some((2, total_row_groups))).await;

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

    let plan = native_datafusion_sql_plan(&path, SQL_FILTERED_GLOBAL_COUNT_QUERY, native_datafusion_config()).await;

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

async fn assert_default_sql_native_parity(
    path: &std::path::Path,
    sql: &str,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches = cpu_sql_output(path, sql).await;
    let native_physical_plan = sql_plan(path, SessionConfig::new(), sql).await;
    let lowering = lower_physical_plan_to_native(native_physical_plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let context = bounded_execution_context();
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();

    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(total_row_groups, 4);
        assert_eq!(metrics.row_groups_selected, selected_row_groups as u64);
        assert!(metrics.row_groups_selected < total_row_groups as u64);
    }
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.host_to_device_uploaded_rows, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.source_chunks_planned, metrics.source_chunks_committed);
    assert!(metrics.source_chunks_committed > 0);
    assert_eq!(metrics.row_groups_planned, metrics.row_groups_committed);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));
}

async fn assert_native_physical_plan_parity(
    plan: Arc<dyn ExecutionPlan>,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches =
        collect(Arc::clone(&plan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let lowering = lower_physical_plan_to_native(plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(runtime);
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();

    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.host_to_device_uploaded_rows, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));
}

async fn cpu_sql_output(path: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = sql_plan(path, SessionConfig::new(), sql).await;
    collect(plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap()
}

async fn cpu_lineitem_sql_output(path: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = lineitem_sql_plan(path, direct_friendly_sql_session_config(), sql).await;
    collect(plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap()
}

async fn cpu_tpch_sql_output(root: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = tpch_sql_plan(root, tpch_report_session_config(), sql).await;
    collect(plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap()
}

async fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df = ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn lineitem_sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("lineitem", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn tpch_sql_plan(root: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    register_tiny_tpch_tables(&ctx, root).await;
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
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

async fn native_datafusion_sql_plan(
    path: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features().with_cudf_gpu_islands(config).build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn native_datafusion_lineitem_sql_plan(
    path: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_gpu_islands(config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("lineitem", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn native_datafusion_tpch_sql_plan(
    root: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_gpu_islands(config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tiny_tpch_tables(&ctx, root).await;
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

async fn register_tiny_tpch_tables(ctx: &SessionContext, root: &std::path::Path) {
    for table in TPCH_TABLES {
        let path = root.join(format!("{table}.parquet"));
        ctx.register_parquet(*table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
}

fn native_datafusion_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(512 * 1024 * 1024, 256 * 1024 * 1024, 1, 3, 1)
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

fn unbounded_native_datafusion_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::unbounded().with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

fn constrained_native_datafusion_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(1, 1, 1, 1, 1).with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

fn bounded_execution_context() -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 256 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

fn adapter_error(error: &DataFusionError) -> &Error {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source.downcast_ref::<Error>().expect("DataFusion external error should preserve datafusion_cudf::Error")
}

fn assert_native_cudf_exec(plan: &Arc<dyn ExecutionPlan>, context: &str) {
    assert_eq!(plan.name(), "NativeCudfExec", "{context}");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some(), "{context}");
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

    let native_plan = native_datafusion_lineitem_sql_plan(lineitem_path, q6.sql, native_datafusion_config()).await;
    assert!(
        displayable(native_plan.as_ref()).indent(false).to_string().contains("NativeCudfExec"),
        "TPC-H Q6 should be replaced by NativeCudfExec"
    );

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);
    assert_batches_equal(&cpu_batches, &native_batches);
}

const SQL_SUM_EXPRESSION_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_MIN_MAX_COUNT_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

const SQL_COUNT_STAR_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00, COUNT(*) AS count_rows
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

const SQL_COUNT_ONLY_QUERY: &str = "
SELECT COUNT(*) AS count_rows
FROM bench
";

const SQL_AVG_AGGREGATE_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 3
";

const SQL_AVG_BY_KEY_NO_FILTER_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
GROUP BY key
ORDER BY key ASC
LIMIT 3
";

const SQL_AVG_ORDER_BY_VALUE_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 3
";

const SQL_FILTERED_GLOBAL_COUNT_QUERY: &str = "
SELECT COUNT(*) FILTER (WHERE payload >= 10) AS count_rows
FROM bench
";

const SQL_Q6_SHAPED_QUERY: &str = "
SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM bench
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
";

const SQL_GLOBAL_DECIMAL_AVG_QUERY: &str = "
SELECT
    AVG(l_discount) AS avg_discount
FROM bench
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
";

const SQL_PHASE2_SCALAR_EXPRESSIONS_QUERY: &str = "
SELECT status_prefix, ship_year, ship_month, status_label, SUM(order_mod) AS sum_mod
FROM (
    SELECT
        SUBSTRING(status FROM 1 FOR 1) AS status_prefix,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        shipdate + INTERVAL '1' MONTH AS ship_month,
        CASE WHEN status IN ('O', 'P') THEN 'open' ELSE 'other' END AS status_label,
        orderkey % 2 AS order_mod
    FROM bench
    WHERE status LIKE 'O%'
      AND status IS NOT NULL
      AND orderkey IN (1, 3)
      AND orderkey % 2 = 1
) phase2
GROUP BY status_prefix, ship_year, ship_month, status_label
ORDER BY status_prefix ASC, ship_year ASC, ship_month ASC, status_label ASC
LIMIT 10
";

const SQL_PHASE2_NULLABLE_SCALAR_EXPRESSIONS_QUERY: &str = "
SELECT
    status_prefix,
    ship_year,
    ship_month,
    status_label,
    not_in_label,
    like_match,
    not_like_match,
    status_is_null,
    status_is_not_null,
    SUM(order_mod) AS sum_mod
FROM (
    SELECT
        SUBSTRING(status FROM 1 FOR 1) AS status_prefix,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        shipdate + INTERVAL '1' MONTH AS ship_month,
        CASE
            WHEN status IS NULL THEN 'missing'
            WHEN status IN ('O', 'P') THEN 'listed'
            WHEN status NOT LIKE 'O%' THEN 'not_open'
            ELSE 'other'
        END AS status_label,
        CASE WHEN status NOT IN ('F') THEN 'not_f' ELSE 'f_or_null' END AS not_in_label,
        status LIKE 'O%' AS like_match,
        status NOT LIKE 'O%' AS not_like_match,
        status IS NULL AS status_is_null,
        status IS NOT NULL AS status_is_not_null,
        orderkey % 2 AS order_mod
    FROM bench
) phase2
GROUP BY
    status_prefix,
    ship_year,
    ship_month,
    status_label,
    not_in_label,
    like_match,
    not_like_match,
    status_is_null,
    status_is_not_null
ORDER BY
    status_is_null ASC,
    status_prefix ASC NULLS FIRST,
    ship_year ASC NULLS FIRST,
    ship_month ASC NULLS FIRST,
    status_label ASC,
    not_in_label ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_LIKE_QUERY: &str = "
SELECT row_id, like_match, not_like_match, COUNT(*) AS row_count
FROM (
    SELECT row_id, status LIKE 'O%' AS like_match, status NOT LIKE 'O%' AS not_like_match
    FROM bench
) null_semantics
GROUP BY row_id, like_match, not_like_match
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_SUBSTRING_QUERY: &str = "
SELECT row_id, status_prefix, COUNT(*) AS row_count
FROM (
    SELECT row_id, SUBSTRING(status FROM 1 FOR 1) AS status_prefix
    FROM bench
) null_semantics
GROUP BY row_id, status_prefix
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_IN_LIST_QUERY: &str = "
SELECT row_id, status_in_list, status_not_in_list, COUNT(*) AS row_count
FROM (
    SELECT row_id, status IN ('O', 'P') AS status_in_list, status NOT IN ('F') AS status_not_in_list
    FROM bench
) null_semantics
GROUP BY row_id, status_in_list, status_not_in_list
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_CASE_QUERY: &str = "
SELECT row_id, status_label, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        CASE
            WHEN status IS NULL THEN 'missing'
            WHEN status IN ('O', 'P') THEN 'listed'
            ELSE 'other'
        END AS status_label
    FROM bench
) null_semantics
GROUP BY row_id, status_label
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_CASE_NO_ELSE_QUERY: &str = "
SELECT row_id, open_code, COUNT(*) AS row_count
FROM (
    SELECT row_id, CASE WHEN status = 'O' THEN 1 END AS open_code
    FROM bench
) null_semantics
GROUP BY row_id, open_code
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_EXTRACT_QUERY: &str = "
SELECT row_id, ship_year, ship_month, ship_day, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        EXTRACT(YEAR FROM shipdate) AS ship_year,
        EXTRACT(MONTH FROM shipdate) AS ship_month,
        EXTRACT(DAY FROM shipdate) AS ship_day
    FROM bench
) null_semantics
GROUP BY row_id, ship_year, ship_month, ship_day
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_INTERVAL_QUERY: &str = "
SELECT row_id, shipdate_plus_month, shipdate_minus_day, event_ts_plus_second, event_ts_minus_second, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        shipdate + INTERVAL '1' MONTH AS shipdate_plus_month,
        shipdate - INTERVAL '1' DAY AS shipdate_minus_day,
        event_ts + INTERVAL '1' SECOND AS event_ts_plus_second,
        event_ts - INTERVAL '1' SECOND AS event_ts_minus_second
    FROM bench
) null_semantics
GROUP BY row_id, shipdate_plus_month, shipdate_minus_day, event_ts_plus_second, event_ts_minus_second
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_MODULO_QUERY: &str = "
SELECT row_id, order_mod, COUNT(*) AS row_count
FROM (
    SELECT row_id, orderkey % 2 AS order_mod
    FROM bench
) null_semantics
GROUP BY row_id, order_mod
ORDER BY row_id ASC
LIMIT 20
";

const SQL_PHASE2_NULL_SEMANTICS_IS_NULL_QUERY: &str = "
SELECT row_id, status_is_null, status_is_not_null, shipdate_is_null, COUNT(*) AS row_count
FROM (
    SELECT
        row_id,
        status IS NULL AS status_is_null,
        status IS NOT NULL AS status_is_not_null,
        shipdate IS NULL AS shipdate_is_null
    FROM bench
) null_semantics
GROUP BY row_id, status_is_null, status_is_not_null, shipdate_is_null
ORDER BY row_id ASC
LIMIT 20
";

const SQL_INVALID_SUBSTRING_ZERO_START_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 0 FOR 1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_INVALID_SUBSTRING_DYNAMIC_START_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM orderkey FOR 1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_INVALID_SUBSTRING_DYNAMIC_LENGTH_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 1 FOR orderkey) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_INVALID_SUBSTRING_NEGATIVE_LENGTH_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 1 FOR -1) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_SUBSTRING_MAX_START_LENGTH_ONE_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 2147483647 FOR 1) AS status_prefix
    FROM bench
) boundary_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_INVALID_SUBSTRING_STOP_OVERFLOW_QUERY: &str = "
SELECT status_prefix, COUNT(*) AS count_rows
FROM (
    SELECT SUBSTRING(status FROM 2147483647 FOR 2) AS status_prefix
    FROM bench
) invalid_substring
GROUP BY status_prefix
ORDER BY status_prefix ASC
LIMIT 10
";

const SQL_SORT_NULLS_ASC_FIRST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS FIRST, tie_key DESC
LIMIT 8
";

const SQL_SORT_NULLS_ASC_LAST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS LAST, tie_key DESC
LIMIT 8
";

const SQL_SORT_NULLS_DESC_FIRST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 DESC NULLS FIRST, tie_key DESC
LIMIT 8
";

const SQL_SORT_NULLS_DESC_LAST_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 DESC NULLS LAST, tie_key DESC
LIMIT 8
";

const SQL_SORT_NULLS_TOPK_QUERY: &str = "
SELECT group_key, tie_key, AVG(m00) AS avg_m00
FROM bench
GROUP BY group_key, tie_key
ORDER BY avg_m00 ASC NULLS LAST, tie_key DESC
LIMIT 4
";

fn write_row_group_pruning_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 10, 11, 20, 21])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_phase2_scalar_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322, 18_353])),
            Arc::new(StringArray::from(vec!["O", "F", "O", "P"])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_q6_lineitem_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_quantity", DataType::Decimal128(15, 2), false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                Decimal128Array::from_iter_values([2_300, 2_500, 1_000, 1_200, 1_200, 1_200, 1_200])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(Decimal128Array::from_iter_values([5, 6, 7, 6, 4, 8, 6]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Date32Array::from(vec![
                date32(1994, 1, 15),
                date32(1994, 6, 1),
                date32(1994, 9, 30),
                date32(1995, 1, 1),
                date32(1994, 3, 15),
                date32(1994, 11, 15),
                date32(1993, 12, 31),
            ])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_nullable_phase2_scalar_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("orderkey", DataType::Int64, true),
        Field::new("shipdate", DataType::Date32, true),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("status", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(4), Some(5), None])),
            Arc::new(Date32Array::from(vec![Some(18_262), None, Some(18_322), Some(18_353), None, Some(18_414)])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(100_000_000),
                None,
                Some(300_000_000),
                Some(400_000_000),
                None,
                Some(600_000_000),
            ])),
            Arc::new(StringArray::from(vec![Some("O"), Some("F"), None, Some("P"), Some("OX"), Some("R")])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_nullable_sort_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("tie_key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 30, 20, 10, 50, 60])),
            Arc::new(Int32Array::from(vec![3, 1, 2, 1, 3, 2, 2, 1, 1])),
            Arc::new(Int64Array::from(vec![
                None,
                Some(20),
                Some(12),
                None,
                Some(13),
                Some(22),
                None,
                Some(31),
                Some(11),
            ])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_nullable_avg_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![None, None, Some(10), None, Some(30)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_uneven_avg_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 1, 1, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30, 300])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

fn assert_batches_equal(left: &[RecordBatch], right: &[RecordBatch]) {
    assert!(!left.is_empty());
    assert!(!right.is_empty());
    assert_schema_names_and_types_equal(&left[0], &right[0]);
    assert_eq!(batch_rows(left), batch_rows(right));
}

fn assert_batches_equal_allow_empty(left: &[RecordBatch], right: &[RecordBatch]) {
    if left.is_empty() {
        assert_eq!(
            right.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            0,
            "native output should have zero rows when CPU output is empty"
        );
    } else {
        assert_batches_equal(left, right);
    }
}

fn decimal128_array(batch: &RecordBatch, index: usize) -> &Decimal128Array {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array")
}

fn assert_schema_names_and_types_equal(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_columns(), right.num_columns());
    for index in 0..left.num_columns() {
        assert_eq!(left.schema().field(index).name(), right.schema().field(index).name());
        assert_eq!(left.schema().field(index).data_type(), right.schema().field(index).data_type());
    }
}

fn batch_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());
            for column in 0..batch.num_columns() {
                values.push(array_value(batch.column(column), row));
            }
            rows.push(values);
        }
    }
    rows
}

fn array_value(array: &ArrayRef, row: usize) -> String {
    array_value_to_string(array.as_ref(), row).unwrap()
}

fn parquet_row_group_count(path: &std::path::Path) -> usize {
    SerializedFileReader::try_from(path).unwrap().metadata().num_row_groups()
}

fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let x00 = binary(
        col("m00", &filter_schema).unwrap(),
        projection_operator,
        col("m01", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn count_only_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(datafusion::common::ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&scan_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(count)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("count_rows", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}

fn aggregate_sort_limit_plan_with_projected_x00(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    x00: Arc<dyn PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
