use std::{fs, sync::Arc};

use arrow::{
    array::{ArrayRef, Decimal128Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use datafusion::{
    datasource::{
        physical_plan::{
            FileGroup, FileScanConfigBuilder, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
        source::DataSourceExec,
    },
    execution::{SessionStateBuilder, context::TaskContext},
    functions_aggregate::{count::count_udaf, sum::sum_udaf},
    logical_expr::Operator,
    parquet::file::reader::{FileReader, SerializedFileReader},
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
use datafusion_nexus::{
    backend::{GpuDeviceMemoryLimit, GpuMemoryOwnership, NexusGpuBackend, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    native::lower_physical_plan_to_native_for_test,
    native_exec::{NexusNativeExec, NexusNativeExecutionConfig},
};
use nexus_query_engine::{
    admission::QueryPlanCapabilityProof,
    exec::{execute_query_plan, output::QueryOutput},
    plan::{PlanNode, QueryPlan},
    runtime::{AdmittedQueryContext, ExecutionContext, QueryHandle},
    source::SourceNode,
};
use nexus_workloads::tpch::{TPCH_QUERIES, TPCH_TABLES, write_tiny_tpch_parquet_dataset};
use tokio::sync::Mutex;

static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::const_new(());

#[path = "support/tpch_session.rs"]
mod tpch_session;

#[path = "native_runtime_pruning_tests/datafusion_exec.rs"]
mod datafusion_exec;
#[path = "native_runtime_pruning_tests/fixtures.rs"]
mod fixtures;
#[path = "native_runtime_pruning_tests/scalar_expressions.rs"]
mod scalar_expressions;
#[path = "native_runtime_pruning_tests/sql.rs"]
mod sql;
#[path = "native_runtime_pruning_tests/support.rs"]
mod support;
#[path = "native_runtime_pruning_tests/tpch.rs"]
mod tpch;

use fixtures::*;
use sql::*;
use support::*;
use tpch_session::tpch_report_session_config;

#[tokio::test]
async fn native_runtime_default_sql_sum_expression_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-default-sum");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_min_max_count_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-default-min-max-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_MIN_MAX_COUNT_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_count_star_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-default-count-star");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_COUNT_STAR_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-default-avg");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_AGGREGATE_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_default_sql_distributed_filtered_aggregate_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-default-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_FILTERED_AGGREGATE_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_mixed_filtered_and_unfiltered_aggregates_match_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_sql_native_parity(&path, direct_friendly_sql_session_config(), SQL_MIXED_FILTERED_AGGREGATES_QUERY, None)
        .await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_median_aggregate_matches_cpu_single_stage_bounded_without_partials() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-median-aggregate-bounded");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_MEDIAN_AGGREGATE_QUERY).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_MEDIAN_AGGREGATE_QUERY).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());

    let context = bounded_execution_context(&lowering.plans[0]);
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.partial_aggregate_count, 0, "median must stay single-stage in bounded execution");
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_reader_pushdown_reduces_source_rows_and_retained_bytes() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-reader-pushdown-selective");
    let path = dir.join("input.parquet");
    write_reader_pushdown_selective_parquet_file(&path);

    let mut post_scan_config = SessionConfig::new();
    post_scan_config.options_mut().execution.parquet.pushdown_filters = false;
    let post_scan = execute_sql_native_query(&path, post_scan_config, SQL_SELECTIVE_READER_PUSHDOWN_QUERY).await;
    let mut reader_pushdown_config = SessionConfig::new();
    reader_pushdown_config.options_mut().execution.parquet.pushdown_filters = true;
    let reader_pushdown =
        execute_sql_native_query(&path, reader_pushdown_config, SQL_SELECTIVE_READER_PUSHDOWN_QUERY).await;

    assert_batches_equal(&post_scan.output, &reader_pushdown.output);
    assert_eq!(post_scan.predicate_disposition, "pruning_only");
    assert_eq!(reader_pushdown.predicate_disposition, "reader_pushdown");
    assert!(
        reader_pushdown.input_rows < post_scan.input_rows,
        "reader pushdown should reduce source rows: reader={} post_scan={}",
        reader_pushdown.input_rows,
        post_scan.input_rows
    );
    assert!(
        reader_pushdown.max_task_retained_bytes < post_scan.max_task_retained_bytes,
        "reader pushdown should reduce retained source bytes: reader={} post_scan={}",
        reader_pushdown.max_task_retained_bytes,
        post_scan.max_task_retained_bytes
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_decimal32_reader_pushdown_matches_post_scan() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-decimal32-reader-pushdown");
    let path = dir.join("input.parquet");
    write_decimal32_reader_pushdown_parquet_file(&path);

    let mut post_scan_config = SessionConfig::new();
    post_scan_config.options_mut().execution.parquet.pushdown_filters = false;
    let post_scan = execute_sql_native_query(&path, post_scan_config, SQL_DECIMAL32_READER_PUSHDOWN_QUERY).await;
    let mut reader_pushdown_config = SessionConfig::new();
    reader_pushdown_config.options_mut().execution.parquet.pushdown_filters = true;
    let reader_pushdown =
        execute_sql_native_query(&path, reader_pushdown_config, SQL_DECIMAL32_READER_PUSHDOWN_QUERY).await;

    assert_batches_equal(&post_scan.output, &reader_pushdown.output);
    assert_eq!(post_scan.predicate_disposition, "pruning_only");
    assert_eq!(reader_pushdown.predicate_disposition, "reader_pushdown");
    assert!(
        reader_pushdown.input_rows < post_scan.input_rows,
        "Decimal32-backed reader pushdown should reduce source rows: reader={} post_scan={}",
        reader_pushdown.input_rows,
        post_scan.input_rows
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_decimal32_equality_reader_pushdown_matches_post_scan() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-decimal32-equality-reader-pushdown");
    let path = dir.join("input.parquet");
    write_decimal32_reader_pushdown_parquet_file(&path);

    let mut post_scan_config = SessionConfig::new();
    post_scan_config.options_mut().execution.parquet.pushdown_filters = false;
    let post_scan =
        execute_sql_native_query(&path, post_scan_config, SQL_DECIMAL32_EQUALITY_READER_PUSHDOWN_QUERY).await;
    let mut reader_pushdown_config = SessionConfig::new();
    reader_pushdown_config.options_mut().execution.parquet.pushdown_filters = true;
    let reader_pushdown =
        execute_sql_native_query(&path, reader_pushdown_config, SQL_DECIMAL32_EQUALITY_READER_PUSHDOWN_QUERY).await;

    assert_batches_equal(&post_scan.output, &reader_pushdown.output);
    assert_eq!(post_scan.predicate_disposition, "pruning_only");
    assert_eq!(reader_pushdown.predicate_disposition, "reader_pushdown");
    assert!(
        reader_pushdown.input_rows < post_scan.input_rows,
        "Decimal32-backed equality reader pushdown should reduce source rows: reader={} post_scan={}",
        reader_pushdown.input_rows,
        post_scan.input_rows
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_pushed_predicate_source_limit_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-pushed-predicate-source-limit");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan_with_source_limit(
        parquet_scan_with_pushed_down_payload_predicate(parquet_scan(&path).await, 10),
        3,
    );
    let cpu_batches =
        collect(Arc::clone(&scan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let lowering = lower_physical_plan_to_native_for_test(scan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(lowering.source_diagnostics[0].rejected_features.is_empty());
    let source = lowering.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) | SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .expect("lowered plan should contain a local parquet source");
    assert_eq!(source.scan_limit, None, "scan limit must be applied after the pushed predicate");
    assert!(source.reader_predicate.is_some());

    let context = bounded_execution_context(&lowering.plans[0]);
    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    cudf::testing::arrow::reset_arrow_host_interop_observation_for_test();
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_empty_local_parquet_aggregates_match_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-empty-local-aggregates");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let empty_scan = parquet_scan_without_files(parquet_scan(&path).await);
    assert_empty_local_parquet_aggregate_parity(grouped_sum_aggregate_plan(Arc::clone(&empty_scan))).await;
    assert_empty_local_parquet_aggregate_parity(global_sum_aggregate_plan(empty_scan)).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_physically_empty_parquet_files_match_cpu_without_replay() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-physical-empty-parquet");
    let zero_row_file = dir.join("zero-rows.parquet");
    let zero_row_group_file = dir.join("zero-row-group.parquet");
    write_physically_empty_parquet_file(&zero_row_file);
    write_zero_row_group_parquet_file(&zero_row_group_file);

    assert_eq!(parquet_row_group_count(&zero_row_file), 0, "empty file fixture must have no row groups");
    assert_eq!(parquet_row_group_count(&zero_row_group_file), 1, "zero-row-group fixture must retain one row group");

    for path in [&zero_row_file, &zero_row_group_file] {
        for sql in [SQL_COUNT_ONLY_QUERY, SQL_COUNT_STAR_QUERY] {
            let cpu_batches = cpu_sql_output(path, sql).await;
            let native_plan = native_datafusion_sql_plan(path, sql).await;
            assert_nexus_native_exec(&native_plan, "physical empty parquet aggregate should lower to NexusNativeExec");
            let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
            assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_filtered_empty_rollup_grand_total_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-filtered-empty-rollup");
    let path = dir.join("bench.parquet");
    write_filtered_empty_rollup_parquet_file(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_FILTERED_EMPTY_ROLLUP_SUM_QUERY).await;
    assert_eq!(
        cpu_batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1,
        "CPU DataFusion must emit the ROLLUP grand-total row over filtered-empty input"
    );

    let native_physical_plan = native_datafusion_sql_plan(&path, SQL_FILTERED_EMPTY_ROLLUP_SUM_QUERY).await;
    assert_nexus_native_exec(&native_physical_plan, "runtime-filtered empty ROLLUP should lower to NexusNativeExec");

    let task_ctx = Arc::new(TaskContext::default());
    let native_batches = collect(native_physical_plan, task_ctx).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_bare_boolean_filter_stays_post_decode() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-bare-boolean-filter");
    let path = dir.join("input.parquet");
    write_bare_boolean_filter_parquet_file(&path);

    let mut config = SessionConfig::new();
    config.options_mut().execution.parquet.pushdown_filters = true;
    let native_physical_plan = sql_plan(&path, config, SQL_BARE_BOOLEAN_FILTER_QUERY).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    let sources = lowering.plans[0]
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 1);
    assert!(
        sources[0].reader_predicate.is_none(),
        "bare boolean filters must not become top-level column reader predicates"
    );
    assert_eq!(lowering.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");

    let context = bounded_execution_context(&lowering.plans[0]);
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let (output, metrics) = result.into_parts();
    let QueryOutput::HostArrow(batch) = output else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_eq!(batch_rows(&[batch]), vec![vec!["3".to_string()]]);
    assert_eq!(metrics.input_rows, 6);

    fs::remove_dir_all(dir).unwrap();
}

struct NativeQueryObservation {
    output: Vec<RecordBatch>,
    predicate_disposition: String,
    input_rows: u64,
    max_task_retained_bytes: u64,
}

async fn execute_sql_native_query(path: &std::path::Path, config: SessionConfig, sql: &str) -> NativeQueryObservation {
    let native_physical_plan = sql_plan(path, config, sql).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    assert_eq!(lowering.source_diagnostics.len(), 1);

    let context = bounded_execution_context(&lowering.plans[0]);
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let (output, metrics) = result.into_parts();
    let QueryOutput::HostArrow(batch) = output else {
        panic!("native host sink should return a RecordBatch");
    };

    NativeQueryObservation {
        output: vec![batch],
        predicate_disposition: lowering.source_diagnostics[0].predicate_disposition.as_str().to_string(),
        input_rows: metrics.input_rows,
        max_task_retained_bytes: metrics
            .max_task_retained_bytes
            .expect("bounded source read should report retained bytes"),
    }
}

#[tokio::test]
async fn native_runtime_nullable_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-nullable-avg");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_uneven_row_group_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-uneven-avg");
    let path = dir.join("input.parquet");
    write_uneven_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_sort_null_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-sort-nulls");
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
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-topk-nulls");
    let path = dir.join("input.parquet");
    write_nullable_sort_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_SORT_NULLS_TOPK_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_nullable_avg_output_ordering_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-nullable-avg-sort");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_ORDER_BY_VALUE_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_discounted_revenue_global_decimal_sum_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-discounted-revenue");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_Q6_SHAPED_QUERY).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 4);

    let context = bounded_execution_context(&lowering.plans[0]);
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
async fn native_runtime_order_sensitive_source_scan_limit_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-order-sensitive-scan-limit");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan_with_order_sensitive_limit(parquet_scan(&path).await, 3);
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(datafusion::common::ScalarValue::Int32(Some(0))),
        &scan_schema,
    )
    .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);
    let cpu_batches =
        collect(Arc::clone(&plan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let lowering = lower_physical_plan_to_native_for_test(plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    assert_eq!(lowering.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");
    assert!(
        lowering.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "preserve_order")
    );

    let context = bounded_execution_context(&lowering.plans[0]);
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
async fn native_runtime_parquet_sorting_columns_scan_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-parquet-sorting-columns");
    let path = dir.join("input.parquet");
    write_sorting_columns_parquet_file(&path);

    let cpu_plan = sql_plan(&path, SessionConfig::new(), "SELECT key, payload FROM bench LIMIT 3").await;
    assert!(
        cpu_plan.properties().output_ordering().is_some(),
        "Parquet sorting_columns metadata should infer scan output ordering"
    );
    let cpu_batches =
        collect(Arc::clone(&cpu_plan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let native_plan = native_datafusion_sql_plan(&path, "SELECT key, payload FROM bench LIMIT 3").await;
    assert_nexus_native_exec(&native_plan, "sorting_columns scan should optimize to native exec");
    assert_eq!(native_plan.properties().output_ordering(), cpu_plan.properties().output_ordering());

    let native_batches =
        collect(native_plan, Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_global_decimal_avg_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-global-decimal-avg");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let native_physical_plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);

    let context = bounded_execution_context(&lowering.plans[0]);
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
async fn native_runtime_zero_column_count_matches_cpu_without_parquet_data_columns() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-zero-column-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let native_physical_plan = count_only_sort_limit_plan(parquet_scan(&path).await);
    let cpu_batches =
        collect(Arc::clone(&native_physical_plan), Arc::new(datafusion::execution::context::TaskContext::default()))
            .await
            .unwrap();
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.source_diagnostics[0].read_column_count, 0);
    assert_eq!(lowering.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");

    let context = bounded_execution_context(&lowering.plans[0]);
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
async fn native_runtime_default_sql_not_filter_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-not-filter");
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
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-cast");
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
async fn native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_runtime_partial_byte_range_access_plan_intersection_matches_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-runtime-byte-range-access-plan");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 4, "fixture must have at least four row groups");
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
    ]);
    let scan = parquet_scan_with_explicit_byte_ranges_and_access_plan(
        parquet_scan(&path).await,
        vec![(row_group_offsets[1], row_group_offsets[3])],
        access_plan,
    );
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(datafusion::common::ScalarValue::Int32(Some(0))),
        &scan_schema,
    )
    .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);
    assert_native_physical_plan_parity(plan, Some((1, 4))).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn assert_default_sql_native_parity(
    path: &std::path::Path,
    sql: &str,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    assert_sql_native_parity(path, SessionConfig::new(), sql, expected_selected_row_groups).await;
}

async fn assert_sql_native_parity(
    path: &std::path::Path,
    config: SessionConfig,
    sql: &str,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches = cpu_sql_output(path, sql).await;
    let native_physical_plan = sql_plan(path, config, sql).await;
    let lowering = lower_physical_plan_to_native_for_test(native_physical_plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let context = bounded_execution_context(&lowering.plans[0]);
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
    assert!(metrics.source_chunks_committed > 0);
    assert!(metrics.source_chunks_committed <= metrics.source_chunks_planned);
    assert_eq!(metrics.row_groups_planned, metrics.row_groups_committed);
    assert_eq!(cudf::testing::arrow::arrow_host_import_call_count_for_test(), 0);
    assert_eq!(cudf::testing::arrow::arrow_host_export_call_count_for_test(), 1);

    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal(&cpu_batches, std::slice::from_ref(native_batch));
}

async fn assert_empty_local_parquet_aggregate_parity(plan: Arc<dyn ExecutionPlan>) {
    let cpu_batches =
        collect(Arc::clone(&plan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let lowering = lower_physical_plan_to_native_for_test(plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert!(lowering.source_diagnostics.is_empty());
    assert!(lowering.plans[0].nodes().iter().any(|node| {
        matches!(
            &node.kind,
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LiteralBatch(_))
        )
    }));
    assert!(lowering.plans[0].nodes().iter().all(|node| {
        !matches!(
            &node.kind,
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_))
        )
    }));

    let context = bounded_execution_context(&lowering.plans[0]);
    let result = execute_query_plan(&context, &lowering.plans[0]).unwrap();
    let QueryOutput::HostArrow(native_batch) = result.output() else {
        panic!("native host sink should return a RecordBatch");
    };
    assert_batches_equal_allow_empty(&cpu_batches, std::slice::from_ref(native_batch));
}

async fn assert_native_physical_plan_parity(
    plan: Arc<dyn ExecutionPlan>,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches =
        collect(Arc::clone(&plan), Arc::new(datafusion::execution::context::TaskContext::default())).await.unwrap();
    let lowering = lower_physical_plan_to_native_for_test(plan);
    assert_eq!(lowering.selected_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.not_selected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.plans.len(), 1);
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(lowering.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let context = bounded_execution_context(&lowering.plans[0]);
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

fn parquet_scan_with_order_sensitive_limit(scan: Arc<dyn ExecutionPlan>, limit: usize) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan =
        FileScanConfigBuilder::from(file_scan.clone()).with_limit(Some(limit)).with_preserve_order(true).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_source_limit(scan: Arc<dyn ExecutionPlan>, limit: usize) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_limit(Some(limit)).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_without_files(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan =
        FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(vec![FileGroup::new(vec![])]).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_pushed_down_payload_predicate(scan: Arc<dyn ExecutionPlan>, value: i32) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(datafusion::common::ScalarValue::Int32(Some(value))),
        &scan_schema,
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

fn grouped_sum_aggregate_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &scan_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], scan, scan_schema)
            .unwrap(),
    )
}

fn global_sum_aggregate_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(sum)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    )
}

fn parquet_scan_with_explicit_byte_ranges_and_access_plan(
    scan: Arc<dyn ExecutionPlan>,
    ranges: Vec<(i64, i64)>,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group
                    .files()
                    .iter()
                    .flat_map(|file| {
                        ranges
                            .iter()
                            .map(|(start, end)| {
                                file.clone().with_range(*start, *end).with_extension(access_plan.clone())
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

fn row_group_range_pruning_offsets(path: &std::path::Path) -> Vec<i64> {
    let reader = SerializedFileReader::try_from(path).unwrap();
    reader
        .metadata()
        .row_groups()
        .iter()
        .map(|row_group| {
            let column = row_group.column(0);
            column.dictionary_page_offset().unwrap_or_else(|| column.data_page_offset())
        })
        .collect()
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
