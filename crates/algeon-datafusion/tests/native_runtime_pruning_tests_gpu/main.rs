use std::{fs, sync::Arc};

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit,
        GpuMemoryOwnership,
    },
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, SourceChunking, below_native_admission},
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule},
    report::PlanningReport,
};
use arrow::{
    array::{ArrayRef, Decimal128Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::display::array_value_to_string,
};
use cudf::{
    testing,
    testing::arrow::{
        arrow_host_export_call_count_for_test, arrow_host_import_call_count_for_test,
        reset_arrow_host_interop_observation_for_test,
    },
};
use datafusion::{
    common::ScalarValue,
    config::ConfigOptions,
    datasource::{
        physical_plan::{
            FileGroup, FileScanConfig, FileScanConfigBuilder, ParquetSource,
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
        metrics::MetricsSet,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use fixtures::*;
use query_planning::{optimizer::NativeOptimizerDiagnostics, source::SourceDiagnostics};
use sql::*;
use support::*;
use tokio::{runtime::Builder as TestRuntimeBuilder, sync::Mutex};
use tpch_session::tpch_report_session_config;
use workloads::tpch::{TPCH_QUERIES, TPCH_TABLES, write_tiny_tpch_parquet_dataset};

static ARROW_HOST_OBSERVATION_LOCK: Mutex<()> = Mutex::const_new(());

#[path = "../support/tpch_session.rs"]
mod tpch_session;

mod datafusion_exec;
mod fixtures;
mod scalar_expressions;
mod sql;
mod support;
mod tpch;

async fn native_runtime_default_sql_min_max_count_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-default-min-max-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_MIN_MAX_COUNT_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_default_sql_count_star_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-default-count-star");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_default_sql_native_parity(&path, SQL_COUNT_STAR_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_runtime_pruning_cases_1() {
    testing::run_named_cases(&[
        ("native_runtime_default_sql_distributed_avg_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_distributed_avg_matches_cpu())
        }),
        ("native_runtime_default_sql_distributed_filtered_aggregate_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_distributed_filtered_aggregate_matches_cpu())
        }),
        ("native_runtime_mixed_filtered_and_unfiltered_aggregates_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_mixed_filtered_and_unfiltered_aggregates_match_cpu())
        }),
        ("native_runtime_median_aggregate_matches_cpu_single_stage_bounded_without_partials", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_median_aggregate_matches_cpu_single_stage_bounded_without_partials())
        }),
        ("native_runtime_reader_pushdown_reduces_source_rows_and_retained_bytes", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_reader_pushdown_reduces_source_rows_and_retained_bytes())
        }),
        ("native_runtime_decimal32_reader_pushdown_matches_post_scan", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal32_reader_pushdown_matches_post_scan())
        }),
    ]);
}

#[test]
fn native_runtime_pruning_cases_2() {
    testing::run_named_cases(&[
        ("native_runtime_decimal32_equality_reader_pushdown_matches_post_scan", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_decimal32_equality_reader_pushdown_matches_post_scan())
        }),
        ("native_runtime_pushed_predicate_source_limit_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_pushed_predicate_source_limit_matches_cpu())
        }),
        ("native_runtime_empty_local_parquet_aggregates_match_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_empty_local_parquet_aggregates_match_cpu())
        }),
        ("native_runtime_physically_empty_parquet_files_match_cpu_without_replay", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_physically_empty_parquet_files_match_cpu_without_replay())
        }),
        ("native_runtime_filtered_empty_rollup_grand_total_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_filtered_empty_rollup_grand_total_matches_cpu())
        }),
        ("native_runtime_bare_boolean_filter_stays_post_decode", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_bare_boolean_filter_stays_post_decode())
        }),
    ]);
}

#[test]
fn native_runtime_pruning_cases_3() {
    testing::run_named_cases(&[
        ("native_runtime_nullable_distributed_avg_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_nullable_distributed_avg_matches_cpu())
        }),
        ("native_runtime_grouped_single_distinct_split_matches_cpu_with_null_values", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_grouped_single_distinct_split_matches_cpu_with_null_values())
        }),
        ("native_runtime_global_single_distinct_split_matches_cpu_with_null_values", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_global_single_distinct_split_matches_cpu_with_null_values())
        }),
        ("native_runtime_global_single_distinct_split_over_empty_input_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_global_single_distinct_split_over_empty_input_matches_cpu())
        }),
        ("native_runtime_global_count_join_pushdown_over_empty_join_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_global_count_join_pushdown_over_empty_join_matches_cpu())
        }),
        ("native_runtime_uneven_row_group_distributed_avg_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_uneven_row_group_distributed_avg_matches_cpu())
        }),
    ]);
}

#[test]
fn native_runtime_pruning_cases_4() {
    testing::run_named_cases(&[
        ("native_runtime_sort_null_ordering_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_sort_null_ordering_matches_cpu())
        }),
        ("native_runtime_bounded_topk_null_ordering_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_bounded_topk_null_ordering_matches_cpu())
        }),
        ("native_runtime_nullable_avg_output_ordering_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_nullable_avg_output_ordering_matches_cpu())
        }),
        ("native_runtime_discounted_revenue_global_decimal_sum_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_discounted_revenue_global_decimal_sum_matches_cpu())
        }),
        ("native_runtime_order_sensitive_source_scan_limit_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_order_sensitive_source_scan_limit_matches_cpu())
        }),
    ]);
}

#[test]
fn native_runtime_pruning_cases_5() {
    testing::run_named_cases(&[
        ("native_runtime_parquet_sorting_columns_scan_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_parquet_sorting_columns_scan_matches_cpu())
        }),
        ("native_runtime_sorted_file_split_into_byte_ranges_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_sorted_file_split_into_byte_ranges_matches_cpu())
        }),
        ("native_runtime_global_decimal_avg_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_global_decimal_avg_matches_cpu())
        }),
        ("native_runtime_zero_column_count_matches_cpu_without_parquet_data_columns", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_zero_column_count_matches_cpu_without_parquet_data_columns())
        }),
        ("native_runtime_default_sql_not_filter_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_not_filter_matches_cpu())
        }),
    ]);
}

#[test]
fn native_runtime_pruning_cases_6() {
    testing::run_named_cases(&[
        ("native_runtime_widening_numeric_cast_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_widening_numeric_cast_matches_cpu())
        }),
        ("native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate())
        }),
        ("native_runtime_partial_byte_range_access_plan_intersection_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_partial_byte_range_access_plan_intersection_matches_cpu())
        }),
        ("native_runtime_default_sql_min_max_count_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_min_max_count_matches_cpu())
        }),
        ("native_runtime_default_sql_count_star_matches_cpu", || {
            TestRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("case runtime")
                .block_on(native_runtime_default_sql_count_star_matches_cpu())
        }),
    ]);
}

async fn native_runtime_default_sql_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-default-avg");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_AGGREGATE_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_default_sql_distributed_filtered_aggregate_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-default-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_FILTERED_AGGREGATE_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_mixed_filtered_and_unfiltered_aggregates_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    assert_sql_native_parity(&path, direct_friendly_sql_session_config(), SQL_MIXED_FILTERED_AGGREGATES_QUERY, None)
        .await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_median_aggregate_matches_cpu_single_stage_bounded_without_partials() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-median-aggregate-bounded");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_MEDIAN_AGGREGATE_QUERY).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_MEDIAN_AGGREGATE_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "median aggregate must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "partial_aggregate_count"), 0, "median must stay single-stage");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_reader_pushdown_reduces_source_rows_and_retained_bytes() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-reader-pushdown-selective");
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

async fn native_runtime_decimal32_reader_pushdown_matches_post_scan() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal32-reader-pushdown");
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

async fn native_runtime_decimal32_equality_reader_pushdown_matches_post_scan() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-decimal32-equality-reader-pushdown");
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

async fn native_runtime_pushed_predicate_source_limit_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-pushed-predicate-source-limit");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan_with_source_limit(
        parquet_scan_with_pushed_down_payload_predicate(parquet_scan(&path).await, 10),
        3,
    );
    let cpu_batches = collect(Arc::clone(&scan), Arc::new(TaskContext::default())).await.unwrap();
    let optimization = optimize_with_pruning_template(scan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(optimization.source_diagnostics[0].rejected_features.is_empty());
    assert_algeon_native_exec(&optimization.plan, "pushed-predicate plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_empty_local_parquet_aggregates_match_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-empty-local-aggregates");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let empty_scan = parquet_scan_without_files(parquet_scan(&path).await);
    assert_empty_local_parquet_aggregate_parity(grouped_sum_aggregate_plan(Arc::clone(&empty_scan))).await;
    assert_empty_local_parquet_aggregate_parity(global_sum_aggregate_plan(empty_scan)).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_physically_empty_parquet_files_match_cpu_without_replay() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-physical-empty-parquet");
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
            assert_algeon_native_exec(
                &native_plan,
                "physical empty parquet aggregate should lower to AlgeonNativeExec",
            );
            let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();
            assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_filtered_empty_rollup_grand_total_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-filtered-empty-rollup");
    let path = dir.join("bench.parquet");
    write_filtered_empty_rollup_parquet_file(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_FILTERED_EMPTY_ROLLUP_SUM_QUERY).await;
    assert_eq!(
        cpu_batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1,
        "CPU DataFusion must emit the ROLLUP grand-total row over filtered-empty input"
    );

    let native_physical_plan = native_datafusion_sql_plan(&path, SQL_FILTERED_EMPTY_ROLLUP_SUM_QUERY).await;
    assert_algeon_native_exec(&native_physical_plan, "runtime-filtered empty ROLLUP should lower to AlgeonNativeExec");

    let task_ctx = Arc::new(TaskContext::default());
    let native_batches = collect(native_physical_plan, task_ctx).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_bare_boolean_filter_stays_post_decode() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-bare-boolean-filter");
    let path = dir.join("input.parquet");
    write_bare_boolean_filter_parquet_file(&path);

    let mut config = SessionConfig::new();
    config.options_mut().execution.parquet.pushdown_filters = true;
    let native_physical_plan = sql_plan(&path, config, SQL_BARE_BOOLEAN_FILTER_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    assert_algeon_native_exec(&optimization.plan, "bare-boolean plan must rewrite to AlgeonNativeExec");

    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(batch_rows(&native_batches), vec![vec!["3".to_string()]]);
    assert_eq!(metric_usize(&metrics, "input_rows"), 6);

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
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics.len(), 1);
    assert_algeon_native_exec(&optimization.plan, "reader-pushdown query must rewrite to AlgeonNativeExec");

    let output = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");

    NativeQueryObservation {
        output,
        predicate_disposition: optimization.source_diagnostics[0].predicate_disposition.as_str().to_string(),
        input_rows: metric_usize(&metrics, "input_rows") as u64,
        max_task_retained_bytes: metric_usize(&metrics, "max_task_retained_bytes") as u64,
    }
}

async fn native_runtime_nullable_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-nullable-avg");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_grouped_single_distinct_split_matches_cpu_with_null_values() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-grouped-single-distinct");
    let path = dir.join("input.parquet");
    write_nullable_single_distinct_input_parquet(&path);

    assert_single_distinct_split_parity(&path, SQL_SINGLE_DISTINCT_GROUPED_QUERY).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_global_single_distinct_split_matches_cpu_with_null_values() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-global-single-distinct");
    let path = dir.join("input.parquet");
    write_nullable_single_distinct_input_parquet(&path);

    assert_single_distinct_split_parity(&path, SQL_SINGLE_DISTINCT_GLOBAL_QUERY).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_global_single_distinct_split_over_empty_input_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-empty-single-distinct");
    let path = dir.join("input.parquet");
    write_nullable_single_distinct_input_parquet(&path);

    assert_single_distinct_split_parity(&path, SQL_SINGLE_DISTINCT_EMPTY_GLOBAL_QUERY).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_global_count_join_pushdown_over_empty_join_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-empty-join-global-count");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    assert_aggregate_join_pushdown_parity(&dir, SQL_EMPTY_JOIN_GLOBAL_COUNT_QUERY).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_uneven_row_group_distributed_avg_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-uneven-avg");
    let path = dir.join("input.parquet");
    write_uneven_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_BY_KEY_NO_FILTER_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_sort_null_ordering_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-sort-nulls");
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

async fn native_runtime_bounded_topk_null_ordering_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-topk-nulls");
    let path = dir.join("input.parquet");
    write_nullable_sort_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_SORT_NULLS_TOPK_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_nullable_avg_output_ordering_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-nullable-avg-sort");
    let path = dir.join("input.parquet");
    write_nullable_avg_input_parquet(&path);

    assert_default_sql_native_parity(&path, SQL_AVG_ORDER_BY_VALUE_QUERY, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_discounted_revenue_global_decimal_sum_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-discounted-revenue");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_Q6_SHAPED_QUERY).await;
    let native_physical_plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].read_column_count, 4);
    assert_algeon_native_exec(&optimization.plan, "discounted-revenue plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);
    let revenue = decimal128_array(&native_batches[0], 0);
    assert_eq!(revenue.value(0), 260_000);
    assert_eq!(revenue.scale(), 4);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_order_sensitive_source_scan_limit_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-order-sensitive-scan-limit");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan_with_order_sensitive_limit(parquet_scan(&path).await, 3);
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(0))), &scan_schema)
            .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);
    let cpu_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();
    let optimization = optimize_with_pruning_template(plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");
    assert!(
        optimization.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "preserve_order")
    );
    assert_algeon_native_exec(&optimization.plan, "order-sensitive plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_parquet_sorting_columns_scan_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-parquet-sorting-columns");
    let path = dir.join("input.parquet");
    write_sorting_columns_parquet_file(&path);

    let cpu_plan = sql_plan(&path, SessionConfig::new(), "SELECT key, payload FROM bench LIMIT 3").await;
    assert!(
        cpu_plan.properties().output_ordering().is_some(),
        "Parquet sorting_columns metadata should infer scan output ordering"
    );
    let cpu_batches = collect(Arc::clone(&cpu_plan), Arc::new(TaskContext::default())).await.unwrap();
    let native_plan = native_datafusion_sql_plan(&path, "SELECT key, payload FROM bench LIMIT 3").await;
    assert_algeon_native_exec(&native_plan, "sorting_columns scan should optimize to native exec");
    assert_eq!(native_plan.properties().output_ordering(), cpu_plan.properties().output_ordering());

    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion splits one sorted file into byte-range groups under the default
/// `target_partitions`. The native reader coalesces the ranges back into that
/// file and reads its row groups in order, so the declared ordering holds.
async fn native_runtime_sorted_file_split_into_byte_ranges_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-sorted-byte-ranges");
    let path = dir.join("input.parquet");
    write_multi_row_group_sorting_columns_parquet_file(&path);
    // Lowers the 10 MiB repartition floor so this small file splits like a
    // large one; `target_partitions` keeps its default.
    let config = SessionConfig::new().with_repartition_file_min_size(1);
    let sql = "SELECT key, payload FROM bench WHERE payload >= 500 ORDER BY key LIMIT 7";

    let cpu_plan = sql_plan(&path, config.clone(), sql).await;
    let scans = collect_file_scans(&cpu_plan);
    assert_eq!(scans.len(), 1, "one Parquet scan");
    assert!(scans[0].file_groups.len() > 1, "the scan should split into byte-range groups: {:?}", scans[0].file_groups);
    assert!(!scans[0].output_ordering.is_empty(), "sorting_columns should declare the scan ordering");
    let cpu_batches = collect(Arc::clone(&cpu_plan), Arc::new(TaskContext::default())).await.unwrap();

    let native_plan = native_datafusion_sql_plan_with_config(&path, config, sql).await;
    assert_algeon_native_exec(&native_plan, "a single sorted file split into byte ranges should run natively");
    let native_batches = collect(native_plan, Arc::new(TaskContext::default())).await.unwrap();

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

fn collect_file_scans(plan: &Arc<dyn ExecutionPlan>) -> Vec<FileScanConfig> {
    let mut scans = plan.children().into_iter().flat_map(collect_file_scans).collect::<Vec<_>>();
    if let Some((file_scan, _parquet_source)) = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|data_source| data_source.downcast_to_file_source::<ParquetSource>())
    {
        scans.push(file_scan.clone());
    }
    scans
}

async fn native_runtime_global_decimal_avg_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-global-decimal-avg");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet(&path);

    let cpu_batches = cpu_sql_output(&path, SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let native_physical_plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_DECIMAL_AVG_QUERY).await;
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "decimal-avg plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_zero_column_count_matches_cpu_without_parquet_data_columns() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-zero-column-count");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let native_physical_plan = count_only_sort_limit_plan(parquet_scan(&path).await);
    let cpu_batches = collect(Arc::clone(&native_physical_plan), Arc::new(TaskContext::default())).await.unwrap();
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.source_diagnostics[0].read_column_count, 0);
    assert_eq!(optimization.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");
    assert_algeon_native_exec(&optimization.plan, "zero-column plan must rewrite to AlgeonNativeExec");

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");
    assert_eq!(metric_usize(&metrics, "read_columns"), 0);
    assert_eq!(metric_usize(&metrics, "input_rows"), 8);
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_default_sql_not_filter_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-not-filter");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan(&path).await;
    let scan_schema = scan.schema();
    let payload_lt_10 =
        binary(col("payload", &scan_schema).unwrap(), Operator::Lt, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let predicate = Arc::new(NotExpr::new(payload_lt_10));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);
    assert_native_physical_plan_parity(plan, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_widening_numeric_cast_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-cast");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);

    let scan = parquet_scan(&path).await;
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let cast_payload = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int64, None));
    let x00 = binary(cast_payload, Operator::Plus, col("m00", &scan_schema).unwrap(), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_projected_x00(scan, predicate, x00);
    assert_native_physical_plan_parity(plan, None).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_default_sql_prunes_metadata_row_groups_for_pushed_parquet_predicate() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);

    assert_default_sql_native_parity(&path, SQL_SUM_EXPRESSION_QUERY, Some((2, total_row_groups))).await;

    fs::remove_dir_all(dir).unwrap();
}

async fn native_runtime_partial_byte_range_access_plan_intersection_matches_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-runtime-byte-range-access-plan");
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
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(0))), &scan_schema)
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
    let optimization = optimize_with_pruning_template(native_physical_plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "pruning template must rewrite the plan to AlgeonNativeExec");
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(optimization.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");

    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(total_row_groups, 4);
        assert_eq!(metric_usize(&metrics, "row_groups_selected"), selected_row_groups);
        assert!(selected_row_groups < total_row_groups);
    }
    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "host_to_device_uploaded_rows"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert!(metric_usize(&metrics, "source_chunks_committed") > 0);
    assert!(metric_usize(&metrics, "source_chunks_committed") <= metric_usize(&metrics, "source_chunks_planned"));
    assert_eq!(metric_usize(&metrics, "row_groups_planned"), metric_usize(&metrics, "row_groups_committed"));
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);
}

/// DataFusion keeps one aggregate for `COUNT(DISTINCT x)` beside `COUNT`, so
/// the native optimizer must split it for the comparison to cover the split.
async fn assert_single_distinct_split_parity(path: &std::path::Path, sql: &str) {
    let cpu_batches = cpu_sql_output(path, sql).await;
    let optimization = optimize_with_pruning_template_requiring_native(sql_plan(path, SessionConfig::new(), sql).await);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "single-distinct aggregate must rewrite to AlgeonNativeExec");
    assert_eq!(
        optimization.native_optimizer.single_distinct_aggregate_split.applied_count, 1,
        "{:?}",
        optimization.native_optimizer.single_distinct_aggregate_split
    );

    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
}

/// A global aggregate merges the pushed-down partial state over no rows when
/// the join is empty, which DataFusion answers with zero counts and null sums.
async fn assert_aggregate_join_pushdown_parity(root: &std::path::Path, sql: &str) {
    let cpu_batches = cpu_tpch_sql_output(root, sql).await;
    let optimization =
        optimize_with_pruning_template_requiring_native(tpch_sql_plan(root, tpch_report_session_config(), sql).await);
    assert_eq!(optimization.report.selected_count(), 1, "sql={sql} outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "aggregate over a join must rewrite to AlgeonNativeExec");
    assert_eq!(
        optimization.native_optimizer.aggregate_join_pushdown.applied_count, 1,
        "sql={sql} {:?}",
        optimization.native_optimizer.aggregate_join_pushdown
    );

    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal(&cpu_batches, &native_batches);
}

async fn assert_empty_local_parquet_aggregate_parity(plan: Arc<dyn ExecutionPlan>) {
    let cpu_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();
    // A scan without files lowers to a zero-row literal, which a plan that
    // allows DataFusion CPU execution keeps there by cost.
    let optimization = optimize_with_pruning_template_requiring_native(plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert!(optimization.source_diagnostics.is_empty());
    assert_algeon_native_exec(&optimization.plan, "empty scan aggregate must rewrite to AlgeonNativeExec");

    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    assert_batches_equal_allow_empty(&cpu_batches, &native_batches);
}

async fn assert_native_physical_plan_parity(
    plan: Arc<dyn ExecutionPlan>,
    expected_selected_row_groups: Option<(usize, usize)>,
) {
    let cpu_batches = collect(Arc::clone(&plan), Arc::new(TaskContext::default())).await.unwrap();
    let optimization = optimize_with_pruning_template(plan);
    assert_eq!(optimization.report.selected_count(), 1, "outcomes={:?}", optimization.report.outcomes());
    assert_eq!(optimization.report.not_selected_count(), 0, "outcomes={:?}", optimization.report.outcomes());
    assert_algeon_native_exec(&optimization.plan, "pruning template must rewrite the plan to AlgeonNativeExec");
    if let Some((selected_row_groups, total_row_groups)) = expected_selected_row_groups {
        assert_eq!(optimization.source_diagnostics[0].selected_row_groups, Some(selected_row_groups));
        assert!(selected_row_groups < total_row_groups);
    }

    let _observation_guard = ARROW_HOST_OBSERVATION_LOCK.lock().await;
    reset_arrow_host_interop_observation_for_test();
    let native_batches = collect(Arc::clone(&optimization.plan), Arc::new(TaskContext::default())).await.unwrap();
    let metrics = below_native_admission(&optimization.plan)
        .metrics()
        .expect("AlgeonNativeExec should expose DataFusion metrics");

    assert_eq!(metric_usize(&metrics, "host_to_device_uploads"), 0);
    assert_eq!(metric_usize(&metrics, "host_to_device_uploaded_rows"), 0);
    assert_eq!(metric_usize(&metrics, "device_to_host_materializations"), 1);
    assert_eq!(arrow_host_import_call_count_for_test(), 0);
    assert_eq!(arrow_host_export_call_count_for_test(), 1);

    assert_batches_equal(&cpu_batches, &native_batches);
}

async fn cpu_sql_output(path: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = sql_plan(path, SessionConfig::new(), sql).await;
    collect(plan, Arc::new(TaskContext::default())).await.unwrap()
}

async fn cpu_lineitem_sql_output(path: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = lineitem_sql_plan(path, direct_friendly_sql_session_config(), sql).await;
    collect(plan, Arc::new(TaskContext::default())).await.unwrap()
}

async fn cpu_tpch_sql_output(root: &std::path::Path, sql: &str) -> Vec<RecordBatch> {
    let plan = tpch_sql_plan(root, tpch_report_session_config(), sql).await;
    collect(plan, Arc::new(TaskContext::default())).await.unwrap()
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
        lit(ScalarValue::Int32(Some(value))),
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
