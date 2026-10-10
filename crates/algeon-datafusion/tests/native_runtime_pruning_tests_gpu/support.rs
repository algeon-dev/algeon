use std::{
    fs,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use super::*;

pub(super) fn direct_friendly_sql_session_config() -> SessionConfig {
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

pub(super) async fn native_datafusion_sql_plan(path: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    native_datafusion_sql_plan_with_config(path, SessionConfig::new(), sql).await
}

pub(super) async fn native_datafusion_sql_plan_with_config(
    path: &std::path::Path,
    config: SessionConfig,
    sql: &str,
) -> Arc<dyn ExecutionPlan> {
    let state = pruning_test_backend()
        .install_on(SessionStateBuilder::new_with_default_features().with_config(config))
        .expect("pruning backend must install on the DataFusion session")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn native_datafusion_lineitem_sql_plan(path: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let state = pruning_test_backend()
        .install_on(SessionStateBuilder::new_with_default_features().with_config(tpch_report_session_config()))
        .expect("pruning backend must install on the DataFusion session")
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("lineitem", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn native_datafusion_tpch_sql_plan(root: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let state = pruning_test_backend()
        .install_on(SessionStateBuilder::new_with_default_features().with_config(tpch_report_session_config()))
        .expect("pruning backend must install on the DataFusion session")
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tiny_tpch_tables(&ctx, root).await;
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn register_tiny_tpch_tables(ctx: &SessionContext, root: &std::path::Path) {
    for table in TPCH_TABLES {
        let path = root.join(format!("{table}.parquet"));
        ctx.register_parquet(*table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
}

const TEST_DEVICE_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const TEST_MAX_SOURCE_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

fn pruning_test_backend() -> &'static AlgeonGpuBackend {
    static BACKEND: std::sync::OnceLock<AlgeonGpuBackend> = std::sync::OnceLock::new();

    BACKEND.get_or_init(|| {
        let profile = AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: TEST_DEVICE_BUDGET_BYTES })
            .with_default_query_limit_bytes(TEST_DEVICE_BUDGET_BYTES);
        let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
        AlgeonGpuBackend::builder()
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(TEST_MAX_SOURCE_CHUNK_BYTES, 1, 3, 1).expect("valid source chunking"),
            ))
            .devices([device])
            .queue_policy(AlgeonGpuQueuePolicy::default())
            .build()
            .expect("pruning test backend must initialize")
    })
}

pub(super) struct TemplatedNativeOptimization {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) report: PlanningReport,
    pub(super) native_optimizer: NativeOptimizerDiagnostics,
    pub(super) source_diagnostics: Vec<SourceDiagnostics>,
}

pub(super) fn optimize_with_pruning_template(plan: Arc<dyn ExecutionPlan>) -> TemplatedNativeOptimization {
    optimize_with_optimizer_config(plan, pruning_template_optimizer_config())
}

/// The same optimization without a DataFusion CPU fallback, so a plan that
/// reads no column data still lowers to native.
pub(super) fn optimize_with_pruning_template_requiring_native(
    plan: Arc<dyn ExecutionPlan>,
) -> TemplatedNativeOptimization {
    optimize_with_optimizer_config(
        plan,
        pruning_template_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
    )
}

fn optimize_with_optimizer_config(
    plan: Arc<dyn ExecutionPlan>,
    config: AlgeonNativeOptimizerConfig,
) -> TemplatedNativeOptimization {
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);
    let (optimized, diagnostics) = rule
        .optimize_with_diagnostics(plan, &ConfigOptions::default())
        .expect("pruning template rule must optimize the plan");
    let diagnostics = diagnostics.expect("pruning template rule must retain a planning report");
    TemplatedNativeOptimization {
        plan: optimized,
        report: diagnostics.report,
        native_optimizer: diagnostics.native_optimizer,
        source_diagnostics: diagnostics.source_diagnostics,
    }
}

fn pruning_template_optimizer_config() -> AlgeonNativeOptimizerConfig {
    pruning_test_backend().optimizer_config().clone()
}

pub(super) fn metric_usize(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).unwrap().as_usize()
}

pub(super) fn assert_algeon_native_exec(plan: &Arc<dyn ExecutionPlan>, context: &str) {
    let plan = below_native_admission(plan);
    assert_eq!(plan.name(), "AlgeonNativeExec", "{context}");
    assert!(plan.downcast_ref::<AlgeonNativeExec>().is_some(), "{context}");
}

pub(super) fn assert_batches_equal(left: &[RecordBatch], right: &[RecordBatch]) {
    assert!(!left.is_empty());
    assert!(!right.is_empty());
    assert_schema_names_and_types_equal(&left[0], &right[0]);
    assert_eq!(batch_rows(left), batch_rows(right));
}

pub(super) fn assert_batches_equal_allow_empty(left: &[RecordBatch], right: &[RecordBatch]) {
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

pub(super) fn decimal128_array(batch: &RecordBatch, index: usize) -> &Decimal128Array {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array")
}

fn assert_schema_names_and_types_equal(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_columns(), right.num_columns());
    for index in 0..left.num_columns() {
        assert_eq!(left.schema().field(index).name(), right.schema().field(index).name());
        assert_eq!(left.schema().field(index).data_type(), right.schema().field(index).data_type());
    }
}

pub(super) fn batch_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
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

pub(super) fn array_value(array: &ArrayRef, row: usize) -> String {
    array_value_to_string(array.as_ref(), row).unwrap()
}

pub(super) fn parquet_row_group_count(path: &std::path::Path) -> usize {
    SerializedFileReader::try_from(path).unwrap().metadata().num_row_groups()
}

pub(super) fn aggregate_sort_limit_plan_with_filter(
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

pub(super) fn count_only_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
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

pub(super) fn aggregate_sort_limit_plan_with_projected_x00(
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

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
