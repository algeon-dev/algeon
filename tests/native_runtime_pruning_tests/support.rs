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

pub(super) async fn native_datafusion_sql_plan(
    path: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features().with_cudf_native(config).build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn native_datafusion_lineitem_sql_plan(
    path: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_native(config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("lineitem", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn native_datafusion_tpch_sql_plan(
    root: &std::path::Path,
    sql: &str,
    native_config: NativeCudfExecutionConfig,
) -> Arc<dyn ExecutionPlan> {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_config);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpch_report_session_config())
        .with_cudf_native(config)
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

pub(super) fn native_datafusion_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(512 * 1024 * 1024, 256 * 1024 * 1024, 1, 3, 1)
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

pub(super) fn unbounded_native_datafusion_config() -> NativeCudfExecutionConfig {
    // These pruning tests cover legacy whole-plan execution without bounded
    // source chunking, so the unbounded config is intentional.
    NativeCudfExecutionConfig::unbounded().with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

pub(super) fn constrained_native_datafusion_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(1, 1024 * 1024, 1, 1, 1)
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
}

pub(super) fn bounded_execution_context() -> ExecutionContext {
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

pub(super) fn adapter_error(error: &DataFusionError) -> &Error {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source.downcast_ref::<Error>().expect("DataFusion external error should preserve datafusion_cudf::Error")
}

pub(super) fn assert_native_cudf_exec(plan: &Arc<dyn ExecutionPlan>, context: &str) {
    assert_eq!(plan.name(), "NativeCudfExec", "{context}");
    assert!(plan.as_any().downcast_ref::<NativeCudfExec>().is_some(), "{context}");
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
