use super::*;

pub(super) fn report_row<'a>(
    report: &'a NativeFallbackPressureReport,
    query_id: &str,
) -> &'a NativeFallbackPressureRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) fn planning_rows<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> Vec<&'a NativeDataFusionPlanningRow> {
    report.rows.iter().filter(|row| row.query_id == query_id).collect()
}

pub(super) fn whole_plan_planning_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a NativeDataFusionPlanningRow {
    report.rows.iter().find(|row| row.query_id == query_id).expect("report row should exist")
}

pub(super) fn planning_replacement_row<'a>(
    report: &'a NativeDataFusionPlanningReport,
    query_id: &str,
) -> &'a NativeDataFusionPlanningRow {
    let rows = report.rows.iter().filter(|row| row.query_id == query_id);
    // Replacement assertions use the last accepted fragment row when bounded
    // whole-plan admission fails closed and child native fragments still
    // replace part of the DataFusion plan. Whole-plan assertions must use
    // `whole_plan_planning_row` so those rejections remain visible.
    rows.clone()
        .rev()
        .find(|row| row.outcome == Some(CandidateOutcomeKind::Selected))
        .or_else(|| rows.into_iter().next())
        .expect("report row should exist")
}

pub(super) fn planning_query_count(report: &NativeDataFusionPlanningReport) -> usize {
    report.rows.iter().map(|row| row.query_id.as_str()).collect::<std::collections::BTreeSet<_>>().len()
}

pub(super) async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) async fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    let df = ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn typed_null_and_try_cast_reject_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let typed_null_comparison =
        binary(lit(ScalarValue::Int64(None)), Operator::Eq, lit(ScalarValue::Int64(None)), &scan_schema).unwrap();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int32));
    let try_cast_comparison =
        binary(try_cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let predicate = binary(typed_null_comparison, Operator::And, try_cast_comparison, &scan_schema).unwrap();
    aggregate_sort_limit_plan_with_filter(scan, predicate)
}

pub(super) fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_m00")
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
        PhysicalSortExpr::new(col("sum_m00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(2)));
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) async fn native_datafusion_planning_report(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
) -> NativeDataFusionPlanningReport {
    native_datafusion_planning_report_with_session(query_id, path, sql, direct_friendly_sql_session_config()).await
}

pub(super) async fn native_datafusion_planning_report_with_session(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
) -> NativeDataFusionPlanningReport {
    let optimizer_config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(
        AlgeonNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1),
    );
    native_datafusion_planning_report_with_optimizer_config(query_id, path, sql, session_config, optimizer_config).await
}

pub(super) async fn native_datafusion_planning_report_with_optimizer_config(
    query_id: &str,
    path: &std::path::Path,
    sql: &str,
    session_config: SessionConfig,
    optimizer_config: AlgeonNativeOptimizerConfig,
) -> NativeDataFusionPlanningReport {
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    let (state, logical) = df.into_parts();
    let (plan, diagnostics) = plan_with_native_diagnostics(&state, &logical).await.unwrap();
    NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(query_id, plan, diagnostics)])
}

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

pub(super) fn count_only_sql_session_config() -> SessionConfig {
    let mut config = direct_friendly_sql_session_config().with_collect_statistics(false);
    config.options_mut().execution.collect_statistics = false;
    config
}

pub(super) fn write_numeric_parquet_file(path: &std::path::Path) {
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

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
