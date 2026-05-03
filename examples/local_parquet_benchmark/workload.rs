use super::*;

pub(super) async fn build_cpu_comparable_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let source = parquet_scan_plan(path).await?;
    build_gpu_friendly_plan(source, config)
}

pub(super) async fn build_cpu_default_plan(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ctx = SessionContext::new();
    let path =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    ctx.register_parquet("bench", path, ParquetReadOptions::default()).await?;
    let df = ctx.sql(&benchmark_sql(config)).await?;
    df.create_physical_plan().await
}

pub(super) async fn parquet_scan_plan(path: &Path) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let ctx = SessionContext::new_with_config(benchmark_session_config());
    let path =
        path.to_str().ok_or_else(|| plan_error(format!("benchmark path is not valid UTF-8: {}", path.display())))?;
    let df = ctx.read_parquet(path, ParquetReadOptions::default()).await?;
    df.create_physical_plan().await
}

pub(super) fn build_gpu_friendly_plan(
    source: Arc<dyn ExecutionPlan>,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let input_schema = source.schema();
    let predicate =
        binary(col("payload", &input_schema)?, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &input_schema)?;
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, source)?);

    let filter_schema = filter.schema();
    let mut projection_exprs = Vec::with_capacity(config.projected_measure_pairs() + 1);
    projection_exprs.push(ProjectionExpr { expr: col("key", &filter_schema)?, alias: "key".to_string() });
    for pair in 0..config.projected_measure_pairs() {
        let left = measure_column_name(pair * 2);
        let right = measure_column_name(pair * 2 + 1);
        let expr = binary(col(&left, &filter_schema)?, Operator::Plus, col(&right, &filter_schema)?, &filter_schema)?;
        projection_exprs.push(ProjectionExpr { expr, alias: projected_column_name(pair) });
    }
    let projection: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(projection_exprs, filter)?);

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema)?, "key".to_string())]);
    let mut aggregate_exprs = Vec::with_capacity(config.projected_measure_pairs());
    for pair in 0..config.projected_measure_pairs() {
        let input_name = projected_column_name(pair);
        let output_name = aggregate_column_name(pair);
        let aggregate = AggregateExprBuilder::new(sum_udaf(), vec![col(&input_name, &projection_schema)?])
            .schema(Arc::clone(&projection_schema))
            .alias(output_name)
            .build()?;
        aggregate_exprs.push(Arc::new(aggregate));
    }
    let aggregate_filter_exprs = vec![None; aggregate_exprs.len()];
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        group_by,
        aggregate_exprs,
        aggregate_filter_exprs,
        projection,
        projection_schema,
    )?);

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema)?, ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema)?, ArrowSortOptions::new(false, false)),
    ])
    .ok_or_else(|| plan_error("benchmark sort ordering cannot be empty"))?;
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));

    Ok(Arc::new(LocalLimitExec::new(sort, config.limit)))
}

pub(super) fn benchmark_sql(config: &LocalParquetBenchmarkConfig) -> String {
    let aggregates = (0..config.projected_measure_pairs())
        .map(|pair| {
            format!(
                "SUM({} + {}) AS {}",
                measure_column_name(pair * 2),
                measure_column_name(pair * 2 + 1),
                aggregate_column_name(pair)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "SELECT key, {aggregates} FROM bench WHERE payload >= 10 GROUP BY key ORDER BY sum_x00 DESC, key ASC LIMIT {}",
        config.limit
    )
}

pub(super) async fn local_parquet_scan_diagnostic(path: &Path) -> DataFusionResult<String> {
    let plan = parquet_scan_plan(path).await?;
    let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return Ok(format!("root_plan={}", plan.name()));
    };
    let Some((file_scan, parquet_source)) = data_source.downcast_to_file_source::<ParquetSource>() else {
        return Ok("root_plan=DataSourceExec, source=not_parquet".to_string());
    };
    let ranged_files =
        file_scan.file_groups.iter().flat_map(|group| group.files().iter()).filter(|file| file.range.is_some()).count();
    let total_files = file_scan.file_groups.iter().map(|group| group.files().len()).sum::<usize>();

    Ok(format!(
        "object_store_url={}, partition_cols={}, expr_adapter={}, scan_limit={}, preserve_order={}, parquet_filter={}, file_groups={}, files={}, ranged_files={}",
        file_scan.object_store_url,
        file_scan.table_partition_cols().len(),
        file_scan.expr_adapter_factory.is_some(),
        file_scan.limit.is_some(),
        file_scan.preserve_order,
        parquet_source.filter().is_some(),
        file_scan.file_groups.len(),
        total_files,
        ranged_files,
    ))
}

pub(super) fn benchmark_session_config() -> SessionConfig {
    SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1)
}

pub(super) fn native_datafusion_session_config() -> SessionConfig {
    let mut config = benchmark_session_config().with_repartition_aggregations(false).with_repartition_sorts(false);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

pub(super) fn measure_column_name(index: usize) -> String {
    format!("m{index:02}")
}

pub(super) fn projected_column_name(pair: usize) -> String {
    format!("x{pair:02}")
}

pub(super) fn aggregate_column_name(pair: usize) -> String {
    format!("sum_x{pair:02}")
}
