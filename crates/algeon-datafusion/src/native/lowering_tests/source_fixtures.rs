use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    datasource::{listing::PartitionedFile, physical_plan::ParquetFileReaderFactory},
    parquet::arrow::async_reader::AsyncFileReader,
    physical_plan::metrics::ExecutionPlanMetricsSet,
};
use futures::executor::block_on;

use super::*;

pub(super) fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df = block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn parquet_scan_with_projection_indices(
    path: &std::path::Path,
    projected_indices: Vec<usize>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_projection_indices(Some(projected_indices))
        .unwrap()
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_order_sensitive_limit(path: &std::path::Path, limit: usize) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan =
        FileScanConfigBuilder::from(file_scan.clone()).with_limit(Some(limit)).with_preserve_order(true).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_order_sensitive_limit_one_file_per_group(
    path: &std::path::Path,
    limit: usize,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .flat_map(|group| group.files().iter().cloned().map(|file| FileGroup::new(vec![file])))
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_file_groups(file_groups)
        .with_limit(Some(limit))
        .with_preserve_order(true)
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_preserve_order_one_file_per_group(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .flat_map(|group| group.files().iter().cloned().map(|file| FileGroup::new(vec![file])))
        .collect();
    let file_scan =
        FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).with_preserve_order(true).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_declared_output_ordering(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let ordering = declared_key_ordering(&scan.schema());
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_output_ordering(vec![ordering]).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_declared_output_ordering_one_file_per_group(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let ordering = declared_key_ordering(&scan.schema());
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .flat_map(|group| group.files().iter().cloned().map(|file| FileGroup::new(vec![file])))
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_file_groups(file_groups)
        .with_output_ordering(vec![ordering])
        .build();
    DataSourceExec::from_data_source(file_scan)
}

fn declared_key_ordering(schema: &SchemaRef) -> LexOrdering {
    LexOrdering::new([PhysicalSortExpr::new(col("key", schema).unwrap(), ArrowSortOptions::new(false, false))])
        .expect("declared key ordering should be non-empty")
}

pub(super) fn parquet_scan_with_missing_nullable_column(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let mut fields = file_scan.file_schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new("missing_payload", DataType::Int32, true)));
    let table_schema = Arc::new(Schema::new(fields));
    let projected_indices = (0..table_schema.fields().len()).collect::<Vec<_>>();
    let parquet_source = ParquetSource::new(Arc::clone(&table_schema));
    let file_scan = FileScanConfigBuilder::new(file_scan.object_store_url.clone(), Arc::new(parquet_source))
        .with_file_groups(file_scan.file_groups.clone())
        .with_projection_indices(Some(projected_indices))
        .unwrap()
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn filtered_payload_scan(
    scan: Arc<dyn ExecutionPlan>,
    payload_column: &str,
    minimum: i32,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate = binary(
        col(payload_column, &scan_schema).unwrap(),
        Operator::GtEq,
        lit(ScalarValue::Int32(Some(minimum))),
        &scan_schema,
    )
    .unwrap();
    Arc::new(FilterExec::try_new(predicate, scan).unwrap())
}

pub(super) fn physical_projection_field_with_name(plan: &dyn ExecutionPlan, field_name: &str) -> Field {
    let mut fields = Vec::new();
    collect_physical_projection_fields_with_name(plan, field_name, &mut fields);
    assert_eq!(
        fields.len(),
        1,
        "expected exactly one DataFusion ProjectionExec field named {field_name}, found {fields:?}"
    );
    fields.pop().unwrap()
}

fn collect_physical_projection_fields_with_name(plan: &dyn ExecutionPlan, field_name: &str, fields: &mut Vec<Field>) {
    if plan.is::<ProjectionExec>()
        && let Ok(field) = plan.schema().field_with_name(field_name)
    {
        fields.push(field.clone());
    }
    for child in plan.children() {
        collect_physical_projection_fields_with_name(child.as_ref(), field_name, fields);
    }
}

pub(super) async fn sql_benchmark_plan(path: &std::path::Path, config: SessionConfig) -> Arc<dyn ExecutionPlan> {
    sql_plan(path, config, SQL_BENCHMARK_QUERY).await
}

pub(super) async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

pub(super) fn sql_plan_without_scan_predicate(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let (root_projection, sort_plan) = if let Some(proj) = plan.downcast_ref::<ProjectionExec>() {
        (Some(Arc::clone(&plan)), Arc::clone(proj.input()))
    } else {
        (None, Arc::clone(&plan))
    };
    let sort = sort_plan.downcast_ref::<SortExec>().expect("SQL plan should contain SortExec");
    let (post_projection, aggregate_plan) = if let Some(proj) = sort.input().downcast_ref::<ProjectionExec>() {
        (Some(Arc::clone(sort.input())), Arc::clone(proj.input()))
    } else {
        (None, Arc::clone(sort.input()))
    };
    let aggregate = aggregate_plan
        .downcast_ref::<AggregateExec>()
        .expect("SQL plan should have an AggregateExec below post projection or sort");
    let filter_plan = Arc::clone(aggregate.input());
    let filter = filter_plan.downcast_ref::<FilterExec>().expect("SQL plan should have a FilterExec below aggregate");
    let source_plan = Arc::clone(filter.input());
    let source =
        source_plan.downcast_ref::<DataSourceExec>().expect("SQL plan filter input should be a DataSourceExec");

    let source_without_predicate = parquet_source_exec_without_predicate(source);
    let filter_without_predicate = filter_plan
        .replace_children(
            vec![source_without_predicate],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let aggregate_without_predicate = aggregate_plan
        .replace_children(
            vec![filter_without_predicate],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    let post_projection_without_predicate = if let Some(post_proj) = post_projection {
        post_proj
            .replace_children(
                vec![aggregate_without_predicate],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap()
    } else {
        aggregate_without_predicate
    };
    let sort_without_predicate = sort_plan
        .replace_children(
            vec![post_projection_without_predicate],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    if let Some(root_proj) = root_projection {
        root_proj
            .replace_children(
                vec![sort_without_predicate],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap()
    } else {
        sort_without_predicate
    }
}

pub(super) fn parquet_source_exec_without_predicate(source: &DataSourceExec) -> Arc<dyn ExecutionPlan> {
    let (file_scan, _parquet_source) =
        source.downcast_to_file_source::<ParquetSource>().expect("SQL scan should use ParquetSource");
    let projection_indices = file_scan.file_source().projection().map(|projection| projection.column_indices());
    let parquet_source = ParquetSource::new(file_scan.file_source().table_schema().clone());
    let mut builder =
        FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source) as Arc<dyn FileSource>);
    if let Some(projection_indices) = projection_indices {
        builder = builder.with_projection_indices(Some(projection_indices)).unwrap();
    }
    DataSourceExec::from_data_source(builder.build())
}

#[derive(Debug)]
struct RejectingParquetFileReaderFactory;

impl ParquetFileReaderFactory for RejectingParquetFileReaderFactory {
    fn create_reader(
        &self,
        _partition_index: usize,
        _partitioned_file: PartitionedFile,
        _metadata_size_hint: Option<usize>,
        _metrics: &ExecutionPlanMetricsSet,
    ) -> DataFusionResult<Box<dyn AsyncFileReader + Send>> {
        Err(DataFusionError::Execution("test reader factory should not be invoked during native lowering".to_string()))
    }
}

/// Wraps the rejecting factory under a name that starts with the DataFusion
/// default factory's, so a prefix match on `Debug` output would accept it.
#[derive(Debug)]
struct DefaultParquetFileReaderFactoryProxy {
    inner: RejectingParquetFileReaderFactory,
}

impl ParquetFileReaderFactory for DefaultParquetFileReaderFactoryProxy {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> DataFusionResult<Box<dyn AsyncFileReader + Send>> {
        self.inner.create_reader(partition_index, partitioned_file, metadata_size_hint, metrics)
    }
}

pub(super) fn parquet_scan_with_custom_reader_factory(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_reader_factory(path, Arc::new(RejectingParquetFileReaderFactory))
}

pub(super) fn parquet_scan_with_default_lookalike_reader_factory(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_reader_factory(
        path,
        Arc::new(DefaultParquetFileReaderFactoryProxy { inner: RejectingParquetFileReaderFactory }),
    )
}

fn parquet_scan_with_reader_factory(
    path: &std::path::Path,
    factory: Arc<dyn ParquetFileReaderFactory>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let parquet_source =
        ParquetSource::new(file_scan.file_source().table_schema().clone()).with_parquet_file_reader_factory(factory);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_source(Arc::new(parquet_source) as Arc<dyn FileSource>)
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) struct DefaultDistributedSqlPlanParts {
    root: Arc<dyn ExecutionPlan>,
    sort: Arc<dyn ExecutionPlan>,
    projection: Arc<dyn ExecutionPlan>,
    final_aggregate: Arc<dyn ExecutionPlan>,
    between_aggregate_repartition: Arc<dyn ExecutionPlan>,
    partial_aggregate: Arc<dyn ExecutionPlan>,
    scan_input: Arc<dyn ExecutionPlan>,
}

pub(super) fn default_distributed_sql_plan_parts(plan: Arc<dyn ExecutionPlan>) -> DefaultDistributedSqlPlanParts {
    let root = plan;
    let root_spm = root.downcast_ref::<SortPreservingMergeExec>().expect("default SQL root should be SPM");
    let (sort, projection, final_aggregate) = if let Some(sort_exec) = root_spm.input().downcast_ref::<SortExec>() {
        let sort = Arc::clone(root_spm.input());
        let projection = Arc::clone(sort_exec.input());
        let projection_exec =
            projection.downcast_ref::<ProjectionExec>().expect("default SQL sort input should be ProjectionExec");
        let final_aggregate = Arc::clone(projection_exec.input());
        (sort, projection, final_aggregate)
    } else if let Some(proj_exec) = root_spm.input().downcast_ref::<ProjectionExec>() {
        let projection = Arc::clone(root_spm.input());
        let sort = Arc::clone(proj_exec.input());
        let sort_exec = sort.downcast_ref::<SortExec>().expect("default SQL projection input should be SortExec");
        let final_aggregate = Arc::clone(sort_exec.input());
        (sort, projection, final_aggregate)
    } else {
        panic!("default SQL SPM input should be SortExec or ProjectionExec");
    };
    let final_aggregate_exec =
        final_aggregate.downcast_ref::<AggregateExec>().expect("default SQL plan should contain final AggregateExec");
    assert!(matches!(final_aggregate_exec.mode(), AggregateMode::FinalPartitioned));
    let between_aggregate_repartition = Arc::clone(final_aggregate_exec.input());
    let repartition = between_aggregate_repartition
        .downcast_ref::<RepartitionExec>()
        .expect("default SQL final aggregate input should be RepartitionExec");
    let partial_aggregate = Arc::clone(repartition.input());
    let partial_aggregate_exec = partial_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL repartition input should be partial AggregateExec");
    assert!(matches!(partial_aggregate_exec.mode(), AggregateMode::Partial));
    let filter = partial_aggregate_exec
        .input()
        .downcast_ref::<FilterExec>()
        .expect("default SQL partial aggregate input should be FilterExec");
    let source_repartition =
        filter.input().downcast_ref::<RepartitionExec>().expect("default SQL filter input should be RepartitionExec");
    let scan_input = Arc::clone(source_repartition.input());

    DefaultDistributedSqlPlanParts {
        root,
        sort,
        projection,
        final_aggregate,
        between_aggregate_repartition,
        partial_aggregate,
        scan_input,
    }
}

pub(super) fn rebuild_default_distributed_sql_plan(
    parts: &DefaultDistributedSqlPlanParts,
    final_aggregate: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    if parts.root.children()[0].downcast_ref::<ProjectionExec>().is_some() {
        let sort = Arc::clone(&parts.sort)
            .replace_children(vec![final_aggregate], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap();
        let projection = Arc::clone(&parts.projection)
            .replace_children(vec![sort], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap();
        Arc::clone(&parts.root)
            .replace_children(vec![projection], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap()
    } else {
        let projection = Arc::clone(&parts.projection)
            .replace_children(vec![final_aggregate], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap();
        let sort = Arc::clone(&parts.sort)
            .replace_children(vec![projection], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap();
        Arc::clone(&parts.root)
            .replace_children(vec![sort], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap()
    }
}

pub(super) fn with_rewritten_distributed_final_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate = parts
        .final_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite(final_aggregate)]));
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

pub(super) fn with_rewritten_distributed_final_aggregate_exec(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate = parts
        .final_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    rebuild_default_distributed_sql_plan(&parts, rewrite(final_aggregate))
}

pub(super) fn with_rewritten_distributed_partial_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let partial_aggregate = parts
        .partial_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let rewritten_partial = rewrite(partial_aggregate);
    let repartition = Arc::clone(&parts.between_aggregate_repartition)
        .replace_children(vec![rewritten_partial], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    let final_aggregate = Arc::clone(&parts.final_aggregate)
        .replace_children(vec![repartition], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    rebuild_default_distributed_sql_plan(&parts, final_aggregate)
}

pub(super) fn with_rewritten_distributed_partial_and_final_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite_partial: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
    rewrite_final: impl FnOnce(&AggregateExec) -> Arc<AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let partial_aggregate = parts
        .partial_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let final_aggregate = parts
        .final_aggregate
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_partial = rewrite_partial(partial_aggregate);
    let repartition = Arc::clone(&parts.between_aggregate_repartition)
        .replace_children(vec![rewritten_partial], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite_final(final_aggregate)]));
    let rewritten_final = rewritten_final
        .replace_children(vec![repartition], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

pub(super) fn with_distributed_final_input_replaced_by_scan(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate = Arc::clone(&parts.final_aggregate)
        .replace_children(
            vec![Arc::clone(&parts.scan_input)],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    rebuild_default_distributed_sql_plan(&parts, final_aggregate)
}

pub(super) fn parquet_scan_with_pushed_down_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_pushed_down_predicate_value(path, 10)
}

pub(super) fn parquet_scan_with_source_limit(scan: Arc<dyn ExecutionPlan>, limit: usize) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_limit(Some(limit)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_without_files(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan =
        FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(vec![FileGroup::new(vec![])]).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn replace_parquet_scans_with_empty_file_groups(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if let Some(data_source) = plan.downcast_ref::<DataSourceExec>()
        && data_source.downcast_to_file_source::<ParquetSource>().is_some()
    {
        return parquet_scan_without_files(Arc::clone(&plan));
    }
    let children = plan.children();
    if children.is_empty() {
        return plan;
    }
    let children = children
        .into_iter()
        .map(|child| replace_parquet_scans_with_empty_file_groups(Arc::clone(child)))
        .collect::<Vec<_>>();
    plan.replace_children(children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute)).unwrap()
}

pub(super) fn parquet_scan_with_pushed_down_predicate_value(
    path: &std::path::Path,
    value: i32,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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

pub(super) fn parquet_scan_with_pushed_down_decimal_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(ScalarValue::Decimal128(Some(1_000), 15, 2)),
        &scan_schema,
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_decimal_equality_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_decimal_in_list_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate = in_list(
        col("payload", &scan_schema).unwrap(),
        vec![lit(ScalarValue::Decimal128(Some(500), 7, 2)), lit(ScalarValue::Decimal128(Some(1_000), 7, 2))],
        &false,
        &scan_schema,
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_case_decimal_divide_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let table_schema = scan.schema();
    let vehicle_count = col("hd_vehicle_count", &table_schema).unwrap();
    let dep_count = col("hd_dep_count", &table_schema).unwrap();
    let zero = lit(ScalarValue::Int32(Some(0)));
    let vehicle_positive = binary(Arc::clone(&vehicle_count), Operator::Gt, zero, &table_schema).unwrap();
    let dep_decimal: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(dep_count, DataType::Decimal128(18, 3), None));
    let vehicle_decimal: Arc<dyn PhysicalExpr> =
        Arc::new(CastExpr::new(vehicle_count, DataType::Decimal128(18, 0), None));
    let ratio = binary(dep_decimal, Operator::Divide, vehicle_decimal, &table_schema).unwrap();
    let case_ratio: Arc<dyn PhysicalExpr> =
        Arc::new(CaseExpr::try_new(None, vec![(vehicle_positive, ratio)], None).unwrap());
    let ratio_type = case_ratio.data_type(table_schema.as_ref()).unwrap();
    let DataType::Decimal128(precision, scale) = ratio_type else {
        panic!("CASE decimal divide should declare Decimal128 output, got {ratio_type:?}");
    };
    let threshold = lit(ScalarValue::Decimal128(Some(10_i128.pow(u32::try_from(scale).unwrap())), precision, scale));
    let predicate = binary(case_ratio, Operator::Gt, threshold, &table_schema).unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_projection_indices(Some(vec![0, 2, 4]))
        .unwrap()
        .with_source(Arc::new(parquet_source))
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_integer_divide_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let table_schema = scan.schema();
    // Build an integer divide pushed predicate to cover the scan-only fallback path.
    let dep_count = col("hd_dep_count", &table_schema).unwrap();
    let vehicle_count = col("hd_vehicle_count", &table_schema).unwrap();
    let ratio = binary(dep_count, Operator::Divide, vehicle_count, &table_schema).unwrap();
    let predicate = binary(ratio, Operator::Gt, lit(ScalarValue::Int32(Some(0))), &table_schema).unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone())
        .with_projection_indices(Some(vec![0, 2, 4]))
        .unwrap()
        .with_source(Arc::new(parquet_source))
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_projection_and_pushed_label_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_projection_indices(path, vec![0, 4, 2, 3]);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate = binary(
        col("label", &scan_schema).unwrap(),
        Operator::NotEq,
        lit(ScalarValue::Utf8View(Some("z".to_string()))),
        &scan_schema,
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_edge_scan_with_projection_and_pushed_src_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_projection_indices(path, vec![1, 2, 3]);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate =
        binary(col("src", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int64(Some(10))), &scan_schema)
            .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_label_prefix_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_projection_indices(path, vec![0, 4, 2, 3]);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let prefix: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
        "substr",
        unicode::substr(),
        vec![col("label", &scan_schema).unwrap(), lit(ScalarValue::Int64(Some(1))), lit(ScalarValue::Int64(Some(1)))],
        Arc::new(Field::new("substr", DataType::Utf8View, true)),
        Arc::new(ConfigOptions::new()),
    ));
    let predicate = in_list(
        prefix,
        vec![lit(ScalarValue::Utf8View(Some("a".to_string()))), lit(ScalarValue::Utf8View(Some("b".to_string())))],
        &false,
        scan_schema.as_ref(),
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_pushed_down_partial_reader_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let payload_eq =
        binary(col("payload", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let label_like: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
        false,
        false,
        col("label", &scan_schema).unwrap(),
        lit(ScalarValue::Utf8(Some("a%".to_string()))),
    ));
    let predicate = binary(payload_eq, Operator::And, label_like, &scan_schema).unwrap();
    let parquet_source = parquet_source.with_predicate(predicate).with_pushdown_filters(true);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_access_plan(
    path: &std::path::Path,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(group.files().iter().cloned().map(|file| file.with_extension(access_plan.clone())).collect())
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_byte_range(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| vec![(0, size)])
}

pub(super) fn parquet_scan_with_explicit_byte_ranges(
    path: &std::path::Path,
    ranges: Vec<(i64, i64)>,
) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, move |_size| ranges.clone())
}

pub(super) fn parquet_scan_with_explicit_byte_ranges_and_access_plan(
    path: &std::path::Path,
    ranges: Vec<(i64, i64)>,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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

pub(super) fn parquet_scan_with_named_file_byte_ranges(
    path: &std::path::Path,
    file_name: &str,
    ranges: Vec<(i64, i64)>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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
                        let location = file.object_meta.location.as_ref().to_string();
                        if location.ends_with(file_name) {
                            ranges.iter().map(|(start, end)| file.clone().with_range(*start, *end)).collect::<Vec<_>>()
                        } else {
                            vec![file.clone()]
                        }
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_partial_byte_range_access_plan_and_extra_extension(
    path: &std::path::Path,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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
                    .cloned()
                    .map(|file| {
                        let size = file.object_meta.size as i64;
                        file.with_range(0, size / 2)
                            .with_extension(access_plan.clone())
                            .with_extension(String::from("unsupported"))
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_split_full_byte_ranges(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| {
        let midpoint = size / 2;
        vec![(0, midpoint), (midpoint, size)]
    })
}

pub(super) fn parquet_scan_with_overlapping_byte_ranges(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| {
        let midpoint = size / 2;
        vec![(0, midpoint + 1), (midpoint, size)]
    })
}

pub(super) fn parquet_scan_with_byte_ranges(
    path: &std::path::Path,
    ranges: impl Fn(i64) -> Vec<(i64, i64)>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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
                        let size = file.object_meta.size as i64;
                        ranges(size)
                            .into_iter()
                            .map(|(start, end)| file.clone().with_range(start, end))
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_object_store_url(
    path: &std::path::Path,
    object_store_url: ObjectStoreUrl,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::new(object_store_url, Arc::clone(file_scan.file_source()))
        .with_file_groups(file_scan.file_groups.clone())
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_object_store_url_and_location(
    path: &std::path::Path,
    object_store_url: ObjectStoreUrl,
    location: &str,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
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
                    .cloned()
                    .map(|mut file| {
                        file.object_meta.location = location.into();
                        file
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::new(object_store_url, Arc::clone(file_scan.file_source()))
        .with_file_groups(file_groups)
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_partition_columns(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_partition_field(path, Field::new("part", DataType::Int32, false))
}

pub(super) fn parquet_scan_with_partition_value(path: &std::path::Path, value: ScalarValue) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_typed_partition_value(path, Field::new("part", DataType::Int32, false), value)
}

pub(super) fn parquet_scan_with_typed_partition_value(
    path: &std::path::Path,
    partition_field: Field,
    value: ScalarValue,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_partition_field(path, partition_field);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group.files().iter().cloned().map(|file| file.with_partition_values(vec![value.clone()])).collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_partition_field(
    path: &std::path::Path,
    partition_field: Field,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let projected_indices = (0..=file_scan.file_schema().fields().len()).collect::<Vec<_>>();
    let table_schema = TableSchema::builder(Arc::clone(file_scan.file_schema()))
        .with_table_partition_cols(vec![Arc::new(partition_field)])
        .build();
    let parquet_source = ParquetSource::new(table_schema);
    let file_scan = FileScanConfigBuilder::new(file_scan.object_store_url.clone(), Arc::new(parquet_source))
        .with_file_groups(file_scan.file_groups.clone())
        .with_projection_indices(Some(projected_indices))
        .unwrap()
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn partition_column_group_by_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("part", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int32(Some(7))), &scan_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("part", &filter_schema).unwrap(), alias: "part".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("part", &projection_schema).unwrap(), "part".to_string())]);
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
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("part", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}
