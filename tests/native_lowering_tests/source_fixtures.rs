use super::*;

pub(super) fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn parquet_scan_with_projection_indices(
    path: &std::path::Path,
    projected_indices: Vec<usize>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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

pub(super) fn parquet_scan_with_missing_nullable_column(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    if plan.as_any().is::<ProjectionExec>()
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
    let sort_plan = plan;
    let sort = sort_plan.as_any().downcast_ref::<SortExec>().expect("SQL plan root should be SortExec");
    let post_projection_plan = Arc::clone(sort.input());
    let post_projection = post_projection_plan
        .as_any()
        .downcast_ref::<ProjectionExec>()
        .expect("SQL plan should have a post-aggregate ProjectionExec");
    let aggregate_plan = Arc::clone(post_projection.input());
    let aggregate = aggregate_plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("SQL plan should have an AggregateExec below post projection");
    let filter_plan = Arc::clone(aggregate.input());
    let filter =
        filter_plan.as_any().downcast_ref::<FilterExec>().expect("SQL plan should have a FilterExec below aggregate");
    let source_plan = Arc::clone(filter.input());
    let source = source_plan
        .as_any()
        .downcast_ref::<DataSourceExec>()
        .expect("SQL plan filter input should be a DataSourceExec");

    let source_without_predicate = parquet_source_exec_without_predicate(source);
    let filter_without_predicate = filter_plan.with_new_children(vec![source_without_predicate]).unwrap();
    let aggregate_without_predicate = aggregate_plan.with_new_children(vec![filter_without_predicate]).unwrap();
    let projection_without_predicate =
        post_projection_plan.with_new_children(vec![aggregate_without_predicate]).unwrap();
    sort_plan.with_new_children(vec![projection_without_predicate]).unwrap()
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

impl datafusion::datasource::physical_plan::ParquetFileReaderFactory for RejectingParquetFileReaderFactory {
    fn create_reader(
        &self,
        _partition_index: usize,
        _partitioned_file: datafusion::datasource::listing::PartitionedFile,
        _metadata_size_hint: Option<usize>,
        _metrics: &datafusion::physical_plan::metrics::ExecutionPlanMetricsSet,
    ) -> datafusion::common::Result<Box<dyn datafusion::parquet::arrow::async_reader::AsyncFileReader + Send>> {
        Err(datafusion::common::DataFusionError::Execution(
            "test reader factory should not be invoked during native lowering".to_string(),
        ))
    }
}

pub(super) fn parquet_scan_with_custom_reader_factory(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let parquet_source = ParquetSource::new(file_scan.file_source().table_schema().clone())
        .with_parquet_file_reader_factory(Arc::new(RejectingParquetFileReaderFactory));
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
    let root_spm = root.as_any().downcast_ref::<SortPreservingMergeExec>().expect("default SQL root should be SPM");
    let sort = Arc::clone(root_spm.input());
    let sort_exec = sort.as_any().downcast_ref::<SortExec>().expect("default SQL SPM input should be SortExec");
    let projection = Arc::clone(sort_exec.input());
    let projection_exec =
        projection.as_any().downcast_ref::<ProjectionExec>().expect("default SQL sort input should be ProjectionExec");
    let final_aggregate = Arc::clone(projection_exec.input());
    let final_aggregate_exec = final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL projection input should be final AggregateExec");
    assert!(matches!(final_aggregate_exec.mode(), AggregateMode::FinalPartitioned));
    let between_aggregate_repartition = Arc::clone(final_aggregate_exec.input());
    let repartition = between_aggregate_repartition
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .expect("default SQL final aggregate input should be RepartitionExec");
    let partial_aggregate = Arc::clone(repartition.input());
    let partial_aggregate_exec = partial_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL repartition input should be partial AggregateExec");
    assert!(matches!(partial_aggregate_exec.mode(), AggregateMode::Partial));
    let filter = partial_aggregate_exec
        .input()
        .as_any()
        .downcast_ref::<FilterExec>()
        .expect("default SQL partial aggregate input should be FilterExec");
    let source_repartition = filter
        .input()
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .expect("default SQL filter input should be RepartitionExec");
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
    let projection = Arc::clone(&parts.projection).with_new_children(vec![final_aggregate]).unwrap();
    let sort = Arc::clone(&parts.sort).with_new_children(vec![projection]).unwrap();
    Arc::clone(&parts.root).with_new_children(vec![sort]).unwrap()
}

pub(super) fn with_rewritten_distributed_final_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate = parts
        .final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite(final_aggregate)]));
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

pub(super) fn with_rewritten_distributed_partial_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let partial_aggregate = parts
        .partial_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let rewritten_partial = rewrite(partial_aggregate);
    let repartition =
        Arc::clone(&parts.between_aggregate_repartition).with_new_children(vec![rewritten_partial]).unwrap();
    let final_aggregate = Arc::clone(&parts.final_aggregate).with_new_children(vec![repartition]).unwrap();
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
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let final_aggregate = parts
        .final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_partial = rewrite_partial(partial_aggregate);
    let repartition =
        Arc::clone(&parts.between_aggregate_repartition).with_new_children(vec![rewritten_partial]).unwrap();
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite_final(final_aggregate)]));
    let rewritten_final = rewritten_final.with_new_children(vec![repartition]).unwrap();
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

pub(super) fn with_distributed_final_input_replaced_by_scan(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate =
        Arc::clone(&parts.final_aggregate).with_new_children(vec![Arc::clone(&parts.scan_input)]).unwrap();
    rebuild_default_distributed_sql_plan(&parts, final_aggregate)
}

pub(super) fn parquet_scan_with_pushed_down_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_pushed_down_predicate_value(path, 10)
}

pub(super) fn parquet_scan_with_pushed_down_predicate_value(
    path: &std::path::Path,
    value: i32,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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

pub(super) fn parquet_scan_with_projection_and_pushed_label_predicate(
    path: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_projection_indices(path, vec![0, 4, 2, 3]);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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

pub(super) fn parquet_scan_with_access_plan(
    path: &std::path::Path,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group.files().iter().cloned().map(|file| file.with_extensions(Arc::new(access_plan.clone()))).collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_byte_range(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| vec![(0, size)])
}

pub(super) fn parquet_scan_with_partial_byte_range(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| vec![(0, size / 2)])
}

pub(super) fn parquet_scan_with_partial_byte_range_and_access_plan(
    path: &std::path::Path,
    access_plan: ParquetAccessPlan,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
                        file.with_range(0, size / 2).with_extensions(Arc::new(access_plan.clone()))
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::new(object_store_url, Arc::clone(file_scan.file_source()))
        .with_file_groups(file_scan.file_groups.clone())
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
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
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let projected_indices = (0..=file_scan.file_schema().fields().len()).collect::<Vec<_>>();
    let table_schema = TableSchema::new(Arc::clone(file_scan.file_schema()), vec![Arc::new(partition_field)]);
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
