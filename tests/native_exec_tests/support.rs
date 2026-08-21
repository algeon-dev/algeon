use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array, StringViewArray},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, NullEquality, ScalarValue},
    datasource::{
        physical_plan::{FileScanConfigBuilder, ParquetSource},
        source::DataSourceExec,
        table_schema::TableSchema,
    },
    execution::object_store::ObjectStoreUrl,
    functions_aggregate::{count::count_udaf, sum::sum_udaf},
    logical_expr::Operator,
    parquet::arrow::ArrowWriter,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{TryCastExpr, binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        joins::{HashJoinExec, NestedLoopJoinExec, PartitionMode},
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{native_exec::NexusNativeExecutionConfig, planner::NexusNativeOptimizerConfig};
use nexus_query_engine::{
    admission::{ComposedGraphProducerCandidate, QueryPlanCapabilityProof},
    expr::ColumnRef,
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlan, QueryPlanBuilder},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};
use nexus_workloads::tpch::TPCH_TABLES;
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::tpch_session::tpch_report_session_config;

pub(super) fn bounded_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

pub(super) fn native_optimizer_config() -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::default().with_native_execution_config(bounded_config())
}

pub(super) fn host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_with_host_sink(source)
}

pub(super) fn gpu_resident_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_gpu_resident(source)
}

pub(super) fn bounded_left_build_left_semi_join_host_sink_plan_with_paths(
    left_path: &std::path::Path,
    right_path: &std::path::Path,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source_with_exact_metadata(
        left_path,
        join_left_schema(),
        vec!["key".to_string(), "l_payload".to_string(), "l_m00".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source_with_exact_metadata(
        right_path,
        join_right_schema(),
        vec!["key".to_string(), "r_payload".to_string(), "r_m01".to_string()],
    )));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
        .with_build_side(JoinSide::Left);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

pub(super) fn admitted_native_config(
    plan: &QueryPlan,
    config: NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    let proof = QueryPlanCapabilityProof::for_query_plan(plan).expect("exact native fixture plan must issue a proof");
    let handle = super::test_gpu_backend()
        .admit_query_with_plan_capability_proof(proof)
        .expect("exact native fixture proof must admit");
    config.with_query_service_handle(Arc::new(handle))
}

pub(super) fn admitted_composed_config(
    consumer_plan: &QueryPlan,
    producer_input_name: &str,
    producer_plan: &QueryPlan,
    config: NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    let producer = QueryPlanCapabilityProof::for_query_plan(producer_plan)
        .expect("exact composed producer fixture must issue a proof");
    let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(consumer_plan)
        .expect("exact composed consumer fixture must issue a proof");
    let proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        consumer_plan,
        consumer,
        [ComposedGraphProducerCandidate::new(Some(producer_input_name.to_string()), None, Some(producer))],
    )
    .expect("exact composed fixture must issue a complete proof");
    let handle = super::test_gpu_backend()
        .admit_query_with_plan_capability_proof(proof)
        .expect("exact composed fixture proof must admit");
    config.with_query_service_handle(Arc::new(handle))
}

pub(super) fn local_parquet_gpu_resident_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = local_parquet_source_with_exact_metadata(
        path,
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_gpu_resident(source)
}

pub(super) fn local_parquet_host_sink_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = local_parquet_source_with_exact_metadata(
        path,
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_with_host_sink(source)
}

pub(super) fn local_parquet_source_with_exact_metadata(
    path: &std::path::Path,
    schema: Arc<Schema>,
    read_columns: Vec<String>,
) -> LocalParquetSource {
    LocalParquetSource::from_files(vec![exact_local_parquet_file(path)], schema, read_columns)
}

fn exact_local_parquet_file(path: &std::path::Path) -> LocalParquetFile {
    let reader = SerializedFileReader::try_from(path).unwrap_or_else(|error| {
        panic!("written native fixture must be readable as Parquet: {}: {error}", path.display())
    });
    let metadata = reader.metadata();
    let file_size = fs::metadata(path)
        .unwrap_or_else(|error| panic!("written native fixture must be stat-able: {}: {error}", path.display()))
        .len();
    let row_group_rows = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows()).expect("fixture row count fits u64"))
        .collect::<Vec<_>>();
    let row_group_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.compressed_size()).expect("fixture compressed bytes fit u64"))
        .collect::<Vec<_>>();
    let projected_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.total_byte_size()).expect("fixture uncompressed bytes fit u64"))
        .collect::<Vec<_>>();
    LocalParquetFile::new(path)
        .with_file_size_bytes(file_size)
        .with_row_groups((0..metadata.num_row_groups()).collect())
        .with_row_group_rows(row_group_rows)
        .with_row_group_bytes(row_group_bytes.clone())
        .with_projected_row_group_bytes(row_group_bytes, projected_uncompressed_bytes)
}

pub(super) fn gpu_resident_host_sink_plan(input_name: &str, schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

pub(super) fn bounded_gpu_resident_left_build_left_semi_join_host_sink_plan(
    left_input_name: &str,
    right_input_name: &str,
    input_schema: Arc<Schema>,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(left_input_name, Arc::clone(&input_schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(right_input_name, input_schema)));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
        .with_build_side(JoinSide::Left);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

fn local_parquet_source() -> LocalParquetSource {
    LocalParquetSource::new("/tmp/datafusion-nexus-native-exec-test.parquet", schema(), vec!["key".to_string()])
}

pub(super) fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().unwrap()
}

pub(super) fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().unwrap()
}

pub(super) fn adapter_error(error: &DataFusionError) -> &datafusion_nexus::error::Error {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source
        .downcast_ref::<datafusion_nexus::error::Error>()
        .expect("DataFusion external error should preserve adapter Error")
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]))
}

pub(super) fn parquet_file_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]))
}

pub(super) fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_target_partitions(path, 1)
}

pub(super) fn parquet_scan_with_target_partitions(
    path: &std::path::Path,
    target_partitions: usize,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(target_partitions),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
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

/// A partition-grouped scan in its observable DataFusion 55 shape: the
/// partition column lives in the table schema (not the file schema) and the
/// scan declares the matching hash output partitioning.
pub(super) fn parquet_scan_with_partition_field(
    path: &std::path::Path,
    partition_field: Field,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let partition_field_name = partition_field.name().clone();
    let table_schema = TableSchema::builder(Arc::clone(file_scan.file_schema()))
        .with_table_partition_cols(vec![Arc::new(partition_field)])
        .build();
    let parquet_source = ParquetSource::new(table_schema.clone());
    let projected_indices = (0..=file_scan.file_schema().fields().len()).collect::<Vec<_>>();
    let partition_expr = col(&partition_field_name, table_schema.table_schema()).unwrap();
    let file_scan = FileScanConfigBuilder::new(file_scan.object_store_url.clone(), Arc::new(parquet_source))
        .with_file_groups(file_scan.file_groups.clone())
        .with_projection_indices(Some(projected_indices))
        .unwrap()
        .with_output_partitioning(Some(datafusion::physical_expr::Partitioning::Hash(vec![partition_expr], 1)))
        .build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn parquet_scan_with_declared_output_ordering(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let ordering = LexOrdering::new([PhysicalSortExpr::new(
        col("key", &scan.schema()).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .expect("declared key ordering should be non-empty");
    let data_source = scan.downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_output_ordering(vec![ordering]).build();
    DataSourceExec::from_data_source(file_scan)
}

pub(super) fn join_aggregate_sort_limit_sql_plan(
    left: &std::path::Path,
    right: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    join_aggregate_sort_sql_plan_with_limit(left, right, true)
}

pub(super) fn join_aggregate_sort_sql_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    join_aggregate_sort_sql_plan_with_limit(left, right, false)
}

pub(super) fn single_partition_sql_plan_with_parquet_tables(
    tables: &[(&str, &std::path::Path)],
    sql: &str,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
            .with_repartition_aggregations(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    for (name, path) in tables {
        futures::executor::block_on(ctx.register_parquet(*name, path.to_str().unwrap(), ParquetReadOptions::default()))
            .unwrap();
    }
    let df = futures::executor::block_on(ctx.sql(sql)).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

fn join_aggregate_sort_sql_plan_with_limit(
    left: &std::path::Path,
    right: &std::path::Path,
    limit: bool,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
            .with_repartition_aggregations(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    futures::executor::block_on(ctx.register_parquet(
        "left_input",
        left.to_str().unwrap(),
        ParquetReadOptions::default(),
    ))
    .unwrap();
    futures::executor::block_on(ctx.register_parquet(
        "right_input",
        right.to_str().unwrap(),
        ParquetReadOptions::default(),
    ))
    .unwrap();
    let limit_clause = if limit { "LIMIT 2" } else { "" };
    let sql = format!(
        "
SELECT l.key, SUM(l.m00 + r.m01) AS sum_x00
FROM left_input l
JOIN right_input r ON l.key = r.key
WHERE l.payload >= 10 AND r.payload >= 10
GROUP BY l.key
ORDER BY sum_x00 DESC, l.key ASC
 {limit_clause}
"
    );
    let df = futures::executor::block_on(ctx.sql(&sql)).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn stddev_sort_sql_plan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_aggregations(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    futures::executor::block_on(ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()))
        .unwrap();
    let sql = "
SELECT key, stddev_samp(m00) AS stdev
FROM bench
GROUP BY key
ORDER BY key
";
    let df = futures::executor::block_on(ctx.sql(sql)).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn tpch_sql_plan(root: &std::path::Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(tpch_report_session_config());
    for table in TPCH_TABLES {
        let path = root.join(format!("{table}.parquet"));
        futures::executor::block_on(ctx.register_parquet(
            *table,
            path.to_str().unwrap(),
            ParquetReadOptions::default(),
        ))
        .unwrap();
    }
    let df = futures::executor::block_on(ctx.sql(sql)).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn root_hash_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    let left_schema = left.schema();
    let right_schema = right.schema();
    Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &datafusion::logical_expr::JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

pub(super) fn root_left_semi_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    let left_schema = left.schema();
    let right_schema = right.schema();
    Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &datafusion::logical_expr::JoinType::LeftSemi,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

pub(super) fn root_left_anti_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    let left_schema = left.schema();
    let right_schema = right.schema();
    Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &datafusion::logical_expr::JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

pub(super) fn root_nested_loop_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    Arc::new(NestedLoopJoinExec::try_new(left, right, None, &datafusion::logical_expr::JoinType::Inner, None).unwrap())
}

pub(super) fn aggregate_sort_limit_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
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

pub(super) fn grouped_sum_aggregate_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
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

pub(super) fn unsupported_try_cast_projection_aggregate_sort_limit_plan(
    scan: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let unsupported_x00 = Arc::new(TryCastExpr::new(col("m00", &filter_schema).unwrap(), DataType::Int16));
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: unsupported_x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&projection_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(count)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("count_rows", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

pub(super) fn write_parquet_file(path: &std::path::Path) {
    let schema = parquet_file_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_ilike_parquet_file(path: &std::path::Path) {
    write_ilike_parquet_file_with_labels(
        path,
        &[
            Some("Alpha".to_string()),
            Some("beta".to_string()),
            Some("ALMANAC".to_string()),
            None,
            Some("zeta".to_string()),
        ],
    );
}

pub(super) fn write_ilike_parquet_file_with_labels(path: &std::path::Path, labels: &[Option<String>]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("label", DataType::Utf8View, true),
    ]));
    let ids =
        (1..=i32::try_from(labels.len()).expect("mixed-boundary fixture row count must fit i32")).collect::<Vec<_>>();
    let labels = labels.iter().map(|label| label.as_deref()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringViewArray::from(labels))],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_sorted_parquet_file(path: &std::path::Path) {
    let schema = parquet_file_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Int32Array::from(vec![5, 20, 10])),
            Arc::new(Int64Array::from(vec![100, 300, 200])),
            Arc::new(Int64Array::from(vec![1, 3, 2])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_left_parquet_file(path: &std::path::Path) {
    let schema = join_left_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn join_left_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_payload", DataType::Int32, false),
        Field::new("l_m00", DataType::Int64, false),
    ]))
}

pub(super) fn write_join_right_parquet_file(path: &std::path::Path) {
    let schema = join_right_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn join_right_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_payload", DataType::Int32, false),
        Field::new("r_m01", DataType::Int64, false),
    ]))
}

fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
