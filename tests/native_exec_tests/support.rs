use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, NullEquality, ScalarValue},
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
use datafusion_nexus::{
    CudfOptimizerConfig, GpuFallbackPolicy, NativeCudfExecutionConfig, TPCH_TABLES, tpch_report_session_config,
};
use nexus_query_engine::{
    ColumnRef, GpuResidentSource, JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LocalParquetFile,
    LocalParquetSource, QueryPlan, QueryPlanBuilder, SourceNode,
};

pub(super) fn bounded_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

pub(super) fn native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(bounded_config())
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

pub(super) fn bounded_left_build_left_semi_join_host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")])
        .with_build_side(JoinSide::Left);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

pub(super) fn local_parquet_gpu_resident_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_gpu_resident(source)
}

pub(super) fn gpu_resident_host_sink_plan(input_name: &str, schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
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

pub(super) fn adapter_error(error: &DataFusionError) -> &datafusion_nexus::Error {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source.downcast_ref::<datafusion_nexus::Error>().expect("DataFusion external error should preserve adapter Error")
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
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

pub(super) fn join_aggregate_sort_limit_sql_plan(
    left: &std::path::Path,
    right: &std::path::Path,
) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
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
    let sql = "
SELECT l.key, SUM(l.m00 + r.m01) AS sum_x00
FROM left_input l
JOIN right_input r ON l.key = r.key
WHERE l.payload >= 10 AND r.payload >= 10
GROUP BY l.key
ORDER BY sum_x00 DESC, l.key ASC
LIMIT 2
";
    let df = futures::executor::block_on(ctx.sql(sql)).unwrap();
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

pub(super) fn write_join_left_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_payload", DataType::Int32, false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
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

pub(super) fn write_join_right_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_payload", DataType::Int32, false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
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
