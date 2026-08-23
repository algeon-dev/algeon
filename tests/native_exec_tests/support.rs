use std::sync::Arc;

use arrow::{
    array::{Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{NullEquality, ScalarValue},
    datasource::{
        physical_plan::{FileScanConfigBuilder, ParquetSource},
        source::DataSourceExec,
        table_schema::TableSchema,
    },
    execution::object_store::ObjectStoreUrl,
    functions_aggregate::{count::count_udaf, sum::sum_udaf},
    logical_expr::Operator,
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
use nexus_workloads::tpch::TPCH_TABLES;

use crate::tpch_session::tpch_report_session_config;

pub(super) fn bounded_config() -> NexusNativeExecutionConfig {
    NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

pub(super) fn native_optimizer_config() -> NexusNativeOptimizerConfig {
    NexusNativeOptimizerConfig::default().with_native_execution_config(bounded_config())
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

pub(super) fn join_aggregate_sort_sql_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    super::support_core::join_aggregate_sort_sql_plan_with_limit(left, right, false)
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
    let aggregate = count_only_plan(scan);
    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("count_rows", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}

pub(super) fn count_only_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&scan_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(count)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    )
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

pub(super) use super::support_core::{
    adapter_error, ilike_filter_over_scan_with_pattern, ilike_filter_plan, ilike_filter_plan_with_pattern, int32_array,
    int64_array, join_aggregate_sort_limit_sql_plan, parquet_file_schema, parquet_scan,
    parquet_scan_with_target_partitions, unique_temp_dir, write_batch, write_ilike_parquet_file,
    write_ilike_parquet_file_with_labels, write_join_left_parquet_file, write_join_right_parquet_file,
    write_parquet_file,
};

pub(super) fn ilike_filter_over_scan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    ilike_filter_over_scan_with_pattern(scan, "al%".to_string())
}
