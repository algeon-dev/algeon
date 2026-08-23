use std::{path::Path, sync::Arc};

use super::*;

#[test]
fn optimized_aggregate_join_pushdown_matches_plain_sum_inner_join_with_duplicate_keys_and_nulls() {
    let dir = unique_temp_dir("nexus-query-engine-aggregate-join-pushdown-parity");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

    let left_schema = write_aggregate_join_pushdown_left_parquet(&runtime, &left_path);
    let right_schema = write_aggregate_join_pushdown_right_parquet(&runtime, &right_path);
    let plan = aggregate_join_pushdown_execution_plan(&left_path, &left_schema, &right_path, &right_schema);
    let optimizer_result = nexus_query_engine::optimizer::optimize_query_plan(plan.clone(), &Default::default());
    assert_eq!(optimizer_result.change, nexus_query_engine::optimizer::NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let original = execute_query_plan(&context, &plan).unwrap();
    let optimized = execute_query_plan(&context, &optimizer_result.plan).unwrap();

    let QueryOutput::HostArrow(original_batch) = original.output() else {
        panic!("original aggregate-join plan should materialize host output");
    };
    let QueryOutput::HostArrow(optimized_batch) = optimized.output() else {
        panic!("optimized aggregate-join plan should materialize host output");
    };
    assert_eq!(original_batch.schema(), optimized_batch.schema());
    assert_eq!(
        string_array(original_batch, 0).iter().collect::<Vec<_>>(),
        string_array(optimized_batch, 0).iter().collect::<Vec<_>>()
    );
    assert_eq!(
        int64_array(original_batch, 1).iter().collect::<Vec<_>>(),
        int64_array(optimized_batch, 1).iter().collect::<Vec<_>>()
    );
    assert_eq!(string_array(optimized_batch, 0).iter().collect::<Vec<_>>(), vec![Some("A"), Some("B")]);
    assert_eq!(int64_array(optimized_batch, 1).iter().collect::<Vec<_>>(), vec![Some(25), Some(14)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn optimized_aggregate_join_pushdown_matches_left_join_count_with_unmatched_rows() {
    let dir = unique_temp_dir("nexus-query-engine-aggregate-join-pushdown-left-count-parity");
    let left_path = dir.join("customer.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

    let left_schema = write_left_count_customer_parquet(&runtime, &left_path);
    let right_schema = write_left_count_orders_parquet(&runtime, &right_path);
    let plan = aggregate_join_pushdown_left_count_plan(&left_path, &left_schema, &right_path, &right_schema);
    let optimizer_result = nexus_query_engine::optimizer::optimize_query_plan(plan.clone(), &Default::default());
    assert_eq!(optimizer_result.change, nexus_query_engine::optimizer::NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let original = execute_query_plan(&context, &plan).unwrap();
    let optimized = execute_query_plan(&context, &optimizer_result.plan).unwrap();

    let QueryOutput::HostArrow(original_batch) = original.output() else {
        panic!("original left count aggregate-join plan should materialize host output");
    };
    let QueryOutput::HostArrow(optimized_batch) = optimized.output() else {
        panic!("optimized left count aggregate-join plan should materialize host output");
    };
    assert_eq!(original_batch.schema(), optimized_batch.schema());
    assert_eq!(
        int32_array(original_batch, 0).iter().collect::<Vec<_>>(),
        int32_array(optimized_batch, 0).iter().collect::<Vec<_>>()
    );
    assert_eq!(
        int64_array(original_batch, 1).iter().collect::<Vec<_>>(),
        int64_array(optimized_batch, 1).iter().collect::<Vec<_>>()
    );
    assert_eq!(int32_array(optimized_batch, 0).iter().collect::<Vec<_>>(), vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(int64_array(optimized_batch, 1).iter().collect::<Vec<_>>(), vec![Some(1), Some(4), Some(0), Some(0)]);

    fs::remove_dir_all(dir).unwrap();
}

fn aggregate_join_pushdown_execution_plan(
    left_path: &Path,
    left_schema: &Arc<Schema>,
    right_path: &Path,
    right_schema: &Arc<Schema>,
) -> nexus_query_engine::plan::QueryPlan {
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_net_profit", DataType::Int64, true),
        Field::new("s_store_name", DataType::Utf8, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("s_store_name", DataType::Utf8, true),
        Field::new("sum_profit", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_source(
        left_path,
        left_schema,
        &["zip_prefix", "s_store_name", "ss_net_profit"],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(local_source(right_path, right_schema, &["zip_prefix"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "zip_prefix", "zip_prefix"),
                nexus_query_engine::expr::ColumnRef::input(1, "zip_prefix", "zip_prefix"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "ss_net_profit", "ss_net_profit"),
                "ss_net_profit",
            ),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "s_store_name", "s_store_name"),
                "s_store_name",
            ),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let projection = builder.add_projection(
        join,
        vec![
            ProjectionOutput::new(Expr::column("s_store_name"), Field::new("s_store_name", DataType::Utf8, true)),
            ProjectionOutput::new(
                Expr::column("ss_net_profit"),
                Field::new("__nexus_native_aggregate_arg_0", DataType::Int64, true),
            ),
        ],
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("s_store_name")],
            vec![AggregateExpr::new(
                AggregateFunction::Sum,
                Expr::column("__nexus_native_aggregate_arg_0"),
                "sum_profit",
            )],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("s_store_name"), false, false)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
}

fn aggregate_join_pushdown_left_count_plan(
    left_path: &Path,
    left_schema: &Arc<Schema>,
    right_path: &Path,
    right_schema: &Arc<Schema>,
) -> nexus_query_engine::plan::QueryPlan {
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_source(left_path, left_schema, &["c_custkey"])));
    let right = builder.add_source(SourceNode::LocalParquet(local_source(
        right_path,
        right_schema,
        &["o_custkey", "o_orderkey"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "c_custkey", "c_custkey"),
                nexus_query_engine::expr::ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "c_custkey", "c_custkey"),
                "c_custkey",
            ),
            JoinOutputColumn::right(
                nexus_query_engine::expr::ColumnRef::input(1, "o_orderkey", "o_orderkey"),
                "o_orderkey",
            ),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("c_custkey"), false, true)]),
        aggregate_schema,
    );
    builder.finish_with_host_sink(sort)
}

fn local_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn write_aggregate_join_pushdown_left_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("zip_prefix", DataType::Int32, true),
        Field::new("s_store_name", DataType::Utf8, true),
        Field::new("ss_net_profit", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(3), Some(3), None])),
            Arc::new(StringArray::from(vec![Some("A"), Some("A"), Some("A"), Some("B"), Some("B"), Some("B")])),
            Arc::new(Int64Array::from(vec![Some(10), None, Some(5), Some(7), None, Some(100)])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_aggregate_join_pushdown_right_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("zip_prefix", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(3), Some(3), None]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_left_count_customer_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(2), Some(3), Some(4)]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_left_count_orders_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(2), Some(2), Some(5)])),
            Arc::new(Int64Array::from(vec![Some(10), None, Some(20), Some(21), None, Some(50)])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}
