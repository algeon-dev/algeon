use std::sync::Arc;

use query_engine::expr::ColumnRef;

use super::*;

#[test]
fn bounded_native_query_executes_pure_hash_multi_join_pipeline_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-pure-hash-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let orders_path = dir.join("orders.parquet");
    let region_path = dir.join("regions.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let orders_schema = write_join_right_parquet(&runtime, &orders_path);
    let region_schema = write_join_region_parquet(&runtime, &region_path);
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Utf8, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("region"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let lineitem =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &orders_path,
        &orders_schema,
        &["order_id", "category"],
    )));
    let regions = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &region_path,
        &region_schema,
        &["order_id", "region"],
    )));
    let first_join = builder
        .add_join(
            lineitem,
            orders,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_build_side(JoinSide::Right),
            first_join_schema,
        )
        .unwrap();
    let second_join = builder
        .add_join(
            first_join,
            regions,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "region", "region"), "region"),
                ])
                .with_build_side(JoinSide::Right),
            second_join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            second_join,
            AggregateSpec::new(
                vec![Expr::column("region")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 2);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    assert!(metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).all(|join| {
        join.execution_strategy == "bounded_reusable_hash_join_single_pass_probe"
            && join.key_count == 1
            && !join.residual_predicate
            && join.gather_map_bytes > 0
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("east"), Some("west")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_keeps_partitioned_strategy_for_streamed_multi_join_probe() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-partitioned-streamed-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let orders_path = dir.join("orders.parquet");
    let region_path = dir.join("regions.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let orders_schema = write_join_right_parquet(&runtime, &orders_path);
    let region_schema = write_join_region_parquet(&runtime, &region_path);
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));

    let mut region_source = join_right_source(&region_path, &region_schema, &["order_id", "region"]);
    region_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let lineitem =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&orders_path, &orders_schema, &["order_id"])));
    let regions = builder.add_source(SourceNode::LocalParquet(region_source));
    let first_join = builder
        .add_join(
            lineitem,
            orders,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ])
                .with_build_side(JoinSide::Right),
            first_join_schema,
        )
        .unwrap();
    let second_join = builder
        .add_join(
            first_join,
            regions,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::right(ColumnRef::input(1, "region", "region"), "region"),
                ])
                .with_build_side(JoinSide::Right),
            second_join_schema,
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(second_join).unwrap();
    let context = backend_context_with_grant(8 * 1024 * 1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.one_shot_join_count, 0);
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join")
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("east"), Some("east"), Some("west")]);

    fs::remove_dir_all(dir).unwrap();
}
