use std::sync::Arc;

use query_engine::{expr::ColumnRef, rapids_backend::rapids_analysis_context};

use super::*;

#[test]
fn bounded_native_query_executes_mixed_inner_multi_join_pipeline_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-mixed-inner-multi-join");
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
    let first_join = builder.add_join(
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
    );
    let second_join = builder.add_join(
        first_join,
        regions,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(ColumnRef::input(1, "region", "region"), "region"),
            ])
            .with_build_side(JoinSide::Right),
        second_join_schema,
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("region")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("region"), false, false)]),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_reusable_hash_join_single_pass_probe");
    assert_eq!(join_metrics[1].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[1].join_type, "inner");
    assert_eq!(join_metrics[1].mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert!(join_metrics[1].gather_map_bytes > 0);
    assert_eq!(join_metrics[1].output_rows, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("east")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1]);

    fs::remove_dir_all(dir).unwrap();
}
