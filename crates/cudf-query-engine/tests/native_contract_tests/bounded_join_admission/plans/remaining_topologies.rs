use super::*;

pub(in crate::bounded_join_admission) fn aggregate_over_join_subplan_topk_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "aggregate-over-join-topk-a");
    let b = source(&mut builder, "aggregate-over-join-topk-b");
    let c = source(&mut builder, "aggregate-over-join-topk-c");
    let d = source(&mut builder, "aggregate-over-join-topk-d");
    let e = source(&mut builder, "aggregate-over-join-topk-e");
    let f = source(&mut builder, "aggregate-over-join-topk-f");

    let ab = join(&mut builder, a, b, "ab");
    let abc = join(&mut builder, ab, c, "abc");
    let left_tree = join(&mut builder, abc, d, "left_tree");
    let aggregate_join_input = join(&mut builder, e, f, "aggregate_join_input");
    let aggregate = grouped_aggregate(&mut builder, aggregate_join_input);
    let aggregate_projection = project_key_value(&mut builder, aggregate);
    let root = join(&mut builder, left_tree, aggregate_projection, "root");
    let projection = project_key_value(&mut builder, root);
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
        key_value_schema(),
    );
    builder.finish_with_host_sink(sort)
}

pub(in crate::bounded_join_admission) fn resident_grouped_aggregate_input_post_join_aggregate_topk_plan()
-> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "resident-aggregate-post-aggregate-topk-a");
    let b = source(&mut builder, "resident-aggregate-post-aggregate-topk-b");
    let c = source(&mut builder, "resident-aggregate-post-aggregate-topk-c");
    let lineitem = source(&mut builder, "resident-aggregate-post-aggregate-topk-lineitem");

    let grouped_input = grouped_aggregate(&mut builder, lineitem);
    let ab = join(&mut builder, a, b, "ab");
    let right_tree = join(&mut builder, ab, c, "right_tree");
    let root = join(&mut builder, grouped_input, right_tree, "root");
    let post_join_aggregate = grouped_aggregate(&mut builder, root);
    let projection = project_key_value(&mut builder, post_join_aggregate);
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)]),
        key_value_schema(),
    );
    builder.finish_with_host_sink(sort)
}

pub(in crate::bounded_join_admission) fn resident_grouped_aggregate_subquery_filter_plan()
-> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "resident-aggregate-subquery-filter-a");
    let b = source(&mut builder, "resident-aggregate-subquery-filter-b");
    let c = source(&mut builder, "resident-aggregate-subquery-filter-c");
    let d = source(&mut builder, "resident-aggregate-subquery-filter-d");
    let lineitem = source(&mut builder, "resident-aggregate-subquery-filter-lineitem");

    let left_tree = join(&mut builder, a, b, "left_tree");
    let grouped_input = grouped_aggregate(&mut builder, lineitem);
    let grouped_projection = project_key_value(&mut builder, grouped_input);
    let cd = join(&mut builder, c, d, "cd");
    let right_tree = join(&mut builder, grouped_projection, cd, "right_tree");
    let root = join(&mut builder, left_tree, right_tree, "root");
    let sort =
        builder.add_sort(root, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]), key_value_schema());
    builder.finish_with_host_sink(sort)
}

fn source(builder: &mut QueryPlanBuilder, label: &str) -> cudf_query_engine::NodeId {
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/{label}.parquet"),
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn join(
    builder: &mut QueryPlanBuilder,
    left: cudf_query_engine::NodeId,
    right: cudf_query_engine::NodeId,
    _value_name: &str,
) -> cudf_query_engine::NodeId {
    let output_schema = key_value_schema();
    builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    )
}

fn grouped_aggregate(builder: &mut QueryPlanBuilder, input: cudf_query_engine::NodeId) -> cudf_query_engine::NodeId {
    builder.add_aggregate(
        input,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
        ),
        key_value_schema(),
    )
}

fn project_key_value(builder: &mut QueryPlanBuilder, input: cudf_query_engine::NodeId) -> cudf_query_engine::NodeId {
    builder.add_projection(input, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]))
}
