use super::*;

pub(crate) fn aggregate_over_join_subplan_topk_plan() -> nexus_query_engine::QueryPlan {
    aggregate_over_join_subplan_topk_plan_inner(AggregateOverJoinSubplanTopKVariant::Admitted)
}

pub(crate) fn aggregate_over_join_subplan_topk_variant_plan(
    variant: AggregateOverJoinSubplanTopKVariant,
) -> nexus_query_engine::QueryPlan {
    aggregate_over_join_subplan_topk_plan_inner(variant)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AggregateOverJoinSubplanTopKVariant {
    Admitted,
    MissingFetchLimit,
    PostRootFilteringJoin,
    ScalarAggregateSubplan,
    ShallowMultiJoinSide,
    UnsupportedPostRootProjectionExpression,
}

fn aggregate_over_join_subplan_topk_plan_inner(
    variant: AggregateOverJoinSubplanTopKVariant,
) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "aggregate-over-join-topk-a");
    let b = source(&mut builder, "aggregate-over-join-topk-b");
    let c = source(&mut builder, "aggregate-over-join-topk-c");
    let d = source(&mut builder, "aggregate-over-join-topk-d");
    let e = source(&mut builder, "aggregate-over-join-topk-e");
    let f = source(&mut builder, "aggregate-over-join-topk-f");

    let ab = join(&mut builder, a, b, "ab");
    let left_tree = match variant {
        AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide => ab,
        AggregateOverJoinSubplanTopKVariant::Admitted
        | AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin
        | AggregateOverJoinSubplanTopKVariant::MissingFetchLimit
        | AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan
        | AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression => {
            let abc = join(&mut builder, ab, c, "abc");
            join(&mut builder, abc, d, "left_tree")
        },
    };
    let aggregate_join_input = join(&mut builder, e, f, "aggregate_join_input");
    let aggregate = match variant {
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan => {
            scalar_aggregate(&mut builder, aggregate_join_input)
        },
        AggregateOverJoinSubplanTopKVariant::Admitted
        | AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin
        | AggregateOverJoinSubplanTopKVariant::MissingFetchLimit
        | AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression
        | AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide => {
            grouped_aggregate(&mut builder, aggregate_join_input)
        },
    };
    let aggregate_projection = project_key_value(&mut builder, aggregate);
    let root = join(&mut builder, left_tree, aggregate_projection, "root");
    let root = if matches!(variant, AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin) {
        let post_root_filter = source(&mut builder, "aggregate-over-join-topk-post-root-filter");
        join(&mut builder, root, post_root_filter, "post_root_filter")
    } else {
        root
    };
    let projection = if matches!(variant, AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression)
    {
        builder.add_projection(
            root,
            projection_outputs(vec![Expr::column("k"), Expr::case(Vec::new(), Expr::column("v"))], key_value_schema()),
        )
    } else {
        project_key_value(&mut builder, root)
    };
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
        key_value_schema(),
    );
    match variant {
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit => builder.finish_with_host_sink(sort),
        AggregateOverJoinSubplanTopKVariant::Admitted
        | AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin
        | AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan
        | AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide
        | AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100), key_value_schema());
            builder.finish_with_host_sink(limit)
        },
    }
}

pub(crate) fn resident_grouped_aggregate_input_post_join_aggregate_topk_plan() -> nexus_query_engine::QueryPlan {
    resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted,
    )
}

pub(crate) fn resident_grouped_aggregate_input_post_join_aggregate_topk_variant_plan(
    variant: ResidentGroupedAggregateInputPostJoinAggregateTopKVariant,
) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "resident-aggregate-post-aggregate-topk-a");
    let b = source(&mut builder, "resident-aggregate-post-aggregate-topk-b");
    let c = source(&mut builder, "resident-aggregate-post-aggregate-topk-c");
    let lineitem = source(&mut builder, "resident-aggregate-post-aggregate-topk-lineitem");

    let grouped_input = grouped_aggregate(&mut builder, lineitem);
    let mut grouped_input = builder.add_projection(
        grouped_input,
        projection_outputs(
            vec![Expr::column("k")],
            Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)])),
        ),
    );
    if matches!(variant, ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter) {
        grouped_input = builder.add_filter(
            grouped_input,
            Expr::in_list(Expr::column("k"), Vec::new()),
            Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)])),
        );
    }
    let ab = join(&mut builder, a, b, "ab");
    let right_tree = join(&mut builder, ab, c, "right_tree");
    let root = semi_join(&mut builder, right_tree, grouped_input);
    let root = project_key_value(&mut builder, root);
    let post_join_aggregate = match variant {
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate => {
            scalar_aggregate(&mut builder, root)
        },
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter => {
            grouped_aggregate(&mut builder, root)
        },
    };
    let projection = project_key_value(&mut builder, post_join_aggregate);
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)]),
        key_value_schema(),
    );
    match variant {
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit => {
            builder.finish_with_host_sink(sort)
        },
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100), key_value_schema());
            builder.finish_with_host_sink(limit)
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ResidentGroupedAggregateInputPostJoinAggregateTopKVariant {
    Admitted,
    MissingFetchLimit,
    ScalarPostJoinAggregate,
    UnsupportedGroupedInputFilter,
}

pub(crate) fn resident_grouped_aggregate_subquery_filter_plan() -> nexus_query_engine::QueryPlan {
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
    let right_tree = residual_join_against_grouped_aggregate(&mut builder, grouped_projection, cd);
    let root = semi_join(&mut builder, left_tree, right_tree);
    let sort =
        builder.add_sort(root, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]), key_value_schema());
    builder.finish_with_host_sink(sort)
}

pub(crate) fn shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source(&mut builder, "shared-grouped-aggregate-fanout-a");
    let b = source(&mut builder, "shared-grouped-aggregate-fanout-b");
    let lineitem = source(&mut builder, "shared-grouped-aggregate-fanout-lineitem");

    let shared_grouped = grouped_aggregate(&mut builder, lineitem);
    let shared_for_join = project_key_value(&mut builder, shared_grouped);
    let ab = join(&mut builder, a, b, "ab");
    let joined_branch = join(&mut builder, ab, shared_for_join, "joined_shared");
    let joined_branch = project_key_value(&mut builder, joined_branch);

    let shared_for_aggregate = project_key_value(&mut builder, shared_grouped);
    let aggregate_branch = grouped_aggregate(&mut builder, shared_for_aggregate);
    let aggregate_branch = project_key_value(&mut builder, aggregate_branch);

    let root = join(&mut builder, joined_branch, aggregate_branch, "root");
    let sort =
        builder.add_sort(root, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]), key_value_schema());
    builder.finish_with_host_sink(sort)
}

fn source(builder: &mut QueryPlanBuilder, label: &str) -> nexus_query_engine::NodeId {
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/{label}.parquet"),
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn join(
    builder: &mut QueryPlanBuilder,
    left: nexus_query_engine::NodeId,
    right: nexus_query_engine::NodeId,
    _value_name: &str,
) -> nexus_query_engine::NodeId {
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

fn semi_join(
    builder: &mut QueryPlanBuilder,
    left: nexus_query_engine::NodeId,
    right: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        key_value_schema(),
    )
}

fn residual_join_against_grouped_aggregate(
    builder: &mut QueryPlanBuilder,
    grouped: nexus_query_engine::NodeId,
    subtree: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    builder.add_join(
        grouped,
        subtree,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(1, "v", "v"),
            BinaryOp::Gt,
            Expr::input_column(0, "v", "v"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "k", "k"), "k"),
            JoinOutputColumn::right(ColumnRef::input(1, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Left),
        key_value_schema(),
    )
}

fn grouped_aggregate(builder: &mut QueryPlanBuilder, input: nexus_query_engine::NodeId) -> nexus_query_engine::NodeId {
    builder.add_aggregate(
        input,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
        ),
        key_value_schema(),
    )
}

fn scalar_aggregate(builder: &mut QueryPlanBuilder, input: nexus_query_engine::NodeId) -> nexus_query_engine::NodeId {
    builder.add_aggregate(
        input,
        AggregateSpec::new(vec![], vec![AggregateExpr::new(AggregateFunction::Min, Expr::column("v"), "v")]),
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
    )
}

fn project_key_value(builder: &mut QueryPlanBuilder, input: nexus_query_engine::NodeId) -> nexus_query_engine::NodeId {
    builder.add_projection(input, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]))
}
