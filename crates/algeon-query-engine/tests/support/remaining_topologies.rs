use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use query_planning::{
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, ColumnRef, Expr, LimitSpec, LiteralValue, SortExpr,
        SortSpec,
    },
    plan::{
        JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, ProjectionOutput, QueryPlan, QueryPlanBuilder,
        WindowFunction, WindowSpec,
    },
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

pub(crate) fn aggregate_over_join_subplan_topk_variant_plan(variant: AggregateOverJoinSubplanTopKVariant) -> QueryPlan {
    aggregate_over_join_subplan_topk_plan_inner(variant, false)
}

/// Mutates the multi-join TopK shape where a grouped aggregate is itself built
/// over a join subplan.
#[derive(Debug, Clone, Copy)]
#[expect(dead_code, reason = "shared topology fixtures are compiled by test binaries with different coverage")]
pub(crate) enum AggregateOverJoinSubplanTopKVariant {
    Admitted,
    /// Drops the fetch from the final TopK so the final-sort contract is not
    /// bounded by a row count.
    MissingFetchLimit,
    /// Inserts a single-output keyed filtering join above the root; this is an
    /// admitted post-root stage, not a second root.
    PostRootFilteringJoin,
    /// Replaces the grouped aggregate subplan with a scalar aggregate, removing
    /// the grouping key needed by this proof.
    ScalarAggregateSubplan,
    /// Leaves one side of the root too shallow to prove the intended multi-join
    /// topology.
    ShallowMultiJoinSide,
    /// Keeps the shape close while forcing admission to report the unsupported
    /// projection expression instead of a generic topology failure.
    UnsupportedPostRootProjectionExpression,
}

pub(crate) fn aggregate_over_join_subplan_topk_plan_inner(
    variant: AggregateOverJoinSubplanTopKVariant,
    with_source_metadata: bool,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-a", with_source_metadata);
    let b = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-b", with_source_metadata);
    let e = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-e", with_source_metadata);
    let f = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-f", with_source_metadata);

    let ab = join(&mut builder, a, b, "ab");
    let left_tree = match variant {
        AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide => ab,
        AggregateOverJoinSubplanTopKVariant::Admitted
        | AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin
        | AggregateOverJoinSubplanTopKVariant::MissingFetchLimit
        | AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan
        | AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression => {
            let c = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-c", with_source_metadata);
            let d = source_with_optional_metadata(&mut builder, "aggregate-over-join-topk-d", with_source_metadata);
            let abc = join(&mut builder, ab, c, "abc");
            join(&mut builder, abc, d, "left_tree")
        },
    };
    let aggregate_join_input = join(&mut builder, e, f, "aggregate_join_input");
    let aggregate = match variant {
        AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan => {
            scalar_key_value_aggregate(&mut builder, aggregate_join_input)
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
        builder
            .add_projection(
                root,
                projection_outputs(
                    vec![Expr::column("k"), Expr::case(Vec::new(), Expr::column("v"))],
                    key_value_schema(),
                ),
            )
            .unwrap()
    } else {
        project_key_value(&mut builder, root)
    };
    let sort =
        builder.add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
    match variant {
        AggregateOverJoinSubplanTopKVariant::MissingFetchLimit => builder.finish_with_host_sink(sort).unwrap(),
        AggregateOverJoinSubplanTopKVariant::Admitted
        | AggregateOverJoinSubplanTopKVariant::PostRootFilteringJoin
        | AggregateOverJoinSubplanTopKVariant::ScalarAggregateSubplan
        | AggregateOverJoinSubplanTopKVariant::ShallowMultiJoinSide
        | AggregateOverJoinSubplanTopKVariant::UnsupportedPostRootProjectionExpression => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
            builder.finish_with_host_sink(limit).unwrap()
        },
    }
}

pub(crate) fn resident_grouped_aggregate_input_post_join_aggregate_topk_plan_inner(
    variant: ResidentGroupedAggregateInputPostJoinAggregateTopKVariant,
    with_source_metadata: bool,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a =
        source_with_optional_metadata(&mut builder, "resident-aggregate-post-aggregate-topk-a", with_source_metadata);
    let b =
        source_with_optional_metadata(&mut builder, "resident-aggregate-post-aggregate-topk-b", with_source_metadata);
    let c =
        source_with_optional_metadata(&mut builder, "resident-aggregate-post-aggregate-topk-c", with_source_metadata);
    let lineitem = source_with_optional_metadata(
        &mut builder,
        "resident-aggregate-post-aggregate-topk-lineitem",
        with_source_metadata,
    );

    let grouped_input = grouped_aggregate(&mut builder, lineitem);
    let mut grouped_input = if matches!(
        variant,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput
            | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput
    ) {
        builder.add_filter(
            grouped_input,
            Expr::binary(Expr::column("v"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0)))).unwrap()
    } else {
        builder.add_projection(
            grouped_input,
            projection_outputs(
                vec![Expr::column("k")],
                Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)])),
            ),
        ).unwrap()
    };
    if matches!(variant, ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter) {
        grouped_input = builder.add_filter(grouped_input, Expr::in_list(Expr::column("k"), Vec::new())).unwrap();
    }
    let ab = join(&mut builder, a, b, "ab");
    let right_tree = join(&mut builder, ab, c, "right_tree");
    let right_tree = if matches!(
        variant,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput
    ) {
        grouped_aggregate(&mut builder, right_tree)
    } else {
        right_tree
    };
    let root = semi_join(&mut builder, right_tree, grouped_input);
    let root = if matches!(
        variant,
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::DirectPostJoinAggregateInput
            | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput
            | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput
    ) {
        root
    } else {
        project_key_value(&mut builder, root)
    };
    let post_join_aggregate = match variant {
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate => {
            scalar_key_value_aggregate(&mut builder, root)
        },
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::DirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter => {
            grouped_aggregate(&mut builder, root)
        },
    };
    let projection = project_key_value(&mut builder, post_join_aggregate);
    let sort =
        builder.add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    match variant {
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::MissingFetchLimit => {
            builder.finish_with_host_sink(sort).unwrap()
        },
        ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::DirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::FilterWrappedGroupedInputDirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::ScalarPostJoinAggregate
        | ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::UnsupportedGroupedInputFilter => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
            builder.finish_with_host_sink(limit).unwrap()
        },
    }
}

#[derive(Debug, Clone, Copy)]
#[expect(dead_code, reason = "shared topology fixtures are compiled by test binaries with different coverage")]
pub(crate) enum ResidentGroupedAggregateInputPostJoinAggregateTopKVariant {
    Admitted,
    DirectPostJoinAggregateInput,
    FilterWrappedGroupedInputDirectPostJoinAggregateInput,
    PreAggregatedPreservedInputFilterWrappedDirectPostJoinAggregateInput,
    MissingFetchLimit,
    ScalarPostJoinAggregate,
    UnsupportedGroupedInputFilter,
}

pub(crate) fn resident_grouped_aggregate_subquery_filter_plan(with_source_metadata: bool) -> QueryPlan {
    resident_grouped_aggregate_subquery_filter_plan_inner(with_source_metadata)
}

fn resident_grouped_aggregate_subquery_filter_plan_inner(with_source_metadata: bool) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = source_with_optional_metadata(&mut builder, "resident-aggregate-subquery-filter-a", with_source_metadata);
    let b = source_with_optional_metadata(&mut builder, "resident-aggregate-subquery-filter-b", with_source_metadata);
    let c = source_with_optional_metadata(&mut builder, "resident-aggregate-subquery-filter-c", with_source_metadata);
    let d = source_with_optional_metadata(&mut builder, "resident-aggregate-subquery-filter-d", with_source_metadata);
    let lineitem = source_with_optional_metadata(
        &mut builder,
        "resident-aggregate-subquery-filter-lineitem",
        with_source_metadata,
    );

    let left_tree = join(&mut builder, a, b, "left_tree");
    let grouped_input = grouped_aggregate(&mut builder, lineitem);
    let grouped_projection = project_key_value(&mut builder, grouped_input);
    let cd = join(&mut builder, c, d, "cd");
    let right_tree = residual_join_against_grouped_aggregate(&mut builder, grouped_projection, cd);
    let root = semi_join(&mut builder, left_tree, right_tree);
    let sort = builder.add_sort(root, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

pub(crate) fn root_topk_grouped_aggregate_input_left_join_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let edges = source(&mut builder, "root-topk-grouped-aggregate-input-edges");
    let grouped = grouped_aggregate(&mut builder, edges);
    let grouped_sort =
        builder.add_sort(grouped, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let grouped_topk = builder.add_limit(grouped_sort, LimitSpec::fetch(30)).unwrap();
    let nodes = source(&mut builder, "root-topk-grouped-aggregate-input-nodes");
    let joined = builder
        .add_join(
            grouped_topk,
            nodes,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            ])
            .with_build_side(JoinSide::Right),
            key_value_schema(),
        )
        .unwrap();
    let joined = project_key_value(&mut builder, joined);
    let final_sort =
        builder.add_sort(joined, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let final_topk = builder.add_limit(final_sort, LimitSpec::fetch(30)).unwrap();
    builder.finish_with_host_sink(final_topk).unwrap()
}

pub(crate) fn root_topk_keyless_conditional_direct_source_aggregates_plan(
    missing_right_grouped_source_reduction: bool,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left_grouped_source = source_with_reduction(&mut builder, "root-topk-direct-grouped-left", true);
    let left_scalar_source = source_with_reduction(&mut builder, "root-topk-direct-scalar-left", true);
    let right_grouped_source =
        source_with_reduction(&mut builder, "root-topk-direct-grouped-right", !missing_right_grouped_source_reduction);
    let right_scalar_source = source_with_reduction(&mut builder, "root-topk-direct-scalar-right", true);

    let left = root_topk_keyless_conditional_aggregate_branch(&mut builder, left_grouped_source, left_scalar_source);
    let right = root_topk_keyless_conditional_aggregate_branch(&mut builder, right_grouped_source, right_scalar_source);
    let root = join(&mut builder, left, right, "direct_source_root");
    let projection = project_key_value(&mut builder, root);
    let sort =
        builder.add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn root_topk_keyless_conditional_aggregate_branch(
    builder: &mut QueryPlanBuilder,
    grouped_source: NodeId,
    scalar_source: NodeId,
) -> NodeId {
    let grouped_aggregate = grouped_aggregate(builder, grouped_source);
    let grouped = project_key_value(builder, grouped_aggregate);
    let scalar_aggregate = scalar_key_value_aggregate(builder, scalar_source);
    let scalar = project_key_value(builder, scalar_aggregate);
    builder
        .add_join(
            grouped,
            scalar,
            JoinSpec::new(JoinType::Inner, Vec::new())
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "v", "grouped_v"),
                    BinaryOp::Gt,
                    Expr::input_column(1, "v", "scalar_v"),
                ))
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
            key_value_schema(),
        )
        .unwrap()
}

pub(crate) fn root_topk_join_aggregate_variant_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    if variant.is_null_filtered_aggregate_fan_in() {
        return root_topk_null_filtered_aggregate_fan_in_plan(variant);
    }
    if variant.is_aggregate_window_filter_topk() {
        return root_topk_aggregate_window_filter_topk_plan(variant);
    }
    if variant.is_single_hash_join_aggregate() {
        return root_topk_single_hash_join_aggregate_plan(variant);
    }
    if variant.is_keyless_outer_cross_window() {
        return root_topk_keyless_outer_cross_window_plan(variant);
    }
    if variant.is_shared_computed_fanout() {
        return root_topk_shared_computed_fanout_plan(variant);
    }

    let mut builder = QueryPlanBuilder::new();
    let fact = source_with_reduction(
        &mut builder,
        "root-topk-join-aggregate-fact",
        !matches!(
            variant,
            RootTopKJoinAggregateVariant::MissingSourceReduction
                | RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput
                | RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate
        ),
    );
    let aggregate_all_branch_reduction = matches!(variant, RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate);
    let date = if aggregate_all_branch_reduction {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-date", true)
    } else {
        source(&mut builder, "root-topk-join-aggregate-date")
    };
    let customer = if matches!(variant, RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput)
        || aggregate_all_branch_reduction
    {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-customer", true)
    } else {
        source(&mut builder, "root-topk-join-aggregate-customer")
    };

    let fact_date = join(&mut builder, fact, date, "fact_date");
    let fact_store = if matches!(variant, RootTopKJoinAggregateVariant::SharedSourceFanout) {
        join(&mut builder, fact_date, fact, "fact_self")
    } else {
        let store = if aggregate_all_branch_reduction {
            source_with_reduction(&mut builder, "root-topk-join-aggregate-store", true)
        } else {
            source(&mut builder, "root-topk-join-aggregate-store")
        };
        join(&mut builder, fact_date, store, "fact_store")
    };
    let (mut grouped_projection, unrelated_scalar_projection) =
        if matches!(variant, RootTopKJoinAggregateVariant::MissingGroupedAggregateFragment) {
            (project_key_value(&mut builder, fact_store), None)
        } else {
            let unrelated_scalar_projection =
                if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate) {
                    let scalar = scalar_key_value_aggregate(&mut builder, fact_store);
                    Some(project_key_value(&mut builder, scalar))
                } else {
                    None
                };
            let grouped = if matches!(
                variant,
                RootTopKJoinAggregateVariant::DistinctAggregate
                    | RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin
            ) {
                builder
                    .add_aggregate(
                        fact_store,
                        AggregateSpec::new(
                            vec![Expr::column("k")],
                            vec![
                                AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v")
                                    .with_distinct(true),
                            ],
                        ),
                        key_value_schema(),
                    )
                    .unwrap()
            } else if matches!(variant, RootTopKJoinAggregateVariant::FilteredDistinctAggregate) {
                builder
                    .add_aggregate(
                        fact_store,
                        AggregateSpec::new(
                            vec![Expr::column("k")],
                            vec![
                                AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v")
                                    .with_distinct(true)
                                    .with_filter(Some(Expr::binary(
                                        Expr::column("v"),
                                        BinaryOp::Gt,
                                        Expr::literal(LiteralValue::Int64(0)),
                                    ))),
                            ],
                        ),
                        key_value_schema(),
                    )
                    .unwrap()
            } else {
                grouped_aggregate(&mut builder, fact_store)
            };
            (project_key_value(&mut builder, grouped), unrelated_scalar_projection)
        };
    if matches!(variant, RootTopKJoinAggregateVariant::BranchLocalWindow) {
        grouped_projection = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
    }
    let root = if matches!(variant, RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin) {
        let left = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
        let right = sorted_rank_window_project_key_value(&mut builder, customer);
        full_join(&mut builder, left, right, FullJoinOutput::BothSides)
    } else if matches!(variant, RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection) {
        let left = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
        let right = sorted_rank_window_project_key_value(&mut builder, customer);
        full_join(&mut builder, left, right, FullJoinOutput::LeftOnly)
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin) {
        let unrelated_outer = builder
            .add_join(
                fact_store,
                customer,
                JoinSpec::new(
                    JoinType::Left,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap();
        join(&mut builder, grouped_projection, unrelated_outer, "root_distinct_unrelated_outer")
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin) {
        builder
            .add_join(
                customer,
                grouped_projection,
                JoinSpec::new(
                    JoinType::Left,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin) {
        let right_aggregate = grouped_aggregate(&mut builder, customer);
        let right_aggregate = project_key_value(&mut builder, right_aggregate);
        builder
            .add_join(
                grouped_projection,
                right_aggregate,
                JoinSpec::new(
                    JoinType::Left,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::Left,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate) {
        let unrelated_scalar_projection =
            unrelated_scalar_projection.expect("unrelated scalar variant should build a scalar projection");
        let left = join(&mut builder, grouped_projection, unrelated_scalar_projection, "left_with_unrelated_scalar");
        builder
            .add_join(
                left,
                customer,
                JoinSpec::new(JoinType::Inner, vec![])
                    .with_residual_predicate(Expr::binary(
                        Expr::input_column(0, "k", "k"),
                        BinaryOp::LtEq,
                        Expr::input_column(1, "k", "k"),
                    ))
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                        JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                    ])
                    .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "v", "v"),
                    BinaryOp::Gt,
                    Expr::input_column(1, "v", "v"),
                ))
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput) {
        semi_join(&mut builder, grouped_projection, customer)
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::LeftSemi,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "v", "v"),
                    BinaryOp::Gt,
                    Expr::input_column(1, "v", "v"),
                ))
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::LeftSemi,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Left),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftMarkJoinInput) {
        // A disjunctive EXISTS (`EXISTS a OR EXISTS b`) cannot decorrelate to a
        // semi join, so the frontend emits a mark join per branch and filters on
        // the boolean columns. The executor synthesizes the mark column, so it
        // never appears in the output projection.
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::LeftMark { mark_column: "mark".to_string() },
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_mark_schema("mark"),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::LeftSemi,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::right(ColumnRef::input(1, "k", "k"), "k"),
                    JoinOutputColumn::right(ColumnRef::input(1, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(
                    JoinType::LEFT_ANTI,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(
        variant,
        RootTopKJoinAggregateVariant::KeylessJoin | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
    ) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(JoinType::Inner, vec![])
                    .with_residual_predicate(Expr::binary(
                        Expr::input_column(0, "k", "k"),
                        BinaryOp::LtEq,
                        Expr::input_column(1, "k", "k"),
                    ))
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                        JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                    ])
                    .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(
        variant,
        RootTopKJoinAggregateVariant::ProjectedCrossJoin
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
    ) {
        builder
            .add_join(
                grouped_projection,
                customer,
                JoinSpec::new(JoinType::Cross, vec![])
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                        JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                    ])
                    .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossJoin) {
        keyless_outer_cross_join(&mut builder, grouped_projection, customer)
    } else {
        join(&mut builder, grouped_projection, customer, "root")
    };
    let root = if matches!(variant, RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate) {
        grouped_aggregate(&mut builder, root)
    } else {
        root
    };
    let mut topk_projection = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchHashJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftMarkJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
            | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
    ) {
        root
    } else {
        project_key_value(&mut builder, root)
    };
    if matches!(variant, RootTopKJoinAggregateVariant::Window) {
        topk_projection = sorted_rank_window_project_key_value(&mut builder, topk_projection);
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortedAggregateWindow) {
        let sorted = builder
            .add_sort(topk_projection, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]))
            .unwrap();
        topk_projection = unordered_aggregate_window_project_key_value(&mut builder, sorted);
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow) {
        let sorted = builder
            .add_sort(topk_projection, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]))
            .unwrap();
        topk_projection = unordered_global_aggregate_window_project_key_value(&mut builder, sorted);
    } else if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
        topk_projection = rank_filter_window_project_key_value(&mut builder, topk_projection);
    } else if matches!(variant, RootTopKJoinAggregateVariant::UnsortedRankWindow) {
        topk_projection = rank_window_project_key_value(&mut builder, topk_projection);
    }
    let mut sort =
        builder.add_sort(topk_projection, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
        sort = project_key_value(&mut builder, sort);
    }
    match variant {
        RootTopKJoinAggregateVariant::MissingFetchLimit => {
            let limit = builder.add_limit(sort, LimitSpec::offset(0)).unwrap();
            builder.finish_with_host_sink(limit).unwrap()
        },
        RootTopKJoinAggregateVariant::Admitted
        | RootTopKJoinAggregateVariant::MissingSourceReduction
        | RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput
        | RootTopKJoinAggregateVariant::SharedSourceFanout
        | RootTopKJoinAggregateVariant::MissingGroupedAggregateFragment
        | RootTopKJoinAggregateVariant::DistinctAggregate
        | RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin
        | RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin
        | RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin
        | RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin
        | RootTopKJoinAggregateVariant::FilteredDistinctAggregate
        | RootTopKJoinAggregateVariant::KeylessJoin
        | RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate
        | RootTopKJoinAggregateVariant::ProjectedCrossJoin
        | RootTopKJoinAggregateVariant::KeylessOuterCrossJoin
        | RootTopKJoinAggregateVariant::BranchLocalWindow
        | RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin
        | RootTopKJoinAggregateVariant::SortedAggregateWindow
        | RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow
        | RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection
        | RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection
        | RootTopKJoinAggregateVariant::UnsortedRankWindow
        | RootTopKJoinAggregateVariant::SortFetchHashJoinInput
        | RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiResidualJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftMarkJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput
        | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
        | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
        | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild
        | RootTopKJoinAggregateVariant::Window => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
            builder.finish_with_host_sink(limit).unwrap()
        },
        // Dispatched by `is_aggregate_window_filter_topk` before this function body
        // runs; listed here only to keep this match exhaustive over the enum.
        RootTopKJoinAggregateVariant::AggregateWindowFilterTopK
        | RootTopKJoinAggregateVariant::AggregateWindowFilterTopKMissingInputSort
        | RootTopKJoinAggregateVariant::CumulativeWindowFullOuterTopK => {
            unreachable!("handled by root_topk_aggregate_window_filter_topk_plan")
        },
        // Dispatched by `is_single_hash_join_aggregate`, `is_keyless_outer_cross_window`,
        // and `is_shared_computed_fanout`, whose plans share no spine with this body.
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer
        | RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow
        | RootTopKJoinAggregateVariant::SharedComputedFanout
        | RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout
        | RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout
        | RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout => {
            unreachable!("handled by a dedicated root TopK plan builder")
        },
    }
}

fn root_topk_single_hash_join_aggregate_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let (fact, customer) = match variant {
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters => {
            let fact = source_with_reduction(&mut builder, "root-topk-join-aggregate-fact", false);
            let customer = source(&mut builder, "root-topk-join-aggregate-customer");
            (filter_source(&mut builder, fact), filter_source(&mut builder, customer))
        },
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact => (
            source_with_known_size(&mut builder, "root-topk-join-aggregate-small-exact-fact", 4096),
            source_with_known_size(&mut builder, "root-topk-join-aggregate-small-exact-customer", 4096),
        ),
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan => (
            source_with_known_size(&mut builder, "root-topk-join-aggregate-large-full-scan-fact", 512 << 20),
            source_with_known_size(&mut builder, "root-topk-join-aggregate-small-exact-customer", 4096),
        ),
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan => (
            source_with_reduction(&mut builder, "root-topk-join-aggregate-fact", false),
            source(&mut builder, "root-topk-join-aggregate-customer"),
        ),
        _ => (
            source_with_reduction(&mut builder, "root-topk-join-aggregate-fact", true),
            source(&mut builder, "root-topk-join-aggregate-customer"),
        ),
    };
    let single_join = join(&mut builder, fact, customer, "single_join_aggregate");
    let aggregate = grouped_aggregate(&mut builder, single_join);
    finish_root_topk(builder, aggregate)
}

/// A keyless left join over a grouped aggregate that reads the fact source
/// directly, followed by a window and a post-window join against the store.
fn root_topk_keyless_outer_cross_window_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = source_with_reduction(
        &mut builder,
        "root-topk-join-aggregate-fact",
        !matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction),
    );
    let store = source(&mut builder, "root-topk-join-aggregate-store");
    let customer = source(&mut builder, "root-topk-join-aggregate-customer");

    let direct_grouped = grouped_aggregate(&mut builder, fact);
    let grouped_projection = project_key_value(&mut builder, direct_grouped);
    let root = keyless_outer_cross_join(&mut builder, grouped_projection, customer);
    let root_projection = project_key_value(&mut builder, root);
    let post_window_input = match variant {
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer => {
            let window_branch = sorted_rank_window_project_key_value(&mut builder, root_projection);
            let direct_branch = join(&mut builder, root, store, "direct_keyless_consumer");
            join(&mut builder, window_branch, direct_branch, "post_window_root")
        },
        RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow => {
            let window = unordered_aggregate_window_project_key_value(&mut builder, root_projection);
            join(&mut builder, window, store, "post_window_root")
        },
        _ => {
            let window = sorted_rank_window_project_key_value(&mut builder, root_projection);
            join(&mut builder, window, store, "post_window_root")
        },
    };
    let topk_projection = project_key_value(&mut builder, post_window_input);
    finish_root_topk(builder, topk_projection)
}

/// Shapes where one computed node (the fact join, the grouped aggregate, or its
/// projection) has two consumers. They never read the customer source.
fn root_topk_shared_computed_fanout_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let (fact, date, store) =
        if matches!(variant, RootTopKJoinAggregateVariant::SmallExactSharedProjectionAggregateFanout) {
            (
                source_with_known_size(&mut builder, "root-topk-join-aggregate-fact-small-exact", 4096),
                source_with_known_size(&mut builder, "root-topk-join-aggregate-date-small-exact", 4096),
                source_with_known_size(&mut builder, "root-topk-join-aggregate-store-small-exact", 4096),
            )
        } else {
            (
                source_with_reduction(&mut builder, "root-topk-join-aggregate-fact", true),
                source(&mut builder, "root-topk-join-aggregate-date"),
                source(&mut builder, "root-topk-join-aggregate-store"),
            )
        };
    let fact_date = join(&mut builder, fact, date, "fact_date");
    let fact_store = join(&mut builder, fact_date, store, "fact_store");
    let grouped = grouped_aggregate(&mut builder, fact_store);
    let grouped_projection = project_key_value(&mut builder, grouped);
    let root = match variant {
        RootTopKJoinAggregateVariant::SharedComputedFanout => {
            join(&mut builder, grouped_projection, fact_store, "root_shared_computed")
        },
        RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout => {
            let aggregate_branch = grouped_aggregate(&mut builder, grouped_projection);
            let aggregate_branch = project_key_value(&mut builder, aggregate_branch);
            join(&mut builder, grouped, aggregate_branch, "root_shared_grouped_aggregate")
        },
        _ => {
            let aggregate_branch = grouped_aggregate(&mut builder, grouped_projection);
            let aggregate_branch = project_key_value(&mut builder, aggregate_branch);
            join(&mut builder, grouped_projection, aggregate_branch, "root_shared_projection_aggregate")
        },
    };
    let topk_projection = project_key_value(&mut builder, root);
    finish_root_topk(builder, topk_projection)
}

fn finish_root_topk(mut builder: QueryPlanBuilder, input: NodeId) -> QueryPlan {
    let sort = builder.add_sort(input, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn keyless_outer_cross_join(builder: &mut QueryPlanBuilder, left: NodeId, right: NodeId) -> NodeId {
    builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Left, vec![])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
            key_value_schema(),
        )
        .unwrap()
}

/// Root TopK join/aggregate fixture variants. Each variant either proves a
/// modeled subcontract or introduces one targeted gap so admission reports the
/// intended local reason.
#[derive(Debug, Clone, Copy)]
#[expect(dead_code, reason = "shared topology fixtures are compiled by test binaries with different coverage")]
pub(crate) enum RootTopKJoinAggregateVariant {
    /// Canonical root TopK shape: reduced source -> grouped aggregate fragment ->
    /// keyed root join -> projection -> sort(fetch).
    Admitted,
    /// No source under the aggregate input carries a reduction proof; scan size
    /// is a capacity question owned by the allocator, so this stays admitted.
    MissingSourceReduction,
    /// A source reduction exists, but only on a branch unrelated to the grouped
    /// aggregate input that bounds the TopK result; the branch a dynamic filter
    /// lands on must not change admission.
    SourceReductionOutsideAggregateInput,
    // Shared-input fixtures distinguish harmless source reuse from computed
    // producer fanout that needs an explicit shared-subplan proof.
    SharedSourceFanout,
    SharedComputedFanout,
    SharedGroupedAggregateFanout,
    SharedProjectionAggregateFanout,
    /// Shared projection fanout whose joined aggregate inputs have exact small
    /// sizes; this exercises captured one-shot strategy thresholds.
    SmallExactSharedProjectionAggregateFanout,
    /// Removes the grouped aggregate fragment so generic fallback must not mask
    /// the root TopK-specific shape miss.
    MissingGroupedAggregateFragment,
    /// Uses `LIMIT 0, None` instead of a fetch limit; TopK admission requires a
    /// real fetch bound.
    MissingFetchLimit,
    // Distinct aggregate fixtures cover the explicit distinct subcontract
    // and the unsafe variants that would need unmodeled distinct state.
    DistinctAggregate,
    DistinctAggregateBuildSideKeyedOuterJoin,
    DistinctAggregatePreservedSideAggregateOuterJoin,
    /// Distinct aggregate connected through an extra outer join; state
    /// preservation is no longer direct enough for the subcontract.
    DistinctAggregateTransitiveKeyedOuterJoin,
    /// Distinct aggregate plus an unrelated keyed outer join, which should be
    /// rejected by the local join-contract check.
    DistinctAggregateUnrelatedKeyedOuterJoin,
    /// Distinct aggregate with a filter; filtered distinct state is intentionally
    /// kept out until the state model is explicit.
    FilteredDistinctAggregate,
    // Keyless/cross fixtures exercise the keyless-or-cross subcontract and the
    // narrower keyless outer-cross window subcontract.
    KeylessJoin,
    /// Adds an unrelated scalar aggregate branch (over an unreduced fact scan)
    /// before the keyless join; neither must break the keyless subcontract.
    KeylessJoinUnrelatedScalarAggregate,
    ProjectedCrossJoin,
    /// Bare keyless outer-cross is rejected; only the modeled window variant has
    /// enough output and gather-map contracts.
    KeylessOuterCrossJoin,
    KeylessOuterCrossWindow,
    KeylessOuterCrossWindowMissingSourceReduction,
    /// Sends the keyless outer-cross output to both the modeled window path and
    /// a direct join consumer; the extra consumer invalidates the subcontract.
    KeylessOuterCrossWindowWithDirectJoinConsumer,
    /// Uses an unordered window input after the keyless outer-cross, so the
    /// window subcontract cannot prove ordered input.
    KeylessOuterCrossUnorderedWindow,
    // Window fixtures distinguish admitted ordered/rank windows from unsupported
    // global, left-only full-outer, and unordered-rank variants.
    BranchLocalWindow,
    WindowInputSortFullOuterJoin,
    SortedAggregateWindow,
    SortedGlobalAggregateWindow,
    RankFilterWindowFinalProjection,
    WindowFullOuterLeftOnlyProjection,
    UnsortedRankWindow,
    /// Grouped aggregate -> sort -> whole-partition aggregate window -> filter ->
    /// final sort -> limit, with no projection reinserted before the final sort.
    /// Pins that the window/aggregate composition this contract already admits
    /// whenever an ORDER BY forces a projection into the chain (q47/q57/q89) is
    /// admitted just as well when no such projection appears (q53/q63).
    AggregateWindowFilterTopK,
    /// Same composition as `AggregateWindowFilterTopK`, but the window is an
    /// ordered rank fed directly from the (unsorted) aggregate instead of from a
    /// `Sort`, and the join fragment uses the keyless conditional subcontract so
    /// no generic envelope admits the plan. Pins that this near-miss keeps its
    /// own `root_topk_join_aggregate_requires_ordered_window_input` diagnosis
    /// rather than falling back to the generic pre-window rejection.
    AggregateWindowFilterTopKMissingInputSort,
    /// Two grouped-aggregate branches, each followed by a sort and a cumulative
    /// window, joined by a keyed FULL OUTER join retaining both sides, then
    /// projection -> sort -> cumulative window -> filter -> final sort -> limit,
    /// with no projection between the final window/filter and the final sort.
    CumulativeWindowFullOuterTopK,
    SortFetchHashJoinInput,
    SortFetchKeyedResidualJoinInput,
    SortFetchLeftSemiJoinInput,
    SortFetchLeftSemiResidualJoinInput,
    SortFetchLeftSemiWrongBuildSideJoinInput,
    SortFetchLeftSemiRightOutputJoinInput,
    SortFetchLeftMarkJoinInput,
    SortFetchLeftAntiJoinInput,
    SortFetchKeylessJoinInput,
    SortFetchProjectedCrossJoinInput,
    // Sort-fetch join-aggregate fixtures. Full-scan and large-exact variants
    // admit alongside the reduced and exact-small ones: how much a plan reads is
    // capacity, owned by the admitted grant's allocator.
    SortFetchSingleHashJoinAggregate,
    SortFetchSingleHashJoinAggregateSmallExact,
    SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
    SortFetchSingleHashJoinAggregateLargeFullScan,
    SortFetchSingleHashJoinAggregateFullScan,
    SortFetchHashJoinAggregate,
    // Null-filtered fan-in fixtures model left joins whose right-side marker is
    // filtered with IS NULL and then rewritten to a bounded anti-join contract.
    SortFetchNullFilteredLeftAggregateFanIn,
    SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection,
    SortFetchNullFilteredLeftAggregateFanInMissingFilter,
    SortFetchNullFilteredLeftAggregateFanInWrongBuildSide,
    SortFetchNullFilteredLeftAggregateFanInMarkerEscapes,
    SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse,
    SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight,
    SortFetchNullFilteredLeftAggregateFanInNonUniqueRight,
    SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction,
    SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds,
    SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild,
    Window,
}

impl RootTopKJoinAggregateVariant {
    fn is_null_filtered_aggregate_fan_in(self) -> bool {
        matches!(
            self,
            Self::SortFetchNullFilteredLeftAggregateFanIn
                | Self::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection
                | Self::SortFetchNullFilteredLeftAggregateFanInMissingFilter
                | Self::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
                | Self::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
                | Self::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
                | Self::SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight
                | Self::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight
                | Self::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction
                | Self::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
                | Self::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild
        )
    }

    fn is_single_hash_join_aggregate(self) -> bool {
        matches!(
            self,
            Self::SortFetchSingleHashJoinAggregate
                | Self::SortFetchSingleHashJoinAggregateSmallExact
                | Self::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
                | Self::SortFetchSingleHashJoinAggregateLargeFullScan
                | Self::SortFetchSingleHashJoinAggregateFullScan
        )
    }

    fn is_keyless_outer_cross_window(self) -> bool {
        matches!(
            self,
            Self::KeylessOuterCrossWindow
                | Self::KeylessOuterCrossWindowMissingSourceReduction
                | Self::KeylessOuterCrossWindowWithDirectJoinConsumer
                | Self::KeylessOuterCrossUnorderedWindow
        )
    }

    fn is_shared_computed_fanout(self) -> bool {
        matches!(
            self,
            Self::SharedComputedFanout
                | Self::SharedGroupedAggregateFanout
                | Self::SharedProjectionAggregateFanout
                | Self::SmallExactSharedProjectionAggregateFanout
        )
    }

    fn is_aggregate_window_filter_topk(self) -> bool {
        matches!(
            self,
            Self::AggregateWindowFilterTopK
                | Self::AggregateWindowFilterTopKMissingInputSort
                | Self::CumulativeWindowFullOuterTopK
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum NullFilteredReturnBuildShape {
    Reduced,
    FullBounded,
    FullLarge,
}

fn root_topk_null_filtered_aggregate_fan_in_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let store_return_shape = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
    ) {
        NullFilteredReturnBuildShape::FullBounded
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild) {
        NullFilteredReturnBuildShape::FullLarge
    } else {
        NullFilteredReturnBuildShape::Reduced
    };
    let other_return_shape = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
    ) {
        NullFilteredReturnBuildShape::FullBounded
    } else {
        NullFilteredReturnBuildShape::Reduced
    };
    let store = null_filtered_aggregate_branch(
        &mut builder,
        "root-topk-null-filtered-store",
        !matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction),
        store_return_shape,
        matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter),
        matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide),
        matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes),
        matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse),
    );
    let web = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInUniqueNonAggregateRight
    ) {
        builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("root-topk-null-filtered-web", key_value_schema())
                .with_row_count_upper_bound(100)
                .with_unique_key_columns(std::collections::BTreeSet::from(["k".to_owned()])),
        ))
    } else {
        null_filtered_aggregate_branch(
            &mut builder,
            "root-topk-null-filtered-web",
            true,
            other_return_shape,
            false,
            false,
            false,
            false,
        )
    };
    let web = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInIdentityArithmeticProjection
    ) {
        builder
            .add_projection(
                web,
                projection_outputs(
                    vec![
                        Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
                        Expr::column("v"),
                    ],
                    key_value_schema(),
                ),
            )
            .unwrap()
    } else {
        web
    };
    let catalog = null_filtered_aggregate_branch(
        &mut builder,
        "root-topk-null-filtered-catalog",
        true,
        other_return_shape,
        false,
        false,
        false,
        false,
    );
    let web_fan_in_input =
        if matches!(variant, RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight) {
            grouped_aggregate_with_extra_key(&mut builder, web)
        } else {
            web
        };
    let store_web = aggregate_fan_in_left_join(&mut builder, store, web_fan_in_input);
    let joined = aggregate_fan_in_left_join(&mut builder, store_web, catalog);
    let filtered = builder
        .add_filter(joined, Expr::binary(Expr::column("v"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))))
        .unwrap();
    let projected = project_key_value(&mut builder, filtered);
    let sort = builder.add_sort(projected, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

#[expect(clippy::too_many_arguments, reason = "test fixture parameters are independent fixture axes")]
fn null_filtered_aggregate_branch(
    builder: &mut QueryPlanBuilder,
    label: &str,
    reduced_fact: bool,
    return_build_shape: NullFilteredReturnBuildShape,
    missing_filter: bool,
    wrong_build_side: bool,
    marker_escapes: bool,
    marker_filter_reuse: bool,
) -> NodeId {
    let fact = if reduced_fact {
        source_with_reduction(builder, &format!("{label}-fact"), true)
    } else {
        source(builder, &format!("{label}-fact"))
    };
    let returns = match return_build_shape {
        NullFilteredReturnBuildShape::Reduced => source_with_reduction(builder, &format!("{label}-returns"), true),
        NullFilteredReturnBuildShape::FullBounded => {
            source_with_known_size(builder, &format!("{label}-returns"), 460_691_082)
        },
        NullFilteredReturnBuildShape::FullLarge => {
            source_with_known_size(builder, &format!("{label}-returns"), 600 << 20)
        },
    };
    let marker_name = format!("{label}_return_marker");
    let joined = builder
        .add_join(
            fact,
            returns,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                JoinOutputColumn::right(ColumnRef::input(1, "k", "k"), marker_name.as_str()),
            ])
            .with_build_side(if wrong_build_side { JoinSide::Left } else { JoinSide::Right }),
            key_value_marker_schema(&marker_name),
        )
        .unwrap();
    let filtered = if missing_filter {
        joined
    } else if marker_filter_reuse {
        builder
            .add_filter(
                joined,
                Expr::binary(
                    Expr::is_null(Expr::column(&marker_name)),
                    BinaryOp::And,
                    Expr::binary(Expr::column(&marker_name), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
                ),
            )
            .unwrap()
    } else {
        builder.add_filter(joined, Expr::is_null(Expr::column(&marker_name))).unwrap()
    };
    let projected = if marker_escapes {
        builder
            .add_projection(
                filtered,
                projection_outputs(
                    vec![Expr::column("k"), Expr::column("v"), Expr::column(&marker_name)],
                    key_value_marker_schema(&marker_name),
                ),
            )
            .unwrap()
    } else {
        project_key_value(builder, filtered)
    };
    let date = source_with_reduction(builder, &format!("{label}-date"), true);
    let dated = join(builder, projected, date, "date");
    let aggregate = grouped_aggregate(builder, dated);
    project_key_value(builder, aggregate)
}

fn aggregate_fan_in_left_join(builder: &mut QueryPlanBuilder, left: NodeId, right: NodeId) -> NodeId {
    builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                JoinOutputColumn::right(ColumnRef::input(1, "v", "v"), "right_v"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int32, false),
                Field::new("v", DataType::Int64, true),
                Field::new("right_v", DataType::Int64, true),
            ])),
        )
        .unwrap()
}

fn grouped_aggregate_with_extra_key(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("right_v", DataType::Int64, true),
    ]));
    let grouped = builder
        .add_aggregate(
            input,
            AggregateSpec::new(
                vec![Expr::column("k"), Expr::column("v")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "right_v")],
            ),
            Arc::clone(&schema),
        )
        .unwrap();
    builder
        .add_projection(grouped, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
        .unwrap()
}

fn root_topk_aggregate_window_filter_topk_plan(variant: RootTopKJoinAggregateVariant) -> QueryPlan {
    if matches!(variant, RootTopKJoinAggregateVariant::CumulativeWindowFullOuterTopK) {
        return root_topk_cumulative_window_full_outer_topk_plan();
    }
    let missing_input_sort = matches!(variant, RootTopKJoinAggregateVariant::AggregateWindowFilterTopKMissingInputSort);

    let mut builder = QueryPlanBuilder::new();
    let fact = source(&mut builder, "root-topk-aggregate-window-filter-fact");
    let date = source(&mut builder, "root-topk-aggregate-window-filter-date");
    let store = source(&mut builder, "root-topk-aggregate-window-filter-store");
    let fact_date = join(&mut builder, fact, date, "fact_date");
    let fact_store = if missing_input_sort {
        // The near-miss fixture keeps its join fragment on the keyless conditional
        // subcontract, which the root TopK envelope models but the generic closed
        // multi-join subtree does not. Otherwise the whole plan is admitted by the
        // generic envelope and the window near-miss never projects a rejection.
        builder
            .add_join(
                fact_date,
                store,
                JoinSpec::new(JoinType::Inner, vec![])
                    .with_residual_predicate(Expr::binary(
                        Expr::input_column(0, "k", "k"),
                        BinaryOp::LtEq,
                        Expr::input_column(1, "k", "k"),
                    ))
                    .with_output_projection(vec![
                        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                        JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                    ])
                    .with_build_side(JoinSide::Right),
                key_value_schema(),
            )
            .unwrap()
    } else {
        join(&mut builder, fact_date, store, "fact_store")
    };
    let grouped = grouped_aggregate(&mut builder, fact_store);

    let filter = if missing_input_sort {
        // Ordered window fed directly from the aggregate, without an input sort:
        // pins that this near-miss keeps its own `requires_ordered_window_input`
        // diagnosis instead of the generic pre-window rejection.
        let window = rank_window(&mut builder, grouped);
        builder
            .add_filter(
                window,
                Expr::binary(Expr::column("rank"), BinaryOp::Gt, Expr::literal(LiteralValue::UInt64(0))),
            )
            .unwrap()
    } else {
        let sorted =
            builder.add_sort(grouped, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
        let window = unordered_aggregate_window(&mut builder, sorted);
        builder
            .add_filter(
                window,
                Expr::binary(Expr::column("window_v"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
            )
            .unwrap()
    };
    let sort = builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn root_topk_cumulative_window_full_outer_topk_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = cumulative_window_grouped_aggregate_branch(&mut builder, "root-topk-cumulative-full-outer-left");
    let right = cumulative_window_grouped_aggregate_branch(&mut builder, "root-topk-cumulative-full-outer-right");
    let full_outer = full_join(&mut builder, left, right, FullJoinOutput::BothSides);
    let projected = project_key_value(&mut builder, full_outer);
    let sorted =
        builder.add_sort(projected, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let window = cumulative_aggregate_window(&mut builder, sorted);
    let filter = builder
        .add_filter(window, Expr::binary(Expr::column("window_v"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))))
        .unwrap();
    let final_sort =
        builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn cumulative_window_grouped_aggregate_branch(builder: &mut QueryPlanBuilder, label: &str) -> NodeId {
    let fact = source(builder, &format!("{label}-fact"));
    let date = source(builder, &format!("{label}-date"));
    let fact_date = join(builder, fact, date, "fact_date");
    let grouped = grouped_aggregate(builder, fact_date);
    let sorted = builder.add_sort(grouped, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    cumulative_aggregate_window(builder, sorted)
}

#[expect(dead_code, reason = "shared across test binaries; not all path-includes call this constructor")]
pub(crate) fn shared_grouped_aggregate_fanout_to_join_and_grouped_aggregate_plan() -> QueryPlan {
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
    let sort = builder.add_sort(root, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

fn source(builder: &mut QueryPlanBuilder, label: &str) -> NodeId {
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/{label}.parquet"),
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn source_with_optional_metadata(builder: &mut QueryPlanBuilder, label: &str, with_source_metadata: bool) -> NodeId {
    if !with_source_metadata {
        return source(builder, label);
    }
    let path = format!("/tmp/{label}.parquet");
    let file = LocalParquetFile::new(&path).with_file_size_bytes(std::fs::metadata(&path).unwrap().len());
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn source_with_reduction(builder: &mut QueryPlanBuilder, label: &str, selected_subset: bool) -> NodeId {
    let file = if selected_subset {
        LocalParquetFile::new(format!("/tmp/{label}.parquet"))
            .with_file_size_bytes(4096)
            .with_row_groups(vec![0])
            .with_projected_row_group_bytes(vec![1024, 1024], vec![1024, 1024])
            .with_row_group_rows(vec![1000, 1000])
    } else {
        LocalParquetFile::new(format!("/tmp/{label}.parquet")).with_file_size_bytes(4096)
    };
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn source_with_known_size(builder: &mut QueryPlanBuilder, label: &str, file_size_bytes: u64) -> NodeId {
    let file = LocalParquetFile::new(format!("/tmp/{label}.parquet")).with_file_size_bytes(file_size_bytes);
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn filter_source(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_filter(input, Expr::binary(Expr::column("k"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))))
        .unwrap()
}

fn join(builder: &mut QueryPlanBuilder, left: NodeId, right: NodeId, _value_name: &str) -> NodeId {
    let output_schema = key_value_schema();
    builder
        .add_join(
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
        .unwrap()
}

fn full_join(builder: &mut QueryPlanBuilder, left: NodeId, right: NodeId, output: FullJoinOutput) -> NodeId {
    let output_projection = match output {
        FullJoinOutput::BothSides => vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::right(ColumnRef::input(1, "v", "v"), "v"),
        ],
        FullJoinOutput::LeftOnly => vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ],
    };
    builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Full,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(output_projection)
            .with_build_side(JoinSide::Left),
            key_value_schema(),
        )
        .unwrap()
}

enum FullJoinOutput {
    BothSides,
    LeftOnly,
}

fn semi_join(builder: &mut QueryPlanBuilder, left: NodeId, right: NodeId) -> NodeId {
    builder
        .add_join(
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
        .unwrap()
}

fn residual_join_against_grouped_aggregate(builder: &mut QueryPlanBuilder, grouped: NodeId, subtree: NodeId) -> NodeId {
    builder
        .add_join(
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
        .unwrap()
}

fn grouped_aggregate(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_aggregate(
            input,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
            ),
            key_value_schema(),
        )
        .unwrap()
}

fn scalar_key_value_aggregate(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_aggregate(
            input,
            AggregateSpec::new(
                vec![],
                vec![
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("k"), "k"),
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("v"), "v"),
                ],
            ),
            key_value_schema(),
        )
        .unwrap()
}

fn rank_window(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_window(
            input,
            WindowSpec::new(
                vec![Expr::column("k")],
                vec![SortExpr::new(Expr::column("v"), true, false)],
                vec![WindowFunction::rank("rank")],
            ),
        )
        .unwrap()
}

fn rank_window_project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let window = rank_window(builder, input);
    project_key_value(builder, window)
}

fn sorted_rank_window_project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let sort = builder
        .add_sort(
            input,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("k"), false, false),
                SortExpr::new(Expr::column("v"), true, false),
            ]),
        )
        .unwrap();
    rank_window_project_key_value(builder, sort)
}

fn unordered_aggregate_window(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_window(
            input,
            WindowSpec::new(
                vec![Expr::column("k")],
                vec![],
                vec![WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("v"),
                    Field::new("window_v", DataType::Int64, true),
                )],
            ),
        )
        .unwrap()
}

fn unordered_aggregate_window_project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let window = unordered_aggregate_window(builder, input);
    project_key_value(builder, window)
}

fn unordered_global_aggregate_window_project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let window = builder
        .add_window(
            input,
            WindowSpec::new(
                vec![],
                vec![],
                vec![WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("v"),
                    Field::new("window_v", DataType::Int64, true),
                )],
            ),
        )
        .unwrap();
    project_key_value(builder, window)
}

fn cumulative_aggregate_window(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_window(
            input,
            WindowSpec::new(
                vec![Expr::column("k")],
                vec![SortExpr::new(Expr::column("v"), true, false)],
                vec![WindowFunction::cumulative_aggregate(
                    AggregateFunction::Sum,
                    Expr::column("v"),
                    Field::new("window_v", DataType::Int64, true),
                )],
            ),
        )
        .unwrap()
}

fn rank_filter_window_project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    let sorted = builder
        .add_sort(
            input,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("k"), false, false),
                SortExpr::new(Expr::column("v"), true, false),
            ]),
        )
        .unwrap();
    let window = builder
        .add_window(
            sorted,
            WindowSpec::new(
                vec![Expr::column("k")],
                vec![SortExpr::new(Expr::column("v"), true, false)],
                vec![WindowFunction::rank("rank")],
            ),
        )
        .unwrap();
    let filter = builder
        .add_filter(window, Expr::binary(Expr::column("rank"), BinaryOp::Gt, Expr::literal(LiteralValue::UInt64(0))))
        .unwrap();
    project_key_value(builder, filter)
}

fn project_key_value(builder: &mut QueryPlanBuilder, input: NodeId) -> NodeId {
    builder
        .add_projection(input, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
        .unwrap()
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]))
}

fn key_value_mark_schema(mark_name: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new(mark_name, DataType::Boolean, false),
    ]))
}

fn key_value_marker_schema(marker_name: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new(marker_name, DataType::Int32, true),
    ]))
}
