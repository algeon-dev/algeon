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

pub(crate) fn root_topk_join_aggregate_plan() -> nexus_query_engine::QueryPlan {
    root_topk_join_aggregate_variant_plan(RootTopKJoinAggregateVariant::Admitted)
}

pub(crate) fn root_topk_join_aggregate_variant_plan(
    variant: RootTopKJoinAggregateVariant,
) -> nexus_query_engine::QueryPlan {
    if variant.is_null_filtered_aggregate_fan_in() {
        return root_topk_null_filtered_aggregate_fan_in_plan(variant);
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
                | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
                | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
                | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
        ),
    );
    let aggregate_all_branch_reduction = matches!(variant, RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate);
    let date = if aggregate_all_branch_reduction {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-date", true)
    } else {
        source(&mut builder, "root-topk-join-aggregate-date")
    };
    let store = if aggregate_all_branch_reduction {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-store", true)
    } else {
        source(&mut builder, "root-topk-join-aggregate-store")
    };
    let customer = if matches!(variant, RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput) {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-customer", true)
    } else if aggregate_all_branch_reduction {
        source_with_reduction(&mut builder, "root-topk-join-aggregate-customer", true)
    } else {
        source(&mut builder, "root-topk-join-aggregate-customer")
    };

    let fact_date = join(&mut builder, fact, date, "fact_date");
    let fact_store = if matches!(variant, RootTopKJoinAggregateVariant::SharedSourceFanout) {
        join(&mut builder, fact_date, fact, "fact_self")
    } else {
        join(&mut builder, fact_date, store, "fact_store")
    };
    let (grouped_source, mut grouped_projection, unrelated_scalar_projection) =
        if matches!(variant, RootTopKJoinAggregateVariant::MissingGroupedAggregateFragment) {
            (None, project_key_value(&mut builder, fact_store), None)
        } else {
            let unrelated_scalar_projection =
                if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate) {
                    let scalar = scalar_key_value_aggregate(&mut builder, fact_store);
                    Some(project_key_value(&mut builder, scalar))
                } else {
                    None
                };
            let grouped = grouped_aggregate(&mut builder, fact_store);
            let grouped = if matches!(
                variant,
                RootTopKJoinAggregateVariant::DistinctAggregate
                    | RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin
                    | RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin
            ) {
                builder.add_aggregate(
                    fact_store,
                    AggregateSpec::new(
                        vec![Expr::column("k")],
                        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "v").with_distinct(true)],
                    ),
                    key_value_schema(),
                )
            } else if matches!(variant, RootTopKJoinAggregateVariant::FilteredDistinctAggregate) {
                builder.add_aggregate(
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
            } else {
                grouped
            };
            (Some(grouped), project_key_value(&mut builder, grouped), unrelated_scalar_projection)
        };
    if matches!(variant, RootTopKJoinAggregateVariant::BranchLocalWindow) {
        grouped_projection = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
    }
    if matches!(
        variant,
        RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer
            | RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow
    ) {
        let direct_grouped = grouped_aggregate(&mut builder, fact);
        grouped_projection = project_key_value(&mut builder, direct_grouped);
    }
    let root = if matches!(variant, RootTopKJoinAggregateVariant::WindowInputSortFullOuterJoin) {
        let left = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
        let right = sorted_rank_window_project_key_value(&mut builder, customer);
        full_join(&mut builder, left, right, FullJoinOutput::BothSides)
    } else if matches!(variant, RootTopKJoinAggregateVariant::WindowFullOuterLeftOnlyProjection) {
        let left = sorted_rank_window_project_key_value(&mut builder, grouped_projection);
        let right = sorted_rank_window_project_key_value(&mut builder, customer);
        full_join(&mut builder, left, right, FullJoinOutput::LeftOnly)
    } else if matches!(variant, RootTopKJoinAggregateVariant::SharedComputedFanout) {
        join(&mut builder, grouped_projection, fact_store, "root_shared_computed")
    } else if matches!(variant, RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout) {
        let grouped = grouped_source.expect("shared grouped aggregate variant should build an aggregate");
        let aggregate_branch = grouped_aggregate(&mut builder, grouped_projection);
        let aggregate_branch = project_key_value(&mut builder, aggregate_branch);
        join(&mut builder, grouped, aggregate_branch, "root_shared_grouped_aggregate")
    } else if matches!(variant, RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout) {
        let aggregate_branch = grouped_aggregate(&mut builder, grouped_projection);
        let aggregate_branch = project_key_value(&mut builder, aggregate_branch);
        join(&mut builder, grouped_projection, aggregate_branch, "root_shared_projection_aggregate")
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateUnrelatedKeyedOuterJoin) {
        let unrelated_outer = builder.add_join(
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
        );
        join(&mut builder, grouped_projection, unrelated_outer, "root_distinct_unrelated_outer")
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateBuildSideKeyedOuterJoin) {
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregatePreservedSideAggregateOuterJoin) {
        let right_aggregate = grouped_aggregate(&mut builder, customer);
        let right_aggregate = project_key_value(&mut builder, right_aggregate);
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::DistinctAggregateTransitiveKeyedOuterJoin) {
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessJoinUnrelatedScalarAggregate) {
        let unrelated_scalar_projection =
            unrelated_scalar_projection.expect("unrelated scalar variant should build a scalar projection");
        let left = join(&mut builder, grouped_projection, unrelated_scalar_projection, "left_with_unrelated_scalar");
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput) {
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput) {
        semi_join(&mut builder, grouped_projection, customer)
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput) {
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput) {
        builder.add_join(
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
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput) {
        builder.add_join(
            grouped_projection,
            customer,
            JoinSpec::new(
                JoinType::LeftAnti,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            ])
            .with_build_side(JoinSide::Right),
            key_value_schema(),
        )
    } else if matches!(
        variant,
        RootTopKJoinAggregateVariant::KeylessJoin | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
    ) {
        builder.add_join(
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
    } else if matches!(
        variant,
        RootTopKJoinAggregateVariant::ProjectedCrossJoin
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
    ) {
        builder.add_join(
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
    } else if matches!(
        variant,
        RootTopKJoinAggregateVariant::KeylessOuterCrossJoin
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
            | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer
            | RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow
    ) {
        builder.add_join(
            grouped_projection,
            customer,
            JoinSpec::new(JoinType::Left, vec![])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
            key_value_schema(),
        )
    } else {
        join(&mut builder, grouped_projection, customer, "root")
    };
    let root = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
    ) {
        let fact =
            if matches!(variant, RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters) {
                filter_source(&mut builder, fact)
            } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact) {
                source_with_known_size(&mut builder, "root-topk-join-aggregate-small-exact-fact", 4096)
            } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan) {
                source_with_known_size(&mut builder, "root-topk-join-aggregate-large-full-scan-fact", 512 << 20)
            } else {
                fact
            };
        let customer =
            if matches!(variant, RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters) {
                filter_source(&mut builder, customer)
            } else if matches!(
                variant,
                RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
                    | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
            ) {
                source_with_known_size(&mut builder, "root-topk-join-aggregate-small-exact-customer", 4096)
            } else {
                customer
            };
        let single_join = join(&mut builder, fact, customer, "single_join_aggregate");
        grouped_aggregate(&mut builder, single_join)
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate) {
        grouped_aggregate(&mut builder, root)
    } else {
        root
    };
    let mut topk_projection = if matches!(
        variant,
        RootTopKJoinAggregateVariant::SortFetchHashJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeyedResidualJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput
            | RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput
            | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
            | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
            | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
            | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
    ) {
        root
    } else {
        project_key_value(&mut builder, root)
    };
    let topk_schema = if matches!(variant, RootTopKJoinAggregateVariant::Window) {
        topk_projection = sorted_rank_window_project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindow) {
        topk_projection = sorted_rank_window_project_key_value(&mut builder, topk_projection);
        topk_projection = join(&mut builder, topk_projection, store, "post_window_root");
        topk_projection = project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction) {
        topk_projection = sorted_rank_window_project_key_value(&mut builder, topk_projection);
        topk_projection = join(&mut builder, topk_projection, store, "post_window_root");
        topk_projection = project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer) {
        let window_branch = sorted_rank_window_project_key_value(&mut builder, topk_projection);
        let direct_branch = join(&mut builder, root, store, "direct_keyless_consumer");
        topk_projection = join(&mut builder, window_branch, direct_branch, "post_window_root");
        topk_projection = project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow) {
        topk_projection = unordered_aggregate_window_project_key_value(&mut builder, topk_projection);
        topk_projection = join(&mut builder, topk_projection, store, "post_window_root");
        topk_projection = project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortedAggregateWindow) {
        let sorted = builder.add_sort(
            topk_projection,
            SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
            key_value_schema(),
        );
        topk_projection = unordered_aggregate_window_project_key_value(&mut builder, sorted);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::SortedGlobalAggregateWindow) {
        let sorted = builder.add_sort(
            topk_projection,
            SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
            key_value_schema(),
        );
        topk_projection = unordered_global_aggregate_window_project_key_value(&mut builder, sorted);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
        topk_projection = rank_filter_window_project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else if matches!(variant, RootTopKJoinAggregateVariant::UnsortedRankWindow) {
        topk_projection = rank_window_project_key_value(&mut builder, topk_projection);
        key_value_schema()
    } else {
        key_value_schema()
    };
    let mut sort = builder.add_sort(
        topk_projection,
        SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)]),
        Arc::clone(&topk_schema),
    );
    if matches!(variant, RootTopKJoinAggregateVariant::RankFilterWindowFinalProjection) {
        sort = project_key_value(&mut builder, sort);
    }
    match variant {
        RootTopKJoinAggregateVariant::MissingFetchLimit => {
            let limit = builder.add_limit(sort, LimitSpec::offset(0), topk_schema);
            builder.finish_with_host_sink(limit)
        },
        RootTopKJoinAggregateVariant::Admitted
        | RootTopKJoinAggregateVariant::MissingSourceReduction
        | RootTopKJoinAggregateVariant::SourceReductionOutsideAggregateInput
        | RootTopKJoinAggregateVariant::SharedSourceFanout
        | RootTopKJoinAggregateVariant::SharedComputedFanout
        | RootTopKJoinAggregateVariant::SharedGroupedAggregateFanout
        | RootTopKJoinAggregateVariant::SharedProjectionAggregateFanout
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
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindow
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowMissingSourceReduction
        | RootTopKJoinAggregateVariant::KeylessOuterCrossWindowWithDirectJoinConsumer
        | RootTopKJoinAggregateVariant::KeylessOuterCrossUnorderedWindow
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
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiWrongBuildSideJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftSemiRightOutputJoinInput
        | RootTopKJoinAggregateVariant::SortFetchLeftAntiJoinInput
        | RootTopKJoinAggregateVariant::SortFetchKeylessJoinInput
        | RootTopKJoinAggregateVariant::SortFetchProjectedCrossJoinInput
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregate
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSmallExact
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateSourceAdjacentFilters
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateLargeFullScan
        | RootTopKJoinAggregateVariant::SortFetchSingleHashJoinAggregateFullScan
        | RootTopKJoinAggregateVariant::SortFetchHashJoinAggregate
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanIn
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingFilter
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
        | RootTopKJoinAggregateVariant::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild
        | RootTopKJoinAggregateVariant::Window => {
            let limit = builder.add_limit(sort, LimitSpec::fetch(100), topk_schema);
            builder.finish_with_host_sink(limit)
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RootTopKJoinAggregateVariant {
    Admitted,
    MissingSourceReduction,
    SourceReductionOutsideAggregateInput,
    SharedSourceFanout,
    SharedComputedFanout,
    SharedGroupedAggregateFanout,
    SharedProjectionAggregateFanout,
    MissingGroupedAggregateFragment,
    MissingFetchLimit,
    DistinctAggregate,
    DistinctAggregateBuildSideKeyedOuterJoin,
    DistinctAggregatePreservedSideAggregateOuterJoin,
    DistinctAggregateTransitiveKeyedOuterJoin,
    DistinctAggregateUnrelatedKeyedOuterJoin,
    FilteredDistinctAggregate,
    KeylessJoin,
    KeylessJoinUnrelatedScalarAggregate,
    ProjectedCrossJoin,
    KeylessOuterCrossJoin,
    KeylessOuterCrossWindow,
    KeylessOuterCrossWindowMissingSourceReduction,
    KeylessOuterCrossWindowWithDirectJoinConsumer,
    KeylessOuterCrossUnorderedWindow,
    BranchLocalWindow,
    WindowInputSortFullOuterJoin,
    SortedAggregateWindow,
    SortedGlobalAggregateWindow,
    RankFilterWindowFinalProjection,
    WindowFullOuterLeftOnlyProjection,
    UnsortedRankWindow,
    SortFetchHashJoinInput,
    SortFetchKeyedResidualJoinInput,
    SortFetchLeftSemiJoinInput,
    SortFetchLeftSemiWrongBuildSideJoinInput,
    SortFetchLeftSemiRightOutputJoinInput,
    SortFetchLeftAntiJoinInput,
    SortFetchKeylessJoinInput,
    SortFetchProjectedCrossJoinInput,
    SortFetchSingleHashJoinAggregate,
    SortFetchSingleHashJoinAggregateSmallExact,
    SortFetchSingleHashJoinAggregateSourceAdjacentFilters,
    SortFetchSingleHashJoinAggregateLargeFullScan,
    SortFetchSingleHashJoinAggregateFullScan,
    SortFetchHashJoinAggregate,
    SortFetchNullFilteredLeftAggregateFanIn,
    SortFetchNullFilteredLeftAggregateFanInMissingFilter,
    SortFetchNullFilteredLeftAggregateFanInWrongBuildSide,
    SortFetchNullFilteredLeftAggregateFanInMarkerEscapes,
    SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse,
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
                | Self::SortFetchNullFilteredLeftAggregateFanInMissingFilter
                | Self::SortFetchNullFilteredLeftAggregateFanInWrongBuildSide
                | Self::SortFetchNullFilteredLeftAggregateFanInMarkerEscapes
                | Self::SortFetchNullFilteredLeftAggregateFanInMarkerFilterReuse
                | Self::SortFetchNullFilteredLeftAggregateFanInNonUniqueRight
                | Self::SortFetchNullFilteredLeftAggregateFanInMissingBranchReduction
                | Self::SortFetchNullFilteredLeftAggregateFanInFullBoundedReturnBuilds
                | Self::SortFetchNullFilteredLeftAggregateFanInLargeReturnBuild
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum NullFilteredReturnBuildShape {
    Reduced,
    FullBounded,
    FullLarge,
}

fn root_topk_null_filtered_aggregate_fan_in_plan(
    variant: RootTopKJoinAggregateVariant,
) -> nexus_query_engine::QueryPlan {
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
    let web = null_filtered_aggregate_branch(
        &mut builder,
        "root-topk-null-filtered-web",
        true,
        other_return_shape,
        false,
        false,
        false,
        false,
    );
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
    let filtered = builder.add_filter(
        joined,
        Expr::binary(Expr::column("v"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
        key_value_schema(),
    );
    let projected = project_key_value(&mut builder, filtered);
    let sort = builder.add_sort(
        projected,
        SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)]),
        key_value_schema(),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), key_value_schema());
    builder.finish_with_host_sink(limit)
}

fn null_filtered_aggregate_branch(
    builder: &mut QueryPlanBuilder,
    label: &str,
    reduced_fact: bool,
    return_build_shape: NullFilteredReturnBuildShape,
    missing_filter: bool,
    wrong_build_side: bool,
    marker_escapes: bool,
    marker_filter_reuse: bool,
) -> nexus_query_engine::NodeId {
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
    let joined = builder.add_join(
        fact,
        returns,
        JoinSpec::new(JoinType::Left, vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                JoinOutputColumn::right(ColumnRef::input(1, "k", "k"), marker_name.as_str()),
            ])
            .with_build_side(if wrong_build_side { JoinSide::Left } else { JoinSide::Right }),
        key_value_marker_schema(&marker_name),
    );
    let filtered = if missing_filter {
        joined
    } else if marker_filter_reuse {
        builder.add_filter(
            joined,
            Expr::binary(
                Expr::is_null(Expr::column(&marker_name)),
                BinaryOp::And,
                Expr::binary(Expr::column(&marker_name), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
            ),
            key_value_marker_schema(&marker_name),
        )
    } else {
        builder.add_filter(joined, Expr::is_null(Expr::column(&marker_name)), key_value_marker_schema(&marker_name))
    };
    let projected = if marker_escapes {
        builder.add_projection(
            filtered,
            projection_outputs(
                vec![Expr::column("k"), Expr::column("v"), Expr::column(&marker_name)],
                key_value_marker_schema(&marker_name),
            ),
        )
    } else {
        project_key_value(builder, filtered)
    };
    let date = source_with_reduction(builder, &format!("{label}-date"), true);
    let dated = join(builder, projected, date, "date");
    let aggregate = grouped_aggregate(builder, dated);
    project_key_value(builder, aggregate)
}

fn aggregate_fan_in_left_join(
    builder: &mut QueryPlanBuilder,
    left: nexus_query_engine::NodeId,
    right: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))])
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
}

fn grouped_aggregate_with_extra_key(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("right_v", DataType::Int64, true),
    ]));
    let grouped = builder.add_aggregate(
        input,
        AggregateSpec::new(
            vec![Expr::column("k"), Expr::column("v")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "right_v")],
        ),
        Arc::clone(&schema),
    );
    builder.add_projection(grouped, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
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

fn source_with_reduction(
    builder: &mut QueryPlanBuilder,
    label: &str,
    selected_subset: bool,
) -> nexus_query_engine::NodeId {
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

fn source_with_known_size(
    builder: &mut QueryPlanBuilder,
    label: &str,
    file_size_bytes: u64,
) -> nexus_query_engine::NodeId {
    let file = LocalParquetFile::new(format!("/tmp/{label}.parquet")).with_file_size_bytes(file_size_bytes);
    builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        key_value_schema(),
        vec!["k".into(), "v".into()],
    )))
}

fn filter_source(builder: &mut QueryPlanBuilder, input: nexus_query_engine::NodeId) -> nexus_query_engine::NodeId {
    builder.add_filter(
        input,
        Expr::binary(Expr::column("k"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        key_value_schema(),
    )
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

fn full_join(
    builder: &mut QueryPlanBuilder,
    left: nexus_query_engine::NodeId,
    right: nexus_query_engine::NodeId,
    output: FullJoinOutput,
) -> nexus_query_engine::NodeId {
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
    builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Full, vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))])
            .with_output_projection(output_projection)
            .with_build_side(JoinSide::Left),
        key_value_schema(),
    )
}

enum FullJoinOutput {
    BothSides,
    LeftOnly,
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

fn scalar_key_value_aggregate(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    builder.add_aggregate(
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
}

fn rank_window_project_key_value(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("rank", DataType::UInt64, false),
    ]));
    let window = builder.add_window(
        input,
        WindowSpec::new(
            vec![Expr::column("k")],
            vec![SortExpr::new(Expr::column("v"), true, false)],
            vec![WindowFunction::Rank { alias: "rank".to_string() }],
        ),
        window_schema,
    );
    project_key_value(builder, window)
}

fn sorted_rank_window_project_key_value(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let sort = builder.add_sort(
        input,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("k"), false, false),
            SortExpr::new(Expr::column("v"), true, false),
        ]),
        key_value_schema(),
    );
    rank_window_project_key_value(builder, sort)
}

fn unordered_aggregate_window_project_key_value(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("window_v", DataType::Int64, true),
    ]));
    let window = builder.add_window(
        input,
        WindowSpec::new(
            vec![Expr::column("k")],
            vec![],
            vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("v"), "window_v")],
        ),
        window_schema,
    );
    project_key_value(builder, window)
}

fn unordered_global_aggregate_window_project_key_value(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("window_v", DataType::Int64, true),
    ]));
    let window = builder.add_window(
        input,
        WindowSpec::new(
            vec![],
            vec![],
            vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("v"), "window_v")],
        ),
        window_schema,
    );
    project_key_value(builder, window)
}

fn rank_filter_window_project_key_value(
    builder: &mut QueryPlanBuilder,
    input: nexus_query_engine::NodeId,
) -> nexus_query_engine::NodeId {
    let sorted = builder.add_sort(
        input,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("k"), false, false),
            SortExpr::new(Expr::column("v"), true, false),
        ]),
        key_value_schema(),
    );
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new("rank", DataType::UInt64, false),
    ]));
    let window = builder.add_window(
        sorted,
        WindowSpec::new(
            vec![Expr::column("k")],
            vec![SortExpr::new(Expr::column("v"), true, false)],
            vec![WindowFunction::Rank { alias: "rank".to_string() }],
        ),
        Arc::clone(&window_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rank"), BinaryOp::Gt, Expr::literal(LiteralValue::UInt64(0))),
        window_schema,
    );
    project_key_value(builder, filter)
}

fn project_key_value(builder: &mut QueryPlanBuilder, input: nexus_query_engine::NodeId) -> nexus_query_engine::NodeId {
    builder.add_projection(input, projection_outputs(vec![Expr::column("k"), Expr::column("v")], key_value_schema()))
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]))
}

fn key_value_marker_schema(marker_name: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Int64, true),
        Field::new(marker_name, DataType::Int32, true),
    ]))
}
