use std::sync::Arc;

use query_engine::{
    dimension_pruning::local_dimension_pruning_resolver,
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    plan::{NodeId, QueryPlan},
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_admits_variants_through_composed_path()
{
    let plan =
        keyless_conditional_aggregate_over_join_subplan_inputs_plan(AggregateOverJoinSubplanInputsVariant::Admitted);

    assert!(
        unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_none()
    );
    let preaggregate_leaf_plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        AggregateOverJoinSubplanInputsVariant::PreAggregateLeaf,
    );
    assert!(
        unsupported_facts(
            &preaggregate_leaf_plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context()
        )
        .is_none()
    );

    for variant in [
        AggregateOverJoinSubplanInputsVariant::GroupedAggregateOverUnsupportedInput,
        AggregateOverJoinSubplanInputsVariant::ScalarAggregateOverUnsupportedInput,
        AggregateOverJoinSubplanInputsVariant::SortBelowAggregateInput,
        AggregateOverJoinSubplanInputsVariant::LimitBelowAggregateInput,
        AggregateOverJoinSubplanInputsVariant::BareGroupedAggregateInput,
        AggregateOverJoinSubplanInputsVariant::BareJoinInput,
        AggregateOverJoinSubplanInputsVariant::AggregateOverAggregateShortcut,
        AggregateOverJoinSubplanInputsVariant::NestedUnsupportedJoinInput,
        AggregateOverJoinSubplanInputsVariant::PostRootReversedStages,
        AggregateOverJoinSubplanInputsVariant::PostRootRepeatedProjection,
        AggregateOverJoinSubplanInputsVariant::PostRootRepeatedSort,
        AggregateOverJoinSubplanInputsVariant::RootOutputIncludesRight,
        AggregateOverJoinSubplanInputsVariant::PreAggregateLeafWithNonSum,
    ] {
        // S3 admits these variants through the compiled task contracts once
        // they compile.
        let plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(variant);

        assert!(
            unsupported_facts(&plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).is_none(),
            "{variant:?}"
        );
    }

    // A fetch limit after the root turns the tail into sort+fetch over the same
    // keyless conditional and grouped-aggregate fragments. It is executable and
    // admitted; capacity is the allocator's, enforced as a typed cap failure at
    // runtime, not an admission gate.
    let limit_after_root_plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        AggregateOverJoinSubplanInputsVariant::LimitAfterRoot,
    );
    assert!(
        bounded_pipeline_admission(&limit_after_root_plan).expect("bounded admission should not fail").is_some(),
        "LimitAfterRoot should be admitted"
    );
    assert!(
        unsupported_facts(
            &limit_after_root_plan,
            StatisticalAggregateExecutionMode::default(),
            rapids_analysis_context()
        )
        .is_none(),
        "LimitAfterRoot is admitted, not a support gap"
    );
}

#[test]
fn bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_survives_native_optimizer() {
    let plan =
        keyless_conditional_aggregate_over_join_subplan_inputs_plan(AggregateOverJoinSubplanInputsVariant::Admitted);
    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );

    assert!(
        unsupported_facts(&optimized.plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
            .is_none(),
        "projection simplification must preserve bounded aggregate-over-join input wrappers"
    );
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AggregateOverJoinSubplanInputsVariant {
    Admitted,
    GroupedAggregateOverUnsupportedInput,
    ScalarAggregateOverUnsupportedInput,
    SortBelowAggregateInput,
    LimitBelowAggregateInput,
    BareGroupedAggregateInput,
    BareJoinInput,
    AggregateOverAggregateShortcut,
    NestedUnsupportedJoinInput,
    PostRootReversedStages,
    PostRootRepeatedProjection,
    PostRootRepeatedSort,
    RootOutputIncludesRight,
    LimitAfterRoot,
    PreAggregateLeaf,
    PreAggregateLeafWithNonSum,
}

pub(crate) fn keyless_conditional_aggregate_over_join_subplan_inputs_plan(
    variant: AggregateOverJoinSubplanInputsVariant,
) -> QueryPlan {
    let left_join_schema = aggregate_over_join_subplan_join_subtree_schema();
    let right_join_schema = aggregate_over_join_subplan_join_subtree_schema();
    let left_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let right_aggregate_schema = Arc::new(Schema::new(vec![Field::new("threshold", DataType::Int64, true)]));
    let root_output_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let root_output_with_threshold_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
        Field::new("threshold", DataType::Int64, true),
    ]));
    let preaggregate_leaf_function = match variant {
        AggregateOverJoinSubplanInputsVariant::PreAggregateLeaf => Some(AggregateFunction::Sum),
        AggregateOverJoinSubplanInputsVariant::PreAggregateLeafWithNonSum => Some(AggregateFunction::Min),
        _ => None,
    };

    let mut builder = QueryPlanBuilder::new();
    let left_join_input =
        if matches!(variant, AggregateOverJoinSubplanInputsVariant::GroupedAggregateOverUnsupportedInput) {
            builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/aggregate-subplan-left-unsupported-partsupp.parquet",
                Arc::clone(&left_join_schema),
                vec!["ps_partkey".into(), "ps_supplycost".into(), "ps_availqty".into()],
            )))
        } else {
            build_aggregate_over_join_subplan_bounded_join_subtree(
                &mut builder,
                "left",
                matches!(variant, AggregateOverJoinSubplanInputsVariant::NestedUnsupportedJoinInput),
                preaggregate_leaf_function,
            )
        };
    let left_join_input = match variant {
        AggregateOverJoinSubplanInputsVariant::SortBelowAggregateInput => builder.add_sort(
            left_join_input,
            SortSpec::new(vec![SortExpr::new(Expr::column("ps_partkey"), false, false)]),
            Arc::clone(&left_join_schema),
        ),
        AggregateOverJoinSubplanInputsVariant::LimitBelowAggregateInput => {
            let sort = builder.add_sort(
                left_join_input,
                SortSpec::new(vec![SortExpr::new(Expr::column("ps_partkey"), false, false)]),
                Arc::clone(&left_join_schema),
            );
            builder.add_limit(sort, LimitSpec::fetch(10), Arc::clone(&left_join_schema))
        },
        AggregateOverJoinSubplanInputsVariant::AggregateOverAggregateShortcut => builder.add_aggregate(
            left_join_input,
            AggregateSpec::new(
                vec![Expr::column("ps_partkey")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ps_supplycost"), "value")],
            ),
            Arc::clone(&left_aggregate_schema),
        ),
        AggregateOverJoinSubplanInputsVariant::Admitted
        | AggregateOverJoinSubplanInputsVariant::PreAggregateLeaf
        | AggregateOverJoinSubplanInputsVariant::PreAggregateLeafWithNonSum
        | AggregateOverJoinSubplanInputsVariant::GroupedAggregateOverUnsupportedInput
        | AggregateOverJoinSubplanInputsVariant::ScalarAggregateOverUnsupportedInput
        | AggregateOverJoinSubplanInputsVariant::BareGroupedAggregateInput
        | AggregateOverJoinSubplanInputsVariant::BareJoinInput
        | AggregateOverJoinSubplanInputsVariant::NestedUnsupportedJoinInput
        | AggregateOverJoinSubplanInputsVariant::PostRootReversedStages
        | AggregateOverJoinSubplanInputsVariant::PostRootRepeatedProjection
        | AggregateOverJoinSubplanInputsVariant::PostRootRepeatedSort
        | AggregateOverJoinSubplanInputsVariant::RootOutputIncludesRight
        | AggregateOverJoinSubplanInputsVariant::LimitAfterRoot => left_join_input,
    };
    let left_input = if matches!(variant, AggregateOverJoinSubplanInputsVariant::BareJoinInput) {
        left_join_input
    } else {
        let left_aggregate = builder.add_aggregate(
            left_join_input,
            AggregateSpec::new(
                vec![Expr::column("ps_partkey")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ps_supplycost"), "value")],
            ),
            Arc::clone(&left_aggregate_schema),
        );
        if matches!(variant, AggregateOverJoinSubplanInputsVariant::BareGroupedAggregateInput) {
            left_aggregate
        } else {
            builder.add_projection(
                left_aggregate,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&left_aggregate_schema),
                ),
            )
        }
    };

    let right_join_input =
        if matches!(variant, AggregateOverJoinSubplanInputsVariant::ScalarAggregateOverUnsupportedInput) {
            builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/aggregate-subplan-right-unsupported-partsupp.parquet",
                Arc::clone(&right_join_schema),
                vec!["ps_partkey".into(), "ps_supplycost".into(), "ps_availqty".into()],
            )))
        } else {
            build_aggregate_over_join_subplan_bounded_join_subtree(&mut builder, "right", false, None)
        };
    let right_aggregate = builder.add_aggregate(
        right_join_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ps_supplycost"), "threshold")],
        ),
        Arc::clone(&right_aggregate_schema),
    );
    let right_input = builder.add_projection(
        right_aggregate,
        projection_outputs(vec![Expr::column("threshold")], Arc::clone(&right_aggregate_schema)),
    );

    let root_projection = if matches!(variant, AggregateOverJoinSubplanInputsVariant::RootOutputIncludesRight) {
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
            JoinOutputColumn::right(ColumnRef::input(1, "threshold", "threshold"), "threshold"),
        ]
    } else {
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ]
    };
    let root_schema = if matches!(variant, AggregateOverJoinSubplanInputsVariant::RootOutputIncludesRight) {
        Arc::clone(&root_output_with_threshold_schema)
    } else {
        Arc::clone(&root_output_schema)
    };
    let root = builder.add_join(
        left_input,
        right_input,
        JoinSpec::new(JoinType::Inner, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "value", "value"),
                BinaryOp::Gt,
                Expr::input_column(1, "threshold", "threshold"),
            ))
            .with_output_projection(root_projection)
            .with_build_side(JoinSide::Left),
        root_schema,
    );
    let output = match variant {
        AggregateOverJoinSubplanInputsVariant::PostRootReversedStages => {
            let sort = builder.add_sort(
                root,
                SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
                Arc::clone(&root_output_schema),
            );
            builder.add_projection(
                sort,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&root_output_schema),
                ),
            )
        },
        AggregateOverJoinSubplanInputsVariant::PostRootRepeatedProjection => {
            let projection = builder.add_projection(
                root,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&root_output_schema),
                ),
            );
            builder.add_projection(
                projection,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&root_output_schema),
                ),
            )
        },
        AggregateOverJoinSubplanInputsVariant::PostRootRepeatedSort => {
            let sort = builder.add_sort(
                root,
                SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
                Arc::clone(&root_output_schema),
            );
            builder.add_sort(
                sort,
                SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
                Arc::clone(&root_output_schema),
            )
        },
        AggregateOverJoinSubplanInputsVariant::LimitAfterRoot => {
            let projection = builder.add_projection(
                root,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&root_output_schema),
                ),
            );
            let sort = builder.add_sort(
                projection,
                SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
                Arc::clone(&root_output_schema),
            );
            builder.add_limit(sort, LimitSpec::fetch(100), root_output_schema)
        },
        AggregateOverJoinSubplanInputsVariant::Admitted
        | AggregateOverJoinSubplanInputsVariant::PreAggregateLeaf
        | AggregateOverJoinSubplanInputsVariant::PreAggregateLeafWithNonSum
        | AggregateOverJoinSubplanInputsVariant::GroupedAggregateOverUnsupportedInput
        | AggregateOverJoinSubplanInputsVariant::ScalarAggregateOverUnsupportedInput
        | AggregateOverJoinSubplanInputsVariant::SortBelowAggregateInput
        | AggregateOverJoinSubplanInputsVariant::LimitBelowAggregateInput
        | AggregateOverJoinSubplanInputsVariant::BareGroupedAggregateInput
        | AggregateOverJoinSubplanInputsVariant::BareJoinInput
        | AggregateOverJoinSubplanInputsVariant::AggregateOverAggregateShortcut
        | AggregateOverJoinSubplanInputsVariant::NestedUnsupportedJoinInput
        | AggregateOverJoinSubplanInputsVariant::RootOutputIncludesRight => {
            let projection = builder.add_projection(
                root,
                projection_outputs(
                    vec![Expr::column("ps_partkey"), Expr::column("value")],
                    Arc::clone(&root_output_schema),
                ),
            );
            builder.add_sort(
                projection,
                SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
                Arc::clone(&root_output_schema),
            )
        },
    };
    builder.finish_with_host_sink(output)
}

fn build_aggregate_over_join_subplan_bounded_join_subtree(
    builder: &mut QueryPlanBuilder,
    label: &str,
    nested_unsupported_join: bool,
    preaggregate_leaf_function: Option<AggregateFunction>,
) -> NodeId {
    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
    ]));
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_nationkey", DataType::Int32, false),
    ]));
    let nation_schema = Arc::new(Schema::new(vec![Field::new("n_nationkey", DataType::Int32, false)]));
    let nation_preaggregate_schema = Arc::new(Schema::new(vec![
        Field::new("n_nationkey", DataType::Int32, false),
        Field::new("n_sum", DataType::Int64, true),
    ]));
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
        Field::new("s_nationkey", DataType::Int32, false),
    ]));
    let final_join_schema = aggregate_over_join_subplan_join_subtree_schema();

    let partsupp = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/aggregate-subplan-{label}-partsupp.parquet"),
        partsupp_schema,
        vec!["ps_partkey".into(), "ps_suppkey".into(), "ps_supplycost".into(), "ps_availqty".into()],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/aggregate-subplan-{label}-supplier.parquet"),
        supplier_schema,
        vec!["s_suppkey".into(), "s_nationkey".into()],
    )));
    let nation = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/aggregate-subplan-{label}-nation.parquet"),
        nation_schema,
        vec!["n_nationkey".into()],
    )));
    let nation = if let Some(function) = preaggregate_leaf_function {
        builder.add_aggregate(
            nation,
            AggregateSpec::new(
                vec![Expr::column("n_nationkey")],
                vec![AggregateExpr::new(function, Expr::column("n_nationkey"), "n_sum")],
            ),
            nation_preaggregate_schema,
        )
    } else {
        nation
    };

    let first_join_spec = if nested_unsupported_join {
        JoinSpec::new(JoinType::Cross, Vec::new())
    } else {
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                ColumnRef::input(1, "s_suppkey", "s_suppkey"),
            )],
        )
    };
    let first_join = builder.add_join(
        partsupp,
        supplier,
        first_join_spec
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
                JoinOutputColumn::left(ColumnRef::input(0, "ps_availqty", "ps_availqty"), "ps_availqty"),
                JoinOutputColumn::right(ColumnRef::input(1, "s_nationkey", "s_nationkey"), "s_nationkey"),
            ])
            .with_build_side(JoinSide::Right),
        first_join_schema,
    );
    builder.add_join(
        first_join,
        nation,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_nationkey", "s_nationkey"),
                ColumnRef::input(1, "n_nationkey", "n_nationkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
            JoinOutputColumn::left(ColumnRef::input(0, "ps_availqty", "ps_availqty"), "ps_availqty"),
        ])
        .with_build_side(JoinSide::Right),
        final_join_schema,
    )
}

fn aggregate_over_join_subplan_join_subtree_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int64, false),
        Field::new("ps_availqty", DataType::Int32, false),
    ]))
}
