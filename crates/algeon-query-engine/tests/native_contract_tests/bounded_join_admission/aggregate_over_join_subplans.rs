use std::sync::Arc;

use query_engine::{
    capability::CapabilityEnvelope,
    dimension_pruning::local_dimension_pruning_resolver,
    optimizer::{NativeOptimizerOptions, ProjectionSimplificationRejectedReason, optimize_query_plan},
    plan::{NodeId, QueryPlan},
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_admits_variants_through_composed_path()
{
    let plan =
        keyless_conditional_aggregate_over_join_subplan_inputs_plan(AggregateOverJoinSubplanInputsVariant::Admitted);

    assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    assert!(
        !plan.capability_analysis(rapids_analysis_context()).admits_keyless_conditional_join_input_subplan_pipeline()
    );
    assert!(
        plan.capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
    );
    assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none());
    let preaggregate_leaf_plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        AggregateOverJoinSubplanInputsVariant::PreAggregateLeaf,
    );
    assert!(
        preaggregate_leaf_plan
            .capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
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
        // they compile; what stays narrow is the aggregate-over-join envelope
        // itself.
        let plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(variant);

        assert!(!plan.capability_analysis(rapids_analysis_context()).admits_root_join(), "{variant:?}");
        assert!(
            !plan
                .capability_analysis(rapids_analysis_context())
                .admits_keyless_conditional_join_input_subplan_pipeline(),
            "{variant:?}"
        );
        assert!(
            !plan
                .capability_analysis(rapids_analysis_context())
                .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(unsupported_facts(&plan, rapids_analysis_context()).is_none(), "{variant:?}");
    }

    // A fetch limit after the root turns the tail into sort+fetch, which is the
    // root TopK join/aggregate composition over the same keyless conditional and
    // grouped-aggregate fragments. It stays outside this envelope's post-root
    // stage contract but is executable and admitted by the root TopK envelope;
    // capacity is the allocator's, enforced as a typed cap failure at runtime,
    // not an admission gate.
    let limit_after_root_plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        AggregateOverJoinSubplanInputsVariant::LimitAfterRoot,
    );
    assert!(!limit_after_root_plan.capability_analysis(rapids_analysis_context()).admits_root_join());
    assert!(
        !limit_after_root_plan
            .capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_input_subplan_pipeline()
    );
    assert!(
        !limit_after_root_plan
            .capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
    );
    let admission = bounded_pipeline_admission(&limit_after_root_plan)
        .expect("bounded admission should not fail")
        .expect("LimitAfterRoot should be admitted through the root TopK join/aggregate envelope");
    assert_eq!(admission.envelope(), CapabilityEnvelope::RootTopKJoinAggregate);
    assert!(
        unsupported_facts(&limit_after_root_plan, rapids_analysis_context()).is_none(),
        "LimitAfterRoot is admitted, not a support gap"
    );
}

#[test]
fn bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_admission_uses_capability_evidence() {
    let plan =
        keyless_conditional_aggregate_over_join_subplan_inputs_plan(AggregateOverJoinSubplanInputsVariant::Admitted);
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert_eq!(
        plan.capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
        analysis.admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
    );
    assert_eq!(
        analysis.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission,
        CapabilityDecision::Satisfied
    );
    assert!(analysis.invariants.iter().any(|invariant| {
        matches!(invariant.kind, SubgraphInvariantKind::PostRootStageOrder { .. })
            && invariant.decision == CapabilityDecision::Satisfied
    }));
    // Sort acyclicity is no longer a plan invariant: the compiled schedule's
    // barrier dominance check owns it, so no FinalBarrierOrder evidence is
    // produced here.
    assert!(
        !analysis
            .invariants
            .iter()
            .any(|invariant| { matches!(invariant.kind, SubgraphInvariantKind::FinalBarrierOrder) })
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
        optimized
            .plan
            .capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
        "projection simplification must preserve bounded aggregate-over-join input wrappers"
    );
    assert!(
        optimized.diagnostics.projection_simplification.candidates.iter().any(|candidate| {
            candidate.rejected_reason == Some(ProjectionSimplificationRejectedReason::ShapeContract)
        }),
        "protected projection wrappers should be reported as shape-contract rejections"
    );
}

#[test]
fn bounded_keyless_conditional_join_aggregate_over_join_subplan_inputs_rejects_bad_subplan_locally() {
    let plan = keyless_conditional_aggregate_over_join_subplan_inputs_plan(
        AggregateOverJoinSubplanInputsVariant::SortBelowAggregateInput,
    );
    let analysis = plan.capability_analysis(rapids_analysis_context());

    assert!(
        !plan
            .capability_analysis(rapids_analysis_context())
            .admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline()
    );
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.keyless_conditional_join_aggregate_over_join_subplan_inputs_admission
    else {
        panic!("sort below aggregate input should reject aggregate-over-join subplan inputs admission");
    };
    assert_eq!(*reason, "aggregate_over_join_topk_input_requires_join_subplan");
    assert!(facts.contains_key("node_id"));
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
