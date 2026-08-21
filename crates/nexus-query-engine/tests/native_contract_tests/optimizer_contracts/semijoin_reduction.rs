#[test]
fn semijoin_reduction_pushes_left_semi_into_left_inner_join_child() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Left);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.crossed_join_node_ids.len(), 1);
    assert_eq!(
        candidate.cost_gate.as_ref().map(|gate| &gate.signal),
        Some(&SemijoinReductionCostSignal::AggregateKeysetShape)
    );
    assert_eq!(candidate.key_mappings[0].before_column, "x_key");
    assert_eq!(candidate.key_mappings[0].after_column.as_deref(), Some("x_key"));

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_left_semi_input(&result.plan, root_join.inputs[0], "x_key");
    assert_source_input(&result.plan, root_join.inputs[1], "y");
}

#[test]
fn semijoin_reduction_pushes_left_semi_into_right_inner_join_child() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, root_join.inputs[0], "x");
    assert_left_semi_input(&result.plan, root_join.inputs[1], "y_key");
}

#[test]
fn semijoin_reduction_pushes_to_deepest_safe_inner_join_subtree() {
    let plan = root_semijoin_over_nested_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].crossed_join_node_ids.len(), 2);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the outer crossed inner join");
    };
    assert_eq!(root_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, root_join.inputs[1], "c");

    let PlanNode::Join(inner_join) = &result.plan.node(root_join.inputs[0]).expect("inner join").kind else {
        panic!("outer left input should be the rebuilt inner join");
    };
    assert_eq!(inner_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, inner_join.inputs[0], "a");
    assert_left_semi_input(&result.plan, inner_join.inputs[1], "b_key");
}

#[test]
fn semijoin_reduction_pushes_across_projections_and_filters_between_joins() {
    let plan = root_semijoin_over_projection_filter_nested_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].crossed_join_node_ids.len(), 2);
    assert!(result.plan.unreachable_node_ids().is_empty());
}

#[test]
fn semijoin_reduction_accepts_filtered_dimension_keyset_for_transitive_fact_filtering() {
    let plan = root_semijoin_over_fact_date_dimension_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.crossed_join_node_ids.len(), 1);
    assert_eq!(candidate.key_mappings[0].before_column, "d_date");
    assert_eq!(candidate.key_mappings[0].after_column.as_deref(), Some("d_date"));
    assert!(matches!(
        candidate.cost_gate.as_ref().map(|gate| &gate.signal),
        Some(SemijoinReductionCostSignal::FilteredSourceKeysetShape {
            filtered_source_count: 1,
            target_rows: Some(73_049),
            crossed_join_count: 1,
            ..
        })
    ));

    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(&route.producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. })
                && route.producer_columns == vec!["d_date_sk".to_string()]
                && route.consumer_columns == vec!["ss_sold_date_sk".to_string()]
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.consumer_lowering == DynamicFilterConsumerLowering::ReaderPredicate
        }),
        "optimized dimension-side semijoin should expose a reusable date_sk dynamic-filter route to the fact source"
    );
}

#[test]
fn semijoin_reduction_accepts_filtered_left_semi_dimension_keyset_for_transitive_fact_filtering() {
    let plan = root_semijoin_over_fact_date_dimension_plan_with_left_semi_week_keyset();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 2);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(result.plan.unreachable_node_ids().is_empty());

    let date_candidate = result
        .diagnostics
        .semijoin_reduction
        .candidates
        .iter()
        .find(|candidate| {
            candidate
                .key_mappings
                .iter()
                .any(|mapping| mapping.before_column == "d_date" && mapping.after_column.as_deref() == Some("d_date"))
        })
        .expect("outer date semijoin candidate should be considered");
    assert!(date_candidate.applied);
    assert_eq!(date_candidate.crossed_join_node_ids.len(), 1);
    assert!(matches!(
        date_candidate.cost_gate.as_ref().map(|gate| &gate.signal),
        Some(SemijoinReductionCostSignal::FilteredSourceKeysetShape {
            filtered_source_count: 1,
            target_rows: Some(73_049),
            crossed_join_count: 1,
            ..
        })
    ));

    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(&route.producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. })
                && route.producer_columns == vec!["d_date_sk".to_string()]
                && route.consumer_columns == vec!["ss_sold_date_sk".to_string()]
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.consumer_lowering == DynamicFilterConsumerLowering::ReaderPredicate
        }),
        "optimized left-semi dimension keyset should expose a reusable date_sk dynamic-filter route to the fact source"
    );
}

#[test]
fn semijoin_reduction_bounded_gate_allows_later_passes_to_route_dimension_keyset_filter() {
    let plan = root_semijoin_over_fact_date_dimension_plan();
    let options = NativeOptimizerOptions {
        semijoin_reduction: SemijoinReductionOptions::default()
            .with_admission_mode(SemijoinReductionAdmissionMode::Bounded),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(
        candidate.admission_decision,
        Some(nexus_query_engine::optimizer::SemijoinReductionAdmissionDecision::BoundedKept)
    );
    assert_eq!(candidate.original_admitted, Some(true));
    assert_eq!(candidate.rewritten_admitted, Some(true));
    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(&route.producer, DynamicFilterProducer::HashJoinBuild { build_side: JoinSide::Right, .. })
                && route.producer_columns == vec!["d_date_sk".to_string()]
                && route.consumer_columns == vec!["ss_sold_date_sk".to_string()]
                && matches!(&route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.consumer_lowering == DynamicFilterConsumerLowering::ReaderPredicate
        }),
        "strict-gated semijoin rewrite should remain available to downstream dynamic-filter routing"
    );
}

#[test]
fn semijoin_reduction_rejects_unestimated_non_keyset_shape() {
    let plan = root_semijoin_over_inner_join_plan_with_raw_keyset(SemijoinTarget::Right);

    let result = optimize_query_plan(plan, &semijoin_reduction_only_options(SemijoinReductionOptions::default()));

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(SemijoinReductionRejectedReason::CostGateRejected));
    assert_eq!(candidate.cost_gate.as_ref().map(|gate| &gate.signal), Some(&SemijoinReductionCostSignal::Unavailable));
}

#[test]
fn semijoin_reduction_bounded_admission_gate_keeps_rewrite_when_original_is_not_yet_admitted() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);
    let options = semijoin_reduction_only_options(
        SemijoinReductionOptions::default().with_admission_mode(SemijoinReductionAdmissionMode::Bounded),
    );

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 1);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(
        candidate.admission_decision,
        Some(nexus_query_engine::optimizer::SemijoinReductionAdmissionDecision::BoundedKept)
    );
    assert_eq!(candidate.original_admitted, Some(false));
    assert_eq!(candidate.rewritten_admitted, Some(false));
}

#[test]
fn semijoin_reduction_preserves_plan_captured_join_strategy_options_through_rewrite() {
    let keyed_hash_join_strategy =
        KeyedHashJoinStrategyOptions::default().with_small_reusable_hash_build_row_threshold(11);
    let stamped_plan = optimize_query_plan(
        root_semijoin_over_inner_join_plan(SemijoinTarget::Right),
        &NativeOptimizerOptions { keyed_hash_join_strategy, ..NativeOptimizerOptions::disabled() },
    )
    .plan;

    let result = optimize_semijoin_reduction(stamped_plan, &SemijoinReductionOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.applied_count, 1);
    assert_eq!(result.plan.keyed_hash_join_strategy_options(), keyed_hash_join_strategy);
}

#[test]
fn semijoin_reduction_rejects_outer_crossed_joins() {
    for join_type in [JoinType::Left, JoinType::Full] {
        assert_semijoin_rejection(
            root_semijoin_over_crossed_join_type_plan(join_type),
            SemijoinReductionRejectedReason::UnsupportedCrossedJoinType,
        );
    }
}

#[test]
fn semijoin_reduction_rejects_ambiguous_lineage_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_duplicate_output_name_plan(),
        SemijoinReductionRejectedReason::AmbiguousKeyLineage,
    );
}

#[test]
fn semijoin_reduction_rejects_expression_derived_key_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_expression_key_projection_plan(),
        SemijoinReductionRejectedReason::ExpressionDerivedKey,
    );
}

#[test]
fn semijoin_reduction_rejects_dropped_key_lineage_candidate() {
    assert_semijoin_rejection(
        root_semijoin_over_dropped_key_projection_plan(),
        SemijoinReductionRejectedReason::DroppedKeyLineage,
    );
}

#[test]
fn semijoin_reduction_rejects_unsupported_null_semantics() {
    assert_semijoin_rejection(
        root_semijoin_over_inner_join_plan_with_null_equality(
            JoinNullEquality::NullsMatch,
            JoinNullEquality::NullsDoNotMatch,
        ),
        SemijoinReductionRejectedReason::UnsupportedNullEquality,
    );
    assert_semijoin_rejection(
        root_semijoin_over_inner_join_plan_with_null_equality(
            JoinNullEquality::NullsDoNotMatch,
            JoinNullEquality::NullsMatch,
        ),
        SemijoinReductionRejectedReason::UnsupportedNullEquality,
    );
}

enum SemijoinTarget {
    Left,
    Right,
}

fn root_semijoin_over_inner_join_plan(target: SemijoinTarget) -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, true)
}

fn root_semijoin_over_inner_join_plan_with_raw_keyset(target: SemijoinTarget) -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, false)
}

fn root_semijoin_over_inner_join_plan_impl(
    target: SemijoinTarget,
    aggregate_keyset: bool,
) -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        target,
        aggregate_keyset,
        JoinType::Inner,
        JoinNullEquality::NullsDoNotMatch,
        JoinNullEquality::NullsDoNotMatch,
    )
}

fn root_semijoin_over_inner_join_plan_with_null_equality(
    semijoin_null_equality: JoinNullEquality,
    crossed_join_null_equality: JoinNullEquality,
) -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        JoinType::Inner,
        semijoin_null_equality,
        crossed_join_null_equality,
    )
}

fn root_semijoin_over_crossed_join_type_plan(join_type: JoinType) -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        join_type,
        JoinNullEquality::NullsDoNotMatch,
        JoinNullEquality::NullsDoNotMatch,
    )
}

fn root_semijoin_over_inner_join_plan_with_options(
    target: SemijoinTarget,
    aggregate_keyset: bool,
    crossed_join_type: JoinType,
    semijoin_null_equality: JoinNullEquality,
    crossed_join_null_equality: JoinNullEquality,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key", "x_payload"]));
    let y = builder.add_source(gpu_source("y", &["y_key", "y_payload"]));
    let key_name = match target {
        SemijoinTarget::Left => "x_key",
        SemijoinTarget::Right => "y_key",
    };
    let keyset = if aggregate_keyset {
        aggregate_keyset_source(&mut builder, "keyset", key_name)
    } else {
        builder.add_source(gpu_source("keyset", &[key_name]))
    };
    let xy_schema = schema(&["x_key", "x_payload", "y_key", "y_payload"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            crossed_join_type,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "x_payload", "x_payload"), "x_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_payload", "y_payload"), "y_payload"),
        ])
        .with_null_equality(crossed_join_null_equality),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, key_name, key_name), ColumnRef::input(1, key_name, key_name))],
        )
        .with_output_projection(left_projection(&["x_key", "x_payload", "y_key", "y_payload"]))
        .with_null_equality(semijoin_null_equality),
        xy_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_duplicate_output_name_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "dup");
    let duplicate_schema = schema(&["dup", "dup"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "dup"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "dup"),
        ]),
        Arc::clone(&duplicate_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "dup", "dup"), ColumnRef::input(1, "dup", "dup"))],
        )
        .with_output_projection(left_projection(&["dup", "dup"])),
        duplicate_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_expression_key_projection_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "y_key");
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key")]),
        schema(&["y_key"]),
    );
    let projected = builder.add_projection(
        xy,
        vec![ProjectionOutput::new(
            Expr::binary(Expr::column("y_key"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0))),
            Field::new("y_key", DataType::Int32, false),
        )],
    );
    let semi = builder.add_join(
        projected,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "y_key", "y_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(left_projection(&["y_key"])),
        schema(&["y_key"]),
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_dropped_key_projection_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let x = builder.add_source(gpu_source("x", &["x_key"]));
    let y = builder.add_source(gpu_source("y", &["y_key"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "y_key");
    let xy_schema = schema(&["x_key", "y_key"]);
    let xy = builder.add_join(
        x,
        y,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key")]),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "y_key", "y_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(left_projection(&["x_key", "y_key"])),
        xy_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_nested_inner_join_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(gpu_source("a", &["a_key", "a_payload"]));
    let b = builder.add_source(gpu_source("b", &["b_key", "b_payload"]));
    let c = builder.add_source(gpu_source("c", &["c_key", "c_payload"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "b_key");

    let ab_schema = schema(&["a_key", "a_payload", "b_key", "b_payload"]);
    let ab = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_key", "a_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_payload", "b_payload"), "b_payload"),
        ]),
        ab_schema,
    );

    let abc_schema = schema(&["a_key", "a_payload", "b_key", "b_payload", "c_key", "c_payload"]);
    let abc = builder.add_join(
        ab,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "c_key", "c_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_payload", "b_payload"), "b_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_key", "c_key"), "c_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_payload", "c_payload"), "c_payload"),
        ]),
        Arc::clone(&abc_schema),
    );
    let semi = builder.add_join(
        abc,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(left_projection(&[
            "a_key",
            "a_payload",
            "b_key",
            "b_payload",
            "c_key",
            "c_payload",
        ])),
        abc_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_projection_filter_nested_inner_join_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(gpu_source("a", &["a_key", "a_payload"]));
    let b = builder.add_source(gpu_source("b", &["b_key", "b_payload"]));
    let c = builder.add_source(gpu_source("c", &["c_key", "c_payload"]));
    let keyset = aggregate_keyset_source(&mut builder, "keyset", "b_key");

    let ab_schema = schema(&["a_key", "a_payload", "b_key", "b_payload"]);
    let ab = builder.add_join(
        a,
        b,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "a_key", "a_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "b_payload", "b_payload"), "b_payload"),
        ]),
        ab_schema,
    );

    let ab_filter = builder.add_filter(
        ab,
        Expr::binary(Expr::column("a_key"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        schema(&["a_key", "a_payload", "b_key", "b_payload"]),
    );
    let ab_proj = builder.add_projection(
        ab_filter,
        vec![
            ProjectionOutput::new(Expr::column("a_key"), Field::new("a_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("a_payload"), Field::new("a_payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("b_key"), Field::new("b_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("b_payload"), Field::new("b_payload", DataType::Int32, false)),
        ],
    );

    let abc_schema = schema(&["a_key", "a_payload", "b_key", "b_payload", "c_key", "c_payload"]);
    let abc = builder.add_join(
        ab_proj,
        c,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "c_key", "c_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "a_payload", "a_payload"), "a_payload"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_key", "b_key"), "b_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "b_payload", "b_payload"), "b_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_key", "c_key"), "c_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "c_payload", "c_payload"), "c_payload"),
        ]),
        Arc::clone(&abc_schema),
    );

    let abc_filter = builder.add_filter(
        abc,
        Expr::binary(Expr::column("c_key"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&abc_schema),
    );
    let abc_proj = builder.add_projection(
        abc_filter,
        vec![
            ProjectionOutput::new(Expr::column("a_key"), Field::new("a_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("a_payload"), Field::new("a_payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("b_key"), Field::new("b_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("b_payload"), Field::new("b_payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("c_key"), Field::new("c_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("c_payload"), Field::new("c_payload", DataType::Int32, false)),
        ],
    );

    let semi = builder.add_join(
        abc_proj,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "b_key", "b_key"))],
        )
        .with_output_projection(left_projection(&[
            "a_key",
            "a_payload",
            "b_key",
            "b_payload",
            "c_key",
            "c_payload",
        ])),
        abc_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn root_semijoin_over_fact_date_dimension_plan() -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_fact_date_dimension_plan_with_week_keyset(week_dates_from_filtered_date_dimension)
}

fn root_semijoin_over_fact_date_dimension_plan_with_left_semi_week_keyset() -> nexus_query_engine::plan::QueryPlan {
    root_semijoin_over_fact_date_dimension_plan_with_week_keyset(week_dates_from_filtered_date_dimension_left_semi)
}

fn root_semijoin_over_fact_date_dimension_plan_with_week_keyset(
    week_keyset: fn(&mut QueryPlanBuilder, nexus_query_engine::plan::NodeId) -> nexus_query_engine::plan::NodeId,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["ss_sold_date_sk", "ss_ext_sales_price"],
        256 * 1024 * 1024,
        10_000_000,
    ));
    let date_dim = builder.add_source(estimated_local_source_with_fields(
        "date_dim",
        &["d_date_sk", "d_date", "d_week_seq"],
        1024 * 1024,
        73_049,
    ));

    let sales_with_date_schema = schema(&["ss_sold_date_sk", "ss_ext_sales_price", "d_date_sk", "d_date"]);
    let sales_with_date = builder.add_join(
        fact,
        date_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(
                ColumnRef::input(0, "ss_ext_sales_price", "ss_ext_sales_price"),
                "ss_ext_sales_price",
            ),
            JoinOutputColumn::right(ColumnRef::input(1, "d_date_sk", "d_date_sk"), "d_date_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_date", "d_date"), "d_date"),
        ]),
        Arc::clone(&sales_with_date_schema),
    );

    let week_dates = week_keyset(&mut builder, date_dim);
    let semi = builder.add_join(
        sales_with_date,
        week_dates,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "d_date", "d_date"), ColumnRef::input(1, "d_date", "d_date"))],
        )
        .with_output_projection(left_projection(&[
            "ss_sold_date_sk",
            "ss_ext_sales_price",
            "d_date_sk",
            "d_date",
        ])),
        sales_with_date_schema,
    );
    builder.finish_gpu_resident(semi)
}

fn week_dates_from_filtered_date_dimension(
    builder: &mut QueryPlanBuilder,
    date_dim: nexus_query_engine::plan::NodeId,
) -> nexus_query_engine::plan::NodeId {
    let all_dates = builder.add_projection(
        date_dim,
        vec![
            ProjectionOutput::new(Expr::column("d_date"), Field::new("d_date", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false)),
        ],
    );
    let filtered_date = builder.add_filter(
        date_dim,
        Expr::binary(Expr::column("d_date"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(10_959))),
        schema(&["d_date_sk", "d_date", "d_week_seq"]),
    );
    let filtered_week = builder.add_projection(
        filtered_date,
        vec![ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false))],
    );
    builder.add_join(
        all_dates,
        filtered_week,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_week_seq", "d_week_seq"),
                ColumnRef::input(1, "d_week_seq", "d_week_seq"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "d_date", "d_date"), "d_date")]),
        schema(&["d_date"]),
    )
}

fn week_dates_from_filtered_date_dimension_left_semi(
    builder: &mut QueryPlanBuilder,
    date_dim: nexus_query_engine::plan::NodeId,
) -> nexus_query_engine::plan::NodeId {
    let all_dates = builder.add_projection(
        date_dim,
        vec![
            ProjectionOutput::new(Expr::column("d_date"), Field::new("d_date", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false)),
        ],
    );
    let filtered_date = builder.add_filter(
        date_dim,
        Expr::binary(Expr::column("d_date"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(10_959))),
        schema(&["d_date_sk", "d_date", "d_week_seq"]),
    );
    let filtered_week = builder.add_projection(
        filtered_date,
        vec![ProjectionOutput::new(Expr::column("d_week_seq"), Field::new("d_week_seq", DataType::Int32, false))],
    );
    builder.add_join(
        all_dates,
        filtered_week,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_week_seq", "d_week_seq"),
                ColumnRef::input(1, "d_week_seq", "d_week_seq"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "d_date", "d_date"), "d_date")]),
        schema(&["d_date"]),
    )
}

fn assert_left_semi_input(
    plan: &nexus_query_engine::plan::QueryPlan,
    node_id: nexus_query_engine::plan::NodeId,
    key_name: &str,
) {
    let node = plan.node(node_id).expect("node should exist");
    let PlanNode::Join(join) = &node.kind else {
        panic!("expected pushed left-semi join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftSemi);
    assert_eq!(join.spec.keys.len(), 1);
    assert_eq!(join.spec.keys[0].left.name(), key_name);
    assert_eq!(join.spec.keys[0].right.name(), key_name);
    assert!(
        join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left),
        "pushed left-semi must expose only target-side columns"
    );
}

fn assert_semijoin_rejection(plan: nexus_query_engine::plan::QueryPlan, reason: SemijoinReductionRejectedReason) {
    let result = optimize_query_plan(plan, &semijoin_reduction_only_options(SemijoinReductionOptions::default()));
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].rejected_reason, Some(reason));
}

fn semijoin_reduction_only_options(semijoin_reduction: SemijoinReductionOptions) -> NativeOptimizerOptions {
    NativeOptimizerOptions { semijoin_reduction, ..NativeOptimizerOptions::disabled() }
}
