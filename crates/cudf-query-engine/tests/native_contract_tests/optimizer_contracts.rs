use cudf_query_engine::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinBuildSideSelectionOptions, NativeOptimizerChange, NativeOptimizerOptions,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionRejectedReason, SEMIJOIN_REDUCTION_RULE_NAME,
    SemijoinReductionAdmissionMode, SemijoinReductionCostSignal, SemijoinReductionOptions,
    SemijoinReductionRejectedReason, optimize_query_plan,
};

use super::*;

#[test]
fn native_optimizer_phase1_semijoin_entry_point_returns_unchanged_plan_and_empty_diagnostics() {
    let plan = single_source_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(!result.changed());
    assert!(result.diagnostics.is_empty());
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert!(result.diagnostics.semijoin_reduction.candidates.is_empty());
}

#[test]
fn native_optimizer_options_can_disable_semijoin_reduction_pass() {
    let plan = single_source_plan();
    let options = NativeOptimizerOptions::disabled();

    let result = optimize_query_plan(plan.clone(), &options);

    assert!(!options.semijoin_reduction.enabled);
    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn native_optimizer_flips_inner_equi_join_to_build_smaller_estimated_left_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.join_type, JoinType::Inner);
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_disabled_keeps_default_inner_equi_join_build_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::disabled());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn semijoin_reduction_phase1_options_and_rule_name_are_stable() {
    assert_eq!(SEMIJOIN_REDUCTION_RULE_NAME, "semijoin_reduction");
    assert!(SemijoinReductionOptions::default().enabled);
    assert_eq!(
        SemijoinReductionOptions::default().cost_threshold_multiplier,
        DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER
    );
    assert_eq!(SemijoinReductionOptions::default().admission_mode, SemijoinReductionAdmissionMode::NotEvaluated);
    assert!(!SemijoinReductionOptions::disabled().enabled);
}

#[test]
fn join_build_side_selection_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_BUILD_SIDE_SELECTION_RULE_NAME, "join_build_side_selection");
    assert!(JoinBuildSideSelectionOptions::default().enabled);
    assert!(!JoinBuildSideSelectionOptions::disabled().enabled);
    assert_eq!(JoinBuildSideSelectionOptions::default().small_build_side_bytes, 64 * 1024 * 1024);
    assert_eq!(JoinBuildSideSelectionOptions::default().small_build_side_rows, 1_000_000);
}

#[test]
fn outer_join_null_rejection_options_and_rule_name_are_stable() {
    assert_eq!(OUTER_JOIN_NULL_REJECTION_RULE_NAME, "outer_join_null_rejection");
    assert!(cudf_query_engine::OuterJoinNullRejectionOptions::default().enabled);
    assert!(!cudf_query_engine::OuterJoinNullRejectionOptions::disabled().enabled);
}

#[test]
fn join_filter_reassociation_options_and_rule_name_are_stable() {
    assert_eq!(JOIN_FILTER_REASSOCIATION_RULE_NAME, "join_filter_reassociation");
    assert!(cudf_query_engine::JoinFilterReassociationOptions::default().enabled);
    assert!(!cudf_query_engine::JoinFilterReassociationOptions::disabled().enabled);
}

#[test]
fn outer_join_null_rejection_converts_left_join_when_inner_join_key_rejects_right_nulls() {
    let plan = inner_join_over_projected_left_join_right_key_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: cudf_query_engine::JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    assert!(result.diagnostics.outer_join_null_rejection.candidates[0].applied);
    assert_eq!(result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason, None);

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("root should remain an inner join");
    };
    let PlanNode::Projection(projection) = &result.plan.node(root_join.inputs[0]).expect("projection").kind else {
        panic!("root left input should remain a projection");
    };
    let PlanNode::Join(rewritten_join) = &result.plan.node(projection.input).expect("left join").kind else {
        panic!("projection input should be rewritten join");
    };
    assert_eq!(rewritten_join.spec.join_type, JoinType::Inner);
}

#[test]
fn outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side() {
    let plan = inner_join_over_projected_left_join_left_key_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: cudf_query_engine::JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn outer_join_null_rejection_rejects_shared_outer_join_output() {
    let plan = shared_left_join_output_with_one_null_rejecting_consumer_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: cudf_query_engine::JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::SharedOuterJoinOutput)
    );
}

#[test]
fn join_filter_reassociation_pushes_dimension_filter_below_fact_join_without_losing_multiplicity() {
    let plan = inner_join_filter_over_projected_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_filter_reassociation.considered_count, 1);
    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 1);
    let candidate = &result.diagnostics.join_filter_reassociation.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let PlanNode::Projection(parent_projection) = &result.plan.root_node().kind else {
        panic!("rewritten parent join should become a projection preserving the original output schema");
    };
    let PlanNode::Projection(case_projection) =
        &result.plan.node(parent_projection.input).expect("case projection").kind
    else {
        panic!("parent projection should consume rebuilt case projection");
    };
    let PlanNode::Join(fact_join) = &result.plan.node(case_projection.input).expect("rebuilt fact join").kind else {
        panic!("case projection should consume rebuilt fact join");
    };
    assert_eq!(fact_join.spec.join_type, JoinType::Inner);
    let PlanNode::Join(filter_join) = &result.plan.node(fact_join.inputs[1]).expect("filter join").kind else {
        panic!("fact join right input should be prefiltered by dimension");
    };
    assert_eq!(filter_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, filter_join.inputs[0], "returns");
    assert_source_input(&result.plan, filter_join.inputs[1], "reason");
}

#[test]
fn join_build_side_selection_switches_inner_equi_join_to_small_left_input() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert!(result.diagnostics.semijoin_reduction.is_empty());
    assert_eq!(result.diagnostics.join_build_side_selection.considered_count, 1);
    assert_eq!(result.diagnostics.join_build_side_selection.applied_count, 1);
    let candidate = &result.diagnostics.join_build_side_selection.candidates[0];
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);
    assert_eq!(candidate.left_estimated_bytes, Some(1_024));
    assert_eq!(candidate.right_estimated_bytes, Some(9 * 1024 * 1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain an inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn disabled_native_optimizer_keeps_join_build_side_selection_off() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
}

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
fn semijoin_reduction_rejects_unestimated_non_keyset_shape() {
    let plan = root_semijoin_over_inner_join_plan_with_raw_keyset(SemijoinTarget::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(SemijoinReductionRejectedReason::CostGateRejected));
    assert_eq!(candidate.cost_gate.as_ref().map(|gate| &gate.signal), Some(&SemijoinReductionCostSignal::Unavailable));
}

#[test]
fn semijoin_reduction_strict_admission_gate_keeps_original_when_rewrite_is_not_admitted() {
    let plan = root_semijoin_over_inner_join_plan(SemijoinTarget::Right);
    let options = NativeOptimizerOptions {
        semijoin_reduction: SemijoinReductionOptions::default()
            .with_admission_mode(SemijoinReductionAdmissionMode::StrictBounded),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    let candidate = &result.diagnostics.semijoin_reduction.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(SemijoinReductionRejectedReason::AdmissionRejectedRewrite));
    assert_eq!(candidate.original_admitted, Some(false));
    assert_eq!(candidate.rewritten_admitted, Some(false));
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

fn single_source_plan() -> cudf_query_engine::QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source)
}

fn estimated_inner_equi_join_plan(left_bytes: u64, right_bytes: u64) -> cudf_query_engine::QueryPlan {
    estimated_inner_join_plan(JoinSide::Right, left_bytes, 1, right_bytes, 1)
}

fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", left_bytes, left_rows));
    let right = builder.add_source(estimated_local_source("right", "right_key", right_bytes, right_rows));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(build_side),
        output_schema,
    );
    builder.finish_gpu_resident(join)
}

fn estimated_local_source(name: &str, key_name: &str, bytes: u64, rows: u64) -> SourceNode {
    let source_schema = schema(&[key_name]);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], source_schema, vec![key_name.to_string()]))
}

#[derive(Debug, Clone, Copy)]
enum SemijoinTarget {
    Left,
    Right,
}

fn root_semijoin_over_inner_join_plan(target: SemijoinTarget) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, true)
}

fn root_semijoin_over_inner_join_plan_with_raw_keyset(target: SemijoinTarget) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_impl(target, false)
}

fn root_semijoin_over_inner_join_plan_impl(
    target: SemijoinTarget,
    aggregate_keyset: bool,
) -> cudf_query_engine::QueryPlan {
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
) -> cudf_query_engine::QueryPlan {
    root_semijoin_over_inner_join_plan_with_options(
        SemijoinTarget::Right,
        true,
        JoinType::Inner,
        semijoin_null_equality,
        crossed_join_null_equality,
    )
}

fn root_semijoin_over_crossed_join_type_plan(join_type: JoinType) -> cudf_query_engine::QueryPlan {
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
) -> cudf_query_engine::QueryPlan {
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

fn root_semijoin_over_duplicate_output_name_plan() -> cudf_query_engine::QueryPlan {
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

fn root_semijoin_over_expression_key_projection_plan() -> cudf_query_engine::QueryPlan {
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

fn root_semijoin_over_dropped_key_projection_plan() -> cudf_query_engine::QueryPlan {
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

fn root_semijoin_over_nested_inner_join_plan() -> cudf_query_engine::QueryPlan {
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

fn inner_join_over_projected_left_join_right_key_plan() -> cudf_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("r_key")
}

fn inner_join_over_projected_left_join_left_key_plan() -> cudf_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("l_key")
}

fn inner_join_over_projected_left_join_key_plan(rejecting_key: &str) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key", "payload"]));
    let right = builder.add_source(gpu_source("right", &["r_key", "r_reason"]));
    let dim = builder.add_source(gpu_source("dim", &[rejecting_key]));
    let left_join_schema = schema(&["l_key", "payload", "r_key", "r_reason"]);
    let left_join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_reason", "r_reason"), "r_reason"),
        ]),
        Arc::clone(&left_join_schema),
    );
    let projected = builder.add_projection(
        left_join,
        vec![
            ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
            ProjectionOutput::new(Expr::column("r_reason"), Field::new("r_reason", DataType::Int32, true)),
        ],
    );
    let root_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
        Field::new("r_reason", DataType::Int32, true),
    ]));
    let root = builder.add_join(
        projected,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, rejecting_key, rejecting_key),
                ColumnRef::input(1, rejecting_key, rejecting_key),
            )],
        )
        .with_output_projection(left_projection(&["l_key", "payload", "r_key", "r_reason"])),
        root_schema,
    );
    builder.finish_gpu_resident(root)
}

fn shared_left_join_output_with_one_null_rejecting_consumer_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key"]));
    let right = builder.add_source(gpu_source("right", &["r_key"]));
    let dim = builder.add_source(gpu_source("dim", &["r_key"]));
    let left_join_schema = schema(&["l_key", "r_key"]);
    let left_join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
        ]),
        Arc::clone(&left_join_schema),
    );
    let rejecting = builder.add_join(
        left_join,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "r_key", "r_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        Arc::clone(&left_join_schema),
    );
    let root = builder.add_join(
        left_join,
        rejecting,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "l_key", "l_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        left_join_schema,
    );
    builder.finish_gpu_resident(root)
}

fn inner_join_filter_over_projected_inner_join_plan() -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["s_item", "s_ticket", "customer", "amount"]));
    let returns = builder.add_source(gpu_source("returns", &["r_item", "r_ticket", "reason_key", "returned"]));
    let reason = builder.add_source(gpu_source("reason", &["reason_key"]));
    let sales_returns_schema =
        schema(&["s_item", "s_ticket", "customer", "amount", "r_item", "r_ticket", "reason_key", "returned"]);
    let sales_returns = builder.add_join(
        sales,
        returns,
        JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(ColumnRef::input(0, "s_item", "s_item"), ColumnRef::input(1, "r_item", "r_item")),
                JoinKey::new(ColumnRef::input(0, "s_ticket", "s_ticket"), ColumnRef::input(1, "r_ticket", "r_ticket")),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_item", "s_item"), "s_item"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_ticket", "s_ticket"), "s_ticket"),
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_item", "r_item"), "r_item"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_ticket", "r_ticket"), "r_ticket"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_key", "reason_key"), "reason_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "returned", "returned"), "returned"),
        ]),
        sales_returns_schema,
    );
    let projected_schema = schema(&["customer", "net_amount", "reason_key"]);
    let projected = builder.add_projection(
        sales_returns,
        vec![
            ProjectionOutput::new(Expr::column("customer"), Field::new("customer", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::binary(Expr::column("amount"), BinaryOp::Subtract, Expr::column("returned")),
                Field::new("net_amount", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("reason_key"), Field::new("reason_key", DataType::Int32, false)),
        ],
    );
    let filtered = builder.add_join(
        projected,
        reason,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_key", "reason_key"),
                ColumnRef::input(1, "reason_key", "reason_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "net_amount", "net_amount"), "net_amount"),
        ]),
        schema(&["customer", "net_amount"]),
    );
    assert_eq!(projected_schema.fields().len(), 3);
    builder.finish_gpu_resident(filtered)
}

fn gpu_source(name: &str, fields: &[&str]) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> cudf_query_engine::NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name]))
}

fn schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn left_projection(fields: &[&str]) -> Vec<JoinOutputColumn> {
    fields.iter().map(|name| JoinOutputColumn::left(ColumnRef::input(0, *name, *name), *name)).collect()
}

fn assert_source_input(plan: &cudf_query_engine::QueryPlan, node_id: cudf_query_engine::NodeId, name: &str) {
    let node = plan.node(node_id).expect("node should exist");
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::GpuResident(source) => assert_eq!(source.input_name, name),
            other => panic!("expected GPU-resident source {name}, got {other:?}"),
        },
        other => panic!("expected source {name}, got {other:?}"),
    }
}

fn assert_left_semi_input(plan: &cudf_query_engine::QueryPlan, node_id: cudf_query_engine::NodeId, key_name: &str) {
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

fn assert_semijoin_rejection(plan: cudf_query_engine::QueryPlan, reason: SemijoinReductionRejectedReason) {
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.candidates[0].rejected_reason, Some(reason));
}
