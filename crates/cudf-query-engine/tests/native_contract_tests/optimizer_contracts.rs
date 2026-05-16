use cudf_query_engine::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, NativeOptimizerChange, NativeOptimizerOptions,
    SEMIJOIN_REDUCTION_RULE_NAME, SemijoinReductionAdmissionMode, SemijoinReductionCostSignal,
    SemijoinReductionOptions, SemijoinReductionRejectedReason, optimize_query_plan,
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

fn single_source_plan() -> cudf_query_engine::QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source)
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
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "x_key", "x_key"), ColumnRef::input(1, "y_key", "y_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "x_key", "x_key"), "x_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "x_payload", "x_payload"), "x_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_key", "y_key"), "y_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "y_payload", "y_payload"), "y_payload"),
        ]),
        Arc::clone(&xy_schema),
    );
    let semi = builder.add_join(
        xy,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, key_name, key_name), ColumnRef::input(1, key_name, key_name))],
        )
        .with_output_projection(left_projection(&["x_key", "x_payload", "y_key", "y_payload"])),
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
