use cudf_query_engine::{
    NativeOptimizerChange, NativeOptimizerOptions, SEMIJOIN_REDUCTION_RULE_NAME, SemijoinReductionOptions,
    optimize_query_plan,
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
    assert!(!SemijoinReductionOptions::disabled().enabled);
}

fn single_source_plan() -> cudf_query_engine::QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source)
}
