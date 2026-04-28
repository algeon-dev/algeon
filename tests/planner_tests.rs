use std::sync::Arc;

use arrow::{
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use datafusion::{
    common::{DataFusionError, ScalarValue, config::ConfigOptions},
    execution::SessionStateBuilder,
    functions_aggregate::min_max::max_udaf,
    logical_expr::Operator,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{CastExpr, binary, col, lit},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        displayable,
        empty::EmptyExec,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
};
use datafusion_cudf::{
    CandidateOutcome, CudfOptimizerConfig, CudfPhysicalOptimizerRule, CudfSessionStateBuilderExt, Error, ErrorCode,
    FallbackReason, GpuFallbackPolicy, cudf_physical_optimizer_rules, optimize_plan,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("value", DataType::Int32, false)]))
}

fn input() -> Arc<dyn ExecutionPlan> {
    Arc::new(EmptyExec::new(schema()))
}

#[derive(Debug)]
struct MarkerPhysicalOptimizerRule;

impl PhysicalOptimizerRule for MarkerPhysicalOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(plan)
    }

    fn name(&self) -> &str {
        "marker_rule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn filter_plan() -> Arc<dyn ExecutionPlan> {
    let schema = schema();
    let predicate =
        binary(col("id", &schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(2))), &schema).unwrap();
    Arc::new(FilterExec::try_new(predicate, input()).unwrap())
}

fn projection_plan(child: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let input_schema = child.schema();
    let total =
        binary(col("id", &input_schema).unwrap(), Operator::Plus, col("value", &input_schema).unwrap(), &input_schema)
            .unwrap();
    Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("id", &input_schema).unwrap(), alias: "id".to_string() },
                ProjectionExpr { expr: total, alias: "total".to_string() },
            ],
            child,
        )
        .unwrap(),
    )
}

fn sort_plan(child: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let input_schema = child.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("total", &input_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    Arc::new(SortExec::new(sort_exprs, child))
}

fn aggregate_plan(child: Arc<dyn ExecutionPlan>, mode: AggregateMode) -> Arc<dyn ExecutionPlan> {
    let input_schema = child.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("id", &input_schema).unwrap(), "id".to_string())]);
    let total = AggregateExprBuilder::new(max_udaf(), vec![col("total", &input_schema).unwrap()])
        .schema(Arc::clone(&input_schema))
        .alias("total_max")
        .build()
        .unwrap();
    Arc::new(AggregateExec::try_new(mode, group_by, vec![Arc::new(total)], vec![None], child, input_schema).unwrap())
}

#[test]
fn supported_projection_filter_is_rewritten_to_gpu_island() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);
    let result = optimize_plan(projection_plan(filter_plan()), &config).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(result.report.outcomes().len(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count("ProjectionExec(FilterExec(child))", 2, 2,),
        },
    );
}

#[test]
fn supported_aggregate_projection_filter_is_rewritten_to_larger_gpu_island() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let plan = aggregate_plan(projection_plan(filter_plan()), AggregateMode::Single);

    let result = optimize_plan(plan, &config).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(result.report.outcomes().len(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "AggregateExec(ProjectionExec(FilterExec(child)))",
                2,
                3,
            ),
        },
    );
}

#[test]
fn unsupported_partial_aggregate_is_reported_before_rewrite() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);
    let plan = aggregate_plan(projection_plan(filter_plan()), AggregateMode::Partial);

    let result = optimize_plan(plan, &config).unwrap();

    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "AggregateExec(ProjectionExec(FilterExec(child)))",
                2,
                3,
            ),
            reason: FallbackReason::UnsupportedOperator,
        },
    );
}

#[test]
fn supported_local_limit_projection_filter_is_rewritten_to_larger_gpu_island() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(projection_plan(filter_plan()), 2));

    let result = optimize_plan(plan, &config).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(result.report.outcomes().len(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "LocalLimitExec(ProjectionExec(FilterExec(child)))",
                2,
                3,
            ),
        },
    );
}

#[test]
fn supported_sort_projection_filter_is_rewritten_to_larger_gpu_island() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let plan = sort_plan(projection_plan(filter_plan()));

    let result = optimize_plan(plan, &config).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(result.report.outcomes().len(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "SortExec(ProjectionExec(FilterExec(child)))",
                2,
                3,
            ),
        },
    );
}

#[test]
fn supported_local_limit_sort_projection_filter_is_rewritten_to_topk_gpu_island() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(sort_plan(projection_plan(filter_plan())), 2));

    let result = optimize_plan(plan, &config).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(result.report.outcomes().len(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "LocalLimitExec(SortExec(ProjectionExec(FilterExec(child))))",
                2,
                4,
            ),
        },
    );
}

#[test]
fn unsupported_sort_expression_is_reported_before_rewrite() {
    let projection = projection_plan(filter_plan());
    let input_schema = projection.schema();
    let sort_key =
        binary(col("id", &input_schema).unwrap(), Operator::Plus, col("total", &input_schema).unwrap(), &input_schema)
            .unwrap();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(sort_key, ArrowSortOptions::new(false, false))]).unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, projection));
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);

    let result = optimize_plan(sort, &config).unwrap();

    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "SortExec(ProjectionExec(FilterExec(child)))",
                2,
                3,
            ),
            reason: FallbackReason::UnsupportedExpression,
        },
    );
}

#[test]
fn gpu_island_explain_names_fused_operators_and_boundaries() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(LocalLimitExec::new(sort_plan(projection_plan(filter_plan())), 2));

    let result = optimize_plan(plan, &config).unwrap();

    let one_line = displayable(result.plan.as_ref()).one_line().to_string();
    assert!(one_line.contains("operators=[filter,projection,sort,local_limit]"));
    assert!(one_line.contains("internal_batch=CudfBatch"));
    assert!(one_line.contains("internal_execution=operator_chain"));
    assert!(one_line.contains("projection_exprs=2"));
    assert!(one_line.contains("sort_exprs=1"));
    assert!(one_line.contains("local_limit=2"));
    assert!(one_line.contains("input_boundary=host_to_device"));
    assert!(one_line.contains("output_boundary=device_to_host"));

    let tree = displayable(result.plan.as_ref()).tree_render().to_string();
    assert!(tree.contains("operators"));
    assert!(tree.contains("filter"));
    assert!(tree.contains("projection"));
    assert!(tree.contains("sort"));
    assert!(tree.contains("local_limit"));
    assert!(tree.contains("internal_batch"));
    assert!(tree.contains("CudfBatch"));
    assert!(tree.contains("internal_execution"));
    assert!(tree.contains("operator_chain"));
    assert!(tree.contains("input_boundary"));
    assert!(tree.contains("host_to_device"));
    assert!(tree.contains("output_boundary"));
    assert!(tree.contains("device_to_host"));
}

#[test]
fn default_policy_rejects_single_operator_island_as_too_small() {
    let result = optimize_plan(filter_plan(), &CudfOptimizerConfig::default()).unwrap();

    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count("FilterExec(child)", 2, 1),
            reason: FallbackReason::InsufficientIslandSize,
        },
    );
}

#[test]
fn cpu_only_policy_reports_explicit_cpu_decision() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::CpuOnly).with_min_operators_per_island(1);
    let result = optimize_plan(projection_plan(filter_plan()), &config).unwrap();

    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count("ProjectionExec(FilterExec(child))", 2, 2,),
            reason: FallbackReason::ExplicitCpuPolicy,
        },
    );
}

#[test]
fn unsupported_expression_is_reported_before_rewrite() {
    let schema = schema();
    let cast = Arc::new(CastExpr::new(col("id", &schema).unwrap(), DataType::Int64, None));
    let predicate = binary(cast, Operator::GtEq, lit(ScalarValue::Int64(Some(2))), &schema).unwrap();
    let filter = Arc::new(FilterExec::try_new(predicate, input()).unwrap());
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(1);

    let result = optimize_plan(filter, &config).unwrap();

    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count("FilterExec(child)", 2, 1),
            reason: FallbackReason::UnsupportedExpression,
        },
    );
}

#[test]
fn gpu_only_turns_candidate_rejection_into_planning_error() {
    let result = optimize_plan(filter_plan(), &CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly));

    let error = result.expect_err("GpuOnly must fail when a candidate cannot be rewritten");
    let DataFusionError::External(source) = &error else {
        panic!("GpuOnly candidate rejection should preserve the typed adapter error");
    };
    let adapter_error = source.downcast_ref::<Error>().expect("expected datafusion-cudf Error");

    assert_eq!(adapter_error.code(), ErrorCode::GpuCandidateRejected);
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("insufficient_island_size"));
    assert_eq!(adapter_error.fact_value("candidate_shape"), Some("FilterExec(child)"));
    assert!(error.to_string().contains("insufficient_island_size"));
}

#[test]
fn physical_optimizer_rule_stores_last_planning_report() {
    let rule = CudfPhysicalOptimizerRule::new(CudfOptimizerConfig::new(GpuFallbackPolicy::CpuOnly));
    let plan = rule.optimize(filter_plan(), &datafusion::common::config::ConfigOptions::new()).unwrap();

    assert_eq!(plan.name(), "FilterExec");
    assert_eq!(rule.name(), "datafusion_cudf_gpu_island");
    assert!(rule.schema_check());
    assert_eq!(
        rule.last_report().unwrap().outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count("FilterExec(child)", 2, 1),
            reason: FallbackReason::ExplicitCpuPolicy,
        },
    );
}

#[test]
fn cudf_default_physical_optimizer_rules_insert_gpu_rewrite_before_final_sanity_check() {
    let rules = cudf_physical_optimizer_rules(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred));
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let cudf_position =
        names.iter().position(|name| *name == "datafusion_cudf_gpu_island").expect("cudf rule should be registered");
    let sanity_position = names
        .iter()
        .position(|name| *name == "SanityCheckPlan")
        .expect("DataFusion final sanity check should be preserved");

    assert!(
        cudf_position < sanity_position,
        "cudf GPU rewrite must run before DataFusion's final sanity check: {names:?}",
    );
    assert_eq!(
        names.last().copied(),
        Some("SanityCheckPlan"),
        "the final DataFusion plan validation rule must remain last",
    );
}

#[test]
fn session_state_builder_extension_registers_cudf_before_final_validation() {
    let state = SessionStateBuilder::new()
        .with_cudf_gpu_islands(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .build();
    let names = state.physical_optimizers().iter().map(|rule| rule.name()).collect::<Vec<_>>();

    assert_eq!(names.last().copied(), Some("SanityCheckPlan"));
    let cudf_position =
        names.iter().position(|name| *name == "datafusion_cudf_gpu_island").expect("cudf rule should be registered");
    let sanity_position = names
        .iter()
        .position(|name| *name == "SanityCheckPlan")
        .expect("DataFusion final sanity check should be preserved");

    assert!(cudf_position < sanity_position, "cudf rule must run before final validation: {names:?}");
}

#[test]
fn session_state_builder_extension_keeps_final_validation_after_pending_rules() {
    let state = SessionStateBuilder::new()
        .with_physical_optimizer_rule(Arc::new(MarkerPhysicalOptimizerRule))
        .with_cudf_gpu_islands(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred))
        .build();
    let names = state.physical_optimizers().iter().map(|rule| rule.name()).collect::<Vec<_>>();

    assert_eq!(names[names.len() - 3], "marker_rule");
    assert_eq!(names[names.len() - 2], "datafusion_cudf_gpu_island");
    assert_eq!(names[names.len() - 1], "SanityCheckPlan");
}
