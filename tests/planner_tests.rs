use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    common::{DataFusionError, ScalarValue},
    logical_expr::Operator,
    physical_expr::expressions::{CastExpr, binary, col, lit},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        empty::EmptyExec,
        filter::FilterExec,
        projection::{ProjectionExec, ProjectionExpr},
    },
};
use datafusion_cudf::{
    CandidateOutcome, CudfOptimizerConfig, CudfPhysicalOptimizerRule, Error, ErrorCode, FallbackReason,
    GpuFallbackPolicy, optimize_plan,
};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("value", DataType::Int32, false)]))
}

fn input() -> Arc<dyn ExecutionPlan> {
    Arc::new(EmptyExec::new(schema()))
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
