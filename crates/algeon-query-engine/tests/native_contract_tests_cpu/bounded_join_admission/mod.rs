use std::collections::BTreeMap;

use query_engine::{
    dimension_pruning::local_dimension_pruning_resolver, exec::admit_query_plan,
    rapids_backend::rapids_analysis_context,
};
use query_planning::{
    admission::{
        AdmittedPlan, CompiledBoundedPipelineSupport, admit, bounded_join_modeled_facts,
        bounded_join_unsupported_shape_facts, compiled_bounded_pipeline_support, compiled_bounded_query_plan_support,
    },
    backend::AnalysisContext,
    capability::{CapabilityDecision, EdgeConsumer, JoinForm, JoinStrategy, Reason},
    error::{QueryError, QueryErrorCode, Result},
    expr::Expr,
    mode::StatisticalAggregateExecutionMode,
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    pipeline::{
        AcceptedInputForm, CompiledPipeline, JoinPlan, OrderingScope, PipelineTask, PipelineTaskKind, ResidentDataForm,
        ResourceSharing, SortStrategy, SourceOutput, SourceReadPlan, TaskPhysical, compile_pipeline,
    },
    plan::{EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner, GraphEdgeColumnBindings, QueryPlan},
    policy::PlanningPolicy,
};
use root_topk::{compositional_inner_join_base, compositional_join_row_number_plan};
use support::{assert_rejection_facts_are_family_free, assert_rejection_reason_is_family_free};

use super::*;

mod conditional_subplan;
mod resident_subplan;
mod root_topk;
mod strategy;
mod support;

fn bounded_pipeline_admission(plan: &QueryPlan) -> Result<Option<Arc<AdmittedPlan>>> {
    Ok(admit(
        plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )?
    .admission()
    .cloned())
}

/// The error `admit` returns for a plan the validator finds malformed.
fn malformed_plan_error(plan: &QueryPlan) -> QueryError {
    admit(
        plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect_err("a malformed plan has no admission decision")
}

fn modeled_facts(
    plan: &QueryPlan,
    mode: StatisticalAggregateExecutionMode,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    let decision = admit(plan, mode, ctx.policy(), ctx).unwrap();
    bounded_join_modeled_facts(plan, &decision, ctx)
}

fn unsupported_facts(
    plan: &QueryPlan,
    mode: StatisticalAggregateExecutionMode,
    ctx: AnalysisContext<'_>,
) -> Option<BTreeMap<&'static str, String>> {
    let decision = admit(plan, mode, ctx.policy(), ctx).unwrap();
    bounded_join_unsupported_shape_facts(plan, &decision, ctx)
}

mod aggregate_over_join_subplans;
mod mixed_residual;
#[expect(dead_code, reason = "fixtures shared with the GPU pipeline target are not all used by this target")]
pub(crate) mod plans;
mod residual_resident_grouped_aggregate;
use plans::*;
