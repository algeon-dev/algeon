use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::Schema;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    datasource::source::DataSourceExec,
    logical_expr::JoinType,
    physical_expr::{PhysicalExpr, ScalarFunctionExpr, expressions::BinaryExpr},
    physical_plan::{
        ExecutionPlan,
        aggregates::AggregateExec,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::sort::SortExec,
        union::UnionExec,
        windows::{BoundedWindowAggExec, WindowAggExec},
    },
};
use nexus_query_engine::capability::BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS;

use super::{
    super::{CudfOptimizerConfig, candidate_rejected_error},
    NativeAdaptivePipelineDiagnostics, NativeStrictBoundedContractDiagnostics,
    adaptive::{adaptive_pipeline_was_rejected, plan_adaptive_pipeline_diagnostics},
    strict_bounded::{
        accepted_plan_contract_diagnostics, rejected_plan_contract_diagnostics, strict_bounded_admission_is_rejected,
    },
};
use crate::{
    error::{Error, ErrorCode},
    native_exec::{
        NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
        gpu_resident_producer_input_name,
    },
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{CandidateOutcome, GpuCandidate, PlanningReport},
    table_format::CudfTableFormatScanExec,
};

/// Optimizer attempt output, including the replacement plan and all report evidence.
#[derive(Debug, Clone)]
pub(super) struct NativeOptimizationAttempt {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) report: PlanningReport,
    pub(super) native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics,
    pub(super) source_diagnostics: Vec<nexus_query_engine::SourceDiagnostics>,
    pub(super) strict_bounded_contract: NativeStrictBoundedContractDiagnostics,
    pub(super) adaptive_pipeline: NativeAdaptivePipelineDiagnostics,
    pub(super) rewritten: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostArrowChildReplacementMode {
    Normal,
    OuterJoinBoundary,
}

pub(super) fn apply_native_fallback_policy(
    original_plan: Arc<dyn ExecutionPlan>,
    attempt: NativeOptimizationAttempt,
    config: &CudfOptimizerConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    match config.fallback_policy {
        GpuFallbackPolicy::GpuPreferred => Ok(attempt),
        GpuFallbackPolicy::CpuOnly => Ok(NativeOptimizationAttempt {
            plan: original_plan,
            report: native_cpu_only_report(&attempt.report),
            native_optimizer: attempt.native_optimizer,
            source_diagnostics: attempt.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        }),
        GpuFallbackPolicy::GpuOnly => {
            if attempt.plan.downcast_ref::<NativeCudfExec>().is_some()
                || attempt.plan.downcast_ref::<NativeCudfComposedExec>().is_some()
            {
                return Ok(attempt);
            }
            if let Some((candidate, reason)) = first_native_rejection(&attempt.report) {
                return Err(candidate_rejected_error(candidate, reason, config).into());
            }
            Ok(attempt)
        },
    }
}

fn native_cpu_only_report(report: &PlanningReport) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } | CandidateOutcome::Rejected { candidate, .. } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: FallbackReason::ExplicitCpuPolicy }
        },
    }))
}

fn first_native_rejection(report: &PlanningReport) -> Option<(&GpuCandidate, FallbackReason)> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { candidate, reason } => Some((candidate, *reason)),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

fn native_preference_rejected(report: &PlanningReport) -> bool {
    first_native_rejection(report).is_some_and(|(_, reason)| {
        matches!(
            reason,
            FallbackReason::NativeSmallSelectiveTopkCpuPreferred
                | FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred
                | FallbackReason::NativeSmallSelectiveFullSortCpuPreferred
        )
    })
}

fn native_preference_fallback_reason(reason: nexus_query_engine::NativePlanCpuPreferenceReason) -> FallbackReason {
    match reason {
        nexus_query_engine::NativePlanCpuPreferenceReason::SmallSelectiveTopK => {
            FallbackReason::NativeSmallSelectiveTopkCpuPreferred
        },
        nexus_query_engine::NativePlanCpuPreferenceReason::SmallSelectiveResidualJoin => {
            FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred
        },
        nexus_query_engine::NativePlanCpuPreferenceReason::SmallSelectiveFullSort => {
            FallbackReason::NativeSmallSelectiveFullSortCpuPreferred
        },
    }
}

pub(super) fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering = crate::native::lower_count_only_physical_plan_to_native(lowering_input);
    replace_lowered_root_or_keep_plan(
        plan,
        lowering,
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
    )
}

pub(super) fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    optimize_final_native_plan_with_composition(
        plan,
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
        true,
    )
}

fn optimize_final_native_plan_with_composition(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
    allow_fragment_composition: bool,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    if lowering.plans.len() == 1 && lowering.rewritten_count() == 1 {
        let attempt = replace_lowered_root_or_keep_plan(
            Arc::clone(&plan),
            lowering,
            native_execution_config,
            native_optimizer_options,
            native_plan_preference_options,
        )?;
        if attempt.rewritten {
            return Ok(attempt);
        }
        if native_preference_rejected(&attempt.report) {
            return Ok(attempt);
        }
        if strict_bounded_rejection_blocks_child_rewrites(&attempt.strict_bounded_contract) {
            return Ok(attempt);
        }

        let mut report = attempt.report.clone();
        let mut native_optimizer = attempt.native_optimizer.clone();
        let mut source_diagnostics = attempt.source_diagnostics.clone();
        let mut strict_bounded_contract = attempt.strict_bounded_contract.clone();
        let mut adaptive_pipeline = attempt.adaptive_pipeline.clone();
        let mode = if subtree_contains_outer_join(plan.as_ref()) {
            HostArrowChildReplacementMode::OuterJoinBoundary
        } else {
            HostArrowChildReplacementMode::Normal
        };
        let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
            Arc::clone(&plan),
            native_execution_config,
            native_optimizer_options,
            native_plan_preference_options,
            mode,
            &mut report,
            &mut native_optimizer,
            &mut source_diagnostics,
            &mut strict_bounded_contract,
            &mut adaptive_pipeline,
        )?
        else {
            return Ok(attempt);
        };

        let rewritten_plan = if allow_fragment_composition {
            compose_host_remainder_over_native_fragments_if_possible(
                rewritten_plan,
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
                &mut report,
                &mut native_optimizer,
                &mut source_diagnostics,
                &mut strict_bounded_contract,
                &mut adaptive_pipeline,
            )?
        } else {
            rewritten_plan
        };

        return Ok(NativeOptimizationAttempt {
            plan: rewritten_plan,
            report,
            native_optimizer,
            source_diagnostics,
            strict_bounded_contract,
            adaptive_pipeline,
            rewritten: true,
        });
    }

    let mut report = lowering.report.clone();
    let mut native_optimizer = nexus_query_engine::NativeOptimizerDiagnostics::default();
    let mut source_diagnostics = lowering.source_diagnostics.clone();
    let mut strict_bounded_contract = NativeStrictBoundedContractDiagnostics::default();
    let mut adaptive_pipeline = NativeAdaptivePipelineDiagnostics::default();
    if first_native_rejection(&report).is_some_and(|(_, reason)| reason == FallbackReason::UnsupportedExpression) {
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics,
            strict_bounded_contract,
            adaptive_pipeline,
            rewritten: false,
        });
    }
    let Some(rewritten_plan) = optimize_native_children_for_host_arrow_boundary(
        Arc::clone(&plan),
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
        HostArrowChildReplacementMode::Normal,
        &mut report,
        &mut native_optimizer,
        &mut source_diagnostics,
        &mut strict_bounded_contract,
        &mut adaptive_pipeline,
    )?
    else {
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics,
            strict_bounded_contract,
            adaptive_pipeline,
            rewritten: false,
        });
    };

    let rewritten_plan = if allow_fragment_composition {
        compose_host_remainder_over_native_fragments_if_possible(
            rewritten_plan,
            native_execution_config,
            native_optimizer_options,
            native_plan_preference_options,
            &mut report,
            &mut native_optimizer,
            &mut source_diagnostics,
            &mut strict_bounded_contract,
            &mut adaptive_pipeline,
        )?
    } else {
        rewritten_plan
    };

    Ok(NativeOptimizationAttempt {
        plan: rewritten_plan,
        report,
        native_optimizer,
        source_diagnostics,
        strict_bounded_contract,
        adaptive_pipeline,
        rewritten: true,
    })
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    lowering: crate::native::NativeLoweringResult,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    if lowering.plans.len() != 1 || lowering.rewritten_count() != 1 {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics::default(),
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        });
    }
    let candidate = single_rewritten_candidate(&lowering.report)
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;
    let optimizer_options = native_optimizer_options_for_execution(
        native_optimizer_options,
        native_execution_config.native_pipeline_execution_mode(),
    );
    let optimizer_result = nexus_query_engine::optimize_query_plan(lowering.plans[0].clone(), &optimizer_options);
    let native_optimizer = optimizer_result.diagnostics;
    let native_plan = optimizer_result.plan;
    if let nexus_query_engine::NativePlanPreferenceDecision::CpuPreferred { reason, .. } =
        nexus_query_engine::native_plan_preference_decision(&native_plan, native_plan_preference_options)
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_rejection_report(&lowering.report, native_preference_fallback_reason(reason)),
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        });
    }

    if native_execution_config.native_pipeline_execution_mode()
        == nexus_query_engine::NativePipelineExecutionMode::Adaptive
    {
        let adaptive_pipeline =
            plan_adaptive_pipeline_diagnostics(&native_plan, candidate.clone(), native_execution_config)?
                .unwrap_or_default();
        if adaptive_pipeline_was_rejected(&adaptive_pipeline) {
            return Ok(NativeOptimizationAttempt {
                plan,
                report: native_execution_rejection_report_for_candidate(
                    candidate.clone(),
                    FallbackReason::NativeAdaptivePipelineUnsupported,
                ),
                native_optimizer,
                source_diagnostics: lowering.source_diagnostics,
                strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
                adaptive_pipeline,
                rewritten: false,
            });
        }

        let plan = native_datafusion_exec_for_plan(Arc::clone(&plan), native_plan, native_execution_config)?;
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline,
            rewritten: true,
        });
    }

    if native_execution_boundary_is_rejected(&native_plan, native_execution_config)? {
        let report = native_execution_rejection_report_for_candidate(
            candidate.clone(),
            FallbackReason::NativeBoundedJoinUnsupported,
        );
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            strict_bounded_contract: rejected_plan_contract_diagnostics(
                &native_plan,
                candidate,
                native_execution_config,
            )?,
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
            rewritten: false,
        });
    }

    let strict_bounded_contract = accepted_plan_contract_diagnostics(&native_plan, candidate, native_execution_config)?;
    let plan = native_datafusion_exec_for_plan(Arc::clone(&plan), native_plan, native_execution_config)?;
    Ok(NativeOptimizationAttempt {
        plan,
        report: lowering.report,
        native_optimizer,
        source_diagnostics: lowering.source_diagnostics,
        strict_bounded_contract,
        adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
        rewritten: true,
    })
}

fn native_optimizer_options_for_execution(
    options: &nexus_query_engine::NativeOptimizerOptions,
    execution_mode: nexus_query_engine::NativePipelineExecutionMode,
) -> nexus_query_engine::NativeOptimizerOptions {
    let mut options = options.clone();
    options.semijoin_reduction.admission_mode = match execution_mode {
        nexus_query_engine::NativePipelineExecutionMode::StrictBounded => {
            nexus_query_engine::SemijoinReductionAdmissionMode::StrictBounded
        },
        nexus_query_engine::NativePipelineExecutionMode::Adaptive => {
            nexus_query_engine::SemijoinReductionAdmissionMode::Adaptive
        },
    };
    options
}

fn native_datafusion_exec_for_plan(
    original_plan: Arc<dyn ExecutionPlan>,
    native_plan: nexus_query_engine::QueryPlan,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let gpu_input_names = native_plan_gpu_resident_source_names(&native_plan);
    if gpu_input_names.is_empty() {
        return NativeCudfExec::try_new(native_plan, native_execution_config.clone())
            .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
            .map_err(DataFusionError::from);
    }

    let gpu_inputs = gpu_resident_producer_children(&original_plan, &gpu_input_names)?;
    NativeCudfComposedExec::try_new(native_plan, gpu_inputs, native_execution_config.clone())
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from)
}

fn native_plan_gpu_resident_source_names(plan: &nexus_query_engine::QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            nexus_query_engine::PlanNode::Source(source) => match source.as_ref() {
                nexus_query_engine::SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                nexus_query_engine::SourceNode::LocalParquet(_) => None,
                nexus_query_engine::SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

fn gpu_resident_producer_children(
    plan: &Arc<dyn ExecutionPlan>,
    required_inputs: &BTreeSet<String>,
) -> DataFusionResult<Vec<Arc<dyn ExecutionPlan>>> {
    let mut producers = BTreeMap::<String, Arc<dyn ExecutionPlan>>::new();
    collect_gpu_resident_producer_children(plan, required_inputs, &mut producers);
    let missing =
        required_inputs.iter().filter(|name| !producers.contains_key(*name)).map(String::as_str).collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native.compose_gpu_resident",
                "native plan requires GPU-resident inputs that were not produced by child execution plans",
            )
            .with_fact("missing_inputs", missing.join(",")),
        ));
    }
    Ok(required_inputs.iter().filter_map(|name| producers.remove(name)).collect())
}

fn collect_gpu_resident_producer_children(
    plan: &Arc<dyn ExecutionPlan>,
    required_inputs: &BTreeSet<String>,
    producers: &mut BTreeMap<String, Arc<dyn ExecutionPlan>>,
) {
    if let Some(input_name) = gpu_resident_producer_input_name(plan.as_ref()) {
        if required_inputs.contains(&input_name) {
            producers.entry(input_name).or_insert_with(|| Arc::clone(plan));
        }
        return;
    }
    for child in plan.children() {
        collect_gpu_resident_producer_children(child, required_inputs, producers);
    }
}

fn compose_host_remainder_over_native_fragments_if_possible(
    rewritten_plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
    report: &mut PlanningReport,
    native_optimizer: &mut nexus_query_engine::NativeOptimizerDiagnostics,
    source_diagnostics: &mut Vec<nexus_query_engine::SourceDiagnostics>,
    strict_bounded_contract: &mut NativeStrictBoundedContractDiagnostics,
    adaptive_pipeline: &mut NativeAdaptivePipelineDiagnostics,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let Some(composed_attempt) = try_compose_host_remainder_over_native_fragments(
        Arc::clone(&rewritten_plan),
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
    )?
    else {
        return Ok(rewritten_plan);
    };
    if !host_arrow_child_replacement_preserves_schema(rewritten_plan.as_ref(), composed_attempt.plan.as_ref()) {
        return Ok(rewritten_plan);
    }

    append_planning_report(report, &composed_attempt.report);
    native_optimizer.merge(composed_attempt.native_optimizer);
    source_diagnostics.extend(composed_attempt.source_diagnostics);
    strict_bounded_contract.merge(composed_attempt.strict_bounded_contract);
    adaptive_pipeline.merge(composed_attempt.adaptive_pipeline);
    Ok(composed_attempt.plan)
}

fn try_compose_host_remainder_over_native_fragments(
    rewritten_plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
) -> DataFusionResult<Option<NativeOptimizationAttempt>> {
    if rewritten_plan.downcast_ref::<NativeCudfExec>().is_some()
        || rewritten_plan.downcast_ref::<NativeCudfComposedExec>().is_some()
        || gpu_resident_producer_input_name(rewritten_plan.as_ref()).is_some()
    {
        return Ok(None);
    }

    let mut state = NativeFragmentProducerization::from_plan(rewritten_plan.as_ref());
    let Some(producerized_plan) =
        producerize_native_fragments(Arc::clone(&rewritten_plan), &mut state, native_execution_config)?
    else {
        return Ok(None);
    };

    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&producerized_plan));
    let attempt = replace_lowered_root_or_keep_plan(
        producerized_plan,
        lowering,
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
    )?;
    if !attempt.rewritten || attempt.plan.downcast_ref::<NativeCudfComposedExec>().is_none() {
        return Ok(None);
    }
    Ok(Some(attempt))
}

#[derive(Debug, Default)]
struct NativeFragmentProducerization {
    next_fragment_index: usize,
    used_input_names: BTreeSet<String>,
}

impl NativeFragmentProducerization {
    fn from_plan(plan: &dyn ExecutionPlan) -> Self {
        let mut state = Self::default();
        collect_gpu_resident_input_names(plan, &mut state.used_input_names);
        state
    }

    fn next_input_name(&mut self) -> String {
        loop {
            let candidate = format!("__native_fragment_{}", self.next_fragment_index);
            self.next_fragment_index += 1;
            if self.used_input_names.insert(candidate.clone()) {
                return candidate;
            }
        }
    }
}

fn collect_gpu_resident_input_names(plan: &dyn ExecutionPlan, names: &mut BTreeSet<String>) {
    if let Some(input_name) = gpu_resident_producer_input_name(plan) {
        names.insert(input_name);
        return;
    }
    for child in plan.children() {
        collect_gpu_resident_input_names(child.as_ref(), names);
    }
}

fn producerize_native_fragments(
    plan: Arc<dyn ExecutionPlan>,
    state: &mut NativeFragmentProducerization,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(native) = plan.downcast_ref::<NativeCudfExec>() {
        let native_plan = native.native_plan().clone_with_gpu_resident_sink();
        if native_gpu_resident_producer_boundary_is_rejected(&native_plan, native_execution_config)? {
            return Ok(None);
        }
        let input_name = state.next_input_name();
        let producer = NativeCudfGpuProducerExec::try_new(input_name, native_plan).map_err(DataFusionError::from)?;
        return Ok(Some(Arc::new(producer)));
    }

    if plan.downcast_ref::<NativeCudfComposedExec>().is_some()
        || gpu_resident_producer_input_name(plan.as_ref()).is_some()
    {
        return Ok(None);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(None);
    }

    let mut changed = false;
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        if let Some(new_child) = producerize_native_fragments(Arc::clone(child), state, native_execution_config)? {
            changed = true;
            new_children.push(new_child);
        } else {
            new_children.push(Arc::clone(child));
        }
    }
    if !changed {
        return Ok(None);
    }
    plan.with_new_children(new_children).map(Some)
}

fn native_gpu_resident_producer_boundary_is_rejected(
    plan: &nexus_query_engine::QueryPlan,
    native_execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<bool> {
    match native_execution_config.native_pipeline_execution_mode() {
        nexus_query_engine::NativePipelineExecutionMode::StrictBounded => {
            native_execution_boundary_is_rejected(plan, native_execution_config)
        },
        nexus_query_engine::NativePipelineExecutionMode::Adaptive => {
            let diagnostics = plan_adaptive_pipeline_diagnostics(
                plan,
                GpuCandidate::new("NativeCudfGpuProducerExec", plan.output_schema().fields().len()),
                native_execution_config,
            )?
            .unwrap_or_default();
            Ok(adaptive_pipeline_was_rejected(&diagnostics))
        },
    }
}

fn optimize_native_child_without_composing_host_remainder(
    child: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    optimize_final_native_plan_with_composition(
        child,
        native_execution_config,
        native_optimizer_options,
        native_plan_preference_options,
        false,
    )
}

fn optimize_native_children_for_host_arrow_boundary(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NativeCudfExecutionConfig,
    native_optimizer_options: &nexus_query_engine::NativeOptimizerOptions,
    native_plan_preference_options: &nexus_query_engine::NativePlanPreferenceOptions,
    mode: HostArrowChildReplacementMode,
    report: &mut PlanningReport,
    native_optimizer: &mut nexus_query_engine::NativeOptimizerDiagnostics,
    source_diagnostics: &mut Vec<nexus_query_engine::SourceDiagnostics>,
    strict_bounded_contract: &mut NativeStrictBoundedContractDiagnostics,
    adaptive_pipeline: &mut NativeAdaptivePipelineDiagnostics,
) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
    if !host_arrow_boundary_allows_child_replacement(plan.as_ref()) {
        return Ok(None);
    }
    if precision_sensitive_aggregate_requires_whole_subtree_rejection(plan.as_ref())
        || subtree_contains_unsafe_join_residual(plan.as_ref())
    {
        return Ok(None);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(None);
    }

    let mut rewritten = false;
    let mut child_reports = Vec::new();
    let mut child_native_optimizer = nexus_query_engine::NativeOptimizerDiagnostics::default();
    let mut child_source_diagnostics = Vec::new();
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        if window_boundary_should_preserve_sort_child(plan.as_ref(), child.as_ref()) {
            if let Some(rewritten_child) = optimize_native_children_for_host_arrow_boundary(
                Arc::clone(child),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
                mode,
                report,
                native_optimizer,
                source_diagnostics,
                strict_bounded_contract,
                adaptive_pipeline,
            )? {
                rewritten = true;
                new_children.push(rewritten_child);
            } else {
                new_children.push(Arc::clone(child));
            }
            continue;
        }
        if mode == HostArrowChildReplacementMode::Normal
            && join_boundary_should_keep_datafusion_child(plan.as_ref(), child.as_ref())
        {
            new_children.push(Arc::clone(child));
            continue;
        }
        if subtree_contains_outer_join(child.as_ref()) {
            if let Some(rewritten_child) = optimize_native_children_for_host_arrow_boundary(
                Arc::clone(child),
                native_execution_config,
                native_optimizer_options,
                native_plan_preference_options,
                HostArrowChildReplacementMode::OuterJoinBoundary,
                report,
                native_optimizer,
                source_diagnostics,
                strict_bounded_contract,
                adaptive_pipeline,
            )? {
                rewritten = true;
                new_children.push(rewritten_child);
            } else {
                new_children.push(Arc::clone(child));
            }
            continue;
        }
        let attempt = optimize_native_child_without_composing_host_remainder(
            Arc::clone(child),
            native_execution_config,
            native_optimizer_options,
            native_plan_preference_options,
        )?;
        if attempt.rewritten && !host_arrow_child_replacement_preserves_schema(child.as_ref(), attempt.plan.as_ref()) {
            new_children.push(Arc::clone(child));
            continue;
        }
        rewritten |= attempt.rewritten;
        child_reports.push(attempt.report);
        child_native_optimizer.merge(attempt.native_optimizer);
        child_source_diagnostics.extend(attempt.source_diagnostics);
        strict_bounded_contract.merge(attempt.strict_bounded_contract);
        adaptive_pipeline.merge(attempt.adaptive_pipeline);
        new_children.push(attempt.plan);
    }

    if !rewritten {
        return Ok(None);
    }

    for child_report in child_reports {
        append_planning_report(report, &child_report);
    }
    native_optimizer.merge(child_native_optimizer);
    source_diagnostics.extend(child_source_diagnostics);
    plan.with_new_children(new_children).map(Some)
}

fn host_arrow_child_replacement_preserves_schema(
    original: &dyn ExecutionPlan,
    replacement: &dyn ExecutionPlan,
) -> bool {
    original.schema().as_ref() == replacement.schema().as_ref()
}

fn append_planning_report(report: &mut PlanningReport, extra: &PlanningReport) {
    *report = PlanningReport::from_outcomes(report.outcomes().iter().cloned().chain(extra.outcomes().iter().cloned()));
}

fn host_arrow_boundary_allows_child_replacement(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<SortExec>().is_some()
        || plan.downcast_ref::<GlobalLimitExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some()
        || plan.downcast_ref::<AggregateExec>().is_some()
        || plan.downcast_ref::<CoalescePartitionsExec>().is_some()
        || plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<HashJoinExec>().is_some()
        || plan.downcast_ref::<CrossJoinExec>().is_some()
        || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.downcast_ref::<UnionExec>().is_some()
        || plan.downcast_ref::<WindowAggExec>().is_some()
        || plan.downcast_ref::<BoundedWindowAggExec>().is_some()
}

fn join_boundary_should_keep_datafusion_child(parent: &dyn ExecutionPlan, child: &dyn ExecutionPlan) -> bool {
    if parent.downcast_ref::<HashJoinExec>().is_none() && parent.downcast_ref::<NestedLoopJoinExec>().is_none() {
        return false;
    }
    if subtree_contains_table_format_scan(child) {
        return false;
    }
    child.downcast_ref::<DataSourceExec>().is_some() || child.downcast_ref::<FilterExec>().is_some()
}

fn subtree_contains_table_format_scan(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<CudfTableFormatScanExec>().is_some()
        || plan.children().into_iter().any(|child| subtree_contains_table_format_scan(child.as_ref()))
}

fn window_boundary_should_preserve_sort_child(parent: &dyn ExecutionPlan, child: &dyn ExecutionPlan) -> bool {
    (parent.downcast_ref::<WindowAggExec>().is_some() || parent.downcast_ref::<BoundedWindowAggExec>().is_some())
        && child.downcast_ref::<SortExec>().is_some()
}

// q39-style final full-sort/stddev subtrees are precision-sensitive: replacing
// children below the CPU stddev aggregate changes the input order/batching that
// DataFusion's Welford accumulator observes and can produce final-bit drift.
fn precision_sensitive_aggregate_requires_whole_subtree_rejection(plan: &dyn ExecutionPlan) -> bool {
    let Some(sort) = plan.downcast_ref::<SortExec>() else {
        return false;
    };
    sort.fetch().is_none() && subtree_contains_precision_sensitive_unsupported_aggregate(plan)
}

fn subtree_contains_precision_sensitive_unsupported_aggregate(plan: &dyn ExecutionPlan) -> bool {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
        && aggregate.aggr_expr().iter().any(|expr| {
            let function_name = expr.fun().name();
            function_name.eq_ignore_ascii_case("stddev") || function_name.eq_ignore_ascii_case("stddev_samp")
        })
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_precision_sensitive_unsupported_aggregate(child.as_ref()))
}

fn subtree_contains_unsafe_join_residual(plan: &dyn ExecutionPlan) -> bool {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>()
        && join
            .filter()
            .is_some_and(|filter| join_residual_contains_nullable_scalar_or(filter.expression(), filter.schema()))
    {
        return true;
    }
    if let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>()
        && join
            .filter()
            .is_some_and(|filter| join_residual_contains_nullable_scalar_or(filter.expression(), filter.schema()))
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_unsafe_join_residual(child.as_ref()))
}

fn subtree_contains_outer_join(plan: &dyn ExecutionPlan) -> bool {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>()
        && matches!(join.join_type(), JoinType::Left | JoinType::Right | JoinType::Full)
    {
        return true;
    }
    if let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>()
        && matches!(join.join_type(), JoinType::Left | JoinType::Right | JoinType::Full)
    {
        return true;
    }
    plan.children().iter().any(|child| subtree_contains_outer_join(child.as_ref()))
}

fn join_residual_contains_nullable_scalar_or(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>()
        && matches!(binary.op(), datafusion::logical_expr::Operator::Or)
        && expr.nullable(schema).unwrap_or(true)
        && expr_contains_scalar_function(expr)
    {
        return true;
    }
    expr.children().into_iter().any(|child| join_residual_contains_nullable_scalar_or(child, schema))
}

fn expr_contains_scalar_function(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<ScalarFunctionExpr>().is_some()
        || expr.children().into_iter().any(expr_contains_scalar_function)
}

fn native_execution_boundary_is_rejected(
    plan: &nexus_query_engine::QueryPlan,
    execution_config: &NativeCudfExecutionConfig,
) -> DataFusionResult<bool> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(false);
    }
    strict_bounded_admission_is_rejected(plan, execution_config)
}

fn strict_bounded_rejection_blocks_child_rewrites(diagnostics: &NativeStrictBoundedContractDiagnostics) -> bool {
    diagnostics.rejected_candidates.iter().any(|candidate| {
        candidate.blocking_contract.required_contract == BLOCKING_ROLLUP_RANK_FULL_SCAN_REQUIRES_EXACT_CONTRACTS
    })
}

fn native_execution_rejection_report(report: &PlanningReport, rejection_reason: FallbackReason) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: rejection_reason }
        },
        CandidateOutcome::Rejected { candidate, reason } => {
            CandidateOutcome::Rejected { candidate: candidate.clone(), reason: *reason }
        },
    }))
}

fn native_execution_rejection_report_for_candidate(
    candidate: GpuCandidate,
    rejection_reason: FallbackReason,
) -> PlanningReport {
    PlanningReport::from_outcomes([CandidateOutcome::Rejected { candidate, reason: rejection_reason }])
}

fn single_rewritten_candidate(report: &PlanningReport) -> Option<&GpuCandidate> {
    let mut rewritten = report.outcomes().iter().filter_map(|outcome| match outcome {
        CandidateOutcome::Rewritten { candidate } => Some(candidate),
        CandidateOutcome::Rejected { .. } => None,
    });
    let candidate = rewritten.next()?;
    rewritten.next().is_none().then_some(candidate)
}

fn native_candidate_error(message: &'static str) -> DataFusionError {
    Error::new(
        crate::error::ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.native.pipeline_candidate_diagnostics",
        message,
    )
    .into()
}

fn peel_output_requirement_exec(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.name() == "OutputRequirementExec"
        && let [child] = plan.children().as_slice()
    {
        return Arc::clone(child);
    }
    plan
}

pub(super) fn peel_cooperative_native_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.name() != "CooperativeExec" {
        return None;
    }
    let children = plan.children();
    let [child] = children.as_slice() else {
        return None;
    };
    child.downcast_ref::<NativeCudfExec>().is_some().then(|| Arc::clone(child))
}
