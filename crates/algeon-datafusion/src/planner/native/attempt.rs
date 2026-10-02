use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use datafusion::{
    common::Result as DataFusionResult,
    physical_expr::EquivalenceProperties,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
};
use query_engine::{
    admission::{AdmissionDecision, AdmissionRejection, AdmittedPlan},
    capability::CapabilityRejectionCategory,
    dimension_pruning::local_dimension_pruning_resolver,
    error::QueryError,
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions, optimize_query_plan},
    plan::{
        NativePlanCostDecision, NativePlanCostOptions, NativePlanCostReason, PlanNode, QueryPlan, QueryPlanBuilder,
        native_plan_cost_decision,
    },
    rapids_backend::rapids_analysis_context,
    source::{SourceDiagnostics, SourceNode},
};

use super::{
    NativeCapabilityContractDiagnostics,
    capability_envelope::{
        accepted_plan_contract_diagnostics, native_admission_decision, rejected_plan_contract_diagnostics,
    },
};
use crate::{
    capability::CapabilityReason,
    error::{Error, ErrorCode, Result},
    native::{
        FooterScope, NativeLoweringResult, floating_statistical_aggregate_input_is_partitioned,
        lower_count_only_physical_plan_to_native, lower_physical_plan_to_native,
        native_plan_reproduces_declared_output_ordering, reestablish_declared_output_ordering_if_needed,
        wrap_candidate_with_declared_output_ordering_sort,
    },
    native_exec::{
        AlgeonNativeExec, AlgeonNativeExecutionConfig, NativeOutput, gpu_resident_producer,
        gpu_resident_producer_input_name, rewrite_gpu_resident_producer_for_subsumed_composed_consumer,
    },
    report::{CandidateOutcome, GpuCandidate, NativeCandidateCostEvidence, NotSupportedDetail, PlanningReport},
};

/// Optimizer attempt output, including the replacement plan and all report evidence.
#[derive(Debug, Clone)]
pub(super) struct NativeOptimizationAttempt {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) report: PlanningReport,
    pub(super) native_optimizer: NativeOptimizerDiagnostics,
    pub(super) source_diagnostics: Vec<SourceDiagnostics>,
    pub(super) capability_contract: NativeCapabilityContractDiagnostics,
    pub(super) rewritten: bool,
}

impl NativeOptimizationAttempt {
    /// Retains the DataFusion plan when relational native rewriting is
    /// intentionally disabled. GPU UDTF discovery and admission still run
    /// in the final optimizer stage after this selection result is produced.
    pub(super) fn retained(plan: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            plan,
            report: PlanningReport::new(),
            native_optimizer: NativeOptimizerDiagnostics::default(),
            source_diagnostics: Vec::new(),
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        }
    }
}

/// Plan-cost selection inputs: the configured thresholds, and whether the
/// completed plan may keep a DataFusion operator in place of a native one.
#[derive(Debug, Clone, Copy)]
pub(super) struct NativeCostSelection {
    pub(super) options: NativePlanCostOptions,
    pub(super) datafusion_cpu_allowed: bool,
}

pub(super) fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
    root_sink: &NativeReplacementSink,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let _footers = FooterScope::enter();
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering =
        lower_count_only_physical_plan_to_native(lowering_input, &native_execution_config.source_read_strategy())?;
    replace_lowered_root_or_keep_plan_for_sink(
        plan,
        None,
        lowering,
        native_execution_config,
        native_optimizer_options,
        cost_selection,
        root_sink.clone(),
    )
}

pub(super) fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
    root_sink: &NativeReplacementSink,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let _footers = FooterScope::enter();
    let mut relation_inputs = LoweredRelationInputs::default();
    let plan = lower_gpu_resident_producer_relation_inputs(
        plan,
        native_execution_config,
        native_optimizer_options,
        cost_selection,
        &mut relation_inputs,
    )?;
    let mut attempt =
        optimize_final_native_root(plan, native_execution_config, native_optimizer_options, cost_selection, root_sink)?;
    // The root attempt selects at most one whole-plan candidate, so relation
    // input outcomes are appended only after it has consumed its own report.
    attempt.report.append(relation_inputs.report);
    attempt.source_diagnostics.extend(relation_inputs.source_diagnostics);
    Ok(attempt)
}

/// Planning outcomes and source diagnostics of relation inputs lowered into
/// GPU-resident producers.
#[derive(Debug, Default)]
struct LoweredRelationInputs {
    report: PlanningReport,
    source_diagnostics: Vec<SourceDiagnostics>,
}

/// Lowers each DataFusion child of a GPU relation function into a
/// GPU-resident producer bound under the function's input name.
///
/// Children are rewritten bottom-up, so a relation input that itself contains
/// a relation function lowers into a composed producer over the already
/// rewritten inner function. A child that does not lower keeps its DataFusion
/// plan and is host-staged at execution; its planning outcome stays in the
/// report.
fn lower_gpu_resident_producer_relation_inputs(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
    lowered: &mut LoweredRelationInputs,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let relation_input_names = match gpu_resident_producer(plan.as_ref()) {
        Some(producer) => match producer.native_relation_input_names() {
            Some(names) => Some(names.to_vec()),
            None => return Ok(plan),
        },
        None => None,
    };
    let children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    let mut rewritten_children = Vec::with_capacity(children.len());
    let mut children_changed = false;
    for (index, child) in children.iter().enumerate() {
        let mut rewritten = lower_gpu_resident_producer_relation_inputs(
            Arc::clone(child),
            native_execution_config,
            native_optimizer_options,
            cost_selection,
            lowered,
        )?;
        if let Some(input_name) = relation_input_names.as_ref().and_then(|names| names.get(index))
            && gpu_resident_producer(rewritten.as_ref()).is_none()
        {
            let lowering =
                lower_physical_plan_to_native(Arc::clone(&rewritten), &native_execution_config.source_read_strategy())?;
            let mut attempt = replace_lowered_root_or_keep_plan_for_sink(
                Arc::clone(&rewritten),
                None,
                lowering,
                native_execution_config,
                native_optimizer_options,
                cost_selection,
                NativeReplacementSink::GpuResidentProducer { input_name: input_name.clone() },
            )?;
            attempt.report.wrap_selected_candidates_with_outer_operator(plan.name());
            lowered.report.append(attempt.report);
            lowered.source_diagnostics.extend(attempt.source_diagnostics);
            if attempt.rewritten {
                rewritten = attempt.plan;
            }
        }
        children_changed |= !Arc::ptr_eq(child, &rewritten);
        rewritten_children.push(rewritten);
    }
    if !children_changed {
        return Ok(plan);
    }
    plan.replace_children(rewritten_children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
}

fn optimize_final_native_root(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
    root_sink: &NativeReplacementSink,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let source_reads = native_execution_config.source_read_strategy();
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan), &source_reads)?;
    if whole_plan_has_unsupported_expression_miss(&lowering)
        && let Some(boundary) = super::mixed::CertifiedMixedFilterBoundary::certify(Arc::clone(&plan))
    {
        let native_leaf_lowering = lower_physical_plan_to_native(Arc::clone(boundary.source()), &source_reads)?;
        if native_leaf_lowering.plan.as_ref().is_some_and(|native_leaf| boundary.accepts_native_leaf(native_leaf)) {
            let mut native_leaf = replace_lowered_root_or_keep_plan(
                Arc::clone(boundary.source()),
                Some(boundary.native_equivalence_properties()),
                native_leaf_lowering,
                native_execution_config,
                native_optimizer_options,
                cost_selection,
            )?;
            if native_leaf.rewritten
                && let Some(plan) = boundary.rebuild(Arc::clone(&native_leaf.plan))
            {
                native_leaf.plan = plan;
                return Ok(native_leaf);
            }
        }
    }
    replace_lowered_root_or_keep_plan_for_sink(
        plan,
        None,
        lowering,
        native_execution_config,
        native_optimizer_options,
        cost_selection,
        root_sink.clone(),
    )
}

/// Result boundary of a lowered native replacement.
#[derive(Debug, Clone)]
pub(super) enum NativeReplacementSink {
    /// The replacement streams Arrow batches to its DataFusion parent.
    HostArrow,
    /// The replacement stays on the device and binds into its parent relation
    /// function under `input_name`.
    GpuResidentProducer { input_name: String },
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    output_equivalence_properties: Option<EquivalenceProperties>,
    lowering: NativeLoweringResult,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
) -> DataFusionResult<NativeOptimizationAttempt> {
    replace_lowered_root_or_keep_plan_for_sink(
        plan,
        output_equivalence_properties,
        lowering,
        native_execution_config,
        native_optimizer_options,
        cost_selection,
        NativeReplacementSink::HostArrow,
    )
}

fn replace_lowered_root_or_keep_plan_for_sink(
    plan: Arc<dyn ExecutionPlan>,
    output_equivalence_properties: Option<EquivalenceProperties>,
    lowering: NativeLoweringResult,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    cost_selection: NativeCostSelection,
    sink: NativeReplacementSink,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let mut lowering = lowering;
    stamp_planning_report_correlation(&mut lowering.report, native_execution_config);
    let Some(lowered_plan) = lowering.plan.take() else {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer: NativeOptimizerDiagnostics::default(),
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        });
    };
    lowering.report.anchor_selected_candidates_to_plan_node(lowered_plan.root_id());
    let candidate_before_declared_output_sort = lowering
        .report
        .single_selected_candidate()
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;
    let optimizer_options = native_optimizer_options_for_execution(native_optimizer_options, native_execution_config)?;
    let optimizer_result = optimize_query_plan(
        lowered_plan,
        &optimizer_options,
        rapids_analysis_context(native_execution_config.planning_policy()),
        local_dimension_pruning_resolver(),
    )
    .map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native_optimizer.optimize",
            "native optimizer produced a malformed plan",
            error,
        )
    })?;
    let native_optimizer = optimizer_result.diagnostics;
    let shared_resident_producers = optimizer_result.shared_resident_producers;
    let ordered_output_sort_needed = plan.properties().output_ordering().is_some()
        && !native_plan_reproduces_declared_output_ordering(&optimizer_result.plan);
    let (native_plan, appended_declared_output_sort) = match reestablish_declared_output_ordering_if_needed(
        optimizer_result.plan,
        plan.properties().output_ordering(),
    ) {
        Ok(native_plan) => native_plan,
        Err(reason) => {
            let candidate = if ordered_output_sort_needed {
                wrap_candidate_with_declared_output_ordering_sort(candidate_before_declared_output_sort)
            } else {
                candidate_before_declared_output_sort
            };
            return Ok(NativeOptimizationAttempt {
                plan,
                report: native_execution_rejection_report_for_candidate(
                    candidate,
                    reason,
                    NotSupportedDetail::default(),
                    &lowering.report,
                ),
                native_optimizer,
                source_diagnostics: lowering.source_diagnostics,
                capability_contract: NativeCapabilityContractDiagnostics::default(),
                rewritten: false,
            });
        },
    };
    if appended_declared_output_sort {
        lowering.report.wrap_selected_candidates_with_outer_operator("SortExec");
    }
    lowering.report.anchor_selected_candidates_to_plan_node(native_plan.root_id());
    if let NativePlanCostDecision::NotSelectedByCost { reason, facts } =
        native_plan_cost_decision(&native_plan, &cost_selection.options, cost_selection.datafusion_cpu_allowed)
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_cost_selection_report(
                &lowering.report,
                reason,
                NativeCandidateCostEvidence::from_cost_facts(facts),
            ),
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        });
    }

    lowering.report.anchor_selected_candidates_to_plan_node(native_plan.root_id());
    let candidate = lowering
        .report
        .single_selected_candidate()
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;

    if native_execution_config.statistical_aggregate_execution_mode().is_datafusion_exact()
        && floating_statistical_aggregate_input_is_partitioned(plan.as_ref())
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_rejection_report_for_candidate(
                candidate,
                CapabilityReason::NativeExactStatisticalInputPartitioned,
                NotSupportedDetail::default(),
                &lowering.report,
            ),
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        });
    }

    // The candidate is decided once, on the sink form execution runs; the
    // boundary check and both contract-diagnostics branches read this decision.
    let native_plan = match &sink {
        NativeReplacementSink::HostArrow => native_plan.clone_with_deferred_host_materialization_sink(),
        NativeReplacementSink::GpuResidentProducer { .. } => native_plan.clone_with_gpu_resident_sink(),
    };
    let admission = match native_admission_decision(&native_plan, native_execution_config)? {
        AdmissionDecision::Admitted(admission) => admission,
        AdmissionDecision::Rejected(rejection) => {
            return rejected_attempt(
                plan,
                candidate,
                (&native_plan, &rejection),
                native_optimizer,
                lowering,
                native_execution_config,
            );
        },
    };
    // Each shared producer the optimizer extracted executes as its own plan,
    // so it is decided here too; a refused producer refuses the candidate.
    let mut shared_producers = Vec::with_capacity(shared_resident_producers.len());
    for producer in shared_resident_producers {
        match native_admission_decision(&producer.plan, native_execution_config)? {
            AdmissionDecision::Admitted(admitted) => shared_producers.push((producer.input_name, admitted)),
            AdmissionDecision::Rejected(rejection) => {
                return rejected_attempt(
                    plan,
                    candidate,
                    (&producer.plan, &rejection),
                    native_optimizer,
                    lowering,
                    native_execution_config,
                );
            },
        }
    }

    let datafusion_shape = candidate.shape.to_string();
    let capability_contract = accepted_plan_contract_diagnostics(candidate, native_execution_config, &admission)?;
    let plan = native_datafusion_exec_for_plan(
        Arc::clone(&plan),
        admission,
        native_execution_config,
        &datafusion_shape,
        output_equivalence_properties.unwrap_or_else(|| plan.properties().equivalence_properties().clone()),
        shared_producers,
        sink,
    )?;
    Ok(NativeOptimizationAttempt {
        plan,
        report: lowering.report,
        native_optimizer,
        source_diagnostics: lowering.source_diagnostics,
        capability_contract,
        rewritten: true,
    })
}

fn rejected_attempt(
    plan: Arc<dyn ExecutionPlan>,
    candidate: GpuCandidate,
    (rejected_plan, rejection): (&QueryPlan, &AdmissionRejection),
    native_optimizer: NativeOptimizerDiagnostics,
    lowering: NativeLoweringResult,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let report = native_execution_rejection_report_for_candidate(
        candidate.clone(),
        rejection_capability_reason(rejected_plan, rejection),
        NotSupportedDetail { admission: Some(rejection.rejection().clone()), ..NotSupportedDetail::default() },
        &lowering.report,
    );
    Ok(NativeOptimizationAttempt {
        plan,
        report,
        native_optimizer,
        source_diagnostics: lowering.source_diagnostics,
        capability_contract: rejected_plan_contract_diagnostics(candidate, native_execution_config, rejection)?,
        rewritten: false,
    })
}

fn rejection_capability_reason(plan: &QueryPlan, rejection: &AdmissionRejection) -> CapabilityReason {
    if rejection.is_execution_mode_refusal() || rejection.validation_error().is_some() {
        return match rejection.rejection().category() {
            CapabilityRejectionCategory::UnsupportedArrowDtype => CapabilityReason::UnsupportedType,
            CapabilityRejectionCategory::UnsupportedExpressionFamily => CapabilityReason::UnsupportedExpression,
            _ => CapabilityReason::UnsupportedOperator,
        };
    }
    if plan.contains_join() {
        CapabilityReason::NativeBoundedJoinUnsupported
    } else {
        CapabilityReason::NativeCompiledPipelineUnsupported
    }
}

fn whole_plan_has_unsupported_expression_miss(lowering: &NativeLoweringResult) -> bool {
    lowering.plan.is_none()
        && matches!(
            lowering.report.outcomes(),
            [CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }]
        )
}

fn native_optimizer_options_for_execution(
    options: &NativeOptimizerOptions,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> Result<NativeOptimizerOptions> {
    native_execution_config.validate_query_policies()?;
    let options = options.clone();
    options.validate().map_err(|error| {
        Error::from_native(
            "algeon_datafusion.native_exec.config.validate",
            "invalid native optimizer option policy",
            stamp_native_error_correlation(error, native_execution_config),
        )
    })?;
    Ok(options)
}

fn native_datafusion_exec_for_plan(
    original_plan: Arc<dyn ExecutionPlan>,
    admitted: Arc<AdmittedPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    datafusion_shape: &str,
    eq_properties: EquivalenceProperties,
    shared_producers: Vec<(String, Arc<AdmittedPlan>)>,
    sink: NativeReplacementSink,
) -> Result<Arc<dyn ExecutionPlan>> {
    let native_execution_config =
        native_execution_config.clone().with_plan_task_mapping_datafusion_shape(datafusion_shape);
    let output = match sink {
        NativeReplacementSink::HostArrow => NativeOutput::Host,
        NativeReplacementSink::GpuResidentProducer { input_name } => NativeOutput::GpuResident { input_name },
    };
    let gpu_input_names = native_plan_gpu_resident_source_names(admitted.plan());
    if gpu_input_names.is_empty() {
        return AlgeonNativeExec::try_new(admitted, Vec::new(), output, native_execution_config, Some(eq_properties))
            .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>);
    }

    let shared_input_names = shared_producers.iter().map(|(input_name, _)| input_name.clone()).collect::<BTreeSet<_>>();
    let external_gpu_input_names = gpu_input_names.difference(&shared_input_names).cloned().collect();
    let mut gpu_inputs = gpu_resident_producer_children(&original_plan, &external_gpu_input_names)?;
    for (input_name, producer) in shared_producers {
        gpu_inputs.push(Arc::new(AlgeonNativeExec::try_new(
            producer,
            Vec::new(),
            NativeOutput::GpuResident { input_name },
            native_execution_config.clone(),
            None,
        )?));
    }
    gpu_inputs.sort_by(|left, right| {
        gpu_resident_producer_input_name(left.as_ref()).cmp(&gpu_resident_producer_input_name(right.as_ref()))
    });
    let (admitted, gpu_inputs) = if output == NativeOutput::Host {
        remove_subsumed_composed_consumer(admitted, gpu_inputs, &native_execution_config)?
    } else {
        (admitted, gpu_inputs)
    };
    AlgeonNativeExec::try_new(admitted, gpu_inputs, output, native_execution_config, Some(eq_properties))
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
}

/// A composed consumer's admitted plan and the producers it reads.
type ComposedConsumer = (Arc<AdmittedPlan>, Vec<Arc<dyn ExecutionPlan>>);

fn remove_subsumed_composed_consumer(
    admitted: Arc<AdmittedPlan>,
    mut gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> Result<ComposedConsumer> {
    let [producer] = gpu_inputs.as_slice() else {
        return Ok((admitted, gpu_inputs));
    };
    let Some(rewritten_producer) =
        rewrite_gpu_resident_producer_for_subsumed_composed_consumer(producer.as_ref(), admitted.plan())?
    else {
        return Ok((admitted, gpu_inputs));
    };
    gpu_inputs[0] = rewritten_producer;
    let source = admitted
        .plan()
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => Some(source.as_ref().clone()),
            _ => None,
        })
        .ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.native_optimizer.subsumed_consumer",
                "a subsumed composed consumer is missing its GPU-resident source",
            )
        })?;
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source);
    // The producer now runs the consumer's work, so the composed node only
    // reads its output; that pass-through plan is decided like any other.
    let pass_through = builder
        .finish_with_host_sink(source)
        .map_err(|error| {
            Error::from_native(
                "algeon_datafusion.native_optimizer.subsumed_consumer",
                "a subsumed composed consumer pass-through plan is malformed",
                error,
            )
        })?
        .clone_with_deferred_host_materialization_sink();
    match native_admission_decision(&pass_through, native_execution_config)? {
        AdmissionDecision::Admitted(pass_through) => Ok((pass_through, gpu_inputs)),
        AdmissionDecision::Rejected(rejection) => Err(Error::from_native(
            "algeon_datafusion.native_optimizer.subsumed_consumer",
            "the pass-through consumer of a subsumed composed plan is not admitted",
            rejection.execution_error(&pass_through),
        )),
    }
}

fn native_plan_gpu_resident_source_names(plan: &QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                SourceNode::LocalParquet(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

fn gpu_resident_producer_children(
    plan: &Arc<dyn ExecutionPlan>,
    required_inputs: &BTreeSet<String>,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    let mut producers = BTreeMap::<String, Arc<dyn ExecutionPlan>>::new();
    collect_gpu_resident_producer_children(plan, required_inputs, &mut producers);
    let missing =
        required_inputs.iter().filter(|name| !producers.contains_key(*name)).map(String::as_str).collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.native.compose_gpu_resident",
            "native plan requires GPU-resident inputs that were not produced by child execution plans",
        )
        .with_fact("missing_inputs", missing.join(",")));
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

fn native_execution_cost_selection_report(
    report: &PlanningReport,
    cost_reason: NativePlanCostReason,
    evidence: NativeCandidateCostEvidence,
) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Selected { candidate } => CandidateOutcome::NotSelectedByCost {
            candidate: candidate.clone(),
            cost_reason,
            evidence: evidence.clone(),
        },
        CandidateOutcome::NotSupported { candidate, capability_reason, detail } => CandidateOutcome::NotSupported {
            candidate: candidate.clone(),
            capability_reason: *capability_reason,
            detail: detail.clone(),
        },
        CandidateOutcome::NotSelectedByCost { candidate, cost_reason, evidence } => {
            CandidateOutcome::NotSelectedByCost {
                candidate: candidate.clone(),
                cost_reason: *cost_reason,
                evidence: evidence.clone(),
            }
        },
    }))
    .with_correlation_from(report)
}

fn native_execution_rejection_report_for_candidate(
    candidate: GpuCandidate,
    rejection_reason: CapabilityReason,
    detail: NotSupportedDetail,
    source_report: &PlanningReport,
) -> PlanningReport {
    PlanningReport::from_outcomes([CandidateOutcome::NotSupported {
        candidate,
        capability_reason: rejection_reason,
        detail,
    }])
    .with_correlation_from(source_report)
}

fn stamp_planning_report_correlation(
    report: &mut PlanningReport,
    native_execution_config: &AlgeonNativeExecutionConfig,
) {
    if let Some(query_id) = native_execution_config.query_id() {
        report.set_query_id(query_id.clone());
    }
    if let Some(trace_id) = native_execution_config.trace_id() {
        report.set_trace_id(trace_id.clone());
    }
}

fn stamp_native_error_correlation(
    mut error: QueryError,
    native_execution_config: &AlgeonNativeExecutionConfig,
) -> QueryError {
    if let Some(query_id) = native_execution_config.query_id() {
        error = error.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = native_execution_config.trace_id() {
        error = error.with_trace_id(trace_id.clone());
    }
    error
}

fn native_candidate_error(message: &'static str) -> Error {
    Error::new(ErrorCode::InvalidExecutionPlan, "algeon_datafusion.native.pipeline_candidate_diagnostics", message)
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
    child
        .downcast_ref::<AlgeonNativeExec>()
        .is_some_and(|native| native.materializes_host() && native.gpu_inputs().is_empty())
        .then(|| Arc::clone(child))
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{ArrayRef, Int32Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use query_engine::{
        capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions, Reason},
        chunk::SourceChunkPlanningOptions,
        dynamic_filter::DynamicFilterExecutionOptions,
        expr::{Expr, SortExpr},
        optimizer::SemijoinReductionOptions,
        plan::{NativePlanCostFacts, WindowFunction, WindowSpec},
        source::LiteralBatchSource,
    };

    use super::*;
    use crate::native_exec::SourceChunking;

    #[test]
    fn non_join_candidate_is_rejected_at_planning_with_its_missing_contract() {
        // Row numbers under ORDER BY over a source that records no ordering:
        // no compiled contract covers the window, so the candidate stays on
        // DataFusion with that reason instead of running on a fallback.
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 1])) as ArrayRef])
            .expect("literal batch");
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        let window = builder
            .add_window(
                source,
                WindowSpec::new(
                    Vec::new(),
                    vec![SortExpr::new(Expr::column("value"), false, false)],
                    vec![WindowFunction::row_number("row_number")],
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(window).unwrap().clone_with_deferred_host_materialization_sink();
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"));

        let AdmissionDecision::Rejected(rejection) =
            native_admission_decision(&plan, &execution_config).expect("the candidate is decidable")
        else {
            panic!("a row number over unordered input has no compiled contract");
        };

        assert_eq!(rejection.rejection().reason, Reason::WindowInputOrderingNotEstablished);
        assert_eq!(rejection_capability_reason(&plan, &rejection), CapabilityReason::NativeCompiledPipelineUnsupported);
    }

    #[test]
    fn cost_selection_preserves_full_cost_evidence() {
        let facts = NativePlanCostFacts {
            node_count: 7,
            source_count: 1,
            local_parquet_source_count: 1,
            selected_row_groups_known: true,
            selected_row_groups: Some(2),
            estimated_source_rows: Some(128),
            estimated_source_bytes: Some(4_096),
            selected_source_bytes_known: true,
            join_count: 1,
            residual_join_count: 0,
            aggregate_count: 1,
            grouped_aggregate_count: 1,
            sort_count: 1,
            limit_count: 1,
            min_topk_fetch: Some(10),
        };
        let report = PlanningReport::from_outcomes([CandidateOutcome::Selected {
            candidate: GpuCandidate::new("SortExec(HashJoinExec)", 3),
        }]);

        let report = native_execution_cost_selection_report(
            &report,
            NativePlanCostReason::SmallSelectiveTopK,
            NativeCandidateCostEvidence::from_cost_facts(facts.clone()),
        );

        assert!(matches!(
            report.outcomes(),
            [CandidateOutcome::NotSelectedByCost { cost_reason, evidence, .. }]
                if *cost_reason == NativePlanCostReason::SmallSelectiveTopK
                    && evidence.node_count == facts.node_count
                    && evidence.selected_row_groups == facts.selected_row_groups
                    && evidence.estimated_source_bytes == facts.estimated_source_bytes
                    && evidence.min_topk_fetch == facts.min_topk_fetch
        ));
        let json = report.to_json();
        assert_eq!(json["outcomes"][0]["outcome"], "not_selected_by_cost");
        assert_eq!(json["outcomes"][0]["cost_reason_code"], "native_small_selective_topk_not_selected_by_cost");
        assert_eq!(json["outcomes"][0]["cost_evidence"]["estimated_source_bytes"], 4_096);
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_expert_query_policy() {
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_dynamic_filter_execution_options(
                    DynamicFilterExecutionOptions::default()
                        .with_max_build_rows(10)
                        .with_large_local_parquet_range_producer_rows(11),
                );

        let error =
            native_optimizer_options_for_execution(&NativeOptimizerOptions::disabled(), &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
        assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_keyed_hash_join_policy() {
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_keyed_hash_join_strategy_options(
                    KeyedHashJoinStrategyOptions::default().with_large_keyed_hash_build_row_threshold(10),
                );

        let error =
            native_optimizer_options_for_execution(&NativeOptimizerOptions::disabled(), &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("large_field"), Some("large_keyed_hash_build_row_threshold"));
        assert_eq!(error.fact_value("reason"), Some("large_threshold_below_small_threshold"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_aggregate_strategy_policy() {
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_aggregate_strategy_execution_options(
                    AggregateStrategyExecutionOptions::default()
                        .with_encoded_key_partial_low_cardinality_threshold_ppm(1_000_001),
                );

        let error =
            native_optimizer_options_for_execution(&NativeOptimizerOptions::disabled(), &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("encoded_key_partial_low_cardinality_threshold_ppm"));
        assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_source_chunk_planning_policy() {
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"))
                .with_source_chunk_planning_options(
                    SourceChunkPlanningOptions::default().with_local_parquet_throughput_min_device_budget_bytes(0),
                );
        let error =
            native_optimizer_options_for_execution(&NativeOptimizerOptions::disabled(), &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_min_device_budget_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_optimizer_policy() {
        let optimizer_options = NativeOptimizerOptions {
            semijoin_reduction: SemijoinReductionOptions { cost_threshold_multiplier: 0, ..Default::default() },
            ..NativeOptimizerOptions::disabled()
        };
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"));

        let error = native_optimizer_options_for_execution(&optimizer_options, &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("cost_threshold_multiplier"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }
}
