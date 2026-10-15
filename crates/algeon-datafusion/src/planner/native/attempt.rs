//! One lowering candidate, from native IR to a decided replacement.
//!
//! `replace_lowered_root_or_keep_plan_for_sink` applies the decisions in a
//! fixed order: native optimizer, ordering reproduction, sink form, plan cost,
//! the exact statistical-aggregate partition rule, then capability admission
//! of the candidate and of every shared producer the optimizer extracted. A
//! decline at any step returns the original DataFusion plan with
//! `rewritten: false` and a report that names the step; malformed IR or an
//! invalid configuration is an error. The decision is made on the sink form
//! execution runs (deferred host materialization or GPU-resident), so the
//! admitted plan and the executed plan are the same object.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use datafusion::{
    common::{Result as DataFusionResult, config::OptimizerOptions},
    physical_expr::EquivalenceProperties,
    physical_plan::{ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions},
};
use query_engine::{dimension_pruning::local_dimension_pruning_resolver, rapids_backend::rapids_analysis_context};
use query_planning::{
    admission::{AdmissionDecision, AdmissionRejection, AdmittedPlan},
    capability::CapabilityRejectionCategory,
    error::QueryError,
    optimizer::{NativeOptimizerDiagnostics, NativeOptimizerOptions, optimize_query_plan},
    plan::{NativePlanCostReason, PlanNode, QueryPlan, QueryPlanBuilder, native_plan_cost_decision},
    source::{SourceDiagnostics, SourceNode},
};

use super::{
    NativeCapabilityContractDiagnostics,
    capability_envelope::{
        accepted_plan_contract_diagnostics, native_admission_decision, rejected_plan_contract_diagnostics,
    },
    fragment::FragmentSearch,
    shared_resident::share_island_subplans,
};
use crate::{
    capability::CapabilityReason,
    error::{Error, ErrorCode, Result},
    native::{
        FooterScope, NativeLoweringResult, first_join_operator, floating_statistical_aggregate_input_is_partitioned,
        lower_count_only_physical_plan_to_native, lower_physical_plan_to_native,
        native_plan_reproduces_declared_output_ordering, reestablish_declared_output_ordering_if_needed,
        wrap_candidate_with_declared_output_ordering_sort,
    },
    native_exec::{
        AlgeonNativeExec, AlgeonNativeExecutionConfig, NativeOutput, gpu_resident_producer,
        gpu_resident_producer_input_name, rewrite_gpu_resident_producer_for_subsumed_composed_consumer,
    },
    report::{CandidateOutcome, GpuCandidate, NotSupportedDetail, PlanningReport},
};

#[derive(Debug, Clone)]
pub(super) struct NativeOptimizationAttempt {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) report: PlanningReport,
    pub(super) native_optimizer: NativeOptimizerDiagnostics,
    pub(super) source_diagnostics: Vec<SourceDiagnostics>,
    pub(super) capability_contract: NativeCapabilityContractDiagnostics,
    /// Whether `plan` replaces the input; `false` keeps the DataFusion plan.
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

pub(super) fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    datafusion_cpu_allowed: bool,
    root_sink: &NativeReplacementSink,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let _footers = FooterScope::enter();
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering =
        lower_count_only_physical_plan_to_native(lowering_input, &native_execution_config.source_read_strategy())?;
    replace_lowered_root_or_keep_plan_for_sink(
        plan,
        lowering,
        native_execution_config,
        native_optimizer_options,
        datafusion_cpu_allowed,
        root_sink.clone(),
    )
}

pub(super) fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    datafusion_cpu_allowed: bool,
    root_sink: &NativeReplacementSink,
    optimizer_options: &OptimizerOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let _footers = FooterScope::enter();
    let mut relation_inputs = LoweredRelationInputs::default();
    let plan = lower_gpu_resident_producer_relation_inputs(
        plan,
        native_execution_config,
        native_optimizer_options,
        datafusion_cpu_allowed,
        &mut relation_inputs,
    )?;
    let mut attempt = optimize_final_native_root(
        plan,
        native_execution_config,
        native_optimizer_options,
        datafusion_cpu_allowed,
        root_sink,
        optimizer_options,
    )?;
    if let Some(island) = share_island_subplans(&attempt.plan, native_execution_config)? {
        attempt.plan = island.root;
        attempt.native_optimizer.merge(island.native_optimizer);
        if let Some(root_admission) = island.root_admission
            && let [accepted] = attempt.capability_contract.accepted_candidates.as_slice()
        {
            attempt.capability_contract = accepted_plan_contract_diagnostics(
                accepted.candidate.clone(),
                native_execution_config,
                &root_admission,
            )?;
        }
    }
    // Each lowering selects at most one candidate, so relation input outcomes
    // are appended only after the root and fragment lowerings consumed their
    // own reports.
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
    datafusion_cpu_allowed: bool,
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
            datafusion_cpu_allowed,
            lowered,
        )?;
        if let Some(input_name) = relation_input_names.as_ref().and_then(|names| names.get(index))
            && gpu_resident_producer(rewritten.as_ref()).is_none()
        {
            let lowering =
                lower_physical_plan_to_native(Arc::clone(&rewritten), &native_execution_config.source_read_strategy())?;
            let mut attempt = replace_lowered_root_or_keep_plan_for_sink(
                Arc::clone(&rewritten),
                lowering,
                native_execution_config,
                native_optimizer_options,
                datafusion_cpu_allowed,
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
    datafusion_cpu_allowed: bool,
    root_sink: &NativeReplacementSink,
    optimizer_options: &OptimizerOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan), &native_execution_config.source_read_strategy())?;
    let attempt = replace_lowered_root_or_keep_plan_for_sink(
        plan,
        lowering,
        native_execution_config,
        native_optimizer_options,
        datafusion_cpu_allowed,
        root_sink.clone(),
    )?;
    // Fragments are placed below query roots, whose admission this rule
    // binds; the outer run binds a cuGraph input subplan's admission. A plan
    // that must not run DataFusion operators gains nothing from them.
    if !matches!(root_sink, NativeReplacementSink::HostArrow) || !datafusion_cpu_allowed {
        return Ok(attempt);
    }
    FragmentSearch { native_execution_config, native_optimizer_options, datafusion_cpu_allowed, optimizer_options }
        .below_root(attempt)
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

/// Decides `lowering`'s candidate for `sink` and returns either the native unit
/// that replaces `plan` or `plan` itself with the report of the decline.
///
/// `lowering.plan` being `None` means lowering already reported the decline.
/// The optimizer runs before the ordering check because it can remove or add
/// the operators that reproduce the declared ordering.
pub(super) fn replace_lowered_root_or_keep_plan_for_sink(
    plan: Arc<dyn ExecutionPlan>,
    lowering: NativeLoweringResult,
    native_execution_config: &AlgeonNativeExecutionConfig,
    native_optimizer_options: &NativeOptimizerOptions,
    datafusion_cpu_allowed: bool,
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
        Err(rejection) => {
            let candidate = if ordered_output_sort_needed {
                wrap_candidate_with_declared_output_ordering_sort(candidate_before_declared_output_sort)
            } else {
                candidate_before_declared_output_sort
            };
            let (reason, detail) = rejection.into_detail(None);
            return Ok(NativeOptimizationAttempt {
                plan,
                report: native_execution_rejection_report_for_candidate(candidate, reason, detail, &lowering.report),
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
    // The candidate is decided once, on the sink form execution runs; the cost
    // decision, the boundary check and both contract-diagnostics branches read
    // this decision.
    let native_plan = match &sink {
        NativeReplacementSink::HostArrow => native_plan.clone_with_deferred_host_materialization_sink(),
        NativeReplacementSink::GpuResidentProducer { .. } => native_plan.clone_with_gpu_resident_sink(),
    };
    lowering.report.anchor_selected_candidates_to_plan_node(native_plan.root_id());
    if let Some(reason) = native_plan_cost_decision(&native_plan, datafusion_cpu_allowed) {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_cost_selection_report(&lowering.report, reason),
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
        && let Some(blocker) = floating_statistical_aggregate_input_is_partitioned(plan.as_ref())
    {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: native_execution_rejection_report_for_candidate(
                candidate,
                CapabilityReason::NativeExactStatisticalInputPartitioned,
                NotSupportedDetail { blocker: Some(Box::new(blocker)), ..NotSupportedDetail::default() },
                &lowering.report,
            ),
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        });
    }

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
        plan.properties().equivalence_properties().clone(),
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
    let reason = rejection_capability_reason(rejected_plan, rejection);
    // Lowering accepted every DataFusion operator of the candidate, so naming
    // one for a validation refusal would blame an operator that lowered. The
    // refusal's own rule is the subject; only the join contract points back at
    // a DataFusion join.
    let blocker = match reason {
        CapabilityReason::NativeBoundedJoinUnsupported => first_join_operator(plan.as_ref()),
        _ => None,
    };
    let report = native_execution_rejection_report_for_candidate(
        candidate.clone(),
        reason,
        NotSupportedDetail {
            blocker: blocker.map(Box::new),
            admission: Some(rejection.rejection().clone()),
            ..NotSupportedDetail::default()
        },
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
    // Children are sorted by input name, the order `shared_resident` rebinding
    // also uses, so child order depends on the plan alone.
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
) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Selected { candidate } => {
            CandidateOutcome::NotSelectedByCost { candidate: candidate.clone(), cost_reason }
        },
        CandidateOutcome::NotSupported { candidate, capability_reason, detail } => CandidateOutcome::NotSupported {
            candidate: candidate.clone(),
            capability_reason: *capability_reason,
            detail: detail.clone(),
        },
        CandidateOutcome::NotSelectedByCost { candidate, cost_reason } => {
            CandidateOutcome::NotSelectedByCost { candidate: candidate.clone(), cost_reason: *cost_reason }
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

/// `OutputRequirementExec` exists only during optimization; DataFusion removes
/// it before execution, so the probe lowers what it wraps.
fn peel_output_requirement_exec(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.name() == "OutputRequirementExec"
        && let [child] = plan.children().as_slice()
    {
        return Arc::clone(child);
    }
    plan
}

/// The host leaf unit under a `CooperativeExec`, when that is all the wrapper
/// holds. DataFusion's `EnsureCooperative` rule wraps leaves after the
/// count-only probe selected one, so the final rewrite peels the wrapper to
/// recognize its earlier selection and to root admission directly above the
/// unit.
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
    use datafusion::physical_plan::{empty::EmptyExec, union::UnionExec};
    use query_planning::{
        capability::Reason,
        dynamic_filter::DynamicFilterExecutionOptions,
        expr::{AggregateFunction, Expr, SortExpr},
        optimizer::SemijoinReductionOptions,
        plan::{WindowFunction, WindowSpec},
        source::LiteralBatchSource,
    };

    use super::*;
    use crate::{native::NativeLoweringResult, native_exec::SourceChunking};

    /// Plan validation refuses a cumulative COUNT window, which the engine has
    /// no scan kernel for. Every operator of the DataFusion candidate lowered,
    /// so the report must not blame the union at its root.
    #[test]
    fn validation_refusal_is_not_blamed_on_a_datafusion_operator() {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
        let window = builder
            .add_window(
                source,
                WindowSpec::new(
                    Vec::new(),
                    vec![SortExpr::new(Expr::column("value"), false, false)],
                    vec![WindowFunction::cumulative_aggregate(
                        AggregateFunction::Count,
                        Expr::column("value"),
                        Field::new("running_count", DataType::Int64, false),
                    )],
                ),
            )
            .unwrap();
        let native_plan =
            builder.finish_with_host_sink(window).unwrap().clone_with_deferred_host_materialization_sink();
        let execution_config =
            AlgeonNativeExecutionConfig::new(SourceChunking::try_new(512, 1, 3, 1).expect("valid source chunking"));
        let AdmissionDecision::Rejected(rejection) =
            native_admission_decision(&native_plan, &execution_config).expect("the candidate is decidable")
        else {
            panic!("a cumulative COUNT window is refused by plan validation");
        };
        assert!(rejection.validation_error().is_some());
        let union = UnionExec::try_new(vec![
            Arc::new(EmptyExec::new(Arc::clone(&schema))) as Arc<dyn ExecutionPlan>,
            Arc::new(EmptyExec::new(schema)),
        ])
        .unwrap();
        let lowering =
            NativeLoweringResult { plan: None, report: PlanningReport::new(), source_diagnostics: Vec::new() };

        let attempt = rejected_attempt(
            union,
            GpuCandidate::new("UnionExec", 1),
            (&native_plan, &rejection),
            NativeOptimizerDiagnostics::default(),
            lowering,
            &execution_config,
        )
        .unwrap();

        let [CandidateOutcome::NotSupported { capability_reason, detail, .. }] = attempt.report.outcomes() else {
            panic!("the candidate is not supported: {:?}", attempt.report.outcomes());
        };
        assert_eq!(*capability_reason, CapabilityReason::UnsupportedOperator);
        let blocking = detail.blocking().expect("the refusal names a subject");
        assert_eq!(blocking.kind(), "compiled_contract", "{}", blocking.text());
    }

    #[test]
    fn non_join_candidate_is_rejected_at_planning_with_its_missing_contract() {
        // Row numbers under ORDER BY over a source that records no ordering:
        // no compiled contract covers the window, so planning rejects the
        // candidate and the plan stays on DataFusion with that reason.
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
    fn cost_selection_declines_the_selected_candidate_with_its_reason() {
        let report = PlanningReport::from_outcomes([CandidateOutcome::Selected {
            candidate: GpuCandidate::new("ProjectionExec(PlaceholderRowExec)", 2),
        }]);

        let report = native_execution_cost_selection_report(&report, NativePlanCostReason::NoColumnData);

        assert!(matches!(
            report.outcomes(),
            [CandidateOutcome::NotSelectedByCost { cost_reason: NativePlanCostReason::NoColumnData, .. }]
        ));
        let json = report.to_json();
        assert_eq!(json["outcomes"][0]["outcome"], "not_selected_by_cost");
        assert_eq!(json["outcomes"][0]["cost_reason_code"], "native_no_column_data_not_selected_by_cost");
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

        assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("native_code"), Some("invalid_configuration"));
        assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
        assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
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

        assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("native_code"), Some("invalid_configuration"));
        assert_eq!(error.fact_value("field"), Some("cost_threshold_multiplier"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }
}
