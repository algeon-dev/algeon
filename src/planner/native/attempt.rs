use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    physical_expr::EquivalenceProperties,
    physical_plan::ExecutionPlan,
};

use super::{
    NativeCapabilityContractDiagnostics,
    capability_envelope::{
        accepted_plan_contract_diagnostics, capability_envelope_admission_is_rejected,
        rejected_plan_contract_diagnostics,
    },
};
use crate::{
    capability::CapabilityReason,
    error::{Error, ErrorCode},
    native_exec::{
        NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig,
        producer::{gpu_resident_producer_input_name, rewrite_gpu_resident_producer_for_subsumed_composed_consumer},
    },
    report::{CandidateOutcome, GpuCandidate, NativeCandidateCostEvidence, PlanningReport},
};

/// Optimizer attempt output, including the replacement plan and all report evidence.
#[derive(Debug, Clone)]
pub(super) struct NativeOptimizationAttempt {
    pub(super) plan: Arc<dyn ExecutionPlan>,
    pub(super) report: PlanningReport,
    pub(super) native_optimizer: nexus_query_engine::optimizer::NativeOptimizerDiagnostics,
    pub(super) source_diagnostics: Vec<nexus_query_engine::source::SourceDiagnostics>,
    pub(super) capability_contract: NativeCapabilityContractDiagnostics,
    pub(super) rewritten: bool,
}

pub(super) fn optimize_count_only_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NexusNativeExecutionConfig,
    native_optimizer_options: &nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_plan_cost_options: &nexus_query_engine::plan::NativePlanCostOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering_input = peel_output_requirement_exec(Arc::clone(&plan));
    let lowering = crate::native::lower_count_only_physical_plan_to_native(lowering_input);
    replace_lowered_root_or_keep_plan(
        plan,
        None,
        lowering,
        native_execution_config,
        native_optimizer_options,
        native_plan_cost_options,
    )
}

pub(super) fn optimize_final_native_plan(
    plan: Arc<dyn ExecutionPlan>,
    native_execution_config: &NexusNativeExecutionConfig,
    native_optimizer_options: &nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_plan_cost_options: &nexus_query_engine::plan::NativePlanCostOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let lowering = crate::native::lower_physical_plan_to_native(Arc::clone(&plan));
    if whole_plan_has_unsupported_expression_miss(&lowering)
        && let Some(boundary) = super::mixed::CertifiedMixedFilterBoundary::certify(Arc::clone(&plan))
    {
        let native_leaf_lowering = crate::native::lower_physical_plan_to_native(Arc::clone(boundary.source()));
        if native_leaf_lowering.plans.len() == 1
            && native_leaf_lowering.selected_count() == 1
            && boundary.accepts_native_leaf(&native_leaf_lowering.plans[0])
        {
            let mut native_leaf = replace_lowered_root_or_keep_plan(
                Arc::clone(boundary.source()),
                Some(boundary.native_equivalence_properties()),
                native_leaf_lowering,
                native_execution_config,
                native_optimizer_options,
                native_plan_cost_options,
            )?;
            if native_leaf.rewritten
                && let Some(plan) = boundary.rebuild(Arc::clone(&native_leaf.plan))
            {
                native_leaf.plan = plan;
                return Ok(native_leaf);
            }
        }
    }
    replace_lowered_root_or_keep_plan(
        plan,
        None,
        lowering,
        native_execution_config,
        native_optimizer_options,
        native_plan_cost_options,
    )
}

fn replace_lowered_root_or_keep_plan(
    plan: Arc<dyn ExecutionPlan>,
    output_equivalence_properties: Option<EquivalenceProperties>,
    lowering: crate::native::NativeLoweringResult,
    native_execution_config: &NexusNativeExecutionConfig,
    native_optimizer_options: &nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_plan_cost_options: &nexus_query_engine::plan::NativePlanCostOptions,
) -> DataFusionResult<NativeOptimizationAttempt> {
    let mut lowering = lowering;
    stamp_planning_report_correlation(&mut lowering.report, native_execution_config);
    if lowering.plans.len() != 1 || lowering.selected_count() != 1 {
        return Ok(NativeOptimizationAttempt {
            plan,
            report: lowering.report,
            native_optimizer: nexus_query_engine::optimizer::NativeOptimizerDiagnostics::default(),
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: NativeCapabilityContractDiagnostics::default(),
            rewritten: false,
        });
    }
    lowering.report.anchor_selected_candidates_to_plan_node(lowering.plans[0].root_id());
    let candidate_before_declared_output_sort = lowering
        .report
        .single_selected_candidate()
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;
    let optimizer_options = native_optimizer_options_for_execution(native_optimizer_options, native_execution_config)
        .map_err(DataFusionError::from)?;
    let optimizer_result =
        nexus_query_engine::optimizer::optimize_query_plan(lowering.plans[0].clone(), &optimizer_options);
    let native_optimizer = optimizer_result.diagnostics;
    let ordered_output_sort_needed = plan.properties().output_ordering().is_some()
        && !crate::native::native_plan_reproduces_declared_output_ordering(&optimizer_result.plan);
    let (native_plan, appended_declared_output_sort) =
        match crate::native::reestablish_declared_output_ordering_if_needed(
            optimizer_result.plan,
            plan.properties().output_ordering(),
        ) {
            Ok(native_plan) => native_plan,
            Err(reason) => {
                let candidate = if ordered_output_sort_needed {
                    crate::native::wrap_candidate_with_declared_output_ordering_sort(
                        candidate_before_declared_output_sort,
                    )
                } else {
                    candidate_before_declared_output_sort
                };
                return Ok(NativeOptimizationAttempt {
                    plan,
                    report: native_execution_rejection_report_for_candidate(candidate, reason, &lowering.report),
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
    let candidate = lowering
        .report
        .single_selected_candidate()
        .cloned()
        .ok_or_else(|| native_candidate_error("native rewrite has no rewritten candidate"))?;
    if let nexus_query_engine::plan::NativePlanCostDecision::NotSelectedByCost { reason, facts } =
        nexus_query_engine::plan::native_plan_cost_decision(&native_plan, native_plan_cost_options)
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

    if native_execution_boundary_is_rejected(&native_plan, native_execution_config)? {
        let report = native_execution_rejection_report_for_candidate(
            candidate.clone(),
            CapabilityReason::NativeBoundedJoinUnsupported,
            &lowering.report,
        );
        return Ok(NativeOptimizationAttempt {
            plan,
            report,
            native_optimizer,
            source_diagnostics: lowering.source_diagnostics,
            capability_contract: rejected_plan_contract_diagnostics(&native_plan, candidate, native_execution_config)?,
            rewritten: false,
        });
    }

    let datafusion_shape = candidate.shape.clone();
    let capability_contract = accepted_plan_contract_diagnostics(&native_plan, candidate, native_execution_config)?;
    let plan = native_datafusion_exec_for_plan(
        Arc::clone(&plan),
        native_plan,
        native_execution_config,
        &datafusion_shape,
        output_equivalence_properties.unwrap_or_else(|| plan.properties().equivalence_properties().clone()),
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

fn whole_plan_has_unsupported_expression_miss(lowering: &crate::native::NativeLoweringResult) -> bool {
    lowering.plans.is_empty()
        && lowering.selected_count() == 0
        && matches!(
            lowering.report.outcomes(),
            [CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }]
        )
}

fn native_optimizer_options_for_execution(
    options: &nexus_query_engine::optimizer::NativeOptimizerOptions,
    native_execution_config: &NexusNativeExecutionConfig,
) -> crate::error::Result<nexus_query_engine::optimizer::NativeOptimizerOptions> {
    native_execution_config.validate_query_policies()?;
    let mut options = options.clone();
    options.semijoin_reduction.admission_mode = nexus_query_engine::optimizer::SemijoinReductionAdmissionMode::Bounded;
    options.dynamic_filter_route_discovery.execution = native_execution_config.dynamic_filter_execution_options();
    options.keyed_hash_join_strategy = native_execution_config.keyed_hash_join_strategy_options();
    options.validate().map_err(|error| {
        Error::from_native(
            "datafusion_nexus.native_exec.config.validate",
            "invalid native optimizer option policy",
            stamp_native_error_correlation(error, native_execution_config),
        )
    })?;
    Ok(options)
}

fn native_datafusion_exec_for_plan(
    original_plan: Arc<dyn ExecutionPlan>,
    native_plan: nexus_query_engine::plan::QueryPlan,
    native_execution_config: &NexusNativeExecutionConfig,
    datafusion_shape: &str,
    eq_properties: EquivalenceProperties,
) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    let native_execution_config =
        native_execution_config.clone().with_plan_task_mapping_datafusion_shape(datafusion_shape);
    let gpu_input_names = native_plan_gpu_resident_source_names(&native_plan);
    if gpu_input_names.is_empty() {
        return NexusNativeExec::try_new_with_equivalence_properties(
            native_plan,
            native_execution_config,
            eq_properties,
        )
        .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
        .map_err(DataFusionError::from);
    }

    let gpu_inputs = gpu_resident_producer_children(&original_plan, &gpu_input_names)?;
    let (native_plan, gpu_inputs) = remove_subsumed_composed_consumer(native_plan, gpu_inputs)?;
    NexusNativeComposedExec::try_new_with_equivalence_properties(
        native_plan,
        gpu_inputs,
        native_execution_config,
        eq_properties,
    )
    .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>)
    .map_err(DataFusionError::from)
}

fn remove_subsumed_composed_consumer(
    plan: nexus_query_engine::plan::QueryPlan,
    mut gpu_inputs: Vec<Arc<dyn ExecutionPlan>>,
) -> DataFusionResult<(nexus_query_engine::plan::QueryPlan, Vec<Arc<dyn ExecutionPlan>>)> {
    let [producer] = gpu_inputs.as_slice() else {
        return Ok((plan, gpu_inputs));
    };
    let Some(rewritten_producer) =
        rewrite_gpu_resident_producer_for_subsumed_composed_consumer(producer.as_ref(), &plan)?
    else {
        return Ok((plan, gpu_inputs));
    };
    gpu_inputs[0] = rewritten_producer;
    let source = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            nexus_query_engine::plan::PlanNode::Source(source) => Some(source.as_ref().clone()),
            _ => None,
        })
        .ok_or_else(|| {
            DataFusionError::from(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.native_optimizer.subsumed_consumer",
                "a subsumed composed consumer is missing its GPU-resident source",
            ))
        })?;
    let mut builder = nexus_query_engine::plan::QueryPlanBuilder::new();
    let source = builder.add_source(source);
    Ok((builder.finish_with_host_sink(source), gpu_inputs))
}

fn native_plan_gpu_resident_source_names(plan: &nexus_query_engine::plan::QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            nexus_query_engine::plan::PlanNode::Source(source) => match source.as_ref() {
                nexus_query_engine::source::SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                nexus_query_engine::source::SourceNode::LocalParquet(_) => None,
                nexus_query_engine::source::SourceNode::LiteralBatch(_) => None,
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

fn native_execution_boundary_is_rejected(
    plan: &nexus_query_engine::plan::QueryPlan,
    execution_config: &NexusNativeExecutionConfig,
) -> DataFusionResult<bool> {
    if !execution_config.memory_control_policy().is_bounded() || !plan.contains_join() {
        return Ok(false);
    }
    capability_envelope_admission_is_rejected(plan, execution_config)
}

fn native_execution_cost_selection_report(
    report: &PlanningReport,
    cost_reason: nexus_query_engine::plan::NativePlanCostReason,
    evidence: NativeCandidateCostEvidence,
) -> PlanningReport {
    PlanningReport::from_outcomes(report.outcomes().iter().map(|outcome| match outcome {
        CandidateOutcome::Selected { candidate } => CandidateOutcome::NotSelectedByCost {
            candidate: candidate.clone(),
            cost_reason,
            evidence: evidence.clone(),
        },
        CandidateOutcome::NotSupported { candidate, capability_reason } => {
            CandidateOutcome::NotSupported { candidate: candidate.clone(), capability_reason: *capability_reason }
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
    .with_unsupported_columns_from(report)
}

fn native_execution_rejection_report_for_candidate(
    candidate: GpuCandidate,
    rejection_reason: CapabilityReason,
    source_report: &PlanningReport,
) -> PlanningReport {
    PlanningReport::from_outcomes([CandidateOutcome::NotSupported { candidate, capability_reason: rejection_reason }])
        .with_correlation_from(source_report)
}

fn stamp_planning_report_correlation(
    report: &mut PlanningReport,
    native_execution_config: &NexusNativeExecutionConfig,
) {
    if let Some(query_id) = native_execution_config.query_id() {
        report.set_query_id(query_id.clone());
    }
    if let Some(trace_id) = native_execution_config.trace_id() {
        report.set_trace_id(trace_id.clone());
    }
}

fn stamp_native_error_correlation(
    mut error: nexus_query_engine::error::QueryError,
    native_execution_config: &NexusNativeExecutionConfig,
) -> nexus_query_engine::error::QueryError {
    if let Some(query_id) = native_execution_config.query_id() {
        error = error.with_query_id(query_id.clone());
    }
    if let Some(trace_id) = native_execution_config.trace_id() {
        error = error.with_trace_id(trace_id.clone());
    }
    error
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
    child.downcast_ref::<NexusNativeExec>().is_some().then(|| Arc::clone(child))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_selection_preserves_full_cost_evidence() {
        let facts = nexus_query_engine::plan::NativePlanCostFacts {
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
            nexus_query_engine::plan::NativePlanCostReason::SmallSelectiveTopK,
            NativeCandidateCostEvidence::from_cost_facts(facts.clone()),
        );

        assert!(matches!(
            report.outcomes(),
            [CandidateOutcome::NotSelectedByCost { cost_reason, evidence, .. }]
                if *cost_reason == nexus_query_engine::plan::NativePlanCostReason::SmallSelectiveTopK
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
    fn optimizer_options_for_execution_copy_dynamic_filter_policy_from_execution_config() {
        let dynamic_filter_options =
            nexus_query_engine::dynamic_filter::DynamicFilterExecutionOptions::default().with_max_build_rows(10_000);
        let keyed_hash_join_strategy_options = nexus_query_engine::capability::KeyedHashJoinStrategyOptions::default()
            .with_small_reusable_hash_build_row_threshold(11);
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
            .with_dynamic_filter_execution_options(dynamic_filter_options)
            .with_keyed_hash_join_strategy_options(keyed_hash_join_strategy_options);

        let options = native_optimizer_options_for_execution(
            &nexus_query_engine::optimizer::NativeOptimizerOptions::disabled(),
            &execution_config,
        )
        .expect("valid execution policy should produce optimizer options");

        assert_eq!(options.dynamic_filter_route_discovery.execution, dynamic_filter_options);
        assert_eq!(options.keyed_hash_join_strategy, keyed_hash_join_strategy_options);
        assert_eq!(
            options.semijoin_reduction.admission_mode,
            nexus_query_engine::optimizer::SemijoinReductionAdmissionMode::Bounded
        );
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_expert_query_policy() {
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
            .with_dynamic_filter_execution_options(
                nexus_query_engine::dynamic_filter::DynamicFilterExecutionOptions::default()
                    .with_max_build_rows(10)
                    .with_large_local_parquet_range_producer_rows(11),
            );

        let error = native_optimizer_options_for_execution(
            &nexus_query_engine::optimizer::NativeOptimizerOptions::disabled(),
            &execution_config,
        )
        .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("large_local_parquet_range_producer_rows"));
        assert_eq!(error.fact_value("reason"), Some("range_producer_rows_exceeds_runtime_build_cap"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_keyed_hash_join_policy() {
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
            .with_keyed_hash_join_strategy_options(
                nexus_query_engine::capability::KeyedHashJoinStrategyOptions::default()
                    .with_large_keyed_hash_build_row_threshold(10),
            );

        let error = native_optimizer_options_for_execution(
            &nexus_query_engine::optimizer::NativeOptimizerOptions::disabled(),
            &execution_config,
        )
        .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("large_field"), Some("large_keyed_hash_build_row_threshold"));
        assert_eq!(error.fact_value("reason"), Some("large_threshold_below_small_threshold"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_aggregate_strategy_policy() {
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
            .with_aggregate_strategy_execution_options(
                nexus_query_engine::capability::AggregateStrategyExecutionOptions::default()
                    .with_encoded_key_partial_low_cardinality_threshold_ppm(1_000_001),
            );

        let error = native_optimizer_options_for_execution(
            &nexus_query_engine::optimizer::NativeOptimizerOptions::disabled(),
            &execution_config,
        )
        .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("encoded_key_partial_low_cardinality_threshold_ppm"));
        assert_eq!(error.fact_value("reason"), Some("ratio_ppm_above_one_hundred_percent"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_source_chunk_planning_policy() {
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1)
            .with_source_chunk_planning_options(
                nexus_query_engine::chunk::SourceChunkPlanningOptions::default()
                    .with_local_parquet_throughput_min_device_budget_bytes(0),
            );
        let error = native_optimizer_options_for_execution(
            &nexus_query_engine::optimizer::NativeOptimizerOptions::disabled(),
            &execution_config,
        )
        .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("local_parquet_throughput_min_device_budget_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn optimizer_options_for_execution_rejects_invalid_optimizer_policy() {
        let optimizer_options = nexus_query_engine::optimizer::NativeOptimizerOptions {
            semijoin_reduction: nexus_query_engine::optimizer::SemijoinReductionOptions {
                cost_threshold_multiplier: 0,
                ..Default::default()
            },
            ..nexus_query_engine::optimizer::NativeOptimizerOptions::disabled()
        };
        let execution_config = NexusNativeExecutionConfig::bounded(1024, 512, 1, 3, 1);

        let error = native_optimizer_options_for_execution(&optimizer_options, &execution_config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
        assert_eq!(error.fact_value("native_code"), Some("invalid_plan"));
        assert_eq!(error.fact_value("field"), Some("cost_threshold_multiplier"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }
}
