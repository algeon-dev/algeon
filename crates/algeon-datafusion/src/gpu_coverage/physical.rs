//! Final-plan coverage projections and candidate-selection diagnostics.

use datafusion::physical_plan::ExecutionPlan;

use super::{
    model::{
        QueryGpuCandidateOutcome, QueryGpuCandidateReasonKind, QueryGpuCugraphCall, QueryGpuRuntimeCaveat,
        QueryGpuValidatedUnder,
    },
    remedy::{capability_remedy, cost_remedy},
};
#[cfg(feature = "cugraph")]
use crate::cugraph_sql::CugraphAlgoExec;
use crate::{
    capability::CapabilityReason,
    native_report::candidate_shape_family,
    planner::{FinalPlanCoverage, NativePlanningDiagnostics},
    report::{CandidateOutcome, DEFERRED_PRODUCER_BINDING_REASON_CODE, PlanningReport},
};

pub(super) fn cugraph_calls(plan: &dyn ExecutionPlan) -> Vec<QueryGpuCugraphCall> {
    let mut calls = Vec::new();
    collect_cugraph_calls(plan, &mut calls);
    calls
}

#[cfg(feature = "cugraph")]
fn collect_cugraph_calls(plan: &dyn ExecutionPlan, calls: &mut Vec<QueryGpuCugraphCall>) {
    if let Some(cugraph) = plan.downcast_ref::<CugraphAlgoExec>() {
        calls.push(QueryGpuCugraphCall {
            algorithm: cugraph.coverage_algorithm_name().to_string(),
            statically_valid: true,
            notes: vec![
                "physical cuGraph plan constructed".to_string(),
                "runtime seed, personalization, and device-side validation are not decided at plan time".to_string(),
            ],
        });
    }
    for child in plan.children() {
        collect_cugraph_calls(child.as_ref(), calls);
    }
}

#[cfg(not(feature = "cugraph"))]
fn collect_cugraph_calls(_plan: &dyn ExecutionPlan, _calls: &mut Vec<QueryGpuCugraphCall>) {}

pub(super) fn candidate_outcomes_from_report(report: &PlanningReport) -> Vec<QueryGpuCandidateOutcome> {
    report
        .outcomes()
        .iter()
        .map(|outcome| match outcome {
            CandidateOutcome::Selected { candidate } => QueryGpuCandidateOutcome {
                selection: outcome.kind(),
                candidate_family: candidate_shape_family(candidate),
                reason_code: None,
                reason_kind: None,
                detail: candidate.shape.clone(),
                category: None,
                remedy: None,
                cost_evidence: None,
            },
            CandidateOutcome::Deferred { candidate, .. } => QueryGpuCandidateOutcome {
                selection: outcome.kind(),
                candidate_family: candidate_shape_family(candidate),
                reason_code: Some(DEFERRED_PRODUCER_BINDING_REASON_CODE.to_string()),
                reason_kind: Some(QueryGpuCandidateReasonKind::PlanShape),
                detail: candidate.shape.clone(),
                category: Some("producer_binding".to_string()),
                remedy: None,
                cost_evidence: None,
            },
            CandidateOutcome::NotSupported { candidate, capability_reason } => QueryGpuCandidateOutcome {
                selection: outcome.kind(),
                candidate_family: candidate_shape_family(candidate),
                reason_code: Some(capability_reason.code().to_string()),
                reason_kind: Some(capability_reason_kind(*capability_reason)),
                detail: candidate.shape.clone(),
                category: Some(capability_reason.triage_code().as_str().to_string()),
                remedy: Some(capability_remedy(*capability_reason)),
                cost_evidence: None,
            },
            CandidateOutcome::NotSelectedByCost { candidate, cost_reason, evidence } => QueryGpuCandidateOutcome {
                selection: outcome.kind(),
                candidate_family: candidate_shape_family(candidate),
                reason_code: Some(cost_reason.code().to_string()),
                reason_kind: Some(QueryGpuCandidateReasonKind::Cost),
                detail: candidate.shape.clone(),
                category: Some("cost".to_string()),
                remedy: Some(cost_remedy(*cost_reason)),
                cost_evidence: Some(evidence.clone()),
            },
        })
        .collect()
}

fn capability_reason_kind(reason: CapabilityReason) -> QueryGpuCandidateReasonKind {
    match reason {
        CapabilityReason::UnsupportedExpression => QueryGpuCandidateReasonKind::Expression,
        CapabilityReason::UnsupportedType => QueryGpuCandidateReasonKind::ArrowType,
        CapabilityReason::NativeSourceNonLocalObjectStore
        | CapabilityReason::NativeSourceExprAdapter
        | CapabilityReason::NativeSourceParquetFileReaderFactory
        | CapabilityReason::NativeSourcePreserveOrder
        | CapabilityReason::NativeSourceOutputOrdering
        | CapabilityReason::NativeSourcePushedDownParquetPredicate
        | CapabilityReason::NativeSourceEmpty
        | CapabilityReason::NativeSourcePartitionValues
        | CapabilityReason::NativeSourceByteRange
        | CapabilityReason::NativeSourceRowSelection
        | CapabilityReason::NativeSourceSchemaMapping
        | CapabilityReason::TableFormatUnsupported
        | CapabilityReason::TableFormatPlanningFailed
        | CapabilityReason::IcebergNonParquetDataFile
        | CapabilityReason::IcebergDeleteFilesUnsupported
        | CapabilityReason::IcebergFieldIdMappingUnsupported
        | CapabilityReason::IcebergNameMappingUnsupported
        | CapabilityReason::IcebergNestedProjectionUnsupported
        | CapabilityReason::IcebergPartitionTransformUnsupported
        | CapabilityReason::IcebergResidualPredicateUnsupported
        | CapabilityReason::IcebergRowSelectionUnsupported
        | CapabilityReason::IcebergObjectStoreReadUnsupported
        | CapabilityReason::IcebergFileIoSourceAccessUnverified
        | CapabilityReason::IcebergEncryptedFileUnsupported
        | CapabilityReason::IcebergScalarValueUnsupported => QueryGpuCandidateReasonKind::Source,
        CapabilityReason::NativeSortPreservingMergeUnsupported
        | CapabilityReason::NativeDistributedAggregateUnsupported
        | CapabilityReason::NativePartitionPreservingSortUnsupported
        | CapabilityReason::NativeBoundedJoinUnsupported
        | CapabilityReason::NativeOrderedOutputUnsupported => QueryGpuCandidateReasonKind::PlanShape,
        CapabilityReason::UnsupportedOperator => QueryGpuCandidateReasonKind::Operator,
    }
}

pub(super) fn runtime_caveats(
    observation: FinalPlanCoverage,
    cugraph_calls: &[QueryGpuCugraphCall],
    validated_under: &QueryGpuValidatedUnder,
) -> Vec<QueryGpuRuntimeCaveat> {
    if !observation.has_gpu_runtime() {
        return Vec::new();
    }
    let mut caveats = vec![
        QueryGpuRuntimeCaveat::new(
            "runtime_gpu_admission_unknown",
            "query-service grants are acquired at execution time and can wait or fail under concurrency; admitted grants are immutable",
        ),
        QueryGpuRuntimeCaveat::new(
            "device_memory_runtime_unknown",
            "device allocation, reservation, and retry outcomes are data- and concurrency-dependent",
        ),
        QueryGpuRuntimeCaveat::new(
            "native_runtime_failure_has_no_cpu_rescue",
            "once native GPU execution starts, native failures are surfaced as failures rather than CPU fallback",
        ),
    ];
    if observation.host_materialization_count() > 0 {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "host_output_materialization",
            "DataFusion-facing query output is materialized back to host Arrow outside the GPU-execution promise",
        ));
    }
    if !cugraph_calls.is_empty() {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "cugraph_runtime_validation_unknown",
            "cuGraph seed, personalization, weight, and device-side checks can still fail at runtime",
        ));
    }
    if let Some(execution_mode) = &validated_under.execution_mode {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "execution_mode_plan_time_only",
            format!(
                "validated under native pipeline execution mode {execution_mode}; bounded runtime validation still executes later"
            ),
        ));
    }
    caveats
}

pub(super) fn planning_diagnostics_did_io(diagnostics: &NativePlanningDiagnostics) -> bool {
    diagnostics.input_plan_root.contains("AlgeonIcebergScanExec")
        || diagnostics.input_plan_shape.contains("AlgeonIcebergScanExec")
        || diagnostics.report.outcomes().iter().any(|outcome| match outcome {
            CandidateOutcome::NotSupported { capability_reason, .. } => {
                capability_reason_is_table_format(*capability_reason)
            },
            CandidateOutcome::Selected { .. }
            | CandidateOutcome::Deferred { .. }
            | CandidateOutcome::NotSelectedByCost { .. } => false,
        })
}

fn capability_reason_is_table_format(reason: CapabilityReason) -> bool {
    matches!(
        reason,
        CapabilityReason::TableFormatUnsupported
            | CapabilityReason::TableFormatPlanningFailed
            | CapabilityReason::IcebergNonParquetDataFile
            | CapabilityReason::IcebergDeleteFilesUnsupported
            | CapabilityReason::IcebergFieldIdMappingUnsupported
            | CapabilityReason::IcebergNameMappingUnsupported
            | CapabilityReason::IcebergNestedProjectionUnsupported
            | CapabilityReason::IcebergPartitionTransformUnsupported
            | CapabilityReason::IcebergResidualPredicateUnsupported
            | CapabilityReason::IcebergRowSelectionUnsupported
            | CapabilityReason::IcebergObjectStoreReadUnsupported
            | CapabilityReason::IcebergFileIoSourceAccessUnverified
            | CapabilityReason::IcebergEncryptedFileUnsupported
            | CapabilityReason::IcebergScalarValueUnsupported
    )
}
