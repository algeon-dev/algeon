//! Physical-plan GPU coverage walk and report conversion.

use datafusion::physical_plan::ExecutionPlan;

use super::model::{
    QueryGpuCoverageOverall, QueryGpuCugraphCall, QueryGpuFallbackReason, QueryGpuFallbackReasonKind,
    QueryGpuRuntimeCaveat, QueryGpuValidatedUnder,
};
#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphAlgoExec;
use crate::{
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeGpuProducerExec},
    policy::{FallbackReason, GpuFallbackPolicy},
    report::{CandidateOutcome, PlanningReport},
};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct QueryGpuCoverageWalk {
    pub(super) has_gpu: bool,
    pub(super) all_gpu_covered: bool,
    pub(super) gpu_fragment_count: usize,
    pub(super) cpu_boundary_count: usize,
    pub(super) host_output_materialization: bool,
    pub(super) cugraph_present: bool,
    pub(super) cugraph_calls: Vec<QueryGpuCugraphCall>,
}

impl QueryGpuCoverageWalk {
    fn cpu_only() -> Self {
        Self { all_gpu_covered: true, ..Self::default() }
    }

    fn merge_child(&mut self, child: QueryGpuCoverageWalk) {
        self.has_gpu |= child.has_gpu;
        self.all_gpu_covered &= child.all_gpu_covered;
        self.gpu_fragment_count += child.gpu_fragment_count;
        self.cpu_boundary_count += child.cpu_boundary_count;
        self.host_output_materialization |= child.host_output_materialization;
        self.cugraph_present |= child.cugraph_present;
        self.cugraph_calls.extend(child.cugraph_calls);
    }
}

pub(super) fn walk_gpu_coverage(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    if plan.downcast_ref::<NexusNativeExec>().is_some() {
        return QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
    }
    if plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some() {
        return QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: false,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
    }
    if plan.downcast_ref::<NexusNativeComposedExec>().is_some() {
        let mut walk = QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: false,
            cugraph_calls: Vec::new(),
        };
        for child in plan.children() {
            let child_walk = walk_gpu_coverage(child.as_ref());
            walk.merge_child(child_walk);
        }
        return walk;
    }
    walk_cugraph_or_generic(plan)
}

#[cfg(feature = "cugraph")]
fn walk_cugraph_or_generic(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    if let Some(cugraph) = plan.downcast_ref::<CugraphAlgoExec>() {
        let mut walk = QueryGpuCoverageWalk {
            has_gpu: true,
            all_gpu_covered: true,
            gpu_fragment_count: 1,
            cpu_boundary_count: 0,
            host_output_materialization: true,
            cugraph_present: true,
            cugraph_calls: vec![QueryGpuCugraphCall {
                algorithm: cugraph.coverage_algorithm_name().to_string(),
                statically_valid: true,
                notes: vec![
                    "physical cuGraph plan constructed".to_string(),
                    "runtime seed, personalization, and device-side validation are not decided at plan time"
                        .to_string(),
                ],
            }],
        };
        for child in plan.children() {
            let child_walk = walk_gpu_coverage(child.as_ref());
            if !cugraph_child_stays_gpu_resident(child.as_ref()) {
                walk.cpu_boundary_count += 1;
                walk.all_gpu_covered = false;
            }
            walk.merge_child(child_walk);
        }
        return walk;
    }
    walk_generic_datafusion(plan)
}

#[cfg(not(feature = "cugraph"))]
fn walk_cugraph_or_generic(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    walk_generic_datafusion(plan)
}

fn walk_generic_datafusion(plan: &dyn ExecutionPlan) -> QueryGpuCoverageWalk {
    let mut walk = QueryGpuCoverageWalk::cpu_only();
    let mut direct_gpu_child = false;
    for child in plan.children() {
        let child_walk = walk_gpu_coverage(child.as_ref());
        direct_gpu_child |= child_walk.has_gpu;
        walk.merge_child(child_walk);
    }
    if walk.has_gpu {
        walk.all_gpu_covered = false;
        if direct_gpu_child {
            walk.cpu_boundary_count += 1;
        }
    }
    walk
}

#[cfg(feature = "cugraph")]
pub(super) fn cugraph_child_stays_gpu_resident(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<NexusNativeExec>().is_some() || plan.downcast_ref::<NexusNativeComposedExec>().is_some() {
        return true;
    }
    if plan.name() == "CooperativeExec" {
        let children = plan.children();
        return matches!(
            children.as_slice(),
            [child]
                if child.downcast_ref::<NexusNativeExec>().is_some()
                    || child.downcast_ref::<NexusNativeComposedExec>().is_some()
        );
    }
    false
}

pub(super) fn root_is_gpu_runtime_node(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some()
        || root_is_cugraph_node(plan)
}

#[cfg(feature = "cugraph")]
fn root_is_cugraph_node(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<CugraphAlgoExec>().is_some()
}

#[cfg(not(feature = "cugraph"))]
fn root_is_cugraph_node(_plan: &dyn ExecutionPlan) -> bool {
    false
}

pub(super) fn fallback_reasons_from_report(report: &PlanningReport) -> Vec<QueryGpuFallbackReason> {
    report
        .outcomes()
        .iter()
        .filter_map(|outcome| match outcome {
            CandidateOutcome::Rejected { candidate, reason } => Some(QueryGpuFallbackReason {
                reason_code: reason.code().to_string(),
                kind: fallback_reason_kind(*reason),
                detail: candidate.shape.clone(),
                category: reason.triage_code().as_str().to_string(),
            }),
            CandidateOutcome::Rewritten { .. } => None,
        })
        .collect()
}

fn fallback_reason_kind(reason: FallbackReason) -> QueryGpuFallbackReasonKind {
    match reason {
        FallbackReason::UnsupportedExpression => QueryGpuFallbackReasonKind::Expression,
        FallbackReason::UnsupportedType => QueryGpuFallbackReasonKind::ArrowType,
        FallbackReason::ExplicitCpuPolicy => QueryGpuFallbackReasonKind::Policy,
        FallbackReason::NativeSmallSelectiveTopkCpuPreferred
        | FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred
        | FallbackReason::NativeSmallSelectiveFullSortCpuPreferred => QueryGpuFallbackReasonKind::Preference,
        FallbackReason::NativeSourceNonLocalObjectStore
        | FallbackReason::NativeSourcePartitionColumns
        | FallbackReason::NativeSourceExprAdapter
        | FallbackReason::NativeSourceParquetFileReaderFactory
        | FallbackReason::NativeSourceScanLimit
        | FallbackReason::NativeSourcePreserveOrder
        | FallbackReason::NativeSourceOutputOrdering
        | FallbackReason::NativeSourcePushedDownParquetPredicate
        | FallbackReason::NativeSourceEmpty
        | FallbackReason::NativeSourcePartitionValues
        | FallbackReason::NativeSourceByteRange
        | FallbackReason::NativeSourceFileOrdering
        | FallbackReason::NativeSourceRowSelection
        | FallbackReason::NativeSourceSchemaMapping
        | FallbackReason::TableFormatUnsupported
        | FallbackReason::TableFormatPlanningFailed
        | FallbackReason::IcebergNonParquetDataFile
        | FallbackReason::IcebergDeleteFilesUnsupported
        | FallbackReason::IcebergFieldIdMappingUnsupported
        | FallbackReason::IcebergNameMappingUnsupported
        | FallbackReason::IcebergNestedProjectionUnsupported
        | FallbackReason::IcebergPartitionTransformUnsupported
        | FallbackReason::IcebergResidualPredicateUnsupported
        | FallbackReason::IcebergRowSelectionUnsupported
        | FallbackReason::IcebergObjectStoreReadUnsupported
        | FallbackReason::IcebergSnapshotUnsupported => QueryGpuFallbackReasonKind::Source,
        FallbackReason::NativeSortPreservingMergeUnsupported
        | FallbackReason::NativeDistributedAggregateUnsupported
        | FallbackReason::NativePartitionPreservingSortUnsupported
        | FallbackReason::NativeBoundedJoinUnsupported
        | FallbackReason::NativeAdaptivePipelineUnsupported
        | FallbackReason::NativeOrderedOutputUnsupported
        | FallbackReason::TableFormatGpuOnlyLeafNotRewritten => QueryGpuFallbackReasonKind::PlanShape,
        FallbackReason::UnsupportedOperator => QueryGpuFallbackReasonKind::Operator,
    }
}

pub(super) fn runtime_caveats(
    walk: &QueryGpuCoverageWalk,
    overall: &QueryGpuCoverageOverall,
    validated_under: &QueryGpuValidatedUnder,
) -> Vec<QueryGpuRuntimeCaveat> {
    if !walk.has_gpu {
        return runtime_caveats_for_overall(overall, validated_under, Vec::new());
    }
    let mut caveats = vec![
        QueryGpuRuntimeCaveat::new(
            "runtime_gpu_admission_unknown",
            "query-service grants are acquired at execution time and can wait, shrink, or fail under concurrency",
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
    if walk.host_output_materialization {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "host_output_materialization",
            "DataFusion-facing query output is materialized back to host Arrow outside the GPU-execution promise",
        ));
    }
    if walk.cugraph_present {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "cugraph_runtime_validation_unknown",
            "cuGraph seed, personalization, weight, and device-side checks can still fail at runtime",
        ));
    }
    caveats.push(QueryGpuRuntimeCaveat::new(
        "execution_mode_plan_time_only",
        format!(
            "validated under native pipeline execution mode {}; strict/adaptive runtime gates still execute later",
            validated_under.execution_mode
        ),
    ));
    runtime_caveats_for_overall(overall, validated_under, caveats)
}

pub(super) fn runtime_caveats_for_overall(
    overall: &QueryGpuCoverageOverall,
    validated_under: &QueryGpuValidatedUnder,
    mut caveats: Vec<QueryGpuRuntimeCaveat>,
) -> Vec<QueryGpuRuntimeCaveat> {
    if validated_under.actual_policy == GpuFallbackPolicy::GpuOnly.to_string()
        && !matches!(overall, QueryGpuCoverageOverall::NoCpuInPlannedPath)
    {
        caveats.push(QueryGpuRuntimeCaveat::new(
            "gpu_only_policy_would_error",
            "the caller's GpuOnly policy treats non-full-GPU validation outcomes as hard planning failures; validation plans diagnostically under GpuPreferred to retain fallback evidence",
        ));
    }
    caveats
}

pub(super) fn planning_diagnostics_did_io(diagnostics: &crate::planner::NativePlanningDiagnostics) -> bool {
    diagnostics.input_plan_root.contains("NexusTableFormatScanExec")
        || diagnostics.input_plan_shape.contains("NexusTableFormatScanExec")
        || diagnostics.report.outcomes().iter().any(|outcome| match outcome {
            CandidateOutcome::Rejected { reason, .. } => fallback_reason_is_table_format(*reason),
            CandidateOutcome::Rewritten { .. } => false,
        })
}

fn fallback_reason_is_table_format(reason: FallbackReason) -> bool {
    matches!(
        reason,
        FallbackReason::TableFormatUnsupported
            | FallbackReason::TableFormatPlanningFailed
            | FallbackReason::IcebergNonParquetDataFile
            | FallbackReason::IcebergDeleteFilesUnsupported
            | FallbackReason::IcebergFieldIdMappingUnsupported
            | FallbackReason::IcebergNameMappingUnsupported
            | FallbackReason::IcebergNestedProjectionUnsupported
            | FallbackReason::IcebergPartitionTransformUnsupported
            | FallbackReason::IcebergResidualPredicateUnsupported
            | FallbackReason::IcebergRowSelectionUnsupported
            | FallbackReason::IcebergObjectStoreReadUnsupported
            | FallbackReason::IcebergSnapshotUnsupported
            | FallbackReason::TableFormatGpuOnlyLeafNotRewritten
    )
}
