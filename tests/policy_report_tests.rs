use datafusion_cudf::{CandidateOutcome, FallbackReason, GpuCandidate, GpuFallbackPolicy, PlanningReport};

#[test]
fn gpu_preferred_is_default_development_policy() {
    assert_eq!(GpuFallbackPolicy::default(), GpuFallbackPolicy::GpuPreferred);
}

#[test]
fn fallback_reason_codes_are_stable_manifest_values() {
    let codes = [
        (FallbackReason::UnsupportedOperator, "unsupported_operator"),
        (FallbackReason::UnsupportedExpression, "unsupported_expression"),
        (FallbackReason::UnsupportedType, "unsupported_type"),
        (FallbackReason::InsufficientIslandSize, "insufficient_island_size"),
        (FallbackReason::PlanPropertyMismatch, "plan_property_mismatch"),
        (FallbackReason::ExplicitCpuPolicy, "explicit_cpu_policy"),
    ];

    for (reason, expected) in codes {
        assert_eq!(reason.code(), expected);
    }
}

#[test]
fn planning_report_keeps_rewritten_and_rejected_candidates_in_order() {
    let mut report = PlanningReport::new();
    let filter = GpuCandidate::new("FilterExec", 2);
    let projection_filter = GpuCandidate::new("ProjectionExec(FilterExec(child))", 3);

    report.push_rejected(filter.clone(), FallbackReason::InsufficientIslandSize);
    report.push_rewritten(projection_filter.clone());

    assert_eq!(report.rejected_count(), 1);
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(
        report.outcomes(),
        &[
            CandidateOutcome::Rejected { candidate: filter, reason: FallbackReason::InsufficientIslandSize },
            CandidateOutcome::Rewritten { candidate: projection_filter },
        ],
    );
}
