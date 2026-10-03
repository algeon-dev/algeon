//! Flight SQL native-planning diagnostics surface.
//!
//! The retained native optimizer rule keeps the stable
//! [`crate::server::report::PlanningReport`] for "why was this not native?". This module
//! logs that evidence without adding a public reporting API:
//!
//!   selection counts plus the first unsupported-capability class.
//! - [`log_native_planning_report_for_tables`] also emits a warn line for
//!   data-layout-fixable capability misses, with the referenced table list and reason.
//! - [`FlightSqlNativeNotSupportedClass`], read from [`CapabilityReason::facts`], buckets
//!   common capability-miss codes so log searches do not need to enumerate every variant.

use crate::server::{
    capability::{CapabilityReason, FlightSqlNativeNotSupportedClass},
    report::{CandidateOutcome, DataLayoutFixableNotSupported, PlanningReport},
};

/// Emit the native planning diagnostics for a Flight SQL query, including a
/// WARN line when the first relevant capability miss is fixable by changing table
/// layout metadata.
pub fn log_native_planning_report_for_tables(query_id: &str, report: Option<&PlanningReport>, tables: &str) {
    let Some(report) = report else {
        return;
    };
    if report.outcomes().is_empty() {
        return;
    }

    let first_not_supported = first_not_supported_reason(report);
    let not_supported_class = first_not_supported.map(|reason| reason.facts().flight_class);
    tracing::debug!(
        query_id,
        selected_count = report.selected_count(),
        not_selected_count = report.not_selected_count(),
        native_rule_first_not_supported_context = first_not_supported.map(CapabilityReason::code).unwrap_or(""),
        not_supported_class = not_supported_class.map(FlightSqlNativeNotSupportedClass::as_str).unwrap_or(""),
        "algeon-datafusion native planning report"
    );

    if let Some(fields) = data_layout_fixable_planning_event_fields(report, tables) {
        tracing::warn!(
            query_id,
            tables = fields.tables,
            reason_code = fields.reason_code,
            layout_property = fields.layout_property,
            native_candidate_shape = fields.native_candidate_shape,
            outcome_index = fields.outcome_index,
            "algeon-datafusion native planning found a data-layout-fixable capability miss"
        );
    }
}

/// Structured field set for the data-layout-fixable native-planning warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DataLayoutFixablePlanningEventFields<'a> {
    pub tables: &'a str,
    pub reason_code: &'static str,
    pub layout_property: &'static str,
    pub native_candidate_shape: &'a str,
    pub outcome_index: usize,
}

pub(crate) fn data_layout_fixable_planning_event_fields<'a>(
    report: &'a PlanningReport,
    tables: &'a str,
) -> Option<DataLayoutFixablePlanningEventFields<'a>> {
    let DataLayoutFixableNotSupported { outcome_index, candidate, capability_reason, property } =
        report.data_layout_fixable_not_supported()?;
    Some(DataLayoutFixablePlanningEventFields {
        tables,
        reason_code: capability_reason.code(),
        layout_property: property,
        native_candidate_shape: candidate.shape.as_str(),
        outcome_index,
    })
}

fn first_not_supported_reason(report: &PlanningReport) -> Option<CapabilityReason> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::NotSupported { capability_reason, .. } => Some(*capability_reason),
        CandidateOutcome::Selected { .. } | CandidateOutcome::NotSelectedByCost { .. } => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::report::{GpuCandidate, PlanningReport};

    #[test]
    fn class_labels_are_stable_strings() {
        assert_eq!(
            FlightSqlNativeNotSupportedClass::NativeSortPreservingMerge.as_str(),
            "native_sort_preserving_merge_unsupported"
        );
        assert_eq!(FlightSqlNativeNotSupportedClass::Other.as_str(), "other");
    }

    #[test]
    fn first_not_supported_reason_returns_first_not_supported_outcome() {
        let mut report = PlanningReport::new();
        report.push_selected(GpuCandidate::new("AggregateExec", 2));
        report.push_not_supported(
            GpuCandidate::new("SortPreservingMergeExec", 2),
            CapabilityReason::NativeSortPreservingMergeUnsupported,
        );
        report.push_not_supported(GpuCandidate::new("AggregateExec", 2), CapabilityReason::UnsupportedExpression);

        assert_eq!(first_not_supported_reason(&report), Some(CapabilityReason::NativeSortPreservingMergeUnsupported));
    }

    #[test]
    fn first_not_supported_reason_is_none_when_all_selected() {
        let mut report = PlanningReport::new();
        report.push_selected(GpuCandidate::new("AggregateExec", 2));
        assert_eq!(first_not_supported_reason(&report), None);
    }

    #[test]
    fn log_native_planning_report_is_noop_for_empty_inputs() {
        // The helper unconditionally checks emptiness before logging, so we
        // only need to assert it does not panic for the no-op paths.
        log_native_planning_report_for_tables("q-empty", None, "");
        log_native_planning_report_for_tables("q-empty", Some(&PlanningReport::new()), "");
    }

    #[test]
    fn data_layout_fixable_planning_event_fields_pin_warn_payload() {
        let mut report = PlanningReport::new();
        report.push_not_supported(
            GpuCandidate::new("AggregateExec(LocalParquetScan)", 2),
            CapabilityReason::UnsupportedType,
        );
        report.push_not_supported(
            GpuCandidate::new("AggregateExec(LocalParquetScan)", 2),
            CapabilityReason::NativeOrderedOutputUnsupported,
        );

        let fields = data_layout_fixable_planning_event_fields(&report, "public.sorted_lineitem")
            .expect("ordered-output capability miss should be classified as data-layout-fixable");

        assert_eq!(fields.tables, "public.sorted_lineitem");
        assert_eq!(fields.reason_code, "native_ordered_output_unsupported");
        assert_eq!(fields.layout_property, "declared_output_ordering");
        assert_eq!(fields.native_candidate_shape, "AggregateExec(LocalParquetScan)");
        assert_eq!(fields.outcome_index, 1);
    }
}
