//! Flight SQL native-planning diagnostics surface.
//!
//! The retained native optimizer rule keeps the stable
//! [`crate::report::PlanningReport`] for "why was this not native?". This module
//! logs that evidence without adding a public reporting API:
//!
//! - [`log_native_planning_report`] emits one debug line per query with rewrite
//!   and rejection counts plus the first rejection class.
//! - [`log_native_planning_report_for_tables`] also emits a warn line for
//!   data-layout-fixable rejections, with the referenced table list and reason.
//! [`FlightSqlNativeRejectClass`] buckets common [`FallbackReason`] codes so log
//! searches do not need to enumerate every variant.

use crate::{
    policy::FallbackReason,
    report::{CandidateOutcome, DataLayoutFixableRejection, PlanningReport},
};

/// Stable bucket for common Flight SQL native-rejection codes.
///
/// The named variants cover common Iceberg-backed Flight SQL rejections; all
/// other [`FallbackReason`] codes collapse into [`FlightSqlNativeRejectClass::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightSqlNativeRejectClass {
    /// DataFusion left a `SortPreservingMergeExec` edge case that native
    /// lowering could not safely reproduce.
    NativeSortPreservingMerge,
    /// The Iceberg source is backed by an object-store read that native
    /// lowering cannot reproduce as a local Parquet read.
    IcebergObjectStoreRead,
    /// The Iceberg source has delete files attached, which native lowering
    /// does not yet support.
    IcebergDeleteFiles,
    /// Any rejection code outside the four common Flight SQL classes.
    Other,
}

impl FlightSqlNativeRejectClass {
    /// Stable snake-case label used by log lines and tests.
    ///
    /// The first four labels intentionally match the corresponding
    /// [`FallbackReason::code`] values so a single keyword search across server
    /// logs covers both the rejection code and the bucket label. The
    /// residual bucket returns `"other"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeSortPreservingMerge => "native_sort_preserving_merge_unsupported",
            Self::IcebergObjectStoreRead => "iceberg_object_store_read_unsupported",
            Self::IcebergDeleteFiles => "iceberg_delete_files_unsupported",
            Self::Other => "other",
        }
    }

    pub const fn from_reason(reason: FallbackReason) -> Self {
        match reason {
            FallbackReason::NativeSortPreservingMergeUnsupported => Self::NativeSortPreservingMerge,
            FallbackReason::IcebergObjectStoreReadUnsupported => Self::IcebergObjectStoreRead,
            FallbackReason::IcebergDeleteFilesUnsupported => Self::IcebergDeleteFiles,
            _ => Self::Other,
        }
    }
}

/// Emit one `tracing::debug!` line summarising a Flight SQL native planning
/// report.
///
/// The line carries the per-query identifier plus rewritten/rejected counts,
/// the first native-rule rejection as context (when any candidate was
/// rejected), and the classifier bucket for that contextual rejection. No-op
/// for an empty report so pure-CPU sessions do not generate noise.
pub fn log_native_planning_report(query_id: &str, report: Option<&PlanningReport>) {
    log_native_planning_report_for_tables(query_id, report, "");
}

/// Emit the native planning diagnostics for a Flight SQL query, including a
/// WARN line when the first relevant rejection is fixable by changing table
/// layout metadata.
pub fn log_native_planning_report_for_tables(query_id: &str, report: Option<&PlanningReport>, tables: &str) {
    let Some(report) = report else {
        return;
    };
    if report.outcomes().is_empty() {
        return;
    }

    let first_rejection = first_rejection_reason(report);
    let rejection_class = first_rejection.map(FlightSqlNativeRejectClass::from_reason);
    tracing::debug!(
        query_id,
        rewritten_count = report.rewritten_count(),
        rejected_count = report.rejected_count(),
        native_rule_first_rejection_context = first_rejection.map(FallbackReason::code).unwrap_or(""),
        rejection_class = rejection_class.map(FlightSqlNativeRejectClass::as_str).unwrap_or(""),
        "datafusion-nexus native planning report"
    );

    if let Some(fields) = data_layout_fixable_planning_event_fields(report, tables) {
        tracing::warn!(
            query_id,
            tables = fields.tables,
            reason_code = fields.reason_code,
            layout_property = fields.layout_property,
            native_candidate_shape = fields.native_candidate_shape,
            outcome_index = fields.outcome_index,
            "datafusion-nexus native planning rejected a data-layout-fixable candidate"
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
    let DataLayoutFixableRejection { outcome_index, candidate, reason, property } =
        report.data_layout_fixable_rejection()?;
    Some(DataLayoutFixablePlanningEventFields {
        tables,
        reason_code: reason.code(),
        layout_property: property,
        native_candidate_shape: candidate.shape.as_str(),
        outcome_index,
    })
}

fn first_rejection_reason(report: &PlanningReport) -> Option<FallbackReason> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { reason, .. } => Some(*reason),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{GpuCandidate, PlanningReport};

    #[test]
    fn class_labels_are_stable_strings() {
        assert_eq!(
            FlightSqlNativeRejectClass::NativeSortPreservingMerge.as_str(),
            "native_sort_preserving_merge_unsupported"
        );
        assert_eq!(
            FlightSqlNativeRejectClass::IcebergObjectStoreRead.as_str(),
            "iceberg_object_store_read_unsupported"
        );
        assert_eq!(FlightSqlNativeRejectClass::IcebergDeleteFiles.as_str(), "iceberg_delete_files_unsupported");
        assert_eq!(FlightSqlNativeRejectClass::Other.as_str(), "other");
    }

    #[test]
    fn classifies_named_buckets_from_fallback_reason() {
        assert_eq!(
            FlightSqlNativeRejectClass::from_reason(FallbackReason::NativeSortPreservingMergeUnsupported),
            FlightSqlNativeRejectClass::NativeSortPreservingMerge,
        );
        assert_eq!(
            FlightSqlNativeRejectClass::from_reason(FallbackReason::IcebergObjectStoreReadUnsupported),
            FlightSqlNativeRejectClass::IcebergObjectStoreRead,
        );
        assert_eq!(
            FlightSqlNativeRejectClass::from_reason(FallbackReason::IcebergDeleteFilesUnsupported),
            FlightSqlNativeRejectClass::IcebergDeleteFiles,
        );
    }

    #[test]
    fn other_bucket_covers_unrelated_rejections() {
        for reason in [
            FallbackReason::UnsupportedOperator,
            FallbackReason::UnsupportedExpression,
            FallbackReason::UnsupportedType,
            FallbackReason::NativeBoundedJoinUnsupported,
            FallbackReason::IcebergFieldIdMappingUnsupported,
        ] {
            assert_eq!(
                FlightSqlNativeRejectClass::from_reason(reason),
                FlightSqlNativeRejectClass::Other,
                "{reason:?} should fall through to the residual class",
            );
        }
    }

    #[test]
    fn first_rejection_reason_returns_first_rejected_outcome() {
        let mut report = PlanningReport::new();
        report.push_rewritten(GpuCandidate::new("AggregateExec", 2));
        report.push_rejected(
            GpuCandidate::new("SortPreservingMergeExec", 2),
            FallbackReason::NativeSortPreservingMergeUnsupported,
        );
        report.push_rejected(GpuCandidate::new("AggregateExec", 2), FallbackReason::UnsupportedExpression);

        assert_eq!(first_rejection_reason(&report), Some(FallbackReason::NativeSortPreservingMergeUnsupported));
    }

    #[test]
    fn first_rejection_reason_is_none_when_all_rewritten() {
        let mut report = PlanningReport::new();
        report.push_rewritten(GpuCandidate::new("AggregateExec", 2));
        assert_eq!(first_rejection_reason(&report), None);
    }

    #[test]
    fn log_native_planning_report_is_noop_for_empty_inputs() {
        // The helper unconditionally checks emptiness before logging, so we
        // only need to assert it does not panic for the no-op paths.
        log_native_planning_report("q-empty", None);
        log_native_planning_report("q-empty", Some(&PlanningReport::new()));
    }

    #[test]
    fn data_layout_fixable_planning_event_fields_pin_warn_payload() {
        let mut report = PlanningReport::new();
        report.push_rejected(GpuCandidate::new("AggregateExec(LocalParquetScan)", 2), FallbackReason::UnsupportedType);
        report.push_rejected(
            GpuCandidate::new("AggregateExec(LocalParquetScan)", 2),
            FallbackReason::NativeOrderedOutputUnsupported,
        );

        let fields = data_layout_fixable_planning_event_fields(&report, "public.sorted_lineitem")
            .expect("ordered-output rejection should be classified as data-layout-fixable");

        assert_eq!(fields.tables, "public.sorted_lineitem");
        assert_eq!(fields.reason_code, "native_ordered_output_unsupported");
        assert_eq!(fields.layout_property, "declared_output_ordering");
        assert_eq!(fields.native_candidate_shape, "AggregateExec(LocalParquetScan)");
        assert_eq!(fields.outcome_index, 1);
    }
}
