//! Flight SQL native-planning diagnostics surface.
//!
//! The retained native optimizer rule keeps the stable
//! [`crate::report::PlanningReport`] for "why was this not native?". This module
//! logs that evidence without adding a public reporting API:
//!
//! - [`log_native_planning_report`] emits one debug line per query with
//!   selection counts plus the first unsupported-capability class.
//! - [`log_native_planning_report_for_tables`] also emits a warn line for
//!   data-layout-fixable capability misses, with the referenced table list and reason.
//! - [`FlightSqlNativeNotSupportedClass`] buckets common [`CapabilityReason`] codes so log
//!   searches do not need to enumerate every variant.

use crate::{
    capability::CapabilityReason,
    report::{CandidateOutcome, DataLayoutFixableNotSupported, PlanningReport},
};

/// Stable bucket for common Flight SQL native capability-miss codes.
///
/// The named variants cover common Iceberg-backed Flight SQL capability misses;
/// all other [`CapabilityReason`] codes collapse into
/// [`FlightSqlNativeNotSupportedClass::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightSqlNativeNotSupportedClass {
    /// DataFusion left a `SortPreservingMergeExec` edge case that native
    /// lowering could not safely reproduce.
    NativeSortPreservingMerge,
    /// The Iceberg source is backed by an object-store read that native
    /// lowering cannot reproduce as a local Parquet read.
    IcebergObjectStoreRead,
    /// The Iceberg source has delete files attached, which native lowering
    /// does not yet support.
    IcebergDeleteFiles,
    /// Any capability-miss code outside the common Flight SQL classes.
    Other,
}

impl FlightSqlNativeNotSupportedClass {
    /// Stable snake-case label used by log lines and tests.
    ///
    /// The first four labels intentionally match the corresponding
    /// [`CapabilityReason::code`] values so a single keyword search across server
    /// logs covers both the capability-miss code and the bucket label. The
    /// residual bucket returns `"other"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeSortPreservingMerge => "native_sort_preserving_merge_unsupported",
            Self::IcebergObjectStoreRead => "iceberg_object_store_read_unsupported",
            Self::IcebergDeleteFiles => "iceberg_delete_files_unsupported",
            Self::Other => "other",
        }
    }

    pub const fn from_capability_reason(reason: CapabilityReason) -> Self {
        match reason {
            CapabilityReason::NativeSortPreservingMergeUnsupported => Self::NativeSortPreservingMerge,
            CapabilityReason::IcebergObjectStoreReadUnsupported => Self::IcebergObjectStoreRead,
            CapabilityReason::IcebergDeleteFilesUnsupported => Self::IcebergDeleteFiles,
            _ => Self::Other,
        }
    }
}

/// Emit one `tracing::debug!` line summarising a Flight SQL native planning
/// report.
///
/// The line carries the per-query identifier plus selection counts, the first
/// unsupported native candidate as context, and its classifier bucket. No-op
/// for an empty report so pure-CPU sessions do not generate noise.
pub fn log_native_planning_report(query_id: &str, report: Option<&PlanningReport>) {
    log_native_planning_report_for_tables(query_id, report, "");
}

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
    let not_supported_class = first_not_supported.map(FlightSqlNativeNotSupportedClass::from_capability_reason);
    tracing::debug!(
        query_id,
        selected_count = report.selected_count(),
        not_selected_count = report.not_selected_count(),
        native_rule_first_not_supported_context = first_not_supported.map(CapabilityReason::code).unwrap_or(""),
        not_supported_class = not_supported_class.map(FlightSqlNativeNotSupportedClass::as_str).unwrap_or(""),
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
            "datafusion-nexus native planning found a data-layout-fixable capability miss"
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
    use crate::report::{GpuCandidate, PlanningReport};

    #[test]
    fn class_labels_are_stable_strings() {
        assert_eq!(
            FlightSqlNativeNotSupportedClass::NativeSortPreservingMerge.as_str(),
            "native_sort_preserving_merge_unsupported"
        );
        assert_eq!(
            FlightSqlNativeNotSupportedClass::IcebergObjectStoreRead.as_str(),
            "iceberg_object_store_read_unsupported"
        );
        assert_eq!(FlightSqlNativeNotSupportedClass::IcebergDeleteFiles.as_str(), "iceberg_delete_files_unsupported");
        assert_eq!(FlightSqlNativeNotSupportedClass::Other.as_str(), "other");
    }

    #[test]
    fn classifies_named_buckets_from_capability_reason() {
        assert_eq!(
            FlightSqlNativeNotSupportedClass::from_capability_reason(
                CapabilityReason::NativeSortPreservingMergeUnsupported
            ),
            FlightSqlNativeNotSupportedClass::NativeSortPreservingMerge,
        );
        assert_eq!(
            FlightSqlNativeNotSupportedClass::from_capability_reason(
                CapabilityReason::IcebergObjectStoreReadUnsupported
            ),
            FlightSqlNativeNotSupportedClass::IcebergObjectStoreRead,
        );
        assert_eq!(
            FlightSqlNativeNotSupportedClass::from_capability_reason(CapabilityReason::IcebergDeleteFilesUnsupported),
            FlightSqlNativeNotSupportedClass::IcebergDeleteFiles,
        );
    }

    #[test]
    fn other_bucket_covers_unrelated_capability_misses() {
        for reason in [
            CapabilityReason::UnsupportedOperator,
            CapabilityReason::UnsupportedExpression,
            CapabilityReason::UnsupportedType,
            CapabilityReason::NativeBoundedJoinUnsupported,
            CapabilityReason::IcebergFieldIdMappingUnsupported,
        ] {
            assert_eq!(
                FlightSqlNativeNotSupportedClass::from_capability_reason(reason),
                FlightSqlNativeNotSupportedClass::Other,
                "{reason:?} should fall through to the residual class",
            );
        }
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
        log_native_planning_report("q-empty", None);
        log_native_planning_report("q-empty", Some(&PlanningReport::new()));
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
