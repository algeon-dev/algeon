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
//! - [`log_leaf_guard_rejection`] emits a warn line for the GPU-only leaf guard,
//!   including the guard error facts needed to inspect the rejected plan.
//!
//! [`FlightSqlNativeRejectClass`] buckets common [`FallbackReason`] codes so log
//! searches do not need to enumerate every variant.

use datafusion::common::DataFusionError;

use super::config::ServerSessionProfile;
use crate::{
    error::DataFusionErrorExt,
    policy::FallbackReason,
    report::{CandidateOutcome, DataLayoutFixableRejection, PlanningReport},
};

/// Hint appended to the leaf-guard log line when the server is running under
/// [`ServerSessionProfile::CpuFallback`]. The string is stable for log-grep
/// tooling.
///
/// Under `CpuFallback`, this means Iceberg is still using the GPU-only provider
/// because the binary was built without `iceberg-cpu`.
const CPU_FALLBACK_PROFILE_HINT: &str = "expected in cpu_fallback session profile over Iceberg when the \
     `iceberg-cpu` feature was not compiled in — rebuild with --features iceberg-cpu, or switch to \
     NEXUS_SERVER_SESSION_PROFILE=native";

/// Stable bucket for common Flight SQL native-rejection codes.
///
/// The named variants cover common Iceberg-backed Flight SQL rejections; all
/// other [`FallbackReason`] codes collapse into [`FlightSqlNativeRejectClass::Other`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightSqlNativeRejectClass {
    /// DataFusion left a `SortPreservingMergeExec` edge case that native
    /// lowering could not safely reproduce.
    NativeSortPreservingMerge,
    /// A GPU-only table-format scan leaf survived native optimization, so the
    /// surrounding plan cannot fall back to DataFusion CPU execution.
    TableFormatGpuOnlyLeaf,
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
            Self::TableFormatGpuOnlyLeaf => "table_format_gpu_only_leaf_not_rewritten",
            Self::IcebergObjectStoreRead => "iceberg_object_store_read_unsupported",
            Self::IcebergDeleteFiles => "iceberg_delete_files_unsupported",
            Self::Other => "other",
        }
    }

    pub const fn from_reason(reason: FallbackReason) -> Self {
        match reason {
            FallbackReason::NativeSortPreservingMergeUnsupported => Self::NativeSortPreservingMerge,
            FallbackReason::TableFormatGpuOnlyLeafNotRewritten => Self::TableFormatGpuOnlyLeaf,
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

/// Emit a `tracing::warn!` line when planning fails because the Milestone 1
/// GPU-only leaf guard fired.
///
/// Looks for `fallback_reason=table_format_gpu_only_leaf_not_rewritten` and
/// quotes the relevant adapter error facts. Other error shapes are ignored.
///
/// Under [`ServerSessionProfile::CpuFallback`], also emits a `profile_hint`
/// explaining the testing-only Iceberg/GPU-leaf combination.
pub fn log_leaf_guard_rejection(query_id: &str, error: &DataFusionError, profile: ServerSessionProfile) {
    let Some(fields) = leaf_guard_event_fields(error) else {
        return;
    };
    tracing::warn!(
        query_id,
        fallback_reason = fields.fallback_reason,
        leaf_guard_first_rejection = fields.leaf_guard_first_rejection,
        leaf_guard_context_source = fields.leaf_guard_context_source,
        native_rule_blocking_context = fields.native_rule_blocking_context,
        native_rule_blocking_rejection = fields.native_rule_blocking_rejection,
        physical_plan_root = fields.physical_plan_root,
        physical_plan_shape = fields.physical_plan_shape,
        table_format = fields.table_format,
        snapshot_id = fields.snapshot_id,
        catalog_type = fields.catalog_type,
        planned_data_file_count = fields.planned_data_file_count,
        planned_delete_file_count = fields.planned_delete_file_count,
        rejected_feature_list = fields.rejected_feature_list,
        source_read_strategy = fields.source_read_strategy,
        credential_mode = fields.credential_mode,
        lakehouse_cache_status = fields.lakehouse_cache_status,
        direct_device_read_disposition = fields.direct_device_read_disposition,
        unsupported_column = fields.unsupported_column,
        unsupported_column_type = fields.unsupported_column_type,
        unsupported_column_role = fields.unsupported_column_role,
        rejection_class = fields.rejection_class.as_str(),
        session_profile = profile.as_str(),
        profile_hint = profile_hint(profile),
        "datafusion-nexus Flight SQL planning rejected by GPU-only table-format leaf guard"
    );
}

/// Operator-facing hint paired with a leaf-guard rejection.
///
/// Returns the stable [`CPU_FALLBACK_PROFILE_HINT`] string for the
/// testing-only [`ServerSessionProfile::CpuFallback`] profile and an empty
/// string otherwise. Exposed at crate visibility so unit tests can pin the
/// stable value without going through `tracing-subscriber` capture.
pub(crate) fn profile_hint(profile: ServerSessionProfile) -> &'static str {
    match profile {
        ServerSessionProfile::CpuFallback => CPU_FALLBACK_PROFILE_HINT,
        ServerSessionProfile::Native => "",
    }
}

/// Structured field set extracted from a leaf-guard adapter error.
///
/// Returned by [`leaf_guard_event_fields`] so tests can verify the exact field
/// set that the `tracing::warn!` macro consumes. The lifetime is borrowed
/// directly from the inspected error's fact strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LeafGuardEventFields<'a> {
    pub fallback_reason: &'a str,
    pub leaf_guard_first_rejection: &'a str,
    pub leaf_guard_context_source: &'a str,
    pub native_rule_blocking_context: &'a str,
    pub native_rule_blocking_rejection: &'a str,
    pub physical_plan_root: &'a str,
    pub physical_plan_shape: &'a str,
    pub table_format: &'a str,
    pub snapshot_id: &'a str,
    pub catalog_type: &'a str,
    pub planned_data_file_count: &'a str,
    pub planned_delete_file_count: &'a str,
    pub rejected_feature_list: &'a str,
    pub source_read_strategy: &'a str,
    pub credential_mode: &'a str,
    pub lakehouse_cache_status: &'a str,
    pub direct_device_read_disposition: &'a str,
    pub unsupported_column: &'a str,
    pub unsupported_column_type: &'a str,
    pub unsupported_column_role: &'a str,
    pub rejection_class: FlightSqlNativeRejectClass,
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

/// Extract the structured leaf-guard fields from a [`DataFusionError`] when it
/// wraps an adapter [`Error`] tagged with the leaf-guard fallback reason.
///
/// Returns `None` for any other error shape so the caller knows to keep the
/// generic planning-failure log line. Empty-string defaults are used for facts
/// the guard didn't attach (for example `native_rule_blocking_rejection` is absent
/// when the leaf survives because of a structural mixed-plan context rather
/// than a native-rule rejection).
pub(crate) fn leaf_guard_event_fields(error: &DataFusionError) -> Option<LeafGuardEventFields<'_>> {
    let adapter = error.find_nexus_error()?;
    let facts = adapter.facts();
    let fallback_reason = facts.get("fallback_reason")?;
    if fallback_reason != FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code() {
        return None;
    }
    let leaf_guard_first_rejection = facts
        .get("leaf_guard_first_rejection")
        .or_else(|| facts.get("native_rule_blocking_rejection"))
        .unwrap_or(fallback_reason);
    let native_rule_blocking_rejection = facts.get("native_rule_blocking_rejection").unwrap_or("");
    Some(LeafGuardEventFields {
        fallback_reason,
        leaf_guard_first_rejection,
        leaf_guard_context_source: facts.get("leaf_guard_context_source").unwrap_or(""),
        native_rule_blocking_context: facts.get("native_rule_blocking_context").unwrap_or(""),
        native_rule_blocking_rejection,
        physical_plan_root: facts.get("physical_plan_root").unwrap_or(""),
        physical_plan_shape: facts.get("physical_plan_shape").unwrap_or(""),
        table_format: facts.get("table_format").unwrap_or(""),
        snapshot_id: facts.get("snapshot_id").unwrap_or(""),
        catalog_type: facts.get("catalog_type").unwrap_or(""),
        planned_data_file_count: facts.get("planned_data_file_count").unwrap_or(""),
        planned_delete_file_count: facts.get("planned_delete_file_count").unwrap_or(""),
        rejected_feature_list: facts.get("rejected_feature_list").unwrap_or(""),
        source_read_strategy: facts.get("source_read_strategy").unwrap_or(""),
        credential_mode: facts.get("credential_mode").unwrap_or(""),
        lakehouse_cache_status: facts.get("lakehouse_cache_status").unwrap_or(""),
        direct_device_read_disposition: facts.get("direct_device_read_disposition").unwrap_or(""),
        unsupported_column: facts.get("unsupported_column").unwrap_or(""),
        unsupported_column_type: facts.get("unsupported_column_type").unwrap_or(""),
        unsupported_column_role: facts.get("unsupported_column_role").unwrap_or(""),
        rejection_class: FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf,
    })
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
    use crate::{
        error::{Error, ErrorCode},
        report::{GpuCandidate, PlanningReport},
    };

    #[test]
    fn class_labels_are_stable_strings() {
        assert_eq!(
            FlightSqlNativeRejectClass::NativeSortPreservingMerge.as_str(),
            "native_sort_preserving_merge_unsupported"
        );
        assert_eq!(
            FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf.as_str(),
            "table_format_gpu_only_leaf_not_rewritten"
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
            FlightSqlNativeRejectClass::from_reason(FallbackReason::TableFormatGpuOnlyLeafNotRewritten),
            FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf,
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

    #[test]
    fn find_nexus_error_recurses_through_context_and_collection() {
        let inner = Error::new(
            ErrorCode::NativeCandidateRejected,
            "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
            "rejected by leaf guard",
        )
        .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
        let df_error: DataFusionError = inner.into();
        let wrapped = DataFusionError::Context("optimizer rule".to_string(), Box::new(df_error));

        let found = wrapped.find_nexus_error().expect("nested adapter error must be discoverable");
        assert_eq!(
            found.fact_value("fallback_reason"),
            Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code()),
        );
    }

    fn leaf_guard_error(native_rule_blocking_rejection: Option<&str>) -> DataFusionError {
        let mut error = Error::new(
            ErrorCode::NativeCandidateRejected,
            "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
            "table-format scan leaf was not rewritten to native",
        )
        .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
        .with_fact("leftover_operator", "CudfTableFormatScanExec")
        .with_fact("table_format", "iceberg")
        .with_fact("cpu_arrow_reader_fallback", false)
        .with_fact("physical_plan_root", "SortPreservingMergeExec")
        .with_fact(
            "physical_plan_shape",
            "SortPreservingMergeExec(SortExec(AggregateExec(RepartitionExec(CudfTableFormatScanExec))))",
        );
        if let Some(reason) = native_rule_blocking_rejection {
            error = error.with_fact("leaf_guard_first_rejection", reason);
            error = error.with_fact("leaf_guard_context_source", "native_rule");
            error = error.with_fact("native_rule_blocking_rejection", reason);
            error = error.with_fact("native_rule_blocking_context", reason);
        }
        error.into()
    }

    #[test]
    fn leaf_guard_event_fields_extracts_all_facts() {
        let error = leaf_guard_error(Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()));
        let fields = leaf_guard_event_fields(&error).expect("leaf-guard error must produce structured fields");

        assert_eq!(fields.fallback_reason, FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
        assert_eq!(fields.native_rule_blocking_context, FallbackReason::NativeSortPreservingMergeUnsupported.code());
        assert_eq!(fields.native_rule_blocking_rejection, FallbackReason::NativeSortPreservingMergeUnsupported.code());
        assert_eq!(fields.physical_plan_root, "SortPreservingMergeExec");
        assert!(fields.physical_plan_shape.starts_with("SortPreservingMergeExec("));
        assert!(fields.physical_plan_shape.contains("CudfTableFormatScanExec"));
        assert_eq!(fields.table_format, "iceberg");
        assert_eq!(fields.leaf_guard_first_rejection, FallbackReason::NativeSortPreservingMergeUnsupported.code());
        assert_eq!(fields.unsupported_column, "");
        assert_eq!(fields.unsupported_column_type, "");
        assert_eq!(fields.unsupported_column_role, "");
        assert_eq!(fields.rejection_class, FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf);
    }

    #[test]
    fn leaf_guard_event_fields_defaults_missing_blocking_rejection_to_empty() {
        // The Milestone 1 guard skips the `native_rule_blocking_rejection` fact when the
        // native rule did not produce a rejection - for example when the leaf
        // survives because an unrelated optimizer rule wrapped it without the
        // native rule ever being invoked.
        let error = leaf_guard_error(None);
        let fields = leaf_guard_event_fields(&error).expect("leaf-guard error must produce structured fields");
        assert_eq!(fields.native_rule_blocking_context, "");
        assert_eq!(fields.native_rule_blocking_rejection, "");
    }

    #[test]
    fn leaf_guard_event_fields_preserves_structural_context_without_native_rejection() {
        let mut error = Error::new(
            ErrorCode::NativeCandidateRejected,
            "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
            "table-format scan leaf was not rewritten to native",
        )
        .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
        .with_fact("leftover_operator", "CudfTableFormatScanExec")
        .with_fact("table_format", "iceberg")
        .with_fact("cpu_arrow_reader_fallback", false)
        .with_fact("physical_plan_root", "UnionExec")
        .with_fact("physical_plan_shape", "UnionExec(CudfTableFormatScanExec,NativeCudfExec)");
        error = error.with_fact("native_rule_blocking_context", "union_branch_not_rewritten");
        let df_error: DataFusionError = error.into();

        let fields = leaf_guard_event_fields(&df_error).expect("leaf-guard error must produce structured fields");
        assert_eq!(fields.native_rule_blocking_context, "union_branch_not_rewritten");
        assert_eq!(fields.native_rule_blocking_rejection, "");
        assert_eq!(fields.physical_plan_root, "UnionExec");
    }

    #[test]
    fn leaf_guard_event_fields_prefers_first_rejection_and_extracts_scan_diagnostics() {
        let error = Error::new(
            ErrorCode::NativeCandidateRejected,
            "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
            "table-format scan leaf was not rewritten to native",
        )
        .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
        .with_fact("leaf_guard_first_rejection", FallbackReason::IcebergFieldIdMappingUnsupported.code())
        .with_fact("leaf_guard_context_source", "scan_plan")
        .with_fact("native_rule_blocking_context", FallbackReason::IcebergFieldIdMappingUnsupported.code())
        .with_fact("leftover_operator", "CudfTableFormatScanExec")
        .with_fact("table_format", "iceberg")
        .with_fact("snapshot_id", "s-123")
        .with_fact("catalog_type", "glue")
        .with_fact("planned_data_file_count", "2")
        .with_fact("planned_delete_file_count", "1")
        .with_fact("rejected_feature_list", FallbackReason::IcebergFieldIdMappingUnsupported.code())
        .with_fact("source_read_strategy", "remote_direct")
        .with_fact("credential_mode", "scoped_resolved")
        .with_fact("lakehouse_cache_status", "miss")
        .with_fact("direct_device_read_disposition", "rejected")
        .with_fact("unsupported_column", "countDetail")
        .with_fact("unsupported_column_type", "Map(\"entries\": non-null Struct(\"key\": non-null Utf8), unsorted)")
        .with_fact("unsupported_column_role", "projected");
        let df_error: DataFusionError = error.into();

        let fields = leaf_guard_event_fields(&df_error).expect("leaf-guard error must produce structured fields");

        assert_eq!(fields.leaf_guard_first_rejection, FallbackReason::IcebergFieldIdMappingUnsupported.code());
        assert_eq!(fields.leaf_guard_context_source, "scan_plan");
        assert_eq!(fields.table_format, "iceberg");
        assert_eq!(fields.snapshot_id, "s-123");
        assert_eq!(fields.catalog_type, "glue");
        assert_eq!(fields.planned_data_file_count, "2");
        assert_eq!(fields.planned_delete_file_count, "1");
        assert_eq!(fields.rejected_feature_list, FallbackReason::IcebergFieldIdMappingUnsupported.code());
        assert_eq!(fields.source_read_strategy, "remote_direct");
        assert_eq!(fields.credential_mode, "scoped_resolved");
        assert_eq!(fields.lakehouse_cache_status, "miss");
        assert_eq!(fields.direct_device_read_disposition, "rejected");
        assert_eq!(fields.unsupported_column, "countDetail");
        assert_eq!(
            fields.unsupported_column_type,
            "Map(\"entries\": non-null Struct(\"key\": non-null Utf8), unsorted)"
        );
        assert_eq!(fields.unsupported_column_role, "projected");
        assert_eq!(fields.rejection_class, FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf);
    }

    #[test]
    fn leaf_guard_event_fields_is_none_for_unrelated_errors() {
        assert!(leaf_guard_event_fields(&DataFusionError::Plan("not a leaf guard error".to_string())).is_none());

        let other_adapter =
            Error::new(ErrorCode::UnsupportedExpression, "datafusion_nexus.test", "some other adapter error")
                .with_fact("fallback_reason", FallbackReason::UnsupportedExpression.code());
        let df_error: DataFusionError = other_adapter.into();
        assert!(leaf_guard_event_fields(&df_error).is_none());
    }

    #[test]
    fn leaf_guard_event_fields_recurses_through_context() {
        let error = leaf_guard_error(Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()));
        let wrapped = DataFusionError::Context("optimizer rule: leaf guard".to_string(), Box::new(error));
        let fields = leaf_guard_event_fields(&wrapped).expect("Context-wrapped leaf-guard error must extract");
        assert_eq!(fields.fallback_reason, FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
    }

    #[test]
    fn log_leaf_guard_rejection_does_not_panic_on_either_path() {
        // The macro path is hard to assert on without a tracing-subscriber,
        // but we can at least confirm both the "fires" and "no-op" branches
        // do not panic. Cover both profiles so the profile-hint branch is
        // exercised.
        let leaf_guard = leaf_guard_error(Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()));
        log_leaf_guard_rejection("q-leaf-guard-native", &leaf_guard, ServerSessionProfile::Native);
        log_leaf_guard_rejection("q-leaf-guard-cpu", &leaf_guard, ServerSessionProfile::CpuFallback);

        let unrelated = DataFusionError::Plan("unrelated".to_string());
        log_leaf_guard_rejection("q-unrelated", &unrelated, ServerSessionProfile::Native);
    }

    #[test]
    fn profile_hint_fires_only_for_cpu_fallback() {
        assert_eq!(profile_hint(ServerSessionProfile::Native), "");
        let cpu_hint = profile_hint(ServerSessionProfile::CpuFallback);
        assert!(!cpu_hint.is_empty(), "cpu_fallback profile must carry a hint");
        assert!(
            cpu_hint.contains("NEXUS_SERVER_SESSION_PROFILE=native"),
            "the hint must point operators back at the env var to fix the profile; got {cpu_hint:?}",
        );
        assert!(
            cpu_hint.contains("cpu_fallback"),
            "the hint must mention the profile name so log greps surface it; got {cpu_hint:?}",
        );
    }
}
