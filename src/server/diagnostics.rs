//! Flight SQL native-planning diagnostics surface.
//!
//! The Flight SQL server runs DataFusion planning on every statement. When the
//! cudf native physical optimizer rule rejects part or all of the plan, the
//! single stable surface for "why was this not native?" is the
//! [`crate::report::PlanningReport`] kept by the retained native rule. This
//! module exposes two helpers that the server's planning path can call without
//! adding new public reporting surface area:
//!
//! - [`log_native_planning_report`] writes a structured `tracing::debug!` line
//!   per query that summarises the planning outcomes (rewritten / rejected
//!   counts, first rejection code, classifier bucket). Operators can flip the
//!   server's log filter to `debug` and get one line per statement without
//!   recompiling.
//! - [`log_leaf_guard_rejection`] writes a `tracing::warn!` line when planning
//!   fails because the Milestone 1 GPU-only leaf guard fired. The line quotes the
//!   structured facts attached to the guard's [`crate::Error`] (fallback
//!   reason, first native rejection, physical plan shape, table format, …) so
//!   the immediate audit follow-up — "what was the planner trying to lower?" —
//!   is answerable from server logs alone.
//!
//! The fifth public surface is [`FlightSqlNativeRejectClass`], which buckets
//! [`FallbackReason`] codes into the four common Flight SQL rejection classes
//! the Milestone 4 plan calls out plus a residual `Other` bucket. The bucket label
//! goes onto every log line so an operator can grep for one keyword instead of
//! enumerating every reject code variant.

#![deny(missing_docs)]

use datafusion::common::DataFusionError;

use super::config::ServerSessionProfile;
use crate::{
    Error,
    policy::FallbackReason,
    report::{CandidateOutcome, PlanningReport},
};

/// Hint appended to the leaf-guard log line when the server is running under
/// [`ServerSessionProfile::CpuFallback`]. The string is stable for log-grep
/// tooling.
///
/// The leaf-guard fires when the GPU-only `CudfTableFormatScanExec` survives
/// to execution. Under `CpuFallback` that means the catalog handed out the
/// GPU-only provider — which happens when the binary was built **without**
/// the `iceberg-cpu` feature. With `iceberg-cpu` compiled in the catalog
/// routes Iceberg through `iceberg_datafusion::IcebergStaticTableProvider`
/// and this guard does not fire.
const CPU_FALLBACK_PROFILE_HINT: &str = "expected in cpu_fallback session profile over Iceberg when the \
     `iceberg-cpu` feature was not compiled in — rebuild with --features iceberg-cpu, or switch to \
     DATAFUSION_NEXUS_SERVER_SESSION_PROFILE=native";

/// Stable bucket for common Flight SQL native-rejection codes.
///
/// The four named variants correspond to the rejection reasons the Milestone 4
/// plan explicitly listed as common when serving Iceberg-backed Flight SQL
/// queries. Anything else collapses into [`FlightSqlNativeRejectClass::Other`]
/// so log queries can grep by class without enumerating the full set of
/// [`FallbackReason`] codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlightSqlNativeRejectClass {
    /// DataFusion left a full `SortPreservingMergeExec` (no `fetch`) at the
    /// plan root, which native lowering does not yet support.
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

    /// Bucket a [`FallbackReason`] into one of the common Flight SQL classes.
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
/// the first rejection code (when any candidate was rejected), and the
/// classifier bucket for that first rejection. No-op for an empty report so
/// pure-CPU sessions do not generate noise.
pub fn log_native_planning_report(query_id: &str, report: Option<&PlanningReport>) {
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
        first_native_rejection = first_rejection.map(FallbackReason::code).unwrap_or(""),
        rejection_class = rejection_class.map(FlightSqlNativeRejectClass::as_str).unwrap_or(""),
        "datafusion-nexus native planning report"
    );
}

/// Emit a `tracing::warn!` line when planning fails because the Milestone 1
/// GPU-only leaf guard fired.
///
/// The function walks the [`DataFusionError`] looking for an adapter
/// [`Error`] tagged with `fallback_reason=table_format_gpu_only_leaf_not_rewritten`
/// and quotes its facts (`first_native_rejection`, `physical_plan_shape`,
/// `physical_plan_root`, `table_format`). It is a no-op for any other error
/// shape so non-leaf-guard failures keep their existing log line.
///
/// When `profile` is [`ServerSessionProfile::CpuFallback`], the emitted line
/// additionally carries a `profile_hint` field pointing operators back to the
/// native profile — this combination is the testing-only configuration
/// where the audit shape is reproducible, and the hint short-circuits the
/// usual "why did this fail?" follow-up.
pub fn log_leaf_guard_rejection(query_id: &str, error: &DataFusionError, profile: ServerSessionProfile) {
    let Some(fields) = leaf_guard_event_fields(error) else {
        return;
    };
    tracing::warn!(
        query_id,
        fallback_reason = fields.fallback_reason,
        first_native_rejection = fields.first_native_rejection,
        physical_plan_root = fields.physical_plan_root,
        physical_plan_shape = fields.physical_plan_shape,
        table_format = fields.table_format,
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
    pub first_native_rejection: &'a str,
    pub physical_plan_root: &'a str,
    pub physical_plan_shape: &'a str,
    pub table_format: &'a str,
    pub rejection_class: FlightSqlNativeRejectClass,
}

/// Extract the structured leaf-guard fields from a [`DataFusionError`] when it
/// wraps an adapter [`Error`] tagged with the leaf-guard fallback reason.
///
/// Returns `None` for any other error shape so the caller knows to keep the
/// generic planning-failure log line. Empty-string defaults are used for facts
/// the guard didn't attach (for example `first_native_rejection` is absent
/// when the leaf survives without a native rejection).
pub(crate) fn leaf_guard_event_fields(error: &DataFusionError) -> Option<LeafGuardEventFields<'_>> {
    let adapter = find_adapter_error(error)?;
    let facts = adapter.facts();
    let fallback_reason = facts.get("fallback_reason")?;
    if fallback_reason != FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code() {
        return None;
    }
    Some(LeafGuardEventFields {
        fallback_reason,
        first_native_rejection: facts.get("first_native_rejection").unwrap_or(""),
        physical_plan_root: facts.get("physical_plan_root").unwrap_or(""),
        physical_plan_shape: facts.get("physical_plan_shape").unwrap_or(""),
        table_format: facts.get("table_format").unwrap_or(""),
        rejection_class: FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf,
    })
}

fn first_rejection_reason(report: &PlanningReport) -> Option<FallbackReason> {
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { reason, .. } => Some(*reason),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

fn find_adapter_error(error: &DataFusionError) -> Option<&Error> {
    match error {
        DataFusionError::External(source) => source.downcast_ref::<Error>(),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => find_adapter_error(source),
        DataFusionError::Collection(errors) => errors.iter().find_map(find_adapter_error),
        DataFusionError::Shared(source) => find_adapter_error(source),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        error::ErrorCode,
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
    fn find_adapter_error_recurses_through_context_and_collection() {
        let inner = Error::new(
            ErrorCode::NativeCandidateRejected,
            "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
            "rejected by leaf guard",
        )
        .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
        let df_error: DataFusionError = inner.into();
        let wrapped = DataFusionError::Context("optimizer rule".to_string(), Box::new(df_error));

        let found = find_adapter_error(&wrapped).expect("nested adapter error must be discoverable");
        assert_eq!(
            found.fact_value("fallback_reason"),
            Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code()),
        );
    }

    fn leaf_guard_error(first_native_rejection: Option<&str>) -> DataFusionError {
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
        if let Some(reason) = first_native_rejection {
            error = error.with_fact("first_native_rejection", reason);
        }
        error.into()
    }

    #[test]
    fn leaf_guard_event_fields_extracts_all_facts() {
        let error = leaf_guard_error(Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()));
        let fields = leaf_guard_event_fields(&error).expect("leaf-guard error must produce structured fields");

        assert_eq!(fields.fallback_reason, FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
        assert_eq!(fields.first_native_rejection, FallbackReason::NativeSortPreservingMergeUnsupported.code());
        assert_eq!(fields.physical_plan_root, "SortPreservingMergeExec");
        assert!(fields.physical_plan_shape.starts_with("SortPreservingMergeExec("));
        assert!(fields.physical_plan_shape.contains("CudfTableFormatScanExec"));
        assert_eq!(fields.table_format, "iceberg");
        assert_eq!(fields.rejection_class, FlightSqlNativeRejectClass::TableFormatGpuOnlyLeaf);
    }

    #[test]
    fn leaf_guard_event_fields_defaults_missing_first_native_rejection_to_empty() {
        // The Milestone 1 guard skips the `first_native_rejection` fact when the
        // native rule did not produce a rejection — for example when the leaf
        // survives because an unrelated optimizer rule wrapped it without the
        // native rule ever being invoked.
        let error = leaf_guard_error(None);
        let fields = leaf_guard_event_fields(&error).expect("leaf-guard error must produce structured fields");
        assert_eq!(fields.first_native_rejection, "");
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
            cpu_hint.contains("DATAFUSION_NEXUS_SERVER_SESSION_PROFILE=native"),
            "the hint must point operators back at the env var to fix the profile; got {cpu_hint:?}",
        );
        assert!(
            cpu_hint.contains("cpu_fallback"),
            "the hint must mention the profile name so log greps surface it; got {cpu_hint:?}",
        );
    }
}
