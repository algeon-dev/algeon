//! Final-stage guard for adapter-owned GPU-only table-format scan leaves.
//!
//! [`CudfTableFormatScanExec`] is a metadata-only planning leaf. Native
//! lowering is expected to consume it; if the native physical optimizer rule
//! leaves it in place, the surrounding plan cannot fall back to DataFusion CPU
//! execution because the leaf has no CPU Arrow reader. Letting such a plan
//! reach `ExecutionPlan::execute` produces a placeholder runtime error.
//!
//! This rule runs immediately after the native rewrite rule. If any
//! `CudfTableFormatScanExec` survives, the rule turns that into a structured
//! planning error tagged with [`FallbackReason::TableFormatGpuOnlyLeafNotRewritten`]
//! so callers see a real native rejection reason rather than the placeholder.

use std::sync::Arc;

use datafusion::{
    common::{Result as DataFusionResult, config::ConfigOptions},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, union::UnionExec},
};

use crate::{
    error::{Error, ErrorCode},
    native_exec::{NativeCudfComposedExec, NativeCudfExec, gpu_resident_producer_input_name},
    planner::CudfNativePhysicalOptimizerRule,
    policy::FallbackReason,
    report::{CandidateOutcome, PlanningReport},
    table_format::CudfTableFormatScanExec,
};

/// Public rule name and deduplication key for the leftover GPU-only leaf guard.
pub const CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME: &str = "datafusion_nexus_table_format_gpu_only_leaf_guard";
const UNION_BRANCH_NOT_REWRITTEN_CONTEXT: &str = "union_branch_not_rewritten";

/// Final-stage physical optimizer rule that converts leftover
/// [`CudfTableFormatScanExec`] nodes into a structured planning error.
///
/// The rule reads diagnostics from a sibling [`CudfNativePhysicalOptimizerRule`]
/// when wired through session installation so the resulting error can quote a
/// native rule rejection when that rejection is the blocking context.
/// Constructed standalone with [`Self::new`], the rule still fires but native
/// rule context facts stay empty.
#[derive(Debug, Clone)]
pub struct CudfTableFormatGpuOnlyLeafGuardRule {
    native_rule: Option<Arc<CudfNativePhysicalOptimizerRule>>,
}

impl CudfTableFormatGpuOnlyLeafGuardRule {
    /// Build a guard rule that does not reference a native rule's diagnostics.
    pub fn new() -> Self {
        Self { native_rule: None }
    }

    /// Build a guard rule that can quote retained native rule diagnostics into
    /// emitted errors when the final plan shape does not provide a more local
    /// blocking context.
    pub fn with_native_rule(native_rule: Arc<CudfNativePhysicalOptimizerRule>) -> Self {
        Self { native_rule: Some(native_rule) }
    }
}

impl Default for CudfTableFormatGpuOnlyLeafGuardRule {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for CudfTableFormatGpuOnlyLeafGuardRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let Some(leaf) = find_leftover_table_format_scan(plan.as_ref()) else {
            return Ok(plan);
        };
        let native_report = self.native_rule.as_ref().and_then(|rule| rule.last_report());
        let blocking_context = leaf_guard_blocking_context(plan.as_ref(), leaf, native_report.as_ref());
        Err(leftover_leaf_error(leaf, plan.as_ref(), blocking_context).into())
    }

    fn name(&self) -> &str {
        CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        false
    }
}

fn find_leftover_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    for child in plan.children() {
        if let Some(scan) = find_leftover_table_format_scan(child.as_ref()) {
            return Some(scan);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafGuardBlockingContext {
    UnionBranchNotRewritten,
    NativeRuleRejection(FallbackReason),
    ScanPlanRejection(FallbackReason),
}

impl LeafGuardBlockingContext {
    fn code(self) -> &'static str {
        match self {
            Self::UnionBranchNotRewritten => UNION_BRANCH_NOT_REWRITTEN_CONTEXT,
            Self::NativeRuleRejection(reason) | Self::ScanPlanRejection(reason) => reason.code(),
        }
    }

    fn source(self) -> &'static str {
        match self {
            Self::UnionBranchNotRewritten => "plan_shape",
            Self::NativeRuleRejection(_) => "native_rule",
            Self::ScanPlanRejection(_) => "scan_plan",
        }
    }

    fn native_rule_rejection(self) -> Option<FallbackReason> {
        match self {
            Self::NativeRuleRejection(reason) => Some(reason),
            Self::UnionBranchNotRewritten | Self::ScanPlanRejection(_) => None,
        }
    }

    fn blocking_rejection(self) -> Option<FallbackReason> {
        match self {
            Self::NativeRuleRejection(reason) | Self::ScanPlanRejection(reason) => Some(reason),
            Self::UnionBranchNotRewritten => None,
        }
    }
}

fn leaf_guard_blocking_context(
    plan: &dyn ExecutionPlan,
    leaf: &CudfTableFormatScanExec,
    report: Option<&PlanningReport>,
) -> Option<LeafGuardBlockingContext> {
    if let Some(reason) = scan_plan_rejection(leaf).filter(|reason| is_table_format_or_dtype_rejection(*reason)) {
        return Some(LeafGuardBlockingContext::ScanPlanRejection(reason));
    }
    if plan_has_mixed_union_branch_with_leftover_leaf(plan, leaf) {
        return Some(LeafGuardBlockingContext::UnionBranchNotRewritten);
    }

    let native_rule_rejection = report.and_then(|report| first_blocking_rejection(report, leaf));
    native_rule_rejection.map(LeafGuardBlockingContext::NativeRuleRejection)
}

fn first_blocking_rejection(report: &PlanningReport, leaf: &CudfTableFormatScanExec) -> Option<FallbackReason> {
    let leftover_leaf_is_proven_empty = table_format_leaf_is_proven_empty(leaf);
    report.outcomes().iter().find_map(|outcome| match outcome {
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceEmpty, .. }
            if !leftover_leaf_is_proven_empty =>
        {
            None
        },
        CandidateOutcome::Rejected { candidate, reason } if is_table_format_or_dtype_rejection(*reason) => {
            candidate_shape_contains_table_format_scan(candidate.shape.as_str()).then_some(*reason)
        },
        CandidateOutcome::Rejected { reason, .. } => Some(*reason),
        CandidateOutcome::Rewritten { .. } => None,
    })
}

fn candidate_shape_contains_table_format_scan(shape: &str) -> bool {
    shape.contains("TableFormatScan(")
}

fn table_format_leaf_is_proven_empty(leaf: &CudfTableFormatScanExec) -> bool {
    let scan_plan = leaf.scan_plan();
    scan_plan.data_files.is_empty() && scan_plan.delete_files.is_empty() && scan_plan.unsupported_features.is_empty()
}

fn scan_plan_rejection(leaf: &CudfTableFormatScanExec) -> Option<FallbackReason> {
    leaf.scan_plan().first_rejection_reason().or_else(|| {
        leaf.schema()
            .fields()
            .iter()
            .any(|field| !nexus_query_engine::is_supported_arrow_type(field.data_type()))
            .then_some(FallbackReason::UnsupportedType)
    })
}

fn is_table_format_or_dtype_rejection(reason: FallbackReason) -> bool {
    matches!(
        reason,
        FallbackReason::UnsupportedType
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
            | FallbackReason::IcebergSnapshotUnsupported
    )
}

fn plan_has_mixed_union_branch_with_leftover_leaf(plan: &dyn ExecutionPlan, leaf: &CudfTableFormatScanExec) -> bool {
    if let Some(union) = plan.downcast_ref::<UnionExec>() {
        let mut native_branch = false;
        let mut leftover_leaf_branch = false;
        for child in union.inputs() {
            let branch_contains_native = subtree_contains_native_execution_boundary(child.as_ref());
            let branch_contains_leftover_leaf = subtree_contains_table_format_leaf(child.as_ref(), leaf);
            native_branch |= branch_contains_native && !branch_contains_leftover_leaf;
            leftover_leaf_branch |= branch_contains_leftover_leaf && !branch_contains_native;
        }
        if native_branch && leftover_leaf_branch {
            return true;
        }
    }

    plan.children().into_iter().any(|child| plan_has_mixed_union_branch_with_leftover_leaf(child.as_ref(), leaf))
}

fn subtree_contains_native_execution_boundary(plan: &dyn ExecutionPlan) -> bool {
    if plan.downcast_ref::<NativeCudfExec>().is_some()
        || plan.downcast_ref::<NativeCudfComposedExec>().is_some()
        || gpu_resident_producer_input_name(plan).is_some()
    {
        return true;
    }

    plan.children().into_iter().any(|child| subtree_contains_native_execution_boundary(child.as_ref()))
}

fn subtree_contains_table_format_leaf(plan: &dyn ExecutionPlan, leaf: &CudfTableFormatScanExec) -> bool {
    if let Some(scan) = plan.downcast_ref::<CudfTableFormatScanExec>() {
        return std::ptr::eq(scan, leaf);
    }

    plan.children().into_iter().any(|child| subtree_contains_table_format_leaf(child.as_ref(), leaf))
}

fn leftover_leaf_error(
    leaf: &CudfTableFormatScanExec,
    plan: &dyn ExecutionPlan,
    blocking_context: Option<LeafGuardBlockingContext>,
) -> Error {
    let diagnostics = leaf.diagnostics();
    let first_rejection = blocking_context
        .and_then(LeafGuardBlockingContext::blocking_rejection)
        .map(FallbackReason::code)
        .unwrap_or(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code());
    let mut error = Error::new(
        ErrorCode::NativeCandidateRejected,
        "datafusion_nexus.planner.table_format_gpu_only_leaf_guard",
        format!(
            "table-format scan leaf was not rewritten to native; first rejection={first_rejection}; no CPU Arrow reader fallback"
        ),
    )
    .with_fact("fallback_reason", FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
    .with_fact("leaf_guard_first_rejection", first_rejection)
    .with_fact("leftover_operator", leaf.name())
    .with_fact("cpu_arrow_reader_fallback", false)
    .with_fact("physical_plan_root", plan.name())
    .with_fact("physical_plan_shape", physical_plan_shape(plan));
    for (key, value) in diagnostics.facts() {
        error = error.with_fact(key, value);
    }
    if let Some(context) = blocking_context {
        error = error.with_fact("leaf_guard_context_source", context.source());
        error = error.with_fact("native_rule_blocking_context", context.code());
        if let Some(reason) = context.native_rule_rejection() {
            error = error.with_fact("native_rule_blocking_rejection", reason.code());
        }
    }
    error
}

fn physical_plan_shape(plan: &dyn ExecutionPlan) -> String {
    let children = plan.children();
    if children.is_empty() {
        return plan.name().to_string();
    }

    let mut shape = String::new();
    shape.push_str(plan.name());
    shape.push('(');
    for (index, child) in children.iter().enumerate() {
        if index > 0 {
            shape.push(',');
        }
        shape.push_str(&physical_plan_shape(child.as_ref()));
    }
    shape.push(')');
    shape
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use nexus_query_engine::{LiteralBatchSource, QueryPlanBuilder, SourceNode};

    use super::*;
    use crate::{
        native_exec::NativeCudfExecutionConfig,
        planner::CudfOptimizerConfig,
        report::GpuCandidate,
        table_format::{
            TableFormat, TableFormatDataFile, TableFormatDataFileFormat, TableFormatReadCapability,
            TableFormatScanPlan, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
        },
    };

    #[test]
    fn first_blocking_rejection_skips_empty_source_when_leftover_leaf_has_files() {
        let leaf = table_format_scan_with_data_file();
        let mut report = PlanningReport::new();
        report.push_rejected(GpuCandidate::new("AggregateExec", 1), FallbackReason::NativeSourceEmpty);
        report.push_rejected(
            GpuCandidate::new("SortPreservingMergeExec", 1),
            FallbackReason::NativeSortPreservingMergeUnsupported,
        );

        assert_eq!(
            first_blocking_rejection(&report, &leaf),
            Some(FallbackReason::NativeSortPreservingMergeUnsupported)
        );
    }

    #[test]
    fn first_blocking_rejection_omits_empty_source_when_non_empty_leaf_has_no_better_reason() {
        let leaf = table_format_scan_with_data_file();
        let mut report = PlanningReport::new();
        report.push_rejected(GpuCandidate::new("AggregateExec", 1), FallbackReason::NativeSourceEmpty);

        assert_eq!(first_blocking_rejection(&report, &leaf), None);
    }

    #[test]
    fn first_blocking_rejection_keeps_empty_source_when_leaf_is_empty() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(Arc::clone(&schema));
        let leaf = CudfTableFormatScanExec::new(plan, schema);
        let mut report = PlanningReport::new();
        report.push_rejected(GpuCandidate::new("AggregateExec", 1), FallbackReason::NativeSourceEmpty);

        assert_eq!(first_blocking_rejection(&report, &leaf), Some(FallbackReason::NativeSourceEmpty));
    }

    #[test]
    fn leaf_guard_rule_emits_union_branch_context_for_mixed_native_union() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let table_format_leaf = Arc::new(table_format_scan_with_schema_and_data_file(Arc::clone(&schema)));
        let native_branch = native_literal_exec(Arc::clone(&schema));
        let table_format_branch: Arc<dyn ExecutionPlan> = table_format_leaf.clone();
        let union = UnionExec::try_new(vec![table_format_branch, native_branch]).expect("schema-compatible union");

        let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(CudfOptimizerConfig::default()));
        let empty_leaf: Arc<dyn ExecutionPlan> =
            Arc::new(CudfTableFormatScanExec::new(empty_scan_plan(Arc::clone(&schema)), Arc::clone(&schema)));
        native_rule
            .optimize(empty_leaf, &ConfigOptions::new())
            .expect("native rule should retain a rejection report for an empty source");
        let seeded_report = native_rule.last_report().expect("empty source native attempt must produce a report");
        assert_eq!(first_blocking_rejection(&seeded_report, table_format_leaf.as_ref()), None);

        let guard = CudfTableFormatGpuOnlyLeafGuardRule::with_native_rule(native_rule);
        let error = guard
            .optimize(union, &ConfigOptions::new())
            .expect_err("mixed native/table-format union must be rejected by the leaf guard");
        let adapter = adapter_error_from(&error);

        assert_eq!(
            adapter.fact_value("fallback_reason"),
            Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
        );
        assert_eq!(adapter.fact_value("leftover_operator"), Some("CudfTableFormatScanExec"));
        assert_eq!(adapter.fact_value("table_format"), Some("iceberg"));
        assert_eq!(adapter.fact_value("cpu_arrow_reader_fallback"), Some("false"));
        assert_eq!(adapter.fact_value("physical_plan_root"), Some("UnionExec"));
        assert_eq!(adapter.fact_value("native_rule_blocking_context"), Some(UNION_BRANCH_NOT_REWRITTEN_CONTEXT));
        assert_eq!(adapter.fact_value("native_rule_blocking_rejection"), None);
        assert_ne!(adapter.fact_value("native_rule_blocking_context"), Some(FallbackReason::NativeSourceEmpty.code()));
        let shape = adapter.fact_value("physical_plan_shape").expect("physical plan shape fact");
        assert!(shape.contains("UnionExec("), "shape must include the union: {shape}");
        assert!(shape.contains("CudfTableFormatScanExec"), "shape must include the leftover leaf: {shape}");
        assert!(shape.contains("NativeCudfExec"), "shape must include the native branch: {shape}");
    }

    #[test]
    fn leaf_guard_context_ignores_uncorrelated_native_rule_table_format_rejection() {
        let leaf = table_format_scan_with_data_file();
        let mut report = PlanningReport::new();
        report.push_rejected(
            GpuCandidate::new("CudfTableFormatScanExec", 1),
            FallbackReason::IcebergFieldIdMappingUnsupported,
        );

        assert_eq!(leaf.scan_plan().first_rejection_reason(), None);
        assert_eq!(leaf_guard_blocking_context(&leaf, &leaf, Some(&report)), None);
    }

    #[test]
    fn leaf_guard_context_keeps_native_rule_table_format_scan_rejection() {
        let leaf = table_format_scan_with_data_file();
        let mut report = PlanningReport::new();
        report.push_rejected(
            GpuCandidate::new("ProjectionExec(TableFormatScan(iceberg))", 1),
            FallbackReason::IcebergFieldIdMappingUnsupported,
        );

        assert_eq!(
            leaf_guard_blocking_context(&leaf, &leaf, Some(&report)),
            Some(LeafGuardBlockingContext::NativeRuleRejection(FallbackReason::IcebergFieldIdMappingUnsupported))
        );
    }

    #[test]
    fn leaf_guard_context_keeps_native_rule_structural_rejection() {
        let leaf = table_format_scan_with_data_file();
        let mut report = PlanningReport::new();
        report.push_rejected(
            GpuCandidate::new("SortPreservingMergeExec", 1),
            FallbackReason::NativeSortPreservingMergeUnsupported,
        );

        assert_eq!(
            leaf_guard_blocking_context(&leaf, &leaf, Some(&report)),
            Some(LeafGuardBlockingContext::NativeRuleRejection(FallbackReason::NativeSortPreservingMergeUnsupported))
        );
    }

    #[test]
    fn leaf_guard_derives_scan_plan_rejection_and_diagnostics_without_native_rule_context() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_snapshot_id("s-123")
            .with_catalog_type("glue")
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(Arc::clone(&schema))
            .with_data_file(
                TableFormatDataFile::new(
                    "data-0001",
                    "s3://bucket/table/data-0001.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_size_bytes(4096)
                .with_row_count(11),
            )
            .with_unsupported_feature(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported,
                "schema evolution requires field-id-safe binding",
            ));
        let scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(scan_plan, schema));

        let error = CudfTableFormatGpuOnlyLeafGuardRule::new()
            .optimize(scan, &ConfigOptions::new())
            .expect_err("leftover table-format scan must be rejected");
        let adapter = adapter_error_from(&error);

        assert_eq!(
            adapter.fact_value("fallback_reason"),
            Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code())
        );
        assert_eq!(
            adapter.fact_value("leaf_guard_first_rejection"),
            Some(FallbackReason::IcebergFieldIdMappingUnsupported.code())
        );
        assert_eq!(adapter.fact_value("leaf_guard_context_source"), Some("scan_plan"));
        assert_eq!(
            adapter.fact_value("native_rule_blocking_context"),
            Some(FallbackReason::IcebergFieldIdMappingUnsupported.code())
        );
        assert_eq!(adapter.fact_value("native_rule_blocking_rejection"), None);
        assert_eq!(adapter.fact_value("table_format"), Some("iceberg"));
        assert_eq!(adapter.fact_value("snapshot_id"), Some("s-123"));
        assert_eq!(adapter.fact_value("catalog_type"), Some("glue"));
        assert_eq!(adapter.fact_value("planned_data_file_count"), Some("1"));
        assert_eq!(adapter.fact_value("planned_delete_file_count"), Some("0"));
        assert_eq!(
            adapter.fact_value("rejected_feature_list"),
            Some(FallbackReason::IcebergFieldIdMappingUnsupported.code())
        );
        assert_eq!(adapter.fact_value("native_source_read_capability"), Some("object_store:s3"));
        assert_eq!(adapter.fact_value("source_read_strategy"), Some("unknown"));
        assert_eq!(adapter.fact_value("source_decode_input"), Some("unknown"));
        assert_eq!(adapter.fact_value("credential_mode"), Some("unknown"));
        assert_eq!(adapter.fact_value("lakehouse_cache_status"), Some("unknown"));
        assert_eq!(adapter.fact_value("direct_device_read_disposition"), Some("rejected"));
        assert!(
            adapter.to_string().contains(FallbackReason::IcebergFieldIdMappingUnsupported.code()),
            "client-visible leaf-guard error must include the stable first rejection code: {adapter}",
        );
    }

    #[test]
    fn leaf_guard_derives_projected_dtype_rejection_without_native_rule_context() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        )]));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(Arc::clone(&schema))
            .with_data_file(TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            ));
        let scan: Arc<dyn ExecutionPlan> = Arc::new(CudfTableFormatScanExec::new(scan_plan, schema));

        let error = CudfTableFormatGpuOnlyLeafGuardRule::new()
            .optimize(scan, &ConfigOptions::new())
            .expect_err("leftover table-format scan must be rejected");
        let adapter = adapter_error_from(&error);

        assert_eq!(adapter.fact_value("leaf_guard_first_rejection"), Some(FallbackReason::UnsupportedType.code()));
        assert_eq!(adapter.fact_value("leaf_guard_context_source"), Some("scan_plan"));
        assert_eq!(adapter.fact_value("native_rule_blocking_context"), Some(FallbackReason::UnsupportedType.code()));
    }

    fn table_format_scan_with_data_file() -> CudfTableFormatScanExec {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        table_format_scan_with_schema_and_data_file(schema)
    }

    fn table_format_scan_with_schema_and_data_file(schema: Arc<Schema>) -> CudfTableFormatScanExec {
        let file = TableFormatDataFile::new(
            "data-0001",
            "/tmp/iceberg/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::local_file_system(),
        );
        let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(Arc::clone(&schema))
            .with_data_file(file);
        CudfTableFormatScanExec::new(plan, schema)
    }

    fn empty_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
        TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&schema))
            .with_native_source_schema(schema)
    }

    fn native_literal_exec(schema: Arc<Schema>) -> Arc<dyn ExecutionPlan> {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
        Arc::new(
            NativeCudfExec::try_new(builder.finish_with_host_sink(source), NativeCudfExecutionConfig::unbounded())
                .expect("literal native plan should be host-sink executable"),
        )
    }

    fn adapter_error_from(error: &datafusion::common::DataFusionError) -> &Error {
        match error {
            datafusion::common::DataFusionError::External(source) => {
                source.downcast_ref::<Error>().expect("leaf guard errors should cross DataFusion as adapter errors")
            },
            datafusion::common::DataFusionError::Context(_, source)
            | datafusion::common::DataFusionError::Diagnostic(_, source) => adapter_error_from(source),
            other => panic!("expected adapter error from leaf guard, got {other:?}"),
        }
    }
}
