//! Phase 3 regression tests for the Flight SQL Iceberg fallback fix
//! (`docs/plan/2026-05-22-flight-sql-iceberg-cpu-fallback-plan.md`).
//!
//! These tests pin the exact regression surface for two scenarios:
//!
//! - Under the native-friendly server session profile, Iceberg-backed
//!   `GROUP BY ... ORDER BY` and `JOIN + GROUP BY + ORDER BY` queries must
//!   rewrite to native and leave no `CudfTableFormatScanExec` behind.
//! - Under a CPU-style session profile that re-enables DataFusion
//!   repartitioning, the same `GROUP BY ... ORDER BY` shape is rejected by
//!   native lowering (full `SortPreservingMergeExec` unsupported) and the
//!   leftover GPU-only leaf is caught by the Phase 1 leaf guard before
//!   `execute()`.
//! - A leaf-guard unit test asserts that a leftover `CudfTableFormatScanExec`
//!   beneath a non-rewritable wrapper produces the structured
//!   `table_format_gpu_only_leaf_not_rewritten` error.
//!
//! All tests are credential-free; the Iceberg provider is driven by a
//! [`TableFormatPlanner`] returning a hand-built native-admissible
//! [`TableFormatScanPlan`], so no AWS Glue/S3 access is required.

#![cfg(all(feature = "iceberg", feature = "server"))]

use std::sync::{Arc, Mutex};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::{
    common::DataFusionError,
    execution::SessionStateBuilder,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, displayable, sorts::sort_preserving_merge::SortPreservingMergeExec},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
    CudfSessionStateBuilderExt, CudfTableFormatGpuOnlyLeafGuardRule, CudfTableFormatScanExec, FallbackReason,
    GpuFallbackPolicy, TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
    TableFormatDataFile, TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability, TableFormatScanPlan,
    TableFormatScanRequest,
    iceberg::CudfIcebergTableProvider,
    server::{ServerSessionProfile, apply_session_profile},
};
use futures::future::BoxFuture;

#[tokio::test]
async fn native_profile_iceberg_group_by_order_by_rewrites_with_no_leftover_leaf() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT value, COUNT(*) AS c FROM iceberg_t GROUP BY value ORDER BY value",
    )
    .await
    .expect("native profile must rewrite GROUP BY + ORDER BY over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_join_group_by_order_by_rewrites_with_no_leftover_leaf() {
    // Both inputs share the same logical schema (id BIGINT NOT NULL, value
    // VARCHAR). The native-friendly profile pins target_partitions=1, so the
    // hash join, grouped aggregate, and final sort all collapse onto a single
    // native pipeline rooted at NativeCudfExec instead of being split across
    // DataFusion exchange nodes.
    let schema = iceberg_test_schema();
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[
            ("iceberg_t", native_admissible_scan_plan(Arc::clone(&schema))),
            ("iceberg_u", native_admissible_scan_plan(Arc::clone(&schema))),
        ],
        "SELECT t.value, COUNT(*) AS c \
         FROM iceberg_t t JOIN iceberg_u u ON t.id = u.id \
         GROUP BY t.value ORDER BY t.value",
    )
    .await
    .expect("native profile must rewrite JOIN + GROUP BY + ORDER BY over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn cpu_style_profile_iceberg_group_by_order_by_rejected_by_leaf_guard() {
    // The CPU-style profile mirrors the pre-Phase-2 audit scenario: DataFusion
    // is allowed to repartition sorts so an unbounded GROUP BY + ORDER BY plan
    // produces a root SortPreservingMergeExec over the GPU-only
    // CudfTableFormatScanExec. The native rule rejects the full SPM shape, and
    // the Phase 1 leaf guard converts the leftover leaf into a structured
    // planning error so DataFusion never reaches the leaf's placeholder
    // execute() failure.
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let error = run_iceberg_sql_under_cpu_profile(
        &[("iceberg_t", plan)],
        "SELECT value, COUNT(*) AS c FROM iceberg_t GROUP BY value ORDER BY value",
    )
    .await
    .expect_err("CPU-style profile must fail in planning rather than at scan execution");

    let adapter_error = adapter_error_from(&error);
    let facts = collect_facts(adapter_error);
    assert_eq!(
        facts.get("fallback_reason").map(String::as_str),
        Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code()),
        "facts={facts:?}",
    );
    assert_eq!(facts.get("leftover_operator").map(String::as_str), Some("CudfTableFormatScanExec"), "facts={facts:?}",);
    assert_eq!(facts.get("table_format").map(String::as_str), Some("iceberg"), "facts={facts:?}");
    assert_eq!(facts.get("cpu_arrow_reader_fallback").map(String::as_str), Some("false"), "facts={facts:?}");

    // Phase 1 quotes the first native rejection reason when the guard fires
    // after a native rejection. Under the CPU-style profile the root is a full
    // SortPreservingMergeExec which the native rule rejects with
    // `native_sort_preserving_merge_unsupported`.
    assert_eq!(
        facts.get("first_native_rejection").map(String::as_str),
        Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()),
        "the guard must quote the SPM rejection that triggered the CPU fallback shape; facts={facts:?}",
    );

    let rendered = error.to_string();
    assert!(
        !rendered.contains("does not execute a CPU Arrow reader"),
        "the placeholder execute() error must not surface under the leaf guard; got: {rendered}",
    );
}

#[test]
fn leaf_guard_rejects_leftover_table_format_scan_under_unsupported_wrapper() {
    // The Phase 1 leaf guard is a `PhysicalOptimizerRule` that runs after the
    // native rewrite. This test exercises it directly with a non-rewritable
    // wrapper around a CudfTableFormatScanExec so the leaf survives any
    // upstream rewrite attempt. The wrapper here is a SortPreservingMergeExec
    // over the leaf, which is exactly the shape the audit identified as the
    // CPU-fallback trigger.
    let schema = iceberg_test_schema();
    let scan: Arc<dyn ExecutionPlan> =
        Arc::new(CudfTableFormatScanExec::new(native_admissible_scan_plan(Arc::clone(&schema)), schema));
    let ordering = datafusion::physical_expr::LexOrdering::new([datafusion::physical_expr::PhysicalSortExpr::new(
        datafusion::physical_expr::expressions::col("id", scan.schema().as_ref()).unwrap(),
        arrow::compute::SortOptions::new(false, false),
    )])
    .expect("non-empty LexOrdering");
    let wrapper: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, scan));

    let rule = CudfTableFormatGpuOnlyLeafGuardRule::new();
    assert_eq!(rule.name(), CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME);

    let error = rule
        .optimize(wrapper, &datafusion::common::config::ConfigOptions::new())
        .expect_err("leaf guard must reject a plan that retains a CudfTableFormatScanExec");

    let adapter_error = adapter_error_from(&error);
    let facts = collect_facts(adapter_error);
    assert_eq!(
        facts.get("fallback_reason").map(String::as_str),
        Some(FallbackReason::TableFormatGpuOnlyLeafNotRewritten.code()),
    );
    assert_eq!(facts.get("leftover_operator").map(String::as_str), Some("CudfTableFormatScanExec"));
    assert_eq!(facts.get("table_format").map(String::as_str), Some("iceberg"));
    assert_eq!(facts.get("cpu_arrow_reader_fallback").map(String::as_str), Some("false"));
    assert_eq!(facts.get("physical_plan_root").map(String::as_str), Some("SortPreservingMergeExec"));
    let shape = facts.get("physical_plan_shape").map(String::as_str).expect("shape fact");
    assert!(shape.contains("SortPreservingMergeExec("), "shape must reproduce the wrapper: {shape}");
    assert!(shape.contains("CudfTableFormatScanExec"), "shape must mention the surviving leaf: {shape}");
}

// ----- helpers ---------------------------------------------------------------

async fn run_iceberg_sql_under_native_profile(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let session_config = apply_session_profile(SessionConfig::new(), ServerSessionProfile::Native);
    run_iceberg_sql_with_session_config(tables, sql, session_config).await
}

async fn run_iceberg_sql_under_cpu_profile(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    // The CPU-fallback profile mirrors the `scripts/run_server.sh` legacy
    // setup the audit captured: DataFusion's defaults plus an explicit
    // multi-partition target so the EnforceDistribution rule actually injects
    // the repartition / SortPreservingMergeExec shape the native rule does not
    // yet support. `apply_session_profile(CpuFallback)` only flips
    // `information_schema`, so the rest is just DataFusion defaults.
    let mut base = SessionConfig::new().with_target_partitions(16);
    base = base.with_repartition_sorts(true);
    let session_config = apply_session_profile(base, ServerSessionProfile::CpuFallback);
    run_iceberg_sql_with_session_config(tables, sql, session_config).await
}

async fn run_iceberg_sql_with_session_config(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
    session_config: SessionConfig,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);

    for (name, plan) in tables {
        let schema = plan.logical_schema.clone();
        let planner = Arc::new(RecordingPlanner::new(plan.clone()));
        let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(schema, planner));
        ctx.register_table(*name, provider).expect("test should register a unique table name once");
    }

    let df = ctx.sql(sql).await?;
    df.create_physical_plan().await
}

fn assert_no_leftover_table_format_scan(plan: &dyn ExecutionPlan) {
    if find_table_format_scan(plan).is_some() {
        let rendered = displayable(plan).indent(false).to_string();
        panic!("native profile rewrite left a CudfTableFormatScanExec in place; plan:\n{rendered}");
    }
}

/// Phase 3 requires the *entire* native-friendly query shape to lower into a
/// single native pipeline rather than only rewriting the source leaf. The
/// optimizer rule does this by replacing the accepted subtree's root with a
/// [`NativeCudfExec`]; everything between that root and DataFusion's
/// post-planning wrappers must be free of unabsorbed CPU operators.
///
/// This walker therefore (1) descends through the thin post-rewrite wrappers
/// DataFusion can still leave above the native root (`CoalescePartitionsExec`,
/// projections, output coercion, etc.) until it finds the [`NativeCudfExec`]
/// boundary, then (2) verifies no aggregate / sort / join / repartition /
/// SortPreservingMergeExec node is sitting above that boundary. A surviving
/// `AggregateExec` or `SortPreservingMergeExec` would mean the native rule
/// only consumed the leaf — that is exactly the regression Phase 3 pins.
fn assert_query_fully_lowered_to_native(plan: &dyn ExecutionPlan) {
    let rendered = displayable(plan).indent(false).to_string();
    let native_root = find_native_cudf_exec_root(plan).unwrap_or_else(|| {
        panic!("native profile rewrite did not produce a NativeCudfExec; plan:\n{rendered}");
    });
    for forbidden in CPU_PIPELINE_OPERATORS_ABSORBED_BY_NATIVE_REWRITE {
        assert!(
            !native_root.ancestor_names.iter().any(|name| name == forbidden),
            "native profile rewrite left `{forbidden}` above the NativeCudfExec root; \
             the whole accepted subtree must collapse into a single native pipeline. plan:\n{rendered}",
        );
    }
}

const CPU_PIPELINE_OPERATORS_ABSORBED_BY_NATIVE_REWRITE: &[&str] =
    &["AggregateExec", "SortExec", "HashJoinExec", "RepartitionExec", "SortPreservingMergeExec"];

struct NativeRootDescent {
    ancestor_names: Vec<String>,
}

fn find_native_cudf_exec_root(plan: &dyn ExecutionPlan) -> Option<NativeRootDescent> {
    let mut ancestor_names = Vec::new();
    let mut cursor: &dyn ExecutionPlan = plan;
    loop {
        if cursor.name() == "NativeCudfExec" {
            return Some(NativeRootDescent { ancestor_names });
        }
        ancestor_names.push(cursor.name().to_string());
        let children = cursor.children();
        if children.len() != 1 {
            // Multi-input or leaf nodes that are not the native root mean the
            // rewrite either did not produce a NativeCudfExec at all or did
            // not collapse the accepted subtree into a single pipeline.
            return None;
        }
        cursor = children[0].as_ref();
    }
}

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.as_any().downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
}

fn adapter_error_from(error: &DataFusionError) -> &datafusion_cudf::Error {
    // `DataFusionError::Context` wraps the inner planning error with the
    // optimizer rule name; recurse into it before downcasting.
    if let DataFusionError::Context(_, inner) = error {
        return adapter_error_from(inner);
    }
    let DataFusionError::External(boxed) = error else {
        panic!("expected datafusion_cudf::Error to cross DataFusion as `External`, got: {error:?}");
    };
    boxed
        .downcast_ref::<datafusion_cudf::Error>()
        .expect("external errors emitted by datafusion-cudf must downcast to datafusion_cudf::Error")
}

fn collect_facts(error: &datafusion_cudf::Error) -> std::collections::BTreeMap<&'static str, String> {
    error.facts().iter().map(|(key, value)| (key, value.to_string())).collect()
}

fn iceberg_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("value", DataType::Utf8, true)]))
}

fn native_admissible_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            Some(1),
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            Some(2),
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

#[derive(Debug)]
struct RecordingPlanner {
    plan: TableFormatScanPlan,
    requests: Mutex<Vec<TableFormatScanRequest>>,
}

impl RecordingPlanner {
    fn new(plan: TableFormatScanPlan) -> Self {
        Self { plan, requests: Mutex::new(Vec::new()) }
    }
}

impl TableFormatPlanner for RecordingPlanner {
    fn plan_scan<'a>(
        &'a self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'a, datafusion_cudf::Result<TableFormatScanPlan>> {
        let scan_limit = request.scan_limit;
        self.requests.lock().expect("request recorder mutex poisoned").push(request);
        let plan = self.plan.clone().with_scan_limit(scan_limit);
        Box::pin(async move { Ok(plan) })
    }
}

// Phase 1 retained a sibling reference on the leaf guard so emitted facts can
// quote the first native rejection. The `last_report` accessor is the dependency
// the guard reads at fire time; the assertion below pins the source of truth so
// future refactors that drop the accessor surface in CI.
#[allow(dead_code)]
fn _last_report_accessor_is_present(rule: Arc<CudfNativePhysicalOptimizerRule>) {
    let _ = rule.last_report();
}
