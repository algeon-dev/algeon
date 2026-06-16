//! Milestone 3 regression tests for the Flight SQL Iceberg fallback fix
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
//!   leftover GPU-only leaf is caught by the Milestone 1 leaf guard before
//!   `execute()`.
//! - A leaf-guard unit test asserts that a leftover `CudfTableFormatScanExec`
//!   beneath a non-rewritable wrapper produces the structured
//!   `table_format_gpu_only_leaf_not_rewritten` error.
//!
//! All tests are credential-free; the Iceberg provider is driven by a
//! [`TableFormatPlanner`] returning a hand-built native-admissible
//! [`TableFormatScanPlan`], so no AWS Glue/S3 access is required.

#![cfg(all(feature = "iceberg", feature = "server"))]

use std::{
    fs::{self, File},
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::DataFusionError,
    execution::SessionStateBuilder,
    parquet::arrow::ArrowWriter,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, displayable, sorts::sort_preserving_merge::SortPreservingMergeExec},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
    CudfSessionStateBuilderExt, CudfTableFormatGpuOnlyLeafGuardRule, CudfTableFormatScanExec, FallbackReason,
    GpuFallbackPolicy, TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
    TableFormatDataFile, TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability, TableFormatScanPlan,
    TableFormatScanRequest,
    iceberg::CudfIcebergTableProvider,
    server::{FlightSqlNativeRejectClass, ServerSessionProfile, apply_session_profile, log_leaf_guard_rejection},
    table_format::scan::{TableFormatPredicate, TableFormatPredicateOp},
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
async fn native_profile_iceberg_wide_aggregate_order_limit_rewrites_as_native_root() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT value, COUNT(*) AS rows, SUM(id) AS total_id, MIN(id) AS min_id, MAX(id) AS max_id \
         FROM iceberg_t \
         GROUP BY value \
         ORDER BY total_id DESC, value ASC \
         LIMIT 5",
    )
    .await
    .expect("native profile must rewrite wide aggregate ORDER BY LIMIT over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_root_is_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_q1_style_cast_round_aggregate_order_limit_rewrites_as_native_root() {
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", q1_style_wide_aggregate_scan_plan())],
        "SELECT COALESCE(idp,'(none)') AS idp_key, \
                COUNT(*) AS n, \
                COUNT(website) AS n_web, \
                CAST(ROUND(AVG(\"crsScore\")*10000) AS BIGINT) AS avg_score_e4, \
                MIN(\"foundedYear\") AS min_year, \
                MAX(\"foundedYear\") AS max_year, \
                SUM(\"supportApi\") AS sum_api \
         FROM iceberg_t \
         WHERE \"foundedYear\" IS NOT NULL \
         GROUP BY COALESCE(idp,'(none)') \
         ORDER BY n DESC, idp_key \
         LIMIT 15",
    )
    .await
    .expect("native profile must rewrite cast/round aggregate ORDER BY LIMIT over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_root_is_native(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "ProjectionExec");
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "SortExec");
}

#[tokio::test]
async fn native_profile_iceberg_cte_aggregate_join_back_rewrites_as_native_root() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "WITH c AS (SELECT value, COUNT(*) AS cnt FROM iceberg_t GROUP BY value) \
         SELECT t.value, COUNT(*) AS rows, MAX(c.cnt) AS max_cnt \
         FROM iceberg_t t JOIN c ON t.value = c.value \
         WHERE c.cnt > 0 \
         GROUP BY t.value \
         ORDER BY max_cnt DESC, t.value ASC \
         LIMIT 5",
    )
    .await
    .expect("native profile must rewrite CTE aggregate joined back to Iceberg fact");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_root_is_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_not_in_count_distinct_keeps_safe_host_boundary() {
    let schema = iceberg_test_schema();
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[
            ("iceberg_t", native_admissible_scan_plan(Arc::clone(&schema))),
            ("iceberg_u", native_admissible_scan_plan(Arc::clone(&schema))),
        ],
        "SELECT COUNT(DISTINCT t.value) AS distinct_values \
         FROM iceberg_t t \
         WHERE t.id NOT IN (SELECT u.id FROM iceberg_u u WHERE u.value IS NOT NULL)",
    )
    .await
    .expect("native profile must plan NOT IN plus COUNT(DISTINCT) over Iceberg without a leftover GPU-only leaf");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_plan_contains_native_cudf_exec(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "NativeCudfGpuProducerExec");
}

#[tokio::test]
async fn native_profile_iceberg_two_derived_aggregates_joined_to_fact_rewrites_as_native_root() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "WITH high AS (SELECT value, COUNT(*) AS high_cnt FROM iceberg_t WHERE id >= 0 GROUP BY value), \
              low AS (SELECT value, SUM(id) AS low_sum FROM iceberg_t WHERE id < 100 GROUP BY value) \
         SELECT t.value, COUNT(*) AS rows, MAX(high.high_cnt) AS high_cnt, MAX(low.low_sum) AS low_sum \
         FROM iceberg_t t \
         JOIN high ON t.value = high.value \
         JOIN low ON t.value = low.value \
         GROUP BY t.value \
         ORDER BY rows DESC, t.value ASC \
         LIMIT 5",
    )
    .await
    .expect("native profile must rewrite fact joined to two derived aggregate branches");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_root_is_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_cross_table_in_semijoin_rewrites_as_native_root() {
    let schema = iceberg_test_schema();
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[
            ("iceberg_t", native_admissible_scan_plan(Arc::clone(&schema))),
            ("iceberg_u", native_admissible_scan_plan(Arc::clone(&schema))),
        ],
        "SELECT t.id, t.value \
         FROM iceberg_t t \
         WHERE t.id IN (SELECT u.id FROM iceberg_u u WHERE u.value IS NOT NULL)",
    )
    .await
    .expect("native profile must rewrite cross-table IN semijoin over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_root_is_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_not_exists_anti_join_rewrites_with_no_leftover_leaf() {
    let schema = iceberg_test_schema();
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[
            ("iceberg_t", native_admissible_scan_plan(Arc::clone(&schema))),
            ("iceberg_u", native_admissible_scan_plan(Arc::clone(&schema))),
        ],
        "SELECT COUNT(*) AS n \
         FROM iceberg_t t \
         WHERE NOT EXISTS (SELECT 1 FROM iceberg_u u WHERE u.id = t.id)",
    )
    .await
    .expect("native profile must rewrite NOT EXISTS anti-join over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_union_all_aggregate_branches_rewrites_with_no_leftover_leaf() {
    let schema = iceberg_test_schema();
    let plan = high_empty_low_nonempty_scan_plan(Arc::clone(&schema));
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT 'high' AS bucket, COUNT(*) AS n FROM iceberg_t WHERE id >= 80 \
         UNION ALL \
         SELECT 'low' AS bucket, COUNT(*) AS n FROM iceberg_t WHERE id < 50 \
         ORDER BY bucket",
    )
    .await
    .expect("native profile must rewrite UNION ALL aggregate branches over Iceberg with an empty branch");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_correlated_scalar_avg_rewrites_with_no_leftover_leaf() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT COUNT(*) AS n \
         FROM iceberg_t s1 \
         WHERE s1.value IS NOT NULL \
           AND s1.id > (SELECT AVG(s2.id) FROM iceberg_t s2 WHERE s2.value = s1.value)",
    )
    .await
    .expect("native profile must rewrite correlated scalar AVG over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_correlated_scalar_max_rewrites_with_no_leftover_leaf() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT COUNT(*) AS n \
         FROM iceberg_t s \
         WHERE s.value IS NOT NULL \
           AND s.id = (SELECT MAX(s2.id) FROM iceberg_t s2 WHERE s2.value = s.value)",
    )
    .await
    .expect("native profile must rewrite correlated scalar MAX over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_cte_aggregate_scalar_filter_plans_with_no_leftover_leaf() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "WITH c AS (SELECT value, COUNT(*) AS cnt FROM iceberg_t GROUP BY value) \
         SELECT COUNT(*) AS n FROM c WHERE cnt > (SELECT AVG(cnt) FROM c)",
    )
    .await
    .expect("native profile must rewrite CTE aggregate scalar filter over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_plan_contains_native_cudf_exec(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "ScalarSubqueryExec");
}

#[tokio::test]
async fn native_profile_iceberg_physical_scalar_subquery_rewrites_children_with_no_leftover_leaf() {
    let schema = iceberg_test_schema();
    let mut session_config = apply_session_profile(SessionConfig::new(), ServerSessionProfile::Native);
    session_config.options_mut().optimizer.enable_physical_uncorrelated_scalar_subquery = true;
    let physical_plan = run_iceberg_sql_with_session_config(
        &[
            ("iceberg_t", native_admissible_scan_plan(Arc::clone(&schema))),
            ("iceberg_u", native_admissible_scan_plan(schema)),
        ],
        "SELECT value, SUM(id) AS total_id \
         FROM iceberg_t \
         GROUP BY value \
         HAVING SUM(id) > (SELECT SUM(id) FROM iceberg_u) \
         ORDER BY total_id DESC",
        session_config,
    )
    .await
    .expect("native profile must rewrite table-format children under physical scalar subqueries");

    assert!(
        plan_contains_operator(physical_plan.as_ref(), "ScalarSubqueryExec"),
        "test must exercise DataFusion's physical scalar-subquery wrapper"
    );
    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_plan_contains_native_cudf_exec(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_scalar_subquery_scope_skips_mixed_regular_parquet_branch() {
    let dir = unique_temp_dir("datafusion-nexus-iceberg-scalar-scope");
    let parquet_path = dir.join("regular.parquet");
    write_scalar_subquery_probe_parquet(&parquet_path);
    let mut session_config = apply_session_profile(SessionConfig::new(), ServerSessionProfile::Native);
    session_config.options_mut().optimizer.enable_physical_uncorrelated_scalar_subquery = true;
    let physical_plan = run_mixed_iceberg_parquet_sql_with_session_config(
        &[
            ("iceberg_t", native_admissible_scan_plan(iceberg_test_schema())),
            ("iceberg_u", native_admissible_scan_plan(iceberg_test_schema())),
        ],
        &[("regular_t", parquet_path.as_path())],
        "SELECT value, SUM(id) AS total_id \
         FROM iceberg_t \
         GROUP BY value \
         HAVING SUM(id) > ( \
             SELECT SUM(x) FROM ( \
                 SELECT id AS x FROM iceberg_u \
                 UNION ALL \
                 SELECT m00 AS x FROM regular_t \
             ) mixed_scalar_input \
         ) \
         ORDER BY total_id DESC",
        session_config,
    )
    .await
    .expect("native profile must rewrite only table-format children under mixed physical scalar subqueries");

    assert!(
        plan_contains_operator(physical_plan.as_ref(), "ScalarSubqueryExec"),
        "test must exercise DataFusion's physical scalar-subquery wrapper"
    );
    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_plan_contains_native_cudf_exec(physical_plan.as_ref());
    let scalar_subquery = find_operator(physical_plan.as_ref(), "ScalarSubqueryExec")
        .expect("physical scalar subquery should remain in the optimized plan");
    let regular_parquet_subquery = scalar_subquery
        .children()
        .into_iter()
        .find(|child| plan_contains_operator(child.as_ref(), "DataSourceExec"))
        .expect("regular Parquet branch should remain under the mixed scalar-subquery child");
    assert!(
        plan_contains_operator(regular_parquet_subquery.as_ref(), "DataSourceExec"),
        "regular Parquet branch should remain a DataFusion subtree even when its sibling table-format branch is rewritten:\n{}",
        displayable(physical_plan.as_ref()).indent(false)
    );

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cpu_style_profile_iceberg_group_by_order_by_rejected_by_leaf_guard() {
    // The CPU-style profile mirrors the pre-Milestone-2 audit scenario: DataFusion
    // is allowed to repartition sorts so an unbounded GROUP BY + ORDER BY plan
    // produces a root SortPreservingMergeExec over the GPU-only
    // CudfTableFormatScanExec. The native rule rejects the full SPM shape, and
    // the Milestone 1 leaf guard converts the leftover leaf into a structured
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

    // Milestone 1 quotes native rule rejection context when the guard fires
    // after a native rejection. Under the CPU-style profile the root is a full
    // SortPreservingMergeExec, which the native rule rejects with
    // `native_sort_preserving_merge_unsupported`.
    assert_eq!(
        facts.get("native_rule_blocking_rejection").map(String::as_str),
        Some(FallbackReason::NativeSortPreservingMergeUnsupported.code()),
        "the guard must quote the SPM rejection that triggered the CPU fallback shape; facts={facts:?}",
    );

    let rendered = error.to_string();
    assert!(
        !rendered.contains("does not execute a CPU Arrow reader"),
        "the placeholder execute() error must not surface under the leaf guard; got: {rendered}",
    );
}

#[tokio::test]
async fn cpu_style_profile_rejection_is_recognised_by_diagnostics_helper() {
    // Milestone 4 ships a Flight SQL diagnostics helper that the server uses to
    // emit a `tracing::warn!` line when a query fails because the Milestone 1
    // GPU-only leaf guard fired. The macro itself is hard to assert on
    // without a tracing-subscriber, but the helper is a thin wrapper around
    // `log_leaf_guard_rejection` which calls a `pub(crate)`
    // `leaf_guard_event_fields` extractor exercised by `server::diagnostics`
    // unit tests. Here we pin the end-to-end contract: the same CPU-style
    // session profile that triggers the audit shape produces an error the
    // public helper recognises (does not panic + classifies correctly).
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let error = run_iceberg_sql_under_cpu_profile(
        &[("iceberg_t", plan)],
        "SELECT value, COUNT(*) AS c FROM iceberg_t GROUP BY value ORDER BY value",
    )
    .await
    .expect_err("CPU-style profile must fail in planning rather than at scan execution");

    log_leaf_guard_rejection("test-query-id", &error, ServerSessionProfile::CpuFallback);

    let adapter_error = adapter_error_from(&error);
    let facts = collect_facts(adapter_error);
    let bucket = FlightSqlNativeRejectClass::from_reason(FallbackReason::TableFormatGpuOnlyLeafNotRewritten);
    assert_eq!(
        bucket.as_str(),
        facts.get("fallback_reason").map(String::as_str).unwrap(),
        "the leaf-guard rejection bucket label must equal the structured fallback_reason",
    );
}

#[test]
fn leaf_guard_rejects_leftover_table_format_scan_under_unsupported_wrapper() {
    // The Milestone 1 leaf guard is a `PhysicalOptimizerRule` that runs after the
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
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
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

async fn run_mixed_iceberg_parquet_sql_with_session_config(
    iceberg_tables: &[(&str, TableFormatScanPlan)],
    parquet_tables: &[(&str, &Path)],
    sql: &str,
    session_config: SessionConfig,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);

    for (name, plan) in iceberg_tables {
        let schema = plan.logical_schema.clone();
        let planner = Arc::new(RecordingPlanner::new(plan.clone()));
        let provider = Arc::new(CudfIcebergTableProvider::from_table_format_planner(schema, planner));
        ctx.register_table(*name, provider).expect("test should register a unique table name once");
    }
    for (name, path) in parquet_tables {
        ctx.register_parquet(*name, path.to_str().expect("test path should be utf-8"), ParquetReadOptions::default())
            .await?;
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

/// Milestone 3 requires the *entire* native-friendly query shape to lower into a
/// single native pipeline rather than only rewriting the source leaf. The
/// optimizer rule does this by replacing the accepted subtree's root with a
/// native execution wrapper; everything between that root and DataFusion's
/// post-planning wrappers must be free of unabsorbed CPU operators.
///
/// This walker therefore (1) descends through the thin post-rewrite wrappers
/// DataFusion can still leave above the native root (`CoalescePartitionsExec`,
/// projections, output coercion, etc.) until it finds a native execution
/// boundary, then (2) verifies no aggregate / sort / join / repartition /
/// SortPreservingMergeExec node is sitting above that boundary. A surviving
/// `AggregateExec` or `SortPreservingMergeExec` would mean the native rule
/// only consumed the leaf — that is exactly the regression Milestone 3 pins.
fn assert_query_fully_lowered_to_native(plan: &dyn ExecutionPlan) {
    let rendered = displayable(plan).indent(false).to_string();
    let native_root = find_native_cudf_exec_root(plan).unwrap_or_else(|| {
        panic!("native profile rewrite did not produce a native execution wrapper; plan:\n{rendered}");
    });
    for forbidden in CPU_PIPELINE_OPERATORS_ABSORBED_BY_NATIVE_REWRITE {
        assert!(
            !native_root.ancestor_names.iter().any(|name| name == forbidden),
            "native profile rewrite left `{forbidden}` above the native execution root; \
             the whole accepted subtree must collapse into a single native pipeline. plan:\n{rendered}",
        );
    }
}

const CPU_PIPELINE_OPERATORS_ABSORBED_BY_NATIVE_REWRITE: &[&str] =
    &["AggregateExec", "SortExec", "HashJoinExec", "RepartitionExec", "SortPreservingMergeExec"];

fn assert_plan_contains_native_cudf_exec(plan: &dyn ExecutionPlan) {
    if !plan_contains_operator(plan, "NativeCudfExec") && !plan_contains_operator(plan, "NativeCudfComposedExec") {
        let rendered = displayable(plan).indent(false).to_string();
        panic!("native profile rewrite did not produce any native execution wrapper; plan:\n{rendered}");
    }
}

fn assert_plan_does_not_contain_operator(plan: &dyn ExecutionPlan, operator: &str) {
    if plan_contains_operator(plan, operator) {
        let rendered = displayable(plan).indent(false).to_string();
        panic!("native profile rewrite left `{operator}` in the plan; plan:\n{rendered}");
    }
}

fn assert_query_root_is_native(plan: &dyn ExecutionPlan) {
    if !matches!(plan.name(), "NativeCudfExec" | "NativeCudfComposedExec") {
        let rendered = displayable(plan).indent(false).to_string();
        panic!(
            "native profile rewrite left a DataFusion host boundary above the native plan root; \
             expected a native root wrapper, got `{}`. plan:\n{rendered}",
            plan.name()
        );
    }
}

fn plan_contains_operator(plan: &dyn ExecutionPlan, operator: &str) -> bool {
    plan.name() == operator || plan.children().into_iter().any(|child| plan_contains_operator(child.as_ref(), operator))
}

fn find_operator<'a>(plan: &'a dyn ExecutionPlan, operator: &str) -> Option<&'a dyn ExecutionPlan> {
    if plan.name() == operator {
        return Some(plan);
    }
    plan.children().into_iter().find_map(|child| find_operator(child.as_ref(), operator))
}

struct NativeRootDescent {
    ancestor_names: Vec<String>,
}

fn find_native_cudf_exec_root(plan: &dyn ExecutionPlan) -> Option<NativeRootDescent> {
    let mut ancestor_names = Vec::new();
    let mut cursor: &dyn ExecutionPlan = plan;
    loop {
        if matches!(cursor.name(), "NativeCudfExec" | "NativeCudfComposedExec") {
            return Some(NativeRootDescent { ancestor_names });
        }
        ancestor_names.push(cursor.name().to_string());
        let children = cursor.children();
        if children.len() != 1 {
            // Multi-input or leaf nodes that are not the native root mean the
            // rewrite either did not produce a native execution wrapper at all
            // or did not collapse the accepted subtree into a single pipeline.
            return None;
        }
        cursor = children[0].as_ref();
    }
}

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&CudfTableFormatScanExec> {
    if let Some(scan) = plan.downcast_ref::<CudfTableFormatScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
}

fn adapter_error_from(error: &DataFusionError) -> &datafusion_nexus::Error {
    // `DataFusionError::Context` wraps the inner planning error with the
    // optimizer rule name; recurse into it before downcasting.
    if let DataFusionError::Context(_, inner) = error {
        return adapter_error_from(inner);
    }
    let DataFusionError::External(boxed) = error else {
        panic!("expected datafusion_nexus::Error to cross DataFusion as `External`, got: {error:?}");
    };
    boxed
        .downcast_ref::<datafusion_nexus::Error>()
        .expect("external errors emitted by datafusion-nexus must downcast to datafusion_nexus::Error")
}

fn collect_facts(error: &datafusion_nexus::Error) -> std::collections::BTreeMap<&'static str, String> {
    error.facts().iter().map(|(key, value)| (key, value.to_string())).collect()
}

fn iceberg_test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("value", DataType::Utf8, true)]))
}

fn write_scalar_subquery_probe_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
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
            None,
            "id",
            Some("id"),
            DataType::Int64,
            false,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "value",
            Some("value"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

fn q1_style_wide_aggregate_scan_plan() -> TableFormatScanPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("idp", DataType::Utf8, true),
        Field::new("website", DataType::Utf8, true),
        Field::new("crsScore", DataType::Float64, true),
        Field::new("foundedYear", DataType::Int32, true),
        Field::new("supportApi", DataType::Int64, true),
    ]));
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/rcg4-q1-style.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(11),
        )
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "idp",
            Some("idp"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "website",
            Some("website"),
            DataType::Utf8,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "crsScore",
            Some("crsScore"),
            DataType::Float64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "foundedYear",
            Some("foundedYear"),
            DataType::Int32,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
        .with_projected_column(TableFormatColumnMapping::new(
            None,
            "supportApi",
            Some("supportApi"),
            DataType::Int64,
            true,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::Projected,
        ))
}

fn empty_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("12345")
        .with_catalog_type("local")
        .with_logical_schema(Arc::clone(&schema))
        .with_native_source_schema(schema)
}

fn high_empty_low_nonempty_scan_plan(schema: Arc<Schema>) -> TableFormatScanPlan {
    native_admissible_scan_plan(schema)
        .with_data_file(
            TableFormatDataFile::new(
                "data-0002",
                "/tmp/iceberg/data-0002.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(4096)
            .with_row_count(7),
        )
        .with_snapshot_id("high-empty")
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
    fn plan_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::Result<TableFormatScanPlan>> {
        let scan_limit = request.scan_limit;
        let high_branch_pruned = self.plan.snapshot_id.as_deref() == Some("high-empty")
            && matches!(
                &request.predicate,
                Some(TableFormatPredicate::Comparison {
                    column,
                    op: TableFormatPredicateOp::GtEq,
                    ..
                }) if column == "id"
            );
        let empty_plan = high_branch_pruned.then(|| empty_scan_plan(Arc::clone(&self.plan.logical_schema)));
        self.requests.lock().expect("request recorder mutex poisoned").push(request);
        let plan = empty_plan.unwrap_or_else(|| self.plan.clone()).with_scan_limit(scan_limit);
        Box::pin(async move { Ok(plan) })
    }
}

// Milestone 1 retained a sibling reference on the leaf guard so emitted facts can
// quote the first native rejection. The `last_report` accessor is the dependency
// the guard reads at fire time; the assertion below pins the source of truth so
// future refactors that drop the accessor surface in CI.
#[allow(dead_code)]
fn _last_report_accessor_is_present(rule: Arc<CudfNativePhysicalOptimizerRule>) {
    let _ = rule.last_report();
}
