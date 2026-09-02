//! Regression tests for Flight SQL Iceberg native planning and CPU execution.
//!
//! These tests pin the exact regression surface for two scenarios:
//!
//! - Under DataFusion's standard session configuration, Iceberg-backed
//!   `GROUP BY ... ORDER BY` and `JOIN + GROUP BY + ORDER BY` queries must
//!   rewrite to native and leave no `NexusIcebergScanExec` behind.
//! - Under an explicit multi-partition configuration with DataFusion sort
//!   repartitioning enabled, the same `GROUP BY ... ORDER BY` shape includes a full
//!   `SortPreservingMergeExec`; native lowering must absorb that SPM as an
//!   explicit native sort and still leave no executable Iceberg wrapper behind.
//!
//! All tests are credential-free; the Iceberg provider is driven by a
//! [`TableFormatPlanner`] returning a hand-built native-admissible
//! [`TableFormatScanPlan`], so no AWS Glue/S3 access is required.

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
    datasource::{MemTable, TableProvider},
    execution::SessionStateBuilder,
    parquet::arrow::ArrowWriter,
    physical_plan::{ExecutionPlan, displayable},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    planner::{
        FinalPlanDisposition, NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule, classify_final_plan,
    },
    session::NexusSessionStateBuilderExt,
    table_format::{
        NexusIcebergScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
        TableFormatDataFile, TableFormatDataFileFormat, TableFormatPlanner, TableFormatReadCapability,
        TableFormatScanPlan, TableFormatScanRequest,
        iceberg::NexusIcebergTableProvider,
        scan::{TableFormatPredicate, TableFormatPredicateOp},
    },
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
async fn native_profile_iceberg_lead_window_rewrites_with_no_leftover_leaf() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT id, LEAD(id) OVER (ORDER BY id) AS next_id FROM iceberg_t",
    )
    .await
    .expect("native profile must rewrite LEAD over Iceberg");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

#[tokio::test]
async fn native_profile_iceberg_join_group_by_order_by_rewrites_with_no_leftover_leaf() {
    // Both inputs share the same logical schema (id BIGINT NOT NULL, value
    // VARCHAR). The native-friendly profile pins target_partitions=1, so the
    // hash join, grouped aggregate, and final sort all collapse onto a single
    // native pipeline rooted at NexusNativeExec instead of being split across
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
async fn native_profile_iceberg_cte_aggregate_join_back_fully_lowers_to_native() {
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
    .expect("native profile must rewrite native-safe CTE aggregate join back");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "HashJoinExec");
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "SortExec");
}

#[tokio::test]
async fn native_profile_iceberg_not_in_count_distinct_remains_executable_on_datafusion() {
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
    .expect("unlowerable NOT IN plus COUNT(DISTINCT) must retain an executable DataFusion plan");

    assert_executable_iceberg_datafusion_plan(physical_plan.as_ref());
    assert_plan_contains_operator(physical_plan.as_ref(), "HashJoinExec");
    assert_plan_contains_operator(physical_plan.as_ref(), "AggregateExec");
}

#[tokio::test]
async fn native_profile_iceberg_two_derived_aggregates_joined_to_fact_fully_lower_to_native() {
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
    .expect("native profile must rewrite native-safe aggregate branches");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "HashJoinExec");
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "SortExec");
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
async fn native_profile_iceberg_union_all_aggregate_branches_remain_executable_on_datafusion() {
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
    .expect("unlowerable UNION ALL aggregate branches must retain an executable DataFusion plan");

    assert_executable_iceberg_datafusion_plan(physical_plan.as_ref());
    assert_plan_contains_operator(physical_plan.as_ref(), "UnionExec");
    assert_plan_contains_operator(physical_plan.as_ref(), "SortPreservingMergeExec");
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
async fn native_profile_iceberg_correlated_scalar_max_fully_lowers_to_native() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "SELECT COUNT(*) AS n \
         FROM iceberg_t s \
         WHERE s.value IS NOT NULL \
           AND s.id = (SELECT MAX(s2.id) FROM iceberg_t s2 WHERE s2.value = s.value)",
    )
    .await
    .expect("native profile must rewrite native-safe correlated scalar MAX fragments");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "HashJoinExec");
    assert_plan_does_not_contain_operator(physical_plan.as_ref(), "AggregateExec");
}

#[tokio::test]
async fn native_profile_iceberg_cte_aggregate_scalar_filter_remains_executable_on_datafusion() {
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_native_profile(
        &[("iceberg_t", plan)],
        "WITH c AS (SELECT value, COUNT(*) AS cnt FROM iceberg_t GROUP BY value) \
         SELECT COUNT(*) AS n FROM c WHERE cnt > (SELECT AVG(cnt) FROM c)",
    )
    .await
    .expect("unlowerable CTE scalar filter must retain an executable DataFusion plan");

    assert_executable_iceberg_datafusion_plan(physical_plan.as_ref());
    assert_plan_contains_operator(physical_plan.as_ref(), "ScalarSubqueryExec");
    assert_plan_contains_operator(physical_plan.as_ref(), "AggregateExec");
}

#[tokio::test]
async fn physical_scalar_subquery_retains_executable_iceberg_datafusion_plan() {
    let schema = iceberg_test_schema();
    let mut session_config = SessionConfig::new();
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
    .expect("physical scalar subqueries must retain an executable DataFusion Iceberg plan");

    assert!(
        plan_contains_operator(physical_plan.as_ref(), "ScalarSubqueryExec"),
        "test must exercise DataFusion's physical scalar-subquery wrapper"
    );
    assert_executable_iceberg_datafusion_plan(physical_plan.as_ref());
}

#[tokio::test]
async fn physical_scalar_subquery_with_mixed_parquet_retains_executable_datafusion_plan() {
    let dir = unique_temp_dir("nexus-iceberg-scalar-scope");
    let parquet_path = dir.join("regular.parquet");
    write_scalar_subquery_probe_parquet(&parquet_path);
    let mut session_config = SessionConfig::new();
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
    .expect("mixed physical scalar subqueries must retain an executable DataFusion Iceberg plan");

    assert!(
        plan_contains_operator(physical_plan.as_ref(), "ScalarSubqueryExec"),
        "test must exercise DataFusion's physical scalar-subquery wrapper"
    );
    assert_executable_iceberg_datafusion_plan(physical_plan.as_ref());
    let scalar_subquery = find_operator(physical_plan.as_ref(), "ScalarSubqueryExec")
        .expect("physical scalar subquery should remain in the optimized plan");
    let regular_parquet_subquery = scalar_subquery
        .children()
        .into_iter()
        .find(|child| plan_contains_operator(child.as_ref(), "DataSourceExec"))
        .expect("regular Parquet branch should remain under the mixed scalar-subquery child");
    assert!(
        plan_contains_operator(regular_parquet_subquery.as_ref(), "DataSourceExec"),
        "regular Parquet branch should remain a DataFusion subtree alongside the executable Iceberg leaf:\n{}",
        displayable(physical_plan.as_ref()).indent(false)
    );

    fs::remove_dir_all(dir).ok();
}

#[tokio::test]
async fn cpu_style_profile_iceberg_group_by_order_by_full_spm_lowers_to_native_sort() {
    // This explicit multi-partition setup lets DataFusion repartition sorts so
    // an unbounded GROUP BY +
    // ORDER BY plan produces a root SortPreservingMergeExec over the Iceberg
    // scan wrapper. Full SPM is lowerable as an explicit native sort, so this
    // configuration should rewrite the whole subtree.
    let plan = native_admissible_scan_plan(iceberg_test_schema());
    let physical_plan = run_iceberg_sql_under_cpu_profile(
        &[("iceberg_t", plan)],
        "SELECT value, COUNT(*) AS c FROM iceberg_t GROUP BY value ORDER BY value",
    )
    .await
    .expect("multi-partition configuration should rewrite GROUP BY + ORDER BY over Iceberg after full SPM support");

    assert_no_leftover_table_format_scan(physical_plan.as_ref());
    assert_query_fully_lowered_to_native(physical_plan.as_ref());
}

// ----- helpers ---------------------------------------------------------------

async fn run_iceberg_sql_under_native_profile(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    run_iceberg_sql_with_session_config(tables, sql, SessionConfig::new()).await
}

async fn run_iceberg_sql_under_cpu_profile(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    // The explicit multi-partition target makes EnforceDistribution inject the
    // repartition / SortPreservingMergeExec shape.
    let mut base = SessionConfig::new().with_target_partitions(16);
    base = base.with_repartition_sorts(true);
    run_iceberg_sql_with_session_config(tables, sql, base).await
}

async fn run_iceberg_sql_with_session_config(
    tables: &[(&str, TableFormatScanPlan)],
    sql: &str,
    session_config: SessionConfig,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let optimizer_config = NexusNativeOptimizerConfig::default();
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);

    for (name, plan) in tables {
        let schema = plan.logical_schema.clone();
        let planner = Arc::new(RecordingPlanner::new(plan.clone()));
        let provider = Arc::new(test_iceberg_provider(schema, planner));
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
    let optimizer_config = NexusNativeOptimizerConfig::default();
    let native_rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_nexus_native_rule(Arc::clone(&native_rule))
        .build();
    let ctx = SessionContext::new_with_state(state);

    for (name, plan) in iceberg_tables {
        let schema = plan.logical_schema.clone();
        let planner = Arc::new(RecordingPlanner::new(plan.clone()));
        let provider = Arc::new(test_iceberg_provider(schema, planner));
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
        panic!("native profile rewrite left a NexusIcebergScanExec in place; plan:\n{rendered}");
    }
}

fn assert_executable_iceberg_datafusion_plan(plan: &dyn ExecutionPlan) {
    let rendered = displayable(plan).indent(false).to_string();
    assert!(
        find_table_format_scan(plan).is_some() || plan_contains_operator(plan, "PlaceholderRowExec"),
        "DataFusion fallback plan must retain the executable Iceberg wrapper; plan:\n{rendered}"
    );
    assert_eq!(classify_final_plan(plan), FinalPlanDisposition::DataFusion, "plan:\n{rendered}");
    assert!(!plan_contains_operator(plan, "NexusNativeExec"), "plan:\n{rendered}");
    assert!(!plan_contains_operator(plan, "NexusNativeComposedExec"), "plan:\n{rendered}");
    assert!(!plan_contains_operator(plan, "NexusNativeGpuProducerExec"), "plan:\n{rendered}");
}

/// The native-lowering contract requires the *entire* native-friendly query shape to lower into a
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
/// only consumed the leaf — that is the regression this test pins.
fn assert_query_fully_lowered_to_native(plan: &dyn ExecutionPlan) {
    let rendered = displayable(plan).indent(false).to_string();
    let native_root = find_nexus_native_exec_root(plan).unwrap_or_else(|| {
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

fn assert_plan_does_not_contain_operator(plan: &dyn ExecutionPlan, operator: &str) {
    if plan_contains_operator(plan, operator) {
        let rendered = displayable(plan).indent(false).to_string();
        panic!("native profile rewrite left `{operator}` in the plan; plan:\n{rendered}");
    }
}

fn assert_plan_contains_operator(plan: &dyn ExecutionPlan, operator: &str) {
    if !plan_contains_operator(plan, operator) {
        let rendered = displayable(plan).indent(false).to_string();
        panic!("native profile rewrite did not include `{operator}` in the plan; plan:\n{rendered}");
    }
}

fn assert_query_root_is_native(plan: &dyn ExecutionPlan) {
    if !matches!(plan.name(), "NexusNativeExec" | "NexusNativeComposedExec") {
        let rendered = displayable(plan).indent(false).to_string();
        panic!(
            "native profile rewrite left a DataFusion host boundary above the native plan root; \
             expected a native root wrapper, got `{}`. plan:\n{rendered}",
            plan.name()
        );
    }
}

fn plan_contains_operator(plan: &dyn ExecutionPlan, operator: &str) -> bool {
    plan.name() == operator
        || plan.name().strip_prefix(operator).is_some_and(|suffix| suffix.starts_with('('))
        || plan.children().into_iter().any(|child| plan_contains_operator(child.as_ref(), operator))
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

fn find_nexus_native_exec_root(plan: &dyn ExecutionPlan) -> Option<NativeRootDescent> {
    let mut ancestor_names = Vec::new();
    let mut cursor: &dyn ExecutionPlan = plan;
    loop {
        if matches!(cursor.name(), "NexusNativeExec" | "NexusNativeComposedExec") {
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

fn find_table_format_scan(plan: &dyn ExecutionPlan) -> Option<&NexusIcebergScanExec> {
    if let Some(scan) = plan.downcast_ref::<NexusIcebergScanExec>() {
        return Some(scan);
    }
    plan.children().into_iter().find_map(|child| find_table_format_scan(child.as_ref()))
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
    fn resolve_scan(
        &self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'_, datafusion_nexus::error::Result<TableFormatScanPlan>> {
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

fn test_iceberg_provider(schema: Arc<Schema>, planner: Arc<RecordingPlanner>) -> NexusIcebergTableProvider {
    let cpu_provider: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(schema, vec![vec![]]).expect("test CPU provider must build"));
    NexusIcebergTableProvider::from_cpu_provider_and_native_planner(cpu_provider, None, planner)
}
