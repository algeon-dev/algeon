//! TPC-DS planning under the GPU execution requirement of the SF100 benchmark.
//!
//! `tpcds_gpu_cpu_bench` plans with the `sf100_gpu_throughput` profile, the
//! tolerant statistical aggregate mode and `native_no_datafusion_cpu`, which
//! fails planning when the completed plan keeps a DataFusion CPU operator.
//! These cases plan the same configuration over the tiny TPC-DS dataset, which
//! has the TPC-DS column types, without a device.

use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{planner::plan_with_native_diagnostics, session::AlgeonSessionStateBuilderExt};
use bench::report_workload::{
    RelationalExecutionSelection,
    tpcds::{
        ALGEON_TPCDS_NATIVE_PROFILE_ENV, TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
        register_tpcds_tables_with_native_metadata, tpcds_native_analytics_optimizer_config_with_profile_settings,
        tpcds_native_profile_settings_with_env, tpcds_native_report_session_config,
    },
};
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use workloads::tpcds::{TPCDS_QUERIES, write_tiny_tpcds_parquet_dataset};

/// q06 and q65 compare a column with a scalar subquery that DataFusion
/// decorrelates into a semi join, and q58 compares the revenues of three
/// inner-joined aggregates. Each join residual casts a `Float64` product such
/// as `1.2 * avg(...)` to `Decimal128(30, 15)`, a cast that fails on values
/// outside that precision, so it runs on the key-matched pairs only.
#[tokio::test]
async fn tpcds_join_residuals_with_checked_decimal_casts_plan_without_datafusion_cpu() {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("algeon-bench-tpcds-required-planning-{id}"));
    fs::create_dir_all(&dir).unwrap();
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let profile = tpcds_native_profile_settings_with_env(|name| {
        (name == ALGEON_TPCDS_NATIVE_PROFILE_ENV).then(|| TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT.to_string())
    });
    let optimizer_config = RelationalExecutionSelection::strict_native()
        .configure_native_optimizer(tpcds_native_analytics_optimizer_config_with_profile_settings(&profile))
        .unwrap();
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpcds_native_report_session_config())
        .with_algeon_native(optimizer_config)
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables_with_native_metadata(&ctx, &dir).await.unwrap();

    let mut failures = Vec::new();
    for query_id in ["q06", "q58", "q65"] {
        let query = TPCDS_QUERIES.iter().find(|query| query.id == query_id).unwrap();
        let (state, logical) = ctx.sql(query.sql).await.unwrap().into_parts();
        if let Err(error) = plan_with_native_diagnostics(&state, &logical).await {
            failures.push(format!("{query_id}: {error}"));
        }
    }
    assert!(failures.is_empty(), "queries kept DataFusion CPU operators: {failures:#?}");

    fs::remove_dir_all(dir).unwrap();
}
