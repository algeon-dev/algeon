//! End-to-end tests for the local Iceberg REST catalog + RustFS stack.
//!
//! These exercise the real `RestIcebergCatalogBackend` against a live REST
//! fixture + S3-compatible store (RustFS). They are gated behind
//! `CUDF_ICEBERG_REST_E2E=1` (in addition to `#[ignore]`) so Docker-less CI
//! skips them, consistent with the existing GPU/Iceberg gating.
//!
//! Bring the stack up and load data first:
//!
//! ```bash
//! nerdctl compose -f docker/iceberg-local/docker-compose.yml up -d
//! docker/iceberg-local/load.sh            # loader (Phase 3c)
//! CUDF_ICEBERG_REST_E2E=1 \
//!   DATAFUSION_CUDF_ICEBERG_REST_URI=http://localhost:8181 \
//!   DATAFUSION_CUDF_ICEBERG_WAREHOUSE=s3://lakehouse/warehouse \
//!   DATAFUSION_CUDF_ICEBERG_S3_ENDPOINT=http://localhost:9000 \
//!   DATAFUSION_CUDF_ICEBERG_S3_PATH_STYLE=true \
//!   DATAFUSION_CUDF_ICEBERG_S3_ACCESS_KEY_ID=cudfadmin \
//!   DATAFUSION_CUDF_ICEBERG_S3_SECRET_ACCESS_KEY=cudfadminsecret \
//!   DATAFUSION_CUDF_ICEBERG_NAMESPACE=tpch_sf1 \
//!   cargo test --features iceberg --test iceberg_rest_e2e_tests -- --ignored --nocapture
//! ```
#![cfg(feature = "iceberg")]

use std::sync::Arc;

use datafusion::catalog::CatalogProvider;
use datafusion_cudf::iceberg::catalog::{
    IcebergCatalogBackend, IcebergProviderKind, RestIcebergCatalogBackend, RestIcebergCatalogConfig,
    build_scoped_catalog_provider_from_backend,
};
use iceberg::{NamespaceIdent, TableIdent};

/// Returns false (and skips) unless `CUDF_ICEBERG_REST_E2E=1` is set.
fn e2e_enabled() -> bool {
    std::env::var("CUDF_ICEBERG_REST_E2E").map(|v| v == "1").unwrap_or(false)
}

/// Builds a REST config from the standard `DATAFUSION_CUDF_ICEBERG_*` env vars.
/// S3 credentials are resolved from `DATAFUSION_CUDF_ICEBERG_S3_*` by the
/// backend itself (never `AWS_*`).
fn rest_config_from_env() -> RestIcebergCatalogConfig {
    let uri = std::env::var("DATAFUSION_CUDF_ICEBERG_REST_URI").unwrap_or_else(|_| "http://localhost:8181".to_string());
    let warehouse =
        std::env::var("DATAFUSION_CUDF_ICEBERG_WAREHOUSE").unwrap_or_else(|_| "s3://lakehouse/warehouse".to_string());

    let mut config = RestIcebergCatalogConfig::new("lake", uri, warehouse);
    if let Ok(endpoint) = std::env::var("DATAFUSION_CUDF_ICEBERG_S3_ENDPOINT") {
        config = config.with_s3_endpoint(endpoint);
    }
    if let Ok(region) = std::env::var("DATAFUSION_CUDF_ICEBERG_S3_REGION") {
        config = config.with_s3_region(region);
    }
    if let Ok(path_style) = std::env::var("DATAFUSION_CUDF_ICEBERG_S3_PATH_STYLE") {
        config = config.with_s3_path_style_access(path_style == "true");
    }
    config
}

/// Read path: list namespaces through the REST backend. Proves the Rust/OpenDAL
/// catalog client can talk to the fixture.
#[tokio::test]
#[ignore = "requires CUDF_ICEBERG_REST_E2E=1 and a running docker/iceberg-local stack"]
async fn rest_backend_lists_namespaces_live() {
    if !e2e_enabled() {
        eprintln!("skipping: set CUDF_ICEBERG_REST_E2E=1 to run");
        return;
    }

    let backend =
        RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.expect("connect to REST catalog");
    let namespaces = backend.list_namespaces().await.expect("list namespaces");
    eprintln!("discovered namespaces: {namespaces:?}");
    assert!(!namespaces.is_empty(), "expected at least one namespace in the local catalog");
}

/// Read path: scoped catalog provider over the configured namespace exposes its
/// tables. Mirrors the Glue live test but against REST + RustFS.
#[tokio::test]
#[ignore = "requires CUDF_ICEBERG_REST_E2E=1 and a running docker/iceberg-local stack"]
async fn rest_scoped_catalog_provider_discovers_tables_live() {
    if !e2e_enabled() {
        eprintln!("skipping: set CUDF_ICEBERG_REST_E2E=1 to run");
        return;
    }

    let namespace = std::env::var("DATAFUSION_CUDF_ICEBERG_NAMESPACE")
        .expect("set DATAFUSION_CUDF_ICEBERG_NAMESPACE for live REST test");

    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.unwrap());
    let provider =
        build_scoped_catalog_provider_from_backend(Arc::clone(&backend), "rest", &namespace, IcebergProviderKind::Gpu)
            .await
            .expect("build scoped catalog provider");

    let schema_names = provider.schema_names();
    eprintln!("discovered scoped namespaces: {schema_names:?}");
    assert_eq!(schema_names, vec![namespace.clone()], "scoped catalog must expose only '{namespace}'");

    let schema = provider.schema(&namespace).unwrap();
    let table_names = schema.table_names();
    eprintln!("discovered tables in '{namespace}': {table_names:?}");
    assert!(!table_names.is_empty(), "expected at least one table in '{namespace}'");
}

/// Full read path: register the REST catalog with a CPU Arrow reader and run a
/// real DataFusion scan (`count(*)` + a `LIMIT` projection) end-to-end through
/// RustFS. Requires the `iceberg-cpu` feature for a GPU-free reader.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// CUDF_ICEBERG_REST_E2E=1 DATAFUSION_CUDF_ICEBERG_NAMESPACE=tpch_sf1 \
///   CUDF_ICEBERG_REST_E2E_TABLE=nation CUDF_ICEBERG_REST_E2E_EXPECTED_ROWS=25 \
///   ... cargo test --features iceberg-cpu --test iceberg_rest_e2e_tests \
///   rest_datafusion_scan_live -- --ignored --nocapture
/// ```
#[cfg(feature = "iceberg-cpu")]
#[tokio::test]
#[ignore = "requires CUDF_ICEBERG_REST_E2E=1, iceberg-cpu, and a loaded docker/iceberg-local stack"]
async fn rest_datafusion_scan_live() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    if !e2e_enabled() {
        eprintln!("skipping: set CUDF_ICEBERG_REST_E2E=1 to run");
        return;
    }

    let namespace = std::env::var("DATAFUSION_CUDF_ICEBERG_NAMESPACE")
        .expect("set DATAFUSION_CUDF_ICEBERG_NAMESPACE for live REST scan test");
    let table = std::env::var("CUDF_ICEBERG_REST_E2E_TABLE").unwrap_or_else(|_| "nation".to_string());
    let expected_rows: Option<i64> =
        std::env::var("CUDF_ICEBERG_REST_E2E_EXPECTED_ROWS").ok().and_then(|v| v.parse().ok());

    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.unwrap());
    let provider =
        build_scoped_catalog_provider_from_backend(Arc::clone(&backend), "rest", &namespace, IcebergProviderKind::Cpu)
            .await
            .expect("build scoped CPU catalog provider");

    let ctx = SessionContext::new_with_config(SessionConfig::new());
    ctx.register_catalog("lake", Arc::new(provider));

    // Step 5: one DataFusion scan end-to-end through the REST backend.
    let count_sql = format!("SELECT count(*) AS n FROM lake.\"{namespace}\".\"{table}\"");
    let batches = ctx.sql(&count_sql).await.expect("plan count").collect().await.expect("execute count");
    let n = batches
        .first()
        .map(|b| b.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().expect("count is Int64").value(0))
        .expect("one count batch");
    eprintln!("scan count for lake.{namespace}.{table} = {n}");
    assert!(n > 0, "expected non-empty table {namespace}.{table}");
    if let Some(expected) = expected_rows {
        assert_eq!(n, expected, "row count mismatch for {namespace}.{table}");
    }

    // A small projecting LIMIT scan exercises the actual data-file read path.
    let limit_sql = format!("SELECT * FROM lake.\"{namespace}\".\"{table}\" LIMIT 3");
    let limited = ctx.sql(&limit_sql).await.expect("plan limit").collect().await.expect("execute limit");
    let rows: usize = limited.iter().map(|b| b.num_rows()).sum();
    eprintln!("limit-3 scan returned {rows} rows");
    assert!(rows > 0 && rows <= 3, "LIMIT 3 must return between 1 and 3 rows, got {rows}");
}

/// Metadata contract: load the `nation` table directly through the REST backend
/// and assert (1) the Iceberg schema field names match the TPC-H `nation`
/// columns and (2) the table root and metadata pointer live under `s3://`
/// (i.e. the write actually landed in RustFS, not on the REST fixture's local
/// disk). This closes Phase 3d steps 2 and 4 without needing a GPU or the
/// `iceberg-cpu` reader.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// CUDF_ICEBERG_REST_E2E=1 DATAFUSION_CUDF_ICEBERG_NAMESPACE=tpch_sf1 \
///   ... cargo test --features iceberg --test iceberg_rest_e2e_tests \
///   rest_table_metadata_contract_live -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "requires CUDF_ICEBERG_REST_E2E=1 and a loaded docker/iceberg-local stack"]
async fn rest_table_metadata_contract_live() {
    if !e2e_enabled() {
        eprintln!("skipping: set CUDF_ICEBERG_REST_E2E=1 to run");
        return;
    }

    let namespace = std::env::var("DATAFUSION_CUDF_ICEBERG_NAMESPACE")
        .expect("set DATAFUSION_CUDF_ICEBERG_NAMESPACE for live REST metadata test");

    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.unwrap());

    let ns = NamespaceIdent::new(namespace.clone());
    let ident = TableIdent::new(ns, "nation".to_string());
    let table = backend.load_table(&ident).await.expect("load nation table");

    // (1) Schema field names match the TPC-H nation columns.
    let schema = table.metadata().current_schema();
    let mut field_names: Vec<String> = schema.as_struct().fields().iter().map(|f| f.name.clone()).collect();
    field_names.sort();
    let mut expected = vec!["n_comment", "n_name", "n_nationkey", "n_regionkey"];
    expected.sort();
    eprintln!("nation schema field names: {field_names:?}");
    assert_eq!(field_names, expected, "TPC-H nation schema field names must match the loader output");

    // (2) Table root and metadata pointer live in S3 (RustFS), not local disk.
    let table_location = table.metadata().location();
    eprintln!("nation table location: {table_location}");
    assert!(table_location.starts_with("s3://"), "table root must be an s3:// location, got {table_location}");

    let metadata_location = table.metadata_location().expect("metadata pointer present");
    eprintln!("nation metadata location: {metadata_location}");
    assert!(
        metadata_location.starts_with("s3://"),
        "metadata pointer must be an s3:// location, got {metadata_location}"
    );
}

/// Phase 4: assert the native **planning rewrite** plus **CPU-reader
/// stability** over a live REST table, with GPU-native correctness as an
/// explicit opt-in.
///
/// What the default path proves (CPU-only — works wherever the stack is up):
/// 1. **Planning rewrite.** Register the REST catalog with a GPU provider and
///    the `CudfNativePhysicalOptimizerRule`, build the optimized physical plan
///    for an aggregate-over-Iceberg shape, and assert the rule rewrote it to a
///    `NativeCudfExec` with no leftover `CudfTableFormatScanExec`. This proves
///    the optimizer *wrapped* the shape — NOT that the lowered native IR or its
///    GPU execution is semantically correct.
/// 2. **CPU-reader stability.** Two independent CPU-reader runs of the same
///    query must agree, proving the REST/RustFS read + aggregate path is
///    deterministic end-to-end.
///
/// The default path does NOT validate the native-rewritten plan's *result*: a
/// broken `NativeCudfExec` or native Iceberg read path can still pass it. Native
/// result correctness is only checked under the GPU-parity opt-in below.
/// GPU parity exercises the native source-storage profile path for the
/// configured REST/RustFS endpoint, path-style flag, region, and static
/// credentials.
///
/// GPU-native parity is gated behind `CUDF_ICEBERG_REST_E2E_GPU_PARITY=1`.
/// When set, the test executes the native-rewritten plan and a GPU execution
/// failure is a real failure — it does NOT silently skip. Enable it for live
/// validation of the configured S3 endpoint.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// CUDF_ICEBERG_REST_E2E=1 DATAFUSION_CUDF_ICEBERG_NAMESPACE=tpch_sf1 \
///   ... cargo test --features iceberg-cpu --test iceberg_rest_e2e_tests \
///   rest_native_planning_and_parity_live -- --ignored --nocapture
/// ```
#[cfg(feature = "iceberg-cpu")]
#[tokio::test]
#[ignore = "requires CUDF_ICEBERG_REST_E2E=1, iceberg-cpu, and a loaded docker/iceberg-local stack"]
async fn rest_native_planning_and_parity_live() {
    use datafusion::{
        execution::session_state::SessionStateBuilder,
        physical_plan::displayable,
        prelude::{SessionConfig, SessionContext},
    };
    use datafusion_cudf::{
        CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy, NativeCudfExecutionConfig,
    };

    if !e2e_enabled() {
        eprintln!("skipping: set CUDF_ICEBERG_REST_E2E=1 to run");
        return;
    }

    let namespace = std::env::var("DATAFUSION_CUDF_ICEBERG_NAMESPACE")
        .expect("set DATAFUSION_CUDF_ICEBERG_NAMESPACE for live REST planning/parity test");
    // `nation` is small and has a `n_regionkey` group key — a stable
    // aggregate-over-Iceberg shape the native rule rewrites.
    let table = std::env::var("CUDF_ICEBERG_REST_E2E_TABLE").unwrap_or_else(|_| "nation".to_string());
    let group_col = std::env::var("CUDF_ICEBERG_REST_E2E_GROUP_COL").unwrap_or_else(|_| "n_regionkey".to_string());

    // No ORDER BY: a top-level sort introduces a SortPreservingMergeExec that the
    // native rule does not absorb (it would reject the whole plan with no CPU
    // fallback under the GPU provider). Parity comparison sorts rows itself.
    let sql = format!(
        "SELECT \"{group_col}\" AS k, count(*) AS n FROM lake.\"{namespace}\".\"{table}\" \
         GROUP BY \"{group_col}\""
    );

    // --- native (GPU) session: catalog + native optimizer rule ---------------
    let rest_config = rest_config_from_env();
    let storage_profile = rest_config.source_storage_profile_from_env();
    let gpu_backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config.clone()).await.unwrap());
    let gpu_provider = build_scoped_catalog_provider_from_backend(
        Arc::clone(&gpu_backend),
        "rest",
        &namespace,
        IcebergProviderKind::Gpu,
    )
    .await
    .expect("build scoped GPU catalog provider");

    // Single partition keeps the plan free of RepartitionExec so the aggregate
    // shape lowers cleanly into one NativeCudfExec pipeline.
    let native_execution_config =
        NativeCudfExecutionConfig::bounded_auto_sized().with_source_storage_profile(storage_profile);
    let optimizer_config =
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(native_execution_config);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .with_cudf_native(optimizer_config)
        .build();
    let gpu_ctx = SessionContext::new_with_state(state);
    gpu_ctx.register_catalog("lake", Arc::new(gpu_provider));

    // Planning outcome: the rule must rewrite to NativeCudfExec.
    let physical = gpu_ctx
        .sql(&sql)
        .await
        .expect("plan native sql")
        .create_physical_plan()
        .await
        .expect("build native physical plan");
    let rendered = displayable(physical.as_ref()).indent(false).to_string();
    eprintln!("native physical plan:\n{rendered}");
    assert!(
        plan_contains(physical.as_ref(), "NativeCudfExec"),
        "native profile must rewrite the aggregate-over-Iceberg shape to NativeCudfExec; plan:\n{rendered}"
    );
    assert!(
        !plan_contains(physical.as_ref(), "CudfTableFormatScanExec"),
        "native rewrite left a leftover CudfTableFormatScanExec; plan:\n{rendered}"
    );

    // --- CPU reference session: GPU-free Arrow reader ------------------------
    let cpu_backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config.clone()).await.unwrap());
    let cpu_provider = build_scoped_catalog_provider_from_backend(
        Arc::clone(&cpu_backend),
        "rest",
        &namespace,
        IcebergProviderKind::Cpu,
    )
    .await
    .expect("build scoped CPU catalog provider");
    let cpu_ctx = SessionContext::new_with_config(SessionConfig::new());
    cpu_ctx.register_catalog("lake", Arc::new(cpu_provider));
    let cpu_rows =
        sorted_rows(&cpu_ctx.sql(&sql).await.expect("plan cpu sql").collect().await.expect("execute cpu reference"));
    eprintln!("cpu reference produced {} rows", cpu_rows.len());
    assert!(!cpu_rows.is_empty(), "cpu reference returned no rows for {namespace}.{table}");

    // CPU-reader stability (default): a second independent CPU-reader run must
    // agree with the first. This is deterministic and storage-profile-independent;
    // it proves the read + aggregate shape is stable end-to-end through RustFS. It
    // does NOT validate the native-rewritten plan's result — that is the GPU-parity
    // opt-in below.
    let cpu_provider2 = build_scoped_catalog_provider_from_backend(
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config).await.unwrap())
            as Arc<dyn IcebergCatalogBackend>,
        "rest",
        &namespace,
        IcebergProviderKind::Cpu,
    )
    .await
    .expect("build second scoped CPU catalog provider");
    let cpu_ctx2 = SessionContext::new_with_config(SessionConfig::new());
    cpu_ctx2.register_catalog("lake", Arc::new(cpu_provider2));
    let cpu_rows2 = sorted_rows(
        &cpu_ctx2.sql(&sql).await.expect("plan cpu sql 2").collect().await.expect("execute cpu reference 2"),
    );
    assert_eq!(cpu_rows, cpu_rows2, "CPU-reader parity mismatch for {namespace}.{table}");
    eprintln!("CPU-reader parity OK: {} rows (GPU-native parity not asserted)", cpu_rows.len());

    // GPU-native parity is opt-in. Set this for live REST/RustFS/local-cache
    // validation of the native source-storage profile path. We do NOT silently
    // skip on error — a failure when the flag is set is a real failure.
    let gpu_parity = std::env::var("CUDF_ICEBERG_REST_E2E_GPU_PARITY").map(|v| v == "1").unwrap_or(false);
    if !gpu_parity {
        eprintln!(
            "GPU-native parity not run: set CUDF_ICEBERG_REST_E2E_GPU_PARITY=1 to enable \
             live REST/RustFS/local-cache native profile validation"
        );
        return;
    }

    let native_batches = gpu_ctx
        .sql(&sql)
        .await
        .expect("plan native sql (exec)")
        .collect()
        .await
        .expect("execute native (GPU) plan — CUDF_ICEBERG_REST_E2E_GPU_PARITY=1 requires it to run");
    let native_rows = sorted_rows(&native_batches);
    eprintln!("native plan produced {} rows", native_rows.len());
    assert_eq!(native_rows, cpu_rows, "native vs CPU row parity mismatch for {namespace}.{table}");
    eprintln!("GPU-native parity OK: native and CPU agree on {} rows", native_rows.len());
}

/// Recursively checks whether any node in the physical plan tree has the given
/// operator name.
#[cfg(feature = "iceberg-cpu")]
fn plan_contains(plan: &dyn datafusion::physical_plan::ExecutionPlan, name: &str) -> bool {
    if plan.name() == name {
        return true;
    }
    plan.children().iter().any(|c| plan_contains(c.as_ref(), name))
}

/// Renders RecordBatches to a sorted Vec of row strings for order-insensitive,
/// engine-agnostic parity comparison.
#[cfg(feature = "iceberg-cpu")]
fn sorted_rows(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    use std::fmt::Write;
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut s = String::new();
            for col in 0..batch.num_columns() {
                let array = batch.column(col);
                if col > 0 {
                    s.push('\u{1f}');
                }
                if array.is_null(row) {
                    s.push_str("\u{0}NULL");
                } else {
                    let _ =
                        write!(s, "{}", arrow::util::display::array_value_to_string(array, row).unwrap_or_default());
                }
            }
            rows.push(s);
        }
    }
    rows.sort();
    rows
}
