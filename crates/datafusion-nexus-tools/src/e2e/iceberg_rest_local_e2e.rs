//! End-to-end runner for the local Iceberg REST catalog + RustFS stack.
//!
//! This exercises the real `RestIcebergCatalogBackend` against a local REST
//! fixture + S3-compatible store (RustFS). It is an explicit `cargo run` tool,
//! not a `cargo test` target. `NEXUS_ICEBERG_REST_E2E=1` is still required so an
//! accidental invocation fails before touching external services.
//!
//! Bring the stack up and load data first:
//!
//! ```bash
//! docker compose -f fixture/iceberg-local/docker-compose.yml up -d
//! fixture/iceberg-local/load.sh            # loader (Milestone 3c)
//! NEXUS_ICEBERG_REST_E2E=1 \
//!   NEXUS_ICEBERG_REST_URI=http://localhost:8181 \
//!   NEXUS_ICEBERG_WAREHOUSE=s3://lakehouse/warehouse \
//!   NEXUS_ICEBERG_S3_ENDPOINT=http://localhost:9000 \
//!   NEXUS_ICEBERG_S3_PATH_STYLE=true \
//!   NEXUS_ICEBERG_S3_ACCESS_KEY_ID=nexusadmin \
//!   NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY=nexusadmin \
//!   NEXUS_ICEBERG_NAMESPACE=tpch_sf1 \
//!   cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
//! ```

use std::sync::Arc;

use datafusion::catalog::CatalogProvider;
use datafusion_nexus::table_format::iceberg::catalog::{
    IcebergCatalogBackend, IcebergProviderKind, RestIcebergCatalogBackend, RestIcebergCatalogConfig,
    build_scoped_catalog_provider_from_backend,
};
use iceberg::{NamespaceIdent, TableIdent};
use url::Url;

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub async fn run() -> E2eResult<()> {
    require_e2e_enabled()?;
    require_local_rest_storage_env()?;
    rest_backend_lists_namespaces_local().await;
    rest_scoped_catalog_provider_discovers_tables_local().await;
    rest_table_metadata_contract_local().await;
    rest_datafusion_scan_local().await;
    rest_native_planning_and_parity_local().await;
    println!("iceberg_rest_local_e2e ok");
    Ok(())
}

fn require_e2e_enabled() -> E2eResult<()> {
    match std::env::var("NEXUS_ICEBERG_REST_E2E").as_deref() {
        Ok("1") => Ok(()),
        _ => Err("iceberg REST E2E requires NEXUS_ICEBERG_REST_E2E=1 and fixture/iceberg-local".into()),
    }
}

fn require_local_rest_storage_env() -> E2eResult<()> {
    let rest_uri = std::env::var("NEXUS_ICEBERG_REST_URI").unwrap_or_else(|_| "http://localhost:8181".to_string());
    require_loopback_http_url("NEXUS_ICEBERG_REST_URI", &rest_uri)?;

    let endpoint = require_env("NEXUS_ICEBERG_S3_ENDPOINT")?;
    require_loopback_http_url("NEXUS_ICEBERG_S3_ENDPOINT", &endpoint)?;

    let path_style = require_env("NEXUS_ICEBERG_S3_PATH_STYLE")?;
    if path_style != "true" {
        return Err("local REST E2E requires NEXUS_ICEBERG_S3_PATH_STYLE=true for RustFS".into());
    }

    require_env("NEXUS_ICEBERG_S3_ACCESS_KEY_ID")?;
    require_env("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY")?;
    reject_non_empty_env("NEXUS_ICEBERG_S3_SESSION_TOKEN")?;
    reject_non_empty_env("AWS_SESSION_TOKEN")?;
    reject_non_empty_env("AWS_SECURITY_TOKEN")?;
    Ok(())
}

fn require_loopback_http_url(name: &str, value: &str) -> E2eResult<()> {
    if is_loopback_http_url(value) {
        return Ok(());
    }
    Err(format!("{name} must be a loopback http:// URL for local REST E2E, got {value:?}").into())
}

fn is_loopback_http_url(value: &str) -> bool {
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<std::net::IpAddr>().map(|addr| addr.is_loopback()).unwrap_or(false)
}

fn require_env(name: &'static str) -> E2eResult<String> {
    std::env::var(name).map_err(|_| format!("local REST E2E requires {name} for fixture/iceberg-local RustFS").into())
}

fn reject_non_empty_env(name: &'static str) -> E2eResult<()> {
    if std::env::var(name).is_ok_and(|value| !value.is_empty()) {
        return Err(
            format!("local REST E2E refuses ambient session token {name}; use static RustFS credentials only").into()
        );
    }
    Ok(())
}

/// Builds a REST config from the standard `NEXUS_ICEBERG_*` env vars.
/// S3 credentials are resolved from `NEXUS_ICEBERG_S3_*` by the
/// backend itself (never `AWS_*`).
fn rest_config_from_env() -> RestIcebergCatalogConfig {
    let uri = std::env::var("NEXUS_ICEBERG_REST_URI").unwrap_or_else(|_| "http://localhost:8181".to_string());
    let warehouse = std::env::var("NEXUS_ICEBERG_WAREHOUSE").unwrap_or_else(|_| "s3://lakehouse/warehouse".to_string());

    let endpoint = std::env::var("NEXUS_ICEBERG_S3_ENDPOINT").expect("validated NEXUS_ICEBERG_S3_ENDPOINT");
    let path_style = std::env::var("NEXUS_ICEBERG_S3_PATH_STYLE").expect("validated NEXUS_ICEBERG_S3_PATH_STYLE");
    let region = std::env::var("NEXUS_ICEBERG_S3_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    RestIcebergCatalogConfig::new("lake", uri, warehouse)
        .with_s3_endpoint(endpoint)
        .with_s3_region(region)
        .with_s3_path_style_access(path_style == "true")
}

/// Read path: list namespaces through the REST backend. Proves the Rust/OpenDAL
/// catalog client can talk to the fixture.
async fn rest_backend_lists_namespaces_local() {
    let backend =
        RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.expect("connect to REST catalog");
    let namespaces = backend.list_namespaces().await.expect("list namespaces");
    eprintln!("discovered namespaces: {namespaces:?}");
    assert!(!namespaces.is_empty(), "expected at least one namespace in the local catalog");
}

/// Read path: scoped catalog provider over the configured namespace exposes its
/// tables. Mirrors the Glue local E2E check but against REST + RustFS.
async fn rest_scoped_catalog_provider_discovers_tables_local() {
    let namespace = std::env::var("NEXUS_ICEBERG_NAMESPACE").expect("set NEXUS_ICEBERG_NAMESPACE for local REST E2E");

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

/// Full CPU-reader path through REST/RustFS: `count(*)` plus a small `LIMIT`.
/// Requires `iceberg-cpu`.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// NEXUS_ICEBERG_REST_E2E=1 NEXUS_ICEBERG_NAMESPACE=tpch_sf1 \
///   NEXUS_ICEBERG_REST_E2E_TABLE=nation NEXUS_ICEBERG_REST_E2E_EXPECTED_ROWS=25 \
///   cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
/// ```
async fn rest_datafusion_scan_local() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    let namespace =
        std::env::var("NEXUS_ICEBERG_NAMESPACE").expect("set NEXUS_ICEBERG_NAMESPACE for local REST scan E2E");
    let table = std::env::var("NEXUS_ICEBERG_REST_E2E_TABLE").unwrap_or_else(|_| "nation".to_string());
    let expected_rows: Option<i64> =
        std::env::var("NEXUS_ICEBERG_REST_E2E_EXPECTED_ROWS").ok().and_then(|v| v.parse().ok());

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

    if let Some(expected_ids) = expected_i64_list_env("NEXUS_ICEBERG_REST_E2E_EXPECTED_IDS") {
        let ids_sql = format!("SELECT id FROM lake.\"{namespace}\".\"{table}\" ORDER BY id");
        let id_batches = ctx.sql(&ids_sql).await.expect("plan id scan").collect().await.expect("execute id scan");
        let ids = collect_i64_column(&id_batches, 0);
        assert_eq!(ids, expected_ids, "visible id set mismatch for {namespace}.{table}");
    }

    // A small projecting LIMIT scan exercises the actual data-file read path.
    let limit_sql = format!("SELECT * FROM lake.\"{namespace}\".\"{table}\" LIMIT 3");
    let limited = ctx.sql(&limit_sql).await.expect("plan limit").collect().await.expect("execute limit");
    let rows: usize = limited.iter().map(|b| b.num_rows()).sum();
    eprintln!("limit-3 scan returned {rows} rows");
    assert!(rows > 0 && rows <= 3, "LIMIT 3 must return between 1 and 3 rows, got {rows}");
}

/// Metadata contract: `nation` has the expected schema and S3-backed metadata
/// locations, without requiring GPU or `iceberg-cpu`.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// NEXUS_ICEBERG_REST_E2E=1 NEXUS_ICEBERG_NAMESPACE=tpch_sf1 \
///   cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
/// ```
async fn rest_table_metadata_contract_local() {
    let namespace =
        std::env::var("NEXUS_ICEBERG_NAMESPACE").expect("set NEXUS_ICEBERG_NAMESPACE for local REST metadata E2E");
    let table_name = std::env::var("NEXUS_ICEBERG_REST_E2E_TABLE").unwrap_or_else(|_| "nation".to_string());

    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(rest_config_from_env()).await.unwrap());

    let ns = NamespaceIdent::new(namespace.clone());
    let ident = TableIdent::new(ns, table_name.clone());
    let table = backend.load_table(&ident).await.expect("load REST E2E table");

    // (1) Schema field names match the selected fixture table.
    let schema = table.metadata().current_schema();
    let mut field_names: Vec<String> = schema.as_struct().fields().iter().map(|f| f.name.clone()).collect();
    field_names.sort();
    if table_name == "nation" {
        let mut expected = vec!["n_comment", "n_name", "n_nationkey", "n_regionkey"];
        expected.sort();
        eprintln!("nation schema field names: {field_names:?}");
        assert_eq!(field_names, expected, "TPC-H nation schema field names must match the loader output");
    } else if table_name == "eq_deletes" {
        let mut expected = vec!["grp", "id", "payload"];
        expected.sort();
        eprintln!("eq_deletes schema field names: {field_names:?}");
        assert_eq!(field_names, expected, "equality-delete fixture schema field names must match");
    } else {
        eprintln!("{table_name} schema field names: {field_names:?}");
    }

    if let Some(expected_delete_files) = expected_i64_env("NEXUS_ICEBERG_REST_E2E_EXPECTED_EQ_DELETE_FILES") {
        let added_equality_delete_files = snapshot_summary_i64_total(table.metadata(), "added-equality-delete-files");
        assert_eq!(
            added_equality_delete_files, expected_delete_files,
            "equality-delete snapshot metadata mismatch for {namespace}.{table_name}"
        );
        let total_delete_files = table
            .metadata()
            .current_snapshot()
            .and_then(|snapshot| snapshot.summary().additional_properties.get("total-delete-files"))
            .and_then(|value| value.parse::<i64>().ok());
        assert_eq!(
            total_delete_files,
            Some(expected_delete_files),
            "current snapshot total-delete-files mismatch for {namespace}.{table_name}"
        );
    }

    // (2) Table root and metadata pointer live in S3 (RustFS), not local disk.
    let table_location = table.metadata().location();
    eprintln!("{table_name} table location: {table_location}");
    assert!(table_location.starts_with("s3://"), "table root must be an s3:// location, got {table_location}");

    let metadata_location = table.metadata_location().expect("metadata pointer present");
    eprintln!("{table_name} metadata location: {metadata_location}");
    assert!(
        metadata_location.starts_with("s3://"),
        "metadata pointer must be an s3:// location, got {metadata_location}"
    );
}

/// Checks native planning rewrite and CPU-reader stability over a local REST
/// table; GPU-native correctness is opt-in.
///
/// Default CPU-only path proves:
/// 1. **Planning rewrite.** Register the REST catalog with a GPU provider and
///    the `CudfNativePhysicalOptimizerRule`, build the optimized physical plan
///    for an aggregate-over-Iceberg shape, and assert the rule rewrote it to a
///    `NativeCudfExec` with no leftover `CudfTableFormatScanExec`. This proves
///    the optimizer wrapped the shape, not GPU result correctness.
/// 2. **CPU-reader stability.** Two independent CPU-reader runs of the same
///    query must agree.
///
/// GPU parity exercises the native source-storage profile path and validates
/// the native-rewritten result.
///
/// GPU-native parity is gated behind `NEXUS_ICEBERG_REST_E2E_GPU_PARITY=1`.
/// When set, GPU execution failures are real failures, not skips.
///
/// Run against a loaded `tpch_sf1` namespace:
/// ```bash
/// NEXUS_ICEBERG_REST_E2E=1 NEXUS_ICEBERG_NAMESPACE=tpch_sf1 \
///   cargo run -p datafusion-nexus-tools --features iceberg-e2e -- iceberg-rest-local-e2e
/// ```
async fn rest_native_planning_and_parity_local() {
    use datafusion::{
        execution::session_state::SessionStateBuilder,
        physical_plan::displayable,
        prelude::{SessionConfig, SessionContext},
    };
    use datafusion_nexus::{
        native_exec::NativeCudfExecutionConfig, planner::CudfOptimizerConfig, policy::GpuFallbackPolicy,
        session::CudfSessionStateBuilderExt,
    };

    let namespace = std::env::var("NEXUS_ICEBERG_NAMESPACE")
        .expect("set NEXUS_ICEBERG_NAMESPACE for local REST planning/parity E2E");
    // `nation` is small and has a `n_regionkey` group key — a stable
    // aggregate-over-Iceberg shape the native rule rewrites.
    let table = std::env::var("NEXUS_ICEBERG_REST_E2E_TABLE").unwrap_or_else(|_| "nation".to_string());
    let group_col = std::env::var("NEXUS_ICEBERG_REST_E2E_GROUP_COL").unwrap_or_else(|_| "n_regionkey".to_string());

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

    // GPU-native parity is opt-in. Set this for local REST/RustFS/local-cache
    // validation of the native source-storage profile path. We do NOT silently
    // skip on error — a failure when the flag is set is a real failure.
    let gpu_parity = std::env::var("NEXUS_ICEBERG_REST_E2E_GPU_PARITY").map(|v| v == "1").unwrap_or(false);
    if !gpu_parity {
        eprintln!(
            "GPU-native parity not run: set NEXUS_ICEBERG_REST_E2E_GPU_PARITY=1 to enable \
             local REST/RustFS/local-cache native profile validation"
        );
        return;
    }

    let native_batches = gpu_ctx
        .sql(&sql)
        .await
        .expect("plan native sql (exec)")
        .collect()
        .await
        .expect("execute native (GPU) plan — NEXUS_ICEBERG_REST_E2E_GPU_PARITY=1 requires it to run");
    let native_rows = sorted_rows(&native_batches);
    eprintln!("native plan produced {} rows", native_rows.len());
    assert_eq!(native_rows, cpu_rows, "native vs CPU row parity mismatch for {namespace}.{table}");
    eprintln!("GPU-native parity OK: native and CPU agree on {} rows", native_rows.len());
}

/// Recursively checks whether any node in the physical plan tree has the given
/// operator name.
fn plan_contains(plan: &dyn datafusion::physical_plan::ExecutionPlan, name: &str) -> bool {
    if plan.name() == name {
        return true;
    }
    plan.children().iter().any(|c| plan_contains(c.as_ref(), name))
}

/// Renders RecordBatches to a sorted Vec of row strings for order-insensitive,
/// engine-agnostic parity comparison.
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

fn expected_i64_env(name: &str) -> Option<i64> {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty()).map(|value| {
        value.parse::<i64>().unwrap_or_else(|error| panic!("{name} must be an integer, got {value:?}: {error}"))
    })
}

fn expected_i64_list_env(name: &str) -> Option<Vec<i64>> {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty()).map(|value| {
        value
            .split(',')
            .map(|part| {
                let part = part.trim();
                part.parse::<i64>().unwrap_or_else(|error| panic!("{name} contains non-integer {part:?}: {error}"))
            })
            .collect()
    })
}

fn collect_i64_column(batches: &[arrow::record_batch::RecordBatch], column_index: usize) -> Vec<i64> {
    use arrow::array::Array;

    let mut values = Vec::new();
    for batch in batches {
        let column = batch
            .column(column_index)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("expected Int64 column");
        for row in 0..column.len() {
            values.push(column.value(row));
        }
    }
    values
}

fn snapshot_summary_i64_total(metadata: &iceberg::spec::TableMetadata, key: &str) -> i64 {
    metadata
        .snapshots()
        .filter_map(|snapshot| snapshot.summary().additional_properties.get(key))
        .map(|value| {
            value.parse::<i64>().unwrap_or_else(|error| panic!("invalid snapshot summary {key}={value}: {error}"))
        })
        .sum()
}
