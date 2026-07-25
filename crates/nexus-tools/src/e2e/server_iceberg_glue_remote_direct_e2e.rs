//! Glue/S3 KvikIO-only Parquet E2E.
//!
//! The parent process launches one independent child process for each
//! `KVIKIO_COMPAT_MODE` value (`ON`, `OFF`, and `AUTO`). KvikIO initialization
//! is process-global, so changing the environment inside one server process is
//! not valid coverage. Each child exercises remote S3 URI, generated local
//! Parquet path, and fill-before-read local-cache path inputs through Flight SQL
//! and returns normalized rows for cross-mode parity checking.
//!
//! By default the tool targets the read-only TPC-DS SF100 fixture in `us-west-2`:
//! `s3://datafusion-nexus-tpcds-sf100-123456789012-us-west-2/warehouse/nexus_tpcds_sf100.db/`.
//! Credentials are inherited from the AWS default chain (normally the EC2
//! instance profile); no credential value is accepted in arguments or output.
//! Local Glue/S3 endpoints remain available only through explicit environment
//! overrides.
//!
//! GPU and read access to Glue/S3 are required. Serialize with
//! `flock /tmp/cudf-gpu.lock` per repository policy.
//!
//! ```bash
//! NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E=1 \
//!   flock /tmp/cudf-gpu.lock \
//!   cargo run -p nexus-tools --features server-iceberg-e2e -- \
//!   server-iceberg-glue-remote-direct-e2e
//! ```

use std::{
    collections::HashMap,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, ArrayRef, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::{
    Action, FlightInfo,
    sql::{Any, TicketStatementQuery, client::FlightSqlServiceClient},
};
use datafusion_nexus::{
    server::{
        NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV, ServerConfig, serve_with_listener,
    },
    table_format::{
        NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, NEXUS_LAKEHOUSE_CACHE_READ_ENV,
        NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
    },
};
use futures::TryStreamExt;
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use prost::Message as _;
use url::Url;

use crate::connection_retry::{BeforeConnect, ConnectionRetryConfig, connect_with_retry};

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn e2e_error(message: impl Into<String>) -> Box<dyn std::error::Error + Send + Sync> {
    message.into().into()
}

const DEFAULT_TABLE: &str = "reason";
const DEFAULT_INT_COLUMN: &str = "r_reason_sk";
const DEFAULT_STRING_COLUMN: &str = "r_reason_id";
const DEFAULT_EXPECTED_ROWS: i64 = 55;
const DEFAULT_NAMESPACE: &str = "nexus_tpcds_sf100";
const DEFAULT_WAREHOUSE: &str =
    "s3://datafusion-nexus-tpcds-sf100-123456789012-us-west-2/warehouse/nexus_tpcds_sf100.db/";
const DEFAULT_REGION: &str = "us-west-2";
const CHILD_MODE_ENV: &str = "NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E_CHILD_MODE";
const CHILD_RESULT_PREFIX: &str = "NEXUS_KVIKIO_E2E_RESULT=";
const COMMAND: &str = "server-iceberg-glue-remote-direct-e2e";
const KVIKIO_MODES: [&str; 3] = ["ON", "OFF", "AUTO"];

pub async fn run() -> E2eResult<()> {
    require_remote_direct_e2e_enabled()?;
    if let Ok(mode) = std::env::var(CHILD_MODE_ENV) {
        return run_child(&mode).await;
    }
    run_independent_mode_processes().await
}

async fn run_child(mode: &str) -> E2eResult<()> {
    if !KVIKIO_MODES.contains(&mode) {
        return Err(format!("invalid {CHILD_MODE_ENV} value {mode:?}").into());
    }
    if std::env::var("KVIKIO_COMPAT_MODE").as_deref() != Ok(mode) {
        return Err(format!("child {mode} must start with KVIKIO_COMPAT_MODE={mode}").into());
    }
    let params = RunParams::from_env()?;
    let _env_guard = EnvGuard::install(&params)?;
    let local_dir = OwnedTempDir::new("datafusion-nexus-kvikio-local", mode)?;
    let local_path = local_dir.path().join("local-fixture.parquet");
    write_local_parquet_fixture(&local_path)?;
    let remote_and_local = run_remote_and_local_server_pass(&params, &local_path).await?;

    let cache_dir = OwnedTempDir::new("datafusion-nexus-kvikio-cache", mode)?;
    let cache = run_cache_server_pass(&params, cache_dir.path()).await?;
    if cache != remote_and_local.remote {
        return Err(format!("KvikIO {mode} fill-before-read cache results differ from remote URI results").into());
    }
    let result = ModeResults { remote: remote_and_local.remote, local: remote_and_local.local, cache };
    println!(
        "{CHILD_RESULT_PREFIX}{}",
        serde_json::json!({
            "mode": mode,
            "remote": query_result_json(&result.remote),
            "local": query_result_json(&result.local),
            "cache": query_result_json(&result.cache),
        })
    );
    Ok(())
}

async fn run_independent_mode_processes() -> E2eResult<()> {
    let executable = std::env::current_exe()?;
    let mut baseline: Option<ModeResults> = None;
    for mode in KVIKIO_MODES {
        eprintln!("[{mode}] launching independent KvikIO process");
        let output = tokio::process::Command::new(&executable)
            .arg(COMMAND)
            .env(CHILD_MODE_ENV, mode)
            .env("KVIKIO_COMPAT_MODE", mode)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;
        if !output.status.success() {
            return Err(format!(
                "KvikIO {mode} child failed with {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
            .into());
        }
        let result = parse_child_result(mode, &output.stdout)?;
        if let Some(expected) = &baseline {
            if result != *expected {
                return Err(format!("KvikIO {mode} remote/local/cache results differ from ON baseline").into());
            }
        } else {
            baseline = Some(result);
        }
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    println!("server_iceberg_glue_remote_direct_e2e ok modes=ON,OFF,AUTO substrates=remote,local,cache parity=exact");
    Ok(())
}

fn query_result_json(result: &QueryResults) -> serde_json::Value {
    serde_json::json!({
        "aggregate_rows": result.aggregate_rows,
        "projection_rows": result.projection_rows,
    })
}

fn parse_child_result(mode: &str, stdout: &[u8]) -> E2eResult<ModeResults> {
    let stdout = String::from_utf8(stdout.to_vec())?;
    let payload = stdout
        .lines()
        .find_map(|line| line.strip_prefix(CHILD_RESULT_PREFIX))
        .ok_or_else(|| format!("KvikIO {mode} child did not emit {CHILD_RESULT_PREFIX}"))?;
    let value: serde_json::Value = serde_json::from_str(payload)?;
    if value["mode"].as_str() != Some(mode) {
        return Err(format!("KvikIO child result mode mismatch for {mode}").into());
    }
    let query_results = |substrate: &str| -> E2eResult<QueryResults> {
        let substrate_value = value
            .get(substrate)
            .ok_or_else(|| e2e_error(format!("KvikIO {mode} child result is missing {substrate}")))?;
        let rows = |field: &str| -> E2eResult<Vec<String>> {
            substrate_value[field]
                .as_array()
                .ok_or_else(|| e2e_error(format!("KvikIO {mode} child {substrate} result is missing {field}")))?
                .iter()
                .map(|row| {
                    row.as_str().map(str::to_string).ok_or_else(|| {
                        e2e_error(format!("KvikIO {mode} child {substrate} {field} contains a non-string row"))
                    })
                })
                .collect()
        };
        Ok(QueryResults { aggregate_rows: rows("aggregate_rows")?, projection_rows: rows("projection_rows")? })
    };
    Ok(ModeResults { remote: query_results("remote")?, local: query_results("local")?, cache: query_results("cache")? })
}

fn require_remote_direct_e2e_enabled() -> E2eResult<()> {
    match std::env::var("NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E").as_deref() {
        Ok("1") => Ok(()),
        _ => Err("server Iceberg Glue remote-direct E2E requires NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E=1, \
                  read access to the configured Glue/S3 fixture, default-chain AWS credentials, and a GPU \
                  (serialize with `flock /tmp/cudf-gpu.lock`)"
            .into()),
    }
}

// ---------------------------------------------------------------------------
// Parameters and environment contract
// ---------------------------------------------------------------------------

struct RunParams {
    glue_uri: Option<String>,
    s3_endpoint: Option<String>,
    warehouse: String,
    s3_region: String,
    namespace: String,
    table: String,
    int_column: String,
    string_column: String,
    expected_rows: Option<i64>,
}

impl RunParams {
    fn from_env() -> E2eResult<Self> {
        let glue_uri = std::env::var(NEXUS_ICEBERG_GLUE_URI_ENV).ok().filter(|value| !value.is_empty());
        let s3_endpoint = std::env::var(NEXUS_ICEBERG_S3_ENDPOINT_ENV).ok().filter(|value| !value.is_empty());
        let warehouse = std::env::var("NEXUS_ICEBERG_WAREHOUSE").unwrap_or_else(|_| DEFAULT_WAREHOUSE.to_string());
        let s3_region = std::env::var(NEXUS_ICEBERG_S3_REGION_ENV).unwrap_or_else(|_| DEFAULT_REGION.to_string());
        if let Some(glue_uri) = &glue_uri {
            require_loopback_http_url(NEXUS_ICEBERG_GLUE_URI_ENV, glue_uri)?;
        }
        if let Some(s3_endpoint) = &s3_endpoint {
            require_loopback_http_url(NEXUS_ICEBERG_S3_ENDPOINT_ENV, s3_endpoint)?;
        }
        require_s3_warehouse("NEXUS_ICEBERG_WAREHOUSE", &warehouse)?;

        // The regression under test is the credential DEFAULT CHAIN: RustFS
        // credentials must come only from the ambient AWS env pair the
        // object-store provider resolves, never from static NEXUS_ICEBERG_S3_*.
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV)?;

        let namespace = std::env::var(NEXUS_ICEBERG_NAMESPACE_ENV).unwrap_or_else(|_| DEFAULT_NAMESPACE.to_string());
        let table =
            std::env::var("NEXUS_ICEBERG_REMOTE_DIRECT_E2E_TABLE").unwrap_or_else(|_| DEFAULT_TABLE.to_string());
        let int_column =
            std::env::var("NEXUS_ICEBERG_REMOTE_DIRECT_E2E_INT_COL").unwrap_or_else(|_| DEFAULT_INT_COLUMN.to_string());
        let string_column = std::env::var("NEXUS_ICEBERG_REMOTE_DIRECT_E2E_STR_COL")
            .unwrap_or_else(|_| DEFAULT_STRING_COLUMN.to_string());
        let expected_rows = match std::env::var("NEXUS_ICEBERG_REMOTE_DIRECT_E2E_EXPECTED_ROWS") {
            Ok(raw) => Some(raw.parse::<i64>().map_err(|_| {
                format!("NEXUS_ICEBERG_REMOTE_DIRECT_E2E_EXPECTED_ROWS must be an integer, got {raw:?}")
            })?),
            Err(_) if table == DEFAULT_TABLE => Some(DEFAULT_EXPECTED_ROWS),
            Err(_) => None,
        };

        Ok(Self {
            glue_uri,
            s3_endpoint,
            warehouse,
            s3_region,
            namespace,
            table,
            int_column,
            string_column,
            expected_rows,
        })
    }

    fn server_env(&self, cache_root: Option<&Path>) -> HashMap<&'static str, String> {
        let mut values = HashMap::from([
            (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
            (NEXUS_ICEBERG_NAMESPACE_ENV, self.namespace.clone()),
            (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, self.warehouse.clone()),
            (NEXUS_ICEBERG_S3_REGION_ENV, self.s3_region.clone()),
            (
                NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN.to_string(),
            ),
            (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, self.s3_region.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_DEFAULT_REGION_ENV, self.s3_region.clone()),
        ]);
        if let Some(glue_uri) = &self.glue_uri {
            values.insert(NEXUS_ICEBERG_GLUE_URI_ENV, glue_uri.clone());
        }
        if let Some(s3_endpoint) = &self.s3_endpoint {
            values.insert(NEXUS_ICEBERG_S3_ENDPOINT_ENV, s3_endpoint.clone());
            values.insert(NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string());
        }
        if let Some(cache_root) = cache_root {
            values.insert(NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, cache_root.display().to_string());
            values.insert(NEXUS_LAKEHOUSE_CACHE_READ_ENV, "enabled".to_string());
            values.insert(NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, "fill_before_read".to_string());
            values.insert(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, (1024_u64 * 1024 * 1024).to_string());
        }
        values
    }

    fn table_ref(&self) -> String {
        format!("glue.\"{}\".\"{}\"", self.namespace, self.table)
    }

    fn aggregate_sql(&self) -> String {
        format!(
            "SELECT count(\"{str_col}\") AS non_null_strs, min(\"{int_col}\") AS min_int, \
             max(\"{int_col}\") AS max_int FROM {table}",
            str_col = self.string_column,
            int_col = self.int_column,
            table = self.table_ref(),
        )
    }

    fn projection_sql(&self) -> String {
        format!(
            "SELECT \"{int_col}\", \"{str_col}\" FROM {table}",
            int_col = self.int_column,
            str_col = self.string_column,
            table = self.table_ref(),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
struct QueryResults {
    aggregate_rows: Vec<String>,
    projection_rows: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct ModeResults {
    remote: QueryResults,
    local: QueryResults,
    cache: QueryResults,
}

struct RemoteAndLocalResults {
    remote: QueryResults,
    local: QueryResults,
}

// ---------------------------------------------------------------------------
// Server passes for remote/local and local-cache KvikIO contracts
// ---------------------------------------------------------------------------

async fn run_remote_and_local_server_pass(params: &RunParams, local_path: &Path) -> E2eResult<RemoteAndLocalResults> {
    let label = std::env::var(CHILD_MODE_ENV).unwrap_or_else(|_| "unknown".to_string());
    let server_env = params.server_env(None);
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|name| server_env.get(name).cloned())?;
    config.bind_addr = addr;

    eprintln!("[{label}] starting Flight SQL server on {addr}");
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let result = validate_remote_and_local_kvikio(params, local_path, addr, &label).await;
    server.abort();
    let _ = server.await;
    result
}

async fn run_cache_server_pass(params: &RunParams, cache_root: &Path) -> E2eResult<QueryResults> {
    let mode = std::env::var(CHILD_MODE_ENV).unwrap_or_else(|_| "unknown".to_string());
    let label = format!("{mode}:cache");
    let server_env = params.server_env(Some(cache_root));
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|name| server_env.get(name).cloned())?;
    config.bind_addr = addr;

    eprintln!("[{label}] starting Flight SQL server on {addr}");
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let result = validate_cache_kvikio(params, addr, &label).await;
    server.abort();
    let _ = server.await;
    result
}

async fn validate_remote_and_local_kvikio(
    params: &RunParams,
    local_path: &Path,
    addr: std::net::SocketAddr,
    label: &str,
) -> E2eResult<RemoteAndLocalResults> {
    let mut client = connect_flight_sql(addr).await?;
    let remote = execute_remote_data_queries(params, &mut client, label).await?;
    validate_remote_evidence(params, &mut client, label).await?;
    let local = validate_local_kvikio(local_path, &mut client, label).await?;
    Ok(RemoteAndLocalResults { remote, local })
}

async fn execute_remote_data_queries(
    params: &RunParams,
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    label: &str,
) -> E2eResult<QueryResults> {
    // COUNT(*) is sanity only: MetadataRowCount sources answer it from Iceberg
    // metadata without opening the remote Parquet source.
    let count_sql = format!("SELECT count(*) AS row_count FROM {}", params.table_ref());
    let row_count = query_single_int(client, &count_sql).await?;
    eprintln!("[{label}] {} row_count={row_count}", params.table_ref());
    match params.expected_rows {
        Some(expected) if row_count != expected => {
            return Err(format!("[{label}] expected {expected} rows in {}, got {row_count}", params.table_ref()).into());
        },
        None if row_count <= 0 => {
            return Err(format!(
                "[{label}] remote-direct E2E requires a data-bearing table, got {row_count} rows in {}",
                params.table_ref()
            )
            .into());
        },
        _ => {},
    }

    // The real regression queries: a column aggregate and a projection must
    // open the Parquet source through the substrate under test.
    let aggregate_rows = sorted_rows(&query_batches(client, &params.aggregate_sql()).await?);
    if aggregate_rows.len() != 1 {
        return Err(format!("[{label}] expected one aggregate row, got {aggregate_rows:?}").into());
    }
    eprintln!("[{label}] column aggregate: {aggregate_rows:?}");

    let projection_batches = query_batches(client, &params.projection_sql()).await?;
    let projection_rows = sorted_rows(&projection_batches);
    if projection_rows.len() != row_count as usize {
        return Err(
            format!("[{label}] projection returned {} rows, expected {row_count}", projection_rows.len()).into()
        );
    }

    let limit_sql = format!("{} LIMIT 5", params.projection_sql());
    let limit_rows = sorted_rows(&query_batches(client, &limit_sql).await?);
    let expected_limit_rows = (row_count as usize).min(5);
    if limit_rows.len() != expected_limit_rows {
        return Err(
            format!("[{label}] LIMIT 5 returned {} rows, expected {expected_limit_rows}", limit_rows.len()).into()
        );
    }

    Ok(QueryResults { aggregate_rows, projection_rows })
}

async fn validate_remote_evidence(
    params: &RunParams,
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    label: &str,
) -> E2eResult<()> {
    // Planning evidence: the projection must rewrite natively and plan the
    // requested read strategy. Unlike min/max aggregates, it cannot be folded
    // entirely from Parquet statistics.
    let explain_text = query_plan_text(client, &format!("EXPLAIN VERBOSE {}", params.projection_sql())).await?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "NexusNativeExec")?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "source_read_strategy=remote_direct")?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "credential_mode=scoped_resolved")?;

    // Runtime evidence comes from the retained query record because DataFusion's
    // EXPLAIN ANALYZE display only renders the count of nested source-read rows.
    let (_batches, runtime_text) = query_batches_with_stats(client, &params.projection_sql()).await?;
    require_contains(label, "GetQueryStats", &runtime_text, "[source_read_runtime]")?;
    require_contains(label, "GetQueryStats", &runtime_text, "read_strategy=remote_direct")?;
    require_contains(label, "GetQueryStats", &runtime_text, "remote_read_mode=require_kvikio")?;
    require_any_contains(
        label,
        "GetQueryStats",
        &runtime_text,
        &["parquet_read_input_kind=remote_path", "parquet_read_input_kind=remote_path_multi"],
    )?;
    require_removed_labels_absent(label, "GetQueryStats", &runtime_text)?;
    Ok(())
}

async fn validate_local_kvikio(
    local_path: &Path,
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    mode_label: &str,
) -> E2eResult<QueryResults> {
    let label = format!("{mode_label}:local");
    let table = "kvikio_local_fixture";
    query_batches(
        client,
        &format!("CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{}'", local_path.to_string_lossy()),
    )
    .await?;
    let aggregate_sql = format!("SELECT count(local_name), min(local_id), max(local_id) FROM {table}");
    let projection_sql = format!("SELECT local_id, local_name FROM {table}");
    let aggregate_rows = sorted_rows(&query_batches(client, &aggregate_sql).await?);
    let projection_rows = sorted_rows(&query_batches(client, &projection_sql).await?);
    if aggregate_rows != ["4\u{1f}3\u{1f}13"] || projection_rows.len() != 4 {
        return Err(
            format!("[{label}] unexpected local fixture results: {aggregate_rows:?} {projection_rows:?}").into()
        );
    }

    let explain_text = query_plan_text(client, &format!("EXPLAIN VERBOSE {projection_sql}")).await?;
    require_contains(&label, "EXPLAIN VERBOSE", &explain_text, "NexusNativeExec")?;
    let (_batches, runtime_text) = query_batches_with_stats(client, &projection_sql).await?;
    require_contains(&label, "GetQueryStats", &runtime_text, "[source_read_runtime]")?;
    require_contains(&label, "GetQueryStats", &runtime_text, "read_strategy=local_kvikio")?;
    require_contains(&label, "GetQueryStats", &runtime_text, "parquet_read_input_kind=local_path")?;
    require_removed_labels_absent(&label, "GetQueryStats", &runtime_text)?;

    Ok(QueryResults { aggregate_rows, projection_rows })
}

async fn validate_cache_kvikio(params: &RunParams, addr: std::net::SocketAddr, label: &str) -> E2eResult<QueryResults> {
    let mut client = connect_flight_sql(addr).await?;
    let results = execute_remote_data_queries(params, &mut client, label).await?;

    // The data query above fills every selected file before opening it. A new
    // plan must therefore resolve to a complete local cache path.
    let explain_text = query_plan_text(&mut client, &format!("EXPLAIN VERBOSE {}", params.projection_sql())).await?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "NexusNativeExec")?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "source_read_strategy=local_cache_hit")?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "lakehouse_cache_status=hit")?;
    let (_batches, runtime_text) = query_batches_with_stats(&mut client, &params.projection_sql()).await?;
    require_contains(label, "GetQueryStats", &runtime_text, "[source_read_runtime]")?;
    require_contains(label, "GetQueryStats", &runtime_text, "read_strategy=local_cache_kvikio")?;
    require_contains(label, "GetQueryStats", &runtime_text, "parquet_read_input_kind=local_path")?;
    require_removed_labels_absent(label, "GetQueryStats", &runtime_text)?;

    Ok(results)
}

fn require_removed_labels_absent(label: &str, surface: &str, runtime_text: &str) -> E2eResult<()> {
    for removed in [
        concat!("host_", "buffer"),
        concat!("source_", "fetch_bytes"),
        concat!("source_", "fetch_count"),
        concat!("host_", "fetch_budget_wait_time"),
        concat!("source_", "fetch_concurrency_wait_time"),
        concat!("source_", "fetch_time"),
        concat!("source_", "decode_time"),
        concat!("source_decode_", "input"),
        concat!("source_fallback_", "reason"),
        concat!("parquet_read_fallback_", "reason"),
    ] {
        require_absent(label, surface, runtime_text, removed)?;
    }
    Ok(())
}

fn require_contains(label: &str, surface: &str, haystack: &str, needle: &str) -> E2eResult<()> {
    if haystack.contains(needle) {
        return Ok(());
    }
    Err(format!("[{label}] {surface} output must contain `{needle}`:\n{haystack}").into())
}

fn require_any_contains(label: &str, surface: &str, haystack: &str, needles: &[&str]) -> E2eResult<()> {
    if needles.iter().any(|needle| haystack.contains(needle)) {
        return Ok(());
    }
    Err(format!("[{label}] {surface} output must contain one of {needles:?}:\n{haystack}").into())
}

fn require_absent(label: &str, surface: &str, haystack: &str, needle: &str) -> E2eResult<()> {
    if !haystack.contains(needle) {
        return Ok(());
    }
    Err(format!("[{label}] {surface} output contains removed label `{needle}`:\n{haystack}").into())
}

// ---------------------------------------------------------------------------
// Flight SQL helpers
// ---------------------------------------------------------------------------

async fn query_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<Vec<arrow::record_batch::RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket =
        flight_info.endpoint.first().and_then(|ep| ep.ticket.clone()).ok_or("Flight SQL response missing ticket")?;
    Ok(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?)
}

async fn query_batches_with_stats(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<(Vec<RecordBatch>, String)> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let ticket =
        flight_info.endpoint.first().and_then(|ep| ep.ticket.clone()).ok_or("Flight SQL response missing ticket")?;
    let batches = client.do_get(ticket).await?.try_collect::<Vec<_>>().await?;
    let action = Action { r#type: "GetQueryStats".to_string(), body: query_id.as_bytes().to_vec().into() };
    let mut stream = client.do_action(action).await?;
    let result = stream.message().await?.ok_or("GetQueryStats returned no result")?;
    if stream.message().await?.is_some() {
        return Err("GetQueryStats returned more than one result".into());
    }
    Ok((batches, String::from_utf8(result.body.to_vec())?))
}

fn query_id_from_statement_flight_info(flight_info: &FlightInfo) -> E2eResult<String> {
    let ticket = flight_info
        .endpoint
        .first()
        .and_then(|endpoint| endpoint.ticket.as_ref())
        .ok_or("FlightInfo missing statement ticket")?;
    let any = Any::decode(&*ticket.ticket)?;
    let statement = any.unpack::<TicketStatementQuery>()?.ok_or("FlightInfo ticket is not TicketStatementQuery")?;
    Ok(String::from_utf8(statement.statement_handle.to_vec())?)
}

async fn query_single_int(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, sql: &str) -> E2eResult<i64> {
    let batches = query_batches(client, sql).await?;
    if batches.len() != 1 || batches[0].num_rows() != 1 {
        return Err(format!(
            "expected one single-row batch for {sql:?}, got {} batches and first rows {:?}",
            batches.len(),
            batches.first().map(|batch| batch.num_rows())
        )
        .into());
    }
    int_value(batches[0].column(0)).ok_or_else(|| format!("could not read integer result of {sql:?}").into())
}

/// Renders every Utf8 cell of an EXPLAIN result into one searchable string.
async fn query_plan_text(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<String> {
    let batches = query_batches(client, sql).await?;
    let mut text = String::new();
    for batch in &batches {
        for column in batch.columns() {
            if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                for row in 0..strings.len() {
                    if !strings.is_null(row) {
                        text.push_str(strings.value(row));
                        text.push('\n');
                    }
                }
            } else if let Some(strings) = column.as_any().downcast_ref::<LargeStringArray>() {
                for row in 0..strings.len() {
                    if !strings.is_null(row) {
                        text.push_str(strings.value(row));
                        text.push('\n');
                    }
                }
            }
        }
    }
    if text.is_empty() {
        return Err(format!("{sql:?} returned no plan text").into());
    }
    Ok(text)
}

fn int_value(array: &arrow::array::ArrayRef) -> Option<i64> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|a| a.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|a| i64::from(a.value(0))))
        .or_else(|| array.as_any().downcast_ref::<UInt64Array>().map(|a| a.value(0) as i64))
        .or_else(|| array.as_any().downcast_ref::<UInt32Array>().map(|a| i64::from(a.value(0))))
}

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it before spawning the server keeps that
/// cost off the connect window, and the `OnceLock` collapses concurrent callers
/// onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = datafusion_nexus::server::init_cuda_device(0);
    });
}

/// Reserve a loopback port and hand back the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where another process could steal the port.
fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

async fn connect_flight_sql(
    addr: std::net::SocketAddr,
) -> E2eResult<FlightSqlServiceClient<tonic::transport::Channel>> {
    let uri = format!("http://{addr}");
    connect_with_retry(
        &uri,
        ConnectionRetryConfig::exponential_timeout(
            Duration::from_secs(15),
            Duration::from_millis(10),
            Duration::from_millis(500),
        ),
        || Ok(BeforeConnect::Continue),
        FlightSqlServiceClient::new,
        |error| format!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}"),
    )
    .await
}

/// Renders RecordBatches to a sorted Vec of row strings for order-insensitive,
/// substrate-agnostic parity comparison.
fn sorted_rows(batches: &[arrow::record_batch::RecordBatch]) -> Vec<String> {
    use std::fmt::Write;
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let mut rendered = String::new();
            for col in 0..batch.num_columns() {
                let array = batch.column(col);
                if col > 0 {
                    rendered.push('\u{1f}');
                }
                if array.is_null(row) {
                    rendered.push_str("\u{0}NULL");
                } else {
                    let _ = write!(
                        rendered,
                        "{}",
                        arrow::util::display::array_value_to_string(array, row).unwrap_or_default()
                    );
                }
            }
            rows.push(rendered);
        }
    }
    rows.sort();
    rows
}

// ---------------------------------------------------------------------------
// Per-child local fixtures
// ---------------------------------------------------------------------------

struct OwnedTempDir {
    path: PathBuf,
}

impl OwnedTempDir {
    fn new(prefix: &str, mode: &str) -> E2eResult<Self> {
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("{prefix}-{}-{mode}-{suffix}", std::process::id()));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for OwnedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write_local_parquet_fixture(path: &Path) -> E2eResult<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("local_id", DataType::Int64, false),
        Field::new("local_name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![3_i64, 5, 8, 13])) as ArrayRef,
            Arc::new(StringArray::from(vec!["three", "five", "eight", "thirteen"])) as ArrayRef,
        ],
    )?;
    let file = fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build()))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Environment validation helpers
// ---------------------------------------------------------------------------

fn require_loopback_http_url(name: &str, value: &str) -> E2eResult<()> {
    if is_loopback_http_url(value) {
        return Ok(());
    }
    Err(format!("{name} must be a loopback http:// URL for local Glue/Moto remote-direct E2E, got {value:?}").into())
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

fn require_s3_warehouse(name: &str, value: &str) -> E2eResult<()> {
    if value.strip_prefix("s3://").and_then(|rest| rest.split('/').next()).is_some_and(|bucket| !bucket.is_empty()) {
        return Ok(());
    }
    Err(format!("{name} must be an s3://bucket URI for local Glue/Moto remote-direct E2E, got {value:?}").into())
}

fn reject_non_empty_env(name: &str) -> E2eResult<()> {
    if std::env::var(name).is_ok_and(|value| !value.is_empty()) {
        return Err(format!(
            "the KvikIO-only E2E validates the AWS default credential chain; unset static Nexus credential {name}"
        )
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Process-env guard
// ---------------------------------------------------------------------------

struct EnvGuard {
    original: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    /// Pins both standard AWS region spellings before any AWS/KvikIO client is
    /// initialized. Credentials remain entirely owned by the inherited
    /// default chain (environment, profile, web identity, or instance role).
    fn install(params: &RunParams) -> E2eResult<Self> {
        let values: HashMap<&'static str, String> = HashMap::from([
            (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, params.s3_region.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_DEFAULT_REGION_ENV, params.s3_region.clone()),
        ]);
        let mut names: Vec<&'static str> = values.keys().copied().collect();
        names.sort_unstable();
        names.dedup();

        let original = names.iter().map(|name| (*name, std::env::var(name).ok())).collect::<Vec<_>>();
        for name in names {
            let value = values.get(name).expect("region value exists");
            // This single-purpose E2E child installs process env before
            // starting any background work that reads AWS configuration.
            unsafe { std::env::set_var(name, value) };
        }
        Ok(Self { original })
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.original.iter().rev() {
            match value {
                Some(value) => {
                    // Restore process env after the local servers have been
                    // aborted; no other threads in this tool should read it.
                    unsafe { std::env::set_var(name, value) };
                },
                None => {
                    unsafe { std::env::remove_var(name) };
                },
            }
        }
    }
}
