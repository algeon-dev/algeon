//! Credentialless Glue/S3 remote-direct regression E2E (local Moto + RustFS).
//!
//! Regression coverage for the 2026-06-09 QA finding that GPU remote-direct
//! (KvikIO) column reads failed without static AWS credentials while the
//! host-buffer substrate worked: the engine now resolves the AWS default
//! credential chain through the same `object_store` provider host-buffer
//! fetches use. This runner validates that plumbing end to end through the
//! Flight SQL server against a data-bearing local Glue/Moto + RustFS table:
//!
//! 1. RustFS credentials are supplied ONLY through the provider path
//!    (process-env `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`), never via the
//!    static `NEXUS_ICEBERG_S3_*` credential env vars, and the storage profile
//!    is explicitly opted into the default chain with
//!    `NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE=default_chain`.
//! 2. A `remote_kvikio` + `require_direct` server must run real column
//!    aggregates and projections (not only `COUNT(*)`, which is served from
//!    Iceberg metadata without opening the source), with EXPLAIN ANALYZE
//!    evidence `source_read_boundary_direct_device >= 1` and
//!    `source_fetch_bytes == 0`.
//! 3. A `host_buffer` control server with the same credentialless environment
//!    must return identical results, with `source_fetch_count >= 1` host
//!    object-store fetch evidence, so the two substrates can never diverge on
//!    credential behavior. (`source_read_boundary_direct_device` records the
//!    selected decode-boundary policy and increments on both substrates; only
//!    the fetch counters distinguish them.)
//!
//! GPU is required (native remote-direct execution). Serialize with
//! `flock /tmp/cudf-gpu.lock` per repo policy. Load a data-bearing namespace
//! first (`fixture/scripts/e2e.sh iceberg-local --load-glue ...` or
//! `fixture/scripts/fixture.sh iceberg glue-moto load`).
//!
//! ```bash
//! NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E=1 \
//!   NEXUS_ICEBERG_NAMESPACE=tpch_sf1 \
//!   AWS_ACCESS_KEY_ID=cudfadmin \
//!   AWS_SECRET_ACCESS_KEY=cudfadminsecret \
//!   flock /tmp/cudf-gpu.lock \
//!   cargo run -p datafusion-nexus-tools --features server-iceberg-e2e -- \
//!   server-iceberg-glue-remote-direct-e2e
//! ```

use std::{collections::HashMap, net::TcpListener, time::Duration};

use arrow::array::{Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion_nexus::server::{
    NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
    NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
    NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV, ServerConfig, serve_with_listener,
};
use futures::TryStreamExt;
use tonic::transport::Endpoint;
use url::Url;

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const DEFAULT_TABLE: &str = "nation";
const DEFAULT_INT_COLUMN: &str = "n_nationkey";
const DEFAULT_STRING_COLUMN: &str = "n_name";
const DEFAULT_NATION_ROWS: i64 = 25;

pub async fn run() -> E2eResult<()> {
    require_remote_direct_e2e_enabled()?;
    let params = RunParams::from_env()?;
    let _env_guard = EnvGuard::install(&params)?;

    let remote = run_server_pass(&params, ReadSubstrate::RemoteDirect).await?;
    let control = run_server_pass(&params, ReadSubstrate::HostBufferControl).await?;

    if remote.aggregate_rows != control.aggregate_rows {
        return Err(format!(
            "remote-direct vs host-buffer column aggregate mismatch:\n  remote : {:?}\n  control: {:?}",
            remote.aggregate_rows, control.aggregate_rows
        )
        .into());
    }
    if remote.projection_rows != control.projection_rows {
        return Err(format!(
            "remote-direct vs host-buffer projection mismatch ({} vs {} rows)",
            remote.projection_rows.len(),
            control.projection_rows.len()
        )
        .into());
    }
    eprintln!(
        "remote-direct/host-buffer parity OK: {} aggregate row(s), {} projection row(s)",
        remote.aggregate_rows.len(),
        remote.projection_rows.len()
    );
    println!("server_iceberg_glue_remote_direct_e2e ok");
    Ok(())
}

fn require_remote_direct_e2e_enabled() -> E2eResult<()> {
    match std::env::var("NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E").as_deref() {
        Ok("1") => Ok(()),
        _ => Err("server Iceberg Glue remote-direct E2E requires NEXUS_ICEBERG_GLUE_REMOTE_DIRECT_E2E=1, \
                  fixture/iceberg-local Moto + RustFS, a loaded data-bearing namespace, and a GPU \
                  (serialize with `flock /tmp/cudf-gpu.lock`)"
            .into()),
    }
}

// ---------------------------------------------------------------------------
// Parameters and environment contract
// ---------------------------------------------------------------------------

struct RunParams {
    glue_uri: String,
    s3_endpoint: String,
    warehouse: String,
    s3_region: String,
    namespace: String,
    table: String,
    int_column: String,
    string_column: String,
    expected_rows: Option<i64>,
    aws_access_key_id: String,
    aws_secret_access_key: String,
}

impl RunParams {
    fn from_env() -> E2eResult<Self> {
        let glue_uri =
            std::env::var(NEXUS_ICEBERG_GLUE_URI_ENV).unwrap_or_else(|_| "http://localhost:5001".to_string());
        let s3_endpoint =
            std::env::var(NEXUS_ICEBERG_S3_ENDPOINT_ENV).unwrap_or_else(|_| "http://localhost:9000".to_string());
        let warehouse =
            std::env::var("NEXUS_ICEBERG_WAREHOUSE").unwrap_or_else(|_| "s3://lakehouse/warehouse".to_string());
        let s3_region = std::env::var(NEXUS_ICEBERG_S3_REGION_ENV).unwrap_or_else(|_| "us-east-1".to_string());
        require_loopback_http_url(NEXUS_ICEBERG_GLUE_URI_ENV, &glue_uri)?;
        require_loopback_http_url(NEXUS_ICEBERG_S3_ENDPOINT_ENV, &s3_endpoint)?;
        require_s3_warehouse("NEXUS_ICEBERG_WAREHOUSE", &warehouse)?;

        // The regression under test is the credential DEFAULT CHAIN: RustFS
        // credentials must come only from the ambient AWS env pair the
        // object-store provider resolves, never from static NEXUS_ICEBERG_S3_*.
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::AWS_SESSION_TOKEN_ENV)?;
        reject_non_empty_env(datafusion_nexus::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)?;

        let namespace = std::env::var(NEXUS_ICEBERG_NAMESPACE_ENV).map_err(|_| {
            format!("set {NEXUS_ICEBERG_NAMESPACE_ENV} to a loaded data-bearing namespace such as tpch_sf1")
        })?;
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
            Err(_) if table == DEFAULT_TABLE => Some(DEFAULT_NATION_ROWS),
            Err(_) => None,
        };

        let aws_access_key_id = std::env::var(datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)
            .unwrap_or_else(|_| "cudfadmin".to_string());
        let aws_secret_access_key = std::env::var(datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV)
            .unwrap_or_else(|_| "cudfadminsecret".to_string());

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
            aws_access_key_id,
            aws_secret_access_key,
        })
    }

    fn server_env(&self, substrate: ReadSubstrate) -> HashMap<&'static str, String> {
        let (read_strategy, read_mode) = match substrate {
            ReadSubstrate::RemoteDirect => ("remote_kvikio", "require_direct"),
            ReadSubstrate::HostBufferControl => ("host_buffer", "allow_host_fallback"),
        };
        HashMap::from([
            (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
            (NEXUS_ICEBERG_NAMESPACE_ENV, self.namespace.clone()),
            (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, self.warehouse.clone()),
            (NEXUS_ICEBERG_GLUE_URI_ENV, self.glue_uri.clone()),
            (NEXUS_ICEBERG_S3_ENDPOINT_ENV, self.s3_endpoint.clone()),
            (NEXUS_ICEBERG_S3_REGION_ENV, self.s3_region.clone()),
            (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
            (
                NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN.to_string(),
            ),
            (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, self.s3_region.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, self.aws_access_key_id.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, self.aws_secret_access_key.clone()),
            (datafusion_nexus::table_format::NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, read_strategy.to_string()),
            (datafusion_nexus::table_format::NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV, read_mode.to_string()),
        ])
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadSubstrate {
    RemoteDirect,
    HostBufferControl,
}

impl ReadSubstrate {
    fn label(self) -> &'static str {
        match self {
            Self::RemoteDirect => "remote_direct",
            Self::HostBufferControl => "host_buffer_control",
        }
    }
}

struct SubstrateResults {
    aggregate_rows: Vec<String>,
    projection_rows: Vec<String>,
}

// ---------------------------------------------------------------------------
// One server pass per read substrate
// ---------------------------------------------------------------------------

async fn run_server_pass(params: &RunParams, substrate: ReadSubstrate) -> E2eResult<SubstrateResults> {
    let label = substrate.label();
    let server_env = params.server_env(substrate);
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|name| server_env.get(name).cloned())?;
    config.bind_addr = addr;

    eprintln!("[{label}] starting Flight SQL server on {addr}");
    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let result = validate_substrate(params, substrate, addr).await;
    server.abort();
    result
}

async fn validate_substrate(
    params: &RunParams,
    substrate: ReadSubstrate,
    addr: std::net::SocketAddr,
) -> E2eResult<SubstrateResults> {
    let label = substrate.label();
    let mut client = connect_with_retry(addr).await?;

    // COUNT(*) is sanity only: MetadataRowCount sources answer it from Iceberg
    // metadata without opening the remote Parquet source.
    let count_sql = format!("SELECT count(*) AS row_count FROM {}", params.table_ref());
    let row_count = query_single_int(&mut client, &count_sql).await?;
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
    let aggregate_rows = sorted_rows(&query_batches(&mut client, &params.aggregate_sql()).await?);
    if aggregate_rows.len() != 1 {
        return Err(format!("[{label}] expected one aggregate row, got {aggregate_rows:?}").into());
    }
    eprintln!("[{label}] column aggregate: {aggregate_rows:?}");

    let projection_batches = query_batches(&mut client, &params.projection_sql()).await?;
    let projection_rows = sorted_rows(&projection_batches);
    if projection_rows.len() != row_count as usize {
        return Err(
            format!("[{label}] projection returned {} rows, expected {row_count}", projection_rows.len()).into()
        );
    }

    let limit_sql = format!("{} LIMIT 5", params.projection_sql());
    let limit_rows = sorted_rows(&query_batches(&mut client, &limit_sql).await?);
    let expected_limit_rows = (row_count as usize).min(5);
    if limit_rows.len() != expected_limit_rows {
        return Err(
            format!("[{label}] LIMIT 5 returned {} rows, expected {expected_limit_rows}", limit_rows.len()).into()
        );
    }

    // Planning evidence: the column aggregate must rewrite natively and plan
    // the requested read strategy.
    let explain_text = query_plan_text(&mut client, &format!("EXPLAIN VERBOSE {}", params.aggregate_sql())).await?;
    require_contains(label, "EXPLAIN VERBOSE", &explain_text, "NativeCudfExec")?;
    if substrate == ReadSubstrate::RemoteDirect {
        require_contains(label, "EXPLAIN VERBOSE", &explain_text, "source_read_strategy=remote_direct")?;
        require_contains(label, "EXPLAIN VERBOSE", &explain_text, "credential_mode=scoped_resolved")?;
    }

    // Runtime evidence from EXPLAIN ANALYZE on the column aggregate.
    let analyze_text = query_plan_text(&mut client, &format!("EXPLAIN ANALYZE {}", params.aggregate_sql())).await?;
    let direct_device: u64 = metric_values(&analyze_text, "source_read_boundary_direct_device").iter().sum();
    let fetch_bytes: u64 = metric_values(&analyze_text, "source_fetch_bytes").iter().sum();
    let fetch_count: u64 = metric_values(&analyze_text, "source_fetch_count").iter().sum();
    eprintln!(
        "[{label}] source_read_boundary_direct_device={direct_device} \
         source_fetch_bytes={fetch_bytes} source_fetch_count={fetch_count}"
    );
    match substrate {
        ReadSubstrate::RemoteDirect => {
            if direct_device < 1 {
                return Err(format!(
                    "[{label}] expected source_read_boundary_direct_device >= 1 for remote-direct column reads; \
                     EXPLAIN ANALYZE:\n{analyze_text}"
                )
                .into());
            }
            if fetch_bytes != 0 || fetch_count != 0 {
                return Err(format!(
                    "[{label}] remote-direct column reads must not fall back to host-buffer object-store fetch \
                     (source_fetch_bytes={fetch_bytes}, source_fetch_count={fetch_count}); EXPLAIN ANALYZE:\n{analyze_text}"
                )
                .into());
            }
        },
        ReadSubstrate::HostBufferControl => {
            if fetch_count < 1 {
                return Err(format!(
                    "[{label}] host-buffer control should fetch through the object store \
                     (source_fetch_count={fetch_count}); EXPLAIN ANALYZE:\n{analyze_text}"
                )
                .into());
            }
        },
    }

    Ok(SubstrateResults { aggregate_rows, projection_rows })
}

fn require_contains(label: &str, surface: &str, haystack: &str, needle: &str) -> E2eResult<()> {
    if haystack.contains(needle) {
        return Ok(());
    }
    Err(format!("[{label}] {surface} output must contain `{needle}`:\n{haystack}").into())
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

/// Extracts every `name=<integer>` occurrence from rendered plan metrics.
fn metric_values(plan_text: &str, name: &str) -> Vec<u64> {
    let needle = format!("{name}=");
    let mut values = Vec::new();
    let mut rest = plan_text;
    while let Some(position) = rest.find(&needle) {
        let after = &rest[position + needle.len()..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(value) = digits.parse::<u64>() {
            values.push(value);
        }
        rest = after;
    }
    values
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

async fn connect_with_retry(
    addr: std::net::SocketAddr,
) -> E2eResult<FlightSqlServiceClient<tonic::transport::Channel>> {
    let uri = format!("http://{addr}");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut backoff = Duration::from_millis(10);
    loop {
        match Endpoint::from_shared(uri.clone())?.connect().await {
            Ok(channel) => return Ok(FlightSqlServiceClient::new(channel)),
            Err(error) => {
                if std::time::Instant::now() >= deadline {
                    return Err(format!(
                        "Flight SQL server did not accept connections at {addr} within 15s: {error:?}"
                    )
                    .into());
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_millis(500));
            },
        }
    }
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
            "the remote-direct E2E validates the AWS default credential chain; unset {name} so RustFS \
             credentials flow only through process-env AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
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
    /// Installs the credentialless process environment: the ambient AWS
    /// key pair (for the default chain, the Moto Glue client, and OpenDAL
    /// FileIO config load) plus region, with every static `NEXUS_ICEBERG_S3_*`
    /// credential and ambient session token cleared.
    fn install(params: &RunParams) -> E2eResult<Self> {
        let values: HashMap<&'static str, String> = HashMap::from([
            (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, params.aws_access_key_id.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, params.aws_secret_access_key.clone()),
            (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, params.s3_region.clone()),
        ]);
        let mut names: Vec<&'static str> = values.keys().copied().collect();
        names.extend([
            datafusion_nexus::table_format::iceberg::AWS_SESSION_TOKEN_ENV,
            datafusion_nexus::table_format::iceberg::AWS_SECURITY_TOKEN_ENV,
            datafusion_nexus::table_format::iceberg::AWS_PROFILE_ENV,
            datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV,
            datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV,
            datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
        ]);
        names.sort_unstable();
        names.dedup();

        let original = names.iter().map(|name| (*name, std::env::var(name).ok())).collect::<Vec<_>>();
        for name in names {
            if let Some(value) = values.get(name) {
                // This single-purpose E2E binary installs process env before
                // starting any background work that reads AWS credentials.
                unsafe { std::env::set_var(name, value) };
            } else {
                // Clear static credential overrides and ambient session state
                // so only the default-chain env pair can satisfy reads.
                unsafe { std::env::remove_var(name) };
            }
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
