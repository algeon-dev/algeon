//! Black-box E2E for the embedded dual-frontend backend fixture.
//!
//! This runner intentionally lives outside Cargo test targets. It starts the
//! fixture binary as a child process, registers the citation-network Parquet
//! files through Flight SQL, talks to REST with HTTP and to Flight SQL with
//! gRPC, and keeps all GPU use behind the caller's `flock`.

use std::{
    env,
    error::Error,
    fs,
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{
        Array, ArrayRef, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray,
        UInt32Array, UInt64Array,
    },
    record_batch::RecordBatch,
};
use arrow_flight::{
    FlightInfo, decode::FlightRecordBatchStream, error::FlightError, sql::client::FlightSqlServiceClient,
};
use futures::TryStreamExt;
use reqwest::{Client, StatusCode, header};
use serde_json::{Value, json};
use tokio::{
    process::{Child, Command},
    select,
    task::JoinHandle,
    time::{self, Instant},
};
use tonic::{Code, transport::Channel};

use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;
type FlightClient = FlightSqlServiceClient<Channel>;

const ENABLE_ENV: &str = "ALGEON_EMBEDDED_BACKEND_E2E";
const CITATION_DATA_ENV: &str = "ALGEON_EMBEDDED_BACKEND_E2E_CITATION_DATA";
const DEFAULT_CITATION_DATA: &str = "/mnt/algeon-nvme/algeon/fixture/demo/citation_network/v1/parquet/";
/// Table name and Parquet file registered before the checks, as in the citation-network guide.
const CITATION_TABLES: [(&str, &str); 5] = [
    ("citation_edges", "edges_by_src.parquet"),
    ("citation_edges_by_dst", "edges_by_dst.parquet"),
    ("papers", "profiles_by_paper_id.parquet"),
    ("paper_authors", "paper_authors.parquet"),
    ("paper_fos", "paper_fos.parquet"),
];
const EXPECTED_PAPERS: i64 = 4_894_081;
const EXPECTED_CITATION_EDGES: i64 = 45_564_149;
const QUEUE_FILL_REQUESTS: usize = 64;
const QUEUE_WAIT_MS: u64 = 120_000;
/// A native scan whose result exceeds the transport buffers; an unread stream keeps its attempt
/// admitted.
const HOLD_SQL: &str = "SELECT src, dst, weight FROM citation_edges";

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage();
        return Ok(());
    }
    let params = RunParams::from_args(args)?;
    build_fixture_binary(&params.repo_root).await?;

    let mut child = spawn_fixture(&params)?;
    let result = run_checks(&params, &mut child).await;
    if result.is_err() {
        terminate_child_best_effort(&mut child).await;
    }
    result?;

    println!("embedded_backend_e2e ok");
    Ok(())
}

#[derive(Clone)]
struct RunParams {
    repo_root: PathBuf,
    rest_bind: SocketAddr,
    flight_bind: SocketAddr,
    rest_base: String,
    flight_url: String,
    citation_data: PathBuf,
}

impl RunParams {
    fn from_args(args: Vec<String>) -> E2eResult<Self> {
        require_enabled()?;
        let mut citation_data = None;
        let mut remaining = args.into_iter();
        while let Some(arg) = remaining.next() {
            match arg.as_str() {
                "--citation-data" => {
                    let value = remaining.next().ok_or("missing value for --citation-data")?;
                    citation_data = Some(PathBuf::from(value));
                },
                other => return Err(format!("unknown embedded-backend-e2e argument: {other}").into()),
            }
        }
        let citation_data = citation_data
            .or_else(|| env::var_os(CITATION_DATA_ENV).filter(|value| !value.is_empty()).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CITATION_DATA));
        for (_, file) in CITATION_TABLES {
            let path = citation_data.join(file);
            if !path.is_file() {
                return Err(format!("citation data directory must contain {file}: {}", path.display()).into());
            }
        }

        let rest_bind = free_loopback_addr()?;
        let flight_bind = free_loopback_addr()?;
        Ok(Self {
            repo_root: repo_root(),
            rest_bind,
            flight_bind,
            rest_base: format!("http://{rest_bind}"),
            flight_url: format!("http://{flight_bind}"),
            citation_data,
        })
    }

    fn fixture_binary(&self) -> PathBuf {
        self.target_dir().join("debug").join(format!("embedded_backend{}", env::consts::EXE_SUFFIX))
    }

    fn server_config_path(&self) -> PathBuf {
        self.target_dir().join("embedded_backend_e2e/server.toml")
    }

    fn target_dir(&self) -> PathBuf {
        env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .map(|path| if path.is_absolute() { path } else { self.repo_root.join(path) })
            .unwrap_or_else(|| self.repo_root.join("target"))
    }
}

fn require_enabled() -> E2eResult<()> {
    match env::var(ENABLE_ENV).as_deref() {
        Ok("1") => Ok(()),
        _ => Err(format!(
            "embedded backend E2E requires {ENABLE_ENV}=1, the citation_network Parquet files \
             (--citation-data or {CITATION_DATA_ENV}), and GPU serialization via \
             `flock /tmp/cudf-gpu.lock`"
        )
        .into()),
    }
}

fn print_usage() {
    eprintln!(
        "embedded-backend-e2e\n\
         \n\
         USAGE:\n\
           {ENABLE_ENV}=1 flock /tmp/cudf-gpu.lock cargo run -p tools --features embedded-backend-e2e -- \\\n\
             embedded-backend-e2e [--citation-data DIR]\n\
         \n\
         OPTIONS:\n\
           --citation-data DIR   Directory holding edges_by_src.parquet, edges_by_dst.parquet,\n\
                                 profiles_by_paper_id.parquet, paper_authors.parquet, and paper_fos.parquet\n\
                                 (default: ${CITATION_DATA_ENV}, then {DEFAULT_CITATION_DATA})"
    );
}

async fn build_fixture_binary(repo_root: &Path) -> E2eResult<()> {
    println!("building embedded_backend fixture binary");
    let status = Command::new(cargo_bin())
        .current_dir(repo_root)
        .args(["build", "--workspace", "--all-features", "--bin", "embedded_backend"])
        .status()
        .await?;
    if !status.success() {
        return Err(format!("cargo build for embedded_backend failed with status {status}").into());
    }
    Ok(())
}

/// The backend requires one explicit device profile. A single active attempt makes one held
/// stream saturate admission, and the queue bound is the one the overload check fills.
fn server_config() -> String {
    format!(
        "[admission]\nmax_queued_attempts = {QUEUE_FILL_REQUESTS}\nmax_queue_wait_ms = {QUEUE_WAIT_MS}\n\n\
         [[admission.device_profiles]]\ndevice_ordinal = 0\nmax_active_attempts = 1\n"
    )
}

fn spawn_fixture(params: &RunParams) -> E2eResult<Child> {
    let config_path = params.server_config_path();
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config_path, server_config())?;

    println!("starting embedded_backend rest={} flight={}", params.rest_bind, params.flight_bind);
    let mut command = Command::new(params.fixture_binary());
    command.current_dir(&params.repo_root).stdout(Stdio::inherit()).stderr(Stdio::inherit());
    // Inherited server settings would change the topology under test.
    for (name, _) in env::vars() {
        if name.starts_with("ALGEON_SERVER_") {
            command.env_remove(name);
        }
    }
    command
        .env("EXAMPLE_REST_BIND", params.rest_bind.to_string())
        .env("ALGEON_SERVER_BIND", params.flight_bind.to_string())
        .env("ALGEON_SERVER_CONFIG_FILE", &config_path)
        .env("ALGEON_SERVER_CUGRAPH_ENABLED", "true");
    Ok(command.spawn()?)
}

async fn run_checks(params: &RunParams, child: &mut Child) -> E2eResult<()> {
    let http = Client::builder().pool_max_idle_per_host(128).build()?;
    let mut flight = select! {
        result = connect_flight_sql(&params.flight_url, Duration::from_secs(120)) => result?,
        status = child.wait() => {
            return Err(format!("embedded_backend exited before accepting Flight SQL connections: {}", status?).into());
        },
    };
    register_citation_tables(&mut flight, &params.citation_data).await?;
    wait_for_health(&http, params, child).await?;

    assert_catalog_truth(&mut flight).await?;
    let seed = query_single_int(&mut flight, "SELECT src FROM citation_edges WHERE src IS NOT NULL LIMIT 1").await?;
    let year =
        query_single_int(&mut flight, "SELECT CAST(year AS BIGINT) FROM papers WHERE year IS NOT NULL LIMIT 1").await?;
    let fos_name = query_single_string(
        &mut flight,
        "SELECT fos_name FROM paper_fos WHERE fos_name IS NOT NULL AND fos_name <> '' LIMIT 1",
    )
    .await?;

    assert_rest_pipelines(&http, params, seed, year, &fos_name).await?;
    assert_ad_hoc_graph(&mut flight, seed).await?;
    assert_shared_ledger(&http, params, year).await?;
    assert_overload_contract(&http, params, year).await?;
    assert_graceful_shutdown(child).await?;
    Ok(())
}

async fn register_citation_tables(client: &mut FlightClient, citation_data: &Path) -> E2eResult<()> {
    println!("registering citation-network tables from {}", citation_data.display());
    for (table, file) in CITATION_TABLES {
        let path = citation_data.join(file);
        let location = path.to_str().ok_or_else(|| format!("citation data path is not UTF-8: {}", path.display()))?;
        query_batches(
            client,
            &format!(
                "CREATE OR REPLACE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{}'",
                location.replace('\'', "''")
            ),
        )
        .await?;
    }
    Ok(())
}

async fn wait_for_health(http: &Client, params: &RunParams, child: &mut Child) -> E2eResult<()> {
    let url = format!("{}/healthz", params.rest_base);
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last_body: String;
    loop {
        if let Some(status) = child.try_wait()? {
            return Err(format!("embedded_backend exited before healthz was ready: {status}").into());
        }
        match http.get(&url).send().await {
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_body = body.clone();
                if status.is_success() {
                    let value: Value = serde_json::from_str(&body)?;
                    if value.get("status").and_then(Value::as_str) == Some("ok") {
                        println!("healthz ok");
                        return Ok(());
                    }
                }
            },
            Err(error) => last_body = error.to_string(),
        }
        if Instant::now() >= deadline {
            return Err(format!("embedded_backend did not become healthy at {url}; last response: {last_body}").into());
        }
        time::sleep(Duration::from_secs(1)).await;
    }
}

async fn assert_catalog_truth(client: &mut FlightClient) -> E2eResult<()> {
    println!("checking Flight SQL catalog truth");
    let papers = query_single_int(client, "SELECT count(*) FROM papers").await?;
    if papers != EXPECTED_PAPERS {
        return Err(format!("papers count mismatch: expected {EXPECTED_PAPERS}, got {papers}").into());
    }
    let edges = query_single_int(client, "SELECT count(*) FROM citation_edges").await?;
    if edges != EXPECTED_CITATION_EDGES {
        return Err(format!("citation_edges count mismatch: expected {EXPECTED_CITATION_EDGES}, got {edges}").into());
    }
    Ok(())
}

async fn assert_rest_pipelines(
    http: &Client,
    params: &RunParams,
    seed: i64,
    year: i64,
    fos_name: &str,
) -> E2eResult<()> {
    println!("checking REST pipelines and planning reports");
    let listed = get_json(http, &format!("{}/api/pipelines", params.rest_base)).await?;
    let names = listed
        .get("pipelines")
        .and_then(Value::as_array)
        .ok_or("pipeline listing did not contain pipelines array")?
        .iter()
        .filter_map(|pipeline| pipeline.get("name").and_then(Value::as_str))
        .collect::<Vec<_>>();
    for expected in ["sql_aggregate_topk", "sql_join_facets", "graph_pagerank_topk", "graph_bfs_hops"] {
        if !names.contains(&expected) {
            return Err(format!("pipeline listing missing {expected}: {listed}").into());
        }
        let plan = get_json(http, &format!("{}/api/pipelines/{expected}/plan", params.rest_base)).await?;
        require_selected_plan(expected, &plan)?;
    }

    let aggregate =
        post_pipeline(http, params, "sql_aggregate_topk", json!({"year_from": year, "year_to": year, "limit": 5}))
            .await?;
    require_rows_between("sql_aggregate_topk", &aggregate, 1, 5)?;

    let join = post_pipeline(http, params, "sql_join_facets", json!({"fos_name": fos_name})).await?;
    require_rows_at_least("sql_join_facets", &join, 1)?;

    let pagerank = post_pipeline(http, params, "graph_pagerank_topk", json!({"limit": 5})).await?;
    let rows = require_rows_between("graph_pagerank_topk", &pagerank, 1, 5)?;
    require_monotonic_desc(rows, "pagerank")?;

    let bfs = post_pipeline(http, params, "graph_bfs_hops", json!({"seed_paper_id": seed, "max_depth": 2})).await?;
    let rows = require_rows_at_least("graph_bfs_hops", &bfs, 1)?;
    require_nonnegative_distances(rows)?;
    Ok(())
}

async fn assert_ad_hoc_graph(client: &mut FlightClient, seed: i64) -> E2eResult<()> {
    println!("checking Flight SQL ad-hoc graph query");
    let suffix = unique_suffix();
    let view = format!("embedded_e2e_edges_{suffix}");
    query_batches(
        client,
        &format!(
            "CREATE VIEW {view} AS \
             SELECT src, dst, weight FROM citation_edges \
             WHERE src = {seed} OR dst = {seed}"
        ),
    )
    .await?;
    let count = query_single_int(
        client,
        &format!(
            "SELECT COUNT(*) \
             FROM cugraph_bfs(\
                 edges => (SELECT src, dst FROM {view}), source_vertex => {seed}, \
                 depth_limit => 1, output_mode => 'vertices') \
             WHERE reachable"
        ),
    )
    .await?;
    if count < 1 {
        return Err(format!("ad-hoc BFS over {view} returned no reachable vertices").into());
    }
    Ok(())
}

async fn assert_shared_ledger(http: &Client, params: &RunParams, year: i64) -> E2eResult<()> {
    println!("checking shared REST/Flight admission ledger");
    let held = open_stream(&params.flight_url, HOLD_SQL).await?;
    wait_for_admission(http, params, Duration::from_secs(60), "active_queries >= 1", |active, _| active >= 1).await?;
    let rest_task = {
        let http = http.clone();
        let params = params.clone();
        tokio::spawn(async move {
            post_pipeline(&http, &params, "sql_aggregate_topk", json!({"year_from": year, "year_to": year, "limit": 1}))
                .await
        })
    };
    wait_for_admission(
        http,
        params,
        Duration::from_secs(20),
        "active_queries >= 1 and queued_queries >= 1",
        |active, queued| active >= 1 && queued >= 1,
    )
    .await?;
    // Ending the held stream frees the only active slot for the queued REST request.
    drop(held);
    let rest_result = time::timeout(Duration::from_secs(120), rest_task).await??;
    let rest_json = rest_result?;
    require_rows_between("shared-ledger rest pipeline", &rest_json, 1, 1)?;
    Ok(())
}

async fn assert_overload_contract(http: &Client, params: &RunParams, year: i64) -> E2eResult<()> {
    println!("checking overload contract");
    let held = open_stream(&params.flight_url, HOLD_SQL).await?;
    wait_for_admission(http, params, Duration::from_secs(60), "active_queries >= 1", |active, _| active >= 1).await?;

    let mut queued = Vec::with_capacity(QUEUE_FILL_REQUESTS);
    for _ in 0..QUEUE_FILL_REQUESTS {
        let flight_url = params.flight_url.clone();
        queued.push(tokio::spawn(async move {
            let mut stream = open_stream(&flight_url, HOLD_SQL).await?;
            // The first frame arrives only after admission; until then this task waits in the queue.
            stream.try_next().await?;
            Ok::<(), Box<dyn Error + Send + Sync>>(())
        }));
    }
    let queue_full = format!("queued_queries >= {QUEUE_FILL_REQUESTS}");
    wait_for_admission(http, params, Duration::from_secs(60), &queue_full, |_, queued| {
        queued >= QUEUE_FILL_REQUESTS as u64
    })
    .await?;

    let overflow_body = json!({"year_from": year, "year_to": year, "limit": 1});
    let response =
        http.post(format!("{}/api/pipelines/sql_aggregate_topk", params.rest_base)).json(&overflow_body).send().await?;
    if response.status() != StatusCode::TOO_MANY_REQUESTS {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        abort_queued(queued).await;
        return Err(format!("overflow REST request should return 429, got {status}: {body}").into());
    }
    if response.headers().contains_key(header::RETRY_AFTER) {
        abort_queued(queued).await;
        return Err("overflow REST request fabricated Retry-After without sealed attempt advice".into());
    }

    let mut overflow = open_stream(&params.flight_url, HOLD_SQL).await?;
    match overflow.try_next().await {
        Err(FlightError::Tonic(status)) if status.code() == Code::ResourceExhausted => {},
        Err(error) => {
            abort_queued(queued).await;
            return Err(format!("overflow Flight request should be resource_exhausted, got {error:?}").into());
        },
        Ok(_) => {
            abort_queued(queued).await;
            return Err("overflow Flight request unexpectedly succeeded".into());
        },
    }

    abort_queued(queued).await;
    wait_for_admission(http, params, Duration::from_secs(30), "queued_queries == 0", |_, queued| queued == 0).await?;
    // Ending the held stream frees the only active slot before the recovery requests.
    drop(held);

    let mut recovery_client = connect_flight_sql(&params.flight_url, Duration::from_secs(30)).await?;
    let recovered = query_single_int(&mut recovery_client, "SELECT 3 AS recovered").await?;
    if recovered != 3 {
        return Err(format!("post-overload Flight recovery returned {recovered}, expected 3").into());
    }
    let recovery =
        post_pipeline(http, params, "sql_aggregate_topk", json!({"year_from": year, "year_to": year, "limit": 1}))
            .await?;
    require_rows_between("post-overload REST recovery", &recovery, 1, 1)?;
    Ok(())
}

async fn assert_graceful_shutdown(child: &mut Child) -> E2eResult<()> {
    println!("checking SIGTERM graceful shutdown");
    let pid = child.id().ok_or("embedded_backend child has no pid")?;
    let status = Command::new("kill").arg("-TERM").arg(pid.to_string()).status().await?;
    if !status.success() {
        return Err(format!("failed to send SIGTERM to embedded_backend pid {pid}: {status}").into());
    }
    let status = time::timeout(Duration::from_secs(60), child.wait()).await??;
    if !status.success() {
        return Err(format!("embedded_backend exited non-zero after SIGTERM: {status}").into());
    }
    Ok(())
}

async fn terminate_child_best_effort(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    if let Some(pid) = child.id() {
        let _ = Command::new("kill").arg("-TERM").arg(pid.to_string()).status().await;
        if let Ok(Ok(_)) = time::timeout(Duration::from_secs(10), child.wait()).await {
            return;
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}

async fn abort_queued(handles: Vec<JoinHandle<E2eResult<()>>>) {
    for handle in &handles {
        handle.abort();
    }
    for handle in handles {
        let _ = handle.await;
    }
}

/// Polls `/api/gpu` until `satisfied(active_queries, queued_queries)` holds.
async fn wait_for_admission(
    http: &Client,
    params: &RunParams,
    timeout: Duration,
    expectation: &str,
    satisfied: impl Fn(u64, u64) -> bool,
) -> E2eResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let gpu = get_json(http, &format!("{}/api/gpu", params.rest_base)).await?;
        let active = gpu["admission"]["active_queries"].as_u64().unwrap_or_default();
        let queued = gpu["admission"]["queued_queries"].as_u64().unwrap_or_default();
        if satisfied(active, queued) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for {expectation}; last gpu payload: {gpu}").into());
        }
        time::sleep(Duration::from_millis(100)).await;
    }
}

async fn get_json(http: &Client, url: &str) -> E2eResult<Value> {
    let response = http.get(url).send().await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(format!("GET {url} failed with {status}: {body}").into());
    }
    Ok(serde_json::from_str(&body)?)
}

async fn post_pipeline(http: &Client, params: &RunParams, name: &str, body: Value) -> E2eResult<Value> {
    let url = format!("{}/api/pipelines/{name}", params.rest_base);
    let response = http.post(&url).json(&body).send().await?;
    let status = response.status();
    let body_text = response.text().await?;
    if !status.is_success() {
        return Err(format!("POST {url} failed with {status}: {body_text}").into());
    }
    Ok(serde_json::from_str(&body_text)?)
}

fn require_selected_plan(name: &str, plan: &Value) -> E2eResult<()> {
    let report = plan.get("planning_report").ok_or("plan response missing planning_report")?;
    let selected_count = report.get("selected_count").and_then(Value::as_u64).unwrap_or_default();
    let has_selected = report.get("outcomes").and_then(Value::as_array).is_some_and(|outcomes| {
        outcomes.iter().any(|outcome| outcome.get("outcome").and_then(Value::as_str) == Some("selected"))
    });
    if selected_count == 0 || !has_selected {
        return Err(format!("pipeline {name} did not report a selected native candidate: {plan}").into());
    }
    Ok(())
}

fn require_rows_between<'a>(label: &str, value: &'a Value, min: usize, max: usize) -> E2eResult<&'a Vec<Value>> {
    let rows = value.get("rows").and_then(Value::as_array).ok_or_else(|| format!("{label} missing rows array"))?;
    if rows.len() < min || rows.len() > max {
        return Err(format!("{label} returned {} rows, expected between {min} and {max}: {value}", rows.len()).into());
    }
    Ok(rows)
}

fn require_rows_at_least<'a>(label: &str, value: &'a Value, min: usize) -> E2eResult<&'a Vec<Value>> {
    let rows = value.get("rows").and_then(Value::as_array).ok_or_else(|| format!("{label} missing rows array"))?;
    if rows.len() < min {
        return Err(format!("{label} returned {} rows, expected at least {min}: {value}", rows.len()).into());
    }
    Ok(rows)
}

fn require_monotonic_desc(rows: &[Value], field: &str) -> E2eResult<()> {
    let mut previous = f64::INFINITY;
    for row in rows {
        let value =
            row.get(field).and_then(Value::as_f64).ok_or_else(|| format!("row missing numeric {field}: {row}"))?;
        if value > previous {
            return Err(format!("{field} values are not descending: previous={previous}, current={value}").into());
        }
        previous = value;
    }
    Ok(())
}

fn require_nonnegative_distances(rows: &[Value]) -> E2eResult<()> {
    for row in rows {
        let distance =
            row.get("distance").and_then(Value::as_i64).ok_or_else(|| format!("BFS row missing distance: {row}"))?;
        if distance < 0 {
            return Err(format!("BFS returned negative distance: {row}").into());
        }
    }
    Ok(())
}

async fn connect_flight_sql(url: &str, timeout: Duration) -> E2eResult<FlightClient> {
    const BACKOFF: Duration = Duration::from_millis(100);
    let attempts = usize::try_from(timeout.as_millis() / BACKOFF.as_millis()).unwrap_or(usize::MAX).max(1);
    connect_with_retry(
        url,
        ConnectionRetryConfig::fixed_attempts(attempts, BACKOFF),
        FlightSqlServiceClient::new,
        |error| format!("Flight SQL server did not accept connections at {url}: {error:?}"),
    )
    .await
}

/// Issues the statement and opens its result stream without reading a frame.
async fn open_stream(url: &str, sql: &str) -> E2eResult<FlightRecordBatchStream> {
    let mut client = connect_flight_sql(url, Duration::from_secs(30)).await?;
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket =
        flight_info.endpoint.into_iter().find_map(|endpoint| endpoint.ticket).ok_or("FlightInfo has no ticket")?;
    Ok(client.do_get(ticket).await?)
}

async fn query_batches(client: &mut FlightClient, sql: &str) -> E2eResult<Vec<RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    consume_flight_info(client, flight_info).await
}

async fn consume_flight_info(client: &mut FlightClient, flight_info: FlightInfo) -> E2eResult<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for endpoint in flight_info.endpoint {
        let Some(ticket) = endpoint.ticket else {
            continue;
        };
        batches.extend(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?);
    }
    Ok(batches)
}

async fn query_single_int(client: &mut FlightClient, sql: &str) -> E2eResult<i64> {
    let batches = query_batches(client, sql).await?;
    let batch = single_row_batch(&batches, sql)?;
    int_value(batch.column(0), 0).ok_or_else(|| format!("could not read integer result for {sql:?}").into())
}

async fn query_single_string(client: &mut FlightClient, sql: &str) -> E2eResult<String> {
    let batches = query_batches(client, sql).await?;
    let batch = single_row_batch(&batches, sql)?;
    string_value(batch.column(0), 0).ok_or_else(|| format!("could not read string result for {sql:?}").into())
}

fn single_row_batch<'a>(batches: &'a [RecordBatch], sql: &str) -> E2eResult<&'a RecordBatch> {
    if batches.len() != 1 || batches[0].num_rows() != 1 {
        return Err(format!(
            "expected one single-row batch for {sql:?}, got {} batches and first rows {:?}",
            batches.len(),
            batches.first().map(RecordBatch::num_rows)
        )
        .into());
    }
    Ok(&batches[0])
}

fn int_value(array: &ArrayRef, row: usize) -> Option<i64> {
    if array.is_null(row) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|a| a.value(row))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|a| i64::from(a.value(row))))
        .or_else(|| array.as_any().downcast_ref::<Int16Array>().map(|a| i64::from(a.value(row))))
        .or_else(|| array.as_any().downcast_ref::<UInt64Array>().map(|a| a.value(row) as i64))
        .or_else(|| array.as_any().downcast_ref::<UInt32Array>().map(|a| i64::from(a.value(row))))
}

fn string_value(array: &ArrayRef, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|a| a.value(row).to_string())
        .or_else(|| array.as_any().downcast_ref::<LargeStringArray>().map(|a| a.value(row).to_string()))
        .or_else(|| array.as_any().downcast_ref::<StringViewArray>().map(|a| a.value(row).to_string()))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().and_then(Path::parent).unwrap().to_path_buf()
}

fn cargo_bin() -> String {
    env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn free_loopback_addr() -> E2eResult<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{}_{}", std::process::id(), nanos)
}
