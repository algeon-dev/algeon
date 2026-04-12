//! Black-box E2E for the embedded dual-frontend backend example.
//!
//! This runner intentionally lives outside Cargo test targets. It starts the
//! example binary as a child process, talks to REST with HTTP and to Flight SQL
//! with gRPC, and keeps all GPU use behind the caller's `flock`.

use std::{
    error::Error,
    net::{SocketAddr, TcpListener},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array},
    record_batch::RecordBatch,
};
use arrow_flight::{FlightInfo, error::FlightError, sql::client::FlightSqlServiceClient};
use futures::TryStreamExt;
use reqwest::{StatusCode, header};
use serde_json::{Value, json};
use tokio::{process::Child, time::Instant};
use tonic::{Code, transport::Endpoint};
use url::Url;

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const ENABLE_ENV: &str = "NEXUS_EMBEDDED_BACKEND_E2E";
const DEFAULT_REST_URI: &str = "http://localhost:8181";
const DEFAULT_WAREHOUSE: &str = "s3://lakehouse/warehouse";
const DEFAULT_S3_ENDPOINT: &str = "http://localhost:9000";
const DEFAULT_S3_REGION: &str = "us-east-1";
const DEFAULT_S3_ACCESS_KEY: &str = "cudfadmin";
const DEFAULT_S3_SECRET_KEY: &str = "cudfadminsecret";
const EXPECTED_PAPERS: i64 = 4_894_081;
const EXPECTED_CITATION_EDGES: i64 = 45_564_149;
const QUEUE_FILL_REQUESTS: usize = 64;

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    let params = RunParams::from_args(args)?;
    build_example_binary(&params.repo_root).await?;

    let mut child = spawn_example(&params)?;
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
    rest_catalog_uri: String,
    warehouse: String,
    s3_endpoint: String,
    s3_region: String,
    s3_access_key: String,
    s3_secret_key: String,
}

impl RunParams {
    fn from_args(args: Vec<String>) -> E2eResult<Self> {
        require_enabled()?;
        if let Some(arg) = args.first() {
            match arg.as_str() {
                "--gpu-parity" => {
                    return Err(
                        "embedded-backend-e2e --gpu-parity is not implemented yet; omit the flag for the supported black-box suite"
                            .into(),
                    );
                },
                "-h" | "--help" => {
                    print_usage();
                    return Err("embedded-backend-e2e help requested".into());
                },
                other => return Err(format!("unknown embedded-backend-e2e argument: {other}").into()),
            }
        }

        let repo_root = repo_root();
        let rest_bind = free_loopback_addr()?;
        let flight_bind = free_loopback_addr()?;
        let rest_catalog_uri = env_or_default("NEXUS_ICEBERG_REST_URI", DEFAULT_REST_URI);
        let warehouse = env_or_default("NEXUS_ICEBERG_WAREHOUSE", DEFAULT_WAREHOUSE);
        let s3_endpoint = env_or_default("NEXUS_ICEBERG_S3_ENDPOINT", DEFAULT_S3_ENDPOINT);
        let s3_region = env_or_default("NEXUS_ICEBERG_S3_REGION", DEFAULT_S3_REGION);
        let s3_access_key = env_or_default("NEXUS_ICEBERG_S3_ACCESS_KEY_ID", DEFAULT_S3_ACCESS_KEY);
        let s3_secret_key = env_or_default("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY", DEFAULT_S3_SECRET_KEY);

        require_loopback_http_url("NEXUS_ICEBERG_REST_URI", &rest_catalog_uri)?;
        require_loopback_http_url("NEXUS_ICEBERG_S3_ENDPOINT", &s3_endpoint)?;
        require_s3_warehouse("NEXUS_ICEBERG_WAREHOUSE", &warehouse)?;

        Ok(Self {
            repo_root,
            rest_bind,
            flight_bind,
            rest_base: format!("http://{rest_bind}"),
            flight_url: format!("http://{flight_bind}"),
            rest_catalog_uri,
            warehouse,
            s3_endpoint,
            s3_region,
            s3_access_key,
            s3_secret_key,
        })
    }

    fn example_binary(&self) -> PathBuf {
        target_debug_binary(&self.repo_root, "embedded_backend")
    }
}

fn require_enabled() -> E2eResult<()> {
    match std::env::var(ENABLE_ENV).as_deref() {
        Ok("1") => Ok(()),
        _ => Err(format!(
            "embedded backend E2E requires {ENABLE_ENV}=1, the local Iceberg REST/RustFS stack, \
             citation_network loaded with --load-mode add-files, and GPU serialization via \
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
           crates/datafusion-nexus-examples/scripts/e2e.sh [--skip-stack] [--no-load] [--reload]"
    );
}

async fn build_example_binary(repo_root: &Path) -> E2eResult<()> {
    println!("building embedded_backend example binary");
    let status = tokio::process::Command::new(cargo_bin())
        .current_dir(repo_root)
        .arg("build")
        .arg("-p")
        .arg("datafusion-nexus-examples")
        .arg("--bin")
        .arg("embedded_backend")
        .status()
        .await?;
    if !status.success() {
        return Err(format!("cargo build for embedded_backend failed with status {status}").into());
    }
    Ok(())
}

fn spawn_example(params: &RunParams) -> E2eResult<Child> {
    println!("starting embedded_backend rest={} flight={}", params.rest_bind, params.flight_bind);
    let mut command = tokio::process::Command::new(params.example_binary());
    command
        .current_dir(&params.repo_root)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("EXAMPLE_REST_BIND", params.rest_bind.to_string())
        .env("NEXUS_SERVER_BIND", params.flight_bind.to_string())
        .env("NEXUS_SERVER_MAX_CONCURRENT_QUERIES", "1")
        .env_remove("NEXUS_SERVER_MAX_IN_FLIGHT")
        .env_remove("NEXUS_SERVER_CONFIG_FILE")
        .env("NEXUS_SERVER_SESSION_PROFILE", "native")
        .env("NEXUS_SERVER_CUGRAPH_ENABLED", "true")
        .env("NEXUS_ICEBERG_CATALOG_KIND", "rest")
        .env("NEXUS_ICEBERG_CATALOG_NAME", "lake")
        .env("NEXUS_ICEBERG_NAMESPACE", "citation_network")
        .env("NEXUS_ICEBERG_WAREHOUSE", &params.warehouse)
        .env("NEXUS_ICEBERG_REST_URI", &params.rest_catalog_uri)
        .env("NEXUS_ICEBERG_S3_ENDPOINT", &params.s3_endpoint)
        .env("NEXUS_ICEBERG_S3_REGION", &params.s3_region)
        .env("NEXUS_ICEBERG_S3_PATH_STYLE", "true")
        .env("NEXUS_ICEBERG_S3_ACCESS_KEY_ID", &params.s3_access_key)
        .env("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY", &params.s3_secret_key)
        .env_remove("NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE")
        .env("NEXUS_SERVER_WORKSPACE_BACKING_CATALOG", "lake")
        .env("NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA", "citation_network")
        .env("NEXUS_SERVER_WORKSPACE_BACKING_ALIASES", "papers,paper_fos,citation_edges,citation_edges_by_dst");
    Ok(command.spawn()?)
}

async fn run_checks(params: &RunParams, child: &mut Child) -> E2eResult<()> {
    let http = reqwest::Client::builder().pool_max_idle_per_host(128).build()?;
    wait_for_health(&http, params, child).await?;

    let mut flight = connect_with_retry(&params.flight_url, Duration::from_secs(60)).await?;
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

async fn wait_for_health(http: &reqwest::Client, params: &RunParams, child: &mut Child) -> E2eResult<()> {
    let url = format!("{}/healthz", params.rest_base);
    let deadline = Instant::now() + Duration::from_secs(300);
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
            return Err(format!(
                "embedded_backend did not become healthy at {url}; last response: {last_body}\n\
                 Ensure `crates/datafusion-nexus-examples/scripts/e2e.sh` loaded citation_network with --load-mode add-files."
            )
            .into());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn assert_catalog_truth(client: &mut FlightSqlServiceClient<tonic::transport::Channel>) -> E2eResult<()> {
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
    http: &reqwest::Client,
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
        require_rewritten_plan(expected, &plan)?;
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

async fn assert_ad_hoc_graph(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    seed: i64,
) -> E2eResult<()> {
    println!("checking Flight SQL ad-hoc graph query");
    let suffix = unique_suffix();
    let view = format!("embedded_e2e_edges_{suffix}");
    execute_batches(
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
             FROM cugraph_bfs('{view}', {seed}, 'src', 'dst', NULL, '{{\"depth_limit\":1,\"output_mode\":\"normalized\"}}') \
             WHERE reachable"
        ),
    )
    .await?;
    if count < 1 {
        return Err(format!("ad-hoc BFS over {view} returned no reachable vertices").into());
    }
    Ok(())
}

async fn assert_shared_ledger(http: &reqwest::Client, params: &RunParams, year: i64) -> E2eResult<()> {
    println!("checking shared REST/Flight admission ledger");
    let mut flight = connect_with_retry(&params.flight_url, Duration::from_secs(30)).await?;
    let held = flight.execute("SELECT 1 AS held".to_string(), None).await?;
    let rest_task = {
        let http = http.clone();
        let params = params.clone();
        tokio::spawn(async move {
            post_pipeline(&http, &params, "sql_aggregate_topk", json!({"year_from": year, "year_to": year, "limit": 1}))
                .await
        })
    };
    wait_for_queue_depth(http, params, 1, Duration::from_secs(20)).await?;
    let gpu = get_json(http, &format!("{}/api/gpu", params.rest_base)).await?;
    let active = gpu["admission"]["active_queries"].as_u64().unwrap_or_default();
    let queued = gpu["admission"]["queued_queries"].as_u64().unwrap_or_default();
    if active < 1 || queued < 1 {
        return Err(format!("shared ledger did not show active Flight plus queued REST work: {gpu}").into());
    }
    consume_flight_info(&mut flight, held).await?;
    let rest_result = tokio::time::timeout(Duration::from_secs(120), rest_task).await??;
    let rest_json = rest_result?;
    require_rows_between("shared-ledger rest pipeline", &rest_json, 1, 1)?;
    Ok(())
}

async fn assert_overload_contract(http: &reqwest::Client, params: &RunParams, year: i64) -> E2eResult<()> {
    println!("checking overload contract");
    let mut active_client = connect_with_retry(&params.flight_url, Duration::from_secs(30)).await?;
    let held = active_client.execute("SELECT 1 AS held_for_overload".to_string(), None).await?;

    let mut queued = Vec::with_capacity(QUEUE_FILL_REQUESTS);
    for _ in 0..QUEUE_FILL_REQUESTS {
        let flight_url = params.flight_url.clone();
        queued.push(tokio::spawn(async move {
            let mut client = connect_with_retry(&flight_url, Duration::from_secs(30)).await?;
            client.execute("SELECT 1 AS queued_for_overload".to_string(), None).await?;
            Ok::<(), Box<dyn Error + Send + Sync>>(())
        }));
    }
    wait_for_queue_depth(http, params, QUEUE_FILL_REQUESTS as u64, Duration::from_secs(30)).await?;

    let overflow_body = json!({"year_from": year, "year_to": year, "limit": 1});
    let response =
        http.post(format!("{}/api/pipelines/sql_aggregate_topk", params.rest_base)).json(&overflow_body).send().await?;
    if response.status() != StatusCode::TOO_MANY_REQUESTS {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        abort_queued(queued).await;
        consume_flight_info(&mut active_client, held).await?;
        return Err(format!("overflow REST request should return 429, got {status}: {body}").into());
    }
    if !response.headers().contains_key(header::RETRY_AFTER) {
        abort_queued(queued).await;
        consume_flight_info(&mut active_client, held).await?;
        return Err("overflow REST request returned 429 without Retry-After".into());
    }

    let mut overflow_client = connect_with_retry(&params.flight_url, Duration::from_secs(30)).await?;
    match overflow_client.execute("SELECT 2 AS overflow".to_string(), None).await {
        Err(FlightError::Tonic(status)) if status.code() == Code::ResourceExhausted => {},
        Err(error) => {
            abort_queued(queued).await;
            consume_flight_info(&mut active_client, held).await?;
            return Err(format!("overflow Flight request should be resource_exhausted, got {error:?}").into());
        },
        Ok(_) => {
            abort_queued(queued).await;
            consume_flight_info(&mut active_client, held).await?;
            return Err("overflow Flight request unexpectedly succeeded".into());
        },
    }

    abort_queued(queued).await;
    wait_for_exact_queue_depth(http, params, 0, Duration::from_secs(30)).await?;
    consume_flight_info(&mut active_client, held).await?;

    let mut recovery_client = connect_with_retry(&params.flight_url, Duration::from_secs(30)).await?;
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
    let status = tokio::process::Command::new("kill").arg("-TERM").arg(pid.to_string()).status().await?;
    if !status.success() {
        return Err(format!("failed to send SIGTERM to embedded_backend pid {pid}: {status}").into());
    }
    let status = tokio::time::timeout(Duration::from_secs(60), child.wait()).await??;
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
        let _ = tokio::process::Command::new("kill").arg("-TERM").arg(pid.to_string()).status().await;
        if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
            return;
        }
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}

async fn abort_queued(handles: Vec<tokio::task::JoinHandle<E2eResult<()>>>) {
    for handle in &handles {
        handle.abort();
    }
    for handle in handles {
        let _ = handle.await;
    }
}

async fn wait_for_queue_depth(
    http: &reqwest::Client,
    params: &RunParams,
    min_queued: u64,
    timeout: Duration,
) -> E2eResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let gpu = get_json(http, &format!("{}/api/gpu", params.rest_base)).await?;
        let queued = gpu["admission"]["queued_queries"].as_u64().unwrap_or_default();
        if queued >= min_queued {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for queued_queries >= {min_queued}; last gpu payload: {gpu}").into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn wait_for_exact_queue_depth(
    http: &reqwest::Client,
    params: &RunParams,
    expected: u64,
    timeout: Duration,
) -> E2eResult<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let gpu = get_json(http, &format!("{}/api/gpu", params.rest_base)).await?;
        let queued = gpu["admission"]["queued_queries"].as_u64().unwrap_or_default();
        if queued == expected {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for queued_queries == {expected}; last gpu payload: {gpu}").into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn get_json(http: &reqwest::Client, url: &str) -> E2eResult<Value> {
    let response = http.get(url).send().await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(format!("GET {url} failed with {status}: {body}").into());
    }
    Ok(serde_json::from_str(&body)?)
}

async fn post_pipeline(http: &reqwest::Client, params: &RunParams, name: &str, body: Value) -> E2eResult<Value> {
    let url = format!("{}/api/pipelines/{name}", params.rest_base);
    let response = http.post(&url).json(&body).send().await?;
    let status = response.status();
    let body_text = response.text().await?;
    if !status.is_success() {
        return Err(format!("POST {url} failed with {status}: {body_text}").into());
    }
    Ok(serde_json::from_str(&body_text)?)
}

fn require_rewritten_plan(name: &str, plan: &Value) -> E2eResult<()> {
    let report = plan.get("planning_report").ok_or("plan response missing planning_report")?;
    let rewritten_count = report.get("rewritten_count").and_then(Value::as_u64).unwrap_or_default();
    let has_rewritten = report.get("outcomes").and_then(Value::as_array).is_some_and(|outcomes| {
        outcomes.iter().any(|outcome| outcome.get("outcome").and_then(Value::as_str) == Some("rewritten"))
    });
    if rewritten_count == 0 || !has_rewritten {
        return Err(format!("pipeline {name} did not report a rewritten native candidate: {plan}").into());
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

async fn connect_with_retry(
    url: &str,
    timeout: Duration,
) -> E2eResult<FlightSqlServiceClient<tonic::transport::Channel>> {
    let deadline = Instant::now() + timeout;
    loop {
        match Endpoint::from_shared(url.to_string())?.connect().await {
            Ok(channel) => return Ok(FlightSqlServiceClient::new(channel)),
            Err(error) => {
                if Instant::now() >= deadline {
                    return Err(format!("Flight SQL server did not accept connections at {url}: {error:?}").into());
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            },
        }
    }
}

async fn query_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<Vec<RecordBatch>> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    consume_flight_info(client, flight_info).await
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<Vec<RecordBatch>> {
    query_batches(client, sql).await
}

async fn consume_flight_info(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: FlightInfo,
) -> E2eResult<Vec<RecordBatch>> {
    let mut batches = Vec::new();
    for endpoint in flight_info.endpoint {
        let Some(ticket) = endpoint.ticket else {
            continue;
        };
        batches.extend(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?);
    }
    Ok(batches)
}

async fn query_single_int(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, sql: &str) -> E2eResult<i64> {
    let batches = query_batches(client, sql).await?;
    let batch = single_row_batch(&batches, sql)?;
    int_value(batch.column(0), 0).ok_or_else(|| format!("could not read integer result for {sql:?}").into())
}

async fn query_single_string(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> E2eResult<String> {
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

fn int_value(array: &arrow::array::ArrayRef, row: usize) -> Option<i64> {
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

fn string_value(array: &arrow::array::ArrayRef, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|a| a.value(row).to_string())
        .or_else(|| array.as_any().downcast_ref::<LargeStringArray>().map(|a| a.value(row).to_string()))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().and_then(Path::parent).unwrap().to_path_buf()
}

fn target_debug_binary(repo_root: &Path, name: &str) -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|path| if path.is_absolute() { path } else { repo_root.join(path) })
        .unwrap_or_else(|| repo_root.join("target"));
    target_dir.join("debug").join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn free_loopback_addr() -> E2eResult<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn env_or_default(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

fn require_loopback_http_url(name: &str, value: &str) -> E2eResult<()> {
    if is_loopback_http_url(value) {
        return Ok(());
    }
    Err(format!("{name} must be a loopback http:// URL for embedded backend E2E, got {value:?}").into())
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
    Err(format!("{name} must be an s3://bucket URI for embedded backend E2E, got {value:?}").into())
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    format!("{}_{}", std::process::id(), nanos)
}
