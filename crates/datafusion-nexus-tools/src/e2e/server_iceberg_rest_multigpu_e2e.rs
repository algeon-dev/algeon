//! Production-topology Flight SQL acceptance against local Iceberg REST + RustFS.
//!
//! This is deliberately an explicit tool, not a Cargo test. It launches the
//! release `datafusion_nexus_server` as a child with one Tokio worker and four
//! GPUs, drives concurrent TPC-H statements through Flight SQL, and retains
//! machine-readable evidence under `target/production_acceptance`.
//!
//! Use `fixture/scripts/e2e.sh iceberg-local --server-rest-multigpu-only` so
//! the REST/RustFS stack, fixture load, release build, and GPU flock are all
//! handled consistently.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use arrow_flight::{
    FlightInfo,
    sql::{
        ActionCancelQueryRequest, ActionCancelQueryResult, Any, ProstMessageExt, TicketStatementQuery,
        client::FlightSqlServiceClient,
    },
};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_nexus::{
    ParityFloatTolerance, ParityStatus, compare_record_batches,
    iceberg::catalog::{
        IcebergCatalogBackend, IcebergProviderKind, RestIcebergCatalogBackend, RestIcebergCatalogConfig,
        build_scoped_catalog_provider_from_backend,
    },
};
use futures::TryStreamExt;
use prost::Message as _;
use serde_json::json;
use tokio::{
    process::{Child, Command},
    sync::{Barrier, mpsc},
    task::JoinSet,
};
use tonic::transport::{Channel, Endpoint};
use url::Url;

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const OPT_IN_ENV: &str = "NEXUS_ICEBERG_SERVER_REST_MULTIGPU_E2E";
const SERVER_BINARY_ENV: &str = "NEXUS_SERVER_BINARY";
const DEVICE_LIST_ENV: &str = "NEXUS_PRODUCTION_ACCEPTANCE_GPU_DEVICES";
const ARTIFACT_ROOT_ENV: &str = "NEXUS_PRODUCTION_ACCEPTANCE_ARTIFACT_ROOT";
const EXPECTED_DEVICE_COUNT: usize = 4;
const MAX_CONCURRENT_QUERIES: usize = 16;
const MAX_QUERIES_PER_DEVICE: usize = 4;
const SATURATION_FLOOR: u64 = 8;
const REQUIRED_SATURATED_PROBES: usize = 20;
const Q06_WAVE_SIZE: usize = 64;
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const QUERY_TIMEOUT: Duration = Duration::from_secs(180);
const CHURN_TIMEOUT: Duration = Duration::from_secs(300);
const CANCEL_RESULT_CANCELLING: i32 = 2;

pub async fn run() -> E2eResult<()> {
    require_opt_in()?;
    let params = Params::from_env()?;
    fs::create_dir_all(&params.artifact_dir)?;
    write_run_metadata(&params)?;

    eprintln!("building CPU references through Iceberg REST/RustFS");
    let workloads = tpch_workloads(&params);
    let cpu_references = build_cpu_references(&params, &workloads).await?;

    let mut server = ServerChild::spawn(&params)?;
    let channel = match connect_with_retry(&params.server_url, server.child_mut()).await {
        Ok(channel) => channel,
        Err(error) => {
            let _ = fs::write(params.artifact_dir.join("FAILURE.txt"), format!("{error:#}\n"));
            return Err(error);
        },
    };

    let acceptance = run_acceptance(&params, channel, &workloads, &cpu_references).await;
    if let Err(error) = &acceptance {
        let _ = fs::write(params.artifact_dir.join("FAILURE.txt"), format!("{error:#}\n"));
    }
    let shutdown = server.shutdown().await;
    acceptance?;
    shutdown?;

    println!("server_iceberg_rest_multigpu_e2e ok: {}", params.artifact_dir.display());
    Ok(())
}

#[derive(Clone)]
struct Params {
    server_binary: PathBuf,
    server_addr: std::net::SocketAddr,
    server_url: String,
    rest_uri: String,
    warehouse: String,
    s3_endpoint: String,
    s3_region: String,
    s3_access_key: String,
    s3_secret_key: String,
    namespace: String,
    devices: Vec<usize>,
    artifact_dir: PathBuf,
}

impl Params {
    fn from_env() -> E2eResult<Self> {
        let server_binary = std::env::var_os(SERVER_BINARY_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/release/datafusion_nexus_server"));
        if !server_binary.is_file() {
            return Err(format!(
                "{SERVER_BINARY_ENV} must name the built release server binary; missing {}",
                server_binary.display()
            )
            .into());
        }

        let rest_uri = env_or("NEXUS_ICEBERG_REST_URI", "http://localhost:8181");
        let warehouse = env_or("NEXUS_ICEBERG_WAREHOUSE", "s3://lakehouse/warehouse");
        let s3_endpoint = required_env("NEXUS_ICEBERG_S3_ENDPOINT")?;
        let s3_region = env_or("NEXUS_ICEBERG_S3_REGION", "us-east-1");
        let s3_access_key = required_env("NEXUS_ICEBERG_S3_ACCESS_KEY_ID")?;
        let s3_secret_key = required_env("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY")?;
        let namespace = required_env("NEXUS_ICEBERG_NAMESPACE")?;
        require_loopback_http_url("NEXUS_ICEBERG_REST_URI", &rest_uri)?;
        require_loopback_http_url("NEXUS_ICEBERG_S3_ENDPOINT", &s3_endpoint)?;
        if !warehouse.starts_with("s3://") {
            return Err(format!("NEXUS_ICEBERG_WAREHOUSE must be an s3:// URI, got {warehouse:?}").into());
        }
        for name in ["NEXUS_ICEBERG_S3_SESSION_TOKEN", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"] {
            reject_non_empty_env(name)?;
        }

        let devices = parse_devices(&env_or(DEVICE_LIST_ENV, "0,1,2,3"))?;
        if devices.len() != EXPECTED_DEVICE_COUNT {
            return Err(format!(
                "production acceptance requires exactly {EXPECTED_DEVICE_COUNT} GPU ordinals, got {devices:?}"
            )
            .into());
        }

        let server_addr = reserve_loopback_addr()?;
        let server_url = format!("http://{server_addr}");
        let artifact_root = std::env::var_os(ARTIFACT_ROOT_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/production_acceptance/server_iceberg_rest_multigpu"));
        let run_id = format!("{}-{}", unix_seconds(), std::process::id());
        let artifact_dir = artifact_root.join(run_id);

        Ok(Self {
            server_binary,
            server_addr,
            server_url,
            rest_uri,
            warehouse,
            s3_endpoint,
            s3_region,
            s3_access_key,
            s3_secret_key,
            namespace,
            devices,
            artifact_dir,
        })
    }

    fn rest_config(&self) -> RestIcebergCatalogConfig {
        RestIcebergCatalogConfig::new("rest", self.rest_uri.clone(), self.warehouse.clone())
            .with_s3_endpoint(self.s3_endpoint.clone())
            .with_s3_region(self.s3_region.clone())
            .with_s3_path_style_access(true)
    }

    fn table(&self, name: &str) -> String {
        format!("lake.\"{}\".\"{name}\"", self.namespace)
    }

    fn device_csv(&self) -> String {
        self.devices.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
    }
}

fn require_opt_in() -> E2eResult<()> {
    if std::env::var(OPT_IN_ENV).as_deref() == Ok("1") {
        return Ok(());
    }
    Err(format!(
        "production multi-GPU REST acceptance requires {OPT_IN_ENV}=1, local Iceberg REST/RustFS, four GPUs, and flock /tmp/cudf-gpu.lock"
    )
    .into())
}

fn required_env(name: &'static str) -> E2eResult<String> {
    std::env::var(name).map_err(|_| format!("production REST acceptance requires {name}").into())
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn reject_non_empty_env(name: &str) -> E2eResult<()> {
    if std::env::var(name).is_ok_and(|value| !value.is_empty()) {
        return Err(format!("production REST acceptance refuses ambient session token {name}").into());
    }
    Ok(())
}

fn require_loopback_http_url(name: &str, value: &str) -> E2eResult<()> {
    let parsed = Url::parse(value)?;
    let loopback = parsed.scheme() == "http"
        && parsed.host_str().is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host.parse::<std::net::IpAddr>().is_ok_and(|address| address.is_loopback())
        });
    if loopback { Ok(()) } else { Err(format!("{name} must be a loopback http:// URL, got {value:?}").into()) }
}

fn parse_devices(raw: &str) -> E2eResult<Vec<usize>> {
    let devices = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;
    let unique = devices.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != devices.len() {
        return Err(format!("{DEVICE_LIST_ENV} contains duplicate ordinals: {raw:?}").into());
    }
    Ok(devices)
}

fn reserve_loopback_addr() -> E2eResult<std::net::SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn unix_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_secs())
}

fn write_run_metadata(params: &Params) -> E2eResult<()> {
    let git_sha = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let git_dirty = command_output("git", &["status", "--porcelain"]).is_some_and(|value| !value.is_empty());
    let server_sha256 = sha256_file(&params.server_binary).unwrap_or_else(|| "unknown".to_string());
    let driver_sha256 =
        std::env::current_exe().ok().as_deref().and_then(sha256_file).unwrap_or_else(|| "unknown".to_string());
    let metadata = format!(
        "git_sha={git_sha}\ngit_dirty={git_dirty}\nserver_binary={}\nserver_features=server,iceberg,nvml\n\
         server_binary_sha256={server_sha256}\ndriver_binary_sha256={driver_sha256}\n\
         tokio_worker_threads=1\ndevice_ordinals={}\nmax_concurrent_queries={MAX_CONCURRENT_QUERIES}\n\
         max_queries_per_device={MAX_QUERIES_PER_DEVICE}\nrest_uri={}\nwarehouse={}\nnamespace={}\n",
        params.server_binary.display(),
        params.device_csv(),
        params.rest_uri,
        params.warehouse,
        params.namespace,
    );
    fs::write(params.artifact_dir.join("metadata.txt"), metadata)?;
    fs::write(params.artifact_dir.join("harness-source.rs"), include_str!("server_iceberg_rest_multigpu_e2e.rs"))?;
    if let Some(diff) = command_output(
        "git",
        &[
            "diff",
            "--binary",
            "--",
            "Cargo.lock",
            "crates/datafusion-nexus-tools/Cargo.toml",
            "crates/datafusion-nexus-tools/src/bin/datafusion_nexus_tools.rs",
            "fixture/scripts/e2e.sh",
            "docs/issue/2026-07-12-hardware-limit-roadmap.md",
        ],
    ) {
        fs::write(params.artifact_dir.join("source.diff"), diff)?;
    }
    Ok(())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn sha256_file(path: &Path) -> Option<String> {
    let path = path.to_str()?;
    command_output("sha256sum", &[path]).and_then(|line| line.split_whitespace().next().map(str::to_string))
}

struct ServerChild {
    child: Child,
    stopped: bool,
    artifact_dir: PathBuf,
}

impl ServerChild {
    fn spawn(params: &Params) -> E2eResult<Self> {
        let stdout = File::create(params.artifact_dir.join("server.stdout.log"))?;
        let stderr = File::create(params.artifact_dir.join("server.stderr.log"))?;
        let mut command = Command::new(&params.server_binary);
        command.stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr)).kill_on_drop(true);

        for (name, _) in std::env::vars() {
            if name.starts_with("NEXUS_SERVER_")
                || name.starts_with("NEXUS_ICEBERG_")
                || name.starts_with("NEXUS_LAKEHOUSE_")
                || name == "TOKIO_WORKER_THREADS"
            {
                command.env_remove(name);
            }
        }
        command.envs([
            ("TOKIO_WORKER_THREADS", "1".to_string()),
            ("NEXUS_SERVER_BIND", params.server_addr.to_string()),
            ("NEXUS_SERVER_LOG", "info".to_string()),
            ("NEXUS_SERVER_GPU_DEVICES", params.device_csv()),
            ("NEXUS_SERVER_MAX_CONCURRENT_QUERIES", MAX_CONCURRENT_QUERIES.to_string()),
            ("NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE", MAX_QUERIES_PER_DEVICE.to_string()),
            ("NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY", "least_loaded".to_string()),
            ("NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS", "120000".to_string()),
            ("NEXUS_SERVER_SESSION_PROFILE", "native".to_string()),
            ("NEXUS_SERVER_NATIVE_MEMORY_POLICY", "bounded".to_string()),
            ("NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES", (8_u64 * 1024 * 1024 * 1024).to_string()),
            ("NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS", "60".to_string()),
            ("NEXUS_ICEBERG_CATALOG_KIND", "rest".to_string()),
            ("NEXUS_ICEBERG_CATALOG_NAME", "lake".to_string()),
            ("NEXUS_ICEBERG_NAMESPACE", params.namespace.clone()),
            ("NEXUS_ICEBERG_REST_URI", params.rest_uri.clone()),
            ("NEXUS_ICEBERG_WAREHOUSE", params.warehouse.clone()),
            ("NEXUS_ICEBERG_S3_ENDPOINT", params.s3_endpoint.clone()),
            ("NEXUS_ICEBERG_S3_REGION", params.s3_region.clone()),
            ("NEXUS_ICEBERG_S3_PATH_STYLE", "true".to_string()),
            ("NEXUS_ICEBERG_S3_ACCESS_KEY_ID", params.s3_access_key.clone()),
            ("NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY", params.s3_secret_key.clone()),
        ]);
        command.env_remove("NEXUS_ICEBERG_S3_SESSION_TOKEN");
        command.env_remove("AWS_SESSION_TOKEN");
        command.env_remove("AWS_SECURITY_TOKEN");

        eprintln!("starting release Flight SQL server at {} with devices {}", params.server_addr, params.device_csv());
        let child = command.spawn()?;
        Ok(Self { child, stopped: false, artifact_dir: params.artifact_dir.clone() })
    }

    fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    async fn shutdown(&mut self) -> E2eResult<()> {
        let Some(pid) = self.child.id() else {
            return Err("release server child has no process id".into());
        };
        let signal_status = std::process::Command::new("kill").args(["-TERM", &pid.to_string()]).status()?;
        if !signal_status.success() {
            return Err(format!("failed to send SIGTERM to release server pid {pid}: {signal_status}").into());
        }
        let status = tokio::time::timeout(Duration::from_secs(30), self.child.wait())
            .await
            .map_err(|_| format!("release server pid {pid} did not drain within 30 seconds"))??;
        self.stopped = true;
        fs::write(self.artifact_dir.join("server.exit.txt"), format!("{status}\n"))?;
        if !status.success() {
            return Err(format!("release server exited unsuccessfully after SIGTERM: {status}").into());
        }
        Ok(())
    }
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.child.start_kill();
        }
    }
}

async fn connect_with_retry(url: &str, child: &mut Child) -> E2eResult<Channel> {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Some(status) = child.try_wait()? {
            return Err(format!("release server exited before accepting Flight SQL connections: {status}").into());
        }
        match Endpoint::from_shared(url.to_string())?.connect().await {
            Ok(channel) => return Ok(channel),
            Err(error) if Instant::now() >= deadline => {
                return Err(format!("release server did not accept connections at {url}: {error:?}").into());
            },
            Err(_) => {},
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[derive(Clone)]
struct Workload {
    id: &'static str,
    sql: String,
    ordered: bool,
}

fn tpch_workloads(params: &Params) -> BTreeMap<&'static str, Workload> {
    let lineitem = params.table("lineitem");
    let customer = params.table("customer");
    let orders = params.table("orders");
    let nation = params.table("nation");
    BTreeMap::from([
        (
            "q01",
            Workload {
                id: "q01",
                ordered: true,
                sql: format!(
                    "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, \
                     SUM(l_extendedprice) AS sum_base_price, \
                     SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, \
                     SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, \
                     AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, \
                     AVG(l_discount) AS avg_disc, COUNT(*) AS count_order FROM {lineitem} \
                     WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY \
                     GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus"
                ),
            },
        ),
        (
            "q03",
            Workload {
                id: "q03",
                ordered: true,
                sql: format!(
                    "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
                     o_orderdate, o_shippriority FROM {customer} JOIN {orders} ON c_custkey = o_custkey \
                     JOIN {lineitem} ON l_orderkey = o_orderkey WHERE c_mktsegment = 'BUILDING' \
                     AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15' \
                     GROUP BY l_orderkey, o_orderdate, o_shippriority \
                     ORDER BY revenue DESC, o_orderdate LIMIT 10"
                ),
            },
        ),
        (
            "q06",
            Workload {
                id: "q06",
                ordered: true,
                sql: format!(
                    "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM {lineitem} \
                     WHERE l_shipdate >= DATE '1994-01-01' \
                     AND l_shipdate < DATE '1994-01-01' + INTERVAL '1' YEAR \
                     AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"
                ),
            },
        ),
        (
            "q10",
            Workload {
                id: "q10",
                ordered: true,
                sql: format!(
                    "SELECT c_custkey, c_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue, \
                     c_acctbal, n_name, c_address, c_phone, c_comment FROM {customer} \
                     JOIN {orders} ON c_custkey = o_custkey JOIN {lineitem} ON l_orderkey = o_orderkey \
                     JOIN {nation} ON c_nationkey = n_nationkey \
                     WHERE o_orderdate >= DATE '1993-10-01' \
                     AND o_orderdate < DATE '1993-10-01' + INTERVAL '3' MONTH AND l_returnflag = 'R' \
                     GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment \
                     ORDER BY revenue DESC LIMIT 20"
                ),
            },
        ),
    ])
}

#[derive(Clone)]
struct CpuReference {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

async fn build_cpu_references(
    params: &Params,
    workloads: &BTreeMap<&'static str, Workload>,
) -> E2eResult<BTreeMap<&'static str, CpuReference>> {
    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(RestIcebergCatalogBackend::from_config_env(params.rest_config()).await?);
    let provider =
        build_scoped_catalog_provider_from_backend(backend, "rest", &params.namespace, IcebergProviderKind::Cpu)
            .await?;
    let context = SessionContext::new_with_config(SessionConfig::new());
    context.register_catalog("lake", Arc::new(provider));

    let mut references = BTreeMap::new();
    for workload in workloads.values() {
        let dataframe = context.sql(&workload.sql).await?;
        let batches = dataframe.collect().await?;
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .ok_or_else(|| format!("CPU reference {} returned no batches", workload.id))?;
        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        if rows == 0 {
            return Err(format!("CPU reference {} returned no rows", workload.id).into());
        }
        eprintln!("CPU reference {}: {rows} row(s)", workload.id);
        references.insert(workload.id, CpuReference { schema, batches });
    }
    Ok(references)
}

async fn run_acceptance(
    params: &Params,
    channel: Channel,
    workloads: &BTreeMap<&'static str, Workload>,
    cpu_references: &BTreeMap<&'static str, CpuReference>,
) -> E2eResult<()> {
    let before = get_admission_stats(&channel).await?;
    validate_static_topology(&before, &params.devices)?;
    fs::write(params.artifact_dir.join("admission.before.txt"), &before)?;

    let probe_stop = Arc::new(AtomicBool::new(false));
    let probe_task = tokio::spawn(run_probes(channel.clone(), Arc::clone(&probe_stop)));

    let waves = async {
        eprintln!("running synchronized cold Q06 wave ({Q06_WAVE_SIZE} statements)");
        let q06 = workloads.get("q06").expect("q06 workload").clone();
        let q06_cpu = cpu_references.get("q06").expect("q06 CPU reference").clone();
        let cold = run_wave(&channel, "cold_q06", vec![q06.clone(); Q06_WAVE_SIZE], q06_cpu.clone()).await?;

        eprintln!("running synchronized warm Q06 wave ({Q06_WAVE_SIZE} statements)");
        let warm = run_wave(&channel, "warm_q06", vec![q06; Q06_WAVE_SIZE], q06_cpu).await?;

        eprintln!("running mixed concurrent Q01/Q03/Q06/Q10 wave");
        let mut mixed_workloads = Vec::with_capacity(MAX_CONCURRENT_QUERIES);
        for _ in 0..MAX_QUERIES_PER_DEVICE {
            for query_id in ["q01", "q03", "q06", "q10"] {
                mixed_workloads.push(workloads.get(query_id).expect("mixed workload").clone());
            }
        }
        let mixed = run_mixed_wave(&channel, "mixed_olap", mixed_workloads, cpu_references).await?;
        Ok::<_, Box<dyn std::error::Error + Send + Sync>>((cold, warm, mixed))
    }
    .await;

    probe_stop.store(true, Ordering::Release);
    let probes: E2eResult<ProbeReport> = match probe_task.await {
        Ok(result) => result,
        Err(error) => Err(format!("probe task failed: {error}").into()),
    };
    let (cold, warm, mixed) = waves?;
    let probes = probes?;
    if probes.saturated_count < REQUIRED_SATURATED_PROBES {
        return Err(format!(
            "only {} diagnostics/GetFlightInfo probe pairs completed while active GPU work was at least {SATURATION_FLOOR}; required {REQUIRED_SATURATED_PROBES}",
            probes.saturated_count
        )
        .into());
    }
    if probes.tickets_issued != probes.tickets_retired || probes.tickets_issued != probes.rows.len() {
        return Err(format!(
            "liveness probe ticket accounting mismatch: issued={} retired={} rows={}",
            probes.tickets_issued,
            probes.tickets_retired,
            probes.rows.len()
        )
        .into());
    }
    fs::write(params.artifact_dir.join("admission.peak.txt"), &probes.peak_payload)?;
    write_probe_artifacts(params, &probes)?;

    validate_cold_and_warm_cache(&cold, &warm, &params.devices)?;

    eprintln!("running cancellation and dropped-stream churn");
    let churn = tokio::time::timeout(
        CHURN_TIMEOUT,
        run_churn(
            &channel,
            workloads.get("q03").expect("q03 workload"),
            cpu_references.get("q03").expect("q03 CPU reference"),
        ),
    )
    .await
    .map_err(|_| format!("cancellation/drop churn exceeded {} seconds", CHURN_TIMEOUT.as_secs()))??;

    let after = wait_for_idle(&channel).await?;
    validate_final_diagnostics(&after, &params.devices)?;
    fs::write(params.artifact_dir.join("admission.after.txt"), &after)?;

    let mut successful = Vec::new();
    successful.extend(cold);
    successful.extend(warm);
    successful.extend(mixed);
    successful.extend(churn.completed.iter().cloned());
    validate_observed_devices(&successful, &params.devices)?;
    write_query_artifacts(params, &successful, &churn)?;
    write_summary(params, &successful, &churn, &probes, &after)?;
    Ok(())
}

#[derive(Clone)]
struct QueryEvidence {
    phase: String,
    workload: &'static str,
    query_id: String,
    device_ordinal: usize,
    output_rows: u64,
    source_cache_hits: u64,
    source_cache_misses: u64,
    source_cache_inserts: u64,
    d2h_count: u64,
    d2h_bytes: u64,
    parity_mode: String,
    cpu_fingerprint: String,
    native_fingerprint: String,
    stats_payload: String,
}

async fn run_wave(
    channel: &Channel,
    phase: &'static str,
    workloads: Vec<Workload>,
    cpu_reference: CpuReference,
) -> E2eResult<Vec<QueryEvidence>> {
    let barrier = Arc::new(Barrier::new(workloads.len() + 1));
    let mut tasks = JoinSet::new();
    for workload in workloads {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let cpu_reference = cpu_reference.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            tokio::time::timeout(QUERY_TIMEOUT, execute_success(&channel, phase, workload, cpu_reference))
                .await
                .map_err(|_| format!("{phase} statement exceeded {} seconds", QUERY_TIMEOUT.as_secs()))?
        });
    }
    barrier.wait().await;
    collect_query_tasks(tasks).await
}

async fn run_mixed_wave(
    channel: &Channel,
    phase: &'static str,
    workloads: Vec<Workload>,
    cpu_references: &BTreeMap<&'static str, CpuReference>,
) -> E2eResult<Vec<QueryEvidence>> {
    let barrier = Arc::new(Barrier::new(workloads.len() + 1));
    let mut tasks = JoinSet::new();
    for workload in workloads {
        let channel = channel.clone();
        let barrier = Arc::clone(&barrier);
        let cpu_reference = cpu_references.get(workload.id).expect("mixed CPU reference").clone();
        tasks.spawn(async move {
            barrier.wait().await;
            tokio::time::timeout(QUERY_TIMEOUT, execute_success(&channel, phase, workload, cpu_reference))
                .await
                .map_err(|_| format!("{phase} statement exceeded {} seconds", QUERY_TIMEOUT.as_secs()))?
        });
    }
    barrier.wait().await;
    collect_query_tasks(tasks).await
}

async fn collect_query_tasks(mut tasks: JoinSet<E2eResult<QueryEvidence>>) -> E2eResult<Vec<QueryEvidence>> {
    let mut rows = Vec::new();
    while let Some(result) = tasks.join_next().await {
        rows.push(result.map_err(|error| format!("query task failed: {error}"))??);
    }
    Ok(rows)
}

async fn execute_success(
    channel: &Channel,
    phase: &str,
    workload: Workload,
    cpu_reference: CpuReference,
) -> E2eResult<QueryEvidence> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let flight_info = client.execute(workload.sql.clone(), None).await?;
    collect_success_from_info(channel, phase, workload, cpu_reference, flight_info).await
}

async fn collect_success_from_info(
    channel: &Channel,
    phase: &str,
    workload: Workload,
    cpu_reference: CpuReference,
    flight_info: FlightInfo,
) -> E2eResult<QueryEvidence> {
    let query_id = query_id_from_statement_flight_info(&flight_info)?;
    let (native_schema, native_batches) = collect_flight_info(channel, flight_info).await?;
    let comparison = compare_record_batches(
        cpu_reference.schema.as_ref(),
        &cpu_reference.batches,
        native_schema.as_ref(),
        &native_batches,
        workload.ordered,
        ParityFloatTolerance::exact(),
    );
    if comparison.status != ParityStatus::Matched {
        return Err(format!(
            "{} {} CPU/native parity failed: status={} mode={} detail={:?}",
            phase,
            workload.id,
            comparison.status.as_str(),
            comparison.mode.as_str(),
            comparison.detail
        )
        .into());
    }

    let stats_payload = get_query_stats(channel, &query_id).await?;
    let planning = tsv_section_single_row(&stats_payload, "planning")?;
    let execution = tsv_section_single_row(&stats_payload, "execution")?;
    require_value(&planning, "outcome", "accepted_replaced", &query_id)?;
    require_value(&planning, "post_native_execution_path", "native", &query_id)?;
    require_value(&execution, "planning_outcome", "accepted_replaced", &query_id)?;
    require_value(&execution, "post_native_execution_path", "native", &query_id)?;
    require_value(&execution, "native_runtime_started", "true", &query_id)?;
    require_value(&execution, "runtime_outcome", "succeeded", &query_id)?;
    if !stats_payload.contains("# outcome=succeeded\n") || !stats_payload.contains("# sealed=true\n") {
        return Err(
            format!("successful query {query_id} was not retained as sealed+succeeded:\n{stats_payload}").into()
        );
    }

    let device_ordinal = parse_usize_field(&execution, "device_ordinal", &query_id)?;
    let output_rows = parse_u64_field(&execution, "output_rows", &query_id)?;
    let d2h_count = parse_u64_field(&execution, "device_to_host_materializations", &query_id)?;
    let d2h_bytes = parse_u64_field(&execution, "device_to_host_materialized_bytes", &query_id)?;
    if d2h_count == 0 || d2h_bytes == 0 {
        return Err(format!(
            "successful consumed query {query_id} lacks positive D2H evidence: count={d2h_count} bytes={d2h_bytes}"
        )
        .into());
    }
    let native_rows = native_batches.iter().map(RecordBatch::num_rows).sum::<usize>() as u64;
    if output_rows != native_rows {
        return Err(format!(
            "query {query_id} execution output_rows={output_rows} but Flight stream yielded {native_rows} rows"
        )
        .into());
    }

    Ok(QueryEvidence {
        phase: phase.to_string(),
        workload: workload.id,
        query_id,
        device_ordinal,
        output_rows,
        source_cache_hits: parse_u64_field(&execution, "source_cache_hits", "successful query")?,
        source_cache_misses: parse_u64_field(&execution, "source_cache_misses", "successful query")?,
        source_cache_inserts: parse_u64_field(&execution, "source_cache_inserts", "successful query")?,
        d2h_count,
        d2h_bytes,
        parity_mode: comparison.mode.as_str().to_string(),
        cpu_fingerprint: comparison.cpu_fingerprint,
        native_fingerprint: comparison.native_fingerprint,
        stats_payload,
    })
}

async fn collect_flight_info(channel: &Channel, flight_info: FlightInfo) -> E2eResult<(SchemaRef, Vec<RecordBatch>)> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let mut schema = None;
    let mut batches = Vec::new();
    for endpoint in flight_info.endpoint {
        let ticket = endpoint.ticket.ok_or("Flight SQL endpoint missing ticket")?;
        let mut stream = client.do_get(ticket).await?;
        while let Some(batch) = stream.try_next().await? {
            batches.push(batch);
        }
        schema = stream.schema().cloned().or(schema);
    }
    let schema =
        schema.or_else(|| batches.first().map(RecordBatch::schema)).ok_or("Flight stream did not expose schema")?;
    Ok((schema, batches))
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

async fn get_query_stats(channel: &Channel, query_id: &str) -> E2eResult<String> {
    action_payload(channel, "GetQueryStats", query_id.as_bytes().to_vec()).await
}

async fn get_admission_stats(channel: &Channel) -> E2eResult<String> {
    action_payload(channel, "GetAdmissionStats", Vec::new()).await
}

async fn action_payload(channel: &Channel, action_type: &str, body: Vec<u8>) -> E2eResult<String> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let action = arrow_flight::Action { r#type: action_type.to_string(), body: body.into() };
    let mut stream = client.do_action(action).await?;
    let result = stream.message().await?.ok_or_else(|| format!("{action_type} returned no result"))?;
    if stream.message().await?.is_some() {
        return Err(format!("{action_type} returned more than one result").into());
    }
    Ok(String::from_utf8(result.body.to_vec())?)
}

#[derive(Default)]
struct ProbeReport {
    rows: Vec<ProbeRow>,
    saturated_count: usize,
    peak_score: u64,
    peak_payload: String,
    tickets_issued: usize,
    tickets_retired: usize,
}

struct ProbeRow {
    sequence: usize,
    active_queries: u64,
    queued_queries: u64,
    active_gpu_jobs: u64,
    diagnostics_ms: u128,
    get_flight_info_ms: u128,
    ticket_retire_ms: u128,
    saturated: bool,
}

async fn run_probes(channel: Channel, stop: Arc<AtomicBool>) -> E2eResult<ProbeReport> {
    let mut report = ProbeReport::default();
    while !stop.load(Ordering::Acquire) {
        let diagnostics_started = Instant::now();
        let payload = tokio::time::timeout(PROBE_TIMEOUT, get_admission_stats(&channel))
            .await
            .map_err(|_| "GetAdmissionStats exceeded two-second liveness bound")??;
        let diagnostics_ms = diagnostics_started.elapsed().as_millis();
        let admission = tsv_section_single_row(&payload, "admission")?;
        let gpu_execution = tsv_section_single_row(&payload, "gpu_execution")?;
        let active_queries = parse_u64_field(&admission, "active_queries", "admission probe")?;
        let queued_queries = parse_u64_field(&admission, "queued_queries", "admission probe")?;
        let active_gpu_jobs = parse_u64_field(&gpu_execution, "active_query_jobs", "admission probe")?;
        let score = active_queries.max(active_gpu_jobs);
        if score > report.peak_score || report.peak_payload.is_empty() {
            report.peak_score = score;
            report.peak_payload.clone_from(&payload);
        }

        let info_started = Instant::now();
        let flight_info = tokio::time::timeout(PROBE_TIMEOUT, lightweight_get_flight_info(&channel))
            .await
            .map_err(|_| "GetFlightInfo(SELECT 1) exceeded two-second liveness bound")??;
        let get_flight_info_ms = info_started.elapsed().as_millis();
        report.tickets_issued += 1;
        let retire_started = Instant::now();
        let cancel_result = tokio::time::timeout(PROBE_TIMEOUT, cancel_query(&channel, &flight_info))
            .await
            .map_err(|_| "CancelQuery for liveness ticket exceeded two-second bound")??;
        if cancel_result.result != CANCEL_RESULT_CANCELLING {
            return Err(format!(
                "liveness ticket CancelQuery returned {}, expected {CANCEL_RESULT_CANCELLING}",
                cancel_result.result
            )
            .into());
        }
        let ticket_retire_ms = retire_started.elapsed().as_millis();
        report.tickets_retired += 1;
        let saturated = score >= SATURATION_FLOOR;
        if saturated {
            report.saturated_count += 1;
        }
        report.rows.push(ProbeRow {
            sequence: report.rows.len() + 1,
            active_queries,
            queued_queries,
            active_gpu_jobs,
            diagnostics_ms,
            get_flight_info_ms,
            ticket_retire_ms,
            saturated,
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    Ok(report)
}

async fn lightweight_get_flight_info(channel: &Channel) -> E2eResult<FlightInfo> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let flight_info = client.execute("SELECT 1 AS liveness_probe".to_string(), None).await?;
    if flight_info.endpoint.is_empty() {
        return Err("GetFlightInfo(SELECT 1) returned no endpoint".into());
    }
    Ok(flight_info)
}

type TsvRow = BTreeMap<String, String>;

fn tsv_section(payload: &str, name: &str) -> E2eResult<Vec<TsvRow>> {
    let marker = format!("[{name}]");
    let mut lines = payload.lines().skip_while(|line| *line != marker).skip(1);
    let header = lines.next().ok_or_else(|| format!("missing [{name}] section in payload:\n{payload}"))?;
    if header.is_empty() || header.starts_with('[') {
        return Err(format!("missing [{name}] TSV header in payload:\n{payload}").into());
    }
    let columns = header.split('\t').collect::<Vec<_>>();
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() || line.starts_with('[') {
            break;
        }
        if line.starts_with('#') {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != columns.len() {
            return Err(format!(
                "[{name}] TSV width mismatch: {} columns, {} values in {line:?}",
                columns.len(),
                values.len()
            )
            .into());
        }
        rows.push(
            columns.iter().zip(values).map(|(column, value)| ((*column).to_string(), value.to_string())).collect(),
        );
    }
    Ok(rows)
}

fn tsv_section_single_row(payload: &str, name: &str) -> E2eResult<TsvRow> {
    let rows = tsv_section(payload, name)?;
    if rows.len() != 1 {
        return Err(format!("expected one [{name}] row, got {}", rows.len()).into());
    }
    Ok(rows.into_iter().next().expect("one row"))
}

fn field<'a>(row: &'a TsvRow, name: &str, context: &str) -> E2eResult<&'a str> {
    row.get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("missing TSV field {name:?} for {context}; row={row:?}").into())
}

fn parse_u64_field(row: &TsvRow, name: &str, context: &str) -> E2eResult<u64> {
    let raw = field(row, name, context)?;
    raw.parse::<u64>().map_err(|error| format!("invalid {name}={raw:?} for {context}: {error}").into())
}

fn parse_usize_field(row: &TsvRow, name: &str, context: &str) -> E2eResult<usize> {
    let raw = field(row, name, context)?;
    raw.parse::<usize>().map_err(|error| format!("invalid {name}={raw:?} for {context}: {error}").into())
}

fn require_value(row: &TsvRow, name: &str, expected: &str, context: &str) -> E2eResult<()> {
    let actual = field(row, name, context)?;
    if actual != expected {
        return Err(format!("{context} expected {name}={expected:?}, got {actual:?}; row={row:?}").into());
    }
    Ok(())
}

fn row_device_set(rows: &[TsvRow], context: &str) -> E2eResult<BTreeSet<usize>> {
    rows.iter().map(|row| parse_usize_field(row, "device_ordinal", context)).collect()
}

fn expected_device_set(devices: &[usize]) -> BTreeSet<usize> {
    devices.iter().copied().collect()
}

fn validate_static_topology(payload: &str, devices: &[usize]) -> E2eResult<()> {
    if !payload.contains("# backend=query_service\n")
        || !payload.contains(&format!(
            "# device_ordinals={}\n",
            devices.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
        ))
        || !payload.contains(&format!("# max_concurrent_queries={MAX_CONCURRENT_QUERIES}\n"))
        || !payload.contains(&format!("# max_queries_per_device={MAX_QUERIES_PER_DEVICE}\n"))
        || !payload.contains("# placement_policy=least_loaded\n")
    {
        return Err(format!("admission payload does not describe the requested production topology:\n{payload}").into());
    }
    let expected = expected_device_set(devices);
    for section in [
        "memory_governor_reports",
        "device_memory_failure_reports",
        "gpu_execution_pools",
        "device_admission_slots",
        "device_telemetry",
    ] {
        let rows = tsv_section(payload, section)?;
        if rows.len() != devices.len() || row_device_set(&rows, section)? != expected {
            return Err(format!("[{section}] must contain exactly devices {expected:?}; rows={rows:?}").into());
        }
    }

    let governors = tsv_section(payload, "memory_governor_reports")?;
    let capacities = governors
        .iter()
        .map(|row| parse_u64_field(row, "capacity_bytes", "memory governor"))
        .collect::<E2eResult<BTreeSet<_>>>()?;
    if capacities.len() != 1 {
        return Err(format!("heterogeneous GPU capacities are unsupported by this acceptance: {capacities:?}").into());
    }

    let telemetry = tsv_section(payload, "device_telemetry")?;
    let mut cuda_totals = BTreeSet::new();
    let mut nvml_totals = BTreeSet::new();
    for row in telemetry {
        require_value(&row, "cuda_status", "available", "device telemetry")?;
        require_value(&row, "nvml_status", "available", "device telemetry")?;
        cuda_totals.insert(parse_u64_field(&row, "cuda_total_bytes", "device telemetry")?);
        nvml_totals.insert(parse_u64_field(&row, "nvml_total_bytes", "device telemetry")?);
    }
    if cuda_totals.len() != 1 || nvml_totals.len() != 1 {
        return Err(
            format!("heterogeneous GPU totals are unsupported: CUDA={cuda_totals:?}, NVML={nvml_totals:?}").into()
        );
    }
    Ok(())
}

fn validate_cold_and_warm_cache(cold: &[QueryEvidence], warm: &[QueryEvidence], devices: &[usize]) -> E2eResult<()> {
    for device in devices {
        let cold_rows = cold.iter().filter(|row| row.device_ordinal == *device).collect::<Vec<_>>();
        let warm_rows = warm.iter().filter(|row| row.device_ordinal == *device).collect::<Vec<_>>();
        let cold_misses = cold_rows.iter().map(|row| row.source_cache_misses).sum::<u64>();
        let cold_inserts = cold_rows.iter().map(|row| row.source_cache_inserts).sum::<u64>();
        let warm_hits = warm_rows.iter().map(|row| row.source_cache_hits).sum::<u64>();
        let warm_misses = warm_rows.iter().map(|row| row.source_cache_misses).sum::<u64>();
        if cold_rows.is_empty() || cold_misses == 0 || cold_inserts == 0 {
            return Err(format!(
                "device {device} lacks cold decoded-source-cache evidence: rows={} misses={cold_misses} inserts={cold_inserts}",
                cold_rows.len()
            )
            .into());
        }
        // REST host-buffer staging assigns query-scoped reader locations, so
        // cross-query key stability is not guaranteed on this substrate.
        // Require active cache lookup evidence on the warm wave, but do not
        // mislabel staging-identity misses as a decoded-cache regression.
        if warm_rows.is_empty() || warm_hits.saturating_add(warm_misses) == 0 {
            return Err(format!(
                "device {device} lacks warm Q06 decoded-cache lookup evidence: rows={} hits={warm_hits} misses={warm_misses}",
                warm_rows.len()
            )
            .into());
        }
    }
    Ok(())
}

fn validate_observed_devices(successful: &[QueryEvidence], devices: &[usize]) -> E2eResult<()> {
    let observed = successful.iter().map(|row| row.device_ordinal).collect::<BTreeSet<_>>();
    let expected = expected_device_set(devices);
    if observed != expected {
        return Err(
            format!("successful Flight SQL queries observed devices {observed:?}, expected {expected:?}").into()
        );
    }
    Ok(())
}

struct ChurnReport {
    completed: Vec<QueryEvidence>,
    cancelled: Vec<(String, String)>,
    dropped: Vec<(String, String)>,
}

#[derive(Clone, Copy)]
enum ChurnMode {
    Complete,
    Cancel,
    Drop,
}

enum ChurnTaskOutput {
    Complete(QueryEvidence),
    Cancel(String),
    Drop(String),
}

async fn run_churn(channel: &Channel, workload: &Workload, cpu_reference: &CpuReference) -> E2eResult<ChurnReport> {
    // Twenty-four simultaneous DoGet calls exceed the sixteen-query active
    // ceiling. Eight are cancelled, eight drop their streams, and eight must
    // still complete with exact parity.
    let modes = [vec![ChurnMode::Complete; 8], vec![ChurnMode::Cancel; 8], vec![ChurnMode::Drop; 8]].concat();
    let mut statements = Vec::with_capacity(modes.len());
    for mode in modes {
        let mut client = FlightSqlServiceClient::new(channel.clone());
        let flight_info = tokio::time::timeout(Duration::from_secs(60), client.execute(workload.sql.clone(), None))
            .await
            .map_err(|_| "churn GetFlightInfo exceeded 60 seconds")??;
        let query_id = query_id_from_statement_flight_info(&flight_info)?;
        statements.push((mode, query_id, flight_info));
    }

    let start = Arc::new(Barrier::new(statements.len() + 1));
    let cancel_release = Arc::new(Barrier::new(9));
    let (cancel_ready_tx, mut cancel_ready_rx) = mpsc::channel::<String>(8);
    let mut cancel_infos = Vec::new();
    let mut tasks: JoinSet<E2eResult<ChurnTaskOutput>> = JoinSet::new();

    for (mode, query_id, flight_info) in statements {
        if matches!(mode, ChurnMode::Cancel) {
            cancel_infos.push((query_id.clone(), flight_info.clone()));
        }
        let channel = channel.clone();
        let start = Arc::clone(&start);
        let cancel_release = Arc::clone(&cancel_release);
        let cancel_ready_tx = cancel_ready_tx.clone();
        let workload = workload.clone();
        let cpu_reference = cpu_reference.clone();
        tasks.spawn(async move {
            start.wait().await;
            match mode {
                ChurnMode::Complete => {
                    let evidence =
                        collect_success_from_info(&channel, "churn_complete", workload, cpu_reference, flight_info)
                            .await?;
                    Ok(ChurnTaskOutput::Complete(evidence))
                },
                ChurnMode::Cancel => {
                    let ticket = flight_info
                        .endpoint
                        .first()
                        .and_then(|endpoint| endpoint.ticket.clone())
                        .ok_or("cancel churn FlightInfo missing ticket")?;
                    let mut client = FlightSqlServiceClient::new(channel);
                    let mut stream = client.do_get(ticket).await?;
                    cancel_ready_tx.send(query_id.clone()).await?;
                    cancel_release.wait().await;
                    while stream.try_next().await.is_ok_and(|batch| batch.is_some()) {}
                    Ok(ChurnTaskOutput::Cancel(query_id))
                },
                ChurnMode::Drop => {
                    let ticket = flight_info
                        .endpoint
                        .first()
                        .and_then(|endpoint| endpoint.ticket.clone())
                        .ok_or("drop churn FlightInfo missing ticket")?;
                    let mut client = FlightSqlServiceClient::new(channel);
                    let stream = client.do_get(ticket).await?;
                    drop(stream);
                    Ok(ChurnTaskOutput::Drop(query_id))
                },
            }
        });
    }
    drop(cancel_ready_tx);
    start.wait().await;

    let mut ready = BTreeSet::new();
    while ready.len() < cancel_infos.len() {
        let query_id = tokio::time::timeout(Duration::from_secs(30), cancel_ready_rx.recv())
            .await
            .map_err(|_| "cancel churn streams did not become ready within 30 seconds")?
            .ok_or("cancel churn readiness channel closed early")?;
        ready.insert(query_id);
    }
    for (query_id, flight_info) in &cancel_infos {
        if !ready.contains(query_id) {
            return Err(format!("cancel churn stream {query_id} did not become ready").into());
        }
        let result = cancel_query(channel, flight_info).await?;
        if result.result != CANCEL_RESULT_CANCELLING {
            return Err(format!(
                "CancelQuery for active churn query {query_id} returned {}, expected {CANCEL_RESULT_CANCELLING}",
                result.result
            )
            .into());
        }
    }
    cancel_release.wait().await;

    let mut completed = Vec::new();
    let mut cancelled_ids = Vec::new();
    let mut dropped_ids = Vec::new();
    while let Some(result) = tasks.join_next().await {
        match result.map_err(|error| format!("churn task failed: {error}"))?? {
            ChurnTaskOutput::Complete(row) => completed.push(row),
            ChurnTaskOutput::Cancel(query_id) => cancelled_ids.push(query_id),
            ChurnTaskOutput::Drop(query_id) => dropped_ids.push(query_id),
        }
    }
    if completed.len() != 8 || cancelled_ids.len() != 8 || dropped_ids.len() != 8 {
        return Err(format!(
            "churn outcome count mismatch: completed={} cancelled={} dropped={}",
            completed.len(),
            cancelled_ids.len(),
            dropped_ids.len()
        )
        .into());
    }

    let mut cancelled = Vec::new();
    for query_id in cancelled_ids {
        let payload = wait_for_cancelled_query_stats(channel, &query_id).await?;
        cancelled.push((query_id, payload));
    }
    let mut dropped = Vec::new();
    for query_id in dropped_ids {
        let payload = wait_for_cancelled_query_stats(channel, &query_id).await?;
        dropped.push((query_id, payload));
    }
    Ok(ChurnReport { completed, cancelled, dropped })
}

async fn cancel_query(channel: &Channel, flight_info: &FlightInfo) -> E2eResult<ActionCancelQueryResult> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    let request = ActionCancelQueryRequest { info: flight_info.encode_to_vec().into() };
    let action =
        arrow_flight::Action { r#type: "CancelQuery".to_string(), body: request.as_any().encode_to_vec().into() };
    let mut stream = client.do_action(action).await?;
    let result = stream.message().await?.ok_or("CancelQuery returned no result")?;
    if stream.message().await?.is_some() {
        return Err("CancelQuery returned more than one result".into());
    }
    let any = Any::decode(&*result.body)?;
    any.unpack::<ActionCancelQueryResult>()?.ok_or_else(|| "CancelQuery returned wrong result type".into())
}

async fn wait_for_cancelled_query_stats(channel: &Channel, query_id: &str) -> E2eResult<String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let payload = get_query_stats(channel, query_id).await?;
        if payload.contains("# sealed=true\n") {
            if !payload.contains("# outcome=failed\n")
                || !payload.contains("[structured_error]\n")
                || !payload.contains("code=cancelled\n")
            {
                return Err(format!(
                    "cancelled/dropped query {query_id} lacks structured cancellation evidence:\n{payload}"
                )
                .into());
            }
            if payload.contains("code=internal") || payload.contains("kind=internal") {
                return Err(format!("cancelled/dropped query {query_id} reported internal failure:\n{payload}").into());
            }
            return Ok(payload);
        }
        if Instant::now() >= deadline {
            return Err(format!("query {query_id} was not sealed within 30 seconds:\n{payload}").into());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_idle(channel: &Channel) -> E2eResult<String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let payload = get_admission_stats(channel).await?;
        if diagnostics_are_idle(&payload)? {
            return Ok(payload);
        }
        if Instant::now() >= deadline {
            return Err(format!("server did not return to an idle resource state within 60 seconds:\n{payload}").into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn diagnostics_are_idle(payload: &str) -> E2eResult<bool> {
    let admission = tsv_section_single_row(payload, "admission")?;
    if parse_u64_field(&admission, "active_queries", "final admission")? != 0
        || parse_u64_field(&admission, "queued_queries", "final admission")? != 0
    {
        return Ok(false);
    }
    if !tsv_section(payload, "memory_governor_grants")?.is_empty() {
        return Ok(false);
    }
    let gpu = tsv_section_single_row(payload, "gpu_execution")?;
    if parse_u64_field(&gpu, "active_query_jobs", "final GPU execution")? != 0 {
        return Ok(false);
    }
    for row in tsv_section(payload, "gpu_execution_pools")? {
        for field_name in ["active_jobs", "queued_jobs", "active_query_jobs"] {
            if parse_u64_field(&row, field_name, "final GPU pool")? != 0 {
                return Ok(false);
            }
        }
    }
    for row in tsv_section(payload, "device_admission_slots")? {
        if parse_u64_field(&row, "process_active_query_slots", "final device slots")? != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn validate_final_diagnostics(payload: &str, devices: &[usize]) -> E2eResult<()> {
    validate_static_topology(payload, devices)?;
    if !diagnostics_are_idle(payload)? {
        return Err(format!("final diagnostics are not idle:\n{payload}").into());
    }

    let gpu = tsv_section_single_row(payload, "gpu_execution")?;
    let peak = parse_u64_field(&gpu, "peak_active_query_jobs", "global GPU execution")?;
    if peak != MAX_CONCURRENT_QUERIES as u64 {
        return Err(format!("global GPU executor peak must be exactly {MAX_CONCURRENT_QUERIES}, got {peak}").into());
    }
    for row in tsv_section(payload, "gpu_execution_pools")? {
        require_value(&row, "worker_count", &MAX_QUERIES_PER_DEVICE.to_string(), "GPU execution pool")?;
        require_value(&row, "peak_active_query_jobs", &MAX_QUERIES_PER_DEVICE.to_string(), "GPU execution pool")?;
    }
    for row in tsv_section(payload, "device_memory_failure_reports")? {
        for field_name in
            ["allocation_failure_count", "total_failed_allocation_bytes", "largest_failed_allocation_bytes"]
        {
            require_value(&row, field_name, "0", "device allocation failures")?;
        }
    }
    Ok(())
}

fn write_probe_artifacts(params: &Params, probes: &ProbeReport) -> E2eResult<()> {
    let mut output = String::from(
        "sequence\tactive_queries\tqueued_queries\tactive_gpu_jobs\tdiagnostics_ms\tget_flight_info_ms\t\
         ticket_retire_ms\tsaturated\n",
    );
    for row in &probes.rows {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.sequence,
            row.active_queries,
            row.queued_queries,
            row.active_gpu_jobs,
            row.diagnostics_ms,
            row.get_flight_info_ms,
            row.ticket_retire_ms,
            row.saturated
        ));
    }
    fs::write(params.artifact_dir.join("probes.tsv"), output)?;
    Ok(())
}

fn write_query_artifacts(params: &Params, successful: &[QueryEvidence], churn: &ChurnReport) -> E2eResult<()> {
    let mut output = String::from(
        "phase\tworkload\tquery_id\tdevice_ordinal\toutput_rows\tsource_cache_hits\tsource_cache_misses\t\
         source_cache_inserts\td2h_count\td2h_bytes\tparity_mode\tcpu_fingerprint\tnative_fingerprint\n",
    );
    for row in successful {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            tsv_escape(&row.phase),
            row.workload,
            tsv_escape(&row.query_id),
            row.device_ordinal,
            row.output_rows,
            row.source_cache_hits,
            row.source_cache_misses,
            row.source_cache_inserts,
            row.d2h_count,
            row.d2h_bytes,
            row.parity_mode,
            row.cpu_fingerprint,
            row.native_fingerprint,
        ));
    }
    fs::write(params.artifact_dir.join("queries.tsv"), output)?;

    let mut payloads = String::new();
    for row in successful {
        append_query_payload(&mut payloads, &row.phase, &row.query_id, &row.stats_payload);
    }
    for (query_id, payload) in &churn.cancelled {
        append_query_payload(&mut payloads, "churn_cancelled", query_id, payload);
    }
    for (query_id, payload) in &churn.dropped {
        append_query_payload(&mut payloads, "churn_dropped", query_id, payload);
    }
    fs::write(params.artifact_dir.join("query_stats.txt"), payloads)?;
    Ok(())
}

fn append_query_payload(output: &mut String, phase: &str, query_id: &str, payload: &str) {
    output.push_str(&format!("===== phase={} query_id={} =====\n", tsv_escape(phase), tsv_escape(query_id)));
    output.push_str(payload);
    if !payload.ends_with('\n') {
        output.push('\n');
    }
}

fn tsv_escape(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn write_summary(
    params: &Params,
    successful: &[QueryEvidence],
    churn: &ChurnReport,
    probes: &ProbeReport,
    after: &str,
) -> E2eResult<()> {
    let gpu = tsv_section_single_row(after, "gpu_execution")?;
    let pools = tsv_section(after, "gpu_execution_pools")?;
    let governors = tsv_section(after, "memory_governor_reports")?;
    let telemetry = tsv_section(after, "device_telemetry")?;
    let device_peaks = pools
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary pool")?,
                "worker_count": parse_u64_field(row, "worker_count", "summary pool")?,
                "peak_active_query_jobs": parse_u64_field(row, "peak_active_query_jobs", "summary pool")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let capacities = governors
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary governor")?,
                "capacity_bytes": parse_u64_field(row, "capacity_bytes", "summary governor")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let device_totals = telemetry
        .iter()
        .map(|row| {
            Ok(json!({
                "device_ordinal": parse_usize_field(row, "device_ordinal", "summary telemetry")?,
                "cuda_total_bytes": parse_u64_field(row, "cuda_total_bytes", "summary telemetry")?,
                "nvml_total_bytes": parse_u64_field(row, "nvml_total_bytes", "summary telemetry")?,
            }))
        })
        .collect::<E2eResult<Vec<_>>>()?;
    let observed_devices = successful.iter().map(|row| row.device_ordinal).collect::<BTreeSet<_>>();
    let cache_by_device = params
        .devices
        .iter()
        .map(|device| {
            let rows = successful.iter().filter(|row| row.device_ordinal == *device);
            let (hits, misses, inserts) = rows.fold((0_u64, 0_u64, 0_u64), |totals, row| {
                (
                    totals.0.saturating_add(row.source_cache_hits),
                    totals.1.saturating_add(row.source_cache_misses),
                    totals.2.saturating_add(row.source_cache_inserts),
                )
            });
            json!({
                "device_ordinal": device,
                "hits": hits,
                "misses": misses,
                "inserts": inserts,
            })
        })
        .collect::<Vec<_>>();
    let summary = json!({
        "status": "passed",
        "topology": {
            "tokio_worker_threads": 1,
            "device_ordinals": params.devices,
            "max_concurrent_queries": MAX_CONCURRENT_QUERIES,
            "max_queries_per_device": MAX_QUERIES_PER_DEVICE,
        },
        "queries": {
            "successful": successful.len(),
            "cancelled": churn.cancelled.len(),
            "dropped": churn.dropped.len(),
            "observed_devices": observed_devices,
            "all_exact_parity": true,
            "all_native": true,
            "all_positive_d2h": true,
        },
        "liveness": {
            "probe_pairs": probes.rows.len(),
            "saturated_probe_pairs": probes.saturated_count,
            "saturation_floor": SATURATION_FLOOR,
            "timeout_ms": PROBE_TIMEOUT.as_millis(),
            "peak_observed_active": probes.peak_score,
            "statement_tickets_issued": probes.tickets_issued,
            "statement_tickets_retired": probes.tickets_retired,
        },
        "gpu_execution": {
            "peak_active_query_jobs": parse_u64_field(&gpu, "peak_active_query_jobs", "summary GPU")?,
            "pools": device_peaks,
        },
        "memory_governors": capacities,
        "decoded_source_cache": {
            "configured_max_retained_bytes_per_device": 8_u64 * 1024 * 1024 * 1024,
            "rest_host_buffer_identity": "query_scoped_misses_allowed",
            "devices": cache_by_device,
        },
        "telemetry": device_totals,
        "allocation_failures": 0,
        "final_state_idle": true,
    });
    fs::write(params.artifact_dir.join("summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
