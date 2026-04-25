use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    net::TcpListener,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::table_format::iceberg::catalog::RestIcebergCatalogConfig;
use tokio::process::{Child, Command};
use tonic::transport::Channel;
use url::Url;

use super::E2eResult;
use crate::connection_retry::{BeforeConnect, ConnectionRetryConfig, connect_with_retry};

const OPT_IN_ENV: &str = "NEXUS_ICEBERG_SERVER_REST_MULTIGPU_E2E";
const SERVER_BINARY_ENV: &str = "NEXUS_SERVER_BINARY";
const DEVICE_LIST_ENV: &str = "NEXUS_PRODUCTION_ACCEPTANCE_GPU_DEVICES";
const ARTIFACT_ROOT_ENV: &str = "NEXUS_PRODUCTION_ACCEPTANCE_ARTIFACT_ROOT";
const EXPECTED_DEVICE_COUNT: usize = 4;

pub(super) const MAX_CONCURRENT_QUERIES: usize = 16;
pub(super) const MAX_QUERIES_PER_DEVICE: usize = 4;

#[derive(Clone)]
pub(super) struct Params {
    pub(super) server_binary: PathBuf,
    pub(super) server_addr: std::net::SocketAddr,
    pub(super) server_url: String,
    pub(super) rest_uri: String,
    pub(super) warehouse: String,
    pub(super) s3_endpoint: String,
    pub(super) s3_region: String,
    pub(super) s3_access_key: String,
    pub(super) s3_secret_key: String,
    pub(super) namespace: String,
    pub(super) devices: Vec<usize>,
    pub(super) artifact_dir: PathBuf,
}

impl Params {
    pub(super) fn from_env() -> E2eResult<Self> {
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

    pub(super) fn rest_config(&self) -> RestIcebergCatalogConfig {
        RestIcebergCatalogConfig::new("rest", self.rest_uri.clone(), self.warehouse.clone())
            .with_s3_endpoint(self.s3_endpoint.clone())
            .with_s3_region(self.s3_region.clone())
            .with_s3_path_style_access(true)
    }

    pub(super) fn table(&self, name: &str) -> String {
        format!("lake.\"{}\".\"{name}\"", self.namespace)
    }

    pub(super) fn device_csv(&self) -> String {
        self.devices.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
    }
}

pub(super) fn require_opt_in() -> E2eResult<()> {
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

pub(super) struct ServerChild {
    child: Child,
    stopped: bool,
    artifact_dir: Option<PathBuf>,
}

pub(super) struct ServerSpawnConfig {
    pub(super) server_binary: PathBuf,
    pub(super) stdout_path: PathBuf,
    pub(super) stderr_path: PathBuf,
    pub(super) env: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub(super) struct ProcessExit {
    pub(super) success: bool,
    pub(super) code: Option<i32>,
    pub(super) signal: Option<i32>,
    pub(super) display: String,
}

impl ServerChild {
    pub(super) fn spawn(params: &Params) -> E2eResult<Self> {
        let env = BTreeMap::from([
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
        ])
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect();

        eprintln!("starting release Flight SQL server at {} with devices {}", params.server_addr, params.device_csv());
        let mut child = Self::spawn_config(ServerSpawnConfig {
            server_binary: params.server_binary.clone(),
            stdout_path: params.artifact_dir.join("server.stdout.log"),
            stderr_path: params.artifact_dir.join("server.stderr.log"),
            env,
        })?;
        child.artifact_dir = Some(params.artifact_dir.clone());
        Ok(child)
    }

    pub(super) fn spawn_config(config: ServerSpawnConfig) -> E2eResult<Self> {
        ensure_parent(&config.stdout_path)?;
        ensure_parent(&config.stderr_path)?;
        let stdout = File::create(&config.stdout_path)?;
        let stderr = File::create(&config.stderr_path)?;
        let mut command = Command::new(&config.server_binary);
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
        for name in ["NEXUS_ICEBERG_S3_SESSION_TOKEN", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"] {
            command.env_remove(name);
        }
        command.envs(config.env);
        let child = command.spawn()?;
        Ok(Self { child, stopped: false, artifact_dir: None })
    }

    pub(super) async fn connect(&mut self, url: &str) -> E2eResult<Channel> {
        self.connect_with_timeout(url, Duration::from_secs(120)).await
    }

    pub(super) async fn connect_with_timeout(&mut self, url: &str, timeout: Duration) -> E2eResult<Channel> {
        connect_with_retry(
            url,
            ConnectionRetryConfig::fixed_timeout(timeout, Duration::from_millis(100)),
            || {
                Ok(match self.child.try_wait()? {
                    Some(status) => BeforeConnect::Abort(format!(
                        "release server exited before accepting Flight SQL connections: {status}"
                    )),
                    None => BeforeConnect::Continue,
                })
            },
            |channel| channel,
            |error| format!("release server did not accept connections at {url}: {error:?}"),
        )
        .await
    }

    pub(super) async fn shutdown(&mut self) -> E2eResult<()> {
        let exit = self.terminate("TERM", Duration::from_secs(30)).await?;
        if let Some(artifact_dir) = &self.artifact_dir {
            std::fs::write(artifact_dir.join("server.exit.txt"), format!("{}\n", exit.display))?;
        }
        if !exit.success {
            return Err(format!("release server exited unsuccessfully after SIGTERM: {}", exit.display).into());
        }
        Ok(())
    }

    pub(super) async fn terminate(&mut self, signal: &str, timeout: Duration) -> E2eResult<ProcessExit> {
        let Some(pid) = self.child.id() else {
            return Err("release server child has no process id".into());
        };
        let signal_status =
            std::process::Command::new("kill").args([format!("-{signal}"), pid.to_string()]).status()?;
        if !signal_status.success() {
            return Err(format!("failed to send SIG{signal} to release server pid {pid}: {signal_status}").into());
        }
        let status = tokio::time::timeout(timeout, self.child.wait())
            .await
            .map_err(|_| format!("release server pid {pid} did not exit within {} seconds", timeout.as_secs()))??;
        self.stopped = true;
        let exit = process_exit(status);
        Ok(exit)
    }

    pub(super) fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub(super) fn try_exit(&mut self) -> E2eResult<Option<ProcessExit>> {
        Ok(self.child.try_wait()?.map(process_exit))
    }

    pub(super) fn require_running(&mut self, context: &str) -> E2eResult<()> {
        if let Some(exit) = self.try_exit()? {
            return Err(format!("{context}: server exited unexpectedly: {}", exit.display).into());
        }
        Ok(())
    }
}

fn ensure_parent(path: &Path) -> E2eResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn process_exit(status: std::process::ExitStatus) -> ProcessExit {
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;
    ProcessExit {
        success: status.success(),
        code: status.code(),
        #[cfg(unix)]
        signal: status.signal(),
        #[cfg(not(unix))]
        signal: None,
        display: status.to_string(),
    }
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.child.start_kill();
        }
    }
}
