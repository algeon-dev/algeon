use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) enum ChildProfile {
    Native,
    CpuFallback,
}

impl ChildProfile {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

pub(super) struct RunningServer {
    pub(super) label: String,
    pub(super) profile: ChildProfile,
    pub(super) pid: Option<u32>,
    pub(super) stdout_log: PathBuf,
    pub(super) stderr_log: PathBuf,
    pub(super) started: Instant,
    pub(super) child: ServerChild,
    pub(super) channel: Channel,
}

pub(super) async fn spawn_server(
    params: &RuntimeParams,
    evidence: &mut RunEvidence,
    label: &str,
    profile: ChildProfile,
) -> E2eResult<RunningServer> {
    let addr = reserve_loopback_addr()?;
    let stdout_log = suffixed_log_path(&params.server_log_path, label, "stdout");
    let stderr_log = suffixed_log_path(&params.server_log_path, label, "stderr");
    let mut env = BTreeMap::from([
        ("TOKIO_WORKER_THREADS".to_string(), "1".to_string()),
        ("NEXUS_SERVER_BIND".to_string(), addr.to_string()),
        ("NEXUS_SERVER_LOG".to_string(), "info".to_string()),
        ("NEXUS_SERVER_GPU_DEVICES".to_string(), params.device.to_string()),
        ("NEXUS_SERVER_MAX_CONCURRENT_QUERIES".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY".to_string(), "least_loaded".to_string()),
        ("NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS".to_string(), params.query_timeout.as_millis().to_string()),
        ("NEXUS_SERVER_SESSION_PROFILE".to_string(), profile.label().to_string()),
        ("NEXUS_SERVER_NATIVE_MEMORY_POLICY".to_string(), "bounded".to_string()),
        ("NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE".to_string(), params.bounded_mode.clone()),
        ("NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS".to_string(), params.concurrency.to_string()),
        ("NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS".to_string(), params.query_timeout.as_millis().to_string()),
        ("NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS".to_string(), params.poll_timeout.as_secs().to_string()),
        (OBS_RECORDS_ENV.to_string(), params.observability.per_query_records.to_string()),
        (OBS_QUERY_BYTES_ENV.to_string(), params.observability.per_query_bytes.to_string()),
        (OBS_RETAINED_COUNT_ENV.to_string(), params.observability.retained_count.to_string()),
        (OBS_RETAINED_BYTES_ENV.to_string(), params.observability.retained_bytes.to_string()),
        (OBS_TTL_ENV.to_string(), params.observability.ttl_secs.to_string()),
        (OBS_GRACE_ENV.to_string(), params.observability.shutdown_grace_secs.to_string()),
    ]);
    if matches!(profile, ChildProfile::CpuFallback) {
        env.insert("DATAFUSION_EXECUTION_TARGET_PARTITIONS".to_string(), "1".to_string());
    }
    let started = Instant::now();
    let mut child = ServerChild::spawn_config(ServerSpawnConfig {
        server_binary: params.server_binary.clone(),
        stdout_path: stdout_log.clone(),
        stderr_path: stderr_log.clone(),
        env,
    })?;
    let pid = child.process_id();
    evidence.server_logs.push(stdout_log.clone());
    evidence.server_logs.push(stderr_log.clone());
    let url = format!("http://{addr}");
    let channel = child.connect_with_timeout(&url, params.connect_timeout).await?;
    child.require_running(label)?;
    Ok(RunningServer { label: label.to_string(), profile, pid, stdout_log, stderr_log, started, child, channel })
}

pub(super) async fn terminate_server(
    params: &RuntimeParams,
    server: &mut RunningServer,
    signal: &str,
    require_success: bool,
) -> E2eResult<ProcessEvidence> {
    let exit = server.child.terminate(signal, params.shutdown_timeout).await?;
    if require_success && !exit.success {
        return Err(format!("{} did not exit successfully after SIG{signal}: {}", server.label, exit.display).into());
    }
    Ok(process_evidence(server, exit))
}

pub(super) fn process_evidence(server: &RunningServer, exit: ProcessExit) -> ProcessEvidence {
    ProcessEvidence {
        label: server.label.clone(),
        pid: server.pid,
        profile: server.profile.label().to_string(),
        stdout_log: server.stdout_log.clone(),
        stderr_log: server.stderr_log.clone(),
        exit_success: exit.success,
        exit_code: exit.code,
        exit_signal: exit.signal,
        exit_display: exit.display,
        elapsed_ms: millis(server.started.elapsed()),
    }
}

pub(super) fn reserve_loopback_addr() -> E2eResult<std::net::SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

pub(super) fn suffixed_log_path(base: &Path, label: &str, stream: &str) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new(""));
    let name = base.file_name().and_then(|name| name.to_str()).unwrap_or("server");
    parent.join(format!("{name}.{label}.{stream}.log"))
}

pub(super) async fn setup_local_table(params: &RuntimeParams, channel: &Channel, label: &str) -> E2eResult<()> {
    let mut client = correlated_client(channel, &format!("setup-{label}"));
    for (table, path) in [(TABLE_NAME, params.lineitem_path()), (ORDERS_TABLE_NAME, params.orders_path())] {
        let sql = format!("CREATE EXTERNAL TABLE {table} STORED AS PARQUET LOCATION '{}'", sql_path(&path)?);
        tokio::time::timeout(params.query_timeout, client.execute_update(sql, None))
            .await
            .map_err(|_| format!("{label} table setup exceeded {} seconds", params.query_timeout.as_secs()))??;
    }
    Ok(())
}

pub(super) fn sql_path(path: &Path) -> E2eResult<String> {
    let raw = path.to_str().ok_or_else(|| format!("fixture path is not UTF-8: {}", path.display()))?;
    Ok(raw.replace('\'', "''"))
}

pub(super) fn correlated_client(channel: &Channel, correlation_id: &str) -> FlightSqlServiceClient<Channel> {
    let mut client = FlightSqlServiceClient::new(channel.clone());
    client.set_header(CORRELATION_HEADER, correlation_id);
    client
}

pub(super) async fn wait_for_idle_with_timeout(params: &RuntimeParams, channel: &Channel) -> E2eResult<String> {
    tokio::time::timeout(params.poll_timeout, wait_for_idle(channel))
        .await
        .map_err(|_| format!("server did not become idle within {:?}", params.poll_timeout))?
}
