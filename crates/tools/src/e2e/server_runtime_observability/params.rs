use std::{collections::BTreeSet, env, path::PathBuf, time::Duration};

use super::E2eResult;

const DEFAULT_FIXTURE_ROOT: &str = "/opt/dlami/nvme/fixture/data";
const DEFAULT_SERVER_BINARY: &str = "target/release/algeon_server";
const DEFAULT_EVIDENCE_PATH: &str = "target/runtime_observability_e2e/evidence.json";
const DEFAULT_SERVER_LOG_PATH: &str = "target/runtime_observability_e2e/server";
pub(super) const EVIDENCE_SCHEMA: &str = "algeon_datafusion.server_runtime_observability_e2e";
pub(super) const TABLE_NAME: &str = "runtime_observability_lineitem";
pub(super) const ORDERS_TABLE_NAME: &str = "runtime_observability_orders";
pub(super) const OBS_RECORDS_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS";
pub(super) const OBS_QUERY_BYTES_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES";
pub(super) const OBS_RETAINED_COUNT_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT";
pub(super) const OBS_RETAINED_BYTES_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES";
pub(super) const OBS_TTL_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS";
pub(super) const OBS_GRACE_ENV: &str = "ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS";

pub(super) fn print_usage() {
    eprintln!(
        "server-runtime-observability-e2e\n\
         \n\
         OPTIONS:\n\
           --catalog-path PATH          Directory containing lineitem.parquet and orders.parquet\n\
                                        (default: <fixture-root>/tpch/sf1/parquet)\n\
           --fixture-root PATH          Root of the default catalog path (default: {DEFAULT_FIXTURE_ROOT})\n\
           --server-binary PATH         Built server executable (default: {DEFAULT_SERVER_BINARY})\n\
           --device ORDINAL             CUDA device ordinal (default: 0)\n\
           --concurrency N              Overlapping queries, host gate, and per-device limit (default: 4)\n\
           --waves N                    Native success waves (default: 2)\n\
           --restart-iterations N       Hard termination/restart iterations (default: 1)\n\
           --scenarios matrix|native|datafusion_baseline\n\
           --evidence-json PATH         Machine-readable evidence output (default: {DEFAULT_EVIDENCE_PATH})\n\
           --server-log PATH            Prefix for per-process stdout/stderr logs (default: {DEFAULT_SERVER_LOG_PATH})\n\
           --connect-timeout-secs N\n\
           --query-timeout-secs N\n\
           --poll-timeout-secs N\n\
           --shutdown-timeout-secs N\n\
           --per-query-records N\n\
           --per-query-bytes N\n\
           --retained-count N\n\
           --retained-bytes N\n\
           --retention-ttl-secs N\n\
           --shutdown-grace-secs N\n\
           --native-once                One short overlapping native wave, then exit"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScenarioSelection {
    Matrix,
    Native,
    DataFusionBaseline,
}

impl ScenarioSelection {
    pub(super) fn parse(raw: &str) -> E2eResult<Self> {
        match raw {
            "matrix" => Ok(Self::Matrix),
            "native" => Ok(Self::Native),
            "datafusion_baseline" => Ok(Self::DataFusionBaseline),
            _ => Err(format!("--scenarios must be matrix, native, or datafusion_baseline; got {raw:?}").into()),
        }
    }

    pub(super) fn runs_native(self) -> bool {
        matches!(self, Self::Matrix | Self::Native)
    }

    pub(super) fn runs_datafusion_baseline(self) -> bool {
        matches!(self, Self::Matrix | Self::DataFusionBaseline)
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Matrix => "matrix",
            Self::Native => "native",
            Self::DataFusionBaseline => "datafusion_baseline",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ObservabilityLimits {
    pub(super) per_query_records: u64,
    pub(super) per_query_bytes: u64,
    pub(super) retained_count: usize,
    pub(super) retained_bytes: u64,
    pub(super) ttl_secs: u64,
    pub(super) shutdown_grace_secs: u64,
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeParams {
    pub(super) fixture_root: PathBuf,
    pub(super) catalog_path: PathBuf,
    pub(super) server_binary: PathBuf,
    pub(super) device: usize,
    pub(super) concurrency: usize,
    pub(super) waves: usize,
    pub(super) restart_iterations: usize,
    pub(super) scenario_selection: ScenarioSelection,
    pub(super) evidence_path: PathBuf,
    pub(super) server_log_path: PathBuf,
    pub(super) connect_timeout: Duration,
    pub(super) query_timeout: Duration,
    pub(super) poll_timeout: Duration,
    pub(super) shutdown_timeout: Duration,
    pub(super) native_once: bool,
    pub(super) observability: ObservabilityLimits,
}

impl RuntimeParams {
    pub(super) fn parse(args: Vec<String>) -> E2eResult<Self> {
        let mut fixture_root = PathBuf::from(DEFAULT_FIXTURE_ROOT);
        let mut catalog_path = None;
        let mut server_binary = PathBuf::from(DEFAULT_SERVER_BINARY);
        let mut device = 0usize;
        let mut concurrency = 4usize;
        let mut waves = 2usize;
        let mut restart_iterations = 1usize;
        let mut scenario_selection = ScenarioSelection::Matrix;
        let mut evidence_path = PathBuf::from(DEFAULT_EVIDENCE_PATH);
        let mut server_log_path = PathBuf::from(DEFAULT_SERVER_LOG_PATH);
        let mut connect_timeout = Duration::from_secs(60);
        let mut query_timeout = Duration::from_secs(180);
        let mut poll_timeout = Duration::from_secs(30);
        let mut shutdown_timeout = Duration::from_secs(30);
        let mut native_once = false;
        let mut per_query_records = None;
        let mut per_query_bytes = None;
        let mut retained_count = None;
        let mut retained_bytes = None;
        let mut ttl_secs = None;
        let mut shutdown_grace_secs = None;
        let mut seen = BTreeSet::new();
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            if !seen.insert(flag.to_string()) {
                return Err(format!("duplicate option {flag}").into());
            }
            if flag == "--native-once" {
                native_once = true;
                index += 1;
                continue;
            }
            let value = args.get(index + 1).ok_or_else(|| format!("missing value for {flag}"))?;
            match flag {
                "--fixture-root" => fixture_root = PathBuf::from(value),
                "--catalog-path" => catalog_path = Some(PathBuf::from(value)),
                "--server-binary" => server_binary = PathBuf::from(value),
                "--device" => device = parse_usize(flag, value)?,
                "--concurrency" => concurrency = parse_usize(flag, value)?,
                "--waves" => waves = parse_usize(flag, value)?,
                "--restart-iterations" => restart_iterations = parse_usize(flag, value)?,
                "--scenarios" => scenario_selection = ScenarioSelection::parse(value)?,
                "--evidence-json" => evidence_path = PathBuf::from(value),
                "--server-log" => server_log_path = PathBuf::from(value),
                "--connect-timeout-secs" => connect_timeout = positive_duration(flag, value)?,
                "--query-timeout-secs" => query_timeout = positive_duration(flag, value)?,
                "--poll-timeout-secs" => poll_timeout = positive_duration(flag, value)?,
                "--shutdown-timeout-secs" => shutdown_timeout = positive_duration(flag, value)?,
                "--per-query-records" => per_query_records = Some(positive_u64(flag, value)?),
                "--per-query-bytes" => per_query_bytes = Some(positive_u64(flag, value)?),
                "--retained-count" => retained_count = Some(positive_usize(flag, value)?),
                "--retained-bytes" => retained_bytes = Some(positive_u64(flag, value)?),
                "--retention-ttl-secs" => ttl_secs = Some(positive_u64(flag, value)?),
                "--shutdown-grace-secs" => shutdown_grace_secs = Some(positive_u64(flag, value)?),
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        let catalog_path = catalog_path.unwrap_or_else(|| fixture_root.join("tpch/sf1/parquet"));
        let observability = ObservabilityLimits {
            per_query_records: cli_or_env_u64(per_query_records, OBS_RECORDS_ENV, 256)?,
            per_query_bytes: cli_or_env_u64(per_query_bytes, OBS_QUERY_BYTES_ENV, 65_536)?,
            retained_count: cli_or_env_usize(retained_count, OBS_RETAINED_COUNT_ENV, 4)?,
            retained_bytes: cli_or_env_u64(retained_bytes, OBS_RETAINED_BYTES_ENV, 1_048_576)?,
            ttl_secs: cli_or_env_u64(ttl_secs, OBS_TTL_ENV, 2)?,
            shutdown_grace_secs: cli_or_env_u64(shutdown_grace_secs, OBS_GRACE_ENV, 2)?,
        };
        if !(2..=64).contains(&concurrency) {
            return Err("--concurrency must be between 2 and 64 so overlap can be proven".into());
        }
        if !(1..=100).contains(&waves) {
            return Err("--waves must be between 1 and 100".into());
        }
        if restart_iterations > 20 {
            return Err("--restart-iterations must be between 0 and 20".into());
        }
        if observability.retained_count < concurrency {
            return Err(format!(
                "--retained-count ({}) must be at least --concurrency ({concurrency})",
                observability.retained_count
            )
            .into());
        }
        if observability.retained_bytes < observability.per_query_bytes {
            return Err("--retained-bytes must be at least --per-query-bytes".into());
        }
        if scenario_selection.runs_native() && !native_once && observability.retained_bytes > 2 * 1024 * 1024 {
            return Err(
                "--retained-bytes must not exceed 2 MiB for the explicit serialized-byte pressure scenario".into()
            );
        }
        if native_once && !scenario_selection.runs_native() {
            return Err("--native-once requires --scenarios native or matrix".into());
        }
        if evidence_path == server_log_path {
            return Err("--evidence-json and --server-log must be different paths".into());
        }
        Ok(Self {
            fixture_root,
            catalog_path,
            server_binary,
            device,
            concurrency,
            waves,
            restart_iterations,
            scenario_selection,
            evidence_path,
            server_log_path,
            connect_timeout,
            query_timeout,
            poll_timeout,
            shutdown_timeout,
            native_once,
            observability,
        })
    }

    pub(super) fn validate_filesystem(&self) -> E2eResult<()> {
        if !self.server_binary.is_file() {
            return Err(format!("server binary does not exist: {}", self.server_binary.display()).into());
        }
        let lineitem = self.lineitem_path();
        if !lineitem.is_file() {
            return Err(format!("catalog path must contain lineitem.parquet: {}", lineitem.display()).into());
        }
        let orders = self.orders_path();
        if !orders.is_file() {
            return Err(format!("catalog path must contain orders.parquet: {}", orders.display()).into());
        }
        Ok(())
    }

    pub(super) fn lineitem_path(&self) -> PathBuf {
        self.catalog_path.join("lineitem.parquet")
    }

    pub(super) fn orders_path(&self) -> PathBuf {
        self.catalog_path.join("orders.parquet")
    }
}

fn parse_usize(flag: &str, raw: &str) -> E2eResult<usize> {
    raw.parse::<usize>().map_err(|error| format!("invalid {flag}={raw:?}: {error}").into())
}

fn positive_usize(flag: &str, raw: &str) -> E2eResult<usize> {
    let value = parse_usize(flag, raw)?;
    if value == 0 { Err(format!("{flag} must be greater than zero").into()) } else { Ok(value) }
}

fn positive_u64(flag: &str, raw: &str) -> E2eResult<u64> {
    let value = raw.parse::<u64>().map_err(|error| format!("invalid {flag}={raw:?}: {error}"))?;
    if value == 0 { Err(format!("{flag} must be greater than zero").into()) } else { Ok(value) }
}

fn positive_duration(flag: &str, raw: &str) -> E2eResult<Duration> {
    Ok(Duration::from_secs(positive_u64(flag, raw)?))
}

fn cli_or_env_u64(cli: Option<u64>, name: &str, default: u64) -> E2eResult<u64> {
    match cli {
        Some(value) => Ok(value),
        None => Ok(env::var(name).ok().map(|raw| positive_u64(name, &raw)).transpose()?.unwrap_or(default)),
    }
}

fn cli_or_env_usize(cli: Option<usize>, name: &str, default: usize) -> E2eResult<usize> {
    match cli {
        Some(value) => Ok(value),
        None => Ok(env::var(name).ok().map(|raw| positive_usize(name, &raw)).transpose()?.unwrap_or(default)),
    }
}
