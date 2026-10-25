use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use query_planning::{execution_contracts::SourceChunkPlanningMode, mode::StatisticalAggregateExecutionMode};
use query_runtime::runtime::SourceSchedulerPolicy;
use serde::Deserialize;
use tonic::transport::ServerTlsConfig;

use super::{
    auth::{AuthConfig, AuthMode},
    error::{Result, ServerError},
};
use crate::server::{
    backend::{AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuMemoryOwnership},
    coordinator::QueryCoordinatorConfig,
    cugraph_sql::{CugraphGraphConstructionPolicy, CugraphGraphOptions},
    cuml_sql::CumlModelStoreConfig,
    native_exec::{AlgeonNativeExecutionConfig, S3SourceStorageProfile, SourceChunking},
    planner::AlgeonNativeOptimizerConfig,
};

mod admission;
mod cugraph;
mod data;
mod effective;
mod env;
mod file;
mod iceberg;
mod models;
mod native;
mod observability;
mod s3;
mod tls;

use admission::{server_device_profiles, server_queue_policy};
pub use cugraph::CugraphServerConfig;
use cugraph::cugraph_config;
use data::data_root;
pub(crate) use effective::non_default_settings;
pub use env::{ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_LOG_ENV};
use env::{EnvOverride, ServerEnvironment};
use file::{ServerFile, ServerTlsFileConfig, load_file};
use iceberg::iceberg_catalogs;
pub use iceberg::{
    ServerIcebergCatalogConfig, ServerIcebergCatalogKind, ServerIcebergGlueCatalogConfig, ServerIcebergRestAuth,
    ServerIcebergRestCatalogConfig, ServerIcebergSecret,
};
use models::models_config;
use native::native_optimizer_config;
pub use observability::ServerObservabilityConfig;
use observability::server_observability_config;
use s3::s3_storage;
pub(super) use tls::install_rustls_crypto_provider;
use tls::tls_config;

const DEFAULT_BIND_ADDR: &str = "127.0.0.1:50051";
const DEFAULT_STATEMENT_TICKET_TTL_SECS: u64 = 300;

/// HTTP/2 transport loss detection. Acknowledging clients may retain unread results.
#[derive(Debug, Clone, Copy)]
pub struct Http2Keepalive {
    /// Time without inbound traffic before a ping is sent.
    pub interval: Duration,
    /// Time to acknowledge a ping before the connection closes.
    pub timeout: Duration,
}

impl Default for Http2Keepalive {
    fn default() -> Self {
        Self { interval: Duration::from_secs(30), timeout: Duration::from_secs(10) }
    }
}

#[derive(Debug, Clone)]
/// Flight SQL deployment configuration for transport, GPU resources, and session features.
///
/// Loading from TOML and the environment validates settings without claiming
/// GPU ownership. The launcher or embedding application supplies
/// [`Self::gpu_memory_ownership`] before backend construction.
pub struct ServerConfig {
    /// Address bound when no prebound listener is injected.
    pub bind_addr: SocketAddr,
    /// Queue-only admission policy. Device-local bounds live in
    /// `gpu_device_profiles` and are never duplicated here.
    pub queue_policy: AlgeonGpuQueuePolicy,
    /// One resource profile per selected GPU, in selection order. The
    /// profile ordinals are the selected GPUs; see [`Self::gpu_devices`].
    /// The server never applies one scalar admission setting to several GPUs.
    pub gpu_device_profiles: Vec<AlgeonGpuDeviceProfile>,
    /// Exact deployment ownership declaration consumed when a bounded GPU
    /// backend is constructed. Configuration loading leaves this empty; the
    /// embedding or launch profile makes the ownership promise.
    pub gpu_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    /// Wall-clock time a planned Flight SQL statement ticket may wait for
    /// `DoGet`. Expired tickets are dropped; GPU admission is acquired later
    /// by native plan execution.
    pub statement_ticket_ttl: Duration,
    /// Keepalive intervals for accepted HTTP/2 connections.
    pub http2_keepalive: Http2Keepalive,
    /// Bounded native runtime-artifact capture and sealed retention policy.
    pub observability: ServerObservabilityConfig,
    /// Native optimizer and execution settings installed in server sessions.
    pub native_optimizer_config: AlgeonNativeOptimizerConfig,
    /// Authentication mode applied to incoming Flight SQL RPCs, from the
    /// `[auth]` table and disabled by default.
    pub auth: AuthConfig,
    /// cuGraph SQL registration and staging settings; `None` leaves its SQL family unregistered.
    pub cugraph: Option<CugraphServerConfig>,
    /// Models root for `cuml_forest_predict`, from the `[models]` table.
    /// `None` leaves the function unavailable with `model_root_unconfigured`.
    pub models: Option<CumlModelStoreConfig>,
    /// Directory of Parquet data, from `[data] root`. Loading resolves it to
    /// an absolute path and requires a directory.
    /// [`FlightSqlServerBuilder::build`] registers its first-level entries as
    /// tables in the shared catalog before it runs [`Self::init_sql`]:
    ///
    /// - A file with the extension `parquet`, in any case, becomes a table
    ///   named by the file stem.
    /// - A directory becomes a table named by the directory, read like
    ///   `CREATE EXTERNAL TABLE <name> STORED AS PARQUET LOCATION '<dir>/'`,
    ///   when that location lists at least one Parquet file.
    /// - Names starting with `.` or `_`, other files, and directories
    ///   without Parquet files are skipped.
    ///
    /// Table names keep the case and characters of the entry name. Entries
    /// that cannot be inspected, names that are not UTF-8, and directories
    /// whose listing fails before a Parquet file is found are skipped with a
    /// warning. Two entries with the same table name, or a Parquet entry whose
    /// schema cannot be read or that cannot be registered, fail the build with
    /// `invalid_configuration` naming the paths. `None` registers nothing.
    ///
    /// [`FlightSqlServerBuilder::build`]: crate::server::FlightSqlServerBuilder::build
    pub data_root: Option<PathBuf>,
    /// TLS for the Flight SQL listener, from the `[tls]` table. `None` serves
    /// plaintext. The value must carry an identity, because tonic panics when
    /// it builds a listener from a config without one. A client CA makes the
    /// listener require and verify client certificates.
    /// [`FlightSqlServerBuilder::build`] rejects unparsable PEM or a key that
    /// does not match the certificate with `invalid_configuration` before it
    /// builds a GPU backend.
    ///
    /// [`FlightSqlServerBuilder::build`]: crate::server::FlightSqlServerBuilder::build
    pub tls: Option<ServerTlsConfig>,
    /// SQL file whose DDL statements [`FlightSqlServerBuilder::build`] runs on
    /// the shared session, one statement at a time, before the server
    /// accepts connections. Any statement that is not DDL, or that fails,
    /// fails the build with `invalid_configuration`.
    ///
    /// [`FlightSqlServerBuilder::build`]: crate::server::FlightSqlServerBuilder::build
    pub init_sql: Option<PathBuf>,
    /// Iceberg catalogs from the `[[iceberg.catalogs]]` array, in file order.
    /// Loading requires `[s3]` when any entry is present and rejects
    /// duplicate names and the name `datafusion`.
    pub iceberg_catalogs: Vec<ServerIcebergCatalogConfig>,
}

impl ServerConfig {
    /// Attach the deployment ownership promise required by bounded backend
    /// construction. The backend validates this declaration against
    /// [`Self::gpu_devices`] before acquiring any device lease.
    pub fn with_gpu_memory_ownership(
        mut self,
        ownership: impl IntoIterator<Item = (usize, GpuMemoryOwnership)>,
    ) -> Self {
        self.gpu_memory_ownership = ownership.into_iter().collect();
        self
    }

    /// The `[s3]` storage every `s3://` source reads through, or `None`
    /// without the table.
    pub fn s3_storage(&self) -> Option<&S3SourceStorageProfile> {
        self.native_optimizer_config.native_execution_config().s3_storage()
    }

    /// The selected CUDA ordinals: the ordinals of
    /// [`Self::gpu_device_profiles`], in profile order.
    pub fn gpu_devices(&self) -> Vec<usize> {
        self.gpu_device_profiles.iter().map(AlgeonGpuDeviceProfile::device_ordinal).collect()
    }

    /// Loads the configuration from the process environment and the TOML
    /// file it names; see [`LoadedServerConfig::from_vars`].
    pub fn from_env() -> Result<Self> {
        LoadedServerConfig::from_env().map(|loaded| loaded.config)
    }

    /// Loads the configuration from `vars` in place of the process
    /// environment; see [`LoadedServerConfig::from_vars`].
    pub fn from_vars<K, V>(vars: impl IntoIterator<Item = (K, V)>) -> Result<Self>
    where
        K: Into<String>,
        V: Into<String>,
    {
        LoadedServerConfig::from_vars(vars).map(|loaded| loaded.config)
    }

    fn from_file(file: ServerFile, sources: &ConfigSources) -> Result<(Self, bool)> {
        let bind_addr = parse_bind_addr(file.bind.as_deref().unwrap_or(DEFAULT_BIND_ADDR), sources)?;
        let statement_ticket_ttl = positive_secs(
            sources,
            "statement_ticket_ttl_secs",
            file.statement_ticket_ttl_secs.unwrap_or(DEFAULT_STATEMENT_TICKET_TTL_SECS),
        )?;
        let keepalive_defaults = Http2Keepalive::default();
        let http2_keepalive = Http2Keepalive {
            interval: positive_secs(
                sources,
                "http2_keepalive_interval_secs",
                file.http2_keepalive_interval_secs.unwrap_or(keepalive_defaults.interval.as_secs()),
            )?,
            timeout: positive_secs(
                sources,
                "http2_keepalive_timeout_secs",
                file.http2_keepalive_timeout_secs.unwrap_or(keepalive_defaults.timeout.as_secs()),
            )?,
        };
        let queue_policy = server_queue_policy(&file.admission, sources)?;
        let (gpu_device_profiles, device_profiles_defaulted) = server_device_profiles(&file.admission)?;
        let observability = server_observability_config(&file.observability, sources)?;
        let s3_storage = s3_storage(file.s3.as_ref(), sources)?;
        let iceberg_catalogs = iceberg_catalogs(file.iceberg, s3_storage.as_ref(), sources)?;
        let native_optimizer_config = native_optimizer_config(&file.native, s3_storage, sources)?;
        let cugraph = cugraph_config(&file.cugraph, sources)?;
        let models = models_config(&file.models, sources)?;
        let data_root = data_root(&file.data, sources)?;
        let tls = tls_config(file.tls.as_ref(), sources)?;
        let init_sql = file.init_sql.map(|path| readable_init_sql(path, sources)).transpose()?;
        file.auth.mode.validate()?;

        let config = Self {
            bind_addr,
            queue_policy,
            gpu_device_profiles,
            gpu_memory_ownership: Vec::new(),
            statement_ticket_ttl,
            http2_keepalive,
            observability,
            native_optimizer_config,
            auth: file.auth,
            cugraph,
            models,
            data_root,
            tls,
            init_sql,
            iceberg_catalogs,
        };
        config.validate_gpu_device_profiles()?;
        config.check_admission_within_live_query_capacity(sources)?;
        // Reject a default capture budget that overflows at load time.
        config.active_logical_capture_budget_bytes()?;
        Ok((config, device_profiles_defaulted))
    }

    /// Logical capture budget of the query coordinator under this config's
    /// admission bounds; see [`ServerObservabilityConfig::active_logical_capture_max_bytes`].
    pub(crate) fn active_logical_capture_budget_bytes(&self) -> Result<u64> {
        self.observability
            .active_logical_capture_budget_bytes(admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles))
    }

    /// Logical capture bytes of every attempt this config's admission bounds
    /// hold at once.
    pub(crate) fn admission_logical_capture_bytes(&self) -> u64 {
        self.observability
            .admission_logical_capture_bytes(admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles))
    }

    /// The coordinator registers at most `MAX_LIVE_QUERIES` live statements of
    /// every kind; admission bounds must stay below that cap. Statements that
    /// hold no admission slot (planning, ticketed, CPU-only, metadata,
    /// finalizing) count against the same cap, so registration can still
    /// refuse a statement while admission has room. Config loading and
    /// `FlightSqlServerBuilder::build` both run this one check.
    pub(crate) fn validate_admission_within_live_query_capacity(&self) -> Result<()> {
        self.check_admission_within_live_query_capacity(&ConfigSources::default())
    }

    fn check_admission_within_live_query_capacity(&self, sources: &ConfigSources) -> Result<()> {
        let admission_live_attempts = admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles);
        let max_live_queries = QueryCoordinatorConfig::MAX_LIVE_QUERIES;
        let capacity = u64::try_from(max_live_queries).unwrap_or(u64::MAX);
        if admission_live_attempts < capacity {
            return Ok(());
        }
        // The active attempts the device profiles allow fill registration on
        // their own, or the queue added to them does.
        let active_attempts = admission_live_attempts
            .saturating_sub(u64::try_from(self.queue_policy.max_queued_attempts()).unwrap_or(u64::MAX));
        let field = if active_attempts >= capacity {
            "admission.device_profiles.max_active_attempts"
        } else {
            "admission.max_queued_attempts"
        };
        Err(sources
            .invalid(field, format!("leaves no room for other statements: the admission bounds hold {admission_live_attempts} attempts at once and the server registers at most {max_live_queries} live statements"))
            .with_fact("reason", "admission_exceeds_live_query_capacity")
            .with_fact("admission_live_attempts", admission_live_attempts)
            .with_fact("max_live_queries", max_live_queries))
    }

    /// Rejects two profiles for one GPU. Loading already applies this check,
    /// but callers can construct `ServerConfig` directly before injecting or
    /// constructing a Flight backend.
    pub(crate) fn validate_gpu_device_profiles(&self) -> Result<()> {
        let mut configured = BTreeSet::new();
        for profile in &self.gpu_device_profiles {
            if !configured.insert(profile.device_ordinal()) {
                return Err(ServerError::invalid_config(
                    "admission.device_profiles",
                    format!("has two entries for GPU {}; give each GPU one entry", profile.device_ordinal()),
                )
                .with_fact("reason", "duplicate_device")
                .with_fact("device_ordinal", profile.device_ordinal()));
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_gpu_devices(&self) -> Result<Vec<AlgeonGpuDeviceConfig>> {
        self.validate_gpu_device_profiles()?;

        let selected = self.gpu_devices().into_iter().collect::<BTreeSet<_>>();
        let mut declared = BTreeMap::new();
        for (ordinal, ownership) in self.gpu_memory_ownership.iter().copied() {
            if declared.insert(ordinal, ownership).is_some() {
                return Err(ServerError::invalid_config(
                    "gpu_memory_ownership",
                    format!("contains duplicate GPU memory ownership declaration for device ordinal {ordinal}"),
                )
                .with_fact("reason", "duplicate_device")
                .with_fact("device_ordinal", ordinal));
            }
        }
        let declared_ordinals = declared.keys().copied().collect::<BTreeSet<_>>();
        let missing = selected.difference(&declared_ordinals).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ServerError::invalid_config(
                "gpu_memory_ownership",
                "must declare GPU memory ownership for every selected device",
            )
            .with_fact("reason", "missing_devices")
            .with_fact("device_ordinals", format_device_ordinals(&missing)));
        }
        let extra = declared_ordinals.difference(&selected).copied().collect::<Vec<_>>();
        if !extra.is_empty() {
            return Err(ServerError::invalid_config(
                "gpu_memory_ownership",
                "must not declare GPU memory ownership for an unselected device",
            )
            .with_fact("reason", "extra_devices")
            .with_fact("device_ordinals", format_device_ordinals(&extra)));
        }

        let devices = self
            .gpu_device_profiles
            .iter()
            .map(|profile| {
                let ownership = declared[&profile.device_ordinal()];
                AlgeonGpuDeviceConfig::new(profile.clone(), ownership)
            })
            .collect();
        Ok(devices)
    }
}

/// Where one configuration setting took its value from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigValueSource {
    /// No file entry and no environment override; the built-in default applies.
    Default,
    /// The TOML file named by `--config` or `ALGEON_SERVER_CONFIG_FILE`.
    File,
    /// The named `ALGEON_SERVER_*` environment variable.
    Env(String),
}

impl fmt::Display for ConfigValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("default"),
            Self::File => f.write_str("file"),
            Self::Env(name) => write!(f, "env {name}"),
        }
    }
}

/// A validated [`ServerConfig`] together with where each setting came from.
///
/// The loader starts from built-in defaults, applies the TOML file named by
/// `ALGEON_SERVER_CONFIG_FILE`, then applies environment overrides. Each
/// scalar TOML leaf has one override named `ALGEON_SERVER_` followed by its
/// path in upper case with `.` written as `_`; `native.execution_mode` is
/// `ALGEON_SERVER_NATIVE_EXECUTION_MODE`. Arrays of tables
/// (`admission.device_profiles`, `auth.mode.users`) are file-only. Any other
/// `ALGEON_SERVER_*` variable fails loading with `invalid_configuration`,
/// as an unknown TOML field does, and the error names the closest known
/// variable.
#[derive(Debug, Clone)]
pub struct LoadedServerConfig {
    /// The validated configuration. Ownership declarations are still empty.
    pub config: ServerConfig,
    sources: ConfigSources,
}

impl LoadedServerConfig {
    /// Loads the configuration from the process environment and the TOML
    /// file it names.
    ///
    /// Returns `invalid_configuration` for malformed values, unknown
    /// `ALGEON_SERVER_*` variables, invalid file content, unreadable TLS or
    /// `init_sql` files, or a non-UTF-8 `ALGEON_SERVER_*` value. TLS identity
    /// parsing and key matching occur at server build.
    pub fn from_env() -> Result<Self> {
        let mut vars = Vec::new();
        for (name, value) in std::env::vars_os() {
            let Some(name) = name.to_str().filter(|name| name.starts_with("ALGEON_SERVER_")) else {
                continue;
            };
            let value = value.into_string().map_err(|_| {
                ServerError::configuration(format!("environment variable {name} must be valid UTF-8"))
                    .with_fact("config_env", name)
            })?;
            vars.push((name.to_string(), value));
        }
        Self::from_vars(vars)
    }

    /// Loads the configuration from `vars`, a complete set of environment
    /// variables, in place of the process environment. Variables without the
    /// `ALGEON_SERVER_` prefix are ignored. File I/O still happens for the
    /// config file, TLS files, the models and data roots, and `init_sql`.
    pub fn from_vars<K, V>(vars: impl IntoIterator<Item = (K, V)>) -> Result<Self>
    where
        K: Into<String>,
        V: Into<String>,
    {
        let environment =
            ServerEnvironment::from_vars(vars.into_iter().map(|(name, value)| (name.into(), value.into())))?;
        let file_table = match &environment.config_file {
            Some(path) => load_file(path)?,
            None => toml::Table::new(),
        };
        let mut merged = file_table.clone();
        environment.apply_to(&mut merged);
        let file = merged.try_into::<ServerFile>().map_err(|error| {
            let detail = error.to_string();
            let names =
                environment.overrides.values().map(|value| value.env_name.as_str()).collect::<Vec<_>>().join(", ");
            ServerError::configuration_with_source(
                format!("the environment overrides {names} leave an incomplete configuration table: {detail}"),
                error,
            )
        })?;
        let mut sources = ConfigSources {
            config_file: environment.config_file,
            file_table,
            env_overrides: environment.overrides,
            tls: file.tls.clone(),
            device_profiles_defaulted: false,
        };
        let (config, device_profiles_defaulted) = ServerConfig::from_file(file, &sources)?;
        sources.device_profiles_defaulted = device_profiles_defaulted;
        if device_profiles_defaulted {
            tracing::info!(
                device_ordinals = %format_device_ordinals(&config.gpu_devices()),
                "no [[admission.device_profiles]] entry; using one default profile for GPU 0"
            );
        }
        Ok(Self { config, sources })
    }

    /// The TOML file the configuration was read from, if any.
    pub fn config_file(&self) -> Option<&Path> {
        self.sources.config_file.as_deref()
    }

    /// Where the setting at the TOML `path`, such as
    /// `native.execution_mode`, took its value from.
    pub fn source(&self, path: &str) -> ConfigValueSource {
        self.sources.source(path)
    }

    /// Whether the file had no `[[admission.device_profiles]]` entry, so the
    /// configuration holds one default profile for GPU 0.
    pub fn device_profiles_defaulted(&self) -> bool {
        self.sources.device_profiles_defaulted
    }

    /// The effective configuration as TOML text, every setting followed by a
    /// `# source: default`, `# source: file` or `# source: env NAME` comment.
    /// Unset optional settings appear as comments. Password hashes are
    /// omitted.
    pub fn effective_toml(&self) -> String {
        effective::render(&self.config, &self.sources)
    }

    pub(crate) fn into_parts(self) -> (ServerConfig, ConfigSources) {
        (self.config, self.sources)
    }
}

/// Provenance of a loaded configuration: the raw file table and the
/// environment overrides. A config built in Rust has empty sources.
#[derive(Debug, Clone, Default)]
pub(crate) struct ConfigSources {
    config_file: Option<PathBuf>,
    file_table: toml::Table,
    env_overrides: BTreeMap<String, EnvOverride>,
    tls: Option<ServerTlsFileConfig>,
    device_profiles_defaulted: bool,
}

impl ConfigSources {
    pub(crate) fn config_file(&self) -> Option<&Path> {
        self.config_file.as_deref()
    }

    pub(crate) fn device_profiles_defaulted(&self) -> bool {
        self.device_profiles_defaulted
    }

    fn source(&self, path: &str) -> ConfigValueSource {
        if let Some(env) = self.env_overrides.get(path) {
            return ConfigValueSource::Env(env.env_name.clone());
        }
        if table_lookup(&self.file_table, path).is_some() {
            return ConfigValueSource::File;
        }
        ConfigValueSource::Default
    }

    /// `invalid_configuration` naming the TOML `path`, and the environment
    /// variable too when an override supplied the value.
    fn invalid(&self, path: &'static str, reason: impl Into<String>) -> ServerError {
        match self.env_overrides.get(path) {
            None => ServerError::invalid_config(path, reason),
            Some(env) => ServerError::configuration(format!(
                "{path} {} (set by environment variable {})",
                reason.into(),
                env.env_name
            ))
            .with_fact("config_field", path)
            .with_fact("config_env", &env.env_name),
        }
    }

    fn invalid_with_source(
        &self,
        path: &'static str,
        reason: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> ServerError {
        match self.env_overrides.get(path) {
            None => ServerError::invalid_config_with_source(path, reason, source),
            Some(env) => ServerError::configuration_with_source(
                format!("{path} {} (set by environment variable {})", reason.into(), env.env_name),
                source,
            )
            .with_fact("config_field", path)
            .with_fact("config_env", &env.env_name),
        }
    }
}

fn table_lookup<'a>(table: &'a toml::Table, path: &str) -> Option<&'a toml::Value> {
    let (parent, key) = match path.rsplit_once('.') {
        Some((parent, key)) => (Some(parent), key),
        None => (None, path),
    };
    let table = match parent {
        Some(parent) => table_lookup(table, parent)?.as_table()?,
        None => table,
    };
    table.get(key)
}

fn parse_bind_addr(raw: &str, sources: &ConfigSources) -> Result<SocketAddr> {
    raw.parse().map_err(|error| {
        sources.invalid_with_source(
            "bind",
            format!(
                "must be an IP socket address such as 127.0.0.1:50051 or 0.0.0.0:50051; host names are not resolved, got `{raw}`"
            ),
            error,
        )
    })
}

fn positive_secs(sources: &ConfigSources, path: &'static str, seconds: u64) -> Result<Duration> {
    if seconds == 0 {
        return Err(sources.invalid(path, "must be greater than zero"));
    }
    checked_config_duration(sources, path, Duration::from_secs(seconds))
}

fn checked_config_duration(sources: &ConfigSources, path: &'static str, duration: Duration) -> Result<Duration> {
    if Instant::now().checked_add(duration).is_none() {
        return Err(sources.invalid(path, "is too large for the monotonic clock"));
    }
    Ok(duration)
}

fn readable_init_sql(path: PathBuf, sources: &ConfigSources) -> Result<PathBuf> {
    match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => Ok(path),
        Ok(_) => Err(sources.invalid("init_sql", format!("`{}` is not a file", path.display()))),
        Err(error) => {
            Err(sources.invalid_with_source("init_sql", format!("`{}` cannot be read", path.display()), error))
        },
    }
}

/// Attempts admission can hold at once: every device's active attempts plus
/// the queue shared by all devices.
fn admission_live_attempts(queue_policy: &AlgeonGpuQueuePolicy, profiles: &[AlgeonGpuDeviceProfile]) -> u64 {
    let attempts = profiles
        .iter()
        .map(AlgeonGpuDeviceProfile::max_active_attempts)
        .fold(queue_policy.max_queued_attempts(), usize::saturating_add);
    u64::try_from(attempts).unwrap_or(u64::MAX)
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}
