use serde::Serialize;
use toml::de::Error as TomlDeError;

use super::{AuthConfig, AuthMode, Deserialize, Path, PathBuf, Result, ServerError, fs};

/// Raw shape of the server TOML file. Every server setting has exactly one
/// location here; environment overrides address the scalar leaves of this
/// shape by name (see `env.rs`).
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerFile {
    pub(super) bind: Option<String>,
    pub(super) statement_ticket_ttl_secs: Option<u64>,
    pub(super) http2_keepalive_interval_secs: Option<u64>,
    pub(super) http2_keepalive_timeout_secs: Option<u64>,
    pub(super) init_sql: Option<PathBuf>,
    pub(super) admission: ServerAdmissionFileConfig,
    pub(super) native: ServerNativeFileConfig,
    pub(super) observability: ServerObservabilityFileConfig,
    pub(super) cugraph: ServerCugraphFileConfig,
    pub(super) models: ServerModelsFileConfig,
    pub(super) auth: AuthConfig,
    pub(super) tls: Option<ServerTlsFileConfig>,
}

/// `[cugraph]`: registration, the default construction backend, and
/// cumulative Arrow staging limits for explicit graph functions.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerCugraphFileConfig {
    pub(super) enabled: Option<bool>,
    pub(super) construction_policy: Option<String>,
    pub(super) max_input_staging_bytes: Option<usize>,
    pub(super) max_output_staging_bytes: Option<usize>,
}

/// `[tls]`: PEM files for the Flight SQL listener. Setting `client_ca_path`
/// makes the listener require client certificates signed by that CA.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServerTlsFileConfig {
    pub(super) cert_path: PathBuf,
    pub(super) key_path: PathBuf,
    pub(super) client_ca_path: Option<PathBuf>,
}

/// `[models]`: the directory `cuml_forest_predict` reads model files from.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerModelsFileConfig {
    pub(super) root: Option<PathBuf>,
    pub(super) max_model_bytes: Option<u64>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerObservabilityFileConfig {
    pub(super) per_query_max_records: Option<u64>,
    pub(super) per_query_max_bytes: Option<u64>,
    pub(super) active_logical_capture_max_bytes: Option<u64>,
    pub(super) global_sealed_max_count: Option<usize>,
    pub(super) global_sealed_max_bytes: Option<u64>,
    pub(super) sealed_ttl_secs: Option<u64>,
    pub(super) shutdown_grace_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerAdmissionFileConfig {
    pub(super) max_queued_attempts: Option<usize>,
    pub(super) max_queue_wait_ms: Option<u64>,
    pub(super) device_profiles: Vec<ServerAdmissionDeviceProfileFileConfig>,
}

/// One `[[admission.device_profiles]]` entry: the resource bounds of one GPU.
///
/// It contains resource inputs only. Deployment ownership and device
/// capabilities come from the embedding process through `ServerConfig`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServerAdmissionDeviceProfileFileConfig {
    pub(super) device_ordinal: usize,
    pub(super) memory_limit_bytes: Option<u64>,
    pub(super) backend_reserve_bytes: Option<u64>,
    pub(super) cache_cap_bytes: Option<u64>,
    pub(super) default_query_limit_bytes: Option<u64>,
    pub(super) source_work_per_attempt: Option<usize>,
    pub(super) max_active_attempts: Option<usize>,
    pub(super) stream_pool_size: Option<usize>,
    pub(super) worker_count: Option<usize>,
    pub(super) lane_count: Option<usize>,
    pub(super) install_telemetry: Option<bool>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerNativeFileConfig {
    pub(super) execution_mode: Option<String>,
    pub(super) max_source_chunk_bytes: Option<u64>,
    pub(super) max_row_groups_per_chunk: Option<usize>,
    pub(super) max_retry_attempts: Option<usize>,
    pub(super) max_source_reads_in_flight: Option<usize>,
    pub(super) source_chunk_planning_mode: Option<String>,
    pub(super) statistical_aggregate_execution_mode: Option<String>,
}

impl ServerFile {
    /// Populates scalar leaves with their TOML types for environment overrides.
    /// Exhaustive literals require new schema fields to appear in this template.
    pub(super) fn leaf_type_template() -> Self {
        Self {
            bind: Some(String::new()),
            statement_ticket_ttl_secs: Some(0),
            http2_keepalive_interval_secs: Some(0),
            http2_keepalive_timeout_secs: Some(0),
            init_sql: Some(PathBuf::new()),
            admission: ServerAdmissionFileConfig {
                max_queued_attempts: Some(0),
                max_queue_wait_ms: Some(0),
                device_profiles: Vec::new(),
            },
            native: ServerNativeFileConfig {
                execution_mode: Some(String::new()),
                max_source_chunk_bytes: Some(0),
                max_row_groups_per_chunk: Some(0),
                max_retry_attempts: Some(0),
                max_source_reads_in_flight: Some(0),
                source_chunk_planning_mode: Some(String::new()),
                statistical_aggregate_execution_mode: Some(String::new()),
            },
            observability: ServerObservabilityFileConfig {
                per_query_max_records: Some(0),
                per_query_max_bytes: Some(0),
                active_logical_capture_max_bytes: Some(0),
                global_sealed_max_count: Some(0),
                global_sealed_max_bytes: Some(0),
                sealed_ttl_secs: Some(0),
                shutdown_grace_secs: Some(0),
            },
            cugraph: ServerCugraphFileConfig {
                enabled: Some(false),
                construction_policy: Some(String::new()),
                max_input_staging_bytes: Some(0),
                max_output_staging_bytes: Some(0),
            },
            models: ServerModelsFileConfig { root: Some(PathBuf::new()), max_model_bytes: Some(0) },
            auth: AuthConfig { mode: AuthMode::BasicBearer { users: Vec::new(), token_ttl_secs: 0 } },
            tls: Some(ServerTlsFileConfig {
                cert_path: PathBuf::new(),
                key_path: PathBuf::new(),
                client_ca_path: Some(PathBuf::new()),
            }),
        }
    }
}

impl ServerAdmissionDeviceProfileFileConfig {
    /// Every field set, for the same reason as [`ServerFile::leaf_type_template`].
    pub(super) fn leaf_type_template() -> Self {
        Self {
            device_ordinal: 0,
            memory_limit_bytes: Some(0),
            backend_reserve_bytes: Some(0),
            cache_cap_bytes: Some(0),
            default_query_limit_bytes: Some(0),
            source_work_per_attempt: Some(0),
            max_active_attempts: Some(0),
            stream_pool_size: Some(0),
            worker_count: Some(0),
            lane_count: Some(0),
            install_telemetry: Some(false),
        }
    }
}

/// Validates the file against the typed schema before returning its TOML table.
/// Direct parsing preserves line numbers in type and unknown-field errors;
/// deserialization after environment overrides loses those locations.
pub(super) fn load_file(path: &Path) -> Result<toml::Table> {
    let raw = fs::read_to_string(path).map_err(|error| {
        ServerError::configuration_with_source(format!("failed to read server config file `{}`", path.display()), error)
    })?;
    let parse_error = |error: TomlDeError| {
        let detail = error.to_string();
        ServerError::configuration_with_source(
            format!("failed to parse server config file `{}`: {detail}", path.display()),
            error,
        )
    };
    toml::from_str::<ServerFile>(&raw).map_err(parse_error)?;
    toml::from_str::<toml::Table>(&raw).map_err(parse_error)
}
