use std::fs;

use super::*;

/// Raw shape of the optional TOML overlay loaded from
/// [`NEXUS_SERVER_CONFIG_FILE_ENV`].
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerFileOverlay {
    pub(super) auth: AuthConfig,
    pub(super) admission: ServerAdmissionFileConfig,
    pub(super) native: ServerNativeFileConfig,
    pub(super) observability: ServerObservabilityFileConfig,
}

#[derive(Debug, Default, Deserialize)]
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

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerAdmissionFileConfig {
    #[serde(default)]
    pub(super) device_profiles: Vec<ServerAdmissionDeviceProfileFileConfig>,
    pub(super) max_queued_attempts: Option<usize>,
    pub(super) max_queue_wait_ms: Option<u64>,
    pub(super) max_overtakes_per_waiter: Option<u64>,
}

/// One explicit ordinal-keyed device envelope in `[admission]`.
///
/// It intentionally contains resource inputs only.  Deployment ownership and
/// exact capability restrictions are supplied by the embedding process via
/// `ServerConfig`, never invented by TOML parsing.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ServerAdmissionDeviceProfileFileConfig {
    pub(super) device_ordinal: usize,
    pub(super) memory_limit_bytes: Option<u64>,
    pub(super) backend_reserve_bytes: Option<u64>,
    pub(super) cache_cap_bytes: Option<u64>,
    pub(super) attempt_floor_bytes: Option<u64>,
    pub(super) source_work_per_attempt: Option<usize>,
    pub(super) max_active_attempts: Option<usize>,
    pub(super) stream_pool_size: Option<usize>,
    pub(super) worker_count: Option<usize>,
    pub(super) lane_count: Option<usize>,
    pub(super) install_telemetry: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerNativeFileConfig {
    pub(super) max_source_chunk_bytes: Option<u64>,
    pub(super) max_row_groups_per_chunk: Option<usize>,
    pub(super) max_retry_attempts: Option<usize>,
    pub(super) max_source_reads_in_flight: Option<usize>,
    pub(super) source_chunk_planning_mode: Option<String>,
    pub(super) statistical_aggregate_execution_mode: Option<String>,
    pub(super) final_plan_requirement: Option<String>,
}
pub(super) fn load_overlay(path: &std::path::Path) -> Result<ServerFileOverlay> {
    let raw = fs::read_to_string(path).map_err(|error| {
        ServerError::configuration_with_source(format!("failed to read server config file `{}`", path.display()), error)
    })?;
    toml::from_str::<ServerFileOverlay>(&raw).map_err(|error| {
        // The structured source is retained for diagnostics, while the parser
        // detail remains in the operator-facing configuration error.
        let detail = error.to_string();
        ServerError::configuration_with_source(
            format!("failed to parse server config file `{}`: {detail}", path.display()),
            error,
        )
    })
}
