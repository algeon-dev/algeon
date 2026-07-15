use super::*;

/// Raw shape of the optional TOML overlay loaded from
/// [`NEXUS_SERVER_CONFIG_FILE_ENV`].
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerFileOverlay {
    pub(super) auth: AuthConfig,
    pub(super) admission: ServerAdmissionFileConfig,
    pub(super) fallback: ServerFallbackFileConfig,
    pub(super) native: ServerNativeFileConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerAdmissionFileConfig {
    pub(super) workload_profile: Option<String>,
    pub(super) max_concurrent_queries: Option<usize>,
    pub(super) max_queries_per_device: Option<usize>,
    pub(super) placement_policy: Option<String>,
    pub(super) pinned_device_ordinal: Option<usize>,
    pub(super) grant_min: Option<ServerMemoryShareFileValue>,
    pub(super) grant_want: Option<ServerMemoryShareFileValue>,
    pub(super) slot_guarantee: Option<String>,
    pub(super) wait_timeout_ms: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerFallbackFileConfig {
    pub(super) max_concurrent_statements: Option<usize>,
    pub(super) wait_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum ServerMemoryShareFileValue {
    Integer(u64),
    Float(f64),
    String(String),
}

impl ServerMemoryShareFileValue {
    pub(super) fn as_raw(&self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::String(value) => value.clone(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub(super) struct ServerNativeFileConfig {
    pub(super) memory_policy: Option<String>,
    pub(super) device_budget_bytes: Option<u64>,
    pub(super) max_source_chunk_bytes: Option<u64>,
    pub(super) max_row_groups_per_chunk: Option<usize>,
    pub(super) max_retry_attempts: Option<usize>,
    pub(super) min_retry_chunk_row_groups: Option<usize>,
    pub(super) max_source_reads_in_flight: Option<usize>,
    pub(super) source_chunk_planning_mode: Option<String>,
    pub(super) host_fetch_acquire_timeout_secs: Option<u64>,
    pub(super) source_cache_max_retained_bytes: Option<u64>,
    pub(super) graph_cache_max_retained_bytes: Option<u64>,
    pub(super) query_memory_domain_ceiling_multiplier: Option<u64>,
    pub(super) pipeline_execution_mode: Option<String>,
    pub(super) statistical_aggregate_execution_mode: Option<String>,
}
pub(super) fn load_overlay(path: &std::path::Path) -> Result<ServerFileOverlay> {
    let raw = fs::read_to_string(path).map_err(|error| {
        ServerError::configuration_with_source(format!("failed to read server config file `{}`", path.display()), error)
    })?;
    toml::from_str::<ServerFileOverlay>(&raw).map_err(|error| {
        ServerError::configuration_with_source(
            format!("failed to parse server config file `{}`", path.display()),
            error,
        )
    })
}
