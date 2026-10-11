use super::{
    ConfigSources, CugraphGraphConstructionPolicy, CugraphGraphOptions, Result, ServerError,
    file::ServerCugraphFileConfig,
};
use crate::cugraph_sql::GpuIslandHostBoundaryConfig;

#[derive(Debug, Clone, PartialEq)]
/// Graph construction and host staging settings for the registered server cuGraph SQL family.
pub struct CugraphServerConfig {
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Named graph arguments still override these values for one call.
    pub default_graph_options: CugraphGraphOptions,
    /// Cumulative Arrow staging limits, independent of the query device grant.
    pub host_boundary_config: GpuIslandHostBoundaryConfig,
}

/// `[cugraph]`. `enabled = false` leaves the SQL family unregistered, so a
/// `cugraph_*` relation fails as not found; any other `[cugraph]` setting
/// then contradicts it and fails loading. With `enabled = true` and a binary
/// built without the `cugraph` feature, the functions register and execution
/// returns `required_feature_disabled`.
pub(super) fn cugraph_config(
    file: &ServerCugraphFileConfig,
    sources: &ConfigSources,
) -> Result<Option<CugraphServerConfig>> {
    let requested = file.construction_policy.is_some()
        || file.max_input_staging_bytes.is_some()
        || file.max_output_staging_bytes.is_some();
    if !file.enabled.unwrap_or(true) {
        if requested {
            return Err(sources.invalid(
                "cugraph.enabled",
                "is false while other [cugraph] settings are set; remove them or enable cuGraph",
            ));
        }
        return Ok(None);
    }

    let mut default_graph_options = CugraphGraphOptions::default();
    if let Some(raw) = &file.construction_policy {
        default_graph_options.construction_policy = parse_cugraph_construction_policy(sources, raw)?;
    }

    let defaults = GpuIslandHostBoundaryConfig::default();
    let input_bytes = file.max_input_staging_bytes.unwrap_or(defaults.max_input_staging_bytes());
    let output_bytes = file.max_output_staging_bytes.unwrap_or(defaults.max_output_staging_bytes());
    if input_bytes == 0 {
        return Err(sources.invalid("cugraph.max_input_staging_bytes", "must be greater than zero"));
    }
    if output_bytes == 0 {
        return Err(sources.invalid("cugraph.max_output_staging_bytes", "must be greater than zero"));
    }
    let host_boundary_config = GpuIslandHostBoundaryConfig::try_new(input_bytes, output_bytes)
        .map_err(|error| ServerError::configuration_with_source("invalid cuGraph host staging limits", error))?;
    Ok(Some(CugraphServerConfig { default_graph_options, host_boundary_config }))
}

fn parse_cugraph_construction_policy(sources: &ConfigSources, raw: &str) -> Result<CugraphGraphConstructionPolicy> {
    let raw = raw.trim().to_ascii_lowercase();
    [CugraphGraphConstructionPolicy::PythonCugraph, CugraphGraphConstructionPolicy::RawLibcugraph]
        .into_iter()
        .find(|policy| policy.as_str() == raw)
        .ok_or_else(|| {
            sources.invalid(
                "cugraph.construction_policy",
                format!("must be `python_cugraph` or `raw_libcugraph`, got `{raw}`"),
            )
        })
}
