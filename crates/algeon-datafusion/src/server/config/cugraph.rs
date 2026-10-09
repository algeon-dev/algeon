use super::{env::EnvValues, *};
use crate::cugraph_sql::GpuIslandHostBoundaryConfig;

/// Enables registration of cuGraph SQL functions. Defaults to `true`. When
/// `false`, the cuGraph SQL family is not installed, so `cugraph_*` relations
/// fail with a not-found error. In a binary built without the `cugraph`
/// feature, `true` registers the functions and execution returns the
/// structured `required_feature_disabled` capability error.
pub const ALGEON_SERVER_CUGRAPH_ENABLED_ENV: &str = "ALGEON_SERVER_CUGRAPH_ENABLED";
/// Default cuGraph graph construction backend. Use `python_cugraph` or
/// `raw_libcugraph`; named graph arguments can override it for one call.
pub const ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV: &str = "ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY";
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphServerConfig {
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Named graph arguments still override these values for one call.
    pub default_graph_options: CugraphGraphOptions,
    /// Cumulative Arrow staging limits, independent of the query device grant.
    pub host_boundary_config: GpuIslandHostBoundaryConfig,
}
pub(super) fn cugraph_config(
    values: &mut EnvValues<'_>,
    file: &ServerCugraphFileConfig,
) -> Result<Option<CugraphServerConfig>> {
    let requested = values.any_present([ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV])
        || file.max_input_staging_bytes.is_some()
        || file.max_output_staging_bytes.is_some();
    let enabled = values.optional_bool(ALGEON_SERVER_CUGRAPH_ENABLED_ENV)?.unwrap_or(true);
    if !enabled {
        if requested {
            return Err(ServerError::invalid_config(
                ALGEON_SERVER_CUGRAPH_ENABLED_ENV,
                "must be true when cuGraph server settings are provided",
            ));
        }
        return Ok(None);
    }

    let mut default_graph_options = CugraphGraphOptions::default();
    if let Some(raw) = values.string(ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV) {
        default_graph_options.construction_policy =
            parse_cugraph_construction_policy(ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, &raw)?;
    }

    let defaults = GpuIslandHostBoundaryConfig::default();
    let input_bytes = file.max_input_staging_bytes.unwrap_or(defaults.max_input_staging_bytes());
    let output_bytes = file.max_output_staging_bytes.unwrap_or(defaults.max_output_staging_bytes());
    if input_bytes == 0 {
        return Err(ServerError::invalid_config("cugraph.max_input_staging_bytes", "must be greater than zero"));
    }
    if output_bytes == 0 {
        return Err(ServerError::invalid_config("cugraph.max_output_staging_bytes", "must be greater than zero"));
    }
    let host_boundary_config = GpuIslandHostBoundaryConfig::try_new(input_bytes, output_bytes)
        .map_err(|error| ServerError::configuration_with_source("invalid cuGraph host staging limits", error))?;
    Ok(Some(CugraphServerConfig { default_graph_options, host_boundary_config }))
}

pub(super) fn parse_cugraph_construction_policy(
    env_name: &'static str,
    raw: &str,
) -> Result<CugraphGraphConstructionPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "python_cugraph" => Ok(CugraphGraphConstructionPolicy::PythonCugraph),
        "raw_libcugraph" => Ok(CugraphGraphConstructionPolicy::RawLibcugraph),
        other => Err(ServerError::invalid_config(
            env_name,
            format!("must be `python_cugraph` or `raw_libcugraph`, got `{other}`"),
        )),
    }
}
