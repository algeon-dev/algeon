use super::{env::EnvValues, *};

/// Enables registration of cuGraph SQL functions. Defaults to `true`; set it
/// to `false` to hide the SQL frontend. A feature-off server still
/// registers the public SQL frontend when enabled, but execution then returns
/// its structured `required_feature_disabled` capability error.
pub const ALGEON_SERVER_CUGRAPH_ENABLED_ENV: &str = "ALGEON_SERVER_CUGRAPH_ENABLED";
/// Default cuGraph graph construction backend. Use `python_cugraph` or
/// `raw_libcugraph`; named graph arguments can override it for one call.
pub const ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV: &str = "ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY";
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphServerConfig {
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Named graph arguments still override these values for one call.
    pub default_graph_options: CugraphGraphOptions,
}
pub(super) fn cugraph_config(values: &mut EnvValues<'_>) -> Result<Option<CugraphServerConfig>> {
    let requested = values.any_present([ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV]);
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

    Ok(Some(CugraphServerConfig { default_graph_options }))
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
