use super::{env::EnvValues, *};

/// Enables registration of cuGraph SQL functions. Defaults to `true`; set it
/// to `false` to hide the SQL frontend. A feature-off server still
/// registers the public SQL frontend when enabled, but execution then returns
/// its structured `required_feature_disabled` capability error.
pub const NEXUS_SERVER_CUGRAPH_ENABLED_ENV: &str = "NEXUS_SERVER_CUGRAPH_ENABLED";
/// Default cuGraph edge direction setting for SQL calls. Accepts boolean
/// values; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_DIRECTED_ENV: &str = "NEXUS_SERVER_CUGRAPH_DIRECTED";
/// Default cuGraph vertex renumbering setting for SQL calls. Accepts boolean
/// values; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_RENUMBER_ENV: &str = "NEXUS_SERVER_CUGRAPH_RENUMBER";
/// Default cuGraph graph construction backend. Use `python_cugraph` or
/// `raw_libcugraph`; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV: &str = "NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY";
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphServerConfig {
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Per-call `options_json` still overrides these values.
    pub default_graph_options: CugraphGraphOptions,
}
pub(super) fn cugraph_config(values: &mut EnvValues<'_>) -> Result<Option<CugraphServerConfig>> {
    let requested = values.any_present([
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]);
    let enabled = values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)?.unwrap_or(true);
    if !enabled {
        if requested {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
                "must be true when cuGraph server settings are provided",
            ));
        }
        return Ok(None);
    }

    let mut default_graph_options = CugraphGraphOptions::default();
    if let Some(directed) = values.optional_bool(NEXUS_SERVER_CUGRAPH_DIRECTED_ENV)? {
        default_graph_options.directed = directed;
    }
    if let Some(renumber) = values.optional_bool(NEXUS_SERVER_CUGRAPH_RENUMBER_ENV)? {
        default_graph_options.renumber = renumber;
    }
    if let Some(raw) = values.string(NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV) {
        default_graph_options.construction_policy =
            parse_cugraph_construction_policy(NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, &raw)?;
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
