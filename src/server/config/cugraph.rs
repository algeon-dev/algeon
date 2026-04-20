use super::{env::EnvValues, *};

/// Enables registration of cuGraph SQL functions. Must be `true` before any
/// other cuGraph server setting is accepted. On a build without the `cugraph`
/// feature an explicit `false` is accepted (it requests no capability), so one
/// env file can serve both artifact flavors; `true` and every other cuGraph
/// setting are still rejected there.
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
#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphServerConfig {
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Per-call `options_json` still overrides these values.
    pub default_graph_options: CugraphGraphOptions,
}
#[cfg(feature = "cugraph")]
pub(super) fn cugraph_config(values: &mut EnvValues<'_>) -> Result<Option<CugraphServerConfig>> {
    let requested = values.any_present([
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]);
    let enabled = values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)?;
    if enabled != Some(true) {
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

#[cfg(feature = "cugraph")]
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
#[cfg(not(feature = "cugraph"))]
pub(super) fn reject_cugraph_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    // An explicit `NEXUS_SERVER_CUGRAPH_ENABLED=false` requests no capability
    // and is accepted, so one env file can serve both artifact flavors.
    if values.any_present([
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]) || values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)? == Some(true)
    {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
            "cuGraph server registration requires building with --features server,cugraph or --features server-cugraph",
        ));
    }
    Ok(())
}
