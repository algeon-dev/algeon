use super::*;

#[test]
fn server_cugraph_config_accepts_enabled_env() {
    use algeon_datafusion::cugraph_sql::CugraphGraphConstructionPolicy;
    let env = BTreeMap::from([
        (ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);

    let config = server_config_with_default_device_profile(env).unwrap();
    let cugraph = config.cugraph.expect("enabled env must materialize cuGraph server config");

    assert_eq!(cugraph.default_graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);
}

#[test]
fn server_cugraph_config_rejects_removed_graph_option_env() {
    for name in ["ALGEON_SERVER_CUGRAPH_DIRECTED", "ALGEON_SERVER_CUGRAPH_RENUMBER"] {
        let env = BTreeMap::from([(name, "false".to_string())]);
        let error = server_config_with_default_device_profile(env).unwrap_err();
        assert_eq!(error.fact_value("config_field"), Some(name), "{error}");
        assert!(error.message().contains("removed by the resource-only admission contract"), "{error}");
    }
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_feature_off_cugraph_config_is_retained_for_unavailable_registration() {
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert!(
        config.cugraph.is_some(),
        "the server retains the request so the SQL layer can expose typed unavailability"
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_rejects_settings_when_explicitly_disabled() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string()),
        (ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains(ALGEON_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_materializes_when_enabled() {
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert!(config.cugraph.is_some());
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_defaults_to_enabled() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = server_config_with_default_device_profile(env).unwrap();
    let cugraph = config.cugraph.expect("cuGraph must be enabled by default");

    assert_eq!(cugraph.default_graph_options, CugraphGraphOptions::default());
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_remains_absent_when_explicitly_disabled() {
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert!(config.cugraph.is_none());
}

#[test]
fn server_cugraph_staging_limits_load_from_toml_and_require_positive_values() {
    let (_dir, mut env) = env_with_server_toml(
        "[cugraph]\nmax_input_staging_bytes = 8589934592\nmax_output_staging_bytes = 2147483648\n",
    );
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let limits = config.cugraph.unwrap().host_boundary_config;
    assert_eq!(limits.max_input_staging_bytes(), 8_589_934_592);
    assert_eq!(limits.max_output_staging_bytes(), 2_147_483_648);

    env.insert(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string());
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
    assert_eq!(error.fact_value("config_field"), Some(ALGEON_SERVER_CUGRAPH_ENABLED_ENV));

    for field in ["max_input_staging_bytes", "max_output_staging_bytes"] {
        let (_dir, env) = env_with_server_toml(&format!("[cugraph]\n{field} = 0\n"));
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();
        assert_eq!(error.fact_value("config_field"), Some(format!("cugraph.{field}").as_str()));
    }
}
