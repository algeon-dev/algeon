#[test]
fn server_cugraph_config_accepts_enabled_env() {
    use datafusion_nexus::cugraph_sql::CugraphGraphConstructionPolicy;
    let env = BTreeMap::from([
        (NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
        (NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string()),
        (NEXUS_SERVER_CUGRAPH_RENUMBER_ENV, "false".to_string()),
        (NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);

    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
    let cugraph = config.cugraph.expect("enabled env must materialize cuGraph server config");

    assert!(!cugraph.default_graph_options.directed);
    assert!(!cugraph.default_graph_options.renumber);
    assert_eq!(cugraph.default_graph_options.construction_policy, CugraphGraphConstructionPolicy::RawLibcugraph);
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_feature_off_cugraph_config_is_retained_for_unavailable_registration() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(
        config.cugraph.is_some(),
        "the server retains the request so the SQL layer can expose typed unavailability"
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_rejects_settings_without_enabled_true() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, "false".to_string())]);
    let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap_err();

    assert!(error.to_string().contains(NEXUS_SERVER_CUGRAPH_ENABLED_ENV));
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_materializes_when_enabled() {
    let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(config.cugraph.is_some());
}

#[cfg(feature = "cugraph")]
#[test]
fn server_cugraph_config_remains_absent_when_disabled() {
    let env: BTreeMap<&'static str, String> = BTreeMap::new();
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();

    assert!(config.cugraph.is_none());
}
