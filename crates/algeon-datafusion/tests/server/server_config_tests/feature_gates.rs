#[cfg(not(feature = "iceberg"))]
#[test]
fn server_config_rejects_iceberg_env_without_iceberg_feature() {
    let env = BTreeMap::from([(ALGEON_ICEBERG_NAMESPACE_ENV, "tpch_sf1".to_string())]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains("-p server --features iceberg"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_accepts_cugraph_env_without_cugraph_feature() {
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert!(config.cugraph.is_some(), "an explicit enabled setting must install the unavailable SQL frontend");
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_accepts_explicit_cugraph_disabled_without_cugraph_feature() {
    // One env file must be able to serve both artifact flavors: an explicit
    // `false` requests no capability, so a no-cugraph build accepts it.
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string())]);
    let config = server_config_with_default_device_profile(env).unwrap();

    assert_eq!(config.bind_addr.to_string(), "0.0.0.0:50051");
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_cugraph_settings_alongside_disabled_env_without_cugraph_feature() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "false".to_string()),
        (algeon_datafusion::server::ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, "raw_libcugraph".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains(ALGEON_SERVER_CUGRAPH_ENABLED_ENV));
    assert!(error.to_string().contains("must be true"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_malformed_cugraph_enabled_without_cugraph_feature() {
    let env = BTreeMap::from([(ALGEON_SERVER_CUGRAPH_ENABLED_ENV, "maybe".to_string())]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains("must be a boolean"));
}
