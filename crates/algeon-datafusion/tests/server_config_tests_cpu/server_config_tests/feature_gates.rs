#[cfg(not(feature = "cugraph"))]
use super::*;

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_accepts_cugraph_env_without_cugraph_feature() {
    let env = BTreeMap::from([("ALGEON_SERVER_CUGRAPH_ENABLED", "true".to_string())]);
    let config = load(env).unwrap();

    assert!(config.cugraph.is_some(), "an explicit enabled setting must install the unavailable SQL frontend");
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_accepts_explicit_cugraph_disabled_without_cugraph_feature() {
    // One env file must be able to serve both artifact flavors. An explicit
    // `false` requests no capability, so a no-cugraph build accepts it.
    let env = BTreeMap::from([("ALGEON_SERVER_CUGRAPH_ENABLED", "false".to_string())]);
    let config = load(env).unwrap();

    assert_eq!(config.bind_addr.to_string(), "127.0.0.1:50051");
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_cugraph_settings_alongside_disabled_env_without_cugraph_feature() {
    let env = BTreeMap::from([
        ("ALGEON_SERVER_CUGRAPH_ENABLED", "false".to_string()),
        ("ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY", "raw_libcugraph".to_string()),
    ]);
    let error = load(env).unwrap_err();

    assert!(error.to_string().contains("ALGEON_SERVER_CUGRAPH_ENABLED"));
    assert!(error.to_string().contains("is false while other [cugraph] settings are set"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn server_config_rejects_malformed_cugraph_enabled_without_cugraph_feature() {
    let env = BTreeMap::from([("ALGEON_SERVER_CUGRAPH_ENABLED", "maybe".to_string())]);
    let error = load(env).unwrap_err();

    assert!(error.to_string().contains("must be a boolean"));
}
