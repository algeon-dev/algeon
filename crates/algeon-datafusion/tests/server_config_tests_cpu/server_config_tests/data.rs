use super::*;

#[test]
fn data_root_loads_from_the_toml_table_or_its_environment_override() {
    let root = tempfile::tempdir().expect("data root");
    let (_dir, env) = env_with_server_toml(&format!("[data]\nroot = {:?}\n", root.path().join(".")));
    let config = ServerConfig::from_vars(env).expect("load data TOML");
    assert_eq!(config.data_root, Some(root.path().canonicalize().unwrap()));

    let loaded =
        LoadedServerConfig::from_vars(BTreeMap::from([("ALGEON_SERVER_DATA_ROOT", root.path().display().to_string())]))
            .expect("load data root override");
    assert_eq!(loaded.config.data_root, Some(root.path().canonicalize().unwrap()));
    assert_eq!(loaded.source("data.root"), ConfigValueSource::Env("ALGEON_SERVER_DATA_ROOT".to_string()));
    let rendered = loaded.effective_toml();
    let line = format!(
        "root = {:?}  # source: env ALGEON_SERVER_DATA_ROOT",
        root.path().canonicalize().unwrap().display().to_string()
    );
    assert!(rendered.contains(&line), "missing `{line}` in\n{rendered}");

    assert!(load(BTreeMap::new()).unwrap().data_root.is_none());
}

#[test]
fn data_root_must_be_an_existing_directory() {
    let root = tempfile::tempdir().expect("data root");
    let file = root.path().join("orders.parquet");
    std::fs::write(&file, b"PAR1").unwrap();
    for path in [root.path().join("missing"), file] {
        let error = load(BTreeMap::from([("ALGEON_SERVER_DATA_ROOT", path.display().to_string())]))
            .expect_err("a data root that is not a directory");
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
        assert_eq!(error.fact_value("config_field"), Some("data.root"), "{error}");
        assert_eq!(error.fact_value("config_env"), Some("ALGEON_SERVER_DATA_ROOT"), "{error}");
        assert!(error.message().contains(&path.display().to_string()), "{error}");
    }
}
