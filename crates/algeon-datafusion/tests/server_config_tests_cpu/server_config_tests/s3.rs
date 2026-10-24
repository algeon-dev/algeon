use algeon_datafusion::native_exec::S3SourceStorageProfile;

use super::*;

fn s3_storage(config: &ServerConfig) -> Option<&S3SourceStorageProfile> {
    config.native_optimizer_config.native_execution_config().s3_storage()
}

#[test]
fn s3_table_sets_the_storage_for_s3_sources() {
    let config = load_toml(
        r#"
[s3]
endpoint = "http://127.0.0.1:9000"
region = "eu-west-2"
path_style = true
unsigned = true
"#,
    )
    .unwrap();
    let expected = S3SourceStorageProfile::default_chain()
        .with_endpoint("http://127.0.0.1:9000")
        .with_region("eu-west-2")
        .with_path_style_access(true)
        .with_unsigned_requests();
    assert_eq!(s3_storage(&config), Some(&expected));

    let config = load_toml("[s3]\n").unwrap();
    assert_eq!(s3_storage(&config), Some(&S3SourceStorageProfile::default_chain()));

    assert!(s3_storage(&load(BTreeMap::new()).unwrap()).is_none());
}

#[test]
fn s3_environment_variables_override_the_table_or_enable_it() {
    let (_dir, mut env) = env_with_server_toml("[s3]\nregion = \"eu-west-2\"\nunsigned = true\n");
    env.insert("ALGEON_SERVER_S3_REGION", "us-west-2".to_string());
    env.insert("ALGEON_SERVER_S3_UNSIGNED", "false".to_string());
    let loaded = LoadedServerConfig::from_vars(env).unwrap();
    assert_eq!(s3_storage(&loaded.config), Some(&S3SourceStorageProfile::default_chain().with_region("us-west-2")));
    assert_eq!(loaded.source("s3.region"), ConfigValueSource::Env("ALGEON_SERVER_S3_REGION".to_string()));

    let config = load(BTreeMap::from([
        ("ALGEON_SERVER_S3_ENDPOINT", "http://127.0.0.1:9000".to_string()),
        ("ALGEON_SERVER_S3_PATH_STYLE", "true".to_string()),
    ]))
    .unwrap();
    assert_eq!(
        s3_storage(&config),
        Some(
            &S3SourceStorageProfile::default_chain()
                .with_endpoint("http://127.0.0.1:9000")
                .with_path_style_access(true)
        )
    );
}

#[test]
fn s3_custom_endpoint_requires_path_style_addressing() {
    let error = load_toml("[s3]\nendpoint = \"http://127.0.0.1:9000\"\npath_style = false\n")
        .expect_err("a custom endpoint with virtual-hosted addressing");
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("s3.path_style"), "{error}");
    assert_eq!(error.fact_value("reason"), Some("s3_custom_endpoint_requires_path_style"), "{error}");
    assert!(error.message().contains("set `path_style = true`"), "{error}");

    let (_dir, mut env) = env_with_server_toml("[s3]\nendpoint = \"http://127.0.0.1:9000\"\n");
    env.insert("ALGEON_SERVER_S3_PATH_STYLE", "false".to_string());
    let error = ServerConfig::from_vars(env).expect_err("virtual-hosted addressing set by the environment");
    assert_eq!(error.fact_value("config_env"), Some("ALGEON_SERVER_S3_PATH_STYLE"), "{error}");

    // KvikIO and object_store address a custom endpoint path-style when the
    // flag is unset.
    let config = load_toml("[s3]\nendpoint = \"http://127.0.0.1:9000\"\n").unwrap();
    assert_eq!(
        s3_storage(&config),
        Some(&S3SourceStorageProfile::default_chain().with_endpoint("http://127.0.0.1:9000"))
    );
}

#[test]
fn effective_toml_prints_the_s3_table_with_sources() {
    let (_dir, mut env) = env_with_server_toml("[s3]\nregion = \"eu-west-2\"\n");
    env.insert("ALGEON_SERVER_S3_PATH_STYLE", "true".to_string());
    let rendered = LoadedServerConfig::from_vars(env).unwrap().effective_toml();
    for line in [
        "\n[s3]\n",
        "region = \"eu-west-2\"  # source: file",
        "path_style = true  # source: env ALGEON_SERVER_S3_PATH_STYLE",
        "unsigned = false  # source: default",
        "# endpoint is unset  # source: default",
    ] {
        assert!(rendered.contains(line), "missing `{line}` in\n{rendered}");
    }
    let parsed = rendered.parse::<toml::Table>().expect("the rendering is valid TOML");
    assert!(parsed.contains_key("s3"), "{rendered}");

    let defaulted = LoadedServerConfig::from_vars(BTreeMap::<&str, String>::new()).unwrap().effective_toml();
    assert!(defaulted.contains("\n# [s3]\n"), "{defaulted}");
}
