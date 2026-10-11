use super::*;

#[test]
fn derived_environment_overrides_reach_leaves_in_every_section() {
    let models = tempfile::tempdir().expect("models root");
    let env = BTreeMap::from([
        ("ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS", "45".to_string()),
        ("ALGEON_SERVER_NATIVE_EXECUTION_MODE", "functions_only".to_string()),
        ("ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS", "60".to_string()),
        ("ALGEON_SERVER_CUGRAPH_MAX_OUTPUT_STAGING_BYTES", "4096".to_string()),
        ("ALGEON_SERVER_MODELS_ROOT", models.path().to_string_lossy().into_owned()),
        ("ALGEON_SERVER_MODELS_MAX_MODEL_BYTES", "8192".to_string()),
        // A blank value counts as unset.
        ("ALGEON_SERVER_BIND", " ".to_string()),
    ]);
    let loaded = LoadedServerConfig::from_vars(env).expect("derived names load");
    let config = &loaded.config;

    assert_eq!(config.statement_ticket_ttl, Duration::from_secs(45));
    assert_eq!(config.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::FunctionsOnly);
    assert_eq!(config.observability.sealed_ttl, Duration::from_secs(60));
    assert_eq!(config.cugraph.as_ref().unwrap().host_boundary_config.max_output_staging_bytes(), 4096);
    assert_eq!(config.models.as_ref().unwrap().max_model_bytes, 8192);
    assert_eq!(config.bind_addr.to_string(), "127.0.0.1:50051");
    assert_eq!(
        loaded.source("models.max_model_bytes"),
        ConfigValueSource::Env("ALGEON_SERVER_MODELS_MAX_MODEL_BYTES".to_string())
    );
    assert_eq!(loaded.source("bind"), ConfigValueSource::Default);
}

#[test]
fn an_unknown_server_variable_fails_and_names_the_closest_known_name() {
    for (name, suggestion) in [
        ("ALGEON_SERVER_NATIVE_EXECUTON_MODE", "ALGEON_SERVER_NATIVE_EXECUTION_MODE"),
        ("ALGEON_SERVER_EXECUTION_MODE", "ALGEON_SERVER_NATIVE_EXECUTION_MODE"),
        ("ALGEON_SERVER_ADMISSION_MAX_QUEUED", "ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS"),
        ("ALGEON_SERVER_TLS_CERT", "ALGEON_SERVER_TLS_CERT_PATH"),
    ] {
        let error = load(BTreeMap::from([(name, "x".to_string())])).expect_err(name);

        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
        assert_eq!(error.fact_value("reason"), Some("unknown_environment_variable"), "{error}");
        assert_eq!(error.fact_value("config_env"), Some(name), "{error}");
        assert_eq!(error.fact_value("suggestion"), Some(suggestion), "{error}");
        assert!(error.message().contains(name) && error.message().contains(suggestion), "{error}");
    }
    // Arrays of tables have no environment form.
    let error = load(BTreeMap::from([("ALGEON_SERVER_ADMISSION_DEVICE_PROFILES", "0".to_string())]))
        .expect_err("device profiles are file-only");
    assert_eq!(error.fact_value("reason"), Some("unknown_environment_variable"));
    // A name far from every setting gets no guess, so a removed setting is
    // not mistaken for an unrelated one.
    for name in ["ALGEON_SERVER_GPU_DEVICES", "ALGEON_SERVER_S3_REGION", "ALGEON_SERVER_CUGRAPH_DIRECTED"] {
        let error = load(BTreeMap::from([(name, "x".to_string())])).expect_err(name);
        assert_eq!(error.fact_value("suggestion"), None, "{error}");
        assert!(error.message().contains("no known setting has a similar name"), "{error}");
    }
}

#[test]
fn values_record_their_source_and_errors_name_the_toml_path_and_the_variable() {
    let (_dir, mut env) =
        env_with_server_toml("[admission]\nmax_queued_attempts = 4\n\n[native]\nmax_retry_attempts = 5\n");
    env.insert("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "8".to_string());
    let loaded = LoadedServerConfig::from_vars(env.clone()).unwrap();
    assert_eq!(
        loaded.source("admission.max_queued_attempts"),
        ConfigValueSource::Env("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS".to_string())
    );
    assert_eq!(loaded.source("native.max_retry_attempts"), ConfigValueSource::File);
    assert_eq!(loaded.source("native.max_row_groups_per_chunk"), ConfigValueSource::Default);

    env.insert("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "0".to_string());
    let error = ServerConfig::from_vars(env.clone()).unwrap_err();
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));
    assert_eq!(error.fact_value("config_env"), Some("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS"));

    env.insert("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS", "many".to_string());
    let error = ServerConfig::from_vars(env).unwrap_err();
    assert_eq!(error.fact_value("config_field"), Some("admission.max_queued_attempts"));
    assert!(error.message().contains("`many`"), "{error}");
}

#[test]
fn effective_toml_prints_every_value_with_its_source() {
    let (_dir, mut env) = env_with_server_toml(
        "[native]\nmax_retry_attempts = 5\n\n[[admission.device_profiles]]\ndevice_ordinal = 0\nmax_active_attempts = 3\n",
    );
    env.insert("ALGEON_SERVER_NATIVE_EXECUTION_MODE", "native_required".to_string());
    let rendered = LoadedServerConfig::from_vars(env).unwrap().effective_toml();

    for line in [
        "bind = \"127.0.0.1:50051\"  # source: default",
        "max_retry_attempts = 5  # source: file",
        "execution_mode = \"native_required\"  # source: env ALGEON_SERVER_NATIVE_EXECUTION_MODE",
        "# max_source_chunk_bytes is unset  # source: default",
        "[[admission.device_profiles]]",
        "max_active_attempts = 3  # source: file",
    ] {
        assert!(rendered.contains(line), "missing `{line}` in\n{rendered}");
    }
    let parsed = rendered.parse::<toml::Table>().expect("the rendering is valid TOML");
    assert!(parsed.contains_key("native"));

    let defaulted = LoadedServerConfig::from_vars(BTreeMap::<&str, String>::new()).unwrap().effective_toml();
    assert!(defaulted.contains("[[admission.device_profiles]]  # defaulted"), "{defaulted}");
}

#[test]
fn bind_requires_an_ip_socket_address() {
    let error = load(BTreeMap::from([("ALGEON_SERVER_BIND", "localhost:50051".to_string())])).unwrap_err();

    assert_eq!(error.fact_value("config_field"), Some("bind"));
    assert!(
        error.message().contains("must be an IP socket address such as 127.0.0.1:50051 or 0.0.0.0:50051"),
        "{error}"
    );
}

#[test]
fn init_sql_must_name_a_readable_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("absent.sql");
    let error = load_toml(&format!("init_sql = {missing:?}\n")).expect_err("a missing init_sql file");
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("init_sql"));
    assert!(error.message().contains("absent.sql"), "{error}");

    let present = dir.path().join("init.sql");
    std::fs::write(&present, "CREATE VIEW one AS SELECT 1;\n").unwrap();
    let config = load_toml(&format!("init_sql = {present:?}\n")).expect("a readable init_sql file");
    assert_eq!(config.init_sql.as_deref(), Some(present.as_path()));
}
