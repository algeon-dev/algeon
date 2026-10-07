use super::*;

#[test]
fn models_toml_table_resolves_the_root_and_applies_the_default_size_bound() {
    let models = tempfile::tempdir().expect("models root");
    let (_dir, env) = env_with_server_toml(&format!("[models]\nroot = {:?}\n", models.path().join(".")));
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("load models TOML");
    let store = config.models.expect("configured models root");
    assert_eq!(store.root, models.path().canonicalize().unwrap());
    assert_eq!(store.max_model_bytes, CumlModelStoreConfig::DEFAULT_MAX_MODEL_BYTES);

    let (_dir, env) = env_with_server_toml(&format!("[models]\nroot = {:?}\nmax_model_bytes = 4096\n", models.path()));
    let config = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("load models TOML");
    assert_eq!(config.models.expect("configured models root").max_model_bytes, 4096);

    let (_dir, env) = env_with_default_device_profile();
    assert!(ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap().models.is_none());
}

#[test]
fn models_toml_table_rejects_invalid_settings_at_startup() {
    let models = tempfile::tempdir().expect("models root");
    let file = models.path().join("model.json");
    std::fs::write(&file, b"{}").unwrap();
    for (body, field) in [
        ("[models]\nmax_model_bytes = 4096\n".to_string(), Some("models.max_model_bytes")),
        (format!("[models]\nroot = {:?}\nmax_model_bytes = 0\n", models.path()), Some("models.max_model_bytes")),
        (format!("[models]\nroot = {:?}\n", models.path().join("missing")), Some("models.root")),
        (format!("[models]\nroot = {file:?}\n"), Some("models.root")),
        (format!("[models]\nroot = {:?}\ncache = true\n", models.path()), None),
    ] {
        let (_dir, env) = env_with_server_toml(&body);
        let error = ServerConfig::from_env_get(|name| env.get(name).cloned()).expect_err(&body);
        match field {
            Some(field) => assert_eq!(error.fact_value("config_field"), Some(field), "{body}: {error}"),
            None => assert!(error.to_string().contains("cache"), "{body}: {error}"),
        }
    }
}
