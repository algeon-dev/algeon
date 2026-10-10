use super::*;

/// A config file holding `toml_body` and the environment that names it.
pub(super) fn env_with_server_toml(toml_body: &str) -> (tempfile::TempDir, BTreeMap<&'static str, String>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("server.toml");
    std::fs::write(&path, toml_body).expect("write config file");
    (dir, BTreeMap::from([(ALGEON_SERVER_CONFIG_FILE_ENV, path.to_string_lossy().into_owned())]))
}

/// Loads with `env` as the complete environment and no config file.
pub(super) fn load(env: BTreeMap<&'static str, String>) -> ServerResult<ServerConfig> {
    ServerConfig::from_vars(env)
}

/// Loads from a config file holding `toml_body` and an otherwise empty environment.
pub(super) fn load_toml(toml_body: &str) -> ServerResult<ServerConfig> {
    let (_dir, env) = env_with_server_toml(toml_body);
    ServerConfig::from_vars(env)
}
