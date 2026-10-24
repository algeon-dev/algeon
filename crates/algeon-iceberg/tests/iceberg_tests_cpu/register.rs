use std::{fs, net::TcpListener, sync::Arc};

use algeon_datafusion::{
    error::ErrorCode,
    server::{ALGEON_SERVER_CONFIG_FILE_ENV, LoadedServerConfig, ServerConfig},
};
use algeon_iceberg::register_iceberg_catalogs;
use datafusion::{catalog::MemoryCatalogProvider, prelude::SessionContext};

fn server_config(catalogs: &str) -> (tempfile::TempDir, ServerConfig) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.toml");
    let toml = format!("[s3]\nendpoint = \"http://127.0.0.1:9\"\npath_style = true\n\n{catalogs}");
    fs::write(&path, toml).unwrap();
    let loaded =
        LoadedServerConfig::from_vars([(ALGEON_SERVER_CONFIG_FILE_ENV.to_string(), path.display().to_string())])
            .unwrap();
    (dir, loaded.config)
}

#[tokio::test]
async fn a_catalog_name_the_session_already_holds_is_refused() {
    let (_dir, config) =
        server_config("[[iceberg.catalogs]]\nname = \"lake\"\nkind = \"rest\"\nuri = \"http://127.0.0.1:9\"\n");
    let ctx = SessionContext::new();
    ctx.register_catalog("lake", Arc::new(MemoryCatalogProvider::new()));

    let error = register_iceberg_catalogs(&ctx, &config).await.unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration);
    assert_eq!(error.fact_value("config_field"), Some("iceberg.catalogs.name"));
}

#[tokio::test]
async fn an_unreachable_rest_catalog_fails_registration_with_a_source_error() {
    // A bound and closed port refuses the connection.
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (_dir, config) = server_config(&format!(
        "[[iceberg.catalogs]]\nname = \"lake\"\nkind = \"rest\"\nuri = \"http://127.0.0.1:{port}\"\n"
    ));
    let ctx = SessionContext::new();

    let error = register_iceberg_catalogs(&ctx, &config).await.unwrap_err();
    assert!(
        matches!(error.code(), ErrorCode::SourceTemporaryFailure | ErrorCode::SourceDependencyFailure),
        "{error:?}"
    );
    assert_eq!(error.fact_value("catalog"), Some("lake"));
    assert!(ctx.catalog("lake").is_none());
}
