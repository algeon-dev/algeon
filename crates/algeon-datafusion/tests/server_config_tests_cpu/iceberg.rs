use algeon_datafusion::server::{
    ServerIcebergCatalogConfig, ServerIcebergCatalogKind, ServerIcebergGlueCatalogConfig, ServerIcebergRestAuth,
    ServerIcebergRestCatalogConfig, ServerIcebergSecret,
};

use super::*;

const S3: &str = "[s3]\nendpoint = \"http://127.0.0.1:9000\"\npath_style = true\nregion = \"eu-west-2\"\n";

fn with_s3(catalogs: &str) -> String {
    format!("{S3}\n{catalogs}")
}

#[test]
fn iceberg_rest_and_glue_catalogs_load_from_the_file() {
    let config = load_toml(&with_s3(
        r#"
[[iceberg.catalogs]]
name = "lake"
kind = "rest"
uri = "http://127.0.0.1:8181"
warehouse = "s3://warehouse/"
prefix = "tenant"

[iceberg.catalogs.auth]
kind = "oauth2"
credential = "client:secret"
server_uri = "http://127.0.0.1:8181/v1/oauth/tokens"
scope = "catalog"

[[iceberg.catalogs]]
name = "tokens"
kind = "rest"
uri = "http://127.0.0.1:8182"
auth = { kind = "token", token = "bearer-value" }

[[iceberg.catalogs]]
name = "open"
kind = "rest"
uri = "http://127.0.0.1:8183"

[[iceberg.catalogs]]
name = "glue"
kind = "glue"
warehouse = "s3://bucket/warehouse"
catalog_id = "123456789012"
region = "eu-west-2"
endpoint = "http://127.0.0.1:5000"
"#,
    ))
    .unwrap();

    let rest =
        |uri: &str, auth| ServerIcebergRestCatalogConfig { uri: uri.to_string(), warehouse: None, prefix: None, auth };
    let expected = vec![
        ServerIcebergCatalogConfig {
            name: "lake".to_string(),
            kind: ServerIcebergCatalogKind::Rest(ServerIcebergRestCatalogConfig {
                warehouse: Some("s3://warehouse/".to_string()),
                prefix: Some("tenant".to_string()),
                ..rest(
                    "http://127.0.0.1:8181",
                    ServerIcebergRestAuth::OAuth2 {
                        credential: ServerIcebergSecret::new("client:secret"),
                        server_uri: Some("http://127.0.0.1:8181/v1/oauth/tokens".to_string()),
                        scope: Some("catalog".to_string()),
                    },
                )
            }),
        },
        ServerIcebergCatalogConfig {
            name: "tokens".to_string(),
            kind: ServerIcebergCatalogKind::Rest(rest(
                "http://127.0.0.1:8182",
                ServerIcebergRestAuth::Token(ServerIcebergSecret::new("bearer-value")),
            )),
        },
        ServerIcebergCatalogConfig {
            name: "open".to_string(),
            kind: ServerIcebergCatalogKind::Rest(rest("http://127.0.0.1:8183", ServerIcebergRestAuth::None)),
        },
        ServerIcebergCatalogConfig {
            name: "glue".to_string(),
            kind: ServerIcebergCatalogKind::Glue(ServerIcebergGlueCatalogConfig {
                warehouse: "s3://bucket/warehouse".to_string(),
                catalog_id: Some("123456789012".to_string()),
                region: Some("eu-west-2".to_string()),
                endpoint: Some("http://127.0.0.1:5000".to_string()),
            }),
        },
    ];
    assert_eq!(config.iceberg_catalogs, expected);
    let ServerIcebergCatalogKind::Rest(ServerIcebergRestCatalogConfig {
        auth: ServerIcebergRestAuth::OAuth2 { credential, .. },
        ..
    }) = &config.iceberg_catalogs[0].kind
    else {
        panic!("expected an OAuth2 REST catalog");
    };
    assert_eq!(credential.expose(), "client:secret");
    assert!(config.s3_storage().is_some());
}

#[test]
fn iceberg_catalogs_require_the_s3_table() {
    let error = load_toml("[[iceberg.catalogs]]\nname = \"lake\"\nkind = \"rest\"\nuri = \"http://127.0.0.1:8181\"\n")
        .expect_err("catalogs without [s3]");
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("iceberg.catalogs"), "{error}");
    assert_eq!(error.fact_value("reason"), Some("iceberg_catalogs_require_s3"), "{error}");
    assert!(error.message().contains("add `[s3]`"), "{error}");
}

#[test]
fn glue_endpoint_override_requires_an_s3_endpoint() {
    let glue = "[[iceberg.catalogs]]\nname = \"lake\"\nkind = \"glue\"\nwarehouse = \"s3://b/w\"\nendpoint = \"http://127.0.0.1:5001\"\n";
    let error =
        load_toml(&format!("[s3]\nregion = \"eu-west-2\"\n\n{glue}")).expect_err("Glue endpoint without s3.endpoint");
    assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
    assert_eq!(error.fact_value("config_field"), Some("iceberg.catalogs.endpoint"), "{error}");
    assert_eq!(error.fact_value("reason"), Some("iceberg_glue_endpoint_requires_s3_endpoint"), "{error}");

    let config = load_toml(&format!("[s3]\nendpoint = \"http://127.0.0.1:9000\"\npath_style = true\n\n{glue}"))
        .expect("Glue endpoint with s3.endpoint");
    assert_eq!(config.iceberg_catalogs.len(), 1);
}

#[test]
fn iceberg_catalog_entries_reject_missing_and_foreign_keys() {
    for (entry, key) in [
        ("kind = \"rest\"\n", "uri"),
        ("kind = \"glue\"\n", "warehouse"),
        ("uri = \"http://127.0.0.1:8181\"\n", "kind"),
        ("kind = \"hive\"\n", "hive"),
        ("kind = \"rest\"\nuri = \"http://h\"\nauth = { kind = \"oauth2\" }\n", "credential"),
        ("kind = \"rest\"\nuri = \"http://h\"\nauth = { kind = \"token\" }\n", "token"),
        ("kind = \"rest\"\nuri = \"http://h\"\nauth = { kind = \"none\", token = \"t\" }\n", "token"),
        ("kind = \"rest\"\nuri = \"http://h\"\nauth = { kind = \"token\", token = \"t\", scope = \"s\" }\n", "scope"),
        ("kind = \"glue\"\nwarehouse = \"s3://b/w\"\nuri = \"http://h\"\n", "uri"),
        ("kind = \"rest\"\nuri = \"http://h\"\ncatalog_id = \"1\"\n", "catalog_id"),
        ("kind = \"rest\"\nuri = \"http://h\"\nproperties = { a = \"b\" }\n", "properties"),
    ] {
        let toml_body = with_s3(&format!("[[iceberg.catalogs]]\nname = \"lake\"\n{entry}"));
        let error = load_toml(&toml_body).expect_err(entry);
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{entry}: {error}");
        assert!(error.message().contains(&format!("`{key}`")), "{entry}: {error}");
    }
}

#[test]
fn iceberg_catalog_names_are_unique_and_leave_the_default_catalog() {
    let entry = |name: &str| format!("[[iceberg.catalogs]]\nname = \"{name}\"\nkind = \"rest\"\nuri = \"http://h\"\n");
    for (catalogs, reason) in [
        (format!("{}{}", entry("lake"), entry("lake")), Some("duplicate_iceberg_catalog")),
        (entry("datafusion"), Some("iceberg_catalog_name_reserved")),
        (entry(""), None),
    ] {
        let error = load_toml(&with_s3(&catalogs)).expect_err(&catalogs);
        assert_eq!(error.code(), ErrorCode::InvalidConfiguration, "{error}");
        assert_eq!(error.fact_value("config_field"), Some("iceberg.catalogs.name"), "{error}");
        assert_eq!(error.fact_value("reason"), reason, "{error}");
    }
}

#[test]
fn iceberg_secrets_stay_out_of_effective_toml_and_debug() {
    let (_dir, env) = env_with_server_toml(&with_s3(
        r#"
[[iceberg.catalogs]]
name = "lake"
kind = "rest"
uri = "http://127.0.0.1:8181"
auth = { kind = "oauth2", credential = "client:oauth-secret-value" }

[[iceberg.catalogs]]
name = "tokens"
kind = "rest"
uri = "http://127.0.0.1:8182"
auth = { kind = "token", token = "bearer-secret-value" }

[[iceberg.catalogs]]
name = "glue"
kind = "glue"
warehouse = "s3://bucket/warehouse"
"#,
    ));
    let loaded = LoadedServerConfig::from_vars(env).unwrap();
    let rendered = loaded.effective_toml();
    for line in [
        "\n[[iceberg.catalogs]]\nname = \"lake\"  # source: file\nkind = \"rest\"  # source: file\n",
        "# warehouse is unset  # source: default",
        "\n[iceberg.catalogs.auth]\nkind = \"oauth2\"  # source: file\n# credential is set  # source: file\n",
        "kind = \"token\"  # source: file\n# token is set  # source: file\n",
        "warehouse = \"s3://bucket/warehouse\"  # source: file",
    ] {
        assert!(rendered.contains(line), "missing `{line}` in\n{rendered}");
    }
    let parsed = rendered.parse::<toml::Table>().expect("the rendering is valid TOML");
    let catalogs = parsed["iceberg"]["catalogs"].as_array().expect("catalog array");
    assert_eq!(catalogs.len(), 3, "{rendered}");

    let debug = format!("{:?}", loaded.config);
    for (text, secret) in [(&rendered, "oauth-secret-value"), (&debug, "oauth-secret-value")]
        .into_iter()
        .chain([(&rendered, "bearer-secret-value"), (&debug, "bearer-secret-value")])
    {
        assert!(!text.contains(secret), "`{secret}` leaked into\n{text}");
    }
    assert!(debug.contains("<redacted>"), "{debug}");
}

#[test]
fn iceberg_catalogs_and_their_secrets_have_no_environment_form() {
    let toml_body = with_s3(
        "[[iceberg.catalogs]]\nname = \"lake\"\nkind = \"rest\"\nuri = \"http://h\"\nauth = { kind = \"token\", token = \"t\" }\n",
    );
    let (_dir, env) = env_with_server_toml(&toml_body);
    assert_eq!(ServerConfig::from_vars(env.clone()).unwrap().iceberg_catalogs.len(), 1);

    for name in [
        "ALGEON_SERVER_ICEBERG_CATALOGS",
        "ALGEON_SERVER_ICEBERG_CATALOGS_AUTH_TOKEN",
        "ALGEON_SERVER_ICEBERG_CATALOGS_AUTH_CREDENTIAL",
    ] {
        let mut env = env.clone();
        env.insert(name, "from-env".to_string());
        let error = ServerConfig::from_vars(env).expect_err(name);
        assert_eq!(error.fact_value("reason"), Some("unknown_environment_variable"), "{error}");
        assert_eq!(error.fact_value("config_env"), Some(name), "{error}");
    }
}
