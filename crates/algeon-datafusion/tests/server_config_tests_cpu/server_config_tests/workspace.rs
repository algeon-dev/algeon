use super::*;

#[test]
fn server_config_accepts_workspace_overlay_defaults_and_aliases() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "algeon_graph".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_ALIASES_ENV, "edges_by_src,edges_by_dst,profiles_by_user_id".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let workspace = config.workspace.expect("workspace overlay config");

    assert_eq!(workspace.catalog, "datafusion");
    assert_eq!(workspace.schema, "public");
    assert_eq!(workspace.backing_catalog, "glue");
    assert_eq!(workspace.backing_schema, "algeon_graph");
    assert_eq!(
        workspace.backing_aliases.unwrap(),
        vec!["edges_by_src".to_string(), "edges_by_dst".to_string(), "profiles_by_user_id".to_string()]
    );
}

#[test]
fn server_config_accepts_explicit_workspace_identity() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_WORKSPACE_CATALOG_ENV, "workspace".to_string()),
        (ALGEON_SERVER_WORKSPACE_SCHEMA_ENV, "session".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "algeon_graph".to_string()),
    ]);
    let config = server_config_with_default_device_profile(env).unwrap();
    let workspace = config.workspace.expect("workspace overlay config");

    assert_eq!(workspace.catalog, "workspace");
    assert_eq!(workspace.schema, "session");
    assert_eq!(workspace.backing_aliases, None);
}

#[test]
fn server_config_rejects_incomplete_workspace_backing() {
    let env = BTreeMap::from([(ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string())]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains(ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV), "{error}");
}

#[test]
fn server_config_rejects_workspace_equal_to_backing() {
    let env = BTreeMap::from([
        (ALGEON_SERVER_WORKSPACE_CATALOG_ENV, "glue".to_string()),
        (ALGEON_SERVER_WORKSPACE_SCHEMA_ENV, "algeon_graph".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV, "glue".to_string()),
        (ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, "algeon_graph".to_string()),
    ]);
    let error = server_config_with_default_device_profile(env).unwrap_err();

    assert!(error.to_string().contains("must differ from backing"), "{error}");
}

#[tokio::test]
async fn create_view_fails_when_default_schema_is_read_only_source_catalog() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_default_catalog_and_schema("glue", "algeon_graph"));
    ctx.register_catalog("glue", Arc::new(ReadOnlyCatalog::new("algeon_graph", [("edges_by_src", edge_table())])));

    let error = ctx.sql("CREATE VIEW target_edges AS SELECT src, dst FROM edges_by_src").await.unwrap_err();

    assert!(error.to_string().contains("schema provider does not support registering tables"), "{error}");
}

#[tokio::test]
async fn workspace_default_create_view_can_read_qualified_source_catalog() {
    use datafusion::prelude::{SessionConfig, SessionContext};

    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_default_catalog_and_schema("datafusion", "public"));
    ctx.register_catalog("glue", Arc::new(ReadOnlyCatalog::new("algeon_graph", [("edges_by_src", edge_table())])));

    ctx.sql("CREATE VIEW target_edges AS SELECT src, dst FROM glue.algeon_graph.edges_by_src")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let batches = ctx.sql("SELECT COUNT(*) FROM target_edges").await.unwrap().collect().await.unwrap();

    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batch_i64_value(&batches[0], 0), 2);
}
