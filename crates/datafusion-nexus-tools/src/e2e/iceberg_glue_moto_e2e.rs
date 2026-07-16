//! Local AWS Glue-compatible E2E runner backed by Moto + RustFS.
//!
//! This exercises the real `GlueIcebergCatalogBackend` without requiring
//! a developer-owned AWS Glue catalog. Bring up `fixture/iceberg-local` first,
//! then run with `NEXUS_ICEBERG_GLUE_LOCAL_E2E=1`.

use std::{collections::HashMap, sync::Arc};

use datafusion::catalog::CatalogProvider;
use datafusion_nexus::table_format::iceberg::catalog::{
    GlueIcebergCatalogBackend, GlueIcebergCatalogConfig, IcebergCatalogBackend, IcebergProviderKind,
    build_scoped_catalog_provider_from_backend,
};
use iceberg::{
    Catalog, CatalogBuilder, NamespaceIdent, TableCreation,
    io::{S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_PATH_STYLE_ACCESS, S3_REGION, S3_SECRET_ACCESS_KEY},
    spec::{NestedField, PrimitiveType, Schema, Type},
};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, GLUE_CATALOG_PROP_URI, GLUE_CATALOG_PROP_WAREHOUSE,
    GlueCatalogBuilder,
};
use url::Url;

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub async fn run() -> E2eResult<()> {
    require_local_glue_enabled()?;
    require_local_endpoint_env()?;
    glue_local_backend_and_provider_discover_table().await?;
    println!("iceberg_glue_moto_e2e ok");
    Ok(())
}

fn require_local_glue_enabled() -> E2eResult<()> {
    match std::env::var("NEXUS_ICEBERG_GLUE_LOCAL_E2E").as_deref() {
        Ok("1") => Ok(()),
        _ => {
            Err("local Glue E2E requires NEXUS_ICEBERG_GLUE_LOCAL_E2E=1 and fixture/iceberg-local Moto + RustFS".into())
        },
    }
}

fn local_glue_config() -> GlueIcebergCatalogConfig {
    let glue_uri = std::env::var("NEXUS_ICEBERG_GLUE_URI").unwrap_or_else(|_| "http://localhost:5001".to_string());
    let warehouse = local_warehouse();
    let s3_endpoint =
        std::env::var("NEXUS_ICEBERG_S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".to_string());
    let s3_region = std::env::var("NEXUS_ICEBERG_S3_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let path_style = std::env::var("NEXUS_ICEBERG_S3_PATH_STYLE").map(|value| value == "true").unwrap_or(true);

    GlueIcebergCatalogConfig::new("glue", warehouse, "us-east-1")
        .with_uri(glue_uri)
        .with_s3_endpoint(s3_endpoint)
        .with_s3_region(s3_region)
        .with_s3_path_style_access(path_style)
}

fn require_local_endpoint_env() -> E2eResult<()> {
    let glue_uri = std::env::var("NEXUS_ICEBERG_GLUE_URI").unwrap_or_else(|_| "http://localhost:5001".to_string());
    let s3_endpoint =
        std::env::var("NEXUS_ICEBERG_S3_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".to_string());
    let warehouse = local_warehouse();
    require_loopback_http_url("NEXUS_ICEBERG_GLUE_URI", &glue_uri)?;
    require_loopback_http_url("NEXUS_ICEBERG_S3_ENDPOINT", &s3_endpoint)?;
    require_s3_warehouse("NEXUS_ICEBERG_WAREHOUSE", &warehouse)?;
    Ok(())
}

fn require_loopback_http_url(name: &str, value: &str) -> E2eResult<()> {
    if is_loopback_http_url(value) {
        return Ok(());
    }
    Err(format!("{name} must be a loopback http:// URL for local Glue/Moto E2E, got {value:?}").into())
}

fn is_loopback_http_url(value: &str) -> bool {
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<std::net::IpAddr>().map(|addr| addr.is_loopback()).unwrap_or(false)
}

fn require_s3_warehouse(name: &str, value: &str) -> E2eResult<()> {
    if value.strip_prefix("s3://").and_then(|rest| rest.split('/').next()).is_some_and(|bucket| !bucket.is_empty()) {
        return Ok(());
    }
    Err(format!("{name} must be an s3://bucket URI for local Glue/Moto E2E, got {value:?}").into())
}

fn local_warehouse() -> String {
    std::env::var("NEXUS_ICEBERG_WAREHOUSE").unwrap_or_else(|_| "s3://lakehouse/warehouse".to_string())
}

fn local_env(name: &str) -> Option<String> {
    match name {
        datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some(moto_access_key()),
        datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some(moto_secret_key()),
        datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some(local_s3_access_key()),
        datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some(local_s3_secret_key()),
        _ => None,
    }
}

fn moto_access_key() -> String {
    std::env::var("MOTO_ACCESS_KEY").unwrap_or_else(|_| "moto-access".to_string())
}

fn moto_secret_key() -> String {
    std::env::var("MOTO_SECRET_KEY").unwrap_or_else(|_| "moto-secret".to_string())
}

fn local_s3_access_key() -> String {
    std::env::var(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)
        .unwrap_or_else(|_| "nexusadmin".to_string())
}

fn local_s3_secret_key() -> String {
    std::env::var(datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)
        .unwrap_or_else(|_| "nexusadmin".to_string())
}

async fn local_glue_catalog(config: &GlueIcebergCatalogConfig) -> iceberg_catalog_glue::GlueCatalog {
    let props = HashMap::from([
        (GLUE_CATALOG_PROP_URI.to_string(), config.uri.clone().expect("local Glue URI")),
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), config.warehouse.clone()),
        (AWS_REGION_NAME.to_string(), config.region.clone()),
        (AWS_ACCESS_KEY_ID.to_string(), moto_access_key()),
        (AWS_SECRET_ACCESS_KEY.to_string(), moto_secret_key()),
        (S3_ENDPOINT.to_string(), config.s3_endpoint.clone().expect("local S3 endpoint")),
        (S3_REGION.to_string(), config.s3_region.clone().unwrap_or_else(|| config.region.clone())),
        (S3_PATH_STYLE_ACCESS.to_string(), "true".to_string()),
        (S3_ACCESS_KEY_ID.to_string(), local_s3_access_key()),
        (S3_SECRET_ACCESS_KEY.to_string(), local_s3_secret_key()),
    ]);

    GlueCatalogBuilder::default()
        .load(config.catalog_handle.clone(), props)
        .await
        .expect("connect to local Moto Glue catalog")
}

fn unique_namespace() -> NamespaceIdent {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    NamespaceIdent::new(format!("nexus_glue_local_{nanos}"))
}

fn tiny_table_creation(name: &str) -> TableCreation {
    let schema = Schema::builder()
        .with_fields([
            Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))),
            Arc::new(NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))),
        ])
        .build()
        .unwrap();

    TableCreation::builder().name(name.to_string()).schema(schema).properties(HashMap::<String, String>::new()).build()
}

async fn create_tiny_table(catalog: &iceberg_catalog_glue::GlueCatalog, namespace: &NamespaceIdent, table: &str) {
    let _ = catalog.drop_namespace(namespace).await;
    catalog.create_namespace(namespace, HashMap::new()).await.expect("create local Glue namespace");
    catalog.create_table(namespace, tiny_table_creation(table)).await.expect("create local Glue table");
}

async fn glue_local_backend_and_provider_discover_table() -> E2eResult<()> {
    let config = local_glue_config();
    let catalog = local_glue_catalog(&config).await;
    let namespace = unique_namespace();
    create_tiny_table(&catalog, &namespace, "tiny_table").await;

    let result = validate_provider(&config, &namespace).await;
    cleanup_tiny_table(&catalog, &namespace, "tiny_table").await;
    result
}

async fn validate_provider(config: &GlueIcebergCatalogConfig, namespace: &NamespaceIdent) -> E2eResult<()> {
    let backend: Arc<dyn IcebergCatalogBackend> =
        Arc::new(GlueIcebergCatalogBackend::from_config(config.clone(), local_env).await?);
    let namespace_name = namespace.to_string();
    let provider = build_scoped_catalog_provider_from_backend(
        Arc::clone(&backend),
        "glue",
        &namespace_name,
        IcebergProviderKind::Gpu,
    )
    .await?;

    let schema_names = provider.schema_names();
    if schema_names != vec![namespace_name.clone()] {
        return Err(format!("expected schema {namespace_name:?}, got {schema_names:?}").into());
    }
    let schema = provider.schema(&namespace_name).ok_or_else(|| format!("schema {namespace_name} missing"))?;
    let table_names = schema.table_names();
    if table_names != vec!["tiny_table"] {
        return Err(format!("expected tiny_table, got {table_names:?}").into());
    }
    Ok(())
}

async fn cleanup_tiny_table(catalog: &iceberg_catalog_glue::GlueCatalog, namespace: &NamespaceIdent, table: &str) {
    let _ = catalog.drop_table(&iceberg::TableIdent::new(namespace.clone(), table.to_string())).await;
    let _ = catalog.drop_namespace(namespace).await;
}
