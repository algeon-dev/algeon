//! Local AWS Glue-compatible E2E runner backed by Moto + RustFS.
//!
//! This exercises the real `GlueCatalogBackend` without requiring
//! a developer-owned AWS Glue catalog. Bring up `fixture/iceberg-local` first,
//! then run with `NEXUS_ICEBERG_GLUE_LOCAL_E2E=1`.

use std::{collections::HashMap, sync::Arc};

use datafusion::catalog::CatalogProvider;
use datafusion_nexus::table_format::iceberg::{
    SourceAccessPolicy,
    catalog::{
        CatalogBackend, GlueCatalogBackend, GlueCatalogConfig, NexusIcebergCatalogSource,
        build_scoped_catalog_provider_from_backend,
    },
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
    let fixture = LocalGlueFixture::from_env();
    require_local_endpoint_env(&fixture)?;
    glue_local_backend_and_provider_discover_table(&fixture).await?;
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

#[derive(Debug, Clone)]
struct LocalGlueFixture {
    glue_uri: String,
    warehouse: String,
    s3_endpoint: String,
    s3_region: String,
    path_style: bool,
    catalog_access_key: String,
    catalog_secret_key: String,
    s3_access_key: String,
    s3_secret_key: String,
}

impl LocalGlueFixture {
    fn from_env() -> Self {
        Self {
            glue_uri: std::env::var("NEXUS_ICEBERG_GLUE_URI").unwrap_or_else(|_| "http://localhost:5001".to_string()),
            warehouse: local_warehouse(),
            s3_endpoint: std::env::var("NEXUS_ICEBERG_S3_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9000".to_string()),
            s3_region: std::env::var("NEXUS_ICEBERG_S3_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
            path_style: std::env::var("NEXUS_ICEBERG_S3_PATH_STYLE").map(|value| value == "true").unwrap_or(true),
            catalog_access_key: moto_access_key(),
            catalog_secret_key: moto_secret_key(),
            s3_access_key: local_s3_access_key(),
            s3_secret_key: local_s3_secret_key(),
        }
    }

    fn catalog_config(&self) -> GlueCatalogConfig {
        let source_access = SourceAccessPolicy::S3(
            SourceAccessPolicy::s3_custom_endpoint(
                self.s3_endpoint.clone(),
                Some(self.s3_region.clone()),
                Some(self.path_style),
            )
            .as_s3()
            .clone()
            .with_static_credentials(self.s3_access_key.clone(), self.s3_secret_key.clone(), None),
        );
        GlueCatalogConfig::new("glue", self.warehouse.clone(), "us-east-1")
            .with_uri(self.glue_uri.clone())
            .with_catalog_static_credentials(self.catalog_access_key.clone(), self.catalog_secret_key.clone(), None)
            .with_source_access_policy(source_access)
    }
}

fn require_local_endpoint_env(fixture: &LocalGlueFixture) -> E2eResult<()> {
    require_loopback_http_url("NEXUS_ICEBERG_GLUE_URI", &fixture.glue_uri)?;
    require_loopback_http_url("NEXUS_ICEBERG_S3_ENDPOINT", &fixture.s3_endpoint)?;
    require_s3_warehouse("NEXUS_ICEBERG_WAREHOUSE", &fixture.warehouse)?;
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

async fn local_glue_catalog(fixture: &LocalGlueFixture) -> iceberg_catalog_glue::GlueCatalog {
    let props = HashMap::from([
        (GLUE_CATALOG_PROP_URI.to_string(), fixture.glue_uri.clone()),
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), fixture.warehouse.clone()),
        (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
        (AWS_ACCESS_KEY_ID.to_string(), fixture.catalog_access_key.clone()),
        (AWS_SECRET_ACCESS_KEY.to_string(), fixture.catalog_secret_key.clone()),
        (S3_ENDPOINT.to_string(), fixture.s3_endpoint.clone()),
        (S3_REGION.to_string(), fixture.s3_region.clone()),
        (S3_PATH_STYLE_ACCESS.to_string(), fixture.path_style.to_string()),
        (S3_ACCESS_KEY_ID.to_string(), fixture.s3_access_key.clone()),
        (S3_SECRET_ACCESS_KEY.to_string(), fixture.s3_secret_key.clone()),
    ]);

    GlueCatalogBuilder::default().load("glue", props).await.expect("connect to local Moto Glue catalog")
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

async fn glue_local_backend_and_provider_discover_table(fixture: &LocalGlueFixture) -> E2eResult<()> {
    let config = fixture.catalog_config();
    let catalog = local_glue_catalog(fixture).await;
    let namespace = unique_namespace();
    create_tiny_table(&catalog, &namespace, "tiny_table").await;

    let result = validate_provider(&config, &namespace).await;
    cleanup_tiny_table(&catalog, &namespace, "tiny_table").await;
    result
}

async fn validate_provider(config: &GlueCatalogConfig, namespace: &NamespaceIdent) -> E2eResult<()> {
    let backend: Arc<dyn CatalogBackend> =
        Arc::new(NexusIcebergCatalogSource::new(GlueCatalogBackend::connect(config.clone()).await?.into_backend()));
    let namespace_name = namespace.to_string();
    let provider = build_scoped_catalog_provider_from_backend(Arc::clone(&backend), "glue", &namespace_name).await?;

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
