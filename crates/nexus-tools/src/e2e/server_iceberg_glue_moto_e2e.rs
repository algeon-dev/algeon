use std::{collections::HashMap, net::TcpListener, sync::Arc, time::Duration};

use arrow::array::{Int32Array, Int64Array, UInt32Array, UInt64Array};
use arrow_flight::sql::client::FlightSqlServiceClient;
use datafusion_nexus::server::{
    NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
    NEXUS_ICEBERG_NAMESPACE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
    NEXUS_ICEBERG_S3_REGION_ENV, ServerConfig, serve_with_listener,
};
use futures::TryStreamExt;
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

use crate::connection_retry::{BeforeConnect, ConnectionRetryConfig, connect_with_retry};

type E2eResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub async fn run() -> E2eResult<()> {
    require_local_glue_enabled()?;
    require_local_endpoint_env()?;
    glue_local_three_part_empty_table_count_via_flight_sql().await?;
    println!("server_iceberg_glue_moto_e2e ok");
    Ok(())
}

fn require_local_glue_enabled() -> E2eResult<()> {
    match std::env::var("NEXUS_ICEBERG_GLUE_LOCAL_E2E").as_deref() {
        Ok("1") => Ok(()),
        _ => Err("server Iceberg Glue Moto E2E requires NEXUS_ICEBERG_GLUE_LOCAL_E2E=1".into()),
    }
}

/// Local replacement for the live-AWS Glue smoke path. This uses Moto for the
/// Glue catalog API and RustFS for Iceberg metadata files, so it does not rely
/// on a developer-owned AWS account. The table is intentionally empty; this
/// test validates server config, dynamic catalog discovery, table loading, and
/// Flight SQL execution through the CPU Iceberg reader.
async fn glue_local_three_part_empty_table_count_via_flight_sql() -> E2eResult<()> {
    let namespace = unique_namespace();
    let table = "tiny_table";
    let glue_uri = std::env::var(NEXUS_ICEBERG_GLUE_URI_ENV).unwrap_or_else(|_| "http://localhost:5001".to_string());
    let s3_endpoint =
        std::env::var(NEXUS_ICEBERG_S3_ENDPOINT_ENV).unwrap_or_else(|_| "http://localhost:9000".to_string());
    let warehouse = local_warehouse();
    let local_env = local_server_env(&namespace, &glue_uri, &s3_endpoint, &warehouse);
    let _env_guard = EnvGuard::install(&local_env);
    let catalog = local_glue_catalog(&glue_uri, &s3_endpoint, &warehouse).await;
    create_tiny_table(&catalog, &namespace, table).await;

    let result = validate_flight_sql_server(&local_env, &namespace, table).await;
    cleanup_tiny_table(&catalog, &namespace, table).await;
    result
}

fn local_server_env(
    namespace: &NamespaceIdent,
    glue_uri: &str,
    s3_endpoint: &str,
    warehouse: &str,
) -> HashMap<&'static str, String> {
    HashMap::from([
        (NEXUS_ICEBERG_CATALOG_KIND_ENV, "glue".to_string()),
        (NEXUS_ICEBERG_NAMESPACE_ENV, namespace.to_string()),
        (NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, warehouse.to_string()),
        (NEXUS_ICEBERG_GLUE_URI_ENV, glue_uri.to_string()),
        (NEXUS_ICEBERG_S3_ENDPOINT_ENV, s3_endpoint.to_string()),
        (NEXUS_ICEBERG_S3_REGION_ENV, "us-east-1".to_string()),
        (NEXUS_ICEBERG_S3_PATH_STYLE_ENV, "true".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_REGION_ENV, "us-east-1".to_string()),
        (datafusion_nexus::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV, moto_access_key()),
        (datafusion_nexus::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV, moto_secret_key()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, local_s3_access_key()),
        (datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, local_s3_secret_key()),
    ])
}

async fn validate_flight_sql_server(
    local_env: &HashMap<&'static str, String>,
    namespace: &NamespaceIdent,
    table: &str,
) -> E2eResult<()> {
    let (listener, addr) = free_local_addr();
    let mut config = ServerConfig::from_env_get(|name| local_env.get(name).cloned())?;
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_flight_sql(addr).await;

    let sql = format!("SELECT count(*) AS row_count FROM glue.\"{}\".{table}", namespace);
    let row_count = query_count(&mut client, &sql).await?;
    if row_count != 0 {
        server.abort();
        return Err(format!("newly-created local Glue table should be empty, got {row_count}").into());
    }

    server.abort();
    Ok(())
}

async fn cleanup_tiny_table(catalog: &iceberg_catalog_glue::GlueCatalog, namespace: &NamespaceIdent, table: &str) {
    let _ = catalog.drop_table(&iceberg::TableIdent::new(namespace.clone(), table.to_string())).await;
    let _ = catalog.drop_namespace(namespace).await;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn query_count(client: &mut FlightSqlServiceClient<tonic::transport::Channel>, sql: &str) -> E2eResult<i64> {
    let flight_info = client.execute(sql.to_string(), None).await?;
    let ticket =
        flight_info.endpoint.first().and_then(|ep| ep.ticket.clone()).ok_or("Flight SQL response missing ticket")?;
    let batches = client.do_get(ticket).await?.try_collect::<Vec<_>>().await?;
    if batches.len() != 1 || batches[0].num_rows() != 1 {
        return Err(format!(
            "expected one single-row batch for count query, got {} batches and first rows {:?}",
            batches.len(),
            batches.first().map(|batch| batch.num_rows())
        )
        .into());
    }
    int_value(batches[0].column(0)).ok_or_else(|| "could not read count from result".into())
}

/// Warm the process-wide CUDA context exactly once. The first CUDA call pays a
/// multi-second `cuInit()` cost; doing it before spawning the server keeps that
/// cost off the connect window, and the `OnceLock` collapses concurrent callers
/// onto a single init.
fn warm_cuda_once() {
    static CUDA_WARM: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CUDA_WARM.get_or_init(|| {
        // Best-effort: serve_with_listener re-runs (and surfaces) any error.
        let _ = datafusion_nexus::server::init_cuda_device(0);
    });
}

/// Reserve a loopback port and hand back the live listener so the caller can
/// pass it straight to `serve_with_listener`. Keeping the socket bound closes
/// the bind/serve gap where another process could steal the port.
fn free_local_addr() -> (TcpListener, std::net::SocketAddr) {
    warm_cuda_once();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    (listener, addr)
}

async fn connect_flight_sql(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    connect_with_retry(
        &uri,
        ConnectionRetryConfig::exponential_timeout(
            Duration::from_secs(15),
            Duration::from_millis(10),
            Duration::from_millis(500),
        ),
        || Ok(BeforeConnect::Continue),
        FlightSqlServiceClient::new,
        |error| format!("Flight SQL server did not accept connections at {addr} within 15s: {error:?}"),
    )
    .await
    .unwrap_or_else(|error| panic!("{error}"))
}

fn int_value(array: &arrow::array::ArrayRef) -> Option<i64> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|a| a.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|a| i64::from(a.value(0))))
        .or_else(|| array.as_any().downcast_ref::<UInt64Array>().map(|a| a.value(0) as i64))
        .or_else(|| array.as_any().downcast_ref::<UInt32Array>().map(|a| i64::from(a.value(0))))
}

fn unique_namespace() -> NamespaceIdent {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    NamespaceIdent::new(format!("nexus_glue_server_local_{nanos}"))
}

async fn local_glue_catalog(glue_uri: &str, s3_endpoint: &str, warehouse: &str) -> iceberg_catalog_glue::GlueCatalog {
    let props = HashMap::from([
        (GLUE_CATALOG_PROP_URI.to_string(), glue_uri.to_string()),
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), warehouse.to_string()),
        (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
        (AWS_ACCESS_KEY_ID.to_string(), moto_access_key()),
        (AWS_SECRET_ACCESS_KEY.to_string(), moto_secret_key()),
        (S3_ENDPOINT.to_string(), s3_endpoint.to_string()),
        (S3_REGION.to_string(), "us-east-1".to_string()),
        (S3_PATH_STYLE_ACCESS.to_string(), "true".to_string()),
        (S3_ACCESS_KEY_ID.to_string(), local_s3_access_key()),
        (S3_SECRET_ACCESS_KEY.to_string(), local_s3_secret_key()),
    ]);

    GlueCatalogBuilder::default().load("glue", props).await.expect("connect to local Moto Glue catalog")
}

fn require_local_endpoint_env() -> E2eResult<()> {
    let glue_uri = std::env::var(NEXUS_ICEBERG_GLUE_URI_ENV).unwrap_or_else(|_| "http://localhost:5001".to_string());
    let s3_endpoint =
        std::env::var(NEXUS_ICEBERG_S3_ENDPOINT_ENV).unwrap_or_else(|_| "http://localhost:9000".to_string());
    let warehouse = local_warehouse();
    require_loopback_http_url(NEXUS_ICEBERG_GLUE_URI_ENV, &glue_uri)?;
    require_loopback_http_url(NEXUS_ICEBERG_S3_ENDPOINT_ENV, &s3_endpoint)?;
    require_s3_warehouse(NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, &warehouse)?;
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
    std::env::var(NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV).unwrap_or_else(|_| "s3://lakehouse/warehouse".to_string())
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

struct EnvGuard {
    original: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn install(values: &HashMap<&'static str, String>) -> Self {
        let mut names: Vec<&'static str> = values.keys().copied().collect();
        names.extend([
            datafusion_nexus::table_format::iceberg::AWS_SESSION_TOKEN_ENV,
            datafusion_nexus::table_format::iceberg::AWS_SECURITY_TOKEN_ENV,
            datafusion_nexus::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
        ]);
        names.sort_unstable();
        names.dedup();

        let original = names.iter().map(|name| (*name, std::env::var(name).ok())).collect::<Vec<_>>();
        for name in names {
            if let Some(value) = values.get(name) {
                // This single-purpose E2E binary installs process env before
                // starting any background work that reads Iceberg credentials.
                unsafe { std::env::set_var(name, value) };
            } else {
                // Clear ambient session tokens so Moto validation cannot mix
                // developer AWS session state with local static credentials.
                unsafe { std::env::remove_var(name) };
            }
        }
        Self { original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.original.iter().rev() {
            match value {
                Some(value) => {
                    // Restore process env after the local server has been
                    // aborted; no other threads in this tool should read it.
                    unsafe { std::env::set_var(name, value) };
                },
                None => {
                    unsafe { std::env::remove_var(name) };
                },
            }
        }
    }
}
