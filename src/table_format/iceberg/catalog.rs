//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`IcebergCatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`] or
//! [`build_scoped_catalog_provider_from_backend`]; individual tables are loaded
//! lazily on first access.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error as StdError,
    fmt,
    sync::{Arc, RwLock as StdRwLock},
    time::Duration,
};

use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::DataFusionError,
    datasource::TableProvider,
};
use iceberg::{
    Catalog, CatalogBuilder as _, NamespaceIdent, TableIdent,
    io::{S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_ENDPOINT, S3_PATH_STYLE_ACCESS, S3_REGION},
};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_URI, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};
use iceberg_catalog_rest::{REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder};
use iceberg_storage_opendal::OpenDalStorageFactory;
use nexus_query_engine::{S3SourceStorageProfile, SourceStorageProfile};
use tokio::sync::RwLock;

use crate::{
    Error, ErrorCode, Result,
    table_format::scan::{
        DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, TableFormatParquetReadStrategyConfig,
        credential_free_endpoint_identity,
    },
};

const CATALOG_BACKEND_OP: &str = "datafusion_nexus.iceberg.catalog_backend";

fn catalog_backend_error(message: &'static str, catalog_type: &'static str, source: iceberg::Error) -> Error {
    let aws_facts = AwsCatalogErrorFacts::from_iceberg_error(&source);
    let mut error = Error::new(catalog_backend_error_code(&source, aws_facts.as_ref()), CATALOG_BACKEND_OP, message)
        .with_fact("catalog_type", catalog_type)
        .with_fact("iceberg_error_kind", source.kind().into_static())
        .with_fact("iceberg_retryable", source.retryable());
    if let Some(aws_facts) = aws_facts {
        error = aws_facts.apply(error);
    }
    error.with_source(source)
}

fn catalog_backend_error_code(source: &iceberg::Error, aws_facts: Option<&AwsCatalogErrorFacts>) -> ErrorCode {
    if let Some(aws_facts) = aws_facts {
        if matches!(
            aws_facts.error_code,
            Some("ExpiredTokenException" | "InvalidAccessKeyId" | "UnrecognizedClientException")
        ) || matches!(aws_facts.detail, Some("http_401" | "missing_credentials"))
        {
            return ErrorCode::SourceUnauthenticated;
        }
        if matches!(aws_facts.error_code, Some("AccessDeniedException" | "SignatureDoesNotMatch"))
            || matches!(aws_facts.detail, Some("http_403" | "access_denied"))
        {
            return ErrorCode::SourceAccessDenied;
        }
        if matches!(aws_facts.error_code, Some("EntityNotFoundException")) {
            return ErrorCode::SourceObjectNotFound;
        }
    }
    match source.kind() {
        iceberg::ErrorKind::TableNotFound | iceberg::ErrorKind::NamespaceNotFound => ErrorCode::ObjectNotFound,
        _ if source.retryable() => ErrorCode::SourceTemporaryFailure,
        _ => ErrorCode::CudfDependencyFailure,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AwsCatalogErrorFacts {
    error_code: Option<&'static str>,
    request_id: Option<String>,
    http_status: Option<&'static str>,
    retryable: Option<&'static str>,
    detail: Option<&'static str>,
}

impl AwsCatalogErrorFacts {
    fn from_iceberg_error(source: &iceberg::Error) -> Option<Self> {
        let text = error_chain_text(source);
        if !text.contains("aws sdk error") && !text.contains("ErrorMetadata") && !text.contains("aws_request_id") {
            return None;
        }
        let error_code = aws_error_code_fact(&text);
        let detail = aws_error_detail_fact(&text, error_code);
        Some(Self {
            error_code,
            request_id: aws_request_id_fact(&text),
            http_status: aws_http_status_fact(&text),
            retryable: aws_retryable_fact(&text),
            detail,
        })
    }

    fn apply(self, mut error: Error) -> Error {
        if let Some(error_code) = self.error_code {
            error = error.with_fact("aws_error_code", error_code);
        }
        if let Some(request_id) = self.request_id {
            error = error.with_fact("aws_request_id", request_id);
        }
        if let Some(http_status) = self.http_status {
            error = error.with_fact("aws_http_status", http_status);
        }
        if let Some(retryable) = self.retryable {
            error = error.with_fact("aws_retryable", retryable);
        }
        if let Some(detail) = self.detail {
            error = error.with_fact("aws_error_detail", detail);
        }
        error
    }
}

fn error_chain_text(source: &(dyn StdError + 'static)) -> String {
    let mut text = source.to_string();
    let mut current = source.source();
    while let Some(source) = current {
        text.push_str(": ");
        text.push_str(&source.to_string());
        current = source.source();
    }
    text
}

fn aws_error_code_fact(text: &str) -> Option<&'static str> {
    if contains_any(text, &["ExpiredTokenException", "ExpiredToken"]) {
        Some("ExpiredTokenException")
    } else if text.contains("InvalidAccessKeyId") {
        Some("InvalidAccessKeyId")
    } else if text.contains("UnrecognizedClientException") {
        Some("UnrecognizedClientException")
    } else if text.contains("AccessDeniedException") || text.contains("AccessDenied") {
        Some("AccessDeniedException")
    } else if text.contains("SignatureDoesNotMatch") {
        Some("SignatureDoesNotMatch")
    } else if text.contains("EntityNotFoundException") {
        Some("EntityNotFoundException")
    } else {
        None
    }
}

fn aws_error_detail_fact(text: &str, error_code: Option<&'static str>) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    match error_code {
        Some("ExpiredTokenException") => return Some("expired_token"),
        Some("InvalidAccessKeyId") => return Some("invalid_access_key"),
        Some("UnrecognizedClientException") => return Some("invalid_client"),
        Some("AccessDeniedException") => return Some("access_denied"),
        Some("SignatureDoesNotMatch") => return Some("signature_mismatch"),
        Some("EntityNotFoundException") => return Some("not_found"),
        _ => {},
    }
    if contains_any(&lower, &["401 unauthorized", "status code: 401", "http 401"]) {
        Some("http_401")
    } else if contains_any(&lower, &["403 forbidden", "status code: 403", "http 403"]) {
        Some("http_403")
    } else if contains_any(&lower, &["no credentials", "missing credentials", "could not load credentials"]) {
        Some("missing_credentials")
    } else {
        None
    }
}

fn aws_request_id_fact(text: &str) -> Option<String> {
    extract_quoted_marker(text, "aws_request_id\": \"")
        .or_else(|| extract_quoted_marker(text, "aws_request_id: \""))
        .or_else(|| extract_value_after(text, "request id: "))
}

fn aws_http_status_fact(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if contains_any(&lower, &["status code: 401", "http 401", "401 unauthorized"]) {
        Some("401")
    } else if contains_any(&lower, &["status code: 403", "http 403", "403 forbidden"]) {
        Some("403")
    } else if contains_any(&lower, &["status code: 404", "http 404", "404 not found"]) {
        Some("404")
    } else {
        None
    }
}

fn aws_retryable_fact(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("retryable: true") {
        Some("true")
    } else if lower.contains("retryable: false") {
        Some("false")
    } else {
        None
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn extract_quoted_marker(text: &str, marker: &str) -> Option<String> {
    let start = text.find(marker)? + marker.len();
    let value = &text[start..];
    let end = value.find('"')?;
    let value = &value[..end];
    (!value.is_empty()).then(|| value.to_string())
}

fn extract_value_after(text: &str, marker: &str) -> Option<String> {
    let start = text.to_ascii_lowercase().find(marker)? + marker.len();
    let value = text[start..]
        .split(|ch: char| ch == ',' || ch == ')' || ch == '}' || ch.is_whitespace())
        .next()
        .unwrap_or("")
        .trim_matches('"');
    (!value.is_empty()).then(|| value.to_string())
}

async fn catalog_backend_list_namespaces(
    catalog: &Arc<dyn Catalog>,
    catalog_type: &'static str,
    message: &'static str,
) -> Result<Vec<NamespaceIdent>> {
    catalog.list_namespaces(None).await.map_err(|error| catalog_backend_error(message, catalog_type, error))
}

async fn catalog_backend_list_tables(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    catalog_type: &'static str,
    message: &'static str,
) -> Result<Vec<TableIdent>> {
    catalog.list_tables(namespace).await.map_err(|error| {
        catalog_backend_error(message, catalog_type, error).with_fact("namespace", namespace.to_string())
    })
}

async fn catalog_backend_load_table(
    catalog: &Arc<dyn Catalog>,
    table: &TableIdent,
    catalog_type: &'static str,
    message: &'static str,
) -> Result<iceberg::table::Table> {
    catalog.load_table(table).await.map_err(|error| {
        catalog_backend_error(message, catalog_type, error)
            .with_fact("namespace", table.namespace().to_string())
            .with_fact("table", table.name())
    })
}

#[derive(Clone, Copy)]
struct S3FileIoConfig<'a> {
    endpoint: Option<&'a str>,
    region: Option<&'a str>,
    path_style_access: Option<bool>,
}

impl<'a> S3FileIoConfig<'a> {
    fn from_glue(config: &'a GlueIcebergCatalogConfig) -> Self {
        Self {
            endpoint: config.s3_endpoint.as_deref(),
            region: config.s3_region.as_deref(),
            path_style_access: config.s3_path_style_access,
        }
    }

    fn from_rest(config: &'a RestIcebergCatalogConfig) -> Self {
        Self {
            endpoint: config.s3_endpoint.as_deref(),
            region: config.s3_region.as_deref(),
            path_style_access: config.s3_path_style_access,
        }
    }

    fn apply_parquet_read_strategy_config(
        self,
        mut config: TableFormatParquetReadStrategyConfig,
        force_scoped_resolved: bool,
    ) -> TableFormatParquetReadStrategyConfig {
        config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
        if let Some(endpoint) = self.non_empty_endpoint() {
            config.storage_endpoint = Some(credential_free_endpoint_identity(endpoint));
        }
        if force_scoped_resolved
            || self.non_empty_region().is_some()
            || self.non_empty_endpoint().is_some()
            || self.path_style_access.is_some()
        {
            config.credential_mode = nexus_query_engine::CredentialMode::ScopedResolved;
        }
        if self.non_empty_endpoint().is_some() && self.path_style_access == Some(false) {
            config.remote_direct_profile_unsupported_reason = Some("remote_direct_virtual_hosted_endpoint_unsupported");
        }
        config
    }

    fn apply_catalog_props(self, props: &mut HashMap<String, String>) {
        if let Some(endpoint) = self.non_empty_endpoint() {
            props.insert(S3_ENDPOINT.to_string(), endpoint.to_string());
        }
        if let Some(region) = self.non_empty_region() {
            props.insert(S3_REGION.to_string(), region.to_string());
        }
        if let Some(path_style) = self.path_style_access {
            props.insert(S3_PATH_STYLE_ACCESS.to_string(), path_style.to_string());
        }
    }

    fn apply_source_storage_profile(
        self,
        mut s3: S3SourceStorageProfile,
        fallback_region: Option<&str>,
    ) -> (S3SourceStorageProfile, bool) {
        if let Some(region) = self.non_empty_region().or(fallback_region) {
            s3 = s3.with_region(region.to_string());
        }
        let has_s3_endpoint = self.non_empty_endpoint().is_some();
        if let Some(endpoint) = self.non_empty_endpoint() {
            s3 = s3.with_endpoint(endpoint.to_string());
        }
        if let Some(path_style) = self.path_style_access {
            s3 = s3.with_path_style_access(path_style);
        }
        (s3, has_s3_endpoint)
    }

    fn non_empty_endpoint(self) -> Option<&'a str> {
        self.endpoint.filter(|endpoint| !endpoint.trim().is_empty())
    }

    fn non_empty_region(self) -> Option<&'a str> {
        self.region.filter(|region| !region.trim().is_empty())
    }
}

#[cfg(feature = "server")]
fn log_iceberg_metadata_lookup(
    operation: &'static str,
    catalog_type: &str,
    namespace: &NamespaceIdent,
    table: Option<&str>,
    elapsed: Duration,
    cache_hit: Option<bool>,
    outcome: &'static str,
) {
    tracing::debug!(
        operation,
        catalog_type,
        namespace = %namespace.as_ref().join("."),
        table = table.unwrap_or(""),
        elapsed_ms = elapsed.as_millis(),
        cache_hit = ?cache_hit,
        outcome,
        "Iceberg catalog metadata lookup"
    );
}

/// Which DataFusion [`TableProvider`] implementation to wrap loaded Iceberg
/// tables in.
///
/// The dynamic catalog stays neutral — the choice is decided once at server
/// startup based on [`crate::server::config::ServerSessionProfile`] (when the
/// server is built with both `iceberg` and `iceberg-cpu`).
///
/// `Gpu` wraps loaded tables in the adapter's native
/// `CudfIcebergTableProvider` (GPU-only `CudfTableFormatScanExec` leaf).
/// `Cpu` wraps them in upstream `iceberg-datafusion`'s
/// `IcebergStaticTableProvider`, which provides a real CPU Arrow reader and
/// is only available when the `iceberg-cpu` feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergProviderKind {
    /// Native GPU planning leaf via [`crate::table_format::iceberg::CudfIcebergTableProvider`].
    Gpu,
    /// CPU Arrow reader via `iceberg_datafusion::IcebergStaticTableProvider`.
    ///
    /// Requires the `iceberg-cpu` feature. Building [`CudfIcebergSchemaProvider`]
    /// with this variant when the feature is off returns an error at startup
    /// rather than at first-query time.
    Cpu,
}

impl IcebergProviderKind {
    /// Returns a stable string label suitable for tracing.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Cpu => "cpu",
        }
    }
}

/// Options applied when wrapping loaded Iceberg tables in DataFusion table providers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IcebergProviderOptions {
    /// Optional wall-clock budget for GPU table-format scan planning.
    pub scan_planning_timeout: Option<Duration>,
}

impl IcebergProviderOptions {
    /// Sets the optional GPU scan-planning timeout.
    pub fn with_scan_planning_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.scan_planning_timeout = timeout;
        self
    }
}

/// Wraps a loaded Iceberg table in the [`TableProvider`] selected by `kind`.
///
/// `Gpu` always succeeds (the native provider has no runtime preconditions).
/// `Cpu` requires the `iceberg-cpu` feature; without it this returns
/// [`ErrorCode::CudfInvalidInput`] so server startup fails loudly instead of
/// silently downgrading.
async fn build_table_provider(
    table: iceberg::table::Table,
    kind: IcebergProviderKind,
    catalog_type: &str,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    provider_options: &IcebergProviderOptions,
) -> Result<Arc<dyn TableProvider>> {
    match kind {
        IcebergProviderKind::Gpu => {
            let provider = match parquet_read_strategy_config {
                Some(config) => {
                    crate::table_format::iceberg::CudfIcebergTableProvider::try_new_with_parquet_read_strategy_config(
                        table, config,
                    )?
                },
                None => crate::table_format::iceberg::CudfIcebergTableProvider::try_new(table)?,
            };
            Ok(Arc::new(
                provider
                    .with_catalog_type(catalog_type)
                    .with_scan_planning_timeout(provider_options.scan_planning_timeout),
            ))
        },
        IcebergProviderKind::Cpu => {
            #[cfg(feature = "iceberg-cpu")]
            {
                let _ = catalog_type;
                let provider = iceberg_datafusion::IcebergStaticTableProvider::try_new_from_table(table)
                    .await
                    .map_err(|error| {
                        Error::new(
                            ErrorCode::CudfDependencyFailure,
                            CATALOG_BACKEND_OP,
                            "failed to build iceberg-datafusion CPU table provider",
                        )
                        .with_source(error)
                    })?;
                Ok(Arc::new(provider))
            }
            #[cfg(not(feature = "iceberg-cpu"))]
            {
                let _ = (table, catalog_type);
                Err(Error::new(
                    ErrorCode::CudfInvalidInput,
                    CATALOG_BACKEND_OP,
                    "IcebergProviderKind::Cpu requires the `iceberg-cpu` feature to be enabled at build time",
                ))
            }
        },
    }
}

/// Backend-neutral Iceberg catalog listing and table-loading interface.
///
/// All methods are `async` because catalog operations require network calls.
/// DataFusion's synchronous [`CatalogProvider`] and [`SchemaProvider`] methods
/// read from a startup cache populated by
/// [`build_catalog_provider_from_backend`] or
/// [`build_scoped_catalog_provider_from_backend`]; [`SchemaProvider::table`]
/// (which *is* async) loads tables lazily.
///
/// Implementations should avoid leaking credentials in errors. `load_table`
/// must accept identifiers previously returned by `list_tables`.
#[async_trait]
pub trait IcebergCatalogBackend: fmt::Debug + Send + Sync {
    /// List all top-level namespaces exposed by this catalog.
    async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>>;

    /// List all tables in the given namespace.
    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>>;

    /// Load the Iceberg table for the given identifier.
    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table>;

    /// Table-format Parquet read-strategy config associated with this catalog.
    ///
    /// Backends that know S3 FileIO endpoint/region/addressing facts return a
    /// credential-free config so GPU scan facts can reference the matching
    /// execution-time storage profile. Secret material is registered only on the
    /// native execution config.
    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        None
    }
}

/// DataFusion [`CatalogProvider`] backed by a startup-cached Iceberg namespace
/// list.
///
/// Built by [`build_catalog_provider_from_backend`] or
/// [`build_scoped_catalog_provider_from_backend`]. Each schema maps to one
/// Iceberg namespace; individual tables are loaded lazily through the backend
/// when first queried via [`SchemaProvider::table`].
#[derive(Debug)]
pub struct CudfIcebergCatalogProvider {
    schemas: BTreeMap<String, Arc<CudfIcebergSchemaProvider>>,
}

impl CudfIcebergCatalogProvider {
    fn new(schemas: BTreeMap<String, Arc<CudfIcebergSchemaProvider>>) -> Self {
        Self { schemas }
    }

    /// Returns the concrete schema provider for `name`, if it exists.
    ///
    /// Use this to seed pre-loaded alias tables before registering the catalog
    /// in a [`datafusion::prelude::SessionContext`].
    pub fn schema_concrete(&self, name: &str) -> Option<Arc<CudfIcebergSchemaProvider>> {
        self.schemas.get(name).cloned()
    }
}

impl CatalogProvider for CudfIcebergCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).map(|s| Arc::clone(s) as Arc<dyn SchemaProvider>)
    }
}

/// DataFusion [`SchemaProvider`] backed by a single Iceberg namespace with lazy
/// table loading.
///
/// `table_names()` returns the list discovered at startup.  `table()` loads
/// the Iceberg table on first access through the backend and caches the result.
/// The cache has no refresh policy; concurrent first access can call the
/// backend more than once, with the first inserted provider retained.
#[derive(Debug)]
pub struct CudfIcebergSchemaProvider {
    namespace: NamespaceIdent,
    known_table_names: Vec<String>,
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    provider_kind: IcebergProviderKind,
    provider_options: IcebergProviderOptions,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    table_cache: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
    seeded_table_names: StdRwLock<BTreeSet<String>>,
}

impl CudfIcebergSchemaProvider {
    fn new(
        namespace: NamespaceIdent,
        known_table_names: Vec<String>,
        backend: Arc<dyn IcebergCatalogBackend>,
        catalog_type: impl Into<String>,
        provider_kind: IcebergProviderKind,
        provider_options: IcebergProviderOptions,
        parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    ) -> Self {
        Self {
            namespace,
            known_table_names,
            backend,
            catalog_type: catalog_type.into(),
            provider_kind,
            provider_options,
            parquet_read_strategy_config,
            table_cache: RwLock::new(HashMap::new()),
            seeded_table_names: StdRwLock::new(BTreeSet::new()),
        }
    }

    /// Seeds a pre-loaded table provider under `name`.
    ///
    /// Call this before registering the parent [`CudfIcebergCatalogProvider`]
    /// in the session to inject alias mappings without triggering a backend
    /// load.  A name already in the cache is left unchanged (first write wins).
    /// Seeded names do not appear in `table_names()`.
    pub async fn seed_table(&self, name: String, provider: Arc<dyn TableProvider>) {
        let mut cache = self.table_cache.write().await;
        cache.entry(name.clone()).or_insert(provider);
        if let Ok(mut seeded) = self.seeded_table_names.write() {
            seeded.insert(name);
        }
    }

    /// Returns whether `name` is either startup-discovered or already seeded in
    /// the synchronous cache. This exists for catalog-overlay collision checks
    /// that cannot await [`SchemaProvider::table`].
    #[cfg(feature = "server")]
    pub(crate) fn known_or_seeded_table_exists(&self, name: &str) -> bool {
        self.table_exist(name) || self.seeded_table_names.read().is_ok_and(|seeded| seeded.contains(name))
    }

    /// Returns startup-discovered table names plus any synchronously visible
    /// seeded aliases.
    #[cfg(feature = "server")]
    pub(crate) fn known_or_seeded_table_names(&self) -> Vec<String> {
        let mut names = self.known_table_names.iter().cloned().collect::<BTreeSet<_>>();
        if let Ok(seeded) = self.seeded_table_names.read() {
            names.extend(seeded.iter().cloned());
        }
        names.into_iter().collect()
    }
}

#[async_trait]
impl SchemaProvider for CudfIcebergSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        self.known_table_names.clone()
    }

    async fn table(&self, name: &str) -> datafusion::common::Result<Option<Arc<dyn TableProvider>>> {
        #[cfg(feature = "server")]
        let lookup_started_at = std::time::Instant::now();

        // Fast path: check the cache (covers pre-seeded aliases and already-loaded tables).
        {
            let cache = self.table_cache.read().await;
            if let Some(provider) = cache.get(name) {
                #[cfg(feature = "server")]
                log_iceberg_metadata_lookup(
                    "load_table",
                    &self.catalog_type,
                    &self.namespace,
                    Some(name),
                    lookup_started_at.elapsed(),
                    Some(true),
                    "hit",
                );
                return Ok(Some(Arc::clone(provider)));
            }
        }

        // Only load tables that were discovered during startup.
        if !self.table_exist(name) {
            #[cfg(feature = "server")]
            log_iceberg_metadata_lookup(
                "load_table",
                &self.catalog_type,
                &self.namespace,
                Some(name),
                lookup_started_at.elapsed(),
                Some(false),
                "not_found",
            );
            return Ok(None);
        }

        let ident = TableIdent::new(self.namespace.clone(), name.to_string());
        #[cfg(feature = "server")]
        let load_started_at = std::time::Instant::now();
        let table = match self.backend.load_table(&ident).await {
            Ok(table) => {
                #[cfg(feature = "server")]
                log_iceberg_metadata_lookup(
                    "load_table",
                    &self.catalog_type,
                    &self.namespace,
                    Some(name),
                    load_started_at.elapsed(),
                    Some(false),
                    "loaded",
                );
                table
            },
            Err(error) => {
                #[cfg(feature = "server")]
                log_iceberg_metadata_lookup(
                    "load_table",
                    &self.catalog_type,
                    &self.namespace,
                    Some(name),
                    load_started_at.elapsed(),
                    Some(false),
                    "error",
                );
                return Err(DataFusionError::from(error));
            },
        };
        let provider = build_table_provider(
            table,
            self.provider_kind,
            &self.catalog_type,
            self.parquet_read_strategy_config.clone(),
            &self.provider_options,
        )
        .await
        .map_err(DataFusionError::from)?;

        // Insert into cache; under concurrent first-access races both paths
        // produce equivalent providers — first write wins, the loser's provider
        // is discarded, subsequent callers see the winner.
        {
            let mut cache = self.table_cache.write().await;
            cache.entry(name.to_string()).or_insert_with(|| Arc::clone(&provider));
        }

        Ok(Some(provider))
    }

    /// Returns the table type without loading the full Iceberg metadata.
    ///
    /// Startup-discovered names always resolve to `TableType::Base`: the
    /// Iceberg backend only enumerates physical Iceberg tables, so this
    /// shortcut avoids the round-trip the default `table_type()` would do via
    /// `table()`. For seeded providers we defer to `TableProvider::table_type()`
    /// so callers that seed a view (or anything else non-base) are reported
    /// honestly.
    async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<datafusion::datasource::TableType>> {
        if self.table_exist(name) {
            return Ok(Some(datafusion::datasource::TableType::Base));
        }
        let cache = self.table_cache.read().await;
        Ok(cache.get(name).map(|provider| provider.table_type()))
    }

    fn table_exist(&self, name: &str) -> bool {
        self.known_table_names.iter().any(|n| n == name)
    }
}

/// Builds a [`CudfIcebergCatalogProvider`] by discovering all namespaces and
/// their table names from `backend` at startup.
///
/// This populates the synchronous `schema_names()` / `table_names()` caches.
/// Individual tables are loaded lazily on first access via `table()`.
///
/// Multi-level namespace identifiers (two or more parts) are joined with `.`
/// to form a flat DataFusion schema name. If two namespaces flatten to the
/// same name, the later insertion replaces the earlier one.
pub async fn build_catalog_provider_from_backend(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    provider_kind: IcebergProviderKind,
) -> Result<CudfIcebergCatalogProvider> {
    build_catalog_provider_from_backend_with_options(
        backend,
        catalog_type,
        provider_kind,
        IcebergProviderOptions::default(),
    )
    .await
}

/// Builds a [`CudfIcebergCatalogProvider`] with explicit provider options.
pub async fn build_catalog_provider_from_backend_with_options(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    provider_kind: IcebergProviderKind,
    provider_options: IcebergProviderOptions,
) -> Result<CudfIcebergCatalogProvider> {
    let catalog_type = catalog_type.into();
    let namespaces = backend.list_namespaces().await?;
    build_catalog_provider_for_namespaces(backend, catalog_type, namespaces, provider_kind, provider_options).await
}

/// Builds a [`CudfIcebergCatalogProvider`] scoped to one configured namespace.
///
/// Unlike [`build_catalog_provider_from_backend`], this does not enumerate all
/// namespaces the backend can see. It lists tables only for `namespace`, so the
/// resulting DataFusion catalog exposes a single schema and information-schema
/// queries cannot fan out into unrelated namespaces.
///
/// Dotted namespace strings are interpreted as multi-level Iceberg namespaces.
pub async fn build_scoped_catalog_provider_from_backend(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    namespace: &str,
    provider_kind: IcebergProviderKind,
) -> Result<CudfIcebergCatalogProvider> {
    build_scoped_catalog_provider_from_backend_with_options(
        backend,
        catalog_type,
        namespace,
        provider_kind,
        IcebergProviderOptions::default(),
    )
    .await
}

/// Builds a namespace-scoped [`CudfIcebergCatalogProvider`] with explicit provider options.
pub async fn build_scoped_catalog_provider_from_backend_with_options(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    namespace: &str,
    provider_kind: IcebergProviderKind,
    provider_options: IcebergProviderOptions,
) -> Result<CudfIcebergCatalogProvider> {
    let namespace = namespace_ident_from_flat_name(namespace)?;
    build_catalog_provider_for_namespaces(
        backend,
        catalog_type.into(),
        vec![namespace],
        provider_kind,
        provider_options,
    )
    .await
}

async fn build_catalog_provider_for_namespaces(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    namespaces: Vec<NamespaceIdent>,
    provider_kind: IcebergProviderKind,
    provider_options: IcebergProviderOptions,
) -> Result<CudfIcebergCatalogProvider> {
    let mut schemas: BTreeMap<String, Arc<CudfIcebergSchemaProvider>> = BTreeMap::new();

    for namespace in namespaces {
        let ns_parts: &Vec<String> = namespace.as_ref();
        let schema_name = if ns_parts.len() == 1 { ns_parts[0].clone() } else { ns_parts.join(".") };

        #[cfg(feature = "server")]
        let list_tables_started_at = std::time::Instant::now();
        let table_idents = match backend.list_tables(&namespace).await {
            Ok(table_idents) => {
                #[cfg(feature = "server")]
                log_iceberg_metadata_lookup(
                    "list_tables",
                    &catalog_type,
                    &namespace,
                    None,
                    list_tables_started_at.elapsed(),
                    None,
                    "listed",
                );
                table_idents
            },
            Err(error) => {
                #[cfg(feature = "server")]
                log_iceberg_metadata_lookup(
                    "list_tables",
                    &catalog_type,
                    &namespace,
                    None,
                    list_tables_started_at.elapsed(),
                    None,
                    "error",
                );
                return Err(error);
            },
        };
        let known_table_names: Vec<String> = table_idents.into_iter().map(|t| t.name).collect();

        let schema_provider = Arc::new(CudfIcebergSchemaProvider::new(
            namespace,
            known_table_names,
            Arc::clone(&backend),
            &catalog_type,
            provider_kind,
            provider_options.clone(),
            backend.parquet_read_strategy_config(),
        ));
        schemas.insert(schema_name, schema_provider);
    }

    Ok(CudfIcebergCatalogProvider::new(schemas))
}

/// Builds a [`TableIdent`] from a flat dotted namespace scope and a bare table
/// name. Backend-agnostic helper used by server alias seeding to route table
/// loads through [`IcebergCatalogBackend::load_table`] regardless of catalog
/// kind.
#[cfg(feature = "server")]
pub(crate) fn table_ident_from_flat_name(namespace: &str, table: &str) -> Result<TableIdent> {
    let namespace = namespace_ident_from_flat_name(namespace)?;
    Ok(TableIdent::new(namespace, table.to_string()))
}

fn namespace_ident_from_flat_name(namespace: &str) -> Result<NamespaceIdent> {
    let parts = namespace.split('.').collect::<Vec<_>>();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
        return Err(Error::new(
            ErrorCode::CudfInvalidInput,
            CATALOG_BACKEND_OP,
            "Iceberg namespace scope must not be empty",
        ));
    }
    NamespaceIdent::from_strs(parts).map_err(|error| {
        Error::new(ErrorCode::CudfInvalidInput, CATALOG_BACKEND_OP, "invalid Iceberg namespace scope")
            .with_fact("namespace", namespace)
            .with_source(error)
    })
}

/// Catalog-level configuration for the AWS Glue Iceberg backend.
///
/// This covers the Glue catalog connection (warehouse, region, optional catalog
/// ID and profile).  Table identity is not part of this config; it is resolved
/// at runtime via [`IcebergCatalogBackend::list_tables`] and
/// [`IcebergCatalogBackend::load_table`].
#[derive(Clone, PartialEq, Eq)]
pub struct GlueIcebergCatalogConfig {
    /// Internal catalog handle passed to `GlueCatalogBuilder::load`.
    /// Independent of the DataFusion catalog name chosen by the operator.
    pub catalog_handle: String,
    /// Iceberg warehouse URI, typically an `s3://...` root.
    pub warehouse: String,
    /// AWS region used by Glue/S3 calls.
    pub region: String,
    /// Optional Glue catalog id for cross-account catalog access.
    pub catalog_id: Option<String>,
    /// Optional AWS profile selector; empty values are ignored at connect time.
    pub profile: Option<String>,
    /// Optional Glue-compatible endpoint URI, e.g. a local Moto server.
    ///
    /// This maps to `iceberg-catalog-glue`'s `uri` prop and may contain
    /// endpoint credentials, so it is redacted from [`Debug`].
    pub uri: Option<String>,
    /// Optional FileIO `s3.endpoint`, e.g. a local RustFS/MinIO server.
    ///
    /// This is independent of the Glue endpoint URI and is redacted from
    /// [`Debug`] because endpoint URLs can carry userinfo.
    pub s3_endpoint: Option<String>,
    /// Optional FileIO `s3.region`. Defaults to [`Self::region`] when unset.
    pub s3_region: Option<String>,
    /// Optional FileIO `s3.path-style-access`.
    pub s3_path_style_access: Option<bool>,
}

impl fmt::Debug for GlueIcebergCatalogConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueIcebergCatalogConfig")
            .field("catalog_handle", &self.catalog_handle)
            .field("warehouse", &self.warehouse)
            .field("region", &self.region)
            .field("catalog_id", &self.catalog_id)
            .field("profile", &self.profile)
            .field("uri", &self.uri.as_ref().map(|_| "<redacted>"))
            .field("s3_endpoint", &self.s3_endpoint.as_ref().map(|_| "<redacted>"))
            .field("s3_region", &self.s3_region)
            .field("s3_path_style_access", &self.s3_path_style_access)
            .finish()
    }
}

impl GlueIcebergCatalogConfig {
    /// Builds catalog-level Glue configuration.
    pub fn new(catalog_handle: impl Into<String>, warehouse: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            catalog_handle: catalog_handle.into(),
            warehouse: warehouse.into(),
            region: region.into(),
            catalog_id: None,
            profile: None,
            uri: None,
            s3_endpoint: None,
            s3_region: None,
            s3_path_style_access: None,
        }
    }

    /// Sets an explicit Glue catalog id.
    pub fn with_catalog_id(mut self, catalog_id: impl Into<String>) -> Self {
        self.catalog_id = Some(catalog_id.into());
        self
    }

    /// Sets an AWS profile selector.
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Sets a Glue-compatible endpoint URI, such as a local Moto server.
    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }

    /// Sets the FileIO `s3.endpoint` used by tables loaded from this catalog.
    pub fn with_s3_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.s3_endpoint = Some(endpoint.into());
        self
    }

    /// Sets the FileIO `s3.region` used by tables loaded from this catalog.
    pub fn with_s3_region(mut self, region: impl Into<String>) -> Self {
        self.s3_region = Some(region.into());
        self
    }

    /// Sets the FileIO `s3.path-style-access` flag.
    pub fn with_s3_path_style_access(mut self, path_style: bool) -> Self {
        self.s3_path_style_access = Some(path_style);
        self
    }

    /// Builds the table-format Parquet read-strategy config for this Glue
    /// catalog. The config is credential-free and references the native source
    /// storage profile by id.
    pub fn parquet_read_strategy_config(&self) -> TableFormatParquetReadStrategyConfig {
        self.parquet_read_strategy_config_from(TableFormatParquetReadStrategyConfig::from_env())
    }

    /// Applies this catalog's credential-free endpoint/profile metadata to a
    /// caller-owned table-format Parquet read-strategy config.
    pub fn parquet_read_strategy_config_from(
        &self,
        config: TableFormatParquetReadStrategyConfig,
    ) -> TableFormatParquetReadStrategyConfig {
        S3FileIoConfig::from_glue(self).apply_parquet_read_strategy_config(config, !self.region.trim().is_empty())
    }

    /// Builds the native source storage profile for this Glue catalog using the
    /// same AWS credential environment accepted by the Glue catalog loader.
    pub fn source_storage_profile_with_env(&self, mut get: impl FnMut(&str) -> Option<String>) -> SourceStorageProfile {
        let (mut s3, has_s3_endpoint) = S3FileIoConfig::from_glue(self)
            .apply_source_storage_profile(S3SourceStorageProfile::default_chain(), Some(&self.region));

        let explicit_s3_credentials = (
            super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)),
            super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV)),
        );
        let aws_credentials = (
            super::non_empty(get(crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
            super::non_empty(
                get(crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
                    .or_else(|| get(crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
            ),
        );
        if let (Some(access_key_id), Some(secret_access_key), session_token) = explicit_s3_credentials {
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
        } else if super::s3_credential_source_default_chain_opt_in(&mut get) {
            // Explicit operator opt-in: keep the default chain even with a
            // custom endpoint. See NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV.
        } else if has_s3_endpoint {
            // Local/S3-compatible endpoints must not silently reuse a
            // developer's ambient AWS credentials. Operators that need signed
            // requests for such endpoints should set NEXUS_ICEBERG_S3_* or opt
            // into the default chain via NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE.
            s3 = s3.with_unsigned_requests();
        } else if let (Some(access_key_id), Some(secret_access_key), session_token) = aws_credentials {
            // Preserve the existing real-AWS behavior while still allowing
            // local S3-compatible tests to override with NEXUS_ICEBERG_S3_*.
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
        }
        SourceStorageProfile::s3(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, s3)
    }

    /// Builds the native source storage profile for this Glue catalog from
    /// process environment.
    pub fn source_storage_profile_from_env(&self) -> SourceStorageProfile {
        self.source_storage_profile_with_env(|name| std::env::var(name).ok())
    }
}

fn glue_catalog_props(
    config: &GlueIcebergCatalogConfig,
    mut get: impl FnMut(&str) -> Option<String>,
) -> HashMap<String, String> {
    let mut props = HashMap::from([
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), config.warehouse.clone()),
        (AWS_REGION_NAME.to_string(), config.region.clone()),
    ]);
    if let Some(uri) = config.uri.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert(GLUE_CATALOG_PROP_URI.to_string(), uri.clone());
    }
    if let Some(id) = config.catalog_id.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert(GLUE_CATALOG_PROP_CATALOG_ID.to_string(), id.clone());
    }
    if let Some(profile) = config.profile.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert(AWS_PROFILE_NAME.to_string(), profile.clone());
    }
    let s3_file_io = S3FileIoConfig::from_glue(config);
    if s3_file_io.non_empty_endpoint().is_some() {
        s3_file_io.apply_catalog_props(&mut props);
        // Custom endpoints default-deny ambient credential discovery unless
        // the operator explicitly opted into the AWS default chain via
        // NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE=default_chain.
        if !super::s3_credential_source_default_chain_opt_in(&mut get) {
            props.insert(S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
            props.insert(S3_DISABLE_EC2_METADATA.to_string(), "true".to_string());
        }
    } else {
        s3_file_io.apply_catalog_props(&mut props);
    }

    if let (Some(key_id), Some(secret)) = (
        super::non_empty(get(crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)),
        super::non_empty(get(crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
    ) {
        props.insert(AWS_ACCESS_KEY_ID.to_string(), key_id);
        props.insert(AWS_SECRET_ACCESS_KEY.to_string(), secret);
    }
    if let Some(token) = super::non_empty(
        get(crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
            .or_else(|| get(crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
    ) {
        props.insert(AWS_SESSION_TOKEN.to_string(), token);
    }

    super::insert_explicit_nexus_s3_credentials(&mut props, |name| get(name), true);

    props
}

/// AWS Glue implementation of [`IcebergCatalogBackend`].
///
/// Holds a connected `iceberg-catalog-glue` catalog handle.  Namespace and
/// table listing, and individual table loading, all delegate to the
/// `iceberg-rust` [`Catalog`] trait.
pub struct GlueIcebergCatalogBackend {
    catalog: Arc<dyn Catalog>,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
}

impl fmt::Debug for GlueIcebergCatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueIcebergCatalogBackend").finish_non_exhaustive()
    }
}

impl GlueIcebergCatalogBackend {
    /// Builds a connected Glue backend from pre-assembled catalog properties.
    ///
    /// Properties are passed through to `GlueCatalogBuilder` as-is and may
    /// contain secrets; callers should avoid logging them.
    pub async fn from_props(catalog_handle: impl Into<String>, props: HashMap<String, String>) -> Result<Self> {
        let glue = GlueCatalogBuilder::default()
            .load(catalog_handle, props)
            .await
            .map_err(|error| catalog_backend_error("failed to connect to AWS Glue Iceberg catalog", "glue", error))?;
        Ok(Self { catalog: Arc::new(glue), parquet_read_strategy_config: None })
    }

    /// Builds a connected Glue backend from a [`GlueIcebergCatalogConfig`].
    ///
    /// Empty catalog id/profile values are ignored. Access key and secret key
    /// must both be non-empty before static credentials are passed through; a
    /// session token can come from `AWS_SESSION_TOKEN` or `AWS_SECURITY_TOKEN`.
    pub async fn from_config(
        config: GlueIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let props = glue_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config()
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(config, props, parquet_read_strategy_config).await
    }

    /// Builds a connected Glue backend from a config and caller-owned
    /// table-format Parquet read-strategy config.
    pub async fn from_config_with_parquet_read_strategy_config(
        config: GlueIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let props = glue_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config_from(parquet_read_strategy_config)
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(config, props, parquet_read_strategy_config).await
    }

    async fn from_config_with_resolved_parquet_read_strategy_config(
        config: GlueIcebergCatalogConfig,
        props: HashMap<String, String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let mut backend = Self::from_props(config.catalog_handle, props).await?;
        backend.parquet_read_strategy_config = Some(parquet_read_strategy_config);
        Ok(backend)
    }

    /// Builds a connected Glue backend from a [`GlueIcebergCatalogConfig`]
    /// using `std::env::var` for credential lookup.
    pub async fn from_config_env(config: GlueIcebergCatalogConfig) -> Result<Self> {
        Self::from_config(config, |name| std::env::var(name).ok()).await
    }
}

#[async_trait]
impl IcebergCatalogBackend for GlueIcebergCatalogBackend {
    async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>> {
        catalog_backend_list_namespaces(
            &self.catalog,
            "glue",
            "failed to list namespaces from AWS Glue Iceberg catalog",
        )
        .await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        catalog_backend_list_tables(
            &self.catalog,
            namespace,
            "glue",
            "failed to list tables from AWS Glue Iceberg catalog",
        )
        .await
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        catalog_backend_load_table(&self.catalog, table, "glue", "failed to load table from AWS Glue Iceberg catalog")
            .await
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
    }
}

/// Catalog-level configuration for the Iceberg REST catalog backend.
///
/// This covers the REST catalog connection (`uri`, `warehouse`, optional
/// `prefix`/`token`) plus the S3 FileIO settings used to reach the object store
/// holding the table data (e.g. RustFS or MinIO for local stacks, or real S3).
/// Table identity is not part of this config; it is resolved at runtime via
/// [`IcebergCatalogBackend::list_tables`] and
/// [`IcebergCatalogBackend::load_table`].
///
/// Unlike the Glue backend, REST has **no default storage factory**, so the
/// backend constructed from this config always injects an S3 `OpenDalStorage`
/// factory. The static S3 credentials are read at connect time from explicit
/// `NEXUS_ICEBERG_S3_*` environment variables (never the ambient
/// `AWS_*` chain) and are not retained in this struct.
#[derive(Clone, PartialEq, Eq)]
pub struct RestIcebergCatalogConfig {
    /// Internal catalog handle passed to `RestCatalogBuilder::load`.
    /// Independent of the DataFusion catalog name chosen by the operator.
    pub catalog_handle: String,
    /// REST catalog base URI, e.g. `http://localhost:8181`. Required.
    pub uri: String,
    /// Iceberg warehouse root, typically an `s3://bucket/...` location.
    /// Required by the validated server config; the underlying REST builder
    /// itself treats it as optional.
    pub warehouse: String,
    /// Optional REST `prefix` prop for multi-tenant REST servers. `None` omits
    /// the prop entirely.
    pub prefix: Option<String>,
    /// Optional bearer token passed as the REST `token` prop. Secret: never
    /// logged or stored in diagnostics/`Facts`. `None` omits the prop.
    pub token: Option<String>,
    /// FileIO `s3.endpoint`, e.g. `http://localhost:9000`. `None` lets OpenDAL
    /// use its default (real AWS S3).
    pub s3_endpoint: Option<String>,
    /// FileIO `s3.region`, e.g. `us-east-1`. `None` omits the prop.
    pub s3_region: Option<String>,
    /// FileIO `s3.path-style-access`. `Some(true)` is required for RustFS/MinIO
    /// path-style addressing; `None` omits the prop (OpenDAL default).
    pub s3_path_style_access: Option<bool>,
}

impl fmt::Debug for RestIcebergCatalogConfig {
    /// Redacting `Debug`: never renders the bearer `token`, the `uri`, or the
    /// `s3_endpoint`. The URI and the endpoint are URLs that may embed
    /// credentials (`https://user:secret@host`), and the token is a secret, so
    /// all three are replaced with a redaction marker regardless of content.
    /// All other fields are non-secret connection metadata and printed as-is.
    /// (This is also why the struct does not `derive(Debug)`: the derived impl
    /// would print them verbatim.)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestIcebergCatalogConfig")
            .field("catalog_handle", &self.catalog_handle)
            .field("uri", &"<redacted>")
            .field("warehouse", &self.warehouse)
            .field("prefix", &self.prefix)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .field("s3_endpoint", &self.s3_endpoint.as_ref().map(|_| "<redacted>"))
            .field("s3_region", &self.s3_region)
            .field("s3_path_style_access", &self.s3_path_style_access)
            .finish()
    }
}

impl RestIcebergCatalogConfig {
    /// Builds REST catalog configuration with the required `uri` and
    /// `warehouse`. Optional fields default to `None`.
    pub fn new(catalog_handle: impl Into<String>, uri: impl Into<String>, warehouse: impl Into<String>) -> Self {
        Self {
            catalog_handle: catalog_handle.into(),
            uri: uri.into(),
            warehouse: warehouse.into(),
            prefix: None,
            token: None,
            s3_endpoint: None,
            s3_region: None,
            s3_path_style_access: None,
        }
    }

    /// Sets the optional REST `prefix` prop.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Sets the optional bearer `token` prop. Secret; not retained in diagnostics.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets the FileIO `s3.endpoint`.
    pub fn with_s3_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.s3_endpoint = Some(endpoint.into());
        self
    }

    /// Sets the FileIO `s3.region`.
    pub fn with_s3_region(mut self, region: impl Into<String>) -> Self {
        self.s3_region = Some(region.into());
        self
    }

    /// Sets the FileIO `s3.path-style-access` flag.
    pub fn with_s3_path_style_access(mut self, path_style: bool) -> Self {
        self.s3_path_style_access = Some(path_style);
        self
    }

    /// Builds the table-format Parquet read-strategy config for this REST
    /// catalog. The config is credential-free and references the native source
    /// storage profile by id.
    pub fn parquet_read_strategy_config(&self) -> TableFormatParquetReadStrategyConfig {
        self.parquet_read_strategy_config_from(TableFormatParquetReadStrategyConfig::from_env())
    }

    /// Applies this catalog's credential-free endpoint/profile metadata to a
    /// caller-owned table-format Parquet read-strategy config.
    pub fn parquet_read_strategy_config_from(
        &self,
        config: TableFormatParquetReadStrategyConfig,
    ) -> TableFormatParquetReadStrategyConfig {
        S3FileIoConfig::from_rest(self).apply_parquet_read_strategy_config(config, false)
    }

    /// Builds the native source storage profile for this REST catalog using the
    /// explicit `NEXUS_ICEBERG_S3_*` credential environment.
    pub fn source_storage_profile_with_env(&self, mut get: impl FnMut(&str) -> Option<String>) -> SourceStorageProfile {
        let (mut s3, has_s3_endpoint) =
            S3FileIoConfig::from_rest(self).apply_source_storage_profile(S3SourceStorageProfile::default_chain(), None);
        if let (Some(access_key_id), Some(secret_access_key)) = (
            super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)),
        ) {
            let session_token = super::non_empty(get(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV));
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
        } else if super::s3_credential_source_default_chain_opt_in(&mut get) {
            // Explicit operator opt-in: keep the default chain even with a
            // custom endpoint. See NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV.
        } else if has_s3_endpoint {
            s3 = s3.with_unsigned_requests();
        }
        SourceStorageProfile::s3(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, s3)
    }

    /// Builds the native source storage profile for this REST catalog from
    /// process environment.
    pub fn source_storage_profile_from_env(&self) -> SourceStorageProfile {
        self.source_storage_profile_with_env(|name| std::env::var(name).ok())
    }
}

/// Builds the REST catalog property map from a [`RestIcebergCatalogConfig`] and
/// an env lookup for static S3 credentials.
///
/// Non-empty `s3.access-key-id` and `s3.secret-access-key` are inserted only
/// when both are present; a session token is inserted from
/// [`crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV`] when present. Empty
/// optional values are omitted rather than inserted as empty strings.
///
/// S3 credentials come ONLY from the explicit `NEXUS_ICEBERG_S3_*`
/// environment variables read here. To enforce that, the prop map sets
/// `s3.disable-config-load=true` and `s3.disable-ec2-metadata=true`, which turn
/// off OpenDAL's ambient credential discovery (the `AWS_*` environment chain,
/// `~/.aws/*` config files, and the EC2 IMDS endpoint). Without these flags
/// OpenDAL would silently fall back to the ambient AWS chain when no explicit
/// creds are provided — exactly the silent `AWS_*` pickup this project forbids.
/// The single exception is an explicit operator opt-in via
/// `NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE=default_chain`, which deliberately
/// selects that ambient chain and therefore skips the default-deny flags.
///
/// Separated from [`RestIcebergCatalogBackend::from_config`] so it can be
/// unit-tested without a live REST server.
fn rest_catalog_props(
    config: &RestIcebergCatalogConfig,
    mut get: impl FnMut(&str) -> Option<String>,
) -> HashMap<String, String> {
    let mut props = HashMap::from([
        (REST_CATALOG_PROP_URI.to_string(), config.uri.clone()),
        (REST_CATALOG_PROP_WAREHOUSE.to_string(), config.warehouse.clone()),
    ]);
    // Default-deny ambient AWS credential discovery; see the doc comment.
    if !super::s3_credential_source_default_chain_opt_in(&mut get) {
        props.insert(S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
        props.insert(S3_DISABLE_EC2_METADATA.to_string(), "true".to_string());
    }
    if let Some(prefix) = config.prefix.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert("prefix".to_string(), prefix.clone());
    }
    if let Some(token) = config.token.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert("token".to_string(), token.clone());
    }
    S3FileIoConfig::from_rest(config).apply_catalog_props(&mut props);
    super::insert_explicit_nexus_s3_credentials(&mut props, |name| get(name), false);
    props
}

/// Builds the S3 `OpenDalStorageFactory` that the REST backend always injects.
///
/// REST has no default storage factory (it errors at file-IO time without one),
/// so every `RestIcebergCatalogBackend` constructor routes through this. Exposed
/// at crate scope so tests can assert the factory is wired (the upstream error
/// only fires at read time, not at construction).
pub(crate) fn rest_storage_factory() -> Arc<OpenDalStorageFactory> {
    Arc::new(OpenDalStorageFactory::S3 { customized_credential_load: None })
}

/// Iceberg REST catalog implementation of [`IcebergCatalogBackend`].
///
/// Holds a connected `iceberg-catalog-rest` catalog handle with an injected S3
/// storage factory. Namespace/table listing and table loading delegate to the
/// `iceberg-rust` [`Catalog`] trait, mirroring [`GlueIcebergCatalogBackend`]
/// with the `catalog_type` fact set to `"rest"`.
pub struct RestIcebergCatalogBackend {
    catalog: Arc<dyn Catalog>,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
}

impl fmt::Debug for RestIcebergCatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RestIcebergCatalogBackend").finish_non_exhaustive()
    }
}

impl RestIcebergCatalogBackend {
    /// Builds a connected REST backend from pre-assembled catalog properties.
    ///
    /// Always injects the S3 storage factory via `with_storage_factory` before
    /// `load`, because REST has no default factory. Properties are passed
    /// through as-is and may contain secrets (`uri`, `token`,
    /// `s3.secret-access-key`); callers must avoid logging them.
    ///
    /// Secret-handling boundary: our own [`Error`] wrapper here carries only a
    /// static message and non-secret facts (`catalog_type`). The upstream
    /// `iceberg-rust` error is attached via [`Error::with_source`], and
    /// [`Error`]'s `Display`/`Debug` render `source.to_string()`. We do not
    /// control that upstream rendering and deliberately do not wrap it in a
    /// sanitizing boundary, so if a future `iceberg-rust` release folds the
    /// request URL or prop map into its error text, that text would surface
    /// through our error. Treat REST-backend errors as potentially
    /// secret-bearing at log sites; do not log them at INFO/WARN in untrusted
    /// environments without scrubbing.
    pub async fn from_props(catalog_handle: impl Into<String>, props: HashMap<String, String>) -> Result<Self> {
        let rest = RestCatalogBuilder::default()
            .with_storage_factory(rest_storage_factory())
            .load(catalog_handle, props)
            .await
            .map_err(|error| catalog_backend_error("failed to connect to Iceberg REST catalog", "rest", error))?;
        Ok(Self { catalog: Arc::new(rest), parquet_read_strategy_config: None })
    }

    /// Builds a connected REST backend from a [`RestIcebergCatalogConfig`],
    /// resolving static S3 credentials through `get`.
    pub async fn from_config(
        config: RestIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let props = rest_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config()
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(config, props, parquet_read_strategy_config).await
    }

    /// Builds a connected REST backend from a config and caller-owned
    /// table-format Parquet read-strategy config.
    pub async fn from_config_with_parquet_read_strategy_config(
        config: RestIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let props = rest_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config_from(parquet_read_strategy_config)
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(config, props, parquet_read_strategy_config).await
    }

    async fn from_config_with_resolved_parquet_read_strategy_config(
        config: RestIcebergCatalogConfig,
        props: HashMap<String, String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let mut backend = Self::from_props(config.catalog_handle, props).await?;
        backend.parquet_read_strategy_config = Some(parquet_read_strategy_config);
        Ok(backend)
    }

    /// Builds a connected REST backend from a [`RestIcebergCatalogConfig`]
    /// using `std::env::var` for credential lookup.
    pub async fn from_config_env(config: RestIcebergCatalogConfig) -> Result<Self> {
        Self::from_config(config, |name| std::env::var(name).ok()).await
    }
}

#[async_trait]
impl IcebergCatalogBackend for RestIcebergCatalogBackend {
    async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>> {
        catalog_backend_list_namespaces(&self.catalog, "rest", "failed to list namespaces from Iceberg REST catalog")
            .await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        catalog_backend_list_tables(&self.catalog, namespace, "rest", "failed to list tables from Iceberg REST catalog")
            .await
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        catalog_backend_load_table(&self.catalog, table, "rest", "failed to load table from Iceberg REST catalog").await
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
    }
}

#[cfg(test)]
mod rest_tests {
    use iceberg::io::{S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN, StorageConfig, StorageFactory};

    use super::*;

    fn upstream_iceberg_error() -> iceberg::Error {
        iceberg::Error::new(iceberg::ErrorKind::Unexpected, "upstream catalog failure")
    }

    fn glue_expired_token_error() -> iceberg::Error {
        iceberg::Error::new(iceberg::ErrorKind::Unexpected, "Operation failed for hitting aws sdk error")
            .with_retryable(true)
            .with_source(std::io::Error::other(
                "aws sdk error: ServiceError(ServiceError { source: Unhandled(Unhandled { \
                 source: ErrorMetadata { code: Some(\"ExpiredTokenException\"), \
                 message: Some(\"The security token included in the request is expired\"), \
                 extras: Some({\"aws_request_id\": \"req-expired-123\"}) }, retryable: true }), \
                 status code: 401 Unauthorized })",
            ))
    }

    fn no_env(_name: &str) -> Option<String> {
        None
    }

    #[test]
    fn catalog_backend_error_preserves_glue_contract() {
        let error =
            catalog_backend_error("failed to connect to AWS Glue Iceberg catalog", "glue", upstream_iceberg_error());

        assert_eq!(error.code(), ErrorCode::CudfDependencyFailure);
        assert_eq!(error.operation(), CATALOG_BACKEND_OP);
        assert_eq!(error.message(), "failed to connect to AWS Glue Iceberg catalog");
        assert_eq!(error.fact_value("catalog_type"), Some("glue"));
        assert_eq!(error.fact_value("iceberg_error_kind"), Some("Unexpected"));
        assert_eq!(error.fact_value("iceberg_retryable"), Some("false"));
        assert!(error.source_error().is_some());
    }

    #[test]
    fn catalog_backend_error_preserves_rest_contract() {
        let error =
            catalog_backend_error("failed to connect to Iceberg REST catalog", "rest", upstream_iceberg_error());

        assert_eq!(error.code(), ErrorCode::CudfDependencyFailure);
        assert_eq!(error.operation(), CATALOG_BACKEND_OP);
        assert_eq!(error.message(), "failed to connect to Iceberg REST catalog");
        assert_eq!(error.fact_value("catalog_type"), Some("rest"));
        assert_eq!(error.fact_value("iceberg_error_kind"), Some("Unexpected"));
        assert_eq!(error.fact_value("iceberg_retryable"), Some("false"));
        assert!(error.source_error().is_some());
    }

    #[test]
    fn catalog_backend_error_extracts_safe_aws_metadata_facts() {
        let error =
            catalog_backend_error("failed to connect to AWS Glue Iceberg catalog", "glue", glue_expired_token_error());

        assert_eq!(error.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(error.fact_value("catalog_type"), Some("glue"));
        assert_eq!(error.fact_value("iceberg_retryable"), Some("true"));
        assert_eq!(error.fact_value("aws_error_code"), Some("ExpiredTokenException"));
        assert_eq!(error.fact_value("aws_request_id"), Some("req-expired-123"));
        assert_eq!(error.fact_value("aws_http_status"), Some("401"));
        assert_eq!(error.fact_value("aws_retryable"), Some("true"));
        assert_eq!(error.fact_value("aws_error_detail"), Some("expired_token"));
    }

    #[test]
    fn catalog_backend_error_allows_namespace_and_table_facts() {
        let error =
            catalog_backend_error("failed to load table from Iceberg REST catalog", "rest", upstream_iceberg_error())
                .with_fact("namespace", "tpch_sf1")
                .with_fact("table", "lineitem");

        assert_eq!(error.fact_value("catalog_type"), Some("rest"));
        assert_eq!(error.fact_value("namespace"), Some("tpch_sf1"));
        assert_eq!(error.fact_value("table"), Some("lineitem"));
    }

    #[test]
    fn rest_props_include_uri_and_warehouse() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let props = rest_catalog_props(&config, no_env);
        assert_eq!(props.get(REST_CATALOG_PROP_URI).map(String::as_str), Some("http://localhost:8181"));
        assert_eq!(props.get(REST_CATALOG_PROP_WAREHOUSE).map(String::as_str), Some("s3://lakehouse/warehouse"));
    }

    #[test]
    fn rest_props_omit_unset_optionals() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let props = rest_catalog_props(&config, no_env);
        assert!(!props.contains_key("prefix"));
        assert!(!props.contains_key("token"));
        assert!(!props.contains_key(S3_ENDPOINT));
        assert!(!props.contains_key(S3_REGION));
        assert!(!props.contains_key(S3_PATH_STYLE_ACCESS));
        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
    }

    #[test]
    fn rest_props_include_s3_filio_settings() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_region("us-east-1")
            .with_s3_path_style_access(true)
            .with_prefix("tenant_a");
        let props = rest_catalog_props(&config, no_env);
        assert_eq!(props.get(S3_ENDPOINT).map(String::as_str), Some("http://localhost:9000"));
        assert_eq!(props.get(S3_REGION).map(String::as_str), Some("us-east-1"));
        assert_eq!(props.get(S3_PATH_STYLE_ACCESS).map(String::as_str), Some("true"));
        assert_eq!(props.get("prefix").map(String::as_str), Some("tenant_a"));
    }

    #[test]
    fn rest_props_resolve_static_s3_credentials_from_env() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rest-secret".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rest-token-value".to_string()),
            _ => None,
        };
        let props = rest_catalog_props(&config, env);
        assert_eq!(props.get(S3_ACCESS_KEY_ID).map(String::as_str), Some("AKIA_LOCAL"));
        assert_eq!(props.get(S3_SECRET_ACCESS_KEY).map(String::as_str), Some("rest-secret"));
        assert_eq!(props.get(S3_SESSION_TOKEN).map(String::as_str), Some("rest-token-value"));
    }

    #[test]
    fn rest_props_skip_partial_s3_credentials() {
        // Only an access key, no secret: neither should be inserted.
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
            _ => None,
        };
        let props = rest_catalog_props(&config, env);
        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
    }

    #[test]
    fn rest_props_always_disable_ambient_aws_credential_discovery() {
        // No explicit creds in env: the prop map must STILL default-deny the
        // ambient AWS chain so OpenDAL cannot silently pick up AWS_* / IMDS.
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let props = rest_catalog_props(&config, no_env);
        assert_eq!(props.get(S3_DISABLE_CONFIG_LOAD).map(String::as_str), Some("true"));
        assert_eq!(props.get(S3_DISABLE_EC2_METADATA).map(String::as_str), Some("true"));
    }

    #[test]
    fn rest_props_ignore_ambient_aws_credentials() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("aws-access-must-not-be-used".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("aws-secret-must-not-be-used".to_string()),
            crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV => Some("aws-token-must-not-be-used".to_string()),
            _ => None,
        };

        let props = rest_catalog_props(&config, env);

        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
        assert!(!props.contains_key(S3_SESSION_TOKEN));
        assert_eq!(props.get(S3_DISABLE_CONFIG_LOAD).map(String::as_str), Some("true"));
        assert_eq!(props.get(S3_DISABLE_EC2_METADATA).map(String::as_str), Some("true"));
    }

    #[test]
    fn rest_props_default_chain_opt_in_omits_default_deny_flags() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV => {
                Some(crate::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN.to_string())
            },
            _ => None,
        };

        let props = rest_catalog_props(&config, env);

        assert!(!props.contains_key(S3_DISABLE_CONFIG_LOAD));
        assert!(!props.contains_key(S3_DISABLE_EC2_METADATA));
    }

    #[test]
    fn rest_props_preserve_s3_session_token_without_static_pair() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rest-token-value".to_string()),
            _ => None,
        };

        let props = rest_catalog_props(&config, env);

        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
        assert_eq!(props.get(S3_SESSION_TOKEN).map(String::as_str), Some("rest-token-value"));
    }

    #[test]
    fn rest_parquet_strategy_config_references_sanitized_source_storage_profile() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://user:secret@localhost:9000/catalog?signature=secret#frag");

        let strategy = config.parquet_read_strategy_config();

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/catalog"));
    }

    #[test]
    fn rest_parquet_strategy_config_uses_injected_server_strategy() {
        let injected = TableFormatParquetReadStrategyConfig {
            preference: crate::TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/rest-cache")),
            lakehouse_cache_policy: crate::TableFormatLakehouseCachePolicy::FillBeforeRead,
            lakehouse_cache_max_bytes: Some(1024),
            ..Default::default()
        };
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000");

        let strategy = config.parquet_read_strategy_config_from(injected);

        assert_eq!(strategy.preference, crate::TableFormatParquetReadStrategyPreference::LocalCache);
        assert_eq!(strategy.lakehouse_cache_root, Some(std::path::PathBuf::from("/tmp/rest-cache")));
        assert_eq!(strategy.lakehouse_cache_policy, crate::TableFormatLakehouseCachePolicy::FillBeforeRead);
        assert_eq!(strategy.lakehouse_cache_max_bytes, Some(1024));
        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000"));
    }

    #[derive(Debug)]
    struct StrategyMockBackend {
        parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    }

    #[async_trait]
    impl IcebergCatalogBackend for StrategyMockBackend {
        async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>> {
            Ok(vec![NamespaceIdent::new("tpch_sf1".to_string())])
        }

        async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
            Ok(vec![TableIdent::new(namespace.clone(), "nation".to_string())])
        }

        async fn load_table(&self, _table: &TableIdent) -> Result<iceberg::table::Table> {
            unreachable!("this test only inspects schema-provider construction")
        }

        fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
            self.parquet_read_strategy_config.clone()
        }
    }

    #[tokio::test]
    async fn catalog_provider_carries_backend_strategy_into_schema_provider() {
        let strategy = TableFormatParquetReadStrategyConfig {
            preference: crate::TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/provider-cache")),
            lakehouse_cache_policy: crate::TableFormatLakehouseCachePolicy::FillBeforeRead,
            ..Default::default()
        };
        let backend = Arc::new(StrategyMockBackend { parquet_read_strategy_config: Some(strategy.clone()) });

        let provider =
            build_scoped_catalog_provider_from_backend(backend, "rest", "tpch_sf1", IcebergProviderKind::Gpu)
                .await
                .unwrap();
        let schema = provider.schema_concrete("tpch_sf1").expect("scoped schema provider");

        assert_eq!(schema.parquet_read_strategy_config, Some(strategy));
    }

    #[test]
    fn rest_source_storage_profile_carries_s3_endpoint_path_style_and_static_credentials() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_region("us-east-1")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rest-secret".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rest-token-value".to_string()),
            _ => None,
        };

        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(profile.id(), DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID);
        let s3 = profile.s3_profile();
        assert_eq!(s3.region.as_deref(), Some("us-east-1"));
        assert_eq!(s3.endpoint.as_deref(), Some("http://localhost:9000"));
        assert_eq!(s3.virtual_hosted_style_request, Some(false));
        match &s3.credential {
            nexus_query_engine::S3SourceStorageCredential::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                assert_eq!(access_key_id, "AKIA_LOCAL");
                assert_eq!(secret_access_key, "rest-secret");
                assert_eq!(session_token.as_deref(), Some("rest-token-value"));
            },
            other => panic!("expected static credentials, got {other:?}"),
        }

        let rendered = format!("{profile:?}");
        assert!(!rendered.contains("AKIA_LOCAL"), "access key id must be redacted: {rendered}");
        assert!(!rendered.contains("rest-secret"), "secret values must be redacted: {rendered}");
        assert!(!rendered.contains("rest-token-value"), "session token values must be redacted: {rendered}");
        assert!(!rendered.contains("localhost:9000"), "endpoint must be redacted: {rendered}");
    }

    #[test]
    fn rest_source_storage_profile_does_not_use_ambient_aws_credentials_for_local_s3_endpoint() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("aws-access-must-not-be-used".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("aws-secret-must-not-be-used".to_string()),
            _ => None,
        };

        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(profile.s3_profile().credential, nexus_query_engine::S3SourceStorageCredential::Unsigned);
    }

    #[test]
    fn rest_parquet_strategy_marks_region_override_as_scoped() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_region("us-west-2");

        let strategy = config.parquet_read_strategy_config();
        let profile = config.source_storage_profile_with_env(|_| None);

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.credential_mode, nexus_query_engine::CredentialMode::ScopedResolved);
        assert_eq!(profile.s3_profile().region.as_deref(), Some("us-west-2"));
        assert!(profile.requires_scoped_remote_direct_profile());
    }

    #[test]
    fn glue_source_storage_profile_uses_glue_region_and_aws_static_credentials() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-west-2");
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("AKIA_GLUE".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("glue-secret".to_string()),
            crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV => Some("glue-token".to_string()),
            _ => None,
        };

        let strategy = config.parquet_read_strategy_config();
        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.credential_mode, nexus_query_engine::CredentialMode::ScopedResolved);
        assert_eq!(profile.id(), DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID);
        assert_eq!(profile.s3_profile().region.as_deref(), Some("us-west-2"));
        match &profile.s3_profile().credential {
            nexus_query_engine::S3SourceStorageCredential::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                assert_eq!(access_key_id, "AKIA_GLUE");
                assert_eq!(secret_access_key, "glue-secret");
                assert_eq!(session_token.as_deref(), Some("glue-token"));
            },
            other => panic!("expected static credentials, got {other:?}"),
        }
    }

    #[test]
    fn glue_props_include_moto_uri_s3_endpoint_and_explicit_s3_credentials() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_uri("http://localhost:5001")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_region("us-east-1")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("moto-access".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("moto-secret".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("rustfs-access".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rustfs-secret".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rustfs-token".to_string()),
            _ => None,
        };

        let props = glue_catalog_props(&config, env);

        assert_eq!(props.get(GLUE_CATALOG_PROP_URI).map(String::as_str), Some("http://localhost:5001"));
        assert_eq!(props.get(S3_ENDPOINT).map(String::as_str), Some("http://localhost:9000"));
        assert_eq!(props.get(S3_REGION).map(String::as_str), Some("us-east-1"));
        assert_eq!(props.get(S3_PATH_STYLE_ACCESS).map(String::as_str), Some("true"));
        assert_eq!(props.get(AWS_ACCESS_KEY_ID).map(String::as_str), Some("moto-access"));
        assert_eq!(props.get(AWS_SECRET_ACCESS_KEY).map(String::as_str), Some("moto-secret"));
        assert_eq!(props.get(S3_ACCESS_KEY_ID).map(String::as_str), Some("rustfs-access"));
        assert_eq!(props.get(S3_SECRET_ACCESS_KEY).map(String::as_str), Some("rustfs-secret"));
        assert_eq!(props.get(S3_SESSION_TOKEN).map(String::as_str), Some("rustfs-token"));
        assert_eq!(props.get(S3_DISABLE_CONFIG_LOAD).map(String::as_str), Some("true"));
        assert_eq!(props.get(S3_DISABLE_EC2_METADATA).map(String::as_str), Some("true"));
    }

    #[test]
    fn glue_props_accept_ambient_aws_credentials_without_s3_endpoint() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-west-2");
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("AKIA_GLUE".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("glue-secret".to_string()),
            crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV => Some("glue-legacy-token".to_string()),
            _ => None,
        };

        let props = glue_catalog_props(&config, env);

        assert_eq!(props.get(AWS_ACCESS_KEY_ID).map(String::as_str), Some("AKIA_GLUE"));
        assert_eq!(props.get(AWS_SECRET_ACCESS_KEY).map(String::as_str), Some("glue-secret"));
        assert_eq!(props.get(AWS_SESSION_TOKEN).map(String::as_str), Some("glue-legacy-token"));
        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
        assert!(!props.contains_key(S3_DISABLE_CONFIG_LOAD));
        assert!(!props.contains_key(S3_DISABLE_EC2_METADATA));
    }

    #[test]
    fn glue_props_default_chain_opt_in_skips_endpoint_default_deny_without_static_s3_credentials() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_s3_endpoint("http://localhost:9000");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV => {
                Some(crate::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN.to_string())
            },
            _ => None,
        };

        let props = glue_catalog_props(&config, env);

        assert_eq!(props.get(S3_ENDPOINT).map(String::as_str), Some("http://localhost:9000"));
        assert!(!props.contains_key(S3_DISABLE_CONFIG_LOAD));
        assert!(!props.contains_key(S3_DISABLE_EC2_METADATA));
    }

    #[test]
    fn glue_props_preserve_explicit_s3_session_token_without_static_pair() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1");
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rustfs-token".to_string()),
            _ => None,
        };

        let props = glue_catalog_props(&config, env);

        assert!(!props.contains_key(S3_ACCESS_KEY_ID));
        assert!(!props.contains_key(S3_SECRET_ACCESS_KEY));
        assert_eq!(props.get(S3_SESSION_TOKEN).map(String::as_str), Some("rustfs-token"));
        assert!(!props.contains_key(S3_DISABLE_CONFIG_LOAD));
        assert!(!props.contains_key(S3_DISABLE_EC2_METADATA));
    }

    #[test]
    fn glue_parquet_strategy_and_source_profile_include_local_s3_endpoint() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_s3_endpoint("http://user:secret@localhost:9000/catalog?signature=secret#frag")
            .with_s3_region("us-west-2")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("rustfs-access".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rustfs-secret".to_string()),
            crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rustfs-token".to_string()),
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("aws-access-ignored".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("aws-secret-ignored".to_string()),
            _ => None,
        };

        let strategy = config.parquet_read_strategy_config();
        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/catalog"));
        let s3 = profile.s3_profile();
        assert_eq!(s3.region.as_deref(), Some("us-west-2"));
        assert_eq!(s3.endpoint.as_deref(), Some("http://user:secret@localhost:9000/catalog?signature=secret#frag"));
        assert_eq!(s3.virtual_hosted_style_request, Some(false));
        match &s3.credential {
            nexus_query_engine::S3SourceStorageCredential::Static {
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                assert_eq!(access_key_id, "rustfs-access");
                assert_eq!(secret_access_key, "rustfs-secret");
                assert_eq!(session_token.as_deref(), Some("rustfs-token"));
            },
            other => panic!("expected explicit S3 static credentials, got {other:?}"),
        }
    }

    #[test]
    fn glue_parquet_strategy_config_uses_injected_server_strategy() {
        let injected = TableFormatParquetReadStrategyConfig {
            preference: crate::TableFormatParquetReadStrategyPreference::RemoteKvikio,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/glue-cache")),
            lakehouse_cache_policy: crate::TableFormatLakehouseCachePolicy::ReadThrough,
            lakehouse_cache_max_bytes: Some(2048),
            storage_endpoint: Some("https://ambient.example.invalid".to_string()),
            ..Default::default()
        };
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_s3_endpoint("http://user:secret@localhost:9000/catalog?signature=secret#frag");

        let strategy = config.parquet_read_strategy_config_from(injected);

        assert_eq!(strategy.preference, crate::TableFormatParquetReadStrategyPreference::RemoteKvikio);
        assert_eq!(strategy.lakehouse_cache_root, Some(std::path::PathBuf::from("/tmp/glue-cache")));
        assert_eq!(strategy.lakehouse_cache_policy, crate::TableFormatLakehouseCachePolicy::ReadThrough);
        assert_eq!(strategy.lakehouse_cache_max_bytes, Some(2048));
        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/catalog"));
    }

    #[test]
    fn glue_source_storage_profile_does_not_use_ambient_aws_credentials_for_local_s3_endpoint() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("aws-access-must-not-be-used".to_string()),
            crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("aws-secret-must-not-be-used".to_string()),
            _ => None,
        };

        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(profile.s3_profile().credential, nexus_query_engine::S3SourceStorageCredential::Unsigned);
    }

    #[test]
    fn glue_config_debug_redacts_uri_and_s3_endpoint() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_uri("https://user:glue-secret@glue.local")
            .with_s3_endpoint("https://user:s3-secret@s3.local");

        let rendered = format!("{config:?}");

        assert!(!rendered.contains("glue-secret"), "Glue URI must be redacted: {rendered}");
        assert!(!rendered.contains("glue.local"), "Glue URI host must be redacted: {rendered}");
        assert!(!rendered.contains("s3-secret"), "S3 endpoint must be redacted: {rendered}");
        assert!(!rendered.contains("s3.local"), "S3 endpoint host must be redacted: {rendered}");
        assert!(rendered.contains("us-east-1"), "non-secret region should be visible: {rendered}");
        assert!(rendered.contains("<redacted>"), "redaction marker should be visible: {rendered}");
    }

    #[test]
    fn rest_config_debug_redacts_token_and_uri() {
        // The URI and the S3 endpoint are URLs that may embed credentials, and
        // the token is a secret; none may appear in Debug output. Other
        // connection metadata may.
        let config = RestIcebergCatalogConfig::new("lake", "https://user:supersecret@catalog.example", "s3://w")
            .with_token("bearer-abc123")
            .with_s3_endpoint("https://endpointuser:endpointsecret@minio.internal")
            .with_s3_region("us-east-1");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("supersecret"), "uri must be redacted: {rendered}");
        assert!(!rendered.contains("catalog.example"), "uri must be redacted: {rendered}");
        assert!(!rendered.contains("bearer-abc123"), "token must be redacted: {rendered}");
        assert!(!rendered.contains("endpointsecret"), "s3_endpoint must be redacted: {rendered}");
        assert!(!rendered.contains("minio.internal"), "s3_endpoint must be redacted: {rendered}");
        assert!(rendered.contains("<redacted>"), "expected a redaction marker: {rendered}");
        // Non-secret metadata is still visible.
        assert!(rendered.contains("us-east-1"), "non-secret region should be printed: {rendered}");
        assert!(rendered.contains("lake"), "non-secret catalog handle should be printed: {rendered}");
    }

    /// The REST backend has no default storage factory; the upstream error only
    /// fires at file-IO time, so this asserts the seam itself wires an S3
    /// storage rather than asserting a constructor panic.
    #[test]
    fn rest_storage_factory_builds_s3_storage() {
        let factory = rest_storage_factory();
        let storage_config = StorageConfig::from_props(HashMap::from([
            (S3_ENDPOINT.to_string(), "http://localhost:9000".to_string()),
            (S3_REGION.to_string(), "us-east-1".to_string()),
            (S3_PATH_STYLE_ACCESS.to_string(), "true".to_string()),
            (S3_ACCESS_KEY_ID.to_string(), "AKIA_LOCAL".to_string()),
            (S3_SECRET_ACCESS_KEY.to_string(), "secret".to_string()),
        ]));
        // build() returning Ok proves the S3 variant is the one wired in.
        assert!(factory.build(&storage_config).is_ok(), "REST backend must inject a working S3 storage factory");
    }
}
