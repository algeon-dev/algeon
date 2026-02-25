//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`IcebergCatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`] or
//! [`build_scoped_catalog_provider_from_backend`]; individual tables are loaded
//! lazily on first access.
//!
//! # Backend implementations
//!
//! - [`GlueIcebergCatalogBackend`] — AWS Glue via `iceberg-catalog-glue`
//! - REST and Hive backends can be added as additional implementations of
//!   [`IcebergCatalogBackend`] without changing the DataFusion-facing layer.

#![deny(missing_docs)]

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt,
    sync::Arc,
};

use async_trait::async_trait;
use cudf_query_engine::{S3SourceStorageProfile, SourceStorageProfile};
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::DataFusionError,
    datasource::TableProvider,
};
use iceberg::{
    Catalog, CatalogBuilder as _, NamespaceIdent, TableIdent,
    io::{
        S3_ACCESS_KEY_ID, S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_ENDPOINT, S3_PATH_STYLE_ACCESS,
        S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
    },
};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};
use iceberg_catalog_rest::{REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder};
use iceberg_storage_opendal::OpenDalStorageFactory;
use tokio::sync::RwLock;

use crate::{
    Error, ErrorCode, Result,
    table_format::scan::{
        DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, TableFormatParquetReadStrategyConfig,
        credential_free_endpoint_identity,
    },
};

const CATALOG_BACKEND_OP: &str = "datafusion_cudf.iceberg.catalog_backend";

// ---------------------------------------------------------------------------
// Provider kind
// ---------------------------------------------------------------------------

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
            Ok(Arc::new(provider.with_catalog_type(catalog_type)))
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

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// DataFusion catalog provider
// ---------------------------------------------------------------------------

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
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema_names(&self) -> Vec<String> {
        self.schemas.keys().cloned().collect()
    }

    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.schemas.get(name).map(|s| Arc::clone(s) as Arc<dyn SchemaProvider>)
    }
}

// ---------------------------------------------------------------------------
// DataFusion schema provider
// ---------------------------------------------------------------------------

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
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    table_cache: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
}

impl CudfIcebergSchemaProvider {
    fn new(
        namespace: NamespaceIdent,
        known_table_names: Vec<String>,
        backend: Arc<dyn IcebergCatalogBackend>,
        catalog_type: impl Into<String>,
        provider_kind: IcebergProviderKind,
        parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    ) -> Self {
        Self {
            namespace,
            known_table_names,
            backend,
            catalog_type: catalog_type.into(),
            provider_kind,
            parquet_read_strategy_config,
            table_cache: RwLock::new(HashMap::new()),
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
        cache.entry(name).or_insert(provider);
    }
}

#[async_trait]
impl SchemaProvider for CudfIcebergSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.known_table_names.clone()
    }

    async fn table(&self, name: &str) -> datafusion::common::Result<Option<Arc<dyn TableProvider>>> {
        // Fast path: check the cache (covers pre-seeded aliases and already-loaded tables).
        {
            let cache = self.table_cache.read().await;
            if let Some(provider) = cache.get(name) {
                return Ok(Some(Arc::clone(provider)));
            }
        }

        // Only load tables that were discovered during startup.
        if !self.table_exist(name) {
            return Ok(None);
        }

        let ident = TableIdent::new(self.namespace.clone(), name.to_string());
        let table = self.backend.load_table(&ident).await.map_err(DataFusionError::from)?;
        let provider = build_table_provider(
            table,
            self.provider_kind,
            &self.catalog_type,
            self.parquet_read_strategy_config.clone(),
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

// ---------------------------------------------------------------------------
// Startup builder
// ---------------------------------------------------------------------------

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
    let catalog_type = catalog_type.into();
    let namespaces = backend.list_namespaces().await?;
    build_catalog_provider_for_namespaces(backend, catalog_type, namespaces, provider_kind).await
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
    let namespace = namespace_ident_from_flat_name(namespace)?;
    build_catalog_provider_for_namespaces(backend, catalog_type.into(), vec![namespace], provider_kind).await
}

async fn build_catalog_provider_for_namespaces(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    namespaces: Vec<NamespaceIdent>,
    provider_kind: IcebergProviderKind,
) -> Result<CudfIcebergCatalogProvider> {
    let mut schemas: BTreeMap<String, Arc<CudfIcebergSchemaProvider>> = BTreeMap::new();

    for namespace in namespaces {
        let ns_parts: &Vec<String> = namespace.as_ref();
        let schema_name = if ns_parts.len() == 1 { ns_parts[0].clone() } else { ns_parts.join(".") };

        let table_idents = backend.list_tables(&namespace).await?;
        let known_table_names: Vec<String> = table_idents.into_iter().map(|t| t.name).collect();

        let schema_provider = Arc::new(CudfIcebergSchemaProvider::new(
            namespace,
            known_table_names,
            Arc::clone(&backend),
            &catalog_type,
            provider_kind,
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

// ---------------------------------------------------------------------------
// Glue backend configuration
// ---------------------------------------------------------------------------

/// Catalog-level configuration for the AWS Glue Iceberg backend.
///
/// This covers the Glue catalog connection (warehouse, region, optional catalog
/// ID and profile).  Table identity is not part of this config; it is resolved
/// at runtime via [`IcebergCatalogBackend::list_tables`] and
/// [`IcebergCatalogBackend::load_table`].
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Builds the table-format Parquet read-strategy config for this Glue
    /// catalog. The config is credential-free and references the native source
    /// storage profile by id.
    pub fn parquet_read_strategy_config(&self) -> TableFormatParquetReadStrategyConfig {
        let mut config = TableFormatParquetReadStrategyConfig::from_env();
        config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
        config
    }

    /// Builds the native source storage profile for this Glue catalog using the
    /// same AWS credential environment accepted by the Glue catalog loader.
    pub fn source_storage_profile_with_env(&self, mut get: impl FnMut(&str) -> Option<String>) -> SourceStorageProfile {
        let mut s3 = S3SourceStorageProfile::default_chain().with_region(self.region.clone());
        if let (Some(access_key_id), Some(secret_access_key)) = (
            super::non_empty(get(crate::iceberg::AWS_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
        ) {
            let session_token = super::non_empty(
                get(crate::iceberg::AWS_SESSION_TOKEN_ENV).or_else(|| get(crate::iceberg::AWS_SECURITY_TOKEN_ENV)),
            );
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

// ---------------------------------------------------------------------------
// Glue backend implementation
// ---------------------------------------------------------------------------

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
        let glue = GlueCatalogBuilder::default().load(catalog_handle, props).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to connect to AWS Glue Iceberg catalog",
            )
            .with_fact("catalog_type", "glue")
            .with_source(error)
        })?;
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
        let mut props = HashMap::from([
            (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), config.warehouse.clone()),
            (AWS_REGION_NAME.to_string(), config.region.clone()),
        ]);
        if let Some(id) = config.catalog_id.as_ref().filter(|v| !v.trim().is_empty()) {
            props.insert(GLUE_CATALOG_PROP_CATALOG_ID.to_string(), id.clone());
        }
        if let Some(profile) = config.profile.as_ref().filter(|v| !v.trim().is_empty()) {
            props.insert(AWS_PROFILE_NAME.to_string(), profile.clone());
        }
        if let (Some(key_id), Some(secret)) = (
            super::non_empty(get(crate::iceberg::AWS_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
        ) {
            props.insert(AWS_ACCESS_KEY_ID.to_string(), key_id);
            props.insert(AWS_SECRET_ACCESS_KEY.to_string(), secret);
        }
        if let Some(token) = super::non_empty(
            get(crate::iceberg::AWS_SESSION_TOKEN_ENV).or_else(|| get(crate::iceberg::AWS_SECURITY_TOKEN_ENV)),
        ) {
            props.insert(AWS_SESSION_TOKEN.to_string(), token);
        }
        let parquet_read_strategy_config = config.parquet_read_strategy_config();
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
        self.catalog.list_namespaces(None).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to list namespaces from AWS Glue Iceberg catalog",
            )
            .with_fact("catalog_type", "glue")
            .with_source(error)
        })
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.catalog.list_tables(namespace).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to list tables from AWS Glue Iceberg catalog",
            )
            .with_fact("catalog_type", "glue")
            .with_fact("namespace", namespace.to_string())
            .with_source(error)
        })
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        self.catalog.load_table(table).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to load table from AWS Glue Iceberg catalog",
            )
            .with_fact("catalog_type", "glue")
            .with_fact("namespace", table.namespace().to_string())
            .with_fact("table", table.name())
            .with_source(error)
        })
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
    }
}

// ---------------------------------------------------------------------------
// REST backend configuration
// ---------------------------------------------------------------------------

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
/// `DATAFUSION_CUDF_ICEBERG_S3_*` environment variables (never the ambient
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
        let mut config = TableFormatParquetReadStrategyConfig::from_env();
        config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
        if let Some(endpoint) = self.s3_endpoint.as_ref().filter(|endpoint| !endpoint.trim().is_empty()) {
            config.storage_endpoint = Some(credential_free_endpoint_identity(endpoint));
        }
        config
    }

    /// Builds the native source storage profile for this REST catalog using the
    /// explicit `DATAFUSION_CUDF_ICEBERG_S3_*` credential environment.
    pub fn source_storage_profile_with_env(&self, mut get: impl FnMut(&str) -> Option<String>) -> SourceStorageProfile {
        let mut s3 = S3SourceStorageProfile::default_chain();
        if let Some(region) = self.s3_region.as_ref().filter(|region| !region.trim().is_empty()) {
            s3 = s3.with_region(region.clone());
        }
        if let Some(endpoint) = self.s3_endpoint.as_ref().filter(|endpoint| !endpoint.trim().is_empty()) {
            s3 = s3.with_endpoint(endpoint.clone());
        }
        if let Some(path_style) = self.s3_path_style_access {
            s3 = s3.with_path_style_access(path_style);
        }
        if let (Some(access_key_id), Some(secret_access_key)) = (
            super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV)),
            super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)),
        ) {
            let session_token = super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_SESSION_TOKEN_ENV));
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
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
/// [`crate::iceberg::CUDF_ICEBERG_S3_SESSION_TOKEN_ENV`] when present. Empty
/// optional values are omitted rather than inserted as empty strings.
///
/// S3 credentials come ONLY from the explicit `DATAFUSION_CUDF_ICEBERG_S3_*`
/// environment variables read here. To enforce that, the prop map always sets
/// `s3.disable-config-load=true` and `s3.disable-ec2-metadata=true`, which turn
/// off OpenDAL's ambient credential discovery (the `AWS_*` environment chain,
/// `~/.aws/*` config files, and the EC2 IMDS endpoint). Without these flags
/// OpenDAL would silently fall back to the ambient AWS chain when no explicit
/// creds are provided — exactly the silent `AWS_*` pickup this project forbids.
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
        // Default-deny ambient AWS credential discovery; see the doc comment.
        (S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string()),
        (S3_DISABLE_EC2_METADATA.to_string(), "true".to_string()),
    ]);
    if let Some(prefix) = config.prefix.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert("prefix".to_string(), prefix.clone());
    }
    if let Some(token) = config.token.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert("token".to_string(), token.clone());
    }
    if let Some(endpoint) = config.s3_endpoint.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert(S3_ENDPOINT.to_string(), endpoint.clone());
    }
    if let Some(region) = config.s3_region.as_ref().filter(|v| !v.trim().is_empty()) {
        props.insert(S3_REGION.to_string(), region.clone());
    }
    if let Some(path_style) = config.s3_path_style_access {
        props.insert(S3_PATH_STYLE_ACCESS.to_string(), path_style.to_string());
    }
    if let (Some(key_id), Some(secret)) = (
        super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV)),
        super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)),
    ) {
        props.insert(S3_ACCESS_KEY_ID.to_string(), key_id);
        props.insert(S3_SECRET_ACCESS_KEY.to_string(), secret);
    }
    if let Some(token) = super::non_empty(get(crate::iceberg::CUDF_ICEBERG_S3_SESSION_TOKEN_ENV)) {
        props.insert(S3_SESSION_TOKEN.to_string(), token);
    }
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

// ---------------------------------------------------------------------------
// REST backend implementation
// ---------------------------------------------------------------------------

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
            .map_err(|error| {
                Error::new(
                    ErrorCode::CudfDependencyFailure,
                    CATALOG_BACKEND_OP,
                    "failed to connect to Iceberg REST catalog",
                )
                .with_fact("catalog_type", "rest")
                .with_source(error)
            })?;
        Ok(Self { catalog: Arc::new(rest), parquet_read_strategy_config: None })
    }

    /// Builds a connected REST backend from a [`RestIcebergCatalogConfig`],
    /// resolving static S3 credentials through `get`.
    pub async fn from_config(
        config: RestIcebergCatalogConfig,
        get: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let props = rest_catalog_props(&config, get);
        let parquet_read_strategy_config = config.parquet_read_strategy_config();
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
        self.catalog.list_namespaces(None).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to list namespaces from Iceberg REST catalog",
            )
            .with_fact("catalog_type", "rest")
            .with_source(error)
        })
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.catalog.list_tables(namespace).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to list tables from Iceberg REST catalog",
            )
            .with_fact("catalog_type", "rest")
            .with_fact("namespace", namespace.to_string())
            .with_source(error)
        })
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        self.catalog.load_table(table).await.map_err(|error| {
            Error::new(
                ErrorCode::CudfDependencyFailure,
                CATALOG_BACKEND_OP,
                "failed to load table from Iceberg REST catalog",
            )
            .with_fact("catalog_type", "rest")
            .with_fact("namespace", table.namespace().to_string())
            .with_fact("table", table.name())
            .with_source(error)
        })
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
    }
}

#[cfg(test)]
mod rest_tests {
    use iceberg::io::{StorageConfig, StorageFactory};

    use super::*;

    fn no_env(_name: &str) -> Option<String> {
        None
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
            crate::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
            crate::iceberg::CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rest-secret".to_string()),
            crate::iceberg::CUDF_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rest-token-value".to_string()),
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
            crate::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
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
    fn rest_parquet_strategy_config_references_sanitized_source_storage_profile() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://user:secret@localhost:9000/catalog?signature=secret#frag");

        let strategy = config.parquet_read_strategy_config();

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000/catalog"));
    }

    #[test]
    fn rest_source_storage_profile_carries_s3_endpoint_path_style_and_static_credentials() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000")
            .with_s3_region("us-east-1")
            .with_s3_path_style_access(true);
        let env = |name: &str| match name {
            crate::iceberg::CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV => Some("AKIA_LOCAL".to_string()),
            crate::iceberg::CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV => Some("rest-secret".to_string()),
            crate::iceberg::CUDF_ICEBERG_S3_SESSION_TOKEN_ENV => Some("rest-token-value".to_string()),
            _ => None,
        };

        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(profile.id(), DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID);
        let s3 = profile.s3_profile();
        assert_eq!(s3.region.as_deref(), Some("us-east-1"));
        assert_eq!(s3.endpoint.as_deref(), Some("http://localhost:9000"));
        assert_eq!(s3.virtual_hosted_style_request, Some(false));
        match &s3.credential {
            cudf_query_engine::S3SourceStorageCredential::Static {
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
    fn glue_source_storage_profile_uses_glue_region_and_aws_static_credentials() {
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-west-2");
        let env = |name: &str| match name {
            crate::iceberg::AWS_ACCESS_KEY_ID_ENV => Some("AKIA_GLUE".to_string()),
            crate::iceberg::AWS_SECRET_ACCESS_KEY_ENV => Some("glue-secret".to_string()),
            crate::iceberg::AWS_SESSION_TOKEN_ENV => Some("glue-token".to_string()),
            _ => None,
        };

        let strategy = config.parquet_read_strategy_config();
        let profile = config.source_storage_profile_with_env(env);

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(profile.id(), DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID);
        assert_eq!(profile.s3_profile().region.as_deref(), Some("us-west-2"));
        match &profile.s3_profile().credential {
            cudf_query_engine::S3SourceStorageCredential::Static {
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
