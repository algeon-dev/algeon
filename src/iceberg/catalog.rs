//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`IcebergCatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`]; individual tables are
//! loaded lazily on first access.
//!
//! # Backend implementations
//!
//! - [`GlueIcebergCatalogBackend`] — AWS Glue via `iceberg-catalog-glue`
//! - REST and Hive backends can be added as additional implementations of
//!   [`IcebergCatalogBackend`] without changing the DataFusion-facing layer.

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::DataFusionError,
    datasource::TableProvider,
};
use iceberg::{Catalog, CatalogBuilder as _, NamespaceIdent, TableIdent};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};
use tokio::sync::RwLock;

use crate::{Error, ErrorCode, Result};

const CATALOG_BACKEND_OP: &str = "datafusion_cudf.iceberg.catalog_backend";

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Backend-neutral Iceberg catalog listing and table-loading interface.
///
/// All methods are `async` because catalog operations require network calls.
/// DataFusion's synchronous [`CatalogProvider`] and [`SchemaProvider`] methods
/// read from a startup cache populated by [`build_catalog_provider_from_backend`];
/// [`SchemaProvider::table`] (which *is* async) loads tables lazily.
#[async_trait]
pub trait IcebergCatalogBackend: fmt::Debug + Send + Sync {
    /// List all top-level namespaces exposed by this catalog.
    async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>>;

    /// List all tables in the given namespace.
    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>>;

    /// Load the Iceberg table for the given identifier.
    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table>;
}

// ---------------------------------------------------------------------------
// DataFusion catalog provider
// ---------------------------------------------------------------------------

/// DataFusion [`CatalogProvider`] backed by a startup-cached Iceberg namespace
/// list.
///
/// Built by [`build_catalog_provider_from_backend`].  Each schema maps to one
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
#[derive(Debug)]
pub struct CudfIcebergSchemaProvider {
    namespace: NamespaceIdent,
    known_table_names: Vec<String>,
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    table_cache: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
}

impl CudfIcebergSchemaProvider {
    fn new(
        namespace: NamespaceIdent,
        known_table_names: Vec<String>,
        backend: Arc<dyn IcebergCatalogBackend>,
        catalog_type: impl Into<String>,
    ) -> Self {
        Self {
            namespace,
            known_table_names,
            backend,
            catalog_type: catalog_type.into(),
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
        let provider: Arc<dyn TableProvider> = Arc::new(
            crate::table_format::iceberg::CudfIcebergTableProvider::try_new(table)
                .map_err(DataFusionError::from)?
                .with_catalog_type(&self.catalog_type),
        );

        // Insert into cache; under concurrent first-access races both paths
        // produce equivalent providers — first write wins, the loser's provider
        // is discarded, subsequent callers see the winner.
        {
            let mut cache = self.table_cache.write().await;
            cache.entry(name.to_string()).or_insert_with(|| Arc::clone(&provider));
        }

        Ok(Some(provider))
    }

    /// Returns `TableType::Base` for any known or seeded table without loading
    /// the full Iceberg metadata.  All Iceberg tables are base tables; this
    /// override avoids the backend round-trip that the default `table_type()`
    /// implementation would trigger via `table()`.
    async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<datafusion::datasource::TableType>> {
        if self.table_exist(name) {
            return Ok(Some(datafusion::datasource::TableType::Base));
        }
        let cache = self.table_cache.read().await;
        if cache.contains_key(name) { Ok(Some(datafusion::datasource::TableType::Base)) } else { Ok(None) }
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
/// to form a flat DataFusion schema name.
pub async fn build_catalog_provider_from_backend(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
) -> Result<CudfIcebergCatalogProvider> {
    let catalog_type = catalog_type.into();
    let namespaces = backend.list_namespaces().await?;
    let mut schemas: BTreeMap<String, Arc<CudfIcebergSchemaProvider>> = BTreeMap::new();

    for namespace in namespaces {
        let ns_parts: &Vec<String> = namespace.as_ref();
        let schema_name = if ns_parts.len() == 1 { ns_parts[0].clone() } else { ns_parts.join(".") };

        let table_idents = backend.list_tables(&namespace).await?;
        let known_table_names: Vec<String> = table_idents.into_iter().map(|t| t.name).collect();

        let schema_provider =
            Arc::new(CudfIcebergSchemaProvider::new(namespace, known_table_names, Arc::clone(&backend), &catalog_type));
        schemas.insert(schema_name, schema_provider);
    }

    Ok(CudfIcebergCatalogProvider::new(schemas))
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
    pub warehouse: String,
    pub region: String,
    pub catalog_id: Option<String>,
    pub profile: Option<String>,
}

impl GlueIcebergCatalogConfig {
    pub fn new(catalog_handle: impl Into<String>, warehouse: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            catalog_handle: catalog_handle.into(),
            warehouse: warehouse.into(),
            region: region.into(),
            catalog_id: None,
            profile: None,
        }
    }

    pub fn with_catalog_id(mut self, catalog_id: impl Into<String>) -> Self {
        self.catalog_id = Some(catalog_id.into());
        self
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
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
}

impl fmt::Debug for GlueIcebergCatalogBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueIcebergCatalogBackend").finish_non_exhaustive()
    }
}

impl GlueIcebergCatalogBackend {
    /// Builds a connected Glue backend from pre-assembled catalog properties.
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
        Ok(Self { catalog: Arc::new(glue) })
    }

    /// Builds a connected Glue backend from a [`GlueIcebergCatalogConfig`],
    /// reading AWS credentials from the provided environment accessor.
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
        Self::from_props(config.catalog_handle, props).await
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
}
