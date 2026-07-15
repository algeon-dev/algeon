use super::*;

pub(super) async fn catalog_backend_list_namespaces(
    catalog: &Arc<dyn Catalog>,
    catalog_type: &'static str,
    message: &'static str,
) -> Result<Vec<NamespaceIdent>> {
    catalog.list_namespaces(None).await.map_err(|error| catalog_backend_error(message, catalog_type, error))
}

pub(super) async fn catalog_backend_list_tables(
    catalog: &Arc<dyn Catalog>,
    namespace: &NamespaceIdent,
    catalog_type: &'static str,
    message: &'static str,
) -> Result<Vec<TableIdent>> {
    catalog.list_tables(namespace).await.map_err(|error| {
        catalog_backend_error(message, catalog_type, error).with_fact("namespace", namespace.to_string())
    })
}

pub(super) async fn catalog_backend_load_table(
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
pub(super) struct S3FileIoConfig<'a> {
    endpoint: Option<&'a str>,
    region: Option<&'a str>,
    path_style_access: Option<bool>,
}

impl<'a> S3FileIoConfig<'a> {
    pub(super) fn from_glue(config: &'a GlueIcebergCatalogConfig) -> Self {
        Self {
            endpoint: config.s3_endpoint.as_deref(),
            region: config.s3_region.as_deref(),
            path_style_access: config.s3_path_style_access,
        }
    }

    pub(super) fn from_rest(config: &'a RestIcebergCatalogConfig) -> Self {
        Self {
            endpoint: config.s3_endpoint.as_deref(),
            region: config.s3_region.as_deref(),
            path_style_access: config.s3_path_style_access,
        }
    }

    pub(super) fn apply_parquet_read_strategy_config(
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
            config.credential_mode = nexus_query_engine::source::CredentialMode::ScopedResolved;
        }
        if self.non_empty_endpoint().is_some() && self.path_style_access == Some(false) {
            config.remote_direct_profile_unsupported_reason = Some("remote_direct_virtual_hosted_endpoint_unsupported");
        }
        config
    }

    pub(super) fn apply_catalog_props(self, props: &mut HashMap<String, String>) {
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

    pub(super) fn apply_source_storage_profile(
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

    pub(super) fn non_empty_endpoint(self) -> Option<&'a str> {
        self.endpoint.filter(|endpoint| !endpoint.trim().is_empty())
    }

    pub(super) fn non_empty_region(self) -> Option<&'a str> {
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
/// startup based on [`crate::server::ServerSessionProfile`] (when the
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

#[cfg(test)]
mod tests {
    use super::*;

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
            preference: crate::table_format::TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/provider-cache")),
            lakehouse_cache_policy: crate::table_format::TableFormatLakehouseCachePolicy::FillBeforeRead,
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
}
