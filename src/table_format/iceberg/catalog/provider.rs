use super::*;

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

/// Options applied when wrapping loaded Iceberg tables in DataFusion table providers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IcebergProviderOptions {
    /// Optional wall-clock budget for each table-format scan-planning phase.
    pub scan_planning_timeout: Option<Duration>,
}

impl IcebergProviderOptions {
    /// Sets the optional source-resolution and native-inspection timeout.
    pub fn with_scan_planning_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.scan_planning_timeout = timeout;
        self
    }
}

/// Wraps a loaded Iceberg table in the unified dual-mode provider.
fn build_table_provider(
    resolved: nexus_iceberg::ResolvedIcebergTable,
    catalog_type: &str,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    provider_options: &IcebergProviderOptions,
) -> Result<Arc<dyn TableProvider>> {
    let provider = match parquet_read_strategy_config {
        Some(config) => {
            crate::table_format::iceberg::NexusIcebergTableProvider::try_new_resolved_with_parquet_read_strategy_config(
                resolved, config,
            )?
        },
        None => crate::table_format::iceberg::NexusIcebergTableProvider::try_new_resolved(resolved)?,
    };
    Ok(Arc::new(
        provider.with_catalog_type(catalog_type).with_scan_planning_timeout(provider_options.scan_planning_timeout),
    ))
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

    /// Load and freeze the Iceberg table for the given identifier.
    ///
    /// The DataFusion wrapper consumes the returned resolution directly, so
    /// the CPU delegate, leaf scan candidate, and native source policy share
    /// the same snapshot and source-access facts.
    async fn load_table(&self, table: &TableIdent) -> Result<nexus_iceberg::ResolvedIcebergTable>;

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

/// DataFusion-facing view of a connected leaf catalog backend.
///
/// The leaf owns catalog connection, FileIO, and the immutable
/// [`nexus_iceberg::SourceAccessPolicy`] stored in every resolved
/// table. The root owns only the DataFusion conversion and its independent
/// table-format read strategy. Keeping that strategy here prevents a dynamic
/// catalog from silently dropping server cache or native read settings.
#[derive(Debug, Clone)]
pub struct NexusIcebergCatalogSource {
    backend: Arc<nexus_iceberg::IcebergCatalogBackend>,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
}

impl NexusIcebergCatalogSource {
    /// Adapts an owned, already-connected leaf backend for DataFusion.
    pub fn new(backend: nexus_iceberg::IcebergCatalogBackend) -> Self {
        Self::from_shared_backend(Arc::new(backend))
    }

    /// Adapts a shared, already-connected leaf backend for DataFusion.
    pub fn from_shared_backend(backend: Arc<nexus_iceberg::IcebergCatalogBackend>) -> Self {
        Self { backend, parquet_read_strategy_config: None }
    }

    /// Attaches root-owned native read strategy facts for all dynamic tables.
    pub fn with_parquet_read_strategy_config(mut self, config: TableFormatParquetReadStrategyConfig) -> Self {
        self.parquet_read_strategy_config = Some(config);
        self
    }

    /// Returns the connected leaf backend without exposing its credentials.
    pub const fn backend(&self) -> &Arc<nexus_iceberg::IcebergCatalogBackend> {
        &self.backend
    }
}

/// Converts leaf catalog identities and normalized failures at the one
/// DataFusion boundary. The leaf backend intentionally has no implementation
/// of this root trait: callers must choose an explicit root read strategy.
#[async_trait]
impl IcebergCatalogBackend for NexusIcebergCatalogSource {
    async fn list_namespaces(&self) -> Result<Vec<NamespaceIdent>> {
        self.backend
            .list_namespaces()
            .await
            .map_err(|error| {
                crate::table_format::iceberg::adapter::root_error_from_iceberg(
                    error,
                    "datafusion_nexus.table_format.iceberg.catalog.list_namespaces",
                    "failed to list Iceberg catalog namespaces for DataFusion adaptation",
                )
            })?
            .into_iter()
            .map(|namespace| {
                NamespaceIdent::from_vec(namespace.parts).map_err(|error| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.table_format.iceberg.catalog.list_namespaces",
                        "Iceberg leaf catalog returned an invalid namespace identity",
                    )
                    .with_source(error)
                })
            })
            .collect()
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.backend
            .list_tables(nexus_iceberg::IcebergNamespace { parts: namespace.as_ref().clone() })
            .await
            .map_err(|error| {
                crate::table_format::iceberg::adapter::root_error_from_iceberg(
                    error,
                    "datafusion_nexus.table_format.iceberg.catalog.list_tables",
                    "failed to list Iceberg catalog tables for DataFusion adaptation",
                )
            })?
            .into_iter()
            .map(|table| {
                let namespace = NamespaceIdent::from_vec(table.namespace).map_err(|error| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.table_format.iceberg.catalog.list_tables",
                        "Iceberg leaf catalog returned a table with an invalid namespace identity",
                    )
                    .with_source(error)
                })?;
                Ok(TableIdent::new(namespace, table.name))
            })
            .collect()
    }

    async fn load_table(&self, table: &TableIdent) -> Result<nexus_iceberg::ResolvedIcebergTable> {
        self.backend
            .load_table(nexus_iceberg::IcebergTableIdentifier {
                namespace: table.namespace().as_ref().clone(),
                name: table.name().to_string(),
            })
            .await
            .map_err(|error| {
                crate::table_format::iceberg::adapter::root_error_from_iceberg(
                    error,
                    "datafusion_nexus.table_format.iceberg.catalog.load_table",
                    "failed to load and freeze an Iceberg table for DataFusion adaptation",
                )
            })
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
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
pub struct NexusIcebergCatalogProvider {
    schemas: BTreeMap<String, Arc<NexusIcebergSchemaProvider>>,
}

impl NexusIcebergCatalogProvider {
    fn new(schemas: BTreeMap<String, Arc<NexusIcebergSchemaProvider>>) -> Self {
        Self { schemas }
    }

    /// Returns the concrete schema provider for `name`, if it exists.
    ///
    /// Use this to seed pre-loaded alias tables before registering the catalog
    /// in a [`datafusion::prelude::SessionContext`].
    pub fn schema_concrete(&self, name: &str) -> Option<Arc<NexusIcebergSchemaProvider>> {
        self.schemas.get(name).cloned()
    }
}

impl CatalogProvider for NexusIcebergCatalogProvider {
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
pub struct NexusIcebergSchemaProvider {
    namespace: NamespaceIdent,
    known_table_names: Vec<String>,
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    provider_options: IcebergProviderOptions,
    parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    table_cache: RwLock<HashMap<String, Arc<dyn TableProvider>>>,
    seeded_table_names: StdRwLock<BTreeSet<String>>,
}

impl NexusIcebergSchemaProvider {
    fn new(
        namespace: NamespaceIdent,
        known_table_names: Vec<String>,
        backend: Arc<dyn IcebergCatalogBackend>,
        catalog_type: impl Into<String>,
        provider_options: IcebergProviderOptions,
        parquet_read_strategy_config: Option<TableFormatParquetReadStrategyConfig>,
    ) -> Self {
        Self {
            namespace,
            known_table_names,
            backend,
            catalog_type: catalog_type.into(),
            provider_options,
            parquet_read_strategy_config,
            table_cache: RwLock::new(HashMap::new()),
            seeded_table_names: StdRwLock::new(BTreeSet::new()),
        }
    }

    /// Seeds a pre-loaded table provider under `name`.
    ///
    /// Call this before registering the parent [`NexusIcebergCatalogProvider`]
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
impl SchemaProvider for NexusIcebergSchemaProvider {
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
            &self.catalog_type,
            self.parquet_read_strategy_config.clone(),
            &self.provider_options,
        )
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

/// Builds a [`NexusIcebergCatalogProvider`] by discovering all namespaces and
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
) -> Result<NexusIcebergCatalogProvider> {
    build_catalog_provider_from_backend_with_options(backend, catalog_type, IcebergProviderOptions::default()).await
}

/// Builds a [`NexusIcebergCatalogProvider`] with explicit provider options.
pub async fn build_catalog_provider_from_backend_with_options(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    provider_options: IcebergProviderOptions,
) -> Result<NexusIcebergCatalogProvider> {
    let catalog_type = catalog_type.into();
    let namespaces = backend.list_namespaces().await?;
    build_catalog_provider_for_namespaces(backend, catalog_type, namespaces, provider_options).await
}

/// Builds a [`NexusIcebergCatalogProvider`] scoped to one configured namespace.
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
) -> Result<NexusIcebergCatalogProvider> {
    build_scoped_catalog_provider_from_backend_with_options(
        backend,
        catalog_type,
        namespace,
        IcebergProviderOptions::default(),
    )
    .await
}

/// Builds a namespace-scoped [`NexusIcebergCatalogProvider`] with explicit provider options.
pub async fn build_scoped_catalog_provider_from_backend_with_options(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: impl Into<String>,
    namespace: &str,
    provider_options: IcebergProviderOptions,
) -> Result<NexusIcebergCatalogProvider> {
    let namespace = namespace_ident_from_flat_name(namespace)?;
    build_catalog_provider_for_namespaces(backend, catalog_type.into(), vec![namespace], provider_options).await
}

async fn build_catalog_provider_for_namespaces(
    backend: Arc<dyn IcebergCatalogBackend>,
    catalog_type: String,
    namespaces: Vec<NamespaceIdent>,
    provider_options: IcebergProviderOptions,
) -> Result<NexusIcebergCatalogProvider> {
    let mut schemas: BTreeMap<String, Arc<NexusIcebergSchemaProvider>> = BTreeMap::new();

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

        let schema_provider = Arc::new(NexusIcebergSchemaProvider::new(
            namespace,
            known_table_names,
            Arc::clone(&backend),
            &catalog_type,
            provider_options.clone(),
            backend.parquet_read_strategy_config(),
        ));
        schemas.insert(schema_name, schema_provider);
    }

    Ok(NexusIcebergCatalogProvider::new(schemas))
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

        async fn load_table(&self, _table: &TableIdent) -> Result<nexus_iceberg::ResolvedIcebergTable> {
            unreachable!("this test only inspects schema-provider construction")
        }

        fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
            self.parquet_read_strategy_config.clone()
        }
    }

    #[tokio::test]
    async fn catalog_provider_carries_backend_strategy_into_schema_provider() {
        let strategy = TableFormatParquetReadStrategyConfig {
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/provider-cache")),
            lakehouse_cache_policy: crate::table_format::TableFormatLakehouseCachePolicy::FillBeforeRead,
            ..Default::default()
        };
        let backend = Arc::new(StrategyMockBackend { parquet_read_strategy_config: Some(strategy.clone()) });

        let provider = build_scoped_catalog_provider_from_backend(backend, "rest", "tpch_sf1").await.unwrap();
        let schema = provider.schema_concrete("tpch_sf1").expect("scoped schema provider");

        assert_eq!(schema.parquet_read_strategy_config, Some(strategy));
    }
}
