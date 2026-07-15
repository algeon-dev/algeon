use super::*;

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
            crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV,
            )),
            crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV,
            )),
        ) {
            let session_token = crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
            ));
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
        } else if crate::table_format::iceberg::s3_credential_source_default_chain_opt_in(&mut get) {
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
    if !crate::table_format::iceberg::s3_credential_source_default_chain_opt_in(&mut get) {
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
    crate::table_format::iceberg::insert_explicit_nexus_s3_credentials(&mut props, |name| get(name), false);
    props
}

/// Builds the S3 `OpenDalStorageFactory` that the REST backend always injects.
///
/// REST has no default storage factory (it errors at file-IO time without one),
/// so every `RestIcebergCatalogBackend` constructor routes through this. Exposed
/// at crate scope so tests can assert the factory is wired (the upstream error
/// only fires at read time, not at construction).
fn rest_storage_factory() -> Arc<OpenDalStorageFactory> {
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
            preference: crate::table_format::TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/rest-cache")),
            lakehouse_cache_policy: crate::table_format::TableFormatLakehouseCachePolicy::FillBeforeRead,
            lakehouse_cache_max_bytes: Some(1024),
            ..Default::default()
        };
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_endpoint("http://localhost:9000");

        let strategy = config.parquet_read_strategy_config_from(injected);

        assert_eq!(strategy.preference, crate::table_format::TableFormatParquetReadStrategyPreference::LocalCache);
        assert_eq!(strategy.lakehouse_cache_root, Some(std::path::PathBuf::from("/tmp/rest-cache")));
        assert_eq!(
            strategy.lakehouse_cache_policy,
            crate::table_format::TableFormatLakehouseCachePolicy::FillBeforeRead
        );
        assert_eq!(strategy.lakehouse_cache_max_bytes, Some(1024));
        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.storage_endpoint.as_deref(), Some("http://localhost:9000"));
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
            nexus_query_engine::runtime::S3SourceStorageCredential::Static {
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

        assert_eq!(profile.s3_profile().credential, nexus_query_engine::runtime::S3SourceStorageCredential::Unsigned);
    }

    #[test]
    fn rest_parquet_strategy_marks_region_override_as_scoped() {
        let config = RestIcebergCatalogConfig::new("lake", "http://localhost:8181", "s3://lakehouse/warehouse")
            .with_s3_region("us-west-2");

        let strategy = config.parquet_read_strategy_config();
        let profile = config.source_storage_profile_with_env(|_| None);

        assert_eq!(strategy.storage_profile_id.as_deref(), Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID));
        assert_eq!(strategy.credential_mode, nexus_query_engine::source::CredentialMode::ScopedResolved);
        assert_eq!(profile.s3_profile().region.as_deref(), Some("us-west-2"));
        assert!(profile.requires_scoped_remote_direct_profile());
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
