use super::*;

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
            crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV,
            )),
            crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV,
            )),
            crate::table_format::iceberg::non_empty(get(
                crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
            )),
        );
        let aws_credentials = (
            crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)),
            crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
            crate::table_format::iceberg::non_empty(
                get(crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
                    .or_else(|| get(crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
            ),
        );
        if let (Some(access_key_id), Some(secret_access_key), session_token) = explicit_s3_credentials {
            s3 = s3.with_static_credentials(access_key_id, secret_access_key, session_token);
        } else if crate::table_format::iceberg::s3_credential_source_default_chain_opt_in(&mut get) {
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct GlueCatalogCredentialFacts {
    profile: Option<String>,
    credential_source: crate::table_format::iceberg::GlueIcebergCredentialSource,
}

impl GlueCatalogCredentialFacts {
    fn from_config_env(
        config: &GlueIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
    ) -> GlueCatalogCredentialFacts {
        let profile = config
            .profile
            .clone()
            .or_else(|| crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_PROFILE_ENV)));
        let has_access_key =
            crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)).is_some();
        let has_secret_key =
            crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV))
                .is_some();
        let has_session_token = crate::table_format::iceberg::non_empty(
            get(crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
                .or_else(|| get(crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
        )
        .is_some();
        let credential_source = crate::table_format::iceberg::GlueIcebergCredentialSource::from_parts(
            profile.as_deref(),
            has_access_key,
            has_secret_key,
            has_session_token,
        );
        Self { profile, credential_source }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct GlueCatalogBackendErrorContext {
    catalog_handle: Option<String>,
    catalog_id: Option<String>,
    region: Option<String>,
    credential_facts: Option<GlueCatalogCredentialFacts>,
}

impl GlueCatalogBackendErrorContext {
    fn from_config(config: &GlueIcebergCatalogConfig, credential_facts: GlueCatalogCredentialFacts) -> Self {
        Self {
            catalog_handle: crate::table_format::iceberg::non_empty(Some(config.catalog_handle.clone())),
            catalog_id: config
                .catalog_id
                .clone()
                .and_then(|value| crate::table_format::iceberg::non_empty(Some(value))),
            region: crate::table_format::iceberg::non_empty(Some(config.region.clone())),
            credential_facts: Some(credential_facts),
        }
    }

    pub(super) fn apply(&self, mut error: Error) -> Error {
        if let Some(catalog_handle) = self.catalog_handle.as_ref() {
            error = error.with_fact("catalog_handle", catalog_handle);
        }
        if let Some(catalog_id) = self.catalog_id.as_ref() {
            error = error.with_fact("catalog_id", catalog_id);
        }
        if let Some(region) = self.region.as_ref() {
            error = error.with_fact("region", region);
        }
        if let Some(credential_facts) = self.credential_facts.as_ref() {
            error = error.with_fact("credential_source", credential_facts.credential_source.code());
            if let Some(profile) = credential_facts.profile.as_ref() {
                error = error.with_fact("aws_profile", profile);
            }
        }
        error
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
        if !crate::table_format::iceberg::s3_credential_source_default_chain_opt_in(&mut get) {
            props.insert(S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
            props.insert(S3_DISABLE_EC2_METADATA.to_string(), "true".to_string());
        }
    } else {
        s3_file_io.apply_catalog_props(&mut props);
    }

    if let (Some(key_id), Some(secret)) = (
        crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV)),
        crate::table_format::iceberg::non_empty(get(crate::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV)),
    ) {
        props.insert(AWS_ACCESS_KEY_ID.to_string(), key_id);
        props.insert(AWS_SECRET_ACCESS_KEY.to_string(), secret);
    }
    if let Some(token) = crate::table_format::iceberg::non_empty(
        get(crate::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
            .or_else(|| get(crate::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
    ) {
        props.insert(AWS_SESSION_TOKEN.to_string(), token);
    }

    crate::table_format::iceberg::insert_explicit_nexus_s3_credentials(&mut props, |name| get(name), true);

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
    error_context: GlueCatalogBackendErrorContext,
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
        Ok(Self {
            catalog: Arc::new(glue),
            parquet_read_strategy_config: None,
            error_context: GlueCatalogBackendErrorContext::default(),
        })
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
        let credential_facts = GlueCatalogCredentialFacts::from_config_env(&config, |name| get(name));
        let props = glue_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config()
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(
            config,
            props,
            parquet_read_strategy_config,
            credential_facts,
        )
        .await
    }

    /// Builds a connected Glue backend from a config and caller-owned
    /// table-format Parquet read-strategy config.
    pub async fn from_config_with_parquet_read_strategy_config(
        config: GlueIcebergCatalogConfig,
        mut get: impl FnMut(&str) -> Option<String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let credential_facts = GlueCatalogCredentialFacts::from_config_env(&config, |name| get(name));
        let props = glue_catalog_props(&config, |name| get(name));
        let source_storage_profile = config.source_storage_profile_with_env(|name| get(name));
        let parquet_read_strategy_config = config
            .parquet_read_strategy_config_from(parquet_read_strategy_config)
            .with_source_storage_profile_remote_direct_capability(&source_storage_profile);
        Self::from_config_with_resolved_parquet_read_strategy_config(
            config,
            props,
            parquet_read_strategy_config,
            credential_facts,
        )
        .await
    }

    async fn from_config_with_resolved_parquet_read_strategy_config(
        config: GlueIcebergCatalogConfig,
        props: HashMap<String, String>,
        parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
        credential_facts: GlueCatalogCredentialFacts,
    ) -> Result<Self> {
        let error_context = GlueCatalogBackendErrorContext::from_config(&config, credential_facts);
        let glue = GlueCatalogBuilder::default().load(config.catalog_handle.clone(), props).await.map_err(|error| {
            glue_catalog_backend_error("failed to connect to AWS Glue Iceberg catalog", &error_context, error)
        })?;
        let mut backend = Self { catalog: Arc::new(glue), parquet_read_strategy_config: None, error_context };
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
        .map_err(|error| self.error_context.apply(error))
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        catalog_backend_list_tables(
            &self.catalog,
            namespace,
            "glue",
            "failed to list tables from AWS Glue Iceberg catalog",
        )
        .await
        .map_err(|error| self.error_context.apply(error))
    }

    async fn load_table(&self, table: &TableIdent) -> Result<iceberg::table::Table> {
        catalog_backend_load_table(&self.catalog, table, "glue", "failed to load table from AWS Glue Iceberg catalog")
            .await
            .map_err(|error| self.error_context.apply(error))
    }

    fn parquet_read_strategy_config(&self) -> Option<TableFormatParquetReadStrategyConfig> {
        self.parquet_read_strategy_config.clone()
    }
}

#[cfg(test)]
mod tests {
    use iceberg::io::{S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN};

    use super::*;

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
    fn glue_catalog_backend_error_includes_safe_credential_context_for_profile() {
        let config = GlueIcebergCatalogConfig::new("glue-handle", "s3://lakehouse/warehouse", "us-west-2")
            .with_catalog_id("123456789012")
            .with_profile("crem-int");
        let credential_facts = GlueCatalogCredentialFacts::from_config_env(&config, no_env);
        let error_context = GlueCatalogBackendErrorContext::from_config(&config, credential_facts);

        let error = glue_catalog_backend_error(
            "failed to connect to AWS Glue Iceberg catalog",
            &error_context,
            glue_expired_token_error(),
        );

        assert_eq!(error.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(error.fact_value("credential_source"), Some("aws_profile_env"));
        assert_eq!(error.fact_value("aws_profile"), Some("crem-int"));
        assert_eq!(error.fact_value("catalog_handle"), Some("glue-handle"));
        assert_eq!(error.fact_value("catalog_id"), Some("123456789012"));
        assert_eq!(error.fact_value("region"), Some("us-west-2"));
        assert_eq!(error.fact_value("aws_error_detail"), Some("expired_token"));
    }

    #[test]
    fn glue_catalog_backend_error_includes_default_chain_without_profile_fact() {
        let config = GlueIcebergCatalogConfig::new("glue-handle", "s3://lakehouse/warehouse", "us-west-2");
        let credential_facts = GlueCatalogCredentialFacts::from_config_env(&config, no_env);
        let error_context = GlueCatalogBackendErrorContext::from_config(&config, credential_facts);

        let error = glue_catalog_backend_error(
            "failed to connect to AWS Glue Iceberg catalog",
            &error_context,
            glue_expired_token_error(),
        );

        assert_eq!(error.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(error.fact_value("credential_source"), Some("aws_default_chain"));
        assert_eq!(error.fact_value("aws_profile"), None);
    }

    #[test]
    fn glue_catalog_backend_error_context_preserves_namespace_fact() {
        let config = GlueIcebergCatalogConfig::new("glue-handle", "s3://lakehouse/warehouse", "us-west-2")
            .with_profile("crem-int");
        let credential_facts = GlueCatalogCredentialFacts::from_config_env(&config, no_env);
        let error_context = GlueCatalogBackendErrorContext::from_config(&config, credential_facts);

        let error = error_context.apply(
            catalog_backend_error(
                "failed to list tables from AWS Glue Iceberg catalog",
                "glue",
                glue_expired_token_error(),
            )
            .with_fact("namespace", "nexus_graph"),
        );

        assert_eq!(error.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(error.fact_value("namespace"), Some("nexus_graph"));
        assert_eq!(error.fact_value("credential_source"), Some("aws_profile_env"));
        assert_eq!(error.fact_value("aws_profile"), Some("crem-int"));
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
        assert_eq!(strategy.credential_mode, nexus_query_engine::source::CredentialMode::ScopedResolved);
        assert_eq!(profile.id(), DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID);
        assert_eq!(profile.s3_profile().region.as_deref(), Some("us-west-2"));
        match &profile.s3_profile().credential {
            nexus_query_engine::runtime::S3SourceStorageCredential::Static {
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
            nexus_query_engine::runtime::S3SourceStorageCredential::Static {
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
            preference: crate::table_format::TableFormatParquetReadStrategyPreference::RemoteKvikio,
            lakehouse_cache_root: Some(std::path::PathBuf::from("/tmp/glue-cache")),
            lakehouse_cache_policy: crate::table_format::TableFormatLakehouseCachePolicy::ReadThrough,
            lakehouse_cache_max_bytes: Some(2048),
            storage_endpoint: Some("https://ambient.example.invalid".to_string()),
            ..Default::default()
        };
        let config = GlueIcebergCatalogConfig::new("glue", "s3://lakehouse/warehouse", "us-east-1")
            .with_s3_endpoint("http://user:secret@localhost:9000/catalog?signature=secret#frag");

        let strategy = config.parquet_read_strategy_config_from(injected);

        assert_eq!(strategy.preference, crate::table_format::TableFormatParquetReadStrategyPreference::RemoteKvikio);
        assert_eq!(strategy.lakehouse_cache_root, Some(std::path::PathBuf::from("/tmp/glue-cache")));
        assert_eq!(strategy.lakehouse_cache_policy, crate::table_format::TableFormatLakehouseCachePolicy::ReadThrough);
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

        assert_eq!(profile.s3_profile().credential, nexus_query_engine::runtime::S3SourceStorageCredential::Unsigned);
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
}
