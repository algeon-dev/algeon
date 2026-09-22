#[cfg(feature = "iceberg")]
use std::{fmt, path::PathBuf, time::Duration};

use super::{env::EnvValues, *};
#[cfg(not(feature = "iceberg"))]
use crate::table_format::{
    ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV, ALGEON_LAKEHOUSE_CACHE_POLICY_ENV, ALGEON_LAKEHOUSE_CACHE_READ_ENV,
    ALGEON_LAKEHOUSE_CACHE_ROOT_ENV,
};

/// DataFusion catalog name exposed to SQL for Iceberg tables.
pub const ALGEON_ICEBERG_CATALOG_NAME_ENV: &str = "ALGEON_ICEBERG_CATALOG_NAME";
/// Iceberg namespace exposed to SQL and required when Iceberg server
/// registration is configured.
pub const ALGEON_ICEBERG_NAMESPACE_ENV: &str = "ALGEON_ICEBERG_NAMESPACE";
/// Optional comma-separated Iceberg table registrations. Entries may be
/// `table` or `alias=table`; when unset, tables resolve lazily through the
/// dynamic catalog.
pub const ALGEON_ICEBERG_TABLES_ENV: &str = "ALGEON_ICEBERG_TABLES";

/// Selects which Iceberg catalog backend the server connects to. Accepts
/// `glue` (default when unset) or `rest`. Any other value is rejected at
/// config load time.
pub const ALGEON_ICEBERG_CATALOG_KIND_ENV: &str = "ALGEON_ICEBERG_CATALOG_KIND";

/// Glue catalog handle passed to iceberg-rust.
pub const ALGEON_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "ALGEON_ICEBERG_GLUE_CATALOG_NAME";
/// Iceberg warehouse location, typically an `s3://...` URI. Required for both
/// Glue and REST Iceberg backends.
pub const ALGEON_ICEBERG_WAREHOUSE_ENV: &str = "ALGEON_ICEBERG_WAREHOUSE";
/// Optional AWS Glue catalog ID used to scope Glue requests.
pub const ALGEON_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "ALGEON_ICEBERG_GLUE_CATALOG_ID";
/// Optional Glue-compatible endpoint URI, mainly for local or test catalogs.
pub const ALGEON_ICEBERG_GLUE_URI_ENV: &str = "ALGEON_ICEBERG_GLUE_URI";

/// REST Iceberg catalog base URI. Required when `ALGEON_ICEBERG_CATALOG_KIND=rest`.
pub const ALGEON_ICEBERG_REST_URI_ENV: &str = "ALGEON_ICEBERG_REST_URI";
/// Optional REST Iceberg catalog prefix, commonly used for tenant or warehouse routing.
pub const ALGEON_ICEBERG_REST_PREFIX_ENV: &str = "ALGEON_ICEBERG_REST_PREFIX";
/// Optional REST Iceberg bearer token. Treated as a secret and redacted from
/// diagnostics.
pub const ALGEON_ICEBERG_REST_TOKEN_ENV: &str = "ALGEON_ICEBERG_REST_TOKEN";
/// Internal REST catalog handle passed to iceberg-rust. Independent of the
/// DataFusion catalog name and defaults to `rest`.
pub const ALGEON_ICEBERG_REST_CATALOG_HANDLE_ENV: &str = "ALGEON_ICEBERG_REST_CATALOG_HANDLE";
/// S3-compatible object-store endpoint for Iceberg data, such as MinIO or
/// RustFS. Applies to both Glue and REST Iceberg backends.
pub const ALGEON_ICEBERG_S3_ENDPOINT_ENV: &str = "ALGEON_ICEBERG_S3_ENDPOINT";
/// S3 region for Iceberg object storage. Applies to both Glue and REST Iceberg backends.
pub const ALGEON_ICEBERG_S3_REGION_ENV: &str = "ALGEON_ICEBERG_S3_REGION";
/// Enables path-style S3 addressing for Iceberg object storage. Use `true` for
/// many local S3-compatible stores.
pub const ALGEON_ICEBERG_S3_PATH_STYLE_ENV: &str = "ALGEON_ICEBERG_S3_PATH_STYLE";
/// Explicit S3 credential source for Iceberg object storage. The only accepted
/// value is `default_chain`: an operator opt-in to the ambient AWS default
/// credential chain (env vars, profile, IMDS instance role), including with a
/// custom `ALGEON_ICEBERG_S3_ENDPOINT` where the default is otherwise unsigned
/// requests. Mutually exclusive with static `ALGEON_ICEBERG_S3_*` credentials.
pub const ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV: &str = "ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE";
/// Seconds allowed for Iceberg table-format scan planning before Flight SQL
/// returns a structured `planning_timeout` error. Defaults to `60`.
pub const ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV: &str = "ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS";
#[cfg(feature = "iceberg")]
const DEFAULT_GLUE_CATALOG_NAME: &str = "glue";
#[cfg(feature = "iceberg")]
const DEFAULT_REST_CATALOG_HANDLE: &str = "rest";
#[cfg(feature = "iceberg")]
const DEFAULT_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS: u64 = 60;
#[cfg(feature = "iceberg")]
#[derive(Clone, PartialEq, Eq)]
pub struct IcebergServerConfig {
    /// DataFusion catalog name (e.g. "iceberg" or "glue").  Exposed as the
    /// first part of `catalog.namespace.table` three-part addressing.
    pub catalog_name: String,
    /// Iceberg namespace / Glue database name (e.g. "tpch_sf1"). This is the
    /// dynamic catalog visibility scope and the second part of three-part addressing.
    pub namespace: String,
    /// Pre-declared table alias registrations seeded before catalog
    /// registration. Empty means dynamic-catalog-only (all tables resolve
    /// lazily via three-part addressing).
    pub tables: Vec<IcebergTableRegistrationConfig>,
    /// Backend-specific connection payload. The enum makes invalid
    /// combinations unrepresentable (a Glue payload cannot carry a REST URI and
    /// vice versa). Selected by [`ALGEON_ICEBERG_CATALOG_KIND_ENV`].
    pub backend: IcebergCatalogBackendConfig,
    /// The one frozen storage policy used by the leaf catalog/FileIO and
    /// converted by the root exactly once into the native source profile.
    /// It owns all credential selection; no server path may rediscover those
    /// credentials from the process environment later.
    source_access: crate::server::table_format::iceberg::SourceAccessPolicy,
    /// Server-validated table-format Parquet strategy passed into catalog
    /// planning paths. Server mode must not let those paths reread process env.
    pub parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
    /// Wall-clock budget applied to Iceberg scan planning.
    pub scan_planning_timeout: Duration,
}

#[cfg(feature = "iceberg")]
impl fmt::Debug for IcebergServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergServerConfig")
            .field("catalog_name", &self.catalog_name)
            .field("namespace", &self.namespace)
            .field("tables", &self.tables)
            .field("backend", &self.backend)
            .field("source_access", &self.source_access)
            .field("parquet_read_strategy_config", &"<redacted>")
            .field("scan_planning_timeout", &self.scan_planning_timeout)
            .finish()
    }
}

#[cfg(feature = "iceberg")]
impl IcebergServerConfig {
    /// Returns the source-access policy frozen while parsing server config.
    ///
    /// The connected leaf catalog places this exact policy in every
    /// `ResolvedTable`; the root uses the same value to build its
    /// native source-storage profile.
    pub const fn source_access_policy(&self) -> &crate::server::table_format::iceberg::SourceAccessPolicy {
        &self.source_access
    }
}

/// Backend-specific Iceberg catalog connection config.
///
/// Validation lives per-variant: the Glue branch requires a region; the REST
/// branch requires a URI and never requires a region.
#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcebergCatalogBackendConfig {
    Glue(GlueServerBackendConfig),
    Rest(crate::server::table_format::iceberg::catalog::RestCatalogConfig),
}

/// Glue-backend connection config carried by [`IcebergCatalogBackendConfig::Glue`].
///
/// `catalog_handle` is the internal handle passed to iceberg-rust's
/// `GlueCatalogBuilder`; it is independent of the DataFusion `catalog_name` so
/// operators can choose a Trino-style connector name without changing the Glue
/// catalog identifier. `region` is required for Glue. `catalog_id` / `profile`
/// are optional AWS scoping knobs. `uri` and `s3_*` fields support local
/// Glue-compatible tests against Moto plus an S3-compatible object store.
#[cfg(feature = "iceberg")]
#[derive(Clone, PartialEq, Eq)]
pub struct GlueServerBackendConfig {
    pub catalog_handle: String,
    pub warehouse: String,
    pub region: String,
    pub catalog_id: Option<String>,
    pub profile: Option<String>,
    pub uri: Option<String>,
    pub s3_endpoint: Option<String>,
    pub s3_region: Option<String>,
    pub s3_path_style_access: Option<bool>,
}

#[cfg(feature = "iceberg")]
impl fmt::Debug for GlueServerBackendConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueServerBackendConfig")
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

#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTableRegistrationConfig {
    pub table: String,
    pub alias: String,
}
#[cfg(feature = "iceberg")]
pub(super) fn iceberg_source_storage_profile(
    config: &IcebergServerConfig,
) -> query_engine::runtime::SourceStorageProfile {
    crate::server::table_format::iceberg::source_access_policy_to_native_source_storage_profile(
        config.source_access_policy(),
    )
}

#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogKind {
    Glue,
    Rest,
}

#[cfg(feature = "iceberg")]
fn iceberg_catalog_kind(values: &mut EnvValues<'_>) -> Result<CatalogKind> {
    match values.string(ALGEON_ICEBERG_CATALOG_KIND_ENV) {
        // A defaulted backend selects the wrong catalog for someone: fail
        // closed instead of optimizing for one deployment style.
        None => Err(ServerError::invalid_config(
            ALGEON_ICEBERG_CATALOG_KIND_ENV,
            "is required when Iceberg settings are present; set `glue` or `rest`",
        )),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "glue" => Ok(CatalogKind::Glue),
            "rest" => Ok(CatalogKind::Rest),
            other => Err(ServerError::invalid_config(
                ALGEON_ICEBERG_CATALOG_KIND_ENV,
                format!("must be `glue` or `rest`, got `{other}`"),
            )),
        },
    }
}

#[cfg(feature = "iceberg")]
pub(super) fn iceberg_config(values: &mut EnvValues<'_>) -> Result<Option<IcebergServerConfig>> {
    let requested = values.any_present([
        ALGEON_ICEBERG_CATALOG_KIND_ENV,
        ALGEON_ICEBERG_CATALOG_NAME_ENV,
        ALGEON_ICEBERG_NAMESPACE_ENV,
        ALGEON_ICEBERG_TABLES_ENV,
        ALGEON_ICEBERG_GLUE_CATALOG_NAME_ENV,
        ALGEON_ICEBERG_WAREHOUSE_ENV,
        ALGEON_ICEBERG_GLUE_CATALOG_ID_ENV,
        ALGEON_ICEBERG_GLUE_URI_ENV,
        ALGEON_ICEBERG_REST_URI_ENV,
        ALGEON_ICEBERG_REST_PREFIX_ENV,
        ALGEON_ICEBERG_REST_TOKEN_ENV,
        ALGEON_ICEBERG_REST_CATALOG_HANDLE_ENV,
        ALGEON_ICEBERG_S3_ENDPOINT_ENV,
        ALGEON_ICEBERG_S3_REGION_ENV,
        ALGEON_ICEBERG_S3_PATH_STYLE_ENV,
        ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
        ALGEON_LAKEHOUSE_CACHE_ROOT_ENV,
        ALGEON_LAKEHOUSE_CACHE_READ_ENV,
        ALGEON_LAKEHOUSE_CACHE_POLICY_ENV,
        ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
        ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    let kind = iceberg_catalog_kind(values)?;

    let namespace = values.required_string(ALGEON_ICEBERG_NAMESPACE_ENV)?;

    let warehouse = values.required_string(ALGEON_ICEBERG_WAREHOUSE_ENV)?;

    // When unset, the dynamic catalog serves all tables and aliases are not
    // required (an empty vector is valid).
    let raw_tables = values.string(ALGEON_ICEBERG_TABLES_ENV);
    let tables = match raw_tables {
        Some(ref raw) => parse_iceberg_tables(ALGEON_ICEBERG_TABLES_ENV, raw)?,
        None => vec![],
    };

    let default_catalog_name = match kind {
        CatalogKind::Glue => DEFAULT_GLUE_CATALOG_NAME,
        CatalogKind::Rest => DEFAULT_REST_CATALOG_HANDLE,
    };
    let catalog_name =
        values.string(ALGEON_ICEBERG_CATALOG_NAME_ENV).unwrap_or_else(|| default_catalog_name.to_string());

    let backend = match kind {
        CatalogKind::Glue => IcebergCatalogBackendConfig::Glue(glue_backend_config(values, &warehouse)?),
        CatalogKind::Rest => IcebergCatalogBackendConfig::Rest(rest_backend_config(values, warehouse.clone())?),
    };
    let source_access = iceberg_source_access_policy(values, &backend)?;
    let parquet_read_strategy_config = iceberg_parquet_read_strategy_config(values, &source_access)?;
    let scan_planning_timeout = Duration::from_secs(
        values
            .positive_u64(ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, DEFAULT_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS)?,
    );

    Ok(Some(IcebergServerConfig {
        catalog_name,
        namespace,
        tables,
        backend,
        source_access,
        parquet_read_strategy_config,
        scan_planning_timeout,
    }))
}

#[cfg(feature = "iceberg")]
pub(super) fn iceberg_parquet_read_strategy_config(
    values: &mut EnvValues<'_>,
    source_access_policy: &crate::server::table_format::iceberg::SourceAccessPolicy,
) -> Result<TableFormatParquetReadStrategyConfig> {
    let mut config = table_format_parquet_read_strategy_config(values)?;
    config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
    let source_access = source_access_policy.s3_access();
    config.storage_endpoint =
        source_access.endpoint.filter(|endpoint| !endpoint.trim().is_empty()).map(credential_free_endpoint_identity);
    if source_access.endpoint.is_some()
        || source_access.region.is_some()
        || source_access.path_style_access.is_some()
        || !matches!(source_access.credential, iceberg_catalog::S3CredentialRef::DefaultChain)
    {
        config.credential_mode = query_engine::source::CredentialMode::ScopedResolved;
    }
    Ok(config)
}

#[cfg(feature = "iceberg")]
fn iceberg_source_access_policy(
    values: &mut EnvValues<'_>,
    backend: &IcebergCatalogBackendConfig,
) -> Result<crate::server::table_format::iceberg::SourceAccessPolicy> {
    let (configured_endpoint, region, path_style_access, allow_aws_static_credentials) = match backend {
        IcebergCatalogBackendConfig::Glue(glue) => (
            glue.s3_endpoint.clone(),
            glue.s3_region.clone().or_else(|| Some(glue.region.clone())),
            glue.s3_path_style_access,
            true,
        ),
        IcebergCatalogBackendConfig::Rest(_) => (
            values.string(ALGEON_ICEBERG_S3_ENDPOINT_ENV),
            values.string(ALGEON_ICEBERG_S3_REGION_ENV),
            values.optional_bool(ALGEON_ICEBERG_S3_PATH_STYLE_ENV)?,
            false,
        ),
    };
    // Standard AWS endpoint overrides are accepted by the existing server
    // strategy parser. Fold them into the leaf policy too, so FileIO, CPU,
    // and native access cannot select different endpoints.
    let endpoint = configured_endpoint
        .or_else(|| values.string(AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| values.string(AWS_ENDPOINT_URL_ENV));
    let mut s3 = match endpoint {
        Some(endpoint) if !endpoint.trim().is_empty() => {
            crate::server::table_format::iceberg::SourceAccessPolicy::s3_custom_endpoint(
                endpoint,
                region,
                path_style_access,
            )
            .as_s3()
            .clone()
        },
        _ => crate::server::table_format::iceberg::SourceAccessPolicy::s3_default_chain(region).as_s3().clone(),
    };

    let explicit_credentials = (
        values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_ACCESS_KEY_ID_ENV),
        values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY_ENV),
        values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_SESSION_TOKEN_ENV),
    );
    let explicit_default_chain = values.string(ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV).is_some_and(|raw| {
        raw.trim().eq_ignore_ascii_case(
            crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN,
        )
    });
    let aws_credentials = (
        values.string(crate::server::table_format::iceberg::AWS_ACCESS_KEY_ID_ENV),
        values.string(crate::server::table_format::iceberg::AWS_SECRET_ACCESS_KEY_ENV),
        values
            .string(crate::server::table_format::iceberg::AWS_SESSION_TOKEN_ENV)
            .or_else(|| values.string(crate::server::table_format::iceberg::AWS_SECURITY_TOKEN_ENV)),
    );

    s3 = match explicit_credentials {
        (Some(access_key_id), Some(secret_access_key), session_token) => {
            s3.with_static_credentials(access_key_id, secret_access_key, session_token)
        },
        _ if explicit_default_chain => s3.with_default_chain(),
        _ if allow_aws_static_credentials && s3.endpoint().is_none() => match aws_credentials {
            (Some(access_key_id), Some(secret_access_key), session_token) => {
                s3.with_static_credentials(access_key_id, secret_access_key, session_token)
            },
            _ => s3,
        },
        _ => s3,
    };
    Ok(crate::server::table_format::iceberg::SourceAccessPolicy::S3(s3))
}

#[cfg(feature = "iceberg")]
pub(super) fn table_format_parquet_read_strategy_config(
    values: &mut EnvValues<'_>,
) -> Result<TableFormatParquetReadStrategyConfig> {
    let lakehouse_cache_root = values.string(ALGEON_LAKEHOUSE_CACHE_ROOT_ENV).map(PathBuf::from);
    let lakehouse_cache_read_policy = values
        .string(ALGEON_LAKEHOUSE_CACHE_READ_ENV)
        .map(|raw| parse_lakehouse_cache_read_policy(&raw))
        .transpose()?
        .unwrap_or_default();
    let lakehouse_cache_policy = values
        .string(ALGEON_LAKEHOUSE_CACHE_POLICY_ENV)
        .map(|raw| parse_lakehouse_cache_policy(&raw))
        .transpose()?
        .unwrap_or_default();
    let lakehouse_cache_max_bytes = match values.optional_u64(ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV)? {
        Some(0) => {
            return Err(ServerError::invalid_config(ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "must be greater than zero"));
        },
        Some(value) => Some(value),
        None => None,
    };
    if lakehouse_cache_root.is_some()
        && lakehouse_cache_policy != TableFormatLakehouseCachePolicy::Disabled
        && lakehouse_cache_max_bytes.is_none()
    {
        return Err(ServerError::invalid_config(
            ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
            "is required when a lakehouse cache root is configured with a filling policy; \
             an unbounded disk cache has no retention target",
        ));
    }
    let storage_endpoint = values
        .string(ALGEON_ICEBERG_S3_ENDPOINT_ENV)
        .or_else(|| values.string(AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| values.string(AWS_ENDPOINT_URL_ENV))
        .map(|endpoint| credential_free_endpoint_identity(&endpoint));
    let has_explicit_s3_credentials =
        values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_ACCESS_KEY_ID_ENV).is_some()
            || values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY_ENV).is_some()
            || values.string(crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_SESSION_TOKEN_ENV).is_some();
    let credential_source = values.string(ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV);
    if let Some(raw) = credential_source.as_deref() {
        if !raw.eq_ignore_ascii_case(
            crate::server::table_format::iceberg::ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN,
        ) {
            return Err(ServerError::invalid_config(
                ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                format!("must be `default_chain` when set, got `{raw}`"),
            ));
        }
        if has_explicit_s3_credentials {
            return Err(ServerError::invalid_config(
                ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                "cannot be combined with static ALGEON_ICEBERG_S3_ACCESS_KEY_ID / \
                 ALGEON_ICEBERG_S3_SECRET_ACCESS_KEY / ALGEON_ICEBERG_S3_SESSION_TOKEN credentials",
            ));
        }
    }
    let has_storage_region = values.string(ALGEON_ICEBERG_S3_REGION_ENV).is_some();
    let credential_mode = if storage_endpoint.is_some()
        || has_storage_region
        || has_explicit_s3_credentials
        || credential_source.is_some()
    {
        query_engine::source::CredentialMode::ScopedResolved
    } else {
        query_engine::source::CredentialMode::SharedProcessEnv
    };

    Ok(TableFormatParquetReadStrategyConfig {
        lakehouse_cache_root,
        lakehouse_cache_read_policy,
        lakehouse_cache_policy,
        lakehouse_cache_max_bytes,
        storage_endpoint,
        credential_mode,
        ..Default::default()
    })
}

#[cfg(feature = "iceberg")]
pub(super) fn parse_lakehouse_cache_read_policy(raw: &str) -> Result<TableFormatLakehouseCacheReadPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "enabled" => Ok(TableFormatLakehouseCacheReadPolicy::Enabled),
        "disabled" => Ok(TableFormatLakehouseCacheReadPolicy::Disabled),
        other => Err(ServerError::invalid_config(
            ALGEON_LAKEHOUSE_CACHE_READ_ENV,
            format!("must be `enabled` or `disabled`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
pub(super) fn parse_lakehouse_cache_policy(raw: &str) -> Result<TableFormatLakehouseCachePolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "disabled" => Ok(TableFormatLakehouseCachePolicy::Disabled),
        "read_through" => Ok(TableFormatLakehouseCachePolicy::ReadThrough),
        "fill_before_read" => Ok(TableFormatLakehouseCachePolicy::FillBeforeRead),
        other => Err(ServerError::invalid_config(
            ALGEON_LAKEHOUSE_CACHE_POLICY_ENV,
            format!("must be `disabled`, `read_through`, or `fill_before_read`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
pub(super) fn glue_backend_config(values: &mut EnvValues<'_>, warehouse: &str) -> Result<GlueServerBackendConfig> {
    let region = values
        .string(crate::server::table_format::iceberg::AWS_REGION_ENV)
        .or_else(|| values.string(crate::server::table_format::iceberg::AWS_DEFAULT_REGION_ENV))
        .ok_or_else(|| {
            ServerError::invalid_config(
                crate::server::table_format::iceberg::AWS_REGION_ENV,
                "is required for Glue Iceberg",
            )
        })?;

    // catalog_handle is the internal handle passed to iceberg-rust's
    // GlueCatalogBuilder; it is independent of the DataFusion catalog_name so
    // users can choose a different connector name without changing the Glue
    // catalog identifier.  Controlled only by the Glue-specific env var.
    let catalog_handle =
        values.string(ALGEON_ICEBERG_GLUE_CATALOG_NAME_ENV).unwrap_or_else(|| DEFAULT_GLUE_CATALOG_NAME.to_string());

    Ok(GlueServerBackendConfig {
        catalog_handle,
        warehouse: warehouse.to_string(),
        region,
        catalog_id: values.string(ALGEON_ICEBERG_GLUE_CATALOG_ID_ENV),
        profile: values.string(crate::server::table_format::iceberg::AWS_PROFILE_ENV),
        uri: values.string(ALGEON_ICEBERG_GLUE_URI_ENV),
        s3_endpoint: values.string(ALGEON_ICEBERG_S3_ENDPOINT_ENV),
        s3_region: values.string(ALGEON_ICEBERG_S3_REGION_ENV),
        s3_path_style_access: values.optional_bool(ALGEON_ICEBERG_S3_PATH_STYLE_ENV)?,
    })
}

#[cfg(feature = "iceberg")]
pub(super) fn rest_backend_config(
    values: &mut EnvValues<'_>,
    warehouse: String,
) -> Result<crate::server::table_format::iceberg::catalog::RestCatalogConfig> {
    let uri = values.required_string(ALGEON_ICEBERG_REST_URI_ENV)?;
    let catalog_handle = values
        .string(ALGEON_ICEBERG_REST_CATALOG_HANDLE_ENV)
        .unwrap_or_else(|| DEFAULT_REST_CATALOG_HANDLE.to_string());

    let mut config =
        crate::server::table_format::iceberg::catalog::RestCatalogConfig::new(catalog_handle, uri, warehouse);
    if let Some(prefix) = values.string(ALGEON_ICEBERG_REST_PREFIX_ENV) {
        config = config.with_prefix(prefix);
    }
    if let Some(token) = values.string(ALGEON_ICEBERG_REST_TOKEN_ENV) {
        config = config.with_bearer_token(token);
    }
    Ok(config)
}

#[cfg(feature = "iceberg")]
pub(super) fn parse_iceberg_tables(env_name: &'static str, raw: &str) -> Result<Vec<IcebergTableRegistrationConfig>> {
    let mut tables = Vec::new();
    for entry in raw.split(',').map(str::trim).filter(|entry| !entry.is_empty()) {
        let (alias, table) = match entry.split_once('=') {
            Some((alias, table)) => (alias.trim(), table.trim()),
            None => (entry, entry),
        };
        if alias.is_empty() || table.is_empty() {
            return Err(ServerError::invalid_config(env_name, "entries must be table or alias=table"));
        }
        tables.push(IcebergTableRegistrationConfig { table: table.to_string(), alias: alias.to_string() });
    }
    if tables.is_empty() {
        return Err(ServerError::invalid_config(env_name, "must contain at least one table"));
    }
    Ok(tables)
}

#[cfg(not(feature = "iceberg"))]
pub(super) fn reject_iceberg_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    if values.any_present([
        ALGEON_ICEBERG_CATALOG_KIND_ENV,
        ALGEON_ICEBERG_CATALOG_NAME_ENV,
        ALGEON_ICEBERG_NAMESPACE_ENV,
        ALGEON_ICEBERG_TABLES_ENV,
        ALGEON_ICEBERG_GLUE_CATALOG_NAME_ENV,
        ALGEON_ICEBERG_WAREHOUSE_ENV,
        ALGEON_ICEBERG_GLUE_CATALOG_ID_ENV,
        ALGEON_ICEBERG_GLUE_URI_ENV,
        ALGEON_ICEBERG_REST_URI_ENV,
        ALGEON_ICEBERG_REST_PREFIX_ENV,
        ALGEON_ICEBERG_REST_TOKEN_ENV,
        ALGEON_ICEBERG_REST_CATALOG_HANDLE_ENV,
        ALGEON_ICEBERG_S3_ENDPOINT_ENV,
        ALGEON_ICEBERG_S3_REGION_ENV,
        ALGEON_ICEBERG_S3_PATH_STYLE_ENV,
        ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
        ALGEON_LAKEHOUSE_CACHE_ROOT_ENV,
        ALGEON_LAKEHOUSE_CACHE_READ_ENV,
        ALGEON_LAKEHOUSE_CACHE_POLICY_ENV,
        ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
        ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            ALGEON_ICEBERG_NAMESPACE_ENV,
            "Iceberg server registration requires building with -p algeon-server --features iceberg",
        ));
    }
    Ok(())
}
