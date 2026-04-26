//! User-facing Iceberg integration helpers.
//!
//! This module is available with the `iceberg` Cargo feature. Lower-level,
//! format-neutral scan facts remain under [`crate::table_format`].

mod adapter;
pub mod catalog;
mod planner;

use std::{fmt, sync::Arc};

pub use adapter::iceberg_scan_candidate_to_table_format_scan_plan;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, TableReference},
    datasource::TableProvider,
    prelude::SessionContext,
};
pub use datafusion_nexus_iceberg::{
    IcebergTableIdentifier, ResolvedIcebergTable, S3CredentialMode, SourceAccessPolicy,
};
use iceberg::table::Table;

pub use self::planner::{
    IcebergTableFormatPlanner, NexusIcebergTableProvider, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema, source_access_policy_to_native_source_storage_profile,
};

const LOAD_GLUE_TABLE_OPERATION: &str = "datafusion_nexus.iceberg.load_glue_table";

/// Glue catalog name used when constructing an Iceberg catalog from env.
pub const NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_NAME";
/// Glue database name used by env-driven Iceberg helpers.
pub const NEXUS_ICEBERG_GLUE_DATABASE_ENV: &str = "NEXUS_ICEBERG_GLUE_DATABASE";
/// Glue table name used by env-driven Iceberg helpers.
pub const NEXUS_ICEBERG_GLUE_TABLE_ENV: &str = "NEXUS_ICEBERG_GLUE_TABLE";
/// Iceberg warehouse URI, typically an `s3://...` root.
pub const NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "NEXUS_ICEBERG_GLUE_WAREHOUSE";
/// Optional Glue catalog id for cross-account catalog targeting.
pub const NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_ID";

/// Optional AWS profile name. Non-secret; preferred over raw env credentials.
pub const AWS_PROFILE_ENV: &str = "AWS_PROFILE";
/// Preferred AWS region env var.
pub const AWS_REGION_ENV: &str = "AWS_REGION";
/// Fallback AWS region env var when `AWS_REGION` is unset.
pub const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";
/// AWS access key id consumed at load time and never retained in diagnostics.
pub const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";
/// AWS secret key consumed at load time and never retained in diagnostics.
pub const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";
/// AWS session token consumed at load time and never retained in diagnostics.
pub const AWS_SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";
/// Legacy session-token env var accepted as a fallback.
pub const AWS_SECURITY_TOKEN_ENV: &str = "AWS_SECURITY_TOKEN";

/// Explicit S3 access key id for the REST FileIO path.
///
/// Deliberately distinct from `AWS_ACCESS_KEY_ID` so a local REST/RustFS stack
/// never silently picks up a developer's real AWS credentials. Consumed at
/// connect time and never retained in diagnostics.
pub const NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV: &str = "NEXUS_ICEBERG_S3_ACCESS_KEY_ID";
/// Explicit S3 secret access key for the REST FileIO path. Secret; consumed at
/// connect time and never retained in diagnostics. See
/// [`NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV`] for why this is not `AWS_*`.
pub const NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV: &str = "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY";
/// Optional explicit S3 session token for the REST FileIO path. Secret;
/// consumed at connect time and never retained in diagnostics.
pub const NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV: &str = "NEXUS_ICEBERG_S3_SESSION_TOKEN";
/// Optional S3-compatible endpoint for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_ENDPOINT_ENV: &str = "NEXUS_ICEBERG_S3_ENDPOINT";
/// Optional S3 region for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_REGION_ENV: &str = "NEXUS_ICEBERG_S3_REGION";
/// Optional S3 path-style flag for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_PATH_STYLE_ENV: &str = "NEXUS_ICEBERG_S3_PATH_STYLE";
/// Optional explicit S3 credential source for Glue/REST object storage.
///
/// The only accepted value is `default_chain`. Setting it is an explicit
/// operator opt-in to the ambient AWS default credential chain (env vars,
/// profile, IMDS instance role) for the configured object storage — including
/// when a custom `NEXUS_ICEBERG_S3_ENDPOINT` is set, where the default is
/// otherwise unsigned requests so a local S3-compatible stack never silently
/// picks up developer AWS credentials. Mutually exclusive with the static
/// `NEXUS_ICEBERG_S3_ACCESS_KEY_ID`/`NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY` pair.
pub const NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV: &str = "NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE";
/// Accepted value of [`NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV`] selecting the
/// ambient AWS default credential chain.
pub const NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN: &str = "default_chain";

/// Whether the env explicitly opts object storage into the AWS default
/// credential chain. Only the exact value `default_chain` (case-insensitive)
/// opts in; validation of other values is the caller's responsibility.
pub(crate) fn s3_credential_source_default_chain_opt_in(mut get: impl FnMut(&str) -> Option<String>) -> bool {
    non_empty(get(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV))
        .is_some_and(|value| value.trim().eq_ignore_ascii_case(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN))
}

/// Builds the adapter-owned Iceberg provider that plans scans into native
/// table-format facts instead of delegating to an Arrow reader.
pub fn table_provider(table: Table) -> crate::error::Result<NexusIcebergTableProvider> {
    NexusIcebergTableProvider::try_new(table)
}

/// Registers an Iceberg table provider on a DataFusion session context.
///
/// The returned provider still goes through native table-format scan lowering;
/// this helper only wires the table into DataFusion's catalog.
pub fn register_table(
    ctx: &SessionContext,
    table_ref: impl Into<TableReference>,
    table: Table,
) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
    let provider = table_provider(table).map_err(DataFusionError::from)?;
    ctx.register_table(table_ref, Arc::new(provider))
}

/// AWS Glue Iceberg table-load configuration.
///
/// This struct stores table/catalog options and non-secret auth selectors such
/// as profile and region. Raw AWS key material is read only at load time and is
/// not retained here.
#[derive(Clone, PartialEq, Eq)]
pub struct GlueIcebergTableConfig {
    /// Iceberg catalog name passed to `iceberg-catalog-glue`.
    pub catalog_name: String,
    /// Glue database namespace.
    pub database: String,
    /// Glue table name within `database`; distinct from DataFusion registration name.
    pub table: String,
    /// Iceberg warehouse URI passed to the Glue catalog builder.
    pub warehouse: String,
    /// Optional AWS account/catalog id for cross-account Glue access.
    pub catalog_id: Option<String>,
    /// AWS region used for Glue and S3 calls.
    pub region: String,
    /// Optional AWS profile selector; raw credentials are still read only at load time.
    pub profile: Option<String>,
    /// Optional Glue-compatible endpoint URI, e.g. a local Moto server.
    pub uri: Option<String>,
    /// Optional FileIO `s3.endpoint` for S3-compatible object stores.
    pub s3_endpoint: Option<String>,
    /// Optional FileIO `s3.region`; defaults to [`Self::region`] when unset.
    pub s3_region: Option<String>,
    /// Optional FileIO `s3.path-style-access`.
    pub s3_path_style_access: Option<bool>,
}

impl fmt::Debug for GlueIcebergTableConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueIcebergTableConfig")
            .field("catalog_name", &self.catalog_name)
            .field("database", &self.database)
            .field("table", &self.table)
            .field("warehouse", &self.warehouse)
            .field("catalog_id", &self.catalog_id)
            .field("region", &self.region)
            .field("profile", &self.profile)
            .field("uri", &self.uri.as_ref().map(|_| "<redacted>"))
            .field("s3_endpoint", &self.s3_endpoint.as_ref().map(|_| "<redacted>"))
            .field("s3_region", &self.s3_region)
            .field("s3_path_style_access", &self.s3_path_style_access)
            .finish()
    }
}

impl GlueIcebergTableConfig {
    /// Builds a Glue table config with required catalog/table identity.
    pub fn new(
        catalog_name: impl Into<String>,
        database: impl Into<String>,
        table: impl Into<String>,
        warehouse: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        Self {
            catalog_name: catalog_name.into(),
            database: database.into(),
            table: table.into(),
            warehouse: warehouse.into(),
            catalog_id: None,
            region: region.into(),
            profile: None,
            uri: None,
            s3_endpoint: None,
            s3_region: None,
            s3_path_style_access: None,
        }
    }

    pub fn with_catalog_name(mut self, catalog_name: impl Into<String>) -> Self {
        self.catalog_name = catalog_name.into();
        self
    }

    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    pub fn with_warehouse(mut self, warehouse: impl Into<String>) -> Self {
        self.warehouse = warehouse.into();
        self
    }

    pub fn with_catalog_id(mut self, catalog_id: impl Into<String>) -> Self {
        self.catalog_id = Some(catalog_id.into());
        self
    }

    pub fn without_catalog_id(mut self) -> Self {
        self.catalog_id = None;
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    pub fn without_profile(mut self) -> Self {
        self.profile = None;
        self
    }

    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }

    pub fn without_uri(mut self) -> Self {
        self.uri = None;
        self
    }

    pub fn with_s3_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.s3_endpoint = Some(endpoint.into());
        self
    }

    pub fn without_s3_endpoint(mut self) -> Self {
        self.s3_endpoint = None;
        self
    }

    pub fn with_s3_region(mut self, region: impl Into<String>) -> Self {
        self.s3_region = Some(region.into());
        self
    }

    pub fn without_s3_region(mut self) -> Self {
        self.s3_region = None;
        self
    }

    pub fn with_s3_path_style_access(mut self, path_style: bool) -> Self {
        self.s3_path_style_access = Some(path_style);
        self
    }

    pub fn without_s3_path_style_access(mut self) -> Self {
        self.s3_path_style_access = None;
        self
    }

    pub fn table_ident_display(&self) -> String {
        format!("{}.{}", self.database, self.table)
    }

    /// Builds credential-safe diagnostics from process environment.
    pub fn diagnostics_from_env(&self) -> GlueIcebergTableDiagnostics {
        self.diagnostics_with_env(|name| std::env::var(name).ok())
    }

    /// Freezes this legacy single-table configuration into the leaf-owned
    /// source policy. Callers that must pre-register a native S3 profile use
    /// this policy rather than rebuilding endpoint or credential choices.
    pub fn source_access_policy_from_env(&self) -> SourceAccessPolicy {
        let environment = GlueLoadEnvironment::from_env(|name| std::env::var(name).ok());
        let diagnostics = self.diagnostics_from_environment(&environment);
        self.leaf_catalog_config(&environment, &diagnostics).source_access_policy().clone()
    }

    /// Builds credential-safe diagnostics from an injected env lookup.
    pub fn diagnostics_with_env(&self, mut get: impl FnMut(&str) -> Option<String>) -> GlueIcebergTableDiagnostics {
        let profile = self.profile.clone().or_else(|| non_empty(get(AWS_PROFILE_ENV)));
        let has_access_key = non_empty(get(AWS_ACCESS_KEY_ID_ENV)).is_some();
        let has_secret_key = non_empty(get(AWS_SECRET_ACCESS_KEY_ENV)).is_some();
        let has_session_token = non_empty(get(AWS_SESSION_TOKEN_ENV).or_else(|| get(AWS_SECURITY_TOKEN_ENV))).is_some();
        let credential_source = GlueIcebergCredentialSource::from_parts(
            profile.as_deref(),
            has_access_key,
            has_secret_key,
            has_session_token,
        );

        GlueIcebergTableDiagnostics {
            catalog_name: self.catalog_name.clone(),
            database: self.database.clone(),
            table: self.table.clone(),
            warehouse: self.warehouse.clone(),
            catalog_id: self.catalog_id.clone(),
            region: self.region.clone(),
            profile,
            credential_source,
        }
    }

    fn diagnostics_from_environment(&self, environment: &GlueLoadEnvironment) -> GlueIcebergTableDiagnostics {
        let profile = self.profile.clone().or_else(|| environment.aws_profile.clone());
        let credential_source = GlueIcebergCredentialSource::from_parts(
            profile.as_deref(),
            environment.aws_access_key_id.is_some(),
            environment.aws_secret_access_key.is_some(),
            environment.aws_session_token.is_some(),
        );
        GlueIcebergTableDiagnostics {
            catalog_name: self.catalog_name.clone(),
            database: self.database.clone(),
            table: self.table.clone(),
            warehouse: self.warehouse.clone(),
            catalog_id: self.catalog_id.clone(),
            region: self.region.clone(),
            profile,
            credential_source,
        }
    }

    /// Produces the leaf-owned catalog configuration from one frozen process
    /// environment snapshot. The policy drives both the leaf FileIO and root
    /// native-profile conversion through the resulting `ResolvedIcebergTable`.
    fn leaf_catalog_config(
        &self,
        environment: &GlueLoadEnvironment,
        diagnostics: &GlueIcebergTableDiagnostics,
    ) -> catalog::GlueIcebergCatalogConfig {
        let storage_region = non_empty(self.s3_region.clone()).or_else(|| non_empty(Some(self.region.clone())));
        let endpoint = non_empty(self.s3_endpoint.clone());
        let mut s3 = match endpoint {
            Some(endpoint) => {
                SourceAccessPolicy::s3_custom_endpoint(endpoint, storage_region.clone(), self.s3_path_style_access)
                    .as_s3()
                    .clone()
            },
            None => SourceAccessPolicy::s3_default_chain(storage_region.clone()).as_s3().clone(),
        };
        s3 = match (&environment.nexus_access_key_id, &environment.nexus_secret_access_key) {
            (Some(access_key_id), Some(secret_access_key)) => s3.with_static_credentials(
                access_key_id.clone(),
                secret_access_key.clone(),
                environment.nexus_session_token.clone(),
            ),
            _ if environment.default_chain_opt_in() => s3.with_default_chain(),
            _ if s3.endpoint().is_none() => {
                match (&environment.aws_access_key_id, &environment.aws_secret_access_key) {
                    (Some(access_key_id), Some(secret_access_key)) => s3.with_static_credentials(
                        access_key_id.clone(),
                        secret_access_key.clone(),
                        environment.aws_session_token.clone(),
                    ),
                    _ => s3,
                }
            },
            _ => s3,
        };

        let mut catalog_config = catalog::GlueIcebergCatalogConfig::new(
            self.catalog_name.clone(),
            self.warehouse.clone(),
            self.region.clone(),
        )
        .with_source_access_policy(SourceAccessPolicy::S3(s3));
        if diagnostics.profile.is_none()
            && let (Some(access_key_id), Some(secret_access_key)) =
                (&environment.aws_access_key_id, &environment.aws_secret_access_key)
        {
            catalog_config = catalog_config.with_catalog_static_credentials(
                access_key_id.clone(),
                secret_access_key.clone(),
                environment.aws_session_token.clone(),
            );
        }
        if let Some(catalog_id) = non_empty(self.catalog_id.clone()) {
            catalog_config = catalog_config.with_catalog_id(catalog_id);
        }
        if let Some(profile) = diagnostics.profile.as_ref().filter(|profile| !profile.trim().is_empty()) {
            catalog_config = catalog_config.with_profile(profile.clone());
        }
        if let Some(uri) = self.uri.as_ref().filter(|uri| !uri.trim().is_empty()) {
            catalog_config = catalog_config.with_uri(uri.clone());
        }
        catalog_config
    }
}

/// The environment facts consumed exactly once while resolving a legacy
/// single-table Glue request. This keeps the leaf FileIO and the later native
/// profile on one immutable credential/access-policy decision.
#[derive(Debug, Clone, Default)]
struct GlueLoadEnvironment {
    aws_profile: Option<String>,
    aws_access_key_id: Option<String>,
    aws_secret_access_key: Option<String>,
    aws_session_token: Option<String>,
    nexus_access_key_id: Option<String>,
    nexus_secret_access_key: Option<String>,
    nexus_session_token: Option<String>,
    nexus_credential_source: Option<String>,
}

impl GlueLoadEnvironment {
    fn from_env(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            aws_profile: non_empty(get(AWS_PROFILE_ENV)),
            aws_access_key_id: non_empty(get(AWS_ACCESS_KEY_ID_ENV)),
            aws_secret_access_key: non_empty(get(AWS_SECRET_ACCESS_KEY_ENV)),
            aws_session_token: non_empty(get(AWS_SESSION_TOKEN_ENV).or_else(|| get(AWS_SECURITY_TOKEN_ENV))),
            nexus_access_key_id: non_empty(get(NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)),
            nexus_secret_access_key: non_empty(get(NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)),
            nexus_session_token: non_empty(get(NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV)),
            nexus_credential_source: non_empty(get(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV)),
        }
    }

    fn default_chain_opt_in(&self) -> bool {
        self.nexus_credential_source
            .as_deref()
            .is_some_and(|source| source.trim().eq_ignore_ascii_case(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN))
    }
}

/// Credential source classification for Glue/S3 diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlueIcebergCredentialSource {
    /// Explicit profile from config or `AWS_PROFILE`.
    AwsProfileEnv,
    /// Access key, secret key, and session token env vars were all present.
    StsEnv,
    /// Access key and secret key env vars were present without a session token.
    StaticKeyEnv,
    /// The AWS SDK default chain will decide credentials.
    AwsDefaultChain,
}

impl GlueIcebergCredentialSource {
    /// Stable diagnostic code for this credential source class.
    pub const fn code(self) -> &'static str {
        match self {
            Self::AwsProfileEnv => "aws_profile_env",
            Self::StsEnv => "sts_env",
            Self::StaticKeyEnv => "static_key_env",
            Self::AwsDefaultChain => "aws_default_chain",
        }
    }

    fn from_parts(profile: Option<&str>, has_access_key: bool, has_secret_key: bool, has_session_token: bool) -> Self {
        if matches!(profile, Some(profile) if !profile.is_empty()) {
            Self::AwsProfileEnv
        } else if has_access_key && has_secret_key && has_session_token {
            Self::StsEnv
        } else if has_access_key && has_secret_key {
            Self::StaticKeyEnv
        } else {
            Self::AwsDefaultChain
        }
    }
}

impl fmt::Display for GlueIcebergCredentialSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Credential-safe Glue table-load diagnostics.
///
/// This records catalog/table identity, region, optional profile name, and the
/// credential source class. It intentionally does not store AWS access keys,
/// secret keys, or session token values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlueIcebergTableDiagnostics {
    /// Iceberg catalog name passed to the Glue catalog builder.
    pub catalog_name: String,
    /// Glue database namespace.
    pub database: String,
    /// Glue table name within `database`.
    pub table: String,
    /// Iceberg warehouse URI.
    pub warehouse: String,
    /// Optional Glue catalog id for cross-account access.
    pub catalog_id: Option<String>,
    /// AWS region used for Glue/S3 calls.
    pub region: String,
    /// Selected AWS profile, if any.
    pub profile: Option<String>,
    /// Credential source class; never contains credential values.
    pub credential_source: GlueIcebergCredentialSource,
}

impl GlueIcebergTableDiagnostics {
    /// Returns `database.table` for report rows and error facts.
    pub fn table_ident_display(&self) -> String {
        format!("{}.{}", self.database, self.table)
    }
}

/// Loaded Glue Iceberg table and credential-safe diagnostics.
pub struct LoadedGlueIcebergTable {
    /// Frozen table/snapshot/source-access resolution shared by the CPU
    /// delegate and optional native source inspection.
    pub resolved: ResolvedIcebergTable,
    /// Legacy raw table view of [`Self::resolved`]. New callers should consume
    /// `resolved` so CPU and native planning retain one frozen identity.
    pub table: Table,
    /// Credential-safe load diagnostics.
    pub diagnostics: GlueIcebergTableDiagnostics,
}

/// Result of registering a Glue Iceberg table in a DataFusion session.
pub struct GlueIcebergTableRegistration {
    /// Credential-safe load diagnostics.
    pub diagnostics: GlueIcebergTableDiagnostics,
    /// Previously registered provider for the same DataFusion table reference, if any.
    pub previous: Option<Arc<dyn TableProvider>>,
}

/// Loads an Iceberg table from AWS Glue.
///
/// AWS credential values may be passed to `iceberg-catalog-glue` at runtime
/// through environment variables, but the returned diagnostics never retain
/// those values.
pub async fn load_glue_table(config: GlueIcebergTableConfig) -> crate::error::Result<LoadedGlueIcebergTable> {
    load_glue_table_with_env(config, |name| std::env::var(name).ok()).await
}

/// Loads an AWS Glue Iceberg table and registers it as a native-planned
/// `NexusIcebergTableProvider`.
pub async fn register_glue_table(
    ctx: &SessionContext,
    table_ref: impl Into<TableReference>,
    config: GlueIcebergTableConfig,
) -> DataFusionResult<GlueIcebergTableRegistration> {
    let loaded = load_glue_table(config).await.map_err(DataFusionError::from)?;
    let provider = NexusIcebergTableProvider::try_new_resolved(loaded.resolved)
        .map_err(DataFusionError::from)?
        .with_catalog_type("glue");
    let previous = ctx.register_table(table_ref, Arc::new(provider))?;
    Ok(GlueIcebergTableRegistration { diagnostics: loaded.diagnostics, previous })
}

async fn load_glue_table_with_env(
    config: GlueIcebergTableConfig,
    mut get: impl FnMut(&str) -> Option<String>,
) -> crate::error::Result<LoadedGlueIcebergTable> {
    let environment = GlueLoadEnvironment::from_env(&mut get);
    let diagnostics = config.diagnostics_from_environment(&environment);
    let catalog_config = config.leaf_catalog_config(&environment, &diagnostics);
    let backend = catalog::GlueIcebergCatalogBackend::connect(catalog_config)
        .await
        .map_err(|error| glue_leaf_error("failed to connect AWS Glue Iceberg catalog", &config, &diagnostics, error))?
        .into_backend();
    let resolved = backend
        .load_table(datafusion_nexus_iceberg::IcebergTableIdentifier {
            namespace: vec![config.database.clone()],
            name: config.table.clone(),
        })
        .await
        .map_err(|error| glue_leaf_error("failed to load AWS Glue Iceberg table", &config, &diagnostics, error))?;
    Ok(LoadedGlueIcebergTable { table: resolved.table().clone(), resolved, diagnostics })
}

fn glue_leaf_error(
    message: &'static str,
    config: &GlueIcebergTableConfig,
    diagnostics: &GlueIcebergTableDiagnostics,
    source: datafusion_nexus_iceberg::IcebergError,
) -> crate::error::Error {
    adapter::root_error_from_iceberg(source, LOAD_GLUE_TABLE_OPERATION, message)
        .with_fact("catalog_type", "glue")
        .with_fact("catalog_name", &config.catalog_name)
        .with_fact("database", &config.database)
        .with_fact("table", &config.table)
        .with_fact("region", &config.region)
        .with_fact("credential_source", diagnostics.credential_source.code())
}

pub(crate) fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
    })
}

#[cfg(test)]
mod tests {
    use nexus_query_engine::runtime::S3SourceStorageCredential;

    use super::*;
    use crate::table_format::iceberg::source_access_policy_to_native_source_storage_profile;

    #[test]
    fn glue_leaf_config_and_native_profile_share_static_custom_endpoint_policy() {
        let config = GlueIcebergTableConfig::new("glue", "db", "tbl", "s3://warehouse", "us-east-1")
            .with_s3_endpoint("http://rustfs.example:9000")
            .with_s3_region("eu-west-1")
            .with_s3_path_style_access(true);
        let environment = GlueLoadEnvironment {
            nexus_access_key_id: Some("access".to_string()),
            nexus_secret_access_key: Some("secret".to_string()),
            nexus_session_token: Some("token".to_string()),
            ..Default::default()
        };
        let diagnostics = config.diagnostics_from_environment(&environment);
        let leaf_config = config.leaf_catalog_config(&environment, &diagnostics);
        let policy = leaf_config.source_access_policy();
        let access = policy.s3_access();

        assert_eq!(access.endpoint, Some("http://rustfs.example:9000"));
        assert_eq!(access.region, Some("eu-west-1"));
        assert_eq!(access.path_style_access, Some(true));
        assert!(matches!(access.credential, datafusion_nexus_iceberg::S3CredentialRef::Static { .. }));

        let native = source_access_policy_to_native_source_storage_profile(policy);
        let native_s3 = native.s3_profile();
        assert_eq!(native_s3.endpoint.as_deref(), Some("http://rustfs.example:9000"));
        assert_eq!(native_s3.region.as_deref(), Some("eu-west-1"));
        assert_eq!(native_s3.virtual_hosted_style_request, Some(false));
        assert!(matches!(native_s3.credential, S3SourceStorageCredential::Static { .. }));
    }

    #[test]
    fn custom_endpoint_without_explicit_opt_in_stays_unsigned_for_leaf_and_native() {
        let config = GlueIcebergTableConfig::new("glue", "db", "tbl", "s3://warehouse", "us-east-1")
            .with_s3_endpoint("http://rustfs.example:9000");
        let environment = GlueLoadEnvironment::default();
        let diagnostics = config.diagnostics_from_environment(&environment);
        let leaf_config = config.leaf_catalog_config(&environment, &diagnostics);
        let policy = leaf_config.source_access_policy();

        assert!(matches!(policy.s3_access().credential, datafusion_nexus_iceberg::S3CredentialRef::Unsigned));
        assert!(matches!(
            source_access_policy_to_native_source_storage_profile(policy).s3_profile().credential,
            S3SourceStorageCredential::Unsigned
        ));
    }

    #[test]
    fn default_chain_policy_stays_default_chain_for_leaf_and_native() {
        let config = GlueIcebergTableConfig::new("glue", "db", "tbl", "s3://warehouse", "us-east-1");
        let environment = GlueLoadEnvironment::default();
        let diagnostics = config.diagnostics_from_environment(&environment);
        let leaf_config = config.leaf_catalog_config(&environment, &diagnostics);
        let policy = leaf_config.source_access_policy();

        assert!(matches!(policy.s3_access().credential, datafusion_nexus_iceberg::S3CredentialRef::DefaultChain));
        assert!(matches!(
            source_access_policy_to_native_source_storage_profile(policy).s3_profile().credential,
            S3SourceStorageCredential::DefaultChain
        ));
    }
}
