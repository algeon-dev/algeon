//! User-facing Iceberg integration helpers.
//!
//! This module is available with the `iceberg` Cargo feature. Lower-level,
//! format-neutral scan facts remain under [`crate::table_format`].

pub mod catalog;

use std::{collections::HashMap, fmt, sync::Arc};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, TableReference},
    datasource::TableProvider,
    prelude::SessionContext,
};
use iceberg::{
    Catalog, CatalogBuilder, TableIdent,
    io::{
        S3_ACCESS_KEY_ID, S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_ENDPOINT, S3_PATH_STYLE_ACCESS,
        S3_REGION, S3_SECRET_ACCESS_KEY, S3_SESSION_TOKEN,
    },
    table::Table,
};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID as GLUE_AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME,
    AWS_SECRET_ACCESS_KEY as GLUE_AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN as GLUE_AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_URI, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};

pub use crate::table_format::iceberg::{
    CudfIcebergTableProvider, IcebergTableFormatPlanner, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema,
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

/// Builds the adapter-owned Iceberg provider that plans scans into native
/// table-format facts instead of delegating to an Arrow reader.
pub fn table_provider(table: Table) -> crate::Result<CudfIcebergTableProvider> {
    CudfIcebergTableProvider::try_new(table)
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
pub async fn load_glue_table(config: GlueIcebergTableConfig) -> crate::Result<LoadedGlueIcebergTable> {
    load_glue_table_with_env(config, |name| std::env::var(name).ok()).await
}

/// Loads an AWS Glue Iceberg table and registers it as a native-planned
/// `CudfIcebergTableProvider`.
pub async fn register_glue_table(
    ctx: &SessionContext,
    table_ref: impl Into<TableReference>,
    config: GlueIcebergTableConfig,
) -> DataFusionResult<GlueIcebergTableRegistration> {
    let loaded = load_glue_table(config).await.map_err(DataFusionError::from)?;
    let provider = table_provider(loaded.table).map_err(DataFusionError::from)?.with_catalog_type("glue");
    let previous = ctx.register_table(table_ref, Arc::new(provider))?;
    Ok(GlueIcebergTableRegistration { diagnostics: loaded.diagnostics, previous })
}

async fn load_glue_table_with_env(
    config: GlueIcebergTableConfig,
    mut get: impl FnMut(&str) -> Option<String>,
) -> crate::Result<LoadedGlueIcebergTable> {
    let diagnostics = config.diagnostics_with_env(&mut get);
    let mut props = HashMap::from([
        (GLUE_CATALOG_PROP_WAREHOUSE.to_string(), config.warehouse.clone()),
        (AWS_REGION_NAME.to_string(), config.region.clone()),
    ]);
    if let Some(uri) = config.uri.as_ref().filter(|value| !value.trim().is_empty()) {
        props.insert(GLUE_CATALOG_PROP_URI.to_string(), uri.clone());
    }
    if let Some(catalog_id) = config.catalog_id.as_ref().filter(|value| !value.trim().is_empty()) {
        props.insert(GLUE_CATALOG_PROP_CATALOG_ID.to_string(), catalog_id.clone());
    }
    if let Some(profile) = diagnostics.profile.as_ref().filter(|value| !value.trim().is_empty()) {
        props.insert(AWS_PROFILE_NAME.to_string(), profile.clone());
    }
    if let (Some(access_key_id), Some(secret_access_key)) =
        (non_empty(get(AWS_ACCESS_KEY_ID_ENV)), non_empty(get(AWS_SECRET_ACCESS_KEY_ENV)))
    {
        props.insert(GLUE_AWS_ACCESS_KEY_ID.to_string(), access_key_id);
        props.insert(GLUE_AWS_SECRET_ACCESS_KEY.to_string(), secret_access_key);
    }
    if let Some(session_token) = non_empty(get(AWS_SESSION_TOKEN_ENV).or_else(|| get(AWS_SECURITY_TOKEN_ENV))) {
        props.insert(GLUE_AWS_SESSION_TOKEN.to_string(), session_token);
    }
    if let Some(endpoint) = config.s3_endpoint.as_ref().filter(|value| !value.trim().is_empty()) {
        props.insert(S3_ENDPOINT.to_string(), endpoint.clone());
        props.insert(S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
        props.insert(S3_DISABLE_EC2_METADATA.to_string(), "true".to_string());
    }
    if let Some(region) = config.s3_region.as_ref().filter(|value| !value.trim().is_empty()) {
        props.insert(S3_REGION.to_string(), region.clone());
    }
    if let Some(path_style) = config.s3_path_style_access {
        props.insert(S3_PATH_STYLE_ACCESS.to_string(), path_style.to_string());
    }
    if let (Some(access_key_id), Some(secret_access_key)) =
        (non_empty(get(NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV)), non_empty(get(NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV)))
    {
        props.insert(S3_ACCESS_KEY_ID.to_string(), access_key_id);
        props.insert(S3_SECRET_ACCESS_KEY.to_string(), secret_access_key);
        props.insert(S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string());
        props.insert(S3_DISABLE_EC2_METADATA.to_string(), "true".to_string());
    }
    if let Some(session_token) = non_empty(get(NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV)) {
        props.insert(S3_SESSION_TOKEN.to_string(), session_token);
    }

    let catalog = GlueCatalogBuilder::default().load(config.catalog_name.clone(), props).await.map_err(|error| {
        glue_error("failed to load AWS Glue Iceberg catalog", &config, diagnostics.credential_source, error)
    })?;
    let ident = TableIdent::from_strs([config.database.as_str(), config.table.as_str()]).map_err(|error| {
        glue_error("failed to build AWS Glue Iceberg table identifier", &config, diagnostics.credential_source, error)
    })?;
    let table = catalog.load_table(&ident).await.map_err(|error| {
        glue_error("failed to load AWS Glue Iceberg table", &config, diagnostics.credential_source, error)
    })?;
    Ok(LoadedGlueIcebergTable { table, diagnostics })
}

fn glue_error(
    message: &'static str,
    config: &GlueIcebergTableConfig,
    credential_source: GlueIcebergCredentialSource,
    source: iceberg::Error,
) -> crate::Error {
    crate::Error::new(crate::ErrorCode::CudfDependencyFailure, LOAD_GLUE_TABLE_OPERATION, message)
        .with_fact("catalog_type", "glue")
        .with_fact("catalog_name", &config.catalog_name)
        .with_fact("database", &config.database)
        .with_fact("table", &config.table)
        .with_fact("region", &config.region)
        .with_fact("credential_source", credential_source.code())
        .with_source(source)
}

pub(crate) fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
    })
}
