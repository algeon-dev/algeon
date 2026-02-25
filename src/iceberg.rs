//! User-facing Iceberg integration helpers.
//!
//! This module is available with the `iceberg` Cargo feature. Lower-level,
//! format-neutral scan facts remain under [`crate::table_format`].

#![deny(missing_docs)]

pub mod catalog;

use std::{collections::HashMap, fmt, sync::Arc};

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, TableReference},
    datasource::TableProvider,
    prelude::SessionContext,
};
use iceberg::{Catalog, CatalogBuilder, TableIdent, table::Table};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID as GLUE_AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME,
    AWS_SECRET_ACCESS_KEY as GLUE_AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN as GLUE_AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};

pub use crate::table_format::iceberg::{
    CudfIcebergTableProvider, IcebergTableFormatPlanner, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema,
};

const LOAD_GLUE_TABLE_OPERATION: &str = "datafusion_cudf.iceberg.load_glue_table";

/// Glue catalog name used when constructing an Iceberg catalog from env.
pub const CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "CUDF_ICEBERG_GLUE_CATALOG_NAME";
/// Glue database name used by env-driven Iceberg helpers.
pub const CUDF_ICEBERG_GLUE_DATABASE_ENV: &str = "CUDF_ICEBERG_GLUE_DATABASE";
/// Glue table name used by env-driven Iceberg helpers.
pub const CUDF_ICEBERG_GLUE_TABLE_ENV: &str = "CUDF_ICEBERG_GLUE_TABLE";
/// Iceberg warehouse URI, typically an `s3://...` root.
pub const CUDF_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "CUDF_ICEBERG_GLUE_WAREHOUSE";
/// Optional Glue catalog id for cross-account catalog targeting.
pub const CUDF_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "CUDF_ICEBERG_GLUE_CATALOG_ID";

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
pub const CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_ACCESS_KEY_ID";
/// Explicit S3 secret access key for the REST FileIO path. Secret; consumed at
/// connect time and never retained in diagnostics. See
/// [`CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV`] for why this is not `AWS_*`.
pub const CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_SECRET_ACCESS_KEY";
/// Optional explicit S3 session token for the REST FileIO path. Secret;
/// consumed at connect time and never retained in diagnostics.
pub const CUDF_ICEBERG_S3_SESSION_TOKEN_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_SESSION_TOKEN";

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        }
    }

    /// Replaces the Iceberg catalog name.
    pub fn with_catalog_name(mut self, catalog_name: impl Into<String>) -> Self {
        self.catalog_name = catalog_name.into();
        self
    }

    /// Replaces the Glue database namespace.
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = database.into();
        self
    }

    /// Replaces the Glue table name.
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Replaces the Iceberg warehouse URI.
    pub fn with_warehouse(mut self, warehouse: impl Into<String>) -> Self {
        self.warehouse = warehouse.into();
        self
    }

    /// Sets an explicit Glue catalog id.
    pub fn with_catalog_id(mut self, catalog_id: impl Into<String>) -> Self {
        self.catalog_id = Some(catalog_id.into());
        self
    }

    /// Clears the explicit Glue catalog id.
    pub fn without_catalog_id(mut self) -> Self {
        self.catalog_id = None;
        self
    }

    /// Replaces the AWS region.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Sets an AWS profile selector.
    pub fn with_profile(mut self, profile: impl Into<String>) -> Self {
        self.profile = Some(profile.into());
        self
    }

    /// Clears the AWS profile selector.
    pub fn without_profile(mut self) -> Self {
        self.profile = None;
        self
    }

    /// Returns `database.table` for diagnostics.
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
    /// Loaded `iceberg-rust` table handle.
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
