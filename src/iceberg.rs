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
use iceberg::{Catalog, CatalogBuilder, TableIdent, table::Table};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID as GLUE_AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME,
    AWS_SECRET_ACCESS_KEY as GLUE_AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN as GLUE_AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};

pub use crate::table_format::iceberg::{
    CudfIcebergTableProvider, IcebergTableFormatPlanner, iceberg_file_scan_tasks_to_table_format_scan_plan,
};

const LOAD_GLUE_TABLE_OPERATION: &str = "datafusion_cudf.iceberg.load_glue_table";

pub const CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "CUDF_ICEBERG_GLUE_CATALOG_NAME";
pub const CUDF_ICEBERG_GLUE_DATABASE_ENV: &str = "CUDF_ICEBERG_GLUE_DATABASE";
pub const CUDF_ICEBERG_GLUE_TABLE_ENV: &str = "CUDF_ICEBERG_GLUE_TABLE";
pub const CUDF_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "CUDF_ICEBERG_GLUE_WAREHOUSE";
pub const CUDF_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "CUDF_ICEBERG_GLUE_CATALOG_ID";

pub const AWS_PROFILE_ENV: &str = "AWS_PROFILE";
pub const AWS_REGION_ENV: &str = "AWS_REGION";
pub const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";
pub const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";
pub const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";
pub const AWS_SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";
pub const AWS_SECURITY_TOKEN_ENV: &str = "AWS_SECURITY_TOKEN";

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
    pub catalog_name: String,
    pub database: String,
    pub table: String,
    pub warehouse: String,
    pub catalog_id: Option<String>,
    pub region: String,
    pub profile: Option<String>,
}

impl GlueIcebergTableConfig {
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

    pub fn table_ident_display(&self) -> String {
        format!("{}.{}", self.database, self.table)
    }

    pub fn diagnostics_from_env(&self) -> GlueIcebergTableDiagnostics {
        self.diagnostics_with_env(|name| std::env::var(name).ok())
    }

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
    AwsProfileEnv,
    StsEnv,
    StaticKeyEnv,
    AwsDefaultChain,
}

impl GlueIcebergCredentialSource {
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
    pub catalog_name: String,
    pub database: String,
    pub table: String,
    pub warehouse: String,
    pub catalog_id: Option<String>,
    pub region: String,
    pub profile: Option<String>,
    pub credential_source: GlueIcebergCredentialSource,
}

impl GlueIcebergTableDiagnostics {
    pub fn table_ident_display(&self) -> String {
        format!("{}.{}", self.database, self.table)
    }
}

/// Loaded Glue Iceberg table and credential-safe diagnostics.
pub struct LoadedGlueIcebergTable {
    pub table: Table,
    pub diagnostics: GlueIcebergTableDiagnostics,
}

/// Result of registering a Glue Iceberg table in a DataFusion session.
pub struct GlueIcebergTableRegistration {
    pub diagnostics: GlueIcebergTableDiagnostics,
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
