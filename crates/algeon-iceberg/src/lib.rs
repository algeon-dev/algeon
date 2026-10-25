//! Iceberg tables as native Algeon scans.
//!
//! [`register_iceberg_catalogs`] registers one DataFusion catalog per
//! configured Iceberg REST or Glue catalog. Each namespace is a schema and
//! each table an [`IcebergTableProvider`], which plans the current snapshot
//! with Iceberg's manifest pruning and scans its data files as Parquet. The
//! data files are read with the session's `[s3]` storage, never with
//! credentials a catalog returns.
//!
//! Snapshots the Parquet scan would read incorrectly (delete files, columns
//! bound by field id to other names, partition values kept only in
//! metadata) fail with `unsupported_table_feature`; there is no other read
//! path.

mod binding;
mod catalog;
mod error;
mod predicate;
mod schema;
mod storage;
mod table;

use std::sync::Arc;

use algeon_datafusion::{
    error::{Error, ErrorCode, Result},
    server::ServerConfig,
};
use datafusion::prelude::SessionContext;

pub use crate::{
    catalog::{IcebergCatalogProvider, IcebergSchemaProvider, connect_catalog},
    table::IcebergTableProvider,
};

/// Connects every `[[iceberg.catalogs]]` entry of `config` and registers it
/// on `ctx` under its name. Data files are read with the `[s3]` storage of
/// `config`.
///
/// Catalogs connect in file order on the caller's Tokio runtime; the first
/// failure is returned and leaves the catalogs registered before it in
/// place. A name the session already holds, such as its default catalog,
/// fails with `invalid_configuration` instead of replacing that catalog.
pub async fn register_iceberg_catalogs(ctx: &SessionContext, config: &ServerConfig) -> Result<()> {
    for options in &config.iceberg_catalogs {
        if ctx.catalog(&options.name).is_some() {
            return Err(Error::new(
                ErrorCode::InvalidConfiguration,
                "algeon_iceberg.register_catalogs",
                format!("iceberg.catalogs name `{}` is already a catalog of the server session", options.name),
            )
            .with_fact("config_field", "iceberg.catalogs.name"));
        }
        let provider = IcebergCatalogProvider::connect(options, config.s3_storage()).await?;
        ctx.register_catalog(options.name.clone(), Arc::new(provider));
    }
    Ok(())
}
