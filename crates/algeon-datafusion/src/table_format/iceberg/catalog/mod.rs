//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`CatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`] or
//! [`build_scoped_catalog_provider_from_backend`]; individual tables are loaded
//! lazily on first access.
//!
//! AWS Glue and Iceberg REST catalog connection, FileIO construction, and the
//! immutable [`SourceAccessPolicy`](iceberg_catalog::SourceAccessPolicy) belong
//! to the DataFusion-free Iceberg leaf, which this module re-exports. The root
//! catalog module only adapts a resolved leaf backend to DataFusion's catalog
//! provider traits.

use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::DataFusionError,
    datasource::TableProvider,
};
use iceberg::{NamespaceIdent, TableIdent};
use tokio::sync::RwLock;

use crate::{
    error::{Error, ErrorCode, Result},
    table_format::scan::TableFormatParquetReadStrategyConfig,
};

mod error;
mod provider;

use error::CATALOG_BACKEND_OP;
pub(crate) use error::{
    apply_iceberg_catalog_error_facts, apply_opendal_scan_error_facts, classify_iceberg_scan_error,
    classify_opendal_scan_error,
};
pub use iceberg_catalog::{GlueCatalogBackend, GlueCatalogConfig, RestCatalogBackend, RestCatalogConfig};
pub use provider::{
    AlgeonIcebergCatalogProvider, AlgeonIcebergCatalogSource, AlgeonIcebergSchemaProvider, CatalogBackend,
    IcebergProviderOptions, build_catalog_provider_from_backend, build_catalog_provider_from_backend_with_options,
    build_scoped_catalog_provider_from_backend, build_scoped_catalog_provider_from_backend_with_options,
};
