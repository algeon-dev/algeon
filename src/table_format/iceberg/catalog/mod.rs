//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`IcebergCatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`] or
//! [`build_scoped_catalog_provider_from_backend`]; individual tables are loaded
//! lazily on first access.

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
mod glue;
mod provider;
mod rest;

use error::CATALOG_BACKEND_OP;
pub(crate) use error::{
    apply_iceberg_catalog_error_facts, apply_opendal_scan_error_facts, classify_iceberg_scan_error,
    classify_opendal_scan_error,
};
pub use glue::{GlueIcebergCatalogBackend, GlueIcebergCatalogConfig};
pub use provider::{
    IcebergCatalogBackend, IcebergProviderOptions, NexusIcebergCatalogProvider, NexusIcebergCatalogSource,
    NexusIcebergSchemaProvider, build_catalog_provider_from_backend, build_catalog_provider_from_backend_with_options,
    build_scoped_catalog_provider_from_backend, build_scoped_catalog_provider_from_backend_with_options,
};
pub use rest::{RestIcebergCatalogBackend, RestIcebergCatalogConfig};
