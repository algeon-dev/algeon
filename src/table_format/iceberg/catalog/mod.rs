//! Dynamic Iceberg catalog provider for DataFusion.
//!
//! This module provides a backend-neutral [`IcebergCatalogBackend`] trait and
//! the DataFusion [`CatalogProvider`] / [`SchemaProvider`] implementations that
//! sit on top of it.  Namespace and table lists are populated asynchronously at
//! startup via [`build_catalog_provider_from_backend`] or
//! [`build_scoped_catalog_provider_from_backend`]; individual tables are loaded
//! lazily on first access.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error as StdError,
    fmt,
    sync::{Arc, RwLock as StdRwLock},
    time::Duration,
};

use async_trait::async_trait;
use datafusion::{
    catalog::{CatalogProvider, SchemaProvider},
    common::DataFusionError,
    datasource::TableProvider,
};
use iceberg::{
    Catalog, CatalogBuilder as _, NamespaceIdent, TableIdent,
    io::{S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_ENDPOINT, S3_PATH_STYLE_ACCESS, S3_REGION},
};
use iceberg_catalog_glue::{
    AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
    GLUE_CATALOG_PROP_CATALOG_ID, GLUE_CATALOG_PROP_URI, GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder,
};
use iceberg_catalog_rest::{REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder};
use iceberg_storage_opendal::OpenDalStorageFactory;
use nexus_query_engine::runtime::{S3SourceStorageProfile, SourceStorageProfile};
use tokio::sync::RwLock;

use crate::{
    error::{Error, ErrorCode, Result},
    table_format::scan::{
        DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, TableFormatParquetReadStrategyConfig,
        credential_free_endpoint_identity,
    },
};

mod error;
mod glue;
mod provider;
mod rest;

use error::{CATALOG_BACKEND_OP, catalog_backend_error, glue_catalog_backend_error};
pub(super) use error::{apply_iceberg_catalog_error_facts, classify_iceberg_catalog_error};
use glue::GlueCatalogBackendErrorContext;
pub use glue::{GlueIcebergCatalogBackend, GlueIcebergCatalogConfig};
#[cfg(feature = "server")]
pub(crate) use provider::table_ident_from_flat_name;
pub use provider::{
    IcebergCatalogBackend, IcebergProviderKind, IcebergProviderOptions, NexusIcebergCatalogProvider,
    NexusIcebergSchemaProvider, build_catalog_provider_from_backend, build_catalog_provider_from_backend_with_options,
    build_scoped_catalog_provider_from_backend, build_scoped_catalog_provider_from_backend_with_options,
};
use provider::{
    S3FileIoConfig, catalog_backend_list_namespaces, catalog_backend_list_tables, catalog_backend_load_table,
};
pub use rest::{RestIcebergCatalogBackend, RestIcebergCatalogConfig};
