use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use arrow::datatypes::{DataType, Schema, SchemaRef};
use futures::future::BoxFuture;
use nexus_query_engine::{
    CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, RemoteParquetReadMode,
    ResolvedParquetSource, SourceStorageProfile,
};

use crate::{Result, policy::FallbackReason};

mod columns;
mod config;
mod diagnostics;
mod files;
mod plan;
mod request;
mod source_resolution;

pub use columns::*;
#[cfg(test)]
use config::table_format_parquet_read_strategy_config_from_env;
pub use config::*;
pub use diagnostics::*;
pub use files::*;
#[cfg(test)]
use plan::table_format_source_read_contract;
pub use plan::*;
pub use request::*;
pub use source_resolution::{
    TableFormatParquetSourceMetadata, TableFormatReadCapability, credential_free_endpoint_identity,
    resolve_table_format_parquet_source, resolve_table_format_parquet_source_with_metadata,
};
use source_resolution::{redacted_scan_debug_value, redacted_scan_location_debug_value};
