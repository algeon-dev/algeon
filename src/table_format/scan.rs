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
    runtime::SourceStorageProfile,
    source::{
        CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ResolvedParquetSource,
    },
};

use crate::{capability::CapabilityReason, error::Result};

mod columns;
mod config;
mod diagnostics;
mod files;
mod plan;
mod request;
mod source_resolution;

pub use columns::{
    TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatPartitionTransform,
    TableFormatPartitionValue, TableFormatScalarValue,
};
#[cfg(test)]
use config::table_format_parquet_read_strategy_config_from_env;
pub use config::{
    AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID,
    NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, NEXUS_LAKEHOUSE_CACHE_READ_ENV,
    NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, TableFormatLakehouseCachePolicy, TableFormatLakehouseCacheReadPolicy,
    TableFormatParquetReadStrategyConfig,
};
pub use diagnostics::{
    TableFormatDirectDeviceReadDisposition, TableFormatScanDiagnostics, TableFormatUnsupportedFeature,
    TableFormatUnsupportedFeatureCode,
};
pub use files::{TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent};
pub use plan::TableFormatScanPlan;
#[cfg(test)]
use plan::table_format_source_read_contract;
pub use request::{
    TableFormat, TableFormatLiteral, TableFormatPlanner, TableFormatPlanningDeadline, TableFormatPredicate,
    TableFormatPredicateOp, TableFormatScanRequest,
};
pub use source_resolution::{
    TableFormatParquetSourceMetadata, TableFormatReadCapability, credential_free_endpoint_identity,
    resolve_table_format_parquet_source, resolve_table_format_parquet_source_with_metadata,
};
use source_resolution::{redacted_scan_debug_value, redacted_scan_location_debug_value};
