//! Adapter-owned table-format scan facts.
//!
//! This layer resolves table-format metadata into frontend facts that native
//! lowering can inspect later. It does not execute source reads and it stays out
//! of `nexus-query-engine`, which continues to receive only native source facts.

#[cfg(feature = "iceberg")]
pub mod iceberg;
pub mod scan;
pub mod scan_exec;

#[cfg(feature = "iceberg")]
pub use iceberg::{
    IcebergTableFormatPlanner, NexusIcebergTableProvider, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema,
};
pub use scan::{
    AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, CredentialMode, LakehouseCacheFillMode, LakehouseCacheReadMode,
    LakehouseCacheStatus, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    NEXUS_LAKEHOUSE_CACHE_READ_ENV, NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, ParquetReadStrategy, ResolvedParquetInput,
    ResolvedParquetSource, S3SourceStorageCredential, S3SourceStorageProfile, SourceStorageProfile, TableFormat,
    TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
    TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent,
    TableFormatDirectDeviceReadDisposition, TableFormatLakehouseCachePolicy, TableFormatLakehouseCacheReadPolicy,
    TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
    TableFormatPartitionValue, TableFormatPlanner, TableFormatPlanningDeadline, TableFormatReadCapability,
    TableFormatScalarValue, TableFormatScanDiagnostics, TableFormatScanPlan, TableFormatScanRequest,
    TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode, resolve_table_format_parquet_source,
    resolve_table_format_parquet_source_with_metadata,
};
pub use scan_exec::{NexusIcebergNativeScanOutcome, NexusIcebergScanExec};
