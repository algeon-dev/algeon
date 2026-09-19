//! Adapter-owned table-format scan facts.
//!
//! This layer resolves table-format metadata into frontend facts that native
//! lowering can inspect later. It does not execute source reads and it stays out
//! of `query-engine`, which continues to receive only native source facts.
//!
//! Iceberg is not supported: the `iceberg` feature is absent from every
//! manifest, so the `cfg(feature = "iceberg")` items below cannot be built.
//! The DataFusion integration left apache/iceberg-rust for
//! apache/datafusion-iceberg (iceberg-rust#3029, merged 2026-09-17). That
//! project already targets DataFusion 55, but it is not published to crates.io
//! and still pins `iceberg` by git rev, and a git dependency cannot appear in
//! a published manifest. These items are the restoration point for when it
//! ships a release. See `docs/issue/2026-08-20-crates-io-release-readiness.md`.

#[cfg(feature = "iceberg")]
pub mod iceberg;
#[cfg(all(test, feature = "iceberg"))]
mod provider_tests;
pub mod scan;
pub mod scan_exec;

#[cfg(feature = "iceberg")]
pub use iceberg::{
    AlgeonIcebergTableProvider, IcebergTableFormatPlanner, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema,
};
pub use scan::{
    ALGEON_LAKEHOUSE_CACHE_MAX_BYTES_ENV, ALGEON_LAKEHOUSE_CACHE_POLICY_ENV, ALGEON_LAKEHOUSE_CACHE_READ_ENV,
    ALGEON_LAKEHOUSE_CACHE_ROOT_ENV, AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, CredentialMode,
    LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus, ParquetReadStrategy, ResolvedParquetInput,
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
pub use scan_exec::{AlgeonIcebergNativeScanOutcome, AlgeonIcebergScanExec};
