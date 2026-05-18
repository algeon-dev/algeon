//! Adapter-owned table-format scan facts.
//!
//! This layer resolves table-format metadata into frontend facts that native
//! lowering can inspect later. It does not execute source reads and it stays out
//! of `cudf-query-engine`, which continues to receive only native source facts.

#[cfg(feature = "iceberg")]
pub mod iceberg;
pub mod scan;
pub mod scan_exec;

#[cfg(feature = "iceberg")]
pub use iceberg::{
    CudfIcebergTableProvider, IcebergTableFormatPlanner, iceberg_file_scan_tasks_to_table_format_scan_plan,
};
pub use scan::{
    TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
    TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent,
    TableFormatDirectDeviceReadDisposition, TableFormatPartitionTransform, TableFormatPartitionValue,
    TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue, TableFormatScanDiagnostics,
    TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
};
pub use scan_exec::CudfTableFormatScanExec;
