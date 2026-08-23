use std::{collections::BTreeMap, fmt};

use super::CapabilityReason;

/// Unsupported scan fact category with a stable fallback reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatUnsupportedFeatureCode {
    /// Generic unsupported table-format scan.
    TableFormatUnsupported,
    /// Table-format metadata planning failed.
    TableFormatPlanningFailed,
    /// Iceberg data file is not Parquet.
    IcebergNonParquetDataFile,
    /// Iceberg delete files are present.
    IcebergDeleteFilesUnsupported,
    /// Iceberg field-id mapping is required but unsupported.
    IcebergFieldIdMappingUnsupported,
    /// Iceberg name mapping is required but unsupported.
    IcebergNameMappingUnsupported,
    /// Iceberg nested projection is required but unsupported.
    IcebergNestedProjectionUnsupported,
    /// Iceberg partition transform cannot be materialized natively.
    IcebergPartitionTransformUnsupported,
    /// Iceberg residual predicate is not represented by native scan facts.
    IcebergResidualPredicateUnsupported,
    /// Iceberg row-selection/page-index semantics are unsupported.
    IcebergRowSelectionUnsupported,
    /// Iceberg file range semantics cannot be represented by a whole-file native read.
    IcebergFileRangeUnsupported,
    /// Iceberg object-store read cannot be represented by native source facts.
    IcebergObjectStoreReadUnsupported,
    /// Remote Iceberg S3 FileIO cannot be proven to match the frozen
    /// CPU/native source-access policy.
    IcebergFileIoSourceAccessUnverified,
    /// Iceberg file encryption metadata cannot be materialized natively.
    IcebergEncryptedFileUnsupported,
    /// An Iceberg metadata scalar cannot be materialized natively.
    IcebergScalarValueUnsupported,
}

impl TableFormatUnsupportedFeatureCode {
    /// Stable fallback reason associated with this unsupported feature.
    pub const fn fallback_reason(self) -> CapabilityReason {
        match self {
            Self::TableFormatUnsupported => CapabilityReason::TableFormatUnsupported,
            Self::TableFormatPlanningFailed => CapabilityReason::TableFormatPlanningFailed,
            Self::IcebergNonParquetDataFile => CapabilityReason::IcebergNonParquetDataFile,
            Self::IcebergDeleteFilesUnsupported => CapabilityReason::IcebergDeleteFilesUnsupported,
            Self::IcebergFieldIdMappingUnsupported => CapabilityReason::IcebergFieldIdMappingUnsupported,
            Self::IcebergNameMappingUnsupported => CapabilityReason::IcebergNameMappingUnsupported,
            Self::IcebergNestedProjectionUnsupported => CapabilityReason::IcebergNestedProjectionUnsupported,
            Self::IcebergPartitionTransformUnsupported => CapabilityReason::IcebergPartitionTransformUnsupported,
            Self::IcebergResidualPredicateUnsupported => CapabilityReason::IcebergResidualPredicateUnsupported,
            Self::IcebergRowSelectionUnsupported => CapabilityReason::IcebergRowSelectionUnsupported,
            Self::IcebergFileRangeUnsupported => CapabilityReason::NativeSourceByteRange,
            Self::IcebergObjectStoreReadUnsupported => CapabilityReason::IcebergObjectStoreReadUnsupported,
            Self::IcebergFileIoSourceAccessUnverified => CapabilityReason::IcebergFileIoSourceAccessUnverified,
            Self::IcebergEncryptedFileUnsupported => CapabilityReason::IcebergEncryptedFileUnsupported,
            Self::IcebergScalarValueUnsupported => CapabilityReason::IcebergScalarValueUnsupported,
        }
    }

    /// Stable fallback reason code.
    pub const fn code(self) -> &'static str {
        self.fallback_reason().code()
    }
}

/// One unsupported table-format fact retained for reporting and rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatUnsupportedFeature {
    /// Unsupported feature category.
    pub code: TableFormatUnsupportedFeatureCode,
    /// Human-readable unsupported feature detail.
    pub message: String,
    /// Optional file id when the unsupported feature is file-local.
    pub file_id: Option<String>,
}

impl TableFormatUnsupportedFeature {
    /// Builds a scan-level unsupported feature.
    pub fn new(code: TableFormatUnsupportedFeatureCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into(), file_id: None }
    }

    /// Builds a file-local unsupported feature.
    pub fn for_file(
        code: TableFormatUnsupportedFeatureCode,
        file_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self { code, message: message.into(), file_id: Some(file_id.into()) }
    }

    /// Stable fallback reason associated with this feature.
    pub fn fallback_reason(&self) -> CapabilityReason {
        self.code.fallback_reason()
    }

    /// Stable fallback reason code.
    pub fn code(&self) -> &'static str {
        self.code.code()
    }
}

/// Native direct-device-read disposition for a resolved table-format scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableFormatDirectDeviceReadDisposition {
    /// Scan facts are eligible for direct native device reads.
    Eligible,
    /// Scan facts must be rejected before direct native device reads.
    Rejected,
}

impl TableFormatDirectDeviceReadDisposition {
    /// Stable disposition code for diagnostics.
    pub const fn code(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Rejected => "rejected",
        }
    }
}

/// Credential-free scan diagnostics intended for tests and report rows.
///
/// `planned_bytes = None` means the table metadata did not expose a reliable
/// byte total, not that the plan is empty. String fields are stable report
/// vocabulary and should avoid paths, tokens, or other sensitive values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableFormatScanDiagnostics {
    /// Stable table-format code.
    pub table_format: String,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Number of planned data files.
    pub planned_data_file_count: usize,
    /// Number of planned delete files.
    pub planned_delete_file_count: usize,
    /// Total planned data-file bytes, or `None` if any file size is unknown.
    pub planned_bytes: Option<u64>,
    /// Number of files admitted to native source lowering.
    pub admitted_native_source_file_count: usize,
    /// Stable unsupported-feature codes retained by planning.
    pub rejected_feature_list: Vec<String>,
    /// Comma-separated stable read-capability codes.
    pub native_source_read_capability: String,
    /// Comma-separated high-level source read strategies.
    pub source_read_strategy: String,
    /// Comma-separated credential modes.
    pub credential_mode: String,
    /// Comma-separated local byte-cache statuses.
    pub lakehouse_cache_status: String,
    /// Comma-separated foreground lakehouse cache-read modes.
    pub lakehouse_cache_read_mode: String,
    /// Comma-separated complete-object lakehouse cache fill modes.
    pub lakehouse_cache_fill_mode: String,
    /// Direct device-read eligibility.
    pub direct_device_read_disposition: TableFormatDirectDeviceReadDisposition,
}

impl TableFormatScanDiagnostics {
    /// Credential-free facts suitable for structured errors and tests.
    pub fn facts(&self) -> BTreeMap<&'static str, String> {
        BTreeMap::from([
            ("table_format", self.table_format.clone()),
            ("snapshot_id", self.snapshot_id.clone().unwrap_or_default()),
            ("catalog_type", self.catalog_type.clone().unwrap_or_default()),
            ("planned_data_file_count", self.planned_data_file_count.to_string()),
            ("planned_delete_file_count", self.planned_delete_file_count.to_string()),
            ("planned_bytes", self.planned_bytes.map(|bytes| bytes.to_string()).unwrap_or_default()),
            ("admitted_native_source_file_count", self.admitted_native_source_file_count.to_string()),
            ("rejected_feature_list", self.rejected_feature_list.join(",")),
            ("native_source_read_capability", self.native_source_read_capability.clone()),
            ("source_read_strategy", self.source_read_strategy.clone()),
            ("credential_mode", self.credential_mode.clone()),
            ("lakehouse_cache_status", self.lakehouse_cache_status.clone()),
            ("lakehouse_cache_read_mode", self.lakehouse_cache_read_mode.clone()),
            ("lakehouse_cache_fill_mode", self.lakehouse_cache_fill_mode.clone()),
            ("direct_device_read_disposition", self.direct_device_read_disposition.code().to_string()),
        ])
    }
}

impl fmt::Display for TableFormatScanDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, (key, value)) in self.facts().iter().enumerate() {
            if index > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{key}={value}")?;
        }
        Ok(())
    }
}
