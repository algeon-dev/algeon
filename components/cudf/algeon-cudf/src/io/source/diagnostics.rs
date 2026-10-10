use std::fmt;

use super::{LocalParquetSource, LocalParquetSourceKind, SourceColumnRef, SourceColumnRole};

/// Derived diagnostics for a local Parquet source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    pub(super) direct_device_read: bool,
    pub(super) file_count: usize,
    pub(super) ranged_file_count: usize,
    pub(super) known_bytes: Option<u64>,
    pub(super) selected_row_groups: Option<usize>,
    pub(super) predicate_disposition: PredicateDisposition,
    pub(super) byte_range_disposition: ByteRangeDisposition,
    pub(super) partition_column_disposition: PartitionColumnDisposition,
    pub(super) zero_column_count_disposition: ZeroColumnCountDisposition,
    pub(super) schema_mapping_disposition: SchemaMappingDisposition,
    pub(super) read_column_count: usize,
    pub(super) output_column_count: usize,
    pub(super) pure_filter_column_count: usize,
    pub(super) rejected_features: Vec<RejectedSourceFeature>,
}

impl SourceDiagnostics {
    pub(super) fn from_source(source: &LocalParquetSource) -> Self {
        let known_bytes = source.files.iter().try_fold(0u64, |acc, file| {
            let file_size = file.file_size_bytes?;
            acc.checked_add(file_size)
        });
        let selected_row_groups = if source.files.iter().any(|file| file.row_groups.is_some()) {
            Some(source.files.iter().filter_map(|file| file.row_groups.as_ref()).map(Vec::len).sum())
        } else {
            None
        };
        let pure_filter_column_count =
            source.required_columns.iter().filter(|column| column.role == SourceColumnRole::PureFilter).count();
        let zero_column_count_disposition = match source.kind {
            LocalParquetSourceKind::Data => ZeroColumnCountDisposition::NotRequired,
            LocalParquetSourceKind::MetadataRowCount => ZeroColumnCountDisposition::Represented,
        };

        let schema_mapping_disposition =
            if source.output_columns.iter().any(|column| column.source == SourceColumnRef::MissingNull)
                || source.required_columns.iter().any(|column| column.source == SourceColumnRef::MissingNull)
            {
                SchemaMappingDisposition::MissingNullableColumns
            } else {
                SchemaMappingDisposition::DirectByName
            };

        Self {
            direct_device_read: true,
            file_count: source.files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: if source.reader_predicate.is_some() {
                PredicateDisposition::ReaderPushdown
            } else {
                PredicateDisposition::Absent
            },
            byte_range_disposition: ByteRangeDisposition::FullFiles,
            partition_column_disposition: if source.partition_columns.is_empty() {
                PartitionColumnDisposition::Absent
            } else {
                PartitionColumnDisposition::Represented
            },
            zero_column_count_disposition,
            schema_mapping_disposition,
            read_column_count: source.read_columns.len(),
            output_column_count: source.output_columns.len(),
            pure_filter_column_count,
            rejected_features: Vec::new(),
        }
    }

    /// Whether the current local Parquet execution path can honor all
    /// represented source facts.
    pub fn direct_device_read(&self) -> bool {
        self.direct_device_read
    }

    /// Number of local files.
    pub fn file_count(&self) -> usize {
        self.file_count
    }

    /// Number of ranged files. The local Parquet path currently accepts only
    /// full local files.
    pub fn ranged_file_count(&self) -> usize {
        self.ranged_file_count
    }

    /// Known total bytes when all file sizes are known and addition fits.
    pub fn known_bytes(&self) -> Option<u64> {
        self.known_bytes
    }

    /// Total selected row groups if any file has explicit row-group selection.
    pub fn selected_row_groups(&self) -> Option<usize> {
        self.selected_row_groups
    }

    /// Predicate disposition.
    pub fn predicate_disposition(&self) -> PredicateDisposition {
        self.predicate_disposition
    }

    /// Byte-range disposition.
    pub fn byte_range_disposition(&self) -> ByteRangeDisposition {
        self.byte_range_disposition
    }

    /// Partition-column disposition.
    pub fn partition_column_disposition(&self) -> PartitionColumnDisposition {
        self.partition_column_disposition
    }

    /// Zero-column count disposition.
    pub fn zero_column_count_disposition(&self) -> ZeroColumnCountDisposition {
        self.zero_column_count_disposition
    }

    /// Schema mapping disposition.
    pub fn schema_mapping_disposition(&self) -> SchemaMappingDisposition {
        self.schema_mapping_disposition
    }

    /// Number of decoded read columns.
    pub fn read_column_count(&self) -> usize {
        self.read_column_count
    }

    /// Number of final output columns.
    pub fn output_column_count(&self) -> usize {
        self.output_column_count
    }

    /// Number of pure-filter required columns.
    pub fn pure_filter_column_count(&self) -> usize {
        self.pure_filter_column_count
    }

    /// Neutral rejected source features. Valid constructed sources have no
    /// rejected features.
    pub fn rejected_features(&self) -> &[RejectedSourceFeature] {
        &self.rejected_features
    }
}

/// Predicate support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateDisposition {
    /// No source predicate was requested.
    Absent,
    /// Predicate was used for pruning only.
    PruningOnly,
    /// Predicate was represented as a reader-level Parquet predicate.
    ReaderPushdown,
    /// Predicate remains a native post-scan filter.
    NativePostScanOnly,
    /// Predicate execution could not be represented.
    Rejected,
}

/// Byte-range support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRangeDisposition {
    /// Full local files only.
    FullFiles,
    /// Caller already coalesced ranges into full local files.
    CoalescedFullFiles,
    /// Byte ranges could not be represented.
    Rejected,
}

/// Partition-column support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionColumnDisposition {
    /// No partition columns.
    Absent,
    /// Partition columns are represented explicitly.
    Represented,
    /// Partition columns could not be represented.
    Rejected,
}

/// Metadata-only zero-column count support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroColumnCountDisposition {
    /// Count mode is not requested.
    NotRequired,
    /// Count mode is represented by the synthetic count column.
    Represented,
    /// Count mode could not be represented.
    Rejected,
}

/// Schema mapping support summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaMappingDisposition {
    /// Direct exact by-name mapping.
    DirectByName,
    /// One or more nullable logical columns are absent from the physical file
    /// schema and represented as typed null columns.
    MissingNullableColumns,
    /// Schema mapping could not be represented.
    Rejected,
}

/// Neutral source feature names for rejected planner features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectedSourceFeature {
    /// The input was not a local filesystem source.
    NonLocalSource,
    /// Partition columns or values could not be represented.
    UnsupportedPartitionColumns,
    /// Predicate execution could not be represented.
    UnsupportedPredicateExecution,
    /// Scan limit could not be represented.
    UnsupportedScanLimit,
    /// Ordering requirement could not be represented.
    UnsupportedOrderingRequirement,
    /// The source contained no files.
    EmptySource,
    /// Partition values could not be represented.
    UnsupportedPartitionValues,
    /// Byte ranges could not be represented.
    UnsupportedByteRange,
    /// Row-level selection could not be represented.
    UnsupportedRowSelection,
    /// Metadata-only count could not be represented.
    UnsupportedZeroColumnCount,
    /// Schema mapping could not be represented.
    UnsupportedSchemaMapping,
}

impl RejectedSourceFeature {
    /// Stable diagnostic string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLocalSource => "non_local_source",
            Self::UnsupportedPartitionColumns => "unsupported_partition_columns",
            Self::UnsupportedPredicateExecution => "unsupported_predicate_execution",
            Self::UnsupportedScanLimit => "unsupported_scan_limit",
            Self::UnsupportedOrderingRequirement => "unsupported_ordering_requirement",
            Self::EmptySource => "empty_source",
            Self::UnsupportedPartitionValues => "unsupported_partition_values",
            Self::UnsupportedByteRange => "unsupported_byte_range",
            Self::UnsupportedRowSelection => "unsupported_row_selection",
            Self::UnsupportedZeroColumnCount => "unsupported_zero_column_count",
            Self::UnsupportedSchemaMapping => "unsupported_schema_mapping",
        }
    }
}

macro_rules! stable_display {
    ($type:ty, {$($variant:pat => $name:expr),+ $(,)?}) => {
        impl $type {
            /// Stable diagnostic string.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($variant => $name,)+
                }
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

stable_display!(PredicateDisposition, {
    PredicateDisposition::Absent => "absent",
    PredicateDisposition::PruningOnly => "pruning_only",
    PredicateDisposition::ReaderPushdown => "reader_pushdown",
    PredicateDisposition::NativePostScanOnly => "native_post_scan_only",
    PredicateDisposition::Rejected => "rejected",
});
stable_display!(ByteRangeDisposition, {
    ByteRangeDisposition::FullFiles => "full_files",
    ByteRangeDisposition::CoalescedFullFiles => "coalesced_full_files",
    ByteRangeDisposition::Rejected => "rejected",
});
stable_display!(PartitionColumnDisposition, {
    PartitionColumnDisposition::Absent => "absent",
    PartitionColumnDisposition::Represented => "represented",
    PartitionColumnDisposition::Rejected => "rejected",
});
stable_display!(ZeroColumnCountDisposition, {
    ZeroColumnCountDisposition::NotRequired => "not_required",
    ZeroColumnCountDisposition::Represented => "represented",
    ZeroColumnCountDisposition::Rejected => "rejected",
});
stable_display!(SchemaMappingDisposition, {
    SchemaMappingDisposition::DirectByName => "direct_by_name",
    SchemaMappingDisposition::MissingNullableColumns => "missing_nullable_columns",
    SchemaMappingDisposition::Rejected => "rejected",
});

impl fmt::Display for RejectedSourceFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
