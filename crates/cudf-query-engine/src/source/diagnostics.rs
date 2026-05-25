#![deny(missing_docs)]

use super::{LocalParquetFile, PartitionColumn, SourceColumnRef};

#[derive(Debug, Clone, Copy)]
pub(super) struct LocalParquetDiagnosticColumns<'a> {
    pub(super) read_columns: &'a [String],
    pub(super) read_column_file_indices: &'a [usize],
    pub(super) projected_output_columns: &'a [String],
    pub(super) projected_output_column_source_refs: &'a [SourceColumnRef],
    pub(super) pure_filter_columns: &'a [String],
    pub(super) pure_filter_column_source_refs: &'a [SourceColumnRef],
    pub(super) partition_columns: &'a [PartitionColumn],
    pub(super) missing_columns: &'a [super::MissingColumn],
    pub(super) scan_limit: Option<usize>,
}

/// Frontend-visible source capability and lowering diagnostics.
///
/// Diagnostics explain why a source can or cannot use the native direct device
/// path. They are intentionally structural so adapters and tests can assert on
/// stable fields instead of parsing displayed plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDiagnostics {
    /// Source file/data format.
    pub format: SourceFormat,
    /// True when no rejected features prevent native direct device read.
    pub direct_device_read: bool,
    /// Number of files represented after any frontend coalescing.
    pub file_count: usize,
    /// Number of source file entries that carried byte ranges.
    pub ranged_file_count: usize,
    /// Total known bytes across represented files, or `None` when incomplete.
    pub known_bytes: Option<u64>,
    /// Total selected row groups, or `None` when row-group selection is unknown.
    pub selected_row_groups: Option<usize>,
    /// How pushed-down source predicates were represented.
    pub predicate_disposition: PredicateDisposition,
    /// How byte-range file entries were represented.
    pub byte_range_disposition: ByteRangeDisposition,
    /// How partition columns were represented.
    pub partition_column_disposition: PartitionColumnDisposition,
    /// Whether metadata-only zero-column count was required and represented.
    pub zero_column_count_disposition: ZeroColumnCountDisposition,
    /// Whether a source-level scan limit is absent or represented.
    pub scan_limit_disposition: ScanLimitDisposition,
    /// How logical source columns map to physical or synthetic source columns.
    pub schema_mapping_disposition: SchemaMappingDisposition,
    /// Number of physical Parquet columns read by the native source.
    pub read_column_count: usize,
    /// Number of logical source columns expected to remain after
    /// source-adjacent native filtering/projection.
    pub projected_output_column_count: usize,
    /// Number of emitted source columns used only for source filtering.
    pub pure_filter_column_count: usize,
    /// Stable feature codes explaining why direct device read was rejected.
    pub rejected_features: Vec<RejectedSourceFeature>,
}

impl SourceDiagnostics {
    pub(super) fn for_local_parquet_files(
        files: &[LocalParquetFile],
        columns: LocalParquetDiagnosticColumns<'_>,
    ) -> Self {
        let known_bytes = files
            .iter()
            .map(|file| file.file_size_bytes)
            .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
        let selected_row_groups = files
            .iter()
            .any(|file| file.row_groups.is_some())
            .then(|| files.iter().map(LocalParquetFile::selected_row_group_count).sum());
        let zero_column_count_disposition = if columns.read_columns.is_empty() && columns.missing_columns.is_empty() {
            ZeroColumnCountDisposition::Rejected
        } else {
            ZeroColumnCountDisposition::NotRequired
        };
        let mut rejected_features = Vec::new();
        if zero_column_count_disposition == ZeroColumnCountDisposition::Rejected {
            rejected_features.push(RejectedSourceFeature::ZeroColumnCount);
        }
        let schema_refs_are_valid = columns.read_columns.len() == columns.read_column_file_indices.len()
            && columns.projected_output_columns.len() == columns.projected_output_column_source_refs.len()
            && columns.pure_filter_columns.len() == columns.pure_filter_column_source_refs.len();
        let schema_mapping_disposition = if schema_refs_are_valid {
            if columns
                .projected_output_column_source_refs
                .iter()
                .chain(columns.pure_filter_column_source_refs.iter())
                .any(|source_ref| matches!(source_ref, SourceColumnRef::MissingNull(_)))
            {
                SchemaMappingDisposition::MissingNullableColumns
            } else {
                SchemaMappingDisposition::DirectByName
            }
        } else {
            rejected_features.push(RejectedSourceFeature::SchemaMapping);
            SchemaMappingDisposition::Rejected
        };
        let partition_column_disposition = if columns.partition_columns.is_empty() {
            PartitionColumnDisposition::Absent
        } else {
            PartitionColumnDisposition::Represented
        };
        let scan_limit_disposition =
            if columns.scan_limit.is_some() { ScanLimitDisposition::Represented } else { ScanLimitDisposition::Absent };
        Self {
            format: SourceFormat::Parquet,
            direct_device_read: rejected_features.is_empty(),
            file_count: files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            predicate_disposition: PredicateDisposition::Absent,
            byte_range_disposition: ByteRangeDisposition::FullFiles,
            partition_column_disposition,
            zero_column_count_disposition,
            scan_limit_disposition,
            schema_mapping_disposition,
            read_column_count: columns.read_columns.len(),
            projected_output_column_count: columns.projected_output_columns.len(),
            pure_filter_column_count: columns.pure_filter_columns.len(),
            rejected_features,
        }
    }
}

/// Disposition of a source-level scan limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScanLimitDisposition {
    /// No scan limit was attached to the source.
    Absent,
    /// Scan limit is represented by native source execution.
    Represented,
    /// Scan limit was present but could not be represented safely.
    Rejected,
}

impl ScanLimitDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Represented => "represented",
            Self::Rejected => "rejected",
        }
    }
}

/// Source format represented by native source metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormat {
    /// Apache Parquet files.
    Parquet,
}

/// Disposition of a source-level predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateDisposition {
    /// No predicate was attached to the source.
    Absent,
    /// Predicate is represented only for file or row-group pruning.
    PruningOnly,
    /// Predicate is represented as a reader-level Parquet predicate.
    ReaderPushdown,
    /// Predicate remains represented by native post-scan filtering.
    NativePostScanOnly,
    /// Predicate exists but cannot be represented by the native source.
    Rejected,
}

impl PredicateDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::PruningOnly => "pruning_only",
            Self::ReaderPushdown => "reader_pushdown",
            Self::NativePostScanOnly => "native_post_scan_only",
            Self::Rejected => "rejected",
        }
    }
}

/// Disposition of byte-range file entries.
///
/// This describes the planning-time representation of file entries, not the
/// runtime fetch strategy. When a source carries a `scan_limit` and points at
/// remote object-store URIs, the host-buffer reader transparently switches to
/// footer-first ranged GETs even though entries remain represented as
/// `FullFiles` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteRangeDisposition {
    /// Source entries describe whole files.
    FullFiles,
    /// Byte ranges were safely coalesced back to whole-file reads.
    CoalescedFullFiles,
    /// Byte-ranged source entries are represented by explicit row-group
    /// selections on whole-file local reads.
    RowGroupSelection,
    /// Byte ranges remain and are not supported by the native source.
    Rejected,
}

impl ByteRangeDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullFiles => "full_files",
            Self::CoalescedFullFiles => "coalesced_full_files",
            Self::RowGroupSelection => "row_group_selection",
            Self::Rejected => "rejected",
        }
    }
}

/// Disposition of partition columns used by the native source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartitionColumnDisposition {
    /// No partition columns are required.
    Absent,
    /// Required partition columns or values could not be represented.
    Rejected,
    /// Required partition columns are represented as synthetic source columns.
    Represented,
}

impl PartitionColumnDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

/// Disposition of metadata-only row-count sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeroColumnCountDisposition {
    /// The plan reads at least one source column, so zero-column count is unused.
    NotRequired,
    /// The plan required zero-column count, but it was not proven safe.
    Rejected,
    /// The source represents row counts without reading physical columns.
    Represented,
}

impl ZeroColumnCountDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Rejected => "rejected",
            Self::Represented => "represented",
        }
    }
}

/// Disposition of logical-to-source column mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaMappingDisposition {
    /// Logical columns map directly by name to file or partition columns.
    DirectByName,
    /// Nullable logical columns absent from the physical file are represented
    /// as typed null columns.
    MissingNullableColumns,
    /// Required column mapping could not be represented.
    Rejected,
}

impl SchemaMappingDisposition {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectByName => "direct_by_name",
            Self::MissingNullableColumns => "missing_nullable_columns",
            Self::Rejected => "rejected",
        }
    }
}

/// Stable rejection feature codes for native source diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectedSourceFeature {
    /// Source is not backed by a local filesystem object store.
    NonLocalObjectStore,
    /// Required partition columns could not be represented.
    PartitionColumns,
    /// DataFusion scan expression adapter is present.
    ExprAdapter,
    /// A custom Parquet file reader factory is present.
    ParquetFileReaderFactory,
    /// DataFusion scan limit is present.
    ScanLimit,
    /// Source requires preserved input order.
    PreserveOrder,
    /// Source declares output ordering.
    OutputOrdering,
    /// Pushed-down Parquet predicate cannot be represented safely.
    PushedDownParquetPredicate,
    /// Source has no files.
    EmptySource,
    /// Required partition values are missing or unsupported.
    PartitionValues,
    /// Byte-range file entries are not supported as represented.
    ByteRange,
    /// Per-file ordering metadata is present.
    FileOrdering,
    /// Row-level selection is present.
    RowSelection,
    /// Zero-column count was required but not represented.
    ZeroColumnCount,
    /// Logical-to-source column mapping is invalid.
    SchemaMapping,
}

impl RejectedSourceFeature {
    /// Stable lower-case code used by reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonLocalObjectStore => "non_local_object_store",
            Self::PartitionColumns => "partition_columns",
            Self::ExprAdapter => "expr_adapter",
            Self::ParquetFileReaderFactory => "parquet_file_reader_factory",
            Self::ScanLimit => "scan_limit",
            Self::PreserveOrder => "preserve_order",
            Self::OutputOrdering => "output_ordering",
            Self::PushedDownParquetPredicate => "pushed_down_parquet_predicate",
            Self::EmptySource => "empty_source",
            Self::PartitionValues => "partition_values",
            Self::ByteRange => "byte_range",
            Self::FileOrdering => "file_ordering",
            Self::RowSelection => "row_selection",
            Self::ZeroColumnCount => "zero_column_count",
            Self::SchemaMapping => "schema_mapping",
        }
    }
}
