use std::path::Path;

use super::{LocalParquetFile, ParquetSourceLocation, PartitionColumn, SourceColumnRef};

// Borrowed column-layout snapshot used to build source diagnostics.
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

/// Row-group selection evidence for a local Parquet source.
///
/// `None` on the parent diagnostics means the source does not have enough
/// row-group metadata to distinguish unknown selection from an all-row-group
/// scan. Byte counts are optional because some frontends only know row counts
/// or file sizes at lowering time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedRowGroupStats {
    /// Number of row groups selected for scanning.
    pub selected_row_groups: usize,
    /// Total row groups represented by the source metadata.
    pub total_row_groups: usize,
    /// Selected uncompressed source-read bytes, when known.
    pub selected_uncompressed_bytes: Option<u64>,
    /// Total uncompressed source-read bytes, when known.
    pub total_uncompressed_bytes: Option<u64>,
    /// Selected compressed source-read bytes, when known.
    pub selected_compressed_bytes: Option<u64>,
    /// Total compressed source-read bytes, when known.
    pub total_compressed_bytes: Option<u64>,
    /// Selected source rows, when known.
    pub selected_rows: Option<u64>,
    /// Total source rows, when known.
    pub total_rows: Option<u64>,
}

impl SelectedRowGroupStats {
    /// Returns true when the selection covers every represented row group.
    pub fn selects_all_row_groups(&self) -> bool {
        self.selected_row_groups >= self.total_row_groups
    }

    /// Best available selected byte evidence for source-read sizing.
    pub fn selected_source_bytes(&self) -> Option<u64> {
        self.selected_uncompressed_bytes.or(self.selected_compressed_bytes)
    }

    /// Best available total byte evidence for source-read sizing.
    pub fn total_source_bytes(&self) -> Option<u64> {
        self.total_uncompressed_bytes.or(self.total_compressed_bytes)
    }
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
    /// Comma-separated high-level Parquet read strategies represented by this source.
    pub source_read_strategy: String,
    /// Comma-separated credential modes represented by this source.
    pub credential_mode: String,
    /// Comma-separated local byte-cache statuses represented by this source.
    pub lakehouse_cache_status: String,
    /// Comma-separated foreground lakehouse cache-read modes represented by this source.
    pub lakehouse_cache_read_mode: String,
    /// Comma-separated complete-object lakehouse cache fill modes represented by this source.
    pub lakehouse_cache_fill_mode: String,
    /// True when the source is proven to use a local path/cache reader that
    /// can satisfy the direct-device source-read boundary.
    pub direct_device_read: bool,
    /// Number of files represented after any frontend coalescing.
    pub file_count: usize,
    /// Number of source file entries that carried byte ranges.
    pub ranged_file_count: usize,
    /// Total known bytes across represented files, or `None` when incomplete.
    pub known_bytes: Option<u64>,
    /// Total selected row groups, or `None` when row-group selection is unknown.
    pub selected_row_groups: Option<usize>,
    /// Selected-versus-total row-group, byte, and row evidence.
    pub selected_row_group_stats: Option<SelectedRowGroupStats>,
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
        let selected_row_group_stats = selected_row_group_stats_for_files(files);
        let zero_column_count_disposition = if columns.read_columns.is_empty()
            && columns.missing_columns.is_empty()
            && columns.partition_columns.is_empty()
        {
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
        let has_field_id_bindings = files.iter().any(|file| !file.field_id_read_bindings.is_empty());
        let schema_mapping_disposition = if schema_refs_are_valid {
            if columns
                .projected_output_column_source_refs
                .iter()
                .chain(columns.pure_filter_column_source_refs.iter())
                .any(|source_ref| matches!(source_ref, SourceColumnRef::MissingNull(_)))
            {
                SchemaMappingDisposition::MissingNullableColumns
            } else if has_field_id_bindings {
                SchemaMappingDisposition::FieldIdBindings
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
            source_read_strategy: summarize_source_read_strategy(files),
            credential_mode: summarize_credential_mode(files),
            lakehouse_cache_status: summarize_lakehouse_cache_status(files),
            lakehouse_cache_read_mode: summarize_lakehouse_cache_read_mode(files),
            lakehouse_cache_fill_mode: summarize_lakehouse_cache_fill_mode(files),
            direct_device_read: rejected_features.is_empty() && files_have_proven_direct_device_read(files),
            file_count: files.len(),
            ranged_file_count: 0,
            known_bytes,
            selected_row_groups,
            selected_row_group_stats,
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

pub(crate) fn redacted_path_diagnostic(path: &Path) -> String {
    let path = path.display().to_string();
    redacted_location_diagnostic(path.as_str()).unwrap_or(path)
}

pub(crate) fn parquet_source_path_fact(location: &ParquetSourceLocation, path: &Path) -> String {
    if location.is_local_path() { redacted_path_diagnostic(path) } else { location.diagnostic_location() }
}

pub(crate) fn parquet_source_stable_id_fact(location: &ParquetSourceLocation, stable_id: &str) -> String {
    location.diagnostic_stable_id(stable_id)
}

pub(crate) fn parquet_file_path_fact(file: &LocalParquetFile) -> String {
    parquet_source_path_fact(&file.location, &file.path)
}

pub(crate) fn parquet_file_stable_id_fact(file: &LocalParquetFile) -> String {
    parquet_source_stable_id_fact(&file.location, file.stable_id.as_str())
}

pub(crate) fn local_parquet_file_row_group_metadata_count(file: &LocalParquetFile) -> Option<usize> {
    file.row_group_bytes
        .as_ref()
        .or(file.projected_row_group_uncompressed_bytes.as_ref())
        .or(file.projected_row_group_compressed_bytes.as_ref())
        .or(file.pure_filter_row_group_uncompressed_bytes.as_ref())
        .or(file.pure_filter_row_group_compressed_bytes.as_ref())
        .or(file.row_group_rows.as_ref())
        .map(Vec::len)
}

pub(crate) fn redacted_stable_id_diagnostic(stable_id: &str) -> String {
    redacted_location_diagnostic(stable_id).unwrap_or_else(|| stable_id.to_string())
}

pub(crate) fn redacted_object_store_path_diagnostic(path: &str) -> String {
    format!("object_store_path:fnv64:{:016x}", stable_fnv1a64(path.as_bytes()))
}

pub(super) fn redacted_debug_value(kind: &str, value: &str) -> String {
    format!("{kind}:fnv64:{:016x}:len:{}", stable_fnv1a64(value.as_bytes()), value.len())
}

pub(super) fn redacted_debug_option(kind: &str, value: Option<&str>) -> Option<String> {
    value.map(|value| redacted_debug_value(kind, value))
}

fn redacted_location_diagnostic(location: &str) -> Option<String> {
    let (scheme, _) = location.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    Some(object_store_diagnostic_id(scheme.as_str(), location))
}

pub(super) fn object_store_diagnostic_id(scheme: &str, identity: &str) -> String {
    format!("object_store:{scheme}:fnv64:{:016x}", stable_fnv1a64(identity.as_bytes()))
}

pub(super) fn stable_fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn summarize_source_read_strategy(files: &[LocalParquetFile]) -> String {
    summarize_file_values(files, |file| file.resolved_source.read_strategy().as_str())
}

pub(super) fn files_have_proven_direct_device_read(files: &[LocalParquetFile]) -> bool {
    !files.is_empty()
        && files.iter().all(|file| {
            matches!(
                file.resolved_source.read_strategy(),
                super::ParquetReadStrategy::LocalKvikio | super::ParquetReadStrategy::LocalCacheKvikio
            )
        })
}

fn summarize_credential_mode(files: &[LocalParquetFile]) -> String {
    summarize_file_values(files, |file| file.resolved_source.credential_mode().as_str())
}

fn summarize_lakehouse_cache_status(files: &[LocalParquetFile]) -> String {
    summarize_file_values(files, |file| file.resolved_source.cache_status.as_str())
}

fn summarize_lakehouse_cache_read_mode(files: &[LocalParquetFile]) -> String {
    summarize_file_values(files, |file| file.resolved_source.lakehouse_cache_read_mode.as_str())
}

fn summarize_lakehouse_cache_fill_mode(files: &[LocalParquetFile]) -> String {
    summarize_file_values(files, |file| file.resolved_source.lakehouse_cache_fill_mode.as_str())
}

fn summarize_file_values(files: &[LocalParquetFile], value: impl Fn(&LocalParquetFile) -> &'static str) -> String {
    let mut values = files.iter().map(value).collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    values.join(",")
}

fn selected_row_group_stats_for_files(files: &[LocalParquetFile]) -> Option<SelectedRowGroupStats> {
    let mut aggregate = SelectedRowGroupStatsAccumulator::default();
    for file in files {
        let file_stats = selected_row_group_stats_for_file(file)?;
        aggregate.add(file_stats)?;
    }
    aggregate.finish()
}

fn selected_row_group_stats_for_file(file: &LocalParquetFile) -> Option<SelectedRowGroupStats> {
    let total_row_groups = local_parquet_file_row_group_metadata_count(file)?;
    if total_row_groups == 0 {
        return None;
    }
    let selected_indices = file.row_groups.clone().unwrap_or_else(|| (0..total_row_groups).collect::<Vec<_>>());
    if selected_indices.iter().any(|index| *index >= total_row_groups) {
        return None;
    }

    let total_compressed_bytes = source_read_bytes(
        file.projected_row_group_compressed_bytes.as_ref(),
        file.pure_filter_row_group_compressed_bytes.as_ref(),
        file.row_group_bytes.as_ref(),
        0..total_row_groups,
    )
    .or(file.file_size_bytes);
    let selected_compressed_bytes = source_read_bytes(
        file.projected_row_group_compressed_bytes.as_ref(),
        file.pure_filter_row_group_compressed_bytes.as_ref(),
        file.row_group_bytes.as_ref(),
        selected_indices.iter().copied(),
    )
    .or_else(|| proportional_file_bytes(file.file_size_bytes, selected_indices.len(), total_row_groups));
    let total_uncompressed_bytes = source_read_bytes(
        file.projected_row_group_uncompressed_bytes.as_ref(),
        file.pure_filter_row_group_uncompressed_bytes.as_ref(),
        file.row_group_bytes.as_ref(),
        0..total_row_groups,
    );
    let selected_uncompressed_bytes = source_read_bytes(
        file.projected_row_group_uncompressed_bytes.as_ref(),
        file.pure_filter_row_group_uncompressed_bytes.as_ref(),
        file.row_group_bytes.as_ref(),
        selected_indices.iter().copied(),
    );
    let total_rows = row_group_sum(file.row_group_rows.as_ref(), 0..total_row_groups);
    let selected_rows = row_group_sum(file.row_group_rows.as_ref(), selected_indices.iter().copied());

    Some(SelectedRowGroupStats {
        selected_row_groups: selected_indices.len(),
        total_row_groups,
        selected_uncompressed_bytes,
        total_uncompressed_bytes,
        selected_compressed_bytes,
        total_compressed_bytes,
        selected_rows,
        total_rows,
    })
}

fn source_read_bytes(
    projected: Option<&Vec<u64>>,
    pure_filter: Option<&Vec<u64>>,
    fallback: Option<&Vec<u64>>,
    indexes: impl IntoIterator<Item = usize> + Clone,
) -> Option<u64> {
    match (projected, pure_filter) {
        (Some(projected), Some(pure_filter)) => {
            checked_row_group_sum(projected, indexes.clone())?.checked_add(checked_row_group_sum(pure_filter, indexes)?)
        },
        (Some(projected), None) => checked_row_group_sum(projected, indexes),
        (None, Some(pure_filter)) => checked_row_group_sum(pure_filter, indexes),
        (None, None) => row_group_sum(fallback, indexes),
    }
}

fn row_group_sum(values: Option<&Vec<u64>>, indexes: impl IntoIterator<Item = usize>) -> Option<u64> {
    checked_row_group_sum(values?, indexes)
}

fn checked_row_group_sum(values: &[u64], indexes: impl IntoIterator<Item = usize>) -> Option<u64> {
    indexes.into_iter().try_fold(0_u64, |total, index| total.checked_add(*values.get(index)?))
}

fn proportional_file_bytes(
    file_size_bytes: Option<u64>,
    selected_row_groups: usize,
    total_row_groups: usize,
) -> Option<u64> {
    if total_row_groups == 0 {
        return None;
    }
    let file_size_bytes = file_size_bytes?;
    let numerator = u128::from(file_size_bytes).checked_mul(selected_row_groups as u128)?;
    u64::try_from(numerator / total_row_groups as u128).ok()
}

#[derive(Debug)]
struct SelectedRowGroupStatsAccumulator {
    selected_row_groups: usize,
    total_row_groups: usize,
    selected_uncompressed_bytes: Option<u64>,
    total_uncompressed_bytes: Option<u64>,
    selected_compressed_bytes: Option<u64>,
    total_compressed_bytes: Option<u64>,
    selected_rows: Option<u64>,
    total_rows: Option<u64>,
}

impl Default for SelectedRowGroupStatsAccumulator {
    fn default() -> Self {
        Self {
            selected_row_groups: 0,
            total_row_groups: 0,
            selected_uncompressed_bytes: Some(0),
            total_uncompressed_bytes: Some(0),
            selected_compressed_bytes: Some(0),
            total_compressed_bytes: Some(0),
            selected_rows: Some(0),
            total_rows: Some(0),
        }
    }
}

impl SelectedRowGroupStatsAccumulator {
    fn add(&mut self, stats: SelectedRowGroupStats) -> Option<()> {
        self.selected_row_groups = self.selected_row_groups.checked_add(stats.selected_row_groups)?;
        self.total_row_groups = self.total_row_groups.checked_add(stats.total_row_groups)?;
        self.selected_uncompressed_bytes =
            add_optional_u64(self.selected_uncompressed_bytes, stats.selected_uncompressed_bytes)?;
        self.total_uncompressed_bytes =
            add_optional_u64(self.total_uncompressed_bytes, stats.total_uncompressed_bytes)?;
        self.selected_compressed_bytes =
            add_optional_u64(self.selected_compressed_bytes, stats.selected_compressed_bytes)?;
        self.total_compressed_bytes = add_optional_u64(self.total_compressed_bytes, stats.total_compressed_bytes)?;
        self.selected_rows = add_optional_u64(self.selected_rows, stats.selected_rows)?;
        self.total_rows = add_optional_u64(self.total_rows, stats.total_rows)?;
        Some(())
    }

    fn finish(self) -> Option<SelectedRowGroupStats> {
        (self.total_row_groups > 0).then_some(SelectedRowGroupStats {
            selected_row_groups: self.selected_row_groups,
            total_row_groups: self.total_row_groups,
            selected_uncompressed_bytes: self.selected_uncompressed_bytes,
            total_uncompressed_bytes: self.total_uncompressed_bytes,
            selected_compressed_bytes: self.selected_compressed_bytes,
            total_compressed_bytes: self.total_compressed_bytes,
            selected_rows: self.selected_rows,
            total_rows: self.total_rows,
        })
    }
}

fn add_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<Option<u64>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(Some(left.checked_add(right)?)),
        _ => Some(None),
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
/// lower-level reads performed by the selected Parquet backend.
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
    /// Logical columns map through footer-proven table-format field ids.
    FieldIdBindings,
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
            Self::FieldIdBindings => "field_id_bindings",
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn parquet_source_path_fact_redacts_uri_shaped_local_paths() {
        let path = PathBuf::from("s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc");
        let fact = parquet_source_path_fact(&ParquetSourceLocation::local_path(&path), &path);

        assert!(fact.starts_with("object_store:s3:fnv64:"));
        assert!(!fact.contains("secret-bucket"));
        assert!(!fact.contains("X-Amz-Signature"));
    }

    #[test]
    fn parquet_source_path_fact_uses_object_store_diagnostic_location() {
        let uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
        let location = ParquetSourceLocation::object_store_uri(uri).expect("valid object-store URI");
        let fact = parquet_source_path_fact(&location, PathBuf::from(uri).as_path());

        assert!(fact.starts_with("object_store:s3:fnv64:"));
        assert!(!fact.contains("secret-bucket"));
        assert!(!fact.contains("X-Amz-Signature"));
    }

    #[test]
    fn local_parquet_file_row_group_metadata_count_uses_first_available_vector() {
        let file = LocalParquetFile::new("lineitem.parquet")
            .with_projected_row_group_bytes(vec![10, 20, 30], vec![100, 200, 300])
            .with_row_group_rows(vec![1, 2, 3]);

        assert_eq!(local_parquet_file_row_group_metadata_count(&file), Some(3));
    }
}
