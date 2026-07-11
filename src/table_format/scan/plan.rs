use super::*;

/// Resolved table-format scan facts consumed by later native lowering stages.
///
/// The plan is metadata-only: data files, delete files, projected columns, and
/// pure-filter columns are facts for later source lowering. Unsupported
/// features remain attached so the adapter can produce structured fallback
/// reasons instead of silently dropping table-format semantics. `scan_limit`
/// is an optional source-level upper bound, not a final SQL `LIMIT`; when a
/// parent inexact filter remains, final row limiting is still represented by a
/// separate native or DataFusion limit operator.
#[derive(Clone, PartialEq)]
pub struct TableFormatScanPlan {
    /// Table format that produced these scan facts.
    pub format: TableFormat,
    /// Snapshot id selected by planning, if known.
    pub snapshot_id: Option<String>,
    /// Stable catalog label, if known.
    pub catalog_type: Option<String>,
    /// Optional source-level row upper bound supplied during table-format planning.
    pub scan_limit: Option<usize>,
    /// Logical table schema after projection.
    pub logical_schema: SchemaRef,
    /// Schema expected by native source lowering.
    pub native_source_schema: SchemaRef,
    /// Planned data files.
    pub data_files: Vec<TableFormatDataFile>,
    /// Planned delete files. Supported position deletes may be represented by
    /// native source execution; unsupported delete content remains in
    /// `unsupported_features`.
    pub delete_files: Vec<TableFormatDeleteFile>,
    /// Output column mappings.
    pub projected_columns: Vec<TableFormatColumnMapping>,
    /// Filter-only column mappings.
    pub pure_filter_columns: Vec<TableFormatColumnMapping>,
    /// Unsupported features retained for structured fallback.
    pub unsupported_features: Vec<TableFormatUnsupportedFeature>,
}

impl fmt::Debug for TableFormatScanPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableFormatScanPlan")
            .field("format", &self.format)
            .field("snapshot_id", &self.snapshot_id)
            .field("catalog_type", &self.catalog_type)
            .field("scan_limit", &self.scan_limit)
            .field("logical_schema", &self.logical_schema)
            .field("native_source_schema", &self.native_source_schema)
            .field("data_files", &self.data_files)
            .field("delete_files", &self.delete_files)
            .field("projected_columns", &self.projected_columns)
            .field("pure_filter_columns", &self.pure_filter_columns)
            .field("unsupported_features", &self.unsupported_features)
            .finish()
    }
}

impl TableFormatScanPlan {
    /// Creates an empty metadata-only scan plan for one table format.
    pub fn new(format: TableFormat) -> Self {
        let empty_schema = Arc::new(Schema::empty());
        Self {
            format,
            snapshot_id: None,
            catalog_type: None,
            scan_limit: None,
            logical_schema: Arc::clone(&empty_schema),
            native_source_schema: empty_schema,
            data_files: Vec::new(),
            delete_files: Vec::new(),
            projected_columns: Vec::new(),
            pure_filter_columns: Vec::new(),
            unsupported_features: Vec::new(),
        }
    }

    /// Records the selected table snapshot id when the format exposes one.
    pub fn with_snapshot_id(mut self, snapshot_id: impl Into<String>) -> Self {
        self.snapshot_id = Some(snapshot_id.into());
        self
    }

    /// Records a stable catalog label such as `glue` for diagnostics.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Records an optional source-level row upper bound.
    pub fn with_scan_limit(mut self, scan_limit: Option<usize>) -> Self {
        self.scan_limit = scan_limit;
        self
    }

    /// Records the logical table schema after table-format projection.
    pub fn with_logical_schema(mut self, schema: SchemaRef) -> Self {
        self.logical_schema = schema;
        self
    }

    /// Records the physical/native source schema expected by native lowering.
    pub fn with_native_source_schema(mut self, schema: SchemaRef) -> Self {
        self.native_source_schema = schema;
        self
    }

    /// Adds one planned data file fact.
    pub fn with_data_file(mut self, file: TableFormatDataFile) -> Self {
        self.data_files.push(file);
        self
    }

    /// Adds one delete-file fact that native admission must currently reject.
    pub fn with_delete_file(mut self, file: TableFormatDeleteFile) -> Self {
        self.delete_files.push(file);
        self
    }

    /// Adds one projected output column mapping.
    pub fn with_projected_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.projected_columns.push(column);
        self
    }

    /// Adds one column needed only for source filtering.
    pub fn with_pure_filter_column(mut self, column: TableFormatColumnMapping) -> Self {
        self.pure_filter_columns.push(column);
        self
    }

    /// Adds one unsupported table-format feature retained for structured fallback.
    pub fn with_unsupported_feature(mut self, feature: TableFormatUnsupportedFeature) -> Self {
        self.unsupported_features.push(feature);
        self
    }

    /// Number of planned data files.
    pub fn planned_data_file_count(&self) -> usize {
        self.data_files.len()
    }

    /// Number of planned delete files.
    pub fn planned_delete_file_count(&self) -> usize {
        self.delete_files.len()
    }

    /// Total planned data-file bytes when every file size is known.
    ///
    /// Returns `None` if any data file lacks a byte size; `None` means
    /// "unknown total", not "zero bytes".
    pub fn planned_bytes(&self) -> Option<u64> {
        self.data_files
            .iter()
            .map(|file| file.size_bytes)
            .try_fold(0u64, |total, size| size.map(|size| total.saturating_add(size)))
    }

    /// All-or-nothing native-admission predicate for the current scan plan.
    ///
    /// A plan is admissible only when it has at least one native-readable
    /// Parquet data file, all associated delete files are supported row-level
    /// deletes, and no retained unsupported features remain. Object-store S3
    /// files are native-readable through the host-buffer source pipeline, not
    /// through direct device reads.
    pub fn is_native_admissible(&self) -> bool {
        self.unsupported_features.is_empty()
            && !self.data_files.is_empty()
            && self.data_files.iter().all(TableFormatDataFile::is_native_parquet_representable)
            && self.delete_files.iter().all(TableFormatDeleteFile::is_supported_row_delete)
            && self.delete_files.iter().all(|delete_file| self.row_delete_file_is_associated(delete_file))
    }

    /// Number of source files native lowering may consume.
    ///
    /// This returns zero unless the whole scan is native-admissible.
    pub fn admitted_native_source_file_count(&self) -> usize {
        if self.is_native_admissible() { self.data_files.len() } else { 0 }
    }

    /// Stable fallback reasons for retained unsupported features.
    pub fn unsupported_fallback_reasons(&self) -> Vec<FallbackReason> {
        self.unsupported_features.iter().map(TableFormatUnsupportedFeature::fallback_reason).collect()
    }

    /// Stable unsupported-feature codes for diagnostics/report rows.
    pub fn rejected_feature_codes(&self) -> Vec<String> {
        self.unsupported_features.iter().map(|feature| feature.code().to_string()).collect()
    }

    /// Direct device-read disposition derived from native admissibility.
    pub fn direct_device_read_disposition(&self) -> TableFormatDirectDeviceReadDisposition {
        if self.is_native_admissible()
            && self.data_files.iter().all(TableFormatDataFile::is_direct_native_parquet_representable)
        {
            TableFormatDirectDeviceReadDisposition::Eligible
        } else {
            TableFormatDirectDeviceReadDisposition::Rejected
        }
    }

    /// First local rejection reason implied by retained scan facts.
    ///
    /// This mirrors the native table-format lowering admission order and is
    /// intentionally credential-free. It is used by late diagnostics, such as
    /// the GPU-only leaf guard, when the native rule did not retain enough
    /// context to explain why the scan leaf could not be consumed.
    pub fn first_rejection_reason(&self) -> Option<FallbackReason> {
        if self.data_files.is_empty() && self.delete_files.is_empty() && self.unsupported_features.is_empty() {
            return None;
        }
        if let Some(reason) = self.unsupported_fallback_reasons().into_iter().next() {
            return Some(reason);
        }
        if self.delete_files.iter().any(|file| !file.is_supported_row_delete()) {
            return Some(FallbackReason::IcebergDeleteFilesUnsupported);
        }
        if self.delete_files.iter().any(|file| !self.row_delete_file_is_associated(file)) {
            return Some(FallbackReason::IcebergDeleteFilesUnsupported);
        }
        if self.data_files.is_empty() {
            return Some(FallbackReason::NativeSourceEmpty);
        }
        if self.data_files.iter().any(|file| file.format != TableFormatDataFileFormat::Parquet) {
            return Some(FallbackReason::IcebergNonParquetDataFile);
        }
        if self.data_files.iter().any(|file| !file.read_capability.is_native_parquet_capable()) {
            return Some(FallbackReason::IcebergObjectStoreReadUnsupported);
        }
        if self.first_unsupported_column().is_some() {
            return Some(FallbackReason::UnsupportedType);
        }
        None
    }

    /// First projected or filter-only column whose Arrow type cannot enter native execution.
    pub(crate) fn first_unsupported_column(&self) -> Option<TableFormatUnsupportedColumn<'_>> {
        self.projected_columns
            .iter()
            .chain(self.pure_filter_columns.iter())
            .find(|column| !nexus_query_engine::is_supported_arrow_type(&column.data_type))
            .map(|column| TableFormatUnsupportedColumn {
                logical_name: column.logical_name.as_str(),
                data_type: &column.data_type,
                role: column.role,
            })
    }

    fn row_delete_file_is_associated(&self, delete_file: &TableFormatDeleteFile) -> bool {
        self.data_files.iter().any(|data_file| {
            data_file.id == delete_file.data_file_id
                && (data_file.position_delete_files.iter().any(|associated| associated.id == delete_file.id)
                    || data_file.equality_delete_files.iter().any(|associated| associated.id == delete_file.id))
        })
    }

    /// Comma-separated stable read-capability codes across data files.
    pub fn native_source_read_capability_summary(&self) -> String {
        let mut capabilities =
            self.data_files.iter().map(|file| file.read_capability.code().to_string()).collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        capabilities.join(",")
    }

    /// Comma-separated high-level source read strategies across data files.
    pub fn source_read_strategy_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(table_format_source_read_contract)),
        )
    }

    /// Comma-separated decode input kinds across data files.
    pub fn source_decode_input_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.decode_input.as_str())),
        )
    }

    /// Comma-separated credential modes across data files.
    pub fn credential_mode_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.credential_mode.as_str())),
        )
    }

    /// Comma-separated lakehouse byte-cache statuses across data files.
    pub fn lakehouse_cache_status_summary(&self) -> String {
        summarize_resolved_source_value(
            self.data_files
                .iter()
                .filter_map(|file| file.resolved_parquet_source.as_ref().map(|source| source.cache_status.as_str())),
        )
    }

    /// Comma-separated foreground lakehouse cache-read modes across data files.
    pub fn lakehouse_cache_read_mode_summary(&self) -> String {
        summarize_resolved_source_value(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.lakehouse_cache_read_mode.as_str())
        }))
    }

    /// Comma-separated lakehouse complete-object cache fill modes across data files.
    pub fn lakehouse_cache_fill_mode_summary(&self) -> String {
        summarize_resolved_source_value(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.lakehouse_cache_fill_mode.as_str())
        }))
    }

    /// Comma-separated stable source fallback reasons across data files.
    pub fn source_fallback_reason_summary(&self) -> String {
        summarize_resolved_source_fallback_reason(self.data_files.iter().filter_map(|file| {
            file.resolved_parquet_source.as_ref().map(|source| source.fallback_reason.as_deref().unwrap_or("none"))
        }))
    }

    /// Builds credential-free scan diagnostics for tests and report rows.
    pub fn diagnostics(&self) -> TableFormatScanDiagnostics {
        TableFormatScanDiagnostics {
            table_format: self.format.code().to_string(),
            snapshot_id: self.snapshot_id.clone(),
            catalog_type: self.catalog_type.clone(),
            planned_data_file_count: self.planned_data_file_count(),
            planned_delete_file_count: self.planned_delete_file_count(),
            planned_bytes: self.planned_bytes(),
            admitted_native_source_file_count: self.admitted_native_source_file_count(),
            rejected_feature_list: self.rejected_feature_codes(),
            native_source_read_capability: self.native_source_read_capability_summary(),
            source_read_strategy: self.source_read_strategy_summary(),
            source_decode_input: self.source_decode_input_summary(),
            credential_mode: self.credential_mode_summary(),
            lakehouse_cache_status: self.lakehouse_cache_status_summary(),
            lakehouse_cache_read_mode: self.lakehouse_cache_read_mode_summary(),
            lakehouse_cache_fill_mode: self.lakehouse_cache_fill_mode_summary(),
            source_fallback_reason: self.source_fallback_reason_summary(),
            direct_device_read_disposition: self.direct_device_read_disposition(),
        }
    }
}

fn summarize_resolved_source_value<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    if values.is_empty() { "unknown".to_string() } else { values.join(",") }
}

fn summarize_resolved_source_fallback_reason<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

pub(super) fn table_format_source_read_contract(source: &ResolvedParquetSource) -> &'static str {
    match source.read_strategy {
        nexus_query_engine::ParquetReadStrategy::LocalCacheKvikio
            if source.cache_status == LakehouseCacheStatus::Hit =>
        {
            "local_cache_hit"
        },
        nexus_query_engine::ParquetReadStrategy::RemoteHostBuffer if source.remote_host_buffer_ranged_enabled => {
            "remote_host_buffer_ranged"
        },
        nexus_query_engine::ParquetReadStrategy::RemoteHostBuffer => "remote_host_buffer_full",
        _ => source.read_strategy.as_str(),
    }
}
