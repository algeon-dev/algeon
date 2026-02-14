use std::collections::BTreeSet;

use arrow::datatypes::SchemaRef;
use cudf_query_engine::{
    LiteralValue as NativeLiteralValue, LocalParquetFile as NativeLocalParquetFile,
    PartitionColumnDisposition as NativePartitionColumnDisposition, SourceColumnRef as NativeSourceColumnRef,
};
use datafusion::{
    datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource},
    physical_plan::filter::FilterExec,
};
use parquet::file::{
    metadata::RowGroupMetaData,
    reader::{FileReader, SerializedFileReader},
};

use super::{
    literal::{NativeLiteralContext, native_literal_value},
    pruning::{
        coalesced_local_parquet_scan_files, explicit_row_group_selection, local_object_path_to_filesystem,
        metadata_row_group_selections, scan_predicate_is_dynamic_filter_hint, validate_pruning_only_parquet_predicate,
    },
};
use crate::policy::FallbackReason;

mod columns;
mod diagnostics;

pub(super) use columns::{
    NativeParquetColumnRequirements, native_source_output_schema, required_parquet_column_requirements,
    required_parquet_column_requirements_for_scan_columns,
};
pub(super) use diagnostics::{local_parquet_source_diagnostics, source_diagnostics_for_normalized_plan};

pub(super) fn validate_slice1_parquet_scan(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
) -> Result<(), FallbackReason> {
    if !file_scan.object_store_url.as_str().starts_with("file://") {
        return Err(FallbackReason::NativeSourceNonLocalObjectStore);
    }
    validate_partition_columns(file_scan, &column_requirements.partition_column_indices)?;
    if file_scan.expr_adapter_factory.is_some() {
        return Err(FallbackReason::NativeSourceExprAdapter);
    }
    if !native_parquet_file_reader_factory_is_supported(parquet_source) {
        return Err(FallbackReason::NativeSourceParquetFileReaderFactory);
    }
    if !native_preserve_order_scan_is_supported(file_scan) {
        return Err(FallbackReason::NativeSourcePreserveOrder);
    }
    if !file_scan.output_ordering.is_empty() {
        return Err(FallbackReason::NativeSourceOutputOrdering);
    }
    if parquet_source
        .filter()
        .is_some_and(|predicate| filter.is_none() && scan_predicate_is_dynamic_filter_hint(predicate.as_ref()))
    {
        validate_pruning_only_parquet_predicate(parquet_source, filter, &source_schema)?;
    }
    local_parquet_files(file_scan, parquet_source, filter, source_schema, column_requirements)?;
    Ok(())
}

pub(super) fn native_preserve_order_scan_is_supported(file_scan: &FileScanConfig) -> bool {
    !file_scan.preserve_order || file_scan.file_groups.len() == 1
}

pub(super) fn native_parquet_file_reader_factory_is_supported(parquet_source: &ParquetSource) -> bool {
    let Some(factory) = parquet_source.parquet_file_reader_factory() else {
        return true;
    };

    // DataFusion's normal local-file scans can carry its default or cached
    // object-store reader factory. Native local reads represent that contract
    // directly through filesystem paths; caller-provided factories remain
    // rejected because they may change bytes, metadata, caching, credentials,
    // or IO semantics in ways the native source IR cannot represent.
    let debug = format!("{factory:?}");
    debug.starts_with("DefaultParquetFileReaderFactory") || debug.starts_with("CachedParquetFileReaderFactory")
}

/// Converts DataFusion `PartitionedFile` entries into native local Parquet files.
///
/// The bridge carries over stable file identity, filesystem paths, selected row
/// groups, and required partition values while rejecting scan metadata that the
/// native source contract cannot represent.
pub(super) fn local_parquet_files(
    file_scan: &FileScanConfig,
    parquet_source: &ParquetSource,
    filter: Option<&FilterExec>,
    source_schema: SchemaRef,
    column_requirements: &NativeParquetColumnRequirements,
) -> Result<Vec<NativeLocalParquetFile>, FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if files.is_empty() {
        return Err(FallbackReason::NativeSourceEmpty);
    }
    let scan_files = coalesced_local_parquet_scan_files(&files)?;
    let metadata_row_group_selections =
        metadata_row_group_selections(&files, file_scan, parquet_source, filter, &source_schema);
    let mut native_files = Vec::with_capacity(scan_files.len());
    for (file_index, scan_file) in scan_files.iter().enumerate() {
        let file = scan_file.file;
        if file.ordering.is_some() {
            return Err(FallbackReason::NativeSourceFileOrdering);
        }
        let path = local_object_path_to_filesystem(file.object_meta.location.as_ref());
        let stable_id = format!("local:{}:{}", path.display(), file.object_meta.size);
        let mut native_file =
            NativeLocalParquetFile::new(path).with_stable_id(stable_id).with_file_size_bytes(file.object_meta.size);
        if !column_requirements.partition_column_indices.is_empty() {
            native_file = native_file
                .with_partition_values(native_partition_values(file, &column_requirements.partition_column_indices)?);
        }
        if let Some(row_groups) = explicit_row_group_selection(file)? {
            native_file = native_file.with_row_groups(row_groups);
        } else if let Some(row_groups) =
            metadata_row_group_selections.as_ref().and_then(|selections| selections.get(file_index))
        {
            native_file = native_file.with_row_groups(row_groups.clone());
        }
        native_file = attach_row_group_byte_metadata(native_file, column_requirements);
        native_files.push(native_file);
    }
    Ok(native_files)
}

fn attach_row_group_byte_metadata(
    mut native_file: NativeLocalParquetFile,
    column_requirements: &NativeParquetColumnRequirements,
) -> NativeLocalParquetFile {
    let Ok(reader) = SerializedFileReader::try_from(native_file.path.as_path()) else {
        return native_file;
    };
    let metadata = reader.metadata();
    let read_column_indexes = column_requirements.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
    let projected_column_indexes = column_requirements
        .projected_output_column_source_refs
        .iter()
        .filter_map(file_source_ref_index)
        .collect::<BTreeSet<_>>();
    let pure_filter_column_indexes = column_requirements
        .pure_filter_column_source_refs
        .iter()
        .filter_map(file_source_ref_index)
        .filter(|index| !projected_column_indexes.contains(index))
        .collect::<BTreeSet<_>>();

    if let Some(row_group_bytes) = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &read_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>()
    {
        native_file = native_file.with_row_group_bytes(row_group_bytes);
    }
    let projected_compressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>();
    let projected_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Uncompressed))
        .collect::<Option<Vec<_>>>();
    if let (Some(compressed), Some(uncompressed)) = (projected_compressed_bytes, projected_uncompressed_bytes) {
        native_file = native_file.with_projected_row_group_bytes(compressed, uncompressed);
    }
    let pure_filter_compressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Compressed))
        .collect::<Option<Vec<_>>>();
    let pure_filter_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Uncompressed))
        .collect::<Option<Vec<_>>>();
    if let (Some(compressed), Some(uncompressed)) = (pure_filter_compressed_bytes, pure_filter_uncompressed_bytes) {
        native_file = native_file.with_pure_filter_row_group_bytes(compressed, uncompressed);
    }
    if let Some(row_group_rows) = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows()).ok())
        .collect::<Option<Vec<_>>>()
    {
        native_file = native_file.with_row_group_rows(row_group_rows);
    }

    native_file
}

fn file_source_ref_index(source_ref: &NativeSourceColumnRef) -> Option<usize> {
    match source_ref {
        NativeSourceColumnRef::File(index) => Some(*index),
        NativeSourceColumnRef::Partition(_) | NativeSourceColumnRef::MissingNull(_) => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum ColumnByteKind {
    Compressed,
    Uncompressed,
}

fn row_group_column_bytes(
    row_group: &RowGroupMetaData,
    column_indexes: &BTreeSet<usize>,
    kind: ColumnByteKind,
) -> Option<u64> {
    column_indexes.iter().try_fold(0u64, |total, column_index| {
        let column = row_group.columns().get(*column_index)?;
        let bytes = match kind {
            ColumnByteKind::Compressed => column.compressed_size(),
            ColumnByteKind::Uncompressed => column.uncompressed_size(),
        };
        let bytes = u64::try_from(bytes).ok()?;
        total.checked_add(bytes)
    })
}

fn validate_partition_columns(
    file_scan: &FileScanConfig,
    required_partition_indices: &[usize],
) -> Result<(), FallbackReason> {
    let files = file_scan.file_groups.iter().flat_map(|group| group.files()).collect::<Vec<_>>();
    if required_partition_indices.is_empty() {
        return Ok(());
    }

    let partition_column_count = file_scan.table_partition_cols().len();
    if required_partition_indices.iter().any(|index| *index >= partition_column_count) {
        return Err(FallbackReason::NativeSourcePartitionValues);
    }
    for file in &files {
        if required_partition_indices.iter().any(|index| file.partition_values.get(*index).is_none()) {
            return Err(FallbackReason::NativeSourcePartitionValues);
        }
        native_partition_values(file, required_partition_indices)?;
    }
    Ok(())
}

fn partition_column_disposition(
    file_scan: &FileScanConfig,
    files: &[&datafusion::datasource::listing::PartitionedFile],
    required_partition_indices: &[usize],
) -> NativePartitionColumnDisposition {
    if required_partition_indices.is_empty() {
        return NativePartitionColumnDisposition::Absent;
    }
    let partition_column_count = file_scan.table_partition_cols().len();
    if required_partition_indices.iter().all(|index| *index < partition_column_count)
        && files.iter().all(|file| native_partition_values(file, required_partition_indices).is_ok())
    {
        NativePartitionColumnDisposition::Represented
    } else {
        NativePartitionColumnDisposition::Rejected
    }
}

fn native_partition_values(
    file: &datafusion::datasource::listing::PartitionedFile,
    required_partition_indices: &[usize],
) -> Result<Vec<NativeLiteralValue>, FallbackReason> {
    required_partition_indices
        .iter()
        .map(|index| {
            file.partition_values
                .get(*index)
                .ok_or(FallbackReason::NativeSourcePartitionValues)
                .and_then(|value| native_literal_value(value, NativeLiteralContext::SourceValue))
        })
        .collect()
}
